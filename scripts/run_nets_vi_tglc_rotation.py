#!/usr/bin/env python3
"""Run TGLC and rotation-period checks for the NETS VI target stars."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs, Tesscut
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tglc.quick_lc as quick_lc_module
from tglc.quick_lc import tglc_lc
from tglc.target_lightcurve import epsf as _tglc_epsf


def _quiet_epsf(*args, **kwargs):
    kwargs.setdefault("no_progress_bar", True)
    return _tglc_epsf(*args, **kwargs)


quick_lc_module.epsf = _quiet_epsf


@dataclass(frozen=True)
class Target:
    name: str
    tic: int
    gaia_dr3: int
    expected_rotation_days: tuple[float, ...]
    notes: str


TARGETS = (
    Target("HD 126053", 446436963, 3654496279558010624, (26.0,), "Hempelmann et al. TIGRE rotation period"),
    Target("HD 168009", 416117806, 2115351387048328832, (6.0, 30.0), "TIGRE 6 d; literature/RV activity near 30 d"),
    Target("HD 10780", 373694425, 512167948043650816, (25.6,), "NETS RV activity signal near 25.6 d"),
)

PRIMARY_FLUXES = ("aperture_flux", "psf_flux")
CHECK_FLUXES = PRIMARY_FLUXES + ("cal_aper_flux",)


def target_slug(name: str) -> str:
    return name.replace(" ", "_")


def exact_tic_row(tic: int):
    rows = Catalogs.query_object(f"TIC {tic}", radius=0.01, catalog="TIC")
    if len(rows) == 0:
        raise RuntimeError(f"No TIC rows returned for TIC {tic}")
    matches = rows[np.asarray(rows["ID"], dtype=int) == tic]
    if len(matches) == 0:
        raise RuntimeError(f"TIC {tic} not found in returned TIC rows")
    return matches[0]


def sector_table_for_tic(tic: int) -> pd.DataFrame:
    row = exact_tic_row(tic)
    coord = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)
    sectors = Tesscut.get_sectors(coordinates=coord)
    return sectors.to_pandas().sort_values("sector").reset_index(drop=True)


def lc_files_for_sector(target_dir: Path, sector: int) -> list[Path]:
    pattern = f"hlsp_tglc_tess_ffi_gaiaid-*-s{sector:04d}-*.fits"
    return sorted((target_dir / "lc").glob(pattern)) + sorted((target_dir / "lc").glob(f"*/{pattern}"))


def matching_lc_files(target_dir: Path, tic: int, gaia_dr3: int | None = None, sector: int | None = None) -> list[Path]:
    if sector is None:
        files = sorted((target_dir / "lc").glob("hlsp_tglc_tess_ffi_gaiaid-*-s*.fits"))
        files += sorted((target_dir / "lc").glob("*/hlsp_tglc_tess_ffi_gaiaid-*-s*.fits"))
    else:
        files = lc_files_for_sector(target_dir, sector)

    matches: list[Path] = []
    fallbacks: list[Path] = []
    for path in files:
        try:
            with fits.open(path, mode="denywrite", memmap=False) as hdul:
                hdr_tic = str(hdul[0].header.get("TICID", "")).strip()
                hdr_gaia = str(hdul[0].header.get("GAIADR3", "")).strip()
                gaia_ok = gaia_dr3 is None or hdr_gaia == str(gaia_dr3)
                if hdr_tic == str(tic) and gaia_ok:
                    matches.append(path)
                elif not hdr_tic and gaia_ok:
                    fallbacks.append(path)
        except Exception:
            continue
    return matches if matches else fallbacks


def run_missing_tglc(target: Target, sectors: Iterable[int], outdir: Path, args: argparse.Namespace) -> list[dict]:
    target_dir = outdir / target_slug(target.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for sector in sectors:
        sector = int(sector)
        existing = matching_lc_files(target_dir, target.tic, target.gaia_dr3, sector=sector)
        if existing and not args.force:
            results.append(
                {
                    "target": target.name,
                    "tic": target.tic,
                    "sector": sector,
                    "status": "skipped_existing",
                    "files": [str(p) for p in existing],
                }
            )
            continue

        try:
            tglc_lc(
                target=f"TIC {target.tic}",
                local_directory=str(target_dir.resolve()) + "/",
                size=args.size,
                save_aper=True,
                limit_mag=args.limit_mag,
                get_all_lc=False,
                sector=sector,
                ffi=args.ffi,
                mast_timeout=args.mast_timeout,
            )
            files = matching_lc_files(target_dir, target.tic, target.gaia_dr3, sector=sector)
            results.append(
                {
                    "target": target.name,
                    "tic": target.tic,
                    "sector": sector,
                    "status": "ok" if files else "ran_no_matching_file",
                    "files": [str(p) for p in files],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "target": target.name,
                    "tic": target.tic,
                    "sector": sector,
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return results


def load_flux_series(paths: Iterable[Path], flux_col: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for path in sorted(paths):
        with fits.open(path, mode="denywrite", memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[1].data
            if flux_col not in data.names:
                continue
            time = np.asarray(data["time"], dtype=float)
            flux = np.asarray(data[flux_col], dtype=float)
            tess_flags = np.asarray(data["TESS_flags"], dtype=int)
            tglc_flags = np.asarray(data["TGLC_flags"], dtype=int)
            ok = np.isfinite(time) & np.isfinite(flux) & (tess_flags == 0) & (tglc_flags == 0)
            if ok.sum() < 10:
                continue
            time = time[ok]
            flux = flux[ok]
            med = np.nanmedian(flux)
            if np.isfinite(med) and abs(med) > 0:
                rel_flux = flux / med - 1.0
            else:
                rel_flux = flux - np.nanmedian(flux)
            clipped = sigma_clip(rel_flux, sigma=5.0, maxiters=3, masked=True)
            keep = ~np.asarray(clipped.mask)
            if keep.sum() < 10:
                continue
            chunks.append(
                pd.DataFrame(
                    {
                        "time": time[keep],
                        "rel_flux": rel_flux[keep],
                        "sector": int(hdr["SECTOR"]),
                        "tessmag": float(hdr.get("TESSMAG", np.nan)),
                        "contamrt": float(hdr.get("CONTAMRT", np.nan)),
                        "path": str(path),
                    }
                )
            )
    if not chunks:
        return pd.DataFrame(columns=["time", "rel_flux", "sector", "tessmag", "contamrt", "path"])
    return pd.concat(chunks, ignore_index=True).sort_values("time").reset_index(drop=True)


def local_peak_periods(period: np.ndarray, power: np.ndarray, max_count: int = 5) -> list[tuple[float, float]]:
    if len(period) < 3:
        return []
    peaks, _ = find_peaks(power)
    if len(peaks) == 0:
        peaks = np.arange(len(power))
    order = peaks[np.argsort(power[peaks])[::-1]]
    selected: list[tuple[float, float]] = []
    for idx in order:
        p = float(period[idx])
        if all(abs(p - p0) / p0 > 0.05 for p0, _ in selected):
            selected.append((p, float(power[idx])))
        if len(selected) >= max_count:
            break
    return selected


def lomb_scargle_summary(df: pd.DataFrame, min_period: float, max_period: float, expected: tuple[float, ...]) -> dict:
    if len(df) < 20:
        return {"status": "too_few_points", "n_points": int(len(df))}

    time = np.asarray(df["time"], dtype=float)
    flux = np.asarray(df["rel_flux"], dtype=float)
    flux = flux - np.nanmedian(flux)
    baseline = float(np.nanmax(time) - np.nanmin(time))
    use_max_period = min(max_period, max(min_period * 1.5, baseline * 0.9))
    if use_max_period <= min_period:
        return {"status": "baseline_too_short", "n_points": int(len(df)), "baseline_days": baseline}

    ls = LombScargle(time, flux, center_data=True, fit_mean=True)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / use_max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=15,
    )
    period = 1.0 / freq
    best_index = int(np.nanargmax(power))
    best_period = float(period[best_index])
    best_power = float(power[best_index])
    try:
        fap = float(ls.false_alarm_probability(best_power, method="baluev"))
    except Exception:
        fap = math.nan

    expected_rows = []
    for p in expected:
        if min_period <= p <= use_max_period:
            expected_rows.append(
                {
                    "period_days": float(p),
                    "power": float(ls.power(1.0 / p)),
                }
            )

    return {
        "status": "ok",
        "n_points": int(len(df)),
        "n_sectors": int(df["sector"].nunique()),
        "sectors": ",".join(str(int(s)) for s in sorted(df["sector"].unique())),
        "tessmag_median": float(np.nanmedian(df["tessmag"])),
        "contamrt_median": float(np.nanmedian(df["contamrt"])),
        "first_tbjd": float(np.nanmin(time)),
        "last_tbjd": float(np.nanmax(time)),
        "baseline_days": baseline,
        "search_min_days": float(min_period),
        "search_max_days": float(use_max_period),
        "best_period_days": best_period,
        "best_power": best_power,
        "best_fap": fap,
        "top_peaks": [{"period_days": p, "power": pw} for p, pw in local_peak_periods(period, power)],
        "expected_period_powers": expected_rows,
        "period_grid_days": period.tolist(),
        "power_grid": power.tolist(),
    }


def make_target_plot(target: Target, target_dir: Path, summaries: dict[str, dict], outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(f"{target.name} / TIC {target.tic}", fontsize=14)
    for row, flux_col in enumerate(PRIMARY_FLUXES):
        df = load_flux_series(matching_lc_files(target_dir, target.tic, target.gaia_dr3), flux_col)
        ax_lc = axes[row, 0]
        ax_pg = axes[row, 1]
        ax_lc.set_title(flux_col)
        if len(df) > 0:
            for sector, grp in df.groupby("sector"):
                ax_lc.plot(grp["time"], 1e3 * grp["rel_flux"], ".", ms=1.8, alpha=0.7, label=str(int(sector)))
            ax_lc.set_xlabel("TBJD")
            ax_lc.set_ylabel("relative flux [ppt]")
            if df["sector"].nunique() <= 10:
                ax_lc.legend(title="Sector", fontsize=7, markerscale=3, ncols=2)
        summary = summaries.get(flux_col, {})
        if summary.get("status") == "ok":
            period = np.asarray(summary["period_grid_days"])
            power = np.asarray(summary["power_grid"])
            order = np.argsort(period)
            ax_pg.plot(period[order], power[order], color="0.15", lw=1.0)
            for p in target.expected_rotation_days:
                ax_pg.axvline(p, color="tab:red", ls="--", lw=0.9, alpha=0.8)
            ax_pg.axvline(summary["best_period_days"], color="tab:blue", ls=":", lw=0.9)
            ax_pg.set_xlim(summary["search_min_days"], summary["search_max_days"])
            ax_pg.set_xlabel("Period [d]")
            ax_pg.set_ylabel("GLS power")
            ax_pg.set_title(
                f"best {summary['best_period_days']:.2f} d, FAP {summary['best_fap']:.2g}"
            )
        else:
            ax_pg.text(0.5, 0.5, summary.get("status", "no data"), ha="center", va="center")
            ax_pg.set_axis_off()
    plot_path = outdir / f"{target_slug(target.name)}_tglc_rotation.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def flatten_summary_row(target: Target, flux_col: str, summary: dict) -> dict:
    row = {
        "target": target.name,
        "tic": target.tic,
        "gaia_dr3": target.gaia_dr3,
        "flux": flux_col,
        "expected_rotation_days": ";".join(f"{p:g}" for p in target.expected_rotation_days),
        "notes": target.notes,
        "status": summary.get("status", "missing"),
    }
    for key in (
        "n_points",
        "n_sectors",
        "sectors",
        "tessmag_median",
        "contamrt_median",
        "first_tbjd",
        "last_tbjd",
        "baseline_days",
        "search_min_days",
        "search_max_days",
        "best_period_days",
        "best_power",
        "best_fap",
    ):
        row[key] = summary.get(key)
    row["top_peaks"] = "; ".join(
        f"{item['period_days']:.4g}d(power={item['power']:.3g})" for item in summary.get("top_peaks", [])
    )
    row["expected_period_powers"] = "; ".join(
        f"{item['period_days']:.4g}d(power={item['power']:.3g})"
        for item in summary.get("expected_period_powers", [])
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "nets_vi_tglc_rotation")
    parser.add_argument("--size", type=int, default=90)
    parser.add_argument("--limit-mag", type=float, default=16)
    parser.add_argument("--ffi", choices=("SPOC", "TICA"), default="SPOC")
    parser.add_argument("--mast-timeout", type=int, default=3600)
    parser.add_argument("--min-period", type=float, default=1.0)
    parser.add_argument("--max-period", type=float, default=100.0)
    parser.add_argument("--skip-tglc", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--targets", nargs="*", default=[t.name for t in TARGETS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = [t for t in TARGETS if t.name in set(args.targets)]
    if not selected:
        raise SystemExit("No matching targets selected")

    sector_rows = []
    run_rows = []
    summary_rows = []
    full_summary: dict[str, dict] = {}

    for target in selected:
        print(f"\n=== {target.name} / TIC {target.tic} ===", flush=True)
        sectors_df = sector_table_for_tic(target.tic)
        sector_rows.append(
            {
                "target": target.name,
                "tic": target.tic,
                "n_tess_sectors": int(len(sectors_df)),
                "sectors": ",".join(str(int(s)) for s in sectors_df["sector"]),
            }
        )
        print(sectors_df[["sectorName", "sector", "camera", "ccd"]].to_string(index=False), flush=True)

        if not args.skip_tglc:
            run_rows.extend(run_missing_tglc(target, sectors_df["sector"], args.outdir, args))

        target_dir = args.outdir / target_slug(target.name)
        target_summaries: dict[str, dict] = {}
        for flux_col in CHECK_FLUXES:
            files = matching_lc_files(target_dir, target.tic, target.gaia_dr3)
            df = load_flux_series(files, flux_col)
            summary = lomb_scargle_summary(df, args.min_period, args.max_period, target.expected_rotation_days)
            target_summaries[flux_col] = summary
            summary_rows.append(flatten_summary_row(target, flux_col, summary))
            if summary.get("status") == "ok":
                print(
                    f"{flux_col}: {summary['n_sectors']} sectors, {summary['n_points']} points, "
                    f"best P={summary['best_period_days']:.3g} d, FAP={summary['best_fap']:.3g}",
                    flush=True,
                )
            else:
                print(f"{flux_col}: {summary.get('status')}", flush=True)
        plot_path = make_target_plot(target, target_dir, target_summaries, args.outdir)
        print(f"plot: {plot_path}", flush=True)
        full_summary[target.name] = {
            "tic": target.tic,
            "gaia_dr3": target.gaia_dr3,
            "expected_rotation_days": target.expected_rotation_days,
            "notes": target.notes,
            "sectors": sector_rows[-1]["sectors"],
            "fluxes": target_summaries,
            "plot": str(plot_path),
        }

    pd.DataFrame(sector_rows).to_csv(args.outdir / "target_sector_coverage.csv", index=False)
    pd.DataFrame(run_rows).to_csv(args.outdir / "tglc_run_status.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.outdir / "periodogram_summary.csv", index=False)
    with (args.outdir / "periodogram_summary.json").open("w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\nWrote outputs to {args.outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

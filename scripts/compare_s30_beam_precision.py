"""Compare sector 30 default-TICA and BEAM TGLC light-curve precision."""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

SECTOR = 30
CAMERA = 3
DEFAULT_DEFAULT_ROOT = Path("/pdo/users/tehan/beam_tglc/s0030/default_tica")
DEFAULT_BEAM_ROOT = Path("/pdo/users/tehan/beam_tglc/s0030/beam_likelihood")
DEFAULT_OUT_DIR = Path("/pdo/users/tehan/beam_tglc/s0030/diagnostics")

LC_RE = re.compile(
    r"hlsp_tglc_tess_ffi_gaiaid-(?P<gaia>\d+)-s(?P<sector>\d{4})-cam(?P<camera>\d)-ccd(?P<ccd>\d)_"
)

MAG_BINS = [(8.0, 10.0), (10.0, 12.0), (12.0, 14.0), (14.0, 16.0)]


def lc_key(path: Path) -> Optional[Tuple[int, int, int, int]]:
    match = LC_RE.search(path.name)
    if match is None:
        return None
    sector = int(match.group("sector"))
    camera = int(match.group("camera"))
    if sector != SECTOR or camera != CAMERA:
        return None
    return int(match.group("gaia")), sector, camera, int(match.group("ccd"))


def index_lc_files(root: Path, ccds: Iterable[int]) -> Dict[Tuple[int, int, int, int], str]:
    wanted_ccds = set(int(ccd) for ccd in ccds)
    files: Dict[Tuple[int, int, int, int], str] = {}
    for path in sorted((root / "lc").glob("*/*.fits")):
        key = lc_key(path)
        if key is None:
            continue
        if key[3] not in wanted_ccds:
            continue
        files[key] = str(path)
    return files


def robust_ppm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    median = np.nanmedian(finite)
    if not np.isfinite(median) or median == 0:
        return np.nan
    norm = finite / median
    return float(1e6 * 1.4826 * np.nanmedian(np.abs(norm - np.nanmedian(norm))))


def read_lc(path: str) -> Dict[str, object]:
    with fits.open(path, mode="denywrite", memmap=False) as hdul:
        tab = hdul[1].data
        primary = hdul[0].header
        table_header = hdul[1].header
        data = {
            "path": path,
            "time": np.array(tab["time"], dtype=float),
            "cadence": np.array(tab["cadence_num"], dtype=int),
            "cal_psf_flux": np.array(tab["cal_psf_flux"], dtype=float),
            "cal_aper_flux": np.array(tab["cal_aper_flux"], dtype=float),
            "psf_flux": np.array(tab["psf_flux"], dtype=float),
            "aper_flux": np.array(tab["aperture_flux"], dtype=float),
            "tess_flags": np.array(tab["TESS_flags"], dtype=int),
            "tglc_flags": np.array(tab["TGLC_flags"], dtype=int),
            "tess_mag": float(primary.get("TESSMAG", np.nan)),
            "ticid": str(primary.get("TICID", "")),
            "gaia": int(primary.get("GAIADR3", 0)),
            "ccd": int(primary.get("CCD", 0)),
            "contam": float(primary.get("CONTAMRT", np.nan)),
            "cpsf_err": float(table_header.get("CPSF_ERR", np.nan)),
            "cape_err": float(table_header.get("CAPE_ERR", np.nan)),
        }
    good = (data["tess_flags"] == 0) & (data["tglc_flags"] == 0)
    data["good"] = good
    return data


def aligned_good_values(default_lc: Dict[str, object], beam_lc: Dict[str, object],
                        column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    default_cadence = np.asarray(default_lc["cadence"], dtype=int)
    beam_cadence = np.asarray(beam_lc["cadence"], dtype=int)
    common = np.intersect1d(default_cadence, beam_cadence)
    if common.size == 0:
        return common, np.array([], dtype=float), np.array([], dtype=float)
    default_pos = {cadence: i for i, cadence in enumerate(default_cadence)}
    beam_pos = {cadence: i for i, cadence in enumerate(beam_cadence)}
    default_idx = np.array([default_pos[cadence] for cadence in common], dtype=int)
    beam_idx = np.array([beam_pos[cadence] for cadence in common], dtype=int)
    default_good = np.asarray(default_lc["good"])[default_idx]
    beam_good = np.asarray(beam_lc["good"])[beam_idx]
    default_values = np.asarray(default_lc[column], dtype=float)[default_idx]
    beam_values = np.asarray(beam_lc[column], dtype=float)[beam_idx]
    valid = default_good & beam_good & np.isfinite(default_values) & np.isfinite(beam_values)
    return common[valid], default_values[valid], beam_values[valid]


def summarize_pair(key: Tuple[int, int, int, int], default_path: str, beam_path: str) -> Dict[str, object]:
    default_lc = read_lc(default_path)
    beam_lc = read_lc(beam_path)
    common_psf, default_cal_psf, beam_cal_psf = aligned_good_values(default_lc, beam_lc, "cal_psf_flux")
    _, default_cal_aper, beam_cal_aper = aligned_good_values(default_lc, beam_lc, "cal_aper_flux")
    _, default_psf, beam_psf = aligned_good_values(default_lc, beam_lc, "psf_flux")

    ratio = np.nan
    if default_psf.size and np.nanmedian(default_psf) != 0:
        ratio = float(np.nanmedian(beam_psf) / np.nanmedian(default_psf))

    result = {
        "gaia": key[0],
        "sector": key[1],
        "camera": key[2],
        "ccd": key[3],
        "ticid": default_lc["ticid"],
        "tess_mag": float(default_lc["tess_mag"]),
        "contam_default": float(default_lc["contam"]),
        "contam_beam": float(beam_lc["contam"]),
        "n_common_good": int(common_psf.size),
        "default_cal_psf_mad_ppm": robust_ppm(default_cal_psf),
        "beam_cal_psf_mad_ppm": robust_ppm(beam_cal_psf),
        "default_cal_aper_mad_ppm": robust_ppm(default_cal_aper),
        "beam_cal_aper_mad_ppm": robust_ppm(beam_cal_aper),
        "beam_default_psf_median_ratio": ratio,
        "default_path": default_path,
        "beam_path": beam_path,
    }
    default_scatter = result["default_cal_psf_mad_ppm"]
    beam_scatter = result["beam_cal_psf_mad_ppm"]
    if np.isfinite(default_scatter) and default_scatter != 0 and np.isfinite(beam_scatter):
        result["beam_default_cal_psf_scatter_ratio"] = float(beam_scatter / default_scatter)
        result["beam_minus_default_cal_psf_mad_ppm"] = float(beam_scatter - default_scatter)
    else:
        result["beam_default_cal_psf_scatter_ratio"] = np.nan
        result["beam_minus_default_cal_psf_mad_ppm"] = np.nan
    return result


def bin_label(low: float, high: float) -> str:
    return f"{low:.0f}-{high:.0f}"


def magnitude_bin_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for low, high in MAG_BINS:
        selected = [
            row for row in rows
            if np.isfinite(row["tess_mag"]) and low <= float(row["tess_mag"]) < high
        ]
        ratios = np.array([row["beam_default_cal_psf_scatter_ratio"] for row in selected], dtype=float)
        default_scatter = np.array([row["default_cal_psf_mad_ppm"] for row in selected], dtype=float)
        beam_scatter = np.array([row["beam_cal_psf_mad_ppm"] for row in selected], dtype=float)
        summaries.append(
            {
                "tmag_bin": bin_label(low, high),
                "n": len(selected),
                "default_cal_psf_mad_ppm_median": float(np.nanmedian(default_scatter)) if len(selected) else np.nan,
                "beam_cal_psf_mad_ppm_median": float(np.nanmedian(beam_scatter)) if len(selected) else np.nan,
                "beam_default_scatter_ratio_median": float(np.nanmedian(ratios)) if len(selected) else np.nan,
            }
        )
    return summaries


def finite_rows(rows: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def plot_summary(rows: List[Dict[str, object]], out_dir: Path) -> Optional[str]:
    usable = [
        row for row in rows
        if np.isfinite(row["tess_mag"])
        and np.isfinite(row["default_cal_psf_mad_ppm"])
        and np.isfinite(row["beam_cal_psf_mad_ppm"])
    ]
    if not usable:
        return None

    tmag = finite_rows(usable, "tess_mag")
    default_scatter = finite_rows(usable, "default_cal_psf_mad_ppm")
    beam_scatter = finite_rows(usable, "beam_cal_psf_mad_ppm")
    ratio = finite_rows(usable, "beam_default_cal_psf_scatter_ratio")
    ccd = np.array([int(row["ccd"]) for row in usable])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    sc0 = axes[0].scatter(tmag, default_scatter, c=ccd, s=16, alpha=0.7, cmap="tab10", label="default TICA")
    axes[0].scatter(tmag, beam_scatter, c=ccd, s=16, alpha=0.7, cmap="tab10", marker="x", label="BEAM")
    axes[0].set_xlabel("TESS magnitude")
    axes[0].set_ylabel("cal_psf robust scatter (ppm)")
    axes[0].set_title("Precision by magnitude")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc0, ax=axes[0], label="CCD")

    axes[1].axhline(1, color="0.25", lw=1, ls="--")
    axes[1].scatter(tmag, ratio, c=ccd, s=18, alpha=0.75, cmap="tab10")
    axes[1].set_xlabel("TESS magnitude")
    axes[1].set_ylabel("BEAM/default scatter ratio")
    axes[1].set_title("Ratio below 1 favors BEAM")

    diff = beam_scatter - default_scatter
    axes[2].axhline(0, color="0.25", lw=1, ls="--")
    axes[2].scatter(tmag, diff, c=ccd, s=18, alpha=0.75, cmap="tab10")
    axes[2].set_xlabel("TESS magnitude")
    axes[2].set_ylabel("BEAM - default scatter (ppm)")
    axes[2].set_title("Precision delta")

    out_path = out_dir / "s30_beam_default_tica_precision_summary.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return str(out_path)


def plot_representative_lightcurves(rows: List[Dict[str, object]], out_dir: Path, max_plots: int = 6) -> List[str]:
    candidates = [
        row for row in rows
        if row["n_common_good"] > 0
        and np.isfinite(row["beam_default_cal_psf_scatter_ratio"])
    ]
    candidates.sort(key=lambda row: abs(float(row["beam_default_cal_psf_scatter_ratio"]) - 1), reverse=True)
    paths: List[str] = []
    for row in candidates[:max_plots]:
        default_lc = read_lc(row["default_path"])
        beam_lc = read_lc(row["beam_path"])
        common, default_flux, beam_flux = aligned_good_values(default_lc, beam_lc, "cal_psf_flux")
        if common.size == 0:
            continue
        default_pos = {cadence: i for i, cadence in enumerate(np.asarray(default_lc["cadence"], dtype=int))}
        time = np.array([default_lc["time"][default_pos[cadence]] for cadence in common], dtype=float)
        time = time - np.nanmin(time)

        fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        ax.plot(time, default_flux / np.nanmedian(default_flux), ".", ms=2, alpha=0.45, color="0.35", label="default TICA")
        ax.plot(time, beam_flux / np.nanmedian(beam_flux), ".", ms=2, alpha=0.45, color="tab:blue", label="BEAM")
        ax.set_xlabel("time from start (d)")
        ax.set_ylabel("normalized cal_psf_flux")
        ax.set_title(
            f"Gaia DR3 {row['gaia']} ccd{row['ccd']} T={row['tess_mag']:.2f} "
            f"ratio={row['beam_default_cal_psf_scatter_ratio']:.3f}"
        )
        ax.legend(fontsize=8)
        out_path = out_dir / f"s30_beam_default_tica_gaia{row['gaia']}_ccd{row['ccd']}_lc.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        paths.append(str(out_path))
    return paths


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "gaia", "sector", "camera", "ccd", "ticid", "tess_mag", "n_common_good",
        "default_cal_psf_mad_ppm", "beam_cal_psf_mad_ppm",
        "beam_default_cal_psf_scatter_ratio", "beam_minus_default_cal_psf_mad_ppm",
        "default_cal_aper_mad_ppm", "beam_cal_aper_mad_ppm",
        "beam_default_psf_median_ratio", "contam_default", "contam_beam",
        "default_path", "beam_path",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compare(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_files = index_lc_files(args.default_root, args.ccds)
    beam_files = index_lc_files(args.beam_root, args.ccds)
    common_keys = sorted(set(default_files) & set(beam_files))
    missing_default = sorted(set(beam_files) - set(default_files))
    missing_beam = sorted(set(default_files) - set(beam_files))
    if args.max_targets and args.max_targets > 0:
        common_keys = common_keys[:args.max_targets]

    rows = [summarize_pair(key, default_files[key], beam_files[key]) for key in common_keys]
    rows.sort(key=lambda row: (row["ccd"], row["tess_mag"], row["gaia"]))
    mag_bins = magnitude_bin_summary(rows)

    csv_path = args.out_dir / "s30_beam_default_tica_precision.csv"
    json_path = args.out_dir / "s30_beam_default_tica_precision.json"
    write_csv(rows, csv_path)
    summary_plot = plot_summary(rows, args.out_dir)
    lc_plots = plot_representative_lightcurves(rows, args.out_dir, max_plots=args.max_lc_plots)

    payload = {
        "sector": SECTOR,
        "camera": CAMERA,
        "default_root": str(args.default_root),
        "beam_root": str(args.beam_root),
        "n_default_files": len(default_files),
        "n_beam_files": len(beam_files),
        "n_pairs": len(rows),
        "missing_default": [str(key) for key in missing_default[:100]],
        "missing_beam": [str(key) for key in missing_beam[:100]],
        "magnitude_bins": mag_bins,
        "rows": rows,
        "csv": str(csv_path),
        "summary_plot": summary_plot,
        "representative_lc_plots": lc_plots,
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True, sort_keys=True))
    print(json_path, flush=True)
    print(csv_path, flush=True)
    if summary_plot:
        print(summary_plot, flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-root", type=Path, default=DEFAULT_DEFAULT_ROOT)
    parser.add_argument("--beam-root", type=Path, default=DEFAULT_BEAM_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ccds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-targets", type=int, default=0, help="Limit paired targets for smoke checks; 0 means all.")
    parser.add_argument("--max-lc-plots", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(compare(parse_args()))

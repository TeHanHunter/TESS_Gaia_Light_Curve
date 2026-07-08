"""Plot known S30 cam3 TOI signals in paired default-TICA and BEAM light curves."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


DEFAULT_COLOR = "teal"
BEAM_COLOR = "tomato"


def read_selection(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def local_target_dir(lc_root: Path, toi: str) -> Path:
    return lc_root / toi.replace(".", "_")


def find_lc(target_dir: Path, product: str) -> Path:
    matches = sorted(target_dir.glob(f"{product}_*.fits"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {product} FITS in {target_dir}, found {len(matches)}")
    return matches[0]


def read_light_curve(path: Path) -> Dict[str, np.ndarray]:
    with fits.open(path, mode="denywrite", memmap=False) as hdul:
        table = hdul[1].data
        tess_flags = np.asarray(table["TESS_flags"], dtype=int)
        tglc_flags = np.asarray(table["TGLC_flags"], dtype=int)
        flux = np.asarray(table["cal_psf_flux"], dtype=float)
        time = np.asarray(table["time"], dtype=float)
    good = (tess_flags == 0) & (tglc_flags == 0) & np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    time = time[good]
    flux = flux[good]
    median = np.nanmedian(flux)
    if not np.isfinite(median) or median == 0:
        raise ValueError(f"bad median flux for {path}")
    return {"time": time, "ppt": (flux / median - 1.0) * 1e3}


def phase_hours(time_btjd: np.ndarray, epoch_bjd: float, period_days: float) -> np.ndarray:
    time_bjd = time_btjd + 2457000.0
    phase_days = ((time_bjd - epoch_bjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    return phase_days * 24.0


def binned_median(x: np.ndarray, y: np.ndarray, bins: np.ndarray, min_count: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(centers.shape, np.nan, dtype=float)
    idx = np.digitize(x, bins) - 1
    for i in range(centers.size):
        selected = idx == i
        if np.count_nonzero(selected) >= min_count:
            med[i] = np.nanmedian(y[selected])
    return centers, med


def event_times_btjd(time_btjd: np.ndarray, epoch_bjd: float, period_days: float) -> np.ndarray:
    epoch_btjd = epoch_bjd - 2457000.0
    start = float(np.nanmin(time_btjd))
    stop = float(np.nanmax(time_btjd))
    first = int(np.floor((start - epoch_btjd) / period_days)) - 1
    last = int(np.ceil((stop - epoch_btjd) / period_days)) + 1
    events = epoch_btjd + np.arange(first, last + 1) * period_days
    return events[(events >= start) & (events <= stop)]


def zoom_ylim(values: Iterable[np.ndarray]) -> Tuple[float, float]:
    joined = np.concatenate([value[np.isfinite(value)] for value in values])
    if joined.size == 0:
        return -10.0, 10.0
    lo, hi = np.nanpercentile(joined, [1, 99])
    pad = max(1.0, 0.18 * (hi - lo))
    return float(lo - pad), float(hi + pad)


def plot_grid(rows: List[Dict[str, str]], lc_root: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 12.5), sharex=False)
    axes = axes.ravel()

    for ax, row in zip(axes, rows):
        target_dir = local_target_dir(lc_root, row["toi"])
        default = read_light_curve(find_lc(target_dir, "default_tica"))
        beam = read_light_curve(find_lc(target_dir, "beam_likelihood"))

        period = float(row["period"])
        epoch = float(row["epoch_bjd"])
        duration = float(row["duration_hr"])
        default_phase = phase_hours(default["time"], epoch, period)
        beam_phase = phase_hours(beam["time"], epoch, period)
        window = max(4.0, 4.0 * duration)
        selected_default = np.abs(default_phase) <= window
        selected_beam = np.abs(beam_phase) <= window
        bins = np.linspace(-window, window, 54)

        ax.axhline(0, color="0.25", ls=":", lw=1)
        ax.axvspan(-0.5 * duration, 0.5 * duration, color="0.85", zorder=0)
        ax.plot(default_phase[selected_default], default["ppt"][selected_default], ".", c=DEFAULT_COLOR, ms=2.2, alpha=0.28)
        ax.plot(beam_phase[selected_beam], beam["ppt"][selected_beam], ".", c=BEAM_COLOR, ms=2.2, alpha=0.28)

        for phase, ppt, color, label in [
            (default_phase, default["ppt"], DEFAULT_COLOR, "default TICA"),
            (beam_phase, beam["ppt"], BEAM_COLOR, "BEAM"),
        ]:
            selected = np.abs(phase) <= window
            centers, med = binned_median(phase[selected], ppt[selected], bins)
            ax.plot(centers, med, color=color, lw=1.8, label=label)

        ax.set_xlim(-window, window)
        ax.set_ylim(zoom_ylim([default["ppt"][selected_default], beam["ppt"][selected_beam]]))
        label = "stellar FP" if row["class"] == "stellar_fp" else row["disp"]
        ax.set_title(
            f"{row['toi']} ({label})  T={float(row['tess_mag']):.2f}  "
            f"P={period:.4g} d  ratio={float(row['scatter_ratio']):.2f}",
            fontsize=10,
        )

    for ax in axes[len(rows):]:
        ax.axis("off")

    axes[0].legend(loc="best", fontsize=8, markerscale=2)
    fig.supxlabel("Hours from catalog transit center")
    fig.supylabel("cal_psf_flux / median - 1 (ppt)")
    fig.suptitle("Sector 30 cam3 known transit-like signals: default TICA vs BEAM", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_individual(rows: List[Dict[str, str]], lc_root: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for row in rows:
        target_dir = local_target_dir(lc_root, row["toi"])
        default = read_light_curve(find_lc(target_dir, "default_tica"))
        beam = read_light_curve(find_lc(target_dir, "beam_likelihood"))

        period = float(row["period"])
        epoch = float(row["epoch_bjd"])
        duration = float(row["duration_hr"])
        default_phase = phase_hours(default["time"], epoch, period)
        beam_phase = phase_hours(beam["time"], epoch, period)
        window = max(4.0, 4.0 * duration)
        bins = np.linspace(-window, window, 60)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 6.0),
            gridspec_kw=dict(height_ratios=[1.2, 1.8], hspace=0.25),
        )
        label = "stellar FP" if row["class"] == "stellar_fp" else row["disp"]
        fig.suptitle(
            f"{row['toi']} ({label}) TIC {row['ticid']}  "
            f"T={float(row['tess_mag']):.2f}, CCD {row['ccd']}, "
            f"P={period:.6g} d",
            y=0.98,
        )

        for lc, color, name in [(default, DEFAULT_COLOR, "default TICA"), (beam, BEAM_COLOR, "BEAM")]:
            axes[0].plot(lc["time"], lc["ppt"], ".", ms=2.0, alpha=0.28, color=color, label=name)
        for event in event_times_btjd(default["time"], epoch, period):
            axes[0].axvspan(event - duration / 48.0, event + duration / 48.0, color="0.85", zorder=0)
        axes[0].axhline(0, color="0.25", ls=":", lw=1)
        axes[0].set_xlabel("BTJD")
        axes[0].set_ylabel("ppt")
        axes[0].legend(loc="best", fontsize=8, markerscale=2)
        axes[0].set_ylim(zoom_ylim([default["ppt"], beam["ppt"]]))

        selected_default = np.abs(default_phase) <= window
        selected_beam = np.abs(beam_phase) <= window
        axes[1].axhline(0, color="0.25", ls=":", lw=1)
        axes[1].axvspan(-0.5 * duration, 0.5 * duration, color="0.85", zorder=0)
        axes[1].plot(default_phase[selected_default], default["ppt"][selected_default], ".", c=DEFAULT_COLOR, ms=2.5, alpha=0.32)
        axes[1].plot(beam_phase[selected_beam], beam["ppt"][selected_beam], ".", c=BEAM_COLOR, ms=2.5, alpha=0.32)
        for phase, ppt, color, name in [
            (default_phase, default["ppt"], DEFAULT_COLOR, "default TICA median"),
            (beam_phase, beam["ppt"], BEAM_COLOR, "BEAM median"),
        ]:
            selected = np.abs(phase) <= window
            centers, med = binned_median(phase[selected], ppt[selected], bins)
            axes[1].plot(centers, med, color=color, lw=2.0, label=name)
        axes[1].set_xlim(-window, window)
        axes[1].set_ylim(zoom_ylim([default["ppt"][selected_default], beam["ppt"][selected_beam]]))
        axes[1].set_xlabel("Hours from catalog transit center")
        axes[1].set_ylabel("ppt")
        axes[1].legend(loc="best", fontsize=8)

        out_path = out_dir / f"{row['toi'].replace('.', '_')}_default_tica_vs_beam.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--lc-root", type=Path, required=True)
    parser.add_argument("--out-grid", type=Path, required=True)
    parser.add_argument("--individual-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = read_selection(args.selection)
    plot_grid(rows, args.lc_root, args.out_grid)
    print(args.out_grid)
    if args.individual_dir is not None:
        for path in plot_individual(rows, args.lc_root, args.individual_dir):
            print(path)


if __name__ == "__main__":
    main()

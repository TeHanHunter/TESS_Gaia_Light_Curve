"""Plot sector 30 BEAM/default precision with TGLC-style running medians."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


def read_precision_csv(path: Path) -> Dict[str, np.ndarray]:
    rows: Dict[str, List[float]] = {
        "tess_mag": [],
        "default_cal_psf_mad_ppm": [],
        "beam_cal_psf_mad_ppm": [],
        "beam_default_cal_psf_scatter_ratio": [],
    }
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values = {key: float(row[key]) for key in rows}
            except (KeyError, TypeError, ValueError):
                continue
            if all(np.isfinite(value) for value in values.values()):
                for key, value in values.items():
                    rows[key].append(value)
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def running_medians(
    mag: np.ndarray,
    default_ppm: np.ndarray,
    beam_ppm: np.ndarray,
    *,
    mag_min: float,
    mag_max: float,
    step: float,
    half_width: float,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(mag_min, mag_max + 0.5 * step, step)
    default_med = np.full_like(grid, np.nan, dtype=float)
    beam_med = np.full_like(grid, np.nan, dtype=float)
    counts = np.zeros_like(grid, dtype=int)
    for i, center in enumerate(grid):
        selected = np.abs(mag - center) <= half_width
        counts[i] = int(np.count_nonzero(selected))
        if counts[i] < min_count:
            continue
        default_med[i] = np.nanmedian(default_ppm[selected])
        beam_med[i] = np.nanmedian(beam_ppm[selected])
    return grid, default_med, beam_med, counts


def plot_precision(
    data: Dict[str, np.ndarray],
    out_path: Path,
    *,
    mag_min: float,
    mag_max: float,
    max_points: int,
    seed: int,
    window_mag: float,
    step_mag: float,
    min_count: int,
) -> None:
    mag = data["tess_mag"]
    default_ppm = data["default_cal_psf_mad_ppm"]
    beam_ppm = data["beam_cal_psf_mad_ppm"]
    ratio = data["beam_default_cal_psf_scatter_ratio"]

    finite = (
        np.isfinite(mag)
        & np.isfinite(default_ppm)
        & np.isfinite(beam_ppm)
        & np.isfinite(ratio)
        & (default_ppm > 0)
        & (beam_ppm > 0)
        & (mag >= mag_min)
        & (mag <= mag_max)
    )
    mag = mag[finite]
    default_ppm = default_ppm[finite]
    beam_ppm = beam_ppm[finite]
    ratio = ratio[finite]

    grid, default_med, beam_med, counts = running_medians(
        mag,
        default_ppm,
        beam_ppm,
        mag_min=mag_min,
        mag_max=mag_max,
        step=step_mag,
        half_width=0.5 * window_mag,
        min_count=min_count,
    )
    median_ratio = beam_med / default_med
    valid_grid = np.isfinite(default_med) & np.isfinite(beam_med) & (counts >= min_count)

    rng = np.random.default_rng(seed)
    if mag.size > max_points:
        sample = rng.choice(mag.size, size=max_points, replace=False)
    else:
        sample = np.arange(mag.size)

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.6], hspace=0.08),
        figsize=(6.8, 7.2),
    )

    default_color = "teal"
    beam_color = "tomato"
    stroke = [pe.Stroke(linewidth=4.0, foreground="k"), pe.Normal()]

    ax[0].plot(
        mag[sample],
        default_ppm[sample] / 1e6,
        ".",
        c=default_color,
        ms=1.8,
        alpha=0.12,
        label="default TICA",
    )
    ax[0].plot(
        mag[sample],
        beam_ppm[sample] / 1e6,
        ".",
        c=beam_color,
        ms=1.8,
        alpha=0.12,
        label="BEAM",
    )
    ax[0].plot(
        grid[valid_grid],
        default_med[valid_grid] / 1e6,
        c=default_color,
        lw=2.3,
        label="default TICA median",
        path_effects=stroke,
    )
    ax[0].plot(
        grid[valid_grid],
        beam_med[valid_grid] / 1e6,
        c=beam_color,
        lw=2.3,
        label="BEAM median",
        path_effects=stroke,
    )
    ax[0].hlines(y=0.1, xmin=mag_min, xmax=mag_max, colors="k", linestyles="dotted", lw=1)
    ax[0].hlines(y=0.01, xmin=mag_min, xmax=mag_max, colors="k", linestyles="dotted", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1)
    ax[0].set_ylabel("MAD Photometric Precision")
    ax[0].set_title("Sector 30 cam3 full camera")
    leg = ax[0].legend(loc=4, markerscale=4, fontsize=8)
    for handle in leg.legend_handles:
        handle.set_alpha(1)

    ax[1].plot(
        mag[sample],
        ratio[sample],
        ".",
        c="0.5",
        ms=2.0,
        alpha=0.08,
        label="target ratio",
    )
    ax[1].plot(
        grid[valid_grid],
        median_ratio[valid_grid],
        c=beam_color,
        lw=2.4,
        label="BEAM/default median ratio",
        path_effects=stroke,
    )
    ax[1].hlines(y=1, xmin=mag_min, xmax=mag_max, colors="k", linestyles="dotted", lw=1)
    ax[1].set_ylim(0.2, 1.25)
    ax[1].set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
    ax[1].set_xlabel("TESS magnitude")
    ax[1].set_ylabel("Precision Ratio")
    ax[1].legend(loc=1, markerscale=2, fontsize=8)

    ax[1].set_xlim(mag_min, mag_max)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="s30_beam_default_tica_precision.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--mag-min", type=float, default=2.5)
    parser.add_argument("--mag-max", type=float, default=16.1)
    parser.add_argument("--window-mag", type=float, default=0.35)
    parser.add_argument("--step-mag", type=float, default=0.03)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    out_path = args.out or args.csv.with_name(args.csv.stem + "_running_median.png")
    data = read_precision_csv(args.csv)
    plot_precision(
        data,
        out_path,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        max_points=args.max_points,
        seed=args.seed,
        window_mag=args.window_mag,
        step_mag=args.step_mag,
        min_count=args.min_count,
    )
    print(out_path)


if __name__ == "__main__":
    main()

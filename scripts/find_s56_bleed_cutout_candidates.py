"""Find sector 56 cutouts with overexposure/bleed-like mask candidates.

This is a diagnostic helper for the experimental sector 56 ePSF branch. It
loads source pickles from the read-only baseline archive, applies the current
image-only overexposure detector to a bounded cadence median, and writes
side-by-side raw/mask plots for non-empty candidate cutouts.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tglc.effective_psf import build_overexposure_mask

SECTOR = 56
N_CUTS_SIDE = 14
DEFAULT_BASELINE_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056")
DEFAULT_OUT_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")


def source_path(root, cam, ccd, cut_x, cut_y):
    base = Path(root) / "source" / f"{cam}-{ccd}"
    candidates = [
        base / f"source_{cut_x:02d}_{cut_y:02d}.pkl",
        base / f"source_{cut_x}_{cut_y}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_cut_token(token):
    token = str(token)
    if "_" in token:
        cut_x, cut_y = [int(part) for part in token.split("_", 1)]
        return [cut_x * N_CUTS_SIDE + cut_y]
    if ":" in token:
        start_text, end_text = token.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else N_CUTS_SIDE ** 2
        return list(range(start, end))
    return [int(token)]


def parse_cuts(tokens):
    if not tokens:
        return list(range(N_CUTS_SIDE ** 2))
    cuts = []
    for token in tokens:
        cuts.extend(parse_cut_token(token))
    cuts = sorted(set(cuts))
    for cut in cuts:
        if not 0 <= cut < N_CUTS_SIDE ** 2:
            raise ValueError(f"cut {cut} outside 0..{N_CUTS_SIDE ** 2 - 1}")
    return cuts


def limit_source_cadences(source, max_cadences):
    if max_cadences is None:
        return source
    n_cadences = min(int(max_cadences), len(source.time))
    source.time = source.time[:n_cadences]
    source.flux = source.flux[:n_cadences]
    if hasattr(source, "quality"):
        source.quality = source.quality[:n_cadences]
    if hasattr(source, "cadence"):
        source.cadence = source.cadence[:n_cadences]
    return source


def max_run_length(mask, axis):
    scan = np.moveaxis(np.asarray(mask, dtype=bool), axis, 0)
    best = 0
    for index in np.ndindex(scan.shape[1:]):
        line = scan[(slice(None),) + index]
        padded = np.concatenate(([False], line, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        if changes.size:
            best = max(best, int(np.max(changes[1::2] - changes[::2])))
    return best


def summarize_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {
            "mask_fraction": 0.0,
            "mask_pixels": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "max_vertical_run": 0,
            "max_horizontal_run": 0,
        }
    return {
        "mask_fraction": float(np.mean(mask)),
        "mask_pixels": int(xs.size),
        "x_min": int(np.min(xs)),
        "x_max": int(np.max(xs)),
        "y_min": int(np.min(ys)),
        "y_max": int(np.max(ys)),
        "max_vertical_run": max_run_length(mask, axis=0),
        "max_horizontal_run": max_run_length(mask, axis=1),
    }


def plot_candidate(source, image, mask, row, out_path):
    finite = np.isfinite(image)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(image[finite], [1, 99.7])
    else:
        vmin, vmax = 0, 1

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), constrained_layout=True)
    title_id = f"cam{row['cam']} ccd{row['ccd']} cut {row['cut_x']:02d}_{row['cut_y']:02d}"
    cadence_text = f"first {len(source.time)} cadences"

    im0 = axes[0].imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"raw median cutout\n{title_id}, {cadence_text}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="flux")

    axes[1].imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(float) * 0.55
    axes[1].imshow(overlay, origin="lower")
    if np.any(mask):
        axes[1].contour(mask.astype(float), levels=[0.5], colors="yellow", linewidths=0.7, origin="lower")
    axes[1].set_title(f"overexposure mask\nmasked fraction {row['mask_fraction']:.4f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    for ax in axes:
        ax.set_xlim(0, image.shape[1] - 1)
        ax.set_ylim(0, image.shape[0] - 1)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def scan_cut(args, cam, ccd, cut):
    cut_x = cut // N_CUTS_SIDE
    cut_y = cut % N_CUTS_SIDE
    path = source_path(args.baseline_root, cam, ccd, cut_x, cut_y)
    row = {
        "cam": cam,
        "ccd": ccd,
        "cut": cut,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "path": str(path),
        "status": "missing",
    }
    if not path.exists():
        return row

    with path.open("rb") as handle:
        source = pickle.load(handle)
    source = limit_source_cadences(source, args.max_cadences)
    image = np.nanmedian(source.flux, axis=0)
    mask = build_overexposure_mask(
        source,
        seed_sigma=args.seed_sigma,
        grow_sigma=args.grow_sigma,
        dilation=args.dilation,
        max_mask_fraction=args.max_mask_fraction,
        min_bleed_length=args.min_bleed_length,
    )

    row.update(summarize_mask(mask))
    row["status"] = "candidate" if row["mask_pixels"] > 0 else "empty"
    row["median_flux"] = float(np.nanmedian(image))
    row["p997_flux"] = float(np.nanpercentile(image[np.isfinite(image)], 99.7))

    if row["mask_pixels"] > 0:
        out_dir = args.out_root / "diagnostics" / "bleed_candidates" / f"{cam}-{ccd}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"raw_median_vs_overexposure_mask_{cut_x:02d}_{cut_y:02d}_"
            f"sector_{SECTOR}_{cam}-{ccd}_candidate.png"
        )
        plot_candidate(source, image, mask, row, out_path)
        row["plot"] = str(out_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cams", type=int, nargs="+", default=[1])
    parser.add_argument("--ccds", type=int, nargs="+", default=[1])
    parser.add_argument("--cuts", nargs="*", help="Cut indices, ranges like 0:14, or cut ids like 00_00.")
    parser.add_argument("--max-cadences", type=int, default=20)
    parser.add_argument("--seed-sigma", type=float, default=100)
    parser.add_argument("--grow-sigma", type=float, default=20)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--min-bleed-length", type=int, default=12)
    parser.add_argument("--max-mask-fraction", type=float, default=0.2)
    parser.add_argument("--max-plots", type=int, default=6)
    parser.add_argument("--print-empty", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    cuts = parse_cuts(args.cuts)
    rows = []
    for cam in args.cams:
        for ccd in args.ccds:
            for cut in cuts:
                row = scan_cut(args, cam, ccd, cut)
                rows.append(row)
                if args.print_empty or row.get("status") not in ("empty", "missing"):
                    print(json.dumps(row, sort_keys=True), flush=True)
                candidates = [item for item in rows if item.get("status") == "candidate"]
                if args.max_plots and len(candidates) >= args.max_plots:
                    break
            if args.max_plots and len([item for item in rows if item.get("status") == "candidate"]) >= args.max_plots:
                break
        if args.max_plots and len([item for item in rows if item.get("status") == "candidate"]) >= args.max_plots:
            break

    candidates = sorted(
        [row for row in rows if row.get("status") == "candidate"],
        key=lambda row: (row["mask_pixels"], row["max_vertical_run"], row["max_horizontal_run"]),
        reverse=True,
    )
    summary = {
        "n_scanned": len(rows),
        "n_candidates": len(candidates),
        "candidates": candidates[: args.max_plots or None],
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

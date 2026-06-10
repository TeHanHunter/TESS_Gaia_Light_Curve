"""Plot stitched sector 56 CCD image with mask outlines and ePSF scale map."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECTOR = 56
N_CUTS_SIDE = 14
SIZE = 150
STEP = SIZE - 4
CCD_SIZE = 2048
OVER_SIZE = 23
PSF_COLS = OVER_SIZE ** 2
DEFAULT_BASELINE_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056")
DEFAULT_EXPERIMENT_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")


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


def mask_path(root, cam, ccd, cut_x, cut_y):
    return (
        Path(root)
        / "epsf"
        / f"{cam}-{ccd}"
        / f"overexposure_mask_{cut_x:02d}_{cut_y:02d}_sector_{SECTOR}_{cam}-{ccd}_tmag10_unit.npy"
    )


def epsf_path(root, cam, ccd, cut_x, cut_y):
    return (
        Path(root)
        / "epsf"
        / f"{cam}-{ccd}"
        / f"epsf_{cut_x:02d}_{cut_y:02d}_sector_{SECTOR}_{cam}-{ccd}_tmag10_unit.npy"
    )


def diagnostic_json_path(root, cam, ccd):
    return Path(root) / "diagnostics" / f"s56_tmag10_unit_bleedmask_cam{cam}_ccd{ccd}_brightest_corrected.json"


def stitch_image_and_mask(baseline_root, experiment_root, cam, ccd, max_cadences):
    image_sum = np.zeros((CCD_SIZE, CCD_SIZE), dtype=np.float64)
    image_count = np.zeros((CCD_SIZE, CCD_SIZE), dtype=np.uint16)
    mask_canvas = np.zeros((CCD_SIZE, CCD_SIZE), dtype=bool)

    for cut_y in range(N_CUTS_SIDE):
        for cut_x in range(N_CUTS_SIDE):
            path = source_path(baseline_root, cam, ccd, cut_x, cut_y)
            if not path.exists():
                continue
            with path.open("rb") as handle:
                source = pickle.load(handle)

            flux = source.flux[:max_cadences] if max_cadences is not None else source.flux
            cut_image = np.nanmedian(flux, axis=0)
            finite = np.isfinite(cut_image)

            x0 = cut_x * STEP
            y0 = cut_y * STEP
            yslice = slice(y0, y0 + SIZE)
            xslice = slice(x0, x0 + SIZE)
            image_sum[yslice, xslice][finite] += cut_image[finite]
            image_count[yslice, xslice][finite] += 1

            mpath = mask_path(experiment_root, cam, ccd, cut_x, cut_y)
            if mpath.exists():
                mask_canvas[yslice, xslice] |= np.load(mpath).astype(bool)

    image = np.full((CCD_SIZE, CCD_SIZE), np.nan, dtype=np.float32)
    valid = image_count > 0
    image[valid] = (image_sum[valid] / image_count[valid]).astype(np.float32)
    return image, mask_canvas


def build_epsf_shape_canvas(experiment_root, cam, ccd):
    canvas = np.full((N_CUTS_SIDE * OVER_SIZE, N_CUTS_SIDE * OVER_SIZE), np.nan, dtype=np.float32)
    shape_sums = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan, dtype=np.float32)

    for cut_y in range(N_CUTS_SIDE):
        for cut_x in range(N_CUTS_SIDE):
            path = epsf_path(experiment_root, cam, ccd, cut_x, cut_y)
            if not path.exists():
                continue
            epsf = np.load(path, mmap_mode="r")
            median_shape = np.nanmedian(epsf[:, :PSF_COLS], axis=0).reshape(OVER_SIZE, OVER_SIZE)
            shape_sums[cut_y, cut_x] = np.nansum(median_shape)
            y0 = cut_y * OVER_SIZE
            x0 = cut_x * OVER_SIZE
            canvas[y0 : y0 + OVER_SIZE, x0 : x0 + OVER_SIZE] = median_shape.astype(np.float32)
    return canvas, shape_sums


def plot(args):
    image, mask_canvas = stitch_image_and_mask(
        args.baseline_root,
        args.experiment_root,
        args.cam,
        args.ccd,
        args.max_cadences,
    )
    psf_canvas, shape_sums = build_epsf_shape_canvas(args.experiment_root, args.cam, args.ccd)

    with diagnostic_json_path(args.experiment_root, args.cam, args.ccd).open() as handle:
        diag = json.load(handle)
    scale = np.array(diag["scale_median"], dtype=float)
    scale_rel = scale / np.nanmedian(scale)

    finite_image = image[np.isfinite(image)]
    vmin, vmax = np.nanpercentile(finite_image, [args.vmin_percentile, args.vmax_percentile])

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.0), constrained_layout=True)

    im0 = axes[0].imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    if np.any(mask_canvas):
        axes[0].contour(mask_canvas.astype(float), levels=[0.5], colors="tab:red", linewidths=0.35, origin="lower")
    axes[0].set_title(
        f"cam{args.cam} ccd{args.ccd} stitched median image\n"
        f"mask outlines, first {args.max_cadences} cadences"
    )
    axes[0].set_xlabel("CCD x")
    axes[0].set_ylabel("CCD y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="flux")

    finite_psf = psf_canvas[np.isfinite(psf_canvas)]
    psf_vmin, psf_vmax = np.nanpercentile(
        finite_psf,
        [args.psf_vmin_percentile, args.psf_vmax_percentile],
    )
    im1 = axes[1].imshow(psf_canvas, origin="lower", cmap="viridis", vmin=psf_vmin, vmax=psf_vmax)
    for edge in range(OVER_SIZE, N_CUTS_SIDE * OVER_SIZE, OVER_SIZE):
        axes[1].axhline(edge - 0.5, color="white", lw=0.18, alpha=0.55)
        axes[1].axvline(edge - 0.5, color="white", lw=0.18, alpha=0.55)
    axes[1].set_title(
        f"median unit-sum ePSF shapes\n"
        f"shared linear scale, p{args.psf_vmin_percentile:g}-p{args.psf_vmax_percentile:g}"
    )
    axes[1].set_xlabel("cut x")
    axes[1].set_ylabel("cut y")
    centers = np.arange(N_CUTS_SIDE) * OVER_SIZE + (OVER_SIZE - 1) / 2
    axes[1].set_xticks(centers)
    axes[1].set_yticks(centers)
    axes[1].set_xticklabels(range(N_CUTS_SIDE))
    axes[1].set_yticklabels(range(N_CUTS_SIDE))
    axes[1].set_xlim(-0.5, N_CUTS_SIDE * OVER_SIZE - 0.5)
    axes[1].set_ylim(-0.5, N_CUTS_SIDE * OVER_SIZE - 0.5)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="unit-sum ePSF value")

    finite_scale = scale_rel[np.isfinite(scale_rel)]
    delta = max(abs(float(np.nanmin(finite_scale)) - 1.0), abs(float(np.nanmax(finite_scale)) - 1.0), 0.02)
    norm = TwoSlopeNorm(vmin=1.0 - delta, vcenter=1.0, vmax=1.0 + delta)
    im2 = axes[2].imshow(scale_rel, origin="lower", cmap="coolwarm", norm=norm)
    axes[2].set_title(
        f"ePSF Tmag=10 scale\n"
        f"relative to median, {np.nanmin(finite_scale):.2f}-{np.nanmax(finite_scale):.2f}x"
    )
    axes[2].set_xlabel("cut x")
    axes[2].set_ylabel("cut y")
    for y in range(scale_rel.shape[0]):
        for x in range(scale_rel.shape[1]):
            value = scale_rel[y, x]
            if np.isfinite(value):
                axes[2].text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=4.9, color="black")
    axes[2].set_xticks(range(N_CUTS_SIDE))
    axes[2].set_yticks(range(N_CUTS_SIDE))
    axes[2].set_xlim(-0.5, N_CUTS_SIDE - 0.5)
    axes[2].set_ylim(-0.5, N_CUTS_SIDE - 0.5)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="relative scale")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi)
    plt.close(fig)

    print(args.out_path, flush=True)
    print(
        json.dumps(
            {
                "masked_pixels": int(np.sum(mask_canvas)),
                "masked_fraction": float(np.mean(mask_canvas)),
                "psf_canvas_vmin_vmax": [float(psf_vmin), float(psf_vmax)],
                "psf_shape_sum_min_p05_median_p95_max": [
                    float(np.nanmin(shape_sums)),
                    float(np.nanpercentile(shape_sums, 5)),
                    float(np.nanmedian(shape_sums)),
                    float(np.nanpercentile(shape_sums, 95)),
                    float(np.nanmax(shape_sums)),
                ],
                "scale_rel_min_p05_median_p95_max": [
                    float(np.nanmin(scale_rel)),
                    float(np.nanpercentile(scale_rel, 5)),
                    float(np.nanmedian(scale_rel)),
                    float(np.nanpercentile(scale_rel, 95)),
                    float(np.nanmax(scale_rel)),
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--cam", type=int, default=1)
    parser.add_argument("--ccd", type=int, default=1)
    parser.add_argument("--max-cadences", type=int, default=20)
    parser.add_argument("--vmin-percentile", type=float, default=1)
    parser.add_argument("--vmax-percentile", type=float, default=99.7)
    parser.add_argument("--psf-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--psf-vmax-percentile", type=float, default=99.8)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT
        / "diagnostics"
        / "s56_cam1_ccd1_image_mask_outlines_epsf_shapes_and_scale.png",
    )
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()

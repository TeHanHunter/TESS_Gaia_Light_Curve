"""Plot cut-level ePSF shape/scale and LC precision A/B summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SECTOR = 56
OVER_SIZE = 23
PSF_COLS = OVER_SIZE ** 2
DEFAULT_MASK_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")
DEFAULT_NOMASK_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_nomask_singlecut")


def epsf_path(root, cam, ccd, cut):
    return Path(root) / "epsf" / f"{cam}-{ccd}" / f"epsf_{cut}_sector_{SECTOR}_{cam}-{ccd}_tmag10_unit.npy"


def scale_path(root, cam, ccd, cut):
    return Path(root) / "epsf" / f"{cam}-{ccd}" / f"epsf_scale_{cut}_sector_{SECTOR}_{cam}-{ccd}_tmag10_unit.npy"


def summary_path(mask_root, cam, ccd, cut):
    return (
        Path(mask_root)
        / "diagnostics"
        / "lc_ab"
        / f"s56_cam{cam}_ccd{ccd}_cut_{cut}_mask_ab_lc_summary.json"
    )


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def load_psf_products(mask_root, nomask_root, cam, ccd, cut):
    mask_epsf = np.load(epsf_path(mask_root, cam, ccd, cut), mmap_mode="r")
    nomask_epsf = np.load(epsf_path(nomask_root, cam, ccd, cut), mmap_mode="r")
    mask_scale = np.load(scale_path(mask_root, cam, ccd, cut))
    nomask_scale = np.load(scale_path(nomask_root, cam, ccd, cut))

    mask_shape = np.nanmedian(mask_epsf[:, :PSF_COLS], axis=0).reshape(OVER_SIZE, OVER_SIZE)
    nomask_shape = np.nanmedian(nomask_epsf[:, :PSF_COLS], axis=0).reshape(OVER_SIZE, OVER_SIZE)
    shape_diff = mask_shape - nomask_shape

    finite = np.isfinite(mask_shape) & np.isfinite(nomask_shape)
    scale_ratio = mask_scale / nomask_scale
    raw_nomask = np.nanmedian(nomask_epsf[:, :PSF_COLS] * nomask_scale[:, np.newaxis], axis=0)
    raw_mask = np.nanmedian(mask_epsf[:, :PSF_COLS] * mask_scale[:, np.newaxis], axis=0)

    metrics = {
        "unit_sum_nomask": float(np.nansum(nomask_shape)),
        "unit_sum_mask": float(np.nansum(mask_shape)),
        "unit_shape_l1_frac": float(np.nansum(np.abs(shape_diff)) / np.nansum(np.abs(nomask_shape))),
        "unit_shape_l2_frac": float(np.linalg.norm(shape_diff[finite]) / np.linalg.norm(nomask_shape[finite])),
        "unit_shape_cosine": float(
            np.dot(nomask_shape[finite], mask_shape[finite])
            / (np.linalg.norm(nomask_shape[finite]) * np.linalg.norm(mask_shape[finite]))
        ),
        "scale_ratio_p05_med_p95": [float(x) for x in np.nanpercentile(scale_ratio[np.isfinite(scale_ratio)], [5, 50, 95])],
        "raw_sum_ratio": float(np.nansum(raw_mask) / np.nansum(raw_nomask)),
    }
    return nomask_shape, mask_shape, shape_diff, scale_ratio, metrics


def load_lc_summary(path):
    with Path(path).open() as handle:
        payload = json.load(handle)

    rows = []
    for label, item in payload["targets"].items():
        nomask_scatter = finite_float(item["nomask"]["cal_psf_mad_ppm"])
        mask_scatter = finite_float(item["mask"]["cal_psf_mad_ppm"])
        if not np.isfinite(nomask_scatter) or not np.isfinite(mask_scatter):
            continue
        rows.append(
            {
                "label": label,
                "tmag": float(item["target"]["tess_mag"]),
                "distance": float(item["target"]["mask_distance_pix"]),
                "nomask_scatter": nomask_scatter,
                "mask_scatter": mask_scatter,
                "delta_scatter": mask_scatter - nomask_scatter,
                "diff_mad": finite_float(item["mask_minus_nomask"]["mad_ppm"]),
                "diff_median": finite_float(item["mask_minus_nomask"]["median_ppm"]),
            }
        )
    rows.sort(key=lambda row: (row["label"].startswith("C"), row["label"]))
    return rows, payload.get("missing", [])


def plot(args):
    nomask_shape, mask_shape, shape_diff, scale_ratio, metrics = load_psf_products(
        args.mask_root,
        args.nomask_root,
        args.cam,
        args.ccd,
        args.cut,
    )
    lc_rows, missing = load_lc_summary(summary_path(args.mask_root, args.cam, args.ccd, args.cut))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    psf_vmin, psf_vmax = np.nanpercentile(np.r_[nomask_shape.ravel(), mask_shape.ravel()], [0.5, 99.8])
    im0 = axes[0, 0].imshow(nomask_shape, origin="lower", cmap="viridis", vmin=psf_vmin, vmax=psf_vmax)
    axes[0, 0].set_title(f"no-mask median unit ePSF\nsum={metrics['unit_sum_nomask']:.4f}")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(mask_shape, origin="lower", cmap="viridis", vmin=psf_vmin, vmax=psf_vmax)
    axes[0, 1].set_title(f"mask median unit ePSF\nsum={metrics['unit_sum_mask']:.4f}")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    diff_limit = max(abs(float(np.nanpercentile(shape_diff, 1))), abs(float(np.nanpercentile(shape_diff, 99))))
    im2 = axes[0, 2].imshow(
        shape_diff,
        origin="lower",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-diff_limit, vcenter=0, vmax=diff_limit),
    )
    axes[0, 2].set_title(
        "mask - no-mask unit ePSF\n"
        f"L2={100 * metrics['unit_shape_l2_frac']:.2f}%, cos={metrics['unit_shape_cosine']:.5f}"
    )
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    finite_ratio = scale_ratio[np.isfinite(scale_ratio)]
    axes[1, 0].hist(finite_ratio, bins=45, color="0.35")
    p05, p50, p95 = metrics["scale_ratio_p05_med_p95"]
    axes[1, 0].axvline(1, color="black", lw=1, ls="--", alpha=0.65)
    axes[1, 0].axvline(p50, color="tab:red", lw=1.5)
    axes[1, 0].set_title(f"ePSF scale ratio mask/no-mask\np05-med-p95={p05:.4f}, {p50:.4f}, {p95:.4f}")
    axes[1, 0].set_xlabel("scale ratio")
    axes[1, 0].set_ylabel("cadences")

    labels = [row["label"] for row in lc_rows]
    x = np.arange(len(lc_rows))
    nomask = np.array([row["nomask_scatter"] for row in lc_rows])
    mask = np.array([row["mask_scatter"] for row in lc_rows])
    axes[1, 1].plot(x, nomask, "o-", color="0.45", label="no mask")
    axes[1, 1].plot(x, mask, "o-", color="tab:red", label="mask")
    for idx, row in enumerate(lc_rows):
        axes[1, 1].text(idx, max(nomask[idx], mask[idx]) * 1.025, f"T={row['tmag']:.1f}\nd={row['distance']:.0f}", ha="center", va="bottom", fontsize=7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel("cal_psf robust scatter (ppm)")
    axes[1, 1].set_title("Precision for all usable paired LCs")
    axes[1, 1].legend(fontsize=8)

    delta = np.array([row["delta_scatter"] for row in lc_rows])
    colors = ["tab:red" if value > 0 else "tab:blue" for value in delta]
    axes[1, 2].bar(x, delta, color=colors, alpha=0.78)
    axes[1, 2].axhline(0, color="black", lw=0.8)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_ylabel("mask - no-mask scatter (ppm)")
    axes[1, 2].set_title("Precision change")

    for ax in axes[0, :]:
        ax.set_xticks([])
        ax.set_yticks([])

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Sector 56 cam{args.cam} ccd{args.ccd} cut {args.cut}: ePSF and LC A/B", fontsize=14)
    fig.savefig(args.out_path, dpi=args.dpi)
    plt.close(fig)

    result = {"epsf_metrics": metrics, "lc_rows": lc_rows, "missing": missing, "plot": str(args.out_path)}
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=True, sort_keys=True))
    print(args.out_path, flush=True)
    print(args.json_out, flush=True)
    print(json.dumps(result, sort_keys=True, allow_nan=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--nomask-root", type=Path, default=DEFAULT_NOMASK_ROOT)
    parser.add_argument("--cam", type=int, default=1)
    parser.add_argument("--ccd", type=int, default=1)
    parser.add_argument("--cut", default="07_00")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_MASK_ROOT / "diagnostics" / "lc_ab" / "s56_cam1_ccd1_cut_07_00_psf_precision_summary.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_MASK_ROOT / "diagnostics" / "lc_ab" / "s56_cam1_ccd1_cut_07_00_psf_precision_summary.json",
    )
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()

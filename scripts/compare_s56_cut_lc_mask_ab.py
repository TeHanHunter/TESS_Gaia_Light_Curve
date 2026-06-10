"""Run and compare a single-cut LC A/B for the bleed-mask ePSF option."""

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
from astropy.io import fits

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tglc.effective_psf import build_overexposure_mask
from tglc.target_lightcurve import epsf

SECTOR = 56
DEFAULT_BASELINE_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056")
DEFAULT_MASK_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")
DEFAULT_NOMASK_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_nomask_singlecut")


def normalize_schema(source):
    if "designation" in source.gaia.colnames and "DESIGNATION" not in source.gaia.colnames:
        source.gaia.rename_column("designation", "DESIGNATION")
    if hasattr(source.tic, "colnames") and "gaia3" in source.tic.colnames and "dr3_source_id" not in source.tic.colnames:
        source.tic.rename_column("gaia3", "dr3_source_id")
    if not hasattr(source, "transient"):
        source.transient = None
    return source


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


def prepare_root(root, baseline_root):
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("epsf", "lc", "log", "diagnostics"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    source_link = root / "source"
    baseline_source = baseline_root / "source"
    if not source_link.exists() and not source_link.is_symlink():
        source_link.symlink_to(baseline_source, target_is_directory=True)


def gaia_id_from_designation(designation):
    return [int(part) for part in str(designation).split() if part.isdigit()][0]


def lc_path(root, cam, ccd, designation):
    gaia_id = gaia_id_from_designation(designation)
    return (
        Path(root)
        / "lc"
        / f"{cam}-{ccd}"
        / f"hlsp_tglc_tess_ffi_gaiaid-{gaia_id}-s{SECTOR:04d}-cam{cam}-ccd{ccd}_tess_v2.1_llc.fits"
    )


def robust_ppm(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    median = np.nanmedian(finite)
    if not np.isfinite(median) or median == 0:
        return np.nan
    norm = finite / median
    return float(1e6 * 1.4826 * np.nanmedian(np.abs(norm - np.nanmedian(norm))))


def amplitude_ppm(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    median = np.nanmedian(finite)
    if not np.isfinite(median) or median == 0:
        return np.nan
    p05, p95 = np.nanpercentile(finite / median, [5, 95])
    return float(1e6 * (p95 - p05))


def read_lc(path):
    with fits.open(path, mode="denywrite", memmap=False) as hdul:
        tab = hdul[1].data
        header = hdul[0].header
        table_header = hdul[1].header
        data = {
            "time": np.array(tab["time"], dtype=float),
            "cadence": np.array(tab["cadence_num"], dtype=int),
            "psf_flux": np.array(tab["psf_flux"], dtype=float),
            "cal_psf_flux": np.array(tab["cal_psf_flux"], dtype=float),
            "aper_flux": np.array(tab["aperture_flux"], dtype=float),
            "cal_aper_flux": np.array(tab["cal_aper_flux"], dtype=float),
            "tess_flags": np.array(tab["TESS_flags"], dtype=int),
            "tglc_flags": np.array(tab["TGLC_flags"], dtype=int),
            "tess_mag": float(header["TESSMAG"]),
            "contam": float(header["CONTAMRT"]),
            "psf_err": table_header.get("PSF_ERR", np.nan),
            "cpsf_err": table_header.get("CPSF_ERR", np.nan),
        }
    good = (data["tess_flags"] == 0) & (data["tglc_flags"] == 0)
    for key in ("psf_flux", "cal_psf_flux", "aper_flux", "cal_aper_flux"):
        arr = np.array(data[key], copy=True)
        arr[~good] = np.nan
        data[f"{key}_good"] = arr
    return data


def summarize_pair(nomask_lc, mask_lc):
    common = np.intersect1d(nomask_lc["cadence"], mask_lc["cadence"])
    result = {"n_common": int(common.size)}
    for label, lc in (("nomask", nomask_lc), ("mask", mask_lc)):
        result[label] = {
            "cal_psf_mad_ppm": robust_ppm(lc["cal_psf_flux_good"]),
            "cal_psf_p05_p95_amp_ppm": amplitude_ppm(lc["cal_psf_flux_good"]),
            "psf_mad_ppm": robust_ppm(lc["psf_flux_good"]),
            "psf_p05_p95_amp_ppm": amplitude_ppm(lc["psf_flux_good"]),
            "median_psf_flux": float(np.nanmedian(lc["psf_flux_good"])),
            "finite_good": int(np.sum(np.isfinite(lc["cal_psf_flux_good"]))),
            "contam": float(lc["contam"]),
        }
    if common.size:
        nomask_index = {cad: i for i, cad in enumerate(nomask_lc["cadence"])}
        mask_index = {cad: i for i, cad in enumerate(mask_lc["cadence"])}
        a = np.array([nomask_lc["cal_psf_flux_good"][nomask_index[cad]] for cad in common], dtype=float)
        b = np.array([mask_lc["cal_psf_flux_good"][mask_index[cad]] for cad in common], dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        diff = b[valid] - a[valid]
        result["mask_minus_nomask"] = {
            "median_ppm": float(1e6 * np.nanmedian(diff)) if diff.size else np.nan,
            "mad_ppm": float(1e6 * 1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff)))) if diff.size else np.nan,
            "p05_p95_ppm": [float(x) for x in (1e6 * np.nanpercentile(diff, [5, 95]))] if diff.size else [np.nan, np.nan],
        }
    return result


def select_targets(source, mask, max_targets, max_mag, near_radius, control_min_distance):
    x = np.asarray(source.gaia[f"sector_{source.sector}_x"], dtype=float)
    y = np.asarray(source.gaia[f"sector_{source.sector}_y"], dtype=float)
    mag = np.asarray(source.gaia["tess_mag"], dtype=float)
    designation = np.asarray(source.gaia["DESIGNATION"])
    in_cut = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < source.size) & (y >= 0) & (y < source.size)
    bright = np.isfinite(mag) & (mag <= max_mag)
    eligible = in_cut & bright

    if np.any(mask) and ndimage is not None:
        dist_image = ndimage.distance_transform_edt(~mask.astype(bool))
        xi = np.clip(np.rint(x).astype(int), 0, source.size - 1)
        yi = np.clip(np.rint(y).astype(int), 0, source.size - 1)
        distance = dist_image[yi, xi]
    elif np.any(mask):
        yi_mask, xi_mask = np.nonzero(mask)
        distance = np.full(len(x), np.inf)
        for i in np.where(eligible)[0]:
            distance[i] = np.sqrt(np.min((xi_mask - x[i]) ** 2 + (yi_mask - y[i]) ** 2))
    else:
        distance = np.full(len(x), np.inf)

    rows = []
    for i in np.where(eligible)[0]:
        xi = int(np.clip(round(x[i]), 0, source.size - 1))
        yi = int(np.clip(round(y[i]), 0, source.size - 1))
        rows.append(
            {
                "index": int(i),
                "designation": str(designation[i]),
                "label": "",
                "tess_mag": float(mag[i]),
                "x": float(x[i]),
                "y": float(y[i]),
                "mask_distance_pix": float(distance[i]),
                "inside_mask": bool(mask[yi, xi]) if np.any(mask) else False,
            }
        )

    near = [row for row in rows if row["mask_distance_pix"] <= near_radius]
    near.sort(key=lambda row: (row["inside_mask"], row["mask_distance_pix"], row["tess_mag"]))
    if len(near) < max_targets:
        fallback = sorted(rows, key=lambda row: (row["mask_distance_pix"], row["tess_mag"]))
        for row in fallback:
            if row not in near:
                near.append(row)
            if len(near) >= max_targets:
                break

    n_near = min(max_targets, max(1, max_targets - 2))
    selected = near[:n_near]
    controls = [
        row
        for row in sorted(rows, key=lambda row: row["tess_mag"])
        if row["mask_distance_pix"] >= control_min_distance and row not in selected
    ]
    selected.extend(controls[: max_targets - len(selected)])

    near_index = 1
    control_index = 1
    for row in selected:
        if row["mask_distance_pix"] < control_min_distance:
            row["label"] = f"N{near_index}"
            near_index += 1
        else:
            row["label"] = f"C{control_index}"
            control_index += 1
    return selected


def run_target_lcs(source, roots, selected, cut_x, cut_y, overwrite_epsf):
    for mode, root in roots.items():
        overexposure_mask = mode == "mask"
        for target_index, target in enumerate(selected):
            epsf(
                source,
                psf_size=11,
                factor=2,
                cut_x=cut_x,
                cut_y=cut_y,
                sector=SECTOR,
                power=1.4,
                local_directory=f"{root}/",
                limit_mag=0,
                save_aper=False,
                no_progress_bar=True,
                flux_scale="tmag10",
                epsf_normalization="unit_sum",
                overexposure_mask=overexposure_mask,
                overwrite=overwrite_epsf and target_index == 0,
                name=target["designation"],
            )
            print(f"[{mode}] wrote LC for {target['label']} {target['designation']}", flush=True)


def plot_comparison(source, mask, selected, summaries, lc_pairs, output_path, cam, ccd, cut_x, cut_y):
    image = np.nanmedian(source.flux[:20], axis=0)
    finite = np.isfinite(image)
    vmin, vmax = np.nanpercentile(image[finite], [1, 99.7])

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_img = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_amp = fig.add_subplot(grid[1, 0])
    ax_lc = fig.add_subplot(grid[1, 1])

    ax_img.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    if np.any(mask):
        ax_img.contour(mask.astype(float), levels=[0.5], colors="tab:red", linewidths=0.7, origin="lower")
    for row in selected:
        color = "tab:orange" if row["label"].startswith("N") else "tab:cyan"
        ax_img.scatter(row["x"], row["y"], marker="o", s=35, facecolors="none", edgecolors=color, linewidths=1.2)
        ax_img.text(row["x"] + 2, row["y"] + 2, row["label"], color=color, fontsize=8, weight="bold")
    ax_img.set_title(f"cam{cam} ccd{ccd} cut {cut_x:02d}_{cut_y:02d}\nmask outline and selected targets")
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")

    labels = [row["label"] for row in selected]
    xloc = np.arange(len(labels))
    width = 0.38
    nomask_mad = [summaries[row["label"]]["nomask"]["cal_psf_mad_ppm"] for row in selected]
    mask_mad = [summaries[row["label"]]["mask"]["cal_psf_mad_ppm"] for row in selected]
    ax_scatter.bar(xloc - width / 2, nomask_mad, width, label="no mask", color="0.65")
    ax_scatter.bar(xloc + width / 2, mask_mad, width, label="mask", color="tab:red", alpha=0.75)
    ax_scatter.set_xticks(xloc)
    ax_scatter.set_xticklabels(labels)
    ax_scatter.set_ylabel("cal_psf robust scatter (ppm)")
    ax_scatter.set_title("Precision proxy")
    ax_scatter.legend(fontsize=8)

    nomask_amp = [summaries[row["label"]]["nomask"]["cal_psf_p05_p95_amp_ppm"] for row in selected]
    mask_amp = [summaries[row["label"]]["mask"]["cal_psf_p05_p95_amp_ppm"] for row in selected]
    ax_amp.bar(xloc - width / 2, nomask_amp, width, label="no mask", color="0.65")
    ax_amp.bar(xloc + width / 2, mask_amp, width, label="mask", color="tab:red", alpha=0.75)
    ax_amp.set_xticks(xloc)
    ax_amp.set_xticklabels(labels)
    ax_amp.set_ylabel("cal_psf p95-p05 amplitude (ppm)")
    ax_amp.set_title("Amplitude proxy")

    primary = selected[0]
    nomask_lc, mask_lc = lc_pairs[primary["label"]]
    time = nomask_lc["time"] - np.nanmin(nomask_lc["time"])
    a = nomask_lc["cal_psf_flux_good"]
    b = mask_lc["cal_psf_flux_good"]
    ax_lc.plot(time, a / np.nanmedian(a), ".", ms=1.3, alpha=0.35, color="0.45", label="no mask")
    ax_lc.plot(time, b / np.nanmedian(b), ".", ms=1.3, alpha=0.35, color="tab:red", label="mask")
    ax_lc.set_title(f"{primary['label']} normalized cal_psf LC\nT={primary['tess_mag']:.2f}, dist={primary['mask_distance_pix']:.1f}px")
    ax_lc.set_xlabel("time from start (d)")
    ax_lc.set_ylabel("normalized flux")
    ax_lc.legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--nomask-root", type=Path, default=DEFAULT_NOMASK_ROOT)
    parser.add_argument("--cam", type=int, default=1)
    parser.add_argument("--ccd", type=int, default=1)
    parser.add_argument("--cut", default="07_00")
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-target-mag", type=float, default=14.0)
    parser.add_argument("--near-radius", type=float, default=20.0)
    parser.add_argument("--control-min-distance", type=float, default=35.0)
    parser.add_argument("--overwrite-epsf", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MASK_ROOT / "diagnostics" / "lc_ab")
    args = parser.parse_args()

    cut_x, cut_y = [int(part) for part in str(args.cut).split("_", 1)]
    prepare_root(args.nomask_root, args.baseline_root)
    prepare_root(args.mask_root, args.baseline_root)

    with source_path(args.baseline_root, args.cam, args.ccd, cut_x, cut_y).open("rb") as handle:
        source = normalize_schema(pickle.load(handle))

    mask = build_overexposure_mask(source)
    selected = select_targets(
        source,
        mask,
        max_targets=args.max_targets,
        max_mag=args.max_target_mag,
        near_radius=args.near_radius,
        control_min_distance=args.control_min_distance,
    )
    print(json.dumps({"selected": selected}, indent=2), flush=True)

    roots = {"nomask": args.nomask_root, "mask": args.mask_root}
    run_target_lcs(source, roots, selected, cut_x, cut_y, overwrite_epsf=args.overwrite_epsf)

    summaries = {}
    lc_pairs = {}
    available = []
    missing = []
    for row in selected:
        nomask_path = lc_path(args.nomask_root, args.cam, args.ccd, row["designation"])
        mask_path = lc_path(args.mask_root, args.cam, args.ccd, row["designation"])
        if not nomask_path.exists() or not mask_path.exists():
            missing.append(
                {
                    "target": row,
                    "nomask_path": str(nomask_path),
                    "mask_path": str(mask_path),
                    "nomask_exists": nomask_path.exists(),
                    "mask_exists": mask_path.exists(),
                }
            )
            continue
        nomask_lc = read_lc(nomask_path)
        mask_lc = read_lc(mask_path)
        summaries[row["label"]] = summarize_pair(nomask_lc, mask_lc)
        summaries[row["label"]]["target"] = row
        lc_pairs[row["label"]] = (nomask_lc, mask_lc)
        available.append(row)

    if not available:
        raise RuntimeError(f"No matching LC pairs were written. Missing: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"s56_cam{args.cam}_ccd{args.ccd}_cut_{cut_x:02d}_{cut_y:02d}_mask_ab_lc_summary.json"
    summary_path.write_text(
        json.dumps(
            {"cut": f"{cut_x:02d}_{cut_y:02d}", "targets": summaries, "missing": missing},
            indent=2,
            allow_nan=True,
        )
    )

    plot_path = args.out_dir / f"s56_cam{args.cam}_ccd{args.ccd}_cut_{cut_x:02d}_{cut_y:02d}_mask_ab_lc_comparison.png"
    plot_comparison(source, mask, available, summaries, lc_pairs, plot_path, args.cam, args.ccd, cut_x, cut_y)

    print(summary_path, flush=True)
    print(plot_path, flush=True)
    print(json.dumps({"summary": summaries}, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

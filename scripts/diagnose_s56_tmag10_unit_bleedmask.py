"""Diagnostics for the sector 56 Tmag=10/unit-sum/bleed-mask experiment."""

import argparse
import json
import pickle
import sys
import types
from multiprocessing import Pool
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm

SECTOR = 56
N_CUTS_SIDE = 14
OVER_SIZE = 23
PSF_COLS = OVER_SIZE ** 2
DEFAULT_BASELINE_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056")
DEFAULT_EXPERIMENT_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")


def install_pickle_shims():
    tglc = types.ModuleType("tglc")
    ffi = types.ModuleType("tglc.ffi")
    ffi_cut = types.ModuleType("tglc.ffi_cut")
    Source = type("Source", (), {})
    Source.__module__ = "tglc.ffi"
    Source_cut = type("Source_cut", (), {})
    Source_cut.__module__ = "tglc.ffi_cut"
    ffi.Source = Source
    ffi_cut.Source_cut = Source_cut
    sys.modules["tglc"] = tglc
    sys.modules["tglc.ffi"] = ffi
    sys.modules["tglc.ffi_cut"] = ffi_cut

    try:
        import astropy.coordinates.earth as astropy_earth
        from astropy.coordinates import representation as astropy_rep

        for name in (
            "BaseGeodeticRepresentation",
            "WGS84GeodeticRepresentation",
            "WGS72GeodeticRepresentation",
            "GRS80GeodeticRepresentation",
        ):
            if not hasattr(astropy_earth, name) and hasattr(astropy_rep, name):
                setattr(astropy_earth, name, getattr(astropy_rep, name))
    except Exception:
        pass


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


def epsf_path(root, cam, ccd, cut_x, cut_y, suffix):
    base = Path(root) / "epsf" / f"{cam}-{ccd}"
    candidates = [
        base / f"epsf_{cut_x:02d}_{cut_y:02d}_sector_{SECTOR}_{cam}-{ccd}{suffix}.npy",
        base / f"epsf_{cut_x}_{cut_y}_sector_{SECTOR}_{cam}-{ccd}{suffix}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def scale_path(root, cam, ccd, cut_x, cut_y, suffix):
    base = Path(root) / "epsf" / f"{cam}-{ccd}"
    return base / f"epsf_scale_{cut_x:02d}_{cut_y:02d}_sector_{SECTOR}_{cam}-{ccd}{suffix}.npy"


def mask_path(root, cam, ccd, cut_x, cut_y, suffix):
    base = Path(root) / "epsf" / f"{cam}-{ccd}"
    return base / f"overexposure_mask_{cut_x:02d}_{cut_y:02d}_sector_{SECTOR}_{cam}-{ccd}{suffix}.npy"


def read_source_info(args):
    root, cam, ccd, cut_x, cut_y = args
    path = source_path(root, cam, ccd, cut_x, cut_y)
    if not path.exists():
        return {"x": cut_x, "y": cut_y, "missing": True, "path": str(path)}
    with path.open("rb") as handle:
        source = pickle.load(handle)
    tess_mag = np.asarray(source.gaia["tess_mag"], dtype=float)
    if "tess_flux" in source.gaia.colnames:
        tess_flux = np.asarray(source.gaia["tess_flux"], dtype=float)
    else:
        tess_flux = 10 ** (-tess_mag / 2.5)
    brightest = int(np.nanargmin(tess_mag))
    return {
        "x": cut_x,
        "y": cut_y,
        "missing": False,
        "path": str(path),
        "brightest_tmag": float(tess_mag[brightest]),
        "brightest_flux": float(np.nanmax(tess_flux)),
        "n_gaia": int(len(tess_mag)),
    }


def read_source_grid(root, cam, ccd, processes):
    jobs = [(root, cam, ccd, x, y) for y in range(N_CUTS_SIDE) for x in range(N_CUTS_SIDE)]
    if processes > 1:
        with Pool(processes=processes) as pool:
            rows = list(pool.imap_unordered(read_source_info, jobs))
    else:
        rows = [read_source_info(job) for job in jobs]
    brightest_flux = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    brightest_tmag = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    for row in rows:
        if row["missing"]:
            continue
        brightest_flux[row["y"], row["x"]] = row["brightest_flux"]
        brightest_tmag[row["y"], row["x"]] = row["brightest_tmag"]
    return brightest_flux, brightest_tmag, rows


def read_epsf_cut(root, cam, ccd, cut_x, cut_y, suffix):
    path = epsf_path(root, cam, ccd, cut_x, cut_y, suffix)
    if not path.exists():
        return {
            "epsf_sum": np.nan,
            "n_time": 0,
            "nan_cadences": 0,
            "scale_p05": np.nan,
            "scale_median": np.nan,
            "scale_p95": np.nan,
            "unit_sum_median": np.nan,
            "missing": True,
            "path": str(path),
        }
    arr = np.load(path)
    scale_file = scale_path(root, cam, ccd, cut_x, cut_y, suffix)
    if scale_file.exists():
        scale = np.load(scale_file)
        psf_block = arr[:, :PSF_COLS] * scale[:, np.newaxis]
        unit_sums = np.nansum(arr[:, :PSF_COLS], axis=1)
        scale_finite = scale[np.isfinite(scale)]
        if scale_finite.size:
            scale_p05, scale_median, scale_p95 = np.nanpercentile(scale_finite, [5, 50, 95])
        else:
            scale_p05 = scale_median = scale_p95 = np.nan
    else:
        scale = None
        psf_block = arr[:, :PSF_COLS]
        unit_sums = np.full(arr.shape[0], np.nan)
        scale_p05 = scale_median = scale_p95 = np.nan
    med_img = np.nanmedian(psf_block, axis=0)
    nan_cadences = np.isnan(psf_block).all(axis=1)
    return {
        "epsf_sum": float(np.nansum(med_img)),
        "n_time": int(arr.shape[0]),
        "nan_cadences": int(np.sum(nan_cadences)),
        "scale_p05": float(scale_p05),
        "scale_median": float(scale_median),
        "scale_p95": float(scale_p95),
        "unit_sum_median": float(np.nanmedian(unit_sums)),
        "missing": False,
        "path": str(path),
    }


def read_epsf_grid(root, cam, ccd, suffix):
    epsf_sum = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    n_time = np.zeros((N_CUTS_SIDE, N_CUTS_SIDE), dtype=int)
    nan_cadences = np.zeros((N_CUTS_SIDE, N_CUTS_SIDE), dtype=int)
    scale_median = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    scale_p05 = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    scale_p95 = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    unit_sum_median = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    missing = []
    for y in range(N_CUTS_SIDE):
        for x in range(N_CUTS_SIDE):
            row = read_epsf_cut(root, cam, ccd, x, y, suffix)
            if row["missing"]:
                missing.append({"x": x, "y": y, "path": row["path"]})
                continue
            epsf_sum[y, x] = row["epsf_sum"]
            n_time[y, x] = row["n_time"]
            nan_cadences[y, x] = row["nan_cadences"]
            scale_median[y, x] = row["scale_median"]
            scale_p05[y, x] = row["scale_p05"]
            scale_p95[y, x] = row["scale_p95"]
            unit_sum_median[y, x] = row["unit_sum_median"]
    return {
        "epsf_sum": epsf_sum,
        "n_time": n_time,
        "nan_cadences": nan_cadences,
        "scale_median": scale_median,
        "scale_p05": scale_p05,
        "scale_p95": scale_p95,
        "unit_sum_median": unit_sum_median,
        "missing_epsf": missing,
    }


def read_mask_fraction_grid(root, cam, ccd, suffix):
    fractions = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    for y in range(N_CUTS_SIDE):
        for x in range(N_CUTS_SIDE):
            path = mask_path(root, cam, ccd, x, y, suffix)
            if path.exists():
                fractions[y, x] = float(np.mean(np.load(path).astype(bool)))
    return fractions


def finite_summary(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": np.nan, "p05": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    p05, median, p95 = np.nanpercentile(finite, [5, 50, 95])
    return {
        "min": float(np.nanmin(finite)),
        "p05": float(p05),
        "median": float(median),
        "p95": float(p95),
        "max": float(np.nanmax(finite)),
    }


def compute_ccd(root, source_root, label, suffix, cam, ccd, out_dir, processes):
    brightest_flux, brightest_tmag, source_rows = read_source_grid(source_root, cam, ccd, processes)
    epsf_grid = read_epsf_grid(root, cam, ccd, suffix)
    corrected = epsf_grid["epsf_sum"] / brightest_flux
    corrected_rel = corrected / np.nanmedian(corrected)
    mask_fraction = read_mask_fraction_grid(root, cam, ccd, suffix)
    payload = {
        "label": label,
        "root": str(root),
        "source_root": str(source_root),
        "sector": SECTOR,
        "cam": cam,
        "ccd": ccd,
        "suffix": suffix,
        "epsf_sum": epsf_grid["epsf_sum"].tolist(),
        "brightest_flux": brightest_flux.tolist(),
        "brightest_tmag": brightest_tmag.tolist(),
        "corrected_map": corrected.tolist(),
        "corrected_rel_map": corrected_rel.tolist(),
        "mask_fraction": mask_fraction.tolist(),
        "n_time": epsf_grid["n_time"].tolist(),
        "nan_cadences": epsf_grid["nan_cadences"].tolist(),
        "scale_median": epsf_grid["scale_median"].tolist(),
        "scale_p05": epsf_grid["scale_p05"].tolist(),
        "scale_p95": epsf_grid["scale_p95"].tolist(),
        "unit_sum_median": epsf_grid["unit_sum_median"].tolist(),
        "missing_epsf": epsf_grid["missing_epsf"],
        "source_rows": source_rows,
        "summary": {
            "corrected_rel": finite_summary(corrected_rel),
            "mask_fraction": finite_summary(mask_fraction),
            "nan_cadences_total": int(np.nansum(epsf_grid["nan_cadences"])),
            "scale_median": finite_summary(epsf_grid["scale_median"]),
            "unit_sum_median": finite_summary(epsf_grid["unit_sum_median"]),
        },
    }
    out_path = out_dir / f"s56_{label}_cam{cam}_ccd{ccd}_brightest_corrected.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"[{label} cam{cam} ccd{ccd}] {payload['summary']}", flush=True)
    return out_path


def make_nodes(size):
    step = size - 4
    centers = np.arange(N_CUTS_SIDE, dtype=float) * step + (size - 1) / 2
    nodes = np.empty(N_CUTS_SIDE + 1, dtype=float)
    nodes[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    nodes[0] = centers[0] - 0.5 * step
    nodes[-1] = centers[-1] + 0.5 * step
    return nodes


def world_from_wcs(wcs, x_crop, y_crop):
    x_full = np.asarray(x_crop, dtype=float) + 44.0
    y_full = np.asarray(y_crop, dtype=float)
    world = wcs.pixel_to_world(x_full, y_full)
    return np.asarray(world.ra.deg, dtype=float), np.asarray(world.dec.deg, dtype=float)


def extract_wcs_grid(source_root, out_json, size=150):
    ccds = []
    nodes = make_nodes(size)
    for cam in range(1, 5):
        for ccd in range(1, 5):
            path = source_path(source_root, cam, ccd, 0, 0)
            if not path.exists():
                ccds.append({"cam": cam, "ccd": ccd, "missing": True, "path": str(path)})
                continue
            with path.open("rb") as handle:
                source = pickle.load(handle)
            cells = []
            centers_ra = []
            centers_dec = []
            for y in range(N_CUTS_SIDE):
                for x in range(N_CUTS_SIDE):
                    x_nodes = [nodes[x], nodes[x + 1], nodes[x + 1], nodes[x]]
                    y_nodes = [nodes[y], nodes[y], nodes[y + 1], nodes[y + 1]]
                    ra, dec = world_from_wcs(source.wcs, x_nodes, y_nodes)
                    center_ra, center_dec = world_from_wcs(
                        source.wcs,
                        [x * (size - 4) + (size - 1) / 2],
                        [y * (size - 4) + (size - 1) / 2],
                    )
                    centers_ra.append(float(center_ra[0]))
                    centers_dec.append(float(center_dec[0]))
                    cells.append(
                        {
                            "x": x,
                            "y": y,
                            "ra": ra.tolist(),
                            "dec": dec.tolist(),
                            "center_ra": float(center_ra[0]),
                            "center_dec": float(center_dec[0]),
                        }
                    )
            corners_ra, corners_dec = world_from_wcs(
                source.wcs,
                [nodes[0], nodes[-1], nodes[-1], nodes[0]],
                [nodes[0], nodes[0], nodes[-1], nodes[-1]],
            )
            ccds.append(
                {
                    "cam": cam,
                    "ccd": ccd,
                    "missing": False,
                    "path": str(path),
                    "size": size,
                    "corner_ra": corners_ra.tolist(),
                    "corner_dec": corners_dec.tolist(),
                    "center_ra_median": float(np.nanmedian(centers_ra)),
                    "center_dec_median": float(np.nanmedian(centers_dec)),
                    "cells": cells,
                }
            )
    out = {"root": str(source_root), "sector": SECTOR, "size": size, "ccds": ccds}
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


def mean_skycoord(ra_deg, dec_deg):
    coords = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    cart = coords.cartesian
    x = np.nanmean(cart.x.value)
    y = np.nanmean(cart.y.value)
    z = np.nanmean(cart.z.value)
    return SkyCoord(x=x, y=y, z=z, representation_type="cartesian", frame="icrs").spherical


def project_polygon(frame, ra, dec):
    coords = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg, frame="icrs")
    projected = coords.transform_to(frame)
    return np.column_stack([projected.lon.deg, projected.lat.deg])


def load_map(out_dir, label, cam, ccd):
    path = out_dir / f"s56_{label}_cam{cam}_ccd{ccd}_brightest_corrected.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text())
    return path, np.asarray(payload["corrected_rel_map"], dtype=float)


def plot_wcs_maps(out_dir, labels, wcs_json, annotate=True):
    wcs_payload = json.loads(wcs_json.read_text())
    ccds = {(int(entry["cam"]), int(entry["ccd"])): entry for entry in wcs_payload["ccds"]}
    center_ra = []
    center_dec = []
    for entry in ccds.values():
        if not entry.get("missing"):
            center_ra.extend([cell["center_ra"] for cell in entry["cells"]])
            center_dec.extend([cell["center_dec"] for cell in entry["cells"]])
    ref = mean_skycoord(center_ra, center_dec)
    frame = SkyCoord(ra=ref.lon, dec=ref.lat, frame="icrs").skyoffset_frame()

    all_values = []
    panels = []
    for label in labels:
        polygons = []
        values = []
        value_labels = []
        outlines = []
        ccd_labels = []
        for cam in range(1, 5):
            for ccd in range(1, 5):
                wcs_entry = ccds[(cam, ccd)]
                outline = project_polygon(frame, wcs_entry["corner_ra"], wcs_entry["corner_dec"])
                outlines.append(outline)
                path, arr = load_map(out_dir, label, cam, ccd)
                center = project_polygon(
                    frame,
                    [wcs_entry["center_ra_median"]],
                    [wcs_entry["center_dec_median"]],
                )[0]
                if arr is None:
                    ccd_labels.append((center, f"C{cam} CCD{ccd}\nmissing"))
                    continue
                finite = arr[np.isfinite(arr)]
                ccd_labels.append((center, f"C{cam} CCD{ccd}\n{np.nanmin(finite):.2f}-{np.nanmax(finite):.2f}x"))
                cell_by_xy = {(int(cell["x"]), int(cell["y"])): cell for cell in wcs_entry["cells"]}
                for y in range(N_CUTS_SIDE):
                    for x in range(N_CUTS_SIDE):
                        value = arr[y, x]
                        if not np.isfinite(value):
                            continue
                        cell = cell_by_xy[(x, y)]
                        polygons.append(project_polygon(frame, cell["ra"], cell["dec"]))
                        values.append(value)
                        if annotate:
                            label_center = project_polygon(frame, [cell["center_ra"]], [cell["center_dec"]])[0]
                            value_labels.append((label_center, f"{value:.2f}"))
        panels.append((label, polygons, np.asarray(values), value_labels, outlines, ccd_labels))
        all_values.extend(values)

    all_values = np.asarray(all_values, dtype=float)
    delta = max(
        abs(np.nanpercentile(all_values, 1) - 1.0),
        abs(np.nanpercentile(all_values, 99) - 1.0),
        0.03,
    )
    norm = TwoSlopeNorm(vmin=1.0 - delta, vcenter=1.0, vmax=1.0 + delta)

    fig, axes = plt.subplots(1, len(panels), figsize=(7.8 * len(panels), 20.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    collection = None
    for ax, (label, polygons, values, value_labels, outlines, ccd_labels) in zip(axes, panels):
        collection = PolyCollection(polygons, array=values, cmap="coolwarm", norm=norm, edgecolors="none")
        ax.add_collection(collection)
        for outline in outlines:
            closed = np.vstack([outline, outline[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="black", lw=0.65, alpha=0.85)
        for center, ccd_label in ccd_labels:
            ax.text(
                center[0],
                center[1],
                ccd_label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            )
        if annotate:
            for center, value_label in value_labels:
                ax.text(center[0], center[1], value_label, ha="center", va="center", fontsize=2.6, color="black")
        all_xy = np.concatenate(polygons + outlines)
        pad_x = 0.025 * (np.nanmax(all_xy[:, 0]) - np.nanmin(all_xy[:, 0]))
        pad_y = 0.025 * (np.nanmax(all_xy[:, 1]) - np.nanmin(all_xy[:, 1]))
        ax.set_xlim(np.nanmin(all_xy[:, 0]) - pad_x, np.nanmax(all_xy[:, 0]) + pad_x)
        ax.set_ylim(np.nanmin(all_xy[:, 1]) - pad_y, np.nanmax(all_xy[:, 1]) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_xaxis()
        ax.set_title(f"{label}\nraw sum / brightest flux relative to median")
        ax.set_xlabel("Delta RA cos Dec (deg, east left)")
        ax.set_ylabel("Delta Dec")
    cbar = fig.colorbar(collection, ax=axes, fraction=0.045, pad=0.025)
    cbar.set_label("relative")
    suffix = "_annotated" if annotate else ""
    png = out_dir / f"s56_baseline_vs_tmag10_unit_bleedmask_wcs_linear{suffix}.png"
    fig.savefig(png, dpi=260)
    plt.close(fig)
    print(png, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT / "diagnostics")
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--ccds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--wcs-json", type=Path)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--plot-wcs", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    install_pickle_shims()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for cam in args.cams:
        for ccd in args.ccds:
            if not args.skip_baseline:
                compute_ccd(
                    args.baseline_root,
                    args.baseline_root,
                    "baseline",
                    "",
                    cam,
                    ccd,
                    args.out_dir,
                    args.processes,
                )
            if not args.skip_experiment:
                compute_ccd(
                    args.experiment_root,
                    args.baseline_root,
                    "tmag10_unit_bleedmask",
                    "_tmag10_unit",
                    cam,
                    ccd,
                    args.out_dir,
                    args.processes,
                )
    if not args.skip_baseline:
        labels.append("baseline")
    if not args.skip_experiment:
        labels.append("tmag10_unit_bleedmask")

    if args.plot_wcs:
        wcs_json = args.wcs_json or (args.out_dir / "s56_sector56_wcs_grid.json")
        if not wcs_json.exists():
            extract_wcs_grid(args.baseline_root, wcs_json)
        plot_wcs_maps(args.out_dir, labels, wcs_json, annotate=not args.no_annotate)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare legacy and production A2v1 Sector 56 ePSF integrals.

The production ePSFs use stellar catalog weights relative to the brightest
Gaia-predicted TESS flux in each cut.  This diagnostic therefore divides the
integral of the median ePSF shape by that brightest-star flux before making
spatial maps.  It reads the compact Gaia CCD catalogs and a TICA WCS rather
than unpickling the multi-terabyte source tree.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


SECTOR = 56
ORBITS = (119, 120)
N_CUTS_SIDE = 14
CUT_SIZE = 150
CUT_STEP = CUT_SIZE - 4
PSF_COLS = 23**2
DEFAULT_A2V1_ROOT = Path("/pdo/users/tehan/tglc-gpu-production-A2v1")
DEFAULT_LEGACY_ROOT = Path("/pdo/users/tehan/tglc-gpu-production")


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: np.nan for key in ("min", "p05", "median", "p95", "max")}
    p05, median, p95 = np.nanpercentile(finite, [5, 50, 95])
    return {
        "min": float(np.nanmin(finite)),
        "p05": float(p05),
        "median": float(median),
        "p95": float(p95),
        "max": float(np.nanmax(finite)),
    }


def catalog_path(root: Path, orbit: int, cam: int, ccd: int) -> Path:
    return root / f"orbit-{orbit}" / "ffi" / "catalogs" / f"Gaia_cam{cam}_ccd{ccd}.ecsv"


def ffi_directory(root: Path, orbit: int, cam: int, ccd: int) -> Path:
    return root / f"orbit-{orbit}" / "ffi" / f"cam{cam}" / f"ccd{ccd}" / "ffi"


def epsf_path(root: Path, orbit: int, cam: int, ccd: int, x: int, y: int) -> Path:
    return (
        root
        / f"orbit-{orbit}"
        / "ffi"
        / f"cam{cam}"
        / f"ccd{ccd}"
        / "epsf"
        / f"epsf_{x}_{y}.npy"
    )


def tess_magnitudes(gaia: QTable) -> np.ndarray:
    g = np.ma.asarray(gaia["phot_g_mean_mag"], dtype=float).filled(np.nan)
    bp = np.ma.asarray(gaia["phot_bp_mean_mag"], dtype=float).filled(np.nan)
    rp = np.ma.asarray(gaia["phot_rp_mean_mag"], dtype=float).filled(np.nan)
    color = bp - rp
    tmag = g - 0.00522555 * color**3 + 0.0891337 * color**2 - 0.633923 * color + 0.0324473
    tmag[~np.isfinite(tmag)] = g[~np.isfinite(tmag)] - 0.430
    return tmag


def first_tica_wcs(root: Path, orbit: int, cam: int, ccd: int) -> tuple[WCS, str]:
    paths = sorted(ffi_directory(root, orbit, cam, ccd).glob("*img.fits"))
    if not paths:
        raise FileNotFoundError(f"No TICA FFIs found for cam{cam}/ccd{ccd}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        header = fits.getheader(paths[0], 0)
    return WCS(header), str(paths[0])


def brightest_flux_job(args: tuple[str, int, int, int]) -> dict:
    legacy_root_string, orbit, cam, ccd = args
    legacy_root = Path(legacy_root_string)
    gaia_file = catalog_path(legacy_root, orbit, cam, ccd)
    gaia = QTable.read(gaia_file)
    wcs, ffi_file = first_tica_wcs(legacy_root, orbit, cam, ccd)
    sky = SkyCoord(ra=gaia["ra"], dec=gaia["dec"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaia_x, gaia_y = wcs.world_to_pixel(sky)
    g = np.ma.asarray(gaia["phot_g_mean_mag"], dtype=float).filled(np.nan)
    tmag = tess_magnitudes(gaia)
    valid_catalog = np.isfinite(g) & (g < 25) & np.isfinite(tmag)

    brightest_flux = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
    brightest_tmag = np.full_like(brightest_flux, np.nan)
    n_gaia = np.zeros_like(brightest_flux, dtype=int)
    for y in range(N_CUTS_SIDE):
        y0 = y * CUT_STEP
        in_y = (gaia_y >= y0) & (gaia_y <= y0 + CUT_SIZE)
        for x in range(N_CUTS_SIDE):
            x0 = 44 + x * CUT_STEP
            selected = valid_catalog & in_y & (gaia_x >= x0) & (gaia_x <= x0 + CUT_SIZE)
            n_gaia[y, x] = int(np.count_nonzero(selected))
            if n_gaia[y, x] == 0:
                continue
            cut_tmag = tmag[selected]
            brightest_tmag[y, x] = float(np.nanmin(cut_tmag))
            brightest_flux[y, x] = float(10 ** (-brightest_tmag[y, x] / 2.5))
    return {
        "cam": cam,
        "ccd": ccd,
        "catalog": str(gaia_file),
        "ffi_wcs": ffi_file,
        "brightest_flux": brightest_flux.tolist(),
        "brightest_tmag": brightest_tmag.tolist(),
        "n_gaia": n_gaia.tolist(),
    }


def load_psf_block(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] < PSF_COLS:
        raise ValueError(f"Unexpected ePSF shape {array.shape} in {path}")
    return array[:, :PSF_COLS]


def integral_of_median_shape(*blocks: np.ndarray) -> float:
    if len(blocks) == 1:
        block = blocks[0]
    else:
        block = np.concatenate([np.asarray(item) for item in blocks], axis=0)
    return float(np.nansum(np.nanmedian(block, axis=0)))


def median_cadence_integral(*blocks: np.ndarray) -> float:
    values = [np.nansum(item, axis=1) for item in blocks]
    return float(np.nanmedian(np.concatenate(values)))


def epsf_integral_job(args: tuple[str, str, int, int, int, int]) -> dict:
    a2v1_root_string, legacy_root_string, cam, ccd, x, y = args
    roots = {"a2v1": Path(a2v1_root_string), "legacy": Path(legacy_root_string)}
    paths = {
        label: {orbit: epsf_path(root, orbit, cam, ccd, x, y) for orbit in ORBITS}
        for label, root in roots.items()
    }
    missing = [str(path) for group in paths.values() for path in group.values() if not path.exists()]
    if missing:
        return {"cam": cam, "ccd": ccd, "x": x, "y": y, "missing": missing}

    legacy_blocks = {orbit: load_psf_block(paths["legacy"][orbit]) for orbit in ORBITS}
    a2v1_blocks = {}
    for orbit in ORBITS:
        if os.path.samefile(paths["a2v1"][orbit], paths["legacy"][orbit]):
            a2v1_blocks[orbit] = legacy_blocks[orbit]
        else:
            a2v1_blocks[orbit] = load_psf_block(paths["a2v1"][orbit])

    row = {
        "cam": cam,
        "ccd": ccd,
        "x": x,
        "y": y,
        "missing": [],
        # Orbit 120 had no pre-existing A2v1 ePSFs, so regular files are the
        # exact nonempty-mask refits and symlinks are exact empty-mask reuse.
        "masked_refit": not paths["a2v1"][120].is_symlink(),
        "n_cadences": {
            label: {str(orbit): int(blocks[orbit].shape[0]) for orbit in ORBITS}
            for label, blocks in (("legacy", legacy_blocks), ("a2v1", a2v1_blocks))
        },
    }
    for label, blocks in (("legacy", legacy_blocks), ("a2v1", a2v1_blocks)):
        row[label] = {
            "orbit119": integral_of_median_shape(blocks[119]),
            "orbit120": integral_of_median_shape(blocks[120]),
            "combined": integral_of_median_shape(blocks[119], blocks[120]),
            "combined_median_cadence_integral": median_cadence_integral(blocks[119], blocks[120]),
        }
    return row


def compute_payload(a2v1_root: Path, legacy_root: Path, processes: int) -> dict:
    ccd_jobs = [(str(legacy_root), 119, cam, ccd) for cam in range(1, 5) for ccd in range(1, 5)]
    print("Reconstructing brightest-star factors from Gaia catalogs and TICA WCS", flush=True)
    with ProcessPoolExecutor(max_workers=processes) as executor:
        source_rows = list(executor.map(brightest_flux_job, ccd_jobs))
    source_by_ccd = {(row["cam"], row["ccd"]): row for row in source_rows}

    cut_jobs = [
        (str(a2v1_root), str(legacy_root), cam, ccd, x, y)
        for cam in range(1, 5)
        for ccd in range(1, 5)
        for y in range(N_CUTS_SIDE)
        for x in range(N_CUTS_SIDE)
    ]
    print(f"Reading {len(cut_jobs)} legacy/A2v1 cut pairs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for index, row in enumerate(executor.map(epsf_integral_job, cut_jobs), start=1):
            rows.append(row)
            if index % 196 == 0:
                print(f"  completed {index}/{len(cut_jobs)} cuts", flush=True)

    ccds = []
    for cam in range(1, 5):
        for ccd in range(1, 5):
            source = source_by_ccd[(cam, ccd)]
            brightest_flux = np.asarray(source["brightest_flux"], dtype=float)
            entry = {
                "cam": cam,
                "ccd": ccd,
                "catalog": source["catalog"],
                "ffi_wcs": source["ffi_wcs"],
                "brightest_flux": source["brightest_flux"],
                "brightest_tmag": source["brightest_tmag"],
                "n_gaia": source["n_gaia"],
            }
            ccd_rows = [row for row in rows if row["cam"] == cam and row["ccd"] == ccd]
            missing = [item for row in ccd_rows for item in row["missing"]]
            entry["missing_epsfs"] = missing
            masked = np.zeros((N_CUTS_SIDE, N_CUTS_SIDE), dtype=bool)
            for label in ("legacy", "a2v1"):
                for time_key in ("orbit119", "orbit120", "combined"):
                    integral = np.full((N_CUTS_SIDE, N_CUTS_SIDE), np.nan)
                    median_cadence = np.full_like(integral, np.nan)
                    for row in ccd_rows:
                        y, x = row["y"], row["x"]
                        if not row["missing"]:
                            integral[y, x] = row[label][time_key]
                            masked[y, x] = row["masked_refit"]
                            if time_key == "combined":
                                median_cadence[y, x] = row[label]["combined_median_cadence_integral"]
                    corrected = integral / brightest_flux
                    relative = corrected / np.nanmedian(corrected)
                    entry[f"{label}_{time_key}_integral"] = integral.tolist()
                    entry[f"{label}_{time_key}_corrected"] = corrected.tolist()
                    entry[f"{label}_{time_key}_relative"] = relative.tolist()
                    if time_key == "combined":
                        entry[f"{label}_combined_median_cadence_integral"] = median_cadence.tolist()
            entry["masked_refit"] = masked.tolist()
            legacy = np.asarray(entry["legacy_combined_integral"], dtype=float)
            a2v1 = np.asarray(entry["a2v1_combined_integral"], dtype=float)
            delta = 100 * (a2v1 / legacy - 1)
            entry["a2v1_vs_legacy_percent"] = delta.tolist()
            entry["summary"] = {
                "legacy_relative": finite_summary(entry["legacy_combined_relative"]),
                "a2v1_relative": finite_summary(entry["a2v1_combined_relative"]),
                "a2v1_vs_legacy_percent": finite_summary(delta),
                "masked_refit_count": int(np.count_nonzero(masked)),
                "missing_epsf_count": len(missing),
            }
            ccds.append(entry)

    all_delta = np.concatenate(
        [np.asarray(entry["a2v1_vs_legacy_percent"], dtype=float).ravel() for entry in ccds]
    )
    all_masked = np.concatenate(
        [np.asarray(entry["masked_refit"], dtype=bool).ravel() for entry in ccds]
    )
    payload = {
        "schema": "s56_a2v1_epsf_integral_diagnostic_v1",
        "sector": SECTOR,
        "orbits": list(ORBITS),
        "a2v1_root": str(a2v1_root),
        "legacy_root": str(legacy_root),
        "quantity": "integral of median ePSF shape / brightest Gaia-predicted TESS flux",
        "normalization": "each CCD divided by its own median for relative maps",
        "mask_authority": "orbit-120 A2v1 ePSF symlink state: regular=nonempty-mask refit, symlink=empty-mask legacy reuse",
        "interpretation_note": (
            "The frozen legacy and A2v1 fit paths both pass source.mask.mask to fit_epsf. "
            "A2v1 regular files record recomputation, but do not by themselves establish a new "
            "pixel-mask algorithm relative to this legacy root."
        ),
        "ccds": ccds,
        "summary": {
            "masked_refit_count": int(np.count_nonzero(all_masked)),
            "total_cut_count": int(all_masked.size),
            "masked_refit_fraction": float(np.mean(all_masked)),
            "a2v1_vs_legacy_percent_all": finite_summary(all_delta),
            "a2v1_vs_legacy_percent_masked": finite_summary(all_delta[all_masked]),
            "a2v1_vs_legacy_percent_empty_mask": finite_summary(all_delta[~all_masked]),
        },
    }
    return payload


def mean_skycoord(ra_deg: list[float], dec_deg: list[float]):
    coords = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg)
    xyz = coords.cartesian.xyz.value.mean(axis=1)
    return SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], representation_type="cartesian").spherical


def project_polygon(frame, ra, dec) -> np.ndarray:
    coords = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
    projected = coords.transform_to(frame)
    return np.column_stack([projected.lon.deg, projected.lat.deg])


def collect_panel(payload: dict, wcs_payload: dict, field: str):
    data_by_ccd = {(int(item["cam"]), int(item["ccd"])): item for item in payload["ccds"]}
    wcs_by_ccd = {(int(item["cam"]), int(item["ccd"])): item for item in wcs_payload["ccds"]}
    polygons, values, labels, masked_edges, outlines, ccd_labels = [], [], [], [], [], []
    for cam in range(1, 5):
        for ccd in range(1, 5):
            entry = data_by_ccd[(cam, ccd)]
            wcs_entry = wcs_by_ccd[(cam, ccd)]
            array = np.asarray(entry[field], dtype=float)
            masked = np.asarray(entry["masked_refit"], dtype=bool)
            cell_by_xy = {(int(cell["x"]), int(cell["y"])): cell for cell in wcs_entry["cells"]}
            outlines.append((wcs_entry["corner_ra"], wcs_entry["corner_dec"]))
            ccd_labels.append(
                (
                    wcs_entry["center_ra_median"],
                    wcs_entry["center_dec_median"],
                    f"C{cam} CCD{ccd}",
                )
            )
            for y in range(N_CUTS_SIDE):
                for x in range(N_CUTS_SIDE):
                    value = array[y, x]
                    if not np.isfinite(value):
                        continue
                    cell = cell_by_xy[(x, y)]
                    polygons.append((cell["ra"], cell["dec"]))
                    values.append(value)
                    labels.append((cell["center_ra"], cell["center_dec"], value))
                    masked_edges.append(bool(masked[y, x]))
    return polygons, np.asarray(values), labels, masked_edges, outlines, ccd_labels


def draw_wcs_figure(
    payload: dict,
    wcs_payload: dict,
    fields: list[str],
    titles: list[str],
    output_stem: Path,
    annotate: bool,
):
    center_ra = [cell["center_ra"] for entry in wcs_payload["ccds"] for cell in entry["cells"]]
    center_dec = [cell["center_dec"] for entry in wcs_payload["ccds"] for cell in entry["cells"]]
    ref = mean_skycoord(center_ra, center_dec)
    frame = SkyCoord(ra=ref.lon, dec=ref.lat).skyoffset_frame()
    panels = [collect_panel(payload, wcs_payload, field) for field in fields]

    relative_panels = [panel[1] for field, panel in zip(fields, panels) if "percent" not in field]
    if relative_panels:
        relative_values = np.concatenate(relative_panels)
        relative_delta = max(
            abs(np.nanpercentile(relative_values, 1) - 1),
            abs(np.nanpercentile(relative_values, 99) - 1),
            0.03,
        )
        relative_norm = TwoSlopeNorm(vmin=1 - relative_delta, vcenter=1, vmax=1 + relative_delta)
    else:
        relative_norm = None
    percent_panels = [panel[1] for field, panel in zip(fields, panels) if "percent" in field]
    if percent_panels:
        percent_values = np.concatenate(percent_panels)
        percent_delta = max(
            abs(np.nanpercentile(percent_values, 1)),
            abs(np.nanpercentile(percent_values, 99)),
            0.5,
        )
        percent_norm = TwoSlopeNorm(vmin=-percent_delta, vcenter=0, vmax=percent_delta)
    else:
        percent_norm = None

    fig, axes = plt.subplots(1, len(fields), figsize=(6.6 * len(fields), 17), constrained_layout=True)
    axes = np.atleast_1d(axes)
    collections = []
    for ax, field, title, panel in zip(axes, fields, titles, panels):
        polygons, values, labels, masked_edges, outlines, ccd_labels = panel
        projected = [project_polygon(frame, ra, dec) for ra, dec in polygons]
        is_percent = "percent" in field
        norm = percent_norm if is_percent else relative_norm
        cmap = "RdBu_r" if is_percent else "coolwarm"
        edgecolors = ["#b2182b" if flag else "none" for flag in masked_edges]
        linewidths = [0.32 if flag else 0 for flag in masked_edges]
        collection = PolyCollection(
            projected,
            array=values,
            cmap=cmap,
            norm=norm,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
        ax.add_collection(collection)
        collections.append(collection)
        projected_outlines = []
        for ra, dec in outlines:
            outline = project_polygon(frame, ra, dec)
            projected_outlines.append(outline)
            closed = np.vstack([outline, outline[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="black", lw=0.65)
        for ra, dec, label in ccd_labels:
            center = project_polygon(frame, [ra], [dec])[0]
            ax.text(
                center[0],
                center[1],
                label,
                ha="center",
                va="center",
                fontsize=6.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.8},
            )
        if annotate:
            for ra, dec, value in labels:
                center = project_polygon(frame, [ra], [dec])[0]
                text = f"{value:+.1f}" if is_percent else f"{value:.2f}"
                ax.text(center[0], center[1], text, ha="center", va="center", fontsize=2.15)
        all_xy = np.concatenate(projected + projected_outlines)
        xpad = 0.025 * np.ptp(all_xy[:, 0])
        ypad = 0.025 * np.ptp(all_xy[:, 1])
        ax.set_xlim(np.nanmin(all_xy[:, 0]) - xpad, np.nanmax(all_xy[:, 0]) + xpad)
        ax.set_ylim(np.nanmin(all_xy[:, 1]) - ypad, np.nanmax(all_xy[:, 1]) + ypad)
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Delta RA cos Dec (deg, east left)")
        ax.set_ylabel("Delta Dec (deg)")
        colorbar = fig.colorbar(collection, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label("A2v1 - legacy (%)" if is_percent else "relative to CCD median")
        ax.legend(
            handles=[Line2D([0], [0], color="#b2182b", lw=1.4, label="A2v1 regular refit file")],
            loc="lower right",
            framealpha=0.9,
            fontsize=7,
        )
    fig.suptitle(
        "Sector 56 ePSF integral / brightest Gaia-predicted TESS flux\n"
        "median ePSF shape across both orbits; linear color scales",
        fontsize=13,
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=260)
    plt.close(fig)


def plot_payload(payload_path: Path, wcs_path: Path, output_dir: Path, annotate: bool):
    payload = json.loads(payload_path.read_text())
    wcs_payload = json.loads(wcs_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_wcs_figure(
        payload,
        wcs_payload,
        ["a2v1_combined_relative"],
        ["S56 A2v1 production ePSF normalization"],
        output_dir / "s56_A2v1_wcs_epsf_integral_brightest_corrected_linear_annotated",
        annotate,
    )
    draw_wcs_figure(
        payload,
        wcs_payload,
        ["legacy_combined_relative", "a2v1_combined_relative", "a2v1_vs_legacy_percent"],
        ["Legacy production", "A2v1 production", "Direct integral change"],
        output_dir / "s56_legacy_vs_A2v1_wcs_epsf_integral_comparison",
        annotate,
    )
    draw_wcs_figure(
        payload,
        wcs_payload,
        ["a2v1_orbit119_relative", "a2v1_orbit120_relative"],
        ["A2v1 orbit 119", "A2v1 orbit 120"],
        output_dir / "s56_A2v1_orbit119_vs_orbit120_wcs_epsf_integral",
        annotate,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2v1-root", type=Path, default=DEFAULT_A2V1_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--wcs-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-annotate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.plot_only:
        payload = compute_payload(args.a2v1_root, args.legacy_root, args.processes)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=True))
        print(json.dumps(payload["summary"], indent=2), flush=True)
    if args.wcs_json is not None:
        if args.output_dir is None:
            raise ValueError("--output-dir is required with --wcs-json")
        plot_payload(args.output_json, args.wcs_json, args.output_dir, not args.no_annotate)


if __name__ == "__main__":
    main()

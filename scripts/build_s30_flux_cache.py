"""Build per-cut flux cube caches for the sector-30 cam3 BEAM/TICA run."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from astropy.io import fits
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_s30_beam_tica import (
    ACTIVE_SIZE,
    CAMERA,
    DEFAULT_BEAM_ROOT,
    DEFAULT_SIZE,
    DEFAULT_SPOC_ROOT,
    DEFAULT_TICA_ROOT,
    N_CUTS_SIDE,
    SECTOR,
    TICA_ACTIVE_COLUMN_OFFSET,
    build_spoc_index,
    candidate_cadences_for_metadata,
    cut_origin,
    cut_to_xy,
    flux_cache_paths,
    index_beam_files,
    index_tica_files,
    parse_cuts,
    read_tica_midtime,
    select_common_cadences,
    split_beam_mosaic_to_ccds,
)

DEFAULT_FULL_ROOT = Path("/pdo/users/tehan/beam_tglc/s0030/full")
DEFAULT_CACHE_ROOT = DEFAULT_FULL_ROOT / "flux_cache"


def cache_log(cache_root: Path, product: str, ccd: int) -> Path:
    path = cache_root / product / f"ccd{ccd}" / "cache_build.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_log(path: Path, message: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z ") + message
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def open_cut_memmaps(cache_root: Path, product: str, ccd: int, cuts: Sequence[int],
                     n_cadences: int, size: int, overwrite: bool) -> Dict[int, np.memmap]:
    maps: Dict[int, np.memmap] = {}
    for cut in cuts:
        cut_x, cut_y = cut_to_xy(cut, size=size)
        flux_path, _ = flux_cache_paths(cache_root, product, ccd, cut_x, cut_y)
        assert flux_path is not None
        flux_path.parent.mkdir(parents=True, exist_ok=True)
        if flux_path.exists() and not overwrite:
            maps[cut] = np.load(flux_path, mmap_mode="r+")
        else:
            maps[cut] = open_memmap(flux_path, mode="w+", dtype=np.float32, shape=(n_cadences, size, size))
    return maps


def write_metadata(cache_root: Path, product: str, ccd: int, cadences: Sequence[int],
                   time_values: np.ndarray, quality: np.ndarray, exposure_s: np.ndarray,
                   input_units: str) -> Path:
    _, metadata_path = flux_cache_paths(cache_root, product, ccd, 0, 0)
    assert metadata_path is not None
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        metadata_path,
        cadence=np.array(cadences, dtype=np.int32),
        time=np.array(time_values, dtype=float),
        quality=np.array(quality, dtype=np.int16),
        exposure_s=np.array(exposure_s, dtype=float),
        input_units=np.array(input_units),
    )
    return metadata_path


def read_default_active(path: str) -> np.ndarray:
    with fits.open(path, mode="denywrite", memmap=True) as hdul:
        data = hdul[0].data
        return np.asarray(data[:ACTIVE_SIZE, TICA_ACTIVE_COLUMN_OFFSET:TICA_ACTIVE_COLUMN_OFFSET + ACTIVE_SIZE],
                          dtype=np.float32)


def read_beam_active(path: str, ccd: int) -> np.ndarray:
    with fits.open(path, mode="denywrite", memmap=True) as hdul:
        ccds = split_beam_mosaic_to_ccds(hdul[0].data)
        return np.asarray(ccds[ccd], dtype=np.float32)


def build_cache_for_ccd(args, ccd: int) -> int:
    product = args.product
    log_path = cache_log(args.cache_root, product, ccd)
    manifest_path = args.cache_root / product / f"ccd{ccd}" / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        write_log(log_path, f"skip product={product} ccd={ccd} manifest={manifest_path}")
        return 0
    cuts = parse_cuts(args.cuts)
    tica_index = index_tica_files(args.tica_root, ccd)
    beam_index = index_beam_files(args.beam_root) if product == "beam_likelihood" else None
    required = candidate_cadences_for_metadata(
        product,
        tica_index,
        beam_index,
        max_cadences=args.max_cadences if args.max_cadences > 0 else None,
    )
    spoc_index = build_spoc_index(
        args.spoc_root,
        ccd,
        cache_dir=args.metadata_cache_root / product / "log",
        refresh_cache=args.refresh_metadata_cache,
        required_cadences=required,
    )
    cadences = select_common_cadences(
        product,
        tica_index,
        spoc_index,
        beam_index,
        max_cadences=args.max_cadences if args.max_cadences > 0 else None,
    )
    write_log(log_path, f"start product={product} ccd={ccd} cuts={len(cuts)} cadences={len(cadences)}")

    maps = open_cut_memmaps(args.cache_root, product, ccd, cuts, len(cadences), args.size, args.overwrite)
    time_values = np.empty(len(cadences), dtype=float)
    quality = np.empty(len(cadences), dtype=np.int16)
    exposure_s = np.empty(len(cadences), dtype=float)

    try:
        for i, cadence in enumerate(cadences):
            meta = spoc_index[cadence]
            if product == "default_tica":
                active = read_default_active(tica_index[cadence])
            else:
                if beam_index is None:
                    raise ValueError("beam_index is required for beam_likelihood")
                active = read_beam_active(beam_index[cadence], ccd=ccd)

            scale = meta.exposure_s if args.input_units == "e_per_cadence" else 1.0
            for cut in cuts:
                cut_x, cut_y = cut_to_xy(cut, size=args.size)
                x0, y0 = cut_origin(cut_x, cut_y, size=args.size)
                maps[cut][i] = active[y0:y0 + args.size, x0:x0 + args.size] / scale

            time_values[i] = read_tica_midtime(tica_index[cadence], fallback=meta.midpoint)
            quality[i] = meta.quality
            exposure_s[i] = meta.exposure_s
            if (i + 1) % args.log_every == 0 or i == len(cadences) - 1:
                write_log(log_path, f"cadence {i + 1}/{len(cadences)} ffiindex={cadence}")
    finally:
        for mmap in maps.values():
            mmap.flush()

    metadata_path = write_metadata(args.cache_root, product, ccd, cadences, time_values, quality, exposure_s, args.input_units)
    manifest = {
        "product": product,
        "sector": SECTOR,
        "camera": CAMERA,
        "ccd": ccd,
        "cuts": [f"{cut_to_xy(cut)[0]:02d}_{cut_to_xy(cut)[1]:02d}" for cut in cuts],
        "n_cadences": len(cadences),
        "cadence_min": int(min(cadences)),
        "cadence_max": int(max(cadences)),
        "input_units": args.input_units,
        "metadata": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_log(log_path, f"done product={product} ccd={ccd} manifest={manifest_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["default_tica", "beam_likelihood"], required=True)
    parser.add_argument("--ccds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cuts", nargs="+", default=["all"])
    parser.add_argument("--max-cadences", type=int, default=0)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--metadata-cache-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--tica-root", type=Path, default=DEFAULT_TICA_ROOT)
    parser.add_argument("--beam-root", type=Path, default=DEFAULT_BEAM_ROOT)
    parser.add_argument("--spoc-root", type=Path, default=DEFAULT_SPOC_ROOT)
    parser.add_argument("--input-units", choices=["e_per_cadence", "e_per_s"], default="e_per_cadence")
    parser.add_argument("--refresh-metadata-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for ccd in args.ccds:
        build_cache_for_ccd(args, ccd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

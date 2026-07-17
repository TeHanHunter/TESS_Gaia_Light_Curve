"""
Extract light curves from FFI cutouts using best-fit ePSFs.

Assumes `tglc cutouts` and `tglc epsfs` have already been run.
"""

import argparse
from functools import partial
import logging
from pathlib import Path
import pickle

import numpy as np
from astropy.table import Table

from tglc.ffi import Source
from tglc.light_curve import generate_light_curves
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger()


def _source_tic_overlay_path(source_file: Path) -> Path:
    return source_file.parent.parent / "source_tic" / f"{source_file.stem}.ecsv"


def _apply_source_tic_overlay(source: Source, source_file: Path) -> None:
    overlay_path = _source_tic_overlay_path(source_file)
    if not overlay_path.is_file():
        return
    source.tic = Table.read(overlay_path)
    logger.debug("Using source TIC overlay %s with %d rows", overlay_path, len(source.tic))


def read_source_and_epsf_and_save_light_curves(
    source_and_epsf_files: tuple[Path, Path],
    manifest: Manifest,
    replace: bool,
    psf_size: int,
    oversample_factor: int,
    tic_ids: list[int] | None = None,
):
    """
    Read a pickled `Source` object and a numpy-saved ePSF, and extract and save light curves.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.
    """
    source_file, epsf_file = source_and_epsf_files
    with source_file.open("rb") as source_pickle:
        source: Source = pickle.load(source_pickle)
    _apply_source_tic_overlay(source, source_file)
    epsf = np.load(epsf_file)
    for light_curve in generate_light_curves(source, epsf, psf_size, oversample_factor, tic_ids):
        manifest.tic_id = light_curve.meta["tic_id"]
        if replace or not manifest.light_curve_file.is_file():
            light_curve.write_hdf5(manifest.light_curve_file)
        else:
            logger.debug(
                f"Light curve file {manifest.light_curve_file.resolve()} exists and will not be"
                " overwritten"
            )


def make_light_curves_main(args: argparse.Namespace):
    """
    Extract light curves from FFI cutouts using best-fit ePSFs.

    Assumes `tglc cutouts` and `tglc epsfs` have already been run.
    """
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        ccd_source_files = list(manifest.source_directory.iterdir())
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue

        ccd_source_and_epsf_files = []
        for source_file in ccd_source_files:
            epsf_file = (
                manifest.epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.npy"
            )
            if epsf_file.is_file():
                ccd_source_and_epsf_files.append((source_file, epsf_file))
            else:
                logger.warning(f"ePSF for source file {source_file.resolve()} not found, skipping")
        if len(ccd_source_and_epsf_files) == 0:
            logger.warning(f"No ePSF files found for camera {camera} CCD {ccd}, skipping")
            continue

        manifest.light_curve_directory.mkdir(exist_ok=True)

        if args.tic is not None:
            logger.info(
                "Light curves for the ONLY the following TIC IDs will be produced: "
                + ", ".join(map(str, args.tic))
            )

        save_light_curves_with_argparse_args = partial(
            read_source_and_epsf_and_save_light_curves,
            manifest=manifest,
            replace=args.replace,
            psf_size=args.psf_size,
            oversample_factor=args.oversample,
            tic_ids=args.tic,
        )
        consume_iterator_with_progress_bar(
            pool_map_if_multiprocessing(
                save_light_curves_with_argparse_args,
                ccd_source_and_epsf_files,
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
            ),
            desc=f"Extracting light curves for {camera}-{ccd}",
            unit="cutout",
            total=len(ccd_source_and_epsf_files),
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )

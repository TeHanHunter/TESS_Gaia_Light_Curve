"""
Fit and save ePSFs for FFI cutouts.

Assumes `tglc cutouts` has already been run.
"""

import argparse
from functools import partial
import logging
import multiprocessing
from pathlib import Path
import pickle
import re

import numpy as np

from tglc.epsf import (
    build_overexposure_mask,
    fit_epsf,
    get_star_flux_ratios,
    make_tglc_design_matrix,
    normalize_epsf_unit_sum,
)
from tglc.ffi import Source
from tglc.utils._optional_deps import HAS_CUPY
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


def get_epsf_scale_file(epsf_output_file: Path) -> Path:
    """Return the sidecar scale path for a saved unit-sum ePSF file."""
    suffix = epsf_output_file.stem.removeprefix("epsf")
    return epsf_output_file.with_name(f"epsf_scale{suffix}.npy")


def fit_epsf_for_source(
    source: Source,
    psf_size: int,
    oversample_factor: int,
    edge_compression_factor: float,
    flux_uncertainty_power: float,
    flux_scale: str = "relative",
    overexposure_mask: bool = False,
    overexposure_seed_sigma: float = 100,
    overexposure_grow_sigma: float = 20,
    overexposure_dilation: int = 2,
    overexposure_max_mask_fraction: float = 0.2,
    overexposure_min_bleed_length: int = 12,
    use_gpu: bool = True,
):
    """
    Fit an ePSF for each cadence in a `Source` object.

    Parameters
    ----------
    source : Source
        FFI cutout `Source` object with observed flux, star positions, and star brightnesses.
    psf_size : int
        Side length of ePSF in pixels.
    oversample_factor : int
        Factor by which to oversample the ePSF compared to image pixels.
    flux_uncertainty_power : float
        Power of pixel value used as observational uncertainty in ePSF fit. <1 emphasizes
        contributions from dimmer stars, 1 means all contributions are equal.
    flux_scale : str
        Catalog flux scale for the ePSF design matrix. ``relative`` is the historical per-cut
        brightest-star normalization; ``absolute`` is T=10 -> 15000 e-/s; ``tmag10`` uses a
        unitless T=10 reference.
    overexposure_mask : bool
        If `True`, build a static overexposure/bleed-like pixel mask and exclude those image rows
        from the ePSF fit.
    use_gpu : bool
        If `True`, use `cupy` to run the ePSF parameter fit on the GPU. Requires `cupy` to be
        installed and at least one CUDA device to be available.

    Returns
    -------
    epsf : array
        2D array where first dimension corresponds to cadences in `source` and second dimension
        contains the best-fit ePSF parameters per cadence.
    """
    logger.debug(
        f"Fitting ePSF for source in {source.camera}-{source.ccd} at {source.ccd_x}, {source.ccd_y}"
    )
    star_positions = np.array(
        [source.gaia[f"sector_{source.sector}_x"], source.gaia[f"sector_{source.sector}_y"]]
    ).T
    star_flux_ratios = get_star_flux_ratios(source.gaia, flux_scale=flux_scale)
    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        source.flux.shape[1:],
        (psf_size, psf_size),
        oversample_factor,
        star_positions,
        star_flux_ratios,
        source.mask.data,
        edge_compression_factor,
    )
    flux = source.flux
    # Mask out saturated pixels as a base
    base_flux_mask = source.mask.mask
    extra_flux_mask = None
    if overexposure_mask:
        extra_flux_mask = build_overexposure_mask(
            source,
            seed_sigma=overexposure_seed_sigma,
            grow_sigma=overexposure_grow_sigma,
            dilation=overexposure_dilation,
            max_mask_fraction=overexposure_max_mask_fraction,
            min_bleed_length=overexposure_min_bleed_length,
        )

    if use_gpu and HAS_CUPY:
        import cupy as cp

        design_matrix = cp.asarray(design_matrix)
        flux = cp.asarray(flux)
        base_flux_mask = cp.asarray(base_flux_mask)
        if extra_flux_mask is not None:
            extra_flux_mask = cp.asarray(extra_flux_mask)
        xp = cp
    else:
        xp = np

    e_psf = xp.zeros((flux.shape[0], design_matrix.shape[1]))
    # JIT-ing this loop using numba did not give much performance benefit. Maybe vectorizing would?
    for i in range(flux.shape[0]):
        try:
            # fit_epsf will automatically use the appropriate lstsq method.
            e_psf[i] = fit_epsf(
                design_matrix,
                flux[i],
                base_flux_mask,
                flux_uncertainty_power,
                regularization_extension_size,
                extra_flux_mask=extra_flux_mask,
            )
        except np.linalg.LinAlgError as e:
            logger.warning(f"Error while fitting ePSF: {e}")
            e_psf[i] = np.nan
    if xp != np:
        e_psf = e_psf.get()
    return e_psf


def read_source_and_fit_and_save_epsf(
    source_and_epsf_files: tuple[Path, Path],
    replace: bool,
    psf_size: int,
    oversample_factor: int,
    edge_compression_factor: float,
    flux_uncertainty_power: float,
    flux_scale: str = "relative",
    epsf_normalization: str = "none",
    overexposure_mask: bool = False,
    overexposure_seed_sigma: float = 100,
    overexposure_grow_sigma: float = 20,
    overexposure_dilation: int = 2,
    overexposure_max_mask_fraction: float = 0.2,
    overexposure_min_bleed_length: int = 12,
    use_gpu: bool = True,
):
    """
    Read a pickled `Source` object, fit an ePSF for each of its cadences, and save the results.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.

    Most arguments are passed to `fit_epsf_for_source`.
    """
    source_file, epsf_output_file = source_and_epsf_files
    scale_output_file = get_epsf_scale_file(epsf_output_file)
    if (
        not replace
        and epsf_output_file.is_file()
        and (epsf_normalization != "unit_sum" or scale_output_file.is_file())
    ):
        logger.debug(f"ePSF file {epsf_output_file.resolve()} exists and will not be overwritten")
        return
    with source_file.open("rb") as source_pickle:
        source: Source = pickle.load(source_pickle)

    process_name = multiprocessing.current_process().name
    pool_worker_name_match = re.match(r".*PoolWorker-(\d+)", process_name)
    if pool_worker_name_match:
        pool_worker_id = int(pool_worker_name_match[1])
    else:
        pool_worker_id = -1

    if use_gpu and HAS_CUPY:
        # Figure out which GPU to use, making sure they're evenly disributed
        import cupy

        if pool_worker_id > 0:
            cuda_device = (pool_worker_id - 1) % cupy.cuda.runtime.getDeviceCount()
            logger.debug(f"Pool worker {pool_worker_id} using GPU {cuda_device}")
        else:
            cuda_device = 0
            logger.debug(f"Non-pool process {process_name} using GPU 0")
        cuda_device_context = cupy.cuda.Device(cuda_device)
    else:
        from contextlib import nullcontext

        cuda_device = None
        cuda_device_context = nullcontext()

        if pool_worker_id > 0:
            logger.debug(f"Pool worker {pool_worker_id} using CPU")
        else:
            logger.debug(f"Non-pool process {process_name} using CPU")

    with cuda_device_context:
        epsf = fit_epsf_for_source(
            source,
            psf_size,
            oversample_factor,
            edge_compression_factor,
            flux_uncertainty_power,
            flux_scale=flux_scale,
            overexposure_mask=overexposure_mask,
            overexposure_seed_sigma=overexposure_seed_sigma,
            overexposure_grow_sigma=overexposure_grow_sigma,
            overexposure_dilation=overexposure_dilation,
            overexposure_max_mask_fraction=overexposure_max_mask_fraction,
            overexposure_min_bleed_length=overexposure_min_bleed_length,
            use_gpu=use_gpu,
        )
    if epsf_normalization == "unit_sum":
        over_size = psf_size * oversample_factor + 1
        epsf_unit, epsf_scale = normalize_epsf_unit_sum(epsf, over_size)
        np.save(epsf_output_file, epsf_unit)
        np.save(scale_output_file, epsf_scale)
    elif epsf_normalization == "none":
        np.save(epsf_output_file, epsf)
    else:
        raise ValueError("epsf_normalization must be 'none' or 'unit_sum'.")


def make_epsfs_main(args: argparse.Namespace):
    """
    Fit and save ePSFs for FFI cutouts.

    Assumes `tglc cutouts` has already been run.
    """
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        ccd_source_files = list(manifest.source_directory.iterdir())
        if args.cutout is not None:
            # Filter `ccd_source_files` by cutouts specified by user
            args_cutout_source_files = []
            # The `Manifest` class doesn't support temporary parameters, so there's no good way to
            # make this a list comprehension, which it should be.
            for cutout_x, cutout_y in args.cutout:
                manifest.cutout_x = cutout_x
                manifest.cutout_y = cutout_y
                args_cutout_source_files.append(manifest.source_file.resolve())
            ccd_source_files = [
                file for file in ccd_source_files if file.resolve() in args_cutout_source_files
            ]
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue

        manifest.epsf_directory.mkdir(exist_ok=True)
        ccd_epsf_files = [
            manifest.epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.npy"
            for source_file in ccd_source_files
        ]

        fit_and_save_epsf_with_argparse_args = partial(
            read_source_and_fit_and_save_epsf,
            replace=args.replace,
            psf_size=args.psf_size,
            oversample_factor=args.oversample,
            edge_compression_factor=args.edge_compression_factor,
            flux_uncertainty_power=args.uncertainty_power,
            flux_scale=args.flux_scale,
            epsf_normalization=args.epsf_normalization,
            overexposure_mask=args.overexposure_mask,
            overexposure_seed_sigma=args.overexposure_seed_sigma,
            overexposure_grow_sigma=args.overexposure_grow_sigma,
            overexposure_dilation=args.overexposure_dilation,
            overexposure_max_mask_fraction=args.overexposure_max_mask_fraction,
            overexposure_min_bleed_length=args.overexposure_min_bleed_length,
            use_gpu=not args.no_gpu,
        )
        # For GPU multiprocessing, the "spawn" start method is necessary
        # TODO logging from workers is ignored with the "spawn" method
        mp_start_method = "spawn" if not args.no_gpu else None
        consume_iterator_with_progress_bar(
            pool_map_if_multiprocessing(
                fit_and_save_epsf_with_argparse_args,
                zip(ccd_source_files, ccd_epsf_files, strict=True),
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
                mp_start_method=mp_start_method,
            ),
            desc=f"Fitting ePSFs for {camera}-{ccd}",
            unit="cutout",
            total=len(ccd_source_files),
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )

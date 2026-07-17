"""ePSF helper functions."""

from math import ceil, floor
import warnings

from numba import jit
import numpy as np

from tglc.utils._optional_deps import HAS_CUPY

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - scipy is part of the PDO runtime
    ndimage = None


@jit
def get_xy_coordinates_centered_at_zero(shape: tuple[int, int]):
    """
    Returns coordinates for an array with the given shape with (0, 0) at the center of the array.

    Returns
    -------
    x, y : tuple[array, array]
        X and Y coordinates.
    """
    x_coordinates = np.arange(shape[1]) - (shape[1] - 1) / 2
    y_coordinates = np.arange(shape[0]) - (shape[0] - 1) / 2
    return np.repeat(x_coordinates, shape[0]).reshape(shape[::-1]).T, np.repeat(
        y_coordinates, shape[1]
    ).reshape(shape)


@jit
def make_tglc_design_matrix(
    image_shape: tuple[int, int],
    psf_shape_pixels: tuple[int, int],
    oversample_factor: int,
    star_positions: np.ndarray,
    star_flux_ratios: np.ndarray,
    background_strap_mask: np.ndarray | None = None,
    edge_compression_scale_factor: float | None = None,
):
    """
    Construct the TGLC design matrix from equation (3) of Han & Brandt, 2023.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of image (FFI cutout) that will be used as observed data.
    psf_shape_pixels : tuple[int, int]
        Extent of ePSF array in pixels.
    oversample_factor : int
        Factor by which to oversample the ePSF compared to image pixels.
    star_positions : array
        Positions of stars in image with shape (n, 2). The first column is `x` and the second column
        is `y`. Same order as `star_flux_ratios`.
    star_flux_ratios : array
        Ratio of flux from each star to maximum flux from any star, where flux is calculated using
        catalog brightness for each star. Shape (n,) and same order as `star_positions`.
    background_strap_mask : array | None
        Mask giving the background strap values for each pixel. If omitted or set to `None`, no
        columns for background modeling are added to the design matrix.
    edge_compression_scale_factor : float | None
        Scale factor used when forcing edges of ePSF to 0. This is only needed during fitting (not
        forward modeling) and produces extra rows in the output. If omitted or set to `None`, no
        extra rows are added to the design matrix. If included, `background_strap_mask` must also be
        given.

    Returns
    -------
    design_matrix, regularization_extension_size : tuple[array, int]
        Design matrix and amount that observed vectors need to be extended by for regularization
        during fitting. If `edge_compression_scale_factor` is `None`, then
        `regularization_extension_size` will be `0`.
    """
    oversampled_psf_shape = (
        psf_shape_pixels[0] * oversample_factor + 1,
        psf_shape_pixels[1] * oversample_factor + 1,
    )
    # epsf_contributions_to_pixels[iy, ix, py, px] is the contribution of point (px, py) in the
    # oversampled PSF to pixel (ix, iy) in the image.
    epsf_contributions_to_pixels = np.zeros(
        (image_shape[0], image_shape[1], oversampled_psf_shape[0], oversampled_psf_shape[1])
    )
    pixels_in_epsf_x = (
        np.arange(psf_shape_pixels[1], dtype=np.int64) - (psf_shape_pixels[1] - 1) // 2
    )
    pixels_in_epsf_y = (
        np.arange(psf_shape_pixels[0], dtype=np.int64) - (psf_shape_pixels[0] - 1) // 2
    )
    for (x, y), flux_ratio in zip(star_positions, star_flux_ratios):  # noqa: B905 (for JIT)
        nearest_pixel_x, nearest_pixel_y = (round(x), round(y))
        for pixel_x in pixels_in_epsf_x + nearest_pixel_x:
            if pixel_x < 0 or pixel_x >= image_shape[1]:
                continue
            for pixel_y in pixels_in_epsf_y + nearest_pixel_y:
                if pixel_y < 0 or pixel_y >= image_shape[0]:
                    continue
                # Get the coordinate of the nearest pixel center in coordinates of the PSF grid,
                # with the bottom left PSF point at (0, 0) and distance 1 between adjacent PSF
                # points.
                pixel_psf_x, pixel_psf_y = (
                    (pixel_x - x) * oversample_factor + oversampled_psf_shape[1] // 2,
                    (pixel_y - y) * oversample_factor + oversampled_psf_shape[0] // 2,
                )
                # The four closest PSF points are bilinearly interpolated to give the PSF model
                # value of the pixel, and their coordinates are given by rounding the pixel center
                # coordinates up and down. The contribution from each pixel is the weight it is
                # given in the bilinear interpolation, which is the product of the distances in the
                # x and y directions. We further weight the contribution in importance by the flux
                # ratio of the current star.
                for psf_x, psf_y in [
                    (floor(pixel_psf_x), floor(pixel_psf_y)),
                    (floor(pixel_psf_x), ceil(pixel_psf_y)),
                    (ceil(pixel_psf_x), floor(pixel_psf_y)),
                    (ceil(pixel_psf_x), ceil(pixel_psf_y)),
                ]:
                    # Naively, the interpolation weight is:
                    #   np.abs(pixel_psf_x - psf_x) * np.abs(pixel_psf_y - psf_y)
                    # If the pixel lies on a PSF pixel boundary, one of these terms will vanish. But
                    # that actually means we are only interpolating between two pixel centers on a
                    # line, instead of four on a square. Those points will get double counted
                    # because ceil and floor will give the same result, so we use 0.5 as the weight
                    # to correct that.
                    x_interpolation_weight = np.abs(pixel_psf_x - psf_x) or 0.5
                    y_interpolation_weight = np.abs(pixel_psf_y - psf_y) or 0.5
                    epsf_contributions_to_pixels[pixel_y, pixel_x, psf_y, psf_x] += (
                        flux_ratio * x_interpolation_weight * y_interpolation_weight
                    )

    design_matrix = epsf_contributions_to_pixels.reshape(
        image_shape[0] * image_shape[1],
        oversampled_psf_shape[0] * oversampled_psf_shape[1],
    )
    if background_strap_mask is not None:
        # To calculate the linear gradients, we need the x and y coordinates of each pixel.
        image_pixel_xs, image_pixel_ys = get_xy_coordinates_centered_at_zero(image_shape)
        # background_contributions_to_pixels[iy, ix, b] is the contribution of background parameter
        # b to pixel (ix, iy) in the image.
        background_contribution_to_pixels = np.stack(
            (
                # This order is for historical compatibility
                background_strap_mask * image_pixel_ys,  # y-dependent background straps
                background_strap_mask * image_pixel_xs,  # x-dependent background straps
                background_strap_mask,  # flat background straps
                image_pixel_xs,  # x component of linear gradient => use y coordinate of each point
                image_pixel_ys,  # y component of linear gradient => use x coordinate of each point
                np.ones(image_shape),  # flat background level => same contribution to each point
            ),
            axis=-1,
        )

        # Construct the full design matrix by flattening image coordinates.
        design_matrix = np.hstack(
            (
                design_matrix,
                background_contribution_to_pixels.reshape(image_shape[0] * image_shape[1], -1),
            )
        )

    regularization_extension_size = 0
    if edge_compression_scale_factor is not None:
        # With the current set up, the flat background level could be partly fitted in the ePSF by
        # having a constant background level:
        # [[10 11 10]               [[0 1 0]
        #  [11 13 11]   instead of   [1 3 1]
        #  [10 11 10]]               [0 1 0]]
        # In the case shown here, the background level should be 10 higher than whatever was fitted.
        # To implement this, add rows to the design matrix that pick out a specific PSFpoint and give
        # it a weight based on its distance to the center of the PSF. The vector of observations should
        # have an appropriate number of zeros appended to it at fitting time.
        psf_point_x, psf_point_y = get_xy_coordinates_centered_at_zero(oversampled_psf_shape)
        psf_distance_from_center_weight = edge_compression_scale_factor * (
            1
            - np.exp(
                -0.5
                * (
                    (psf_point_x / psf_shape_pixels[1]) ** 4
                    + (psf_point_y / psf_shape_pixels[0]) ** 4
                )
            )
        )
        edge_compression_block = np.hstack(
            (
                np.diag(
                    psf_distance_from_center_weight.reshape(
                        oversampled_psf_shape[0] * oversampled_psf_shape[1]
                    )
                ),
                np.zeros(
                    (
                        oversampled_psf_shape[0] * oversampled_psf_shape[1],
                        background_contribution_to_pixels.shape[-1],
                    )
                ),
            )
        )
        design_matrix = np.vstack((design_matrix, edge_compression_block))
        regularization_extension_size = oversampled_psf_shape[0] * oversampled_psf_shape[1]

    return design_matrix, regularization_extension_size


def get_star_flux_ratios(gaia_table, flux_scale: str = "relative") -> np.ndarray:
    """Return the catalog flux scale used in the ePSF design matrix."""
    if flux_scale == "relative":
        return np.asarray(gaia_table["tess_flux_ratio"].data)

    tess_mag = np.asarray(gaia_table["tess_mag"], dtype=float)
    if flux_scale == "absolute":
        return 15000 * 10 ** ((tess_mag - 10) / -2.5)
    if flux_scale == "tmag10":
        return 10 ** ((10 - tess_mag) / 2.5)
    raise ValueError("flux_scale must be 'relative', 'absolute', or 'tmag10'.")


def _long_run_mask(mask: np.ndarray, min_length: int = 12) -> np.ndarray:
    run_mask = np.zeros_like(mask, dtype=bool)
    for axis in (0, 1):
        scan = np.moveaxis(mask, axis, 0)
        marked = np.zeros_like(scan, dtype=bool)
        for index in np.ndindex(scan.shape[1:]):
            line = scan[(slice(None),) + index]
            padded = np.concatenate(([False], line, [False]))
            changes = np.flatnonzero(padded[1:] != padded[:-1])
            for start, stop in zip(changes[::2], changes[1::2], strict=False):
                if stop - start >= min_length:
                    marked[(slice(start, stop),) + index] = True
        run_mask |= np.moveaxis(marked, 0, axis)
    return run_mask


def build_overexposure_mask(
    source,
    seed_sigma: float = 100,
    grow_sigma: float = 20,
    dilation: int = 2,
    max_mask_fraction: float = 0.2,
    min_bleed_length: int = 12,
) -> np.ndarray:
    """
    Build a static image mask for overexposed bleed-like regions.

    The mask removes contaminated image rows from ePSF fitting without removing
    stars from the model. Compact bright stars are rejected by requiring a long
    contiguous bright run.
    """
    image = np.nanmedian(source.flux, axis=0)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.ones(source.flux.shape[1:], dtype=bool)

    background = np.nanmedian(image[finite])
    scatter = 1.4826 * np.nanmedian(np.abs(image[finite] - background))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.nanstd(image[finite])
    if not np.isfinite(scatter) or scatter <= 0:
        return ~finite

    seed = finite & (image > background + seed_sigma * scatter)
    grow = finite & (image > background + grow_sigma * scatter)
    if ndimage is None:
        mask = seed & _long_run_mask(grow, min_length=min_bleed_length)
    else:
        candidate = ndimage.binary_propagation(seed, mask=grow)
        mask = candidate & _long_run_mask(candidate, min_length=min_bleed_length)
        if dilation > 0:
            mask = ndimage.binary_dilation(mask, iterations=int(dilation))
    mask = np.asarray(mask, dtype=bool) | ~finite

    mask_fraction = float(np.mean(mask))
    if mask_fraction > max_mask_fraction:
        warnings.warn(
            f"Overexposure mask covers {mask_fraction:.3f} of the cut, above "
            f"the configured {max_mask_fraction:.3f} limit."
        )
    return mask


def normalize_epsf_unit_sum(epsf: np.ndarray, over_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Split fitted ePSFs into a unit-sum shape and per-cadence scale."""
    psf_cols = over_size**2
    epsf_unit = np.array(epsf, copy=True)
    psf_block = epsf_unit[:, :psf_cols]
    scale = np.nansum(psf_block, axis=1)
    scale[np.isnan(psf_block).all(axis=1)] = np.nan
    valid = np.isfinite(scale) & (scale != 0)
    epsf_unit[valid, :psf_cols] /= scale[valid, np.newaxis]
    epsf_unit[~valid, :psf_cols] = np.nan
    return epsf_unit, scale


def reconstruct_epsf_unit_sum(
    epsf_unit: np.ndarray, scale: np.ndarray, over_size: int
) -> np.ndarray:
    """Reconstruct full fitted ePSF parameters from unit-sum shape and scale."""
    psf_cols = over_size**2
    epsf = np.array(epsf_unit, copy=True)
    epsf[:, :psf_cols] *= np.asarray(scale)[:, np.newaxis]
    return epsf


def fit_epsf(
    design_matrix: np.ndarray,
    flux: np.ndarray,
    base_flux_mask: np.ndarray,
    flux_uncertainty_power: float,
    regularization_dimensions: int,
    extra_flux_mask: np.ndarray | None = None,
):
    """
    Find the best-fit ePSF parameters given a design matrix and observed flux values.

    Uses `xp.linalg.lstsq` where `xp` is numpy or cupy depending on the whether `design_matrix` is
    on the CPU or GPU.

    Parameters
    ----------
    design_matrix : array
        2D matrix with shape `(m + r, n)` where `m` is the number of pixels in image, `r` is the
        number of extra dimensions used for regularization, and `n` is the number of parameters in
        the ePSF model.
    flux : array
        2D array of observed flux values with shape `(a, b)` where `a * b == m`.
    base_flux_mask : array[bool]
        2D mask array indicating bad (e.g., saturated) pixels. Pixels lower than 0.8 times the
        median flux are masked in addition.
    flux_uncertainty_power : float
        Power of pixel value used as observational uncertainty in fit. <1 emphasizes contributions
        from dimmer stars, 1 means all contributions are equal.
    regularization_dimensions : int
        Number of extra dimensions used for regularization. Must be added to observed vector.
    extra_flux_mask : array[bool] | None
        Additional image-pixel mask, for example a static overexposure/bleed mask.

    Returns
    -------
    epsf_parameters : array
        Array of best-fit ePSF parameters.
    """
    if HAS_CUPY:
        import cupy as cp

        xp = cp.get_array_module(design_matrix, flux)
    else:
        xp = np

    finite_flux = xp.isfinite(flux)
    flux_uncertainty_scale = 1 / (xp.abs(flux) ** flux_uncertainty_power)
    flux_uncertainty_scale = xp.where(finite_flux, flux_uncertainty_scale, 1)
    flux_mask = base_flux_mask | ~finite_flux | (flux < 0.8 * xp.nanmedian(flux))
    if extra_flux_mask is not None:
        flux_mask = flux_mask | extra_flux_mask

    # Set up observed vector accounting for regularization.
    observed_vector = xp.concatenate((flux.ravel(), xp.zeros(regularization_dimensions)))
    uncertainty_scale = xp.concatenate(
        (flux_uncertainty_scale.ravel(), xp.ones(regularization_dimensions))
    )
    mask = xp.concatenate(
        (flux_mask.ravel(), xp.zeros(regularization_dimensions, dtype=bool))
    )

    A = (design_matrix * uncertainty_scale[:, np.newaxis])[~mask]
    b = (observed_vector * uncertainty_scale)[~mask]

    try:
        # Solve the normal equation instead of running a least squares fit directly. This is much
        # faster because `alpha` is the size of the number of dimensions of the PSF model, which is
        # much smaller than the number of dimensions of the observed flux. In the usual case, this
        # amounts to solving a 535-dimesnional linear equation instead of running a least squares
        # fit on a 23029x535 matrix.
        # Using the normal equation is valid because A has *many* more rows than columns, so the
        # chance that A has linearly dependent columns is negligible.
        alpha = A.T @ A
        beta = A.T @ b
        return xp.linalg.solve(alpha, beta)
    except xp.linalg.LinAlgError:
        # Just in case - this is useful eg for testing
        return xp.linalg.lstsq(A, b)[0]

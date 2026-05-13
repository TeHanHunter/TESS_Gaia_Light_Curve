"""Aperture photometry for TGLC light curves with 3 apertures."""

from astropy import units as u
from astropy.table import QTable
import numpy as np
from scipy.ndimage import center_of_mass

from tglc.utils.constants import (
    TESS_PIXEL_SATURATION_LEVEL,
    convert_tess_flux_to_tess_magnitude,
    convert_tess_magnitude_to_tess_flux,
)


def get_aperture_limits(
    aperture_size: int, x: int, y: int, top_limit: int, right_limit: int
) -> tuple[int, int, int, int]:
    """Get (bottom, top, left, right) limits for aperture within 5x5 pixel grid."""
    bottom = max(0, y - aperture_size // 2)
    top = min(top_limit, y + aperture_size // 2 + 1)
    left = max(0, x - aperture_size // 2)
    right = min(right_limit, x + aperture_size // 2 + 1)
    return bottom, top, left, right


def get_normalized_aperture_photometry(
    images: np.ndarray,
    quality_flags: np.ndarray,
    aperture_size: int,
    x: int,
    y: int,
    tmag: float,
    exposure_time: u.Quantity,
    flux_portion: np.ndarray,
    column_name_prefix: str = "",
) -> QTable:
    """
    Extract normalized magnitude light curve from time series of images.

    Flux is extracted via aperture photometry from the images and converted to TESS magnitude based
    on the reference flux of 15,000 e-/s for a star of TESS magnitude 10 given in the TESS Instrument
    Handbook, p. 37. The flux is then normalized to have its median at the expected flux for the
    target TESS magnitude, and the residual is recorded as the local background level for the light
    curve.

    Saturated points are removed. Points are considered saturated if any of the 2 second
    integrations would have been saturated over the course of the exposure, using the saturation
    level of 200,000 e- given in the TESS Instrument Handbook, p. 37.

    See <https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf#page=38>.

    Parameters
    ----------
    images : array_like
        3 dimensional array with time as first dimension and image cutouts as remaining dimensions.
    quality_flags : array_like[int]
        Quality flags for the cadences, where 0 indicates a good value.
    aperture_size : int
        Side length of square aperture to use.
    x, y : int
        Aperture center coordinates in images.
    tmag : float
        TESS magnitude of target star.
    exposure_time : u.Quantity (time)
        Exposure time for each light curve value. Used to determine saturated points.
    flux_portion : array_like
        Proportion of flux in each pixel of the images. Should be a 2D array with shape matching the
        last two dimensions of `images`, and entries that sum to 1.
    column_name_prefix : str
        Prefix inserted into column names. Default is no prefix.

    Returns
    -------
    photometry_data : QTable
        Table with magnitudes extracted from photometry and flux-weighted centroid of the aperture,
        in the coordinate system of the images. The table metadata contains the local background
        level determined during normalization.

        Columns:
        - `"{column_name_prefix}flux"`: Normalized total flux value in aperture, or NaN if saturated
        - `"{column_name_prefix}magnitude"`: Normalized magnitude value for aperture, or NaN if
          saturated
        - `"{column_name_prefix}centroid_x"`: X coordinate in image of flux-weighted aperture centroid
        - `"{column_name_prefix}centroid_y"`: Y coordinate in image of flux-weighted aperture centroid

        Metadata:
        - `"local_background"`: Local background flux level used in normalization.
    """
    bottom, top, left, right = get_aperture_limits(
        aperture_size, x, y, images.shape[1], images.shape[2]
    )
    flux = np.nansum(images[:, bottom:top, left:right], axis=(1, 2)) * u.electron
    centroids = (
        np.array([center_of_mass(image[bottom:top, left:right]) for image in images]) * u.pixel
    )
    centroids[:, 0] += bottom * u.pixel
    centroids[:, 1] += left * u.pixel

    is_saturated = flux > TESS_PIXEL_SATURATION_LEVEL * (aperture_size**2) * exposure_time / (
        2.0 * u.second
    )
    flux[is_saturated] = np.nan
    centroids[is_saturated, :] = np.nan

    expected_total_flux_per_cadence = convert_tess_magnitude_to_tess_flux(tmag) * exposure_time
    flux_portion_in_aperture = np.nansum(flux_portion[bottom:top, left:right])
    expected_aperture_flux = expected_total_flux_per_cadence * flux_portion_in_aperture

    # Local background is the average amount of flux above the expected amount
    local_background = np.nanmedian(flux[quality_flags == 0]) - expected_aperture_flux
    if not np.isnan(local_background):
        # `local_background` is NaN if there are no good-flagged, non-NaN points
        flux -= local_background
    # TWIRL fork: preserve negative flux for downstream flux-space cotrending.
    # Magnitude is computed on a positive-only copy (NaN where flux <= 0) so
    # convert_tess_flux_to_tess_magnitude emits no runtime warnings; the
    # magnitude column behaviour is unchanged (NaN at flux <= 0). The linear
    # flux column is preserved including its negative excursions, which are
    # physically real Poisson + background-subtraction fluctuations on faint
    # (T >= 19) targets and are needed for unbiased flux-space detrending.
    pos_flux = flux.copy()
    pos_flux[flux <= 0 * flux.unit] = np.nan * flux.unit

    table = QTable(
        {
            f"{column_name_prefix}flux": flux,
            f"{column_name_prefix}magnitude": convert_tess_flux_to_tess_magnitude(
                pos_flux / flux_portion_in_aperture / exposure_time
            ),
            f"{column_name_prefix}centroid_x": centroids[:, 1],
            f"{column_name_prefix}centroid_y": centroids[:, 0],
        },
        meta={f"{column_name_prefix}local_background": local_background},
    )

    return table

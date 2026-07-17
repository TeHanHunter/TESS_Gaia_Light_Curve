"""Light curve extraction functionality."""

from collections.abc import Generator
import logging
from math import ceil, floor

from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
from astropy.table import QTable, hstack
from astropy.time import Time
import astropy.units as u
import numpy as np

from tglc.aperture_light_curve import ApertureLightCurve, ApertureLightCurveMetadata
from tglc.aperture_photometry import get_normalized_aperture_photometry
from tglc.epsf import get_star_flux_ratios, make_tglc_design_matrix
from tglc.ffi import Source
from tglc.utils.constants import TESSJD, apply_barycentric_correction  # noqa: F401 for tjd format
from tglc.utils.tess_ephemeris import get_tess_spacecraft_position


logger = logging.getLogger(__name__)


def get_cutout_for_light_curve(
    flux: np.ndarray,
    epsf: np.ndarray,
    full_design_matrix: np.ndarray,
    target_x: float,
    target_y: float,
    target_flux_ratio: float,
    psf_shape: tuple[int, int],
    psf_oversample_factor: int,
    cutout_size: int = 5,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Make a decontaminated flux cutout suitable for light curve extraction.

    Parameters
    ----------
    flux : array
        Time series of images, with shape `(t, n, m)`.
    epsf : array
        Best-fit PSF and background parameters, with shape `(t, k)`.
    full_design_matrix : array
        Design matrix modeling all stars in the image (and background), with shape `(n * m, k)`.
    target_x, target_y : float
        Coordinates of target in images.
    target_flux_ratio : float
        Ratio of flux from target star to max flux from any star in the image, according to Gaia
        catalog data. Used to create a design matrix specifc to the target star.
    psf_shape : tuple[int, int]
        Shape of the PSF in pixels. Used to create a design matrix specific to the target star.
    psf_oversample_factor: int
        Factor by which to oversample the PSF compared to image pixels. Used to create a design
        matrix specific to the target star.
    cutout_size : int
        Side length of the square cutout in pixels. Cutout may be smaller if target star is near the
        edge of the images.

    Returns
    -------
    cutout, target_x_cutout, target_y_cutout, psf_portions : tuple[array, float, float, array]
        Tuple containing a time series of cutout images, with shape `(t, cutout_size, cutout_size)`
        (last two dimensions may differ for cutouts near image edges), the coordinates of the target
        star within the cutout, and the portion of the ePSF contained in each pixel of the cutout.
    """
    points_in_oversampled_psf = (psf_shape[0] * psf_oversample_factor + 1) * (
        psf_shape[1] * psf_oversample_factor + 1
    )
    cutout_left = max(0, round(target_x) - floor(cutout_size / 2))
    cutout_right = min(flux.shape[2], round(target_x) + ceil(cutout_size / 2))
    cutout_bottom = max(0, round(target_y) - floor(cutout_size / 2))
    cutout_top = min(flux.shape[1], round(target_y) + ceil(cutout_size / 2))
    cutout_shape = (cutout_top - cutout_bottom, cutout_right - cutout_left)
    target_x_in_cutout = target_x - cutout_left
    target_y_in_cutout = target_y - cutout_bottom
    cutout_flux = flux[:, cutout_bottom:cutout_top, cutout_left:cutout_right]

    # We need a design matrix that models everything in the cutout *except* the target star. To do
    # this, we get the relevant part of the complete design matrix and a design matrix for
    # *just* the target star, and subtract the target design matrix from the complete design matrix.
    # Note: it would be simpler to do this for the entire image and then get the cutout at the end,
    # but that's *much* slower because the matrices involved are huge.
    cutout_x, cutout_y = np.meshgrid(
        np.arange(cutout_left, cutout_right), np.arange(cutout_bottom, cutout_top)
    )
    cutout_coordinate_rows_in_design_matrix = (cutout_x + cutout_y * flux.shape[1]).flatten()
    full_design_matrix_for_cutout = full_design_matrix[cutout_coordinate_rows_in_design_matrix]

    target_design_matrix_for_cutout, _ = make_tglc_design_matrix(
        cutout_shape,
        psf_shape,
        psf_oversample_factor,
        np.array([[target_x_in_cutout, target_y_in_cutout]]),
        np.array([target_flux_ratio]),
    )

    field_design_matrix_for_cutout = full_design_matrix_for_cutout.copy()
    field_design_matrix_for_cutout[:, :points_in_oversampled_psf] -= target_design_matrix_for_cutout
    cutout_field_model = np.dot(field_design_matrix_for_cutout, epsf.T).T.reshape(
        flux.shape[0], *cutout_shape
    )
    decontaminated_cutout_flux = cutout_flux - cutout_field_model

    cutout_target_psf = np.dot(
        target_design_matrix_for_cutout, epsf[:, :points_in_oversampled_psf].T
    ).T.reshape(flux.shape[0], *cutout_shape)
    psf_portion_in_cutout = np.nansum(cutout_target_psf, axis=0) / np.nansum(cutout_target_psf)

    return decontaminated_cutout_flux, target_x_in_cutout, target_y_in_cutout, psf_portion_in_cutout


def generate_light_curves(
    source: Source,
    epsf: np.ndarray,
    psf_size: int,
    psf_oversample_factor: int,
    tic_ids: list[int] | None = None,
    flux_scale: str = "relative",
) -> Generator[ApertureLightCurve, None, None]:
    """
    Generator function that yields aperture light curves extracted from the source cutout.

    Parameters
    ----------
    source : Source
        Cutout `Source` object including flux data and positions of stars in the flux images.
    epsf : array
        Best-fit PSF and background parameters, with shape `(t, k)`.
    psf_size : array
        Side length of square PSF in pixels. Used to construct design matrices.
    psf_oversample_factor : int
        Factor by which to oversample the PSF compared to image pixels. Used to construct design
        matrices.
    max_magnitude : float
        Maximum magnitude of target stars for which light curves should be extracted.
    tic_ids : list[int] | None
        Optional list of TIC IDs that should have light curves made. If specified, all other targets
        will be ignored. By default, all targets in the source TIC catalog have light curves made.
    flux_scale : str
        Catalog flux scale used when the ePSFs were fitted. Must match the `tglc epsfs`
        `--flux-scale` setting.

    Yields
    ------
    light_curve : ApertureLightCurve
        Aperture light curves extracted from the source cutout with the ePSF parameters given.
    """
    tic_match_table = source.tic
    if tic_ids is not None:
        tic_match_table = tic_match_table[np.isin(tic_match_table["TIC"], tic_ids)]
    if len(tic_match_table) == 0:
        logger.debug("No targets found, skipping light curve generation")
        return
    logger.debug(f"Making light curves for {tic_match_table} targets")

    star_positions = np.array(
        [source.gaia[f"sector_{source.sector}_x"], source.gaia[f"sector_{source.sector}_y"]]
    ).T
    star_flux_ratios = get_star_flux_ratios(source.gaia, flux_scale=flux_scale)
    design_matrix, _ = make_tglc_design_matrix(
        source.flux.shape[1:],
        (psf_size, psf_size),
        psf_oversample_factor,
        star_positions,
        star_flux_ratios,
        source.mask.data,
    )

    # Use the model's flat background level to determine points that should be ignored during
    # normalization in photometry
    flat_background = epsf[:, -6]
    high_background_points = np.abs(flat_background - np.nanmedian(flat_background)) >= mad_std(
        flat_background, ignore_nan=True
    )

    # These are used for all light curves
    model_background = np.dot(design_matrix[:, -6:], epsf[:, -6:].T).T.reshape(source.flux.shape)
    time = Time(source.time, format="tjd", scale="tdb")
    tess_spacecraft_position = get_tess_spacecraft_position(source.sector, time)

    nearest_pixel_x = np.round(source.gaia[f"sector_{source.sector}_x"]).astype(int)
    nearest_pixel_y = np.round(source.gaia[f"sector_{source.sector}_y"]).astype(int)
    # Targets outside these bounds have too little data to make light curves
    pixel_left_bound = 1.5
    pixel_right_bound = source.size - 2.5
    pixel_bottom_bound = 1.5
    pixel_top_bound = source.size - 2.5

    for tic_id, gaia3_id in tic_match_table:
        try:
            i = np.nonzero(source.gaia["designation"] == f"Gaia DR3 {gaia3_id}")[0][0]
        except IndexError:
            logger.debug(f"No Gaia catalog entry found for TIC {tic_id}/Gaia DR3 {gaia3_id}")
            continue

        if not (
            (pixel_left_bound <= nearest_pixel_x[i] <= pixel_right_bound)
            and (pixel_bottom_bound <= nearest_pixel_y[i] <= pixel_top_bound)
        ):
            continue

        light_curve_cutout, star_x, star_y, psf_portions = get_cutout_for_light_curve(
            source.flux,
            epsf,
            design_matrix,
            star_positions[i][0],
            star_positions[i][1],
            star_flux_ratios[i],
            (psf_size, psf_size),
            psf_oversample_factor,
            cutout_size=5,
        )

        sky_coord = SkyCoord(source.gaia["ra"][i], source.gaia["dec"][i], unit="deg")
        time_btjd = apply_barycentric_correction(time, sky_coord, tess_spacecraft_position)
        aperture_photometry_data = [
            get_normalized_aperture_photometry(
                light_curve_cutout,
                np.array(source.quality) | high_background_points,
                aperture_size,
                round(star_x),
                round(star_y),
                source.gaia["tess_mag"][i],
                source.exposure * u.second,
                psf_portions,
                column_name_prefix=f"{aperture_name}_aperture_",
            )
            for aperture_name, aperture_size in [("primary", 3), ("small", 1), ("large", 5)]
        ]
        for aperture_name, table in zip(
            ["primary", "small", "large"], aperture_photometry_data, strict=False
        ):
            table[f"{aperture_name}_aperture_centroid_x"] += (
                source.ccd_x + nearest_pixel_x[i] - star_x
            ) * u.pixel
            table[f"{aperture_name}_aperture_centroid_y"] += (
                source.ccd_y + nearest_pixel_y[i] - star_y
            ) * u.pixel

        # Background light curve is the background level at the star's location
        background_light_curve = model_background[:, nearest_pixel_y[i], nearest_pixel_x[i]]
        background_quality_flags = np.abs(
            background_light_curve - np.nanmedian(background_light_curve)
        ) >= 5 * mad_std(background_light_curve)

        target_ccd_x = star_positions[i][0] + source.ccd_x
        target_ccd_y = star_positions[i][1] + source.ccd_y

        light_curve_meta = ApertureLightCurveMetadata(
            tic_id=tic_id,
            orbit=source.orbit,
            sector=source.sector,
            camera=source.camera,
            ccd=source.ccd,
            ccd_x=target_ccd_x,
            ccd_y=target_ccd_y,
            sky_coord=sky_coord,
            tess_magnitude=source.gaia["tess_mag"][i],
            exposure_time=source.exposure * u.second,
            primary_aperture_local_background=aperture_photometry_data[0].meta[
                "primary_aperture_local_background"
            ],
            small_aperture_local_background=aperture_photometry_data[1].meta[
                "small_aperture_local_background"
            ],
            large_aperture_local_background=aperture_photometry_data[2].meta[
                "large_aperture_local_background"
            ],
        )

        base_light_curve = QTable(
            {
                "time": time_btjd,
                "cadence": source.cadence,
                # Use 2 for background quality flags to avoid conflicting with QLP quality flags
                "quality_flag": background_quality_flags.astype(int) * 2,
                "background_flux": background_light_curve,
            }
        )

        light_curve = ApertureLightCurve(
            hstack(aperture_photometry_data + [base_light_curve]), meta=light_curve_meta
        )
        yield light_curve

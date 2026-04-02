from collections.abc import Sequence
from functools import partial
from importlib import resources
from itertools import product
import logging
from pathlib import Path
import pickle
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, MaskedColumn, QTable, Table, hstack
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from erfa.core import ErfaWarning
import numba
from numba import float32, jit, prange
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.utils import data
from tglc.utils.constants import get_sector_containing_orbit
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


TICA_QUALITY_HEADER_START_SECTOR = 67


# from Tim
def background_mask(im=None):
    imfilt = im * 1.0
    for i in range(im.shape[1]):
        imfilt[:, i] = ndimage.percentile_filter(im[:, i], 50, size=51)

    ok = im < imfilt
    # Don't use saturated pixels!
    satfactor = 0.4
    ok *= im < satfactor * np.amax(im)
    running_factor = 1
    cal_factor = np.zeros(im.shape[1])
    cal_factor[0] = 1

    di = 1
    i = 0
    while i < im.shape[1] - 1 and i + di < im.shape[1]:
        _ok = ok[:, i] * ok[:, i + di]
        coef = np.median(im[:, i + di][_ok] / im[:, i][_ok])
        if 0.5 < coef < 2:
            running_factor *= coef
            cal_factor[i + di] = running_factor
            i += di
            di = 1  # Reset the stepsize to one.
        else:
            # Label the column as bad, then skip it.
            cal_factor[i + di] = 0
            di += 1

    # cal_factor[im > 0.4 * np.amax(im)] = 0
    return cal_factor


class Source:
    def __init__(
        self,
        x=0,
        y=0,
        flux=None,
        time=None,
        wcs=None,
        quality=None,
        mask=None,
        exposure=1800,
        orbit=0,
        sector=0,
        size=150,
        camera=1,
        ccd=1,
        cadence=None,
        gaia_catalog=None,
        tic_catalog=None,
    ):
        """
        Source object that includes all data from TESS and Gaia DR2
        :param x: int, required
        starting horizontal pixel coordinate
        :param y: int, required
        starting vertical pixel coordinate
        :param flux: np.ndarray, required
        3d data cube, the time series of a all FFI from a CCD
        :param time: np.ndarray, required
        1d array of time
        :param wcs: astropy.wcs.wcs.WCS, required
        WCS Keywords of the TESS FFI
        :param orbit: int, required
        TESS orbit number
        :param sector: int, required
        TESS sector number
        :param size: int, optional
        the side length in pixel  of TESScut image
        :param camera: int, optional
        camera number
        :param ccd: int, optional
        CCD number
        :param cadence: list, required
        list of cadences of TESS FFI
        :param gaia_catalog: QTable, required
        Gaia catalog data
        :param tic_catalog: QTable, required
        TIC catalog data
        """
        if cadence is None:
            cadence = []
        if quality is None:
            quality = []
        if wcs is None:
            wcs = []
        if time is None:
            time = []
        if flux is None:
            flux = []

        self.size = size
        self.orbit = orbit
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.cadence = cadence
        self.quality = quality
        self.exposure = exposure
        self.wcs = wcs
        self.ccd_x = x + 44
        self.ccd_y = y

        # Load catalog files and find relevant stars
        gaia_sky_coordinates = SkyCoord(gaia_catalog["ra"], gaia_catalog["dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            gaia_x, gaia_y = wcs.world_to_pixel(gaia_sky_coordinates)
        gaia_x_in_source = (self.ccd_x <= gaia_x) & (gaia_x <= self.ccd_x + size)
        gaia_y_in_source = (self.ccd_y <= gaia_y) & (gaia_y <= self.ccd_y + size)
        gaia_in_source = gaia_x_in_source & gaia_y_in_source
        catalogdata = gaia_catalog[gaia_in_source]

        tic_sky_coordinates = SkyCoord(tic_catalog["ra"], tic_catalog["dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            tic_x, tic_y = wcs.world_to_pixel(tic_sky_coordinates)
        tic_x_in_source = (self.ccd_x <= tic_x) & (tic_x <= self.ccd_x + size)
        tic_y_in_source = (self.ccd_y <= tic_y) & (tic_y <= self.ccd_y + size)
        tic_in_source = tic_x_in_source & tic_y_in_source
        catalogdata_tic = tic_catalog[tic_in_source]

        # Cross match TIC and Gaia
        tic_match_table = Table()
        tic_match_table.add_column(catalogdata_tic["id"], name="TIC")
        tic_match_table.add_column(catalogdata_tic["gaia3"], name="gaia3")
        self.tic = tic_match_table

        # TODO remove this at some point, but right now units aren't expected downstream
        for name, col in catalogdata.columns.items():
            if np.ma.is_masked(col):
                catalogdata[name] = MaskedColumn(col.data, mask=col.mask)
            else:
                catalogdata[name] = Column(col.data)

        self.flux = flux[:, y : y + size, x : x + size]
        self.mask = mask[y : y + size, x : x + size]
        self.time = np.array(time)
        median_time = np.median(self.time)
        interval = (median_time - 388.5) / 365.25
        # Julian Day Number:	2457000.0 (TBJD=0)
        # Calendar Date/Time:	2014-12-08 12:00:00 388.5 days before J2016

        num_gaia = len(catalogdata)
        x_gaia = np.zeros(num_gaia)
        y_gaia = np.zeros(num_gaia)
        tess_mag = np.zeros(num_gaia)
        in_frame = [True] * num_gaia
        for i, designation in enumerate(catalogdata["designation"]):
            ra = catalogdata["ra"][i]
            dec = catalogdata["dec"][i]
            if not np.isnan(catalogdata["pmra"].mask[i]):  # masked?
                ra += catalogdata["pmra"][i] * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
            if not np.isnan(catalogdata["pmdec"].mask[i]):
                dec += catalogdata["pmdec"][i] * interval / 1000 / 3600
            pixel = self.wcs.all_world2pix(
                np.array([catalogdata["ra"][i], catalogdata["dec"][i]]).reshape((1, 2)),
                0,
                quiet=True,
            )
            x_gaia[i] = pixel[0][0] - self.ccd_x
            y_gaia[i] = pixel[0][1] - self.ccd_y
            if np.isnan(catalogdata["phot_g_mean_mag"][i]):
                in_frame[i] = False
            elif catalogdata["phot_g_mean_mag"][i] >= 25:
                in_frame[i] = False
            elif -4 < x_gaia[i] < self.size + 3 and -4 < y_gaia[i] < self.size + 3:
                dif = catalogdata["phot_bp_mean_mag"][i] - catalogdata["phot_rp_mean_mag"][i]
                with warnings.catch_warnings():
                    # Warnings for for masked value conversion to nan
                    warnings.simplefilter("ignore", UserWarning)
                    tess_mag[i] = (
                        catalogdata["phot_g_mean_mag"][i]
                        - 0.00522555 * dif**3
                        + 0.0891337 * dif**2
                        - 0.633923 * dif
                        + 0.0324473
                    )
                    if np.isnan(tess_mag[i]):
                        tess_mag[i] = catalogdata["phot_g_mean_mag"][i] - 0.430
                    if np.isnan(tess_mag[i]):
                        in_frame[i] = False
            else:
                in_frame[i] = False

        tess_flux = 10 ** (-tess_mag / 2.5)
        t = Table()
        t["tess_mag"] = tess_mag[in_frame]
        t["tess_flux"] = tess_flux[in_frame]
        t["tess_flux_ratio"] = tess_flux[in_frame] / (
            np.nanmax(tess_flux[in_frame]) if len(tess_flux[in_frame]) > 0 else 1
        )
        t[f"sector_{self.sector}_x"] = x_gaia[in_frame]
        t[f"sector_{self.sector}_y"] = y_gaia[in_frame]
        catalogdata = hstack([catalogdata[in_frame], t])
        catalogdata.sort("tess_mag")
        self.gaia = catalogdata


def _get_science_pixel_limits(scipixs_string: str) -> tuple[int, int, int, int]:
    """
    Parse string of the form `"[min_x:max_x,min_y:max_y]"`.

    TICA FFI FITS headers have a "SCIPIXS" keyword indicating which pixels are science pixels
    (rather than buffer rows/columns) of the form indicated above.

    Note
    ----
    The string in TICA headers is 1-indexed, but the pixel coordinates are converted to 0-indexed
    values for use with data arrays in this function. So, the science data from the detector is
    obtained by
    ```python
    science_data = data[min_y:max_y + 1, min_x:max_x + 1]
    ```
    where `data` is the primary extension data from the TICA FFI FITS file.

    Returns
    -------
    (min_x, max_x, min_y, max_y) : tuple[int, int, int, int]
        Tuple containing science pixel limits (inclusive)
    """
    x_range, y_range = scipixs_string.strip("[]").split(",")
    min_x, max_x = map(int, x_range.split(":"))
    min_y, max_y = map(int, y_range.split(":"))
    return (min_x - 1, max_x - 1, min_y - 1, max_y - 1)


def _get_ffi_header_data_and_flux(
    ffi_file: Path, camera: int, sector: int
) -> tuple[int, int, float, np.ndarray]:
    """
    Harvest important header values and pixel flux values from a TICA FFI file.

    Parameters
    ----------
    ffi_file : Path
        Path to FFI FITS file to read.
    camera : int
        Camera of FFI. Needed to read stray light header value.
    sector : int
        TESS sector of FFI. TICA FFIs before Sector 67 do not reliably carry `COARSE`,
        `RW_DESAT`, and `STRAYLT1-4`, so those quality bits are treated as zero there when
        missing.

    Returns
    -------
    (quality, cadence, time, flux) : tuple[int, int, float, array_like]
        Tuple containing quality flag, cadence, and time values pulled from header and flux array
        taken from science pixels.
    """
    try:
        with warnings.catch_warnings():
            # TICA FITS headers get wrangled by Astropy, but it creates no problems
            warnings.simplefilter("ignore", AstropyWarning)
            with fits.open(ffi_file, mode="readonly", memmap=False) as hdulist:
                primary_header = hdulist[0].header
                # Map quality indicators to bits that align with FFI quality flags.
                # See https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014-Rev-F.pdf?page=56
                # The QLP update notes indicate these TICA quality keywords were only added
                # starting in Sector 67, so pre-S67 FFIs fall back to zero when they are absent.
                if sector < TICA_QUALITY_HEADER_START_SECTOR:
                    coarse = primary_header.get("COARSE", 0)
                    rw_desat = primary_header.get("RW_DESAT", 0)
                    straylt = primary_header.get(f"STRAYLT{camera}", 0)
                else:
                    coarse = primary_header["COARSE"]
                    rw_desat = primary_header["RW_DESAT"]
                    straylt = primary_header[f"STRAYLT{camera}"]
                quality = (
                    (coarse << 2)
                    & (rw_desat << 5)
                    & (straylt << 11)
                )
                cadence = primary_header["CADENCE"]
                time = primary_header["MIDTJD"]
                min_x, max_x, min_y, max_y = _get_science_pixel_limits(primary_header["SCIPIXS"])
                flux = hdulist[0].data[min_y : max_y + 1, min_x : max_x + 1]
        return (quality, cadence, time, flux)
    except Exception as e:
        logger.warning(f"Invalid FFI file {ffi_file.resolve()}: got error {e}")
        return (0, 0, np.nan, np.full((2048, 2048), np.nan))


def _make_source_and_write_pickle(
    x_y: tuple[int, int], manifest: Manifest, replace: bool, **kwargs
):
    """
    Construct source object and write pickle file.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks coordinates from the first argument.
    """
    x, y = x_y
    manifest.cutout_x = x
    manifest.cutout_y = y
    if not replace and (manifest.source_file.is_file() and manifest.source_file.stat().st_size > 0):
        logger.debug(
            f"Source file for camera {kwargs['camera']}, CCD {kwargs['ccd']}, {x}_{y} already exists, skipping"
        )
        return
    kwargs["x"] = x * (kwargs["size"] - 4)
    kwargs["y"] = y * (kwargs["size"] - 4)
    source = Source(**kwargs)
    with open(manifest.source_file, "wb") as output:
        pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)


@jit(float32[:, :](float32[:, :, :]), nogil=True, parallel=True)
def _fast_nanmedian_axis0(array):
    """
    Fast JIT-compiled, multithreaded version of np.nanmedian(array, axis=0)

    Computing a nanmedian image from all the FFI data is necessary to detect bad pixels, but on
    arrays with roughly the shape (6000, 2048, 2048), this is incredibly. We use numba here to
    distribute the work to many cores.
    """
    result = np.empty(array.shape[1:], dtype=np.float32)
    for i in prange(result.shape[0]):
        for j in prange(result.shape[1]):
            result[i, j] = np.nanmedian(array[:, i, j])
    return result


def ffi(
    orbit: int,
    camera: int,
    ccd: int,
    cutouts: Sequence[tuple[int, int]] | None,
    manifest: Manifest,
    cutout_size: int = 150,
    cutout_overlap: int = 2,
    produce_mask: bool = False,
    nprocs: int = 1,
    replace: bool = False,
):
    """
    Produce `Source` object pickle file from calibrated FFI files.

    The `source/` directory is created for the given orbit/camera/CCD. If `produce_mask` is `True`,
    the a mask file is created instead.

    Parameters
    ----------
    orbit : int
        TESS orbit of the FFI observations.
    camera, ccd : int
        TESS camera and CCD of FFIs that should be used.
    cutouts : Iterable[tuple[int, int]] | None
        Pairs of `(x, y)` coordinates of cutouts that should be created. If `None`, all cutouts are
        created based on the specified size.
    manifest : Manifest
        Manifest object used to determine input/output file paths. The `orbit`, `camera`, and `ccd`
        fields are populated using arguments.
    cutout_size : int
        Side length of cutouts (pixels). Large numbers recommended for better quality. Default = 150.
    cutout_overlap : int
        Overlap between adjecent cutouts cutouts (pixels). Default = 2.
    produce_mask : bool
        Produce CCD mask instead of making cutout `Source` objects.
    nprocs : int
        Processes to use for in multiprocessing pool. Default = 1.
    replace : bool
        Replace existing files with new data. Default = False.
    """
    manifest.orbit = orbit
    manifest.camera = camera
    manifest.ccd = ccd
    if not manifest.ffi_directory.is_dir():
        logger.warning(
            f"FFI directory for camera {camera} CCD {ccd} not found, skipping."
            f" Expected path: {manifest.ffi_directory.resolve()}"
        )
        return
    elif not manifest.tic_catalog_file.is_file():
        logger.warning(
            f"TIC catalog file for camera {camera} CCD {ccd} not found, skipping."
            f" Expected path: {manifest.tic_catalog_file.resolve()}"
        )
        return
    elif not manifest.gaia_catalog_file.is_file():
        logger.warning(
            f"TIC catalog file for camera {camera} CCD {ccd} not found, skipping."
            f" Expected path: {manifest.gaia_catalog_file.resolve()}"
        )
        return
    ffi_files = list(manifest.ffi_directory.glob(manifest.tica_ffi_file_pattern))

    if len(ffi_files) == 0:
        logger.warning(f"No FFI files found for camera {camera} CCD {ccd}, skipping")
        return
    logger.info(f"Found {len(ffi_files)} FFI files for camera {camera} CCD {ccd}")

    time = np.full_like(ffi_files, np.nan, dtype=np.float64)
    cadence = np.zeros_like(ffi_files, dtype=np.int64)
    quality = np.zeros_like(ffi_files, dtype=np.int32)
    flux = np.full((len(ffi_files), 2048, 2048), np.nan, dtype=np.float32)
    sector = get_sector_containing_orbit(orbit)
    get_ffi_header_data_and_flux_for_camera = partial(
        _get_ffi_header_data_and_flux, camera=camera, sector=sector
    )
    ffi_data_iterator = tqdm(
        pool_map_if_multiprocessing(
            get_ffi_header_data_and_flux_for_camera,
            ffi_files,
            nprocs=nprocs,
            pool_map_method="imap_unordered",
        ),
        desc=f"Reading FFI files for {camera}-{ccd}",
        unit="file",
        total=len(ffi_files),
    )
    with logging_redirect_tqdm():
        for i, (ffi_quality, ffi_cadence, ffi_time, ffi_flux) in enumerate(ffi_data_iterator):
            quality[i] = ffi_quality
            cadence[i] = ffi_cadence
            time[i] = ffi_time
            flux[i] = ffi_flux
    logger.info("Sorting FFI data by timestamp")
    time_order = np.argsort(time)
    time = time[time_order]
    cadence = cadence[time_order]
    quality = quality[time_order]
    flux = flux[time_order, :, :]

    if np.min(np.diff(cadence)) != 1:
        logger.warning(f"{(np.diff(cadence) != 1).sum()} cadence gaps != 1 detected.")

    # Load or save mask
    numba.set_num_threads(nprocs)
    if produce_mask:
        logger.info("Saving background mask")
        median_flux = _fast_nanmedian_axis0(flux)
        mask = background_mask(im=median_flux)
        mask /= ndimage.median_filter(mask, size=51)
        np.save(manifest.ccd_directory / f"mask_orbit{orbit:04d}_cam{camera}_ccd{ccd}.npy", mask)
        return
    logger.info("Loading background mask")
    mask_file = resources.files(data) / "median_mask.fits"
    with fits.open(mask_file) as hdulist:
        mask = hdulist[0].data[(camera - 1) * 4 + (ccd - 1), :]
    mask = np.repeat(mask.reshape(1, 2048), repeats=2048, axis=0)

    logger.info("Detecting bad pixels")
    bad_pixels = np.zeros(flux.shape[1:], dtype=bool)
    median_flux = _fast_nanmedian_axis0(flux)
    bad_pixels[median_flux > 0.8 * np.nanmax(median_flux)] = 1
    bad_pixels[median_flux < 0.2 * np.nanmedian(median_flux)] = 1
    bad_pixels[np.isnan(median_flux)] = 1

    # Mark neighbors of bad pixels as also bad
    bad_y, bad_x = np.nonzero(bad_pixels)
    for x, y in zip(bad_x, bad_y, strict=False):
        bad_pixels[min(y + 1, 2047), x] = 1
        bad_pixels[max(y - 1, 0), x] = 1
        bad_pixels[y, min(x + 1, 2047)] = 1
        bad_pixels[y, max(x - 1, 0)] = 1

    mask = np.ma.masked_array(mask, mask=bad_pixels | (mask == 0))

    logger.info("Loading WCS pixel-to-world solution")
    # Get WCS object from good-quality FFI
    first_good_quality_ffi = ffi_files[np.nonzero(quality == 0)[0][0]]
    with warnings.catch_warnings():
        # TICA FITS headers get wrangled by Astropy, but it creates no problems
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(first_good_quality_ffi) as hdulist:
            wcs = WCS(hdulist[0].header)
            exposure = int(hdulist[0].header["EXPTIME"])

    gaia_catalog = QTable.read(manifest.gaia_catalog_file)
    tic_catalog = QTable.read(manifest.tic_catalog_file)

    logger.info(f"Writing cutout source pickle files to {manifest.source_directory.resolve()}")
    manifest.source_directory.mkdir(exist_ok=True)
    write_source_pickle_from_x_y = partial(
        _make_source_and_write_pickle,
        manifest=manifest,
        replace=replace,
        flux=flux,
        mask=mask,
        orbit=orbit,
        sector=sector,
        time=time,
        size=cutout_size,
        quality=quality,
        wcs=wcs,
        camera=camera,
        ccd=ccd,
        exposure=exposure,
        cadence=cadence,
        gaia_catalog=gaia_catalog,
        tic_catalog=tic_catalog,
    )
    if cutouts is None:
        edge_cutout_size = cutout_size - cutout_overlap
        center_cutout_size = cutout_size - (2 * cutout_overlap)
        num_cutouts = 2 + (2048 - (2 * edge_cutout_size)) // center_cutout_size
        cutouts = list(product(range(num_cutouts), repeat=2))
    # TODO remove this comment when the issue is resolved
    # Currently we don't actually do multiprocessing here because the catalogs can't be pickled.
    # There is a problem pickling astropy `MaskedQuantity` objects. There was some progress in
    # v7.0.1, and there appears to be an issue tracking the remaining problem:
    # https://github.com/astropy/astropy/issues/16352
    consume_iterator_with_progress_bar(
        pool_map_if_multiprocessing(
            write_source_pickle_from_x_y,
            cutouts,
            nprocs=1,  # TODO change to `nprocs=procs` when issue above is resolved
            pool_map_method="imap_unordered",
        ),
        desc=f"Writing source pickle files for {camera}-{ccd}",
        unit="source",
        total=len(cutouts),
    )

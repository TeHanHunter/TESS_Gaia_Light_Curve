"""
Create cached versions of the TIC and Gaia databases with entries relevant to the current orbit.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from logging import getLogger

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import pandas as pd
import sqlalchemy as sa
import tesswcs

from tglc.databases import TIC, Gaia
from tglc.utils.constants import TESS_CCD_SHAPE, get_sector_containing_orbit
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = getLogger(__name__)

TIC_CATALOG_FIELDS = ["ID", "GAIA", "ra", "dec", "Tmag", "pmRA", "pmDEC", "Jmag", "Kmag", "Vmag"]

GAIA_CATALOG_FIELDS = [
    "DESIGNATION",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ra",
    "dec",
    "pmra",
    "pmdec",
]


def _get_camera_query_grid_centers(sector: int, camera: int, ccd: int) -> SkyCoord:
    """Get centers of 5deg-radius cones that will cover a CCD FOV in a given sector."""
    ra, dec, roll = tesswcs.pointings[tesswcs.pointings["Sector"] == sector][0]["RA", "Dec", "Roll"]
    wcs = tesswcs.WCS.predict(ra, dec, roll, camera, ccd, warp=False)
    ccd_rows, ccd_columns = TESS_CCD_SHAPE
    query_center_ccd_x, query_center_ccd_y = np.meshgrid(
        np.arange(ccd_columns / 4, ccd_columns, ccd_columns / 4, dtype=float),
        np.arange(ccd_rows / 4, ccd_rows, ccd_rows / 4, dtype=float),
    )
    return wcs.pixel_to_world(query_center_ccd_x.ravel(), query_center_ccd_y.ravel())


def _run_tic_cone_query(
    ra_dec: tuple[float, float],
    radius: float = 5.0,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
) -> pd.DataFrame:
    """
    Get results of TIC cone query centered at (ra, dec). All arguments have degree units.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks sky coordinates from first argument.
    """
    ra, dec = ra_dec
    if mdwarf_magnitude_cutoff is None:
        mdwarf_magnitude_cutoff = magnitude_cutoff

    tic = TIC("tic_82")
    tic_entries = tic.table("ticentries")
    dr2_to_dr3 = tic.table("dr2_to_dr3")
    base_query = tic.select("ticentries", *(field.lower() for field in TIC_CATALOG_FIELDS))
    magnitude_filter = tic_entries.c.tmag < magnitude_cutoff
    # M dwarfs: magnitude < 15, T_eff < 4,000K, radius < 0.8 solar radii
    mdwarf_filter = sa.and_(
        tic_entries.c.tmag.between(magnitude_cutoff, mdwarf_magnitude_cutoff),
        tic_entries.c.teff < 4_000,
        tic_entries.c.rad < 0.8,
    )
    tic_cone_query = base_query.where(tic.in_cone("ticentries", ra, dec, width=radius)).where(
        sa.or_(magnitude_filter, mdwarf_filter)
    )

    logger.debug(f"Querying TIC via Pyticdb for {radius:.2f}deg cone around {ra=:.2f}, {dec=:.2f}")
    tic_cone_query_results = tic.execute(tic_cone_query)
    tic_with_gaia_dr2 = pd.DataFrame(
        tic_cone_query_results, columns=[field.lower() for field in TIC_CATALOG_FIELDS]
    )
    non_null_gaia_dr2_source_ids = tic_with_gaia_dr2["gaia"].dropna().astype(int)

    logger.debug(f"Querying Gaia DR2 to DR3 table for stars around {ra=:.2f}, {dec=:.2f}")
    gaia_match_query = tic.select("dr2_to_dr3", "dr2_source_id", "dr3_source_id").where(
        dr2_to_dr3.c.dr2_source_id == sa.any_(non_null_gaia_dr2_source_ids.tolist())
    )
    gaia_match_query_results = tic.execute(gaia_match_query)
    gaia_match = pd.DataFrame(gaia_match_query_results, columns=["dr2_source_id", "dr3_source_id"])
    gaia_match["dr2_source_id"] = gaia_match["dr2_source_id"].astype(str)
    gaia_match["dr3_source_id"] = pd.array(gaia_match["dr3_source_id"]).astype("Int64")

    return (
        tic_with_gaia_dr2.merge(gaia_match, how="left", left_on="gaia", right_on="dr2_source_id")
        .drop(columns=["gaia", "dr2_source_id"])
        .rename(columns={"dr3_source_id": "gaia3"})
    )


def get_tic_catalog_data(
    orbit: int,
    camera: int,
    ccd: int,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
    nprocs: int = 1,
) -> Table:
    """
    Query the TESS Input Catalog for stars in a grid of cones covering the camera during the sector.

    Parameters
    ----------
    orbit, camera, ccd : int
        TESS orbit, camera, and CCD identifying the field of view to create a catalog for.
    magnitude_cutoff : float
        Stars brighter than the magnitude cutoff will be included in the query. Default = 13.5
    mdwarf_magnitude_cutoff : float
        Separate magnitude cutoff for M dwarf stars. If excluded, the main magnitude cutoff will be
        used.
    nprocs : int
        Number of processes to use to distribute queries

    Returns
    -------
    tic_data : Table
        Table containing the TIC catalog fields with appropriate units
    """
    sector = get_sector_containing_orbit(orbit)
    query_grid_centers = _get_camera_query_grid_centers(sector, camera, ccd)
    run_tic_cone_query_with_mag_cutoffs = partial(
        _run_tic_cone_query,
        magnitude_cutoff=magnitude_cutoff,
        mdwarf_magnitude_cutoff=mdwarf_magnitude_cutoff,
    )
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        query_results = executor.map(
            run_tic_cone_query_with_mag_cutoffs,
            zip(query_grid_centers.ra.deg, query_grid_centers.dec.deg, strict=True),
        )
    tic_data = Table.from_pandas(
        pd.concat(results for results in query_results if len(results) > 0).drop_duplicates("id")
    )
    tic_data["ra"].unit = u.deg
    tic_data["dec"].unit = u.deg
    tic_data["pmra"].unit = u.mas / u.yr
    tic_data["pmdec"].unit = u.mas / u.yr
    logger.debug(
        f"Found {len(tic_data)} TIC stars for camera {camera}, CCD {ccd} after applying magnitude "
        f"(<{magnitude_cutoff} Tmag) and M dwarf (<{mdwarf_magnitude_cutoff} Tmag, <4,000K T_eff, "
        "<0.8 solar rad) filters"
    )

    return tic_data


def _run_gaia_cone_query(ra_dec: tuple[float, float], radius: float = 5.0) -> pd.DataFrame:
    """
    Get results of Gaia cone query centered at (ra, dec). All arguments have degree units.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks sky coordinates from first argument.
    """
    ra, dec = ra_dec
    gaia = Gaia("gaia3")
    gaia_cone_query = gaia.query_by_loc(
        "gaia_source",
        ra,
        dec,
        radius,
        *(field.lower() for field in GAIA_CATALOG_FIELDS),
        as_query=True,
    )
    logger.debug(f"Querying Gaia via Pyticdb for {radius:.2f}deg cone around {ra=:.2f}, {dec=:.2f}")
    gaia_cone_query_results = gaia.execute(gaia_cone_query)
    return pd.DataFrame(
        gaia_cone_query_results, columns=[field.lower() for field in GAIA_CATALOG_FIELDS]
    )


def get_gaia_catalog_data(orbit: int, camera: int, ccd: int, nprocs: int = 1) -> Table:
    """
    Query Gaia for stars in a grid of cones covering the camera during the sector.

    Parameters
    ----------
    orbit, camera, ccd : int
        TESS orbit, camera, and CCD identifying the field of view to create a catalog for.
    nprocss : int
        Number of processes to use to distribute queries

    Returns
    -------
    gaia_data : Table
        Table containing the Gaia catalog fields with appropriate units
    """
    sector = get_sector_containing_orbit(orbit)
    query_grid_centers = _get_camera_query_grid_centers(sector, camera, ccd)
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        query_results = executor.map(
            _run_gaia_cone_query,
            zip(query_grid_centers.ra.deg, query_grid_centers.dec.deg, strict=True),
        )
    gaia_data = Table.from_pandas(
        pd.concat(results for results in query_results if len(results) > 0).drop_duplicates(
            "designation"
        )
    )
    gaia_data["ra"].unit = u.deg
    gaia_data["dec"].unit = u.deg
    gaia_data["pmra"].unit = u.mas / u.yr
    gaia_data["pmdec"].unit = u.mas / u.yr
    logger.debug(f"Found {len(gaia_data)} Gaia stars for camera {camera}, CCD {ccd}")
    return gaia_data


def make_tic_and_gaia_catalogs(
    camera_ccd: tuple[int, int],
    orbit: int,
    manifest: Manifest,
    tic_magnitude_limit: float,
    mdwarf_magnitude_limit: float,
    nprocs: int = 1,
    replace: bool = False,
    tic_only: bool = False,
    gaia_only: bool = False,
):
    """
    Make TIC and Gaia catalog files for a camera/CCD in a sector.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks camera and CCD from the first argument.
    """
    camera, ccd = camera_ccd
    manifest.orbit = orbit
    manifest.camera = camera
    manifest.ccd = ccd

    if gaia_only:
        logger.debug(f"Skipping TIC catalog creation for camera {camera} CCD {ccd}")
    elif replace or not manifest.tic_catalog_file.is_file():
        tic_results = get_tic_catalog_data(
            orbit,
            camera,
            ccd,
            magnitude_cutoff=tic_magnitude_limit,
            mdwarf_magnitude_cutoff=mdwarf_magnitude_limit,
            nprocs=nprocs,
        )
        # Astropy's fast ascii writer doesn't work with ecsv by default, but we can write the
        # header and then write the data to get an equivalent file.
        tic_results[:0].write(manifest.tic_catalog_file, overwrite=replace)
        with open(manifest.tic_catalog_file, "a") as tic_output:
            tic_results.write(
                tic_output, format="ascii.fast_no_header", delimiter=" ", strip_whitespace=False
            )
    else:
        logger.info(
            f"TIC catalog at {manifest.tic_catalog_file} already exists and will not be overwritten"
        )

    if tic_only:
        logger.debug(f"Skipping Gaia catalog creation for camera {camera} CCD {ccd}")
    elif replace or not manifest.gaia_catalog_file.is_file():
        gaia_results = get_gaia_catalog_data(orbit, camera, ccd, nprocs=nprocs)
        # Astropy's fast ascii writer doesn't work with ecsv by default, but we can write the
        # header and then write the data to get an equivalent file.
        gaia_results[:0].write(manifest.gaia_catalog_file, overwrite=replace)
        with open(manifest.gaia_catalog_file, "a") as gaia_output:
            gaia_results.write(
                gaia_output, format="ascii.fast_no_header", delimiter=" ", strip_whitespace=False
            )
    else:
        logger.info(
            f"Gaia catalog at {manifest.gaia_catalog_file} already exists and will not be overwritten"
        )


def make_catalog_main(args: argparse.Namespace):
    """
    Create cached versions of the TIC and Gaia databases with entries relevant to the current orbit.
    """
    manifest = Manifest(args.tglc_data_dir)
    manifest.orbit = args.orbit
    manifest.catalog_directory.mkdir(exist_ok=True)

    make_tic_and_gaia_catalogs_for_camera_and_ccd = partial(
        make_tic_and_gaia_catalogs,
        orbit=args.orbit,
        manifest=manifest,
        tic_magnitude_limit=args.max_magnitude,
        mdwarf_magnitude_limit=args.mdwarf_magnitude,
        nprocs=max(args.nprocs // 16, 1),  # Controls how many threads to use for queries
        replace=args.replace,
        tic_only=args.tic_only,
        gaia_only=args.gaia_only,
    )
    consume_iterator_with_progress_bar(
        pool_map_if_multiprocessing(
            make_tic_and_gaia_catalogs_for_camera_and_ccd,
            args.ccd,
            nprocs=min(args.nprocs, 16),
            pool_map_method="imap_unordered",
        ),
        desc=f"Creating catalogs for {len(args.ccd)} CCDs",
        unit="ccd",
        total=len(args.ccd),
    )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )

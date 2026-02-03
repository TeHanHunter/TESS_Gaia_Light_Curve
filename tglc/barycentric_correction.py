# Adapted from QLP: https://github.com/havijw/tess-time-correction/

from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta, TimeFromEpoch
import astropy.units as u
import numpy as np
import pandas as pd

def get_ephemeris_file_path(sector: int) -> Path:
    """Get the path to the appropriate TESS ephemeris data file for a given sector."""
    ephemeris_data_directory = Path(__file__).resolve().parent / "ephemeris_data"
    if 1 <= sector <= 5:
        return ephemeris_data_directory / "20180720_tess_ephem.csv"
    elif 6 <= sector <= 19:
        return ephemeris_data_directory / "20190101_tess_ephem.csv"
    elif 19 <= sector <= 32:
        return ephemeris_data_directory / "20200101_tess_ephem.csv"
    elif 33 <= sector <= 45:
        return ephemeris_data_directory / "20210101_tess_ephem.csv"
    elif 46 <= sector <= 59:
        return ephemeris_data_directory / "20211215_tess_ephem.csv"
    elif 60 <= sector <= 73:
        return ephemeris_data_directory / "20221201_tess_ephem.csv"
    elif 74 <= sector <= 87:
        return ephemeris_data_directory / "20231201_tess_ephem.csv"
    elif 88 <= sector <= 101:
        return ephemeris_data_directory / "20241201_tess_ephem.csv"
    else:
        raise ValueError(
            f"No spacecraft ephemeris file assigned for sector {sector}."
        )

class TESSJD(TimeFromEpoch):
    """
    Define TJD as (JD - 2457000) and reported in units of days.

    Importing this class registers the `"tjd"` format with `astropy.time`.
    """

    name = "tjd"
    unit = 1
    epoch_val = 2457000 * u.day
    epoch_val2 = None
    epoch_scale = "tdb"
    epoch_format = "jd"


def apply_barycentric_correction(
        sector: int, tjd: np.typing.ArrayLike, coord: SkyCoord
) -> np.ndarray:
    """
    Apply barycentric time correction to timestamps in from a given sector.

    Uses data from `ephmeris_data/` for TESS spacecraft position.
    Uses vectorized operations so `tjd` and `coord` can be arrays.

    Parameters
    ----------
    sector : int
        Sector containing the time stamps that need to be corrected
    tjd : ArrayLike
        Timestamps (in days) as recorded on the TESS spacecraft
    coord : SkyCoord
        Sky coordinate(s) of the target star(s) for which correction is being applied

    Returns
    -------
    btjd : Array
        Barycentric JD, TDB timestamps.
        If `coord` is a scalar, the array shape matches `tjd`.
        Otherwise, an axis is added before axis 0 which corresponds to objects.
        For instance, if `tjd` is a 1D array and `coord` is an array, the
        returned array will have 1 row per coordinate.
    """
    input_times = Time(tjd, format="tjd", scale="tdb")

    # Linearly interpolate spacecraft position at timestamps
    ephemeris_data_file = get_ephemeris_file_path(sector)
    tess_ephemeris = pd.read_csv(ephemeris_data_file, comment="#")
    tess_spacecraft_x = np.interp(input_times.jd, tess_ephemeris["JDTDB"], tess_ephemeris["X"])
    tess_spacecraft_y = np.interp(input_times.jd, tess_ephemeris["JDTDB"], tess_ephemeris["Y"])
    tess_spacecraft_z = np.interp(input_times.jd, tess_ephemeris["JDTDB"], tess_ephemeris["Z"])
    tess_spacecraft_position = np.array(
        [tess_spacecraft_x, tess_spacecraft_y, tess_spacecraft_z]
    ).T * u.au

    # Calculate difference in light travel time to TESS vs solar system barycenter
    star_vector = coord.cartesian.xyz
    star_projection = np.dot(tess_spacecraft_position, star_vector).T
    light_travel_time_delta = TimeDelta(
        star_projection.to(u.lightsecond).value * u.second,
        format="jd",
        scale="tdb",
    )
    return (input_times + light_travel_time_delta).tjd

if __name__ == "__main__":
    sector = 70
    # Example MID_TJD timestamps from cadences 855745-855747 in camera 1
    tjd = [3208.350463260291, 3208.352778074038, 3208.355092887784]
    # Example values taken for TIC 2761238 and TIC 8939995
    coord = SkyCoord(
        [(356.485772436, -13.4999191877), (354.410412934, -8.16560533748)],
        unit=u.deg,
    )
    print(apply_barycentric_correction(sector, tjd, coord))

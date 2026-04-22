"""
Provides solar system coordinates for TESS throughout the mission. Data is based on JPL Horizons
predicted ephemerides for TESS. See the [ephemeris data readme](util/data/ephemerides/README.md)
for more information.
"""

from importlib import resources
from pathlib import Path

from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd

from tglc.utils.data import ephemerides


def get_ephemeris_file_path(sector: int) -> Path:
    """Get the path to the appropriate TESS ephemeris data file for a given sector."""
    ephemeris_data_directory = resources.files(ephemerides)
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
    elif 102 <= sector <= 115:
        return ephemeris_data_directory / "20260401_tess_ephem.csv"
    else:
        raise ValueError(f"No spacecraft ephemeris file assigned for sector {sector}.")


def get_tess_spacecraft_position(sector: int, time: Time) -> u.Quantity["length"]:  # noqa: F821
    """Get a function that takes timestamps and returns the TESS spacecraft position."""
    ephemeris = pd.read_csv(get_ephemeris_file_path(sector), comment="#")
    spacecraft_x = np.interp(time.jd, ephemeris["JDTDB"], ephemeris["X"])
    spacecraft_y = np.interp(time.jd, ephemeris["JDTDB"], ephemeris["Y"])
    spacecraft_z = np.interp(time.jd, ephemeris["JDTDB"], ephemeris["Z"])
    return np.array([spacecraft_x, spacecraft_y, spacecraft_z]).T * u.au

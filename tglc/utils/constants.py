"""
Astronomical constants and conversions used by TGLC, mostly related to TESS.
"""

from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.time.formats import TimeFromEpoch
import astropy.units as u
import numpy as np
import numpy.typing as npt


TESS_CCD_SHAPE = (2078, 2136)
"""The shape of an array of data from a CCD on TESS: 2078 rows, 2136 columns."""


TESS_PIXEL_SCALE = u.pixel_scale(0.35 * u.arcmin / u.pixel)
"""
Astropy units equivalency for TESS pixels taken from Ricker et al, 2014, S4.1, table 1.

See <https://doi.org/10.1117/1.JATIS.1.1.014003>.
"""

TESS_PIXEL_SATURATION_LEVEL = 2e5 * u.electron
"""
TESS pixel saturation level, from the TESS Instrument Handbook, p37.

See <https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf#page=38>.
"""


def convert_tess_flux_to_tess_magnitude(flux: u.Quantity) -> npt.ArrayLike:
    """
    Convert TESS flux values (e-/s) to TESS magnitude.

    Conversion is based on the reference flux of 15,000 e-/s for a star of TESS magnitude 10 given in
    the TESS Instrument Handbook, p. 37. The conversion is therefore given by
    $$
        m = -2.5 \\log_{10}(F / 15,000) + 10
    $$
    See <https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf#page=38>.
    """
    return (-2.5 * np.log10(flux / (15_000 * u.electron / u.second)) + 10).value


def convert_tess_magnitude_to_tess_flux(magnitude: npt.ArrayLike) -> u.Quantity:
    """
    Convert TESS magnitude to TESS flux values (e-/s).

    Conversion is based on the reference flux of 15,000 e-/s for a star of TESS magnitude 10 given in
    the TESS Instrument Handbook, p. 37. The conversion is therefore given by
    $$
        F = 15,000 \\cdot \\exp_{10}((m - 10) / -2.5)\\ e^-/s
    $$
    See <https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf#page=38>.
    """
    return (15_000 * u.electron / u.second) * 10 ** ((magnitude - 10) / -2.5)


def get_exposure_time_from_sector(sector: int) -> u.Quantity:
    """Get exposure time (in seconds) for the given sector."""
    if sector <= 0:
        raise ValueError(f"No exposure time for sector {sector} - TESS sectors start at 1.")
    if sector < 27:
        # Primary mission
        return 1800 * u.second
    elif sector < 56:
        # First extended mission
        return 600 * u.second
    else:
        # Second extended mission and beyond
        return 200 * u.second


def get_sector_containing_orbit(orbit: int) -> int:
    """Get the TESS sector containing a TESS orbit."""
    if 9 <= orbit <= 200:
        return (orbit - 7) // 2
    elif 201 <= orbit <= 204:
        return 97
    elif 205 <= orbit <= 208:
        return 98
    elif 209 <= orbit <= 226:
        return (orbit - 11) // 2
    else:
        raise ValueError(f"Sector not known for orbit {orbit}")


def get_orbits_in_sector(sector: int) -> list[int]:
    """Get the TESS orbits in a TESS sector."""
    if 1 <= sector <= 96:
        return [sector * 2 + 7, sector * 2 + 8]
    elif sector == 97:
        return [sector * 2 + 7, sector * 2 + 8, sector * 2 + 9, sector * 2 + 10]
    elif sector == 98:
        return [sector * 2 + 9, sector * 2 + 10, sector * 2 + 11, sector * 2 + 12]
    elif 99 <= sector <= 107:
        return [sector * 2 + 11, sector * 2 + 12]
    else:
        raise ValueError(f"Orbits not known for sector {sector}")


def convert_gaia_mags_to_tmag(
    G: npt.ArrayLike, Gbp: npt.ArrayLike, Grp: npt.ArrayLike
) -> np.ma.MaskedArray:
    """
    Convert Gaia magnitudes to Tmag based on the conversion in Stassun et al, 2019, S2.3.1, eq 1.

    See <https://doi.org/10.3847/1538-3881/ab3467>.

    When G_bp and G_rp are available (as indicated by masked arrays), the formula used is

    $$
        T = G - 0.00522555(G_bp - G_rp)^3
            + 0.0891337(G_bp - G_rp)^2
            - 0.633923(G_bp - G_rp)
            + 0.0324473
    $$

    When G_bp or G_rp is not available, the formula used is

    $$
        T = G - 0.430
    $$

    Parameters
    ----------
    G : ArrayLike
        Gaia G passband magnitudes. Masked arrays are supported.
    Gbp : ArrayLike
        Gaia G_bp passband magnitudes. Masked arrays are supported.
    Grp : ArrayLike
        Gaia G_rp passband magnitudes. Masked arrays are supported.

    Returns
    -------
    T : MaskedArray
        Converted Tmag values. Masked where `G` input is masked.
    """
    br_difference = Gbp - Grp
    nominal_conversion = (
        G
        - 0.00522555 * (br_difference**3)
        + 0.0891337 * (br_difference**2)
        - 0.633923 * br_difference
        + 0.0324473
    )

    no_br_conversion = G - 0.430
    br_available = (
        ~br_difference.mask if np.ma.isMaskedArray(br_difference) else np.isfinite(br_difference)
    )
    return np.ma.where(br_available, nominal_conversion, no_br_conversion)


class TESSJD(TimeFromEpoch):
    """
    Astropy time format for TESS Julian Date, TJD = JD - 2457000, reported in TDB.

    Importing this class registers the `"tjd"` format with `astropy.time`.
    """

    name = "tjd"
    unit = 1
    epoch_val = 2457000 * u.day
    epoch_val2 = None
    epoch_scale = "tdb"
    epoch_format = "jd"


def apply_barycentric_correction(
    tjd: npt.ArrayLike,
    coord: SkyCoord,
    spacecraft_position: u.Quantity["length"],  # noqa: F821
) -> Time:
    """
    Apply barycentric time correction to TESS spacecraft timestamps.

    Parameters
    ----------
    tjd : Time
        Timestamps as recorded on TESS spacecraft.
    coord : SkyCoord
        Sky coordinates of target star(s) for which correction is being applied.
    spacecraft_position : Quantity["length"]
        (x, y, z) position of spacecraft relative to solar system barycenter at each time stamp.
        Length should match `tjd`.

    Returns
    -------
    btjd : Time
        Barycentric JD, TDB timestamps.
        If `coord` is a scalar, an array matching the shape of `tjd` is returned.
        If `coord` is an array, a 2D array with one row per object is returned.
    """
    star_vector = coord.cartesian.xyz

    star_projection = np.dot(spacecraft_position, star_vector)
    light_time = TimeDelta(
        star_projection.to(u.lightsecond).value * u.second,
        format="jd",
        scale="tdb",
    )
    if len(coord.shape) == 0:
        # Scalar coordinate
        return tjd + light_time
    return tjd[:, np.newaxis] + light_time

"""
Tests for the tglc.util.constants module, which provides common astronomical constants and
conversions, mostly related to TESS.
"""

from astropy.time import Time
import astropy.units as u
import numpy as np
import pytest

from tglc.utils.constants import (
    TESS_PIXEL_SATURATION_LEVEL,
    TESS_PIXEL_SCALE,
    convert_gaia_mags_to_tmag,
    convert_tess_flux_to_tess_magnitude,
    convert_tess_magnitude_to_tess_flux,
    get_exposure_time_from_sector,
    get_orbits_in_sector,
    get_sector_containing_orbit,
)


def test_tess_pixel_scale():
    single_pixel = 1 * u.pix
    assert single_pixel.to(u.arcmin, equivalencies=TESS_PIXEL_SCALE).value == 0.35
    assert np.isclose(single_pixel.to(u.arcsec, equivalencies=TESS_PIXEL_SCALE).value, 21)
    pixel_on_sky = 0.35 * u.arcmin
    assert pixel_on_sky.to(u.pix, equivalencies=TESS_PIXEL_SCALE).value == 1
    assert np.isclose(pixel_on_sky.to(u.arcsec).to(u.pix, equivalencies=TESS_PIXEL_SCALE).value, 1)
    # Each camera has a 24deg x 24deg field of view and has a 2x2 mosaic of CCD detectors, each a
    # 2048x2048 grid, for an effective 4096x4096 grid. There are gaps and the grid does not exactly
    # correspond to the field of view, so we give this check some tolerance.
    detector_size = 2 * 2048 * u.pix
    assert np.isclose(detector_size.to(u.deg, equivalencies=TESS_PIXEL_SCALE).value, 24, 0.15)


def test_tess_pixel_saturation_level():
    # Mostly testing for unit compatibility
    assert (2e5 + 1) * u.electron > TESS_PIXEL_SATURATION_LEVEL


def test_convert_tess_flux_to_tess_magnitude():
    assert convert_tess_flux_to_tess_magnitude(15_000 * u.electron / u.second) == 10
    assert np.isclose(convert_tess_flux_to_tess_magnitude(1 * u.electron / u.second), 20.44, 0.01)


def test_convert_tess_magnitude_to_tess_flux():
    assert convert_tess_magnitude_to_tess_flux(10) == 15_000 * u.electron / u.second
    assert np.isclose(convert_tess_magnitude_to_tess_flux(20.44), 1 * u.electron / u.second, 0.01)


def test_get_expossure_time_from_sector():
    assert get_exposure_time_from_sector(1) == 1800 * u.second
    assert get_exposure_time_from_sector(26) == 1800 * u.second
    assert get_exposure_time_from_sector(27) == 600 * u.second
    assert get_exposure_time_from_sector(55) == 600 * u.second
    assert get_exposure_time_from_sector(56) == 200 * u.second
    assert get_exposure_time_from_sector(100) == 200 * u.second


@pytest.mark.parametrize("bad_sector", [0, -1])
def test_get_exposure_time_from_sector_with_invalid_sector(bad_sector: int):
    with pytest.raises(ValueError):
        get_exposure_time_from_sector(bad_sector)


def test_get_sector_containing_orbit():
    assert get_sector_containing_orbit(9) == 1
    assert get_sector_containing_orbit(10) == 1
    assert get_sector_containing_orbit(11) == 2
    assert get_sector_containing_orbit(199) == 96
    assert get_sector_containing_orbit(200) == 96
    # 4-orbit sectors
    assert get_sector_containing_orbit(201) == 97
    assert get_sector_containing_orbit(204) == 97
    assert get_sector_containing_orbit(205) == 98
    assert get_sector_containing_orbit(208) == 98
    # Back to 2-orbit sectors
    assert get_sector_containing_orbit(209) == 99
    assert get_sector_containing_orbit(211) == 100
    assert get_sector_containing_orbit(225) == 107
    assert get_sector_containing_orbit(226) == 107


@pytest.mark.parametrize("bad_orbit", [0, -1, 1, 8, 227, 300])
def test_get_sector_containing_orbit_with_invalid_orbit(bad_orbit: int):
    with pytest.raises(ValueError):
        get_sector_containing_orbit(bad_orbit)


def test_get_orbits_in_sector():
    assert get_orbits_in_sector(1) == [9, 10]
    assert get_orbits_in_sector(2) == [11, 12]
    assert get_orbits_in_sector(96) == [199, 200]
    assert get_orbits_in_sector(97) == [201, 202, 203, 204]
    assert get_orbits_in_sector(98) == [205, 206, 207, 208]
    assert get_orbits_in_sector(99) == [209, 210]
    assert get_orbits_in_sector(107) == [225, 226]


@pytest.mark.parametrize("bad_sector", [0, -1, 108, 150])
def test_get_orbits_in_sector_with_invalid_sector(bad_sector: int):
    with pytest.raises(ValueError):
        get_orbits_in_sector(bad_sector)


@pytest.mark.parametrize(
    ["sector", "orbits"],
    [
        (1, [9, 10]),
        (2, [11, 12]),
        (96, [199, 200]),
        (97, [201, 202, 203, 204]),
        (98, [205, 206, 207, 208]),
        (99, [209, 210]),
        (107, [225, 226]),
    ],
)
def test_orbits_sector_round_trip(sector: int, orbits: list[int]):
    for orbit in get_orbits_in_sector(sector):
        assert get_sector_containing_orbit(orbit) == sector
    for orbit in orbits:
        assert orbit in get_orbits_in_sector(get_sector_containing_orbit(orbit))


def test_convert_gaia_mags_to_tmag_no_masks():
    # In the big parametrized test, all the objects are masked arrays, some of which just have the
    # masks as all false. This tests that the function works when normal arrays are passed
    G = np.array([10.0, 12.0, 14.0])
    Gbp = np.array([10.5, 12.5, 14.5])
    Grp = np.array([9.5, 11.5, 13.5])
    expected = np.array([9.48243245, 11.48243245, 13.48243245])
    result = convert_gaia_mags_to_tmag(G, Gbp, Grp)
    np.testing.assert_almost_equal(result, expected, decimal=5)
    assert not np.ma.is_masked(result)


# This should test every combination of having/not having masks
# Result values depend on Gbp, Grp masks
# Result mask depends on G mask
@pytest.mark.parametrize(
    "G_mask,Gbp_mask,Grp_mask,expected",
    [
        (None, None, None, [9.48243245, 11.48243245, 13.48243245]),
        (None, [True, True, True], [True, True, True], [9.57, 11.57, 13.57]),
        (None, [False, True, False], [False, True, False], [9.48243245, 11.57, 13.48243245]),
        (None, [False, True, False], [True, False, False], [9.57, 11.57, 13.48243245]),
        (None, [False, False, True], None, [9.48243245, 11.48243245, 13.57]),
        (None, None, [False, False, True], [9.48243245, 11.48243245, 13.57]),
        ([True, False, False], None, None, [9.48243245, 11.48243245, 13.48243245]),
        ([True, False, False], [False, True, False], None, [9.48243245, 11.57, 13.48243245]),
        ([True, False, False], None, [False, True, False], [9.48243245, 11.57, 13.48243245]),
        (
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [9.57, 11.48243245, 13.48243245],
        ),
    ],
)
def test_convert_gaia_mags_to_tmag(G_mask, Gbp_mask, Grp_mask, expected):
    G = np.ma.masked_array([10.0, 12.0, 14.0], mask=G_mask)
    Gbp = np.ma.masked_array([10.5, 12.5, 14.5], mask=Gbp_mask)
    Grp = np.ma.masked_array([9.5, 11.5, 13.5], mask=Grp_mask)
    # Mask of result should match mask of G, since all calculations involve G, and calculations can
    # be done using only G if other values are missing.
    expected = np.ma.masked_array(expected, mask=G_mask)
    result = convert_gaia_mags_to_tmag(G, Gbp, Grp)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    # When the mask is all False (i.e., no values masked), the mask produced by the function is
    # sometimes a scalar False instead of an array full of False. Using np.array_equiv instead of
    # np.array_equal allows broadcasting
    assert np.array_equiv(result.mask, expected.mask)


def test_tessjd_format():
    tjd = Time(2457000.0, format="jd", scale="tdb")
    assert tjd.tjd == 0.0
    tjd2 = Time(0.0, format="tjd", scale="tdb")
    assert tjd2.jd == 2457000.0

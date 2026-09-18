import numpy as np
import pytest

from tglc.pixel_quality import build_pixel_mask


def test_rate_threshold_does_not_depend_on_brightest_pixel_or_ffi_duration():
    flux = np.array([[100., 50000., 79999., 80000., 100000.]])
    np.testing.assert_array_equal(
        build_pixel_mask(flux, saturation_dilation=0), [[False, False, False, True, True]]
    )
    assert not build_pixel_mask(flux[:, :3], saturation_dilation=0).any()


def test_saturation_grows_neighbors_but_preserves_other_pixels():
    flux = np.ones((7, 7))
    flux[3, 3] = 90000
    mask = build_pixel_mask(flux)
    assert mask.sum() == 5
    assert mask[3, 3] and mask[2, 3] and mask[3, 2]
    assert not mask[0, 0]


def test_missing_zero_and_explicit_mask_are_preserved_when_saturation_disabled():
    flux = np.array([[np.nan, np.inf, 0.], [-10., 90000., 1.]])
    base = np.array([[False, False, False], [False, False, True]])
    np.testing.assert_array_equal(build_pixel_mask(flux, base, saturation_limit=None),
                                  [[True, True, True], [False, False, True]])


@pytest.mark.parametrize('limit', [0, -1, np.nan, np.inf])
def test_bad_threshold_rejected(limit):
    with pytest.raises(ValueError):
        build_pixel_mask(np.ones((2, 2)), saturation_limit=limit)


def test_bad_mask_shape_rejected():
    with pytest.raises(ValueError):
        build_pixel_mask(np.ones((2, 2)), np.zeros((3, 3)))

"""Independent numerical invariants for the public TGLC PSF implementation."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator

from tglc.effective_psf import (
    bg_mod, fit_lc, fit_lc_float_field, fit_psf, get_psf,
)
from tglc.pixel_quality import build_pixel_mask


def source_at(x=15.2, y=15.3, size=32, cadences=3):
    return SimpleNamespace(
        size=size, sector=1,
        gaia=Table({"sector_1_x": np.array([x]), "sector_1_y": np.array([y]),
                    "tess_flux_ratio": np.array([1.]), "tess_mag": np.array([10.])}),
        flux=np.ones((cadences, size, size)), time=np.arange(cadences, dtype=float),
        mask=np.ma.array(np.ones((size, size)), mask=np.zeros((size, size), dtype=bool)),
    )


@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('phase', [0., .13, .49, .5, .51, .87, 1.])
def test_interpolation_matches_independent_asymmetric_kernel(factor, phase):
    source = source_at(15 + phase, 15 - phase)
    matrix, _, over, xr, yr = get_psf(source, factor=factor)
    kernel = np.random.default_rng(31).normal(size=(over, over))
    fitted = (matrix[:source.size ** 2, :over ** 2] @ kernel.ravel()).reshape(source.size, source.size)
    xx, yy = np.meshgrid(np.arange(source.size), np.arange(source.size))
    points = np.stack(((yy - (15 - phase)) * factor + over // 2,
                       (xx - (15 + phase)) * factor + over // 2), axis=-1)
    oracle = RegularGridInterpolator((np.arange(over), np.arange(over)), kernel,
                                     bounds_error=False, fill_value=0)(points)
    support = (np.abs(xx - xr[0]) <= 5) & (np.abs(yy - yr[0]) <= 5)
    oracle[~support] = 0
    np.testing.assert_allclose(fitted, oracle, atol=2e-13)


def gaussian_scene(x=15.2, y=15.3, cadences=3):
    source = source_at(x, y, cadences=cadences)
    matrix, star_info, over, xr, yr = get_psf(source)
    yy, xx = np.indices((over, over))
    kernel = 1000 * np.exp(-((xx - over // 2) ** 2 + (yy - over // 2) ** 2) / 8.)
    parameters = np.concatenate((kernel.ravel(), [0., 0., 100.]))
    epsf = np.repeat(parameters[None, :], cadences, axis=0)
    source.flux[:] = (matrix[:source.size ** 2] @ parameters).reshape(source.size, source.size)
    return source, matrix, star_info, over, xr, yr, epsf


def test_centroid_moves_with_source_and_is_continuous_across_half_pixel():
    centroids = []
    for x in [15., 15.2, 15.49, 15.51]:
        source, matrix, _, over, _, _, epsf = gaussian_scene(x=x, y=15.)
        flux = (matrix[:source.size ** 2, :over ** 2] @ epsf[0, :over ** 2]).reshape(source.size, source.size)
        centroids.append(np.sum(flux * np.arange(source.size)[None, :]) / flux.sum())
    np.testing.assert_allclose(centroids, [15., 15.2, 15.49, 15.51], atol=5e-6)


def test_edge_compression_is_centered_on_the_psf_grid():
    source = source_at()
    matrix, _, over, _, _ = get_psf(source)
    penalties = np.diag(matrix[source.size**2:, :over**2]).reshape(over, over)
    np.testing.assert_array_equal(penalties, penalties[::-1])
    np.testing.assert_array_equal(penalties, penalties[:, ::-1])
    assert penalties[over//2, over//2] == 0


@pytest.mark.parametrize('x', [-6., -5.2, -.2, 0., 31.2, 36.2, 37.])
def test_outside_neighbor_support_is_clipped_without_wraparound(x):
    source = source_at(x=x)
    matrix, _, over, xr, yr = get_psf(source)
    kernel = np.random.default_rng(87).normal(size=(over, over))
    actual = (matrix[:source.size**2, :over**2] @ kernel.ravel()).reshape(source.size, source.size)
    xx, yy = np.meshgrid(np.arange(source.size), np.arange(source.size))
    points = np.stack(((yy-15.3)*2+over//2, (xx-x)*2+over//2), axis=-1)
    expected = RegularGridInterpolator((np.arange(over), np.arange(over)), kernel,
                                       bounds_error=False, fill_value=0)(points)
    expected[(np.abs(xx-xr[0]) > 5) | (np.abs(yy-yr[0]) > 5)] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_invalid_pixels_and_saturation_are_excluded_from_epsf_fit():
    source = source_at(size=3, cadences=1)
    xx = np.arange(9.)
    matrix = np.column_stack((np.ones(9), xx))
    matrix = np.vstack((matrix, [0., 0.]))
    source.flux[0] = (10 + 2 * xx).reshape(3, 3)
    source.flux[0, 0, 0] = np.nan
    source.flux[0, 0, 1] = 90000
    source.flux[0, 0, 2] = 0
    fitted = fit_psf(matrix, source, 1, saturation_dilation=0)
    np.testing.assert_allclose(fitted, [10., 2.], atol=1e-12)


def test_rank_deficient_and_missing_epsf_fits_remain_missing():
    source = source_at(size=3, cadences=1)
    matrix = np.ones((10, 2))
    matrix[-1] = 0
    assert np.isnan(fit_psf(matrix, source, 1)).all()
    source.flux[:] = np.nan
    assert np.isnan(fit_psf(matrix, source, 1)).all()


@pytest.mark.parametrize('cube_mask', [False, True])
def test_explicit_source_pixel_mask_excludes_contamination(cube_mask):
    source = source_at(size=3, cadences=1)
    xx = np.arange(9.)
    matrix = np.vstack((np.column_stack((np.ones(9), xx)), [0., 0.]))
    source.flux[0] = (10 + 2 * xx).reshape(3, 3)
    source.flux[0, 1, 1] = 10000
    source.pixel_mask = np.zeros(source.flux.shape if cube_mask else source.flux.shape[1:], dtype=bool)
    source.pixel_mask[..., 1, 1] = True
    np.testing.assert_allclose(fit_psf(matrix, source, 1), [10., 2.], atol=1e-12)


def test_target_psf_fit_uses_saturation_mask_without_deleting_aperture_pixels():
    source, matrix, info, _, xr, yr, epsf = gaussian_scene()
    reference = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf)
    source.flux[:, yr[0], xr[0]] = 1e6
    masked = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf, saturation_dilation=0)
    unmasked = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf, saturation_limit=None)
    np.testing.assert_allclose(masked[1], reference[1], rtol=1e-12)
    assert np.max(np.abs(unmasked[1] / reference[1] - 1)) > 1
    assert np.max(masked[0]) > 900000  # raw aperture data stay available for separate photometry


def test_missing_first_frame_does_not_remove_other_cadences():
    source, matrix, info, _, xr, yr, epsf = gaussian_scene()
    source.flux[0] = np.nan
    aperture, psf, *_ = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf)
    assert np.isnan(aperture[0]).all() and np.isnan(psf[0])
    assert np.isfinite(psf[1:]).all()


def test_aperture_fraction_uses_full_psf_support():
    source, matrix, info, over, xr, yr, epsf = gaussian_scene()
    result = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf)
    full_image = (matrix[:source.size ** 2, :over ** 2] @ epsf[0, :over ** 2]).reshape(source.size, source.size)
    expected = full_image[yr[0]-1:yr[0]+2, xr[0]-1:xr[0]+2].sum() / full_image.sum()
    np.testing.assert_allclose(result[4], expected)


def test_cutout_edge_fraction_keeps_flux_outside_image_in_denominator():
    source, matrix, info, over, xr, yr, epsf = gaussian_scene(x=.2)
    result = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf, near_edge=True)
    kernel = epsf[0, :over ** 2].reshape(over, over)
    grid = RegularGridInterpolator((np.arange(over), np.arange(over)), kernel)
    xx, yy = np.meshgrid(np.arange(-5, 6) + xr[0], np.arange(-5, 6) + yr[0])
    full_model = grid(np.stack(((yy-15.3)*2+over//2, (xx-.2)*2+over//2), axis=-1))
    included = (xx >= 0) & (np.abs(xx-xr[0]) <= 1) & (np.abs(yy-yr[0]) <= 1)
    np.testing.assert_allclose(result[4], full_model[included].sum()/full_model.sum())


def test_injected_fractional_depth_survives_catalog_background_normalization():
    source, matrix, info, over, xr, yr, epsf = gaussian_scene(cadences=50)
    source.time = np.arange(50) * 200 / 86400
    target = (matrix[:source.size**2, :over**2] @ epsf[0, :over**2]).reshape(source.size, source.size)
    scale = 15000 / target.sum()
    epsf[:, :over**2] *= scale
    target *= scale
    source.flux[:] = target + 100
    source.flux[20:25] -= .01 * target
    aperture, psf, sy, sx, portion, *_ = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=epsf)
    aperture_flux = aperture[:, sy-1:sy+2, sx-1:sx+2].sum(axis=(1, 2))
    result = bg_mod(source, aper_lc=aperture_flux, psf_lc=psf, portion=portion)
    for flux in result[1:3]:
        recovered_depth = 1 - np.median(flux[20:25]) / np.median(flux[:20])
        np.testing.assert_allclose(recovered_depth, .01, atol=2e-14)


def test_float_field_target_mask_handles_single_star_without_prior_index_error():
    source, matrix, info, _, xr, yr, epsf = gaussian_scene()
    reference = fit_lc_float_field(matrix, source, info, xr, yr, e_psf=epsf)
    source.flux[:, yr[0], xr[0]] = 1e6
    masked = fit_lc_float_field(matrix, source, info, xr, yr, e_psf=epsf, saturation_dilation=0)
    np.testing.assert_allclose(masked[1], reference[1], rtol=1e-12)


def test_background_offsets_are_distinct_and_negative_flux_is_preserved():
    source = source_at(cadences=50)
    aperture = np.full(50, 100.)
    psf = np.full(50, 200.)
    aperture[1] = -20000
    psf[1] = -30000
    result = bg_mod(source, aper_lc=aperture, psf_lc=psf, portion=.5, return_offsets=True)
    np.testing.assert_allclose(result[0], [-7400., -14800.])
    assert result[1][1] == -12600 and result[2][1] == -15200


def test_empty_normalization_reference_does_not_invent_good_flux():
    source = source_at(cadences=3)
    result = bg_mod(source, q=np.zeros(3, dtype=bool), aper_lc=np.ones(3), psf_lc=np.ones(3),
                    portion=.5, return_offsets=True)
    assert np.isnan(result[0]).all()
    assert np.isnan(result[1]).all() and np.isnan(result[2]).all()


def test_wholly_missing_flux_stays_missing_through_background_normalization():
    source = source_at(cadences=3)
    result = bg_mod(source, aper_lc=np.full(3, np.nan), psf_lc=np.full(3, np.nan), portion=.5,
                    return_offsets=True)
    assert all(np.isnan(item).all() for item in result)


def independent_crowded_scene(width=1.2, target=(15.13, 15.77), cadences=40, dense=False):
    """Build detector images with SciPy's interpolator, independent of TGLC's matrix."""
    source = source_at(*target, cadences=cadences)
    positions = [target, (target[0] + 2.37, target[1] - 1.29), (7.12, 8.41), (24.21, 23.67)]
    ratios = [1., 3., .7, 1.4]
    if dense:
        positions += [(6.61, 24.14), (23.19, 6.57), (10.41, 18.93), (18.11, 10.68),
                      (8.81, 5.25), (26.71, 16.43), (4.36, 15.62), (18.31, 26.23)]
        ratios += [.4, 1.8, .9, 2., .3, .8, 1.1, .6]
    positions, ratios = np.array(positions), np.array(ratios)
    source.gaia = Table({'sector_1_x': positions[:, 0], 'sector_1_y': positions[:, 1],
                         'tess_flux_ratio': ratios, 'tess_mag': 10 - 2.5 * np.log10(ratios)})
    over = 23
    ky, kx = np.indices((over, over))
    # A shifted secondary component makes the kernel asymmetric as well as phase dependent.
    kernel = (np.exp(-((kx-11)**2+(ky-11)**2)/(8*width**2))
              + .15*np.exp(-((kx-12.3)**2+(ky-10.2)**2)/(8*(.7*width)**2)))
    oracle = RegularGridInterpolator((np.arange(over), np.arange(over)), kernel,
                                     bounds_error=False, fill_value=0)
    full_x, full_y = np.meshgrid(np.arange(-5, 6)+round(target[0]), np.arange(-5, 6)+round(target[1]))
    full_target = oracle(np.stack(((full_y-target[1])*2+11, (full_x-target[0])*2+11), axis=-1))
    kernel *= 15000 / full_target.sum()
    oracle = RegularGridInterpolator((np.arange(over), np.arange(over)), kernel,
                                     bounds_error=False, fill_value=0)
    yy, xx = np.indices((source.size, source.size))
    images = []
    for (x, y), ratio in zip(positions, ratios):
        scene = oracle(np.stack(((yy-y)*2+11, (xx-x)*2+11), axis=-1)) * ratio
        scene[(np.abs(xx-round(x)) > 5) | (np.abs(yy-round(y)) > 5)] = 0
        images.append(scene)
    background = .2*(xx-15.5) - .3*(yy-15.5) + 100
    source.flux[:] = np.sum(images, axis=0) + background
    source.time = np.arange(cadences) * 200 / 86400
    parameters = np.r_[kernel.ravel(), [.2, -.3, 100.]]
    epsf = np.repeat(parameters[None, :], cadences, axis=0)
    matrix, info, _, xr, yr = get_psf(source)
    np.testing.assert_allclose((matrix[:source.size**2] @ parameters).reshape(source.size, source.size),
                               source.flux[0], atol=5e-11)
    return source, matrix, info, xr, yr, epsf, np.array(images)


@pytest.mark.parametrize('width', [.7, 1.2, 1.8])
@pytest.mark.parametrize('target', [(15.13, 15.77), (15.49, 15.51), (.13, 15.77)])
def test_crowded_depth_and_psf_masks_across_width_phase_and_edge(width, target):
    source, matrix, info, xr, yr, epsf, images = independent_crowded_scene(width, target)
    source.flux[20:25] -= .01 * images[0]
    source.flux[2] = np.nan
    source.flux[4, yr[0], xr[0]] = 1e6
    source.pixel_mask = np.zeros(source.flux.shape, dtype=bool)
    source.flux[5, yr[0]+1, xr[0]+1] = -10000
    source.pixel_mask[5, yr[0]+1, xr[0]+1] = True
    near_edge = xr[0] < 2
    aperture, psf, sy, sx, portion, *_ = fit_lc(
        matrix, source, info, xr[0], yr[0], e_psf=epsf, near_edge=near_edge,
        saturation_dilation=1,
    )
    aperture_flux = aperture[:, max(0, sy-1):sy+2, max(0, sx-1):sx+2].sum(axis=(1, 2))
    good = np.ones(len(source.time), dtype=bool)
    good[[2, 4, 5]] = False
    result = bg_mod(source, q=good, aper_lc=aperture_flux, psf_lc=psf, portion=portion,
                    near_edge=near_edge)
    assert np.isnan(aperture[2]).all() and np.isnan(psf[2])
    np.testing.assert_allclose(1-np.median(result[1][20:25])/np.median(result[1][10:20]), .01,
                               atol=5e-13)
    if near_edge:
        assert np.isnan(psf).all()
    else:
        np.testing.assert_allclose(1-np.median(result[2][20:25])/np.median(result[2][10:20]), .01,
                                   atol=5e-13)
        np.testing.assert_allclose(psf[[4, 5]], 15000, rtol=1e-11)


def test_crowded_epsf_refit_recovers_independent_static_scene_with_invalid_pixels():
    source, matrix, info, xr, yr, _, _ = independent_crowded_scene(cadences=2, dense=True)
    expected = source.flux.copy()
    source.flux[:, 0, 0] = np.nan
    source.flux[1, 8, 7] = 1e6
    fitted = np.array([fit_psf(matrix, source, 23, power=1.4, time=i) for i in range(2)])
    assert np.isfinite(fitted).all()
    model = (matrix[:source.size**2] @ fitted.T).T.reshape(source.flux.shape)
    for i in range(2):
        usable = ~build_pixel_mask(source.flux[i])
        relative_residual = np.linalg.norm((model[i]-expected[i])[usable]) / np.linalg.norm(expected[i][usable])
        assert relative_residual < 1e-4
    _, flux, *_ = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=fitted)
    np.testing.assert_allclose(flux, 15000, rtol=1e-3)


@pytest.mark.xfail(strict=True, reason=(
    'Known scientific limitation: refitting a shared ePSF can absorb target variability and '
    'change neighbor subtraction; the independently injected 1% depth is not preserved.'
))
def test_crowded_variable_target_depth_when_shared_epsf_is_refitted():
    source, matrix, info, xr, yr, _, images = independent_crowded_scene(cadences=2, dense=True)
    source.flux[1] -= .01 * images[0]
    source.flux[:, 0, 0] = np.nan
    fitted = np.array([fit_psf(matrix, source, 23, power=1.4, time=i) for i in range(2)])
    assert np.isfinite(fitted).all()
    aperture, psf, sy, sx, portion, *_ = fit_lc(matrix, source, info, xr[0], yr[0], e_psf=fitted)
    aperture_flux = aperture[:, sy-1:sy+2, sx-1:sx+2].sum(axis=(1, 2))
    result = bg_mod(source, q=np.array([True, False]), aper_lc=aperture_flux, psf_lc=psf, portion=portion)
    depths = np.array([1-flux[1]/flux[0] for flux in result[1:3]])
    # Scientific target: no more than 1% relative depth distortion in this noiseless scene.
    np.testing.assert_allclose(depths, .01, atol=1e-4, rtol=0)

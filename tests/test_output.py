from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from tglc.processing import PROCESSING_VERSION
from tglc.target_lightcurve import epsf, lc_output


def test_fits_preserves_flags_signed_raw_flux_and_offsets(tmp_path):
    source = SimpleNamespace(
        sector=56, camera=1, ccd=2, size=10, time=np.array([100., 100.1, 100.2]),
        flux=np.ones((3, 10, 10)),
        gaia=Table({'DESIGNATION': ['Gaia DR3 42'], 'ra': [100.], 'dec': [20.],
                    'tess_mag': [12.], 'phot_g_mean_mag': [12.],
                    'phot_bp_mean_mag': [13.], 'phot_rp_mean_mag': [11.]}),
        tic=Table({'TIC': [123], 'dr3_source_id': [42]}))
    raw = np.array([-4., 10., np.nan])
    flags = np.array([32768, 65536, 1048576], dtype=np.int32)
    path = lc_output(source, local_directory=tmp_path, time=source.time,
                     psf_lc=raw - 2, aper_lc=raw - 3, cal_psf_lc=raw, cal_aper_lc=raw,
                     raw_aper_lc=raw, raw_psf_lc=raw, local_bg=3., psf_local_bg=2.,
                     portion=0.8, bg=np.ones(3), tess_flag=flags, tglc_flag=[0, 4, 10],
                     cadence=np.arange(3), target_5x5=None, field_stars_5x5=None,
                     fit_fingerprint='a' * 64)
    with fits.open(path) as hdus:
        table = hdus[1].data
        np.testing.assert_array_equal(table['TESS_flags'], flags)
        np.testing.assert_array_equal(table['TGLC_flags'], [0, 4, 10])
        np.testing.assert_allclose(table['aperture_flux_raw'], raw, equal_nan=True)
        np.testing.assert_allclose(table['aperture_flux'], (raw - hdus[1].header['LOC_BG']) /
                                   hdus[1].header['PORTION'], equal_nan=True)
        np.testing.assert_allclose(table['psf_flux'], raw - hdus[1].header['PSF_BG'], equal_nan=True)
        assert hdus[0].header['PROCVER'] == PROCESSING_VERSION
        assert hdus[0].header['NEXTEND'] == 2
        assert hdus[1].header['TSTART'] == table['time'][0]
        assert hdus[1].header['TSTOP'] == table['time'][-1]
        assert np.isclose(hdus[1].header['TIMEDEL'], .1)
        assert hdus[1].columns['psf_flux'].unit == 'electron/s'
        assert np.isnan(hdus['MODEL'].data).all()


@pytest.mark.parametrize('prior', [None, 0.001])
def test_extraction_preserves_failed_and_saturated_cadences_and_reuses_cache(tmp_path, monkeypatch, prior):
    import tglc.target_lightcurve as module
    from tglc.effective_psf import get_psf

    count, size = 50, 25
    source = SimpleNamespace(
        sector=56, camera=1, ccd=2, size=size, time=100 + np.arange(count) * .01,
        flux=np.ones((count, size, size)), cadence=np.arange(500, 500 + count),
        quality=np.zeros(count, dtype=np.int32), ffi='SPOC', transient=None,
        mask=np.ma.array(np.ones((size, size)), mask=np.zeros((size, size), bool)),
        gaia=Table({'DESIGNATION': ['Gaia DR3 42'], 'ra': [100.], 'dec': [20.],
                    'tess_mag': [10.], 'phot_g_mean_mag': [10.],
                    'phot_bp_mean_mag': [11.], 'phot_rp_mean_mag': [9.],
                    'sector_56_x': [12.2], 'sector_56_y': [12.3], 'tess_flux_ratio': [1.]}),
        tic=Table({'TIC': [123], 'dr3_source_id': [42]}))
    matrix, _, over, _, _ = get_psf(source)
    yy, xx = np.indices((over, over))
    kernel = 1000 * np.exp(-((xx - over // 2)**2 + (yy - over // 2)**2) / 8.)
    parameters = np.r_[kernel.ravel(), [0., 0., 100.]]
    source.flux[:] = (matrix[:size**2] @ parameters).reshape(size, size)
    source.flux[3, 12, 12] = 1e6
    source.flux[4, 15, 15] = 1e6  # diagonal dilation does not enter the 5x5 target window
    source.quality[5] = 32768

    def fitted(*args, time=0, **kwargs):
        return parameters if time != 8 else np.full_like(parameters, np.nan)

    monkeypatch.setattr(module, 'fit_psf', fitted)
    with pytest.warns(UserWarning, match='1 of 50 ePSF fits failed'):
        outputs = epsf(source, local_directory=tmp_path, name='Gaia DR3 42', no_progress_bar=True, prior=prior)
    assert len(outputs) == 1 and outputs[0].parent == tmp_path / 'lc' / 'SPOC'
    with fits.open(outputs[0]) as hdus:
        table = hdus[1].data
        assert len(table) == count
        assert np.isnan(table['aperture_flux_raw'][3])
        assert np.isfinite(table['psf_flux_raw'][3])  # fit uses unsaturated wings
        assert table['TGLC_flags'][3] & 4
        assert not table['TGLC_flags'][4] & 4
        assert np.isnan(table['aperture_flux_raw'][8]) and np.isnan(table['psf_flux_raw'][8])
        assert table['TGLC_flags'][8] & 2
        assert table['TESS_flags'][5] == 32768
    monkeypatch.setattr(module, 'fit_psf', lambda *a, **k: (_ for _ in ()).throw(AssertionError('cache miss')))
    with pytest.warns(UserWarning, match='1 of 50 ePSF fits failed'):
        assert epsf(source, local_directory=tmp_path, name='Gaia DR3 42', no_progress_bar=True, prior=prior) == outputs

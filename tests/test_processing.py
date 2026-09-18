from types import SimpleNamespace
import pickle

import numpy as np

from tglc.processing import (
    background_outliers,
    epsf_fingerprint,
    load_epsf_cache,
    save_epsf_cache,
)


def test_background_constant_missing_and_outlier():
    values = np.array([10.0] * 12 + [np.nan, 20.0])
    np.testing.assert_array_equal(background_outliers(values),
                                  [False] * 13 + [True])
    assert not background_outliers([np.nan, np.nan]).any()


def test_cache_changes_with_pixels_cadences_model_mask_and_settings(tmp_path):
    source = SimpleNamespace(sector=56, camera=1, ccd=2, time=np.arange(3.0),
                             cadence=np.arange(3), flux=np.ones((3, 2, 2)),
                             mask=np.ma.array(np.ones((2, 2)), mask=np.zeros((2, 2), bool)))
    design = np.eye(4)
    config = {'ffi': 'SPOC', 'power': 1.4}
    fingerprint = epsf_fingerprint(source, design, config)
    parameters = np.ones((3, 4))
    path = tmp_path / 'cache.npz'
    save_epsf_cache(path, parameters, fingerprint, config)
    np.testing.assert_array_equal(load_epsf_cache(path, fingerprint, (3, 4)), parameters)
    assert load_epsf_cache(path, fingerprint, (2, 4)) is None
    assert load_epsf_cache(path, 'other', (3, 4)) is None
    assert fingerprint != epsf_fingerprint(source, design, dict(config, ffi='TICA'))
    assert fingerprint != epsf_fingerprint(source, design * 2, config)
    for array in [source.flux, source.time, source.cadence, source.mask.mask]:
        before = array.flat[0]
        array.flat[0] = 1 if array.dtype == bool else before + 1
        assert fingerprint != epsf_fingerprint(source, design, config)
        array.flat[0] = before
    path.write_bytes(b'interrupted cache')
    assert load_epsf_cache(path, fingerprint, (3, 4)) is None
    save_epsf_cache(path, parameters, fingerprint, config)
    contents = path.read_bytes()
    path.write_bytes(contents[:len(contents) // 2])
    assert load_epsf_cache(path, fingerprint, (3, 4)) is None
    np.savez(path, e_psf=np.full((3, 4), 'bad'), fingerprint=fingerprint)
    assert load_epsf_cache(path, fingerprint, (3, 4)) is None


def test_fits_byte_order_and_source_pickle_keep_the_same_cache_identity():
    source = SimpleNamespace(
        sector=2, camera=2, ccd=4, time=np.array([1354., 1354.02], dtype='>f8'),
        cadence=np.array([123, 124], dtype='>i8'),
        flux=np.array([[[1., np.nan], [3., 4.]], [[5., 6.], [7., 8.]]], dtype='>f4'),
        mask=np.ma.array(np.ones((2, 2)), mask=np.zeros((2, 2), bool)),
    )
    design = np.eye(4, dtype='>f8')
    restored = pickle.loads(pickle.dumps(source))
    restored_design = pickle.loads(pickle.dumps(design))
    config = {'ffi': 'SPOC', 'power': 1.4}
    original = epsf_fingerprint(source, design, config)
    assert original == epsf_fingerprint(restored, restored_design, config)
    restored.flux[0, 0, 0] += 1
    assert original != epsf_fingerprint(restored, restored_design, config)

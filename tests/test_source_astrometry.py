"""Offline checks of catalog propagation, source masks, and source provenance."""

import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from astropy.wcs import WCS

from tglc.astrometry import (
    SOURCE_SCHEMA_VERSION, persistent_bad_pixels, propagate_catalog_positions,
)


def make_wcs(x_origin=0):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [x_origin + 6, 6]
    wcs.wcs.crval = [20, 60]
    wcs.wcs.cdelt = [21 / 3600, 21 / 3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


def make_catalog(wcs, x_origin=0):
    # Third source moves into the 6-pixel halo during the ten-year baseline.
    ra, dec = wcs.all_pix2world([5 + x_origin, -5 + x_origin, -6.3 + x_origin], [5, 5, 5], 0)
    return Table({
        'DESIGNATION': ['Gaia DR3 100', 'Gaia DR3 200', 'Gaia DR3 300'],
        'phot_g_mean_mag': [10., 11., 12.], 'phot_bp_mean_mag': [10., 11., 12.],
        'phot_rp_mean_mag': [10., 11., 12.], 'ra': ra, 'dec': dec,
        'pmra': [1000., 0., 2100.], 'pmdec': [0., 0., 0.],
        'ref_epoch': [2016., 2016., 2016.],
    })


def observation_times():
    return Time(2026., format='jyear', scale='tdb').jd - 2457000 + np.array([-0.01, 0, 0.01])


def test_propagation_preserves_reference_positions_and_uses_row_epochs():
    catalog = Table({'ra': [20., 20.], 'dec': [60., 60.], 'pmra': [1000., 1000.],
                     'pmdec': [0., 0.], 'ref_epoch': [2016., 2021.]})
    propagated = propagate_catalog_positions(catalog, observation_times())
    np.testing.assert_array_equal(propagated['ra'], catalog['ra'])
    np.testing.assert_array_equal(propagated['dec'], catalog['dec'])
    # A 10-year 1000 mas/yr motion at declination 60 moves RA by 20 arcsec.
    np.testing.assert_allclose((propagated['ra_epoch'] - 20) * 3600, [20, 10], rtol=1e-7)
    np.testing.assert_allclose(propagated['position_epoch'], 2026.)
    assert 'ra_epoch' not in catalog.colnames


def test_missing_motion_components_and_invalid_positions_are_explicit():
    catalog = Table({'ra': [20., 20., np.nan], 'dec': [60., 60., 60.],
                     'pmra': MaskedColumn([1000., 0., 0.], mask=[True, True, False]),
                     'pmdec': MaskedColumn([1000., 0., 0.], mask=[False, True, False])})
    propagated = propagate_catalog_positions(catalog, observation_times())
    np.testing.assert_allclose((propagated['dec_epoch'][0] - 60) * 3600, 10, rtol=1e-7)
    np.testing.assert_allclose(propagated['ra_epoch'][:2], [20, 20])
    np.testing.assert_allclose(propagated['dec_epoch'][1], 60)
    assert np.isnan(propagated['ra_epoch'][2])
    assert list(propagated['proper_motion_applied']) == [True, False, False]
    with pytest.raises(ValueError, match='finite observation time'):
        propagate_catalog_positions(catalog, [np.nan])


def test_brightness_and_transient_missing_samples_are_not_persistent_bad_pixels():
    flux = np.array([[[1., 90000.], [np.nan, np.nan]],
                     [[0., 80000.], [2., np.inf]]])
    np.testing.assert_array_equal(persistent_bad_pixels(flux), [[False, False], [False, True]])


def test_fullframe_source_projects_propagated_positions_and_keeps_halo(monkeypatch):
    module = importlib.import_module('tglc.ffi')
    wcs = make_wcs(44)
    catalog = make_catalog(wcs, 44)
    monkeypatch.setattr(module.Source, 'search_gaia', lambda *args: catalog.copy())
    monkeypatch.setattr(module, 'tic_advanced_search_position_rows', lambda **kwargs: Table())
    monkeypatch.setattr(module, 'convert_gaia_id', lambda *args: Table())
    source = module.Source(flux=np.ones((3, 10, 10)), mask=np.ma.array(np.ones((10, 10)), mask=False),
                           time=observation_times(), wcs=wcs, size=10, sector=56)
    assert source.transient is None
    assert source.source_schema_version == SOURCE_SCHEMA_VERSION
    assert len(source.gaia) == 3
    # The known eastward motion is ~0.476 pixels, including spherical geometry.
    assert 5.47 < source.gaia['sector_56_x'][0] < 5.49
    assert -5.4 < source.gaia['sector_56_x'][2] < -5.2
    np.testing.assert_array_equal(source.gaia['ra'], catalog['ra'])


def make_cutout_source():
    module = importlib.import_module('tglc.ffi_cut')
    wcs = make_wcs()
    source = module.Source_cut.__new__(module.Source_cut)
    source.sector = 0
    source.sector_table = Table({'sector': [56]})
    source.sector_list = [56]
    source.ffi = 'SPOC'
    source.size = 10
    source.transient = None
    source.catalogdata = make_catalog(wcs)
    flux = np.ones((3, 10, 10))
    flux[:, 5, 5] = 90000  # Brightest unsaturated source must remain available.
    flux[0, 0, 0] = np.nan
    flux[:, 0, 1] = np.nan
    primary = fits.PrimaryHDU()
    primary.header['SECTOR'] = 56
    primary.header['CAMERA'] = 1
    primary.header['CCD'] = 1
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='TIME', array=observation_times(), format='D'),
        fits.Column(name='QUALITY', array=np.zeros(3, dtype=int), format='J'),
        fits.Column(name='FLUX', array=flux, format='100E', dim='(10,10)'),
        fits.Column(name='FLUX_ERR', array=np.ones_like(flux), format='100E', dim='(10,10)'),
    ])
    source.hdulist = [fits.HDUList([primary, table, fits.ImageHDU(data=np.zeros((10, 10)), header=wcs.to_header())])]
    return source


def test_cutout_selection_uses_epoch_positions_and_finite_support():
    source = make_cutout_source()
    original_ra = source.catalogdata['ra'].copy()
    source.select_sector(56)
    assert len(source.gaia) == 3
    assert 5.47 < source.gaia['sector_56_x'][0] < 5.49
    assert -5.4 < source.gaia['sector_56_x'][2] < -5.2
    assert not source.mask.mask[5, 5]
    assert not source.mask.mask[0, 0]
    assert source.mask.mask[0, 1]
    first_positions = source.gaia['sector_56_x'].copy()
    source.sector = 0
    source.select_sector(56)
    np.testing.assert_array_equal(source.gaia['sector_56_x'], first_positions)
    np.testing.assert_array_equal(source.catalogdata['ra'], original_ra)


@pytest.mark.parametrize('column', ['FFIINDEX', 'CADENCENO'])
def test_cutout_selection_reads_real_cadence_identifiers(column):
    source = make_cutout_source()
    table = source.hdulist[0][1]
    source.hdulist[0][1] = fits.BinTableHDU.from_columns(
        table.columns + fits.ColDefs([fits.Column(name=column, array=[1234, 1235, 1239], format='J')]))
    source.select_sector(56)
    np.testing.assert_array_equal(source.cadence, [1234, 1235, 1239])
    assert source.cadence_origin == column


@pytest.mark.parametrize('placeholder', [None, [0, 0, 0]])
def test_cutout_missing_or_placeholder_cadences_remain_unknown(placeholder):
    source = make_cutout_source()
    if placeholder is not None:
        source.hdulist[0][1] = fits.BinTableHDU.from_columns(
            source.hdulist[0][1].columns + fits.ColDefs([
                fits.Column(name='CADENCENO', array=placeholder, format='J')]))
    source.select_sector(56)
    np.testing.assert_array_equal(source.cadence, [-1, -1, -1])
    assert source.cadence_origin == 'unavailable'


def test_explicit_cadences_are_validated_and_not_reused_in_another_sector():
    source = make_cutout_source()
    source._provided_cadence = [1234, 1235, 1239]
    source.select_sector(56)
    np.testing.assert_array_equal(source.cadence, [1234, 1235, 1239])
    assert source.cadence_origin == 'supplied'
    source.sector_table = Table({'sector': [56, 57]})
    source.sector_list = [56, 57]
    source.hdulist.append(source.hdulist[0].copy())
    with pytest.raises(ValueError, match='applies to one sector'):
        source.select_sector(57)
    assert source.sector == 56


@pytest.mark.parametrize('invalid', [[1, 2], [1, 1, 1], [1, 2, 3.5], [1, -2, 3]])
def test_explicit_cadences_reject_invalid_arrays(invalid):
    source = make_cutout_source()
    source._provided_cadence = invalid
    with pytest.raises(ValueError, match='[Cc]adence'):
        source.select_sector(56)


def test_real_cadence_column_and_explicit_array_must_agree():
    source = make_cutout_source()
    source._provided_cadence = [1234, 1235, 1239]
    source.hdulist[0][1] = fits.BinTableHDU.from_columns(
        source.hdulist[0][1].columns + fits.ColDefs([
            fits.Column(name='CADENCENO', array=[1234, 1235, 1240], format='J')]))
    with pytest.raises(ValueError, match='disagree'):
        source.select_sector(56)


def test_failed_crossmatch_keeps_dr2_but_never_fabricates_dr3(monkeypatch):
    module = importlib.import_module('tglc.ffi')
    def unavailable(*args, **kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr(module.Gaia, 'launch_job_async', unavailable)
    monkeypatch.setattr(module, 'TapPlus', unavailable)
    catalog = Table({'ID': [1, 2, 3], 'GAIA': ['123456789123456789', 'None', '456']})
    with pytest.warns(UserWarning) as emitted:
        result = module.convert_gaia_id(catalog)
    assert any('DR3 identities remain unknown' in str(item.message) for item in emitted)
    np.testing.assert_array_equal(result['dr2_source_id'], [123456789123456789, 456])
    np.testing.assert_array_equal(result['TIC'], [1, 3])
    assert np.all(result['dr3_source_id'].mask)


def test_unambiguous_crossmatch_handles_duplicate_tics_singletons_and_missing_ids(monkeypatch):
    module = importlib.import_module('tglc.ffi')
    queries = []
    def query(text):
        queries.append(text)
        return SimpleNamespace(get_results=lambda: Table({'dr2_source_id': [123], 'dr3_source_id': [456]}))
    monkeypatch.setattr(module.Gaia, 'launch_job_async', query)
    result = module.convert_gaia_id(Table({'ID': [1, 2], 'GAIA': ['123', '123']}))
    np.testing.assert_array_equal(result['dr3_source_id'], [456, 456])
    assert not np.any(result['dr3_source_id'].mask)
    assert 'IN (123)' in queries[0]
    empty = module.convert_gaia_id(Table({'ID': [1], 'GAIA': ['None']}))
    assert len(empty) == 0
    assert len(queries) == 1


def test_ambiguous_crossmatch_stays_unknown(monkeypatch):
    module = importlib.import_module('tglc.ffi')
    bridge = Table({'dr2_source_id': [123, 123], 'dr3_source_id': [456, 789]})
    monkeypatch.setattr(module.Gaia, 'launch_job_async',
                        lambda text: SimpleNamespace(get_results=lambda: bridge))
    result = module.convert_gaia_id(Table({'ID': [1], 'GAIA': ['123']}))
    assert bool(result['dr3_source_id'].mask[0])


def test_dr3_service_failure_does_not_substitute_a_dr2_catalog(monkeypatch):
    module = importlib.import_module('tglc.ffi_cut')
    monkeypatch.setattr(module.Catalogs, 'query_object',
                        lambda *args, **kwargs: Table({'ra': [20.], 'dec': [60.]}))
    def unavailable(*args, **kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr(module.Gaia, 'cone_search_async', unavailable)
    monkeypatch.setattr(module, 'TapPlus', unavailable)
    def forbidden_dr2_fallback(*args, **kwargs):
        pytest.fail('A DR2 catalog must not be substituted for DR3')
    monkeypatch.setattr(module.Catalogs, 'query_region', forbidden_dr2_fallback)
    with pytest.warns(UserWarning, match='Primary Gaia DR3 cone search failed'):
        with pytest.raises(RuntimeError, match='DR2 catalog cannot be used as DR3'):
            module.Source_cut('TIC 123', size=30, ffi='SPOC')


def test_source_cache_validates_request_and_rebuilds_legacy_sources(tmp_path, monkeypatch):
    module = importlib.import_module('tglc.ffi_cut')
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sector_table='test', size=kwargs['size'])
    monkeypatch.setattr(module, 'Source_cut', build)
    kwargs = dict(target='TIC 123', local_directory=tmp_path, sector=56, size=30, limit_mag=16, ffi='spoc')
    source = module.ffi_cut(**kwargs)
    same = module.ffi_cut(**kwargs)
    assert source._tglc_cache_config == same._tglc_cache_config
    assert len(calls) == 1
    for change in ({'size': 40}, {'limit_mag': 15}, {'transient': ('test', 20., 60.)}):
        with pytest.warns(UserWarning, match='Rebuilding incompatible source cache'):
            module.ffi_cut(**(kwargs | change))
    assert len(calls) == 4
    cache = tmp_path / 'source' / 'source_SPOC_TIC 123_sector_56.pkl'
    with cache.open('wb') as stream:
        pickle.dump(SimpleNamespace(sector_table='legacy'), stream)
    with pytest.warns(UserWarning, match='Rebuilding incompatible source cache'):
        module.ffi_cut(**kwargs)
    assert len(calls) == 5
    assert not list(cache.parent.glob('*.tmp'))


def test_failed_source_rebuild_preserves_previous_cache(tmp_path, monkeypatch):
    module = importlib.import_module('tglc.ffi_cut')
    cache = tmp_path / 'source' / 'source_SPOC_TIC 123_sector_56.pkl'
    cache.parent.mkdir()
    with cache.open('wb') as stream:
        pickle.dump(SimpleNamespace(sector_table='legacy'), stream)
    old_bytes = cache.read_bytes()
    def unavailable(*args, **kwargs):
        raise RuntimeError('catalog unavailable')
    monkeypatch.setattr(module, 'Source_cut', unavailable)
    with pytest.warns(UserWarning, match='Rebuilding incompatible source cache'):
        with pytest.raises(RuntimeError, match='catalog unavailable'):
            module.ffi_cut('TIC 123', tmp_path, sector=56, ffi='SPOC')
    assert cache.read_bytes() == old_bytes


@pytest.mark.parametrize('target', [123, np.int64(123), '123', 'TIC 123', 'tic123', ' TIC 00123 '])
@pytest.mark.parametrize('entrypoint', ['Source_cut', 'ffi_cut'])
def test_direct_cutout_tic_forms_normalize_both_name_lookups(tmp_path, monkeypatch, target, entrypoint):
    module = importlib.import_module('tglc.ffi_cut')
    queries = []
    def missing_target(name, **kwargs):
        queries.append((name, kwargs))
        return Table()
    monkeypatch.setattr(module.Catalogs, 'query_object', missing_target)
    # Empty first and wider lookups stop the real constructor before catalog
    # or image downloads, while exercising both remote request boundaries.
    with pytest.raises(RuntimeError, match='MAST name lookup failed for TIC target'):
        if entrypoint == 'Source_cut':
            module.Source_cut(target, size=30, sector=56, ffi='SPOC')
        else:
            module.ffi_cut(target, local_directory=tmp_path, size=30, sector=56, ffi='SPOC')
    assert [query[0] for query in queries] == ['TIC 123', 'TIC 123']
    assert queries[1][1]['radius'] == pytest.approx(5 * queries[0][1]['radius'])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('target', [True, np.bool_(False), 0, np.int64(-1), '0', 'TIC 0', 'TIC wrong'])
@pytest.mark.parametrize('entrypoint', ['Source_cut', 'ffi_cut'])
def test_direct_cutout_rejects_invalid_tic_before_queries(tmp_path, monkeypatch, target, entrypoint):
    module = importlib.import_module('tglc.ffi_cut')
    def forbidden_query(*args, **kwargs):
        pytest.fail('Invalid TIC identifiers must fail before a remote query')
    monkeypatch.setattr(module.Catalogs, 'query_object', forbidden_query)
    with pytest.raises((ValueError, TypeError), match='TIC'):
        if entrypoint == 'Source_cut':
            module.Source_cut(target, ffi='SPOC')
        else:
            module.ffi_cut(target, local_directory=tmp_path, ffi='SPOC')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('target', ['NGC 7654', '351.40691 61.646657'])
def test_direct_cutout_preserves_non_tic_names(monkeypatch, target):
    module = importlib.import_module('tglc.ffi_cut')
    queries = []
    def missing_target(name, **kwargs):
        queries.append(name)
        return Table()
    monkeypatch.setattr(module.Catalogs, 'query_object', missing_target)
    with pytest.raises(RuntimeError, match='Unable to resolve target'):
        module.Source_cut(target, ffi='SPOC')
    assert queries == [target, target]


def test_direct_cutout_tic_forms_share_one_source_cache(tmp_path, monkeypatch):
    module = importlib.import_module('tglc.ffi_cut')
    names = []
    def build(name, **kwargs):
        names.append(name)
        return SimpleNamespace(name=name, sector_table='test')
    monkeypatch.setattr(module, 'Source_cut', build)
    for target in [123, np.int64(123), '123', 'TIC 123', 'tic123', ' TIC 00123 ']:
        source = module.ffi_cut(target, local_directory=tmp_path, sector=56, ffi='SPOC')
        assert source.name == 'TIC 123'
        assert source._tglc_cache_config['target'] == 'TIC 123'
    assert names == ['TIC 123']
    assert [path.name for path in (tmp_path / 'source').iterdir()] == ['source_SPOC_TIC 123_sector_56.pkl']

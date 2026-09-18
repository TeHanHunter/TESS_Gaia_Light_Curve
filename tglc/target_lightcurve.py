# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from numpy import ma
from scipy.ndimage import binary_dilation
from tqdm import trange

import tglc
from tglc.barycentric_correction import apply_barycentric_correction
from tglc.effective_psf import (
    _source_pixel_mask,
    bg_mod,
    fit_lc,
    fit_lc_float_field,
    fit_psf,
    get_psf,
)
from tglc.ffi import Source
from tglc.processing import (
    APERTURE_INVALID,
    EPSF_FAILED,
    HIGH_BACKGROUND,
    NO_GOOD_REFERENCE,
    PROCESSING_VERSION,
    PSF_INVALID,
    SATURATED,
    background_outliers,
    epsf_fingerprint,
    load_epsf_cache,
    save_epsf_cache,
)

warnings.simplefilter('always', UserWarning)


def _header_number(value):
    """FITS headers cannot contain floating-point NaN or infinity."""
    return float(value) if value is not None and np.isfinite(value) else 'NaN'


def _scatter(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return (float(1.4826 * np.median(np.abs(values - np.median(values))))
            if values.size else 'NaN')


def lc_output(source, local_directory='', index=0, time=None, psf_lc=None, cal_psf_lc=None, aper_lc=None,
              cal_aper_lc=None, bg=None, tess_flag=None, tglc_flag=None, cadence=None, aperture=None,
              cut_x=None, cut_y=None, star_x=2, star_y=2, x_aperture=None, y_aperture=None, near_edge=False,
              local_bg=None, save_aper=False, portion=1, prior=None, transient=None, target_5x5=None, field_stars_5x5=None,
              ffi='SPOC', raw_aper_lc=None, raw_psf_lc=None, psf_local_bg=None,
              fit_fingerprint=None, fit_config=None):
    """
    lc output to .FITS file in MAST HLSP standards
    :param tglc_flag: np.array(), required
    TGLC quality flags
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param local_directory: string, required
    output directory
    :param index: int, required
    star index
    :param time: list, required
    epochs of FFI
    :param lc: list, required
    ePSF light curve fluxes
    :param cal_lc: list, required
    ePSF light curve fluxes, detrended
    :param cadence: list, required
    list of cadences of TESS FFI
    :return:
    """
    if transient is None:
        objid = next(int(s) for s in source.gaia[index]['DESIGNATION'].split() if s.isdigit())
    else:
        objid = transient[0]
    local_directory = Path(local_directory)
    local_directory.mkdir(parents=True, exist_ok=True)
    source_path = local_directory / f'hlsp_tglc_tess_ffi_gaiaid-{objid}-s{source.sector:04d}-cam{source.camera}-ccd{source.ccd}_tess_v2_llc.fits'
    # if source_exists and (os.path.getsize(source_path) > 0):
    #     print('LC exists, please (re)move the file if you wish to overwrite.')
    #     return
    if np.isnan(source.gaia[index]['phot_bp_mean_mag']) or ma.is_masked(source.gaia[index]['phot_bp_mean_mag']):
        gaia_bp = 'NaN'
    else:
        gaia_bp = source.gaia[index]['phot_bp_mean_mag']
    if np.isnan(source.gaia[index]['phot_rp_mean_mag']) or ma.is_masked(source.gaia[index]['phot_rp_mean_mag']):
        gaia_rp = 'NaN'
    else:
        gaia_rp = source.gaia[index]['phot_rp_mean_mag']
    psf_err = _scatter(psf_lc)
    aper_err = _scatter(aper_lc)
    cal_psf_err = _scatter(cal_psf_lc)
    cal_aper_err = _scatter(cal_aper_lc)
    try:
        ticid = str(source.tic['TIC'][np.where(source.tic['dr3_source_id'] == objid)][0])
    except (IndexError, KeyError, TypeError, ValueError):
        ticid = ''
    try:
        raw_flux = _header_number(np.nanmedian(source.flux[:, star_y, star_x]))
    except (IndexError, TypeError, ValueError):
        raw_flux = None
    if save_aper:
        primary_hdu = fits.PrimaryHDU(aperture)
    else:
        primary_hdu = fits.PrimaryHDU()
    # Simulated star images based on ePSF, used to estimate contamination ratio and others
    if target_5x5 is None:
        target_5x5 = np.full((5, 5), np.nan)
    if field_stars_5x5 is None:
        field_stars_5x5 = np.full((5, 5), np.nan)
    image_data = np.full((3, 5, 5), np.nan)
    image_data[0] = target_5x5
    image_data[1] = field_stars_5x5
    # This is the pixel-wise contamination ratio
    np.divide(field_stars_5x5, target_5x5, out=image_data[2],
              where=np.isfinite(target_5x5) & (target_5x5 != 0))
    image_hdu = fits.ImageHDU(data=image_data, name='MODEL')
    target_sum = np.sum(target_5x5[1:4, 1:4])
    field_sum = np.sum(field_stars_5x5[1:4, 1:4])
    contamination = (round(float(field_sum / target_sum), 9)
                     if np.isfinite(target_sum) and target_sum != 0 and np.isfinite(field_sum)
                     else 'NaN')

    primary_hdu.header = fits.Header(cards=[
        fits.Card('SIMPLE', True, 'conforms to FITS standard'),
        fits.Card('EXTEND', True),
        fits.Card('NEXTEND', 2, 'number of extensions'),
        fits.Card('EXTNAME', 'PRIMARY', 'name of extension'),
        fits.Card('EXTDATA', 'aperture', 'decontaminated FFI cut for aperture photometry'),
        fits.Card('EXTVER', 1, 'extension version'),
        fits.Card('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'),
        fits.Card('BUNIT', 'e-/s', 'flux unit'),
        fits.Card('STAR_X', x_aperture, 'star x position in cut'),
        fits.Card('STAR_Y', y_aperture, 'star y position in cut'),
        fits.Card('COMMENT', 'hdul[0].data[:,star_y,star_x]=lc'),
        fits.Card('ORIGIN', 'UCSB/TGLC', 'institution responsible for creating this file'),
        fits.Card('TGLCVER', f'{tglc.__version__}', 'TGLC version'),
        fits.Card('PROCVER', PROCESSING_VERSION, 'scientific processing convention'),
        fits.Card('TELESCOP', 'TESS', 'telescope'),
        fits.Card('INSTRUME', 'TESS Photometer', 'detector type'),
        fits.Card('FILTER', 'TESS', 'the filter used for the observations'),
        fits.Card('FFIVER', ffi, 'the FFI product used (SPOC/TICA)'),
        fits.Card('OBJECT', source.gaia[index]['DESIGNATION'], 'string version of Gaia DR3 ID'),
        fits.Card('GAIADR3', objid, 'integer version of Gaia DR3 ID'),
        fits.Card('TICID', ticid, 'TESS Input Catalog ID'),
        fits.Card('SECTOR', source.sector, 'observation sector'),
        fits.Card('CAMERA', source.camera, 'camera No.'),
        fits.Card('CCD', source.ccd, 'CCD No.'),
        fits.Card('CUT_x', cut_x, 'FFI cut x index'),
        fits.Card('CUT_y', cut_y, 'FFI cut y index'),
        fits.Card('CUTSIZE', source.size, 'FFI cut size'),
        fits.Card('RADESYS', 'ICRS', 'reference frame of celestial coordinates'),
        fits.Card('RA_OBJ', source.gaia[index]['ra'], '[deg] Gaia reference right ascension'),
        fits.Card('DEC_OBJ', source.gaia[index]['dec'], '[deg] Gaia reference declination'),
        fits.Card('TESSMAG', source.gaia[index]['tess_mag'], 'TESS magnitude, fitted by Gaia DR3 bands'),
        fits.Card('GAIA_G', source.gaia[index]['phot_g_mean_mag'], 'Gaia DR3 g band magnitude'),
        fits.Card('GAIA_bp', gaia_bp, 'Gaia DR3 bp band magnitude'),
        fits.Card('GAIA_rp', gaia_rp, 'Gaia DR3 rp band magnitude'),
        fits.Card('RAWFLUX', raw_flux, 'median flux of raw FFI'),
        fits.Card('CONTAMRT', contamination, 'field/target model in 3x3; includes background'),
        fits.Card('CALIB', 'TGLC', 'pipeline used for image calibration')])
    if save_aper:
        primary_hdu.header.comments['NAXIS1'] = 'x size of cut'
        primary_hdu.header.comments['NAXIS2'] = 'y size of cut'
        primary_hdu.header.comments['NAXIS3'] = "Time (hdul[1].data['time'])"

    if source.sector < 27:  # primary
        exposure_time = 1800
    elif source.sector < 56:  # first extended
        exposure_time = 600
    else:  # second extended
        exposure_time = 200

    ### barycentric correction for TICA
    if ffi == 'TICA':
        sector = source.sector
        tjd = time
        coord = SkyCoord([(source.gaia[index]['ra'], source.gaia[index]['dec'])], unit=u.deg,)
        time_bcc = apply_barycentric_correction(sector, tjd, coord)[0]
    elif ffi == 'SPOC':
        time_bcc = np.array(time)
    else:
        raise ValueError('ffi must be either TICA or SPOC')

    finite_time = np.asarray(time_bcc)[np.isfinite(time_bcc)]
    if not len(finite_time):
        raise ValueError('Cannot export a light curve without finite times')
    t_start, t_stop = np.min(finite_time), np.max(finite_time)
    time_steps = np.diff(np.unique(finite_time))
    time_step = float(np.median(time_steps)) if len(time_steps) else 0.0
    if fit_fingerprint is not None:
        primary_hdu.header['FITSHASH'] = fit_fingerprint
    if fit_config is not None:
        for keyword, key in [('PSFSIZE', 'psf_size'), ('OVERSAMP', 'factor'),
                             ('FITPOWER', 'power'), ('EDGECOMP', 'edge_compression'),
                             ('SATLIMIT', 'saturation_limit'), ('SATDIL', 'saturation_dilation')]:
            primary_hdu.header[keyword] = fit_config[key]
    for keyword, column, comment in [
            ('PMREFEP', 'ref_epoch', 'Gaia position reference epoch, Julian year'),
            ('PMEPOCH', 'position_epoch', 'propagated position epoch, Julian year'),
            ('RA_EPOCH', 'ra_epoch', '[deg] propagated right ascension'),
            ('DECEPOCH', 'dec_epoch', '[deg] propagated declination')]:
        if column in source.gaia.colnames:
            primary_hdu.header[keyword] = (_header_number(source.gaia[index][column]), comment)

    c1 = fits.Column(name='time', array=np.array(time_bcc), format='D', unit='d')
    c2 = fits.Column(name='psf_flux', array=np.array(psf_lc), format='E', unit='electron/s')
    # c3 = fits.Column(name='psf_flux_err',
    #                  array=1.4826 * np.median(np.abs(psf_lc - np.median(psf_lc))) * np.ones(len(psf_lc)), format='E')
    corrected_aperture = (aper_lc / portion if np.isfinite(portion) and portion > 0
                          else np.full(np.shape(aper_lc), np.nan))
    c4 = fits.Column(name='aperture_flux', array=corrected_aperture, format='E', unit='electron/s')
    # c5 = fits.Column(name='aperture_flux_err',
    #                  array=1.4826 * np.median(np.abs(aper_lc - np.median(aper_lc))) * np.ones(len(aper_lc)), format='E')
    c6 = fits.Column(name='cal_psf_flux', array=np.array(cal_psf_lc), format='E')
    # c7 = fits.Column(name='cal_psf_flux_err',
    #                  array=1.4826 * np.median(np.abs(cal_psf_lc - np.median(cal_psf_lc))) * np.ones(len(cal_psf_lc)),
    #                  format='E')
    c8 = fits.Column(name='cal_aper_flux', array=np.array(cal_aper_lc), format='E')
    # c9 = fits.Column(name='cal_aper_flux_err',
    #                  array=1.4826 * np.median(np.abs(cal_aper_lc - np.median(cal_aper_lc))) * np.ones(len(cal_aper_lc)),
    #                  format='E')
    c10 = fits.Column(name='background', array=bg, format='E', unit='electron/s')
    c11 = fits.Column(name='cadence_num', array=np.asarray(cadence, dtype=np.int32), format='J', null=-1)
    c12 = fits.Column(name='TESS_flags', array=np.asarray(tess_flag, dtype=np.int32), format='J')
    c13 = fits.Column(name='TGLC_flags', array=np.asarray(tglc_flag, dtype=np.int32), format='J')
    columns = [c1, c2, c4, c6, c8, c10, c11, c12, c13]
    if raw_aper_lc is not None:
        columns.append(fits.Column(name='aperture_flux_raw', array=raw_aper_lc,
                                   format='E', unit='electron/s'))
    if raw_psf_lc is not None:
        columns.append(fits.Column(name='psf_flux_raw', array=raw_psf_lc,
                                   format='E', unit='electron/s'))
    table_hdu = fits.BinTableHDU.from_columns(columns)
    table_hdu.header.append(('INHERIT', 'T', 'inherit the primary header'), end=True)
    table_hdu.header.append(('EXTNAME', 'LIGHTCURVE', 'name of extension'), end=True)
    table_hdu.header.append(('EXTVER', 1, 'extension version'),
                            end=True)
    table_hdu.header.append(('TELESCOP', 'TESS', 'telescope'), end=True)
    table_hdu.header.append(('INSTRUME', 'TESS Photometer', 'detector type'), end=True)
    table_hdu.header.append(('FILTER', 'TESS', 'the filter used for the observations'), end=True)
    table_hdu.header.append(('OBJECT', source.gaia[index]['DESIGNATION'], 'string version of Gaia DR3 ID'),
                            end=True)
    table_hdu.header.append(('GAIADR3', objid, 'integer version of GaiaDR3 designation'), end=True)
    table_hdu.header.append(('RADESYS', 'ICRS', 'reference frame of celestial coordinates'), end=True)
    table_hdu.header.append(('RA_OBJ', source.gaia[index]['ra'], '[deg] Gaia reference right ascension'), end=True)
    table_hdu.header.append(('DEC_OBJ', source.gaia[index]['dec'], '[deg] Gaia reference declination'), end=True)
    table_hdu.header.append(('TIMEREF', 'SOLARSYSTEM', 'barycentric correction applied to times'), end=True)
    table_hdu.header.append(('TASSIGN', 'SPACECRAFT', 'where time is assigned'), end=True)
    table_hdu.header.append(('BJDREFI', 2457000, 'integer part of BJD reference date'), end=True)
    table_hdu.header.append(('BJDREFR', 0.0, 'fraction of the day in BJD reference date'), end=True)
    table_hdu.header.append(('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'), end=True)
    table_hdu.header.append(('TIMEUNIT', 'd', 'time unit for TIME'), end=True)
    table_hdu.header.append(('CADORIG', getattr(source, 'cadence_origin', 'supplied'), 'cadence ID source; -1 means unknown'), end=True)
    # table_hdu.header.append(('BUNIT', 'e-/s', 'psf_flux unit'), end=True)
    table_hdu.header.append(('TELAPS', t_stop - t_start, '[d] TSTOP-TSTART'), end=True)
    table_hdu.header.append(('TSTART', t_start, '[d] observation start time in TBJD'), end=True)
    table_hdu.header.append(('TSTOP', t_stop, '[d] observation end time in TBJD'), end=True)
    table_hdu.header.append(('MJD_BEG', t_start + 56999.5, '[d] start time in barycentric MJD'), end=True)
    table_hdu.header.append(('MJD_END', t_stop + 56999.5, '[d] end time in barycentric MJD'), end=True)
    table_hdu.header.append(('TIMEDEL', time_step, '[d] median spacing of finite cadences'),
                            end=True)
    table_hdu.header.append(('XPTIME', exposure_time, '[s] nominal integration, not live exposure'), end=True)
    table_hdu.header.append(('PSF_ERR', psf_err, '[e-/s] MAD scatter, not measurement error'), end=True)
    table_hdu.header.append(('APER_ERR', aper_err, '[e-/s] unscaled-aperture MAD scatter'), end=True)
    table_hdu.header.append(('CPSF_ERR', cal_psf_err, 'dimensionless whole-series MAD scatter'), end=True)
    table_hdu.header.append(('CAPE_ERR', cal_aper_err, 'dimensionless whole-series MAD scatter'), end=True)
    table_hdu.header.append(('NEAREDGE', near_edge, 'distance to edges of FFI <= 2'), end=True)
    table_hdu.header.append(('LOC_BG', _header_number(local_bg), '[e-/s] aperture-sum additive offset'), end=True)
    table_hdu.header.append(('PSF_BG', _header_number(psf_local_bg), '[e-/s] PSF-flux additive offset'), end=True)
    table_hdu.header.append(('PORTION', _header_number(portion), '3x3 fraction of full supported target PSF'), end=True)
    table_hdu.header.append(('COMMENT', 'aperture_flux = (aperture_flux_raw - LOC_BG) / PORTION'), end=True)
    table_hdu.header.append(('COMMENT', 'psf_flux = psf_flux_raw - PSF_BG'), end=True)
    for bit, meaning in [(1, 'background outlier'), (2, 'failed ePSF fit'),
                         (4, 'saturated pixels in target window'), (8, 'invalid aperture flux'),
                         (16, 'invalid PSF flux'), (32, 'no good normalization cadences')]:
        table_hdu.header.append(('COMMENT', f'TGLC_flags {bit}: {meaning}'), end=True)
    table_hdu.header.append(('WOTAN_WL', 1, 'wotan detrending window length'), end=True)
    table_hdu.header.append(('WOTAN_MT', 'biweight', 'wotan detrending method'), end=True)
    if type(prior) == float:
        table_hdu.header.append(('PRIOR', prior, 'prior of field stars'), end=True)

    hdul = fits.HDUList([primary_hdu, table_hdu, image_hdu])
    hdul.writeto(source_path, overwrite=True)
    return source_path



def epsf(source, psf_size=11, factor=2, local_directory='', target=None, cut_x=0, cut_y=0, sector=0, ffi='SPOC',
         limit_mag=16, edge_compression=1e-4, power=1.4, name=None, save_aper=False, no_progress_bar=False, prior=None,
         saturation_limit=80000.0, saturation_dilation=1):
    """Fit a flux-rate source and return the generated light-curve paths.

    Input pixels are in electrons/second. Saturation masking applies to both
    the field ePSF and target amplitude fits. The default cutoff is conservative
    and configurable; masked apertures are missing rather than partial sums.
    Scientific caches include exact input data, geometry, and fit settings.
    """
    ffi = str(ffi).upper()
    if ffi not in ('SPOC', 'TICA'):
        raise ValueError('ffi must be either SPOC or TICA')
    if hasattr(source, 'ffi') and str(source.ffi).upper() != ffi:
        raise ValueError('Requested FFI product does not match source.ffi')
    if not np.isfinite(source.time).any():
        raise ValueError('No finite input cadence times')
    if prior is not None and (not np.isscalar(prior) or not np.isfinite(prior) or prior <= 0):
        raise ValueError('prior must be a positive finite number or None')
    target = f'{cut_x:02d}_{cut_y:02d}' if target is None else str(target)
    # A target name is a label, not a directory path.
    target_label = target.replace('/', '_').replace('\\', '_')
    root = Path(local_directory)
    lc_directory = root / 'lc' / ffi
    epsf_directory = root / 'epsf' / ffi
    full_frame = isinstance(source, Source)
    if full_frame:
        detector = f'{source.camera}-{source.ccd}'
        lc_directory /= detector
        epsf_directory /= detector
    lc_directory.mkdir(parents=True, exist_ok=True)
    epsf_directory.mkdir(parents=True, exist_ok=True)
    A, star_info, over_size, x_round, y_round = get_psf(
        source, psf_size=psf_size, factor=factor, edge_compression=edge_compression)
    bg_dof = A.shape[1] - over_size ** 2
    config = {'ffi': ffi, 'psf_size': psf_size, 'factor': factor, 'edge_compression': edge_compression,
              'power': power, 'saturation_limit': saturation_limit, 'saturation_dilation': saturation_dilation}
    fingerprint = epsf_fingerprint(source, A, config)
    epsf_loc = epsf_directory / f'epsf_{target_label}_sector_{source.sector}_{fingerprint[:16]}.npz'
    shape = (len(source.time), A.shape[1])
    e_psf = load_epsf_cache(epsf_loc, fingerprint, shape)
    mask_options = {'saturation_limit': saturation_limit, 'saturation_dilation': saturation_dilation}
    if e_psf is None:
        e_psf = np.full(shape, np.nan)
        for j in trange(len(source.time), desc='Fitting ePSF', disable=no_progress_bar):
            if np.isfinite(source.time[j]):
                e_psf[j] = fit_psf(A, source, over_size, power=power, time=j, **mask_options)
        save_epsf_cache(epsf_loc, e_psf, fingerprint, config)
    else:
        print(f'Loaded matching ePSF cache for {target}.')
    failed = ~np.isfinite(e_psf).all(axis=1)
    if failed.any():
        warnings.warn(f'{failed.sum()} of {len(failed)} ePSF fits failed for {target}; cadences remain flagged and missing.')
    background = A[:source.size ** 2, -bg_dof:] @ e_psf[:, -bg_dof:].T
    quality_raw = np.zeros(len(source.time), dtype=np.int32)
    quality_raw[background_outliers(e_psf[:, -1], sigma=3)] |= HIGH_BACKGROUND
    quality_raw[failed] |= EPSF_FAILED
    good = (np.asarray(source.quality) == 0) & (quality_raw == 0) & np.isfinite(source.time)
    index = np.flatnonzero(good)
    if not len(index):
        quality_raw |= NO_GOOD_REFERENCE
    # Geometry is determined across all cadences, including a missing first one.
    in_frame = np.where(np.any(np.isfinite(source.flux), axis=0))
    if not len(in_frame[0]):
        raise ValueError('No finite pixels in this source')
    if full_frame:
        x_left = 1.5 if cut_x != 0 else -0.5
        x_right = source.size - (2.5 if cut_x != 13 else 0.5)
        y_left = 1.5 if cut_y != 0 else -0.5
        y_right = source.size - (2.5 if cut_y != 13 else 0.5)
    else:
        x_left, x_right = np.min(in_frame[1]) - 0.5, np.max(in_frame[1]) + 0.5
        y_left, y_right = np.min(in_frame[0]) - 0.5, np.max(in_frame[0]) + 0.5
    targets = np.flatnonzero(np.asarray(source.gaia['tess_mag']) <= limit_mag)
    if name is not None:
        matches = np.flatnonzero(np.asarray(source.gaia['DESIGNATION']) == str(name))
        if not len(matches):
            tic_matches = np.flatnonzero(np.asarray(source.tic['TIC']).astype(str) == str(name))
            if len(tic_matches):
                gaia_id = source.tic['dr3_source_id'][tic_matches[0]]
                matches = np.flatnonzero(np.asarray(source.gaia['DESIGNATION']) == f'Gaia DR3 {gaia_id}')
        if len(matches) != 1:
            raise ValueError(f'Target {name!r} does not have one verified Gaia match in sector {source.sector}')
        targets = matches
    outputs = []
    for i in targets:
        x, y = x_round[i], y_round[i]
        if not (x_left <= x < x_right and y_left <= y < y_right):
            continue
        near_edge = not (x_left + 2 <= x < x_right - 2 and y_left + 2 <= y < y_right - 2)
        fit_options = dict(star_info=star_info, star_num=i, factor=factor, psf_size=psf_size,
                           e_psf=e_psf, near_edge=near_edge, **mask_options)
        if prior is not None:
            aperture, psf_lc, star_y, star_x, portion = fit_lc_float_field(
                A, source, x=x_round, y=y_round, prior=float(prior), **fit_options)
            target_5x5 = field_stars_5x5 = None
        else:
            aperture, psf_lc, star_y, star_x, portion, target_5x5, field_stars_5x5 = fit_lc(
                A, source, x=x, y=y, **fit_options)
        aper_lc = np.sum(aperture[:, max(0, star_y - 1):star_y + 2,
                                     max(0, star_x - 1):star_x + 2], axis=(1, 2))
        saturated_target = np.zeros(len(source.time), dtype=bool)
        for j in range(len(source.time)):
            mask = _source_pixel_mask(source, j, saturation_limit, saturation_dilation)
            if mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                aper_lc[j] = np.nan
            if saturation_limit is not None:
                saturated = np.asarray(source.flux[j]) >= saturation_limit
                if saturation_dilation:
                    saturated = binary_dilation(saturated, iterations=int(saturation_dilation))
                saturated_target[j] = saturated[max(0, y - 2):y + 3, max(0, x - 2):x + 3].any()
        raw_aper_lc, raw_psf_lc = aper_lc.copy(), psf_lc.copy()
        offsets, aper_lc, psf_lc, cal_aper_lc, cal_psf_lc = bg_mod(
            source, q=index, portion=portion, psf_lc=psf_lc, aper_lc=aper_lc,
            near_edge=near_edge, star_num=i, return_offsets=True)
        local_bg, psf_local_bg = offsets
        background_ = background[x + source.size * y]
        quality = quality_raw.copy()
        if not np.isfinite(local_bg) or not np.isfinite(psf_local_bg):
            quality |= NO_GOOD_REFERENCE
        quality[background_outliers(background_, sigma=5)] |= HIGH_BACKGROUND
        quality[~np.isfinite(aper_lc)] |= APERTURE_INVALID
        quality[~np.isfinite(psf_lc)] |= PSF_INVALID
        quality[saturated_target] |= SATURATED
        if not np.isfinite(portion) or portion <= 0:
            quality |= APERTURE_INVALID
        output = lc_output(
            source, local_directory=lc_directory, index=i, tess_flag=source.quality,
            cut_x=cut_x, cut_y=cut_y, cadence=source.cadence, aperture=aperture.astype(np.float32),
            star_y=y, star_x=x, tglc_flag=quality, ffi=ffi, bg=background_, time=source.time,
            psf_lc=psf_lc, cal_psf_lc=cal_psf_lc, aper_lc=aper_lc, cal_aper_lc=cal_aper_lc,
            local_bg=local_bg, psf_local_bg=psf_local_bg,
            x_aperture=float(source.gaia[f'sector_{source.sector}_x'][i] - max(0, x - 2)),
            y_aperture=float(source.gaia[f'sector_{source.sector}_y'][i] - max(0, y - 2)),
            near_edge=near_edge, save_aper=save_aper, portion=portion, prior=prior,
            transient=getattr(source, 'transient', None), target_5x5=target_5x5,
            field_stars_5x5=field_stars_5x5, raw_aper_lc=raw_aper_lc, raw_psf_lc=raw_psf_lc,
            fit_fingerprint=fingerprint, fit_config=config)
        outputs.append(output)
    return outputs

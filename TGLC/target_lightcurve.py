# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import os
from os.path import exists

import numpy as np
from astropy.io import fits
from tqdm import trange
from wotan import flatten
import matplotlib.pyplot as plt

from TGLC.ePSF import *
from TGLC.ffi_cut import *
from TGLC.local_background import *


def lc_output(source, local_directory='', index=0, time=[], lc=[], cal_lc=[], flag=[], cadence=[]):
    """
    lc output to .FITS file in MAST HLSP standards
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
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
    :param flag: list, required
    quality flags
    :param cadence: list, required
    list of cadences of TESS FFI
    :return:
    """
    objid = [int(s) for s in (source.gaia[index]['designation']).split() if s.isdigit()][0]
    if np.isnan(source.gaia[index]['phot_bp_mean_mag']):
        gaia_bp = 'NaN'
    else:
        gaia_bp = source.gaia[index]['phot_bp_mean_mag']
    if np.isnan(source.gaia[index]['phot_rp_mean_mag']):
        gaia_rp = 'NaN'
    else:
        gaia_rp = source.gaia[index]['phot_bp_mean_mag']
    try:
        ticid = source.tic['ID'][np.where(source.tic['GAIA'] == str(objid))][0]
    except:
        ticid = ''
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header = fits.Header(cards=[
        fits.Card('SIMPLE', True, 'conforms to FITS standard'),
        fits.Card('EXTEND', True),
        fits.Card('NEXTEND', 1, 'number of standard extensions'),
        fits.Card('EXTNAME', 'PRIMARY', 'name of extension'),
        fits.Card('ORIGIN', 'UCSB/TGLC', 'institution responsible for creating this file'),
        fits.Card('TELESCOP', 'TESS', 'telescope'),
        fits.Card('INSTRUME', 'TESS Photometer', 'detector type'),
        fits.Card('FILTER', 'TESS', 'the filter used for the observations'),
        fits.Card('OBJECT', source.gaia[index]['designation'], 'string version of Gaia DR2 designation'),
        fits.Card('GAIADR2', objid, 'integer version of Gaia DR2 designation'),
        fits.Card('TICID', ticid, 'TESS Input Catalog ID'),
        fits.Card('SECTOR', source.sector, 'observation sector'),
        fits.Card('CAMERA', source.camera, 'camera No.'),
        fits.Card('CCD', source.ccd, 'CCD No.'),
        fits.Card('RADESYS', 'ICRS', 'reference frame of celestial coordinates'),
        fits.Card('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'),
        fits.Card('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'),
        fits.Card('TESSMAG', source.gaia[index]['tess_mag'], 'TESS magnitude, fitted by Gaia DR2 bands'),
        fits.Card('GAIA_G', source.gaia[index]['phot_g_mean_mag'], 'Gaia DR2 g band magnitude'),
        fits.Card('GAIA_bp', gaia_bp, 'Gaia DR2 bp band magnitude'),
        fits.Card('GAIA_rp', gaia_rp, 'Gaia DR2 rp band magnitude'),
        fits.Card('LOC_BG', source.gaia[index]['tess_mag'] >= 12, 'locally modified background'),
        fits.Card('CALIB', 'TGLC', 'pipeline used for image calibration')])
    t_start = source.time[0]
    t_stop = source.time[-1]
    c1 = fits.Column(name='Time', array=np.array(time), format='D')
    c2 = fits.Column(name='psf_flux', array=np.array(lc), format='D')
    c3 = fits.Column(name='psf_flux_err',
                     array=1.4826 * np.median(np.abs(lc - np.median(lc))) * np.ones(len(lc)), format='D')
    c4 = fits.Column(name='cal_flux', array=np.array(cal_lc), format='D')
    c5 = fits.Column(name='cal_flux_err',
                     array=1.4826 * np.median(np.abs(cal_lc - np.median(cal_lc))) * np.ones(len(cal_lc)), format='D')
    c6 = fits.Column(name='cadence_num', array=np.array(cadence), format='D')
    c7 = fits.Column(name='flags', array=np.array(flag), format='D')
    table_hdu_1 = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
    table_hdu_1.header.append(('INHERIT', 'T', 'inherit the primary header'), end=True)
    table_hdu_1.header.append(('EXTNAME', 'LIGHTCURVE', 'name of extension'), end=True)
    table_hdu_1.header.append(('EXTVER', 'ePSF', 'effective Point Spread Function'),  # TODO: version?
                              end=True)
    table_hdu_1.header.append(('TELESCOP', 'TESS', 'telescope'), end=True)
    table_hdu_1.header.append(('INSTRUME', 'TESS Photometer', 'detector type'), end=True)
    table_hdu_1.header.append(('FILTER', 'TESS', 'the filter used for the observations'), end=True)
    table_hdu_1.header.append(('OBJECT', source.gaia[index]['designation'], 'string version of GaiaDR2 designation'),
                              end=True)
    table_hdu_1.header.append(('GAIADR2', objid, 'integer version of GaiaDR2 designation'), end=True)
    table_hdu_1.header.append(('RADESYS', 'ICRS', 'reference frame of celestial coordinates'), end=True)
    table_hdu_1.header.append(('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'), end=True)
    table_hdu_1.header.append(('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'), end=True)
    table_hdu_1.header.append(('TIMEREF', 'SOLARSYSTEM', 'barycentric correction applied to times'), end=True)
    table_hdu_1.header.append(('TASSIGN', 'SPACECRAFT', 'where time is assigned'), end=True)
    table_hdu_1.header.append(('BJDREFI', 2457000, 'integer part of BJD reference date'), end=True)
    table_hdu_1.header.append(('BJDREFR', 0.0, 'fraction of the day in BJD reference date'), end=True)
    table_hdu_1.header.append(('TIMEUNIT', 'd', 'time unit for TIME'), end=True)
    table_hdu_1.header.append(('TELAPS', t_stop - t_start, '[d] TSTOP-TSTART'), end=True)
    table_hdu_1.header.append(('TSTART', t_start, '[d] observation start time in TBJD'), end=True)
    table_hdu_1.header.append(('TSTOP', t_stop, '[d] observation end time in TBJD'), end=True)
    table_hdu_1.header.append(('TIMEDEL', (t_stop - t_start) / len(source.time), '[d] time resolution of data'),
                              end=True)
    table_hdu_1.header.append(('WOTAN_WL', 1, 'wotan detrending window length'), end=True)
    table_hdu_1.header.append(('WOTAN_MT', 'biweight', 'wotan detrending method'), end=True)

    hdul = fits.HDUList([primary_hdu, table_hdu_1])
    hdul.writeto(f'{local_directory}hlsp_tglc_tess_ffi_s00{source.sector}_gaiaid_{objid}_llc.fits')
    return

    #  1. background fixed (brightest star selected out of frame) use 10 brightest and take the median
    #  2. single core is fast
    #  3. output fits. Check headers
    #  4. do we need to include errors? quality flag
    #  5. for 10-min cadence targets, do we need bigger RAM?


def epsf(source, factor=2, local_directory='', target=None, sector=0, limit_mag=16, edge_compression=1e-4, power=1):
    """
    User function that unites all necessary steps
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param factor: int, optional
    effective PSF oversampling factor
    :param local_directory: string, required
    output directory
    :param target: str, required
    Target identifier (e.g. "NGC 7654" or "M31"),
    or coordinate in the format of ra dec (e.g. '351.40691 61.646657')
    :param sector: int, required
    TESS sector number
    :param limit_mag: int, required
    upper limiting magnitude of the lightcurves that are outputted.
    :param edge_compression: float, optional
    parameter for edge compression
    :param power: float, optional
    power for weighting bright stars' contribution to the fit. 1 means same contribution from all stars,
    <1 means emphasizing dimmer stars
    :return:
    """
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor, edge_compression=edge_compression)

    epsf_exists = exists(f'{local_directory}epsf_{target}_sector_{sector}.npy')
    if epsf_exists:
        e_psf = np.load(f'{local_directory}epsf_{target}_sector_{sector}.npy')
        print('Loaded ePSF from directory. ')
    else:
        e_psf = np.zeros((len(source.time), over_size ** 2 + 1))
        for i in trange(len(source.time), desc='Fitting ePSF'):
            fit = fit_psf(A, source, over_size, power=power, time=i)
            e_psf[i] = fit
        np.save(f'{local_directory}epsf_{target}_sector_{sector}.npy', e_psf)

    lc_exists = exists(f'{local_directory}lc_{target}_sector_{sector}.npy')
    num_stars = np.array(source.gaia['tess_mag']).searchsorted(limit_mag, 'right')
    if lc_exists:
        lightcurve = np.load(f'{local_directory}lc_{target}_sector_{sector}.npy')
        print('Loaded lc from directory. ')
    else:
        lightcurve = np.zeros((num_stars, len(source.time)))
        for i in trange(0, num_stars, desc='Fitting lc'):
            r_A = reduced_A(A, source, star_info=star_info, x=x_round[i], y=y_round[i], star_num=i)
            if 0.5 <= x_round[i] <= source.size - 1.5 and 0.5 <= y_round[
                i] <= source.size - 1.5:  # one pixel width tolerance
                for j in range(len(source.time)):
                    lightcurve[i, j] = source.flux[j][y_round[i], x_round[i]] - np.dot(r_A, e_psf[j])
        np.save(f'{local_directory}lc_{target}_sector_{sector}.npy', lightcurve)

    mod_lc_exists = exists(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
    if mod_lc_exists:
        mod_lightcurve = np.load(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
        print('Loaded mod_lc from directory. ')
    else:
        mod_lightcurve = bg_mod(source, lightcurve=lightcurve, sector=sector, num_stars=num_stars)
        np.save(f'{local_directory}lc_mod_{target}_sector_{sector}.npy', mod_lightcurve)
    # os.mkdir(os.path.join(local_directory, 'psf_lc'))
    # for i in trange(0, num_stars, desc='Saving psf lc'):
    #     if 0.5 <= x_round[i] <= source.size - 1.5 and 0.5 <= y_round[i] <= source.size - 1.5:
    #         lc_output(source, local_directory=local_directory + 'psf_lc/', index=i, time=source.time,
    #                   lc=mod_lightcurve[i], calibration='psf')
    os.mkdir(os.path.join(local_directory, 'lc'))
    for i in trange(num_stars, desc='Flattening lc'):
        flag = np.zeros(len(source.time))
        negative = np.where(mod_lightcurve[i] < 0)[0]
        flag[negative] = 1
        if 0.5 <= x_round[i] <= source.size - 1.5 and 0.5 <= y_round[i] <= source.size - 1.5:
            flatten_lc = flatten(source.time, mod_lightcurve[i] / np.median(mod_lightcurve[i]),
                                 window_length=1, method='biweight', return_trend=False)
            lc_output(source, local_directory=local_directory + 'lc/', index=i, time=source.time,
                      lc=mod_lightcurve[i], cal_lc=flatten_lc, flag=flag)


if __name__ == '__main__':
    target = 'TIC 455784423'  # Target identifier or coordinates TOI-3714
    local_directory = f'/mnt/c/users/tehan/desktop/{target}/'
    # local_directory = os.path.join(os.getcwd(), f'{target}/')
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    size = 90  # int, suggests big cuts
    source = ffi(target=target, size=size, local_directory=local_directory)
    # source.select_sector(sector=24)
    print(source.sector_table)
    epsf(source, factor=2, target=target, sector=source.sector, local_directory=local_directory)  # TODO: power?

# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import os
from os.path import exists

import numpy as np
from astropy.io import fits
from tqdm import trange
from wotan import flatten
import matplotlib.pyplot as plt

from TGLC.effective_psf import *
from TGLC.ffi_cut import *
from TGLC.local_background import *
from matplotlib import colors
import pickle


def lc_output(source, local_directory='', index=0, time=None, lc=None, cal_lc=None, bg=None, tess_flag=None, scale=1.,
              tglc_flag=np.array([]), cadence=None, aperture=None, cut_x=0, cut_y=0, star_x=2, star_y=2):
    """
    lc output to .FITS file in MAST HLSP standards
    :param tglc_flag: np.array(), required
    TGLC quality flags
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
    :param cadence: list, required
    list of cadences of TESS FFI
    :return:
    """
    if aperture is None:
        aperture = []
    if cadence is None:
        cadence = []
    if tess_flag is None:
        tess_flag = []
    if bg is None:
        bg = []
    if cal_lc is None:
        cal_lc = []
    if lc is None:
        lc = []
    if time is None:
        time = []
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
    primary_hdu = fits.PrimaryHDU(aperture)
    primary_hdu.header = fits.Header(cards=[
        fits.Card('SIMPLE', True, 'conforms to FITS standard'),
        fits.Card('EXTEND', True),
        fits.Card('NEXTEND', 1, 'number of standard extensions'),
        fits.Card('EXTNAME', 'PRIMARY', 'name of extension'),
        fits.Card('EXTDATA', 'aperture', 'decontaminated FFI cut for aperture photometry'),
        fits.Card('EXTVER', 1, 'extension version'),
        fits.Card('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'),
        fits.Card('BUNIT', 'e-/s', 'flux unit'),
        fits.Card('STAR_X', star_x, 'star x position in cut'),
        fits.Card('STAR_Y', star_y, 'star y position in cut'),
        fits.Card('COMMENT', 'hdul[0].data[star_y,star_x,:]=lc'),
        fits.Card('ORIGIN', 'UCSB/TGLC', 'institution responsible for creating this file'),
        fits.Card('TELESCOP', 'TESS', 'telescope'),
        fits.Card('INSTRUME', 'TESS Photometer', 'detector type'),
        fits.Card('FILTER', 'TESS', 'the filter used for the observations'),
        fits.Card('OBJECT', source.gaia[index]['designation'], 'string version of Gaia DR2 ID'),
        fits.Card('GAIADR2', objid, 'integer version of Gaia DR2 ID'),
        fits.Card('TICID', ticid, 'TESS Input Catalog ID'),
        fits.Card('SECTOR', source.sector, 'observation sector'),
        fits.Card('CAMERA', source.camera, 'camera No.'),
        fits.Card('CCD', source.ccd, 'CCD No.'),
        fits.Card('CUT_x', cut_x, 'FFI cut x index'),
        fits.Card('CUT_y', cut_y, 'FFI cut y index'),
        fits.Card('CUTSIZE', source.size, 'FFI cut size'),
        fits.Card('RADESYS', 'ICRS', 'reference frame of celestial coordinates'),
        fits.Card('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'),
        fits.Card('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'),
        fits.Card('TESSMAG', source.gaia[index]['tess_mag'], 'TESS magnitude, fitted by Gaia DR2 bands'),
        fits.Card('GAIA_G', source.gaia[index]['phot_g_mean_mag'], 'Gaia DR2 g band magnitude'),
        fits.Card('GAIA_bp', gaia_bp, 'Gaia DR2 bp band magnitude'),
        fits.Card('GAIA_rp', gaia_rp, 'Gaia DR2 rp band magnitude'),
        fits.Card('LOC_BG', source.gaia[index]['tess_mag'] >= 12, 'locally modified background'),
        fits.Card('CALIB', 'TGLC', 'pipeline used for image calibration')])

    primary_hdu.header.comments['NAXIS1'] = "Time (hdul[1].data['time'])"
    primary_hdu.header.comments['NAXIS2'] = 'x size of cut'
    primary_hdu.header.comments['NAXIS3'] = 'y size of cut'

    t_start = source.time[0]
    t_stop = source.time[-1]
    if source.sector < 28:
        exposure_time = 1800
    else:
        exposure_time = 600
    c1 = fits.Column(name='time', array=np.array(time), format='D')
    c2 = fits.Column(name='psf_flux', array=np.array(lc), format='D')
    c3 = fits.Column(name='psf_flux_err',
                     array=1.4826 * np.median(np.abs(lc - np.median(lc))) * np.ones(len(lc)), format='E')
    c4 = fits.Column(name='cal_flux', array=np.array(cal_lc), format='D')
    c5 = fits.Column(name='cal_flux_err',
                     array=1.4826 * np.median(np.abs(cal_lc - np.median(cal_lc))) * np.ones(len(cal_lc)), format='E')
    c6 = fits.Column(name='background', array=bg, format='E')
    c7 = fits.Column(name='cadence_num', array=np.array(cadence), format='J')  # 32 bit int
    c8 = fits.Column(name='TESS_flags', array=np.array(tess_flag), format='I')  # 16 bit int
    c9 = fits.Column(name='TGLC_flags', array=tglc_flag, format='I')
    table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9])
    table_hdu.header.append(('INHERIT', 'T', 'inherit the primary header'), end=True)
    table_hdu.header.append(('EXTNAME', 'LIGHTCURVE', 'name of extension'), end=True)
    table_hdu.header.append(('EXTVER', 1, 'extension version'),  # TODO: version?
                            end=True)
    table_hdu.header.append(('TELESCOP', 'TESS', 'telescope'), end=True)
    table_hdu.header.append(('INSTRUME', 'TESS Photometer', 'detector type'), end=True)
    table_hdu.header.append(('FILTER', 'TESS', 'the filter used for the observations'), end=True)
    table_hdu.header.append(('OBJECT', source.gaia[index]['designation'], 'string version of Gaia DR2 ID'),
                            end=True)
    table_hdu.header.append(('GAIADR2', objid, 'integer version of GaiaDR2 designation'), end=True)
    table_hdu.header.append(('RADESYS', 'ICRS', 'reference frame of celestial coordinates'), end=True)
    table_hdu.header.append(('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'), end=True)
    table_hdu.header.append(('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'), end=True)
    table_hdu.header.append(('TIMEREF', 'SOLARSYSTEM', 'barycentric correction applied to times'), end=True)
    table_hdu.header.append(('TASSIGN', 'SPACECRAFT', 'where time is assigned'), end=True)
    table_hdu.header.append(('BJDREFI', 2457000, 'integer part of BJD reference date'), end=True)
    table_hdu.header.append(('BJDREFR', 0.0, 'fraction of the day in BJD reference date'), end=True)
    table_hdu.header.append(('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'), end=True)
    table_hdu.header.append(('TIMEUNIT', 'd', 'time unit for TIME'), end=True)
    table_hdu.header.append(('BUNIT', 'e-/s', 'psf_flux unit'), end=True)
    table_hdu.header.append(('TELAPS', t_stop - t_start, '[d] TSTOP-TSTART'), end=True)
    table_hdu.header.append(('TSTART', t_start, '[d] observation start time in TBJD'), end=True)
    table_hdu.header.append(('TSTOP', t_stop, '[d] observation end time in TBJD'), end=True)
    table_hdu.header.append(('MJD_BEG', t_start + 56999.5, '[d] start time in barycentric MJD'), end=True)
    table_hdu.header.append(('MJD_END', t_stop + 56999.5, '[d] end time in barycentric MJD'), end=True)
    table_hdu.header.append(('TIMEDEL', (t_stop - t_start) / len(source.time), '[d] time resolution of data'),
                            end=True)
    table_hdu.header.append(('XPTIME', exposure_time, '[s] time resolution of data'), end=True)
    table_hdu.header.append(('SCALE', scale, '[e-/s] normalization factor'), end=True)
    table_hdu.header.append(('WOTAN_WL', 1, 'wotan detrending window length'), end=True)
    table_hdu.header.append(('WOTAN_MT', 'biweight', 'wotan detrending method'), end=True)

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(f'{local_directory}hlsp_tglc_tess_ffi_gaiaid-{objid}-s00{source.sector:02d}_tess_v1_llc.fits',
                 overwrite=True)
    return

    #  1. background fixed (brightest star selected out of frame) use 10 brightest and take the median
    #  2. single core is fast
    #  3. output fits. Check headers
    #  4. do we need to include errors? quality flag
    #  5. for 10-min cadence targets, do we need bigger RAM?


def epsf(source, psf_size=11, factor=2, local_directory='', cut_x=0, cut_y=0, ccd='', sector=0, limit_mag=16,
         edge_compression=1e-4, power=1):
    """
    User function that unites all necessary steps
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param factor: int, optional
    effective PSF oversampling factor
    :param local_directory: string, required
    output directory
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
    target = f'{cut_x:02d}_{cut_y:02d}'
    A, star_info, over_size, x_round, y_round = get_psf(source, psf_size=psf_size, factor=factor,
                                                        edge_compression=edge_compression)
    epsf_loc = f'{local_directory}epsf/{ccd}/epsf_{target}_sector_{sector}.npy'
    epsf_exists = exists(epsf_loc)
    if epsf_exists:
        e_psf = np.load(epsf_loc)
        print('Loaded ePSF from directory. ')
    else:
        e_psf = np.zeros((len(source.time), over_size ** 2 + 1))
        for i in trange(len(source.time), desc='Fitting ePSF'):
            fit = fit_psf(A, source, over_size, power=power, time=i)
            e_psf[i] = fit
        np.save(epsf_loc, e_psf)

    quality = np.zeros(len(source.time), dtype=np.int16)
    quality[abs(e_psf[:, -1] - np.nanmedian(e_psf[:, -1])) >= np.nanstd(e_psf[:, -1]) / 16] += 1
    index_1 = np.where(np.array(source.quality) == 0)[0]
    index_2 = np.where(quality == 0)[0]
    index = np.intersect1d(index_1, index_2)
    x_left = 1.5 if cut_x != 0 else -0.5
    x_right = 2.5 if cut_x != 13 else 0.5
    y_left = 1.5 if cut_y != 0 else -0.5
    y_right = 2.5 if cut_y != 13 else 0.5
    # np.save('/mnt/c/users/tehan/desktop/7654/contamination_8.npy', np.dot(A, e_psf[0])[:50 ** 2].reshape(50, 50))
    # lc_exists = exists(f'{local_directory}lc_{target}_sector_{sector}.npy')
    num_stars = np.array(source.gaia['tess_mag']).searchsorted(limit_mag, 'right')
    # if lc_exists:
    #     lightcurve = np.load(f'{local_directory}lc_{target}_sector_{sector}.npy')
    #     print('Loaded lc from directory. ')
    # else:
    # lightcurve = np.zeros((num_stars, len(source.time)))
    for i in trange(0, num_stars, desc='Fitting lc'):
        if x_left <= x_round[i] <= source.size - x_right and y_left <= y_round[i] <= source.size - y_right:
            aperture, star_y, star_x = reduced_A(A, source, star_info=star_info, x=x_round[i], y=y_round[i], star_num=i,
                                                 e_psf=e_psf)
            lc = aperture[star_y, star_x, :]
            if source.gaia['tess_mag'][i] > 12:
                lc = bg_mod(source, lightcurve=lc, sector=sector, num_stars=num_stars, star_num=i)
            if 1.5 <= x_round[i] <= source.size - 2.5 and 1.5 <= y_round[i] <= source.size - 2.5:
                quality_ = quality
            else:
                quality_ = quality + 2
            flatten_lc = flatten(source.time, lc / np.nanmedian(lc),
                                 window_length=1, method='biweight', return_trend=False)
            lc_output(source, local_directory=local_directory + 'lc/', index=i, tess_flag=source.quality,
                      scale=float(np.nanmedian(lc[index])), cut_x=cut_x, cut_y=cut_y,
                      aperture=aperture.astype(np.float32), star_y=star_y, star_x=star_x, tglc_flag=quality_,
                      bg=e_psf[:, -1], time=source.time, lc=lc, cal_lc=flatten_lc)
        # np.save(f'{local_directory}lc_{target}_sector_{sector}.npy', lightcurve)
    # mod_lc_exists = exists(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
    # if mod_lc_exists:
    #     mod_lightcurve = np.load(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
    #     print('Loaded mod_lc from directory. ')
    # else:

    # os.mkdir(os.path.join(local_directory, 'psf_lc'))
    # for i in trange(0, num_stars, desc='Saving psf lc'):
    #     if 0.5 <= x_round[i] <= source.size - 1.5 and 0.5 <= y_round[i] <= source.size - 1.5:
    #         lc_output(source, local_directory=local_directory + 'psf_lc/', index=i, time=source.time,
    #                   lc=mod_lightcurve[i], calibration='psf')
    # os.mkdir(os.path.join(local_directory, 'lc'))
    # os.makedirs(os.path.join(local_directory, 'lc'), exist_ok=True)

    # mag = []
    # mean_diff = []
    # for i in trange(num_stars, desc='Flattening lc'):
    #     if x_left <= x_round[i] <= source.size - x_right and y_left <= y_round[i] <= source.size - y_right:
    #         if 1.5 <= x_round[i] <= source.size - 2.5 and 1.5 <= y_round[i] <= source.size - 2.5:
    #             quality_ = quality
    #         else:
    #             quality_ = quality + 2
    #         mag.append(source.gaia['tess_mag'][i])
    #         mean_diff.append(np.nanmean(np.abs(np.diff(mod_lightcurve[i][index]))))
    #         flatten_lc = flatten(source.time, mod_lightcurve[i] / np.nanmedian(mod_lightcurve[i]),
    #                              window_length=1, method='biweight', return_trend=False)
    #         lc_output(source, local_directory=local_directory + 'lc/', index=i, tess_flag=source.quality,
    #                   scale=float(np.nanmedian(mod_lightcurve[i][index])), cut_x=cut_x, cut_y=cut_y,
    #                   tglc_flag=quality_, bg=e_psf[:, -1], time=source.time, lc=mod_lightcurve[i], cal_lc=flatten_lc)
    #
    # np.save(local_directory + f'mean_diff{factor}_{target}.npy', np.array([mag, mean_diff]))


if __name__ == '__main__':
    sector = 17
    ccd = '1-1'
    local_directory = f'/mnt/d/TESS_Sector_17/'
    os.makedirs(local_directory + f'epsf/{ccd}/', exist_ok=True)
    for i in range(484):
        target = f'{(i // 22):02d}_{(i % 22):02d}'
        with open(local_directory + f'source/{ccd}/source_{target}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        epsf(source, factor=2, ccd=ccd, sector=source.sector, local_directory=local_directory)

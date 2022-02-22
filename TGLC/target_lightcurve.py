# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import matplotlib.pyplot as plt
from TGLC.ffi_cut import *
from TGLC.ePSF import *
from TGLC.local_background import *

import pickle
import numpy as np
import os
from os.path import exists
from tqdm import trange
from wotan import flatten
from astropy.io import fits
import time


def lc_output(source, local_directory='', index=0, time=[], lc=[], calibration='cal'):
    objid = [int(s) for s in (source.gaia[index]['designation']).split() if s.isdigit()][0]
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
        fits.Card('SECTOR', source.sector, 'observation sector'),
        fits.Card('CAMERA', source.camera, 'camera No.'),
        fits.Card('CCD', source.ccd, 'CCD No.'),
        fits.Card('RADESYS', 'ICRS', 'reference frame of celestial coordinates'),
        fits.Card('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'),
        fits.Card('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'),
        fits.Card('TESSMAG', source.gaia[index]['tess_mag'], 'TESS magnitude, fitted by Gaia DR2 bands'),
        fits.Card('CALIB', 'TGLC', 'pipeline used for image calibration')])
    t_start = source.time[0]
    t_stop = source.time[-1]
    c1 = fits.Column(name='Time', array=np.array(time), format='D')
    c2 = fits.Column(name='Norm_flux', array=np.array(lc), format='D')
    table_hdu = fits.BinTableHDU.from_columns([c1, c2])
    table_hdu.header.append(('INHERIT', 'T', 'inherit the primary header'), end=True)
    table_hdu.header.append(('EXTNAME', 'LIGHTCURVE', 'name of extension'), end=True)
    table_hdu.header.append(('EXTVER', f'{calibration}', 'psf or calibrated (detrended and normalized)'),
                            end=True)
    table_hdu.header.append(('TELESCOP', 'TESS', 'telescope'), end=True)
    table_hdu.header.append(('INSTRUME', 'TESS Photometer', 'detector type'), end=True)
    table_hdu.header.append(('FILTER', 'TESS', 'the filter used for the observations'), end=True)
    table_hdu.header.append(('OBJECT', source.gaia[index]['designation'], 'string version of GaiaDR2 designation'),
                            end=True)
    table_hdu.header.append(('GAIADR2', objid, 'integer version of GaiaDR2 designation'), end=True)
    table_hdu.header.append(('RADESYS', 'ICRS', 'reference frame of celestial coordinates'), end=True)
    table_hdu.header.append(('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'), end=True)
    table_hdu.header.append(('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'), end=True)
    table_hdu.header.append(('TIMEREF', 'SOLARSYSTEM', 'barycentric correction applied to times'), end=True)
    table_hdu.header.append(('TASSIGN', 'SPACECRAFT', 'where time is assigned'), end=True)
    table_hdu.header.append(('BJDREFI', 2457000, 'integer part of BJD reference date'), end=True)
    table_hdu.header.append(('BJDREFR', 0.0, 'fraction of the day in BJD reference date'), end=True)
    table_hdu.header.append(('TIMEUNIT', 'd', 'time unit for TIME'), end=True)
    table_hdu.header.append(('TELAPS', t_stop - t_start, '[d] TSTOP-TSTART'), end=True)
    table_hdu.header.append(('TSTART', t_start, '[d] observation start time in TBJD'), end=True)
    table_hdu.header.append(('TSTOP', t_stop, '[d] observation end time in TBJD'), end=True)
    table_hdu.header.append(('TIMEDEL', (t_stop - t_start) / len(source.time), '[d] time resolution of data'), end=True)
    table_hdu.header.append(('WOTAN_WL', 1, 'wotan detrending window length'), end=True)
    table_hdu.header.append(('WOTAN_MT', 'biweight', 'wotan detrending method'), end=True)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(f'{local_directory}hlsp_tglc_tess_ffi_s00{source.sector}_gaiaid_{objid}_{calibration}_llc.fits')
    return

    #  1. background fixed (brightest star selected out of frame)
    #  2. single core is fast
    #  3. output fits. Check headers


def epsf(source, factor=4, local_directory='', target=None, sector=0, limit_mag=16, edge_compression=1e-4, power=1):
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
            if 1 <= x_round[i] <= source.size - 2 and 1 <= y_round[i] <= source.size - 2:  # one pixel width tolerance
                for j in range(len(source.time)):
                    lightcurve[i, j] = source.flux[j][y_round[i], x_round[i]] - np.dot(r_A, e_psf[j])
        np.save(f'{local_directory}lc_{target}_sector_{sector}.npy', lightcurve)

    mod_lc_exists = exists(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
    if mod_lc_exists:
        mod_lightcurve = np.load(f'{local_directory}lc_mod_{target}_sector_{sector}.npy')
        print('Loaded mod_lc from directory. ')
    else:
        mod_lightcurve = lightcurve

        for i in trange(1, num_stars, desc='Adjusting background'):
            bg_modification, bg_mod_err, bg_arr = bg_mod(source, lightcurve=lightcurve, sector=sector, chosen_index=[i])
            mod_lightcurve[i] = lightcurve[i] + bg_modification
        np.save(f'{local_directory}lc_mod_{target}_sector_{sector}.npy', mod_lightcurve)
    os.mkdir(os.path.join(local_directory, 'psf_lc'))
    for i in trange(0, num_stars, desc='Saving psf lc'):
        lc_output(source, local_directory=local_directory + 'psf_lc/', index=i, time=source.time,
                  lc=mod_lightcurve[i], calibration='psf')
    os.mkdir(os.path.join(local_directory, 'cal_lc'))
    for i in trange(np.shape(lightcurve)[0], desc='Flattening lc'):
        positive = np.where(mod_lightcurve[i] > 0)[0]
        flatten_lc = flatten(source.time[positive],
                             mod_lightcurve[i][positive] / np.median(mod_lightcurve[i][positive]),
                             window_length=1,
                             method='biweight', return_trend=False)
        lc_output(source, local_directory=local_directory + 'cal_lc/', index=i, time=source.time[positive],
                  lc=flatten_lc, calibration='cal')
    #     return flatten_lc, e_psf if return_epsf else flatten_lc
    # else:
    #     return mod_lightcurve, e_psf if return_epsf else mod_lightcurve


if __name__ == '__main__':
    target = 'NGC_7654'  # Target identifier or coordinates
    local_directory = f'/mnt/c/users/tehan/desktop/{target}_new_bilinear/'
    # local_directory = os.path.join(os.getcwd(), f'{target}/')
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    size = 90  # int
    source = ffi(target=target, size=size, local_directory=local_directory)
    source.select_sector(sector=24)
    print(source.sector_table)
    # source.select_sector(sector=18)
    start = time.time()
    epsf(source, factor=2, target=target, sector=source.sector, local_directory=local_directory)
    end = time.time()
    print(f'Runtime of the program is {end - start}')

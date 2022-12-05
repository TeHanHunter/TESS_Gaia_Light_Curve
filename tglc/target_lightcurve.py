# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import os
import warnings
import numpy as np
import numpy.ma as ma
import tglc
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import trange
from os.path import exists
from tglc.effective_psf import get_psf, fit_psf, fit_lc, fit_lc_float_field, bg_mod
from tglc.ffi import Source
from tglc.ffi_cut import Source_cut

warnings.simplefilter('always', UserWarning)


def lc_output(source, local_directory='', index=0, time=None, psf_lc=None, cal_psf_lc=None, aper_lc=None,
              cal_aper_lc=None, bg=None, tess_flag=None, tglc_flag=None, cadence=None, aperture=None,
              cut_x=None, cut_y=None, star_x=2, star_y=2, x_aperture=None, y_aperture=None, near_edge=False,
              local_bg=None, save_aper=False, portion=1, prior=None):
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
    objid = [int(s) for s in (source.gaia[index]['designation']).split() if s.isdigit()][0]
    source_path = f'{local_directory}hlsp_tglc_tess_ffi_gaiaid-{objid}-s{source.sector:04d}-cam{source.camera}-ccd{source.ccd}_tess_v1_llc.fits'
    source_exists = exists(source_path)
    if source_exists and os.path.getsize(source_path) > 0:
        return
    if np.isnan(source.gaia[index]['phot_bp_mean_mag']) or ma.is_masked(source.gaia[index]['phot_bp_mean_mag']):
        gaia_bp = 'NaN'
    else:
        gaia_bp = source.gaia[index]['phot_bp_mean_mag']
    if np.isnan(source.gaia[index]['phot_rp_mean_mag']) or ma.is_masked(source.gaia[index]['phot_rp_mean_mag']):
        gaia_rp = 'NaN'
    else:
        gaia_rp = source.gaia[index]['phot_rp_mean_mag']
    psf_err = 1.4826 * np.median(np.abs(psf_lc - np.median(psf_lc)))
    if np.isnan(psf_err):
        psf_err = 'NaN'
    aper_err = 1.4826 * np.median(np.abs(aper_lc - np.median(aper_lc)))
    if np.isnan(aper_err):
        aper_err = 'NaN'
    cal_psf_err = 1.4826 * np.median(np.abs(cal_psf_lc - np.median(cal_psf_lc)))
    if np.isnan(cal_psf_err):
        cal_psf_err = 'NaN'
    cal_aper_err = 1.4826 * np.median(np.abs(cal_aper_lc - np.median(cal_aper_lc)))
    if np.isnan(cal_aper_err):
        cal_aper_err = 'NaN'
    try:
        ticid = str(source.tic['TIC'][np.where(source.tic['dr3_source_id'] == objid)][0])
    except:
        ticid = ''
    try:
        raw_flux = np.nanmedian(source.flux[:, star_y, star_x])
    except:
        raw_flux = None
    if save_aper:
        primary_hdu = fits.PrimaryHDU(aperture)
    else:
        primary_hdu = fits.PrimaryHDU()

    primary_hdu.header = fits.Header(cards=[
        fits.Card('SIMPLE', True, 'conforms to FITS standard'),
        fits.Card('EXTEND', True),
        fits.Card('NEXTEND', 1, 'number of standard extensions'),
        fits.Card('EXTNAME', 'PRIMARY', 'name of extension'),
        fits.Card('EXTDATA', 'aperture', 'decontaminated FFI cut for aperture photometry'),
        fits.Card('EXTVER', 1, 'extension version'),
        fits.Card('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'),
        fits.Card('BUNIT', 'e-/s', 'flux unit'),
        fits.Card('STAR_X', x_aperture, 'star x position in cut'),
        fits.Card('STAR_Y', y_aperture, 'star y position in cut'),
        fits.Card('COMMENT', 'hdul[0].data[:,star_y,star_x]=lc'),
        fits.Card('ORIGIN', 'UCSB/TGLC', 'institution responsible for creating this file'),
        fits.Card('TELESCOP', 'TESS', 'telescope'),
        fits.Card('INSTRUME', 'TESS Photometer', 'detector type'),
        fits.Card('FILTER', 'TESS', 'the filter used for the observations'),
        fits.Card('OBJECT', source.gaia[index]['designation'], 'string version of Gaia DR3 ID'),
        fits.Card('GAIADR3', objid, 'integer version of Gaia DR3 ID'),
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
        fits.Card('TESSMAG', source.gaia[index]['tess_mag'], 'TESS magnitude, fitted by Gaia DR3 bands'),
        fits.Card('GAIA_G', source.gaia[index]['phot_g_mean_mag'], 'Gaia DR3 g band magnitude'),
        fits.Card('GAIA_bp', gaia_bp, 'Gaia DR3 bp band magnitude'),
        fits.Card('GAIA_rp', gaia_rp, 'Gaia DR3 rp band magnitude'),
        fits.Card('RAWFLUX', raw_flux, 'median flux of raw FFI'),
        fits.Card('CALIB', 'TGLC', 'pipeline used for image calibration')])
    if save_aper:
        primary_hdu.header.comments['NAXIS1'] = "Time (hdul[1].data['time'])"
        primary_hdu.header.comments['NAXIS2'] = 'x size of cut'
        primary_hdu.header.comments['NAXIS3'] = 'y size of cut'

    t_start = source.time[0]
    t_stop = source.time[-1]
    if source.sector < 27:  # primary
        exposure_time = 1800
    elif source.sector < 56:  # first extended
        exposure_time = 600
    else:  # second extended
        exposure_time = 200
    c1 = fits.Column(name='time', array=np.array(time), format='D')
    c2 = fits.Column(name='psf_flux', array=np.array(psf_lc), format='E')  # psf factor
    # c3 = fits.Column(name='psf_flux_err',
    #                  array=1.4826 * np.median(np.abs(psf_lc - np.median(psf_lc))) * np.ones(len(psf_lc)), format='E')
    c4 = fits.Column(name='aperture_flux', array=aper_lc / portion, format='E')
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
    c10 = fits.Column(name='background', array=bg, format='E')  # add tilt
    c11 = fits.Column(name='cadence_num', array=np.array(cadence), format='J')  # 32 bit int
    c12 = fits.Column(name='TESS_flags', array=np.array(tess_flag), format='I')  # 16 bit int
    c13 = fits.Column(name='TGLC_flags', array=tglc_flag, format='I')
    table_hdu = fits.BinTableHDU.from_columns([c1, c2, c4, c6, c8, c10, c11, c12, c13])
    table_hdu.header.append(('INHERIT', 'T', 'inherit the primary header'), end=True)
    table_hdu.header.append(('EXTNAME', 'LIGHTCURVE', 'name of extension'), end=True)
    table_hdu.header.append(('EXTVER', 1, 'extension version'),  # TODO: version?
                            end=True)
    table_hdu.header.append(('TELESCOP', 'TESS', 'telescope'), end=True)
    table_hdu.header.append(('INSTRUME', 'TESS Photometer', 'detector type'), end=True)
    table_hdu.header.append(('FILTER', 'TESS', 'the filter used for the observations'), end=True)
    table_hdu.header.append(('OBJECT', source.gaia[index]['designation'], 'string version of Gaia DR3 ID'),
                            end=True)
    table_hdu.header.append(('GAIADR3', objid, 'integer version of GaiaDR3 designation'), end=True)
    table_hdu.header.append(('RADESYS', 'ICRS', 'reference frame of celestial coordinates'), end=True)
    table_hdu.header.append(('RA_OBJ', source.gaia[index]['ra'], '[deg] right ascension, J2000'), end=True)
    table_hdu.header.append(('DEC_OBJ', source.gaia[index]['dec'], '[deg] declination, J2000'), end=True)
    table_hdu.header.append(('TIMEREF', 'SOLARSYSTEM', 'barycentric correction applied to times'), end=True)
    table_hdu.header.append(('TASSIGN', 'SPACECRAFT', 'where time is assigned'), end=True)
    table_hdu.header.append(('BJDREFI', 2457000, 'integer part of BJD reference date'), end=True)
    table_hdu.header.append(('BJDREFR', 0.0, 'fraction of the day in BJD reference date'), end=True)
    table_hdu.header.append(('TIMESYS', 'TDB', 'TESS Barycentric Dynamical Time'), end=True)
    table_hdu.header.append(('TIMEUNIT', 'd', 'time unit for TIME'), end=True)
    # table_hdu.header.append(('BUNIT', 'e-/s', 'psf_flux unit'), end=True)
    table_hdu.header.append(('TELAPS', t_stop - t_start, '[d] TSTOP-TSTART'), end=True)
    table_hdu.header.append(('TSTART', t_start, '[d] observation start time in TBJD'), end=True)
    table_hdu.header.append(('TSTOP', t_stop, '[d] observation end time in TBJD'), end=True)
    table_hdu.header.append(('MJD_BEG', t_start + 56999.5, '[d] start time in barycentric MJD'), end=True)
    table_hdu.header.append(('MJD_END', t_stop + 56999.5, '[d] end time in barycentric MJD'), end=True)
    table_hdu.header.append(('TIMEDEL', (t_stop - t_start) / len(source.time), '[d] time resolution of data'),
                            end=True)
    table_hdu.header.append(('XPTIME', exposure_time, '[s] exposure time'), end=True)
    table_hdu.header.append(('PSF_ERR', psf_err, '[e-/s] PSF flux error'), end=True)
    table_hdu.header.append(('APER_ERR', aper_err, '[e-/s] aperture flux error'), end=True)
    table_hdu.header.append(('CPSF_ERR', cal_psf_err, '[e-/s] calibrated PSF flux error'), end=True)
    table_hdu.header.append(('CAPE_ERR', cal_aper_err, '[e-/s] calibrated aperture flux error'), end=True)
    table_hdu.header.append(('NEAREDGE', near_edge, 'distance to edges of FFI <= 2'), end=True)
    table_hdu.header.append(('LOC_BG', local_bg, '[e-/s] locally modified background'), end=True)
    table_hdu.header.append(('COMMENT', "TRUE_BG = hdul[1].data['background'] + LOC_BG"), end=True)
    table_hdu.header.append(('WOTAN_WL', 1, 'wotan detrending window length'), end=True)
    table_hdu.header.append(('WOTAN_MT', 'biweight', 'wotan detrending method'), end=True)
    if type(prior) == float:
        table_hdu.header.append(('PRIOR', prior, 'prior of field stars'), end=True)

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(
        f'{local_directory}hlsp_tglc_tess_ffi_gaiaid-{objid}-s{source.sector:04d}-cam{source.camera}-ccd{source.ccd}_tess_v1_llc.fits',
        overwrite=True)
    return


def epsf(source, psf_size=11, factor=2, local_directory='', target=None, cut_x=0, cut_y=0, sector=0,
         limit_mag=16, edge_compression=1e-4, power=1.4, name=None, save_aper=False, no_progress_bar=False, prior=None):
    """
    User function that unites all necessary steps
    :param source: TGLC.ffi_cut.Source or TGLC.ffi_cut.Source_cut, required
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
    if target is None:
        target = f'{cut_x:02d}_{cut_y:02d}'
    A, star_info, over_size, x_round, y_round = get_psf(source, psf_size=psf_size, factor=factor,
                                                        edge_compression=edge_compression)
    lc_directory = f'{local_directory}lc/{source.camera}-{source.ccd}/'
    epsf_loc = f'{local_directory}epsf/{source.camera}-{source.ccd}/epsf_{target}_sector_{sector}_{source.camera}-{source.ccd}.npy'
    if type(source) == Source_cut:
        bg_dof = 3
        lc_directory = f'{local_directory}lc/'
        epsf_loc = f'{local_directory}epsf/epsf_{target}_sector_{sector}.npy'
    else:
        bg_dof = 6
    os.makedirs(lc_directory, exist_ok=True)
    # sim_image = np.dot(A[:source.size ** 2, :], fit_psf(A, source, over_size, power=power, time=0).T)
    # residual = np.abs(source.flux[0].flatten() - sim_image)
    # return residual

    epsf_exists = exists(epsf_loc)
    if epsf_exists:
        e_psf = np.load(epsf_loc)
        print(f'Loaded ePSF {target} from directory. ')
    else:
        e_psf = np.zeros((len(source.time), over_size ** 2 + bg_dof))
        for i in trange(len(source.time), desc='Fitting ePSF', disable=no_progress_bar):
            e_psf[i] = fit_psf(A, source, over_size, power=power, time=i)
        if np.isnan(e_psf).any():
            warnings.warn(
                f"TESS FFI cut includes Nan values. Please shift the center of the cutout to remove Nan near edge. Target: {target}")
        np.save(epsf_loc, e_psf)
    # contamination_8 = np.dot(A[:source.size ** 2, :], e_psf[0].T)
    # np.save('/mnt/c/users/tehan/desktop/7654/contamination_8_.npy', contamination_8)
    # TODO: quality use which background?
    background = np.dot(A[:source.size ** 2, -bg_dof:], e_psf[:, -bg_dof:].T)
    quality_raw = np.zeros(len(source.time), dtype=np.int16)
    sigma = 1.4826 * np.nanmedian(np.abs(e_psf[:, -1] - np.nanmedian(e_psf[:, -1])))
    quality_raw[abs(e_psf[:, -1] - np.nanmedian(e_psf[:, -1])) >= 3 * sigma] += 1
    index_1 = np.where(np.array(source.quality) == 0)[0]
    index_2 = np.where(quality_raw == 0)[0]
    index = np.intersect1d(index_1, index_2)
    if type(source) == Source_cut:
        in_frame = np.where(np.invert(np.isnan(source.flux[0])))
        x_left = np.min(in_frame[1]) - 0.5
        x_right = source.size - np.max(in_frame[1]) + 0.5
        y_left = np.min(in_frame[0]) - 0.5
        y_right = source.size - np.max(in_frame[0]) + 0.5
    else:
        x_left = 1.5 if cut_x != 0 else -0.5
        x_right = 2.5 if cut_x != 13 else 0.5
        y_left = 1.5 if cut_y != 0 else -0.5
        y_right = 2.5 if cut_y != 13 else 0.5

    num_stars = np.array(source.gaia['tess_mag']).searchsorted(limit_mag, 'right')
    x_aperture = source.gaia[f'sector_{source.sector}_x'] - np.maximum(0, x_round - 2)
    y_aperture = source.gaia[f'sector_{source.sector}_y'] - np.maximum(0, y_round - 2)

    start = 0
    end = num_stars
    if name is not None:
        start = int(np.where(source.gaia['designation'] == name)[0][0])
        end = start + 1
    for i in trange(start, end, desc='Fitting lc', disable=no_progress_bar):
        if x_left <= x_round[i] < source.size - x_right and y_left <= y_round[i] < source.size - y_right:
            if type(source) == Source:
                x_left = 1.5
                x_right = 2.5
                y_left = 1.5
                y_right = 2.5
            if x_left + 2 <= x_round[i] < source.size - (x_right + 2) and y_left + 2 <= y_round[i] < source.size - (
                    y_right + 2):
                near_edge = False
            else:
                near_edge = True

            if type(prior) == float:
                aperture, psf_lc, star_y, star_x, portion = \
                    fit_lc_float_field(A, source, star_info=star_info, x=x_round, y=y_round, star_num=i, e_psf=e_psf,
                                       near_edge=near_edge, prior=prior)
            else:
                aperture, psf_lc, star_y, star_x, portion = \
                    fit_lc(A, source, star_info=star_info, x=x_round[i], y=y_round[i], star_num=i, e_psf=e_psf,
                           near_edge=near_edge)

            aper_lc = np.sum(
                aperture[:, max(0, star_y - 1):min(5, star_y + 2), max(0, star_x - 1):min(5, star_x + 2)],
                axis=(1, 2))
            if source.sector < 27:  # primary
                exposure_time = 1800
            elif source.sector < 56:  # first extended
                exposure_time = 600
            else:  # second extended
                exposure_time = 200
            saturated_arg_aper = np.where(aper_lc > 1e5 * 9 * 2e5 / exposure_time)  # saturation is 2e5 e-
            aper_lc[saturated_arg_aper] = np.nan
            saturated_arg_psf = np.where(psf_lc > 1e5 * 9 * 2e5 / exposure_time)
            psf_lc[saturated_arg_psf] = np.nan
            local_bg, aper_lc, psf_lc, cal_aper_lc, cal_psf_lc = bg_mod(source, q=index, portion=portion,
                                                                        psf_lc=psf_lc,
                                                                        aper_lc=aper_lc,
                                                                        near_edge=near_edge, star_num=i)
            background_ = background[x_round[i] + source.size * y_round[i], :]
            quality = np.zeros(len(source.time), dtype=np.int16)
            sigma = 1.4826 * np.nanmedian(np.abs(background_ - np.nanmedian(background_)))
            quality[abs(background_ - np.nanmedian(background_)) >= 5 * sigma] += 1
            if np.isnan(aper_lc).all():
                continue
            else:
                lc_output(source, local_directory=lc_directory, index=i,
                          tess_flag=source.quality, cut_x=cut_x, cut_y=cut_y, cadence=source.cadence,
                          aperture=aperture.astype(np.float32), star_y=y_round[i], star_x=x_round[i], tglc_flag=quality,
                          bg=background_, time=source.time, psf_lc=psf_lc, cal_psf_lc=cal_psf_lc, aper_lc=aper_lc,
                          cal_aper_lc=cal_aper_lc, local_bg=local_bg, x_aperture=x_aperture[i],
                          y_aperture=y_aperture[i], near_edge=near_edge, save_aper=save_aper, portion=portion,
                          prior=prior)

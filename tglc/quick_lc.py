import os
import pickle
from glob import glob
from tqdm import trange
from wotan import flatten
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
import sys
import warnings
# Tesscut._service_api_connection.TIMEOUT = 6000
# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np
import seaborn as sns
import itertools
controller = ThreadpoolController()


@controller.wrap(limits=1, user_api='blas')
def tglc_lc(target='TIC 264468702', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
            first_sector_only=False, last_sector_only=False, sector=None, prior=None, transient=None):
    '''
    Generate light curve for a single target.

    :param target: target identifier
    :kind target: str, required
    :param local_directory: output directory
    :kind local_directory: str, required
    :param size: size of the FFI cut, default size is 90. Recommend large number for better quality. Cannot exceed 100.
    :kind size: int, optional
    '''
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    print(f'Target: {target}')
    target_ = Catalogs.query_object(target, radius=42 * 0.707 / 3600, catalog="Gaia", version=2)
    if len(target_) == 0:
        target_ = Catalogs.query_object(target.name, radius=5 * 21 * 0.707 / 3600, catalog="Gaia", version=2)
    ra = target_[0]['ra']
    dec = target_[0]['dec']
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    sector_table = Tesscut.get_sectors(coordinates=coord)
    print(sector_table)
    if get_all_lc:
        name = None
    else:
        catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
        if target[0:3] == 'TIC':
            name = int(target[4:])
        elif transient is not None:
            name = transient[0]
        else:
            name = int(np.array(catalogdata['ID'])[0])
            print("Since the provided target is not TIC ID, the resulted light curve with get_all_lc=False can not be "
                  "guaranteed to be the target's light curve. Please check the TIC ID of the output file before using "
                  "the light curve or try use TIC ID as the target in the format of 'TIC 12345678'.")
    if type(sector) == int:
        print(f'Only processing Sector {sector}.')
        print('Downloading Data from MAST and Gaia ...')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=sector)
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif first_sector_only:
        print(f'Only processing the first sector the target is observed in: Sector {sector_table["sector"][0]}.')
        print('Downloading Data from MAST and Gaia ...')
        sector = sector_table["sector"][0]
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=source.sector_table['sector'][0])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif last_sector_only:
        print(f'Only processing the last sector the target is observed in: Sector {sector_table["sector"][-1]}.')
        print('Downloading Data from MAST and Gaia ...')
        sector = sector_table["sector"][-1]
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=source.sector_table['sector'][-1])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif sector == None:
        print(f'Processing all available sectors of the target.')
        print('Downloading Data from MAST and Gaia ...')
        for j in range(len(sector_table)):
            print(f'################################################')
            print(f'Downloading Sector {sector_table["sector"][j]}.')
            source = ffi_cut(target=target, size=size, local_directory=local_directory,
                             sector=sector_table['sector'][j],
                             limit_mag=limit_mag, transient=transient)
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    else:
        print(
            f'Processing all available sectors of the target in a single run. Note that if the number of sectors is '
            f'large, the download might cause a timeout error from MAST.')
        print('Downloading Data from MAST and Gaia ...')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        for j in range(len(source.sector_table)):
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)


def search_stars(i, sector=1, tics=None, local_directory=None):
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/hlsp_*.fits')
    for j in trange(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            try:
                if int(hdul[0].header['TICID']) in tics:
                    hdul.writeto(f"{local_directory}{files[j].split('/')[-1]}",
                                 overwrite=True)
            except:
                pass


def timebin(time, meas, meas_err, binsize):
    ind_order = np.argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]
    ct = 0
    while ct < len(time):
        ind = np.where((time >= time[ct]) & (time < time[ct] + binsize))[0]
        num = len(ind)
        wt = (1. / meas_err[ind]) ** 2.  # weights based in errors
        wt = wt / np.sum(wt)  # normalized weights
        if ct == 0:
            time_out = [np.sum(wt * time[ind])]
            meas_out = [np.sum(wt * meas[ind])]
            meas_err_out = [1. / np.sqrt(np.sum(1. / (meas_err[ind]) ** 2))]
        else:
            time_out.append(np.sum(wt * time[ind]))
            meas_out.append(np.sum(wt * meas[ind]))
            meas_err_out.append(1. / np.sqrt(np.sum(1. / (meas_err[ind]) ** 2)))
        ct += num

    return time_out, meas_out, meas_err_out


def star_spliter(server=1,  # or 2
                 tics=None, local_directory=None):
    for i in range(server, 27, 2):
        with Pool(16) as p:
            p.map(partial(search_stars, sector=i, tics=tics, local_directory=local_directory), range(16))
    return


def plot_lc(local_directory=None, kind='cal_aper_flux', xlow=None, xhigh=None, ylow=None, yhigh=None):
    files = glob(f'{local_directory}lc/*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            plt.figure(constrained_layout=False, figsize=(8, 4))
            plt.plot(hdul[1].data['time'], hdul[1].data[kind], '.', c='silver', label=kind)
            plt.plot(hdul[1].data['time'][q], hdul[1].data[kind][q], '.k', label=f'{kind}_flagged')
            plt.xlim(xlow, xhigh)
            plt.ylim(ylow, yhigh)
            plt.title(f'TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}_{kind}')
            plt.legend()
            # plt.show()
            plt.savefig(
                f'{local_directory}plots/TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}_{kind}.png',
                dpi=300)
            plt.close()


def plot_aperture(local_directory=None, kind='cal_aper_flux'):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    portion = [0.9361215204370542, 0.9320709087810205]
    data = np.empty((3, 0))

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            print(files[i], portion[i])
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            plt.figure(constrained_layout=False, figsize=(8, 4))
            plt.plot(hdul[1].data['time'] % 3.79262026, hdul[1].data[kind], '.', c='silver', label=kind)
            plt.plot(hdul[1].data['time'][q] % 3.79262026, hdul[1].data[kind][q], '.k', label=f'{kind}_flagged')
            aperture_bar = 709.5512462444653 * portion[i]
            aper_lc = np.nansum(hdul[0].data, axis=(1, 2))
            local_bg = np.nanmedian(aper_lc) - aperture_bar
            aper_lc = (aper_lc - local_bg) / portion[i]
            cal_aper_lc = aper_lc / np.nanmedian(aper_lc)
            cal_aper_lc[np.where(cal_aper_lc > 100)] = np.nan
            _, trend = flatten(hdul[1].data['time'], cal_aper_lc - np.nanmin(cal_aper_lc) + 1000,
                               window_length=1, method='biweight', return_trend=True)
            cal_aper_lc = (cal_aper_lc - np.nanmin(cal_aper_lc) + 1000 - trend) / np.nanmedian(cal_aper_lc) + 1
            non_outliers = np.where(cal_aper_lc[q] > 0.6)[0]
            plt.plot(hdul[1].data['time'][q][non_outliers] % 3.79262026, cal_aper_lc[q][non_outliers], '.r',
                     label=f'5_5_pixel_flagged')
            plt.xlim(0.5, 1.0)
            plt.ylim(0.95, 1.1)
            plt.title(f'TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}')
            plt.legend()
            # plt.show()
            plt.savefig(
                f'{local_directory}plots/TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}.png',
                dpi=300)
            time = hdul[1].data['time'][q][non_outliers]
            flux = cal_aper_lc[q][non_outliers]
            f_err = 1.4826 * np.nanmedian(np.abs(flux - np.nanmedian(flux)))
            not_nan = np.invert(np.isnan(flux))
            data_ = np.array([time[not_nan],
                              flux[not_nan],
                              np.array([f_err] * len(time[not_nan]))
                              ])
            data = np.append(data, data_, axis=1)
    np.savetxt(f'{local_directory}TESS_TOI-5344_5_5_aper.csv', data, delimiter=',')


def plot_pf_lc(local_directory=None, period=None, mid_transit_tbjd=None, kind='cal_aper_flux'):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    fig = plt.figure(figsize=(13, 5))
    t_all = np.array([])
    f_all = np.array([])
    f_err_all = np.array([])
    for j in range(len(files)):
        not_plotted_num = 0
        with fits.open(files[j], mode='denywrite') as hdul:
            q = [a and b for a, b in
                 zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            # q = [a and b for a, b in zip(q, list(hdul[1].data[kind] > 0.85))]
            # if hdul[0].header['sector'] == 15:
            #     q = [a and b for a, b in zip(q, list(hdul[1].data['time'] < 1736))]
            if len(hdul[1].data['cal_aper_flux']) == len(hdul[1].data['time']):
                if hdul[0].header["SECTOR"] <= 26:
                    t = hdul[1].data['time'][q]
                    f = hdul[1].data[kind][q]
                elif hdul[0].header["SECTOR"] <= 55:
                    t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // 3 * 3].reshape(-1, 3), axis=1)
                    f = np.mean(
                        hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // 3 * 3].reshape(-1, 3), axis=1)
                else:
                    t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // 9 * 9].reshape(-1, 9), axis=1)
                    f = np.mean(
                        hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // 9 * 9].reshape(-1, 9), axis=1)
                t_all = np.append(t_all, t)
                f_all = np.append(f_all, f)
                f_err_all = np.append(f_err_all, np.array([hdul[1].header['CAPE_ERR']] * len(t)))

                # plt.plot(hdul[1].data['time'] % period / period, hdul[1].data[kind], '.', c='silver', ms=3)
                plt.errorbar(t % period / period, f, hdul[1].header['CAPE_ERR'], c='silver', ls='', elinewidth=0.1,
                             marker='.', ms=3, zorder=2)
                # time_out, meas_out, meas_err_out = timebin(time=t % period, meas=f,
                #                                            meas_err=np.array([hdul[1].header['CAPE_ERR']] * len(t)),
                #                                            binsize=600 / 86400)
                # plt.errorbar(np.array(time_out) / period, meas_out, meas_err_out, c=f'C{j}', ls='', elinewidth=1.5,
                #              marker='.', ms=8, zorder=3, label=f'Sector {hdul[0].header["sector"]}')
            else:
                not_plotted_num += 1
            title = f'TIC_{hdul[0].header["TICID"]} with {len(files) - not_plotted_num} sector(s) of data, {kind}'
    # PDCSAP_files = glob('/home/tehan/Documents/GEMS/TIC 172370679/PDCSAP/*.txt')
    # for i in range(len(files)):
    #     PDCSAP = ascii.read(PDCSAP_files[i])
    #     t = np.mean(PDCSAP['col1'][:len(PDCSAP['col1']) // 15 * 15].reshape(-1, 15), axis=1)
    #     f = np.mean(PDCSAP['col2'][:len(PDCSAP['col2']) // 15 * 15].reshape(-1, 15), axis=1)
    #     ferr = np.mean(PDCSAP['col3'][:len(PDCSAP['col3']) // 15 * 15].reshape(-1, 15), axis=1)
    #     plt.errorbar((t - 2457000) % period / period, f, ferr, c='C0', ls='', elinewidth=0, marker='.', ms=2, zorder=1)
    time_out, meas_out, meas_err_out = timebin(time=t_all % period, meas=f_all,
                                               meas_err=f_err_all,
                                               binsize=300 / 86400)
    plt.errorbar(np.array(time_out) / period, meas_out, meas_err_out, c=f'r', ls='', elinewidth=1.5,
                 marker='.', ms=8, zorder=3, label=f'All sectors')

    plt.ylim(0.998, 1.001)
    # plt.xlim(0.3, 0.43)
    plt.legend()
    plt.title(title)
    # plt.xlim(mid_transit_tbjd % period - 0.1 * period, mid_transit_tbjd % period + 0.1 * period)
    # plt.ylim(0.9, 1.1)
    # plt.hlines(y=0.92, xmin=0, xmax=1, ls='dotted', colors='k')
    # plt.hlines(y=0.93, xmin=0, xmax=1, ls='dotted', colors='k')
    plt.vlines(x=(mid_transit_tbjd % period), ymin=0, ymax=2, ls='dotted', colors='grey')
    plt.xlabel('Phase')
    plt.ylabel('Normalized flux')
    plt.savefig(f'{local_directory}/plots/{title}.png', dpi=300)
    plt.close(fig)


# newest
def plot_contamination(local_directory=None, gaia_dr3=None, ymin=None, ymax=None, pm_years=3000, detrend=True):
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.9'})
    files = glob(f'{local_directory}lc/*{gaia_dr3}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            sector = hdul[0].header['SECTOR']
            q = [a and b for a, b in
                 zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            if ymin is None and ymax is None:
                ymin = np.nanmin(hdul[1].data['cal_aper_flux'][q]) - 0.05
                ymax = np.nanmax(hdul[1].data['cal_aper_flux'][q]) + 0.05
            with open(glob(f'{local_directory}source/*_{sector}.pkl')[0], 'rb') as input_:
                source = pickle.load(input_)
                source.select_sector(sector=sector)
                star_num = np.where(source.gaia['DESIGNATION'] == f'Gaia DR3 {gaia_dr3}')

                distances = np.sqrt(
                    (source.gaia[f'sector_{sector}_x'][:500] - source.gaia[star_num][f'sector_{sector}_x']) ** 2 +
                    (source.gaia[f'sector_{sector}_y'][:500] - source.gaia[star_num][f'sector_{sector}_y']) ** 2)

                # Find closest 5 stars (6-self) or those within 5 pixels
                nearby_stars = np.argsort(distances)[:6]
                nearby_stars = nearby_stars[distances[nearby_stars] <= 5]
                star_x = source.gaia[star_num][f'sector_{sector}_x'][0]
                star_y = source.gaia[star_num][f'sector_{sector}_y'][0]
                max_flux = np.nanmax(
                    np.nanmedian(
                        source.flux[:, round(star_y) - 2:round(star_y) + 3, round(star_x) - 2:round(star_x) + 3],
                        axis=0))
                fig = plt.figure(constrained_layout=False, figsize=(15, 9))
                gs = fig.add_gridspec(21, 10)
                gs.update(wspace=0.05, hspace=0.15)
                ax0 = fig.add_subplot(gs[:9, :3])
                ax0.imshow(np.median(source.flux, axis=0), cmap='RdBu', vmin=-max_flux, vmax=max_flux, origin='lower')
                ax0.set_xlabel('x pixel')
                ax0.set_ylabel('y pixel')
                ax0.scatter(star_x, star_y, s=300, c='r', marker='*', label='target star')
                ax0.scatter(source.gaia[f'sector_{sector}_x'][:500], source.gaia[f'sector_{sector}_y'][:500], s=30,
                            c='r', label='background stars')
                ax0.scatter(source.gaia[f'sector_{sector}_x'][nearby_stars[nearby_stars != star_num[0][0]]],
                            source.gaia[f'sector_{sector}_y'][nearby_stars[nearby_stars != star_num[0][0]]],
                            s=30, c='r', edgecolor='black', linewidth=1, label='background stars')
                ax0.grid(False)
                for l in range(len(nearby_stars)):
                    index = np.where(
                        source.tic['dr3_source_id'] == int(source.gaia['DESIGNATION'][nearby_stars[l]].split(' ')[-1]))
                    gaia_targets = source.gaia
                    median_time = np.median(source.time)
                    interval = (median_time - 388.5) / 365.25 + pm_years
                    ra = gaia_targets['ra'][nearby_stars[l]]
                    dec = gaia_targets['dec'][nearby_stars[l]]
                    if not np.isnan(gaia_targets['pmra'][nearby_stars[l]]):
                        ra += gaia_targets['pmra'][nearby_stars[l]] * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
                    if not np.isnan(gaia_targets['pmdec'][nearby_stars[l]]):
                        dec += gaia_targets['pmdec'][nearby_stars[l]] * interval / 1000 / 3600
                    pixel = source.wcs.all_world2pix(np.array([ra, dec]).reshape((1, 2)), 0)
                    x_gaia = pixel[0][0]
                    y_gaia = pixel[0][1]
                    ax0.arrow(source.gaia[f'sector_{sector}_x'][nearby_stars[l]],
                              source.gaia[f'sector_{sector}_y'][nearby_stars[l]],
                              x_gaia - source.gaia[f'sector_{sector}_x'][nearby_stars[l]],
                              y_gaia - source.gaia[f'sector_{sector}_y'][nearby_stars[l]],
                              width=0.02, color='r', edgecolor=None, head_width=0.1)
                    try:
                        txt = ax0.text(source.gaia[f'sector_{sector}_x'][nearby_stars[l]] + 0.5,
                                       source.gaia[f'sector_{sector}_y'][nearby_stars[l]] - 0.05,
                                       f'TIC {int(source.tic["TIC"][index])}', size=7)

                    except TypeError:
                        designation = source.gaia[f"DESIGNATION"][nearby_stars[l]]
                        formatted_text = '\n'.join([designation[i:i + 15] for i in range(0, len(designation), 15)])

                        txt = ax0.text(source.gaia[f'sector_{sector}_x'][nearby_stars[l]] + 0.5,
                                       source.gaia[f'sector_{sector}_y'][nearby_stars[l]] - 0.05,
                                       formatted_text, size=7)
                ax0.set_xlim(round(star_x) - 5.5, round(star_x) + 5.5)
                ax0.set_ylim(round(star_y) - 5.5, round(star_y) + 5.5)
                ax0.set_title(f'TIC_{hdul[0].header["TICID"]}_Sector_{hdul[0].header["SECTOR"]:04d}')
                ax0.vlines(round(star_x) - 2.5, round(star_y) - 2.5, round(star_y) + 2.5, colors='k', lw=1.2)
                ax0.vlines(round(star_x) + 2.5, round(star_y) - 2.5, round(star_y) + 2.5, colors='k', lw=1.2)
                ax0.hlines(round(star_y) - 2.5, round(star_x) - 2.5, round(star_x) + 2.5, colors='k', lw=1.2)
                ax0.hlines(round(star_y) + 2.5, round(star_x) - 2.5, round(star_x) + 2.5, colors='k', lw=1.2)
                try:
                    t_, y_, x_ = np.shape(hdul[0].data)
                except ValueError:
                    warnings.warn('Light curves need to have the primary hdu. Set save_aperture=True when producing the light curve to enable this plot.')
                    sys.exit()
                max_flux = np.max(
                    np.median(source.flux[:, int(star_y) - 2:int(star_y) + 3, int(star_x) - 2:int(star_x) + 3], axis=0))
                arrays = []
                for j in range(y_):
                    for k in range(x_):
                        ax_ = fig.add_subplot(gs[(19 - 2 * j):(21 - 2 * j), (2 * k):(2 + 2 * k)])
                        ax_.patch.set_facecolor('#4682B4')
                        ax_.patch.set_alpha(min(1, max(0, 5 * np.nanmedian(hdul[0].data[:, j, k]) / max_flux)))
                        if detrend:
                            _, trend = flatten(hdul[1].data['time'][q],
                                               hdul[0].data[:, j, k][q] - np.nanmin(hdul[0].data[:, j, k][q]) + 1000,
                                               window_length=1, method='biweight', return_trend=True)
                            cal_aper = (hdul[0].data[:, j, k][q] - np.nanmin(
                                hdul[0].data[:, j, k][q]) + 1000 - trend) / np.nanmedian(
                                hdul[0].data[:, j, k][q]) + 1
                        else:
                            cal_aper = (hdul[0].data[:, j, k][q]) / np.nanmedian(hdul[0].data[:, j, k][q])
                        if 1 <= j <= 3 and 1 <= k <= 3:
                            arrays.append(cal_aper)
                        ax_.plot(hdul[1].data['time'][q], cal_aper, '.k', ms=0.5)
                        # ax_.plot(hdul[1].data['time'][q], hdul[0].data[:, j, k][q], '.k', ms=0.5)
                        ax_.set_ylim(ymin, ymax)
                        ax_.set_xlabel('TBJD')
                        ax_.set_ylabel('')
                        if j != 0:
                            ax_.set_xticklabels([])
                            ax_.set_xlabel('')
                        if k != 0:
                            ax_.set_yticklabels([])
                        if j == 2 and k == 0:
                            ax_.set_ylabel('Normalized and detrended Flux of each pixel')

                combinations = itertools.combinations(arrays, 2)
                median_abs_diffs = []
                for arr_a, arr_b in combinations:
                    abs_diff = np.abs(arr_a - arr_b)
                    median_diff = np.median(abs_diff)
                    median_abs_diffs.append(median_diff)
                median_abs_diffs = np.array(median_abs_diffs)
                iqr = np.percentile(median_abs_diffs, 75) - np.percentile(median_abs_diffs, 25)
                print(f"Interquartile Range (IQR): {iqr}")
                std_dev = np.std(median_abs_diffs)
                print(f"Standard Deviation: {std_dev}")
                ax1 = fig.add_subplot(gs[:9, 3:6])
                ax1.hist(median_abs_diffs, color='k', edgecolor='k', facecolor='none', rwidth=0.8, linewidth=2)
                ax1.set_box_aspect(1)
                # ax1.set_title(f'Distribution of the MADs among combinations of the center 3*3 pixels')
                ax1.set_xlabel('MAD between combinations of fluxes')
                ax1.set_ylabel('Counts')
                text_ax = fig.add_axes([0.6, 0.95, 0.3, 0.3])  # [left, bottom, width, height] in figure coordinates
                text_ax.axis('off')  # Turn off axis lines, ticks, etc.
                text_ax.text(0., 0., f"Gaia DR3 {gaia_dr3} \n"
                                     f" ←← TESS SPOC FFI and TIC/Gaia stars with proper motions. \n"
                                     f"        Arrows show Gaia proper motion after {pm_years} years. \n"
                                     f" ←  Histogram of the MADs between 3*3 pixel fluxes. \n"
                                     f" ↓  Fluxes of each pixels after contaminations are removed. \n"
                                     f"      The fluxes are normalized and detrended. The background \n"
                                     f"      color shows the pixel brightness after the decontamination. \n"
                                     f"How to interpret these plots: \n"
                                     f"     If the signals you are interested in (i.e. transits, \n"
                                     f"     eclipses, variable stars) show similar amplitudes in \n"
                                     f"     all (especially the center 3*3) pixels, then the star \n"
                                     f"     is likely to be the source. The median absolute \n"
                                     f"     differences (MADs) taken between all combinations \n"
                                     f"     of the center pixel fluxes are shown in the histogram \n"
                                     f"     for a quantititive comparison to other possible sources. \n"
                                     f"     The star with smaller distribution width (IQR or \n"
                                     f"     STD) is more likely to be the source of the signal. \n"
                                     f"\n"
                                     f"Interquartile Range (IQR): {iqr:05f} \n"
                                     f"Standard Deviation: {std_dev:05f}", transform=text_ax.transAxes, ha='left',
                             va='top')
                plt.subplots_adjust(top=.97, bottom=0.06, left=0.05, right=0.95)
                plt.savefig(
                    f'{local_directory}plots/contamination_sector_{hdul[0].header["SECTOR"]:04d}_Gaia_DR3_{gaia_dr3}.pdf',
                    dpi=300,)
                # plt.savefig(f'{local_directory}plots/contamination_sector_{hdul[0].header["SECTOR"]:04d}_Gaia_DR3_{gaia_dr3}.png',
                #             dpi=600)
                plt.close()

def plot_epsf(local_directory=None):
    files = glob(f'{local_directory}epsf/*.npy')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        psf = np.load(files[i])
        plt.imshow(psf[0, :23 ** 2].reshape(23, 23), cmap='bone', origin='lower')
        plt.tick_params(axis='x', bottom=False)
        plt.tick_params(axis='y', left=False)
        plt.title(f'{files[i].split("/")[-1].split(".")[0]}')
        plt.savefig(f'{local_directory}plots/{files[i].split("/")[-1]}.png', bbox_inches='tight', dpi=300)


def choose_prior(tics, local_directory=None, priors=np.logspace(-5, 0, 100)):
    mad = np.zeros((2, 100))
    for i in trange(len(priors)):
        resid = get_tglc_lc(tics=tics, method='query', server=1, directory=local_directory, prior=priors[i])
        print(resid)
        mad[:, i] = resid
        # with fits.open(
        #         '/home/tehan/data/cosmos/GEMS/TIC 16005254/lc/hlsp_tglc_tess_ffi_gaiaid-52359538285081728-s0043-cam3-ccd3_tess_v1_llc.fits',
        #         mode='denywrite') as hdul:
        #     mad[0, i] = np.nanmedian(abs(hdul[1].data['cal_psf_flux'] - np.nanmedian(hdul[1].data['cal_psf_flux'])))
        # with fits.open(
        #         '/home/tehan/data/cosmos/GEMS/TIC 16005254/lc/hlsp_tglc_tess_ffi_gaiaid-52359538285081728-s0044-cam1-ccd1_tess_v1_llc.fits',
        #         mode='denywrite') as hdul:
        #     mad[1, i] = np.nanmedian(abs(hdul[1].data['cal_psf_flux'] - np.nanmedian(hdul[1].data['cal_psf_flux'])))
    np.save('/home/tehan/Documents/GEMS/TIC 16005254/mad.npy', mad)
    # plt.plot(priors, mad)
    # plt.xscale('log')
    # plt.title(f'best prior = {priors[np.argmin(mad)]:04d}')
    # plt.show()


def get_tglc_lc(tics=None, method='query', server=1, directory=None, prior=None):
    if method == 'query':
        for i in range(len(tics)):
            target = f'TIC {tics[i]}'
            local_directory = f'{directory}{target}/'
            os.makedirs(local_directory, exist_ok=True)
            tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=True, limit_mag=16,
                    get_all_lc=False, first_sector_only=False, last_sector_only=False, sector=None, prior=prior,
                    transient=None)
            plot_lc(local_directory=f'{directory}TIC {tics[i]}/', kind='cal_aper_flux')
    if method == 'search':
        star_spliter(server=server, tics=tics, local_directory=directory)


if __name__ == '__main__':
    tics = [16005254]
    directory = f'/Users/tehan/Downloads/'
    os.makedirs(directory, exist_ok=True)
    get_tglc_lc(tics=tics, method='query', server=1, directory=directory)
    # plot_lc(local_directory=f'{directory}TIC {tics[0]}/', kind='cal_aper_flux')
    # plot_lc(local_directory=f'/home/tehan/Documents/tglc/TIC 16005254/', kind='cal_aper_flux', ylow=0.9, yhigh=1.1)
    # plot_contamination(local_directory=f'{directory}TIC {tics[0]}/', gaia_dr3=5751990597042725632)
    # plot_epsf(local_directory=f'{directory}TIC {tics[0]}/')
    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', period=0.71912603, mid_transit_tbjd=2790.58344,
    #            kind='cal_psf_flux')
    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', period=0.23818244, mid_transit_tbjd=1738.71248,
    #            kind='cal_aper_flux')
import os
import pickle
import warnings
from glob import glob
from tqdm import trange
from wotan import flatten
from astropy.io import ascii
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from functools import partial
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut, _dot_wait
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
# Tesscut._service_api_connection.TIMEOUT = 6000
import seaborn as sns
# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np
import matplotlib.patheffects as path_effects
import itertools
import sys
from astropy import units
from astroquery.utils.tap.core import TapPlus
from astroquery.mast import Catalogs
import pdb
controller = ThreadpoolController()


@controller.wrap(limits=1, user_api='blas')
def tglc_lc(target='TIC 264468702', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
            first_sector_only=False, last_sector_only=False, sector=None, prior=None, transient=None, ffi='SPOC',
            mast_timeout=3600):
    '''
    Generate light curve for a single target.

    :param target: target identifier
    :kind target: str, required
    :param local_directory: output directory
    :kind local_directory: str, required
    :param size: size of the FFI cut, default size is 90. Recommend large number for better quality. Cannot exceed 100.
    :kind size: int, optional
    :param mast_timeout: timeout in seconds for MAST Tesscut requests
    :kind mast_timeout: int, optional
    '''
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'plots/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    print(f'Target: {target}')
    target_ = Catalogs.query_object(target, radius=42 * 0.707 / 3600, catalog="Gaia", version=2)
    if len(target_) == 0:
        print('TIC not found in Gaia DR2')
        target_ = Catalogs.query_object(target.name, radius=5 * 21 * 0.707 / 3600, catalog="Gaia", version=2)
    ra = target_[0]['ra']
    dec = target_[0]['dec']
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    sector_table = Tesscut.get_sectors(coordinates=coord)
    print(sector_table)
    print(f'Found {len(sector_table)} sector(s) for this target.')
    if get_all_lc:
        name = None
    else:
        catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
        if target[0:3] == 'TIC':
            # name = int(target[4:])
            TIC_ID = int(target[4:])
            ticvals = Catalogs.query_object('TIC {}'.format(TIC_ID), radius=3.0 * units.arcsec.to('degree'),
                                            catalog="tic")
            if len(ticvals) > 1:
                ticvals = ticvals[ticvals['ID'].astype(int) == TIC_ID]
            gaia_id = str(ticvals['GAIA'][0])
            try:
                tmpgaiavals = TapPlus(url="https://gea.esac.esa.int/tap-server/tap").launch_job(
                    "SELECT TOP 1 * FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id = {}".format(
                        gaia_id)).get_results().to_pandas()
                gaiavals = TapPlus(url="https://gea.esac.esa.int/tap-server/tap").launch_job(
                    "SELECT TOP 1 * FROM gaiadr3.gaia_source WHERE source_id = {}".format(
                        tmpgaiavals.loc[0, 'dr3_source_id'])).get_results().to_pandas()
                print('The DR2 ID is {}'.format(tmpgaiavals.loc[0, 'dr2_source_id']))
                print('The DR3 designation is {}'.format(gaiavals.loc[0, 'designation'.upper()]))
                name = f'{gaiavals.loc[0, "designation".upper()]}'
            except Exception as exc:
                warnings.warn(
                    f'Gaia TAP DR2->DR3 lookup failed ({exc}). Falling back to TIC catalog GAIA={gaia_id}.'
                )
                name = f'Gaia DR3 {gaia_id}'
            print(name)
        elif transient is not None:
            name = transient[0]
        else:
            try:
                catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f'MAST name lookup failed for "{target}": {e}')
            name = int(np.array(catalogdata['ID'])[0])
            print("Since the provided target is not TIC ID, the resulted light curve with get_all_lc=False can not be "
                  "guaranteed to be the target's light curve. Please check the TIC ID of the output file before using "
                  "the light curve or try use TIC ID as the target in the format of 'TIC 12345678'.")
    if type(sector) == int:
        print(f'Only processing Sector {sector}.')
        print('Downloading data from MAST and Gaia.')
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient, ffi=ffi, mast_timeout=mast_timeout)  # sector
        source.select_sector(sector=sector)
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior, ffi=ffi)
    elif first_sector_only:
        print(f'Only processing the first sector the target is observed in: Sector {sector_table["sector"][0]}.')
        print('Downloading data from MAST and Gaia.')
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')
        sector = sector_table["sector"][0]
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient, ffi=ffi, mast_timeout=mast_timeout)  # sector
        source.select_sector(sector=source.sector_table['sector'][0])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior, ffi=ffi)
    elif last_sector_only:
        print(f'Only processing the last sector the target is observed in: Sector {sector_table["sector"][-1]}.')
        print('Downloading data from MAST and Gaia.')
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')
        sector = sector_table["sector"][-1]
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient, ffi=ffi, mast_timeout=mast_timeout)  # sector
        source.select_sector(sector=source.sector_table['sector'][-1])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior, ffi=ffi)
    elif sector == None:
        print(f'Processing all available sectors of the target.')
        print('Downloading data from MAST and Gaia.')
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')
        for j in range(len(sector_table)):
            print(f'################################################')
            print(f'Downloading Sector {sector_table["sector"][j]}.')
            attempt = 0
            source = None
            last_error = None
            while attempt < 5:
                try:
                    source = ffi_cut(target=target, size=size, local_directory=local_directory,
                                     sector=sector_table['sector'][j],
                                     limit_mag=limit_mag, transient=transient, ffi=ffi, mast_timeout=mast_timeout)
                    break
                except Exception as exc:
                    attempt += 1
                    last_error = exc
                    print(f'Attempt {attempt}/5 failed for sector {sector_table["sector"][j]}: {exc}')
            if source is None:
                raise RuntimeError(
                    f'Failed to initialize ffi_cut for sector {sector_table["sector"][j]} after 5 attempts.'
                ) from last_error

            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior, ffi=ffi)
    else:
        print(
            f'Processing all available sectors of the target in a single run. Note that if the number of sectors is '
            f'large, the download might cause a timeout error from MAST.')
        print('Downloading data from MAST and Gaia.')
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient, ffi=ffi, mast_timeout=mast_timeout)  # sector
        for j in range(len(source.sector_table)):
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior, ffi=ffi)


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
            # plt.xlim(xlow, xhigh)
            plt.ylim(ylow, yhigh)
            plt.xlim(xlow, xhigh)
            # plt.ylim(0.5, 1.5)
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
def _phase_centered(t, period, t0):
    return ((t - t0 + 0.5 * period) % period) / period - 0.5


def plot_pf_lc(local_directory=None, period=None, mid_transit_tbjd=None, kind='cal_aper_flux',
               min_period=1.7, max_period=1.8, phase_window=0.5, show=False, frequency_factor=1.0,
               manual_period=None, manual_mid_transit_tbjd=None):
    files = sorted(glob(f'{local_directory}*.fits'))
    if len(files) == 0:
        raise FileNotFoundError(f'No FITS files found in: {local_directory}')

    os.makedirs(f'{local_directory}plots/', exist_ok=True)

    sector_series = []
    t_all = []
    f_all = []
    ticid = 'unknown'

    for file in files:
        with fits.open(file, mode='denywrite') as hdul:
            if 'TESS_flags' not in hdul[1].data.names or 'TGLC_flags' not in hdul[1].data.names:
                raise KeyError('Missing TESS_flags or TGLC_flags in light curve table.')
            # Clean for BLS/plotting: only keep cadences that pass both quality flags.
            q = (hdul[1].data['TESS_flags'] == 0) & (hdul[1].data['TGLC_flags'] == 0)
            if len(hdul[1].data[kind]) != len(hdul[1].data['time']):
                continue
            t = np.array(hdul[1].data['time'][q], dtype=float)
            f = np.array(hdul[1].data[kind][q], dtype=float)
            valid = np.isfinite(t) & np.isfinite(f)
            t = t[valid]
            f = f[valid]
            if len(t) < 20:
                continue
            f_med = np.nanmedian(f)
            if not np.isfinite(f_med) or f_med == 0:
                continue
            f = f / f_med
            sector = int(hdul[0].header.get('SECTOR', -1))
            ticid = hdul[0].header.get('TICID', ticid)
            sector_series.append((sector, t, f))
            t_all.append(t)
            f_all.append(f)

    if len(sector_series) == 0:
        raise RuntimeError('No usable sector data found for phase-fold plotting.')

    t_all = np.concatenate(t_all)
    f_all = np.concatenate(f_all)

    sector_series = sorted(sector_series, key=lambda row: row[0])
    bls_sector, bls_t, bls_f = sector_series[-1]

    if manual_period is not None or manual_mid_transit_tbjd is not None:
        if manual_period is None or manual_mid_transit_tbjd is None:
            raise ValueError('manual_period and manual_mid_transit_tbjd must both be provided.')
        period = float(manual_period)
        mid_transit_tbjd = float(manual_mid_transit_tbjd)
        best_duration = 0.1 * float(period)
        ephem_source = 'manual'
    elif period is not None and mid_transit_tbjd is not None:
        period = float(period)
        mid_transit_tbjd = float(mid_transit_tbjd)
        best_duration = 0.1 * float(period)
        ephem_source = 'manual'
    else:
        print(f'Running BLS on last available sector: {bls_sector}')
        if frequency_factor <= 0:
            raise ValueError('frequency_factor must be > 0.')
        f_scatter = 1.4826 * np.nanmedian(np.abs(bls_f - np.nanmedian(bls_f)))
        if not np.isfinite(f_scatter) or f_scatter <= 0:
            f_scatter = np.nanstd(bls_f)
        if not np.isfinite(f_scatter) or f_scatter <= 0:
            f_scatter = 1e-3
        bls = BoxLeastSquares(bls_t, bls_f, dy=np.full_like(bls_f, f_scatter))
        duration_max = min(0.25, 0.5 * float(min_period))
        duration_min = max(0.002, min(0.02, duration_max / 5))
        if duration_min >= duration_max:
            duration_min = 0.5 * duration_max
        durations = np.linspace(duration_min, duration_max, 20)
        periods = bls.autoperiod(
            durations,
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=frequency_factor
        )
        power = bls.power(periods, durations, objective='snr')
        best = int(np.nanargmax(power.power))
        period = float(power.period[best])
        mid_transit_tbjd = float(power.transit_time[best])
        best_duration = float(power.duration[best])
        ephem_source = f'BLS sector {bls_sector}'

    phase_window = float(phase_window)
    if phase_window <= 0 or phase_window > 1:
        raise ValueError('phase_window must be in (0, 1].')
    half_window = phase_window / 2

    fig, ax = plt.subplots(figsize=(9, 5))
    n_sector = len(sector_series)
    if n_sector <= 20:
        cmap = plt.get_cmap('tab20', n_sector)
    else:
        cmap = plt.get_cmap('turbo', n_sector)
    bins = np.linspace(-half_window, half_window, 61)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for i, (sector, t, f) in enumerate(sector_series):
        color = cmap(i)
        phase = _phase_centered(t, period, mid_transit_tbjd)
        keep = np.abs(phase) <= half_window
        if np.any(keep):
            phase_keep = phase[keep]
            flux_keep = f[keep]
            ax.scatter(phase_keep, flux_keep, s=7, alpha=0.2, color=color, rasterized=True)

            # Sector-level median curve
            binned_sector = np.full_like(centers, np.nan, dtype=float)
            idx_sector = np.digitize(phase_keep, bins) - 1
            for k in range(len(centers)):
                in_bin = idx_sector == k
                if np.any(in_bin):
                    binned_sector[k] = np.nanmedian(flux_keep[in_bin])
            valid_sector = np.isfinite(binned_sector)
            if np.any(valid_sector):
                ax.plot(centers[valid_sector], binned_sector[valid_sector], color=color, lw=1.8,
                        label=f'Sector {sector} median')

    phase_all = _phase_centered(t_all, period, mid_transit_tbjd)
    keep_all = np.abs(phase_all) <= half_window
    if np.sum(keep_all) > 40:
        binned = np.full_like(centers, np.nan, dtype=float)
        idx = np.digitize(phase_all[keep_all], bins) - 1
        flux_keep = f_all[keep_all]
        for k in range(len(centers)):
            in_bin = idx == k
            if np.any(in_bin):
                binned[k] = np.nanmedian(flux_keep[in_bin])
        valid = np.isfinite(binned)
        ax.plot(centers[valid], binned[valid], color='k', lw=1.8, label='All-sector median')

    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7)
    ax.set_xlim(-half_window, half_window)
    ax.set_ylim(0.99, 1.01)
    ax.set_xlabel('Phase (cycles)')
    ax.set_ylabel('Normalized flux')
    ax.set_title(
        f'TIC {ticid} phase-folded {kind}\n'
        f'P={period:.6f} d, T0={mid_transit_tbjd:.6f}, duration={best_duration:.4f} d [{ephem_source}]'
    )
    ax.legend(loc='best', fontsize=8, ncol=2)

    out_png = f'{local_directory}/plots/TIC_{ticid}_BLS_phase_folded_{kind}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    print(f'Ephemeris used ({ephem_source}): P={period:.6f} d, T0={mid_transit_tbjd:.6f}')
    print(f'Saved phase-folded plot: {out_png}')
    return period, mid_transit_tbjd


def plot_pf_lc_points(local_directory=None, period=None, mid_transit_tbjd=None, kind='cal_aper_flux'):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    t_all = np.array([])
    f_all = np.array([])

    for j in range(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            q = (hdul[1].data['TESS_flags'] == 0) & (hdul[1].data['TGLC_flags'] == 0)
            if len(hdul[1].data['cal_aper_flux']) == len(hdul[1].data['time']):
                if hdul[0].header["SECTOR"] <= 26:
                    t = hdul[1].data['time'][q]
                    f = hdul[1].data[kind][q]
                elif hdul[0].header["SECTOR"] <= 55:
                    t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // 3 * 3].reshape(-1, 3), axis=1)
                    f = np.mean(hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // 3 * 3].reshape(-1, 3), axis=1)
                else:
                    t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // 9 * 9].reshape(-1, 9), axis=1)
                    f = np.mean(hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // 9 * 9].reshape(-1, 9), axis=1)

                t_all = np.append(t_all, t)
                f_all = np.append(f_all, f)
                idxs = np.where(np.diff(t % period / period) < -0.1)
                id = np.where(np.diff(t % period / period) > 0.1)
                idxs = [0] + list(idxs[0]) + [len(t)] + list(id[0])
                idxs = sorted(idxs)
                print(idxs)
                for i in range(len(idxs)-1):
                    ax.plot(t[idxs[i]+1:idxs[i+1]+1] % period / period, f[idxs[i]+1:idxs[i+1]+1],
                            color='orange', markersize=4, zorder=2, lw=4)

                title = f'TIC_{hdul[0].header["TICID"]}'

    # Remove all decorations
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xlim(0.155, 0.235)
    plt.savefig(f'{local_directory}/plots/{title}_minimal.png',
                dpi=300, bbox_inches='tight', pad_inches=0,
                transparent=True)
    plt.show()
    plt.close(fig)

# newest
def plot_contamination(local_directory=None, gaia_dr3=None, ymin=None, ymax=None, pm_years=3000):
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.9'})
    if gaia_dr3 is None:
        files = glob(f'{local_directory}lc/*.fits')
    else:
        files = glob(f'{local_directory}lc/*{gaia_dr3}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        try:
            with fits.open(files[i], mode='denywrite') as hdul:
                gaia_dr3 = hdul[0].header['GAIADR3']
                TIC = hdul[0].header['TICID']
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
                        np.median(source.flux[:, int(star_y) - 2:int(star_y) + 3, int(star_x) - 2:int(star_x) + 3],
                                  axis=0))
                    arrays = []
                    for j in range(y_):
                        for k in range(x_):
                            ax_ = fig.add_subplot(gs[(19 - 2 * j):(21 - 2 * j), (2 * k):(2 + 2 * k)])
                            ax_.patch.set_facecolor('#4682B4')
                            ax_.patch.set_alpha(min(1, max(0, 5 * np.nanmedian(hdul[0].data[:, j, k]) / max_flux)))

                            _, trend = flatten(hdul[1].data['time'][q],
                                               hdul[0].data[:, j, k][q] - np.nanmin(hdul[0].data[:, j, k][q]) + 1000,
                                               window_length=1, method='biweight', return_trend=True)
                            cal_aper = (hdul[0].data[:, j, k][q] - np.nanmin(
                                hdul[0].data[:, j, k][q]) + 1000 - trend) / np.nanmedian(
                                hdul[0].data[:, j, k][q]) + 1
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
                    try:
                        ax1.hist(median_abs_diffs, color='k', edgecolor='k', facecolor='none', rwidth=0.8, linewidth=2)
                    except ValueError:
                        continue
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
                        f'{local_directory}plots/TIC_{TIC}_contamination_sector_{hdul[0].header["SECTOR"]:04d}_Gaia_DR3_{gaia_dr3}.pdf',
                        dpi=300,)
                    # plt.savefig(f'{local_directory}plots/contamination_sector_{hdul[0].header["SECTOR"]:04d}_Gaia_DR3_{gaia_dr3}.png',
                    #             dpi=600)
                    plt.close()
        except:
            continue


def plot_contamination_shane(local_directory=None, gaia_dr3=None, ymin=None, ymax=None, pm_years=3000,
                             show_colorbar=True, output_directory=None):
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.9'})
    if gaia_dr3 is None:
        files = glob(f'{local_directory}lc/*.fits')
    else:
        files = glob(f'{local_directory}lc/*{gaia_dr3}*.fits')
    plot_root = local_directory if output_directory is None else output_directory
    os.makedirs(f'{plot_root}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            gaia_dr3 = hdul[0].header['GAIADR3']
            sector = hdul[0].header['SECTOR']
            q = [a and b for a, b in
                 zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            if ymin is None and ymax is None:
                ymin = np.nanmin(hdul[1].data['cal_aper_flux'][q]) - 0.05
                ymax = np.nanmax(hdul[1].data['cal_aper_flux'][q]) + 0.05
            fig = plt.figure(constrained_layout=False, figsize=(15, 6))
            gs = fig.add_gridspec(11, 10)
            gs.update(wspace=0.05, hspace=0.15)
            try:
                t_, y_, x_ = np.shape(hdul[0].data)
            except ValueError:
                warnings.warn(
                    'Light curves need to have the primary hdu. Set save_aperture=True when producing the light curve to enable this plot.')
                sys.exit()
            median_aperture = np.nanmedian(hdul[0].data, axis=0)
            max_flux = np.nanmax(median_aperture)
            if not np.isfinite(max_flux) or max_flux == 0:
                max_flux = 1
            pixel_brightness = np.clip(median_aperture / max_flux, 0, 1)
            pixel_brightness = np.nan_to_num(pixel_brightness)
            brightness_cmap = LinearSegmentedColormap.from_list(
                'soft_blues',
                plt.cm.Blues(np.linspace(0.05, 0.78, 256))
            )
            center_axes = []

            for j in range(y_):
                for k in range(x_):
                    ax_ = fig.add_subplot(gs[(9 - 2 * j):(11 - 2 * j), (2 * k):(2 + 2 * k)])
                    ax_.patch.set_facecolor(brightness_cmap(pixel_brightness[j, k]))

                    _, trend = flatten(hdul[1].data['time'][q],
                                       hdul[0].data[:, j, k][q] - np.nanmin(hdul[0].data[:, j, k][q]) + 1000,
                                       window_length=1, method='biweight', return_trend=True)
                    cal_aper = (hdul[0].data[:, j, k][q] - np.nanmin(
                        hdul[0].data[:, j, k][q]) + 1000 - trend) / np.nanmedian(
                        hdul[0].data[:, j, k][q]) + 1
                    if 1 <= j <= 3 and 1 <= k <= 3:
                        center_axes.append(ax_)
                    ax_.plot(hdul[1].data['time'][q], cal_aper, '.k', ms=0.5)
                    ax_.set_ylim(0.3, 1.1)
                    ax_.set_xlim(2814.5, 2818.5)
                    ax_.set_xlabel('TBJD')
                    ax_.set_ylabel('')
                    if j != 0:
                        ax_.set_xticklabels([])
                        ax_.set_xlabel('')
                    if k != 0:
                        ax_.set_yticklabels([])
                    if j == 2 and k == 0:
                        ax_.set_ylabel('Normalized and detrended Flux of each pixel')

            right_margin = 0.92 if show_colorbar else 0.95
            plt.subplots_adjust(top=.97, bottom=0.1, left=0.05, right=right_margin)
            if show_colorbar:
                cax = fig.add_axes([0.93, 0.05, 0.015, 0.87])
                norm = plt.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=brightness_cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label('Relative median pixel brightness')

            if center_axes:
                x0 = min(ax.get_position().x0 for ax in center_axes)
                x1 = max(ax.get_position().x1 for ax in center_axes)
                y0 = min(ax.get_position().y0 for ax in center_axes)
                y1 = max(ax.get_position().y1 for ax in center_axes)
                rect = plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    transform=fig.transFigure, fill=False, edgecolor='red', linewidth=2, zorder=10
                )
                fig.add_artist(rect)

            plt.savefig(
                f'{plot_root}plots/TIC_{hdul[0].header["TICID"]}_contamination_sector_{hdul[0].header["SECTOR"]:04d}_Gaia_DR3_{gaia_dr3}.pdf',
                dpi=300,)
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

def get_tglc_lc(tics=None, sectors=None, method='query', server=1, directory=None, prior=None):
    """
    Downloads and plots TESS light curves for a given list of TICs and their corresponding sectors.

    Parameters
    ----------
    tics : list of int, optional
        List of TESS Input Catalog (TIC) IDs. Defaults to None.
    sectors : list of int, optional
        List of corresponding sectors for each TIC ID. Must have the same length as tics if provided.
        Defaults to None.
    method : str, optional
        Method to use for retrieving light curves. 'query' downloads individual light curves,
        'search' might use a different approach (requires star_spliter function). Defaults to 'query'.
    server : int, optional
        Server number to use if applicable. Defaults to 1.
    directory : str, optional
        Local directory to save the downloaded light curves. Defaults to None.
    prior : str, optional
        Prior information to pass to the light curve retrieval function. Defaults to None.
    """
    if method == 'query':
        if tics is None:
            print("Error: 'tics' list cannot be None when method is 'query'.")
            return
        if sectors is None:
            print("Getting all sectors possible for each target. ")
            sectors = [None] * len(tics)
        if len(tics) != len(sectors):
            print("Error: The 'tics' and 'sectors' lists must have the same length.")
            return

        for i in range(len(tics)):
            target = f'TIC {tics[i]}'
            local_directory = f'{directory}{target}/'
            os.makedirs(local_directory, exist_ok=True)
            tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=True, limit_mag=16,
                    get_all_lc=False, first_sector_only=False, last_sector_only=False, sector=None, prior=prior,
                    transient=None)
            plot_lc(local_directory=f'{directory}TIC {tics[i]}/', kind='cal_aper_flux')
    elif method == 'search':
        star_spliter(server=server, tics=tics, local_directory=directory)
    else:
        print(f"Error: Unknown method '{method}'. Choose either 'query' or 'search'.")


if __name__ == '__main__':
    tics = [389040826, 311276853]
    directory = f'/Users/tehan/Downloads/'
    os.makedirs(directory, exist_ok=True)
    get_tglc_lc(tics=tics, method='query', server=1, directory=directory)
    plot_lc(local_directory=f'{directory}TIC {tics[0]}/', kind='cal_aper_flux', ylow=0.99, yhigh=1.01)
    # plot_lc(local_directory=f'/home/tehan/Documents/tglc/TIC 16005254/', kind='cal_aper_flux', ylow=0.9, yhigh=1.1)
    # plot_contamination(local_directory=f'{directory}TIC {tics[0]}/', gaia_dr3=4652877439164133760)
    # plot_epsf(local_directory=f'{directory}TIC {tics[0]}/')
    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', kind='cal_aper_flux', manual_period=1.7666920094826, manual_mid_transit_tbjd=1817.684965)
    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', period=11, mid_transit_tbjd=1738.71248,
    #            kind='aperture_flux')

    # directory='/home/tehan/data/cosmos/transit_depth_validation_contamrt/'
    # tic_sector = ascii.read(glob(f'{directory}deviation_TGLC_extra.dat')[0])
    # print(len(tic_sector))
    # TIC_183985250_2
    # TIC_288735205_20
    # TIC_339672028_9
    # tics=[183985250,288735205,339672028]
    # sectors=[2,20,9]
    # for i in range(len(tic_sector)):
    #     tics.append(tic_sector[i]['Star_sector'].split('_')[1])
    #     sectors.append(int(tic_sector[i]['Star_sector'].split('_')[2]))

    # for i in range(len(tics)):
    #     # try:
    #     target = f'TIC {tics[i]}'
    #     local_directory = f'{directory}{target}/'
    #     os.makedirs(local_directory, exist_ok=True)
    #     tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=16,
    #             get_all_lc=False, first_sector_only=False, last_sector_only=False, sector=sectors[i], prior=None,
    #             transient=None)
    #     plot_lc(local_directory=f'{directory}TIC {tics[i]}/', kind='cal_aper_flux', xlow=None, xhigh=None, ylow=0.97,
    #             yhigh=1.03)
    #     except Exception as e:
    #         with open(f"{directory}error_log.txt", "w") as file:
    #             file.write(f"An error occurred for {target}: {e}\n")

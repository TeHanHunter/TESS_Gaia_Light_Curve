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

# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np

controller = ThreadpoolController()


@controller.wrap(limits=1, user_api='blas')
def tglc_lc(target='TIC 264468702', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
            first_sector_only=False, sector=None, prior=None):
    '''
    Generate light curve for a single target.

    :param target: target identifier
    :type target: str, required
    :param local_directory: output directory
    :type local_directory: str, required
    :param size: size of the FFI cut, default size is 90. Recommend large number for better quality. Cannot exceed 100.
    :type size: int, optional
    '''
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    if first_sector_only:
        sector = True
    source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector)  # sector
    if get_all_lc:
        name = None
    else:
        catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
        if target[0:3] == 'TIC':
            name = int(target[4:])
        else:
            name = int(np.array(catalogdata['ID'])[0])
            print("Since the provided target is not TIC ID, the resulted light curve with get_all_lc=False can not be "
                  "guaranteed to be the target's light curve. Please check the TIC ID of the output file before using "
                  "the light curve or try use TIC ID as the target in the format of 'TIC 12345678'.")
    if type(sector) == int:
        source.select_sector(sector=sector)
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    else:
        for j in range(len(source.sector_table)):
            # try:
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
            if first_sector_only:
                break


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


def star_spliter(server=1,  # or 2
                 tics=None, local_directory=None):
    for i in range(server, 27, 2):
        with Pool(16) as p:
            p.map(partial(search_stars, sector=i, tics=tics, local_directory=local_directory), range(16))
    return


def plot_lc(local_directory=None):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            plt.figure(constrained_layout=False, figsize=(8, 4))
            # plt.plot(hdul[1].data['time'], hdul[1].data['cal_psf_flux'], '.', label='cal_psf')
            # cal_aper_lc, trend = flatten(hdul[1].data['time'], hdul[1].data['aperture_flux'] + np.min(hdul[1].data['aperture_flux']) + 1000, window_length=1,
            #                             method='biweight', return_trend=True)
            # cal_aper = (hdul[1].data['aperture_flux'] + np.min(hdul[1].data['aperture_flux']) + 1000 - trend) / np.median(hdul[1].data['aperture_flux']) + 1
            plt.plot(hdul[1].data['time'], hdul[1].data['cal_aper_flux'], '.', c='silver', label='cal_aper')
            plt.plot(hdul[1].data['time'][q], hdul[1].data['cal_aper_flux'][q], '.k', label='cal_aper_flagged')
            # plt.xlim(2845, 2855)
            plt.title(f'TIC_{hdul[0].header["TICID"]}')
            plt.legend()
            # plt.show()
            plt.savefig(
                f'{local_directory}plots/TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}.png',
                dpi=300)


def plot_pf_lc(local_directory=None, period=None):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    fig = plt.figure(figsize=(13, 5))
    for j in range(len(files)):
        not_plotted_num = 0
        with fits.open(files[j], mode='denywrite') as hdul:
            q = [a and b for a, b in
                 zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            if len(hdul[1].data['cal_aper_flux']) == len(hdul[1].data['time']):
                if hdul[0].header["SECTOR"] <= 26:
                    t = hdul[1].data['time'][q]
                    f = hdul[1].data['cal_psf_flux'][q]
                else:
                    t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // 3 * 3].reshape(-1, 3), axis=1)
                    f = np.mean(hdul[1].data['cal_psf_flux'][q][:len(hdul[1].data['cal_psf_flux'][q]) // 3 * 3].reshape(-1, 3), axis=1)
                # plt.plot(hdul[1].data['time'] % period / period, hdul[1].data['cal_aper_flux'], '.', c='silver', ms=.8)
                plt.errorbar(t % period / period, f, hdul[1].header['CAPE_ERR'], c='C1', ls='', elinewidth=0, marker='.', ms=2, zorder=2)
                # , label=f'Sector {hdul[0].header["sector"]}'
            else:
                not_plotted_num += 1
            title = f'TIC_{hdul[0].header["TICID"]} with {len(files) - not_plotted_num} sector(s) of data'
    PDCSAP_files = glob('/home/tehan/Documents/GEMS/TIC 172370679/PDCSAP/*.txt')
    for i in range(len(files)):
        PDCSAP = ascii.read(PDCSAP_files[i])
        t = np.mean(PDCSAP['col1'][:len(PDCSAP['col1']) // 15 * 15].reshape(-1, 15), axis=1)
        f = np.mean(PDCSAP['col2'][:len(PDCSAP['col2']) // 15 * 15].reshape(-1, 15), axis=1)
        ferr = np.mean(PDCSAP['col3'][:len(PDCSAP['col3']) // 15 * 15].reshape(-1, 15), axis=1)
        plt.errorbar((t - 2457000) % period / period, f, ferr, c='C0', ls='', elinewidth=0, marker='.', ms=2, zorder=1)
    print(np.median(PDCSAP['col2']))
    plt.ylim(0.94, 1.025)
    plt.xlim(0.84, 0.86)
    plt.legend()
    plt.title(title)
    plt.xlabel('Phase (days)')
    plt.ylabel('Normalized flux')
    plt.savefig(f'{local_directory}/plots/{title}.png', dpi=300)
    plt.close(fig)


def plot_contamination(local_directory=None):
    files = glob(f'{local_directory}source/*.pkl')
    with open(files[0], 'rb') as input_:
        source = pickle.load(input_)
        star_num = np.where(source.gaia['DESIGNATION'] == 'Gaia DR3 2073530190996615424')
        print(source.gaia[500])
        # print(source.gaia[1115])
        print(np.argsort((source.gaia['sector_14_x'][:500] - source.gaia[star_num]['sector_14_x']) ** 2 +
                         (source.gaia['sector_14_y'][:500] - source.gaia[star_num]['sector_14_y']) ** 2))
        print(f'sector = {source.sector}')
        plt.figure(constrained_layout=False, figsize=(5, 5))
        plt.imshow(np.log10(source.flux[0]), cmap='RdBu', vmin=-5, vmax=5)
        # plt.plot(source.gaia['sector_56_y'], source.gaia['sector_56_x'])
        plt.scatter(source.gaia['sector_14_x'][:500], source.gaia['sector_14_y'][:500], s=5, c='r',
                    label='background stars')
        plt.scatter(source.gaia['sector_14_x'][star_num], source.gaia['sector_14_y'][star_num], s=30, c='r', marker='*',
                    label='target star')
        plt.xlim(30, 60)
        plt.ylim(30, 60)
        plt.show()


def choose_prior(local_directory=None, priors=np.logspace(-5, 0, 100)):
    mad = np.zeros((5, 100))
    for i in trange(len(priors)):
        get_tglc_lc(tics=tics, method='query', server=1, directory=local_directory, prior=priors[i])
        with fits.open('/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0014-cam1-ccd1_tess_v1_llc.fits', mode='denywrite') as hdul:
            mad[0,i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open('/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0015-cam1-ccd2_tess_v1_llc.fits', mode='denywrite') as hdul:
            mad[1,i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open('/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0041-cam1-ccd1_tess_v1_llc.fits', mode='denywrite') as hdul:
            mad[2,i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open('/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0054-cam3-ccd2_tess_v1_llc.fits', mode='denywrite') as hdul:
            mad[3,i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open('/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0055-cam3-ccd1_tess_v1_llc.fits',mode='denywrite') as hdul:
            mad[4,i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
    np.save('/home/tehan/data/cosmos/GEMS/TIC 172370679/mad.npy', mad)
    # plt.plot(priors, mad)
    # plt.xscale('log')
    # plt.title(f'best prior = {priors[np.argmin(mad)]:04d}')
    # plt.show()


def get_tglc_lc(tics=None, method='search', server=1, directory=None, prior=None):
    if method == 'query':
        for i in range(len(tics)):
            target = f'TIC {tics[i]}'
            local_directory = f'{directory}{target}/'
            os.makedirs(local_directory, exist_ok=True)
            tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=16,
                    get_all_lc=False, first_sector_only=False, sector=None, prior=prior)
    if method == 'search':
        star_spliter(server=server, tics=tics, local_directory=directory)


if __name__ == '__main__':
    tics = [172370679]
    directory = f'/home/tehan/data/cosmos/GEMS/'
    os.makedirs(directory, exist_ok=True)
    # get_tglc_lc(tics=tics, method='query', server=1, directory=directory, prior=0.0001291549665014884)
    # plot_contamination(local_directory=f'{directory}TIC 172370679/')
    # plot_pf_lc(local_directory=f'{directory}TIC 172370679/lc/', period=29.090312)
    choose_prior(local_directory=directory)
    ####### list of targets example
    # local_directory = '/home/tehan/data/ob_associations/'
    # data = ascii.read(f'{local_directory}Bouret_2021_2013_Ostars.csv')
    # hosts = np.array(data['Gaia EDR3'])
    # for i in range(24, len(hosts)):
    #     tglc_lc(target='Gaia EDR3 ' + str(hosts[i]), local_directory=local_directory, size=90, save_aper=True, get_all_lc=False)

    ####### list of targets
    # local_directory = '/mnt/d/Astro/hpf/'
    # os.makedirs(local_directory + f'logs/', exist_ok=True)
    # os.makedirs(local_directory + f'lc/', exist_ok=True)
    # os.makedirs(local_directory + f'epsf/', exist_ok=True)
    # os.makedirs(local_directory + f'source/', exist_ok=True)
    # data = ascii.read(local_directory + 'hpf_toi_ffi_targets.txt')
    # hosts = np.array(data['TIC'])
    # gaia_name = []
    # for i in range(len(hosts)):
    #     target = hosts[i]  # Target identifier or coordinates TOI-3714
    #     catalogdata = Catalogs.query_object('TIC ' + str(target), radius=0.02, catalog="TIC")
    #     name = 'Gaia DR2 ' + str(np.array(catalogdata['GAIA'])[np.where(catalogdata['ID'] == str(target))[0][0]])
    #     gaia_name.append(name)
    #     print('TIC ' + str(target), name)
    #     size = 90  # int, suggests big cuts
    #     source = ffi_cut(target='TIC ' + str(target), size=size, local_directory=local_directory)
    #     for j in range(len(source.sector_table)):
    #         try:
    #             source.select_sector(sector=source.sector_table['sector'][j])
    #             epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
    #                  name=name)
    #         except:
    #             warnings.warn(f'Skipping sector {source.sector_table["sector"][j]}. (Target not in cut)')
    # np.savetxt('/mnt/d/Astro/hpf/hpf_gaia_dr2.txt', gaia_name)

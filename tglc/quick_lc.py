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
            plt.plot(hdul[1].data['time'], hdul[1].data['cal_aper_flux'], '.', c='silver', label='cal_aper')
            plt.plot(hdul[1].data['time'][q], hdul[1].data['cal_aper_flux'][q], '.k', label='cal_aper_flagged')
            # plt.xlim(2845, 2855)
            plt.title(f'TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}')
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
                    f = np.mean(
                        hdul[1].data['cal_aper_flux'][q][:len(hdul[1].data['cal_aper_flux'][q]) // 3 * 3].reshape(-1,
                                                                                                                  3),
                        axis=1)
                plt.plot(hdul[1].data['time'] % period / period, hdul[1].data['cal_aper_flux'], '.', c=f'C{j}', ms=2)
                plt.errorbar(t % period / period, f, hdul[1].header['CAPE_ERR'], c=f'C{j}', ls='', elinewidth=0,
                             marker='.', ms=2, zorder=2, label=f'Sector {hdul[0].header["sector"]}')
                #
            else:
                not_plotted_num += 1
            title = f'TIC_{hdul[0].header["TICID"]} with {len(files) - not_plotted_num} sector(s) of data'
    # PDCSAP_files = glob('/home/tehan/Documents/GEMS/TIC 172370679/PDCSAP/*.txt')
    # for i in range(len(files)):
    #     PDCSAP = ascii.read(PDCSAP_files[i])
    #     t = np.mean(PDCSAP['col1'][:len(PDCSAP['col1']) // 15 * 15].reshape(-1, 15), axis=1)
    #     f = np.mean(PDCSAP['col2'][:len(PDCSAP['col2']) // 15 * 15].reshape(-1, 15), axis=1)
    #     ferr = np.mean(PDCSAP['col3'][:len(PDCSAP['col3']) // 15 * 15].reshape(-1, 15), axis=1)
    #     plt.errorbar((t - 2457000) % period / period, f, ferr, c='C0', ls='', elinewidth=0, marker='.', ms=2, zorder=1)
    # plt.ylim(0.94, 1.025)
    # plt.xlim(0.84, 0.86)
    plt.legend()
    plt.title(title)
    plt.xlabel('Phase (days)')
    plt.ylabel('Normalized flux')
    plt.savefig(f'{local_directory}/plots/{title}.png', dpi=300)
    plt.close(fig)


def plot_contamination(local_directory=None, gaia_dr3=None):
    files = glob(f'{local_directory}lc/*.fits')
    for i in range(len(files)):
        with open(glob(f'{local_directory}source/*.pkl')[0], 'rb') as input_:
            with fits.open(files[i], mode='denywrite') as hdul:
                sector = hdul[0].header['SECTOR']
                source = pickle.load(input_)
                source.select_sector(sector=sector)
                star_num = np.where(source.gaia['DESIGNATION'] == f'Gaia DR3 {gaia_dr3}')
                # print(source.gaia[891])
                # print(source.gaia[140])
                nearby_stars = np.argsort(
                    (source.gaia[f'sector_{sector}_x'][:1000] - source.gaia[star_num][f'sector_{sector}_x']) ** 2 +
                    (source.gaia[f'sector_{sector}_y'][:1000] - source.gaia[star_num][f'sector_{sector}_y']) ** 2)[1:10]
                # print(f'sector = {source.sector}')
                star_x = source.gaia[star_num][f'sector_{sector}_x'][0]
                star_y = source.gaia[star_num][f'sector_{sector}_y'][0]
                max_flux = np.max(
                    np.median(source.flux[:, round(star_y) - 2:round(star_y) + 3, round(star_x) - 2:round(star_x) + 3],
                              axis=0))
                fig = plt.figure(constrained_layout=False, figsize=(15, 7))
                gs = fig.add_gridspec(5, 10)
                gs.update(wspace=0.5, hspace=0.5)
                ax0 = fig.add_subplot(gs[:5, :5])
                ax0.imshow(source.flux[0], cmap='RdBu', vmin=-max_flux, vmax=max_flux, origin='lower')

                ax0.scatter(source.gaia[f'sector_{sector}_x'][:1000], source.gaia[f'sector_{sector}_y'][:1000], s=50,
                            c='r', label='background stars')
                ax0.scatter(source.gaia[f'sector_{sector}_x'][nearby_stars],
                            source.gaia[f'sector_{sector}_y'][nearby_stars], s=50,
                            c='r', label='background stars')
                for l in range(len(nearby_stars)):
                    index = np.where(
                        source.tic['dr3_source_id'] == int(source.gaia['DESIGNATION'][nearby_stars[l]].split(' ')[-1]))
                    try:
                        ax0.text(source.gaia[f'sector_{sector}_x'][nearby_stars[l]] - 0.1,
                                    source.gaia[f'sector_{sector}_y'][nearby_stars[l]] + 0.2,
                                    f'TIC {int(source.tic["TIC"][index])}', rotation=90)
                    except TypeError:
                        pass
                ax0.scatter(star_x, star_y, s=300, c='r', marker='*', label='target star')

                # ax0.legend()
                ax0.set_xlim(round(star_x) - 5.5, round(star_x) + 5.5)
                ax0.set_ylim(round(star_y) - 5.5, round(star_y) + 5.5)
                ax0.set_title(f'TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}')
                ax0.vlines(round(star_x) - 2.5, round(star_y) - 2.5, round(star_y) + 2.5, colors='k')
                ax0.vlines(round(star_x) + 2.5, round(star_y) - 2.5, round(star_y) + 2.5, colors='k')
                ax0.hlines(round(star_y) - 2.5, round(star_x) - 2.5, round(star_x) + 2.5, colors='k')
                ax0.hlines(round(star_y) + 2.5, round(star_x) - 2.5, round(star_x) + 2.5, colors='k')
                # for j in range(5):
                #     for k in range(5):
                #         ax_ = fig.add_subplot(gs[(4 - j), (5 + k)])
                #         ax_.patch.set_facecolor('C0')
                #         ax_.patch.set_alpha(max(0, np.median(source.flux[:, round(star_y) - 2 + j, round(star_x) - 2 + k]) / max_flux))
                #         cal_lc, trend = flatten(hdul[1].data['time'],
                #                                      source.flux[:, round(star_y) - 2 + j, round(star_x) - 2 + k],
                #                                      window_length=1, method='biweight', return_trend=True)
                #         ax_.plot(hdul[1].data['time'], cal_lc, '.k', ms=1, label='center pixel')

                t_, y_, x_ = np.shape(hdul[0].data)
                max_flux = np.max(
                    np.median(source.flux[:, int(star_y) - 2:int(star_y) + 3, int(star_x) - 2:int(star_x) + 3], axis=0))
                for j in range(y_):
                    for k in range(x_):
                        ax_ = fig.add_subplot(gs[(4 - j), (5 + k)])
                        ax_.patch.set_facecolor('C0')
                        ax_.patch.set_alpha(min(1, max(0, 3 * np.median(hdul[0].data[:, j, k]) / max_flux)))
                        q = [a and b for a, b in
                             zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]

                        _, trend = flatten(hdul[1].data['time'][q],
                                           hdul[0].data[:, j, k][q] - np.min(hdul[0].data[:, j, k][q]) + 1000,
                                           window_length=1, method='biweight', return_trend=True)
                        cal_aper = (hdul[0].data[:, j, k][q] - np.min(
                            hdul[0].data[:, j, k][q]) + 1000 - trend) / np.median(
                            hdul[0].data[:, j, k][q]) + 1
                        ax_.plot(hdul[1].data['time'][q], cal_aper, '.k', ms=1, label='center pixel')
                plt.savefig(f'{local_directory}lc/plots/contamination_sector_{hdul[0].header["SECTOR"]:04d}.pdf',
                            dpi=300)
                plt.show()


def choose_prior(local_directory=None, priors=np.logspace(-5, 0, 100)):
    mad = np.zeros((5, 100))
    for i in trange(len(priors)):
        get_tglc_lc(tics=tics, method='query', server=1, directory=local_directory, prior=priors[i])
        with fits.open(
                '/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0014-cam1-ccd1_tess_v1_llc.fits',
                mode='denywrite') as hdul:
            mad[0, i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open(
                '/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0015-cam1-ccd2_tess_v1_llc.fits',
                mode='denywrite') as hdul:
            mad[1, i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open(
                '/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0041-cam1-ccd1_tess_v1_llc.fits',
                mode='denywrite') as hdul:
            mad[2, i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open(
                '/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0054-cam3-ccd2_tess_v1_llc.fits',
                mode='denywrite') as hdul:
            mad[3, i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
        with fits.open(
                '/home/tehan/data/cosmos/GEMS/TIC 172370679/lc/hlsp_tglc_tess_ffi_gaiaid-2073530190996615424-s0055-cam3-ccd1_tess_v1_llc.fits',
                mode='denywrite') as hdul:
            mad[4, i] = np.median(abs(hdul[1].data['cal_psf_flux'] - np.median(hdul[1].data['cal_psf_flux'])))
    np.save('/home/tehan/data/cosmos/GEMS/TIC 172370679/mad.npy', mad)
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
                    get_all_lc=False, first_sector_only=False, sector=None, prior=prior)
    if method == 'search':
        star_spliter(server=server, tics=tics, local_directory=directory)


if __name__ == '__main__':
    tics = [119585136]
    directory = f'/home/tehan/data/cosmos/GEMS/'
    os.makedirs(directory, exist_ok=True)
    get_tglc_lc(tics=tics, method='query', server=1, directory=directory)
    # plot_contamination(local_directory=f'{directory}TIC 27858644/', gaia_dr3=2091177593123254016)
    # plot_contamination(local_directory=f'{directory}TIC 172370679/', gaia_dr3=2073530190996615424)
    # plot_lc(local_directory=f'{directory}TIC 135272255/lc/')
    # plot_pf_lc(local_directory=f'{directory}TIC 27858644/lc/', period=384)
    # choose_prior(local_directory=directory)
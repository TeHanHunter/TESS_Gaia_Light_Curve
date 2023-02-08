import os
import glob
from tqdm import trange
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
def tglc_lc(target='NGC 7654', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
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
    catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
    name = 'Gaia DR3 ' + str(np.array(catalogdata['GAIA'])[0])
    if get_all_lc:
        name = None
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
        os.makedirs(f'/home/tehan/data/dominic/sector{i:04d}/', exist_ok=True)
        with Pool(16) as p:
            p.map(partial(search_stars, sector=i, tics=tics, local_directory=local_directory), range(16))
    return


def plot_lc(local_directory=None):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            plt.figure(constrained_layout=False, figsize=(8, 4))
            plt.plot(hdul[1].data['time'], hdul[1].data['cal_psf_flux'], '.', label='cal_psf')
            plt.plot(hdul[1].data['time'], hdul[1].data['cal_aper_flux'], '.', label='cal_aper')
            plt.title(f'TIC_{hdul[0].header["TICID"]}')
            plt.legend()
            plt.savefig(f'{local_directory}plots/TIC_{hdul[0].header["TICID"]}.png', dpi=300)


def get_tglc_lc(tics=None, method='search', server=1, local_directory=None):
    if method == 'query':
        for i in range(len(tics)):
            target = f'TIC {tics[i]}'
            local_directory = f'{local_directory}{target}/'
            os.makedirs(local_directory, exist_ok=True)
            tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=16,
                    get_all_lc=False, first_sector_only=False, sector=None, prior=None)
    if method == 'search':
        star_spliter(server=server,  tics=tics, local_directory=local_directory)


if __name__ == '__main__':
    tics = [236785891, 380517859, 72889156, 289666986, 114947483, 264468702, 12938488]
    local_directory = f'/home/tehan/cosmos/GEMS/TIC {tics}/'
    os.makedirs(local_directory, exist_ok=True)
    get_tglc_lc(tics=tics, method='search', server=2, local_directory=local_directory)

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

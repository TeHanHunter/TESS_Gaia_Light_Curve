# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import os
import time
import requests

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from tqdm import trange
from astropy.io import fits
from tglc.target_lightcurve import epsf
from multiprocessing import Pool
from functools import partial
from glob import glob
from astropy.table import Table
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
import multiprocessing as mp


def lc_per_cut(i, camccd='', local_directory='', target_list=None):
    cut_x = i // 14
    cut_y = i % 14
    with open(f'{local_directory}source/{camccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
        source = pickle.load(input_)
    epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, sector=source.sector, power=1.4,
         local_directory=local_directory, limit_mag=20, save_aper=False, no_progress_bar=True, target_list=target_list)


def lc_per_ccd(camccd='1-1', local_directory='', target_list=None):
    os.makedirs(f'{local_directory}epsf/{camccd}/', exist_ok=True)
    with Pool() as p:
        p.map(partial(lc_per_cut, camccd=camccd, local_directory=local_directory, target_list=target_list), range(196))


def plot_epsf(sector=1, camccd='', local_directory=''):
    fig = plt.figure(constrained_layout=False, figsize=(20, 9))
    gs = fig.add_gridspec(14, 30)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(196):
        cut_x = i // 14
        cut_y = i % 14
        psf = np.load(f'{local_directory}epsf/{camccd}/epsf_{cut_x:02d}_{cut_y:02d}_sector_{sector}_{camccd}.npy')
        cmap = 'bone'
        if np.isnan(psf).any():
            cmap = 'inferno'
        ax = fig.add_subplot(gs[13 - cut_y, cut_x])
        ax.imshow(psf[0, :23 ** 2].reshape(23, 23), cmap=cmap, origin='lower')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', left=False)
    input_files = glob(f'{local_directory}ffi/*{camccd}-????-?_ffic.fits')
    with fits.open(input_files[0], mode='denywrite') as hdul:
        flux = hdul[1].data[0:2048, 44:2092]
        ax_1 = fig.add_subplot(gs[:, 16:])
        ax_1.imshow(np.log10(flux), origin='lower')
    fig.text(0.25, 0.08, 'CUT X (0-13)', ha='center')
    fig.text(0.09, 0.5, 'CUT Y (0-13)', va='center', rotation='vertical')
    fig.suptitle(f'ePSF for sector:{sector} camera-ccd:{camccd}', x=0.5, y=0.92, size=20)
    plt.savefig(f'{local_directory}log/epsf_sector_{sector}_{camccd}.png', bbox_inches='tight', dpi=300)


def convert_tic_to_gaia(tic_ids, gaiadr3):
    """
    Convert a list of TIC IDs to Gaia DR3 designations.

    Parameters:
    tic_ids (np.array): Array of TIC IDs.

    Returns:
    tuple: (astropy.table.Table, list) containing TIC IDs and corresponding Gaia DR3 designations.
    """
    gaia_results = []
    # gaia_designations = []

    for i, tic_id in tqdm(enumerate(tic_ids)):
        try:
            catalog_data = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
            gaia_id = gaiadr3[i]
            ra = catalog_data[0]["ra"]
            dec = catalog_data[0]["dec"]
            gaia_results.append((tic_id, gaia_id, ra, dec))
        except:
            continue

    # Convert results to an astropy Table
    table = Table(rows=gaia_results, names=('TIC', 'designation', 'ra', 'dec'))
    return table

def gaiadr3_table(gaiadr3, ra, dec):
    gaia_results = []
    # gaia_designations = []

    for i, gaia in tqdm(enumerate(gaiadr3)):
        try:
            gaia_id = gaia.split(' ')[-1]
            ra_ = ra[i]
            dec_ = dec[i]
            gaia_results.append((gaia_id, ra_, dec_))
        except:
            continue

    # Convert results to an astropy Table
    table = Table(rows=gaia_results, names=('designation', 'ra', 'dec'))
    return table


def get_sectors_for_star(i, table, max_retries=5, delay=10):
    coord = SkyCoord(ra=table['ra'][i], dec=table['dec'][i], unit=(u.degree, u.degree), frame='icrs')

    for attempt in range(max_retries):
        try:
            sector_table = Tesscut.get_sectors(coordinates=coord)
            return i, sector_table
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            return i, None

    print("Max retries exceeded. Could not get sectors for this coordinate.")
    return i, None


def process_star_results(result, table, file_names_odd, file_names_even):
    i, sector_table = result
    if sector_table is not None:
        for j in range(len(sector_table)):
            file_name = (f"hlsp_tglc_tess_ffi_gaiaid-{table['designation'][i]}-s{sector_table['sector'][j]:04d}-"
                         f"cam{sector_table['camera'][j]}-ccd{sector_table['ccd'][j]}_tess_v1_llc.fits")
            if sector_table['sector'][j] % 2 == 0:
                file_names_even.append(file_name)
            elif sector_table['sector'][j] % 2 == 1:
                file_names_odd.append(file_name)


def get_file_name(gaiadr3, ra, dec, max_retries=5, delay=10, dir='/Users/tehan/Documents/TGLC/Jeroen/'):
    table = gaiadr3_table(gaiadr3, ra, dec)
    table.write(f'{dir}Jeroen_new_tic_to_gaia.csv', format='csv', overwrite=True)
    table = Table.read(f'{dir}Jeroen_new_tic_to_gaia.csv', format='csv')
    file_names_odd = mp.Manager().list()
    file_names_even = mp.Manager().list()

    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(get_sectors_for_star, args=(i, table, max_retries, delay)) for i in range(len(table))]

    for result in tqdm(results):
        process_star_results(result.get(), table, file_names_odd, file_names_even)

    pool.close()
    pool.join()
    odd_table = Table([file_names_odd], names=('files',))
    odd_table.write(f'{dir}Jeroen_new_odd_files.csv', format='csv', overwrite=True)
    even_table = Table([file_names_even], names=('files',))
    even_table.write(f'{dir}Jeroen_new_even_files.csv', format='csv', overwrite=True)

    return list(file_names_odd), list(file_names_even)


if __name__ == '__main__':
    file_path = '/Users/tehan/Documents/TGLC/Jeroen/CEPS_Plachy_GaiaID.csv'
    table1 = Table.read(file_path)
    file_path = '/Users/tehan/Documents/TGLC/Jeroen/RRLS_Molnar_GaiaID.csv'
    table2 = Table.read(file_path)
    file_path = '/Users/tehan/Documents/TGLC/Jeroen/gaia_missing_tf.csv'
    table3 = Table.read(file_path)
    file_path = '/Users/tehan/Documents/TGLC/Jeroen/gaia_missing_A_M_hybrid.csv'
    table4 = Table.read(file_path)

    # tic_ids = table3['tic_id'].tolist() + table4['tic_id'].tolist()
    # gaiadr3 = table3['dr3_source_id'].tolist() + table4['dr3_source_id'].tolist()
    # print(len(tic_ids), len(gaiadr3))
    # file_names_odd, file_names_even = get_file_name(tic_ids, gaiadr3)
    gaiadr3 = table1['designation'].tolist() + table2['designation'].tolist()
    ra = table1['ra'].tolist() + table2['ra'].tolist()
    dec = table1['dec'].tolist() + table2['dec'].tolist()
    print(len(gaiadr3))
    file_names_odd, file_names_even = get_file_name(gaiadr3, ra, dec)


    # run cp_files next


    # file_path = '/home/tehan/data/cosmos/mallory/mdwarfs_s1.csv'
    # table = Table.read(file_path, format='csv', delimiter=',')
    # tic_ids = np.array(table['TICID'])
    # t, gaia = convert_tic_to_gaia(tic_ids)
    # t.write('/home/tehan/data/cosmos/mallory/mdwarfs_s1_gaia.csv', overwrite=True)
    # mdwarf_gaia = Table.read("/home/tehan/data/cosmos/mallory/mdwarfs_s1_gaia.csv", format="csv")
    # sector = 1
    # local_directory = f'/home/tehan/data/sector{sector:04d}/'
    # for i in range(16):
    #     name = f'{1 + i // 4}-{1 + i % 4}'
    #     lc_per_ccd(camccd=name, local_directory=local_directory, target_list=mdwarf_gaia['designation'].tolist())

    # For Oddo
    # file_path = '/home/tehan/data/cosmos/oddo/small_dataframe_for_Te.csv'
    # table = Table.read(file_path, format='csv', delimiter=',')
    # result_dict = {}
    # for row in tqdm(table, desc="Processing data"):
    #     designation = row['GAIADR3']
    #     sectors = row['sectors'].split(',')
    #     for sector in sectors:
    #         if sector not in result_dict:
    #             result_dict[sector] = []
    #         result_dict[sector].append(int(designation))
    # sorted_keys = sorted(result_dict.keys())
    # print(sorted_keys)
    # for sector in trange(2, 56, 2):
    #     target_list = result_dict[str(sector)]
    #     print("Number of cpu : ", multiprocessing.cpu_count())
    #     sector = sector
    #     local_directory = f'/home/tehan/data/sector{sector:04d}/'
    #     for i in range(16):
    #         name = f'{1 + i // 4}-{1 + i % 4}'
    #         lc_per_ccd(camccd=name, local_directory=local_directory, target_list=target_list)

    # For TESSminer
    # file_path = '/home/tehan/data/cosmos/GEMS/ListofMdwarfTICs_crossmatch_missing.csv'
    # table = Table.read(file_path, format='csv', delimiter=',')
    # result_dict = {}
    # for row in tqdm(table, desc="Processing data"):
    #     designation = row['designation']
    #     sectors = row['sectors'].split()
    #     for sector in sectors:
    #         if sector not in result_dict:
    #             result_dict[sector] = []
    #         result_dict[sector].append(designation)
    # sorted_keys = sorted(result_dict.keys())
    # print(sorted_keys)
    # for sector in trange(2, 56, 2):
    #     target_list = result_dict[str(sector)]
    #     print("Number of cpu : ", multiprocessing.cpu_count())
    #     sector = sector
    #     local_directory = f'/home/tehan/data/sector{sector:04d}/'
    #     for i in range(16):
    #         name = f'{1 + i // 4}-{1 + i % 4}'
    #         lc_per_ccd(camccd=name, local_directory=local_directory, target_list=target_list)

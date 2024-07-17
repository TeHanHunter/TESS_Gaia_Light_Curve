# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4

import os

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

def convert_tic_to_gaia(tic_ids):
    """
    Convert a list of TIC IDs to Gaia DR3 designations.

    Parameters:
    tic_ids (np.array): Array of TIC IDs.

    Returns:
    tuple: (astropy.table.Table, list) containing TIC IDs and corresponding Gaia DR3 designations.
    """
    gaia_results = []
    gaia_designations = []

    for tic_id in tqdm(tic_ids):
        catalog_data = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
        if len(catalog_data) > 0:
            gaia_id = catalog_data[0]["GAIA"]
            gaia_results.append((tic_id, gaia_id))
            gaia_designations.append(gaia_id)
        else:
            gaia_results.append((tic_id, None))
            gaia_designations.append(None)

    # Convert results to an astropy Table
    table = Table(rows=gaia_results, names=('TIC', 'designation'))
    return table, gaia_designations

if __name__ == '__main__':
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

    # For TESSminer
    file_path = '/home/tehan/data/cosmos/oddo/small_dataframe_for_Te.csv'
    table = Table.read(file_path, format='csv', delimiter=',')
    result_dict = {}
    for row in tqdm(table, desc="Processing data"):
        designation = row['GAIADR3']
        sectors = row['sectors'].split(',')
        for sector in sectors:
            if sector not in result_dict:
                result_dict[sector] = []
            result_dict[sector].append(int(designation))
    sorted_keys = sorted(result_dict.keys())
    print(sorted_keys)
    for sector in trange(1, 56, 2):
        target_list = result_dict[str(sector)]
        print("Number of cpu : ", multiprocessing.cpu_count())
        sector = sector
        local_directory = f'/home/tehan/data/sector{sector:04d}/'
        for i in range(16):
            name = f'{1 + i // 4}-{1 + i % 4}'
            lc_per_ccd(camccd=name, local_directory=local_directory, target_list=target_list)
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
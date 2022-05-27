import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
import pickle
from os.path import exists
from TGLC.target_lightcurve import *
from TGLC.ffi_cut import *
from astropy.io import ascii
import warnings

warnings.simplefilter('always', UserWarning)

if __name__ == '__main__':
    target = 'NGC 7654'  # 368287008 188589164 458419328 126982221
    local_directory = '/mnt/c/users/tehan/desktop/NGC 7654/'
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    size = 90  # int, suggests bigger cuts
    source = ffi(target=target, size=size, local_directory=local_directory, sector=17)
    # 232.3333333 27.17638888, 239.75 35.41667, 131.375 63, 24.25 16.2
    source.select_sector(sector=17)  # 24 25 47 42
    # catalogdata = Catalogs.query_object('TIC ' + str(target), radius=0.02, catalog="TIC")
    # name = 'Gaia DR2 ' + str(np.array(catalogdata['GAIA'])[np.where(catalogdata['ID'] == str(target))[0][0]])
    # print(name)
    epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory)


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
    #     source = ffi(target='TIC ' + str(target), size=size, local_directory=local_directory)
    #     for j in range(len(source.sector_table)):
    #         try:
    #             source.select_sector(sector=source.sector_table['sector'][j])
    #             epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
    #                  name=name)
    #         except:
    #             warnings.warn(f'Skipping sector {source.sector_table["sector"][j]}. (Target not in cut)')
    # np.savetxt('/mnt/d/Astro/hpf/hpf_gaia_dr2.txt', gaia_name)

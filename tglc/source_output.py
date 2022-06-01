import os
from tglc.ffi import *
import multiprocessing
from multiprocessing import Pool
from functools import partial


def ffi_to_source(sector=1, ccd=1, local_directory=''):
    '''
    Cut calibrated FFI to source.pkl
    :param sector: int, required
    TESS sector number
    :param local_directory: string, required
    output directory
    '''
    # local_directory = f'/mnt/d/TESS_Sector_17/'
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'ffi/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)

    with Pool() as p:
      p.map(partial(cut_ffi, sector=sector, size=150, path=local_directory, ccd=ccd), [1, 2, 3, 4])


if __name__ == '__main__':
    sector = 1
    for i in range(4):
        ffi_to_source(sector=sector, ccd=i, local_directory=f'/home/tehan/data/sector{sector:04d}/')

import os

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
from tglc.ffi import *
from multiprocessing import Pool
from functools import partial
import logging
import warnings
import astroquery

logging.getLogger(astroquery.__name__).setLevel(logging.ERROR)
warnings.simplefilter('ignore', UserWarning)


def median_mask(sector_num=10):
    mask = np.ones((sector_num, 16, 2048))
    for i in range(sector_num):
        for j in range(16):
            mask[i, j] = np.load(
                f'/home/tehan/data/sector{i + 1:04d}/mask/mask_sector{i + 1:04d}_cam{1 + j // 4}_ccd{1 + j % 4}.npy')
    med_mask = np.median(mask, axis=0)
    return med_mask


def cut_ffi_(i, sector=1, size=150, local_directory=''):
    ffi(camera=1 + i // 4, ccd=1 + i % 4, sector=sector, size=size, local_directory=local_directory,
        producing_mask=True)


def ffi_to_source(sector=1, local_directory=''):
    """
    Cut calibrated FFI to source.pkl
    :param sector: int, required
    TESS sector number
    :param local_directory: string, required
    output directory
    """
    os.makedirs(f'{local_directory}lc/', exist_ok=True)
    os.makedirs(f'{local_directory}epsf/', exist_ok=True)
    os.makedirs(f'{local_directory}ffi/', exist_ok=True)
    os.makedirs(f'{local_directory}source/', exist_ok=True)
    os.makedirs(f'{local_directory}log/', exist_ok=True)
    os.makedirs(f'{local_directory}mask/', exist_ok=True)

    with Pool(4) as p:
        p.map(partial(cut_ffi_, sector=sector, size=150, local_directory=local_directory), range(16))

    # for i in range(16):
    #     ffi(camera=1 + i // 4, ccd=1 + i % 4, sector=sector, size=150, local_directory=local_directory)


if __name__ == '__main__':
    sector = 1
    ffi_to_source(sector=sector, local_directory=f'/home/tehan/data/sector{sector:04d}/')

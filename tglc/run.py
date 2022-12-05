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

from astropy.io import fits
from tglc.target_lightcurve import epsf
from multiprocessing import Pool
from functools import partial
from glob import glob


def lc_per_cut(i, camccd='', local_directory=''):
    cut_x = i // 14
    cut_y = i % 14
    with open(f'{local_directory}source/{camccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
        source = pickle.load(input_)
    epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, sector=source.sector, power=1.4,
         local_directory=local_directory, limit_mag=16, save_aper=False, no_progress_bar=True)


def lc_per_ccd(camccd='1-1', local_directory=''):
    os.makedirs(f'{local_directory}epsf/{camccd}/', exist_ok=True)
    with Pool() as p:
        p.map(partial(lc_per_cut, camccd=camccd, local_directory=local_directory), range(196))


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


if __name__ == '__main__':
    print("Number of cpu : ", multiprocessing.cpu_count())
    sector = 1
    local_directory = f'/home/tehan/data/sector{sector:04d}/'
    for i in range(16):
        name = f'{1 + i // 4}-{1 + i % 4}'
        lc_per_ccd(camccd=name, local_directory=local_directory)
        plot_epsf(sector=sector, camccd=name, local_directory=local_directory)

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
from tglc.target_lightcurve import *
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from functools import partial


def lc_per_cut(i, ccd='1-1', local_directory=''):
    cut_x = i // 14
    cut_y = i % 14
    with open(local_directory + f'source/{ccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
        source = pickle.load(input_)
    epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, ccd=ccd, sector=source.sector,
         local_directory=local_directory, limit_mag=16, save_aper=False, no_progress_bar=True)  # TODO: power?


def lc_per_ccd(ccd='1-1', local_directory=''):
    os.makedirs(local_directory + f'epsf/{ccd}/', exist_ok=True)
    with Pool() as p:
        p.map(partial(lc_per_cut, ccd=ccd, local_directory=local_directory), range(196))


def plot_epsf(sector=1, ccd='', local_directory=''):
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))
    gs = fig.add_gridspec(14, 14)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(196):
        cut_x = i // 14
        cut_y = i % 14
        psf = np.load(local_directory + f'epsf/{ccd}/epsf_{cut_x:02d}_{cut_y:02d}_sector_{sector}.npy')
        cmap = 'bone'
        if np.isnan(psf).any():
            cmap = 'inferno'
        ax = fig.add_subplot(gs[13 - cut_y, cut_x])
        ax.imshow(psf[0, :23 ** 2].reshape(23, 23), cmap=cmap, origin='lower')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', left=False)
    plt.savefig(local_directory + f'epsf/{ccd}/epsf_sector_{sector}_{ccd}.png', dpi=300)


if __name__ == '__main__':
    print("Number of cpu : ", multiprocessing.cpu_count())
    sector = 1
    local_directory = f'/home/tehan/data/sector{sector:04d}/'
    names = ['1-1', '1-2', '1-3', '1-4']  # , '2-1', '2-2', '2-3', '2-4',
    # '3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3', '4-4']
    for name in names:
        print(name)
        lc_per_ccd(ccd=name, local_directory=local_directory)
        plot_epsf(sector=sector, ccd=name, local_directory=local_directory)

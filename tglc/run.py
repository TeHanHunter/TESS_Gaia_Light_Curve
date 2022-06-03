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
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
from functools import partial

def lc_per_cut(i, ccd='1-1', local_directory=''):
    cut_x = i // 14
    cut_y = i % 14
    with open(local_directory + f'source/{ccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
        source = pickle.load(input_)
    print(ccd)
    epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, ccd=ccd, sector=source.sector,
         local_directory=local_directory, limit_mag=16)  # TODO: power?

def lc_per_ccd(sector = 1, ccd = '1-1'):
    local_directory = f'/home/tehan/data/sector{sector:04d}/'
    os.makedirs(local_directory + f'epsf/{ccd}/', exist_ok=True)
    with Pool() as p:
        p.map(partial(lc_per_cut, ccd=ccd, local_directory=local_directory), range(484))
 
#     for i in range(484):
#         lc_per_cut(i, ccd=ccd, local_directory=local_directory)
        
if __name__ == '__main__':
    # lc_per_ccd(sector=1, ccd='1-1')
    print("Number of cpu : ", multiprocessing.cpu_count())
    sector = 1
    names = ['1-1', '1-2', '1-3', '1-4'] #, '2-1', '2-2', '2-3', '2-4',
             #'3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3', '4-4']
    for name in names:
        lc_per_ccd(sector=sector, ccd = name)

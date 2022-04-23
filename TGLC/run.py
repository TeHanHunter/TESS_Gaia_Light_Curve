# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4
import pickle
from TGLC.target_lightcurve import *

if __name__ == '__main__':
    sector = 17
    ccd = '1-1'
    # local_directory = f'/home/tehan/data/sector{sector:02d}/'
    local_directory = f'/mnt/d/TESS_Sector_17/'
    os.makedirs(local_directory + f'epsf/{ccd}/', exist_ok=True)
    cut_x = 0
    cut_y = 0
    with open(local_directory + f'source/{ccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
        source = pickle.load(input_)
    epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, ccd=ccd, sector=source.sector,
         local_directory=local_directory, limit_mag=18)  # TODO: power?
    # for i in range(484):
    #     target = f'{(i // 14):02d}_{(i % 14):02d}'
    #     with open(local_directory + f'source/{ccd}/source_{target}.pkl', 'rb') as input_:
    #         source = pickle.load(input_)
    #     epsf(source, factor=2, target=target, ccd=ccd, sector=source.sector, local_directory=local_directory)  # TODO: power?

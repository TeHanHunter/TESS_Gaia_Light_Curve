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
    with open(local_directory + f'source/{ccd}/source_11_11.pkl', 'rb') as input_:
        source = pickle.load(input_)
    epsf(source, factor=2, target='11_11', ccd=ccd, sector=source.sector,
         local_directory=local_directory)  # TODO: power?
    # for i in range(484):
    #     target = f'{(i // 22):02d}_{(i % 22):02d}'
    #     with open(local_directory + f'source/{ccd}/source_{target}.pkl', 'rb') as input_:
    #         source = pickle.load(input_)
    #     epsf(source, factor=2, target=target, ccd=ccd, sector=source.sector, local_directory=local_directory)  # TODO: power?

# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4
import pickle
from TGLC.target_lightcurve import *

if __name__ == '__main__':
    ccd = '1-1'
    sector = 17
    local_directory = f'/home/tehan/data/sector{sector}/lc/'
    os.makedirs(local_directory, exist_ok=True)
    for i in range(484):
        target = f'{(i // 4):02d}_{(i % 4):02d}'
        with open(f'/home/tehan/data/sector{sector}/{ccd}/source_{target}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        epsf(source, factor=2, target=target, sector=source.sector, local_directory=local_directory)  # TODO: power?

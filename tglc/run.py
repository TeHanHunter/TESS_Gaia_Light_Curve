# export OPENBLAS_NUM_THREADS=1
# https://dev.to/kapilgorve/set-environment-variable-in-windows-and-wsl-linux-in-terminal-3mg4
from tglc.target_lightcurve import *
import multiprocessing
from multiprocessing import Process

def lc_per_ccd(sector = 1, ccd = '1-1'):
    local_directory = f'/home/tehan/data/sector{sector:04d}/'
    os.makedirs(local_directory + f'epsf/{ccd}/', exist_ok=True)
    for i in range(484):
        cut_x = i // 14
        cut_y = i % 14
        with open(local_directory + f'source/{ccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        epsf(source, psf_size=11, factor=2, cut_x=cut_x, cut_y=cut_y, ccd=ccd, sector=source.sector,
             local_directory=local_directory, limit_mag=16)  # TODO: power?


if __name__ == '__main__':
    # lc_per_ccd(sector=1, ccd='1-1')
    print("Number of cpu : ", multiprocessing.cpu_count())
    sector = 1
    names = ['1-1', '1-2', '1-3', '1-4', '2-1', '2-2', '2-3', '2-4',
             '3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3', '4-4']
    procs = []
    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=lc_per_ccd, args=([sector] * 16, name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

import os
from tglc.ffi import *
import multiprocessing
from multiprocessing import Process

def ffi_to_source(sector=1, local_directory=''):
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

    procs = []
    # instantiating process with arguments
    for i in range(16):
        # print(name)
        proc = Process(target=cut_ffi, args=(sector, 1 + i // 4, 1 + i % 4, local_directory))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()



if __name__ == '__main__':
    sector = 1
    ffi_to_source(sector=sector, local_directory=f'/home/tehan/data/sector{sector:04d}/')

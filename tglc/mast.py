import ftplib
import getpass
from glob import glob
import numpy as np
from tqdm import trange
from multiprocessing import Pool
from functools import partial
from astropy.io import fits
import shutil
import time
import os


def zip_folder(i, sector=1, do_zip=True):
    time.sleep(i)
    cam = 1 + i // 4
    ccd = 1 + i % 4
    zip_file = f'/home/tehan/data/mast/sector{sector:04d}/sector_{sector}_cam_{cam}_ccd_{ccd}'
    original_file = f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/'
    if do_zip:
        shutil.make_archive(zip_file, 'zip', original_file)
        return
    else:
        ftps = ftplib.FTP_TLS('archive.stsci.edu')
        ftps.login('tehanhunter@gmail.com', getpass.getpass())
        ftps.prot_p()
        ftps.cwd('pub/hlsp/tglc/')
        print(f"Sector {sector}")
        sector_dir = f"s{sector:04d}"
        # print current directory
        dir_list = []
        ftps.retrlines('LIST', dir_list.append)
        dir_list = [d.split()[-1] for d in dir_list]
        # check if sector_dir already exists
        if sector_dir in dir_list:
            pass
            # print(f"Directory {sector_dir}/ already exists.")
        # if not, mkdir new sector directory (use relative path, NOT absolute path)
        else:
            print(ftps.mkd(sector_dir))
        # cd into sector directory (use relative path, NOT absolute path)
        ftps.cwd(sector_dir)
        # print('\n')
        with open(f'{zip_file}.zip', 'rb') as f:
            ftps.storbinary(f"STOR sector_{sector}_cam_{cam}_ccd_{ccd}.zip", f)


def hlsp_transfer(sector=1, do_zip=True):
    with Pool(16) as p:
        p.map(partial(zip_folder, sector=sector, do_zip=do_zip), range(16))


def star_finder(i, sector=1, starlist='/home/tehan/data/ben/test_list_sector01.txt'):
    stars = np.loadtxt(starlist, dtype=int)
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/hlsp_*.fits')
    for j in trange(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            try:
                if int(hdul[0].header['TICID']) in stars:
                    hdul.writeto(f"/home/tehan/data/ben/sector0001/{files[j].split('/')[-1]}", overwrite=True)
            except:
                pass


if __name__ == '__main__':
    sector = 1
    hlsp_transfer(sector=sector, do_zip=True)
    hlsp_transfer(sector=sector, do_zip=False)


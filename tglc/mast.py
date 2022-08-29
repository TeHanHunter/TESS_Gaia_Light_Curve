import ftplib
import getpass
from glob import glob
import numpy as np
from tqdm import trange
from multiprocessing import Pool
from functools import partial
from astropy.io import fits
import shutil
import os


def zip_folder(i, sector=1):
    cam = 1 + i // 4
    ccd = 1 + i % 4
    shutil.make_archive(f'sector_{sector}_cam_{cam}_ccd_{ccd}', 'zip',
                        f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/',
                        '')

def hlsp_transfer(i, sector=1):
    cam = 1 + i // 4
    ccd = 1 + i % 4
    ftps = ftplib.FTP_TLS('archive.stsci.edu')
    ftps.login('tehanhunter@gmail.com', getpass.getpass())
    ftps.prot_p()
    ftps.cwd('pub/hlsp/tglc/')
    print(f"Sector {sector}")
    sector_dir = f"s{sector:04d}"
    # print current directory
    # print(ftps.pwd())
    # get list of existing directories
    dir_list = []
    ftps.retrlines('LIST', dir_list.append)
    dir_list = [d.split()[-1] for d in dir_list]
    # check if sector_dir already exists
    # if sector_dir in dir_list:
    #     print(f"Directory {sector_dir}/ already exists.")
    # # if not, mkdir new sector directory (use relative path, NOT absolute path)
    # else:
    #     print(ftps.mkd(sector_dir))
    # cd into sector directory (use relative path, NOT absolute path)
    ftps.cwd(sector_dir)
    # print('\n')

    cam_ccd_dir = f'cam{cam}-ccd{ccd}'
    # get list of existing cam-ccd directories
    subdir_list = []
    ftps.retrlines('LIST', subdir_list.append)
    subdir_list = [d.split()[-1] for d in subdir_list]
    # check if cam_ccd_dir already exists
    if cam_ccd_dir in subdir_list:
        print(f"Subdirectory {cam_ccd_dir} already exists.")
    # if not, mkdir new cam-ccd subdirectory (use relative path, NOT absolute path)
    else:
        ftps.mkd(cam_ccd_dir)
    print(f"Writing files to {cam_ccd_dir}:")
    # cd into new cam-ccd subdirectory
    ftps.cwd(cam_ccd_dir)
    # code goes here to write files to archive.stsci.edu:/pub/hlsp/tglc/<sector>/<cam-ccd>/
    # below is just an example, use whatever working code you have
    file_path = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/hlsp_*.fits')
    for i in trange(len(file_path)):
        file = file_path[i]
        with open(file, 'rb') as f:
            ftps.storbinary(f"STOR {file.split('/')[-1]}", f, 102400)

    # check completeness


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
    with Pool(16) as p:
        p.map(partial(zip_folder, sector=sector), range(16))

    # with Pool(16) as p:
    #     p.map(partial(star_finder, sector=sector), range(16))
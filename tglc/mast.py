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
from astropy.io import ascii
import zipfile


def filter_no_tic_(i, sector=1):
    time.sleep(i)
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/*.fits')
    for j in range(len(files)):
        hdul = fits.open(files[j])
        if hdul[0].header['TICID'] == '':
            shutil.move(files[j], f'/home/tehan/data/sector{sector:04d}/extra_lc/{os.path.basename(files[j])}')


def filter_no_tic(sector=1):
    os.makedirs(f'/home/tehan/data/sector{sector:04d}/extra_lc/', exist_ok=True)
    with Pool(16) as p:
        p.map(partial(filter_no_tic_, sector=sector), range(16))


def zip_folder(i, sector=1, do_zip=True, lc_num_per_zip=1e6):
    time.sleep(i)
    cam = 1 + i // 4
    ccd = 1 + i % 4
    zip_file = f'/home/tehan/data/mast/sector{sector:04d}/sector_{sector}_cam_{cam}_ccd_{ccd}'
    original_file = f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/'
    files = glob(f'{original_file}*.fits')
    if do_zip:
        # num_zips = int(len(files) // lc_num_per_zip + 1)
        # for i in range(num_zips):
            # with zipfile.ZipFile(f'{zip_file}_{i:02d}.zip', 'w') as zipMe:
            #     for file in files[int(i * lc_num_per_zip):int((i + 1) * lc_num_per_zip)]:
            #         zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
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


def search_stars(i, sector=1, star_list=None):
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/hlsp_*.fits')
    for j in trange(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            try:
                if int(hdul[0].header['TICID']) in star_list:
                    hdul.writeto(f"/home/tehan/data/cosmos/dominic_EB/sector{sector:04d}/{files[j].split('/')[-1]}",
                                 overwrite=True)
            except ValueError:
                pass


def star_spliter(server=1,  # or 2
                 star_list='/home/tehan/data/cosmos/dominic_EB/eb_cat.txt'):
    prsa_ebs = ascii.read(star_list)['ID'].data
    # sector_list = tuple([] for _ in range(55))  ##1 extended mission
    # for j in range(len(prsa_ebs)):
    #     try:
    #         sectors = prsa_ebs['sectors'][j].split(',')
    #         for k in range(len(sectors)):
    #             sector_list[int(sectors[k]) - 1].append(prsa_ebs['tess_id'][j])
    #     except AttributeError:
    #         pass
    for i in range(server, 27, 2):
        os.makedirs(f'/home/tehan/data/cosmos/dominic_EB/sector{i:04d}/', exist_ok=True)
        with Pool(16) as p:
            p.map(partial(search_stars, sector=i, star_list=prsa_ebs), range(16))
    return


if __name__ == '__main__':
    sector = 1
    filter_no_tic(sector=sector)
    hlsp_transfer(sector=sector, do_zip=True)
    # hlsp_transfer(sector=sector, do_zip=False)
    # star_spliter(server=1)
    # star_list='/home/tehan/Documents/tglc/dominic_EB/eb_cat.txt'

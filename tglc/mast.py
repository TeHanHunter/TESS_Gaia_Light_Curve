import ftplib
import getpass
from glob import glob
import numpy as np
from tqdm import trange
from multiprocessing import Pool
from functools import partial
from astropy.io import fits


# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive


def star_finder(i, sector=1, starlist='/home/tehan/data/mdwarfs/sector_1_mdwarfs_Tmag_18_TICID.csv'):
    stars = np.loadtxt(starlist, dtype=int)
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}_{ccd}/*.fits')
    for j in trange(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            if int(hdul[0].header['TICID']) in stars:
                hdul.writeto(f"/home/tehan/data/mdwarfs/{files[j].split('/')[-1]}", overwrite=True)


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


def google_drive():
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    upload_file_list = ['1.jpg', '2.jpg']
    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload()  # Upload the file.


if __name__ == '__main__':
    sector = 1
    star_finder(0, sector=1)

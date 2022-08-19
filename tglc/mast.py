import ftplib
import getpass
from glob import glob
from tqdm import trange
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

def hlsp_transfer(sector=1, camera=1, ccd=1):
    ftps = ftplib.FTP_TLS('archive.stsci.edu')
    ftps.login('tehanhunter@gmail.com', getpass.getpass())
    ftps.prot_p()
    ftps.mkd(f'pub/hlsp/tglc/sector{sector:04d}/cam_{camera}_ccd_{ccd}/')
    ftps.cwd(f'pub/hlsp/tglc/sector{sector:04d}/cam_{camera}_ccd_{ccd}/')
    file_path = glob(f'/home/tehan/data/sector{sector:04d}/lc/{camera}-{ccd}/hlsp_*.fits')

    for i in trange(len(file_path)):
        file = file_path[i]
        with open(file, 'rb') as f:
            ftps.storbinary(f"STOR {file.split('/')[-1]}", f)

def google_drive():
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    upload_file_list = ['1.jpg', '2.jpg']
    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload()  # Upload the file.

if __name__=='__main__':
    sector = 3
    for i in range(16):
        hlsp_transfer(sector=sector, camera=1 + i // 4, ccd=1 + i % 4)

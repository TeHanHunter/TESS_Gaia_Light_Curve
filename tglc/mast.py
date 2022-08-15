import ftplib
import getpass
from glob import glob

ftps = ftplib.FTP_TLS('archive.stsci.edu')
ftps.login('tehanhunter@gmail.com', getpass.getpass())
ftps.prot_p()
ftps.cwd('pub/hlsp/tglc/sector0002/1_1/') 

file_path = glob('/home/tehan/data/sector0002/lc/1-1/hlsp_*.fits')
print(file_path)

for file in file_path:
    with open(file, 'rb') as f:
        ftps.storbinary(f'STOR {file}', f)
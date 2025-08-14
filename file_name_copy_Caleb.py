import pandas as pd
import os
import shutil
import tarfile

# Paths
csv_file = '/home/tehan/data/cosmos/missing_tics_081425.csv'
dest_folder = '/home/tehan/data/cosmos/GEMS_missing_081425/'
server = 1
# Make destination folder if not exists
os.makedirs(dest_folder, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

for idx, row in df.iterrows():
    designation = str(row['designation']).split('.')[0]  # int part of designation
    sector = int(row['sector'])  # make sure sector is integer
    camera = int(row['camera'])
    ccd = int(row['ccd'])
    if sector % 2 == server:
        # Original file path
        src_path = f'/home/tehan/data/sector{sector:04d}/lc/{camera}-{ccd}/hlsp_tglc_tess_ffi_gaiaid-{designation}-s{sector:04d}-cam{camera}-ccd{ccd}_tess_v1_llc.fits'

        # Destination path
        dest_path = os.path.join(dest_folder, os.path.basename(src_path))

        # Copy file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f'Copied: {src_path} -> {dest_path}')
        else:
            print(f'File not found: {src_path}')

# Tar the folder
tar_path = dest_folder.rstrip('/') + f'_{server}.tar.gz'
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(dest_folder, arcname=os.path.basename(dest_folder))

print(f'Tar archive created: {tar_path}')
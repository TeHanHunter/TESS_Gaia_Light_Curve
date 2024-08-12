import pandas as pd
import shutil
import os
import time

# Read the CSV file
csv_file = '/home/tehan/data/cosmos/Jeroen/odd_files.csv'  # replace with your CSV file path
df = pd.read_csv(csv_file)  # Assuming the CSV does not have a header
file_names = df['files']  # Assuming the file names are in the first column
print(file_names[0])
# Base directory
base_dir = '/home/tehan/data'

# Function to construct full path based on file name
def construct_full_path(file_name):
    parts = file_name.split('-')
    sector = parts[2][1:5]  # Extract the sector number
    cam_ccd = parts[3][3] + '-' + parts[4][3]
    return os.path.join(base_dir, f'sector{sector}', 'lc', cam_ccd, file_name)

# Destination folder
destination_folder = '/home/tehan/data/cosmos/Jeroen/lc/'  # replace with your destination folder path

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Copy each file to the destination folder
for file_name in file_names:
    try:
        full_path = construct_full_path(file_name)
    except:
        print(file_name)
        continue
    if os.path.isfile(full_path):
        shutil.copy(full_path, destination_folder)
        print(f'Copied {full_path} to {destination_folder}')
    else:
        print(f'File not found: {full_path}')
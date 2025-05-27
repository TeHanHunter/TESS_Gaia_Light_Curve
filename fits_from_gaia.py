import csv
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === CONFIGURATION ===
root_dir = '/home/tehan/data'            # Your root directory containing sector folders
csv_path = '/home/tehan/data/cosmos/mharris/total_gaia_ids_and_ticids.csv'
destination_dir = '/home/tehan/data/cosmos/mharris/lcs'  # Folder to copy found files
max_workers = 32

os.makedirs(destination_dir, exist_ok=True)  # Create destination folder if it doesn't exist

# === Load Gaia IDs from CSV ===
gaia_ids = set()
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip header if exists
    for row in reader:
        if len(row) >= 2:
            gaia_ids.add(row[1].strip())

print(f"Loaded {len(gaia_ids)} Gaia IDs.")

# === Match & Copy Function ===
def match_and_copy(fpath):
    fname = os.path.basename(fpath)
    for gaia in gaia_ids:
        if gaia in fname:
            try:
                dest_path = os.path.join(destination_dir, fname)
                shutil.copy2(fpath, dest_path)
                return dest_path
            except Exception as e:
                print(f"Error copying {fpath}: {e}")
            break
    return None

for i in range(2,56,2):
    # === Glob all .fits files once ===
    print(f"Indexing all .fits files of sector {i}... (this might take a while)")
    all_fits_files = glob.glob(os.path.join(root_dir, f'sector{i:04d}/lc/*/*.fits'), recursive=True)
    print(f"Indexed {len(all_fits_files)} .fits files.")

    # === Parallel Processing ===
    matched_files = []
    with ThreadPoolExecutor(max_workers=1024) as executor:
        for result in tqdm(executor.map(match_and_copy, all_fits_files), total=len(all_fits_files),
                           desc="Matching & Copying", unit="file"):
            if result:
                matched_files.append(result)

    print(f"Copied {len(matched_files)} FITS files to {destination_dir}.")

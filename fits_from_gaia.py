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
    header = next(reader)  # Skip header if exists, comment if no header
    for row in reader:
        if len(row) >= 2:
            gaia_ids.add(row[1].strip())

print(f"Loaded {len(gaia_ids)} Gaia IDs.")

# === Define search + copy function ===
def search_and_copy(gaia_id):
    pattern = os.path.join(root_dir, f'sector00*/lc/*{gaia_id}*.fits')
    found = glob.glob(pattern)
    copied = []
    for fpath in found:
        try:
            fname = os.path.basename(fpath)
            dest_path = os.path.join(destination_dir, fname)
            shutil.copy2(fpath, dest_path)
            copied.append(dest_path)
        except Exception as e:
            print(f"Error copying {fpath}: {e}")
    return copied

# === Run searches + copy in parallel with progress bar ===
all_copied_files = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(search_and_copy, gaia): gaia for gaia in gaia_ids}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Searching & Copying FITS"):
        result = future.result()
        if result:
            all_copied_files.extend(result)

print(f"Copied {len(all_copied_files)} FITS files to {destination_dir}.")

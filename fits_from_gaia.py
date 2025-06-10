import csv
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# === CONFIGURATION ===
root_dir = '/home/tehan/data'
csv_base_path = '/home/tehan/data/cosmos/mharris/sector_csvs'  # Folder with one CSV per sector
destination_dir = '/home/tehan/data/cosmos/mharris/lcs'
max_workers = 1024

os.makedirs(destination_dir, exist_ok=True)

# === Match & Copy Function (to be set per-sector) ===
def get_match_and_copy(gaia_ids, dest_dir):
    def match_and_copy(fpath):
        fname = os.path.basename(fpath)
        for gaia in gaia_ids:
            if gaia in fname:
                try:
                    dest_path = os.path.join(dest_dir, fname)
                    shutil.copy2(fpath, dest_path)
                    return dest_path
                except Exception as e:
                    print(f"Error copying {fpath}: {e}")
                break
        return None
    return match_and_copy

# === Sector Loop ===
for i in range(1, 56, 2):
    csv_path = os.path.join(csv_base_path, f'*sector*s{i:02d}.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found for sector {i}: {csv_path}")
        continue

    # === Load Gaia IDs from sector CSV ===
    gaia_ids = set()
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                gaia_raw = row['Gaia']
                gaia_str = str(int(float(gaia_raw)))  # Convert from sci notation safely
                gaia_ids.add(gaia_str)
            except Exception as e:
                print(f"Error parsing Gaia ID '{row}': {e}")

    print(f"Sector {i:04d}: Loaded {len(gaia_ids)} Gaia IDs.")

    # === Glob all .fits files for this sector ===
    print(f"Indexing all .fits files of sector {i}... (this might take a while)")
    all_fits_files = glob.glob(os.path.join(root_dir, f'sector{i:04d}/lc/*/*.fits'), recursive=True)
    print(f"Indexed {len(all_fits_files)} .fits files.")

    match_and_copy = get_match_and_copy(gaia_ids, destination_dir)

    # === Parallel Processing ===
    matched_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(match_and_copy, all_fits_files),
                           total=len(all_fits_files),
                           desc=f"Sector {i:04d} Matching & Copying", unit="file"):
            if result:
                matched_files.append(result)

    print(f"Sector {i:04d}: Copied {len(matched_files)} FITS files to {destination_dir}.")
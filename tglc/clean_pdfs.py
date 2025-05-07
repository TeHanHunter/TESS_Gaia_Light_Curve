import os
from astropy.table import Table

# Load the .dat file using Astropy
tbl = Table.read("/Users/tehan/Documents/TGLC/deviation_TGLC_2024_kepler.dat", format='ascii')

# Convert to dictionary for fast lookup
rhat_dict = {row['Star_sector']: row['rhat'] for row in tbl}

# Directory containing the PDFs
pdf_dir = "/Users/tehan/Downloads/kepler_plots/"

if __name__ == '__main__':
    # Loop through files
    for fname in os.listdir(pdf_dir):
        if fname.startswith("Plots_TIC_") and fname.endswith("_Fixed_ew.pdf"):
            parts = fname.split("_")
            try:
                tic = parts[2]
                sector = parts[5]
                key = f"TIC_{tic}_{sector}"

                # Check if key exists and rhat == 1.0
                if key not in rhat_dict or rhat_dict[key] != 1.0:
                    os.remove(os.path.join(pdf_dir, fname))
                    print(f"Deleted: {fname}")
            except Exception as e:
                print(f"Skipped: {fname} due to error: {e}")
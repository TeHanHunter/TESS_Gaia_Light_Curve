import pandas as pd
from pathlib import Path
import shutil

# ---- CONFIGURATION ----
csv_path = "/home/tehan/data/cosmos/Oddo_2025/Jul2025_new_EBs_tesspoint_output_filtered.csv"
destination = Path("/home/tehan/data/cosmos/Oddo_2025/lc/")
copy_group = "even"  # options: "odd", "even", "both"
# --- LOAD CSV ---
df = pd.read_csv(csv_path)

# --- Prepare list of missing rows ---
missing_rows = []

# --- COPY LOOP ---
for _, row in df.iterrows():
    outsec = int(row["outSec"])
    if copy_group == "odd" and outsec % 2 == 0:
        continue
    if copy_group == "even" and outsec % 2 != 0:
        continue

    gaia_id = row["GAIA DR3"].split()[-1]
    outcam = row["outCam"]
    outccd = row["outCcd"]

    src_path = Path(
        f"/home/tehan/data/sector{outsec:04d}/lc/{outcam}-{outccd}/"
        f"hlsp_tglc_tess_ffi_gaiaid-{gaia_id}-s{outsec:04d}-"
        f"cam{outcam}-ccd{outccd}_tess_v1_llc.fits"
    )

    if src_path.exists():
        print(f"Copying {src_path.name}")
        shutil.copy2(src_path, destination)
    else:
        print(f"Missing: {src_path}")
        missing_rows.append(row)

# --- SAVE MISSING FILES LOG ---
if missing_rows:
    df_missing = pd.DataFrame(missing_rows)
    df_missing.to_csv(f"/home/tehan/data/cosmos/Oddo_2025/missing_files.csv", index=False)
    print(f"\nSaved {len(missing_rows)} missing entries to missing_files.csv")
else:
    print("\nAll files were copied successfully.")
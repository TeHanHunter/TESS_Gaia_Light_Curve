"""Move selected sector 56 experimental cut products aside before retrying."""

import argparse
import json
import shutil
import time
from pathlib import Path

SECTOR = 56
DEFAULT_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")
DEFAULT_CUTS = ["02_06", "02_08", "05_03", "05_05", "12_12", "13_00", "13_01", "13_04", "13_05"]
KINDS = ["epsf", "epsf_scale", "overexposure_mask"]


def normalize_cut(cut):
    if "_" in cut:
        x_text, y_text = cut.split("_", 1)
        return f"{int(x_text):02d}_{int(y_text):02d}"
    cut = int(cut)
    return f"{cut // 14:02d}_{cut % 14:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cam", type=int, default=1)
    parser.add_argument("--ccd", type=int, default=1)
    parser.add_argument("--cuts", nargs="+", default=DEFAULT_CUTS)
    parser.add_argument("--backup-name", default="retry_backup_20260608")
    args = parser.parse_args()

    base = args.root / "epsf" / f"{args.cam}-{args.ccd}"
    backup = base / args.backup_name
    backup.mkdir(parents=True, exist_ok=True)

    moved = []
    missing = []
    for cut in [normalize_cut(cut) for cut in args.cuts]:
        for kind in KINDS:
            path = base / f"{kind}_{cut}_sector_{SECTOR}_{args.cam}-{args.ccd}_tmag10_unit.npy"
            if not path.exists():
                missing.append(path.name)
                continue
            destination = backup / path.name
            if destination.exists():
                destination = backup / f"{path.stem}.{int(time.time())}{path.suffix}"
            shutil.move(str(path), str(destination))
            moved.append({"from": path.name, "to": destination.name})

    print(json.dumps({"backup": str(backup), "moved": moved, "missing": missing}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

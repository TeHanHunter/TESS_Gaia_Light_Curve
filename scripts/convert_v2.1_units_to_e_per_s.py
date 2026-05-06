"""Convert v2.1 LC FITS flux columns/headers from e- per FFI cadence to e-/s.

The S56 cam4 v2.1 run inherited TWIRL/TICA pickles where source.flux was raw
counts integrated over 200*0.8*0.99 = 158.4 s on-chip. The CPU bg_mod treats
the hardcoded `bar = 15000 * 10^((m-10)/-2.5)` as e-/s, so the resulting LC
columns came out in e- per 158.4-s cadence instead of e-/s.

This script divides aperture_flux, aperture_flux_raw, LOC_BG, PSF_BG by the
exposure factor (default 158.4) in every v2.1 FITS under a target tree, and
adds a `UNIT_FIX` header card so the conversion is auditable. Idempotent: if
UNIT_FIX is already present, the file is skipped.

Usage:
    python convert_v2.1_units_to_e_per_s.py /pdo/users/tehan/tglc_v2.1_s56cam4/sector0056
"""

import argparse
import os
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool

from astropy.io import fits

EXPOSURE_S = 200 * 0.8 * 0.99  # 158.4 s for TICA FFIs
FIX_TAG = "UNIT_FIX"


def convert_one(fp, exposure_s):
    try:
        with fits.open(fp, mode="update") as h:
            hdr = h[1].header
            if FIX_TAG in hdr:
                return "skipped"
            data = h[1].data
            for col in ("aperture_flux", "aperture_flux_raw"):
                if col in data.names:
                    data[col][:] = data[col] / exposure_s
            for k in ("LOC_BG", "PSF_BG"):
                if k in hdr:
                    hdr[k] = float(hdr[k]) / exposure_s
            hdr[FIX_TAG] = (exposure_s, "[s] divisor used to convert e-/cadence -> e-/s")
            hdr["BUNIT"] = ("e-/s", "flux unit (post unit-fix)")
            h.flush()
        return "fixed"
    except Exception as e:
        return f"error: {e!r}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="directory tree containing v2.1 LC FITS")
    p.add_argument("--pattern", default="*v2.1*.fits")
    p.add_argument("--exposure", type=float, default=EXPOSURE_S)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    args = p.parse_args()

    files = []
    for d, _, _ in os.walk(args.root):
        files.extend(glob(os.path.join(d, args.pattern)))
    print(f"found {len(files)} files under {args.root}", flush=True)

    fn = partial(convert_one, exposure_s=args.exposure)
    counts = {"fixed": 0, "skipped": 0}
    errors = []
    with Pool(processes=args.processes) as pool:
        for i, status in enumerate(pool.imap_unordered(fn, files, chunksize=64)):
            if status in counts:
                counts[status] += 1
            else:
                errors.append(status)
            if (i + 1) % 50000 == 0:
                print(f"  progress: {i+1}/{len(files)}  fixed={counts['fixed']} skipped={counts['skipped']} errs={len(errors)}", flush=True)

    print(f"done: fixed={counts['fixed']} skipped={counts['skipped']} errors={len(errors)}", flush=True)
    for e in errors[:5]:
        print(f"  example error: {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

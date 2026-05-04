"""Run the v2.1 CPU LC pipeline for sector 56, camera 4 only.

The GPU production tree on pdogpu1 stores per-orbit Source pickles at
    {prod_root}/orbit-{orbit}/ffi/cam{cam}/ccd{ccd}/source/source_{cx}_{cy}.pkl
This script merges the two orbits per cut into a single Source covering the
whole sector, then calls tglc.target_lightcurve.epsf to produce v2.1 LC FITS.

Set TGLC_PROD_ROOT and TGLC_OUT_ROOT before invoking, e.g.

    export TGLC_PROD_ROOT=/pdo/users/tehan/tglc-gpu-production
    export TGLC_OUT_ROOT=/pdo/users/tehan/tglc_v2.1_s56cam4
    /sw/qlp-environment/.venv/bin/python scripts/run_s56_cam4_v2.1.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

from tglc.target_lightcurve import epsf

SECTOR = 56
ORBITS = (119, 120)
CAM = 4
CCDS = (1, 2, 3, 4)
N_CUTS = 196  # 14 x 14


def _normalize_schema(source):
    """The twirl GPU pipeline writes lowercase `designation` and `tic.gaia3`;
    the CPU LC pipeline expects `DESIGNATION` and `tic.dr3_source_id`."""
    if "designation" in source.gaia.colnames and "DESIGNATION" not in source.gaia.colnames:
        source.gaia.rename_column("designation", "DESIGNATION")
    if hasattr(source.tic, "colnames") and "gaia3" in source.tic.colnames and "dr3_source_id" not in source.tic.colnames:
        source.tic.rename_column("gaia3", "dr3_source_id")
    return source


def merge_orbits(pkl_paths):
    sources = []
    for p in pkl_paths:
        with open(p, "rb") as f:
            sources.append(pickle.load(f))
    base = sources[0]
    base.time = np.concatenate([s.time for s in sources])
    base.cadence = np.concatenate([s.cadence for s in sources])
    base.flux = np.concatenate([s.flux for s in sources], axis=0)
    base.quality = np.concatenate([s.quality for s in sources])
    if not hasattr(base, "transient"):
        base.transient = None
    return _normalize_schema(base)


def run_cut(i, ccd, prod_root, out_root):
    cut_x = i // 14
    cut_y = i % 14
    pkl_paths = [
        f"{prod_root}/orbit-{orbit}/ffi/cam{CAM}/ccd{ccd}/source/source_{cut_x}_{cut_y}.pkl"
        for orbit in ORBITS
    ]
    for p in pkl_paths:
        if not os.path.exists(p):
            print(f"[skip] missing {p}", flush=True)
            return
    source = merge_orbits(pkl_paths)
    local_dir = f"{out_root}/sector{SECTOR:04d}/"
    os.makedirs(f"{local_dir}lc/{CAM}-{ccd}/", exist_ok=True)
    os.makedirs(f"{local_dir}epsf/{CAM}-{ccd}/", exist_ok=True)
    epsf(
        source,
        psf_size=11,
        factor=2,
        cut_x=cut_x,
        cut_y=cut_y,
        sector=SECTOR,
        power=1.4,
        local_directory=local_dir,
        limit_mag=16,
        save_aper=False,
        no_progress_bar=True,
    )


def run_ccd(ccd, prod_root, out_root, n_cuts, processes):
    print(f"[ccd{ccd}] starting {n_cuts} cuts with {processes} workers", flush=True)
    fn = partial(run_cut, ccd=ccd, prod_root=prod_root, out_root=out_root)
    with Pool(processes=processes) as pool:
        pool.map(fn, range(n_cuts))
    print(f"[ccd{ccd}] done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prod-root", default=os.environ.get("TGLC_PROD_ROOT", "/pdo/users/tehan/tglc-gpu-production"))
    p.add_argument("--out-root", default=os.environ.get("TGLC_OUT_ROOT", "/pdo/users/tehan/tglc_v2.1_s56cam4"))
    p.add_argument("--ccds", type=int, nargs="+", default=list(CCDS))
    p.add_argument("--cuts", type=int, default=N_CUTS, help="number of cuts to run per ccd (default all 196)")
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--smoke", action="store_true", help="run a single cut (ccd1, cut 0) and exit")
    args = p.parse_args()

    if args.smoke:
        print("[smoke] running cam4 ccd1 cut 0_0", flush=True)
        run_cut(0, ccd=1, prod_root=args.prod_root, out_root=args.out_root)
        return

    for ccd in args.ccds:
        run_ccd(ccd, args.prod_root, args.out_root, args.cuts, args.processes)


if __name__ == "__main__":
    sys.exit(main())

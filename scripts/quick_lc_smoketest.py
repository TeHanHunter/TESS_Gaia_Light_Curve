#!/usr/bin/env python3
"""
Standalone smoke test for tglc.quick_lc: download all available sectors and plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tglc.quick_lc import tglc_lc, plot_lc, plot_pf_lc


def _ensure_trailing_slash(path: Path) -> str:
    return str(path.resolve()) + "/"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TGLC quick_lc on a target and plot results.")
    parser.add_argument("--target", default="TIC 16005254", help="Target identifier, e.g. 'TIC 16005254'")
    parser.add_argument("--outdir", default="/tmp/tglc_smoketest", help="Base output directory")
    parser.add_argument("--size", type=int, default=90, help="Cutout size")
    parser.add_argument("--limit-mag", type=float, default=16, help="Gaia limiting magnitude")
    parser.add_argument("--ffi", default="SPOC", choices=["SPOC", "TICA"], help="FFI product to use")
    parser.add_argument("--period", type=float, default=None, help="Optional period for phase-folded plot")
    parser.add_argument("--mid-transit-tbjd", type=float, default=None, help="Optional mid-transit TBJD")
    args = parser.parse_args()

    base_dir = Path(args.outdir)
    target_dir = base_dir / args.target
    target_dir.mkdir(parents=True, exist_ok=True)
    local_directory = _ensure_trailing_slash(target_dir)

    print(f"Running tglc_lc for {args.target} (all available sectors)...")
    tglc_lc(
        target=args.target,
        local_directory=local_directory,
        size=args.size,
        save_aper=True,
        limit_mag=args.limit_mag,
        get_all_lc=False,
        first_sector_only=False,
        last_sector_only=False,
        sector=None,
        prior=None,
        transient=None,
        ffi=args.ffi,
    )

    print("Plotting light curves...")
    plot_lc(local_directory=local_directory, kind="cal_aper_flux", ffi=args.ffi)
    plot_lc(local_directory=local_directory, kind="cal_psf_flux", ffi=args.ffi)

    if args.period is not None and args.mid_transit_tbjd is not None:
        print("Plotting phase-folded light curve...")
        plot_pf_lc(
            local_directory=f"{local_directory}lc/{args.ffi}/",
            period=args.period,
            mid_transit_tbjd=args.mid_transit_tbjd,
            kind="cal_aper_flux",
        )

    print(f"Done. Outputs in: {local_directory}")


if __name__ == "__main__":
    main()

"""Run the sector 56 experimental ePSF test.

This uses the existing sector 56 archive source pickles as read-only input and
writes opt-in experimental ePSF/LC products under a separate output root.

Default no-argument invocation runs the smoke cut cam1/ccd1/cut 00_00. Use
``--full`` to run the selected batch, or pass explicit ``--cams``/``--ccds``/
``--cuts`` selections for staged runs.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import sys
import traceback
from functools import partial
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tglc.target_lightcurve import epsf

SECTOR = 56
N_CUTS_SIDE = 14
N_CUTS = N_CUTS_SIDE ** 2
DEFAULT_BASELINE_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056")
DEFAULT_OUT_ROOT = Path("/pdo/users/tehan/_archive/2025_sector0056_variants/sector0056_tmag10_unit_bleedmask")


def _normalize_schema(source):
    """Handle schema differences from older GPU/CPU Source pickle writers."""
    if "designation" in source.gaia.colnames and "DESIGNATION" not in source.gaia.colnames:
        source.gaia.rename_column("designation", "DESIGNATION")
    if hasattr(source.tic, "colnames") and "gaia3" in source.tic.colnames and "dr3_source_id" not in source.tic.colnames:
        source.tic.rename_column("gaia3", "dr3_source_id")
    if not hasattr(source, "transient"):
        source.transient = None
    return source


def source_path(root, cam, ccd, cut_x, cut_y):
    base = Path(root) / "source" / f"{cam}-{ccd}"
    candidates = [
        base / f"source_{cut_x:02d}_{cut_y:02d}.pkl",
        base / f"source_{cut_x}_{cut_y}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_cut_token(token):
    token = str(token)
    if "_" in token:
        cut_x, cut_y = [int(part) for part in token.split("_", 1)]
        if not (0 <= cut_x < N_CUTS_SIDE and 0 <= cut_y < N_CUTS_SIDE):
            raise ValueError(f"cut {token} is outside the 14x14 sector grid")
        return [cut_x * N_CUTS_SIDE + cut_y]
    if ":" in token:
        start_text, end_text = token.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else N_CUTS
        if not (0 <= start <= end <= N_CUTS):
            raise ValueError(f"cut range {token} is outside 0:{N_CUTS}")
        return list(range(start, end))
    cut = int(token)
    if not (0 <= cut < N_CUTS):
        raise ValueError(f"cut {cut} is outside 0..{N_CUTS - 1}")
    return [cut]


def parse_cuts(cut_tokens):
    if not cut_tokens:
        return list(range(N_CUTS))
    cuts = []
    for token in cut_tokens:
        cuts.extend(parse_cut_token(token))
    return sorted(set(cuts))


def prepare_output_tree(baseline_root, out_root, link_source=True):
    out_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("epsf", "lc", "log", "diagnostics"):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)
    source_link = out_root / "source"
    baseline_source = baseline_root / "source"
    if link_source and not source_link.exists() and not source_link.is_symlink():
        source_link.symlink_to(baseline_source, target_is_directory=True)


def limit_source_cadences(source, max_cadences):
    if max_cadences is None:
        return source
    max_cadences = int(max_cadences)
    if max_cadences <= 0:
        raise ValueError("max_cadences must be positive")
    n_cadences = min(max_cadences, len(source.time))
    source.time = source.time[:n_cadences]
    source.flux = source.flux[:n_cadences]
    source.quality = source.quality[:n_cadences]
    source.cadence = source.cadence[:n_cadences]
    return source


def run_cut(cut, cam, ccd, baseline_root, out_root, overwrite=False, limit_mag=16, power=1.4, max_cadences=None):
    cut_x = cut // N_CUTS_SIDE
    cut_y = cut % N_CUTS_SIDE
    pkl_path = source_path(baseline_root, cam, ccd, cut_x, cut_y)
    if not pkl_path.exists():
        return {
            "status": "missing",
            "cam": cam,
            "ccd": ccd,
            "cut": cut,
            "cut_x": cut_x,
            "cut_y": cut_y,
            "path": str(pkl_path),
        }
    try:
        with pkl_path.open("rb") as handle:
            source = _normalize_schema(pickle.load(handle))
        source = limit_source_cadences(source, max_cadences)
        local_dir = f"{out_root}/"
        epsf(
            source,
            psf_size=11,
            factor=2,
            cut_x=cut_x,
            cut_y=cut_y,
            sector=SECTOR,
            power=power,
            local_directory=local_dir,
            limit_mag=limit_mag,
            save_aper=False,
            no_progress_bar=True,
            flux_scale="tmag10",
            epsf_normalization="unit_sum",
            overexposure_mask=True,
            overwrite=overwrite,
        )
        return {
            "status": "ok",
            "cam": cam,
            "ccd": ccd,
            "cut": cut,
            "cut_x": cut_x,
            "cut_y": cut_y,
        }
    except Exception as exc:
        return {
            "status": "error",
            "cam": cam,
            "ccd": ccd,
            "cut": cut,
            "cut_x": cut_x,
            "cut_y": cut_y,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def run_ccd(cam, ccd, cuts, baseline_root, out_root, processes, overwrite, limit_mag, power, max_cadences):
    print(f"[cam{cam} ccd{ccd}] starting {len(cuts)} cuts with {processes} workers", flush=True)
    fn = partial(
        run_cut,
        cam=cam,
        ccd=ccd,
        baseline_root=baseline_root,
        out_root=out_root,
        overwrite=overwrite,
        limit_mag=limit_mag,
        power=power,
        max_cadences=max_cadences,
    )
    if processes <= 1 or len(cuts) == 1:
        results = [fn(cut) for cut in cuts]
    else:
        with Pool(processes=processes) as pool:
            results = list(pool.imap_unordered(fn, cuts))
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] in ("missing", "error"):
            print(row, flush=True)
    print(f"[cam{cam} ccd{ccd}] done {counts}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cams", type=int, nargs="+")
    parser.add_argument("--ccds", type=int, nargs="+")
    parser.add_argument("--cuts", nargs="*", help="Cut indices, ranges like 0:14, or cut ids like 00_00.")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Run cam1/ccd1/cut 00_00 and exit.")
    parser.add_argument("--full", action="store_true", help="Run the selected batch. Needed for no-arg full-sector runs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-source-link", action="store_true")
    parser.add_argument("--limit-mag", type=float, default=16)
    parser.add_argument("--power", type=float, default=1.4)
    parser.add_argument("--max-cadences", type=int, help="Use only the first N cadences for bounded smoke tests.")
    args = parser.parse_args()

    if not args.baseline_root.exists():
        raise FileNotFoundError(f"baseline root does not exist: {args.baseline_root}")
    prepare_output_tree(args.baseline_root, args.out_root, link_source=not args.no_source_link)

    no_selection = args.cams is None and args.ccds is None and args.cuts is None
    if args.smoke or (not args.full and no_selection):
        print("[smoke] running cam1 ccd1 cut 00_00", flush=True)
        result = run_cut(
            0,
            cam=1,
            ccd=1,
            baseline_root=args.baseline_root,
            out_root=args.out_root,
            overwrite=args.overwrite,
            limit_mag=args.limit_mag,
            power=args.power,
            max_cadences=args.max_cadences,
        )
        print(result, flush=True)
        return 0 if result["status"] == "ok" else 1

    cams = args.cams if args.cams is not None else [1, 2, 3, 4]
    ccds = args.ccds if args.ccds is not None else [1, 2, 3, 4]
    cuts = parse_cuts(args.cuts)
    for cam in cams:
        for ccd in ccds:
            run_ccd(
                cam,
                ccd,
                cuts,
                baseline_root=args.baseline_root,
                out_root=args.out_root,
                processes=args.processes,
                overwrite=args.overwrite,
                limit_mag=args.limit_mag,
                power=args.power,
                max_cadences=args.max_cadences,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

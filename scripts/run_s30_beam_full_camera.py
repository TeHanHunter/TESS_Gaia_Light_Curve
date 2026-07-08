"""Orchestrate the full cam3 sector-30 BEAM/default TGLC run.

This wrapper intentionally runs one cut at a time.  The underlying runner keeps
the TGLC behavior unchanged, while this script adds durable per-cut logs and
``.done`` markers so the full camera run can be resumed safely.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_s30_beam_tica import CAMERA, N_CUTS_SIDE, parse_cuts

DEFAULT_FULL_ROOT = Path("/pdo/users/tehan/beam_tglc/s0030/full")
DEFAULT_LOG_ROOT = Path("/pdo/users/tehan/beam_tglc/logs/full_camera")
DEFAULT_FLUX_CACHE_ROOT = DEFAULT_FULL_ROOT / "flux_cache"


def cut_label(cut: int) -> str:
    cut_x = cut // N_CUTS_SIDE
    cut_y = cut % N_CUTS_SIDE
    return f"{cut_x:02d}_{cut_y:02d}"


def product_root(full_root: Path, product: str) -> Path:
    return full_root / product


def done_path(log_root: Path, product: str, ccd: int, cut: int) -> Path:
    return log_root / product / f"ccd{ccd}" / f"cut_{cut_label(cut)}.done"


def cut_log_path(log_root: Path, product: str, ccd: int, cut: int) -> Path:
    return log_root / product / f"ccd{ccd}" / f"cut_{cut_label(cut)}.log"


def status_path(log_root: Path) -> Path:
    return log_root / "full_camera_status.jsonl"


def append_status(log_root: Path, row: Dict[str, object]) -> None:
    path = status_path(log_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def iter_jobs(products: Sequence[str], ccds: Sequence[int], cuts: Sequence[int]) -> Iterable[tuple[str, int, int]]:
    for product in products:
        for ccd in ccds:
            for cut in cuts:
                yield product, ccd, cut


def build_command(args, product: str, ccd: int, cut: int) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_s30_beam_tica.py"),
        "--product",
        product,
        "--ccds",
        str(ccd),
        "--cuts",
        cut_label(cut),
        "--max-cadences",
        str(args.max_cadences),
        "--processes",
        str(args.processes),
        "--limit-mag",
        str(args.limit_mag),
        "--out-root",
        str(product_root(args.full_root, product)),
        "--tica-root",
        str(args.tica_root),
        "--beam-root",
        str(args.beam_root),
        "--spoc-root",
        str(args.spoc_root),
        "--input-units",
        args.input_units,
        "--flux-cache-root",
        str(args.flux_cache_root),
    ]
    if args.refresh_metadata_cache:
        cmd.append("--refresh-metadata-cache")
    if args.force:
        cmd.append("--overwrite")
    if product == "beam_likelihood":
        cmd.extend(["--reference-source-root", str(product_root(args.full_root, "default_tica"))])
    return cmd


def run_job(args, product: str, ccd: int, cut: int) -> int:
    marker = done_path(args.log_root, product, ccd, cut)
    log_path = cut_log_path(args.log_root, product, ccd, cut)
    marker.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists() and not args.force:
        print(f"skip {product} ccd{ccd} cut {cut_label(cut)}", flush=True)
        return 0

    cmd = build_command(args, product, ccd, cut)
    append_status(args.log_root, {
        "event": "start",
        "product": product,
        "camera": CAMERA,
        "ccd": ccd,
        "cut": cut_label(cut),
        "cmd": cmd,
    })
    print(f"run {product} ccd{ccd} cut {cut_label(cut)}", flush=True)
    start = time.monotonic()
    with log_path.open("a") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(time.strftime("START %Y-%m-%dT%H:%M:%S%z\n"))
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - start
        log.write(time.strftime("END %Y-%m-%dT%H:%M:%S%z\n"))
        log.write(f"RETURN_CODE {proc.returncode}\n")
        log.write(f"ELAPSED_S {elapsed:.1f}\n")

    event = {
        "event": "done" if proc.returncode == 0 else "error",
        "product": product,
        "camera": CAMERA,
        "ccd": ccd,
        "cut": cut_label(cut),
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log": str(log_path),
    }
    append_status(args.log_root, event)
    if proc.returncode == 0:
        marker.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    return proc.returncode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", nargs="+", choices=["default_tica", "beam_likelihood"],
                        default=["default_tica", "beam_likelihood"])
    parser.add_argument("--ccds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cuts", nargs="+", default=["all"])
    parser.add_argument("--max-cadences", type=int, default=0,
                        help="0 means all matched cadences.")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of cut subprocesses to run concurrently.")
    parser.add_argument("--limit-mag", type=float, default=16)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--flux-cache-root", type=Path, default=DEFAULT_FLUX_CACHE_ROOT)
    parser.add_argument("--tica-root", type=Path, default=Path("/pdo/qlp-data/tica-delivery/s0030"))
    parser.add_argument("--beam-root", type=Path, default=Path("/pdo/users/djtufto/BEAM/model_outputs/sector30/fits"))
    parser.add_argument("--spoc-root", type=Path, default=Path("/pdo/spoc-data/sector-030"))
    parser.add_argument("--input-units", choices=["e_per_cadence", "e_per_s"], default="e_per_cadence")
    parser.add_argument("--refresh-metadata-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun cuts even if .done markers exist.")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="Run at most this many not-yet-done jobs. 0 means no limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cuts = parse_cuts(args.cuts)
    args.full_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    attempted = 0
    for product in args.products:
        jobs = []
        for ccd in args.ccds:
            for cut in cuts:
                if done_path(args.log_root, product, ccd, cut).exists() and not args.force:
                    continue
                jobs.append((product, ccd, cut))
                if args.stop_after and attempted + len(jobs) >= args.stop_after:
                    break
            if args.stop_after and attempted + len(jobs) >= args.stop_after:
                break

        if args.workers <= 1:
            for job_product, ccd, cut in jobs:
                attempted += 1
                rc = run_job(args, job_product, ccd, cut)
                if rc != 0:
                    return rc
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {
                    executor.submit(run_job, args, job_product, ccd, cut): (job_product, ccd, cut)
                    for job_product, ccd, cut in jobs
                }
                for future in as_completed(future_to_job):
                    attempted += 1
                    rc = future.result()
                    if rc != 0:
                        return rc
        if args.stop_after and attempted >= args.stop_after:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())

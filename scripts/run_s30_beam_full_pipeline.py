"""Run the full sector-30 cam3 default/BEAM TGLC pipeline.

The pipeline is resumable:
* flux cache builders skip CCDs with cache manifests;
* cut runners skip cuts with ``.done`` markers;
* default_tica completes before beam_likelihood so BEAM can reuse catalogs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FULL_ROOT = Path("/pdo/users/tehan/beam_tglc/s0030/full")
DEFAULT_LOG_ROOT = Path("/pdo/users/tehan/beam_tglc/logs/full_camera")
DEFAULT_CACHE_ROOT = DEFAULT_FULL_ROOT / "flux_cache"


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_step(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"{timestamp()} START {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"{timestamp()} END returncode={proc.returncode}\n")
        log.flush()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def product_root(full_root: Path, product: str) -> Path:
    return full_root / product


def run_product_ccd(args, product: str, ccd: int) -> None:
    cache_cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_s30_flux_cache.py"),
        "--product",
        product,
        "--ccds",
        str(ccd),
        "--cuts",
        *args.cuts,
        "--max-cadences",
        str(args.max_cadences),
        "--cache-root",
        str(args.cache_root),
        "--metadata-cache-root",
        str(args.full_root),
        "--log-every",
        str(args.cache_log_every),
    ]
    if args.overwrite_cache:
        cache_cmd.append("--overwrite")
    run_step(cache_cmd, args.log_root / product / f"ccd{ccd}" / "pipeline_cache.log")

    run_cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_s30_beam_full_camera.py"),
        "--products",
        product,
        "--ccds",
        str(ccd),
        "--cuts",
        *args.cuts,
        "--max-cadences",
        str(args.max_cadences),
        "--workers",
        str(args.workers),
        "--processes",
        str(args.processes),
        "--full-root",
        str(args.full_root),
        "--log-root",
        str(args.log_root),
        "--flux-cache-root",
        str(args.cache_root),
    ]
    if args.force_cuts:
        run_cmd.append("--force")
    run_step(run_cmd, args.log_root / product / f"ccd{ccd}" / "pipeline_cuts.log")


def run_compare(args) -> None:
    if not {"default_tica", "beam_likelihood"}.issubset(set(args.products)):
        return
    out_dir = args.full_root / "diagnostics"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compare_s30_beam_precision.py"),
        "--default-root",
        str(product_root(args.full_root, "default_tica")),
        "--beam-root",
        str(product_root(args.full_root, "beam_likelihood")),
        "--out-dir",
        str(out_dir),
        "--ccds",
        *[str(ccd) for ccd in args.ccds],
    ]
    run_step(cmd, args.log_root / "pipeline_compare.log")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", nargs="+", choices=["default_tica", "beam_likelihood"],
                        default=["default_tica", "beam_likelihood"])
    parser.add_argument("--ccds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cuts", nargs="+", default=["all"])
    parser.add_argument("--max-cadences", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--cache-log-every", type=int, default=100)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--force-cuts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    for product in args.products:
        for ccd in args.ccds:
            run_product_ccd(args, product, ccd)
    run_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

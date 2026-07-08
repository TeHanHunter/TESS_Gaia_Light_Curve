"""Run TGLC on sector 30 default TICA or BEAM likelihood frames.

The BEAM sector-30 products are camera mosaics without the WCS/time/quality
metadata that TGLC needs.  This runner combines:

* default TICA flux from ``/pdo/qlp-data/tica-delivery/s0030`` or BEAM flux
  from ``/pdo/users/djtufto/BEAM/model_outputs/sector30/fits``;
* SPOC sector-30 FFIs for WCS, cadence quality, and exposure/livetime;
* the existing TGLC ``epsf`` light-curve machinery.

The default invocation is intentionally a bounded smoke run: cam3/ccd1,
cuts 00_00 and 07_07, first 50 matched cadences.
"""

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table, hstack, unique, vstack
from astropy.wcs import WCS

warnings.simplefilter("ignore", VerifyWarning)

try:
    import importlib_resources
except ModuleNotFoundError:
    from importlib import resources as importlib_resources

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tglc.ffi import Source, convert_gaia_id, tic_advanced_search_position_rows
from tglc.target_lightcurve import epsf

SECTOR = 30
CAMERA = 3
DEFAULT_SIZE = 150
N_CUTS_SIDE = 14
N_CUTS = N_CUTS_SIDE ** 2
ACTIVE_SIZE = 2048
TICA_ACTIVE_COLUMN_OFFSET = 44

DEFAULT_TICA_ROOT = Path("/pdo/qlp-data/tica-delivery/s0030")
DEFAULT_BEAM_ROOT = Path("/pdo/users/djtufto/BEAM/model_outputs/sector30/fits")
DEFAULT_SPOC_ROOT = Path("/pdo/spoc-data/sector-030")
DEFAULT_OUT_BASE = Path("/pdo/users/tehan/beam_tglc/s0030")

TICA_RE = re.compile(
    r"hlsp_tica_tess_ffi_s(?P<sector>\d+)-(?P<cadence>\d{8})-cam(?P<camera>\d)-ccd(?P<ccd>\d)"
)
BEAM_RE = re.compile(
    r"hlsp_tica_tess_ffi_s(?P<sector>\d+)-(?P<cadence>\d{8})-cam(?P<camera>\d)-ccdALL"
)


@dataclass
class FrameMeta:
    cadence: int
    spoc_path: str
    tstart: float
    tstop: float
    quality: int
    exposure_s: float

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.tstart + self.tstop)


def parse_tica_name(path: Path) -> Optional[Tuple[int, int, int, int]]:
    match = TICA_RE.search(path.name)
    if match is None:
        return None
    return (
        int(match.group("cadence")),
        int(match.group("sector")),
        int(match.group("camera")),
        int(match.group("ccd")),
    )


def parse_beam_name(path: Path) -> Optional[Tuple[int, int, int]]:
    match = BEAM_RE.search(path.name)
    if match is None:
        return None
    return (
        int(match.group("cadence")),
        int(match.group("sector")),
        int(match.group("camera")),
    )


def parse_cut_token(token: str, size: int = DEFAULT_SIZE) -> List[int]:
    token = str(token)
    if token.lower() == "all":
        return list(range(N_CUTS))
    if "_" in token:
        cut_x, cut_y = [int(part) for part in token.split("_", 1)]
        if not (0 <= cut_x < N_CUTS_SIDE and 0 <= cut_y < N_CUTS_SIDE):
            raise ValueError(f"cut {token} is outside the {N_CUTS_SIDE}x{N_CUTS_SIDE} grid")
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


def parse_cuts(cut_tokens: Sequence[str]) -> List[int]:
    cuts: List[int] = []
    for token in cut_tokens:
        cuts.extend(parse_cut_token(token))
    return sorted(set(cuts))


def cut_to_xy(cut: int, size: int = DEFAULT_SIZE) -> Tuple[int, int]:
    cut_x = cut // N_CUTS_SIDE
    cut_y = cut % N_CUTS_SIDE
    return cut_x, cut_y


def cut_origin(cut_x: int, cut_y: int, size: int = DEFAULT_SIZE) -> Tuple[int, int]:
    return cut_x * (size - 4), cut_y * (size - 4)


def product_out_root(out_root: Optional[Path], product: str) -> Path:
    if out_root is not None:
        return out_root
    if product == "default_tica":
        return DEFAULT_OUT_BASE / "default_tica"
    if product == "beam_likelihood":
        return DEFAULT_OUT_BASE / "beam_likelihood"
    raise ValueError(f"Unsupported product: {product}")


def prepare_output_tree(root: Path) -> None:
    for subdir in ("source", "epsf", "lc", "log", "diagnostics"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def index_tica_files(tica_root: Path, ccd: int) -> Dict[int, str]:
    folder = tica_root / f"cam{CAMERA}-ccd{ccd}"
    if not folder.exists():
        raise FileNotFoundError(f"Missing TICA folder: {folder}")
    index: Dict[int, str] = {}
    for path in sorted(folder.glob("*.fits*")):
        parsed = parse_tica_name(path)
        if parsed is None:
            continue
        cadence, sector, camera, parsed_ccd = parsed
        if sector == SECTOR and camera == CAMERA and parsed_ccd == ccd:
            index[cadence] = str(path)
    if not index:
        raise FileNotFoundError(f"No sector {SECTOR} cam{CAMERA}/ccd{ccd} TICA files in {folder}")
    return index


def index_beam_files(beam_root: Path) -> Dict[int, str]:
    if not beam_root.exists():
        raise FileNotFoundError(f"Missing BEAM folder: {beam_root}")
    index: Dict[int, str] = {}
    for path in sorted(beam_root.glob("*_img_likelihood.fits")):
        parsed = parse_beam_name(path)
        if parsed is None:
            continue
        cadence, sector, camera = parsed
        if sector == SECTOR and camera == CAMERA:
            index[cadence] = str(path)
    if not index:
        raise FileNotFoundError(f"No sector {SECTOR} cam{CAMERA} BEAM likelihood files in {beam_root}")
    return index


def spoc_cam_dir(spoc_root: Path) -> Path:
    candidates = [
        spoc_root,
        spoc_root / "cam3",
        spoc_root / "ffis" / "cam3",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*_ffic.fits*")):
            return candidate
    raise FileNotFoundError(f"Could not find cam3 SPOC FFI files under {spoc_root}")


def _header_float(header, key: str, default: float = np.nan) -> float:
    try:
        return float(header[key])
    except Exception:
        return default


def _header_int(header, key: str, default: int = 0) -> int:
    try:
        return int(header[key])
    except Exception:
        return default


def read_spoc_meta(path: Path) -> FrameMeta:
    with fits.open(path, mode="denywrite", memmap=False) as hdul:
        primary = hdul[0].header
        image = hdul[1].header
        cadence = _header_int(primary, "FFIINDEX")
        tstart = _header_float(image, "TSTART", _header_float(primary, "TSTART"))
        tstop = _header_float(image, "TSTOP", _header_float(primary, "TSTOP"))
        quality = _header_int(image, "DQUALITY", _header_int(primary, "QUALITY", 0))
        exposure_days = _header_float(image, "LIVETIME", _header_float(image, "EXPOSURE", tstop - tstart))
    exposure_s = float(exposure_days * 86400.0) if np.isfinite(exposure_days) else np.nan
    if not np.isfinite(exposure_s) or exposure_s <= 0:
        exposure_s = float((tstop - tstart) * 86400.0)
    return FrameMeta(
        cadence=cadence,
        spoc_path=str(path),
        tstart=float(tstart),
        tstop=float(tstop),
        quality=int(quality),
        exposure_s=float(exposure_s),
    )


def build_spoc_index(spoc_root: Path, ccd: int, cache_dir: Optional[Path] = None,
                     refresh_cache: bool = False,
                     required_cadences: Optional[Iterable[int]] = None) -> Dict[int, FrameMeta]:
    required = set(required_cadences) if required_cadences is not None else None
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"spoc_s{SECTOR:04d}_cam{CAMERA}_ccd{ccd}_metadata.json"
        if cache_path.exists() and not refresh_cache:
            with cache_path.open() as handle:
                payload = json.load(handle)
            index = {int(k): FrameMeta(**v) for k, v in payload.items()}
            if required is None:
                return index
            selected = {cadence: index[cadence] for cadence in sorted(required) if cadence in index}
            if len(selected) == len(required):
                return selected

    folder = spoc_cam_dir(spoc_root)
    paths = sorted(folder.glob(f"*-s{SECTOR:04d}-{CAMERA}-{ccd}-0195-s_ffic.fits*"))
    if not paths:
        raise FileNotFoundError(f"No SPOC sector {SECTOR} cam{CAMERA}/ccd{ccd} files in {folder}")

    index: Dict[int, FrameMeta] = {}
    for path in paths:
        meta = read_spoc_meta(path)
        if meta.cadence and (required is None or meta.cadence in required):
            index[meta.cadence] = meta
            if required is not None and len(index) == len(required):
                break

    if cache_path is not None and required is None:
        with cache_path.open("w") as handle:
            json.dump({str(k): asdict(v) for k, v in sorted(index.items())}, handle, indent=2)
    return index


def candidate_cadences_for_metadata(product: str, tica_index: Dict[int, str],
                                    beam_index: Optional[Dict[int, str]],
                                    max_cadences: Optional[int]) -> Optional[List[int]]:
    if max_cadences is None or max_cadences <= 0:
        return None
    candidates = set(tica_index)
    if product == "beam_likelihood":
        if beam_index is None:
            raise ValueError("beam_index is required for beam_likelihood")
        candidates &= set(beam_index)
    return sorted(candidates)[:max_cadences]


def read_tica_midtime(path: str, fallback: float) -> float:
    try:
        header = fits.getheader(path, 0)
        return float(header.get("MIDTJD", fallback))
    except Exception:
        return float(fallback)


def select_common_cadences(product: str, tica_index: Dict[int, str], spoc_index: Dict[int, FrameMeta],
                           beam_index: Optional[Dict[int, str]], max_cadences: Optional[int]) -> List[int]:
    common = set(tica_index) & set(spoc_index)
    if product == "beam_likelihood":
        if beam_index is None:
            raise ValueError("beam_index is required for beam_likelihood")
        common &= set(beam_index)
    cadences = sorted(common)
    if max_cadences is not None and max_cadences > 0:
        cadences = cadences[:max_cadences]
    if not cadences:
        raise RuntimeError("No matched cadences found")
    return cadences


def split_beam_mosaic_to_ccds(mosaic: np.ndarray) -> Dict[int, np.ndarray]:
    if mosaic.shape != (ACTIVE_SIZE * 2, ACTIVE_SIZE * 2):
        raise ValueError(f"Expected BEAM mosaic shape {(ACTIVE_SIZE * 2, ACTIVE_SIZE * 2)}, got {mosaic.shape}")
    return {
        3: mosaic[:ACTIVE_SIZE, :ACTIVE_SIZE],
        4: mosaic[:ACTIVE_SIZE, ACTIVE_SIZE:],
        2: np.flip(mosaic[ACTIVE_SIZE:, :ACTIVE_SIZE]),
        1: np.flip(mosaic[ACTIVE_SIZE:, ACTIVE_SIZE:]),
    }


def read_default_tica_cut(path: str, x0: int, y0: int, size: int) -> np.ndarray:
    with fits.open(path, mode="denywrite", memmap=True) as hdul:
        data = hdul[0].data
        cut = data[y0:y0 + size, TICA_ACTIVE_COLUMN_OFFSET + x0:TICA_ACTIVE_COLUMN_OFFSET + x0 + size]
        return np.asarray(cut, dtype=np.float32)


def read_beam_cut(path: str, ccd: int, x0: int, y0: int, size: int) -> np.ndarray:
    with fits.open(path, mode="denywrite", memmap=True) as hdul:
        mosaic = hdul[0].data
        ccds = split_beam_mosaic_to_ccds(mosaic)
        return np.asarray(ccds[ccd][y0:y0 + size, x0:x0 + size], dtype=np.float32)


def flux_cache_paths(flux_cache_root: Optional[Path], product: str, ccd: int,
                     cut_x: int, cut_y: int) -> Tuple[Optional[Path], Optional[Path]]:
    if flux_cache_root is None:
        return None, None
    root = flux_cache_root / product / f"ccd{ccd}"
    return root / f"flux_cut_{cut_x:02d}_{cut_y:02d}.npy", root / "metadata.npz"


def load_flux_cube_from_cache(flux_cache_root: Optional[Path], product: str, ccd: int,
                              cut_x: int, cut_y: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    flux_path, metadata_path = flux_cache_paths(flux_cache_root, product, ccd, cut_x, cut_y)
    if flux_path is None or metadata_path is None or not flux_path.exists() or not metadata_path.exists():
        return None
    flux = np.load(flux_path, mmap_mode="r")
    metadata = np.load(metadata_path)
    return (
        flux,
        np.array(metadata["time"], dtype=float),
        np.array(metadata["quality"], dtype=np.int16),
        np.array(metadata["exposure_s"], dtype=float),
        np.array(metadata["cadence"], dtype=np.int32),
    )


def load_flux_cube(product: str, ccd: int, cut_x: int, cut_y: int, size: int, cadences: Sequence[int],
                   tica_index: Dict[int, str], spoc_index: Dict[int, FrameMeta],
                   beam_index: Optional[Dict[int, str]], input_units: str,
                   flux_cache_root: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = load_flux_cube_from_cache(flux_cache_root, product, ccd, cut_x, cut_y)
    if cached is not None:
        return cached

    x0, y0 = cut_origin(cut_x, cut_y, size=size)
    flux = np.empty((len(cadences), size, size), dtype=np.float32)
    time = np.empty(len(cadences), dtype=float)
    quality = np.empty(len(cadences), dtype=np.int16)
    exposure_s = np.empty(len(cadences), dtype=float)
    for i, cadence in enumerate(cadences):
        meta = spoc_index[cadence]
        if product == "default_tica":
            frame = read_default_tica_cut(tica_index[cadence], x0=x0, y0=y0, size=size)
        elif product == "beam_likelihood":
            if beam_index is None:
                raise ValueError("beam_index is required for beam_likelihood")
            frame = read_beam_cut(beam_index[cadence], ccd=ccd, x0=x0, y0=y0, size=size)
        else:
            raise ValueError(f"Unsupported product: {product}")
        if input_units == "e_per_cadence":
            frame = frame / meta.exposure_s
        elif input_units != "e_per_s":
            raise ValueError("input_units must be 'e_per_cadence' or 'e_per_s'")
        flux[i] = frame
        time[i] = read_tica_midtime(tica_index[cadence], fallback=meta.midpoint)
        quality[i] = meta.quality
        exposure_s[i] = meta.exposure_s
    return flux, time, quality, exposure_s, np.array(cadences, dtype=np.int32)


def representative_wcs(spoc_index: Dict[int, FrameMeta], cadences: Sequence[int]) -> WCS:
    for cadence in cadences:
        meta = spoc_index[cadence]
        if meta.quality == 0:
            with fits.open(meta.spoc_path, mode="denywrite", memmap=False) as hdul:
                return WCS(hdul[1].header)
    first = spoc_index[cadences[0]]
    with fits.open(first.spoc_path, mode="denywrite", memmap=False) as hdul:
        return WCS(hdul[1].header)


def build_cut_mask(flux: np.ndarray, camera: int, ccd: int, cut_x: int, cut_y: int, size: int) -> np.ma.MaskedArray:
    mask_resource = importlib_resources.files("tglc").joinpath("background_mask/median_mask.fits").open("rb")
    try:
        with fits.open(mask_resource) as hdul:
            median_mask = np.array(hdul[0].data[(camera - 1) * 4 + (ccd - 1)], copy=True)
    finally:
        mask_resource.close()
    x0, _ = cut_origin(cut_x, cut_y, size=size)
    mask_data = np.repeat(median_mask[x0:x0 + size].reshape(1, size), repeats=size, axis=0)

    med_flux = np.nanmedian(flux, axis=0)
    finite = np.isfinite(med_flux)
    bad_pixels = np.zeros((size, size), dtype=bool)
    if np.any(finite):
        finite_values = med_flux[finite]
        high = 0.8 * np.nanmax(finite_values)
        low = 0.2 * np.nanmedian(finite_values)
        bad_pixels[med_flux > high] = True
        bad_pixels[med_flux < low] = True
    bad_pixels[~finite] = True

    y_bad, x_bad = np.where(bad_pixels)
    for y, x in zip(y_bad, x_bad):
        if y < size - 1:
            bad_pixels[y + 1, x] = True
        if y > 0:
            bad_pixels[y - 1, x] = True
        if x < size - 1:
            bad_pixels[y, x + 1] = True
        if x > 0:
            bad_pixels[y, x - 1] = True

    mask = np.ma.masked_array(mask_data, mask=bad_pixels)
    return np.ma.masked_equal(mask, 0)


def _table_value(table: Table, column: str, index: int, default: float = np.nan) -> float:
    try:
        value = table[column][index]
    except Exception:
        return default
    if np.ma.is_masked(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _designation_id(designation) -> str:
    return str(designation)


def _build_gaia_tic_for_cut_once(source: Source, x0: int, y0: int, limit_mag: float,
                                 gaia_tap_server: str) -> None:
    co1 = 38.5
    co2 = 116.5
    catalogs = [
        Source.search_gaia(source, x0, y0, co1, co1),
        Source.search_gaia(source, x0, y0, co1, co2),
        Source.search_gaia(source, x0, y0, co2, co1),
        Source.search_gaia(source, x0, y0, co2, co2),
    ]
    catalogs = [catalog for catalog in catalogs if catalog is not None and len(catalog) > 0]
    if not catalogs:
        raise RuntimeError(f"No Gaia sources found for cut origin ({x0}, {y0})")
    catalogdata = unique(vstack(catalogs, join_type="exact"), keys="DESIGNATION")

    coord = source.wcs.pixel_to_world([x0 + (source.size - 1) / 2 + TICA_ACTIVE_COLUMN_OFFSET],
                                      [y0 + (source.size - 1) / 2])[0].to_string()
    ra_center = float(coord.split()[0])
    dec_center = float(coord.split()[1])
    catalogdata_tic = tic_advanced_search_position_rows(
        ra=ra_center,
        dec=dec_center,
        radius=(source.size + 2) * 21 * 0.707 / 3600,
        limit_mag=limit_mag,
    )
    source.tic = convert_gaia_id(catalogdata_tic, gaia_tap_server=gaia_tap_server)

    median_time = np.median(source.time)
    interval = (median_time - 388.5) / 365.25
    num_gaia = len(catalogdata)
    x_gaia = np.zeros(num_gaia)
    y_gaia = np.zeros(num_gaia)
    tess_mag = np.zeros(num_gaia)
    in_frame = np.ones(num_gaia, dtype=bool)

    for i in range(num_gaia):
        ra = _table_value(catalogdata, "ra", i)
        dec = _table_value(catalogdata, "dec", i)
        pmra = _table_value(catalogdata, "pmra", i)
        pmdec = _table_value(catalogdata, "pmdec", i)
        if np.isfinite(pmra) and np.isfinite(dec):
            ra += pmra * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
        if np.isfinite(pmdec):
            dec += pmdec * interval / 1000 / 3600

        pixel = source.wcs.all_world2pix(np.array([ra, dec]).reshape((1, 2)), 0, quiet=True)
        x_gaia[i] = pixel[0][0] - x0 - TICA_ACTIVE_COLUMN_OFFSET
        y_gaia[i] = pixel[0][1] - y0

        g_mag = _table_value(catalogdata, "phot_g_mean_mag", i)
        bp_mag = _table_value(catalogdata, "phot_bp_mean_mag", i)
        rp_mag = _table_value(catalogdata, "phot_rp_mean_mag", i)
        if not np.isfinite(g_mag) or g_mag >= 25:
            in_frame[i] = False
        elif -4 < x_gaia[i] < source.size + 3 and -4 < y_gaia[i] < source.size + 3:
            dif = bp_mag - rp_mag
            tess_mag[i] = g_mag - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
            if not np.isfinite(tess_mag[i]):
                tess_mag[i] = g_mag - 0.430
            if not np.isfinite(tess_mag[i]):
                in_frame[i] = False
        else:
            in_frame[i] = False

    if not np.any(in_frame):
        raise RuntimeError(f"No Gaia sources landed in cut origin ({x0}, {y0})")

    tess_flux = 10 ** (-tess_mag / 2.5)
    t = Table()
    t["tess_mag"] = tess_mag[in_frame]
    t["tess_flux"] = tess_flux[in_frame]
    t["tess_flux_ratio"] = tess_flux[in_frame] / np.nanmax(tess_flux[in_frame])
    t[f"sector_{source.sector}_x"] = x_gaia[in_frame]
    t[f"sector_{source.sector}_y"] = y_gaia[in_frame]
    catalogdata = hstack([catalogdata[in_frame], t])
    catalogdata.sort("tess_mag")
    source.gaia = catalogdata


def build_gaia_tic_for_cut(source: Source, x0: int, y0: int, limit_mag: float,
                           gaia_tap_server: str, attempts: int = 5) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            _build_gaia_tic_for_cut_once(source, x0, y0, limit_mag, gaia_tap_server)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            wait_s = min(300, 30 * attempt)
            print(
                f"catalog build failed for cut origin ({x0}, {y0}) "
                f"attempt {attempt}/{attempts}: {exc!r}; retrying in {wait_s}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_s)
    raise last_error  # type: ignore[misc]


def source_path(root: Path, cam: int, ccd: int, cut_x: int, cut_y: int) -> Path:
    return root / "source" / f"{cam}-{ccd}" / f"source_{cut_x:02d}_{cut_y:02d}.pkl"


def maybe_copy_reference_catalogs(source: Source, reference_root: Optional[Path], cam: int, ccd: int,
                                  cut_x: int, cut_y: int) -> bool:
    if reference_root is None:
        return False
    path = source_path(reference_root, cam, ccd, cut_x, cut_y)
    if not path.exists():
        return False
    with path.open("rb") as handle:
        reference = pickle.load(handle)
    source.gaia = reference.gaia
    source.tic = reference.tic
    return True


def build_source_for_cut(product: str, ccd: int, cut_x: int, cut_y: int, size: int,
                         cadences: Sequence[int], tica_index: Dict[int, str],
                         spoc_index: Dict[int, FrameMeta], beam_index: Optional[Dict[int, str]],
                         input_units: str, limit_mag: float, gaia_tap_server: str,
                         reference_source_root: Optional[Path],
                         flux_cache_root: Optional[Path]) -> Tuple[Source, Dict[str, object]]:
    flux, time, quality, exposure_s, cube_cadences = load_flux_cube(
        product=product,
        ccd=ccd,
        cut_x=cut_x,
        cut_y=cut_y,
        size=size,
        cadences=cadences,
        tica_index=tica_index,
        spoc_index=spoc_index,
        beam_index=beam_index,
        input_units=input_units,
        flux_cache_root=flux_cache_root,
    )
    cadences = [int(cadence) for cadence in cube_cadences]
    wcs = representative_wcs(spoc_index, cadences)
    mask = build_cut_mask(flux, CAMERA, ccd, cut_x, cut_y, size)
    source = Source.__new__(Source)
    source.size = size
    source.sector = SECTOR
    source.camera = CAMERA
    source.ccd = ccd
    source.cadence = np.array(cadences, dtype=np.int32)
    source.quality = quality
    source.exposure = float(np.nanmedian(exposure_s))
    source.wcs = wcs
    source.flux = flux
    source.mask = mask
    source.time = time
    source.transient = None
    source.product = product
    source.input_units = input_units
    source.flux_unit = "e-/s"
    source.exposure_s = exposure_s
    source.ffi = "TICA"

    copied_catalogs = maybe_copy_reference_catalogs(
        source,
        reference_root=reference_source_root,
        cam=CAMERA,
        ccd=ccd,
        cut_x=cut_x,
        cut_y=cut_y,
    )
    if not copied_catalogs:
        x0, y0 = cut_origin(cut_x, cut_y, size=size)
        build_gaia_tic_for_cut(source, x0=x0, y0=y0, limit_mag=limit_mag, gaia_tap_server=gaia_tap_server)

    manifest = {
        "product": product,
        "camera": CAMERA,
        "ccd": ccd,
        "cut": f"{cut_x:02d}_{cut_y:02d}",
        "cut_origin": cut_origin(cut_x, cut_y, size=size),
        "size": size,
        "n_cadences": len(cadences),
        "cadence_min": int(min(cadences)),
        "cadence_max": int(max(cadences)),
        "input_units": input_units,
        "output_flux_unit": "e-/s",
        "exposure_s_median": float(np.nanmedian(exposure_s)),
        "exposure_s_min": float(np.nanmin(exposure_s)),
        "exposure_s_max": float(np.nanmax(exposure_s)),
        "catalogs_copied_from_reference": copied_catalogs,
    }
    return source, manifest


def run_cut(cut: int, ccd: int, product: str, tica_root: Path, beam_root: Path, spoc_root: Path,
            out_root: Path, size: int, max_cadences: Optional[int], input_units: str, limit_mag: float,
            overwrite: bool, refresh_metadata_cache: bool, reference_source_root: Optional[Path],
            gaia_tap_server: str, flux_cache_root: Optional[Path]) -> Dict[str, object]:
    cut_x, cut_y = cut_to_xy(cut, size=size)
    try:
        prepare_output_tree(out_root)
        spath = source_path(out_root, CAMERA, ccd, cut_x, cut_y)
        spath.parent.mkdir(parents=True, exist_ok=True)

        cache_dir = out_root / "log"
        tica_index = index_tica_files(tica_root, ccd)
        beam_index = index_beam_files(beam_root) if product == "beam_likelihood" else None
        required_cadences = candidate_cadences_for_metadata(product, tica_index, beam_index, max_cadences=max_cadences)
        spoc_index = build_spoc_index(
            spoc_root,
            ccd,
            cache_dir=cache_dir,
            refresh_cache=refresh_metadata_cache,
            required_cadences=required_cadences,
        )
        cadences = select_common_cadences(product, tica_index, spoc_index, beam_index, max_cadences=max_cadences)

        if spath.exists() and not overwrite:
            with spath.open("rb") as handle:
                source = pickle.load(handle)
            manifest = {
                "product": product,
                "camera": CAMERA,
                "ccd": ccd,
                "cut": f"{cut_x:02d}_{cut_y:02d}",
                "source_reused": True,
                "n_cadences": int(len(source.time)),
            }
        else:
            source, manifest = build_source_for_cut(
                product=product,
                ccd=ccd,
                cut_x=cut_x,
                cut_y=cut_y,
                size=size,
                cadences=cadences,
                tica_index=tica_index,
                spoc_index=spoc_index,
                beam_index=beam_index,
                input_units=input_units,
                limit_mag=limit_mag,
                gaia_tap_server=gaia_tap_server,
                reference_source_root=reference_source_root,
                flux_cache_root=flux_cache_root,
            )
            with spath.open("wb") as handle:
                pickle.dump(source, handle, pickle.HIGHEST_PROTOCOL)
            manifest["source_reused"] = False

        manifest_path = out_root / "log" / f"manifest_s{SECTOR:04d}_cam{CAMERA}_ccd{ccd}_cut_{cut_x:02d}_{cut_y:02d}_{product}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        epsf(
            source,
            psf_size=11,
            factor=2,
            cut_x=cut_x,
            cut_y=cut_y,
            sector=SECTOR,
            power=1.4,
            local_directory=f"{out_root}/",
            limit_mag=limit_mag,
            save_aper=False,
            no_progress_bar=True,
            overwrite=overwrite,
            ffi="TICA",
        )
        return {
            "status": "ok",
            "product": product,
            "cam": CAMERA,
            "ccd": ccd,
            "cut": cut,
            "cut_x": cut_x,
            "cut_y": cut_y,
            "manifest": str(manifest_path),
            "source": str(spath),
        }
    except Exception as exc:
        return {
            "status": "error",
            "product": product,
            "cam": CAMERA,
            "ccd": ccd,
            "cut": cut,
            "cut_x": cut_x,
            "cut_y": cut_y,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def run_ccd(ccd: int, cuts: Sequence[int], args) -> List[Dict[str, object]]:
    fn = partial(
        run_cut,
        ccd=ccd,
        product=args.product,
        tica_root=args.tica_root,
        beam_root=args.beam_root,
        spoc_root=args.spoc_root,
        out_root=args.out_root,
        size=args.size,
        max_cadences=args.max_cadences if args.max_cadences and args.max_cadences > 0 else None,
        input_units=args.input_units,
        limit_mag=args.limit_mag,
        overwrite=args.overwrite,
        refresh_metadata_cache=args.refresh_metadata_cache,
        reference_source_root=args.reference_source_root,
        gaia_tap_server=args.gaia_tap_server,
        flux_cache_root=args.flux_cache_root,
    )
    if args.processes <= 1 or len(cuts) == 1:
        return [fn(cut) for cut in cuts]
    with Pool(processes=args.processes) as pool:
        return list(pool.imap_unordered(fn, cuts))


def write_run_summary(out_root: Path, results: List[Dict[str, object]], args) -> Path:
    summary = {
        "product": args.product,
        "sector": SECTOR,
        "camera": CAMERA,
        "out_root": str(out_root),
        "results": results,
        "counts": {},
    }
    for row in results:
        summary["counts"][row["status"]] = summary["counts"].get(row["status"], 0) + 1
    path = out_root / "log" / f"run_summary_s{SECTOR:04d}_cam{CAMERA}_{args.product}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["default_tica", "beam_likelihood"], default="default_tica")
    parser.add_argument("--tica-root", type=Path, default=DEFAULT_TICA_ROOT)
    parser.add_argument("--beam-root", type=Path, default=DEFAULT_BEAM_ROOT)
    parser.add_argument("--spoc-root", type=Path, default=DEFAULT_SPOC_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--ccds", type=int, nargs="+", default=[1])
    parser.add_argument("--cuts", nargs="+", default=["00_00", "07_07"],
                        help="Cut ids like 00_00, numeric cut ids, ranges like 0:14, or all.")
    parser.add_argument("--max-cadences", type=int, default=50,
                        help="Maximum matched cadences to use; pass 0 for all matched cadences.")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--limit-mag", type=float, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--input-units", choices=["e_per_cadence", "e_per_s"], default="e_per_cadence",
                        help="TICA/BEAM image units before TGLC. e_per_cadence divides by matched SPOC livetime.")
    parser.add_argument("--refresh-metadata-cache", action="store_true")
    parser.add_argument("--reference-source-root", type=Path,
                        help="Optional source root to copy Gaia/TIC catalogs from, usually the default_tica run.")
    parser.add_argument("--gaia-tap-server", default="https://gea.esac.esa.int/tap-server/tap")
    parser.add_argument("--flux-cache-root", type=Path,
                        help="Optional root containing precomputed per-cut flux cubes from build_s30_flux_cache.py.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root = product_out_root(args.out_root, args.product)
    prepare_output_tree(args.out_root)
    cuts = parse_cuts(args.cuts)
    results: List[Dict[str, object]] = []
    for ccd in args.ccds:
        if ccd not in (1, 2, 3, 4):
            raise ValueError(f"CCD must be 1..4, got {ccd}")
        print(f"[cam{CAMERA} ccd{ccd}] product={args.product} cuts={len(cuts)}", flush=True)
        ccd_results = run_ccd(ccd, cuts, args)
        results.extend(ccd_results)
        counts: Dict[str, int] = {}
        for row in ccd_results:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row["status"] != "ok":
                print(json.dumps(row, sort_keys=True), flush=True)
        print(f"[cam{CAMERA} ccd{ccd}] done {counts}", flush=True)
    summary_path = write_run_summary(args.out_root, results, args)
    print(summary_path, flush=True)
    return 0 if all(row["status"] == "ok" for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())

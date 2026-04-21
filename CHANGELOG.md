# Changelog

## 0.7.2
- Added a configurable `gaia_tap_server` parameter to `tglc_lc`, `ffi_cut`, `Source_cut`, and `convert_gaia_id` so Gaia TAP queries can fall back to a user-specified mirror when the primary ESA server is down. Credit: Caleb Cañas (@cicanas).
- `convert_gaia_id` now retries each 10k-ID batch against the mirror before giving up and using the TIC-GAIA (DR2) fallback.
- `Source_cut` Gaia DR3 cone search retries via the mirror TAP endpoint before falling back to `Catalogs.query_region`.
- `quick_lc` DR2→DR3 designation lookup collapsed to a single `gaia_source` ⨝ `dr2_neighbourhood` join (no intermediate `tmpgaiavals`), with mirror fallback.
- `tglc.ffi_cut` exposes an optional `tesscube` import (guarded by `try/except ImportError`) in preparation for AWS-backed cutouts of `size > 99`.

## 0.7.1
- Spinner in `_dot_wait` now detects non-TTY stdout (PyCharm run console, pipes, CI logs) and prints a single line per call instead of repeating the `\r` animation.
- Gaia DR3 cone search in `ffi_cut` falls back to the MAST Gaia (DR2) catalog when the Gaia TAP is unavailable.
- `convert_gaia_id` (DR2 → DR3 crossmatch) falls back to `catalogdata_tic['GAIA']` on Gaia TAP failure so the pipeline survives archive outages.
- `effective_psf` builds 5×5 cutouts by indexing into a preallocated NaN array (safer near FFI edges).
- `lc_output` default `ffi` is now `'SPOC'`.

## 0.7.0
- Updated dependencies for modern Python/astropy compatibility.
- Added `importlib_resources` and improved package data loading.
- Added richer progress logging and longer default MAST timeout (3600s).
- Added a smoketest script for `quick_lc` in `scripts/quick_lc_smoketest.py`.
- TICA support is **experimental** and still under testing; Tesscut product availability may be limited.

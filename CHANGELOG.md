# Changelog

## 0.7.1
- Spinner in `_dot_wait` now detects non-TTY stdout (PyCharm run console, pipes, CI logs) and prints a single line per call instead of repeating the `\r` animation.
- Gaia DR3 cone search in `ffi_cut` falls back to the MAST Gaia (DR2) catalog when the Gaia TAP is unavailable.
- `convert_gaia_id` (DR2 → DR3 crossmatch) falls back to `catalogdata_tic['GAIA']` on Gaia TAP failure so the pipeline survives archive outages.
- `quick_lc` resolves `"TIC <id>"` / numeric targets explicitly, handles MAST resolver failures with clearer errors, and retries `ffi_cut` up to 5 attempts per sector.
- `effective_psf` builds 5×5 cutouts by indexing into a preallocated NaN array (safer near FFI edges).

## 0.7.0
- Updated dependencies for modern Python/astropy compatibility.
- Added `importlib_resources` and improved package data loading.
- Added richer progress logging and longer default MAST timeout (3600s).
- Added a smoketest script for `quick_lc` in `scripts/quick_lc_smoketest.py`.
- TICA support is **experimental** and still under testing; Tesscut product availability may be limited.

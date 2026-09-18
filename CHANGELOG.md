# Changelog

## 0.8.0

Corrected SPOC extraction and robustness improvements. New products carry
`TGLCVER=0.8.0` and `PROCVER=spoc-0.8.0`. See `docs/release_0_8_0.md` for
migration notes, limitations, and release-validation status.

**Bilinear interpolation correction and ePSF incompatibility:** earlier public
versions contained an error in the subpixel coordinate mapping used for bilinear
interpolation. Version 0.8.0 fixes that error in both ePSF fitting and target
rendering. Fitted ePSFs from 0.8.0 and earlier versions are not interchangeable
in either direction. Refit the ePSF from the science images and rerun extraction
with 0.8.0; renaming or converting an old cache file does not make it compatible.
Previously generated light curves remain readable but retain the earlier
processing and are not corrected by installing the update.

- Correct the bilinear interpolation coordinate mapping described above and
  center the regularization grid consistently. Retain the
  SPOC `edge_compression=1e-4` setting rather than copying TICA's new value.
- Exclude saturated pixels and a configurable surrounding neighborhood from
  global ePSF and target PSF fits to prevent those pixel values from biasing the
  fitted model. Also apply supplied masks and finite-data checks. Use rank-aware least squares and
  preserve failed cadences as missing with processing flags.
- Propagate Gaia positions to the observing epoch, retain outside-neighbor PSF
  support, and keep unknown/ambiguous DR2-to-DR3 matches explicitly unknown.
- Preserve signed raw flux, separate aperture/PSF offsets, full-support aperture
  fractions, 32-bit quality flags, and unknown cadence IDs in FITS products.
  Separate output directories by FFI product and validate scientific caches.
- Consolidate package metadata, Python 3.10–3.12 support, public TIC/path handling,
  returned output paths, product-aware plotting, documentation, and wheel CI.
- Include the integer-TIC regression fix proposed in PR #16 for both `tglc_lc`
  and direct `ffi_cut`/`Source_cut` calls. Equivalent integer and string TIC
  identifiers use the same catalog lookup and source-cache identity.
- Add independent numerical, mixed-failure, public API, and serialization tests.
  A strict expected failure records variable-target feedback in a small
  12-star stress case. Larger-field experiments strongly reduce the effect;
  retain this diagnostic without introducing target exclusion by default.
- Compare real planet transits and a known stellar eclipse on matched inputs.
  PSF reference residuals improve in this small cohort; catalog-normalized
  fractional amplitudes do not improve uniformly. Keep raw output and document
  the normalization limitations rather than claiming universal depth accuracy.

Migration: Python 3.10–3.12 is required. Light curves now live under `lc/SPOC/`
or `lc/TICA/`; use returned output paths or update legacy glob patterns. Source
and ePSF caches are validated and incompatible caches are rebuilt. FITS quality
columns are 32-bit, unknown cadence IDs are null/-1, and two signed raw-flux
columns precede catalog offsets and detrending. Existing MAST products are not
regenerated. Saturation masking is configurable and does not establish precise
saturated-target photometry; TICA remains experimental.

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

Usage
=====

Installation and first extraction
---------------------------------

Use Python 3.10–3.12 in a dedicated environment. The public and MIT QLP packages
both use the import name ``tglc`` and must not be installed together::

    python -m pip install tglc

The examples below describe TGLC 0.8.0. To use this source checkout before or
after publication, install it with ``python -m pip install -e .``.
A first extraction can be limited to one sector::

    from pathlib import Path
    from tglc.quick_lc import tglc_lc, plot_lc

    directory = Path("tglc-output")
    paths = tglc_lc(target=16005254, local_directory=directory,
                    first_sector_only=True, ffi="SPOC")
    plot_lc(local_directory=directory, ffi="SPOC")

``target`` accepts a positive integer TIC ID, a numeric string, or a string such
as ``"TIC 16005254"``. These forms use the same target-resolution path. Other
names retain the legacy resolution path; verify the output TIC and Gaia IDs.
Archive access is required for a fresh target. A supplied directory can be a
string or ``Path`` and needs no trailing slash. The function returns a list of
FITS paths written during the call; it raises an error if no sectors are found.

Extraction options
------------------

* ``sector=56`` selects a sector. With ``sector=None``, all available sectors
  are processed unless ``first_sector_only`` or ``last_sector_only`` is true.
  The two flags cannot both be true. An explicit integer sector takes priority.
* ``size=90`` is the cutout width in detector pixels. The ordinary Tesscut path
  supports cutouts below 100 pixels; larger cutouts need optional infrastructure.
* ``limit_mag=16`` controls the modeling catalog magnitude limit.
* ``get_all_lc=False`` extracts the selected target; true requests field targets.
* ``save_aper=True`` includes the decontaminated 5 by 5 pixel cube in each FITS
  primary HDU. This allows inspecting or summing a different aperture.
* ``prior=None`` keeps catalog neighbor ratios fixed. A positive numeric prior
  selects the floating-neighbor fit. Its output path is covered by tests, but
  an optimized prior or uniformly improved deblending is not guaranteed.
* ``saturation_limit=80000.0`` masks pixels at the conservative cutoff in
  electrons/second. ``None`` disables threshold masking. ``saturation_dilation=1``
  masks an additional pixel neighborhood around detected saturated pixels.
  Saturation masking protects the fit; it does not recover lost charge or
  establish accurate light curves for saturated targets. The cutoff requires
  validation across camera, CCD, and observing conditions.
* ``mast_timeout=3600`` is the Tesscut timeout in seconds.
  ``gaia_tap_server`` can select a fallback Gaia TAP endpoint.
* ``ffi="SPOC"`` is the recommended public workflow. ``ffi="TICA"`` remains
  experimental and needs separate timing, header, unit, and quality validation.

Outputs, time, and quality
--------------------------

New light curves are stored under ``lc/SPOC/`` or ``lc/TICA/``. Existing files in
``lc/`` remain available. ``plot_lc(..., ffi="SPOC")`` selects one product;
omitting ``ffi`` recursively discovers both products and legacy files. For a
phase-folded plot, pass a single product directory to
``plot_pf_lc(local_directory=directory / "lc" / "SPOC", ...)``.

``time`` uses barycentric TDB days relative to JD 2457000 as recorded in the FITS
headers. Inspect ``TESS_flags`` and ``TGLC_flags`` before choosing cadences;
zero in both columns is the conservative selection used by the plotting helper.
Both quality columns use 32-bit integers. ``TGLC_flags`` combines bit values:
1 (background outlier), 2 (failed ePSF fit), 4 (saturation in the target window),
8 (invalid aperture flux), 16 (invalid PSF flux), and 32 (no good normalization
cadences). Saturation can invalidate an aperture sum while leaving a flagged
PSF measurement from usable wings. Missing cadence identifiers are stored as
-1 with FITS null metadata, not fabricated sequence numbers.

Processing flag definitions are maintained in ``tglc.processing`` and written
in the output headers. ``TGLCVER=0.8.0`` identifies the package and
``PROCVER=spoc-0.8.0`` identifies the processing convention; ``FITSHASH`` and
fit-setting headers describe the ePSF inputs/configuration. Check these when
comparing runs. SPOC timestamps retain the cutout's existing barycentric
correction. Precise timing work should verify the remaining reference-to-target
correction; this release does not apply a second full correction.

``aperture_flux_raw`` is the decontaminated 3 by 3 pixel sum and
``psf_flux_raw`` is the fitted target amplitude, both in electrons/second.
These signed columns precede catalog-based additive offsets and detrending;
they are extracted photometry, not untouched detector counts.
The headers record the transformations::

    aperture_flux = (aperture_flux_raw - LOC_BG) / PORTION
    psf_flux = psf_flux_raw - PSF_BG

Thus ``aperture_flux`` and ``psf_flux`` are not detrended but already contain
catalog-based adjustments. ``cal_aper_flux`` and ``cal_psf_flux`` additionally
have normalization and detrending applied. For custom flux calibration or
long-period variability, inspect the raw columns, offsets, aperture fraction
and flags, and account for residual contamination. Catalog estimates and
aperture fractions can introduce amplitude errors; a lower-scatter light curve
does not guarantee a more accurate transit depth. Header scatter statistics
are not independently propagated per-cadence measurement uncertainties.

Caches and reproducibility
--------------------------

Source caches are validated against extraction configuration and schema. ePSF caches use
``.npz`` files containing a fingerprint and fit metadata; old unvalidated
``.npy`` caches are not a valid source for the corrected fit. Version 0.8.0
also invalidates ePSF caches from the preparation processing convention. Changing
inputs or model settings can require refitting. Keep enough disk space for both old and
new outputs while comparing them, and retain raw science inputs for reproducible
validation. ``plot_epsf`` reads the grid size from new cache metadata.

Limitations and migration
-------------------------

Version 0.8.0 delivers corrected extraction and robustness improvements while
retaining power 1.4 and edge_compression 1e-4. Use the raw flux columns and
recorded offsets for custom normalization; lower scatter does not guarantee
an accurate fractional amplitude for every target. The default saturation
cutoff is approximate, and long bleed trails or strongly crowded fields may
require additional masks or tailored analysis.

Existing MAST light curves are not changed by installing a newer package.
Update scripts that assumed ``lc/*.fits`` to use the returned paths or the new
product directories, and allow source/ePSF caches to be rebuilt. The legacy
``plot_aperture``, ``choose_prior``, and archive batch recipes contain
personal-target assumptions and are not supported general-purpose workflows.
See ``docs/release_0_8_0.md`` for migration and release status, and
``docs/release_1_0_0.md`` for the future calibration/API roadmap.

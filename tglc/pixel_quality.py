"""Pixel masks for PSF fitting of calibrated images in electrons per second."""

import numpy as np
from scipy.ndimage import binary_dilation


def build_pixel_mask(flux, base_mask=None, saturation_limit=80000.0, saturation_dilation=1):
    """Return pixels unsuitable for a linear PSF fit.

    ``saturation_limit`` is a rate in e-/s, independent of FFI integration time.
    The default is a conservative 80% of approximately 200,000 electrons per
    two-second detector integration; it is configurable pending detector-specific
    validation. ``None`` disables saturation detection. Saturated pixels and
    their immediate neighbors are excluded; this is not a claim that photometry
    of saturated targets is recovered. Extended bleed trails may need a supplied
    ``base_mask``. Missing and zero-valued pixels are always excluded.
    """
    flux = np.asarray(flux)
    if flux.ndim != 2:
        raise ValueError("flux must be a two-dimensional image")
    mask = ~np.isfinite(flux) | (flux == 0)
    if base_mask is not None:
        base_mask = np.asarray(base_mask, dtype=bool)
        if base_mask.shape != flux.shape:
            raise ValueError("base_mask must have the same shape as flux")
        mask |= base_mask
    if int(saturation_dilation) != saturation_dilation or saturation_dilation < 0:
        raise ValueError("saturation_dilation must be a nonnegative integer")
    if saturation_limit is not None:
        if not np.isfinite(saturation_limit) or saturation_limit <= 0:
            raise ValueError("saturation_limit must be positive and finite, or None")
        saturated = np.isfinite(flux) & (flux >= saturation_limit)
        if saturation_dilation:
            saturated = binary_dilation(saturated, iterations=int(saturation_dilation))
        mask |= saturated
    return mask

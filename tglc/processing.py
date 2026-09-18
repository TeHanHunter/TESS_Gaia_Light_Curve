"""Scientific cache provenance and processing quality conventions."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

# Increment whenever modeling, masking, or extraction semantics change.
PROCESSING_VERSION = "spoc-0.8.0"
HIGH_BACKGROUND = 1
EPSF_FAILED = 2
SATURATED = 4
APERTURE_INVALID = 8
PSF_INVALID = 16
NO_GOOD_REFERENCE = 32


def background_outliers(values, sigma=3):
    """Finite MAD outliers; a constant background is not an outlier."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.zeros(values.shape, dtype=bool)
    if valid.any():
        deviation = np.abs(values - np.median(values[valid]))
        scale = 1.4826 * np.median(deviation[valid])
        result = valid & (deviation > 0) & (deviation >= sigma * scale)
    return result


def _hash_array(digest, array):
    """Hash values in a fixed byte order without copying the entire flux cube.

    FITS arrays are big-endian, while NumPy pickle restores native-endian
    arrays. Their scientific values must retain the same cache identity after
    the source-cache round trip. Preserve precision/dtype and normalize only
    byte order, one cadence at a time.
    """
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise ValueError("Scientific cache inputs must be numeric arrays")
    canonical_dtype = array.dtype.newbyteorder('<')
    digest.update(str((array.shape, canonical_dtype.str)).encode("ascii"))
    if array.ndim == 0:
        digest.update(np.asarray(array, dtype=canonical_dtype).tobytes())
    else:
        for row in array:
            digest.update(np.ascontiguousarray(row, dtype=canonical_dtype).tobytes())


def epsf_fingerprint(source, design_matrix, config):
    """Identify exact pixels, cadences, model, mask, and fit settings."""
    digest = hashlib.sha256()
    metadata = dict(config, processing_version=PROCESSING_VERSION,
                    sector=int(source.sector), camera=int(source.camera),
                    ccd=int(source.ccd), units="electron/s")
    digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for array in (source.time, source.cadence, source.flux,
                  np.ma.getmaskarray(source.mask), design_matrix):
        _hash_array(digest, array)
    if getattr(source, "pixel_mask", None) is not None:
        _hash_array(digest, source.pixel_mask)
    return digest.hexdigest()


def load_epsf_cache(path, fingerprint, shape):
    """Return a matching coefficient array; ignore legacy/corrupt caches."""
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["fingerprint"]) != fingerprint:
                return None
            parameters = cache["e_psf"]
            if parameters.shape != shape or parameters.dtype.kind not in 'fc':
                return None
            return parameters
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_epsf_cache(path, parameters, fingerprint, config):
    """Write via a temporary file so interrupted fits cannot look complete."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, e_psf=parameters, fingerprint=fingerprint,
                     metadata=json.dumps(dict(config, processing_version=PROCESSING_VERSION),
                                         sort_keys=True, allow_nan=False))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

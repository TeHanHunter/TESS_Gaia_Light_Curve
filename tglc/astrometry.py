"""Catalog positions used by the TGLC image model.

Gaia's ``pmra`` is d(alpha)/dt * cos(delta), in mas/year.  Keep the catalog
coordinates unchanged and store observation-epoch coordinates separately so
that repeated sector selection never propagates an already propagated position.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from erfa import ErfaWarning


SOURCE_SCHEMA_VERSION = 2
MODEL_CATALOG_HALO = 6.0  # pixels; includes the default 11 x 11 PSF support


def _numeric_column(catalog, name, default):
    if name not in catalog.colnames:
        return np.full(len(catalog), default, dtype=float)
    return np.asarray(np.ma.asarray(catalog[name], dtype=float).filled(np.nan))


def propagate_catalog_positions(catalog, time, default_ref_epoch=2016.0):
    """Return a copy with angular Gaia positions propagated to median finite TJD.

    ``ra`` and ``dec`` remain reference positions. ``ra_epoch``, ``dec_epoch``
    and ``position_epoch`` describe positions at the observation epoch; all
    epochs are Julian years. Missing proper-motion components are treated as
    zero, while the original catalog values remain available. Gaia DR3's J2016
    reference epoch is used only when a row does not supply ``ref_epoch``.
    Distances/radial velocities are not assumed; this is angular propagation.
    """
    times = np.asarray(time, dtype=float)
    finite_time = times[np.isfinite(times)]
    if not len(finite_time):
        raise ValueError("Cannot propagate catalog positions without a finite observation time")
    observed = Time(np.median(finite_time) + 2457000.0, format="jd", scale="tdb")
    result = catalog.copy(copy_data=True)
    ra = _numeric_column(catalog, "ra", np.nan)
    dec = _numeric_column(catalog, "dec", np.nan)
    epoch = _numeric_column(catalog, "ref_epoch", default_ref_epoch)
    epoch = np.where(np.isfinite(epoch), epoch, default_ref_epoch)
    pmra = _numeric_column(catalog, "pmra", np.nan)
    pmdec = _numeric_column(catalog, "pmdec", np.nan)
    valid = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
    propagated_ra = np.full(len(result), np.nan)
    propagated_dec = np.full(len(result), np.nan)
    if np.any(valid):
        coordinates = SkyCoord(
            ra=ra[valid] * u.deg,
            dec=dec[valid] * u.deg,
            pm_ra_cosdec=np.nan_to_num(pmra[valid], nan=0.0, posinf=0.0, neginf=0.0) * u.mas / u.yr,
            pm_dec=np.nan_to_num(pmdec[valid], nan=0.0, posinf=0.0, neginf=0.0) * u.mas / u.yr,
            obstime=Time(epoch[valid], format="jyear", scale="tdb"),
        )
        # ERFA supplies a unit distance for angular-only propagation. This is
        # expected when no parallax/radial velocity has been requested.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message='ERFA function "pmsafe".*distance overridden', category=ErfaWarning
            )
            propagated = coordinates.apply_space_motion(new_obstime=observed)
        propagated_ra[valid] = propagated.ra.deg
        propagated_dec[valid] = propagated.dec.deg
    result["ref_epoch"] = epoch
    result["ra_epoch"] = propagated_ra
    result["dec_epoch"] = propagated_dec
    result["position_epoch"] = np.full(len(result), observed.jyear)
    result["proper_motion_applied"] = valid & (np.isfinite(pmra) | np.isfinite(pmdec))
    return result


def in_model_catalog(x, y, shape, halo=MODEL_CATALOG_HALO):
    """Select finite centers whose PSF support can overlap a (height, width) image."""
    height, width = shape
    return (
        np.isfinite(x) & np.isfinite(y)
        & (x >= -halo) & (x <= width - 1 + halo)
        & (y >= -halo) & (y <= height - 1 + halo)
    )


def persistent_bad_pixels(flux):
    """Mask pixels that never have a finite sample, independent of brightness.

    Saturation and transient nonfinite samples belong to the per-cadence fit
    mask, not a percentile of the brightest source in the field.
    """
    flux = np.asarray(flux)
    if flux.ndim != 3 or flux.shape[0] == 0:
        raise ValueError("Expected a nonempty time series of 2D images")
    return ~np.any(np.isfinite(flux), axis=0)

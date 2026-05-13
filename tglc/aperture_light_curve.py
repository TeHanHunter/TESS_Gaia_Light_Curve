"""Aperture light curve class."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
from astropy.timeseries import TimeSeries
import h5py
import numpy as np

from tglc.utils.constants import TESSJD


@dataclass
class ApertureLightCurveMetadata:
    """Metadata for an aperture light curve."""

    tic_id: int
    """TIC ID of light curve target star."""

    orbit: int
    """Orbit containing light curve data."""

    sector: int
    """Sector containing light curve data."""

    camera: int
    """Camera containing target star."""

    ccd: int
    """CCD containing target star."""

    ccd_x: float
    """X coordinate on CCD of target star (projected from Gaia coordinates)."""

    ccd_y: float
    """Y coordinate on CCD of target star (projected from Gaia coordinates)."""

    sky_coord: SkyCoord
    """Sky coordinates of target star from the Gaia."""

    tess_magnitude: float
    """Brightness of target star in TESS magnitude."""

    exposure_time: u.Quantity["time"]  # noqa: F821
    """"Exposure time of light curve data points."""

    primary_aperture_local_background: u.Quantity[u.electron]
    """Local background level in primary aperture, subtracted to bring flux median to expect level.
    """

    small_aperture_local_background: u.Quantity[u.electron]
    """Local background level in small aperture, subtracted to bring flux median to expect level."""

    large_aperture_local_background: u.Quantity[u.electron]
    """Local background level in large aperture, subtracted to bring flux median to expect level."""


class ApertureLightCurve(TimeSeries):
    """Aperture light curve."""

    _unordered_required_columns = [
        "cadence",
        "quality_flag",
        "background_flux",
    ] + [
        f"{aperture_name}_aperture_{data_name}"
        for aperture_name in ["primary", "small", "large"]
        for data_name in ["magnitude", "centroid_x", "centroid_y"]
    ]
    _required_metadata = [field.name for field in fields(ApertureLightCurveMetadata)]

    def __init__(self, *args, meta: ApertureLightCurveMetadata | Any = None, **kwargs):
        if isinstance(meta, ApertureLightCurveMetadata):
            meta = asdict(meta)
        super().__init__(*args, meta=meta, **kwargs)

        missing_columns = [
            name for name in self._unordered_required_columns if name not in self.colnames
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for aperture light curve: {', '.join(missing_columns)}"
            )

        missing_metadata = [key for key in self._required_metadata if key not in self.meta]
        if missing_metadata:
            raise ValueError(
                f"Missing required metadata for aperture light curve: {', '.join(missing_metadata)}"
            )

    def write_hdf5(self, output_file: Path):
        with h5py.File(output_file, "w") as file:
            file.attrs["TIC ID"] = self.meta["tic_id"]
            file.attrs["Orbit"] = self.meta["orbit"]
            file.attrs["Sector"] = self.meta["sector"]
            file.attrs["Camera"] = self.meta["camera"]
            file.attrs["CCD"] = self.meta["ccd"]
            file.attrs["RA"] = self.meta["sky_coord"].ra.deg
            file.attrs["Dec"] = self.meta["sky_coord"].dec.deg
            file.attrs["BJDoffset"] = TESSJD.epoch_val.to(u.day)
            file.attrs["TessMag"] = self.meta["tess_magnitude"]

            lc_group = file.create_group("LightCurve")
            lc_group.create_dataset("BJD", data=self.time.tjd, dtype=np.float64)
            lc_group.create_dataset("Cadence", data=self["cadence"], dtype=np.int64)
            lc_group.create_dataset(
                "X",
                data=np.full_like(self.time, self.meta["ccd_x"], dtype=np.float64),
                dtype=np.float64,
            )
            lc_group.create_dataset(
                "Y",
                data=np.full_like(self.time, self.meta["ccd_y"], dtype=np.float64),
                dtype=np.float64,
            )
            lc_group.create_dataset("QualityFlag", data=self["quality_flag"], dtype=np.int64)

            background_group = lc_group.create_group("Background")
            background_group.create_dataset("Value", data=self["background_flux"], dtype=np.float64)
            background_group.create_dataset(
                "Error",
                data=np.full_like(self["background_flux"], mad_std(self["background_flux"])),
                dtype=np.float64,
            )

            photometry_group = lc_group.create_group("AperturePhotometry")
            for aperture_name, aperture_size in [("Primary", 3), ("Small", 1), ("Large", 5)]:
                aperture_group = photometry_group.create_group(f"{aperture_name}Aperture")
                aperture_group.attrs["name"] = f"TGLCAperture{aperture_name}"
                aperture_group.attrs["description"] = f"{aperture_size}x{aperture_size} square"
                aperture_group.attrs["localbackground"] = self.meta[
                    f"{aperture_name.lower()}_aperture_local_background"
                ]

                aperture_data = self[f"{aperture_name.lower()}_aperture_magnitude"]
                aperture_group.create_dataset("RawMagnitude", data=aperture_data, dtype=np.float64)
                aperture_group.create_dataset(
                    "RawMagnitudeError",
                    data=np.full_like(aperture_data, mad_std(aperture_data)),
                    dtype="f",
                )
                # TWIRL fork: emit linear flux alongside magnitude so flux-space
                # detrending downstream can use it (negative cadences preserved).
                # Existing magnitude consumers (QLP detrend + HLSP) are unchanged.
                flux_data = self[f"{aperture_name.lower()}_aperture_flux"]
                rf_ds = aperture_group.create_dataset(
                    "RawFlux", data=flux_data, dtype=np.float64
                )
                rf_ds.attrs["unit"] = "electron"
                aperture_group.create_dataset(
                    "RawFluxError",
                    data=np.full_like(flux_data, mad_std(flux_data)),
                    dtype="f",
                )
                aperture_group.create_dataset(
                    "X", data=self[f"{aperture_name.lower()}_aperture_centroid_x"], dtype="f"
                )
                aperture_group.create_dataset(
                    "Y", data=self[f"{aperture_name.lower()}_aperture_centroid_y"], dtype="f"
                )

"""Offline entry-point checks with archive and fitting boundaries replaced.

Compile the real entry functions to avoid Astroquery's import-time remote client
initialization. These tests cover orchestration, not remote services or science
extraction; the numerical and output tests cover those local stages separately.
"""

import ast
from contextlib import nullcontext
from glob import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import matplotlib
import numpy as np
import pytest
import requests

import tglc


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def api():
    source_path = Path(tglc.__file__).parent / "quick_lc.py"
    parsed = ast.parse(source_path.read_text())
    names = {
        "_output_directory", "_parse_tic_id", "tglc_lc", "get_tglc_lc",
        "plot_lc", "plot_epsf",
    }
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    namespace = dict(
        os=os, np=np, Path=Path, json=json, glob=glob, fits=fits, plt=plt,
        warnings=warnings, requests=requests, SkyCoord=SkyCoord, u=u, units=u,
        _dot_wait=lambda *args, **kwargs: nullcontext(),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)

    def query_object(target, *, catalog, **kwargs):
        if catalog.lower() == "gaia":
            return Table({"ra": [20.0], "dec": [30.0]})
        return Table({"ID": [123], "GAIA": [456]})

    catalog = SimpleNamespace(query_object=Mock(side_effect=query_object))
    gaia = SimpleNamespace(
        MAIN_GAIA_TABLE="gaiadr3.gaia_source",
        launch_job=Mock(return_value=SimpleNamespace(
            get_results=lambda: Table({"dr2_source_id": [456], "DESIGNATION": ["Gaia DR3 789"]})
        )),
    )
    sectors = Table({"sector": [1, 2]})

    def cutout(**kwargs):
        selected = kwargs["sector"]
        source = SimpleNamespace(
            sector=int(selected) if np.isscalar(selected) else 1,
            sector_table=sectors,
        )
        source.select_sector = lambda *, sector: setattr(source, "sector", int(sector))
        return source

    def extract(source, **kwargs):
        return [Path(kwargs["local_directory"]) / "lc" / kwargs["ffi"] / f"sector-{source.sector}.fits"]

    namespace.update(
        Catalogs=catalog, Gaia=gaia,
        Tesscut=SimpleNamespace(get_sectors=Mock(return_value=sectors)),
        ffi_cut=Mock(side_effect=cutout), epsf=Mock(side_effect=extract),
    )
    return namespace


@pytest.mark.parametrize("target", [123, np.int64(123), "123", "TIC 123", "tic123", " TIC 00123 "])
def test_tic_forms_resolve_the_same_identity_and_return_paths(api, tmp_path, target):
    directory = tmp_path / "result"
    result = api["tglc_lc"](target=target, local_directory=directory, sector=1, ffi="spoc")
    assert result == [directory / "lc" / "SPOC" / "sector-1.fits"]
    assert [call.args[0] for call in api["Catalogs"].query_object.call_args_list] == ["TIC 123", "TIC 123"]
    assert api["ffi_cut"].call_args.kwargs["target"] == "TIC 123"
    assert api["epsf"].call_args.kwargs["name"] == "Gaia DR3 789"
    assert (directory / "logs").is_dir()
    assert not (tmp_path / "resultlogs").exists()


@pytest.mark.parametrize("selection, expected", [
    ({"sector": np.int64(1)}, [1]),
    ({"first_sector_only": True}, [1]),
    ({"last_sector_only": True}, [2]),
    ({"sector": None}, [1, 2]),
    ({"sector": [1, 2]}, [1, 2]),
])
def test_all_sector_paths_forward_masking_and_collect_outputs(api, tmp_path, selection, expected):
    result = api["tglc_lc"](
        target=123, local_directory=str(tmp_path), get_all_lc=True,
        saturation_limit=75000.0, saturation_dilation=2, **selection,
    )
    assert [path.name for path in result] == [f"sector-{sector}.fits" for sector in expected]
    for call in api["epsf"].call_args_list:
        assert call.kwargs["saturation_limit"] == 75000.0
        assert call.kwargs["saturation_dilation"] == 2
        assert call.kwargs["name"] is None


@pytest.mark.parametrize("target", [0, -1, True, "TIC wrong", "TIC 0"])
def test_invalid_tic_fails_before_remote_queries(api, tmp_path, target):
    with pytest.raises((ValueError, TypeError), match="TIC"):
        api["tglc_lc"](target=target, local_directory=tmp_path, sector=1)
    api["Catalogs"].query_object.assert_not_called()


def test_no_observed_sectors_has_actionable_error(api, tmp_path):
    api["Tesscut"].get_sectors.return_value = Table({"sector": []})
    with pytest.raises(RuntimeError, match="No TESS sectors"):
        api["tglc_lc"](target=123, local_directory=tmp_path)
    api["ffi_cut"].assert_not_called()


@pytest.mark.parametrize("designations", [[], ["Gaia DR3 789", "Gaia DR3 790"]])
def test_missing_or_ambiguous_dr3_match_cannot_choose_an_arbitrary_target(api, tmp_path, designations):
    api["Gaia"].launch_job.return_value.get_results = lambda: Table({
        "dr2_source_id": [456] * len(designations), "DESIGNATION": designations,
    })
    with pytest.raises(RuntimeError, match="no unique Gaia DR3 match"):
        api["tglc_lc"](target=123, local_directory=tmp_path, sector=1)
    api["ffi_cut"].assert_not_called()


def test_batch_api_passes_supported_plot_product_and_returns_paths(api, tmp_path):
    outputs = [tmp_path / "output.fits"]
    api["tglc_lc"] = Mock(return_value=outputs)
    # Keep the actual plot function; no files means it should simply finish.
    assert api["get_tglc_lc"](tics=[123], directory=tmp_path, ffi="SPOC") == outputs
    assert (tmp_path / "TIC 123" / "plots" / "SPOC").is_dir()


def _write_light_curve(path, product):
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header.update(TICID=123, SECTOR=1, FFIVER=product)
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="time", array=[1000.0, 1001.0], format="D"),
        fits.Column(name="cal_aper_flux", array=[1.0, 0.99], format="E"),
        fits.Column(name="TESS_flags", array=[0, 0], format="J"),
        fits.Column(name="TGLC_flags", array=[0, 0], format="J"),
    ])
    fits.HDUList([primary, table]).writeto(path)


def test_plot_lc_discovers_product_subdirectories(api, tmp_path):
    _write_light_curve(tmp_path / "lc" / "SPOC" / "nested" / "spoc.fits", "SPOC")
    _write_light_curve(tmp_path / "lc" / "TICA" / "tica.fits", "TICA")
    api["plot_lc"](local_directory=tmp_path, ffi="spoc")
    assert len(list((tmp_path / "plots" / "SPOC").glob("*.png"))) == 1
    assert not (tmp_path / "plots" / "TICA").exists()
    api["plot_lc"](local_directory=tmp_path)
    assert len(list((tmp_path / "plots").glob("*.png"))) == 2


def test_plot_epsf_reads_metadata_instead_of_assuming_grid_size(api, tmp_path):
    directory = tmp_path / "epsf" / "SPOC"
    directory.mkdir(parents=True)
    np.savez(directory / "fit.npz", e_psf=np.ones((2, 16 ** 2 + 3)),
             metadata=json.dumps({"psf_size": 5, "factor": 3}))
    api["plot_epsf"](local_directory=tmp_path)
    assert (tmp_path / "plots" / "fit.png").is_file()

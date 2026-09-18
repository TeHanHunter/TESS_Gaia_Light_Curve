from importlib.metadata import metadata, version
from importlib.resources import files

from packaging.specifiers import SpecifierSet

import tglc


def test_runtime_and_distribution_versions_agree():
    assert tglc.__version__ == version("tglc")


def test_python_support_matches_required_astropy():
    supported = SpecifierSet(metadata("tglc")["Requires-Python"])
    assert all(version in supported for version in ("3.10", "3.11", "3.12"))
    assert all(version not in supported for version in ("3.8", "3.9", "3.13"))


def test_installed_package_contains_science_resources():
    package = files("tglc")
    assert package.joinpath("background_mask", "median_mask.fits").is_file()
    assert package.joinpath("ephemeris_data", "20241201_tess_ephem.csv").is_file()

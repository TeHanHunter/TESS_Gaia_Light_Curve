"""Lightweight tests for the sector-30 BEAM/TICA runner helpers.

Run with:
    python scripts/test_s30_beam_tica.py
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_s30_beam_tica import (
    ACTIVE_SIZE,
    FrameMeta,
    build_spoc_index,
    candidate_cadences_for_metadata,
    index_beam_files,
    index_tica_files,
    parse_beam_name,
    parse_tica_name,
    select_common_cadences,
    split_beam_mosaic_to_ccds,
)


def test_beam_mosaic_inverse_mapping():
    ccd1 = np.full((ACTIVE_SIZE, ACTIVE_SIZE), 1, dtype=np.int16)
    ccd2 = np.full((ACTIVE_SIZE, ACTIVE_SIZE), 2, dtype=np.int16)
    ccd3 = np.full((ACTIVE_SIZE, ACTIVE_SIZE), 3, dtype=np.int16)
    ccd4 = np.full((ACTIVE_SIZE, ACTIVE_SIZE), 4, dtype=np.int16)

    mosaic = np.block([[ccd3, ccd4], [np.flip(ccd2), np.flip(ccd1)]])
    split = split_beam_mosaic_to_ccds(mosaic)

    assert np.array_equal(split[1], ccd1)
    assert np.array_equal(split[2], ccd2)
    assert np.array_equal(split[3], ccd3)
    assert np.array_equal(split[4], ccd4)


def test_filename_parsers():
    tica = Path("hlsp_tica_tess_ffi_s30-00127933-cam3-ccd1_tess_v01_img.fits")
    beam = Path("hlsp_tica_tess_ffi_s30-00127933-cam3-ccdALL_tess_v01_img_likelihood.fits")
    assert parse_tica_name(tica) == (127933, 30, 3, 1)
    assert parse_beam_name(beam) == (127933, 30, 3)


def test_common_cadence_selection():
    tica_index = {127933: "a", 127934: "b", 127935: "c"}
    spoc_index = {
        127933: FrameMeta(127933, "s1", 1.0, 2.0, 0, 475.2),
        127935: FrameMeta(127935, "s3", 3.0, 4.0, 0, 475.2),
    }
    beam_index = {127933: "ba", 127934: "bb", 127935: "bc"}
    assert select_common_cadences("default_tica", tica_index, spoc_index, None, max_cadences=None) == [127933, 127935]
    assert select_common_cadences("beam_likelihood", tica_index, spoc_index, beam_index, max_cadences=1) == [127933]


def _write_spoc_header(path: Path, cadence: int):
    primary = fits.PrimaryHDU()
    primary.header["FFIINDEX"] = cadence
    image = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32))
    image.header["TSTART"] = 1700.0 + cadence / 100000
    image.header["TSTOP"] = image.header["TSTART"] + 0.0055
    image.header["DQUALITY"] = 0
    image.header["LIVETIME"] = 0.0055
    fits.HDUList([primary, image]).writeto(path)


def test_synthetic_cadence_index_exact_match():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tica_root = root / "tica"
        beam_root = root / "beam"
        spoc_root = root / "spoc" / "ffis" / "cam3"
        (tica_root / "cam3-ccd1").mkdir(parents=True)
        beam_root.mkdir()
        spoc_root.mkdir(parents=True)

        for cadence in (127933, 127934, 127935):
            (tica_root / "cam3-ccd1" / f"hlsp_tica_tess_ffi_s30-{cadence:08d}-cam3-ccd1_tess_v01_img.fits").touch()
            (beam_root / f"hlsp_tica_tess_ffi_s30-{cadence:08d}-cam3-ccdALL_tess_v01_img_likelihood.fits").touch()
            _write_spoc_header(spoc_root / f"tess{cadence}-s0030-3-1-0195-s_ffic.fits.gz", cadence)

        tica_index = index_tica_files(tica_root, ccd=1)
        beam_index = index_beam_files(beam_root)
        required = candidate_cadences_for_metadata("beam_likelihood", tica_index, beam_index, max_cadences=2)
        spoc_index = build_spoc_index(spoc_root.parent.parent, ccd=1, required_cadences=required)
        cadences = select_common_cadences("beam_likelihood", tica_index, spoc_index, beam_index, max_cadences=2)
        assert cadences == [127933, 127934]


def test_pdo_smoke_outputs_if_present():
    default_root = Path("/pdo/users/tehan/beam_tglc/s0030/default_tica")
    beam_root = Path("/pdo/users/tehan/beam_tglc/s0030/beam_likelihood")
    if not (default_root / "lc").exists() or not (beam_root / "lc").exists():
        return

    from scripts.compare_s30_beam_precision import index_lc_files, summarize_pair

    default_files = index_lc_files(default_root, ccds=[1, 2, 3, 4])
    beam_files = index_lc_files(beam_root, ccds=[1, 2, 3, 4])
    common = sorted(set(default_files) & set(beam_files))
    assert common

    for key in common[:100]:
        row = summarize_pair(key, default_files[key], beam_files[key])
        if row["n_common_good"] > 0:
            assert np.isfinite(row["default_cal_psf_mad_ppm"])
            assert np.isfinite(row["beam_cal_psf_mad_ppm"])
            return
    raise AssertionError("No paired smoke light curve had common good cadences")


def main():
    test_beam_mosaic_inverse_mapping()
    test_filename_parsers()
    test_common_cadence_selection()
    test_synthetic_cadence_index_exact_match()
    test_pdo_smoke_outputs_if_present()
    print("ok")


if __name__ == "__main__":
    main()

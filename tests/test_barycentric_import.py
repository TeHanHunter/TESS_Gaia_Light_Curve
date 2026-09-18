"""Regression coverage for NumPy typing imports (GitHub issue #18)."""

from pathlib import Path
import subprocess
import sys
import textwrap

import tglc


def test_barycentric_import_does_not_depend_on_pandas_loading_numpy_typing():
    script = textwrap.dedent("""
        import importlib
        from pathlib import Path
        import sys
        from typing import get_type_hints

        import numpy as np
        from astropy.coordinates import SkyCoord
        from astropy.time import Time, TimeDelta, TimeFromEpoch
        import astropy.units as u
        import pandas as pd

        # Preload the module's dependencies, then remove the incidental public
        # typing import provided by some pandas versions. Importing TGLC must
        # establish its own typing dependency, independent of that side effect.
        sys.modules.pop("numpy.typing", None)
        if hasattr(np, "typing"):
            delattr(np, "typing")
        assert not hasattr(np, "typing")
        assert "tglc.barycentric_correction" not in sys.modules

        sys.path.insert(0, sys.argv[1])
        module = importlib.import_module("tglc.barycentric_correction")

        # Import only after TGLC has loaded, so this cannot conceal the defect.
        from numpy.typing import ArrayLike
        annotations = get_type_hints(module.apply_barycentric_correction)
        assert annotations["tjd"] == ArrayLike
        assert annotations["return"] is np.ndarray
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script,
         str(Path(tglc.__file__).resolve().parent.parent)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

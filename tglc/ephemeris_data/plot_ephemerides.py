"""
Plot X,Y,Z coordinates from JPL horizons TESS ephemeris files to ensure the
transitions are smooth.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ephemerides(
    ephemeris_directory: Optional[Path] = None, output_file: Optional[Path] = None
) -> None:
    """
    Plot ephemerides from all files in given directory.

    Parameters
    ----------
    ephemeris_directory : Path, optional
        Directory containing files with ephemeris data to plot
    output_file : Path, optional
        File to save plot to. If omitted, calls plt.show()
    """
    if ephemeris_directory is None:
        ephemeris_directory = Path(__file__).parent
    ephemeris_files = sorted(list(ephemeris_directory.iterdir()))
    ephemerides = [
        pd.read_csv(ephemeris_file, comment="#")
        for ephemeris_file in ephemeris_files
        if ephemeris_file.suffix == ".csv"
    ]

    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(9, 6), layout="constrained")

    for ephemeris, path in zip(ephemerides, ephemeris_files):
        ax_x.plot(ephemeris["JDTDB"], ephemeris["X"], label=path.stem, alpha=0.8)
        ax_y.plot(ephemeris["JDTDB"], ephemeris["Y"], label=path.stem, alpha=0.8)
        ax_z.plot(ephemeris["JDTDB"], ephemeris["Z"], label=path.stem, alpha=0.8)

    ax_x.set(xlabel="JD TDB", ylabel="TESS X coordinate [AU]")
    ax_y.set(xlabel="JD TDB", ylabel="TESS Y coordinate [AU]")
    ax_z.set(xlabel="JD TDB", ylabel="TESS Z coordinate [AU]")
    fig.suptitle("JPL TESS Ephemerides for QLP")

    if output_file is None:
        plt.show()
    else:
        fig.savefig(output_file)


if __name__ == "__main__":
    plot_ephemerides(Path(__file__).parent)

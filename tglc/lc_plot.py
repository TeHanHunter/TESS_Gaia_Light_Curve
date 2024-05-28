import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten
from scipy import ndimage
from astropy.io import ascii
from astropy.io import fits
from tqdm import trange
from matplotlib.patches import ConnectionPatch
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut
from tglc.quick_lc import tglc_lc
import matplotlib.patheffects as pe
from astropy.table import Table
from scipy.optimize import curve_fit


def timebin(time, meas, meas_err, binsize=1800 / 3600 / 24):
    # Sort data by time
    ind_order = np.argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]

    time_out = []
    meas_out = []
    meas_err_out = []
    ct = 0

    while ct < len(time):
        # Find indices of data points within the current bin
        ind = np.where((time >= time[ct]) & (time < time[ct] + binsize))[0]
        num = len(ind)

        if num == 0:
            ct += 1
            continue

        # Compute normalized weights based on measurement errors
        wt = (1. / meas_err[ind]) ** 2
        wt = wt / np.sum(wt)

        # Compute weighted averages for time, measurements, and measurement errors
        time_out.append(np.sum(wt * time[ind]))
        meas_out.append(np.sum(wt * meas[ind]))
        meas_err_out.append(1. / np.sqrt(np.sum(1. / (meas_err[ind]) ** 2)))

        # Move to the next bin
        ct += num

    return np.array(time_out), np.array(meas_out), np.array(meas_err_out)


def plot_2d_with_horizontal_arrow(folder='/Users/tehan/Documents/TGLC/TIC 60922830/lc', period=0.374273788):
    """
    Plots 2D data, removes all axes and labels, adds a single horizontal arrow at the bottom
    spanning across all subplots, and adds text labels for each subplot on the arrow.

    Parameters:
    - folder: string, path to the folder containing FITS files.
    """
    fig, axs = plt.subplots(1, 6, figsize=(15, 3), sharey=True)
    files = glob(f'{folder}/*.fits')
    sectors = []
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            sectors.append(hdul[0].header['sector'])
    files = np.array(files)[np.argsort(np.array(sectors))]
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            q = [a and b for a, b in
                 zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            time_out, meas_out, meas_err_out = timebin(hdul[1].data['time'][q], hdul[1].data['cal_aper_flux'][q],
                                                       np.array(
                                                           len(hdul[1].data['time'][q]) * [hdul[1].header['CAPE_ERR']]))
            axs[i].errorbar(time_out % period / period, meas_out, meas_err_out, fmt='o', color='black', ecolor='black',
                            elinewidth=0.5, capsize=0, ms=3)

            # Remove all axes and labels
            axs[i].xaxis.set_visible(False)
            axs[i].yaxis.set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].set_ylim(0.96, 1.04)
            axs[i].text(0.5, 0.97, f'{hdul[0].header["sector"]}', fontsize=10)
    # # Add a single horizontal arrow across all subplots
    # arrow_props = dict(arrowstyle="->", lw=2.5, color='black')
    # fig.text(0, -0.1, '', arrowprops=arrow_props, ha='center')
    #
    # # Add text labels on the arrow for each subplot
    # for i in range(6):
    #     fig.text((i + 0.5) / 6, -0.15, f'Subplot {i+1}', ha='center', va='center', fontsize=12)
    #
    plt.savefig(f'{folder}/6panels.svg', )
    plt.tight_layout()
    plt.show()


def sinusoid(x, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)


def fit_rv(file='/Users/tehan/Documents/SURFSUP/GJ_1111_rv_unbin.csv', period=0.459):
    data = Table.read(file)[:16]
    time_bin = np.mean(data['bjd'][:len(data) // 2 * 2].reshape(-1, 2), axis=1)
    rv_bin = np.mean(data['rv'][:len(data) // 2 * 2].reshape(-1, 2), axis=1)
    time_bin_pf = time_bin % period / period
    arg = np.argsort(time_bin_pf)
    time_bin_pf = time_bin_pf[arg]
    rv_bin = rv_bin[arg]
    popt, pcov = curve_fit(sinusoid, time_bin_pf, rv_bin, p0=[100, 1, 0.3])  # Initial guess: [amplitude, mean, stddev]

    # Plot the data and the fitted curve
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1, wspace=0.05),
                           figsize=(5, 5))
    ax[0].plot(time_bin_pf, rv_bin, '.')
    ax[0].plot(np.linspace(0, 1, 100), sinusoid(np.linspace(0, 1, 100), *popt), 'k', label='Fitted Gaussian')

    ax[1].plot(time_bin_pf, rv_bin-sinusoid(time_bin_pf, *popt), '.k')
    plt.show()


if __name__ == '__main__':
    # plot_2d_with_horizontal_arrow()
    fit_rv()

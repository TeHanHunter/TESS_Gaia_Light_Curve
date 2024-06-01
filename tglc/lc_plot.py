import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
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
from matplotlib import font_manager


def timebin(hdul, q, kind='cal_aper_flux', binsize=1800):
    r = int(1800 / binsize)
    print(f'The binsize is normalized to the multiples of 1800s, currently using {int(1800 / r)}s.')
    if hdul[0].header["SECTOR"] <= 26:
        t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // r * r].reshape(-1, r), axis=1)
        f = np.mean(hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // r * r].reshape(-1, r), axis=1)
        ferr = np.array(len(t) * [1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f)))])
    elif hdul[0].header["SECTOR"] <= 55:
        t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // (3 * r) * (3 * r)].reshape(-1, (3 * r)),
                    axis=1)
        f = np.mean(hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // (3 * r) * (3 * r)].reshape(-1, (3 * r)),
                    axis=1)
        ferr = np.array(len(t) * [1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f)))])
    else:
        t = np.mean(hdul[1].data['time'][q][:len(hdul[1].data['time'][q]) // (9 * r) * (9 * r)].reshape(-1, (9 * r)),
                    axis=1)
        f = np.mean(hdul[1].data[kind][q][:len(hdul[1].data[kind][q]) // (9 * r) * (9 * r)].reshape(-1, (9 * r)),
                    axis=1)
        ferr = np.array(len(t) * [1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f)))])
    print(len(t))
    return t, f, ferr


def bin_time_series(hdul, q, bin_size=1800 / 3600 / 24):
    # Extract time, flux, and error from HDUList
    time = hdul[1].data['time'][q]
    flux = hdul[1].data['cal_aper_flux'][q]
    flux_error = hdul[1].header['CAPE_ERR']

    # Calculate the bin edges
    min_time = np.min(time)
    max_time = np.max(time)
    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

    # Initialize lists to store binned data
    binned_time = []
    binned_flux = []
    binned_flux_err = []

    # Bin the data
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        in_bin = (time >= bin_start) & (time < bin_end)

        if np.any(in_bin):
            binned_time.append(np.nanmean(time[in_bin]))
            binned_flux.append(np.nanmean(flux[in_bin]))
            binned_flux_err.append(
                np.sqrt(np.nansum((flux_error ** 2) / np.nansum(in_bin))))  # Combined error for the bin
        else:
            binned_time.append(bin_start + bin_size / 2)
            binned_flux.append(np.nan)
            binned_flux_err.append(np.nan)

    return np.array(binned_time), np.array(binned_flux), np.array(binned_flux_err)


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
            # time_out, meas_out, meas_err_out = timebin(hdul, q, kind='cal_aper_flux', binsize=1800)
            time_out, meas_out, meas_err_out = bin_time_series(hdul, q)
            meas_err_out = np.array(len(time_out) * [1.4826 * np.nanmedian(np.abs(np.diff(meas_out)))])
            print(np.nanmedian(meas_err_out))
            axs[i].errorbar(time_out % period / period, meas_out, meas_err_out, fmt='o', color='black', ecolor='black',
                            elinewidth=0.2, capsize=0, ms=2, alpha=0.6)
            # print(np.diff(time_out[10:20]))
            # Remove all axes and labels
            axs[i].xaxis.set_visible(False)
            axs[i].yaxis.set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].set_ylim(0.96, 1.042)
            # axs[i].text(0.5, 0.97, f'{hdul[0].header["sector"]}', fontsize=10)
    # # Add a single horizontal arrow across all subplots
    # arrow_props = dict(arrowstyle="->", lw=2.5, color='black')
    # fig.text(0, -0.1, '', arrowprops=arrow_props, ha='center')
    #
    # # Add text labels on the arrow for each subplot
    # for i in range(6):
    #     fig.text((i + 0.5) / 6, -0.15, f'Subplot {i+1}', ha='center', va='center', fontsize=12)
    #
    plt.tight_layout()
    plt.savefig(f'{folder}/6panels.svg', format='svg', transparent=True)
    plt.show()


def sinusoid(x, amplitude, offset, phase):
    return amplitude * np.sin(2 * np.pi * 1 * x + phase) + offset


def fit_rv(file='/Users/tehan/Documents/SURFSUP/GJ_1111_rv_unbin.csv', period=0.459):
    data = Table.read(file)[:16]
    time_bin = np.mean(data['bjd'][:len(data) // 2 * 2].reshape(-1, 2), axis=1)
    rv_bin = np.mean(data['rv'][:len(data) // 2 * 2].reshape(-1, 2), axis=1)
    e_rv_bin = np.mean(data['e_rv'][:len(data) // 2 * 2].reshape(-1, 2), axis=1)
    time_bin_pf = (time_bin + 0.07) % period / period
    arg = np.argsort(time_bin_pf)
    time_bin_pf = time_bin_pf[arg]
    rv_bin = rv_bin[arg]
    popt, pcov = curve_fit(sinusoid, time_bin_pf, rv_bin, p0=[-100, 5, 0.1])  # Initial guess: [amplitude, mean, stddev]
    print(popt)
    # Define two colors in RGB format
    font_path = '/Users/tehan/Library/Fonts/CrimsonPro-Regular.ttf'  # Specify the correct path
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Crimson Pro'
    # Set Seaborn style with custom font
    sns.set(style="whitegrid",
            rc={"axes.edgecolor": "k", "grid.color": "none", "font.family": "serif", "font.serif": "Crimson Pro",
                "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 16,
                "ytick.labelsize": 16, "legend.fontsize": 12})
    # Plot the data and the fitted curve
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1, wspace=0.05),
                           figsize=(5,4))

    ax[0].errorbar(time_bin_pf, rv_bin, e_rv_bin, fmt='o', color='r', ecolor='r', )
    ax[0].plot(np.linspace(0, 1, 100), sinusoid(np.linspace(0, 1, 100), *popt), 'k')
    ax[0].set_ylabel('GJ 1111 RV (m/s)')
    ax[0].set_ylim(-130, 130)

    ax[1].errorbar(time_bin_pf, rv_bin - sinusoid(time_bin_pf, *popt), e_rv_bin, fmt='o', color='r', ecolor='r', )
    ax[1].hlines(0, xmin=0, xmax=1, color='black')  # Add horizontal line at y=0
    ax[1].set_ylabel('Residual')
    ax[1].set_xlabel('Phase')
    ax[1].set_ylim(-22, 22)
    # Remove the background, add axes edges, and ticks
    for axis in ax:
        axis.set_facecolor('none')
        axis.spines['top'].set_visible(True)
        axis.spines['right'].set_visible(True)
        axis.spines['bottom'].set_visible(True)
        axis.spines['left'].set_visible(True)
        axis.tick_params(axis='both', which='both', length=5)

    plt.savefig('/Users/tehan/Documents/SURFSUP/GJ1111_RV.svg', format='svg', transparent=True,bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # plot_2d_with_horizontal_arrow()
    fit_rv()

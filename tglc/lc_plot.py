import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import emcee
import corner

from jedi.inference import param
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
from astropy.table import Table, vstack
import pkg_resources
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd
import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs
from scipy.stats import bootstrap, ks_2samp, norm, gaussian_kde, skewnorm
from scipy.optimize import minimize
import matplotlib
from scipy.optimize import curve_fit  # Add this import
from scipy.odr import ODR, Model, RealData
from scipy.stats import multivariate_normal
from pr_main import Fit
from uncertainties import ufloat
from rapidfuzz import process, fuzz

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # Use Computer Modern (serif font)
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def read_parameter(file=None):
    with open(file, 'r') as file:
        lines = file.readlines()
    table = Table(names=['Parameter', 'Value', 'Upper Error', 'Lower Error'],
                  dtype=['S50', 'f8', 'f8', 'f8'])
    for i in range(1, len(lines)):
        try:
            table.add_row([lines[i].split(',')[0],
                           float(lines[i].split(',')[1].split('$')[0]),
                           float(lines[i].split('+')[1].split('}')[0]),
                           float(lines[i].split('{')[2].split('}')[0])])
        except IndexError:
            table.add_row([lines[i].split(',')[0],
                           float(lines[i].split(',')[1].split('$')[0]),
                           float(lines[i].split(',')[1].split('$')[2]),
                           float(lines[i].split(',')[1].split('$')[2])])
    return table


def fetch_contamrt(folder=None):
    files = glob(os.path.join(folder, '*/lc/*.fits'))
    tic_sec = []
    contamrt = []
    for i in range(len(files)):
        with fits.open(files[i]) as hdul:
            tic_sec.append(f'TIC_{hdul[0].header["TICID"]}_{hdul[0].header["SECTOR"]}')
            contamrt.append(hdul[0].header['CONTAMRT'])
    table = Table([tic_sec, contamrt], names=('tic_sec', 'contamrt'))
    table.write(f'{folder}contamination_ratio_2.dat', format='ascii', overwrite=True)
    return table


def combine_contamrt():
    contamrt_tglc = ascii.read('/Users/tehan/Documents/TGLC/contamination_ratio.dat')
    contamrt_spoc_g = ascii.read('/Users/tehan/Documents/TGLC/deviation_TGLC_spoc_crowdsap_g.dat')
    contamrt_spoc_ng = ascii.read('/Users/tehan/Documents/TGLC/deviation_TGLC_spoc_crowdsap_ng.dat')
    tic_sec = []
    delta_contamrt = []
    for i in range(len(contamrt_tglc)):
        if contamrt_tglc['tic_sec'][i] in contamrt_spoc_g['Star_sector']:
            idx = np.where(contamrt_spoc_g['Star_sector'] == contamrt_tglc['tic_sec'][i])[0][0]
            if contamrt_spoc_g['contamrt_spoc'][idx] != 100:
                tic_sec.append(contamrt_spoc_g['Star_sector'][idx])
                delta_contamrt.append(contamrt_tglc['contamrt'][i] - contamrt_spoc_g['contamrt_spoc'][idx])
        elif contamrt_tglc['tic_sec'][i] in contamrt_spoc_ng['Star_sector']:
            idx = np.where(contamrt_spoc_ng['Star_sector'] == contamrt_tglc['tic_sec'][i])[0][0]
            if contamrt_spoc_ng['contamrt_spoc'][idx] != 100:
                tic_sec.append(contamrt_spoc_ng['Star_sector'][idx])
                delta_contamrt.append(contamrt_tglc['contamrt'][i] - contamrt_spoc_ng['contamrt_spoc'][idx])
    print(len(tic_sec))
    contamrt_combined_table = Table([tic_sec, delta_contamrt], names=('tic_sec', 'contamrt'))
    contamrt_combined_table.write('/Users/tehan/Documents/TGLC/contamination_ratio_combined.dat', format='ascii',
                                  overwrite=True)


def figure_1_collect_result(folder='/home/tehan/Downloads/Data/', param='pl_ratror', r1=0.01, r2=0.4, cmap='Tmag',
                            pipeline='TGLC'):
    param_dict = {'pl_rade': 'r_pl__0', 'pl_ratror': 'ror__0'}
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2025.02.18_10.38.34.csv'))
    # t = ascii.read('/home/tehan/PycharmProjects/TESS_Gaia_Light_Curve/tglc/PSCompPars_2024.02.05_22.52.50.csv')
    tics = [int(s[4:]) for s in t['tic_id']]

    t_ = Table(
        names=['Star_sector', 'Tmag', 'rhat', 'p', f'{param}', f'{param}err1', f'{param}err2', 'value', 'err1', 'err2'],
        dtype=['S20', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
    missed_stars = 0
    for i in trange(len(tics)):
        file = glob(os.path.join(folder, f'*/Photometry/*/*{tics[i]}*.dat'))
        if len(file) == 0:
            missed_stars += 1
        elif len(file) >= 1:
            for j in range(len(file)):
                star = int(os.path.basename(file[j]).split('_')[2])
                sector = int(os.path.basename(file[j]).split('_')[5])
                if star == tics[i]:
                    table_posterior = read_parameter(file[j])
                    table_posterior_row = table_posterior[table_posterior['Parameter'] == param_dict[param]]
                    chain_summary = glob(os.path.join(os.path.dirname(file[j]), 'ChainSummary*.csv'))
                    table_chain = Table.read(chain_summary[0], format='csv')
                    table_chain_row = table_chain[table_chain['Parameter'] == param_dict[param][0:-3] + '[0]']

                    if param == 'pl_rade':
                        t_.add_row(
                            [f'TIC_{star}_{sector}', t['sy_tmag'][i], table_chain_row['r_hat'], t['pl_orbper'][i],
                             t[f'{param}'][i],
                             t[f'{param}err1'][i], t[f'{param}err2'][i], table_posterior_row['Value'][0],
                             table_posterior_row['Upper Error'][0], table_posterior_row['Lower Error'][0]])
                    elif param == 'pl_ratror':
                        ror = t['pl_rade'][i] / t['st_rad'][i] / 109.076
                        sigma_rade = (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2
                        sigma_st_rad = (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                        sigma_ror = ((sigma_rade / t['st_rad'][i] / 109.076) ** 2 +
                                     (t['pl_rade'][i] / t['st_rad'][i] ** 2 / 109.076 * sigma_st_rad) ** 2) ** 0.5
                        t_.add_row(
                            [f'TIC_{star}_{sector}', t['sy_tmag'][i], table_chain_row['r_hat'], t['pl_orbper'][i], ror,
                             sigma_ror, - sigma_ror, table_posterior_row['Value'][0],
                             table_posterior_row['Upper Error'][0], table_posterior_row['Lower Error'][0]])
    print(len(t_))
    t_.write(f'{folder}deviation_{pipeline}_2024_kepler.dat', format='ascii.csv')
    print('missing stars:', missed_stars)
    # colormap = cm.viridis
    # norm = plt.Normalize(t_[cmap].min(), t_[cmap].max())
    # scatter = plt.scatter(t_[f'{param}'], t_['value'], c=t_[cmap], cmap=colormap, facecolors='none', s=0)
    # fig, ax = plt.subplots(figsize=(12, 8))
    # for k in range(len(t_)):
    #     if t_['rhat'][k] < 1.05:
    #         ax.errorbar(t_[f'{param}'][k], t_['value'][k], xerr=t_[f'{param}err1'][k],
    #                     yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]], fmt='o', mec=colormap(norm(t_[cmap][k])),
    #                     mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=10, elinewidth=1, capsize=0.7, alpha=0.5,
    #                     zorder=2)
    #     # else:
    #     #     plt.errorbar(t_[f'{param}'][k], t_['value'][k], yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]],
    #     #                  fmt='o', mec='silver', mfc='none', ecolor='silver',
    #     #                  ms=10, elinewidth=1, capsize=5, alpha=0.8, zorder=1)
    # range_zoom = [0.07, 0.12]
    # axins = inset_axes(ax, width='35%', height='35%', loc='lower right', borderpad=2)
    # for k in range(len(t_)):
    #     if t_['rhat'][k] < 1.05:
    #         axins.errorbar(t_[f'{param}'][k], t_['value'][k], xerr=t_[f'{param}err1'][k],
    #                        yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]], fmt='o', mec=colormap(norm(t_[cmap][k])),
    #                        mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=5, elinewidth=1, capsize=0.7, alpha=0.5,
    #                        zorder=2)
    # axins.set_xlim(range_zoom)
    # axins.set_ylim(range_zoom)
    # axins.set_xscale('log')
    # axins.set_yscale('log')
    # axins.set_xticks([0.07, 0.08, 0.09, 0.1, 0.12])
    # axins.set_xticklabels(['0.07', '', '', '0.1', '0.12'])
    # axins.set_yticks([0.07, 0.08, 0.09, 0.1, 0.12])
    # axins.set_yticklabels(['0.07', '', '', '0.1', '0.12'])
    # axins.plot([0.01, 0.4], [0.01, 0.4], 'k', zorder=0)
    # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='dashed')
    # plt.colorbar(scatter, ax=ax, label='TESS magnitude')
    # ax.plot([0.01, 0.4], [0.01, 0.4], 'k', zorder=0)
    # ax.set_xlim(r1, r2)
    # ax.set_ylim(r1, r2)
    # ax.set_xlabel(r'Literature $R_p/R_*$')
    # ax.set_ylabel(rf'{pipeline}-only fit $R_p/R_*$')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.savefig(os.path.join(folder, f'{param}_diagonal_{pipeline}.png'), bbox_inches='tight', dpi=600)
    # plt.close()
    #
    # plt.figure(figsize=(5, 5))
    # # np.save(f'deviation_{pipeline}.npy', np.array(t_['value'] - t_[f'{param}']))
    # t_.write(f'deviation_{pipeline}_2024.dat', format='ascii.csv')
    # difference_qlp = ascii.read('deviation_QLP.dat')
    # difference_tglc = ascii.read('deviation_TGLC.dat')
    # plt.hist(difference_tglc, edgecolor='C0', histtype='step', linewidth=1.2, bins=np.arange(-0.1, 0.1, 0.005))
    # plt.xlabel(r'fit $R_p/R_*$ - Literature $R_p/R_*$')
    # plt.ylabel(r'Number of stars')
    # median_value = np.median(difference_tglc)
    # print(np.median(np.abs(difference_tglc)))
    # print(len(np.where(difference_tglc < 0)[0]) / len(difference_tglc))
    # percentage = 68
    # lower_bound = np.percentile(difference_tglc, (100 - percentage) / 2)
    # upper_bound = np.percentile(difference_tglc, 100 - (100 - percentage) / 2)
    # print(median_value, lower_bound, upper_bound)
    # # plt.vlines(lower_bound, ymin=0, ymax=250, color='C0', linestyle='dashed')
    # plt.vlines(median_value, ymin=0, ymax=275, color='C0')
    # # plt.vlines(np.mean(difference), ymin=0,ymax=225, color='r')
    # # plt.vlines(upper_bound, ymin=0, ymax=250, color='C0', linestyle='dashed')
    #
    # plt.savefig(os.path.join(folder, f'{param}_hist.png'), bbox_inches='tight', dpi=600)
    # plt.close()

    # plt.figure(figsize=(5, 5))
    # percent_err = (t_['err1'] - t_['err2']) / 2 / t_['value']
    # plt.scatter(t_['Tmag'], percent_err, c='k', s=1)
    # sort = np.argsort(np.array(t_['Tmag']))
    # plt.plot(np.array(t_['Tmag'][sort])[12:-11], np.convolve(percent_err[sort], np.ones(25)/25, mode='valid'))
    # plt.ylim(0,1)
    # plt.xlabel('Tmag')
    # plt.ylabel(r'Percent uncertainty on $R_p/R_*$')
    # plt.savefig(os.path.join(folder, f'{param}_error_{pipeline}.png'), bbox_inches='tight', dpi=600)


def figure_2_collect_result(folder='/home/tehan/Downloads/Data/', ):
    palette = sns.color_palette('colorblind')
    tglc_color = palette[3]
    qlp_color = palette[2]
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2024_kepler.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.01)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # if star_sector in d_qlp['Star_sector']:
        difference_tglc.add_row(d_tglc[i])
        # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    # difference_qlp.write(f'deviation_QLP_common.dat', format='ascii.csv')
    difference_tglc.write(f'{folder}deviation_TGLC_kepler_rhat_limited.dat', format='ascii.csv', overwrite=True)
    print(len(difference_tglc))
    # print(len(difference_qlp))
    # average 491 lcs
    print(np.mean(difference_tglc['pl_ratrorerr1'] / difference_tglc['pl_ratror']))
    # average 160 hosts
    print(np.mean(list(set(difference_tglc['pl_ratrorerr1'].tolist() / difference_tglc['pl_ratror']))))
    # difference = vstack([difference_tglc, difference_qlp])
    # difference['diff'] = (difference['value'] - difference['pl_ratror']) / difference['pl_ratror']
    # difference['Tmag_int'] = np.where(difference['Tmag'] < 12.5, r'$T<12.5$', r'$T>12.5$')
    # print(len(np.where(difference['Tmag'] < 12.5)[0]) / 2)
    # # An outlier of TGLC for <12.5 is making the plot looks clumpy. That single point is removed, but will not affect the statistics.
    # print(difference[np.where(difference['diff'] == np.max(difference['diff']))[0][0]])
    # difference.remove_row(np.where(difference['diff'] == np.max(difference['diff']))[0][0])
    # df = difference.to_pandas()
    # plt.figure(figsize=(6, 6))
    # sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
    #             'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
    #             'axes.facecolor': '0.95', 'grid.color': '0.8'})
    # # sns.violinplot(data=df, x='diff', y='pipeline', bw_adjust=1, palette="Set1")
    # # print(np.sort(difference['diff'][(difference['Tmag_int'] == '$T<12.5$') & (difference['Photometry'] == 'TGLC')]))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.8, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2,
    #                palette=[tglc_color, qlp_color])
    # plt.vlines(0, ymin=-0.5, ymax=1.5, color='k', ls='dashed')
    # plt.xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    # plt.ylabel('')
    # plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
    #            [r'$-60\%$', r'$-40\%$', r'$-20\%$', r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$'])
    # plt.yticks(rotation=90)
    # # plt.xlim(-0.05, 0.05)
    # plt.xlim(-0.75, 0.75)
    # # plt.ylim(-1,2)
    # plt.title('Exoplanet radius ratio fit')
    # plt.savefig(os.path.join(folder, f'ror_ratio_violin.pdf'), bbox_inches='tight', dpi=600)
    # plt.show()
    #
    # # TGLC data
    # rows = np.where(difference_tglc['Tmag'] > 12.5)[0]
    # difference_tglc_values = (difference_tglc['value'][rows] - difference_tglc['pl_ratror'][rows]) / \
    #                          difference_tglc['pl_ratror'][rows]
    # median_value_tglc = np.median(difference_tglc_values)
    # q1_tglc = np.percentile(difference_tglc_values, 25)
    # q3_tglc = np.percentile(difference_tglc_values, 75)
    # iqr_tglc = (median_value_tglc - q1_tglc, q3_tglc - median_value_tglc)
    # negative_ratio_tglc = len(np.where(difference_tglc_values < 0)[0]) / len(difference_tglc)
    #
    # print("TGLC Median:", median_value_tglc)
    # print("TGLC IQR:", f"{median_value_tglc} - {iqr_tglc[0]} + {iqr_tglc[1]}")
    # print("TGLC Negative Ratio:", negative_ratio_tglc)
    #
    # # QLP data
    # rows = np.where(difference_tglc['Tmag'] > 12.5)[0]
    # difference_qlp_values = (difference_qlp['value'][rows] - difference_qlp['pl_ratror'][rows]) / \
    #                         difference_qlp['pl_ratror'][rows]
    # median_value_qlp = np.median(difference_qlp_values)
    # q1_qlp = np.percentile(difference_qlp_values, 25)
    # q3_qlp = np.percentile(difference_qlp_values, 75)
    # iqr_qlp = (median_value_qlp - q1_qlp, q3_qlp - median_value_qlp)
    # negative_ratio_qlp = len(np.where(difference_qlp_values < 0)[0]) / len(difference_qlp)
    #
    # print("QLP Median:", median_value_qlp)
    # print("QLP IQR:", f"{median_value_qlp} - {iqr_qlp[0]} + {iqr_qlp[1]}")
    # print("QLP Negative Ratio:", negative_ratio_qlp)


def compute_weighted_mean(data, tmag_cutoff):
    # Filter rows based on Tmag cutoff
    if tmag_cutoff == 'dim':
        rows = np.where(data['Tmag'] > 12.5)[0]
    elif tmag_cutoff == 'bright':
        rows = np.where(data['Tmag'] < 12.5)[0]
    values = data['value'][rows]
    pl_ratror = data['pl_ratror'][rows]
    errors_value = (data['err1'][rows] - data['err2'][rows]) / 2
    errors_pl_ratror = (data['pl_ratrorerr1'][rows] - data['pl_ratrorerr2'][rows]) / 2

    # correct literature with 0 error
    for i in range(len(errors_pl_ratror)):
        if errors_pl_ratror[i] == 0:
            errors_pl_ratror[i] = errors_value[i]
            # print(errors_pl_ratror[i])
    # Compute the ratio and its propagated error
    difference_values = values - pl_ratror
    errors_ratio = np.sqrt(errors_value ** 2 + errors_pl_ratror ** 2)

    # Compute inverse variance weighted mean
    weights = 1 / (errors_ratio ** 2)
    weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    weighted_mean_error = np.sqrt(1 / np.sum(weights))
    return weighted_mean, weighted_mean_error


def figure_3(folder='/home/tehan/Downloads/Data/', ):
    palette = sns.color_palette('colorblind')
    tglc_color = palette[3]
    qlp_color = palette[2]
    difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = [156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
              445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 289661991, 464300749, 151483286, 335590096,
              17865622, 193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
              395393265, 310002617, 220076110, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137, 243641947,
              419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
              240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
              239816546, 361343239]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            if int(star_sector.split('_')[1]) in ground:
                difference_tglc.add_row(d_tglc[i])
                difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    difference_qlp.write(f'{folder}deviation_QLP_common.dat', format='ascii.csv', overwrite=True)
    difference_tglc.write(f'{folder}deviation_TGLC_common.dat', format='ascii.csv', overwrite=True)
    print(len(difference_tglc))
    print(len(difference_qlp))
    # plt.hist((difference_tglc['value'] - difference_tglc['pl_ratror']) / difference_tglc['pl_ratror'], bins=np.linspace(-0.2, 0.2, 51))
    # plt.show()
    # average 491 lcs
    print(np.mean(difference_tglc['pl_ratrorerr1'] / difference_tglc['pl_ratror']))
    # average 160 hosts
    print(np.mean(list(set(difference_tglc['pl_ratrorerr1'].tolist() / difference_tglc['pl_ratror']))))
    # plt.hist(difference_tglc['pl_ratror'], bins=20)
    # plt.show()
    # TGLC data (dim)
    weighted_mean_tglc_dim, weighted_mean_error_tglc_dim = compute_weighted_mean(difference_tglc, 'dim')
    print("TGLC Weighted Mean Dim:", weighted_mean_tglc_dim)
    print("TGLC Weighted Mean Error Dim:", weighted_mean_error_tglc_dim)
    # QLP data (dim)
    weighted_mean_qlp_dim, weighted_mean_error_qlp_dim = compute_weighted_mean(difference_qlp, 'dim')
    print("QLP Weighted Mean Dim:", weighted_mean_qlp_dim)
    print("QLP Weighted Mean Error Dim:", weighted_mean_error_qlp_dim)
    # TGLC data (bright)
    weighted_mean_tglc_bright, weighted_mean_error_tglc_bright = compute_weighted_mean(difference_tglc, 'bright')
    print("TGLC Weighted Mean Bright:", weighted_mean_tglc_bright)
    print("TGLC Weighted Mean Error Bright:", weighted_mean_error_tglc_bright)
    # QLP data (bright)
    weighted_mean_qlp_bright, weighted_mean_error_qlp_bright = compute_weighted_mean(difference_qlp, 'bright')
    print("QLP Weighted Mean Bright:", weighted_mean_qlp_bright)
    print("QLP Weighted Mean Error Bright:", weighted_mean_error_qlp_bright)

    difference = vstack([difference_tglc, difference_qlp])
    difference['diff'] = difference['value'] - difference['pl_ratror']
    difference['Tmag_int'] = np.where(difference['Tmag'] < 11.5, r'$T<11.5$', r'$T>11.5$')
    print(len(np.where(difference['Tmag'] < 11.5)[0]) / 2)
    # An outlier of TGLC for <12.5 is making the plot looks clumpy. That single point is removed, but will not affect the statistics.
    # print(difference[np.where(difference['diff'] == np.max(difference['diff']))[0][0]])
    difference.remove_row(np.where(difference['diff'] == np.max(difference['diff']))[0][0])
    df = difference.to_pandas()
    plt.figure(figsize=(6, 6))
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    # sns.violinplot(data=df, x='diff', y='pipeline', bw_adjust=1, palette="Set1")
    # print(np.sort(difference['diff'][(difference['Tmag_int'] == '$T<12.5$') & (difference['Photometry'] == 'TGLC')]))
    sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
                   gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    plt.scatter(weighted_mean_tglc_bright, -0.08, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=30,
                zorder=2)
    plt.scatter(weighted_mean_qlp_bright, 0.08, marker='^', color=qlp_color, edgecolors='k', linewidths=0.7, s=30,
                zorder=2)
    plt.scatter(weighted_mean_tglc_dim, 0.92, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=30,
                zorder=2)
    plt.scatter(weighted_mean_qlp_dim, 1.08, marker='^', color=qlp_color, edgecolors='k', linewidths=0.7, s=30,
                zorder=2)
    plt.vlines(0, ymin=-0.7, ymax=1.7, color='k', ls='dashed', lw=1, zorder=1)
    plt.xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    plt.ylabel('')
    plt.xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
               [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    plt.yticks(rotation=90)
    # plt.xlim(-0.05, 0.05)
    plt.xlim(-0.06, 0.06)
    # plt.ylim(-1,2)
    plt.title('Exoplanet radius ratio fit')
    plt.savefig(os.path.join(folder, f'ror_violin_ground.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def compute_weighted_mean_all(data):
    values = data['value']
    pl_ratror = data['pl_ratror']
    errors_value = (data['err1'] - data['err2']) / 2
    errors_pl_ratror = (data['pl_ratrorerr1'] - data['pl_ratrorerr2']) / 2
    # correct literature with 0 error
    for i in range(len(errors_pl_ratror)):
        if errors_pl_ratror[i] == 0:
            errors_pl_ratror[i] = errors_value[i]
            # print(errors_pl_ratror[i])
    # Compute the ratio and its propagated error
    difference_values = (values - pl_ratror) / values
    errors_ratio = np.sqrt(errors_value ** 2)
    # errors_ratio = np.ones(len(errors_ratio))
    # Compute inverse variance weighted mean
    weights = 1 / (errors_ratio ** 2)
    weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    weighted_mean_error = np.sqrt(1 / np.sum(weights))
    # print('###')
    # print(np.median(values))
    # if len(values) == 230:
    #     errors_ratio = errors_ratio[values > 0.0864199]
    #     difference_values = difference_values[values > 0.0864199]
    # elif len(values) == 440:
    #     errors_ratio = errors_ratio[values > 0.06579014999999999]
    #     difference_values = difference_values[values > 0.06579014999999999]
    # plt.hist(errors_ratio, bins=500, color='k', alpha=1, label='error')
    # plt.xlim(0,0.01)
    # plt.show()
    # print(np.sort(weights))
    return difference_values, errors_ratio, weighted_mean, weighted_mean_error


def compute_weighted_mean_bootstrap(data, output_table=False):
    difference_values, errors_ratio = compute_weighted_mean_all(data)[:2]

    # errors_ratio = np.sqrt(errors_value ** 2 + errors_pl_ratror ** 2)
    # errors_ratio = np.ones(len(errors_pl_ratror))
    # errors_ratio = errors_value

    # Compute inverse variance weighted mean
    def weighted_mean(values_, weights_):
        return np.sum(values_ * weights_) / np.sum(weights_)

    def weighted_median(values_, weights_):
        # Sort values and weights by values
        sorted_indices = np.argsort(values_)
        values_ = values_[sorted_indices]
        weights_ = weights_[sorted_indices]

        # Compute cumulative weights
        cumulative_weight = np.cumsum(weights_)
        cutoff = np.sum(weights_) / 2.0

        # Find the index where the cumulative weight crosses the median cutoff
        median_index = np.where(cumulative_weight >= cutoff)[0][0]
        return values_[median_index]

    weights = 1 / (errors_ratio ** 2)

    def weighted_mean_stat(values_):
        return weighted_mean(values_, weights)

    def weighted_median_stat(values_):
        return weighted_median(values_, weights)

    # plt.figure()
    # plt.plot(np.sort(weights), '.')
    # plt.show()
    # weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    # weighted_mean_error = np.sqrt(1 / np.sum(weights))
    res = bootstrap((difference_values,), weighted_median_stat, confidence_level=.95, n_resamples=10000,
                    method='percentile')
    iw_mean = weighted_median(difference_values, weights)
    ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    print(f"Inverse-Variance Weighted Median: {iw_mean}")
    print(f"95% Confidence Interval: ({ci_low}, {ci_high})")
    print(res.standard_error)
    if output_table:
        table = Table([difference_values, weights], names=('diff', 'weights'))
        return table, iw_mean, res.standard_error, res.standard_error
    else:
        return iw_mean, res.standard_error, res.standard_error


def figure_4(folder='/Users/tehan/Documents/TGLC/', ):
    palette = sns.color_palette('bright')
    tglc_color = 'C1'
    qlp_color = 'C0'
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'hspace': 0.1})
    # ground
    difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    print(len(d_tglc))
    difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = [156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
              445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 289661991, 464300749, 151483286, 335590096,
              17865622, 193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
              395393265, 310002617, 220076110, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137, 243641947,
              419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
              240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
              239816546, 361343239]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            if int(star_sector.split('_')[1]) in ground:
                difference_tglc.add_row(d_tglc[i])
                difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    # difference_tglc.write(f'deviation_TGLC_677.dat', format='ascii.csv')
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    # QLP data (dim)
    diff_qlp, errors_qlp, weighted_mean_qlp, weighted_mean_error_qlp = compute_weighted_mean_all(difference_qlp)
    iw_mean_qlp, ci_low_qlp, ci_high_qlp = compute_weighted_mean_bootstrap(difference_qlp)
    print(len(difference_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    ax[0].hist(diff_qlp, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
               color=qlp_color, alpha=0.6, edgecolor=None)

    ax[0].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=tglc_color, alpha=0.6, edgecolor=None)
    stat, p_value = ks_2samp(diff_qlp, diff_tglc)
    print(f"K-S Statistic: {stat}")
    print(f"P-value: {p_value}")
    ax[0].set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    ax[0].scatter(iw_mean_tglc, 2.6, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='TGLC')
    ax[0].errorbar(iw_mean_tglc, 1.6, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[0].scatter(iw_mean_qlp, 2.6, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='QLP')
    ax[0].errorbar(iw_mean_qlp, 1.6, xerr=[[ci_low_qlp], [ci_high_qlp]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )

    ax[0].vlines(0, ymin=0, ymax=52.5, color='k', ls='dashed', lw=1, zorder=3)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Error Weighted Counts')
    ax[0].legend(loc='upper right')
    # ax[0].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #            [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-1,2)
    # no-ground
    difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    no_ground = [428699140, 201248411, 172518755, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                 271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                 351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                 219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                 148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                 29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                 172370679, 116483514, 350153977, 37770169, 162802770, 212957629, 393831507, 207110080, 190496853,
                 404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                 394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                 151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 1003831, 83092282,
                 264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            if int(star_sector.split('_')[1]) in no_ground:
                difference_tglc.add_row(d_tglc[i])
                difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    # QLP data (dim)
    diff_qlp, errors_qlp, weighted_mean_qlp, weighted_mean_error_qlp = compute_weighted_mean_all(difference_qlp)
    iw_mean_qlp, ci_low_qlp, ci_high_qlp = compute_weighted_mean_bootstrap(difference_qlp)
    print(len(difference_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    ax[1].hist(diff_qlp, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
               color=qlp_color, alpha=0.6, edgecolor=None)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=tglc_color, alpha=0.6, edgecolor=None)
    ax[1].set_title(f'TESS-influenced radius ({len(difference_tglc)} light curves)')
    ax[1].scatter(iw_mean_tglc, 6.8, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='TGLC')
    ax[1].errorbar(iw_mean_tglc, 4, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].scatter(iw_mean_qlp, 6.8, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='QLP')
    ax[1].errorbar(iw_mean_qlp, 4, xerr=[[ci_low_qlp], [ci_high_qlp]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].vlines(0, ymin=0, ymax=145, color='k', ls='dashed', lw=1, zorder=3)
    ax[1].set_xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    ax[1].set_ylabel('Error Weighted Counts')
    ax[1].legend(loc='upper right')
    # ax[1].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #                  [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])

    plt.xlim(-0.3, 0.3)

    plt.savefig(os.path.join(folder, f'ror_ground_vs_no_ground.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def figure_radius_bias(folder='/Users/tehan/Documents/TGLC/'):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    g_color = palette[7]
    k_color = palette[0]
    ng_color = palette[3]
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7),
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})    # ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    periods = []

    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] == 1.0)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    contamrt_ground = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # try:
        #     if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
        if int(star_sector.split('_')[1]) in ground:
            difference_tglc.add_row(d_tglc[i])
            periods.append(d_tglc['p'][i])
            # contamrt_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
        # except IndexError:
        #     print(star_sector)
        # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    # difference_tglc.write(f'deviation_TGLC_677.dat', format='ascii.csv')
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_ground = diff_tglc
    difference_tglc_ground = difference_tglc
    # print(difference_tglc[np.argsort(diff_tglc)[0]])
    # print(np.sort(diff_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    # n1, bins1, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #                        color=g_color, alpha=0.8, edgecolor=None, zorder=2)
    # bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    # try:
    #     popt1, _ = curve_fit(gaussian, bin_centers1, n1, p0=[max(n1), 0, 0.1])
    #     x_fit1 = np.linspace(bin_centers1[0], bin_centers1[-1], 200)
    #     y_fit1 = gaussian(x_fit1, *popt1)
    #     ax.plot(x_fit1, y_fit1, color=g_color, linestyle='--', lw=1.5, zorder=3)
    #     print(f"Ground-based Gaussian Fit: Mean = {popt1[1]:.3f}, Sigma = {popt1[2]:.3f}")
    # except RuntimeError:
    #     print("Ground-based Gaussian fit failed")

    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=g_color, alpha=0.1, edgecolor=None, zorder=2)
    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            histtype='step', edgecolor=g_color, linewidth=2, zorder=3, alpha=0.95,
               label=r'TESS-free $f_p$' + f'\n({len(difference_tglc)} fits of 78 planets)')
    # ax.set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    # ax.scatter(iw_mean_tglc, 13, marker='v', color=g_color, edgecolors='k', linewidths=0.7, s=50,
    #            zorder=4, label=r'TESS-free $f_p$' + f'\n({len(difference_tglc)} fits of 79 planets)')
    # ax.errorbar(iw_mean_tglc, 10.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
    #             elinewidth=1, capsize=3, zorder=4, )
    # Vertical line at the mean
    ax1.axvline(iw_mean_tglc, color=g_color, linestyle='--', linewidth=2, zorder=4)

    # Shaded region for error (confidence interval)
    ax1.axvspan(iw_mean_tglc - ci_low_tglc, iw_mean_tglc + ci_high_tglc, color=g_color, alpha=0.5, zorder=3)
    # ax.scatter(iw_mean_qlp, 2.6, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 1.6, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    # ax.vlines(0, ymin=0, ymax=55, color='k', ls='dashed', lw=1, zorder=3)
    # ax.set_xlabel('')
    # ax.set_ylabel('Error Weighted Counts')
    # ax.legend(loc='upper right')
    # ax[0].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #            [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-1,2)
    # no-ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] == 1.0)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])
    contamrt_no_ground = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # try:
        #     if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
        if int(star_sector.split('_')[1]) in no_ground:
            difference_tglc.add_row(d_tglc[i])
            periods.append(d_tglc['p'][i])
            # contamrt_no_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
        # except IndexError:
        #     pass
        # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_no_ground = diff_tglc
    difference_tglc_no_ground = difference_tglc
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    # n2, bins2, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #                        color=ng_color, alpha=0.6, edgecolor=None)
    # bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    # try:
    #     popt2, _ = curve_fit(gaussian, bin_centers2, n2, p0=[max(n2), 0, 0.1])
    #     x_fit2 = np.linspace(bin_centers2[0], bin_centers2[-1], 200)
    #     y_fit2 = gaussian(x_fit2, *popt2)
    #     ax.plot(x_fit2, y_fit2, color=ng_color, linestyle='--', lw=1.5, zorder=3)
    #     print(f"No-ground Gaussian Fit: Mean = {popt2[1]:.3f}, Sigma = {popt2[2]:.3f}")
    # except RuntimeError:
    #     print("No-ground Gaussian fit failed")
    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=ng_color, alpha=0.1, edgecolor=None, zorder=1)
    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            histtype='step', edgecolor=ng_color, linewidth=2, zorder=3, alpha=0.9,
               label=r'TESS-dependent $f_p$ ' + f'\n({len(difference_tglc)} fits of 191 planets)')

    # ax.set_title(f'TESS-influenced radius ({len(difference_tglc)} light curves)')
    # ax.scatter(iw_mean_tglc, 10, marker='v', color=ng_color, edgecolors='k', linewidths=0.7, s=50,
    #            zorder=4, label=r'TESS-dependent $f_p$ ' + f'\n({len(difference_tglc)} fits of 216 planets)')
    # ax.errorbar(iw_mean_tglc, 7.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
    #             elinewidth=1, capsize=3, zorder=4, )
    # Vertical line at the mean
    ax1.axvline(iw_mean_tglc, color=ng_color, linestyle='--', linewidth=2, zorder=4)

    # Shaded region for error (confidence interval)
    ax1.axvspan(iw_mean_tglc - ci_low_tglc, iw_mean_tglc + ci_high_tglc, color=ng_color, alpha=0.5, zorder=3)
    difference_kepler = ascii.read(f'{folder}deviation_TGLC_2024_kepler.dat')
    print(len(difference_kepler))

    d_tglc = difference_kepler[np.where(difference_kepler['rhat'] < 1.01)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])

    contamrt_ground = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        difference_tglc.add_row(d_tglc[i])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_kelper = diff_tglc
    difference_tglc_kelper = difference_tglc
    print(np.sort(diff_tglc))
    print(difference_tglc[np.argsort(diff_tglc)])
    # n3, bins3, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #                        color=k_color, alpha=0.8, edgecolor=None, zorder=3)
    # bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
    # try:
    #     popt3, _ = curve_fit(gaussian, bin_centers3, n3, p0=[max(n3), 0, 0.1])
    #     x_fit3 = np.linspace(bin_centers3[0], bin_centers3[-1], 200)
    #     y_fit3 = gaussian(x_fit3, *popt3)
    #     ax.plot(x_fit3, y_fit3, color=k_color, linestyle='--', lw=1.5, zorder=3)
    #     print(f"Kepler Gaussian Fit: Mean = {popt3[1]:.3f}, Sigma = {popt3[2]:.3f}")
    # except RuntimeError:
    #     print("Kepler Gaussian fit failed")
    ax2.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=k_color, alpha=0.1, edgecolor=None, zorder=3)
    ax2.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            histtype='step', edgecolor=k_color, linewidth=2, zorder=5, alpha=0.9,
               label=r'Kepler $f_p$' + f'\n({len(difference_tglc)} fits of 31 planets)')

    # ax.set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    # ax.scatter(iw_mean_tglc, 4, marker='^', color=k_color, edgecolors='k', linewidths=0.7, s=50,
    #            zorder=4, label=r'Kepler $f_p$' + f'\n({len(difference_tglc)} fits of 31 planets)')
    # ax.errorbar(iw_mean_tglc, 6.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
    #             elinewidth=1, capsize=3, zorder=4, )
    # Vertical line at the mean
    ax2.axvline(iw_mean_tglc, color=k_color, linestyle='--', linewidth=2, zorder=4)

    # Shaded region for error (confidence interval)
    ax2.axvspan(iw_mean_tglc - ci_low_tglc, iw_mean_tglc + ci_high_tglc, color=k_color, alpha=0.5, zorder=3)

    # ax.scatter(iw_mean_qlp, 6.8, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 4, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    # ax1.vlines(0, ymin=0, ymax=175, color='k', ls='-', lw=1, zorder=3)
    # ax2.vlines(0, ymin=0, ymax=40, color='k', ls='-', lw=1, zorder=3)
    ax2.set_xlabel(r'$f_p \equiv (p_{\text{TGLC}} - p_{\text{lit}}) / p_{\text{TGLC}}$')
    ax1.set_ylabel('Error weighted counts')
    ax2.set_ylabel('Error weighted counts')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax2.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                  [f"{x * 100:.0f}%" for x in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]])
    ax1.text(-0.12, 1, "a", transform=ax1.transAxes, fontsize=12, color='k', fontweight='bold')
    ax2.text(-0.12, 1, "b", transform=ax2.transAxes, fontsize=12, color='k', fontweight='bold')
    ax1.set_ylim(0,150)
    ax2.set_ylim(0,35)

    plt.xlim(-0.3, 0.3)
    # plt.ylim(0,15)
    stat, p_value = ks_2samp(diff_tglc_ground, diff_tglc_no_ground)
    print(f"K-S Statistic: {stat}")
    print(f"P-value: {p_value}")
    # plt.title(r'Fractional difference in radius ratio $p$ (TGLC vs. literature)')
    plt.savefig(os.path.join(folder, f'ror_g_ng_kepler.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    # print(len(set(ground+no_ground)))
    # print(len(ground)+len(no_ground))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_kelper['Star_sector']]
    # # print(str() in tics)
    # print(set(ground) - set(tics))
    print(len(set(tics)))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_ground['Star_sector']]
    # # print(str() in tics)
    # print(set(ground) - set(tics))
    print(len(set(tics)))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_no_ground['Star_sector']]
    # # print(str(89020549) in tics)
    # print(set(no_ground) - set(tics))
    print(len(set(tics)))
    print(np.percentile(periods, [0,25,50,75,100]))
    return difference_tglc_ground, difference_tglc_no_ground, contamrt_ground, contamrt_no_ground


def figure_radius_bias_split(folder='/Users/tehan/Documents/TGLC/'):
    # Load and prepare data
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    plot_color = palette[3]
    # Load original datasets
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] == 1.0)]

    # Process ground and no_ground samples
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])
    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])
    # Create combined dataset
    combined_data = Table()
    for sample in ['ground', 'no_ground']:
        temp = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
        tic_list = ground if sample == 'ground' else no_ground

        for i in range(len(d_tglc)):
            star_sector = d_tglc['Star_sector'][i]
            if int(star_sector.split('_')[1]) in tic_list:
                temp.add_row(d_tglc[i])

        temp['sample'] = [sample] * len(temp)
        combined_data = vstack([combined_data, temp])

    # Add sector information
    combined_data['sector'] = [int(entry['Star_sector'].split('_')[2]) for entry in combined_data]

    # Set up plot
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # Parameters to analyze - now includes sector instead of eccentricity
    parameters = [
        ('magnitude', 'Tmag'),
        ('period', 'p'),
        ('sector', 'sector')
    ]

    # Analysis and plotting
    for row_idx, (param_name, col_name) in enumerate(parameters):
        param_values = combined_data[col_name]

        # Determine split criteria
        if param_name == 'sector':
            # Fixed sector split at 26
            lower_mask = param_values <= 26
            upper_mask = param_values > 26
            lower_label = "Sectors ≤26"
            upper_label = "Sectors >26"
        else:
            # Median-based split for magnitude/period
            median_val = np.median(param_values)
            lower_mask = param_values <= median_val
            upper_mask = param_values > median_val
            lower_label = f"{param_name.capitalize()} Lower Half"
            upper_label = f"{param_name.capitalize()} Upper Half"

        for col_idx, (mask, label) in enumerate(zip([lower_mask, upper_mask], [lower_label, upper_label])):
            ax = axes[row_idx, col_idx]
            subset = combined_data[mask]

            if len(subset) < 2:
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
                continue

            # Compute statistics
            diff, errors, _, _ = compute_weighted_mean_all(subset)
            iw_mean, ci_low, ci_high = compute_weighted_mean_bootstrap(subset)

            weights = (1 / errors ** 2) * len(diff) / np.sum(1 / errors ** 2)

            # Plot histograms with original style
            ax.hist(diff, bins=np.linspace(-0.5, 0.5, 41),
                    weights=weights,
                    color=plot_color,
                    alpha=0.1,
                    edgecolor=None,
                    zorder=2)

            ax.hist(diff, bins=np.linspace(-0.5, 0.5, 41),
                    weights=weights,
                    histtype='step',
                    edgecolor=plot_color,
                    linewidth=2,
                    zorder=3,
                    alpha=0.9,
                    label=label + f'\n(N={len(subset)})')

            # Add statistics
            ax.axvline(iw_mean, color=plot_color,
                       linestyle='--', linewidth=2, zorder=4)
            ax.axvspan(iw_mean - ci_low, iw_mean + ci_high,
                       color=plot_color, alpha=0.3, zorder=3)

            # Formatting
            ax.set_xlim(-0.3, 0.3)
            if row_idx == 0:
                ax.set_title(label)
            if row_idx == 2:
                ax.set_xlabel(r'$f_p \equiv (p_{\text{TGLC}} - p_{\text{lit}}) / p_{\text{TGLC}}$')
            if col_idx == 0:
                ax.set_ylabel('Weighted Counts')

            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f'{param_name.capitalize()} {"Upper" if col_idx else "Lower"} Half')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'ror_sector_split.png'), bbox_inches='tight', dpi=600)
    plt.show()

def figure_radius_bias_ecc(folder='/Users/tehan/Documents/TGLC/'):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    g_color = palette[7]
    k_color = palette[0]
    ng_color = palette[3]
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5), gridspec_kw={'hspace': 0.1})
    # ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')

    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.01)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    contamrt_ground = []
    ecc_g = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # try:
        #     if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
        if int(star_sector.split('_')[1]) in ground:
            ecc_g.append(float(t['pl_orbeccen'][np.where(np.array(tics) == int(star_sector.split('_')[1]))[0]].filled(0)))
            if ecc_g[-1] > 0.3:
                difference_tglc.add_row(d_tglc[i])

            # contamrt_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
    # difference_tglc.write(f'deviation_TGLC_677.dat', format='ascii.csv')
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_ground = diff_tglc
    difference_tglc_ground = difference_tglc
    # print(difference_tglc[np.argsort(diff_tglc)[0]])
    # print(np.sort(diff_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    # n1, bins1, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #                        color=g_color, alpha=0.8, edgecolor=None, zorder=2)
    # bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    # try:
    #     popt1, _ = curve_fit(gaussian, bin_centers1, n1, p0=[max(n1), 0, 0.1])
    #     x_fit1 = np.linspace(bin_centers1[0], bin_centers1[-1], 200)
    #     y_fit1 = gaussian(x_fit1, *popt1)
    #     ax.plot(x_fit1, y_fit1, color=g_color, linestyle='--', lw=1.5, zorder=3)
    #     print(f"Ground-based Gaussian Fit: Mean = {popt1[1]:.3f}, Sigma = {popt1[2]:.3f}")
    # except RuntimeError:
    #     print("Ground-based Gaussian fit failed")

    ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=g_color, alpha=0.7, edgecolor=None, zorder=2)
    # ax.set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    ax.scatter(iw_mean_tglc, 13, marker='v', color=g_color, edgecolors='k', linewidths=0.7, s=50,
               zorder=4, label=r'TESS-free $f_p$' + f'\n({len(difference_tglc)} fits of 79 planets)')
    ax.errorbar(iw_mean_tglc, 10.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                elinewidth=1, capsize=3, zorder=4, )
    # ax.scatter(iw_mean_qlp, 2.6, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 1.6, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    # ax.vlines(0, ymin=0, ymax=55, color='k', ls='dashed', lw=1, zorder=3)
    # ax.set_xlabel('')
    # ax.set_ylabel('Error Weighted Counts')
    # ax.legend(loc='upper right')
    # ax[0].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #            [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-1,2)
    # no-ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.01)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])
    # contamrt_no_ground = []
    ecc_ng = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # try:
        #     if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
        if int(star_sector.split('_')[1]) in no_ground:
            ecc_ng.append(
                float(t['pl_orbeccen'][np.where(np.array(tics) == int(star_sector.split('_')[1]))[0]].filled(0)))
            if ecc_ng[-1] > 0.3:
                difference_tglc.add_row(d_tglc[i])

    print(np.median(ecc_g + ecc_ng))
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_no_ground = diff_tglc
    difference_tglc_no_ground = difference_tglc
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    # n2, bins2, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #                        color=ng_color, alpha=0.6, edgecolor=None)
    # bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    # try:
    #     popt2, _ = curve_fit(gaussian, bin_centers2, n2, p0=[max(n2), 0, 0.1])
    #     x_fit2 = np.linspace(bin_centers2[0], bin_centers2[-1], 200)
    #     y_fit2 = gaussian(x_fit2, *popt2)
    #     ax.plot(x_fit2, y_fit2, color=ng_color, linestyle='--', lw=1.5, zorder=3)
    #     print(f"No-ground Gaussian Fit: Mean = {popt2[1]:.3f}, Sigma = {popt2[2]:.3f}")
    # except RuntimeError:
    #     print("No-ground Gaussian fit failed")
    ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=ng_color, alpha=0.8, edgecolor=None)
    # ax.set_title(f'TESS-influenced radius ({len(difference_tglc)} light curves)')
    ax.scatter(iw_mean_tglc, 10, marker='v', color=ng_color, edgecolors='k', linewidths=0.7, s=50,
               zorder=4, label=r'TESS-dependent $f_p$ ' + f'\n({len(difference_tglc)} fits of 216 planets)')
    ax.errorbar(iw_mean_tglc, 7.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                elinewidth=1, capsize=3, zorder=4, )

    # difference_kepler = ascii.read(f'{folder}deviation_TGLC_2024_kepler.dat')
    # print(len(difference_kepler))
    #
    # d_tglc = difference_kepler[np.where(difference_kepler['rhat'] < 1.01)]
    # d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # # print(len(d_tglc))
    # # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    # difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    #
    # contamrt_ground = []
    # for i in range(len(d_tglc)):
    #     star_sector = d_tglc['Star_sector'][i]
    #     difference_tglc.add_row(d_tglc[i])
    # diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    # iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    # diff_tglc_kelper = diff_tglc
    # difference_tglc_kelper = difference_tglc
    # print(np.sort(diff_tglc))
    # print(difference_tglc[np.argsort(diff_tglc)])
    # # n3, bins3, _ = ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    # #                        weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    # #                        color=k_color, alpha=0.8, edgecolor=None, zorder=3)
    # # bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
    # # try:
    # #     popt3, _ = curve_fit(gaussian, bin_centers3, n3, p0=[max(n3), 0, 0.1])
    # #     x_fit3 = np.linspace(bin_centers3[0], bin_centers3[-1], 200)
    # #     y_fit3 = gaussian(x_fit3, *popt3)
    # #     ax.plot(x_fit3, y_fit3, color=k_color, linestyle='--', lw=1.5, zorder=3)
    # #     print(f"Kepler Gaussian Fit: Mean = {popt3[1]:.3f}, Sigma = {popt3[2]:.3f}")
    # # except RuntimeError:
    # #     print("Kepler Gaussian fit failed")
    # ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
    #         weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #         color=k_color, alpha=0.7, edgecolor=None, zorder=3)
    # # ax.set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    # ax.scatter(iw_mean_tglc, 4, marker='^', color=k_color, edgecolors='k', linewidths=0.7, s=50,
    #            zorder=4, label=r'Kepler $f_p$' + f'\n({len(difference_tglc)} fits of 31 planets)')
    # ax.errorbar(iw_mean_tglc, 6.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
    #             elinewidth=1, capsize=3, zorder=4, )

    # ax.scatter(iw_mean_qlp, 6.8, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 4, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    ax.vlines(0, ymin=0, ymax=150, color='k', ls='dashed', lw=1, zorder=3)
    ax.set_xlabel(r'$f_p \equiv (p_{\text{TGLC}} - p_{\text{lit}}) / p_{\text{TGLC}}$')
    ax.set_ylabel('Error weighted counts')
    ax.legend(loc='upper left')
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                  [f"{x * 100:.0f}%" for x in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]])
    plt.xlim(-0.3, 0.3)
    # plt.ylim(0,15)
    stat, p_value = ks_2samp(diff_tglc_ground, diff_tglc_no_ground)
    print(f"K-S Statistic: {stat}")
    print(f"P-value: {p_value}")
    # plt.title(r'Fractional difference in radius ratio $p$ (TGLC vs. literature)')
    plt.savefig(os.path.join(folder, f'ror_g_ng_kepler_ecc.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    # print(len(set(ground+no_ground)))
    # print(len(ground)+len(no_ground))
    # tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_kelper['Star_sector']]
    # # print(str() in tics)
    # print(set(ground) - set(tics))
    print(len(set(tics)))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_ground['Star_sector']]
    # # print(str() in tics)
    # print(set(ground) - set(tics))
    print(len(set(tics)))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_no_ground['Star_sector']]
    # # print(str(89020549) in tics)
    # print(set(no_ground) - set(tics))
    print(len(set(tics)))
    return difference_tglc_ground, difference_tglc_no_ground


def figure_radius_bias_per_planet(folder='/Users/tehan/Documents/TGLC/'):
    palette = sns.color_palette('colorblind')
    g_color = palette[7]
    ng_color = palette[3]
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5), gridspec_kw={'hspace': 0.1})
    # ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    contamrt_ground = []
    for i in range(len(ground)):
        value = []
        err1 = []
        err2 = []
        row = []
        for j in range(len(d_tglc)):
            star_sector = d_tglc['Star_sector'][j]
            if int(star_sector.split('_')[1]) == ground[i]:
                value.append(d_tglc['value'][j])
                err1.append(d_tglc['err1'][j])
                err2.append(d_tglc['err2'][j])
                row = d_tglc[j]
        if type(row) == list:
            continue
        else:
            weights = 1 / ((np.array(err1) - np.array(err2)) / 2) ** 2
            weighted_mean_value = np.sum(value * weights) / np.sum(weights)
            weighted_err1 = np.sqrt(1 / np.sum(weights))

            row['value'] = weighted_mean_value
            row['err1'] = weighted_err1
            row['err2'] = - weighted_err1
            difference_tglc.add_row(row)
            if weighted_mean_value < row['pl_ratror']:
                print(ground[i])


    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_ground = diff_tglc
    difference_tglc_ground = difference_tglc
    # print(difference_tglc[np.argsort(diff_tglc)[0]])
    # print(np.sort(diff_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 51),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=g_color, alpha=0.8, edgecolor=None, zorder=2)
    # ax.set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    ax.scatter(iw_mean_tglc, 10, marker='v', color=g_color, edgecolors='k', linewidths=0.7, s=50,
               zorder=4, label=r'TESS-free $f_p$' + f'\n({len(difference_tglc)} fits of 84 planets)')
    ax.errorbar(iw_mean_tglc, 7, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                elinewidth=1, capsize=3, zorder=2, )
    # ax.scatter(iw_mean_qlp, 2.6, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 1.6, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    # ax.vlines(0, ymin=0, ymax=55, color='k', ls='dashed', lw=1, zorder=3)
    # ax.set_xlabel('')
    # ax.set_ylabel('Error Weighted Counts')
    # ax.legend(loc='upper right')
    # ax[0].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #            [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-1,2)
    # no-ground
    # difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])
    contamrt_no_ground = []
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        # try:
        #     if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
        if int(star_sector.split('_')[1]) in no_ground:
            difference_tglc.add_row(d_tglc[i])
            # contamrt_no_ground.append(contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]])
        # except IndexError:
        #     pass
        # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_no_ground = diff_tglc
    difference_tglc_no_ground = difference_tglc
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax.hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    print(np.sort(diff_tglc))
    ax.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 51),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            color=ng_color, alpha=0.6, edgecolor=None)
    # ax.set_title(f'TESS-influenced radius ({len(difference_tglc)} light curves)')
    ax.scatter(iw_mean_tglc, 10, marker='v', color=ng_color, edgecolors='k', linewidths=0.7, s=50,
               zorder=4, label=r'TESS-dependent $f_p$ ' + f'\n({len(difference_tglc)} fits of 235 planets)')
    ax.errorbar(iw_mean_tglc, 7, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                elinewidth=1, capsize=3, zorder=2, )
    # ax.scatter(iw_mean_qlp, 6.8, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax.errorbar(iw_mean_qlp, 4, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)
    ax.vlines(0, ymin=0, ymax=200, color='k', ls='dashed', lw=1, zorder=3)
    ax.set_xlabel(r'f_p \equiv (p_{\text{TGLC}} - p_{\text{lit}}) / p_{\text{TGLC}}$')
    ax.set_ylabel('Error Weighted Counts')
    ax.legend(loc='upper left')
    # ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02], )
    plt.xlim(-0.3, 0.3)
    stat, p_value = ks_2samp(diff_tglc_ground, diff_tglc_no_ground)
    print(f"K-S Statistic: {stat}")
    print(f"P-value: {p_value}")
    # plt.title(r'Fractional difference in radius ratio $p$ (TGLC vs. literature)')
    # plt.savefig(os.path.join(folder, f'ror_ground_vs_no_ground_TGLC.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    # print(len(set(ground+no_ground)))
    # print(len(ground)+len(no_ground))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_ground['Star_sector']]
    # # print(str() in tics)
    # print(set(ground) - set(tics))
    print(len(set(tics)))
    tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_no_ground['Star_sector']]
    # # print(str(89020549) in tics)
    # print(set(no_ground) - set(tics))
    print(len(set(tics)))
    return difference_tglc_ground, difference_tglc_no_ground, contamrt_ground, contamrt_no_ground


def figure_4_tglc_contamrt_trend(folder='/Users/tehan/Documents/TGLC/', batch_size=150, recalculate=False):
    grid = np.arange(0., 0.6, 0.01)
    # batch = np.arange(0, 1, 0.1)
    difference_tglc_ground, difference_tglc_no_ground, contamrt_ground, contamrt_no_ground \
        = figure_radius_bias()
    arg_g = np.argsort(contamrt_ground)
    arg_ng = np.argsort(contamrt_no_ground)
    difference_tglc_ground = difference_tglc_ground[arg_g]
    difference_tglc_no_ground = difference_tglc_no_ground[arg_ng]
    contamrt_ground = np.array(contamrt_ground)[arg_g]
    contamrt_no_ground = np.array(contamrt_no_ground)[arg_ng]
    # print(contamrt_ground)
    # print(contamrt_no_ground)
    ## running median ###
    tables_g_rm = []
    contam_g_rm = []
    d_ror_g_rm = []
    d_ror_g_rm_err = []
    idx_g = np.linspace(0, len(grid) - 10, 10, dtype=int)
    print(idx_g)
    # for i in batch:
    #     idx = int(np.where((contamrt_ground > i))[0][0])
    #     if idx + batch_size < len(contamrt_ground):
    #         idx_g.append(idx)
    idx_g = np.array(idx_g)
    tables_ng_rm = []
    contam_ng_rm = []
    d_ror_ng_rm = []
    d_ror_ng_rm_err = []
    idx_ng = np.linspace(0, len(grid) - 10, 10, dtype=int)
    # for i in batch:
    #     idx = int(np.where((contamrt_no_ground > i))[0][0])
    #     if idx + batch_size < len(contamrt_no_ground):
    #         idx_ng.append(idx)
    idx_ng = np.array(idx_ng)
    if recalculate:
        for i in grid:
            idx = np.where((contamrt_ground >= i) & (contamrt_ground < i + 0.1))[0]
            contam_g_rm.append(np.median(contamrt_ground[idx]))
            t, d, e = compute_weighted_mean_bootstrap(difference_tglc_ground[idx], output_table=True)[0:3]
            tables_g_rm.append(t)
            d_ror_g_rm.append(d)
            d_ror_g_rm_err.append(e)
        for i in grid:
            idx = np.where((contamrt_no_ground >= i) & (contamrt_no_ground < i + 0.1))[0]
            contam_ng_rm.append(np.median(contamrt_no_ground[idx]))
            t, d, e = compute_weighted_mean_bootstrap(difference_tglc_no_ground[idx], output_table=True)[0:3]
            tables_ng_rm.append(t)
            d_ror_ng_rm.append(d)
            d_ror_ng_rm_err.append(e)

        data = {
            "tables_g_rm": tables_g_rm,
            "contam_g_rm": contam_g_rm,
            "d_ror_g_rm": d_ror_g_rm,
            "d_ror_g_rm_err": d_ror_g_rm_err,
            "tables_ng_rm": tables_ng_rm,
            "contam_ng_rm": contam_ng_rm,
            "d_ror_ng_rm": d_ror_ng_rm,
            "d_ror_ng_rm_err": d_ror_ng_rm_err,
        }
        with open(f"{folder}lists_data.pkl", "wb") as f:
            pickle.dump(data, f)

    with open(f"{folder}lists_data.pkl", "rb") as f:
        data = pickle.load(f)
    tables_g_rm = data['tables_g_rm']
    contam_g_rm = data["contam_g_rm"]
    d_ror_g_rm = data["d_ror_g_rm"]
    d_ror_g_rm_err = data["d_ror_g_rm_err"]
    tables_ng_rm = data['tables_ng_rm']
    contam_ng_rm = data["contam_ng_rm"]
    d_ror_ng_rm = data["d_ror_ng_rm"]
    d_ror_ng_rm_err = data["d_ror_ng_rm_err"]

    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    palette = sns.color_palette('colorblind')
    g_color = palette[2]
    ng_color = palette[3]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.05)
    ax[0].errorbar(d_ror_g_rm, contam_g_rm, xerr=d_ror_g_rm_err, alpha=0.1, linestyle='',
                   marker='o', zorder=1, ms=4, elinewidth=1., mfc=g_color, color='k')
    ax[0].errorbar(d_ror_ng_rm, contam_ng_rm, xerr=d_ror_ng_rm_err, alpha=0.1, linestyle='',
                   marker='o', zorder=1, ms=4, elinewidth=1., mfc=ng_color, color='k')
    ax[0].errorbar(np.array(d_ror_g_rm)[idx_g], np.array(contam_g_rm)[idx_g], xerr=np.array(d_ror_g_rm_err)[idx_g],
                   label='Ground-based-only', alpha=0.9, linestyle='',
                   marker='o', zorder=1000, ms=5, elinewidth=1.5, mfc=g_color, color='k')
    ax[0].errorbar(np.array(d_ror_ng_rm)[idx_ng], np.array(contam_ng_rm)[idx_ng],
                   xerr=np.array(d_ror_ng_rm_err)[idx_ng],
                   label='TESS-influenced', alpha=0.9, linestyle='',
                   marker='o', zorder=1000, ms=5, elinewidth=1.5, mfc=ng_color, color='k')
    x = np.linspace(-0.3, 0.4, 300)
    scale = 100
    for i in idx_g:
        kde = gaussian_kde(tables_g_rm[i]['diff'], weights=tables_g_rm[i]['weights'], bw_method=0.25)
        y = kde(x)
        ax[0].plot(x, y / scale + contam_g_rm[i], c=g_color, alpha=0.8, zorder=999)
        ax[0].fill_between(x, y / scale + contam_g_rm[i], np.zeros(len(x)) + contam_g_rm[i], color=g_color, alpha=0.1,
                           zorder=999)
    x = np.linspace(-0.3, 0.4, 300)
    for i in idx_ng:
        kde = gaussian_kde(tables_ng_rm[i]['diff'], weights=tables_ng_rm[i]['weights'], bw_method=0.25)
        y = kde(x)
        ax[0].plot(x, y / scale + contam_ng_rm[i], c=ng_color, alpha=0.8, zorder=999)
        ax[0].fill_between(x, y / scale + contam_ng_rm[i], np.zeros(len(x)) + contam_ng_rm[i], color=ng_color,
                           alpha=0.1,
                           zorder=999)

    ax[0].legend(loc='upper left')
    ax[0].vlines(0, ymin=0, ymax=0.8, color='k', ls='dashed', lw=1, zorder=1)
    ax[0].set_ylabel(r'Contamination Ratio Median')
    ax[0].set_ylim(0., 0.7)

    ### ground
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(
        difference_tglc_ground)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc_ground)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=g_color, alpha=0.5, edgecolor=None)
    # ax[1].set_title(f'Ground-based-only radius ({len(difference_tglc_ground)} light curves)')
    ax[1].scatter(iw_mean_tglc, 9.5, marker='v', color=g_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4)
    ax[1].errorbar(iw_mean_tglc, 5.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Error Weighted Counts')

    ### no ground
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(
        difference_tglc_no_ground)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc_no_ground)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=ng_color, alpha=0.5, edgecolor=None)
    # ax[1].set_title(f'Ground-based-only radius ({len(difference_tglc_ground)} light curves)')
    ax[1].scatter(iw_mean_tglc, 9.5, marker='v', color=ng_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4)
    ax[1].errorbar(iw_mean_tglc, 5.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].vlines(0, ymin=0, ymax=150, color='k', ls='dashed', lw=1, zorder=3)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Weighted Counts')
    # ax[1].legend(loc='upper right')
    ax[1].set_xlabel(r'Fractional $\Delta(R_{\text{p}}/R_*)$')

    plt.xlim(-0.2, 0.3)
    # plt.xticks([-0.01, -0.005, 0, 0.005, 0.010, 0.015, 0.02])
    plt.savefig(os.path.join(folder, f'ror_v_contamrt_combined.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def figure_4_tglc_contamrt_trend_pad(folder='/Users/tehan/Documents/TGLC/', batch_size=150, recalculate=False):
    pad_size = batch_size // 2
    batch = np.arange(0, 1, 0.1)
    difference_tglc_ground, difference_tglc_no_ground, contamrt_ground, contamrt_no_ground \
        = figure_radius_bias()
    arg_g = np.argsort(contamrt_ground)
    arg_ng = np.argsort(contamrt_no_ground)

    # load and reflect table
    difference_tglc_ground = difference_tglc_ground[arg_g]
    table_rows = list(difference_tglc_ground)
    padded_rows = (
            table_rows[:pad_size][::-1] +  # Reflect top rows
            table_rows +  # Original rows
            table_rows[-pad_size:][::-1]  # Reflect bottom rows
    )
    # Create a new padded Astropy table
    difference_tglc_ground = Table(rows=padded_rows, names=difference_tglc_ground.colnames)

    # load and reflect table
    difference_tglc_no_ground = difference_tglc_no_ground[arg_ng]
    table_rows = list(difference_tglc_no_ground)
    padded_rows = (
            table_rows[:pad_size][::-1] +  # Reflect top rows
            table_rows +  # Original rows
            table_rows[-pad_size:][::-1]  # Reflect bottom rows
    )
    # Create a new padded Astropy table
    difference_tglc_no_ground = Table(rows=padded_rows, names=difference_tglc_no_ground.colnames)

    # sort and reflect contamrt
    contamrt_ground = np.array(contamrt_ground)[arg_g]
    contamrt_ground = np.pad(contamrt_ground, pad_size, mode='reflect')
    contamrt_no_ground = np.array(contamrt_no_ground)[arg_ng]
    contamrt_no_ground = np.pad(contamrt_no_ground, pad_size, mode='reflect')

    # print(contamrt_ground)
    # print(contamrt_no_ground)
    ## running median ###

    tables_g_rm = []
    contam_g_rm = []
    d_ror_g_rm = []
    d_ror_g_rm_err = []
    idx_g = []
    for i in batch:
        idx = int(np.where((contamrt_ground[pad_size:len(contamrt_ground) - pad_size] > i))[0][0])
        # if idx + batch_size < len(contamrt_ground):
        idx_g.append(idx)
        # tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_ground['Star_sector'][idx:idx+batch_size]]
        # print(idx, len(set(tics)))
    idx_g = np.array(idx_g)
    tables_ng_rm = []
    contam_ng_rm = []
    d_ror_ng_rm = []
    d_ror_ng_rm_err = []
    idx_ng = []
    for i in batch:
        idx = int(np.where((contamrt_no_ground[pad_size:len(contamrt_no_ground) - pad_size] > i))[0][0])
        # if idx + batch_size < len(contamrt_no_ground):
        idx_ng.append(idx)
        # tics = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc_ground['Star_sector'][idx:idx+batch_size]]
        # print(idx, len(set(tics)))

    idx_ng = np.array(idx_ng)
    if recalculate:
        for i in range(len(contamrt_ground) - batch_size):
            contam_g_rm.append(np.median(contamrt_ground[i:i + batch_size]))
            t, d, e = compute_weighted_mean_bootstrap(difference_tglc_ground[i:i + batch_size], output_table=True)[0:3]
            tables_g_rm.append(t)
            d_ror_g_rm.append(d)
            d_ror_g_rm_err.append(e)
        for i in range(len(contamrt_no_ground) - batch_size):
            contam_ng_rm.append(np.median(contamrt_no_ground[i:i + batch_size]))
            t, d, e = compute_weighted_mean_bootstrap(difference_tglc_no_ground[i:i + batch_size], output_table=True)[
                      0:3]
            tables_ng_rm.append(t)
            d_ror_ng_rm.append(d)
            d_ror_ng_rm_err.append(e)

        data = {
            "tables_g_rm": tables_g_rm,
            "contam_g_rm": contam_g_rm,
            "d_ror_g_rm": d_ror_g_rm,
            "d_ror_g_rm_err": d_ror_g_rm_err,
            "tables_ng_rm": tables_ng_rm,
            "contam_ng_rm": contam_ng_rm,
            "d_ror_ng_rm": d_ror_ng_rm,
            "d_ror_ng_rm_err": d_ror_ng_rm_err,
        }
        with open(f"{folder}lists_data.pkl", "wb") as f:
            pickle.dump(data, f)

    with open(f"{folder}lists_data.pkl", "rb") as f:
        data = pickle.load(f)
    tables_g_rm = data['tables_g_rm']
    contam_g_rm = data["contam_g_rm"]
    d_ror_g_rm = data["d_ror_g_rm"]
    d_ror_g_rm_err = data["d_ror_g_rm_err"]
    tables_ng_rm = data['tables_ng_rm']
    contam_ng_rm = data["contam_ng_rm"]
    d_ror_ng_rm = data["d_ror_ng_rm"]
    d_ror_ng_rm_err = data["d_ror_ng_rm_err"]

    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    palette = sns.color_palette('colorblind')
    g_color = palette[2]
    ng_color = palette[3]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.05)
    ax[0].errorbar(d_ror_g_rm, contam_g_rm, xerr=d_ror_g_rm_err, alpha=0.1, linestyle='',
                   marker='o', zorder=1, ms=4, elinewidth=1., mfc=g_color, color='k')
    ax[0].errorbar(d_ror_ng_rm, contam_ng_rm, xerr=d_ror_ng_rm_err, alpha=0.1, linestyle='',
                   marker='o', zorder=1, ms=4, elinewidth=1., mfc=ng_color, color='k')
    ax[0].errorbar(np.array(d_ror_g_rm)[idx_g], np.array(contam_g_rm)[idx_g], xerr=np.array(d_ror_g_rm_err)[idx_g],
                   label='Ground-based-only', alpha=0.9, linestyle='',
                   marker='o', zorder=1000, ms=5, elinewidth=1.5, mfc=g_color, color='k')
    ax[0].errorbar(np.array(d_ror_ng_rm)[idx_ng], np.array(contam_ng_rm)[idx_ng],
                   xerr=np.array(d_ror_ng_rm_err)[idx_ng],
                   label='TESS-influenced', alpha=0.9, linestyle='',
                   marker='o', zorder=1000, ms=5, elinewidth=1.5, mfc=ng_color, color='k')
    x = np.linspace(-0.3, 0.4, 300)
    scale = 100
    for i in idx_g:
        kde = gaussian_kde(tables_g_rm[i]['diff'], weights=tables_g_rm[i]['weights'], bw_method=0.25)
        y = kde(x)
        ax[0].plot(x, y / scale + contam_g_rm[i], c=g_color, alpha=0.8, zorder=999)
        ax[0].fill_between(x, y / scale + contam_g_rm[i], np.zeros(len(x)) + contam_g_rm[i], color=g_color, alpha=0.1,
                           zorder=999)
    x = np.linspace(-0.3, 0.4, 300)
    for i in idx_ng:
        kde = gaussian_kde(tables_ng_rm[i]['diff'], weights=tables_ng_rm[i]['weights'], bw_method=0.25)
        y = kde(x)
        ax[0].plot(x, y / scale + contam_ng_rm[i], c=ng_color, alpha=0.8, zorder=999)
        ax[0].fill_between(x, y / scale + contam_ng_rm[i], np.zeros(len(x)) + contam_ng_rm[i], color=ng_color,
                           alpha=0.1,
                           zorder=999)

    ax[0].legend(loc='upper left')
    # ax[0].vlines(0, ymin=0, ymax=0.8, color='k', ls='dashed', lw=1, zorder=1)
    ax[0].set_ylabel(r'TGLC $R_{\text{p}}/R_*$')
    ax[0].set_ylim(0.0, 1)

    ### ground
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(
        difference_tglc_ground)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc_ground)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=g_color, alpha=0.5, edgecolor=None)
    # ax[1].set_title(f'Ground-based-only radius ({len(difference_tglc_ground)} light curves)')
    ax[1].scatter(iw_mean_tglc, 9.5, marker='v', color=g_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4)
    ax[1].errorbar(iw_mean_tglc, 5.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Error Weighted Counts')

    ### no ground
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(
        difference_tglc_no_ground)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc_no_ground)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=ng_color, alpha=0.5, edgecolor=None)
    # ax[1].set_title(f'Ground-based-only radius ({len(difference_tglc_ground)} light curves)')
    ax[1].scatter(iw_mean_tglc, 9.5, marker='v', color=ng_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4)
    ax[1].errorbar(iw_mean_tglc, 5.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    ax[1].vlines(0, ymin=0, ymax=150, color='k', ls='dashed', lw=1, zorder=3)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Weighted Counts')
    # ax[1].legend(loc='upper right')
    ax[1].set_xlabel(r'Fractional $\Delta(R_{\text{p}}/R_*)$')

    plt.xlim(-0.2, 0.3)
    # plt.xticks([-0.01, -0.005, 0, 0.005, 0.010, 0.015, 0.02])
    plt.savefig(os.path.join(folder, f'ror_v_contamrt_combined.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def figure_5(folder='/home/tehan/Downloads/Data/', ):
    contamrt = ascii.read('/Users/tehan/Documents/TGLC/contamination_ratio.dat')
    # plt.plot(np.sort(contamrt['contamrt'].tolist()), '.')
    # plt.yscale('log')
    # plt.show()
    palette = sns.color_palette('bright')
    tglc_color = 'C1'
    qlp_color = 'C0'
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'hspace': 0.1})
    # ground
    difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground = [156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
              445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 289661991, 464300749, 151483286, 335590096,
              17865622, 193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
              395393265, 310002617, 220076110, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137, 243641947,
              419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
              240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
              239816546, 361343239]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            if int(star_sector.split('_')[1]) in ground:
                difference_tglc.add_row(d_tglc[i])
                difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    print("TGLC Weighted Mean:", weighted_mean_tglc)
    print("TGLC Weighted Mean Error:", weighted_mean_error_tglc)
    # QLP data (dim)
    diff_qlp, errors_qlp, weighted_mean_qlp, weighted_mean_error_qlp = compute_weighted_mean_all(difference_qlp)
    print("QLP Weighted Mean:", weighted_mean_qlp)
    print("QLP Weighted Mean Error:", weighted_mean_error_qlp)
    print(len(difference_tglc))
    print(diff_tglc)
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    diff_tglc_contam = []
    diff_qlp_contam = []
    for i in range(len(difference_tglc)):
        i_tglc = np.where(np.array(contamrt['tic_sec'].tolist()) == difference_tglc['Star_sector'][i])[0][0]
        i_qlp = np.where(np.array(contamrt['tic_sec'].tolist()) == difference_qlp['Star_sector'][i])[0][0]
        diff_tglc_contam.append(contamrt['contamrt'].tolist()[i_tglc])
        diff_qlp_contam.append(contamrt['contamrt'].tolist()[i_qlp])
    ax[0].scatter(diff_tglc_contam, diff_tglc)
    ax[0].set_title('Ground-based-only radius')
    ax[0].legend(loc='upper right')
    # ax[0].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #            [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-1,2)
    # no-ground
    difference_qlp = ascii.read(f'{folder}deviation_QLP.dat')
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] < 1.1)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    no_ground = [428699140, 201248411, 172518755, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                 271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                 351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                 219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                 148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                 29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                 172370679, 116483514, 350153977, 37770169, 162802770, 212957629, 393831507, 207110080, 190496853,
                 404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                 394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                 151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 1003831, 83092282,
                 264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            if int(star_sector.split('_')[1]) in no_ground:
                difference_tglc.add_row(d_tglc[i])
                difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    print("TGLC Weighted Mean:", weighted_mean_tglc)
    print("TGLC Weighted Mean Error:", weighted_mean_error_tglc)
    # QLP data (dim)
    diff_qlp, errors_qlp, weighted_mean_qlp, weighted_mean_error_qlp = compute_weighted_mean_all(difference_qlp)
    print("QLP Weighted Mean:", weighted_mean_qlp)
    print("QLP Weighted Mean Error:", weighted_mean_error_qlp)
    print(len(difference_tglc))
    diff_tglc_contam = []
    diff_qlp_contam = []
    for i in range(len(difference_tglc)):
        try:
            i_tglc = np.where(np.array(contamrt['tic_sec'].tolist()) == difference_tglc['Star_sector'][i])[0][0]
            i_qlp = np.where(np.array(contamrt['tic_sec'].tolist()) == difference_qlp['Star_sector'][i])[0][0]
            diff_tglc_contam.append(contamrt['contamrt'].tolist()[i_tglc])
            diff_qlp_contam.append(contamrt['contamrt'].tolist()[i_qlp])
        except IndexError:
            diff_tglc_contam.append(0.)
            diff_qlp_contam.append(0.)
    ax[1].scatter(diff_tglc_contam, diff_tglc)
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax[1].hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)
    #
    # ax[1].hist(diff_tglc, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
    #            color=tglc_color, alpha=0.6, edgecolor=None)
    # ax[1].set_title('TESS and Ground-based radius')
    # ax[1].scatter(weighted_mean_tglc, 4, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=100, zorder=2,
    #               label='TGLC')
    # ax[1].scatter(weighted_mean_qlp, 4, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=100, zorder=2,
    #               label='QLP')
    # ax[1].vlines(0, ymin=0, ymax=145, color='k', ls='dashed', lw=1, zorder=3)
    ax[1].set_xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    ax[1].set_ylabel('Error Weighted Counts')
    ax[1].legend(loc='upper right')

    # ax[1].set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06],
    #                  [r'$-6\%$', r'$-4\%$', r'$-2\%$', r'$0\%$', r'$2\%$', r'$4\%$', r'$6\%$'])

    # plt.xlim(-0.03, 1)
    ax[0].set_ylim(-0.05, 0.05)
    ax[1].set_ylim(-0.05, 0.05)
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')
    plt.savefig(os.path.join(folder, f'ror_contamrt.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def figure_6(folder='/home/tehan/Downloads/Data/', param='pl_rade', r1=0.0001, r2=0.16, cmap='Tmag'):
    param_dict = {'pl_rade': 'r_pl__0', 'pl_ratror': 'ror__0'}
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
    # t = ascii.read('/home/tehan/PycharmProjects/TESS_Gaia_Light_Curve/tglc/PSCompPars_2024.02.05_22.52.50.csv')
    tics = [int(s[4:]) for s in t['tic_id']]

    t_ = Table(names=['Tmag', 'rhat', 'p', f'{param}', f'{param}err1', f'{param}err2', 'value', 'err1', 'err2'],
               dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
    missed_stars = 0
    for i in trange(len(tics)):
        file = glob(os.path.join(folder, f'*/Photometry/*/*{tics[i]}*.dat'))
        if len(file) == 0:
            missed_stars += 1
        elif len(file) >= 1:
            for j in range(len(file)):
                star = int(os.path.basename(file[j]).split('_')[2])
                if star == tics[i]:
                    table_posterior = read_parameter(file[j])
                    table_posterior_row = table_posterior[table_posterior['Parameter'] == param_dict[param]]
                    chain_summary = glob(os.path.join(os.path.dirname(file[j]), 'ChainSummary*.csv'))
                    table_chain = Table.read(chain_summary[0], format='csv')
                    table_chain_row = table_chain[table_chain['Parameter'] == param_dict[param][0:-3] + '[0]']

                    if param == 'pl_rade':
                        t_.add_row([t['sy_tmag'][i], table_chain_row['r_hat'], t['pl_orbper'][i], t[f'{param}'][i],
                                    t[f'{param}err1'][i], t[f'{param}err2'][i], table_posterior_row['Value'][0],
                                    table_posterior_row['Upper Error'][0], table_posterior_row['Lower Error'][0]])
                    elif param == 'pl_ratror':
                        ror = t['pl_rade'][i] / t['st_rad'][i] / 109.076
                        sigma_rade = (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2
                        sigma_st_rad = (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                        sigma_ror = ((sigma_rade / t['st_rad'][i] / 109.076) ** 2 +
                                     (t['pl_rade'][i] / t['st_rad'][i] ** 2 / 109.076 * sigma_st_rad) ** 2) ** 0.5
                        t_.add_row(
                            [t['sy_tmag'][i], table_chain_row['r_hat'], t['pl_orbper'][i], ror,
                             sigma_ror, - sigma_ror, table_posterior_row['Value'][0],
                             table_posterior_row['Upper Error'][0], table_posterior_row['Lower Error'][0]])
    print(len(t_))
    print('missing stars:', missed_stars)
    colormap = cm.viridis
    norm = plt.Normalize(t_[cmap].min(), t_[cmap].max())
    scatter = plt.scatter(t_[f'{param}'], t_['value'], c=t_[cmap], cmap=colormap, facecolors='none', s=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    for k in range(len(t_)):
        if t_['rhat'][k] < 1.05:
            ax.errorbar(t_[f'{param}'][k] ** 2, t_['value'][k] ** 2, xerr=2 * t_[f'{param}'][k] * t_[f'{param}err1'][k],
                        yerr=(t_['err1'][k] - t_['err2'][k]) * t_['value'][k], fmt='o', mec=colormap(norm(t_[cmap][k])),
                        mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=10, elinewidth=1, capsize=0.7, alpha=0.5,
                        zorder=2)
        # else:
        #     plt.errorbar(t_[f'{param}'][k], t_['value'][k], yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]],
        #                  fmt='o', mec='silver', mfc='none', ecolor='silver',
        #                  ms=10, elinewidth=1, capsize=5, alpha=0.8, zorder=1)
    range_zoom = [0.07 ** 2, 0.12 ** 2]
    axins = inset_axes(ax, width='35%', height='35%', loc='upper left', borderpad=2)
    for k in range(len(t_)):
        if t_['rhat'][k] < 1.05:
            ax.errorbar(t_[f'{param}'][k] ** 2, t_['value'][k] ** 2, xerr=2 * t_[f'{param}'][k] * t_[f'{param}err1'][k],
                        yerr=(t_['err1'][k] - t_['err2'][k]) * t_['value'][k], fmt='o', mec=colormap(norm(t_[cmap][k])),
                        mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=10, elinewidth=1, capsize=0.7, alpha=0.5,
                        zorder=2)
    axins.set_xlim(range_zoom)
    axins.set_ylim(range_zoom)
    axins.set_xscale('log')
    axins.set_yscale('log')
    # axins.set_xticks([0.07,0.08,0.09, 0.1, 0.12])
    # axins.set_xticklabels(['0.07','','', '0.1', '0.12'])
    # axins.set_yticks([0.07,0.08,0.09, 0.1, 0.12])
    # axins.set_yticklabels(['0.07','','', '0.1', '0.12'])
    axins.plot([0.0001, 40], [0.0001, 40], 'k', zorder=0)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='dashed')
    plt.colorbar(scatter, ax=ax, label='TESS magnitude')
    ax.plot([0.0001, 40], [0.0001, 40], 'k', zorder=0)
    ax.set_xlim(r1, r2)
    ax.set_ylim(r1, r2)
    ax.set_xlabel(r'Literature transit depth')
    ax.set_ylabel(r'TGLC-only fit transit depth')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(os.path.join(folder, f'{param}_diagonal_transit_depth.png'), bbox_inches='tight', dpi=600)


def figure_7(type='all'):
    palette = sns.color_palette('colorblind')
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '1'})
    data17 = np.loadtxt('/Users/tehan/Downloads/TIC-269820902/Photometry/TESS_Eleanorlarge_S17.csv', delimiter=',')
    data18 = np.loadtxt('/Users/tehan/Downloads/TIC-269820902/Photometry/TESS_Eleanorlarge_S18.csv', delimiter=',')
    data24 = np.loadtxt('/Users/tehan/Downloads/TIC-269820902/Photometry/TESS_Eleanorlarge_S24.csv', delimiter=',')
    t17 = data17[:, 0] - 2457000
    f17 = data17[:, 1]
    # _, trend = flatten(t17, f17, window_length=1, method='biweight', return_trend=True)
    # f17 = (f17 - trend) / np.nanmedian(f17) + 1
    t18 = data18[:, 0][70:] - 2457000
    f18 = data18[:, 1][70:]
    _, trend = flatten(t18, f18, window_length=1, method='biweight', return_trend=True)
    f18 = (f18 - trend) / np.nanmedian(f18) + 1
    t24 = data24[:, 0] - 2457000
    f24 = data24[:, 1]
    _, trend = flatten(t24, f24, window_length=1, method='biweight', return_trend=True)
    f24 = (f24 - trend) / np.nanmedian(f24) + 1

    if type == 'all':
        plt.figure(figsize=(10, 3))
        plt.plot(t17, f17, '.', ms=4, c=palette[0])
        plt.plot(t18, f18, '.', ms=4, c=palette[3])
        plt.plot(t24 - 138, f24, '.', ms=4, c=palette[2])
        plt.ylabel('Flux e-/s')
        plt.xticks([1777, 1803, 1831], ['Sector 17', 'Sector 18', 'Sector 24'])
        plt.savefig('/Users/tehan/Documents/TGLC/eb_eleanor.png', dpi=600)
        plt.show()
    elif type == 'phase-fold':
        plt.figure(figsize=(4, 3), constrained_layout=True)
        plt.plot(t17 % 1.01968 / 1.01968, f17, '.', ms=4, c=palette[0])
        # plt.plot(t18 % 1.01968/1.01968, f18, '.', ms=4, c=palette[3])
        # plt.plot(t24 % 1.01968/1.01968, f24, '.', ms=4, c=palette[2])
        plt.ylabel('Flux e-/s')
        plt.xlabel('Phase')
        # plt.xticks([1777],['Sector 17'])
        plt.savefig('/Users/tehan/Documents/TGLC/eb_eleanor_17_pf.png', dpi=600)
        plt.show()


def figure_8(type='all'):
    palette = sns.color_palette('colorblind')
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '1'})
    hdul17 = fits.open(
        '/Users/tehan/Documents/TGLC/TIC 269820902/lc/hlsp_tglc_tess_ffi_gaiaid-2015648943960251008-s0017-cam3-ccd2_tess_v1_llc.fits')
    hdul18 = fits.open(
        '/Users/tehan/Documents/TGLC/TIC 269820902/lc/hlsp_tglc_tess_ffi_gaiaid-2015648943960251008-s0018-cam3-ccd1_tess_v1_llc.fits')
    hdul24 = fits.open(
        '/Users/tehan/Documents/TGLC/TIC 269820902/lc/hlsp_tglc_tess_ffi_gaiaid-2015648943960251008-s0024-cam4-ccd3_tess_v1_llc.fits')
    q = [a and b for a, b in zip(list(hdul17[1].data['TESS_flags'] == 0), list(hdul17[1].data['TGLC_flags'] == 0))]
    t17 = hdul17[1].data['time'][q]
    f17 = hdul17[1].data['cal_aper_flux'][q]
    f17_aperture = hdul17[1].data['aperture_flux'][q]
    f17_psf = hdul17[1].data['psf_flux'][q]
    q = [a and b for a, b in zip(list(hdul18[1].data['TESS_flags'] == 0), list(hdul18[1].data['TGLC_flags'] == 0))]
    t18 = hdul18[1].data['time'][q][70:]
    f18 = hdul18[1].data['cal_aper_flux'][q][70:]
    f18_aperture = hdul18[1].data['aperture_flux'][q][70:]
    f18_psf = hdul18[1].data['psf_flux'][q][70:]
    q = [a and b for a, b in zip(list(hdul24[1].data['TESS_flags'] == 0), list(hdul24[1].data['TGLC_flags'] == 0))]
    t24 = hdul24[1].data['time'][q]
    f24 = hdul24[1].data['cal_aper_flux'][q]
    f24_aperture = hdul24[1].data['aperture_flux'][q]
    f24_psf = hdul24[1].data['psf_flux'][q]

    if type == 'all':
        plt.figure(figsize=(10, 3))
        plt.plot(t17, f17_aperture, '.', ms=4, c=palette[0])
        plt.plot(t18, f18_aperture, '.', ms=4, c=palette[3])
        plt.plot(t24 - 138, f24_aperture, '.', ms=4, c=palette[2])
        plt.ylabel('Flux e-/s')
        plt.xticks([1777, 1803, 1831], ['Sector 17', 'Sector 18', 'Sector 24'])
        plt.savefig('/Users/tehan/Documents/TGLC/eb_tglc_aperture_portion.png', dpi=600)
        plt.show()
    elif type == 'phase-fold':
        plt.figure(figsize=(4, 3), constrained_layout=True)
        plt.plot(t17 % 1.01968 / 1.01968, f17, '.', ms=4, c=palette[0])
        plt.plot(t18 % 1.01968 / 1.01968, f18, '.', ms=4, c=palette[3])
        plt.plot(t24 % 1.01968 / 1.01968, f24, '.', ms=4, c=palette[2])
        plt.ylabel('Flux e-/s')
        plt.xlabel('Phase')
        # plt.xticks([1777],['Sector 17'])
        plt.savefig('/Users/tehan/Documents/TGLC/eb_tglc_pf_portion.png', dpi=600)
        plt.show()


def mass_radius_to_density(mass, radius, mass_err, radius_err):
    """
    Calculate density and propagate error from mass and radius.

    Parameters:
    - mass: Mass in Earth masses or any consistent unit
    - radius: Radius in Earth radii or any consistent unit
    - mass_err: Error in mass
    - radius_err: Error in radius

    Returns:
    - density: Density in consistent units (e.g., Earth density if mass/radius in Earth units)
    - density_err: Propagated error in density
    """
    # Calculate density (assuming consistent units for mass and radius)
    density = mass / (radius ** 3)

    # Propagate error using partial derivatives
    density_err = density * np.sqrt((mass_err / mass) ** 2 + (3 * radius_err / radius) ** 2)
    if type(density_err) is np.ma.core.MaskedConstant:
        density_err = 0.
    return density, density_err


def zeng_2019_water_world(r):
    mass = (r / 1.24) ** (1 / 0.27)
    rho = mass / r ** 3
    return mass, rho


def rogers_2023_rocky_core(mass):
    # photoevaporation
    a = [1.3104, 0.2862, 0.1329, -0.0174, 0.0002]
    a1 = [1.2131, 0.2326, -0.0139, 0.0367, -0.0065]
    a2 = [1.5776, 0.7713, 0.5921, -0.2325, 0.0301]
    r = a[0] + a[1] * np.log(mass) + a[2] * np.log(mass) ** 2 + a[3] * np.log(mass) ** 3 + a[4] * np.log(mass) ** 4
    r1 = a1[0] + a1[1] * np.log(mass) + a1[2] * np.log(mass) ** 2 + a1[3] * np.log(mass) ** 3 + a1[4] * np.log(
        mass) ** 4
    r2 = a2[0] + a2[1] * np.log(mass) + a2[2] * np.log(mass) ** 2 + a2[3] * np.log(mass) ** 3 + a2[4] * np.log(
        mass) ** 4
    rho = mass / r ** 3
    rho1 = mass / r1 ** 3
    rho2 = mass / r2 ** 3
    return r, r1, r2, rho, rho1, rho2


def fortney_2007_earth_like():
    mass = np.array([0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 10, 31.6, 100, 316])
    r = np.array([0.24, 0.34, 0.50, 0.71, 1., 1.36, 1.8, 2.32, 2.84, 3.29])
    rho = mass / r ** 3
    return mass, rho


def owen_2017_earth_core(mass):
    rho = mass ** 0.25
    return rho


def figure_mr_mrho(folder='/Users/tehan/Documents/TGLC/', recalculate=False):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced.csv'))
    b = t['pl_imppar']
    ror = t['pl_rade'] / t['st_rad'] / 109
    # find grazing
    # for i in range(len(b)):
    #     if 1 - b[i] < ror[i] / 2:
    #         print(t['tic_id'][i])
    #         print(b[i])
    #         print(ror[i])
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    ng_color = palette[3]
    ng_corr_color = palette[2]
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax_ = plt.subplots(1, 2, sharex=True, figsize=(12, 5), gridspec_kw={'hspace': 0.01, 'wspace': 0.17})
    ax = [ax_[1], ax_[0]]
    for spine in ax[0].spines.values():
        spine.set_zorder(5)
    for spine in ax[1].spines.values():
        spine.set_zorder(5)
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])

    if recalculate:
        delta_R = 0.06011182562150113
        mass_g = []
        mass_g_err = []
        r_g = []
        r_g_err = []
        density_g = []
        density_g_err = []
        mass_ng = []
        mass_ng_err = []
        r_ng = []
        r_ng_corr = []
        r_ng_err = []
        r_ng_corr_err = []
        density_ng = []
        density_ng_err = []
        density_ng_corr = []
        density_ng_corr_err = []
        tic_g = []
        tic_ng = []
        for i, tic in enumerate(tics):
            if t['pl_bmassjlim'][i] == 0:
                density, density_err = mass_radius_to_density(t['pl_bmasse'][i], t['pl_rade'][i],
                                                              (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2,
                                                              (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                if t['pl_bmassprov'][i] == 'Mass':
                    # ## overall shift ###
                    # density_ng_corr.append(density * (1 - delta_R) ** 3)
                    ### individual fits ###
                    ror = []
                    ror_err = []
                    weights = []

                    for j in range(len(difference_tglc)):
                        if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                            value = float(difference_tglc['value'][j])
                            err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
                            if err > 0:
                                ror.append(value)
                                ror_err.append(err)
                                weights.append(1 / err ** 2)

                    ror = np.average(ror, weights=weights) if weights else np.nan
                    ror_err = np.sqrt(1 / np.sum(weights)) if weights else np.nan

                    # print(ror is np.nan)
                    density_corr, density_corr_err = mass_radius_to_density(t['pl_bmasse'][i],
                                                                            109.076 * ror * t['st_rad'][i],
                                                                            (t['pl_bmasseerr1'][i] -
                                                                             t['pl_bmasseerr2'][i]) / 2,
                                                                            109.076 * ror * (t['st_raderr1'][i] -
                                                                                             t['st_raderr2'][
                                                                                                 i]) / 2)
                    delta_Rstar = 109.076 * (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                    delta_ror = ror_err
                    if ror is not None and not np.isnan(ror):
                        if (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2 / t['pl_bmasse'][
                            i] < 0.25 and np.sqrt(
                                (109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2) / (
                                109.076 * ror * t['st_rad'][i]) < 0.20:
                            if tic in ground:
                                mass_g.append(t['pl_bmasse'][i])
                                mass_g_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2)
                                r_g.append(t['pl_rade'][i])
                                r_g_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                                density_g.append(density)
                                density_g_err.append(density_err)
                                tic_g.append(str(tic))
                            elif tic in no_ground:
                                mass_ng.append(t['pl_bmasse'][i])
                                mass_ng_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2)
                                r_ng.append(t['pl_rade'][i])
                                r_ng_corr.append(109.076 * ror * t['st_rad'][i])
                                density_ng.append(density)
                                density_ng_err.append(density_err)
                                density_ng_corr_err.append(density_corr_err)
                                tic_ng.append(str(tic))
                                density_ng_corr.append(density_corr)
                                r_ng_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                                r_ng_corr_err.append(
                                    np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2))
                                # if 109.076 * ror * t['st_rad'][i] < 4:
                                    # print(f"Teff = {t['st_teff'][i]}")
                # if density > 10:
                #     print(tic)
                # if tic in ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.9, marker='o', zorder=1, s=10, color=g_color)
                # elif tic in no_ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.5, marker='o', zorder=2, s=10, color='gray')
                #     ax.scatter(t['pl_bmasse'][i], density / 1.21, alpha=0.9, marker='o', zorder=3, s=15,
                #             color=ng_color)
                #     ax.plot([t['pl_bmasse'][i], t['pl_bmasse'][i]], [density, density / 1.21], color='gray', zorder=1,
                #             marker='', alpha=0.5,)
        data = {
            "mass_g": mass_g,
            "mass_g_err": mass_g_err,
            "r_g": r_g,
            "r_g_err": r_g_err,
            "density_g": density_g,
            "density_g_err": density_g_err,
            "mass_ng": mass_ng,
            "mass_ng_err": mass_ng_err,
            "r_ng": r_ng,
            "r_ng_corr": r_ng_corr,
            "r_ng_err": r_ng_err,
            "r_ng_corr_err": r_ng_corr_err,
            "density_ng": density_ng,
            "density_ng_err": density_ng_err,
            "density_ng_corr": density_ng_corr,
            "density_ng_corr_err": density_ng_corr_err,
            "tic_g": tic_g,
            "tic_ng": tic_ng
        }

        with open(f"{folder}mass_density.pkl", "wb") as f:
            pickle.dump(data, f)

    with open(f"{folder}mass_density.pkl", "rb") as f:
        data = pickle.load(f)
    mass_g = data["mass_g"]
    mass_g_err = data["mass_g_err"]
    r_g = data["r_g"]
    r_g_err = data["r_g_err"]
    density_g = data["density_g"]
    density_g_err = data["density_g_err"]
    mass_ng = data["mass_ng"]
    mass_ng_err = data["mass_ng_err"]
    r_ng = data["r_ng"]
    r_ng_corr = data["r_ng_corr"]
    r_ng_err = data["r_ng_err"]
    r_ng_corr_err = data["r_ng_corr_err"]
    density_ng = data["density_ng"]
    density_ng_err = data['density_ng_err']
    density_ng_corr = data["density_ng_corr"]
    density_ng_corr_err = data["density_ng_corr_err"]
    tic_g = data["tic_g"]
    tic_ng = data["tic_ng"]
    print(len(r_ng))
    marker_size = 30
    # density_weight = np.array(r_ng_err)[np.where(np.array(r_ng)<4)]
    # print(density_weight)
    ### mass-density ###
    ax[0].errorbar(mass_g, density_g, xerr=mass_g_err, yerr=density_g_err, fmt='none', alpha=0.5,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, color=palette[7])

    ax[0].scatter(mass_g, density_g, alpha=0.9, marker='D', zorder=2, s=marker_size, color=palette[7], facecolors=palette[7],
                  edgecolors='k', linewidths=0.5)

    ax[0].errorbar(mass_ng, density_ng, xerr=mass_ng_err, yerr=density_ng_err, fmt='none', alpha=0.5,
                   zorder=3, capsize=2.5, capthick=0.7, lw=0.7, color=ng_color)

    ax[0].scatter(mass_ng, density_ng, alpha=0.9, marker='o', zorder=3, s=marker_size, facecolors='none', edgecolors=ng_color)

    ax[0].errorbar(mass_ng, density_ng_corr, xerr=mass_ng_err, yerr=density_ng_corr_err, fmt='none',
                   color=ng_corr_color, lw=0.7, zorder=3, capsize=2.5, capthick=0.7, alpha=0.5)

    ax[0].scatter(mass_ng, density_ng_corr, color=ng_corr_color, alpha=0.9, s=marker_size, zorder=4)
    for i in range(len(density_ng)):
        if density_ng[i] < density_ng_corr[i]:
            ax[0].plot([mass_ng[i], mass_ng[i]], [density_ng[i], density_ng_corr[i]], color=ng_corr_color, zorder=2,
                       marker='', linewidth=3, alpha=0.8)
        elif density_ng[i] > density_ng_corr[i]:
            ax[0].plot([mass_ng[i], mass_ng[i]], [density_ng[i], density_ng_corr[i]], color=ng_color, zorder=2,
                       marker='', linewidth=3, alpha=0.8)
    # for j in range(len(mass_ng)):
    #     ax[0].text(mass_ng[j], density_ng[j], tic_ng[j], fontsize=6)

    ### mass-radius ###
    ax[1].errorbar(mass_g, r_g, xerr=mass_g_err, yerr=r_g_err, fmt='none', alpha=0.5,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, color=palette[7])
    ax[1].scatter(mass_g, r_g, alpha=0.9, marker='D', zorder=2, s=marker_size, color=palette[7], facecolors=palette[7],
                  edgecolors='k', linewidths=0.5)
    ax[1].errorbar(mass_ng, r_ng, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.5,
                   zorder=3, capsize=2.5, capthick=0.7, lw=0.7, color=ng_color)
    ax[1].scatter(mass_ng, r_ng, alpha=0.9, marker='o', zorder=3, s=marker_size, facecolors='none', edgecolors=ng_color)
    ax[1].errorbar(mass_ng, r_ng_corr, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.5,
                   color=ng_corr_color, lw=0.7, zorder=4, capsize=2.5, capthick=0.7)

    ax[1].scatter(mass_ng, r_ng_corr, color=ng_corr_color, alpha=0.9, s=marker_size, zorder=4)
    for i in range(len(r_ng)):
        if r_ng[i] < r_ng_corr[i]:
            ax[1].plot([mass_ng[i], mass_ng[i]], [r_ng[i], r_ng_corr[i]], color=ng_corr_color, zorder=2, marker='', linewidth=3, alpha=0.8, )
        elif r_ng[i] > r_ng_corr[i]:
            ax[1].plot([mass_ng[i], mass_ng[i]], [r_ng[i], r_ng_corr[i]], color=ng_color, zorder=2, marker='', linewidth=3, alpha=0.8, )

    # add manual legend
    ax[1].errorbar(0, 0, xerr=0, yerr=0, fmt='D', alpha=0.9, markerfacecolor=palette[7], markeredgecolor='k', markeredgewidth=0.5,
                   zorder=2, capsize=1.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size), color=palette[7], label='TESS-free')
    # ax[0].errorbar(mass_ng, density_ng_corr, xerr=mass_ng_err, yerr=density_ng_corr_err, fmt='o', alpha=0.3,
    #                color=ng_corr_color, lw=0.7, ms=np.sqrt(15), zorder=4, capsize=3, capthick=0.7)
    ax[1].errorbar(0, 0, xerr=0, yerr=0, fmt='o', alpha=0.9,
                   zorder=3, capsize=1.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size),
                   color=ng_color, markeredgecolor=ng_color, markerfacecolor='none',
                   label='TESS-dependent')
    ax[1].errorbar(0, 0, xerr=0, yerr=0, fmt='o', alpha=0.9,
                   color=ng_corr_color, lw=0.7, zorder=4, capsize=1.5, capthick=0.7, ms=np.sqrt(marker_size), label='TESS-dependent TGLC-fitted')
    ax[1].legend(loc=2, fontsize=10)

    ### water world ###
    r = np.linspace(1.24, 4, 100)
    mass, rho = zeng_2019_water_world(r)
    # ax[0].plot(mass, r, c=palette[0], zorder=4, label='Water world')
    ax[0].plot(mass, rho, c=palette[0], zorder=1, label='Water world', linewidth=2)
    ax[1].plot(mass, r, c=palette[0], zorder=1, label='Water world', linewidth=2)
    ax[0].text(mass[-1] - 22, rho[-1] - 0.14, 'Water world', color=palette[0], fontweight='bold', fontsize=9,
               ha='center', va='center', zorder=1, rotation=23)
    ax[1].text(mass[-1] - 10, r[-1] + 0.05, 'Water world', color=palette[0], fontweight='bold', fontsize=9,
               ha='center', va='center', zorder=1, rotation=25)
    ### rocky core + H/He atmos ###
    mass = np.linspace(1, 30, 100)
    r, r1, r2, rho, rho1, rho2 = rogers_2023_rocky_core(mass)
    # ax[0].plot(mass, r, c=sns.color_palette('muted')[5], zorder=4, label='Rocky core with H/He atmosphere')
    ax[0].plot(mass, rho, c=sns.color_palette('muted')[5], zorder=1, label='Rocky core with H/He atmosphere',
               linewidth=2)
    ax[0].plot(mass, rho1, c=sns.color_palette('muted')[5], zorder=1, ls='--')
    ax[0].plot(mass, rho2, c=sns.color_palette('muted')[5], zorder=1, ls='--')
    ax[1].plot(mass, r, c=sns.color_palette('muted')[5], zorder=1, label='Rocky core with H/He atmosphere', linewidth=2)
    ax[1].plot(mass, r1, c=sns.color_palette('muted')[5], zorder=1, ls='--')
    ax[1].plot(mass, r2, c=sns.color_palette('muted')[5], zorder=1, ls='--')
    ax[0].text(mass[0] + 0.95, rho[0] + 0.01, 'Rocky+atmosphere', color=sns.color_palette('muted')[5],
               fontweight='bold', fontsize=9, ha='center', va='center', zorder=1, rotation=11)
    ax[1].text(mass[0] + 0.9, r[0] + 0.35, 'Rocky+atmosphere', color=sns.color_palette('muted')[5],
               fontweight='bold', fontsize=9, ha='center', va='center', zorder=1, rotation=27)

    ### Earth-like ###
    mass = np.linspace(1, 30, 100)
    rho = owen_2017_earth_core(mass)
    r = (rho / mass) ** (-1 / 3)
    ax[0].plot(mass, rho, c='r', zorder=1, label='Earth-like', linewidth=2)
    ax[1].plot(mass, r, c='r', zorder=1, label='Earth-like', linewidth=2)
    ax[0].text(14, 1.86, 'Earth-like', color='r', fontweight='bold', fontsize=9, ha='center',
               va='center', zorder=1, rotation=45)
    ax[1].text(mass[-1] - 8, r[-1] - 0.35, 'Earth-like', color='r', fontweight='bold', fontsize=9, ha='center',
               va='center', zorder=1, rotation=23)
    ### M-R relation ###
    # mass = np.linspace(2, 30, 100)
    # ax[0].plot(mass, mass / (0.80811874404 * mass ** 0.59)**3, ls='dotted', c='k')
    # ax[0].scatter(2, 2 / (0.80811874404 * 2 ** 0.59)**3, marker=7)
    # ax[0].scatter(1.5838*2, 1.5838*2 / (0.80811874404 * (1.5838*2) ** 0.59)**3, marker=6)
    # mass, rho = fortney_2007_earth_like()
    # ax[0].plot(mass, rho, c='C5')

    ### solar planets ###
    # ax[0].scatter(0.0553,0.985,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(0.0553/1.1, 0.985, 'Mercury', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(0.815,0.951,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(0.815/1.05, 0.951, 'Venus', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    ax[0].scatter(1, 1, facecolors='r', edgecolors='r', zorder=4, marker='o', s=15)
    ax[0].text(1, 1 * 1.08, 'Earth', color='r', fontsize=8, ha='center', va='center', zorder=4, rotation=0)
    ax[1].scatter(1, 1, facecolors='r', edgecolors='r', zorder=4, marker='o', s=15)
    ax[1].text(1, 1 * 1.1, 'Earth', color='r', fontsize=8, ha='center', va='center', zorder=4, rotation=0)
    # ax[0].scatter(0.107,0.714,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(0.107, 0.714, 'Mars', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(95.2, 0.125, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(95.2 / 1.05, 0.125, 'Saturn', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[1].scatter(95.2, 9.45, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[1].text(95.2 / 1.05, 9.45, 'Saturn', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(14.5,0.230,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(14.5/1.05, 0.230, 'Uranus', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    ax[0].scatter(17.1, 0.297, facecolors='k', edgecolors='k', zorder=4, marker='o', s=15)
    ax[0].text(17.1 * 1.08, 0.297, 'Neptune', color='k', fontsize=8, ha='left', va='center', zorder=4, rotation=0)
    ax[1].scatter(17.1, 3.88, facecolors='k', edgecolors='k', zorder=4, marker='o', s=15)
    ax[1].text(17.1 * 1.08, 3.88, 'Neptune', color='k', fontsize=8, ha='left', va='center', zorder=4, rotation=0)
    #####
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylim(-.1, 2)
    ax[1].set_ylim(0.9, 10)
    ax[0].set_xlim(0.8, 100)
    ax[1].set_xlim(0.8, 100)
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks([1, 3, 10, 30, 100])

    ax[0].set_yticks([0, 0.5, 1, 1.5, 2])
    ax[0].set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax[1].set_yticks([1, 2, 3, 5, 10])
    ax[1].set_yticklabels(['1', '2', '3', '5', '10'])
    # plt.yscale('log')
    ax[0].set_xlabel(r'$M_{\text{p}} (\text{M}_{\oplus})$ ')
    ax[1].set_xlabel(r'$M_{\text{p}} (\text{M}_{\oplus})$ ')
    ax[0].set_ylabel(r'$\rho_{\text{p}} (\rho_{\oplus})$ ')
    ax[1].set_ylabel(r'$R_{\text{p}} (R_{\oplus})$ ')
    ax[0].text(-0.12, 1, "b", transform=ax[0].transAxes, fontsize=12, color='k', fontweight='bold')
    ax[1].text(-0.12, 1, "a", transform=ax[1].transAxes, fontsize=12, color='k', fontweight='bold')

    plt.savefig(os.path.join(folder, f'mass_density_individual.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    return


def figure_mr_mrho_save_param(folder='/Users/tehan/Documents/TGLC/', recalculate=False):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.12.07_14.30.50.csv'))
    b = t['pl_imppar']
    ror = t['pl_rade'] / t['st_rad'] / 109
    # Load tables
    t_new = ascii.read(pkg_resources.resource_stream(__name__, 'PS_2025.05.02_14.09.47.csv'))
    pl_rade_like = set(t['pl_rade_reflink'])

    # Fuzzy match setup
    threshold = 95  # adjust as needed
    filtered_rows = []
    print("Fuzzy matches (score between 90 and 99):")

    for refname in t_new['pl_refname']:
        match, score, _ = process.extractOne(refname, pl_rade_like, scorer=fuzz.token_sort_ratio)
        if threshold <= score < 100:
            print(f"{refname}  →  \n"
                  f"{match}  ({score}%)")
            filtered_rows.append(refname)
        elif score == 100:
            filtered_rows.append(refname)

    # Filter and write
    mask = np.isin(t_new['pl_refname'], filtered_rows)
    filtered_table = t_new[mask]
    # filtered_table.write('PS_reduced.csv', format='csv', overwrite=True)
    filtered_table = Table.read('PS_reduced.csv', format='csv')
    print(filtered_table)

    # find grazing
    # for i in range(len(b)):
    #     if 1 - b[i] < ror[i] / 2:
    #         print(t['tic_id'][i])
    #         print(b[i])
    #         print(ror[i])
    difference_tglc = ascii.read(f'{folder}deviation_TGLC.dat')
    difference_tglc = difference_tglc[np.where(difference_tglc['rhat'] == 1.0)]
    d_tglc = Table(
        names=['Star_sector', 'Tmag', 'rhat', 'p', 'pl_ratror', 'pl_ratrorerr1', 'pl_ratrorerr2',
               'value', 'err1', 'err2'],
        dtype=['S20', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
    failed = []
    for i in range(len(difference_tglc)):
        tic = difference_tglc['Star_sector'][i].split('_')[1]
        idx = np.where((f'TIC {tic}') == filtered_table['tic_id'])[0]
        if len(idx) >= 1:
            for j in range(len(idx)):
                # Create ufloats for planet radius and stellar radius
                pl_rade = ufloat(filtered_table['pl_rade'][idx[j]],
                                 (filtered_table['pl_radeerr1'][idx[j]] - filtered_table['pl_radeerr2'][idx[j]]) / 2)
                st_rad = ufloat(filtered_table['st_rad'][idx[j]],
                                (filtered_table['st_raderr1'][idx[j]] - filtered_table['st_raderr2'][idx[j]]) / 2)
                # Compute Rp/Rs with uncertainty
                ror_u = pl_rade / st_rad / 109.076
                # print(f"ror_u: {ror_u}, type(ror_u.s): {type(ror_u.s)}, ror_u.s: {ror_u.s}")
                if np.isnan(ror_u.s):
                    print(tic, filtered_table['st_raderr1'][idx[j]], filtered_table['st_raderr2'][idx[j]])
                else:
                    break
            d_tglc.add_row([
                difference_tglc['Star_sector'][i],
                difference_tglc['Tmag'][i],
                difference_tglc['rhat'][i],
                difference_tglc['p'][i],
                ror_u.n,
                ror_u.s,
                -ror_u.s,
                difference_tglc['value'][i],
                difference_tglc['err1'][i],
                difference_tglc['err2'][i]
            ])
        else:
            print(f'Failed for {tic}.')
            failed.append(tic)
            d_tglc.add_row([
                difference_tglc['Star_sector'][i],
                difference_tglc['Tmag'][i],
                difference_tglc['rhat'][i],
                difference_tglc['p'][i],
                0,
                0,
                0,
                difference_tglc['value'][i],
                difference_tglc['err1'][i],
                difference_tglc['err2'][i]
            ])

    print(d_tglc)
    print(set(failed))
    # d_tglc.write(f'{folder}deviation_TGLC_2025.dat', format='ascii.csv', overwrite=True)


    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    ng_color = palette[3]
    ng_corr_color = palette[2]
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax_ = plt.subplots(1, 2, sharex=True, figsize=(12, 5), gridspec_kw={'hspace': 0.01, 'wspace': 0.17})
    ax = [ax_[1], ax_[0]]
    for spine in ax[0].spines.values():
        spine.set_zorder(5)
    for spine in ax[1].spines.values():
        spine.set_zorder(5)
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])

    if recalculate:
        delta_R = 0.06011182562150113
        mass_g = []
        mass_g_err = []
        r_g = []
        r_g_err = []
        fp_g = []
        density_g = []
        density_g_err = []
        mass_ng = []
        mass_ng_err = []
        r_ng = []
        r_ng_corr = []
        r_ng_err = []
        r_ng_corr_err = []
        fp_ng = []
        density_ng = []
        density_ng_err = []
        density_ng_corr = []
        density_ng_corr_err = []
        tic_g = []
        tic_ng = []
        for i, tic in enumerate(tics):
            # if t['pl_bmassjlim'][i] == 0:
            #     density, density_err = mass_radius_to_density(t['pl_bmasse'][i], t['pl_rade'][i],
            #                                                   (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2,
            #                                                   (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
            if tic in ground:
                ror = []
                ror_err = []
                weights = []

                for j in range(len(difference_tglc)):
                    if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                        value = float(difference_tglc['value'][j])
                        err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
                        if err > 0:
                            ror.append(value)
                            ror_err.append(err)
                            weights.append(1 / err ** 2)
                ror = np.average(ror, weights=weights) if weights else np.nan
                ror_err = np.sqrt(1 / np.sum(weights)) if weights else np.nan
                if ror is not None and not np.isnan(ror):
                    tic_g.append(str(tic))
                    r_g.append(ror)
                    r_g_err.append(ror_err)

            elif tic in no_ground:
                ror = []
                ror_err = []
                weights = []

                for j in range(len(difference_tglc)):
                    if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                        value = float(difference_tglc['value'][j])
                        err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
                        if err > 0:
                            ror.append(value)
                            ror_err.append(err)
                            weights.append(1 / err ** 2)
                ror = np.average(ror, weights=weights) if weights else np.nan
                ror_err = np.sqrt(1 / np.sum(weights)) if weights else np.nan
                if ror is not None and not np.isnan(ror):
                    tic_ng.append(str(tic))
                    r_ng.append(ror)
                    r_ng_err.append(ror_err)

            # if density > 10:
                #     print(tic)
                # if tic in ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.9, marker='o', zorder=1, s=10, color=g_color)
                # elif tic in no_ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.5, marker='o', zorder=2, s=10, color='gray')
                #     ax.scatter(t['pl_bmasse'][i], density / 1.21, alpha=0.9, marker='o', zorder=3, s=15,
                #             color=ng_color)
                #     ax.plot([t['pl_bmasse'][i], t['pl_bmasse'][i]], [density, density / 1.21], color='gray', zorder=1,
                #             marker='', alpha=0.5,)
        data = {
            "mass_g": mass_g,
            "mass_g_err": mass_g_err,
            "r_g": r_g,
            "r_g_err": r_g_err,
            "density_g": density_g,
            "density_g_err": density_g_err,
            "mass_ng": mass_ng,
            "mass_ng_err": mass_ng_err,
            "r_ng": r_ng,
            "r_ng_corr": r_ng_corr,
            "r_ng_err": r_ng_err,
            "r_ng_corr_err": r_ng_corr_err,
            "density_ng": density_ng,
            "density_ng_err": density_ng_err,
            "density_ng_corr": density_ng_corr,
            "density_ng_corr_err": density_ng_corr_err,
            "tic_g": tic_g,
            "tic_ng": tic_ng
        }

        with open(f"{folder}mass_density.pkl", "wb") as f:
            pickle.dump(data, f)

    with open(f"{folder}mass_density.pkl", "rb") as f:
        data = pickle.load(f)
    mass_g = data["mass_g"]
    mass_g_err = data["mass_g_err"]
    r_g = data["r_g"]
    r_g_err = data["r_g_err"]
    density_g = data["density_g"]
    density_g_err = data["density_g_err"]
    mass_ng = data["mass_ng"]
    mass_ng_err = data["mass_ng_err"]
    r_ng = data["r_ng"]
    r_ng_corr = data["r_ng_corr"]
    r_ng_err = data["r_ng_err"]
    r_ng_corr_err = data["r_ng_corr_err"]
    density_ng = data["density_ng"]
    density_ng_err = data['density_ng_err']
    density_ng_corr = data["density_ng_corr"]
    density_ng_corr_err = data["density_ng_corr_err"]
    tic_g = data["tic_g"]
    tic_ng = data["tic_ng"]
    print(len(r_g))
    print(len(r_ng))
    # density_weight = np.array(r_ng_err)[np.where(np.array(r_ng)<4)]
    # print(density_weight)
    ### mass-density ###
    return

# Define the piecewise function Muller et al. 2024
def H(x):
    """Heaviside step function."""
    return np.heaviside(x, 1)

def y_xi(xi, c, alpha1, beta, psi):
    """
    Compute y given xi using the specified equation.

    Parameters:
    - xi: Input variable (scalar or array).
    - c: Constant term.
    - alpha1: Linear coefficient.
    - beta: List of two beta coefficients [β1, β2].
    - psi: List of two psi values [ψ1, ψ2].

    Returns:
    - y: Computed output.
    """
    return c + alpha1 * xi + sum(beta[i] * (xi - psi[i]) * H(xi - psi[i]) for i in range(2))

def radius_1(M):
    if M < 4.37:
        return (1.18 * M**0.00, (1.18 - 1.21) * M**(0.00 - 0.17), (1.18 + 1.21) * M**(0.00 + 0.17))
    elif 4.37 <= M < 127:
        return (0.41 * M**0.71, (0.41 - 1.57) * M**(0.71 - 0.26), (0.41 + 1.57) * M**(0.71 + 0.26))
    else:
        return (14.91 * M**-0.03, (14.91 - 1.11) * M**(-0.03 - 0.26), (14.91 + 1.11) * M**(-0.03 + 0.26))

def radius_2(M):
    if M < 4.37:
        return (1.01 * M**0.26, (1.01 - 1.41) * M**(0.26 - 0.26), (1.01 + 1.41) * M**(0.26 + 0.26))
    elif 4.37 <= M < 127:
        return (0.56 * M**0.66, (0.56 - 1.92) * M**(0.66 - 0.39), (0.56 + 1.92) * M**(0.66 + 0.39))
    else:
        return (14.40 * M**-0.01, (14.40 - 1.10) * M**(-0.01 - 0.39), (14.40 + 1.10) * M**(-0.01 + 0.39))


def piecewise_power_law(M, params):
    c1, alpha1, c2, alpha2, c3, alpha3, M1, M2 = params
    R = np.piecewise(
        M,
        [M < M1, (M1 <= M) & (M < M2), M >= M2],
        [lambda M: c1 * M ** alpha1,
         lambda M: c2 * M ** alpha2,
         lambda M: c3 * M ** alpha3]
    )
    return R


def fit_piecewise_power_law(M, R, M_err, R_err):
    logM, logR = np.log10(M), np.log10(R)
    logM_err, logR_err = M_err / (M * np.log(10)), R_err / (R * np.log(10))

    def model_func(params, x):
        return np.log10(piecewise_power_law(10 ** x, params))

    model = Model(model_func)
    data = RealData(logM, logR, sx=logM_err, sy=logR_err)
    odr = ODR(data, model, beta0=[1.02, 0.27, 0.56, 0.67, 18.6, -0.06, 4.37, 127])
    output = odr.run()

    params, param_errors = output.beta, output.sd_beta
    return params, param_errors


def figure_mr_mrho_all(folder='/Users/tehan/Documents/TGLC/', recalculate=False):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced.csv'))
    b = t['pl_imppar']
    ror = t['pl_rade'] / t['st_rad'] / 109
    # find grazing
    # for i in range(len(b)):
    #     if 1 - b[i] < ror[i] / 2:
    #         print(t['tic_id'][i])
    #         print(b[i])
    #         print(ror[i])
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025.dat')
    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    ng_color = palette[3]
    ng_corr_color = palette[2]
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 8), gridspec_kw={'hspace': 0.1})
    for spine in ax.spines.values():
        spine.set_zorder(5)
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])


    if recalculate:
        delta_R = 0.06011182562150113
        mass_g = []
        mass_g_err = []
        r_g = []
        r_g_err = []
        density_g = []
        density_g_err = []
        mass_ng = []
        mass_ng_err = []
        r_ng = []
        r_ng_corr = []
        r_ng_err = []
        r_ng_corr_err = []
        density_ng = []
        density_ng_err = []
        density_ng_corr = []
        density_ng_corr_err = []
        tic_g = []
        tic_ng = []
        tmag_g = []
        tmag_ng = []
        for i, tic in enumerate(tics):
            if t['pl_bmassjlim'][i] == 0:
                density, density_err = mass_radius_to_density(t['pl_bmasse'][i], t['pl_rade'][i],
                                                              (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2,
                                                              (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                if tic in ground:
                    tmag_g.append(t['sy_tmag'][i])
                    if t['pl_bmassprov'][i] == 'Mass':
                        if ((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2 / t['pl_bmasse'][i] < 0.25 and
                                (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2 / t['pl_rade'][i] < 0.20):

                            mass_g.append(t['pl_bmasse'][i])
                            mass_g_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2)
                            r_g.append(t['pl_rade'][i])
                            r_g_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                            density_g.append(density)
                            density_g_err.append(density_err)
                            tic_g.append(str(tic))

                elif tic in no_ground:
                    tmag_ng.append(t['sy_tmag'][i])
                    if t['pl_bmassprov'][i] == 'Mass':
                        # ## overall shift ###
                        # density_ng_corr.append(density * (1 - delta_R) ** 3)
                        ### individual fits ###
                        ror = []
                        ror_err = []
                        weights = []

                        for j in range(len(difference_tglc)):
                            if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                                value = float(difference_tglc['value'][j])
                                err = float((difference_tglc['err1'][j] - difference_tglc['err2'][j]) / 2)
                                if err > 0:
                                    ror.append(value)
                                    ror_err.append(err)
                                    weights.append(1 / err ** 2)

                        ror = np.average(ror, weights=weights) if weights else np.nan
                        ror_err = np.sqrt(1 / np.sum(weights)) if weights else np.nan

                        # print(ror is np.nan)
                        density_corr, density_corr_err = mass_radius_to_density(t['pl_bmasse'][i], 109.076 * ror * t['st_rad'][i],
                                                                (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2,
                                                                109.076 * ror * (t['st_raderr1'][i] - t['st_raderr2'][i])/2)
                        delta_Rstar = 109.076 * (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                        delta_ror = ror_err
                        if ror is not None and not np.isnan(ror):
                            if (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2 / t['pl_bmasse'][i] < 0.25 and np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2)/ (109.076 * ror * t['st_rad'][i]) < 0.20:
                                mass_ng.append(t['pl_bmasse'][i])
                                mass_ng_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2)
                                r_ng.append(t['pl_rade'][i])
                                r_ng_corr.append(109.076 * ror * t['st_rad'][i])
                                density_ng.append(density)
                                density_ng_err.append(density_err)
                                density_ng_corr_err.append(density_corr_err)
                                tic_ng.append(str(tic))
                                density_ng_corr.append(density_corr)
                                r_ng_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i])/2)
                                r_ng_corr_err.append(np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2))
                                if 109.076 * ror * t['st_rad'][i] < 4:
                                    print(f"Teff = {t['st_teff'][i]}")
                # if density > 10:
                #     print(tic)
                # if tic in ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.9, marker='o', zorder=1, s=10, color=g_color)
                # elif tic in no_ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.5, marker='o', zorder=2, s=10, color='gray')
                #     ax.scatter(t['pl_bmasse'][i], density / 1.21, alpha=0.9, marker='o', zorder=3, s=15,
                #             color=ng_color)
                #     ax.plot([t['pl_bmasse'][i], t['pl_bmasse'][i]], [density, density / 1.21], color='gray', zorder=1,
                #             marker='', alpha=0.5,)
        data = {
            "mass_g": mass_g,
            "mass_g_err": mass_g_err,
            "r_g": r_g,
            "r_g_err": r_g_err,
            "density_g": density_g,
            "density_g_err": density_g_err,
            "mass_ng": mass_ng,
            "mass_ng_err": mass_ng_err,
            "r_ng": r_ng,
            "r_ng_corr": r_ng_corr,
            "r_ng_err": r_ng_err,
            "r_ng_corr_err": r_ng_corr_err,
            "density_ng": density_ng,
            "density_ng_err": density_ng_err,
            "density_ng_corr": density_ng_corr,
            "density_ng_corr_err": density_ng_corr_err,
            "tic_g": tic_g,
            "tic_ng": tic_ng
        }

        with open(f"{folder}mass_density.pkl", "wb") as f:
            pickle.dump(data, f)
    # plt.hist(tmag_ng)
    # plt.hist(tmag_g)
    # plt.show()

    with open(f"{folder}mass_density.pkl", "rb") as f:
        data = pickle.load(f)
    mass_g = data["mass_g"]
    mass_g_err = data["mass_g_err"]
    r_g = data["r_g"]
    r_g_err = data["r_g_err"]
    density_g = data["density_g"]
    density_g_err = data["density_g_err"]
    mass_ng = data["mass_ng"]
    mass_ng_err = data["mass_ng_err"]
    r_ng = data["r_ng"]
    r_ng_corr = data["r_ng_corr"]
    r_ng_err = data["r_ng_err"]
    r_ng_corr_err = data["r_ng_corr_err"]
    density_ng = data["density_ng"]
    density_ng_err = data['density_ng_err']
    density_ng_corr = data["density_ng_corr"]
    density_ng_corr_err = data["density_ng_corr_err"]
    tic_g = data["tic_g"]
    tic_ng = data["tic_ng"]
    print(len(mass_g))
    print(len(mass_ng))
    print(len(r_ng_corr))
    # ### mass-density ###
    # ### mass-radius ###
    # ax.scatter(mass_g, r_g, alpha=0.9, marker='o', zorder=2, s=15, color=palette[7], label='TESS-free')
    # ax.scatter(mass_ng, r_ng, alpha=0.9, marker='o', zorder=3, s=15, facecolors='none', edgecolors=ng_color,
    #               label='TESS-dependent')
    # # for j in range(len(mass_ng)):
    # #     plt.text(mass_ng[j], density_ng[j], tic_ng[j], fontsize=2)
    # ax.errorbar(mass_ng, r_ng_corr, xerr=mass_ng_err, yerr=r_ng_corr_err, fmt='o', alpha=0.9, color=ng_corr_color, lw=0.7,
    #                ms=np.sqrt(15), zorder=4, label='TESS-dependent corrected')
    # ax.legend(loc=2, fontsize=10)
    #
    marker_size = 35
    ax.errorbar(mass_g, r_g, xerr=mass_g_err, yerr=r_g_err, fmt='none', alpha=0.7,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, color=palette[7])
    ax.scatter(mass_g, r_g, alpha=0.9, marker='D', zorder=2, s=marker_size, color=palette[7], facecolors=palette[7],
                  edgecolors='k', linewidths=0.5)
    ax.errorbar(mass_ng, r_ng, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.7,
                   zorder=3, capsize=2.5, capthick=0.7, lw=0.7, color=ng_color)
    ax.scatter(mass_ng, r_ng, alpha=0.9, marker='o', zorder=3, s=marker_size, facecolors='none', edgecolors=ng_color)
    ax.errorbar(mass_ng, r_ng_corr, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.7,
                   color=ng_corr_color, lw=0.7, zorder=4, capsize=2.5, capthick=0.7)
    ax.scatter(mass_ng, r_ng_corr, color=ng_corr_color, alpha=0.9, s=marker_size, zorder=4)
    # ax.plot([mass_ng, mass_ng], [r_ng, r_ng_corr], color='gray', zorder=2, marker='', linewidth=3, alpha=0.5, )

    ax.errorbar(0, 0, xerr=0, yerr=0, fmt='D', alpha=0.9, markerfacecolor=palette[7], markeredgecolor='k',
                   markeredgewidth=0.5,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size), color=palette[7], label='TESS-free')
    # ax[0].errorbar(mass_ng, density_ng_corr, xerr=mass_ng_err, yerr=density_ng_corr_err, fmt='o', alpha=0.3,
    #                color=ng_corr_color, lw=0.7, ms=np.sqrt(15), zorder=4, capsize=2.5, capthick=0.7)
    ax.errorbar(0, 0, xerr=0, yerr=0, fmt='o', alpha=0.9,
                   zorder=3, capsize=2.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size),
                   color=ng_color, markeredgecolor=ng_color, markerfacecolor='none',
                   label='TESS-dependent')
    ax.errorbar(0, 0, xerr=0, yerr=0, fmt='o', alpha=0.9,
                   color=ng_corr_color, lw=0.7, zorder=4, capsize=2.5, capthick=0.7, ms=np.sqrt(marker_size), label='TESS-dependent TGLC-fitted')
    # ax[0].scatter(1, 1, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(1, 1 * 1.08, 'Earth', color='k', fontsize=9, ha='center', va='center', zorder=4, rotation=0)
    # ax.scatter(1, 1, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax.text(1, 1 * 1.1, 'Earth', color='k', fontsize=9, ha='center', va='center', zorder=4, rotation=0)
    # ax[0].scatter(0.107,0.714,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(0.107, 0.714, 'Mars', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(95.2, 0.125, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(95.2 / 1.05, 0.125, 'Saturn', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax.scatter(95.2, 9.45, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax.text(95.2 / 1.05, 9.45, 'Saturn', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(14.5,0.230,facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(14.5/1.05, 0.230, 'Uranus', color='k', fontsize=9, ha='right', va='center', zorder=4, rotation=0)
    # ax[0].scatter(17.1, 0.297, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax[0].text(17.1 * 1.08, 0.297, 'Neptune', color='k', fontsize=9, ha='left', va='center', zorder=4, rotation=0)
    # ax.scatter(17.1, 3.88, facecolors='none', edgecolors='k', zorder=4, marker='o')
    # ax.text(17.1 * 1.08, 3.88, 'Neptune', color='k', fontsize=9, ha='left', va='center', zorder=4, rotation=0)
    #####

    # Generate mass values

    # archival
    c = 0.01
    alpha1 = 0.27
    beta = np.array([0.40, -0.72])
    psi = np.array([0.64, 2.11])
    d_c = 0.01
    d_alpha1 = 0.04
    d_beta = np.array([0.04, 0.02])
    d_psi = np.array([0.07, 0.03])
    xi_values = np.linspace(0, 4, 1000)
    y_values = y_xi(xi_values, c, alpha1, beta, psi)
    y_upper = y_xi(xi_values, c + d_c, alpha1 + d_alpha1, beta+d_beta, psi+d_psi)
    y_lower = y_xi(xi_values, c - d_c, alpha1 - d_alpha1, beta-d_beta, psi-d_psi)
    # Plot the M-R relation
    ax.plot(10 ** xi_values, 10 ** y_values, label=r'Literature $M$-$R$ relation', color='k', zorder=9, lw=4)
    # ax.fill_between(10 ** xi_values, 10 ** y_upper, 10 ** y_lower, color='k', alpha=0.1, zorder=6)

    # #### Original method
    # low_mass = np.where(np.array(mass_g + mass_ng) > 4.37)
    # logx = np.log(np.array(mass_g + mass_ng))[low_mass]
    # logy = np.log(np.array(r_g + r_ng_corr))[low_mass]
    # logx_err = np.array(mass_g_err + mass_ng_err)[low_mass] / np.array(mass_g + mass_ng)[low_mass] / np.log(10)
    # logy_err = np.array(r_g_err + r_ng_corr_err)[low_mass] / np.array(r_g + r_ng_corr)[low_mass] / np.log(10)
    # fit = Fit(logx, logy, sxx=logx_err, syy=logy_err, n_breakpoints=1, verbose=True, n_boot=100)
    # fit.summary()
    #
    # # Retrieve fitted parameters if converged
    # params = fit.get_params()
    # if not params['converged']:
    #     print("Model did not converge. Check summary for details.")
    # else:
    #     # Extract estimates and standard errors
    #     estimates = fit.best_muggeo.best_fit.estimates
    #
    #     # Get parameters in log space
    #     const_est = estimates['const']['estimate']
    #     const_se = estimates['const']['se']
    #     alpha1_est = estimates['alpha1']['estimate']
    #     alpha1_se = estimates['alpha1']['se']
    #     alpha2_est = estimates['alpha2']['estimate']
    #     alpha2_se = estimates['alpha2']['se']
    #     beta1_est = estimates['beta1']['estimate']
    #     beta1_se = estimates['beta1']['se']
    #     bp_log_est = estimates['breakpoint1']['estimate']
    #     bp_log_se = estimates['breakpoint1']['se']
    #
    #     # Convert to original scale
    #     a1_est = np.exp(const_est)
    #     a1_se = a1_est * const_se  # Error propagation: SE(exp(c)) ≈ exp(c) * SE(c)
    #     bp_est = np.exp(bp_log_est)
    #     bp_se = bp_est * bp_log_se  # SE(exp(bp_log)) ≈ exp(bp_log) * SE(bp_log)
    #
    #     # Calculate a2 and its approximate error
    #     a2_est = np.exp(const_est - beta1_est * bp_log_est)
    #     # Approximate error propagation (simplified assuming independence)
    #     var_a2 = (const_se ** 2) + (beta1_est ** 2 * bp_log_se ** 2) + (bp_log_est ** 2 * beta1_se ** 2)
    #     a2_se = a2_est * np.sqrt(var_a2)
    #
    #     # Format results with errors
    #     result_str = (
    #         f"R = ({a1_est:.2f} ±{a1_se:.2f}) M^{{{alpha1_est:.2f}±{alpha1_se:.2f}}} "
    #         f"({a2_est:.2f} ±{a2_se:.2f}) M^{{{alpha2_est:.2f}±{alpha2_se:.2f}}} "
    #         f"\nBreakpoint at M = {bp_est:.1f} ±{bp_se:.1f}"
    #     )
    #     print("\nFormatted Results:")
    #     print(result_str)
    #
    #     # Plot results
    #     x_plot = np.logspace(np.log10(4.37), 4, 200)
    #     logx_plot = np.log(x_plot)
    #     plt.plot(x_plot, np.exp(fit.predict(logx_plot)), 'r-', label='Fitted Model')
    #     plt.axvline(bp_est, color='r', linestyle='-', alpha=0.5, label='Fitted Breakpoint')

    #######
    params, param_errors = fit_piecewise_power_law(np.array(mass_g + mass_ng), np.array(r_g + r_ng),
                                                   np.array(mass_g_err + mass_ng_err), np.array(r_g_err + r_ng_err))

    # Plot results
    M_fine = np.logspace(0, 4, 1000)
    R_fit = piecewise_power_law(M_fine, params)
    R_upper = piecewise_power_law(M_fine, params + param_errors)
    R_lower = piecewise_power_law(M_fine, params - param_errors)

    plt.plot(M_fine, R_fit, label=r'TESS $M$-$R$ relation', color=ng_color, zorder=10, lw=4)
    plt.plot(M_fine, R_fit, color='w', lw=5, alpha=0.5, zorder=9)
    plt.fill_between(M_fine, R_lower, R_upper, color=ng_color, alpha=0.2, zorder=7)

    # Print results
    print(
        f"R=\n({params[0]:.2f} ± {param_errors[0]:.2f}) M^({params[1]:.2f} ± {param_errors[1]:.2f}) for M < ({params[6]:.2f} ± {param_errors[6]:.2f})")
    print(
        f"({params[2]:.2f} ± {param_errors[2]:.2f}) M^({params[3]:.2f} ± {param_errors[3]:.2f}) for {params[6]:.2f} < M < ({params[7]:.2f} ± {param_errors[7]:.2f})")
    print(
        f"({params[4]:.2f} ± {param_errors[4]:.2f}) M^({params[5]:.2f} ± {param_errors[5]:.2f}) for M > ({params[7]:.2f} ± {param_errors[7]:.2f})")
    ########
    params, param_errors = fit_piecewise_power_law(np.array(mass_g + mass_ng), np.array(r_g + r_ng_corr),
                                                   np.array(mass_g_err + mass_ng_err), np.array(r_g_err + r_ng_corr_err))

    # Plot results
    M_fine = np.logspace(0, 4, 1000)
    R_fit = piecewise_power_law(M_fine, params)
    R_upper = piecewise_power_law(M_fine, params + param_errors)
    R_lower = piecewise_power_law(M_fine, params - param_errors)

    plt.plot(M_fine, R_fit, label=r'Corrected TESS $M$-$R$ relation', color=ng_corr_color, zorder=11, lw=4)
    plt.plot(M_fine, R_fit, color='w', lw=5, alpha=0.5, zorder=10)
    plt.fill_between(M_fine, R_lower, R_upper, color=ng_corr_color, alpha=0.2, zorder=8)

    # Print results
    print(
        f"R=\n({params[0]:.2f} ± {param_errors[0]:.2f}) M^({params[1]:.2f} ± {param_errors[1]:.2f}) for M < ({params[6]:.2f} ± {param_errors[6]:.2f})")
    print(
        f"({params[2]:.2f} ± {param_errors[2]:.2f}) M^({params[3]:.2f} ± {param_errors[3]:.2f}) for {params[6]:.2f} < M < ({params[7]:.2f} ± {param_errors[7]:.2f})")
    print(
        f"({params[4]:.2f} ± {param_errors[4]:.2f}) M^({params[5]:.2f} ± {param_errors[5]:.2f}) for M > ({params[7]:.2f} ± {param_errors[7]:.2f})")

    # # Modified fitting call (using fixed breakpoints)
    # result = fit_piecewise_power_law(
    #     mass_g + mass_ng,
    #     r_g + r_ng,
    #     mass_g_err + mass_ng_err,
    #     r_g_err + r_ng_err
    # )
    # # print(np.min(mass_g_err + mass_ng_err))
    # # print(np.max(mass_g_err + mass_ng_err))
    # # print(np.min(r_g_err + r_ng_err))
    # # print(np.max(r_g_err + r_ng_err))
    # # Extract results (now with fixed breakpoints)
    # lower, upper, best_fit, M_fit = result['uncertainty_bands']
    # M1, M2 = result['breakpoints']  # Will be (4.37, 127)
    #
    # # Plotting remains similar but with fixed breakpoint labels
    # plt.fill_between(M_fit, lower, upper, color='red', alpha=0.3,
    #                  label='1σ uncertainty')
    # plt.plot(M_fit, best_fit, 'r-', lw=2, label='Best fit')
    #
    # # Fixed breakpoint lines (no error bars)
    # plt.axvline(M1, color='k', ls=':',
    #             label=f'Rocky-Icy Transition: {M1} M⊕ (fixed)')
    # plt.axvline(M2, color='k', ls=':',
    #             label=f'Icy-Giant Transition: {M2} M⊕ (fixed)')
    # # Print all power-law relations with errors
    # print("\n=== Mass-Radius Relation Results ===")
    # print(f"Small planets (M < 4.37 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['small'][0]:.2f} ± {result['power_laws_err']['small'][0]:.2f}) × M^{result['power_laws']['small'][1]:.2f}±{result['power_laws_err']['small'][1]:.2f}")
    #
    # print(f"\nIntermediate planets (4.37 ≤ M < 127 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['medium'][0]:.2f} ± {result['power_laws_err']['medium'][0]:.2f}) × M^{result['power_laws']['medium'][1]:.2f}±{result['power_laws_err']['medium'][1]:.2f}")
    #
    # print(f"\nGiant planets (M ≥ 127 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['large'][0]:.2f} ± {result['power_laws_err']['large'][0]:.2f}) × M^{result['power_laws']['large'][1]:.2f}±{result['power_laws_err']['large'][1]:.2f}")
    # # Updated print statements for fixed breakpoint version
    # # Modified fitting call (using fixed breakpoints)
    # result = fit_piecewise_power_law(
    #     mass_g + mass_ng,
    #     r_g + r_ng_corr,
    #     mass_g_err + mass_ng_err,
    #     r_g_err + r_ng_corr_err
    # )
    # print(np.min(mass_g_err + mass_ng_err))
    # print(np.max(mass_g_err + mass_ng_err))
    # print(np.min(r_g_err + r_ng_corr_err))
    # print(np.max(r_g_err + r_ng_corr_err))
    # # Extract results (now with fixed breakpoints)
    # lower, upper, best_fit, M_fit = result['uncertainty_bands']
    # M1, M2 = result['breakpoints']  # Will be (4.37, 127)
    #
    # # Plotting remains similar but with fixed breakpoint labels
    # plt.fill_between(M_fit, lower, upper, color='blue', alpha=0.3,
    #                  label='1σ uncertainty')
    # plt.plot(M_fit, best_fit, 'b-', lw=2, label='Best fit')
    # # Print all power-law relations with errors
    # print("\n=== Mass-Radius Relation Results ===")
    # print(f"Small planets (M < 4.37 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['small'][0]:.2f} ± {result['power_laws_err']['small'][0]:.2f}) × M^{result['power_laws']['small'][1]:.2f}±{result['power_laws_err']['small'][1]:.2f}")
    #
    # print(f"\nIntermediate planets (4.37 ≤ M < 127 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['medium'][0]:.2f} ± {result['power_laws_err']['medium'][0]:.2f}) × M^{result['power_laws']['medium'][1]:.2f}±{result['power_laws_err']['medium'][1]:.2f}")
    #
    # print(f"\nGiant planets (M ≥ 127 M⊕):")
    # print(
    #     f"R = ({result['power_laws']['large'][0]:.2f} ± {result['power_laws_err']['large'][0]:.2f}) × M^{result['power_laws']['large'][1]:.2f}±{result['power_laws_err']['large'][1]:.2f}")
    # ax[0].set_xscale('log')
    ax.legend(loc=4, fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax[0].set_ylim(-.1, 2)
    ax.set_ylim(0.9, 10)
    # ax[0].set_xlim(0.8, 100)
    ax.set_xlim(0.8, 100)
    # ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks([1, 10, 100, 1000, 10000])

    # ax[0].set_yticks([0, 0.5, 1, 1.5, 2])
    # ax[0].set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(['1', '3', '10', '30'])
    # plt.yscale('log')
    # ax[0].set_xlabel(r'$M_{\text{p}} (\text{M}_{\oplus})$ ')
    ax.set_xlabel(r'$M_{\text{p}} (\text{M}_{\oplus})$ ')
    # ax[0].set_ylabel(r'$\rho_{\text{p}} (\rho_{\oplus})$ ')
    ax.set_ylabel(r'$R_{\text{p}} (R_{\oplus})$ ')

    plt.savefig(os.path.join(folder, f'mass_density_all.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    return


def calculate_tsm(table, i, rp=1):
    print(rp)
    Ts = table['st_teff'][i]
    ratdor = table['pl_ratdor'][i]
    Rs = table['st_rad'][i]
    # rp = table['pl_rade'][i]
    # if correction:
    #     rp = rp / 0.94
    mj = table['sy_jmag'][i]
    mp = table['pl_bmasse'][i]
    if rp < 1.5:
        f = 0.190
    elif 1.5 <= rp < 2.75:
        f = 1.26
    elif 2.75 <= rp < 4.0:
        f = 1.28
    elif 4.0 <= rp < 10.0:
        f = 1.15
    else:
        f = np.nan
    Teq = Ts * np.sqrt(1 / ratdor) * (1 / 4) ** (1 / 4)
    tsm = f * rp ** 3 * Teq * 10 ** (-mj / 5) / mp / Rs ** 2
    return tsm


def figure_tsm(folder='/Users/tehan/Documents/TGLC/', recalculate=False):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.12.07_14.30.50.csv'))
    b = t['pl_imppar']
    ror = t['pl_rade'] / t['st_rad'] / 109
    # find grazing
    # for i in range(len(b)):
    #     if 1 - b[i] < ror[i] / 2:
    #         print(t['tic_id'][i])
    #         print(b[i])
    #         print(ror[i])
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2024.dat')
    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    ng_color = palette[3]
    ng_corr_color = palette[2]
    sns.set_style("ticks", {'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                            'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                            'axes.facecolor': '1', 'grid.color': '0.'})
    # sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
    #             'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
    #             'axes.facecolor': '1', 'grid.color': '0.'})
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5), gridspec_kw={'hspace': 0.01, 'wspace': 0.17})
    for spine in ax.spines.values():
        spine.set_zorder(5)
    # for spine in ax[1].spines.values():
    #     spine.set_zorder(5)
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])
    if recalculate:
        delta_R = 0.06011182562150113
        mass_g = []
        mass_g_err = []
        r_g = []
        r_g_err = []
        density_g = []

        mass_ng = []
        mass_ng_err = []
        r_ng = []
        r_ng_corr = []
        r_ng_err = []
        r_ng_corr_err = []
        density_ng = []
        density_ng_corr = []
        density_ng_corr_err = []
        tic_ng = []
        tsm_g = []
        tsm_ng = []
        tsm_ng_corr = []
        for i, tic in enumerate(tics):
            if t['pl_bmassjlim'][i] == 0:
                density, density_err = mass_radius_to_density(t['pl_bmasse'][i], t['pl_rade'][i],
                                                              (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2,
                                                              (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                if tic in ground:
                    if t['pl_bmassprov'][i] == 'Mass':
                        if ((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2 / t['pl_bmasse'][i] < 0.25 and
                                (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2 / t['pl_rade'][i] < 0.20):

                            mass_g.append(t['pl_bmasse'][i])
                            mass_g_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2)
                            r_g.append(t['pl_rade'][i])
                            r_g_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                            density_g.append(density)
                            tsm_g.append(calculate_tsm(t, i, rp=t['pl_rade'][i]))
                elif tic in no_ground:
                    if t['pl_bmassprov'][i] == 'Mass':
                        # ## overall shift ###
                        # density_ng_corr.append(density * (1 - delta_R) ** 3)
                        ### individual fits ###
                        ror = []
                        ror_err = []
                        for j in range(len(difference_tglc)):
                            if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                                ror.append(float(difference_tglc['value'][j]))
                                ror_err.append(float((difference_tglc['err1'][j] - difference_tglc['err2'][j])/2))
                        ror = np.nanmedian(np.array(ror))
                        ror_err = np.nanmedian(np.array(ror_err))

                        # print(ror is np.nan)
                        density_corr, density_corr_err = mass_radius_to_density(t['pl_bmasse'][i], 109.076 * ror * t['st_rad'][i],
                                                                (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2,
                                                                109.076 * ror * (t['st_raderr1'][i] - t['st_raderr2'][i])/2)
                        delta_Rstar = 109.076 * (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                        delta_ror = ror_err
                        if ror is not None and not np.isnan(ror):
                            if (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2 / t['pl_bmasse'][i] < 0.25 and np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2)/ (109.076 * ror * t['st_rad'][i]) < 0.20:
                                mass_ng.append(t['pl_bmasse'][i])
                                mass_ng_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2)
                                r_ng.append(t['pl_rade'][i])
                                r_ng_corr.append(109.076 * ror * t['st_rad'][i])
                                density_ng.append(density)
                                density_ng_corr_err.append(density_corr_err)
                                tic_ng.append(str(tic))
                                density_ng_corr.append(density_corr)
                                r_ng_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i])/2)
                                r_ng_corr_err.append(np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2))
                                tsm_ng.append(calculate_tsm(t, i, rp=t['pl_rade'][i]))
                                tsm_ng_corr.append(calculate_tsm(t, i, rp=109.076 * ror * t['st_rad'][i]))
                # if density > 10:
                #     print(tic)
                # if tic in ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.9, marker='o', zorder=1, s=10, color=g_color)
                # elif tic in no_ground:
                #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.5, marker='o', zorder=2, s=10, color='gray')
                #     ax.scatter(t['pl_bmasse'][i], density / 1.21, alpha=0.9, marker='o', zorder=3, s=15,
                #             color=ng_color)
                #     ax.plot([t['pl_bmasse'][i], t['pl_bmasse'][i]], [density, density / 1.21], color='gray', zorder=1,
                #             marker='', alpha=0.5,)
        data = {
            "mass_g": mass_g,
            "mass_g_err": mass_g_err,
            "r_g": r_g,
            "r_g_err": r_g_err,
            "density_g": density_g,
            "mass_ng": mass_ng,
            "mass_ng_err": mass_ng_err,
            "r_ng": r_ng,
            "r_ng_corr": r_ng_corr,
            "r_ng_err": r_ng_err,
            "r_ng_corr_err": r_ng_corr_err,
            "density_ng": density_ng,
            "density_ng_corr": density_ng_corr,
            "density_ng_corr_err": density_ng_corr_err,
            "tic_ng": tic_ng,
            "tsm_g": tsm_g,
            "tsm_ng": tsm_ng,
            "tsm_ng_corr": tsm_ng_corr
        }

        with open(f"{folder}mass_density.pkl", "wb") as f:
            pickle.dump(data, f)

    with open(f"{folder}mass_density.pkl", "rb") as f:
        data = pickle.load(f)
    mass_g = data["mass_g"]
    mass_g_err = data["mass_g_err"]
    r_g = data["r_g"]
    r_g_err = data["r_g_err"]
    density_g = data["density_g"]
    mass_ng = data["mass_ng"]
    mass_ng_err = data["mass_ng_err"]
    r_ng = data["r_ng"]
    r_ng_corr = data["r_ng_corr"]
    r_ng_err = data["r_ng_err"]
    r_ng_corr_err = data["r_ng_corr_err"]
    density_ng = data["density_ng"]
    density_ng_corr = data["density_ng_corr"]
    density_ng_corr_err = data["density_ng_corr_err"]
    tic_ng = data["tic_ng"]
    tsm_g = data["tsm_g"]
    tsm_ng = data["tsm_ng"]
    tsm_ng_corr = data["tsm_ng_corr"]
    # density_weight = np.array(r_ng_err)[np.where(np.array(r_ng)<4)]
    # print(density_weight)
    ### mass-density ###
    ax.scatter(r_g, tsm_g, alpha=0.9, marker='o', zorder=1, s=15, color=palette[7], label='TESS-free')
    ax.scatter(r_ng, tsm_ng, alpha=0.9, marker='o', zorder=2, s=15, facecolors='none', edgecolors=ng_color,
               label='TESS-dependent')
    # for j in range(len(mass_ng)):
    #     plt.text(mass_ng[j], density_ng[j], tic_ng[j], fontsize=2)
    ax.scatter(r_ng_corr, tsm_ng_corr, alpha=0.9, marker='o', zorder=3, s=15, color=ng_corr_color,
               label='TESS-dependent corrected')
    ax.plot([r_ng, r_ng_corr], [tsm_ng, tsm_ng_corr], color='gray',
            zorder=1, marker='', linewidth=0.6, alpha=0.5, )
    count = 0
    for i in range(len(r_ng)):
        if (r_ng[i] < 1.5 and tsm_ng[i] < 12) or (1.5 <= r_ng[i] < 2.75 and tsm_ng[i] < 92) or (
                2.75 <= r_ng[i] < 4 and tsm_ng[i] < 84) \
                or (4 <= r_ng[i] < 10 and tsm_ng[i] < 96):
            if (r_ng_corr[i] < 1.5 and tsm_ng_corr[i] >= 12) or (1.5 <= r_ng_corr[i] < 2.75 and tsm_ng_corr[i] >= 92) \
                    or (2.75 <= r_ng_corr[i] < 4 and tsm_ng_corr[i] >= 84) or \
                    (4 <= r_ng_corr[i] < 10 and tsm_ng_corr[i] >= 96):
                ax.plot([r_ng[i], r_ng_corr[i]], [tsm_ng[i], tsm_ng_corr[i]], color='k',
                        zorder=1, marker='', linewidth=2, alpha=1, )
        elif tsm_ng[i] > 0:
            count += 1
    print(count)
    ax.hlines(12, xmin=0, xmax=1.5, linestyles='dotted', color='k', label='Follow-up cutoff')
    ax.hlines(92, xmin=1.5, xmax=2.75, linestyles='dotted', color='k')
    ax.hlines(84, xmin=2.75, xmax=4, linestyles='dotted', color='k')
    ax.hlines(96, xmin=4, xmax=10, linestyles='dotted', color='k')
    ax.vlines(x=1.5, ymin=12, ymax=92, linestyles='dotted', color='k')
    ax.vlines(x=2.75, ymin=92, ymax=84, linestyles='dotted', color='k')
    ax.vlines(x=4, ymin=84, ymax=96, linestyles='dotted', color='k')

    x_shade = np.array([0, 0, 1.5, 1.5, 2.75, 2.75, 4, 4, 10])
    y_shade = np.array([0, 12, 12, 92, 92, 84, 84, 96, 96])
    ax.fill_between(x_shade, y_shade, 0, color='gray', alpha=0.2, hatch='\\\\', edgecolor='k', zorder=1)
    plt.legend()
    # ax.set_xscale('log')
    # ax[1].set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylim(0, 300)
    # ax[1].set_ylim(4, 10)
    ax.set_xlim(0., 10)
    # ax[1].set_xlim(0.8, 1000)
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.xticks([10, 30, 100, 300, 1000])

    # ax.set_yticks([0, 0.5, 1, 1.5, 2])
    # ax.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    # ax[1].set_yticks([1, 2, 3, 5, 10])
    # ax[1].set_yticklabels(['1', '2', '3', '5', '10'])
    # plt.yscale('log')
    ax.set_xlabel(r'$R_{\text{p}} (\text{R}_{\oplus})$ ')
    # ax[1].set_xlabel(r'$M_{\text{p}} (\text{M}_{\oplus})$ ')
    ax.set_ylabel(r'TSM')
    # ax[1].set_ylabel(r'$R_{\text{p}} (R_{\oplus})$ ')
    # ax.text(0.1, 0.9, "b", transform=ax.transAxes, fontsize=12, color='k', fontweight='bold')
    # ax[1].text(0.1, 0.9, "a", transform=ax[1].transAxes, fontsize=12, color='k', fontweight='bold')
    plt.gca().tick_params(axis='both', direction='in')

    plt.savefig(os.path.join(folder, f'tsm.pdf'), bbox_inches='tight', dpi=600)
    plt.show()
    return


def figure_density_dist(folder='/Users/tehan/Documents/TGLC/', recalculate=False):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.12.07_14.30.50.csv'))
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2024.dat')
    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    ng_color = palette[3]
    ng_corr_color = palette[2]
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0.1})
    for spine in ax.spines.values():
        spine.set_zorder(5)
    ground = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 464300749, 151483286,
               335590096,
               193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
               395393265, 310002617, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137,
               243641947,
               419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
               240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
               239816546, 361343239] +
              [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
               268301217, 455784423] +
              [452006073, 306648160, 165464482, 23769326, 470171739,
               166184428, 259172249, 69356857, 58825110, 154220877,
               119585136, 388076422, 178709444, 241249530, 446549906,
               269333648, 401125028, 439366538])

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 219016883, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 149845414, 293954617, 256722647, 280206394, 468574941,
                  29960110, 141488193, 106402532, 392476080, 158588995, 49428710, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 300038935, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 391903064, 237913194, 160390955, 257060897, 365102760,
                  393818343, 153065527, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  120826158, 99869022, 456862677, 219850915, 380887434, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 391949880, 437011608, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 58542531, 102734241, 268334473,
                  159418353, 18318288, 219857012, 35009898, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 34077285, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  166648874, 368287008, 389900760, 159781361, 21832928, 8348911, 289164482, 158241252, 467651916,
                  201177276, 307958020, 382602147, 317548889, 268532343, 407591297, 1167538, 328081248, 328934463,
                  429358906, 37749396, 305424003, 63898957])

    delta_R = 0.06011182562150113
    mass_g = []
    mass_g_err = []
    r_g = []
    r_g_err = []
    density_g = []
    density_g_err = []
    mass_ng = []
    mass_ng_err = []
    r_ng = []
    r_ng_corr = []
    r_ng_err = []
    r_ng_corr_err = []
    density_ng = []
    density_ng_err = []
    density_ng_corr = []
    density_ng_corr_err = []
    tic_ng = []
    for i, tic in enumerate(tics):
        if t['pl_bmassjlim'][i] == 0:
            density, density_err = mass_radius_to_density(t['pl_bmasse'][i], t['pl_rade'][i],
                                                          (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2,
                                                          (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
            if tic in ground:
                if t['pl_bmassprov'][i] == 'Mass':
                    # if ((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2 / t['pl_bmasse'][i] < 0.25 and
                    #         (t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2 / t['pl_rade'][i] < 0.25):
                    mass_g.append(t['pl_bmasse'][i])
                    mass_g_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i]) / 2)
                    r_g.append(t['pl_rade'][i])
                    r_g_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i]) / 2)
                    density_g.append(density)
                    density_g_err.append(density_err)
            elif tic in no_ground:
                if t['pl_bmassprov'][i] == 'Mass':
                    # ## overall shift ###
                    # density_ng_corr.append(density * (1 - delta_R) ** 3)
                    ### individual fits ###
                    ror = []
                    ror_err = []
                    for j in range(len(difference_tglc)):
                        if tics_fit[j] == tic and difference_tglc['rhat'][j] == 1.0:
                            ror.append(float(difference_tglc['value'][j]))
                            ror_err.append(float((difference_tglc['err1'][j] - difference_tglc['err2'][j])/2))
                    ror = np.nanmedian(np.array(ror))
                    ror_err = np.nanmedian(np.array(ror_err))

                    # print(ror is np.nan)
                    density_corr, density_corr_err = mass_radius_to_density(t['pl_bmasse'][i], 109.076 * ror * t['st_rad'][i],
                                                            (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2,
                                                            109.076 * ror * (t['st_raderr1'][i] - t['st_raderr2'][i])/2)
                    delta_Rstar = 109.076 * (t['st_raderr1'][i] - t['st_raderr2'][i]) / 2
                    delta_ror = ror_err
                    if ror is not None and not np.isnan(ror):
                        # if (t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2 / t['pl_bmasse'][i] < 0.25 and np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2)/ (109.076 * ror * t['st_rad'][i]) < 0.25:
                        mass_ng.append(t['pl_bmasse'][i])
                        mass_ng_err.append((t['pl_bmasseerr1'][i] - t['pl_bmasseerr2'][i])/2)
                        r_ng.append(t['pl_rade'][i])
                        r_ng_corr.append(109.076 * ror * t['st_rad'][i])
                        density_ng.append(density)
                        density_ng_err.append(density_err)
                        density_ng_corr.append(density_corr)
                        density_ng_corr_err.append(density_corr_err)
                        tic_ng.append(str(tic))
                        r_ng_err.append((t['pl_radeerr1'][i] - t['pl_radeerr2'][i])/2)
                        r_ng_corr_err.append(np.sqrt((109.076 * t['st_rad'][i] * delta_ror) ** 2 + (ror * delta_Rstar) ** 2))
            # if density > 10:
            #     print(tic)
            # if tic in ground:
            #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.9, marker='o', zorder=1, s=10, color=g_color)
            # elif tic in no_ground:
            #     ax.scatter(t['pl_bmasse'][i], density, alpha=0.5, marker='o', zorder=2, s=10, color='gray')
            #     ax.scatter(t['pl_bmasse'][i], density / 1.21, alpha=0.9, marker='o', zorder=3, s=15,
            #             color=ng_color)
            #     ax.plot([t['pl_bmasse'][i], t['pl_bmasse'][i]], [density, density / 1.21], color='gray', zorder=1,
            #             marker='', alpha=0.5,)
    radius_cut = 6
    d_ng = np.array(density_ng)[(np.array(r_ng) < radius_cut)]
    d_ng_err = np.array(density_ng_err)[(np.array(r_ng) < radius_cut)]
    d_ng_corr = np.array(density_ng_corr)[(np.array(r_ng) < radius_cut)]
    d_ng_corr_err = np.array(density_ng_corr_err)[(np.array(r_ng) < radius_cut)]
    d_g = np.array(density_g)[(np.array(r_g) < radius_cut)]
    d_g_err = np.array(density_g_err)[(np.array(r_g) < radius_cut)]
    # Plot raw histograms for visualization
    # plt.hist(np.array(density_g)[(np.array(r_g) < 4)], bins=np.linspace(0, 1.5, 31), alpha=0.4, label="Raw")
    # plt.legend()
    # plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1.5, 31)

    # First histogram (Raw)
    axs[0].hist(np.concatenate([d_g, d_ng]), bins=bins, weights=np.concatenate([1 / d_g_err, 1 / d_ng_err]),
                alpha=0.4, label=["d_g", "d_ng"])
    axs[0].legend()
    axs[0].set_title("Raw")

    # Second histogram (Corrected)
    axs[1].hist(np.concatenate([d_g, d_ng_corr]), bins=bins, weights=np.concatenate([1 / d_g_err, 1 / d_ng_corr_err]), alpha=0.4,
                label=["d_g", "d_ng_corr"])
    axs[1].legend()
    axs[1].set_title("Corrected")

    plt.tight_layout()
    plt.show()
    # return
    print(len(d_ng_corr))

    # # Log-likelihood function
    # def log_likelihood(params, data):
    #     w1, w2, mu1, mu2, mu3, sigma1, sigma2, sigma3 = params
    #     w3 = 1 - w1 - w2  # Ensure weights sum to 1
    #     if w3 < 0 or sigma1 <= 0 or sigma2 <= 0 or sigma3 <= 0:
    #         return -np.inf  # Invalid parameters
    #     pdf = (
    #             w1 * norm.pdf(data, mu1, sigma1) +  # Gaussian 1
    #             w2 * norm.pdf(data, mu2, sigma2) +  # Gaussian 2
    #             w3 * norm.pdf(data, mu3, sigma3)  # Gaussian 3
    #     )
    #     return np.sum(np.log(pdf + 1e-10))  # Add small constant to avoid log(0)
    #
    # # Log-prior function
    # def log_prior(params):
    #     w1, w2, mu1, mu2, mu3, sigma1, sigma2, sigma3 = params
    #     w3 = 1 - w1 - w2  # Ensure weights sum to 1
    #     if (0.01 <= w1 <= 0.99 and 0.01 <= w2 <= 0.4 and
    #             w3 >= 0 and
    #             0.2 <= mu1 <= 0.4 and 0.3 <= mu2 <= 0.55 and 0.55 <= mu3 <= 1.2 and
    #             0. <= sigma1 <= 0.05 and 0. <= sigma2 <= 0.05 and 0.01 <= sigma3 <= 0.1):
    #         return 0.0  # Uniform prior
    #     return -np.inf  # Outside bounds
    #
    # # Posterior probability function
    # def log_posterior(params, data):
    #     lp = log_prior(params)
    #     if not np.isfinite(lp):
    #         return -np.inf
    #     return lp + log_likelihood(params, data)
    #
    # # Initial parameter guess
    # initial_params = [0.25, 0.25,  # Weights
    #                   0.3, 0.4, 0.65,  # Means
    #                   0.03, 0.03, 0.08]  # Standard deviations
    #
    # # Setting up the MCMC sampler
    # ndim = len(initial_params)
    # nwalkers = 32  # Number of walkers
    # nsteps = 5000  # Number of steps
    #
    # # Initialize walkers around the initial guess
    # initial_guess = np.array(initial_params) + 0.01 * np.random.randn(nwalkers, ndim)
    #
    # # Run the MCMC sampler
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(d_ng_corr,))
    # sampler.run_mcmc(initial_guess, nsteps, progress=True)
    #
    # # Analyze results
    # samples = sampler.get_chain(discard=10000, thin=10, flat=True)  # Flatten the chain
    # w1, w2, mu1, mu2, mu3, sigma1, sigma2, sigma3 = np.median(samples, axis=0)
    # w3 = 1 - w1 - w2  # Ensure weights sum to 1
    # print(w1, w2, w3, mu1, mu2, mu3, sigma1, sigma2, sigma3)
    #
    # # Generate PDFs
    # x = np.linspace(d_ng_corr.min() - 1, d_ng_corr.max() + 1, 1000)
    # pdf1 = w1 * norm.pdf(x, mu1, sigma1)  # Gaussian 1
    # pdf2 = w2 * norm.pdf(x, mu2, sigma2)  # Gaussian 2
    # pdf3 = w3 * norm.pdf(x, mu3, sigma3)  # Gaussian 3
    # total_pdf = pdf1 + pdf2 + pdf3
    #
    # # Plot data and fitted model
    # plt.hist(d_ng_corr, bins=np.linspace(0, 1.5, 31), density=True, alpha=0.6, color='gray', label="Data")
    # plt.plot(x, pdf1, label=f'Gaussian 1: μ={mu1:.2f}, σ={sigma1:.2f}')
    # plt.plot(x, pdf2, label=f'Gaussian 2: μ={mu2:.2f}, σ={sigma2:.2f}')
    # plt.plot(x, pdf3, label=f'Gaussian 3: μ={mu3:.2f}, σ={sigma3:.2f}')
    # plt.plot(x, total_pdf, 'k-', lw=1, label='Total Mixture Model')
    # plt.xlim(0, 1.5)
    # plt.xlabel(r"Planet Density $(\rho_{\oplus})$")
    # plt.ylabel("Probability Density")
    # plt.legend()
    # plt.title("Fitted Gaussian Mixture Model (MCMC)")
    # plt.savefig(os.path.join(folder, f'density_distribution.pdf'), bbox_inches='tight', dpi=600)
    # plt.show()
    #
    # # Create a corner plot
    # labels = ["w1", "w2", "mu1", "mu2", "mu3", "sigma1", "sigma2", "sigma3"]
    # fig = corner.corner(samples, labels=labels, truths=[w1, w2, mu1, mu2, mu3, sigma1, sigma2, sigma3])
    # plt.show()

    return


if __name__ == '__main__':
    # figure_radius_bias(folder='/Users/tehan/Documents/TGLC/')
    figure_radius_bias_ecc(folder='/Users/tehan/Documents/TGLC/')
    # figure_radius_bias_split(folder='/Users/tehan/Documents/TGLC/')
    # figure_mr_mrho(recalculate=True)
    # figure_mr_mrho_all(recalculate=False)
    # figure_mr_mrho_save_param(recalculate=True)


    # figure_tsm(recalculate=True)
    # figure_1_collect_result(folder='/home/tehan/data/pyexofits/Data/', r1=0.01, param='pl_ratror', cmap='Tmag', pipeline='TGLC')
    # figure_2_collect_result(folder='/Users/tehan/Documents/TGLC/')
    # fetch_contamrt(folder='/home/tehan/data/cosmos/transit_depth_validation_contamrt/')
    # figure_4(folder='/Users/tehan/Documents/TGLC/')
    # figure_radius_bias_per_planet(folder='/Users/tehan/Documents/TGLC/')
    # figure_density_dist(recalculate=True)
    # figure_4_tglc_contamrt_trend(recalculate=True)
    # figure_5(type='phase-fold')
    # figure_9(recalculate=True)
    # figure_10(recalculate=True)
    # figure_11(recalculate=True)
    # combine_contamrt()

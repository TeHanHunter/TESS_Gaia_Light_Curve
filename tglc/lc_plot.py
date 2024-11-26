import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
from scipy.stats import bootstrap, ks_2samp

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # Use Computer Modern (serif font)


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


def figure_1(folder='/home/tehan/Downloads/Data/', param='pl_rade', r1=0.01, r2=0.4, cmap='Tmag', pipeline='TGLC'):
    param_dict = {'pl_rade': 'r_pl__0', 'pl_ratror': 'ror__0'}
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
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
    print('missing stars:', missed_stars)
    colormap = cm.viridis
    norm = plt.Normalize(t_[cmap].min(), t_[cmap].max())
    scatter = plt.scatter(t_[f'{param}'], t_['value'], c=t_[cmap], cmap=colormap, facecolors='none', s=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    for k in range(len(t_)):
        if t_['rhat'][k] < 1.05:
            ax.errorbar(t_[f'{param}'][k], t_['value'][k], xerr=t_[f'{param}err1'][k],
                        yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]], fmt='o', mec=colormap(norm(t_[cmap][k])),
                        mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=10, elinewidth=1, capsize=0.7, alpha=0.5,
                        zorder=2)
        # else:
        #     plt.errorbar(t_[f'{param}'][k], t_['value'][k], yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]],
        #                  fmt='o', mec='silver', mfc='none', ecolor='silver',
        #                  ms=10, elinewidth=1, capsize=5, alpha=0.8, zorder=1)
    range_zoom = [0.07, 0.12]
    axins = inset_axes(ax, width='35%', height='35%', loc='lower right', borderpad=2)
    for k in range(len(t_)):
        if t_['rhat'][k] < 1.05:
            axins.errorbar(t_[f'{param}'][k], t_['value'][k], xerr=t_[f'{param}err1'][k],
                           yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]], fmt='o', mec=colormap(norm(t_[cmap][k])),
                           mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=5, elinewidth=1, capsize=0.7, alpha=0.5,
                           zorder=2)
    axins.set_xlim(range_zoom)
    axins.set_ylim(range_zoom)
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_xticks([0.07, 0.08, 0.09, 0.1, 0.12])
    axins.set_xticklabels(['0.07', '', '', '0.1', '0.12'])
    axins.set_yticks([0.07, 0.08, 0.09, 0.1, 0.12])
    axins.set_yticklabels(['0.07', '', '', '0.1', '0.12'])
    axins.plot([0.01, 0.4], [0.01, 0.4], 'k', zorder=0)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='dashed')
    plt.colorbar(scatter, ax=ax, label='TESS magnitude')
    ax.plot([0.01, 0.4], [0.01, 0.4], 'k', zorder=0)
    ax.set_xlim(r1, r2)
    ax.set_ylim(r1, r2)
    ax.set_xlabel(r'Literature $R_p/R_*$')
    ax.set_ylabel(rf'{pipeline}-only fit $R_p/R_*$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(os.path.join(folder, f'{param}_diagonal_{pipeline}.png'), bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure(figsize=(5, 5))
    # np.save(f'deviation_{pipeline}.npy', np.array(t_['value'] - t_[f'{param}']))
    # t_.write(f'deviation_{pipeline}.dat', format='ascii.csv')
    difference_qlp = ascii.read('deviation_QLP.dat')
    difference_tglc = ascii.read('deviation_TGLC.dat')
    plt.hist(difference_tglc, edgecolor='C0', histtype='step', linewidth=1.2, bins=np.arange(-0.1, 0.1, 0.005))
    plt.xlabel(r'fit $R_p/R_*$ - Literature $R_p/R_*$')
    plt.ylabel(r'Number of stars')
    median_value = np.median(difference_tglc)
    print(np.median(np.abs(difference_tglc)))
    print(len(np.where(difference_tglc < 0)[0]) / len(difference_tglc))
    percentage = 68
    lower_bound = np.percentile(difference_tglc, (100 - percentage) / 2)
    upper_bound = np.percentile(difference_tglc, 100 - (100 - percentage) / 2)
    print(median_value, lower_bound, upper_bound)
    # plt.vlines(lower_bound, ymin=0, ymax=250, color='C0', linestyle='dashed')
    plt.vlines(median_value, ymin=0, ymax=275, color='C0')
    # plt.vlines(np.mean(difference), ymin=0,ymax=225, color='r')
    # plt.vlines(upper_bound, ymin=0, ymax=250, color='C0', linestyle='dashed')

    plt.savefig(os.path.join(folder, f'{param}_hist.png'), bbox_inches='tight', dpi=600)
    plt.close()

    # plt.figure(figsize=(5, 5))
    # percent_err = (t_['err1'] - t_['err2']) / 2 / t_['value']
    # plt.scatter(t_['Tmag'], percent_err, c='k', s=1)
    # sort = np.argsort(np.array(t_['Tmag']))
    # plt.plot(np.array(t_['Tmag'][sort])[12:-11], np.convolve(percent_err[sort], np.ones(25)/25, mode='valid'))
    # plt.ylim(0,1)
    # plt.xlabel('Tmag')
    # plt.ylabel(r'Percent uncertainty on $R_p/R_*$')
    # plt.savefig(os.path.join(folder, f'{param}_error_{pipeline}.png'), bbox_inches='tight', dpi=600)


def figure_2(folder='/home/tehan/Downloads/Data/', ):
    palette = sns.color_palette('colorblind')
    tglc_color = palette[3]
    qlp_color = palette[2]
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
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        if star_sector in d_qlp['Star_sector']:
            difference_tglc.add_row(d_tglc[i])
            difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    # difference_qlp.write(f'deviation_QLP_common.dat', format='ascii.csv')
    # difference_tglc.write(f'deviation_TGLC_common.dat', format='ascii.csv')
    print(len(difference_tglc))
    print(len(difference_qlp))
    # average 491 lcs
    print(np.mean(difference_tglc['pl_ratrorerr1'] / difference_tglc['pl_ratror']))
    # average 160 hosts
    print(np.mean(list(set(difference_tglc['pl_ratrorerr1'].tolist() / difference_tglc['pl_ratror']))))
    difference = vstack([difference_tglc, difference_qlp])
    difference['diff'] = (difference['value'] - difference['pl_ratror']) / difference['pl_ratror']
    difference['Tmag_int'] = np.where(difference['Tmag'] < 12.5, r'$T<12.5$', r'$T>12.5$')
    print(len(np.where(difference['Tmag'] < 12.5)[0]) / 2)
    # An outlier of TGLC for <12.5 is making the plot looks clumpy. That single point is removed, but will not affect the statistics.
    print(difference[np.where(difference['diff'] == np.max(difference['diff']))[0][0]])
    difference.remove_row(np.where(difference['diff'] == np.max(difference['diff']))[0][0])
    df = difference.to_pandas()
    plt.figure(figsize=(6, 6))
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    # sns.violinplot(data=df, x='diff', y='pipeline', bw_adjust=1, palette="Set1")
    # print(np.sort(difference['diff'][(difference['Tmag_int'] == '$T<12.5$') & (difference['Photometry'] == 'TGLC')]))
    sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.8, gap=.04, alpha=0.6,
                   gridsize=500, width=1.2,
                   palette=[tglc_color, qlp_color])
    plt.vlines(0, ymin=-0.5, ymax=1.5, color='k', ls='dashed')
    plt.xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    plt.ylabel('')
    plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
               [r'$-60\%$', r'$-40\%$', r'$-20\%$', r'$0\%$', r'$20\%$', r'$40\%$', r'$60\%$'])
    plt.yticks(rotation=90)
    # plt.xlim(-0.05, 0.05)
    plt.xlim(-0.75, 0.75)
    # plt.ylim(-1,2)
    plt.title('Exoplanet radius ratio fit')
    plt.savefig(os.path.join(folder, f'ror_ratio_violin.pdf'), bbox_inches='tight', dpi=600)
    plt.show()

    # TGLC data
    rows = np.where(difference_tglc['Tmag'] > 12.5)[0]
    difference_tglc_values = (difference_tglc['value'][rows] - difference_tglc['pl_ratror'][rows]) / \
                             difference_tglc['pl_ratror'][rows]
    median_value_tglc = np.median(difference_tglc_values)
    q1_tglc = np.percentile(difference_tglc_values, 25)
    q3_tglc = np.percentile(difference_tglc_values, 75)
    iqr_tglc = (median_value_tglc - q1_tglc, q3_tglc - median_value_tglc)
    negative_ratio_tglc = len(np.where(difference_tglc_values < 0)[0]) / len(difference_tglc)

    print("TGLC Median:", median_value_tglc)
    print("TGLC IQR:", f"{median_value_tglc} - {iqr_tglc[0]} + {iqr_tglc[1]}")
    print("TGLC Negative Ratio:", negative_ratio_tglc)

    # QLP data
    rows = np.where(difference_tglc['Tmag'] > 12.5)[0]
    difference_qlp_values = (difference_qlp['value'][rows] - difference_qlp['pl_ratror'][rows]) / \
                            difference_qlp['pl_ratror'][rows]
    median_value_qlp = np.median(difference_qlp_values)
    q1_qlp = np.percentile(difference_qlp_values, 25)
    q3_qlp = np.percentile(difference_qlp_values, 75)
    iqr_qlp = (median_value_qlp - q1_qlp, q3_qlp - median_value_qlp)
    negative_ratio_qlp = len(np.where(difference_qlp_values < 0)[0]) / len(difference_qlp)

    print("QLP Median:", median_value_qlp)
    print("QLP IQR:", f"{median_value_qlp} - {iqr_qlp[0]} + {iqr_qlp[1]}")
    print("QLP Negative Ratio:", negative_ratio_qlp)


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
    difference_values = values - pl_ratror
    errors_ratio = np.sqrt(errors_value ** 2)

    # Compute inverse variance weighted mean
    weights = 1 / (errors_ratio ** 2)
    weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    weighted_mean_error = np.sqrt(1 / np.sum(weights))
    return difference_values, errors_ratio, weighted_mean, weighted_mean_error


def compute_weighted_mean_bootstrap(data):
    fit_values = data['value']
    pl_ratror = data['pl_ratror']
    errors_value = (data['err1'] - data['err2']) / 2
    errors_pl_ratror = (data['pl_ratrorerr1'] - data['pl_ratrorerr2']) / 2

    # correct literature with 0 error
    for i in range(len(errors_pl_ratror)):
        if errors_pl_ratror[i] == 0:
            errors_pl_ratror[i] = errors_value[i]
            # print(errors_pl_ratror[i])
    # Compute the ratio and its propagated error
    difference_values = fit_values - pl_ratror
    errors_ratio = np.sqrt(errors_value ** 2)

    # errors_ratio = np.sqrt(errors_value ** 2 + errors_pl_ratror ** 2)
    # errors_ratio = np.ones(len(errors_pl_ratror))
    # errors_ratio = errors_value

    # Compute inverse variance weighted mean
    def weighted_mean(values_, weights_):
        return np.sum(values_ * weights_) / np.sum(weights_)

    weights = 1 / (errors_ratio ** 2)

    def weighted_mean_stat(values_):
        return weighted_mean(values_, weights)

    # plt.figure()
    # plt.plot(np.sort(weights), '.')
    # plt.show()
    # weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    # weighted_mean_error = np.sqrt(1 / np.sum(weights))
    res = bootstrap((difference_values,), weighted_mean_stat, confidence_level=.95, n_resamples=10000,
                    method='percentile')
    iw_mean = weighted_mean(difference_values, weights)
    ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    print(f"Inverse-Variance Weighted Mean: {iw_mean}")
    print(f"95% Confidence Interval: ({ci_low}, {ci_high})")
    print(res.standard_error)
    return iw_mean, res.standard_error, res.standard_error
    # return iw_mean, (iw_mean - ci_low)/2, (ci_high - iw_mean)/2


def figure_4(folder='/home/tehan/Downloads/Data/', ):
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
    # difference_tglc.write(f'deviation_TGLC_677.dat', format='ascii.csv')
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    # QLP data (dim)
    diff_qlp, errors_qlp, weighted_mean_qlp, weighted_mean_error_qlp = compute_weighted_mean_all(difference_qlp)
    iw_mean_qlp, ci_low_qlp, ci_high_qlp = compute_weighted_mean_bootstrap(difference_qlp)
    print(len(difference_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    ax[0].hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
               weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
               color=qlp_color, alpha=0.6, edgecolor=None)

    ax[0].hist(diff_tglc, bins=np.linspace(-0.05, 0.05, 41),
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
    ax[1].hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
               weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
               color=qlp_color, alpha=0.6, edgecolor=None)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.05, 0.05, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=tglc_color, alpha=0.6, edgecolor=None)
    ax[1].set_title(f'TESS and Ground-based radius ({len(difference_tglc)} light curves)')
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

    plt.xlim(-0.03, 0.03)

    plt.savefig(os.path.join(folder, f'ror_ground_vs_no_ground.pdf'), bbox_inches='tight', dpi=600)
    plt.show()


def figure_4_tglc(folder='/home/tehan/Downloads/Data/', contamrt_min=0.0):
    contamrt = ascii.read('/Users/tehan/Documents/TGLC/contamination_ratio.dat')
    print(len(set(contamrt['tic_sec'])))
    palette = sns.color_palette('bright')
    tglc_color = 'C1'
    # qlp_color = 'C0'
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'hspace': 0.1})
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
    ground = [156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
              445751830, 7548817, 86263325, 155867025, 198008005, 178162579, 289661991, 464300749, 151483286, 335590096,
              17865622, 193641523, 396562848, 447061717, 124379043, 44792534, 150098860, 179317684, 124029677, 95660472,
              395393265, 310002617, 220076110, 20182780, 70524163, 95057860, 376524552, 394050135, 409794137, 243641947,
              419411415, 281408474, 460984940, 68007716, 39414571, 8599009, 33595516, 458419328, 336128819, 417646390,
              240823272, 147977348, 144700903, 258920431, 280655495, 66561343, 16005254, 375506058, 279947414,
              239816546, 361343239] + [90850770, 97568467, 263179590, 194795551, 139375960, 100389539, 250111245,
                                       268301217, 455784423]
    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        try:
            if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
                if int(star_sector.split('_')[1]) in ground:
                    difference_tglc.add_row(d_tglc[i])
        except IndexError:
            print(star_sector)
            # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    # difference_tglc.write(f'deviation_TGLC_677.dat', format='ascii.csv')
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_ground = diff_tglc
    iw_mean_tglc_ground = iw_mean_tglc
    print(len(difference_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax[0].hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)

    ax[0].hist(diff_tglc, bins=np.linspace(-0.05, 0.05, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=tglc_color, alpha=0.6, edgecolor=None)
    ax[0].set_title(f'Ground-based-only radius ({len(difference_tglc)} light curves)')
    ax[0].scatter(iw_mean_tglc, 2.75, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='TGLC')
    ax[0].errorbar(iw_mean_tglc, 1.5, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    # ax[0].scatter(iw_mean_qlp, 2.6, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax[0].errorbar(iw_mean_qlp, 1.6, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)

    ax[0].vlines(0, ymin=0, ymax=60, color='k', ls='dashed', lw=1, zorder=3)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Error Weighted Counts')
    ax[0].legend(loc='upper right')
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
                 264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] + [
                    370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                    464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                    320004517, 163539739, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                    408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                    407126408, 55650590, 335630746, 55525572, 342642208, 394357918]

    for i in range(len(d_tglc)):
        star_sector = d_tglc['Star_sector'][i]
        try:
            if contamrt['contamrt'][np.where(contamrt['tic_sec'] == star_sector)[0][0]] > contamrt_min:
                if int(star_sector.split('_')[1]) in no_ground:
                    difference_tglc.add_row(d_tglc[i])
        except IndexError:
            pass
                    # difference_qlp.add_row(d_qlp[np.where(d_qlp['Star_sector'] == star_sector)[0][0]])
    diff_tglc, errors_tglc, weighted_mean_tglc, weighted_mean_error_tglc = compute_weighted_mean_all(difference_tglc)
    iw_mean_tglc, ci_low_tglc, ci_high_tglc = compute_weighted_mean_bootstrap(difference_tglc)
    diff_tglc_no_ground = diff_tglc
    iw_mean_tglc_no_ground = iw_mean_tglc
    print(len(difference_tglc))
    # sns.violinplot(data=df, x="diff", y="Tmag_int", hue="Pipeline", split=True, bw_adjust=.6, gap=.04, alpha=0.6,
    #                gridsize=500, width=1.2, palette=[tglc_color, qlp_color])
    # ax[1].hist(diff_qlp, bins=np.linspace(-0.05, 0.05, 41),
    #            weights=(1 / errors_qlp ** 2) * len(diff_qlp) / np.sum(1 / errors_qlp ** 2),
    #            color=qlp_color, alpha=0.6, edgecolor=None)

    ax[1].hist(diff_tglc, bins=np.linspace(-0.05, 0.05, 41),
               weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
               color=tglc_color, alpha=0.6, edgecolor=None)
    ax[1].set_title(f'TESS and Ground-based radius ({len(difference_tglc)} light curves)')
    ax[1].scatter(iw_mean_tglc, 5.5, marker='v', color=tglc_color, edgecolors='k', linewidths=0.7, s=50,
                  zorder=4, label='TGLC')
    ax[1].errorbar(iw_mean_tglc, 3, xerr=[[ci_low_tglc], [ci_high_tglc]], ecolor='k',
                   elinewidth=1, capsize=3, zorder=2, )
    # ax[1].scatter(iw_mean_qlp, 6.8, marker='v', color=qlp_color, edgecolors='k', linewidths=0.7, s=50,
    #               zorder=3, label='QLP')
    # ax[1].errorbar(iw_mean_qlp, 4, xerr=[[iw_mean_qlp-ci_low_qlp], [ci_high_qlp-iw_mean_qlp]], ecolor='k',
    #                elinewidth=1,capsize=3, zorder=2,)
    ax[1].vlines(0, ymin=0, ymax=120, color='k', ls='dashed', lw=1, zorder=3)
    ax[1].set_xlabel(r'$\Delta(R_{\text{p}}/R_*)$')
    ax[1].set_ylabel('Error Weighted Counts')
    ax[1].legend(loc='upper right')
    ax[1].set_xticks([-0.02, -0.01, 0, 0.01, 0.02, 0.06], )
    plt.xlim(-0.025, 0.025)

    stat, p_value = ks_2samp(diff_tglc_ground, diff_tglc_no_ground)
    print(f"K-S Statistic: {stat}")
    print(f"P-value: {p_value}")
    plt.savefig(os.path.join(folder, f'ror_ground_vs_no_ground_TGLC.pdf'), bbox_inches='tight', dpi=600)
    plt.show()

    return iw_mean_tglc_ground, iw_mean_tglc_no_ground


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


if __name__ == '__main__':
    # figure_1(folder='/home/tehan/Downloads/Data_qlp/', r1=0.01, param='pl_ratror', cmap='Tmag', pipeline='QLP')
    # fetch_contamrt(folder='/home/tehan/data/cosmos/transit_depth_validation_contamrt/')
    # figure_4(folder='/Users/tehan/Documents/TGLC/')
    figure_4_tglc(folder='/Users/tehan/Documents/TGLC/', contamrt_min=0.)
    # figure_5(type='phase-fold')

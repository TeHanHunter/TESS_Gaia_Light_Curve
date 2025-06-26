import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import emcee
import corner
from collections import Counter

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
from scipy.odr import ODR, Model, RealData, Data
from scipy.stats import multivariate_normal
# from pr_main import Fit
from uncertainties import ufloat
from rapidfuzz import process, fuzz

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


def compute_weighted_mean_all(data):
    values = data['value']
    pl_ratror = data['pl_ratror']
    errors_value = (data['err1'] - data['err2']) / 2
    errors_pl_ratror = (data['pl_ratrorerr1'] - data['pl_ratrorerr2']) / 2
    values = [ufloat(v, e) for v, e in zip(values, errors_value)]
    pl_ratror = [ufloat(r, e if e > 0 else ve) for r, e, ve in zip(pl_ratror, errors_pl_ratror, errors_value)]

    diff_frac = [(v - p) / v for v, p in zip(values, pl_ratror)]
    diff_vals = np.array([d.n for d in diff_frac])
    diff_errs = np.array([d.s for d in diff_frac])

    weights = 1 / diff_errs ** 2
    weighted_mean = np.sum(diff_vals * weights) / np.sum(weights)
    weighted_mean_err = np.sqrt(1 / np.sum(weights))

    # # correct literature with 0 error
    # for i in range(len(errors_pl_ratror)):
    #     if errors_pl_ratror[i] == 0:
    #         errors_pl_ratror[i] = errors_value[i]
    #         # print(errors_pl_ratror[i])
    # # Compute the ratio and its propagated error
    # difference_values = (values - pl_ratror) / values
    # errors_ratio = np.sqrt(errors_value ** 2)
    # # errors_ratio = np.ones(len(errors_ratio))
    # # Compute inverse variance weighted mean
    # weights = 1 / (errors_ratio ** 2)
    # weighted_mean = np.sum(difference_values * weights) / np.sum(weights)
    # weighted_mean_error = np.sqrt(1 / np.sum(weights))
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
    return diff_vals, diff_errs, weighted_mean, weighted_mean_err


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


def weighted_median(data, weights):
    sorted_idx = np.argsort(data)
    data_sorted = data[sorted_idx]
    weights_sorted = weights[sorted_idx]
    cum_weights = np.cumsum(weights_sorted)
    cutoff = weights_sorted.sum() / 2.0
    return data_sorted[np.searchsorted(cum_weights, cutoff)]


def figure_radius_bias(folder='/Users/tehan/Documents/TGLC/'):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced_cleaned.csv'))
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

    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025_updated.dat', format='csv')
    # d_qlp = difference_qlp[np.where(difference_qlp['rhat'] < 1.1)]
    # d_qlp['Pipeline'] = ['QLP'] * len(d_qlp)
    # print(len(d_qlp))
    d_tglc = difference_tglc[np.where(difference_tglc['rhat'] == 1.0)]
    d_tglc['Pipeline'] = ['TGLC'] * len(d_tglc)
    # print(len(d_tglc))
    # difference_qlp = Table(names=d_qlp.colnames, dtype=[col.dtype for col in d_qlp.columns.values()])
    difference_tglc = Table(names=d_tglc.colnames, dtype=[col.dtype for col in d_tglc.columns.values()])
    ground_old = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 155867025, 198008005, 178162579, 464300749, 151483286,
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
    ground = [16005254, 20182780, 33595516, 44792534, 88992642,
    119585136, 144700903, 150098860, 154220877, 179317684, 193641523,
    243641947, 250111245, 259172249, 271893367, 285048486, 335590096,
    376524552, 388076422, 394050135, 396562848, 409794137, 241249530,
    419411415, 428787891, 445751830, 447061717, 458419328, 460984940,
    464300749]

    ground_diff = list(set(ground_old)-set(ground))

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
    # print(np.sort(diff_tglc))
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
    w_median = weighted_median(diff_tglc, 1 / errors_tglc ** 2)
    print(f"Weighted median: {w_median:.6f}")
    mad = np.median(np.abs(diff_tglc - np.median(diff_tglc)))
    robust_std = 1.4826 * mad  # Approx. conversion to std if data is normal
    fwhm_robust = 2.355 * robust_std
    print(f"Robust std (MAD × 1.4826): {robust_std:.6f}")
    print(f"Robust FWHM ≈ {fwhm_robust:.6f}")
    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            histtype='step', edgecolor=g_color, linewidth=2, zorder=3, alpha=0.95,
               label=r'TESS-free $f_p$' + f'\n({len(difference_tglc)} fits of 29 planets)')
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
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025_updated.dat', format='csv')
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
                  351601843, 24358417, 144193715, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 293954617, 256722647, 280206394, 468574941,
                  29960110, 106402532, 392476080, 158588995, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 237913194, 160390955, 257060897, 365102760,
                  393818343, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  99869022, 456862677, 219850915, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 102734241, 268334473,
                  159418353, 18318288, 219857012, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  368287008, 389900760, 159781361, 21832928, 8348911, 289164482,
                  201177276, 307958020, 382602147, 317548889, 268532343, 1167538, 328081248, 328934463,
                  429358906, 37749396]
                 + [209459275, 130924120, 419523962, 163539739]  # temp
                 + ground_diff)
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
    # print(np.sort(diff_tglc))
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
    w_median = weighted_median(diff_tglc, 1 / errors_tglc ** 2)
    print(f"Weighted median: {w_median:.6f}")
    mad = np.median(np.abs(diff_tglc - np.median(diff_tglc)))
    robust_std = 1.4826 * mad  # Approx. conversion to std if data is normal
    fwhm_robust = 2.355 * robust_std
    print(f"Robust std (MAD × 1.4826): {robust_std:.6f}")
    print(f"Robust FWHM ≈ {fwhm_robust:.6f}")
    ax1.hist(diff_tglc, bins=np.linspace(-0.5, 0.5, 41),
            weights=(1 / errors_tglc ** 2) * len(diff_tglc) / np.sum(1 / errors_tglc ** 2),
            histtype='step', edgecolor=ng_color, linewidth=2, zorder=3, alpha=0.9,
               label=r'TESS-dependent $f_p$ ' + f'\n({len(difference_tglc)} fits of 228 planets)')

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
    # print(np.sort(diff_tglc))
    # print(difference_tglc[np.argsort(diff_tglc)])
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
               label=r'Kepler $f_p$' + f'\n({len(difference_tglc)} fits of 30 planets)')

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

    ax1.vlines(0, ymin=0, ymax=200, color='k', ls='dotted', lw=2, zorder=3)
    ax2.vlines(0, ymin=0, ymax=40, color='k', ls='dotted', lw=2, zorder=3)
    ax2.set_xlabel(r'$f_p \equiv (p_{\text{TGLC}} - p_{\text{lit}}) / p_{\text{TGLC}}$')
    ax1.set_ylabel('Error weighted counts')
    ax2.set_ylabel('Error weighted counts')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax2.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                  [f"{x * 100:.0f}%" for x in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]])
    ax1.text(-0.12, 1, "a", transform=ax1.transAxes, fontsize=12, color='k', fontweight='bold')
    ax2.text(-0.12, 1, "b", transform=ax2.transAxes, fontsize=12, color='k', fontweight='bold')
    ax1.set_ylim(0,200)
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
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_reduced_cleaned.csv'))
    b = t['pl_imppar']
    ror = t['pl_rade'] / t['st_rad'] / 109
    # find grazing
    # for i in range(len(b)):
    #     if 1 - b[i] < ror[i] / 2:
    #         print(t['tic_id'][i])
    #         print(b[i])
    #         print(ror[i])
    difference_tglc = ascii.read(f'{folder}deviation_TGLC_2025_updated.dat')
    tics_fit = [int(tic_sec.split('_')[1]) for tic_sec in difference_tglc['Star_sector']]
    tics = [int(s[4:]) for s in t['tic_id']]
    palette = sns.color_palette('colorblind')
    increase_color = palette[2]
    decrease_color = palette[3]
    ng_color = 'k'
    ng_corr_color = 'k'
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '1', 'grid.color': '0.8'})
    fig, ax_ = plt.subplots(1, 2, sharex=True, figsize=(12, 5), gridspec_kw={'hspace': 0.01, 'wspace': 0.17})
    ax = [ax_[1], ax_[0]]
    for spine in ax[0].spines.values():
        spine.set_zorder(5)
    for spine in ax[1].spines.values():
        spine.set_zorder(5)
    ground_old = ([156648452, 154293917, 271893367, 285048486, 88992642, 454248975, 428787891, 394722182, 395171208,
               445751830, 7548817, 155867025, 198008005, 178162579, 464300749, 151483286,
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
    ground = [16005254, 20182780, 33595516, 44792534, 88992642,
    119585136, 144700903, 150098860, 154220877, 179317684, 193641523,
    243641947, 250111245, 259172249, 271893367, 285048486, 335590096,
    376524552, 388076422, 394050135, 396562848, 409794137, 241249530,
    419411415, 428787891, 445751830, 447061717, 458419328, 460984940,
    464300749]

    ground_diff = list(set(ground_old)-set(ground))

    no_ground = ([428699140, 157698565, 119584412, 262530407, 219854185, 140691463, 237922465,
                  271478281, 29857954, 198485881, 332558858, 376637093, 54002556, 126606859, 231702397, 460205581,
                  351601843, 24358417, 144193715, 445805961, 103633434, 230001847, 70899085, 147950620,
                  219854519, 333657795, 200322593, 287256467, 206541859, 420112589, 261867566, 10837041, 70513361,
                  148673433, 229510866, 321669174, 183120439, 293954617, 256722647, 280206394, 468574941,
                  29960110, 106402532, 392476080, 158588995, 410214986, 441738827, 220479565,
                  172370679, 116483514, 350153977, 37770169, 212957629, 393831507, 207110080, 190496853,
                  404505029, 207141131, 439456714, 394137592, 267263253, 192790476, 169249234, 159873822,
                  394561119, 142394656, 318753380, 422756130, 339672028, 176956893, 348835438, 62483237, 266980320,
                  151825527, 466206508, 288735205, 237104103, 437856897, 73540072, 229742722, 83092282,
                  264678534, 271971130, 204650483, 394918211, 321857016, 290348383, 436873727, 362249359, 372172128] +
                 [370133522, 298663873, 383390264, 329148988, 441462736, 199376584, 257527578, 166527623, 142937186,
                  464646604, 118327550, 234994474, 260004324, 183985250, 349095149, 139285832, 360156606, 200723869,
                  320004517, 89020549, 179034327, 158025009, 333473672, 349576261, 470381900, 218795833,
                  408636441, 76923707, 353475866, 202426247, 387690507, 209464063, 12421862, 296739893, 350618622,
                  407126408, 55650590, 335630746, 55525572, 342642208, 394357918] +
                 [293607057, 332534326, 260708537, 443556801, 52005579, 287145649, 232540264, 404518509, 358070912,
                  352413427, 169765334, 39699648, 305739565, 237913194, 160390955, 257060897, 365102760,
                  393818343, 154872375, 232967440, 154089169, 97766057, 158002130, 22233480, 233087860,
                  99869022, 456862677, 219850915, 232612416, 271169413, 232976128, 49254857,
                  198241702, 282485660, 224297258, 303432813, 198356533, 232982558, 237232044,
                  343628284, 246965431, 417931607, 240968774, 306955329, 219041246, 102734241, 268334473,
                  159418353, 18318288, 219857012, 287080092, 124573851, 289580577, 367858035, 277634430,
                  9348006, 219344917, 21535395, 286916251, 322807371, 142381532, 142387023, 46432937,
                  348755728, 4672985, 91987762, 258514800, 445903569, 71431780, 417931300, 8967242, 441765914,
                  368287008, 389900760, 159781361, 21832928, 8348911, 289164482,
                  201177276, 307958020, 382602147, 317548889, 268532343, 1167538, 328081248, 328934463,
                  429358906, 37749396]
                 + [209459275, 130924120, 419523962, 163539739]  # temp
                 + ground_diff)

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
                            i] < 0.33 and np.sqrt(
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
    print(len(r_g))
    print(len(r_ng))
    marker_size = 25
    # density_weight = np.array(r_ng_err)[np.where(np.array(r_ng)<4)]
    # print(density_weight)
    ### mass-density ###
    ax[0].errorbar(mass_g, density_g, xerr=mass_g_err, yerr=density_g_err, fmt='none', alpha=0.7,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, color=palette[7])

    ax[0].scatter(mass_g, density_g, alpha=0.9, marker='D', zorder=2, s=marker_size-10, color=palette[7], facecolors=palette[7],
                  edgecolors='k', linewidths=0.5)

    # ax[0].errorbar(mass_ng, density_ng, xerr=mass_ng_err, yerr=density_ng_err, fmt='none', alpha=0.5,
    #                zorder=3, capsize=2.5, capthick=0.7, lw=0.7, color=ng_color)
    #
    # ax[0].scatter(mass_ng, density_ng, alpha=0.9, marker='o', zorder=3, s=marker_size, facecolors='none', edgecolors=ng_color)
    #
    ax[0].errorbar(mass_ng, density_ng_corr, xerr=mass_ng_err, yerr=density_ng_corr_err, fmt='none',
                   color=ng_corr_color, lw=0.7, zorder=3, capsize=2, capthick=0.7, alpha=0.8)

    ax[0].scatter(mass_ng, density_ng_corr, facecolors='none', edgecolors=ng_corr_color, alpha=0.9, s=marker_size, zorder=4)
    for i in range(len(density_ng)):
        if density_ng[i] < density_ng_corr[i]:
            ax[0].plot([mass_ng[i], mass_ng[i]], [density_ng[i], density_ng_corr[i]], color=increase_color, zorder=2,
                       marker='', linewidth=4, alpha=0.8)
        elif density_ng[i] > density_ng_corr[i]:
            ax[0].plot([mass_ng[i], mass_ng[i]], [density_ng[i], density_ng_corr[i]], color=decrease_color, zorder=2,
                       marker='', linewidth=4, alpha=0.8)
    # for j in range(len(mass_ng)):
    #     ax[0].text(mass_ng[j], density_ng[j], tic_ng[j], fontsize=6)

    ### mass-radius ###
    ax[1].errorbar(mass_g, r_g, xerr=mass_g_err, yerr=r_g_err, fmt='none', alpha=0.7,
                   zorder=2, capsize=2.5, capthick=0.7, lw=0.7, color=palette[7])
    ax[1].scatter(mass_g, r_g, alpha=0.9, marker='D', zorder=2, s=marker_size-10, color=palette[7], facecolors=palette[7],
                  edgecolors='k', linewidths=0.5)
    # ax[1].errorbar(mass_ng, r_ng, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.5,
    #                zorder=3, capsize=2.5, capthick=0.7, lw=0.7, color=ng_color)
    # ax[1].scatter(mass_ng, r_ng, alpha=0.9, marker='o', zorder=3, s=marker_size, facecolors='none', edgecolors=ng_color)
    ax[1].errorbar(mass_ng, r_ng_corr, xerr=mass_ng_err, yerr=r_ng_err, fmt='none', alpha=0.8,
                   color=ng_corr_color, lw=0.7, zorder=4, capsize=2, capthick=0.7)

    ax[1].scatter(mass_ng, r_ng_corr, facecolors='none', edgecolors=ng_corr_color, alpha=0.9, s=marker_size, zorder=4)
    # for i in range(len(tic_ng)):
    #     ax[1].text(mass_ng[i], r_ng_corr[i], str(tic_ng[i]),
    #                fontsize=6, color=ng_corr_color, alpha=0.9,
    #                ha='left', va='bottom', zorder=5)
    for i in range(len(r_ng)):
        if r_ng[i] < r_ng_corr[i]:
            ax[1].plot([mass_ng[i], mass_ng[i]], [r_ng[i], r_ng_corr[i]], color=increase_color, zorder=2, marker='', linewidth=4, alpha=0.8, )
        elif r_ng[i] > r_ng_corr[i]:
            ax[1].plot([mass_ng[i], mass_ng[i]], [r_ng[i], r_ng_corr[i]], color=decrease_color, zorder=2, marker='', linewidth=4, alpha=0.8, )

    # add manual legend
    ax[1].errorbar(0, 0, xerr=0, yerr=0, fmt='D', alpha=0.9, markerfacecolor=palette[7], markeredgecolor='k', markeredgewidth=0.5,
                   zorder=2, capsize=1.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size-10), color=palette[7], label='TESS-free')
    # ax[0].errorbar(mass_ng, density_ng_corr, xerr=mass_ng_err, yerr=density_ng_corr_err, fmt='o', alpha=0.3,
    #                color=ng_corr_color, lw=0.7, ms=np.sqrt(15), zorder=4, capsize=3, capthick=0.7)
    ax[1].errorbar(0, 0, xerr=0, yerr=0, fmt='o', alpha=0.9,
                   zorder=3, capsize=1.5, capthick=0.7, lw=0.7, ms=np.sqrt(marker_size),
                   color=ng_color, markeredgecolor=ng_color, markerfacecolor='none',
                   label='TESS-dependent TGLC-fitted')
    ax[1].plot([-10, -10], [-10, -5], color=increase_color, zorder=2, marker='',linewidth=4, alpha=0.8,
               label='Increase from literature')
    ax[1].plot([-10, -10], [-10, -5], color=decrease_color, zorder=2, marker='', linewidth=4, alpha=0.8,
               label='Decrease from literature')
    ax[1].legend(loc=2, fontsize=10)

    ### water world ###
    r = np.linspace(1.24, 4, 100)
    mass, rho = zeng_2019_water_world(r)
    # ax[0].plot(mass, r, c=palette[0], zorder=4, label='Water world')
    ax[0].plot(mass, rho, c=palette[0], zorder=1, label='Water world', linewidth=1)
    ax[1].plot(mass, r, c=palette[0], zorder=1, label='Water world', linewidth=1)
    ax[0].text(mass[-1] - 22, rho[-1] - 0.14, 'Water world', color=palette[0], fontweight='bold', fontsize=9,
               ha='center', va='center', zorder=1, rotation=23)
    ax[1].text(mass[-1] - 10, r[-1] + 0.05, 'Water world', color=palette[0], fontweight='bold', fontsize=9,
               ha='center', va='center', zorder=1, rotation=25)
    ### rocky core + H/He atmos ###
    mass = np.linspace(1, 30, 100)
    r, r1, r2, rho, rho1, rho2 = rogers_2023_rocky_core(mass)
    # ax[0].plot(mass, r, c=sns.color_palette('muted')[5], zorder=4, label='Rocky core with H/He atmosphere')
    ax[0].plot(mass, rho, c=sns.color_palette('muted')[5], zorder=1, label='Rocky core with H/He atmosphere',
               linewidth=1)
    ax[0].plot(mass, rho1, c=sns.color_palette('muted')[5], zorder=1, ls='--', linewidth=1)
    ax[0].plot(mass, rho2, c=sns.color_palette('muted')[5], zorder=1, ls='--', linewidth=1)
    ax[1].plot(mass, r, c=sns.color_palette('muted')[5], zorder=1, label='Rocky core with H/He atmosphere', linewidth=1)
    ax[1].plot(mass, r1, c=sns.color_palette('muted')[5], zorder=1, ls='--', linewidth=1)
    ax[1].plot(mass, r2, c=sns.color_palette('muted')[5], zorder=1, ls='--', linewidth=1)
    ax[0].text(mass[0] + 0.95, rho[0] + 0.01, 'Rocky+atmosphere', color=sns.color_palette('muted')[5],
               fontweight='bold', fontsize=9, ha='center', va='center', zorder=1, rotation=11)
    ax[1].text(mass[0] + 0.9, r[0] + 0.35, 'Rocky+atmosphere', color=sns.color_palette('muted')[5],
               fontweight='bold', fontsize=9, ha='center', va='center', zorder=1, rotation=27)

    ### Earth-like ###
    mass = np.linspace(1, 30, 100)
    rho = owen_2017_earth_core(mass)
    r = (rho / mass) ** (-1 / 3)
    ax[0].plot(mass, rho, c='r', zorder=1, label='Earth-like', linewidth=1)
    ax[1].plot(mass, r, c='r', zorder=1, label='Earth-like', linewidth=1)
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



if __name__ == '__main__':
    # Figure 1
    figure_radius_bias(folder='/Users/tehan/Documents/TGLC/')
    # Figure 2
    figure_mr_mrho(recalculate=True)

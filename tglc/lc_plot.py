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
import pkg_resources
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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


def figure_1(folder='/home/tehan/Downloads/Data/', param='pl_rade', r1=0.01, r2=0.4, cmap='Tmag'):
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
    axins.set_xticks([0.07,0.08,0.09, 0.1, 0.12])
    axins.set_xticklabels(['0.07','','', '0.1', '0.12'])
    axins.set_yticks([0.07,0.08,0.09, 0.1, 0.12])
    axins.set_yticklabels(['0.07','','', '0.1', '0.12'])
    axins.plot([0.01, 40], [0.01, 40], 'k', zorder=0)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle='dashed')
    plt.colorbar(scatter, ax=ax, label='TESS magnitude')
    ax.plot([0.01, 40], [0.01, 40], 'k', zorder=0)
    ax.set_xlim(r1, r2)
    ax.set_ylim(r1, r2)
    ax.set_xlabel(r'Literature $R_p/R_*$')
    ax.set_ylabel(r'TGLC-only fit $R_p/R_*$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(os.path.join(folder, f'{param}_diagonal.png'), bbox_inches='tight', dpi=600)


def figure_2(folder='/home/tehan/Downloads/Data/', param='pl_rade', r=25, cmap='Tmag'):
    param_dict = {'pl_rade': 'r_pl__0', 'pl_ratror': 'ror__0'}
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
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
    plt.figure(figsize=(20, 8))
    colormap = cm.viridis
    norm = plt.Normalize(t_[cmap].min(), t_[cmap].max())
    scatter = plt.scatter(t_[f'{param}'], t_['value'], c=t_[cmap], cmap=colormap, facecolors='none', s=0)
    for k in range(len(t_)):
        if t_['rhat'][k] < 1.05:
            plt.errorbar(t_[f'{param}'][k], t_['value'][k], xerr=t_[f'{param}err1'][k],
                         yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]], fmt='o', mec=colormap(norm(t_[cmap][k])),
                         mfc='none', ecolor=colormap(norm(t_[cmap][k])), ms=20, elinewidth=1, capsize=0.7, alpha=0.5,
                         zorder=2)
        # else:
        #     plt.errorbar(t_[f'{param}'][k], t_['value'][k], yerr=[[t_['err2'][k] * -1], [t_['err1'][k]]],
        #                  fmt='o', mec='silver', mfc='none', ecolor='silver',
        #                  ms=10, elinewidth=1, capsize=5, alpha=0.8, zorder=1)
    plt.colorbar(scatter, label=cmap)
    plt.plot([0.01, 40], [0.01, 40], 'k', zorder=0)
    plt.xlim(0.01, r)
    plt.ylim(0.01, r)
    plt.xlabel(param)
    plt.ylabel(param_dict[f'{param}'])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(folder, f'{param}_diagonal.png'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    figure_1(folder='/home/tehan/data/pyexofits/Data/', param='pl_ratror', cmap='Tmag')

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
import seaborn as sns
import matplotlib.cm as cm


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

def figure_1(folder='/home/tehan/data/pyexofits/Data/', param='pl_rade'):
    param_dict = {'pl_rade': 'r_pl__0', 'pl_ratror': 'ror__0'}
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]

    t_ = Table(names=['Tmag', f'{param}', f'{param}err1', f'{param}err2', 'value', 'err1', 'err2'],
               dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
    for i in trange(len(tics)):
        file = glob(os.path.join(folder, f'*/Photometry/*/*{tics[i]}*.dat'))
        if len(file) == 0:
            pass
        elif len(file) >= 1:
            for j in range(len(file)):
                star = int(os.path.basename(file[j]).split('_')[2])
                if star == tics[i]:
                    table = read_parameter(file[j])
                    table_ror = table[table['Parameter'] == param_dict['pl_rade']]
                    # t_.add_row([t['pl_ratror'][i], t['pl_ratrorerr1'][i], t['pl_ratrorerr2'][i],
                    #             table_ror['Value'][0], table_ror['Upper Error'][0], table_ror['Lower Error'][0]])
                    t_.add_row(
                        [t['sy_tmag'][i], t[f'{param}'][i], t[f'{param}err1'][i], t[f'{param}err2'][i],
                         table_ror['Value'][0], table_ror['Upper Error'][0], table_ror['Lower Error'][0]])
    print(len(t_))
    plt.figure(figsize=(10, 10))
    colormap = cm.viridis
    norm = plt.Normalize(t_['Tmag'].min(), t_['Tmag'].max())
    scatter = plt.scatter(t_[f'{param}'], t_['value'], c=t_['Tmag'], cmap=colormap, facecolors='none', s=20)
    plt.errorbar(t_[f'{param}'], t_['value'],
                            yerr=[t_['err2'].data * -1, t_['err1'].data],
                            fmt='o', mec='C1', mfc='none',
                            ecolor='C1', ms=2, elinewidth=0.1, capsize=0.5)
    plt.colorbar(scatter, label='TESS magnitude')
    plt.plot([0, 40], [0, 40], 'k')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.savefig(os.path.join(folder, f'{param}_diagonal.png'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    figure_1()

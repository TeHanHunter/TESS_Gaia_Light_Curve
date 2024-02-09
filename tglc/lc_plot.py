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


def figure_1(folder='/home/tehan/data/pyexofits/Data/', ):
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]

    t_ = Table(names=['pl_ratror', 'pl_ratrorerr1', 'pl_ratrorerr2', 'ror', 'rorerr1', 'rorerr2'],
               dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
    for i in trange(len(tics)):
        file = glob(os.path.join(folder, f'*/Photometry/*/*{tics[i]}*.dat'))
        if len(file) == 0:
            pass
        elif len(file) >= 1:
            for j in range(len(file)):
                star = int(os.path.basename(file[j]).split('_')[2])
                if star == tics[i]:
                    table = read_parameter(file[j])
                    table_ror = table[table['Parameter'] == 'ror__0']
                    # t_.add_row([t['pl_ratror'][i], t['pl_ratrorerr1'][i], t['pl_ratrorerr2'][i],
                    #             table_ror['Value'][0], table_ror['Upper Error'][0], table_ror['Lower Error'][0]])
                    t_.add_row(
                        [t['pl_rade'][i] / t['st_rad'][i] / 109.076, t['pl_ratrorerr1'][i], t['pl_ratrorerr2'][i],
                         table_ror['Value'][0], table_ror['Upper Error'][0], table_ror['Lower Error'][0]])
    print(len(t_))
    plt.figure(figsize=(10, 10))
    plt.plot([0, 0.4], [0, 0.4], 'k')
    plt.errorbar(t_['pl_ratror'], t_['ror'], yerr=[t_['rorerr2'].data * -1, t_['rorerr1'].data * -1], fmt='o', ecolor='C2')
    plt.savefig(os.path.join(folder, 'ror_diagonal.png'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    figure_1()

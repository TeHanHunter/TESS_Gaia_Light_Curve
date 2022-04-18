import os
import sys
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
import pickle
from os.path import exists
from TGLC.target_lightcurve import *
from TGLC.ffi_cut import *
from astropy.io import ascii
from glob import glob

if __name__ == '__main__':
    local_directory = '/home/tehan/data/exoplanets/'
    os.makedirs(local_directory + f'transits/', exist_ok=True)
    data = ascii.read(local_directory + 'PS_2022.04.17_18.17.28.csv')
    hosts = list(set(data['hostname']))
    for i in range(len(hosts)):
        target = hosts[i]  # Target identifier or coordinates TOI-3714
        os.makedirs(local_directory + f'lc/', exist_ok=True)
        os.makedirs(local_directory + f'epsf/test/', exist_ok=True)
        os.makedirs(local_directory + f'source/', exist_ok=True)
        # local_directory = os.path.join(os.getcwd(), f'{target}/')
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
        size = 50  # int, suggests big cuts
        source = ffi(target=target, size=size, local_directory=local_directory)
        for j in range(len(source.sector_table)):
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, ccd='test', local_directory=local_directory)

    for i in range(len(data['pl_name'])):
        files = glob(
            f'{local_directory}{hosts[i]}/lc/hlsp_tglc_tess_ffi_gaiaid-{data["gaia_id"][0].split()[-1]}-s00**_tess_v1_llc.fits')
        period = data['pl_orbper'][i]
        t_0 = data['pl_tranmid'][i]
        phase_fold_mid = (t_0 - 245700) % period
        t = []
        f = []
        for j in range(len(files)):
            with fits.open(files[j], mode='denywrite') as hdul:
                q = hdul[1].data['TGLC_flags'] == 0
                t.append(hdul[1].data['time'][q])
                f.append(hdul[1].data['cal_flux'][q])
        fig = plt.figure(constrained_layout=False, figsize=(10, 4))
        gs = fig.add_gridspec(1, 3)
        gs.update(wspace=1, hspace=0.1)

        # cmap = plt.get_cmap('cmr.fusion')  # MPL
        cmap = 'RdBu'
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(np.array(t), np.array(f), '.')
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(np.array(t) % period, np.array(f), '.')
        ax2.set_ylim(ax1.get_ylim())
        ax2.xlim(0.95 * t_0, 1.05 * t_0)
        ax2.savefig(local_directory + f'transits/{data["pl_name"][i]}')

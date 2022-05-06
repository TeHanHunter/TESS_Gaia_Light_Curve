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

if __name__ == '__main__':
    target = 'TIC 270023061'  # Target identifier or coordinates TOI-3714
    local_directory = f'/mnt/c/users/tehan/desktop/{target}/'

    os.makedirs(local_directory, exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/test/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    # local_directory = os.path.join(os.getcwd(), f'{target}/')

    size = 90  # int, suggests big cuts
    source = ffi(target=target, size=size, local_directory=local_directory)
    # source.select_sector(sector=24)
    # epsf(source, factor=2, sector=source.sector, ccd='test', local_directory=local_directory,
    #      name='Gaia DR2 2015656743621212928')

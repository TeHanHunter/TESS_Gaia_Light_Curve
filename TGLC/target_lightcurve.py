from TGLC.ffi import *
from TGLC.ePSF import *
from TGLC.local_background import *

import pickle
import numpy as np
from os.path import exists
from tqdm import trange
from wotan import flatten


def sector(target='', print_sector=True):
    catalog = Catalogs.query_object(target, radius=0.0618625, catalog="TIC")
    coord = SkyCoord(catalog[0]['ra'], catalog[0]['dec'], unit="deg")
    sector_table = Tesscut.get_sectors(coord)
    if print_sector:
        print(sector_table)
    return sector_table


def ffi(target='', local_directory='', size=90):
    source_exists = exists(f'{local_directory}source_{target}.pkl')
    if source_exists:
        with open(f'{local_directory}source_{target}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        print('Loaded ffi from directory. ')
    else:
        with open(f'{local_directory}source_{target}.pkl', 'wb') as output:
            source = Source(target, size=size)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
    return source


def epsf(factor=4, local_directory='', target=None, sector=0, num_stars=10000, edge_compression=1e-4, power=0.8,
         flat=True, return_epsf=False):
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor, edge_compression=edge_compression)
    epsf_exists = exists(f'{local_directory}epsf_{target}_sector_{sector}.npy')
    if epsf_exists:
        e_psf = np.load(f'{local_directory}epsf_{target}_sector_{sector}.npy')
        print('Loaded ePSF from directory. ')
    else:
        e_psf = np.zeros((len(source.time), (11 * factor + 1) ** 2 + 1))
        for i in trange(len(source.time), desc='Fitting ePSF'):
            fit, fluxfit = fit_psf(A, source, over_size, power=power, time=i)
            e_psf[i] = fit
        np.save(f'{local_directory}epsf_{target}_sector_{sector}.npy', e_psf)
    e_psf = np.median(e_psf, axis=0)
    lc_exists = exists(f'{local_directory}lc_{target}_sector_{sector}.npy')
    if lc_exists:
        lightcurve = np.load(f'{local_directory}lc_{target}_sector_{sector}.npy')
        print('Loaded lc from directory. ')
    else:
        lightcurve = np.zeros((min(num_stars, len(source.gaia)), len(source.time)))
        for i in trange(0, min(num_stars, len(source.gaia)), desc='Fitting lc'):
            r_A = reduced_A(A, source, star_info=star_info, x=x_round[i], y=y_round[i], star_num=i)
            if 0 <= x_round[i] <= source.size - 1 and 0 <= y_round[i] <= source.size - 1:
                for j in range(len(source.time)):
                    lightcurve[i, j] = source.flux[j][y_round[i], x_round[i]] - np.dot(r_A, e_psf)
        np.save(f'{local_directory}lc_{target}_sector_{sector}.npy', lightcurve)

    mod_lightcurve = lightcurve
    for i in trange(1, min(num_stars, len(source.gaia)), desc='Adjusting background'):
        bg_modification, bg_mod_err, bg_arr = bg_mod(source, lightcurve=lightcurve, sector=sector, chosen_index=[i])
        mod_lightcurve[i] = lightcurve[i] + bg_modification
    np.save(f'{local_directory}lc_mod_{target}_sector_{sector}.npy', mod_lightcurve)

    if flat:
        flatten_lc = np.zeros(np.shape(lightcurve))
        for i in trange(np.shape(lightcurve)[0], desc='Flattening lc'):
            flatten_lc[i] = flatten(source.time, mod_lightcurve[i] / np.median(mod_lightcurve[i]), window_length=1,
                                    method='biweight', return_trend=False)
        np.save(f'{local_directory}lc_flatten_{target}_sector_{sector}.npy', flatten_lc)
        return flatten_lc, e_psf if return_epsf else flatten_lc
    else:
        return mod_lightcurve, e_psf if return_epsf else mod_lightcurve


if __name__ == '__main__':
    target = 'NGC_7654'  # Target identifier or coordinates
    # sector_table = sector(target=target)
    local_directory = f'/mnt/c/users/tehan/desktop/{target}/'
    # local_directory = os.path.join(os.getcwd(), f'{target}/')
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    # sector = int(sector_table['sector'][0])
    size = 90  # int
    source = ffi(target=target, size=size, local_directory=local_directory)
    flatten_lc = epsf(factor=4, target=target, sector=source.sector, local_directory=local_directory)

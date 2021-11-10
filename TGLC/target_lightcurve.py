from TGLC.source import *
from TGLC.ePSF import *
from TGLC.local_background import *

import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
import pickle
from wotan import flatten

if __name__ == '__main__':
    target = input('Target identifier or coordinates: ') or 'NGC_7654'
    catalog = Catalogs.query_object(target, radius=0.0618625, catalog="TIC")
    coord = SkyCoord(catalog[0]['ra'], catalog[0]['dec'], unit="deg")
    sector_table = Tesscut.get_sectors(coord)
    print(sector_table)

    preferred_path = input('Local directory to save results: ') or '/mnt/c/Users/tehan/Desktop/NGC_7654/'
    sector = int(input('Which sector to work on?') or sector_table['sector'][0])
    size = int(input('How many pixels to analysis? [default 30]') or 30)
    # None if do not know
    # Fetch TESS and Gaia data
    source_exists = exists(f'{preferred_path}source_{target}_sector_{sector}.pkl')
    if source_exists:
        with open(f'{preferred_path}source_{target}_sector_{sector}.pkl', 'rb') as input_:
            source = pickle.load(input_)
    else:
        with open(f'{preferred_path}source_{target}_sector_{sector}.pkl', 'wb') as output:
            source = Source(target, size=size, sector=sector, search_gaia=True, mag_threshold=15)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)

    factor = 4
    regularization = 5e2 # 5e2  # the smaller, the smoother
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor)
    fit, fluxfit = fit_psf(A, source, over_size, time=0, regularization=regularization)  # 4e2
    plt.imshow(fit[0:-1].reshape(11 * factor + 1, 11 * factor + 1), origin='lower')
    plt.title(str(regularization))
    plt.savefig(f'{preferred_path}epsf_grid_sector_{sector}.png', dpi=300)

    # Plots
    # ePSF fit
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plot0 = ax[0].imshow(np.log10(source.flux[0]),
                         vmin=np.min(np.log10(source.flux[0])),
                         vmax=np.max(np.log10(source.flux[0])),
                         origin='lower')
    plot1 = ax[1].imshow(np.log10(fluxfit[0:source.size ** 2].reshape(source.size, source.size)),
                         vmin=np.min(np.log10(source.flux[0])),
                         vmax=np.max(np.log10(source.flux[0])),
                         origin='lower')
    residual = (source.flux[0] - fluxfit[0:source.size ** 2].reshape(source.size, source.size))
    plot2 = ax[2].imshow(residual, origin='lower', vmin=- np.max(residual), vmax=np.max(residual), cmap='RdBu')
    ax[0].set_title('Raw Data counts in log_10')
    ax[1].set_title('ePSF Model in log_10')
    ax[2].set_title('Residual')
    fig.colorbar(plot0, ax=ax[0])
    fig.colorbar(plot1, ax=ax[1])
    fig.colorbar(plot2, ax=ax[2])
    plt.savefig(f'{preferred_path}{target}_epsf_residual_sector_{sector}.png', dpi=300)
    # plt.imshow(fit[0:-1].reshape(11 * factor + 1, 11 * factor + 1), origin='lower')
    # plt.show()

    # Fit ePSF
    epsf_exists = exists(f'{preferred_path}epsf_{target}_sector_{sector}.npy')
    if epsf_exists:
        epsf = np.load(f'{preferred_path}epsf_{target}_sector_{sector}.npy')
    else:
        epsf = np.zeros((len(source.time), (11 * factor + 1) ** 2 + 1))
        for i in range(len(source.time)):
            fit, fluxfit = fit_psf(A, source, over_size, time=i, regularization=regularization)
            epsf[i] = fit
            print(i)
        np.save(f'{preferred_path}epsf_{target}_sector_{sector}.npy', epsf)

    # Get lightcurves
    lc_exists = exists(f'{preferred_path}lc_{target}_sector_{sector}.npy')
    if lc_exists:
        lightcurve = np.load(f'{preferred_path}lc_{target}_sector_{sector}.npy')
    else:
        lightcurve = np.zeros((min(10000, len(source.gaia)), len(source.time)))
        for i in range(0, min(10000, len(source.gaia))):
            r_A = reduced_A(A, star_info, star_num=i)
            x_shift = x_round[i]
            y_shift = y_round[i]
            if 0 <= x_shift <= source.size - 1 and 0 <= y_shift <= source.size - 1:
                print(i)
                for j in range(len(source.time)):
                    lightcurve[i, j] = source.flux[j][y_shift, x_shift] - \
                                       np.dot(r_A, epsf[j])[0:source.size ** 2].reshape(source.size, source.size)[
                                           y_shift, x_shift]
                source.gaia['variability'][i] = np.std(lightcurve[i] / np.median(lightcurve[i]))
        np.save(f'{preferred_path}lc_{target}_sector_{sector}.npy', lightcurve)
        with open(f'{preferred_path}source_{target}_sector_{sector}.pkl', 'wb') as output:
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)

    # local background
    do_bg = input(
        'Do you wish to refine the local background of a specific target? [y/n] (recommended for dim star in a crowded region)')
    while do_bg == 'y':
        bg_target = input('Star designation to do local background: [format: "Gaia DR2 5615115246375112192"]') or bg_target
        search_range = input(
            'Range to search for comparison stars (degree): [default 0.03 degree, approximately 12*12 pixels]') or 0.03
        target_index, chosen_index = comparison_star(source, target=bg_target, search_range=float(search_range), variability=0.05)
        bg_modification, bg_mod_err, bg_arr = bg_mod(source, lightcurve=lightcurve, sector=sector, chosen_index=chosen_index)
        if np.isnan(bg_modification):
            print(f'!!!Not sufficient stars ({len(chosen_index)}) nearby, consider enlarge the range of search.')
            continue
        # lightcurve[target_index] = lightcurve[target_index] + bg_modification
        print(f'Found {len(chosen_index)} comparison stars. Background modification is {bg_modification}, with error {bg_mod_err}')
        do_bg = input(
            "Do you wish to refine another star's local background? [y/n] (recommended for dim star in a crowded region)")
    # np.save(f'{preferred_path}lc_{target}_sector_{sector}.npy', lightcurve)

    # plt.plot(source.time[:2000] % 5.352446, flatten_lc[:2000], '.', ms=2, label='Sector_34')
    # plt.plot(sect7[0] % 5.352446, sect7[1], '.', ms=2, label='Sector_7')
    # plt.legend()
    # plt.savefig(f'{preferred_path}lc_comparison.png', dpi=300)
    # plt.show()

    # first, find close mag stars, then low variability, and lastly close.70 more computationally efficient

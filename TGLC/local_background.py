import numpy as np


def comparison_star(source, target=None, range=0.03, variability=0.01):
    """
    :param source: source object
    :param target: target designation with the format 'Gaia DR2 5615115246375112192'
    :param range: farthest acceptable comparison stars in arcsec
    :return: indexes of the comparison stars
    """

    index = np.where(target == source.gaia['designation'])
    target_mag = source.gaia['tess_mag'][index]
    target_ra = source.gaia['ra'][index]
    target_dec = source.gaia['dec'][index]

    similar_mag = np.where(source.gaia['tess_mag'] >= target_mag - 1)
    low_variability = np.where(
        np.logical_and(source.gaia['variability'] > 0, source.gaia['variability'] <= variability))
    best_comparison = np.intersect1d(similar_mag, low_variability)
    print(best_comparison)
    close_ra = np.where(
        np.logical_and(source.gaia['ra'] >= target_ra - range, source.gaia['ra'] <= target_ra + range))
    close_dec = np.where(
        np.logical_and(source.gaia['dec'] >= target_dec - range, source.gaia['dec'] <= target_dec + range))
    close_target = np.intersect1d(close_ra, close_dec)
    print(close_target)
    chosen_index = np.intersect1d(best_comparison, close_target)
    return index, chosen_index


def bg_mod(source, lightcurve=None, sector=1, chosen_index=np.zeros(1)):
    x = int(source.gaia[f'Sector_{sector}_x'][0])
    y = int(source.gaia[f'Sector_{sector}_y'][0])
    bg_mod = np.zeros(len(chosen_index))
    f_0 = np.median(source.flux[:, y, x])
    for i, index in enumerate(chosen_index):
        bg_mod[i] = (source.gaia['tess_flux_ratio'][index] * f_0 - np.median(lightcurve[index])) / (
                    1 - source.gaia['tess_flux_ratio'][index])
    return np.mean(bg_mod), np.std(bg_mod), bg_mod

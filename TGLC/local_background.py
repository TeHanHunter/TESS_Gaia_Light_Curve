import numpy as np
from tqdm import trange


def bg_mod(source, lightcurve=np.array([]), sector=1, num_stars=0, mag_lim=12):
    """
    background modification
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param lightcurve: np.ndarray, required
    ePSF lightcurve
    :param sector: int, required
    TESS sector number
    :param num_stars: int, required
    number of stars
    :param mag_lim: int, optional
    lower limiting magnitude of the stars to choose as reference
    :return: modified light curve
    """
    mod_lightcurve = lightcurve
    x = source.gaia[f'sector_{sector}_x']
    y = source.gaia[f'sector_{sector}_y']
    inner_stars = []
    for i in range(num_stars):
        if 0.5 <= x[i] <= source.size - 1.5 and 0.5 <= y[i] <= source.size - 1.5:
            inner_stars.append(i)
        if len(inner_stars) == 5:
            break
    for j in trange(np.array(source.gaia['tess_mag']).searchsorted(mag_lim, 'right'), num_stars,
                    desc='Adjusting background'):
        bg = np.zeros(5)
        for i, index in enumerate(inner_stars):
            bg[i] = source.gaia['tess_flux_ratio'][j] * np.nanmedian(source.flux[:, int(y[index]), int(x[index])]) / \
                    source.gaia['tess_flux_ratio'][index] - np.nanmedian(lightcurve[j])
        mod_lightcurve[j] = lightcurve[j] + np.nanmedian(bg)
    return mod_lightcurve

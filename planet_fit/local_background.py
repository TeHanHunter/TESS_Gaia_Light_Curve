import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def bg_mod(source, q=None, aper_lc=None, psf_lc=None, portion=None, star_num=0, near_edge=False):
    """
    background modification
    :param source: tglc.ffi.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param lightcurve: np.ndarray, required
    ePSF lightcurve
    :param sector: int, required
    TESS sector number
    :param num_stars: int, required
    number of stars
    :return: modified light curve
    """
    bar = 15000 * 10 ** ((source.gaia['tess_mag'][star_num] - 10) / -2.5)
    # med_epsf = np.nanmedian(e_psf[:, :23 ** 2].reshape(len(source.time), 23, 23), axis=0)
    # centroid_to_aper_ratio = 4/9 * np.sum(med_epsf[10:13, 10:13]) / np.sum(med_epsf)
    # centroid_to_aper_ratio = np.nanmedian(ratio)
    # flux_bar = aperture_bar * centroid_to_aper_ratio
    # lightcurve = lightcurve + (flux_bar - np.nanmedian(lightcurve[q]))
    aperture_bar = bar * portion
    local_bg = np.nanmedian(aper_lc[q]) - aperture_bar
    aper_lc = aper_lc - local_bg
    if near_edge:
        return local_bg, aper_lc, psf_lc
    # print(local_bg / aperture_bar)
    psf_bar = bar
    local_bg = np.nanmedian(psf_lc[q]) - psf_bar
    psf_lc = psf_lc - local_bg
    return local_bg, aper_lc, psf_lc

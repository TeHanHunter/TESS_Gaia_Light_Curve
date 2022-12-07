import numpy as np
from wotan import flatten
import tglc


def bilinear(x, y, repeat=1):
    '''
    A bilinear formula
    np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)
    b, d = array[1]
    a, c = array[0]
    :param x: x
    :param y: y
    :param repeat: side length of epsf
    :return: bilinear interpolation
    '''
    return np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)


def get_psf(source, factor=2, psf_size=11, edge_compression=1e-4, c=np.array([0, 0, 0])):
    """
    Generate matrix for PSF fitting
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param factor: int, optional
    effective PSF oversampling factor
    :param psf_size: int, optional
    effective PSF side length
    :param edge_compression: float, optional
    parameter for edge compression
    :param c: np.ndarray, optional
    manual modification of Gaia positions in the format of [x, y, theta]
    :return: A, star_info, over_size, x_round, y_round
    A: 2d matrix for least_square
    star_info: star parameters
    over_size: size of oversampled grid of ePSF
    x_round: star horizontal pixel coordinates rounded
    y_round: star vertical pixel coordinates rounded
    """
    # even only
    if factor % 2 != 0:
        raise ValueError('Factor must be even.')
    psf_size = psf_size
    half_size = int((psf_size - 1) / 2)
    over_size = psf_size * factor + 1
    size = source.size  # TODO: must be even?
    flux_ratio = np.array(source.gaia['tess_flux_ratio'])
    # flux_ratio = 0.9998 * flux_ratio + 0.0002
    # x_shift = np.array(source.gaia[f'sector_{source.sector}_x'])
    # y_shift = np.array(source.gaia[f'sector_{source.sector}_y'])

    x_shift = np.array(source.gaia[f'sector_{source.sector}_x'])
    y_shift = np.array(source.gaia[f'sector_{source.sector}_y'])

    # x_shift = (x_ - c[0]) * np.cos(c[2]) - (y_ - c[1]) * np.sin(c[2])
    # y_shift = (x_ - c[0]) * np.sin(c[2]) + (y_ - c[1]) * np.cos(c[2])

    x_round = np.round(x_shift).astype(int)
    y_round = np.round(y_shift).astype(int)

    left = np.maximum(0, x_round - half_size)
    right = np.minimum(size, x_round + half_size) + 1
    down = np.maximum(0, y_round - half_size)
    up = np.minimum(size, y_round + half_size) + 1
    x_residual = x_shift % (1 / factor) * factor
    y_residual = y_shift % (1 / factor) * factor

    x_p = np.arange(size)
    y_p = np.arange(size)
    coord = np.arange(size ** 2).reshape(size, size)
    xx, yy = np.meshgrid((np.arange(size) - (size - 1) / 2), (np.arange(size) - (size - 1) / 2))

    if type(source) == tglc.ffi.Source:
        bg_dof = 6
        A = np.zeros((size ** 2, over_size ** 2 + bg_dof))
        A[:, -1] = np.ones(size ** 2)
        A[:, -2] = yy.flatten()
        A[:, -3] = xx.flatten()
        A[:, -4] = source.mask.data.flatten()
        A[:, -5] = (source.mask.data * xx).flatten()
        A[:, -6] = (source.mask.data * yy).flatten()
    else:
        bg_dof = 3
        A = np.zeros((size ** 2, over_size ** 2 + bg_dof))
        A[:, -1] = np.ones(size ** 2)
        A[:, -2] = yy.flatten()
        A[:, -3] = xx.flatten()
    star_info = []
    for i in range(len(source.gaia)):
        #     if i == 8:
        #         continue
        x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + (x_shift[i] % 1) // (1 / factor)
        y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + (y_shift[i] % 1) // (1 / factor)
        x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
        a = np.array(x_psf + y_psf * over_size, dtype=np.int64).flatten()
        index = coord[down[i]:up[i], left[i]:right[i]]
        A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
            flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))
        # star_info.append(
        #     (np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
        #      flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))))
        star_info.append(
            (index, a, flux_ratio[i] * bilinear(x_residual[i], y_residual[i])))
    coord_ = np.arange(- psf_size * factor / 2 + 1, psf_size * factor / 2 + 2)
    x_coord, y_coord = np.meshgrid(coord_, coord_)
    variance = psf_size
    dist = (1 - np.exp(- 0.5 * (x_coord ** 4 + y_coord ** 4) / variance ** 4)) * edge_compression  # 1e-3
    A_mod = np.diag(dist.flatten())
    A_mod = np.concatenate((A_mod, (np.zeros((over_size ** 2, bg_dof)))), axis=-1)
    A = np.append(A, A_mod, axis=0)
    return A, star_info, over_size, x_round, y_round


def fit_psf(A, source, over_size, power=0.8, time=0):
    """
    fit_psf using least_square (improved performance by changing to np.linalg.solve)
    :param A: np.ndarray, required
    2d matrix for least_square
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param over_size: int, required
    size of oversampled grid of ePSF
    :param power: float, optional
    power for weighting bright stars' contribution to the fit. 1 means same contribution from all stars,
    <1 means emphasizing dimmer stars
    :param time: int, required
    time index of this ePSF fit
    :return: fit result
    """
    saturated_index = source.mask.mask.flatten()
    flux = source.flux[time].flatten()
    saturated_index[flux < 0.8 * np.nanmedian(flux)] = True

    b = np.delete(flux, saturated_index)
    scaler = np.abs(np.delete(flux, saturated_index)) ** power
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = np.append(scaler, np.ones(over_size ** 2))

    # fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    a = np.delete(A, np.where(saturated_index), 0) / scaler[:, np.newaxis]
    b = b / scaler
    alpha = np.dot(a.T, a)
    beta = np.dot(a.T, b)
    try:
        fit = np.linalg.solve(alpha, beta)
    except np.linalg.LinAlgError:
        fit = np.full(np.shape(a)[1], np.nan)
        # fit = np.linalg.lstsq(a, b, rcond=None)[0]
    return fit


def fit_lc(A, source, star_info=None, x=0., y=0., star_num=0, factor=2, psf_size=11, e_psf=None, near_edge=False):
    """
    Produce matrix for least_square fitting without a certain target
    :param A: np.ndarray, required
    2d matrix for least_square
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param star_info: np.ndarray, required
    star parameters
    :param x: float, required
    target horizontal pixel coordinate
    :param y: float, required
    target vertical pixel coordinate
    :param star_num: int, required
    target star index
    :param factor: int, optional
    effective PSF oversampling factor
    :param psf_size: int, optional
    effective PSF side length
    :param e_psf: np.ndarray, required
    effective PSF as a 3d array as a timeseries
    :param near_edge: boolean, required
    whether the star is 2 pixels or closer to the edge of a CCD
    :return: aperture lightcurve, PSF lightcurve, vertical pixel coord, horizontal pixel coord, portion of light in aperture
    """
    over_size = psf_size * factor + 1
    a = star_info[star_num][1]
    star_info_num = (np.repeat(star_info[star_num][0], 4),
                     np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
                     np.tile(star_info[star_num][2], len(a)))
    size = source.size  # TODO: must be even?
    # star_position = int(x + source.size * y - 5 * size - 5)
    # aper_lc
    cut_size = 5
    in_frame = np.where(np.invert(np.isnan(source.flux[0])))
    left = np.maximum(np.min(in_frame[1]), x - cut_size // 2)
    right = np.minimum(np.max(in_frame[1]), x + cut_size // 2 + 1)
    down = np.maximum(np.min(in_frame[0]), y - cut_size // 2)
    up = np.minimum(np.max(in_frame[0]), y + cut_size // 2 + 1)
    coord = np.arange(size ** 2).reshape(size, size)
    index = np.array(coord[down:up, left:right]).flatten()
    A_cut = np.zeros((len(index), np.shape(A)[1]))
    for i in range(len(index)):
        A_ = np.zeros(np.shape(A)[-1])
        star_pos = np.where(star_info_num[0] == index[i])[0]
        A_[star_info_num[1][star_pos]] = star_info_num[2][star_pos]
        A_cut[i] = A[index[i], :] - A_
    aperture = np.zeros((len(source.time), len(index)))
    for j in range(len(source.time)):
        aperture[j] = np.array(source.flux[j][down:up, left:right]).flatten() - np.dot(A_cut, e_psf[j])
    aperture = aperture.reshape((len(source.time), up - down, right - left))

    # psf_lc
    over_size = psf_size * factor + 1
    if near_edge:  # TODO: near_edge
        psf_lc = np.zeros(len(source.time))
        psf_lc[:] = np.NaN
        e_psf_1d = np.nanmedian(e_psf[:, :over_size ** 2], axis=0).reshape(over_size, over_size)
        portion = (36 / 49) * np.nansum(e_psf_1d[8:15, 8:15]) / np.nansum(e_psf_1d)  # only valid for factor = 2
        return aperture, psf_lc, y - down, x - left, portion
    left_ = left - x + 5
    right_ = right - x + 5
    down_ = down - y + 5
    up_ = up - y + 5

    left_11 = np.maximum(- x + 5, 0)
    right_11 = np.minimum(size - x + 5, 11)
    down_11 = np.maximum(- y + 5, 0)
    up_11 = np.minimum(size - y + 5, 11)
    coord = np.arange(psf_size ** 2).reshape(psf_size, psf_size)
    index = coord[down_11:up_11, left_11:right_11]
    if type(source) == tglc.ffi.Source:
        bg_dof = 6
    else:
        bg_dof = 3
    A = np.zeros((psf_size ** 2, over_size ** 2 + bg_dof))
    A[np.repeat(index, 4), star_info_num[1]] = star_info_num[2]
    psf_shape = np.dot(e_psf, A.T).reshape(len(source.time), psf_size, psf_size)
    psf_sim = psf_shape[:, down_:up_, left_: right_]
    # psf_sim = np.transpose(psf_shape[:, down_:up_, left_: right_], (0, 2, 1))

    psf_lc = np.zeros(len(source.time))
    A_ = np.zeros((cut_size ** 2, 4))
    xx, yy = np.meshgrid((np.arange(cut_size) - (cut_size - 1) / 2),
                         (np.arange(cut_size) - (cut_size - 1) / 2))
    A_[:, -1] = np.ones(cut_size ** 2)
    A_[:, -2] = yy.flatten()
    A_[:, -3] = xx.flatten()
    edge_pixel = np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])
    # edge_pixel = np.array([0, 1, 2, 3, 4, 5, 6,
    #                        7, 8, 9, 10, 11, 12, 13,
    #                        14, 15, 19, 20,
    #                        21, 22, 26, 27,
    #                        28, 29, 33, 34,
    #                        35, 36, 37, 38, 39, 40, 41,
    #                        42, 43, 44, 45, 46, 47, 48])
    med_aperture = np.median(aperture, axis=0).flatten()
    outliers = np.abs(med_aperture[edge_pixel] - np.nanmedian(med_aperture[edge_pixel])) > 1 * np.std(
        med_aperture[edge_pixel])
    epsf_sum = np.sum(np.nanmedian(psf_shape, axis=0))
    for j in range(len(source.time)):
        if np.isnan(psf_sim[j, :, :]).any():
            psf_lc[j] = np.nan
        else:
            aper_flat = aperture[j, :, :].flatten()
            A_[:, 0] = psf_sim[j, :, :].flatten() / epsf_sum
            a = np.delete(A_, edge_pixel[outliers], 0)
            aper_flat = np.delete(aper_flat, edge_pixel[outliers])
            psf_lc[j] = np.linalg.lstsq(a, aper_flat)[0][0]
    portion = np.nansum(psf_shape[:, 4:7, 4:7]) / np.nansum(psf_shape)
    return aperture, psf_lc, y - down, x - left, portion


def fit_lc_float_field(A, source, star_info=None, x=np.array([]), y=np.array([]), star_num=0, factor=2, psf_size=11,
                       e_psf=None, near_edge=False, prior=0.001):
    """
    Produce matrix for least_square fitting without a certain target
    :param A: np.ndarray, required
    2d matrix for least_square
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param star_info: np.ndarray, required
    star parameters
    :param x: float, required
    target horizontal pixel coordinate
    :param y: float, required
    target vertical pixel coordinate
    :param star_num: int, required
    target star index
    :param factor: int, optional
    effective PSF oversampling factor
    :param psf_size: int, optional
    effective PSF side length
    :param e_psf: np.ndarray, required
    effective PSF as a 3d array as a timeseries
    :param near_edge: boolean, required
    whether the star is 2 pixels or closer to the edge of a CCD
    :return: aperture lightcurve, PSF lightcurve, vertical pixel coord, horizontal pixel coord, portion of light in aperture
    """
    over_size = psf_size * factor + 1
    a = star_info[star_num][1]
    star_info_num = (np.repeat(star_info[star_num][0], 4),
                     np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
                     np.tile(star_info[star_num][2], len(a)))
    size = source.size  # TODO: must be even?
    # star_position = int(x + source.size * y - 5 * size - 5)
    # aper_lc
    cut_size = 5
    in_frame = np.where(np.invert(np.isnan(source.flux[0])))
    left = np.maximum(np.min(in_frame[1]), x[star_num] - cut_size // 2)
    right = np.minimum(np.max(in_frame[1]), x[star_num] + cut_size // 2 + 1)
    down = np.maximum(np.min(in_frame[0]), y[star_num] - cut_size // 2)
    up = np.minimum(np.max(in_frame[0]), y[star_num] + cut_size // 2 + 1)
    coord = np.arange(size ** 2).reshape(size, size)
    index = np.array(coord[down:up, left:right]).flatten()
    A_cut = np.zeros((len(index), np.shape(A)[1]))
    for i in range(len(index)):
        A_ = np.zeros(np.shape(A)[-1])
        star_pos = np.where(star_info_num[0] == index[i])[0]
        A_[star_info_num[1][star_pos]] = star_info_num[2][star_pos]
        A_cut[i] = A[index[i], :] - A_
    aperture = np.zeros((len(source.time), len(index)))
    for j in range(len(source.time)):
        aperture[j] = np.array(source.flux[j][down:up, left:right]).flatten() - np.dot(A_cut, e_psf[j])
    aperture = aperture.reshape((len(source.time), up - down, right - left))

    # psf_lc
    over_size = psf_size * factor + 1
    if near_edge:  # TODO: near_edge
        psf_lc = np.zeros(len(source.time))
        psf_lc[:] = np.NaN
        e_psf_1d = np.nanmedian(e_psf[:, :over_size ** 2], axis=0).reshape(over_size, over_size)
        portion = (36 / 49) * np.nansum(e_psf_1d[8:15, 8:15]) / np.nansum(e_psf_1d)  # only valid for factor = 2
        return aperture, psf_lc, y[star_num] - down, x[star_num] - left, portion
    # left_ = left - x[star_num] + 5
    # right_ = right - x[star_num] + 5
    # down_ = down - y[star_num] + 5
    # up_ = up - y[star_num] + 5
    if type(source) == tglc.ffi.Source:
        bg_dof = 6
    else:
        bg_dof = 3
    field_star_num = []
    for j in range(len(source.gaia)):
        if np.abs(x[j] - x[star_num]) < 5 and np.abs(y[j] - y[star_num]) < 5:
            field_star_num.append(j)

    psf_lc = np.zeros(len(source.time))
    A_ = np.zeros((cut_size ** 2 + len(field_star_num), len(field_star_num) + 3))
    xx, yy = np.meshgrid((np.arange(cut_size) - (cut_size - 1) / 2),
                         (np.arange(cut_size) - (cut_size - 1) / 2))
    A_[:(cut_size ** 2), -1] = np.ones(cut_size ** 2)
    A_[:(cut_size ** 2), -2] = yy.flatten()
    A_[:(cut_size ** 2), -3] = xx.flatten()
    psf_sim = np.zeros((len(source.time), 11 ** 2 + len(field_star_num), len(field_star_num)))
    for j, star in enumerate(field_star_num):
        a = star_info[star][1]
        star_info_star = (np.repeat(star_info[star][0], 4),
                          np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
                          np.tile(star_info[star][2], len(a)))
        delta_x = x[star_num] - x[star]
        delta_y = y[star_num] - y[star]
        # for psf_sim
        left_shift = np.maximum(delta_x, 0)
        right_shift = np.minimum(11 + delta_x, 11)
        down_shift = np.maximum(delta_y, 0)
        up_shift = np.minimum(11 + delta_y, 11)
        # for psf_shape
        left_shift_ = np.maximum(-delta_x, 0)
        right_shift_ = np.minimum(11 - delta_x, 11)
        down_shift_ = np.maximum(-delta_y, 0)
        up_shift_ = np.minimum(11 - delta_y, 11)

        left_11 = np.maximum(- x[star] + 5, 0)
        right_11 = np.minimum(size - x[star] + 5, 11)
        down_11 = np.maximum(- y[star] + 5, 0)
        up_11 = np.minimum(size - y[star] + 5, 11)

        coord = np.arange(psf_size ** 2).reshape(psf_size, psf_size)
        index = coord[down_11:up_11, left_11:right_11]
        A = np.zeros((psf_size ** 2, over_size ** 2 + bg_dof))
        A[np.repeat(index, 4), star_info_star[1]] = star_info_star[2]
        psf_shape = np.dot(e_psf, A.T).reshape(len(source.time), psf_size, psf_size)
        epsf_sum = np.sum(np.nanmedian(psf_shape, axis=0))
        psf_sim_index = coord[down_shift:up_shift, left_shift:right_shift].flatten()
        psf_sim[:, psf_sim_index, j] = psf_shape[:, down_shift_:up_shift_, left_shift_:right_shift_].reshape(
            len(source.time), -1) / epsf_sum
        if star != star_num:
            psf_sim[:, 11 ** 2 + j, j] = np.ones(len(source.time)) / (
                    prior * 1.5e4 * 10 ** ((10 - source.gaia[star]['tess_mag']) / 2.5))
        else:
            portion = np.nansum(psf_shape[:, 4:7, 4:7]) / np.nansum(psf_shape)

    star_index = np.where(np.array(field_star_num) == star_num)[0]
    field_star = psf_sim[0, np.arange(11 ** 2).reshape(11, 11)[3:8, 3:8], :].reshape(cut_size ** 2,
                                                                                     len(field_star_num)) * \
                 source.gaia['tess_flux_ratio'][field_star_num]
    field_star[:, star_index] = 0
    for j in range(len(source.time)):
        if np.isnan(psf_sim[j, :, :]).any():
            psf_lc[j] = np.nan
        else:
            aper_flat = aperture[j, :, :].flatten()
            aper_flat = np.append(aper_flat, np.zeros(len(field_star_num) - 1))  # / prior
            aper_flat[cut_size ** 2 + star_index] = 0
            postcards = psf_sim[j, np.arange(11 ** 2).reshape(11, 11)[3:8, 3:8], :].reshape(cut_size ** 2,
                                                                                            len(field_star_num))
            A_[:cut_size ** 2, :len(field_star_num)] = postcards
            field_star = postcards * source.gaia['tess_flux_ratio'][field_star_num]
            field_star[:, star_index] = 0
            # A_[:(cut_size ** 2), -4] = np.sum(field_star, axis=1)
            A_[cut_size ** 2:, :len(field_star_num)] = psf_sim[j, 11 ** 2:, :].reshape(len(field_star_num),
                                                                                       len(field_star_num))
            a = np.delete(A_, cut_size ** 2 + star_index, 0)
            psf_lc[j] = np.linalg.lstsq(a, aper_flat)[0][star_index]
    return aperture, psf_lc, y[star_num] - down, x[star_num] - left, portion


def bg_mod(source, q=None, aper_lc=None, psf_lc=None, portion=None, star_num=0, near_edge=False):
    '''
    background modification
    :param source: tglc.ffi_cut.Source or tglc.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param q: list, optional
    list of booleans that filter the data points
    :param aper_lc: np.ndarray, required
    aperture light curve
    :param psf_lc: np.ndarray, required
    PSF light curve
    :param portion: float, required
    portion of light in aperture
    :param star_num: int, required,
    star index
    :param near_edge: boolean, required
    whether the star is 2 pixels or closer to the edge of a CCD
    :return: local background, modified aperture light curve, modified PSF light curve
    '''
    bar = 15000 * 10 ** ((source.gaia['tess_mag'][star_num] - 10) / -2.5)
    # med_epsf = np.nanmedian(e_psf[:, :23 ** 2].reshape(len(source.time), 23, 23), axis=0)
    # centroid_to_aper_ratio = 4/9 * np.sum(med_epsf[10:13, 10:13]) / np.sum(med_epsf)
    # centroid_to_aper_ratio = np.nanmedian(ratio)
    # flux_bar = aperture_bar * centroid_to_aper_ratio
    # lightcurve = lightcurve + (flux_bar - np.nanmedian(lightcurve[q]))
    aperture_bar = bar * portion
    local_bg = np.nanmedian(aper_lc[q]) - aperture_bar
    if np.isnan(local_bg):
        local_bg = 0
    aper_lc = aper_lc - local_bg
    psf_bar = bar
    local_bg = np.nanmedian(psf_lc[q]) - psf_bar
    if np.isnan(local_bg):
        local_bg = 0
    psf_lc = psf_lc - local_bg

    # removes very large outliers to prevent wotan to freeze
    cal_aper_lc = aper_lc / np.nanmedian(aper_lc)
    cal_aper_lc[np.where(cal_aper_lc > 100)] = np.nan
    if np.isnan(cal_aper_lc).all():
        print('Calibrated aperture flux are not accessible or processed incorrectly. ')
    else:
        cal_aper_lc = flatten(source.time, cal_aper_lc, window_length=1, method='biweight',
                              return_trend=False)
    if near_edge:
        cal_psf_lc = psf_lc
        return local_bg, aper_lc, psf_lc, cal_aper_lc, cal_psf_lc
    else:
        cal_psf_lc = psf_lc / np.nanmedian(psf_lc)
        cal_psf_lc[np.where(cal_psf_lc > 100)] = np.nan
        if np.isnan(cal_psf_lc).all():
            print('Calibrated PSF flux are not accessible or processed incorrectly. ')
        else:
            cal_psf_lc = flatten(source.time, cal_psf_lc, window_length=1, method='biweight',
                                 return_trend=False)
    # aper_mad = 1.4826 * np.nanmedian(np.abs(cal_aper_lc - 1))
    # if aper_mad > 0.02:
    #     psf_mad = 1.4826 * np.nanmedian(np.abs(cal_psf_lc - 1))
    #     cal_psf_lc /= psf_mad / aper_mad
    #     cal_psf_lc += 1 - np.median(cal_psf_lc)
    return local_bg, aper_lc, psf_lc, cal_aper_lc, cal_psf_lc

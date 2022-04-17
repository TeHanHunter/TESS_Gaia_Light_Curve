import numpy as np


def bilinear(x, y, repeat=45):
    """
    A bilinear formula
    np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)
    b, d = array[1]
    a, c = array[0]
    """
    return np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)


def get_psf(source, factor=2, psf_size=11, edge_compression=1e-4, c=np.array([0, 0, 0])):
    """
    Generate matrix for PSF fitting
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param factor: int, optional
    effective PSF oversampling factor
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

    x_ = np.array(source.gaia[f'sector_{source.sector}_x'])
    y_ = np.array(source.gaia[f'sector_{source.sector}_y'])

    x_shift = (x_ - c[0]) * np.cos(c[2]) - (y_ - c[1]) * np.sin(c[2])
    y_shift = (x_ - c[0]) * np.sin(c[2]) + (y_ - c[1]) * np.cos(c[2])

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
    A = np.zeros((size ** 2, over_size ** 2 + 1))
    A[:, -1] = np.ones(size ** 2)
    star_info = []
    for i in range(len(source.gaia)):
        # if i == 8:
        #     pass
        # else:
        x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + (x_shift[i] % 1) // (1 / factor)
        y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + (y_shift[i] % 1) // (1 / factor)
        x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
        a = np.array(x_psf + y_psf * over_size, dtype=np.int64)
        a = a.flatten()
        index = coord[down[i]:up[i], left[i]:right[i]]
        # print(len(np.repeat(index, 4)))
        A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
            flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))
        star_info.append(
            (np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
             flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))))
    coord = np.arange(- psf_size * factor / 2 + 1, psf_size * factor / 2 + 2)
    x_coord, y_coord = np.meshgrid(coord, coord)

    variance = psf_size
    dist = (1 - np.exp(- 0.5 * (x_coord ** 4 + y_coord ** 4) / variance ** 4)) * edge_compression  # 1e-3
    A_mod = np.diag(dist.flatten())
    A_mod = np.concatenate((A_mod, (np.zeros((over_size ** 2, 1)))), axis=-1)
    A = np.append(A, A_mod, axis=0)
    return A, star_info, over_size, x_round, y_round


def reduced_A(A, source, star_info=None, x=0., y=0., star_num=0):
    """
    Produce matrix for least_square fitting without a certain target
    :param A: np.ndarray, required
    2d matrix for least_square
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
    Source or Source_cut object
    :param star_info: np.ndarray, required
    star parameters
    :param x: float, required
    target horizontal pixel coordinate
    :param y: float, required
    target vertical pixel coordinate
    :param star_num: int, required
    target star index
    :return: reduced A, matrix for least_square fix without the target star
    """
    if star_info is None:
        star_info = []
    star_position = int(x + source.size * y)
    A_ = np.zeros(np.shape(A)[-1])
    index = np.where(star_info[star_num][0] == star_position)
    A_[star_info[star_num][1][index]] = star_info[star_num][2][index]
    return A[star_position, :] - A_


def fit_psf(A, source, over_size, power=0.8, time=0):
    """
    fit_psf using least_square (improved performance by changing to np.linalg.solve)
    :param A: np.ndarray, required
    2d matrix for least_square
    :param source: TGLC.ffi.Source or TGLC.ffi_cut.Source_cut, required
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
    b = source.flux[time].flatten()
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = (source.flux[time].flatten()) ** power  # source.flux_err[time].flatten() ** 2 +
    scaler = np.append(scaler, np.ones(over_size ** 2))

    # fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    a = A / scaler[:, np.newaxis]
    b = b / scaler
    alpha = np.dot(a.T, a)
    beta = np.dot(a.T, b)
    fit = np.linalg.solve(alpha, beta)
    return fit

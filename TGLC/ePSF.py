import numpy as np
import matplotlib.pyplot as plt
import pickle


# def bilinear(x, y, repeat=25):
#     """
#     value = a + (b - a) * y + (a - c) * x + (b + d - a - c) * x * y
#     coefficients of a, b, c, d [1 + x - y - x * y, y + x * y, -x - x * y, x * y]
#     (x+1)*(1-y)
#
#     a, c = array[0]
#     b, d = array[1]
#     """
#     return np.array([1 + x - y - x * y, -x - x * y, y + x * y, x * y] * repeat)

def bilinear(x, y, repeat=45):
    """
    np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)
    b, d = array[1]
    a, c = array[0]
    """
    return np.array([1 - x - y + x * y, x - x * y, y - x * y, x * y] * repeat)


def paraboloid(z_0, z_1, z_2, z_3, z_4, z_5, scale=0.1):
    """
    z = c_1 x^2 + c_2 y^2 + c_3 xy + c_4 x + c_5 y + c_6

    Parameters:
        z_0: x = 0, y = 0
        z_1: x = 0.1, y = 0
        z_2: x = 0, y = 0.1
        z_3: x = -0.1, y = 0
        z_4: x = 0, y = -0.1
        z_5: x = 0.1, y = 0.1
    """

    c_1 = (z_3 + z_1 - 2 * z_0) / (2 * scale ** 2)
    c_2 = (z_4 + z_2 - 2 * z_0) / (2 * scale ** 2)
    c_4 = (z_1 - z_3) / (2 * scale)
    c_5 = (z_2 - z_4) / (2 * scale)
    c_6 = z_0
    c_3 = z_5 * scale ** -2 - 100 * c_6 * scale ** -2 - c_4 * scale ** -1 - c_5 * scale ** -1 - c_1 - c_2
    x_max = (2 * c_1 * c_4 - c_3 * c_5) / (c_3 ** 2 - 4 * c_1 * c_2)
    y_max = (2 * c_1 * c_5 - c_3 * c_4) / (c_3 ** 2 - 4 * c_1 * c_2)
    print(c_1, c_2)
    return x_max, y_max


def get_psf(source, factor=2, edge_compression=1e-4, c=np.array([0, 0, 0])):
    # even only
    if factor % 2 != 0:
        raise ValueError('Factor must be even.')
    psf_size = 11
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


def reduced_A(A, source, star_info=None, x=0, y=0, star_num=0):
    if star_info is None:
        star_info = []
    star_position = int(x + source.size * y)
    A_ = np.zeros(np.shape(A)[-1])
    index = np.where(star_info[star_num][0] == star_position)
    A_[star_info[star_num][1][index]] = star_info[star_num][2][index]
    return A[star_position, :] - A_


def fit_psf(A, source, over_size, power=0.8, time=0):
    b = source.flux[time].flatten()
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = (source.flux_err[0].flatten() ** 2 + source.flux[time].flatten()) ** power
    scaler = np.append(scaler, np.ones(over_size ** 2))
    fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    fluxfit = np.dot(A, fit)
    return fit, fluxfit

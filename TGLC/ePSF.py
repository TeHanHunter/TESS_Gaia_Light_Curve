import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import lsq_linear


def bilinear(x, y, repeat=25):
    """
    value = a + (b - a) * y + (a - c) * x + (b + d - a - c) * x * y
    coefficients of a, b, c, d [1 + x - y - x * y, y + x * y, -x - x * y, x * y]
    (x+1)*(1-y)

    a, c = array[0]
    b, d = array[1]
    """
    return np.array([1 + x - y - x * y, -x - x * y, y + x * y, x * y] * repeat)


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


def get_psf(source, factor=2):
    # even only
    if factor % 2 != 0:
        raise ValueError('Factor must be even.')
    psf_size = 11
    half_size = int((psf_size - 1) / 2)
    over_size = psf_size * factor + 1
    # global flux_ratio, x_shift, y_shift, x_round, y_round, x_sign, y_sign
    # nstars = source.nstars
    size = source.size  # TODO: must be even?
    flux_ratio = np.array(source.gaia['tess_flux_ratio'])
    # flux_ratio = 0.9998 * flux_ratio + 0.0002
    x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])
    y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])

    x_round = np.round(x_shift).astype(int)
    y_round = np.round(y_shift).astype(int)

    left = np.maximum(0, x_round - half_size)
    right = np.minimum(size, x_round + half_size) + 1
    down = np.maximum(0, y_round - half_size)
    up = np.minimum(size, y_round + half_size) + 1
    x_residual = x_shift % (1 / factor) * factor
    y_residual = y_shift % (1 / factor) * factor
    # x_sign = np.sign(x_round - x_shift)
    # y_sign = np.sign(y_round - y_shift)

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
    dist = (1 - np.exp(- 0.5 * (x_coord ** 4 + y_coord ** 4) / variance ** 4)) * 1e-4  # 1e-3
    # remove center compression
    remove_index = (np.arange(over_size) + 1) * over_size - 1
    # diag = np.diag(np.ones(over_size ** 2))
    # A_1 = diag - np.concatenate((np.zeros((over_size ** 2, 1)), diag[:, 0: - 1]), axis=-1)
    # A_1 = np.delete(A_1, remove_index, 0)
    # A_1 = np.concatenate((A_1, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
    # A_2 = diag - np.concatenate((np.zeros((over_size ** 2, over_size)), diag[:, 0: - over_size]), axis=-1)
    # A_2 = A_2[0: - over_size]
    # A_2 = np.concatenate((A_2, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
    A_3 = np.diag(dist.flatten())
    A_3 = np.concatenate((A_3, (np.zeros((over_size ** 2, 1)))), axis=-1)
    # A = np.append(A, A_1, axis=0)
    # A = np.append(A, A_2, axis=0)
    A = np.append(A, A_3, axis=0)
    return A, star_info, over_size, x_round, y_round


# def reduced_A(A, star_info, star_num=0):
#     info = star_info[star_num]
#     A_ = np.zeros(np.shape(A))
#     A_[info[0], info[1]] = info[2]
#     return A - A_

def reduced_A(A, source, star_info=[], x=0, y=0, star_num=0):
    star_position = int(x + source.size * y)
    A_ = np.zeros(np.shape(A)[-1])
    index = np.where(star_info[star_num][0] == star_position)
    A_[star_info[star_num][1][index]] = star_info[star_num][2][index]
    return A[star_position, :] - A_


def fit_psf(A, source, over_size, time=0, regularization=8e2):
    b = source.flux[time].flatten()
    # b = np.append(b, np.zeros(2 * over_size * (over_size - 1)))
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = (source.flux_err[0].flatten() ** 2 + source.flux[time].flatten()) ** 0.8
    # scaler = np.append(scaler, regularization * np.ones(2 * over_size * (over_size - 1)))
    scaler = np.append(scaler, np.ones(over_size ** 2))
    fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    fluxfit = np.dot(A, fit)
    fluxfit = np.dot(A, fit)
    return fit, fluxfit


# try moffat for 77
if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    factor = 4
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor)
    fit, fluxfit = fit_psf(A, source, over_size, time=0, regularization=4e2)  # 4e2
    plt.imshow(fit[0:-1].reshape(11 * factor + 1, 11 * factor + 1), origin='lower')
    # plt.savefig('/mnt/c/users/tehan/desktop/epsf_grid.png', dpi=300)
    plt.show()
    epsf = np.load('/mnt/c/users/tehan/desktop/epsf_.npy')

    # epsf = np.zeros((len(source.time), (11 * factor + 1) ** 2 + 1))
    # for i in range(len(source.time)):
    #     fit, fluxfit = fit_psf(A, source, over_size, time=i)
    #     epsf[i] = fit
    #     print(i)
    # np.save('/mnt/c/users/tehan/desktop/epsf_.npy', epsf)

    # lightcurve = np.zeros((5000, len(source.time)))
    # for i in range(0, 5000):
    #     i = i + 15000
    #     r_A = reduced_A(A, star_info, star_num=i)
    #     x_shift = x_round[i]
    #     y_shift = y_round[i]
    #     if 0 <= x_shift <= 89 and 0 <= y_shift <= 89:
    #         print(i)
    #         for j in range(len(source.time)):
    #             lightcurve[i - 15000, j] = source.flux[j][y_shift, x_shift] - \
    #                                np.dot(r_A, epsf[j])[0:source.size ** 2].reshape(source.size, source.size)[
    #                                    y_shift, x_shift]
    # np.save('/mnt/c/users/tehan/desktop/epsf_lc_15000_20000.npy', lightcurve)

    # plt.plot(lightcurve[i])
    # plt.plot(source.flux[:, y_shift, x_shift] - epsf[:, -1])
    # plt.show()

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
    ax[0].set_title('Raw Data')
    ax[1].set_title('ePSF Model')
    ax[2].set_title('Residual')
    fig.colorbar(plot0, ax=ax[0])
    fig.colorbar(plot1, ax=ax[1])
    fig.colorbar(plot2, ax=ax[2])
    plt.savefig('/mnt/c/users/tehan/desktop/epsf_residual.png', dpi=300)
    plt.show()
    # plt.imshow(fit[0:-1].reshape(11 * factor + 1, 11 * factor + 1), origin='lower')
    # plt.show()

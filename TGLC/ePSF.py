import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle


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
    # c_1 = 50 * (z_3 + z_1 - 2 * z_0)
    # c_2 = 50 * (z_4 + z_2 - 2 * z_0)
    # c_4 = 5 * (z_1 - z_3)
    # c_5 = 5 * (z_2 - z_4)
    # c_6 = z_0
    # c_3 = 100 * z_5 - 100 * c_6 - 10 * c_4 - 10 * c_5 - c_1 - c_2
    # x_max = (2 * c_1 * c_4 - c_3 * c_5) / (c_3 ** 2 - 4 * c_1 * c_2)
    # y_max = (2 * c_1 * c_5 - c_3 * c_4) / (c_3 ** 2 - 4 * c_1 * c_2)
    # return x_max, y_max

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


# def get_psf_(source, factor=2, position=0, star_idx=0, x_corr=0, y_corr=0):
#     psf_size = 11
#     half_size = int((psf_size - 1) / 2)
#     over_size = psf_size * factor + 1
#     # global flux_ratio, x_shift, y_shift, x_round, y_round, x_sign, y_sign
#     # nstars = source.nstars
#     size = source.size  # TODO: must be even?
#     flux_ratio = np.array(source.gaia['tess_flux_ratio'])
#     x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])
#     y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])
#     x_shift = x_shift + x_corr
#     y_shift = y_shift + y_corr
#
#     scale = 0.5
#     if position == 0:
#         pass
#     elif position == 1:
#         x_shift[star_idx] = x_shift[star_idx] + scale
#     elif position == 2:
#         y_shift[star_idx] = y_shift[star_idx] + scale
#     elif position == 3:
#         x_shift[star_idx] = x_shift[star_idx] - scale
#     elif position == 4:
#         y_shift[star_idx] = y_shift[star_idx] - scale
#     elif position == 5:
#         x_shift[star_idx] = x_shift[star_idx] + scale
#         y_shift[star_idx] = y_shift[star_idx] + scale
#
#     x_round = np.round(x_shift).astype(int)
#     y_round = np.round(y_shift).astype(int)
#
#     left = np.maximum(0, x_round - half_size)
#     right = np.minimum(size, x_round + half_size) + 1
#     down = np.maximum(0, y_round - half_size)
#     up = np.minimum(size, y_round + half_size) + 1
#     x_residual = x_shift % (1 / factor) * factor
#     y_residual = y_shift % (1 / factor) * factor
#
#     x_p = np.arange(size)
#     y_p = np.arange(size)
#     coord = np.arange(size ** 2).reshape(size, size)
#     A = np.zeros((size ** 2, over_size ** 2 + 1))
#     A[:, -1] = np.ones(size ** 2)
#     b = source.flux[0].flatten()
#
#     for i in range(len(source.gaia)):
#         x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + (x_shift[i] % 1) // (1 / factor)
#         y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + (y_shift[i] % 1) // (1 / factor)
#         # y_psf = y_p + (y_shift[i] % 1) // (1 / factor)
#         # x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + factor // 2 - 0.5 + x_sign[i] * (
#         #             x_offset[i] + 0.5)
#         # y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + factor // 2 - 0.5 + y_sign[i] * (
#         #             y_offset[i] + 0.5)
#         x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
#         a = np.array(x_psf + y_psf * over_size, dtype=np.int64)
#         a = a.flatten()
#         index = coord[down[i]:up[i], left[i]:right[i]]
#         A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
#             flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))
#
#     remove_index = (np.arange(over_size) + 1) * over_size - 1
#     diag = np.diag(np.ones(over_size ** 2))
#     A_1 = diag - np.concatenate((np.zeros((over_size ** 2, 1)), diag[:, 0: - 1]), axis=-1)
#     A_1 = np.delete(A_1, remove_index, 0)
#     A_1 = np.concatenate((A_1, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
#     A_2 = diag - np.concatenate((np.zeros((over_size ** 2, over_size)), diag[:, 0: - over_size]), axis=-1)
#     A_2 = A_2[0: - over_size]
#     A_2 = np.concatenate((A_2, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
#     A = np.append(A, A_1, axis=0)
#     A = np.append(A, A_2, axis=0)
#     b = np.append(b, np.zeros(2 * over_size * (over_size - 1)))
#     scaler = np.sqrt(source.flux_err[0].flatten() ** 2 + source.flux[0].flatten())
#     scaler = np.append(scaler, 1e3 * np.ones(2 * over_size * (over_size - 1)))
#     fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
#     fluxfit = np.dot(A, fit)
#     return fit, fluxfit, x_round, y_round


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
    # x_offset = np.abs(x_round - x_shift) // (1 / factor)
    # y_offset = np.abs(y_round - y_shift) // (1 / factor)
    star_info = []
    for i in range(len(source.gaia)):
        x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + (x_shift[i] % 1) // (1 / factor)
        y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + (y_shift[i] % 1) // (1 / factor)
        # y_psf = y_p + (y_shift[i] % 1) // (1 / factor)
        # x_psf = factor * (x_p[left[i]:right[i]] - x_round[i] + half_size) + factor // 2 - 0.5 + x_sign[i] * (
        #             x_offset[i] + 0.5)
        # y_psf = factor * (y_p[down[i]:up[i]] - y_round[i] + half_size) + factor // 2 - 0.5 + y_sign[i] * (
        #             y_offset[i] + 0.5)
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
    # dist = (x_coord ** 4 + y_coord ** 4) * 8e-7
    dist = (x_coord ** 2 + y_coord ** 2) * 5e-5 # 5e-5
    # # remove center compression
    # dist[18:27, 18:27] = 0
    remove_index = (np.arange(over_size) + 1) * over_size - 1
    diag = np.diag(np.ones(over_size ** 2))
    A_1 = diag - np.concatenate((np.zeros((over_size ** 2, 1)), diag[:, 0: - 1]), axis=-1)
    A_1 = np.delete(A_1, remove_index, 0)
    A_1 = np.concatenate((A_1, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
    A_2 = diag - np.concatenate((np.zeros((over_size ** 2, over_size)), diag[:, 0: - over_size]), axis=-1)
    A_2 = A_2[0: - over_size]
    A_2 = np.concatenate((A_2, (np.zeros((over_size * (over_size - 1), 1)))), axis=-1)
    A_3 = np.diag(dist.flatten())
    A_3 = np.concatenate((A_3, (np.zeros((over_size ** 2, 1)))), axis=-1)
    A = np.append(A, A_1, axis=0)
    A = np.append(A, A_2, axis=0)
    A = np.append(A, A_3, axis=0)
    return A, star_info, over_size, x_round, y_round


def reduced_A(A, star_info, star_num=0):
    info = star_info[star_num]
    A_ = np.zeros(np.shape(A))
    A_[info[0], info[1]] = info[2]
    return A - A_


def fit_psf(A, source, over_size, time=0, regularization=8e2):
    b = source.flux[time].flatten()
    b = np.append(b, np.zeros(2 * over_size * (over_size - 1)))
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = np.sqrt(source.flux_err[0].flatten() ** 2 + source.flux[time].flatten())
    scaler = np.append(scaler, regularization * np.ones(2 * over_size * (over_size - 1)))
    scaler = np.append(scaler, np.ones(over_size ** 2))
    fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    fluxfit = np.dot(A, fit)
    return fit, fluxfit


# try moffat for 77
if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    factor = 4
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor)
    fit, fluxfit = fit_psf(A, source, over_size, time=0, regularization=4e2) # 4e2
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
    # 6476


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


# fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # plot0 = ax[0].imshow(np.log10(source.flux[0]),
    #                      vmin=np.min(np.log10(source.flux[0])),
    #                      vmax=np.max(np.log10(source.flux[0])),
    #                      origin='lower')
    # plot1 = ax[1].imshow(np.log10(np.dot(r_A, epsf[j])[0:source.size ** 2].reshape(source.size, source.size)),
    #                      vmin=np.min(np.log10(source.flux[0])),
    #                      vmax=np.max(np.log10(source.flux[0])),
    #                      origin='lower')
    # residual = (source.flux[0] - np.dot(r_A, epsf[j])[0:source.size ** 2].reshape(source.size, source.size))
    # plot2 = ax[2].imshow(residual, origin='lower', vmin=- np.max(residual), vmax=np.max(residual), cmap='RdBu')
    # ax[0].set_title('Raw Data')
    # ax[1].set_title('ePSF Model')
    # ax[2].set_title('Residual')
    # fig.colorbar(plot0, ax=ax[0])
    # fig.colorbar(plot1, ax=ax[1])
    # fig.colorbar(plot2, ax=ax[2])
    # plt.show()


# fit, fluxfit, x_round, y_round = get_psf(source, factor=factor, position=0)
# left = np.maximum(0, x_round - 2)
# right = np.minimum(source.size, x_round + 2) + 1
# down = np.maximum(0, y_round - 2)
# up = np.minimum(source.size, y_round + 2) + 1
# num = np.arange(-0.05, 0.06, 0.01)
# z = np.zeros((11, 11))
# for i in range(11):
#     for j in range(11):
#         fit, fluxfit = get_psf_(source, factor=factor, star_idx=1, x_corr=num[i], y_corr=num[j])
#         residual = (source.flux[0] - fluxfit[0:source.size ** 2].reshape(source.size, source.size)).reshape(
#                         (source.size, source.size))
#         z[i][j] = np.sqrt(np.sum(np.square(residual[down[j]:up[j], left[j]:right[j]])))
#     print(i)

# fit, fluxfit, x_round, y_round = get_psf(source, factor=factor, position=0)
# left = np.maximum(0, x_round - 2)
# right = np.minimum(source.size, x_round + 2) + 1
# down = np.maximum(0, y_round - 2)
# up = np.minimum(source.size, y_round + 2) + 1
#
# z = np.zeros((6, 10))
# for j in range(10):
#     for i in range(6):
#         fit, fluxfit, x_round, y_round = get_psf(source, factor=factor, position=i, star_idx=j)
#         residual = (source.flux[0] - fluxfit[0:source.size ** 2].reshape(source.size, source.size)).reshape(
#             (source.size, source.size))
#         z[i, j] = np.sqrt(np.sum(np.square(residual[down[j]:up[j], left[j]:right[j]])))
#         plt.imshow(residual[down[j]:up[j], left[j]:right[j]], origin='lower')
#         plt.show()
#     x_max, y_max = paraboloid(z[0, j], z[1, j], z[2, j], z[3, j], z[4, j], z[5, j], scale=0.5)
#     print(z[:, j])
#     source.gaia[f'Sector_{source.sector}_x'][j] = source.gaia[f'Sector_{source.sector}_x'][j] + x_max
#     source.gaia[f'Sector_{source.sector}_y'][j] = source.gaia[f'Sector_{source.sector}_y'][j] + y_max
#     print(j, x_max, y_max)

# def get_psf(source):
#     psf_size = 11
#     half_size = int((psf_size - 1) / 2)
#     over_size = psf_size * 2 + 1
#     # global flux_ratio, x_shift, y_shift, x_round, y_round, x_sign, y_sign
#     # nstars = source.nstars
#     size = source.size  # TODO: must be even?
#     flux_ratio = np.array(source.gaia['tess_flux_ratio'])
#     x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])
#     y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])
#     x_round = np.round(x_shift).astype(int)
#     y_round = np.round(y_shift).astype(int)
#     x_sign = np.sign(x_shift - x_round)
#     y_sign = np.sign(y_shift - y_round)
#
#     left = np.maximum(0, x_round - half_size)
#     right = np.minimum(size, x_round + half_size) + 1
#     down = np.maximum(0, y_round - half_size)
#     up = np.minimum(size, y_round + half_size) + 1
#     x_residual = x_shift % 0.5
#     y_residual = y_shift % 0.5
#
#     x_p = np.arange(size)
#     y_p = np.arange(size)
#     coord = np.arange(size ** 2).reshape(size, size)
#     A = np.zeros((size ** 2, over_size ** 2 + 1))
#     A[:, -1] = np.ones(size ** 2)
#     b = source.flux[0].flatten()
#
#     for i in range(len(flux_ratio)):
#         x_psf = 2 * x_p[left[i]:right[i]] - 2 * x_round[i] + psf_size - 0.5 - 0.5 * x_sign[i]
#         y_psf = 2 * y_p[down[i]:up[i]] - 2 * y_round[i] + psf_size - 0.5 - 0.5 * y_sign[i]
#         x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
#         a = np.array(x_psf + y_psf * over_size, dtype=np.int64)
#         a = a.flatten()
#         index = coord[down[i]:up[i], left[i]:right[i]]
#         A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
#             flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))
#
#     # regularize
#     remove_index = (np.arange(size - 1) + 1) * size - 1
#     A_1 = np.diff(A, axis=0)
#     A_1 = np.delete(A_1, remove_index, 0)
#     b_1 = np.diff(source.flux[0].flatten())
#     b_1 = np.delete(b_1, remove_index)
#     vertical = np.arange(size ** 2).reshape(size, size).flatten(order='F')
#     A_2 = np.diff(A[vertical, :], axis=0)
#     A_2 = np.delete(A_2, remove_index, 0)
#     b_2 = np.diff(source.flux[0].flatten(order='F'))
#     b_2 = np.delete(b_2, remove_index)
#     A = np.append(A, A_1, axis=0)
#     A = np.append(A, A_2, axis=0)
#     b = np.append(b, b_1)
#     b = np.append(b, b_2)
#     scaler = np.sqrt(source.flux_err[0].flatten() ** 2 + source.flux[0].flatten())
#     scaler = np.append(scaler, 1e1 * np.ones(2 * size * (size - 1)))
#     fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
#     # fit = np.linalg.lstsq(A, b, rcond=None)[0]
#     fluxfit = np.dot(A, fit)
#     # np.lstsq(A/err[np.newaxis, :], b/err)
#     # err = np.sqrt(bg**2 + cts)
#     return fit, fluxfit

"""
# for one star, one pixel
size = 20  # TODO: must be even?
# unique for star
flux_ratio = np.array([1])
x_shift = np.array([5.3])
y_shift = np.array([4.6])
x_round = np.round(x_shift)
y_round = np.round(y_shift)
x_sign = np.sign(x_shift - x_round)
y_sign = np.sign(y_shift - y_round)

A = np.zeros((size ** 2, 121))  # all pixels * all x_i
for i in range(size ** 2):
    if x_round[0] - 2 <= x_p[i % size] <= x_round[0] + 2 and y_round[0] - 2 <= y_p[i // size] <= y_round[0] + 2:
        # coordinate of pixel [x_p - x_round + 2, y_p - y_round + 2]
        # coordinate of lower left psf [x_psf, y_psf]
        x_psf = 2 * x_p[i % size] - 2 * x_round[0] + 4.5 - 0.5 * x_sign[0]
        y_psf = 2 * y_p[i // size] - 2 * y_round[0] + 4.5 - 0.5 * y_sign[0]
        a = int(x_psf + y_psf * 11)
        index = [a, a + 1, a + 11, a + 12]
        A[i][np.array(index)] = flux_ratio[0] * bilinear(x, y)
"""

# plot interpolation
# x, y = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(fit[0:-1].reshape(23, 23), extent=[-5.75, 5.75, -5.75, 5.75], origin='lower')
# x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])[7]
# y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])[7]
# # ax.imshow(psf, extent=[-5.5 + x_shift, 5.5 + x_shift,
# #                        -5.5 + y_shift, 5.5 + y_shift],
# #           alpha=1,origin='lower', cmap='bone')
# ax.scatter(x + x_shift % 1 - 0.5, y + y_shift % 1 - 0.5, marker='s', facecolor='None', edgecolors='w', s=480)
# ax.plot(x + x_shift % 1 - 0.5, y + y_shift % 1 - 0.5, '.w', ms=3)
# plt.show()

# # Initializing value of x-axis and y-axis
# # in the range -1 to 1
# size = 20  # TODO: must be even?
# over_sample_size = size * 2 + 1
# x_shift = 15.3
# y_shift = 4.5
# # Intializing sigma and muu
# sigma = 1
# muu = 0.
#
# x_round = round(x_shift)
# y_round = round(y_shift)
#
# flux_cube = np.zeros((1, size, size))
#
# left = max(0, x_round - 2)
# right = min(size, x_round + 2) + 1
# down = max(0, y_round - 2)
# up = min(size, y_round + 2) + 1
#
# # Calculating Gaussian array
# x_psf, y_psf = np.meshgrid(np.linspace(-2.5, 2.5, 11), np.linspace(-2.5, 2.5, 11))
# dst = np.sqrt(x_psf * x_psf + y_psf * y_psf)
# gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
# interp = interpolate.RectBivariateSpline(np.linspace(-2.5, 2.5, 11), np.linspace(-2.5, 2.5, 11), gauss)
# x, y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
# psf = interp.ev(x - x_shift + x_round, y - y_shift + y_round)
#
# flux_cube[0, down:up, left:right] = psf[max(0, 2 - x_round):min(5, size - x_round + 3),
#                                     max(0, 2 - y_round):min(5, size - y_round + 3)]
#
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(flux_cube[0], vmin=0, vmax=1, origin='lower', cmap='gray')
# ax.scatter(x_shift, y_shift)
# plt.show()

# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(gauss, extent=[-2.75, 2.75, -2.75, 2.75], vmin=0, vmax=1, origin='lower', cmap='gray')
# ax.imshow(psf, extent=[-2.5 + x_shift, 2.5 + x_shift, -2.5 + y_shift, 2.5 + y_shift], vmin=-0.1, vmax=1, alpha=1,
#           origin='lower', cmap='bone')
# ax.scatter(x + x_shift, y + y_shift, marker='s', facecolor='None', edgecolors='w', s=2200)
# ax.plot(x + x_shift, y + y_shift, '.w', ms=3)
# ax.scatter(x_psf, y_psf, s=np.log(gauss + 1.01) * 30, c='r')
# plt.show()

# def blockshaped(arr, nrows, ncols):
#     """
#     Return an array of shape (n, nrows, ncols) where
#     n * nrows * ncols = arr.size
#     If arr is a 2D array, the returned array should look like n subblocks with
#     each subblock preserving the "physical" layout of arr.
#     """
#     h, w = arr.shape
#     assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#     assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
#     return arr.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)
#
#
# def unblockshaped(arr, h, w):
#     """
#     Return an array of shape (h, w) where
#     h * w = arr.size
#
#     If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
#     then the returned array preserves the "physical" layout of the sublocks.
#     """
#     n, nrows, ncols = arr.shape
#     return arr.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)

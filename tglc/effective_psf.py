import numpy as np
from scipy.linalg import lstsq
from wotan import flatten
import tglc
from tglc.pixel_quality import build_pixel_mask


def _source_pixel_mask(source, time, saturation_limit, saturation_dilation, extra_pixel_mask=None):
    base = np.ma.getmaskarray(source.mask)
    if base.shape != source.flux[time].shape:
        base = np.zeros(source.flux[time].shape, dtype=bool)
    supplied = getattr(source, 'pixel_mask', None)
    if supplied is not None:
        supplied = np.asarray(supplied, dtype=bool)
        if supplied.shape == source.flux.shape:
            supplied = supplied[time]
        if supplied.shape != base.shape:
            raise ValueError('source.pixel_mask must match an image or the flux cube')
        base = base | supplied
    if extra_pixel_mask is not None:
        extra = np.asarray(extra_pixel_mask, dtype=bool)
        if extra.shape != base.shape:
            raise ValueError('extra_pixel_mask must match the image shape')
        base = base | extra
    return build_pixel_mask(source.flux[time], base, saturation_limit, saturation_dilation)


def _fit_linear_model(matrix, values):
    """Solve finite, full-rank systems without squaring their condition number."""
    matrix, values = np.asarray(matrix), np.asarray(values)
    failed = np.full(matrix.shape[1], np.nan)
    valid = np.isfinite(values) & np.isfinite(matrix).all(axis=1)
    if np.count_nonzero(valid) < matrix.shape[1]:
        return failed
    try:
        fit, _, rank, _ = lstsq(matrix[valid], values[valid], lapack_driver='gelsy', check_finite=False)
    except (ValueError, np.linalg.LinAlgError):
        return failed
    return fit if rank == matrix.shape[1] and np.isfinite(fit).all() else failed


def _psf_axis_samples(pixels, position, factor, over_size):
    """Sample the centered ePSF at detector-pixel minus actual star position."""
    coordinates = (pixels - position) * factor + over_size // 2
    nodes = np.floor(coordinates).astype(int)
    # At the last grid node use the interval to its left with weight one.
    # All detector pixels have the same subpixel phase, so shift the entire
    # sequence together and retain the historical four-weights star_info API.
    if len(nodes) and np.max(nodes) == over_size - 1:
        nodes -= 1
    fraction = coordinates[0] - nodes[0] if len(nodes) else 0.0
    if len(nodes) and (np.min(nodes) < 0 or np.max(nodes) >= over_size - 1):
        raise ValueError('Detector samples fall outside the supported PSF grid')
    return nodes, fraction


def _is_full_frame_source(source):
    # Importing the numerical module must not require catalog/network clients.
    return type(source) is getattr(getattr(tglc, 'ffi', None), 'Source', None)


def _target_psf_model(source, star_num, e_psf, factor, psf_size):
    """Evaluate a target over its full support, including pixels outside the cutout."""
    over_size = psf_size * factor + 1
    x = float(source.gaia[f'sector_{source.sector}_x'][star_num])
    y = float(source.gaia[f'sector_{source.sector}_y'][star_num])
    offsets = np.arange(psf_size) - psf_size // 2
    ix, dx = _psf_axis_samples(offsets + round(x), x, factor, over_size)
    iy, dy = _psf_axis_samples(offsets + round(y), y, factor, over_size)
    nodes = (ix[None, :] + over_size * iy[:, None]).flatten()
    indices = np.stack((nodes, nodes + 1, nodes + over_size, nodes + over_size + 1), axis=1)
    weights = bilinear(dx, dy) * source.gaia['tess_flux_ratio'][star_num]
    return (e_psf[:, indices] @ weights).reshape(len(e_psf), psf_size, psf_size)


def _aperture_portion(psf_shape, source_size, x, y):
    """Fraction in the available central 3x3, relative to complete PSF support."""
    half = psf_shape.shape[1] // 2
    valid = np.isfinite(psf_shape).all(axis=(1, 2))
    model = psf_shape[valid]
    denominator = np.sum(model)
    if not model.size or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    left, right = max(half - 1, half - int(x)), min(half + 2, half + source_size - int(x))
    down, up = max(half - 1, half - int(y)), min(half + 2, half + source_size - int(y))
    return np.sum(model[:, down:up, left:right]) / denominator


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
    if factor <= 0 or int(factor) != factor or factor % 2 != 0:
        raise ValueError('Factor must be a positive even integer.')
    if psf_size <= 0 or int(psf_size) != psf_size or psf_size % 2 != 1:
        raise ValueError('psf_size must be a positive odd integer.')
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

    left = np.clip(x_round - half_size, 0, size)
    right = np.clip(x_round + half_size + 1, 0, size)
    down = np.clip(y_round - half_size, 0, size)
    up = np.clip(y_round + half_size + 1, 0, size)

    x_p = np.arange(size)
    y_p = np.arange(size)
    coord = np.arange(size ** 2).reshape(size, size)
    xx, yy = np.meshgrid((np.arange(size) - (size - 1) / 2), (np.arange(size) - (size - 1) / 2))

    if _is_full_frame_source(source):
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
        x_psf, x_residual = _psf_axis_samples(x_p[left[i]:right[i]], x_shift[i], factor, over_size)
        y_psf, y_residual = _psf_axis_samples(y_p[down[i]:up[i]], y_shift[i], factor, over_size)
        x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
        a = np.array(x_psf + y_psf * over_size, dtype=np.int64).flatten()
        index = coord[down[i]:up[i], left[i]:right[i]]
        A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
            flux_ratio[i] * bilinear(x_residual, y_residual, repeat=len(a))
        # star_info.append(
        #     (np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
        #      flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))))
        star_info.append(
            (index, a, flux_ratio[i] * bilinear(x_residual, y_residual)))
    coord_ = np.arange(over_size) - over_size // 2
    x_coord, y_coord = np.meshgrid(coord_, coord_)
    variance = psf_size
    dist = (1 - np.exp(- 0.5 * (x_coord ** 4 + y_coord ** 4) / variance ** 4)) * edge_compression  # 1e-3
    A_mod = np.diag(dist.flatten())
    A_mod = np.concatenate((A_mod, (np.zeros((over_size ** 2, bg_dof)))), axis=-1)
    A = np.append(A, A_mod, axis=0)
    return A, star_info, over_size, x_round, y_round


def fit_psf(A, source, over_size, power=0.8, time=0, saturation_limit=80000.0,
            saturation_dilation=1, extra_pixel_mask=None):
    """
    Fit the ePSF with weighted, rank-revealing least squares.

    Saturation thresholds are in e-/s. Persistent masks, per-cadence saturation,
    nonfinite pixels, and zero-weight pixels are excluded. An underdetermined or
    numerically failed fit returns NaN parameters rather than plausible values.
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
    :param saturation_limit: float or None, optional
    Pixel saturation threshold in e-/s; None disables the brightness threshold.
    :param saturation_dilation: int, optional
    Number of adjacent-pixel dilation iterations around saturated pixels.
    :param extra_pixel_mask: array, optional
    Additional two-dimensional mask, with True indicating excluded pixels.
    :return: fit result
    """
    saturated_index = _source_pixel_mask(source, time, saturation_limit, saturation_dilation,
                                         extra_pixel_mask).flatten()
    flux = source.flux[time].flatten()
    usable = ~saturated_index
    if not np.any(usable):
        return np.full(A.shape[1], np.nan)
    saturated_index |= flux < 0.8 * np.median(flux[usable])

    b = np.delete(flux, saturated_index)
    scaler = np.abs(np.delete(flux, saturated_index)) ** power
    b = np.append(b, np.zeros(over_size ** 2))
    scaler = np.append(scaler, np.ones(over_size ** 2))

    # fit = np.linalg.lstsq(A / scaler[:, np.newaxis], b / scaler, rcond=None)[0]
    a = np.delete(A, np.where(saturated_index), 0) / scaler[:, np.newaxis]
    b = b / scaler
    return _fit_linear_model(a, b)


def fit_lc(A, source, star_info=None, x=0., y=0., star_num=0, factor=2, psf_size=11, e_psf=None,
           near_edge=False, saturation_limit=80000.0, saturation_dilation=1, extra_pixel_mask=None):
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
    left, right = max(0, int(x) - cut_size // 2), min(size, int(x) + cut_size // 2 + 1)
    down, up = max(0, int(y) - cut_size // 2), min(size, int(y) + cut_size // 2 + 1)
    near_edge = near_edge or right - left != cut_size or up - down != cut_size
    coord = np.arange(size ** 2).reshape(size, size)
    index = np.array(coord[down:up, left:right]).flatten()
    A_cut = np.zeros((len(index), np.shape(A)[1]))
    A_target = np.zeros((len(index), np.shape(A)[1]))
    for i in range(len(index)):
        A_ = np.zeros(np.shape(A)[-1])
        star_pos = np.where(star_info_num[0] == index[i])[0]
        A_[star_info_num[1][star_pos]] = star_info_num[2][star_pos]
        A_target[i] = A_
        A_cut[i] = A[index[i], :] - A_
    aperture = np.zeros((len(source.time), len(index)))
    for j in range(len(source.time)):
        aperture[j] = np.array(source.flux[j][down:up, left:right]).flatten() - np.dot(A_cut, e_psf[j])
    aperture = aperture.reshape((len(source.time), up - down, right - left))
    h = up - down
    w = right - left
    target_plane = np.dot(A_target, np.nanmedian(e_psf, axis=0)).reshape(h, w)
    field_plane = np.dot(A_cut, np.nanmedian(e_psf, axis=0)).reshape(h, w)
    target_5x5 = np.full((cut_size, cut_size), np.nan)
    field_stars_5x5 = np.full((cut_size, cut_size), np.nan)
    y0 = int(down - y + cut_size // 2)
    x0 = int(left - x + cut_size // 2)
    target_5x5[y0:y0 + h, x0:x0 + w] = target_plane
    field_stars_5x5[y0:y0 + h, x0:x0 + w] = field_plane

    # psf_lc
    over_size = psf_size * factor + 1
    psf_shape = _target_psf_model(source, star_num, e_psf, factor, psf_size)
    portion = _aperture_portion(psf_shape, size, x, y)
    if near_edge:  # TODO: near_edge
        psf_lc = np.zeros(len(source.time))
        psf_lc[:] = np.nan
        return aperture, psf_lc, y - down, x - left, portion, target_5x5, field_stars_5x5
    left_ = left - x + psf_size // 2
    right_ = right - x + psf_size // 2
    down_ = down - y + psf_size // 2
    up_ = up - y + psf_size // 2
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
    med_aperture = np.ma.median(np.ma.masked_invalid(aperture), axis=0).filled(np.nan).flatten()
    edge_flux = med_aperture[edge_pixel]
    finite_edge = edge_flux[np.isfinite(edge_flux)]
    outliers = np.zeros(len(edge_pixel), dtype=bool)
    if finite_edge.size:
        outliers = np.abs(edge_flux - np.median(finite_edge)) > np.std(finite_edge)
    epsf_sum = np.sum(np.nanmedian(psf_shape, axis=0))
    for j in range(len(source.time)):
        if np.isnan(psf_sim[j, :, :]).any():
            psf_lc[j] = np.nan
        else:
            aper_flat = aperture[j, :, :].flatten()
            A_[:, 0] = psf_sim[j, :, :].flatten() / epsf_sum
            bad = _source_pixel_mask(source, j, saturation_limit, saturation_dilation,
                                     extra_pixel_mask)[down:up, left:right].flatten()
            bad[edge_pixel[outliers]] = True
            psf_lc[j] = _fit_linear_model(A_[~bad], aper_flat[~bad])[0]
    # print(np.nansum(psf_shape[:, 5, 5]) / np.nansum(psf_shape))
    # np.save(f'toi-5344_psf_{source.sector}.npy', psf_shape)
    return aperture, psf_lc, y - down, x - left, portion, target_5x5, field_stars_5x5


def fit_lc_float_field(A, source, star_info=None, x=np.array([]), y=np.array([]), star_num=0, factor=2, psf_size=11,
                       e_psf=None, near_edge=False, prior=0.001, saturation_limit=80000.0,
                       saturation_dilation=1, extra_pixel_mask=None):
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
    if not np.isfinite(prior) or prior <= 0:
        raise ValueError('prior must be positive and finite')
    over_size = psf_size * factor + 1
    a = star_info[star_num][1]
    star_info_num = (np.repeat(star_info[star_num][0], 4),
                     np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
                     np.tile(star_info[star_num][2], len(a)))
    size = source.size  # TODO: must be even?
    # star_position = int(x + source.size * y - 5 * size - 5)
    # aper_lc
    cut_size = 5
    left = max(0, int(x[star_num]) - cut_size // 2)
    right = min(size, int(x[star_num]) + cut_size // 2 + 1)
    down = max(0, int(y[star_num]) - cut_size // 2)
    up = min(size, int(y[star_num]) + cut_size // 2 + 1)
    near_edge = near_edge or right - left != cut_size or up - down != cut_size
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
    target_model = _target_psf_model(source, star_num, e_psf, factor, psf_size)
    portion = _aperture_portion(target_model, size, x[star_num], y[star_num])
    if near_edge:  # TODO: near_edge
        psf_lc = np.zeros(len(source.time))
        psf_lc[:] = np.nan
        return aperture, psf_lc, y[star_num] - down, x[star_num] - left, portion
    # left_ = left - x[star_num] + 5
    # right_ = right - x[star_num] + 5
    # down_ = down - y[star_num] + 5
    # up_ = up - y[star_num] + 5
    if _is_full_frame_source(source):
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
    psf_sim = np.zeros((len(source.time), psf_size ** 2 + len(field_star_num), len(field_star_num)))
    coord = np.arange(psf_size ** 2).reshape(psf_size, psf_size)
    center = psf_size // 2
    window_indices = coord[center - cut_size // 2:center + cut_size // 2 + 1,
                           center - cut_size // 2:center + cut_size // 2 + 1]
    for j, star in enumerate(field_star_num):
        a = star_info[star][1]
        star_info_star = (np.repeat(star_info[star][0], 4),
                          np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F'),
                          np.tile(star_info[star][2], len(a)))
        delta_x = x[star_num] - x[star]
        delta_y = y[star_num] - y[star]
        # for psf_sim
        left_shift = np.maximum(delta_x, 0)
        right_shift = np.minimum(psf_size + delta_x, psf_size)
        down_shift = np.maximum(delta_y, 0)
        up_shift = np.minimum(psf_size + delta_y, psf_size)
        # for psf_shape
        left_shift_ = np.maximum(-delta_x, 0)
        right_shift_ = np.minimum(psf_size - delta_x, psf_size)
        down_shift_ = np.maximum(-delta_y, 0)
        up_shift_ = np.minimum(psf_size - delta_y, psf_size)

        psf_shape = _target_psf_model(source, star, e_psf, factor, psf_size)
        epsf_sum = np.sum(np.nanmedian(psf_shape, axis=0))
        psf_sim_index = coord[down_shift:up_shift, left_shift:right_shift].flatten()
        psf_sim[:, psf_sim_index, j] = psf_shape[:, down_shift_:up_shift_, left_shift_:right_shift_].reshape(
            len(source.time), -1) / epsf_sum
        if star != star_num:
            psf_sim[:, psf_size ** 2 + j, j] = np.ones(len(source.time)) / (
                    prior * 1.5e4 * 10 ** ((10 - source.gaia[star]['tess_mag']) / 2.5))

    star_index = np.where(np.array(field_star_num) == star_num)[0]
    field_star = psf_sim[0, window_indices, :].reshape(cut_size ** 2,
                                                                                     len(field_star_num)) * \
                 source.gaia['tess_flux_ratio'][field_star_num]
    field_star[:, star_index] = 0
    for j in range(len(source.time)):
        if np.isnan(psf_sim[j, :, :]).any():
            psf_lc[j] = np.nan
        else:
            aper_flat = aperture[j, :, :].flatten()
            aper_flat = np.append(aper_flat, np.zeros(len(field_star_num) - 1))  # / prior
            postcards = psf_sim[j, window_indices, :].reshape(cut_size ** 2,
                                                                                            len(field_star_num))
            A_[:cut_size ** 2, :len(field_star_num)] = postcards
            field_star = postcards * source.gaia['tess_flux_ratio'][field_star_num]
            field_star[:, star_index] = 0
            # A_[:(cut_size ** 2), -4] = np.sum(field_star, axis=1)
            A_[cut_size ** 2:, :len(field_star_num)] = psf_sim[j, psf_size ** 2:, :].reshape(len(field_star_num),
                                                                                       len(field_star_num))
            a = np.delete(A_, cut_size ** 2 + star_index, 0)
            bad = _source_pixel_mask(source, j, saturation_limit, saturation_dilation,
                                     extra_pixel_mask)[down:up, left:right].flatten()
            bad = np.append(bad, np.zeros(len(field_star_num) - 1, dtype=bool))
            psf_lc[j] = _fit_linear_model(a[~bad], aper_flat[~bad])[star_index[0]]
    return aperture, psf_lc, y[star_num] - down, x[star_num] - left, portion


def _detrend_flux(time, flux):
    finite = np.isfinite(flux) & np.isfinite(time)
    if np.count_nonzero(finite) < 3:
        return np.full(len(flux), np.nan)
    median = np.median(flux[finite])
    if median == 0:
        return np.full(len(flux), np.nan)
    normalized = flux / median
    # Keep the legacy detrending guard without censoring the signed flux product.
    normalized[(normalized > 100) | ~finite] = np.nan
    usable = np.isfinite(normalized)
    if np.count_nonzero(usable) < 3:
        return np.full(len(flux), np.nan)
    shifted = normalized - np.min(normalized[usable]) + 1000
    _, trend = flatten(time, shifted, window_length=1, method='biweight', return_trend=True)
    return (shifted - trend) / np.median(normalized[usable]) + 1


def bg_mod(source, q=None, aper_lc=None, psf_lc=None, portion=None, star_num=0, near_edge=False,
           return_offsets=False):
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
    :param return_offsets: boolean, optional
    Return (aperture offset, PSF offset) as the first item. The historical default
    returns just the PSF offset. Linear flux retains negative values; an empty
    valid reference set produces a missing offset and missing normalized flux.
    :return: local background, modified aperture light curve, modified PSF light curve
    '''
    bar = 15000 * 10 ** ((source.gaia['tess_mag'][star_num] - 10) / -2.5)
    # print(bar)
    # med_epsf = np.nanmedian(e_psf[:, :23 ** 2].reshape(len(source.time), 23, 23), axis=0)
    # centroid_to_aper_ratio = 4/9 * np.sum(med_epsf[10:13, 10:13]) / np.sum(med_epsf)
    # centroid_to_aper_ratio = np.nanmedian(ratio)
    # flux_bar = aperture_bar * centroid_to_aper_ratio
    # lightcurve = lightcurve + (flux_bar - np.nanmedian(lightcurve[q]))
    aperture_bar = bar * portion
    # print(bar)
    if q is None:
        q = slice(None)
    aper_reference = np.asarray(aper_lc)[q]
    aper_reference = aper_reference[np.isfinite(aper_reference)]
    aper_bg = np.median(aper_reference) - aperture_bar if aper_reference.size else np.nan
    aper_lc = np.asarray(aper_lc, dtype=float) - aper_bg
    psf_bar = bar
    psf_reference = np.asarray(psf_lc)[q]
    psf_reference = psf_reference[np.isfinite(psf_reference)]
    psf_bg = np.median(psf_reference) - psf_bar if psf_reference.size else np.nan
    psf_lc = np.asarray(psf_lc, dtype=float) - psf_bg
    local_bg = (aper_bg, psf_bg) if return_offsets else psf_bg
    cal_aper_lc = _detrend_flux(source.time, aper_lc)
    cal_psf_lc = psf_lc.copy() if near_edge else _detrend_flux(source.time, psf_lc)
    return local_bg, aper_lc, psf_lc, cal_aper_lc, cal_psf_lc

import os
import pickle
import numpy as np
import astroquery.mast
from glob import glob
from astropy.table import Table, hstack
from astroquery.mast import Catalogs
from tqdm import tqdm, trange
from astropy.io import fits
from astropy.wcs import WCS

class Source(object):
    def __init__(self, x=0, y=0, flux=None, time=None, wcs=None, quality=None, exposure=1800, sector=0, size=150,
                 camera=1, ccd=1, cadence=None):
        """
        Source object that includes all data from TESS and Gaia DR2
        :param x: int, required
        starting horizontal pixel coordinate
        :param y: int, required
        starting vertical pixel coordinate
        :param flux: np.ndarray, required
        3d data cube, the time series of a all FFI from a CCD
        :param time: np.ndarray, required
        1d array of time
        :param wcs: astropy.wcs.wcs.WCS, required
        WCS Keywords of the TESS FFI
        :param sector: int, required
        TESS sector number
        :param size: int, optional
        the side length in pixel  of TESScut image
        :param camera: int, optional
        camera number
        :param ccd: int, optional
        CCD number
        :param cadence: list, required
        list of cadences of TESS FFI
        """
        super(Source, self).__init__()
        if cadence is None:
            cadence = []
        if quality is None:
            quality = []
        if wcs is None:
            wcs = []
        if time is None:
            time = []
        if flux is None:
            flux = []
        coord = wcs.pixel_to_world([x + (size - 1) / 2 + 44], [y + (size - 1) / 2])[0].to_string()
        self.size = size
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.cadence = cadence
        self.quality = quality
        self.exposure = exposure
        with astroquery.mast.conf.set_temp('timeout', 1800):
            catalogdata = Catalogs.query_object(coord, radius=(self.size + 6) * 21 * 0.707 / 3600,
                                            catalog="Gaia", version=2)
            # print(f'Found {len(catalogdata)} Gaia DR2 objects.')
            catalogdata_tic = Catalogs.query_object(coord, radius=(self.size + 6) * 21 * 0.707 / 3600,
                                                catalog="TIC")
            # print(f'Found {len(catalogdata_tic)} TIC objects.')
        self.tic = catalogdata_tic['ID', 'GAIA']
        self.catalogdata = catalogdata
        self.flux = flux[:len(time), y:y + size, x:x + size]
        self.time = np.array(time)
        self.wcs = wcs
        gaia_targets = self.catalogdata[
            'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec']
        x_gaia = np.zeros(len(gaia_targets))
        y_gaia = np.zeros(len(gaia_targets))
        tess_mag = np.zeros(len(gaia_targets))
        in_frame = [True] * len(gaia_targets)
        for i, designation in enumerate(gaia_targets['designation']):
            pixel = self.wcs.all_world2pix(
                np.array([gaia_targets['ra'][i], gaia_targets['dec'][i]]).reshape((1, 2)), 0, quiet=True)
            x_gaia[i] = pixel[0][0] - x - 44
            y_gaia[i] = pixel[0][1] - y
            if -4 < x_gaia[i] < self.size + 3 and -4 < y_gaia[i] < self.size + 3:
                dif = gaia_targets['phot_bp_mean_mag'][i] - gaia_targets['phot_rp_mean_mag'][i]
                tess_mag[i] = gaia_targets['phot_g_mean_mag'][
                                  i] - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
                if np.isnan(tess_mag[i]):
                    tess_mag[i] = gaia_targets['phot_g_mean_mag'][i] - 0.430
            else:
                in_frame[i] = False

        tess_flux = 10 ** (- tess_mag / 2.5)
        t = Table()
        t[f'tess_mag'] = tess_mag[in_frame]
        t[f'tess_flux'] = tess_flux[in_frame]
        t[f'tess_flux_ratio'] = tess_flux[in_frame] / np.max(tess_flux[in_frame])
        t[f'sector_{self.sector}_x'] = x_gaia[in_frame]
        t[f'sector_{self.sector}_y'] = y_gaia[in_frame]
        gaia_targets = hstack([gaia_targets[in_frame], t])  # TODO: sorting not sorting all columns
        gaia_targets.sort('tess_mag')
        self.gaia = gaia_targets


def cut_ffi(ccd=1, camera=1, sector=1, size=150, path=''):
    """
    Generate Source object from the calibrated FFI downloaded directly from MAST
    :param sector: int, required
    TESS sector number
    :param camera: int, required
    camera number
    :param ccd: int, required
    ccd number
    :param size: int, optional
    size of the FFI cut, default size is 150. Recommend large number for better quality.
    :param path: string, required
    path to the FFI folder
    :return:
    """
    input_files = glob(f'{path}ffi/*{camera}-{ccd}-????-?_ffic.fits')
    print('camera: ' + str(camera) + '  ccd: ' + str(ccd) + '  num of files: ' + str(len(input_files)))
    time = []
    quality = []
    cadence = []
    flux = np.empty((len(input_files), 2048, 2048), dtype=np.float32)
    for i, file in enumerate(tqdm(input_files)):
        with fits.open(file, mode='denywrite') as hdul:
            quality.append(hdul[1].header['DQUALITY'])
            cadence.append(hdul[0].header['FFIINDEX'])
            time.append((hdul[1].header['TSTOP'] + hdul[1].header['TSTART']) / 2)
            flux[i] = hdul[1].data[0:2048, 44:2092]  # TODO: might be different for other CCD: seems the same
    time_order = np.argsort(np.array(time))
    time = np.array(time)[time_order]
    flux = flux[time_order, :, :]

    # np.save(path + f'source/sector{sector}_time.npy', time)
    hdul = fits.open(input_files[np.where(np.array(quality) == 0)[0][0]])
    wcs = WCS(hdul[1].header)
    exposure = int((hdul[0].header['TSTART'] - hdul[0].header['TSTOP']) * 86400)

    # 95*95 cuts with 2 pixel redundant, (22*22 cuts)
    # try 77*77 with 4 redundant, (28*28 cuts)
    os.makedirs(path + f'source/{camera}-{ccd}/', exist_ok=True)
    for i in trange(14):  # 22
        for j in range(14):  # 22
            with open(path + f'source/{camera}-{ccd}/source_{i:02d}_{j:02d}.pkl', 'wb') as output:
                source = Source(x=i * (size - 4), y=j * (size - 4), flux=flux, sector=sector, time=time, size=size,
                                quality=quality, wcs=wcs,
                                exposure=exposure, cadence=cadence)  # 93
                pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)

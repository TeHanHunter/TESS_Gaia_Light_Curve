from glob import glob

import numpy as np
from astropy.table import Table, hstack
from astroquery.mast import Catalogs
from tqdm import tqdm

from TGLC.target_lightcurve import *
import pickle


class Source(object):
    def __init__(self, x=0, y=0, flux=[], time=[], wcs=[], sector=0, size=95, camera=1,
                 ccd=1):
        """
        :param name: str or float
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. 351.40691 61.646657)
        :param size: int, optional
        The side length in pixel  of TESScut image
        """
        super(Source, self).__init__()
        coord = wcs.pixel_to_world([x + 47], [y + 47])[0].to_string()
        self.size = size
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        catalogdata = Catalogs.query_object(coord, radius=(self.size + 2) * 21 * 0.707 / 3600,
                                            catalog="Gaia", version=2)
        # TODO: maybe increase search radius
        print(f'Found {len(catalogdata)} Gaia DR2 objects.')
        self.catalogdata = catalogdata
        self.flux = flux[:len(time), y:y + 95, x:x + 95]
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
                np.array([gaia_targets['ra'][i], gaia_targets['dec'][i]]).reshape((1, 2)), 0)
            x_gaia[i] = pixel[0][0] - x - 44
            y_gaia[i] = pixel[0][1] - y
            if -2 < x_gaia[i] < self.size + 1 and -2 < y_gaia[i] < self.size + 1:
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


def cut_ffi(sector=24, ccd='4-3', path='/mnt/d/TESS_Sector_24/'):
    input_files = glob(path + '*' + ccd + '-????-?_ffic.fits')
    time = []
    bad_quality = []
    flux = np.empty((len(input_files), 2048, 2048))
    for i, file in enumerate(tqdm(input_files)):
        with fits.open(file, mode='denywrite') as hdul:
            if hdul[1].header['DQUALITY'] == 0:
                time.append((hdul[1].header['TSTOP'] + hdul[1].header['TSTART']) / 2)
                flux_ = hdul[1].data[0:2048, 44:2092]  # TODO: might be different for other CCD: seems the same
                flux[i - len(bad_quality)] = flux_
            else:
                bad_quality.append(i)
    good_quality = list(set((list(range(0, len(input_files))))) - set(bad_quality))
    np.save(path + f'/sector{sector}_time.npy', time)
    hdul = fits.open(input_files[good_quality[0]])
    wcs = WCS(hdul[1].header)

    # 95*95 cuts with 2 pixel redundant, 22*22 cuts
    for i in trange(22):
        for j in range(22):
            with open(path + ccd + f'/source_{i}_{j}.pkl', 'wb') as output:
                source = Source(x=i * 93, y=j * 93, flux=flux, sector=sector, time=time, wcs=wcs)
                pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cut_ffi(sector=17, ccd='3-2', path='/mnt/d/TESS_Sector_17/')

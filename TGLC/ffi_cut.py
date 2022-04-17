import os
import sys
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
import pickle
from os.path import exists
from TGLC.target_lightcurve import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Source_cut(object):
    def __init__(self, name, size=15, cadence=[]):
        """
        Source_cut object that includes all data from TESS and Gaia DR2
        :param name: str, required
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. '351.40691 61.646657')
        :param size: int, optional
        The side length in pixel  of TESScut image
        :param cadence: list, required
        list of cadences of TESS FFI
        """
        super(Source_cut, self).__init__()
        self.name = name
        self.size = size
        self.sector = 0
        self.wcs = []
        self.time = []
        self.flux = []
        self.flux_err = []
        self.gaia = []
        self.cadence = cadence
        self.quality = []
        catalogdata = Catalogs.query_object(self.name, radius=(self.size + 2) * 21 * 0.707 / 3600,
                                            catalog="Gaia", version=2)
        print(catalogdata[0]['designation'])
        # TODO: maybe increase search radius
        print(f'Found {len(catalogdata)} Gaia DR2 objects.')
        ra = catalogdata[0]['ra']
        dec = catalogdata[0]['dec']
        coord = SkyCoord(ra, dec, unit="deg")
        catalogdata_tic = Catalogs.query_object(coord.to_string(), radius=(self.size + 2) * 21 * 0.707 / 3600,
                                            catalog="TIC")
        print(f'Found {len(catalogdata_tic)} TIC objects.')
        self.tic = catalogdata_tic['ID', 'GAIA']
        sector_table = Tesscut.get_sectors(coord)
        hdulist = Tesscut.get_cutouts(coord, self.size)
        self.catalogdata = catalogdata
        self.sector_table = sector_table
        self.camera = int(sector_table[0]['camera'])
        self.ccd = int(sector_table[0]['ccd'])
        self.hdulist = hdulist
        self.select_sector(sector=sector_table['sector'][0])

    def select_sector(self, sector=1):
        """
        select sector to use if target is in multi-sectors
        :param sector: int, required
        TESS sector number
        :return:
        """
        if self.sector == sector:
            print(f'Already in sector {sector}.')
            return
        elif sector not in self.sector_table['sector']:
            print(f'Sector {sector} does not cover this region. Please refer to sector table.')
            return
        index = list(self.sector_table['sector']).index(sector)
        self.sector = sector
        self.camera = int(self.sector_table[index]['camera'])
        self.ccd = int(self.sector_table[index]['ccd'])
        hdu = self.hdulist[index]
        wcs = WCS(hdu[2].header)
        data_time = hdu[1].data['TIME']
        data_flux = hdu[1].data['FLUX']
        data_flux_err = hdu[1].data['FLUX_ERR']
        data_quality = hdu[1].data['QUALITY']
        data_time = data_time[np.where(data_quality == 0)]
        data_flux = data_flux[np.where(data_quality == 0), :, :][0]
        data_flux_err = data_flux_err[np.where(data_quality == 0), :, :][0]
        self.wcs = wcs
        self.time = data_time
        self.flux = data_flux
        self.flux_err = data_flux_err
        self.quality = np.zeros(len(data_time))

        gaia_targets = self.catalogdata[
            'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec']
        x = np.zeros(len(gaia_targets))
        y = np.zeros(len(gaia_targets))
        tess_mag = np.zeros(len(gaia_targets))
        in_frame = [True] * len(gaia_targets)
        for i, designation in enumerate(gaia_targets['designation']):
            pixel = self.wcs.all_world2pix(
                np.array([gaia_targets['ra'][i], gaia_targets['dec'][i]]).reshape((1, 2)), 0)
            x[i] = pixel[0][0]
            y[i] = pixel[0][1]
            if -2 < x[i] < self.size + 1 and -2 < y[i] < self.size + 1:
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
        t[f'sector_{self.sector}_x'] = x[in_frame]
        t[f'sector_{self.sector}_y'] = y[in_frame]
        gaia_targets = hstack([gaia_targets[in_frame], t])  # TODO: sorting not sorting all columns
        gaia_targets.sort('tess_mag')
        self.gaia = gaia_targets


def ffi(target='', local_directory='', size=90):
    """
    Function to generate Source_cut objects
    :param target: string, required
    target name
    :param local_directory: string, required
    output directory
    :param size: int, required
    FFI cut side length
    :return: TGLC.ffi_cut.Source_cut
    """
    source_exists = exists(f'{local_directory}source_{target}.pkl')
    if source_exists:
        with open(f'{local_directory}source_{target}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        print('Loaded ffi from directory. ')
    else:
        with open(f'{local_directory}source_{target}.pkl', 'wb') as output:
            source = Source_cut(target, size=size)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
    return source


if __name__ == '__main__':
    target = 'NGC 7654'  # Target identifier or coordinates TOI-3714
    # TODO: power?

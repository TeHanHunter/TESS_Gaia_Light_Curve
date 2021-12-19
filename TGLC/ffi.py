import os
import sys
import warnings
import numpy as np
from astropy.wcs import WCS
from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Source(object):
    def __init__(self, name, size=15, sector=None):
        """
        :param name: str or float
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. 351.40691 61.646657)
        :param size: int, optional
        The side length in pixel  of TESScut image
        :param sector: int, optional
        The sector for which data should be returned. If None, returns the earliest observed sector
        """
        super(Source, self).__init__()
        self.name = name
        self.size = size
        self.sector = 0
        catalogdata = Catalogs.query_object(self.name, radius=(self.size + 2) * 21 * 0.707 / 3600,
                                            catalog="Gaia", version=2)
        print(f'Found {len(catalogdata)} Gaia DR2 objects.')
        ra = catalogdata[0]['ra']
        dec = catalogdata[0]['dec']
        coord = SkyCoord(ra, dec, unit="deg")
        sector_table = Tesscut.get_sectors(coord)
        print(sector_table)
        hdulist = Tesscut.get_cutouts(coord, self.size)
        self.catalogdata = catalogdata
        self.sector_table = sector_table
        self.hdulist = hdulist
        self.select_sector(sector=sector_table['sector'][0])

    def select_sector(self, sector=1):
        if self.sector == sector:
            print(f'Already in sector {sector}.')
        elif sector not in self.sector_table['sector']:
            print(f'Sector {sector} does not cover this region. Please refer to sector table.')
        self.sector = sector
        hdu = self.hdulist[list(self.sector_table['sector']).index(sector)]
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

        gaia_targets = self.catalogdata[
            'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec']
        x = np.zeros(len(gaia_targets))
        y = np.zeros(len(gaia_targets))
        tess_mag = np.zeros(len(gaia_targets))
        for i, designation in enumerate(gaia_targets['designation']):
            pixel = self.wcs.all_world2pix(
                np.array([gaia_targets['ra'][i], gaia_targets['dec'][i]]).reshape((1, 2)), 0)
            x[i] = pixel[0][0]
            y[i] = pixel[0][1]
            dif = gaia_targets['phot_bp_mean_mag'][i] - gaia_targets['phot_rp_mean_mag'][i]
            tess_mag[i] = gaia_targets['phot_g_mean_mag'][
                              i] - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
            if np.isnan(tess_mag[i]):
                tess_mag[i] = gaia_targets['phot_g_mean_mag'][i] - 0.430
        tess_flux = 10 ** (- tess_mag / 2.5)
        t = Table()
        t[f'tess_mag'] = tess_mag
        t[f'tess_flux'] = tess_flux
        t[f'tess_flux_ratio'] = tess_flux / np.max(tess_flux)
        t[f'sector_{self.sector}_x'] = x
        t[f'sector_{self.sector}_y'] = y
        t['variability'] = np.zeros(len(gaia_targets), dtype=int)
        gaia_targets = hstack([gaia_targets, t])
        gaia_targets.sort('tess_mag')

        gaia_table = Table(names=gaia_targets.colnames,
                           dtype=('str', 'float64', 'float64', 'float64', 'float64', 'float64',
                                  'float64', 'float64', 'float64', 'float64', 'float64', 'float64'))
        x_tess = gaia_targets[f'sector_{self.sector}_x']
        y_tess = gaia_targets[f'sector_{self.sector}_y']
        for i in range(len(gaia_targets)):
            if -2 < x_tess[i] < self.size + 1 and -2 < y_tess[i] < self.size + 1:
                gaia_table.add_row(gaia_targets[i])
        self.gaia = gaia_table


if __name__ == '__main__':
    source = Source('NGC_7654', size=90)

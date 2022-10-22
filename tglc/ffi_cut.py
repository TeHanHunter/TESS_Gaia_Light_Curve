import warnings
import os
import sys
import pickle
import numpy as np
import astropy.units as u

from os.path import exists
from astroquery.gaia import Gaia
from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, Column
from astropy.wcs import WCS
from tglc.ffi import tic_advanced_search_position_rows, convert_gaia_id

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"


class Source_cut(object):
    def __init__(self, name, size=50, sector=None, cadence=None):
        """
        Source_cut object that includes all data from TESS and Gaia DR3
        :param name: str, required
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. '351.40691 61.646657')
        :param size: int, optional
        The side length in pixel  of TESScut image
        :param cadence: list, required
        list of cadences of TESS FFI
        """
        super(Source_cut, self).__init__()
        if cadence is None:
            cadence = []
        if size < 25:
            warnings.warn('FFI cut size too small, try at least 25*25')
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
        self.mask = []
        target = Catalogs.query_object(self.name, radius=21 * 0.707 / 3600, catalog="Gaia", version=2)
        if len(target) == 0:
            target = Catalogs.query_object(self.name, radius=5 * 21 * 0.707 / 3600, catalog="Gaia", version=2)
        ra = target[0]['ra']
        dec = target[0]['dec']
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity((self.size + 6) * 21 * 0.707 / 3600, u.deg)
        print(f'Target Gaia: {target[0]["designation"]}')
        catalogdata = Gaia.cone_search_async(coord, radius,
                                             columns=['DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                                                      'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec']).get_results()
        print(f'Found {len(catalogdata)} Gaia DR3 objects.')
        catalogdata_tic = tic_advanced_search_position_rows(ra=ra, dec=dec, radius=(self.size + 2) * 21 * 0.707 / 3600)
        print(f'Found {len(catalogdata_tic)} TIC objects.')
        self.tic = convert_gaia_id(catalogdata_tic)
        sector_table = Tesscut.get_sectors(coordinates=coord)
        if len(sector_table) == 0:
            warnings.warn('TESS has not observed this position yet :(')
        print(sector_table)
        if sector is None:
            hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size)
        elif sector is True:
            hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size, sector=sector_table['sector'][0])
            sector = sector_table['sector'][0]
        else:
            hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size, sector=sector)
        self.catalogdata = catalogdata
        self.sector_table = sector_table
        self.camera = int(sector_table[0]['camera'])
        self.ccd = int(sector_table[0]['ccd'])
        self.hdulist = hdulist
        sector_list = []
        for i in range(len(hdulist)):
            sector_list.append(hdulist[i][0].header['SECTOR'])
        self.sector_list = sector_list
        if sector is None:
            self.select_sector(sector=sector_table['sector'][0])
        else:
            self.select_sector(sector=sector)

    def select_sector(self, sector=1):
        """
        select sector to use if target is in multi-sectors
        :param sector: int, required
        TESS sector number
        """
        if self.sector == sector:
            print(f'Already in sector {sector}.')
            return
        elif sector not in self.sector_table['sector']:
            print(f'Sector {sector} does not cover this region. Please refer to sector table.')
            return

        index = self.sector_list.index(sector)
        self.sector = sector
        hdu = self.hdulist[index]
        self.camera = int(hdu[0].header['CAMERA'])
        self.ccd = int(hdu[0].header['CCD'])
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
        median_time = np.median(data_time)
        interval = (median_time - 388.5) / 365.25

        mask = np.ones(np.shape(data_flux[0]))
        bad_pixels = np.zeros(np.shape(data_flux[0]))
        med_flux = np.median(data_flux, axis=0)
        bad_pixels[med_flux > 0.8 * np.nanmax(med_flux)] = 1
        bad_pixels[med_flux < 0.2 * np.nanmedian(med_flux)] = 1
        bad_pixels[np.isnan(med_flux)] = 1
        mask = np.ma.masked_array(mask, mask=bad_pixels)
        self.mask = mask

        gaia_targets = self.catalogdata[
            'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec']
        gaia_targets['phot_bp_mean_mag'].fill_value = np.nan
        gaia_targets['phot_rp_mean_mag'].fill_value = np.nan
        gaia_targets['pmra'].fill_value = np.nan
        gaia_targets['pmdec'].fill_value = np.nan
        gaia_targets = gaia_targets.filled()
        num_gaia = len(gaia_targets)
        # tic_id = np.zeros(num_gaia)
        x_gaia = np.zeros(num_gaia)
        y_gaia = np.zeros(num_gaia)
        tess_mag = np.zeros(num_gaia)
        in_frame = [True] * num_gaia
        for i, designation in enumerate(gaia_targets['designation']):
            ra = gaia_targets['ra'][i]
            dec = gaia_targets['dec'][i]
            if not np.isnan(gaia_targets['pmra'][i]):
                ra += gaia_targets['pmra'][i] * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
            if not np.isnan(gaia_targets['pmdec'][i]):
                dec += gaia_targets['pmdec'][i] * interval / 1000 / 3600
            pixel = self.wcs.all_world2pix(np.array([ra, dec]).reshape((1, 2)), 0)
            x_gaia[i] = pixel[0][0]
            y_gaia[i] = pixel[0][1]
            if np.isnan(gaia_targets['phot_g_mean_mag'][i]):
                in_frame[i] = False
            elif gaia_targets['phot_g_mean_mag'][i] >= 25:
                in_frame[i] = False
            elif -4 < x_gaia[i] < self.size + 3 and -4 < y_gaia[i] < self.size + 3:
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


class Source_cut_pseudo(object):
    def __init__(self, name, size=50, sector=0, cadence=None):
        """
        Source_cut object that includes all data from TESS and Gaia DR3
        :param name: str, required
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. '351.40691 61.646657')
        :param size: int, optional
        The side length in pixel  of TESScut image
        :param cadence: list, required
        list of cadences of TESS FFI
        """
        super(Source_cut_pseudo, self).__init__()
        if cadence is None:
            cadence = []
        self.name = name
        self.size = size
        self.sector = sector
        self.camera = 0
        self.ccd = 0
        self.wcs = []
        self.time = np.arange(10)
        self.flux = 20 * np.ones((100, 50, 50)) + np.random.random(size=(100, 50, 50))
        star_flux = np.random.random(100) * 1000 + 200
        star_x = np.random.random(100) * 50 - 0.5
        star_y = np.random.random(100) * 50 - 0.5
        star_x_round = np.round(star_x)
        star_y_round = np.round(star_y)
        for j in range(100):
            for i in range(100):
                self.flux[j, int(star_y_round[i]), int(star_x_round[i])] += star_flux[i]
                try:
                    self.flux[j, int(star_y_round[i]), int(star_x_round[i]) + 1] += star_flux[i]
                except:
                    continue
        self.flux_err = []
        self.gaia = []
        self.cadence = cadence
        self.quality = []
        self.mask = np.ones(np.shape(self.flux[0]))

        # t_tic = Table()
        # t_tic[f'tic'] = tic_id[in_frame]
        t = Table()
        t[f'tess_mag'] = - star_flux
        t[f'tess_flux'] = star_flux
        t[f'tess_flux_ratio'] = star_flux / np.max(star_flux)
        t[f'sector_{self.sector}_x'] = star_x
        t[f'sector_{self.sector}_y'] = star_y
        gaia_targets = t  # TODO: sorting not sorting all columns
        gaia_targets.sort('tess_mag')
        self.gaia = gaia_targets


def ffi_cut(target='', local_directory='', size=90, sector=None):
    """
    Function to generate Source_cut objects
    :param target: string, required
    target name
    :param local_directory: string, required
    output directory
    :param size: int, required
    FFI cut side length
    :param sector: int, required
    TESS sector number
    :return: tglc.ffi_cut.Source_cut
    """
    if sector is None:
        source_name = f'source_{target}'
    elif sector is True:
        source_name = f'source_{target}_earliest_sector'
    else:
        source_name = f'source_{target}_sector_{sector}'
    source_exists = exists(f'{local_directory}source/{source_name}.pkl')
    if source_exists and os.path.getsize(f'{local_directory}source/{source_name}.pkl') > 0:
        with open(f'{local_directory}source/{source_name}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        print(source.sector_table)
        print('Loaded ffi_cut from directory. ')
    else:
        with open(f'{local_directory}source/{source_name}.pkl', 'wb') as output:
            source = Source_cut(target, size=size, sector=sector)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
    return source

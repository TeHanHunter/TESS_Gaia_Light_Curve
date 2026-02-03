import warnings
import os
import sys
import pickle
import numpy as np
import astropy.units as u
import requests
import time
import contextlib
import threading

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


@contextlib.contextmanager
def _dot_wait(message, interval=0.6, done_format=None):
    stop_event = threading.Event()
    start = time.time()

    def _run():
        dots = 0
        while not stop_event.is_set():
            dots = (dots % 3) + 1
            print(f"\r{message} {'.' * dots}", end="", flush=True)
            time.sleep(interval)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        if done_format is None:
            done_format = "{message} ... done in {elapsed:.1f}s."
        done_line = done_format.format(message=message, elapsed=time.time() - start)
        print(f"\r{done_line}   ")


class Source_cut(object):
    def __init__(self, name, size=50, sector=None, cadence=None, limit_mag=None, transient=None, ffi='TICA',
                 mast_timeout=3600):
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
        self.transient = transient
        self.ffi = ffi
        Tesscut._service_api_connection.TIMEOUT = mast_timeout
        print(f'MAST Tesscut timeout set to {mast_timeout}s.')

        def _parse_tic_id(t):
            if not isinstance(t, str):
                return None
            s = t.strip()
            if s.upper().startswith('TIC'):
                parts = s.split()
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
                s = s[3:].strip()
            return int(s) if s.isdigit() else None

        def _is_tic_id(t):
            if not isinstance(t, str):
                return False
            s = t.strip()
            return s.upper().startswith('TIC') or s.isdigit()

        target = None
        is_tic = _is_tic_id(self.name) and _parse_tic_id(self.name) is not None
        try:
            target = Catalogs.query_object(self.name, radius=21 * 0.707 / 3600, catalog="Gaia", version=2)
        except requests.exceptions.RequestException as e:
            warnings.warn(f'MAST name lookup failed for "{self.name}": {e}')

        if target is None or len(target) == 0:
            try:
                target = Catalogs.query_object(self.name, radius=5 * 21 * 0.707 / 3600, catalog="Gaia", version=2)
            except requests.exceptions.RequestException as e:
                warnings.warn(f'MAST name lookup failed for "{self.name}": {e}')

        if target is None or len(target) == 0:
            if is_tic:
                raise RuntimeError(
                    f'MAST name lookup failed for TIC target "{self.name}". Please retry when MAST is available.'
                )
            raise RuntimeError(
                f'Unable to resolve target "{self.name}". MAST name lookup appears unavailable.'
            )

        ra = target[0]['ra']
        dec = target[0]['dec']
        target_designation = target[0].get('designation', None)
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity((self.size + 6) * 21 * 0.707 / 3600, u.deg)
        if target_designation:
            print(f'Target Gaia: {target_designation}')
        with _dot_wait('Querying Gaia DR3 cone search'):
            catalogdata = Gaia.cone_search_async(
                coord,
                radius=radius,
                columns=['DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                         'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec']
            ).get_results()
        print(f'Found {len(catalogdata)} Gaia DR3 objects.')
        with _dot_wait('Querying TIC around target'):
            catalogdata_tic = tic_advanced_search_position_rows(
                ra=ra,
                dec=dec,
                radius=(self.size + 2) * 21 * 0.707 / 3600,
                limit_mag=limit_mag
            )
        print(f'Found {len(catalogdata_tic)} TIC objects.')
        with _dot_wait('Crossmatching TIC -> Gaia DR3 (this may take a while)'):
            self.tic = convert_gaia_id(catalogdata_tic)
        sector_table = Tesscut.get_sectors(coordinates=coord)
        if len(sector_table) == 0:
            warnings.warn('TESS has not observed this position yet :(')
        wait_note = 'later sectors with 200s cadence can take ~20 minutes'
        if sector is None:
            wait_message = f'Requesting Tesscut cutouts (all sectors). Waiting on MAST response ({wait_note})'
            with _dot_wait(wait_message):
                hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size, product=ffi)
        elif sector == 'first':
            wait_message = f'Requesting Tesscut cutouts (first sector). Waiting on MAST response ({wait_note})'
            with _dot_wait(wait_message):
                hdulist = Tesscut.get_cutouts(
                    coordinates=coord, size=self.size, product=ffi, sector=sector_table['sector'][0]
                )
            sector = sector_table['sector'][0]
        elif sector == 'last':
            wait_message = f'Requesting Tesscut cutouts (last sector). Waiting on MAST response ({wait_note})'
            with _dot_wait(wait_message):
                hdulist = Tesscut.get_cutouts(
                    coordinates=coord, size=self.size, product=ffi, sector=sector_table['sector'][-1]
                )
            sector = sector_table['sector'][-1]
        else:
            wait_message = f'Requesting Tesscut cutouts (sector {sector}). Waiting on MAST response ({wait_note})'
            with _dot_wait(wait_message):
                hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size, product=ffi, sector=sector)
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
        if self.ffi == 'SPOC':
            data_flux = hdu[1].data['FLUX']
            data_flux_err = hdu[1].data['FLUX_ERR']
        elif self.ffi == 'TICA':
            data_flux = hdu[1].data['FLUX'] / (200 * 0.8 * 0.99)
            data_flux_err = hdu[1].data['FLUX_ERR'] / (200 * 0.8 * 0.99)
        else:
            raise Exception(f'FFI type {self.ffi} not supported')
        data_quality = hdu[1].data['QUALITY']
        # data_time = data_time[np.where(data_quality == 0)]
        # data_flux = data_flux[np.where(data_quality == 0), :, :][0]
        # data_flux_err = data_flux_err[np.where(data_quality == 0), :, :][0]
        self.wcs = wcs
        self.time = data_time
        self.flux = data_flux
        self.flux_err = data_flux_err
        self.quality = data_quality
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
            'DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec']

        # inject transients
        if self.transient is not None:
            gaia_targets.add_row([self.transient[0], 20, 20, 20, self.transient[1], self.transient[2], 0, 0])

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
        for i, designation in enumerate(gaia_targets['DESIGNATION']):
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
        gaia_targets = hstack([gaia_targets[in_frame], t])
        if self.transient is not None:
            gaia_targets['tess_flux'][np.where(gaia_targets['DESIGNATION'] == self.transient[0])[0][0]] = 0
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


def ffi_cut(target='', local_directory='', size=90, sector=None, limit_mag=None, transient=None, ffi='TICA',
            mast_timeout=3600):
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
        source_name = f'source_{ffi}_{target}'
    elif sector == 'first':
        source_name = f'source_{ffi}_{target}_earliest_sector'
    elif sector == 'last':
        source_name = f'source_{ffi}_{target}_last_sector'
    else:
        source_name = f'source_{ffi}_{target}_sector_{sector}'
    source_exists = exists(f'{local_directory}source/{source_name}.pkl')
    if source_exists and os.path.getsize(f'{local_directory}source/{source_name}.pkl') > 0:
        with open(f'{local_directory}source/{source_name}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        print(source.sector_table)
        print('Loaded ffi_cut from directory. ')
    else:
        with open(f'{local_directory}source/{source_name}.pkl', 'wb') as output:
            source = Source_cut(target, size=size, sector=sector, limit_mag=limit_mag, transient=transient, ffi=ffi,
                                mast_timeout=mast_timeout)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
    return source

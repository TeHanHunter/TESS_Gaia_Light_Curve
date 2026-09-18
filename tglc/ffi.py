import json
import os
import pickle
import sys
import warnings
import astropy.units as u
import numpy as np
import importlib_resources
import requests
import time

from glob import glob
from os.path import exists
from urllib.parse import quote as urlencode
from astropy.io import fits
from astropy.table import Table, hstack, vstack, unique, Column, MaskedColumn
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus
from scipy import ndimage
from tqdm import tqdm, trange
from tglc.astrometry import (
    MODEL_CATALOG_HALO, SOURCE_SCHEMA_VERSION, in_model_catalog,
    persistent_bad_pixels, propagate_catalog_positions,
)

Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # TODO: dr3 MJD = 2457388.5, TBJD = 388.5


# The next three functions are adopted from astroquery MAST API https://mast.stsci.edu/api/v0/pyex.html#incPy
def mast_query(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

    # Base API url
    request_url = 'https://mast.stsci.edu/api/v0/invoke'
    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))
    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}
    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    # Perform the HTTP request
    resp = requests.post(request_url, data="request=" + req_string, headers=headers)
    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')
    return head, content


def mast_json2table(json_obj):
    data_table = Table()
    for col, atype in [(x['name'], x['type']) for x in json_obj['fields']]:
        if atype == "string":
            atype = "str"
        if atype == "boolean":
            atype = "bool"
        data_table[col] = np.array([x.get(col, None) for x in json_obj['data']], dtype=atype)
    return data_table


def tic_advanced_search_position_rows(ra=1., dec=1., radius=0.5, limit_mag=16):
    request = {"service": "Mast.Catalogs.Filtered.Tic.Position.Rows",
               "format": "json",
               "params": {
                   "columns": 'ID, GAIA',
                   "filters": [
                       {"paramName": "Tmag",
                        "values": [{"min": -10., "max": (limit_mag + 0.5)}]}],
                   "ra": ra,
                   "dec": dec,
                   "radius": radius
               }}

    headers, out_string = mast_query(request)
    out_data = json.loads(out_string)
    return mast_json2table(out_data)


def convert_gaia_id(catalogdata_tic, gaia_tap_server="https://gea.esac.esa.int/tap-server/tap"):
    """Crossmatch TIC DR2 identifiers, retaining unknown DR3 identities as masked.

    An unavailable or ambiguous bridge is not evidence that a DR2 identifier
    identifies the same DR3 source. Keep every valid input TIC/DR2 pair so the
    caller can distinguish an unknown match from a missing TIC record.
    """
    query = """
            SELECT dr2_source_id, dr3_source_id
            FROM gaiadr3.dr2_neighbourhood
            WHERE dr2_source_id IN {gaia_ids}
            """

    def _run_query(gaia_tuple):
        try:
            return Gaia.launch_job_async(query.format(gaia_ids=gaia_tuple)).get_results()
        except Exception as exc:
            warnings.warn(
                f'Primary Gaia TAP crossmatch failed ({exc}). Retrying via mirror {gaia_tap_server}.'
            )
            return TapPlus(url=gaia_tap_server).launch_job_async(
                query.format(gaia_ids=gaia_tuple)
            ).get_results()

    pairs = []
    for tic_id, gaia_id in zip(catalogdata_tic['ID'], catalogdata_tic['GAIA']):
        if np.ma.is_masked(gaia_id) or np.ma.is_masked(tic_id):
            continue
        gaia_id = str(gaia_id).strip()
        if gaia_id.isdigit() and int(gaia_id) > 0:
            pairs.append((int(tic_id), int(gaia_id)))
    results = Table()
    results['dr2_source_id'] = np.array([pair[1] for pair in pairs], dtype=np.int64)
    results['dr3_source_id'] = MaskedColumn(np.zeros(len(pairs), dtype=np.int64), mask=True)
    results['TIC'] = np.array([pair[0] for pair in pairs], dtype=np.int64)
    gaia_ids = sorted({pair[1] for pair in pairs})
    matches = {}
    for start in range(0, len(gaia_ids), 10000):
        # A one-element Python tuple has a trailing comma, which is not valid
        # ADQL. These integers have already been validated above.
        gaia_tuple = '(' + ','.join(map(str, gaia_ids[start:start + 10000])) + ')'
        try:
            bridge = _run_query(gaia_tuple)
        except Exception as exc:
            warnings.warn(f'Gaia DR2->DR3 crossmatch failed ({exc}); DR3 identities remain unknown.')
            continue
        for row in bridge:
            if not np.ma.is_masked(row['dr3_source_id']):
                matches.setdefault(int(row['dr2_source_id']), set()).add(int(row['dr3_source_id']))
    for index, (_, dr2_id) in enumerate(pairs):
        candidates = matches.get(dr2_id, set())
        if len(candidates) == 1:
            results['dr3_source_id'][index] = next(iter(candidates))
    return results


# from Tim
def background_mask(im=None):
    imfilt = im * 1.
    for i in range(im.shape[1]):
        imfilt[:, i] = ndimage.percentile_filter(im[:, i], 50, size=51)

    ok = im < imfilt
    # Don't use saturated pixels!
    satfactor = 0.4
    ok *= im < satfactor * np.amax(im)
    running_factor = 1
    cal_factor = np.zeros(im.shape[1])
    cal_factor[0] = 1

    di = 1
    i = 0
    while i < im.shape[1] - 1 and i + di < im.shape[1]:
        _ok = ok[:, i] * ok[:, i + di]
        coef = np.median(im[:, i + di][_ok] / im[:, i][_ok])
        if 0.5 < coef < 2:
            running_factor *= coef
            cal_factor[i + di] = running_factor
            i += di
            di = 1  # Reset the stepsize to one.
        else:
            # Label the column as bad, then skip it.
            cal_factor[i + di] = 0
            di += 1

    # cal_factor[im > 0.4 * np.amax(im)] = 0
    return cal_factor


class Source(object):
    def __init__(self, x=0, y=0, flux=None, time=None, wcs=None, quality=None, mask=None, exposure=1800, sector=0,
                 size=150,
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

        self.size = size
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.cadence = cadence
        self.quality = quality
        self.exposure = exposure
        self.wcs = wcs
        self.transient = None
        self.ffi = 'SPOC'
        self.source_schema_version = SOURCE_SCHEMA_VERSION
        co1 = (size - 1) / 4
        co2 = 3 * (size - 1) / 4
        catalog_1 = self.search_gaia(x, y, co1, co1)
        catalog_2 = self.search_gaia(x, y, co1, co2)
        catalog_3 = self.search_gaia(x, y, co2, co1)
        catalog_4 = self.search_gaia(x, y, co2, co2)
        catalogdata = vstack([catalog_1, catalog_2, catalog_3, catalog_4], join_type='exact')
        catalogdata = unique(catalogdata, keys='DESIGNATION')
        coord = wcs.pixel_to_world([x + (size - 1) / 2 + 44], [y + (size - 1) / 2])[0].to_string()
        ra = float(coord.split()[0])
        dec = float(coord.split()[1])
        catalogdata_tic = tic_advanced_search_position_rows(ra=ra, dec=dec, radius=(self.size + 2) * 21 * 0.707 / 3600)
        # print(f'no_of_stars={len(catalogdata_tic)}, camera={camera}, ccd={ccd}: ra={ra}, dec={dec}, radius={(self.size + 2) * 21 * 0.707 / 3600}')
        self.tic = convert_gaia_id(catalogdata_tic)
        self.flux = flux[:, y:y + size, x:x + size]
        self.mask = mask[y:y + size, x:x + size]
        self.time = np.array(time)
        catalogdata = propagate_catalog_positions(catalogdata, self.time)
        for column in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
            catalogdata[column] = np.ma.asarray(catalogdata[column], dtype=float).filled(np.nan)

        num_gaia = len(catalogdata)
        tic_id = np.zeros(num_gaia)
        x_gaia = np.zeros(num_gaia)
        y_gaia = np.zeros(num_gaia)
        tess_mag = np.zeros(num_gaia)
        in_frame = [True] * num_gaia
        for i, designation in enumerate(catalogdata['DESIGNATION']):
            pixel = self.wcs.all_world2pix(
                np.array([catalogdata['ra_epoch'][i], catalogdata['dec_epoch'][i]]).reshape((1, 2)), 0, quiet=True)
            x_gaia[i] = pixel[0][0] - x - 44
            y_gaia[i] = pixel[0][1] - y
            try:
                tic_id[i] = catalogdata_tic['ID'][np.where(catalogdata_tic['GAIA'] == designation.split()[2])[0][0]]
            except:
                tic_id[i] = np.nan
            if np.isnan(catalogdata['phot_g_mean_mag'][i]):
                in_frame[i] = False
            elif catalogdata['phot_g_mean_mag'][i] >= 25:
                in_frame[i] = False
            elif in_model_catalog(x_gaia[i], y_gaia[i], self.flux.shape[1:]):
                dif = catalogdata['phot_bp_mean_mag'][i] - catalogdata['phot_rp_mean_mag'][i]
                tess_mag[i] = catalogdata['phot_g_mean_mag'][
                                  i] - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
                if np.isnan(tess_mag[i]):
                    tess_mag[i] = catalogdata['phot_g_mean_mag'][i] - 0.430
                if np.isnan(tess_mag[i]):
                    in_frame[i] = False
            else:
                in_frame[i] = False

        if not np.any(in_frame):
            raise ValueError('No usable Gaia DR3 sources overlap this image and its PSF halo')
        tess_flux = 10 ** (- tess_mag / 2.5)
        t = Table()
        t[f'tess_mag'] = tess_mag[in_frame]
        t[f'tess_flux'] = tess_flux[in_frame]
        t[f'tess_flux_ratio'] = tess_flux[in_frame] / np.nanmax(tess_flux[in_frame])
        t[f'sector_{self.sector}_x'] = x_gaia[in_frame]
        t[f'sector_{self.sector}_y'] = y_gaia[in_frame]
        catalogdata = hstack([catalogdata[in_frame], t])
        catalogdata.sort('tess_mag')
        self.gaia = catalogdata

    def search_gaia(self, x, y, co1, co2):
        coord = self.wcs.pixel_to_world([x + co1 + 44], [y + co2])[0].to_string()
        radius = u.Quantity((self.size / 2 + 2 * MODEL_CATALOG_HALO) * 21 / np.sqrt(2) / 3600, u.deg)
        attempt = 0
        while attempt < 5:
            try:
                catalogdata = Gaia.cone_search_async(coord, radius=radius,
                                             columns=['DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                                                      'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec', 'ref_epoch']).get_results()
                return catalogdata
            except:
                attempt += 1
                time.sleep(10)
                print(f'Trying Gaia search again. Coord = {coord}, radius = {radius}')

def ffi(ccd=1, camera=1, sector=1, size=150, local_directory='', producing_mask=False):
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
    :param local_directory: string, required
    path to the FFI folder
    :return:
    """
    # input_files = glob(f'/pdo/spoc-data/sector-{sector:03d}/ffi*/**/*{camera}-{ccd}-????-?_ffic.fits*')
    input_files = glob(f'{local_directory}ffi/*{camera}-{ccd}-????-?_ffic.fits')
    print('camera: ' + str(camera) + '  ccd: ' + str(ccd) + '  num of files: ' + str(len(input_files)))
    time = []
    quality = []
    cadence = []
    flux = np.empty((len(input_files), 2048, 2048), dtype=np.float32)
    for i, file in enumerate(tqdm(input_files)):
        try:
            with fits.open(file, mode='denywrite', memmap=False) as hdul:
                quality.append(hdul[1].header['DQUALITY'])
                cadence.append(hdul[0].header['FFIINDEX'])
                flux[i] = hdul[1].data[0:2048, 44:2092]
                time.append((hdul[1].header['TSTOP'] + hdul[1].header['TSTART']) / 2)

        except:
            print(f'Corrupted file {file}, download again ...')
            response = requests.get(
                f'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/{os.path.basename(file)}')
            open(file, 'wb').write(response.content)
            with fits.open(file, mode='denywrite', memmap=False) as hdul:
                quality.append(hdul[1].header['DQUALITY'])
                cadence.append(hdul[0].header['FFIINDEX'])
                flux[i] = hdul[1].data[0:2048, 44:2092]
                time.append((hdul[1].header['TSTOP'] + hdul[1].header['TSTART']) / 2)
    time_order = np.argsort(np.array(time))
    input_files = [input_files[index] for index in time_order]
    time = np.array(time)[time_order]
    flux = flux[time_order, :, :]
    quality = np.array(quality)[time_order]
    cadence = np.array(cadence)[time_order]
    # mask = np.array([True] * 2048 ** 2).reshape(2048, 2048)
    # for i in range(len(time)):
    #     mask[np.where(flux[i] > np.percentile(flux[i], 99.95))] = False
    #     mask[np.where(flux[i] < np.median(flux[i]) / 2)] = False
    if np.min(np.diff(cadence)) != 1:
        np.save(f'{local_directory}/Wrong_Cadence_sector{sector:04d}_cam{camera}_ccd{ccd}.npy', np.min(np.diff(cadence)))
    if producing_mask:
        median_flux = np.median(flux, axis=0)
        mask = background_mask(im=median_flux)
        mask /= ndimage.median_filter(mask, size=51)
        np.save(f'{local_directory}mask/mask_sector{sector:04d}_cam{camera}_ccd{ccd}.npy', mask)
        return
    # load mask
    mask = importlib_resources.files(__package__).joinpath("background_mask/median_mask.fits").open("rb")
    mask = fits.open(mask)[0].data[(camera - 1) * 4 + (ccd - 1), :]
    mask = np.repeat(mask.reshape(1, 2048), repeats=2048, axis=0)
    bad_pixels = persistent_bad_pixels(flux)
    mask = np.ma.masked_array(mask, mask=bad_pixels | ~np.isfinite(mask))
    mask = np.ma.masked_equal(mask, 0)

    for i in range(10):
        hdul = fits.open(input_files[np.where(np.array(quality) == 0)[0][i]])
        wcs = WCS(hdul[1].header)
        if wcs.axis_type_names == ['RA', 'DEC']:
            break

    exposure = int((hdul[0].header['TSTOP'] - hdul[0].header['TSTART']) * 86400)

    # 95*95 cuts with 2 pixel redundant, (22*22 cuts)
    # try 77*77 with 4 redundant, (28*28 cuts)
    os.makedirs(f'{local_directory}source/{camera}-{ccd}/', exist_ok=True)
    for i in trange(14):  # 22
        for j in range(14):  # 22
            source_path = f'{local_directory}source/{camera}-{ccd}/source_{i:02d}_{j:02d}.pkl'
            source_exists = exists(source_path)
            source_config = {
                'source_schema': SOURCE_SCHEMA_VERSION, 'ffi': 'SPOC',
                'sector': sector, 'camera': camera, 'ccd': ccd, 'size': size,
                'x': i * (size - 4), 'y': j * (size - 4),
                'cadence': cadence.tolist(), 'time': time.tolist(),
            }
            if source_exists and os.path.getsize(source_path) > 0:
                try:
                    with open(source_path, 'rb') as cached:
                        source = pickle.load(cached)
                    if getattr(source, '_tglc_cache_config', None) == source_config:
                        continue
                except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
                    pass
                warnings.warn(f'Rebuilding incompatible source cache {source_path}')
            source = Source(x=i * (size - 4), y=j * (size - 4), flux=flux, mask=mask, sector=sector,
                            time=time, size=size, quality=quality, wcs=wcs, camera=camera, ccd=ccd,
                            exposure=exposure, cadence=cadence)
            source.cadence_origin = 'FFIINDEX'
            source._tglc_cache_config = source_config
            temporary = f'{source_path}.{os.getpid()}.tmp'
            try:
                with open(temporary, 'wb') as output:
                    pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, source_path)
            finally:
                if exists(temporary):
                    os.remove(temporary)

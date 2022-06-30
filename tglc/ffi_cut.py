import warnings
from astroquery.mast import Tesscut
from os.path import exists
from tglc.ffi import *
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

class Source_cut(object):
    def __init__(self, name, size=15, sector=None, cadence=None):
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
        if cadence is None:
            cadence = []
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
        ra = target[0]['ra']
        dec = target[0]['dec']
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity((self.size + 6) * 21 * 0.707 / 3600, u.deg)
        print(f'Target Gaia: {target[0]["designation"]}')
        catalogdata = Gaia.cone_search_async(coord, radius,
                                             columns=['DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                                                      'phot_rp_mean_mag', 'ra', 'dec']).get_results()
        print(f'Found {len(catalogdata)} Gaia DR2 objects.')
        catalogdata_tic = tic_advanced_search_position_rows(ra=ra, dec=dec, radius=(self.size + 2) * 21 * 0.707 / 3600)
        print(f'Found {len(catalogdata_tic)} TIC objects.')
        self.tic = catalogdata_tic['ID', 'GAIA']
        sector_table = Tesscut.get_sectors(coordinates=coord)
        print(sector_table)
        if sector is None:
            hdulist = Tesscut.get_cutouts(coordinates=coord, size=self.size)
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

        mask = np.ones(np.shape(data_flux))
        for i in trange(len(data_time)):
            mask[i][data_flux[i] > 0.8 * np.nanmax(data_flux[i])] = 0
            mask[i][data_flux[i] < 0.2 * np.nanmedian(data_flux[i])] = 0
            mask[i][np.isnan(mask[i])] = 0
        self.mask = mask

        gaia_targets = self.catalogdata[
            'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec']
        num_gaia = len(gaia_targets)
        # tic_id = np.zeros(num_gaia)
        x_gaia = np.zeros(num_gaia)
        y_gaia = np.zeros(num_gaia)
        tess_mag = np.zeros(num_gaia)
        in_frame = [True] * num_gaia
        # TODO: multiprocess below
        for i, designation in enumerate(gaia_targets['designation']):
            pixel = self.wcs.all_world2pix(
                np.array([gaia_targets['ra'][i], gaia_targets['dec'][i]]).reshape((1, 2)), 0)
            x_gaia[i] = pixel[0][0]
            y_gaia[i] = pixel[0][1]
            # try:
            #     tic_id[i] = self.tic['ID'][np.where(self.tic['GAIA'] == designation.split()[2])[0][0]]
            # except:
            #     tic_id[i] = np.nan
            if np.isnan(gaia_targets['phot_g_mean_mag'][i]):
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
        # t_tic = Table()
        # t_tic[f'tic'] = tic_id[in_frame]
        t = Table()
        t[f'tess_mag'] = tess_mag[in_frame]
        t[f'tess_flux'] = tess_flux[in_frame]
        t[f'tess_flux_ratio'] = tess_flux[in_frame] / np.max(tess_flux[in_frame])
        t[f'sector_{self.sector}_x'] = x_gaia[in_frame]
        t[f'sector_{self.sector}_y'] = y_gaia[in_frame]
        gaia_targets = hstack([gaia_targets[in_frame], t])  # TODO: sorting not sorting all columns
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
    source_exists = exists(f'{local_directory}source/source_{target}.pkl')
    if source_exists and os.path.getsize(f'{local_directory}source/source_{target}.pkl') > 0:
        with open(f'{local_directory}source/source_{target}.pkl', 'rb') as input_:
            source = pickle.load(input_)
        print(source.sector_table)
        print('Loaded ffi_cut from directory. ')
    else:
        with open(f'{local_directory}source/source_{target}.pkl', 'wb') as output:
            source = Source_cut(target, size=size, sector=sector)
            pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
    return source


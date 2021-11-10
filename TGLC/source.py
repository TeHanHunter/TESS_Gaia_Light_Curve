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
    """
    Get FFI cut using TESScut and Gaia DR2 catalog

    Parameters
    ----------
    name : str or float
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. 351.40691 61.646657)
    size : int, optional
        The side length in pixel  of TESScut image
    sector : int, optional
        The sector for which data should be returned. If None, returns the first observed sector
    search_gaia : boolean, optional
        Whether to search gaia targets in the field

    Attributes
    ----------
    z : numpy.ndarray
        Parametrized x (z // size) and y (z % size)
    wcs : astropy.wcs.WCS class
        World Coordinate Systems information of the FFI
    time : numpy.ndarray (1d)
        Time of each frame
    flux : numpy.ndarray (3d)
        Fluxes of each frame, spanning time space
    flux_err : numpy.ndarray (3d)
        Flux errors of each frame, spanning time space
    gaia : astropy.table.table.Table class
        Gaia information including ra, dec, brightness, projection on TESS FFI, etc.
    """
    # variable parameters
    nstars = None
    star_index = [0]
    cguess = [0., 0., 0., 1., 0., 1., 3.]
    var_to_bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.1, 0.1), (0, 10.0), (-0.5, 0.5), (0, 10.0), (0, np.inf)]
    cut_size = 11
    gaia_cut = None
    flux_cut = None
    flux_cut_err = None
    inner_star = []
    # TODO: when close to the edge of CCD, could have nan edge.

    def __init__(self, name, size=15, sector=None, search_gaia=True, mag_threshold=15):
        super(Source, self).__init__()
        self.name = name
        self.size = size
        self.z = np.arange(self.size ** 2)
        self.mag_threshold = mag_threshold
        catalog = Catalogs.query_object(self.name, radius=self.size * 21 * 0.707 / 3600, catalog="TIC")
        ra = catalog[0]['ra']
        dec = catalog[0]['dec']
        coord = SkyCoord(ra, dec, unit="deg")
        hdulist = Tesscut.get_cutouts(coord, self.size)
        sector_table = Tesscut.get_sectors(coord)
        self.sector_table = sector_table
        if sector is None:
            self.sector = sector_table['sector'][0]
            hdu = hdulist[0]
        else:
            self.sector = sector
            hdu = hdulist[list(sector_table['sector']).index(sector)]
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
        if search_gaia:
            catalogdata = Catalogs.query_object(self.name, radius=(self.size + 2) * 21 * 0.707 / 3600, catalog="Gaia")
            # catalogdata.sort("phot_g_mean_mag")
            gaia_targets = catalogdata[
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
            t[f'Sector_{self.sector}_x'] = x
            t[f'Sector_{self.sector}_y'] = y
            t['variability'] = np.zeros(len(gaia_targets), dtype=int)
            gaia_targets = hstack([gaia_targets, t])
            gaia_targets.sort('tess_mag')

            gaia_table = Table(names=gaia_targets.colnames,
                               dtype=('str', 'float64', 'float64', 'float64', 'float64', 'float64',
                                      'float64', 'float64', 'float64', 'float64', 'float64', 'float64'))
            x_tess = gaia_targets[f'Sector_{self.sector}_x']
            y_tess = gaia_targets[f'Sector_{self.sector}_y']
            for i in range(len(gaia_targets)):
                if -2 < x_tess[i] < self.size + 1 and -2 < y_tess[i] < self.size + 1:
                    gaia_table.add_row(gaia_targets[i])
            self.gaia = gaia_table

            if np.min(self.gaia['tess_mag']) > self.mag_threshold:
                print('Magnitude threshold too high. Try a smaller magnitude value.')
                self.nstars = 1
            else:
                nstars = np.where(self.gaia['tess_mag'] < self.mag_threshold)[0][-1]
                self.nstars = nstars

            x_table = gaia_table[f'Sector_{self.sector}_x']
            y_table = gaia_table[f'Sector_{self.sector}_y']
            for i in range(self.nstars):
                if -2 < x_table[i] < self.size + 1 and -2 < y_table[i] < self.size + 1:
                    self.inner_star.append(i)
        else:
            self.gaia = None

    def star_idx(self, star_idx=None):
        """
        Choose stars of interest (primarily for PSF fitting

        Attributes/
        ----------
        nstars : int
            Number of stars of interest, cut by a magnitude threshold
        star_index : list or str
            Star indexes for PSF fitting, list of indexes, int, None, or 'all'
        mag_threshold : int or float
            Min magnitude threshold for stars to fit
        """

        if star_idx is None:
            self.star_index = np.array([], dtype=int)
        elif star_idx == 'all':
            self.star_index = np.arange(self.nstars - 1)
        elif type(star_idx) == int:
            self.star_index = np.array([star_idx])
        elif type(star_idx) == list and all(isinstance(n, (int, np.int64)) for n in star_idx):
            self.star_index = np.array(star_idx)
        elif type(star_idx) == np.ndarray and all(isinstance(n, (int, np.int64)) for n in set(star_idx)):
            self.star_index = star_idx
        else:
            raise TypeError("Star index (star_index) type should be a list or np.array of ints, int, None or 'all'. ")

    def ffi(self):
        self.z = np.arange(self.size ** 2)
        self.star_index = np.array([], dtype=int)
        self.nstars = np.where(self.gaia['tess_mag'] < self.mag_threshold)[0][-1]
        self.gaia_cut = None

    def cut(self, star_idx: int):
        """
        Below is dividing the cut into 9 regions: center, 4 corners and 4 edges. By deciding which
        region a star is at, returns a cut of its neighborhood.
        """

        self.z = np.arange(121)
        x = self.gaia[f'Sector_{self.sector}_x']
        y = self.gaia[f'Sector_{self.sector}_y']
        x_mid = int(min(self.size - 6, max(x[star_idx], 5)))
        y_mid = int(min(self.size - 6, max(y[star_idx], 5)))
        self.flux_cut = self.flux[:, y_mid - 5:y_mid + 6, x_mid - 5:x_mid + 6]
        # self.flux_cut_err = self.flux_err[:, x_mid - 5:x_mid + 6, y_mid - 5:y_mid + 6]
        in_frame = (np.abs(x_mid - x) < 7) & (np.abs(y_mid - y) < 7)
        t = self.gaia[in_frame]

        t[f'Sector_{self.sector}_x'][:] = t[f'Sector_{self.sector}_x'] - x_mid + 5
        t[f'Sector_{self.sector}_y'][:] = t[f'Sector_{self.sector}_y'] - y_mid + 5
        self.gaia_cut = t
        self.star_index = [np.where(self.gaia['designation'][star_idx] == self.gaia_cut['designation'])[0][0]]
        nstars = np.where(self.gaia['tess_mag'] < self.mag_threshold)[0][-1]
        self.nstars = nstars


if __name__ == '__main__':
    # target = Source('NGC 7654')
    source = Source('351.1378304925981 61.311247660185245', size=3, sector=24)

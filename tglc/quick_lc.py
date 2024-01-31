import os
import pickle
from glob import glob
from tqdm import trange
from wotan import flatten
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
import pkg_resources

# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np

controller = ThreadpoolController()


@controller.wrap(limits=1, user_api='blas')
def tglc_lc(target='TIC 264468702', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
            first_sector_only=False, last_sector_only=False, sector=None, prior=None, transient=None):
    '''
    Generate light curve for a single target.

    :param target: target identifier
    :type target: str, required
    :param local_directory: output directory
    :type local_directory: str, required
    :param size: size of the FFI cut, default size is 90. Recommend large number for better quality. Cannot exceed 100.
    :type size: int, optional
    '''
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    if first_sector_only:
        sector = 'first'
    elif last_sector_only:
        sector = 'last'
    print(f'Target: {target}')
    target_ = Catalogs.query_object(target, radius=21 * 0.707 / 3600, catalog="Gaia", version=2)
    if len(target_) == 0:
        target_ = Catalogs.query_object(target.name, radius=5 * 21 * 0.707 / 3600, catalog="Gaia", version=2)
    ra = target_[0]['ra']
    dec = target_[0]['dec']
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    sector_table = Tesscut.get_sectors(coordinates=coord)
    print(sector_table)
    if get_all_lc:
        name = None
    else:
        catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
        if target[0:3] == 'TIC':
            name = int(target[4:])
        elif transient is not None:
            name = transient[0]
        else:
            name = int(np.array(catalogdata['ID'])[0])
            print("Since the provided target is not TIC ID, the resulted light curve with get_all_lc=False can not be "
                  "guaranteed to be the target's light curve. Please check the TIC ID of the output file before using "
                  "the light curve or try use TIC ID as the target in the format of 'TIC 12345678'.")
    if type(sector) == int:
        print(f'Only processing Sector {sector}.')
        print('Downloading Data from MAST and Gaia ...')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=sector)
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif first_sector_only:
        print(f'Only processing the first sector the target is observed in: Sector {sector_table["sector"][0]}.')
        print('Downloading Data from MAST and Gaia ...')

        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=source.sector_table['sector'][0])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif last_sector_only:
        print(f'Only processing the last sector the target is observed in: Sector {sector_table["sector"][-1]}.')
        print('Downloading Data from MAST and Gaia ...')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        source.select_sector(sector=source.sector_table['sector'][-1])
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    elif sector == None:
        print(f'Processing all available sectors of the target.')
        print('Downloading Data from MAST and Gaia ...')
        for j in range(len(sector_table)):
            print(f'################################################')
            print(f'Downloading Sector {sector_table["sector"][j]}.')
            source = ffi_cut(target=target, size=size, local_directory=local_directory,
                             sector=sector_table['sector'][j],
                             limit_mag=limit_mag, transient=transient)
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    else:
        print(
            f'Processing all available sectors of the target in a single run. Note that if the number of sectors is '
            f'large, the download might cause a timeout error from MAST.')
        print('Downloading Data from MAST and Gaia ...')
        source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector,
                         limit_mag=limit_mag, transient=transient)  # sector
        for j in range(len(source.sector_table)):
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)


def search_stars(i, sector=1, tics=None, local_directory=None):
    cam = 1 + i // 4
    ccd = 1 + i % 4
    files = glob(f'/home/tehan/data/sector{sector:04d}/lc/{cam}-{ccd}/hlsp_*.fits')
    for j in trange(len(files)):
        with fits.open(files[j], mode='denywrite') as hdul:
            try:
                if int(hdul[0].header['TICID']) in tics:
                    hdul.writeto(f"{local_directory}{files[j].split('/')[-1]}",
                                 overwrite=True)
            except:
                pass


def timebin(time, meas, meas_err, binsize):
    ind_order = np.argsort(time)
    time = time[ind_order]
    meas = meas[ind_order]
    meas_err = meas_err[ind_order]
    ct = 0
    while ct < len(time):
        ind = np.where((time >= time[ct]) & (time < time[ct] + binsize))[0]
        num = len(ind)
        wt = (1. / meas_err[ind]) ** 2.  # weights based in errors
        wt = wt / np.sum(wt)  # normalized weights
        if ct == 0:
            time_out = [np.sum(wt * time[ind])]
            meas_out = [np.sum(wt * meas[ind])]
            meas_err_out = [1. / np.sqrt(np.sum(1. / (meas_err[ind]) ** 2))]
        else:
            time_out.append(np.sum(wt * time[ind]))
            meas_out.append(np.sum(wt * meas[ind]))
            meas_err_out.append(1. / np.sqrt(np.sum(1. / (meas_err[ind]) ** 2)))
        ct += num

    return time_out, meas_out, meas_err_out


def fits2csv(dir, output_dir=None, gaiadr3=None, star_name=None, sector=None, type='cal_aper_flux', period=None):
    output_dir = f'{output_dir}{star_name}/Photometry/'
    files = glob(f'{dir}*{gaiadr3}*.fits')
    os.makedirs(output_dir, exist_ok=True)
    if sector is None:
        sector = []
        for i in range(len(files)):
            sector.append(int(files[i].split('-')[-3][3:5]))
        print(f'Available sectors: {sector}')
    error_name = {'psf_flux': 'PSF_ERR', 'aperture_flux': 'APER_ERR', 'cal_psf_flux': 'CPSF_ERR',
                  'cal_aper_flux': 'CAPE_ERR'}
    # data = np.empty((3, 0))
    for file in files:
        with fits.open(file, mode='denywrite') as hdul:
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            not_nan = np.invert(np.isnan(hdul[1].data[type][q]))
            data_ = np.array([hdul[1].data['time'][q][not_nan],
                              hdul[1].data[type][q][not_nan],
                              # np.sum(hdul[0].data, axis=(1, 2))[q][not_nan],
                              np.array([hdul[1].header[error_name[type]]] * len(hdul[1].data['time'][q][not_nan]))
                              ])
            if hdul[0].header["SECTOR"] in sector:
                np.savetxt(f'{output_dir}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}_{type}.csv', data_,
                           delimiter=',')
                # data = np.append(data, data_, axis=1)
                # plt.plot(hdul[1].data['time'], hdul[1].data[type], '.', c='silver')
                # plt.plot(data_[0], data_[1], '.')
                # # plt.xlim(0.65, 0.82)
                # plt.ylim(0.5,1.3)
                # plt.title(f'{star_name}_sector_{hdul[0].header["SECTOR"]}')
                # plt.savefig(f'{output_dir}{star_name}_sector_{hdul[0].header["SECTOR"]}.pdf', dpi=300)
                # plt.close()
    # np.savetxt(f'{output_dir}TESS_{star_name}.csv', data, delimiter=',')
    # PlotLSPeriodogram(data[0], data[1], dir=f'{dir}lc/', Title=star_name, MakePlots=True)


def star_spliter(server=1,  # or 2
                 tics=None, local_directory=None):
    for i in range(server, 56, 2):
        with Pool(16) as p:
            p.map(partial(search_stars, sector=i, tics=tics, local_directory=local_directory), range(16))
    return


def plot_lc(local_directory=None, type='cal_aper_flux'):
    files = glob(f'{local_directory}*.fits')
    os.makedirs(f'{local_directory}plots/', exist_ok=True)
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            plt.figure(constrained_layout=False, figsize=(8, 4))
            plt.plot(hdul[1].data['time'], hdul[1].data[type], '.', c='silver', label=type)
            plt.plot(hdul[1].data['time'][q], hdul[1].data[type][q], '.k', label=f'{type}_flagged')
            # plt.xlim(2753, 2755)
            # plt.ylim(0.7, 1.1)
            plt.title(f'TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}_{type}')
            plt.legend()
            # plt.show()
            plt.savefig(
                f'{local_directory}plots/TIC_{hdul[0].header["TICID"]}_sector_{hdul[0].header["SECTOR"]:04d}_{type}.png',
                dpi=300)
            plt.close()


def produce_config(tic=None, gaiadr3=None, nea=None, sector=1):
    name = f'TIC_{tic}'
    dir = f'/home/tehan/data/cosmos/transit_depth_validation/'
    output_dir = '/home/tehan/data/pyexofits/Data/'
    fits2csv(dir, star_name=name, output_dir=output_dir, gaiadr3=gaiadr3, sector=sector, type='cal_aper_flux')
    content = f"""
    [Stellar]
    st_mass = {nea['st_mass']}
    st_masserr1 = {(nea['st_masserr1'] - nea['st_masserr2']) / 2}
    st_rad = {nea['st_rad']}
    st_raderr1 = {(nea['st_raderr1'] - nea['st_raderr2']) / 2}

    [Planet]
    pl_tranmid = {nea['pl_tranmid']}
    pl_tranmiderr1 = {(nea['pl_tranmiderr1'] - nea['pl_tranmiderr2']) / 2}
    pl_orbper = {nea['pl_orbper']}
    pl_orbpererr1 = {(nea['pl_orbpererr1'] - nea['pl_orbpererr2']) / 2}
    pl_trandep = {1000 * -2.5 * np.log10(1 - (109.076 * nea['pl_rade'] / {nea['st_rad']}) ^ 2)}
    pl_masse_expected = 1
    pl_rvamp = 1
    pl_rvamperr1 = 0.1
    ###########################################################################

    [Photometry]
    InstrumentNames = TESS
    ###########################################################################

    [TESS]
    FileName = TESS_TIC_{tic}_sector_{sector}.csv
    Delimiter = ,
    GP_sho = False
    GP_prot = False
    run_masked_gp = False
    subtract_transitmasked_gp = False
    Dilution = False
    ExposureTime = {1800 if sector < 27 else 600}
    RestrictEpoch = False
    SGFilterLen = 101
    OutlierRejection = True
    """

    # Write the content to a file
    with open("output_file.txt", "w") as file:
        file.write(content)


def sort_sectors(t, dir='/home/tehan/data/cosmos/transit_depth_validation/'):
    tics = [int(s[4:]) for s in t['tic_id']]
    files = glob(f'{dir}*.fits')
    tic_sector = np.zeros((len(files), 3))
    for i in range(len(files)):
        hdul = fits.open(files[i])
        tic_sector[i, 0] = int(hdul[0].header['TICID'])
        tic_sector[i, 1] = int(hdul[0].header['GAIADR3'])
        tic_sector[i, 2] = int(hdul[0].header['sector'])
    print('All stars produced:', set(tics) <= set(tic_sector[:, 0]))
    difference_set = set(tics) - set(tic_sector[:, 0])
    print("Elements in NEA but not in folder:", list(difference_set))
    print(f'Stars={len(tics)}, lightcurves={len(np.unique(tic_sector[:, 0]))}')
    unique_elements, counts = np.unique(tic_sector[:, 0], return_counts=True)
    for i in range(56):
        print(i, len(unique_elements[counts == i]))
    return tic_sector


def get_tglc_lc(tics=None, method='query', server=1, directory=None, prior=None):
    if method == 'query':
        for i in range(len(tics)):
            target = f'TIC {tics[i]}'
            local_directory = f'{directory}{target}/'
            os.makedirs(local_directory, exist_ok=True)
            tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=True, limit_mag=16,
                    get_all_lc=False, first_sector_only=False, last_sector_only=False, sector=None, prior=prior,
                    transient=None)
            plot_lc(local_directory=f'{directory}TIC {tics[i]}/lc/', type='cal_aper_flux')
    if method == 'search':
        star_spliter(server=server, tics=tics, local_directory=directory)


if __name__ == '__main__':
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.01.30_16.12.35.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]
    tic_sector = sort_sectors(t, dir='/home/tehan/data/cosmos/transit_depth_validation/')
    for i in trange(len(tic_sector)):
        if tic_sector[i, 0] in tics:
            print(f'TIC {tic_sector[i, 0]}')
            produce_config(tic=tic_sector[i, 0], gaiadr3=tic_sector[i, 1],
                           nea=t[np.where(t['tic_id'] == f'TIC {tic_sector[i, 0]}')[0][0]], sector=tic_sector[i, 2])

    # tics = [21113347, 73848324, 743941, 323094535, 12611594, 38355468, 2521105, 187273748, 158324245, 706595, 70298662,
    #         422334505, 108155949, 187960878, 26417717, 11270200, 677945, 94893626, 120103486, 147677253, 610976842,
    #         90605642, 130162252, 297146957, 119262291, 414843476, 187273811, 416136788, 218299481, 53728859, 70412892,
    #         49040478, 399155300, 440687723, 422349422, 467929202, 171098231, 106352250, 251018878, 442530946, 370736259,
    #         267574918, 153091721, 119605900, 322388624, 91051152, 405673618, 169504920, 102068384, 123362984, 130502317,
    #         31244979, 50171060, 257429687, 420814525, 11023038, 417676990, 195193025, 122298563, 413753029, 122605766,
    #         301031110, 175265494, 164458714, 46020827, 741596, 288246496, 320636129, 126325985, 110178537, 1635721458,
    #         38258419, 159381747, 176966903, 401604346, 270955259, 145368316, 353459965, 741119, 297146115, 49512708,
    #         422325510, 121656582, 335102224, 269217040, 188876052, 244089109, 317520667, 220604190, 341694238, 77202722,
    #         2468648, 405004589, 335322931, 104866616, 167227214, 71841620, 11439959, 175233369, 26547036, 90090343,
    #         130038122, 466884459, 271898990, 271760755, 297373047, 12181371, 68952448, 287333762, 187278212, 158729099,
    #         51664268, 432247186, 92449173, 709015, 165297570, 96847781, 56760743, 218299312, 272213425, 270619059,
    #         441907126, 158635959, 187309502, 427685831, 291751373, 436478932, 453064665, 164730843, 432261603,
    #         276754403, 179012583, 318696424, 37348844, 9385460, 431701493, 49652731]
    # ror = t['pl_ratror']
    # directory = f'/home/tehan/Documents/GEMS/'
    # directory = f'/home/tehan/data/cosmos/transit_depth_validation_extra/'
    # os.makedirs(directory, exist_ok=True)
    # get_tglc_lc(tics=tics, method='search', server=1, directory=directory)
    # plot_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', type='cal_psf_flux')
    # plot_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', type='cal_aper_flux')

    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', period=0.71912603, mid_transit_tbjd=2790.58344,
    #            type='cal_psf_flux')
    # plot_pf_lc(local_directory=f'{directory}TIC {tics[0]}/lc/', period=0.71912603, mid_transit_tbjd=2790.58344,
    #            type='cal_aper_flux')

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
import textwrap

# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np
from astroquery.mast import Observations

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


def produce_config(dir, tic=None, gaiadr3=None, nea=None, sector=1):
    star_name = f'TIC_{tic}'
    output_dir = '/home/tehan/data/pyexofits/Data/'
    version = 'cal_aper_flux'
    output_dir_ = f'{output_dir}{star_name}/Photometry/'
    files = glob(f'{dir}*{gaiadr3}*{sector}*.fits')
    error_name = {'psf_flux': 'PSF_ERR', 'aperture_flux': 'APER_ERR', 'cal_psf_flux': 'CPSF_ERR',
                  'cal_aper_flux': 'CAPE_ERR'}
    # data = np.empty((3, 0))
    if len(files) == 1:
        os.makedirs(output_dir_, exist_ok=True)
        with fits.open(files[0], mode='denywrite') as hdul:
            q = [a and b for a, b in zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
            not_nan = np.invert(np.isnan(hdul[1].data[version][q]))
            cal_aper_err = 1.4826 * np.nanmedian(np.abs(hdul[1].data[version] - np.nanmedian(hdul[1].data[version])))
            data_ = np.array([hdul[1].data['time'][q][not_nan],
                              hdul[1].data[version][q][not_nan],
                              np.array([cal_aper_err] * len(hdul[1].data['time'][q][not_nan]))
                              ])
            # print(f'{output_dir_}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}.csv')
            np.savetxt(f'{output_dir_}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}.csv', data_,
                       delimiter=',')
            # np.savetxt(f'{output_dir}TESS_{star_name}.csv', data, delimiter=',')
            # PlotLSPeriodogram(data[0], data[1], dir=f'{dir}lc/', Title=star_name, MakePlots=True)
            content = textwrap.dedent(f"""\
            [Stellar]
            st_mass = {nea['st_mass']}
            st_masserr1 = {(nea['st_masserr1'] - nea['st_masserr2']) / 2:.3f}
            st_rad = {nea['st_rad']}
            st_raderr1 = {(nea['st_raderr1'] - nea['st_raderr2']) / 2:.3f}

            [Planet]
            pl_tranmid = {nea['pl_tranmid']}
            pl_tranmiderr1 = 0.01 
            pl_orbper = {nea['pl_orbper']}
            pl_orbpererr1 = 0.1 
            pl_trandep = {1000 * -2.5 * np.log10(1 - (nea['pl_rade'] / nea['st_rad'] / 109.076) ** 2):.4f}
            pl_masse_expected = 1
            pl_rvamp = 1
            pl_rvamperr1 = 0.1
            ###########################################################################

            [Photometry]
            InstrumentNames = TESS
            ###########################################################################

            [TESS]
            FileName = TESS_{star_name}_sector_{hdul[0].header['sector']}.csv
            Delimiter = ,
            GP_sho = False
            GP_prot = True
            run_masked_gp = False
            subtract_transitmasked_gp = False
            Dilution = False
            ExposureTime = {1800 if hdul[0].header['sector'] < 27 else 600}
            RestrictEpoch = False
            SGFilterLen = 101
            OutlierRejection = True""")

            # Write the content to a file
            with open(f"{output_dir}{star_name}/{star_name}_config_s{hdul[0].header['sector']:04d}.txt", "w") as file:
                file.write(content)
    elif len(files) > 1:
        os.makedirs(output_dir_, exist_ok=True)
        # data = np.empty((3, 0))
        content = textwrap.dedent(f"""\
                [Stellar]
                st_mass = {nea['st_mass']}
                st_masserr1 = {(nea['st_masserr1'] - nea['st_masserr2']) / 2:.3f}
                st_rad = {nea['st_rad']}
                st_raderr1 = {(nea['st_raderr1'] - nea['st_raderr2']) / 2:.3f}

                [Planet]
                pl_tranmid = {nea['pl_tranmid']}
                pl_tranmiderr1 = 0.01 
                #{(nea['pl_tranmiderr1'] - nea['pl_tranmiderr2']) / 2}
                pl_orbper = {nea['pl_orbper']}
                pl_orbpererr1 = 0.1 
                #{(nea['pl_orbpererr1'] - nea['pl_orbpererr2']) / 2}
                pl_trandep = {1000 * -2.5 * np.log10(1 - (nea['pl_rade'] / nea['st_rad'] / 109.076) ** 2):.4f}
                pl_masse_expected = 1
                pl_rvamp = 1
                pl_rvamperr1 = 0.1
                ###########################################################################

                [Photometry]
                InstrumentNames = {','.join(f'TESS{i}' for i in range(len(files)))}
                """)
        for j in range(len(files)):
            with fits.open(files[j], mode='denywrite') as hdul:
                q = [a and b for a, b in
                     zip(list(hdul[1].data['TESS_flags'] == 0), list(hdul[1].data['TGLC_flags'] == 0))]
                not_nan = np.invert(np.isnan(hdul[1].data[version][q]))
                cal_aper_err = 1.4826 * np.nanmedian(
                    np.abs(hdul[1].data[version] - np.nanmedian(hdul[1].data[version])))
                data_ = np.array([hdul[1].data['time'][q][not_nan],
                                  hdul[1].data[version][q][not_nan],
                                  np.array([cal_aper_err] * len(hdul[1].data['time'][q][not_nan]))
                                  ])
                np.savetxt(f'{output_dir_}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}.csv', data_,
                           delimiter=',')
                content += textwrap.dedent(f"""\
                    ###########################################################################
                    [TESS{j}]
                    FileName = TESS_{star_name}_sector_{hdul[0].header['sector']}.csv
                    Delimiter = ,
                    GP_sho = False
                    GP_prot = True
                    run_masked_gp = False
                    subtract_transitmasked_gp = False
                    Dilution = False
                    ExposureTime = {1800 if hdul[0].header['sector'] < 27 else 600}
                    RestrictEpoch = False
                    SGFilterLen = 101
                    OutlierRejection = True
                    """)
                # data = np.append(data, data_, axis=1)
        # Write the content to a file
        # with open(f"{output_dir}{star_name}/{star_name}_config.txt", "w") as file:
        #     file.write(content)


def produce_config_qlp(dir, tic=None, gaiadr3=None, nea=None, sector=1):
    star_name = f'TIC_{tic}'
    output_dir = '/home/tehan/data/pyexofits/Data_qlp/'
    output_dir_ = f'{output_dir}{star_name}/Photometry/'
    files = glob(f'{dir}*{sector}*{tic}*.fits')
    if len(files) == 1:
        os.makedirs(output_dir_, exist_ok=True)
        try:
            with fits.open(files[0], mode='denywrite') as hdul:
                q = np.where(hdul[1].data['QUALITY'] == 0)
                t = hdul[1].data['TIME'][q]
                flux = hdul[1].data['KSPSAP_FLUX'][q]
                flux_err = hdul[1].data['KSPSAP_FLUX_ERR'][q]
                not_nan = np.invert(np.isnan(flux))
                data_ = np.array([t[not_nan], flux[not_nan], flux_err[not_nan]])
                # print(f'{output_dir_}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}.csv')
                np.savetxt(f'{output_dir_}TESS_{star_name}_sector_{hdul[0].header["SECTOR"]}_qlp.csv', data_,
                           delimiter=',')
                # np.savetxt(f'{output_dir}TESS_{star_name}.csv', data, delimiter=',')
                # PlotLSPeriodogram(data[0], data[1], dir=f'{dir}lc/', Title=star_name, MakePlots=True)
                content = textwrap.dedent(f"""\
                [Stellar]
                st_mass = {nea['st_mass']}
                st_masserr1 = {(nea['st_masserr1'] - nea['st_masserr2']) / 2:.3f}
                st_rad = {nea['st_rad']}
                st_raderr1 = {(nea['st_raderr1'] - nea['st_raderr2']) / 2:.3f}
    
                [Planet]
                pl_tranmid = {nea['pl_tranmid']}
                pl_tranmiderr1 = 0.01 
                pl_orbper = {nea['pl_orbper']}
                pl_orbpererr1 = 0.1 
                pl_trandep = {1000 * -2.5 * np.log10(1 - (nea['pl_rade'] / nea['st_rad'] / 109.076) ** 2):.4f}
                pl_masse_expected = 1
                pl_rvamp = 1
                pl_rvamperr1 = 0.1
                ###########################################################################
    
                [Photometry]
                InstrumentNames = TESS
                ###########################################################################
    
                [TESS]
                FileName = TESS_{star_name}_sector_{hdul[0].header['sector']}_qlp.csv
                Delimiter = ,
                GP_sho = False
                GP_prot = True
                run_masked_gp = False
                subtract_transitmasked_gp = False
                Dilution = False
                ExposureTime = {1800 if hdul[0].header['sector'] < 27 else 600}
                RestrictEpoch = False
                SGFilterLen = 101
                OutlierRejection = True""")

                # Write the content to a file
                with open(f"{output_dir}{star_name}/{star_name}_config_s{hdul[0].header['sector']:04d}.txt", "w") as file:
                    file.write(content)
        except:
            print(files)

def sort_sectors(t, dir='/home/tehan/data/cosmos/transit_depth_validation/'):
    # tics = [int(s[4:]) for s in t['tic_id']]
    tics_string = [s[4:] for s in t['tic_id']]
    files = glob(f'{dir}*.fits')
    tic_sector = []
    for i in range(len(files)):
        hdul = fits.open(files[i])
        if str(hdul[0].header['TICID']) in tics_string:
            tic_sector.append([str(hdul[0].header['TICID']),
                               str(hdul[0].header['GAIADR3']),
                               str(hdul[0].header['sector'])])
    tic_sector = np.array(tic_sector)
    print('All stars produced:', set(tics_string) <= set(tic_sector[:, 0]))
    difference_set = set(tics_string) - set(tic_sector[:, 0])
    print("No. of stars in NEA but not in folder:", len(list(difference_set)))
    print("Stars in NEA but not in folder:", list(difference_set))
    print(f'Stars={len(tics_string)}, lightcurves={len(np.unique(tic_sector[:, 0]))}')
    unique_elements, counts = np.unique(tic_sector[:, 0], return_counts=True)
    for i in range(1, 10):
        print(f'{len(unique_elements[counts == i])} of stars are observed {i} times. ')
    print(f'{len(unique_elements[counts >= 10])} of stars are observed at least {10} times. ')
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
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PSCompPars_2024.02.05_22.52.50.csv'))
    tics = [int(s[4:]) for s in t['tic_id']]
    dir = '/home/tehan/data/cosmos/transit_depth_validation/'
    tic_sector = sort_sectors(t, dir=dir)
    np.savetxt('/home/tehan/data/cosmos/transit_depth_validation/tic_sector.csv', tic_sector, fmt='%s', delimiter=',')

    # for i in trange(len(tic_sector)):
    #     if int(tic_sector[i, 0]) in tics:
    #         produce_config_qlp('/home/tehan/data/cosmos/transit_depth_validation_qlp/', tic=int(tic_sector[i, 0]),
    #                        nea=t[np.where(t['tic_id'] == f'TIC {int(tic_sector[i, 0])}')[0][0]],
    #                        sector=int(tic_sector[i, 2])) # assign sector to '' for generating combined config; or int(tic_sector[i, 2])

    # for i in trange(len(tic_sector)):
    #     if int(tic_sector[i, 0]) in tics:
    #         obs_table = Observations.query_criteria(provenance_name="QLP", target_name=[tic_sector[i, 0]],
    #                                                 sequence_number=int(tic_sector[i, 2]))
    #         try:
    #             data_products = Observations.get_product_list(obs_table)
    #             product = data_products[0]["dataURI"]
    #             result = Observations.download_file(product,
    #                                                 local_path=f'/home/tehan/data/cosmos/transit_depth_validation_qlp/{product.split("/")[-1]}')
    #         except:
    #             if t['sy_tmag'][t['tic_id'] == int(tic_sector[i, 0])] <= 13.5:
    #                 continue
    #             else:
    #                 print(t['sy_tmag'][t['tic_id'] == int(tic_sector[i, 0])])

    # failed_to_fit = []
    # for i in trange(len(tic_sector)):
    #     if int(tic_sector[i, 0]) in tics:
    #         if len(glob(
    #                 f'/home/tehan/Downloads/Data/*/*/*/Plots_*{int(tic_sector[i, 0])}*_{int(tic_sector[i, 2])}_*.pdf')) == 1:
    #             pass
    #         else:
    #             failed_to_fit.append([int(tic_sector[i,0]), int(tic_sector[i,2])])
    # print(len(failed_to_fit))
    # print(failed_to_fit)
    # np.savetxt('/home/tehan/Downloads/Data/failed.csv', np.array(failed_to_fit), fmt='%s', delimiter=',')

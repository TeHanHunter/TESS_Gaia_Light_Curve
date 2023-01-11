import os
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut
from astroquery.mast import Catalogs

# warnings.simplefilter('ignore', UserWarning)
from threadpoolctl import ThreadpoolController, threadpool_limits
import numpy as np
controller = ThreadpoolController()

@controller.wrap(limits=1, user_api='blas')
def tglc_lc(target='NGC 7654', local_directory='', size=90, save_aper=True, limit_mag=16, get_all_lc=False,
            first_sector_only=False, sector=None, prior=None):
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
        sector = True
    source = ffi_cut(target=target, size=size, local_directory=local_directory, sector=sector)  # sector
    catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
    name = 'Gaia DR3 ' + str(np.array(catalogdata['GAIA'])[0])
    if get_all_lc:
        name = None
    if type(sector) == int:
        source.select_sector(sector=sector)
        epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
             name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
    else:
        for j in range(len(source.sector_table)):
            # try:
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                     name=name, limit_mag=limit_mag, save_aper=save_aper, prior=prior)
            if first_sector_only:
                break


if __name__ == '__main__':
    target = 'TIC 270022476'
    local_directory = f'/home/tehan/Downloads/tglc/{target}/'
    os.makedirs(local_directory, exist_ok=True)
    tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=16,
                    get_all_lc=False, first_sector_only=True, sector=None, prior=None)

    ####### list of targets example
    # local_directory = '/home/tehan/data/ob_associations/'
    # data = ascii.read(f'{local_directory}Bouret_2021_2013_Ostars.csv')
    # hosts = np.array(data['Gaia EDR3'])
    # for i in range(24, len(hosts)):
    #     tglc_lc(target='Gaia EDR3 ' + str(hosts[i]), local_directory=local_directory, size=90, save_aper=True, get_all_lc=False)

    ####### list of targets
    # local_directory = '/mnt/d/Astro/hpf/'
    # os.makedirs(local_directory + f'logs/', exist_ok=True)
    # os.makedirs(local_directory + f'lc/', exist_ok=True)
    # os.makedirs(local_directory + f'epsf/', exist_ok=True)
    # os.makedirs(local_directory + f'source/', exist_ok=True)
    # data = ascii.read(local_directory + 'hpf_toi_ffi_targets.txt')
    # hosts = np.array(data['TIC'])
    # gaia_name = []
    # for i in range(len(hosts)):
    #     target = hosts[i]  # Target identifier or coordinates TOI-3714
    #     catalogdata = Catalogs.query_object('TIC ' + str(target), radius=0.02, catalog="TIC")
    #     name = 'Gaia DR2 ' + str(np.array(catalogdata['GAIA'])[np.where(catalogdata['ID'] == str(target))[0][0]])
    #     gaia_name.append(name)
    #     print('TIC ' + str(target), name)
    #     size = 90  # int, suggests big cuts
    #     source = ffi_cut(target='TIC ' + str(target), size=size, local_directory=local_directory)
    #     for j in range(len(source.sector_table)):
    #         try:
    #             source.select_sector(sector=source.sector_table['sector'][j])
    #             epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
    #                  name=name)
    #         except:
    #             warnings.warn(f'Skipping sector {source.sector_table["sector"][j]}. (Target not in cut)')
    # np.savetxt('/mnt/d/Astro/hpf/hpf_gaia_dr2.txt', gaia_name)

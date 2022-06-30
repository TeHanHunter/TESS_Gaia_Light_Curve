from tglc.target_lightcurve import *
from astropy.io import ascii
warnings.simplefilter('ignore', UserWarning)


def tglc_lc(target='NGC 7654', local_directory='', size=90, save_aper=True, get_all_lc=False):
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
    source = ffi_cut(target=target, size=size, local_directory=local_directory)  # sector
    catalogdata = Catalogs.query_object(str(target), radius=0.004, catalog="TIC")
    name = 'Gaia DR2 ' + str(np.array(catalogdata['GAIA'])[0])
    if get_all_lc:
        name = None
    for j in range(len(source.sector_table)):
        try:
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name, save_aper=save_aper)
        except:
            sector_num = source.sector_table["sector"][j]
            warnings.warn(f'Skipping sector {sector_num}. (Target not in cut)')


if __name__ == '__main__':
    local_directory = '/home/tehan/data/ob_associations/'
    data = ascii.read(f'{local_directory}Bouret_2021_2013_Ostars.csv')
    hosts = np.array(data['star ID'])
    for i in range(len(hosts)):
        tglc_lc(target=hosts[i], local_directory=local_directory, size=90, save_aper=True, get_all_lc=False)


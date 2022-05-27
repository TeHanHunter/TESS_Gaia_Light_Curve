from tglc.target_lightcurve import *
warnings.simplefilter('always', UserWarning)


def tglc_lc(target='NGC 7654', local_directory='', size=90):
    '''
    Generate light curve for a single target.
    :param target: str, required
    target identifier
    :param local_directory: string, required
    output directory
    :param size: int, optional
    size of the FFI cut, default size is 90. Recommend large number for better quality. Cannot exceed 100.
    '''
    os.makedirs(local_directory + f'logs/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    source = ffi(target=target, size=size, local_directory=local_directory)  # sector
    catalogdata = Catalogs.query_object(str(target), radius=0.02, catalog="TIC")
    name = 'Gaia DR2 ' + str(np.array(catalogdata['GAIA'])[0])
    # print(name)
    for j in range(len(source.sector_table)):
        try:
            source.select_sector(sector=source.sector_table['sector'][j])
            epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
                 name=name)
        except:
            warnings.warn(f'Skipping sector {source.sector_table["sector"][j]}. (Target not in cut)')


if __name__ == '__main__':
    tglc_lc(target='NGC 7654', local_directory='', size=90)

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
    #     source = ffi(target='TIC ' + str(target), size=size, local_directory=local_directory)
    #     for j in range(len(source.sector_table)):
    #         try:
    #             source.select_sector(sector=source.sector_table['sector'][j])
    #             epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
    #                  name=name)
    #         except:
    #             warnings.warn(f'Skipping sector {source.sector_table["sector"][j]}. (Target not in cut)')
    # np.savetxt('/mnt/d/Astro/hpf/hpf_gaia_dr2.txt', gaia_name)

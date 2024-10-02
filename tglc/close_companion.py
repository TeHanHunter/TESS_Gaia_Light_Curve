import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.io import ascii
import pkg_resources
from astroquery.mast import Catalogs
from tqdm import tqdm

Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"


def query_gaia_for_neighbors(tic_id_list, radius=1, delta_tess_mag=5):
    result = []
    for tic_id in tqdm(tic_id_list):
        target = Catalogs.query_object(f'TIC {tic_id}', radius=radius * 21 * 0.707 / 3600, catalog="TIC")
        star_num = np.where(target['ID'] == str(tic_id))[0][0]
        if star_num != 0:
            print('Star TIC not found. ')
        tess_mag_target = target['Tmag'][star_num]
        for i in range(len(target)):
            if i == star_num:
                continue
            else:
                tess_mag_neighbor = target['Tmag'][i]
                delta_mag = abs(tess_mag_target - tess_mag_neighbor)
                if delta_mag <= delta_tess_mag:
                    result.append([
                        target['ID'][star_num],
                        target['GAIA'][star_num],
                        target['ID'][i],
                        target['GAIA'][i],
                        target['dstArcSec'][i],
                        delta_mag])
                    print(result[-1])
    return result


if __name__ == '__main__':
    t = ascii.read(pkg_resources.resource_stream(__name__, 'PS_2024.10.02_11.45.09.csv'))
    print(t['tic_id'])
    tic_id_list = set([int(s[4:]) for s in t['tic_id']])
    print(len(tic_id_list))
    neighbors_result = query_gaia_for_neighbors(list(tic_id_list))
    ascii.write(np.array(neighbors_result), 'tic_neighbor.csv', format='csv',
                names=['planet_host', 'planet_host_gaia', 'neighbor', 'neighbor_gaia', 'dstArcSec', 'deltaTmag'],
                overwrite=True)

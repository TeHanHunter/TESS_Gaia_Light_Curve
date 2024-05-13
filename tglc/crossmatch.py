import glob
from tqdm import tqdm
import numpy as np
from astropy.io import ascii

def crossmatch(target_list='ListofMdwarfTICs_ForTe.csv', folder='/home/tehan/data/cosmos/GEMS/tessminer/'):
    files = glob.glob(folder + '*.fits')
    files = [file.split('-')[1] for file in files]
    targets = ascii.read(folder + target_list)
    gaia_dr3 = [target.split()[-1] for target in targets['designation']]
    occurances = np.zeros(len(gaia_dr3))
    print(files[:10])
    print(gaia_dr3[:10])
    for i, designation in enumerate(tqdm(gaia_dr3)):
        for j in files:
            if designation == j:
                occurances[i] += 1
                print(occurances)
    targets['occurances'] = occurances
    ascii.write(targets, folder + 'ListofMdwarfTICs_crossmatch.csv', overwrite=True)
    return targets


if __name__ == '__main__':
    targets = crossmatch()
    print(targets[:10])
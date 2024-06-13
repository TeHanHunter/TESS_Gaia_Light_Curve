import glob
from tqdm import tqdm
import numpy as np
from astropy.io import ascii
from tqdm import trange


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
    targets['occurances'] = occurances
    ascii.write(targets, folder + 'ListofMdwarfTICs_crossmatch_even.csv', overwrite=True)
    return targets


def combine(folder='/Users/tehan/Downloads/'):
    all = ascii.read(folder + 'ListofMdwarfTICs_ForTe.csv')
    sector_num = np.zeros((len(all)))
    missing_sector = 0
    for i in trange(len(all)):
        num = 0
        sectors = all['sectors'][i].split()
        for j in sectors:
            try:
                if int(j) < 56:
                    num += 1
            except ValueError:
                missing_sector += 1
                # print(all['sectors'][i])
        sector_num[i] = num
    even = ascii.read(folder + 'ListofMdwarfTICs_crossmatch_even.csv')
    odd = ascii.read(folder + 'ListofMdwarfTICs_crossmatch_odd.csv')
    occurrences = even['occurances'].data + odd['occurances'].data
    print(len(np.where(sector_num - occurrences != 0)[0]))
    print(len(np.where(sector_num - occurrences > 0)[0]))
    complete_miss = np.zeros((len(sector_num)))
    for i in range(len(sector_num)):
        if sector_num[i] > 0 and occurrences[i] == 0:
            complete_miss[i] = 1
    print(np.sum(complete_miss))
    complete_miss_t = all[np.where(complete_miss == 1)[0]]
    complete_miss_t.write(folder + 'ListofMdwarfTICs_crossmatch_complete_miss.csv')

    # print(np.sum(np.where(old_lcs > 0)[0]))
    # print(len(np.where(sector_num == 0)[0]))
    # print(even[np.where(sector_num-occurrences > 0)[0][0]], odd[np.where(sector_num-occurrences > 0)[0][0]])
    # print(len(np.where(occurrences == 0)[0]))
    missing = all[np.where(sector_num - occurrences > 0)[0]]
    print(sum(sector_num - occurrences))
    # missing.write(folder + 'ListofMdwarfTICs_crossmatch_missing.csv')


if __name__ == '__main__':
    # targets = crossmatch()
    # print(targets[:10])
    combine()

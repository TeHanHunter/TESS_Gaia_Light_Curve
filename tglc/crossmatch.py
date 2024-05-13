import glob
from tqdm import tqdm
import numpy as np
from astropy.io import ascii

import os
import glob
import pandas as pd


def crossmatch(target_list='ListofMdwarfTICs_ForTe.csv', folder='/home/tehan/data/cosmos/GEMS/tessminer/'):
    # Get list of files in the folder
    files = glob.glob(os.path.join(folder, '*.fits'))

    # Read the target list into a Pandas DataFrame
    targets = pd.read_csv(os.path.join(folder, target_list))

    # Extract numbers from the 'designation' column
    gaia_dr3 = [target.split()[-1] for target in targets['designation']]

    # Count occurrences using vectorized operations
    occurrences = np.zeros(len(gaia_dr3))
    for i, designation in tqdm(enumerate(gaia_dr3)):
        occurrences[i] = sum(designation in j for j in files)

    # Add occurrences as a new column
    targets['occurrences'] = occurrences

    # Write the updated DataFrame to a CSV file
    targets.to_csv(os.path.join(folder, 'ListofMdwarfTICs_crossmatch.csv'), index=False)

    return targets


if __name__ == '__main__':
    targets = crossmatch()
    print(targets[:10])
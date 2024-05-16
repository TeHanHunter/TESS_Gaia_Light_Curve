import numpy as np
import matplotlib.pyplot as plt
import glob
def get_mad(folder, power=1.0):
    files = glob.glob(f'{folder}*{power}.npy')
    print(files)
    mag = np.array([])
    mad = np.array([])
    for file in files:
        mad_power = np.load(file)
        print(mad_power)
        mag = np.append(mag, mad_power[0])
        mad = np.append(mad, mad_power[1])
    idx = np.argsort(mag)
    mag = mag[idx]
    mad = mad[idx]

    return mag, mad

if __name__ == '__main__':
    plt.plot(get_mad('/Users/tehan/Documents/TGLC/MAD_power/'))
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import glob
def get_mad(folder, power=1.4):
    files = glob.glob(f'{folder}*{power}.npy')
    print(files)
    mag = np.array([])
    mad = np.array([])
    for file in files:
        mad_power = np.load(file)
        mag = np.append(mag, mad_power[0])
        mad = np.append(mad, mad_power[1])
    idx = np.argsort(mag)
    mag = mag[idx]
    mad = mad[idx]
    aper_precision = 1.48 * mad / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - mag) / 2.5))
    print(mad)
    return mag, aper_precision

if __name__ == '__main__':
    mag, aper_precision = get_mad('/Users/tehan/Documents/TGLC/MAD_power/', power=1.0)
    plt.plot(mag, aper_precision, '.')
    plt.ylim(1e-4,1)
    plt.yscale('log')
    plt.show()
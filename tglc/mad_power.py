import numpy as np
import matplotlib.pyplot as plt
import glob
def get_mad(folder, power=1.4):
    files = glob.glob(f'{folder}*{power}.npy')
    # print(files)
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
    print(len(mag))
    return mag, aper_precision

if __name__ == '__main__':
    mag, aper_precision = get_mad('/Users/tehan/Documents/TGLC/MAD_power/', power=1.3)

    bin_size = 200
    tglc_mag = np.median(mag[:len(mag) // bin_size * bin_size].reshape(-1, bin_size), axis=1)
    tglc_binned = np.median(aper_precision[:len(aper_precision) // bin_size * bin_size].reshape(-1, bin_size),
                            axis=1)

    plt.plot(mag, aper_precision, '.k', alpha=0.1)
    plt.plot(tglc_mag, tglc_binned, 'r', zorder=2)
    plt.ylim(1e-4,1)
    plt.yscale('log')
    plt.show()
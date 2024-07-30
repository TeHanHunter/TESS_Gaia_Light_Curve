from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from tglc.lc_plot import get_MAD_qlp
from astropy.io import ascii
from tqdm import trange

def plot_MAD_seaborn(qlp_tic, qlp_precision, tglc_tic, tglc_precision):
    palette = sns.color_palette('colorblind')
    tglc_color = palette[3]
    qlp_color = palette[2]
    # mad_tglc = np.load('/Users/tehan/Documents/TGLC/mad_tglc_30min.npy', allow_pickle=True)
    # mad_qlp = np.load('/Users/tehan/Documents/TGLC/mad_qlp_30min.npy', allow_pickle=True)
    # noise_2015 = ascii.read('/Users/tehan/Documents/TGLC/noisemodel.dat')
    mad_tglc = np.load('/home/tehan/Documents/tglc/mad_tglc_30min.npy', allow_pickle=True)
    mad_qlp = np.load('/home/tehan/Documents/tglc/mad_qlp_30min.npy', allow_pickle=True)
    noise_2015 = ascii.read('/home/tehan/Documents/tglc/prior_mad/noisemodel.dat')

    # Sort data
    sorted_indices_tglc = np.argsort(mad_tglc.tolist()['tics'])
    sorted_indices_qlp = np.argsort(mad_qlp.tolist()['tics'])

    # Interpolate noise model
    noise_interp = interp1d(noise_2015['col1'], noise_2015['col2'], kind='cubic')

    # Bin data
    bin_size = 40000
    tglc_mag = np.median(mad_tglc.tolist()['tics'][sorted_indices_tglc][
                         :len(mad_tglc.tolist()['tics'][sorted_indices_tglc]) // bin_size * bin_size].reshape(-1,
                                                                                                              bin_size),
                         axis=1)
    tglc_binned = np.median(mad_tglc.tolist()['aper_precisions'][sorted_indices_tglc][:len(
        mad_tglc.tolist()['aper_precisions'][sorted_indices_tglc]) // bin_size * bin_size].reshape(-1, bin_size),
                            axis=1)

    bin_size = 10000
    qlp_mag = np.nanmedian(mad_qlp.tolist()['tics'][sorted_indices_qlp][
                           :len(mad_qlp.tolist()['tics'][sorted_indices_qlp]) // bin_size * bin_size].reshape(-1,
                                                                                                              bin_size),
                           axis=1)
    qlp_binned = np.nanmedian(mad_qlp.tolist()['qlp_precision'][sorted_indices_qlp][:len(
        mad_qlp.tolist()['qlp_precision'][sorted_indices_qlp]) // bin_size * bin_size].reshape(-1, bin_size), axis=1)

    # Create Seaborn plot
    # sns.set_style("whitegrid")
    sns.set(rc={'font.family': 'serif', 'font.serif': 'DejaVu Serif', 'font.size': 12,
                'axes.edgecolor': '0.2', 'axes.labelcolor': '0.', 'xtick.color': '0.', 'ytick.color': '0.',
                'axes.facecolor': '0.95', 'grid.color': '0.8'})

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.1})

    # Top panel
    ax[0].scatter(mad_tglc.tolist()['tics'][sorted_indices_tglc],
                  mad_tglc.tolist()['aper_precisions'][sorted_indices_tglc], s=0.01, color=tglc_color, alpha=0.01)
    ax[0].scatter(0,0, s=1, color=tglc_color, alpha=1,label='TGLC Aperture')
    ax[0].scatter(mad_qlp.tolist()['tics'][sorted_indices_qlp], mad_qlp.tolist()['qlp_precision'][sorted_indices_qlp],
                  s=0.01, color=qlp_color, alpha=0.01)
    ax[0].scatter(0,0 ,s=1, color=qlp_color, alpha=1, label='QLP SAP')
    ax[0].plot(noise_2015['col1'], noise_2015['col2'], color='k', label='Sullivan (2015)')
    # ax[0].hlines(y=[0.1, 0.01], xmin=7, xmax=16.5, colors='k', linestyles='dotted')
    ax[0].set_ylabel('Estimated Photometric Precision')
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-4, 1)
    ax[0].set_title('S56 30-min bin')
    ax[0].legend(loc=4, markerscale=4, fontsize=8)

    # Bottom panel
    ax[1].plot(tglc_mag, tglc_binned / noise_interp(tglc_mag), color=tglc_color, label='TGLC Aperture')
    ax[1].plot(qlp_mag, qlp_binned / noise_interp(qlp_mag), color=qlp_color, label='QLP SAP')
    ax[1].plot(qlp_tic, qlp_precision / noise_interp(qlp_tic), color=qlp_color, ls='', marker='o', ms=4)
    # ax[1].plot(tglc_tic, tglc_precision / noise_interp(tglc_tic), color=tglc_color, ls='', marker='o', ms=10)

    ax[1].hlines(y=1, xmin=7, xmax=17, colors='k', label='Sullivan (2015)')
    ax[1].set_ylim(0.5, 2.5)
    ax[1].set_yticks([0.5, 1, 1.5, 2])
    ax[1].set_yticklabels(['0.5', '1', '1.5', '2'])
    ax[1].set_xlabel('TESS magnitude')
    ax[1].set_ylabel('Precision Ratio')
    ax[1].legend(loc=4, markerscale=1, ncol=2, columnspacing=1, fontsize=7.2)

    plt.xlim(7, 16.5)
    # plt.savefig('/Users/tehan/Documents/TGLC/s56_mad_30min_example.png', bbox_inches='tight', dpi=300)
    plt.savefig('/home/tehan/Documents/tglc/s56_mad_30min_example.png', bbox_inches='tight', dpi=300)
    # plt.show()

if __name__ == '__main__':
    files = glob('/home/tehan/data/cosmos/transit_depth_validation_qlp/*.fits')
    qlp_tic = []
    qlp_precision = []
    for i in trange(len(files)):
        t,q = get_MAD_qlp(i,files)
        qlp_tic.append(t)
        qlp_precision.append(q)
    # files = glob('/Users/tehan/Documents/TGLC/s56_mad_tglc/*.fits')
    tglc_tic = []
    tglc_precision = []
    # for i in range(len(files)):
    #     t,q = get_MAD(i,files)
    #     tglc_tic.append(t)
    #     tglc_precision.append(q)
    plot_MAD_seaborn(qlp_tic, qlp_precision, tglc_tic, tglc_precision)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten
from astropy.io import ascii
import pickle
from astropy.io import fits
from matplotlib.patches import ConnectionPatch
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr


def figure_1():
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    period = 2.2895
    t1_ = 435
    t2_ = 455
    t3_ = 940

    t1 = 530
    t2 = 555
    t3 = 1080

    time = np.load('/mnt/c/users/tehan/desktop/eleanor_time.npy')
    eleanor_pca = np.load('/mnt/c/users/tehan/desktop/TIC_270023061.npy')
    eleanor_PSF = np.load('/mnt/c/users/tehan/desktop/eleanor_PSF_1251.npy')
    moffat = np.load('/mnt/c/users/tehan/desktop/moffat_1251.npy')
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')
    qlp = fits.open(
        '/mnt/c/users/tehan/desktop/hlsp_qlp_tess_ffi_s0017-0000000270023089_tess_v01_llc.fits')
    qlp_data = qlp[1].data

    bg_mod = lightcurve[1251][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][1251])
    # epsf
    flatten_lc, trend_lc = flatten(source.time, (lightcurve[1251] - bg_mod) / np.median((lightcurve[1251] - bg_mod)),
                                   window_length=1,
                                   method='biweight',
                                   return_trend=True)
    # moffat
    flatten_lc_, trend_lc_ = flatten(source.time, (moffat - bg_mod) / np.median(moffat - bg_mod), window_length=1,
                                     method='biweight',
                                     return_trend=True)
    # eleanor gaussian
    flatten_lc__, trend_lc__ = flatten(time, eleanor_PSF / np.median(eleanor_PSF), window_length=1, method='biweight',
                                       return_trend=True)

    # eleanor pca
    flatten_lc___, trend_lc___ = flatten(time, eleanor_pca / np.median(eleanor_pca), window_length=1, method='biweight',
                                         return_trend=True)

    # qlp
    # flatten_lc___, trend_lc___ = flatten(qlp_data['TIME'], qlp_data['KSPSAP_FLUX'], window_length=1,
    #                                    method='biweight',
    #                                    return_trend=True)

    fig = plt.figure(constrained_layout=False, figsize=(10, 8))
    gs = fig.add_gridspec(5, 15)
    gs.update(wspace=0.1, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0:5])
    ax2 = fig.add_subplot(gs[1, 0:5])
    ax3 = fig.add_subplot(gs[2, 0:5])
    ax4 = fig.add_subplot(gs[3, 0:5])
    ax5 = fig.add_subplot(gs[4, 0:5])

    ax6 = fig.add_subplot(gs[0, 5:11])
    ax7 = fig.add_subplot(gs[1, 5:11])
    ax8 = fig.add_subplot(gs[2, 5:11])
    ax9 = fig.add_subplot(gs[3, 5:11])
    ax10 = fig.add_subplot(gs[4, 5:11])

    ax11 = fig.add_subplot(gs[0, 12:])
    ax12 = fig.add_subplot(gs[1, 12:])
    ax13 = fig.add_subplot(gs[2, 12:])

    ax1.plot(time, flatten_lc___ / np.median(flatten_lc___), '.k', ms=1)
    ax2.plot(time, flatten_lc__, '.k', ms=1)
    ax3.plot(qlp_data['TIME'], qlp_data['KSPSAP_FLUX'], '.k', ms=1)
    ax4.plot(source.time, flatten_lc_, '.k', ms=1)
    ax5.plot(source.time, flatten_lc, '.k', ms=1)

    ax6.plot(time[0:t1_] % period, flatten_lc___[0:t1_] / np.median(flatten_lc___), '.k', ms=1)
    ax6.plot(time[t2_:t3_] % period, flatten_lc___[t2_:t3_] / np.median(flatten_lc___), '.k', ms=1)
    ax7.plot(time[0:t1_] % period, flatten_lc__[0:t1_], '.k', ms=1)
    ax7.plot(time[t2_:t3_] % period, flatten_lc__[t2_:t3_], '.k', ms=1)
    ax8.plot(qlp_data['TIME'] % period, qlp_data['KSPSAP_FLUX'], '.k', ms=1)
    ax9.plot(source.time[0:t1] % period, flatten_lc_[0:t1], '.k', ms=1)
    ax9.plot(source.time[t2:t3] % period, flatten_lc_[t2:t3], '.k', ms=1)
    ax10.plot(source.time[0:t1] % period, flatten_lc[0:t1], '.k', ms=1)
    ax10.plot(source.time[t2:t3] % period, flatten_lc[t2:t3], '.k', ms=1)
    ax11.plot(time[0:t1_] % period, flatten_lc___[0:t1_] / np.median(flatten_lc___), '.k', ms=0.6, zorder=3)
    ax11.plot(time[t2_:t3_] % period, flatten_lc___[t2_:t3_] / np.median(flatten_lc___), '.k', ms=0.6, zorder=3)
    ax12.plot(time[0:t1_] % period, flatten_lc__[0:t1_], '.k', ms=0.6, zorder=3)
    ax12.plot(time[t2_:t3_] % period, flatten_lc__[t2_:t3_], '.k', ms=0.6, zorder=3)
    ax13.plot(qlp_data['TIME'] % period, qlp_data['KSPSAP_FLUX'], '.k', ms=0.6, label='TESS', zorder=3)

    data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/1251_g.csv')
    data.remove_rows(np.where(data['catflags'] != 0))
    tbjd = data['hjd'] - 2457000
    mag = data['mag']
    flux = 10 ** (- mag / 2.5)  # 3.208e-10 *
    ax6.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax7.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax8.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax9.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax10.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax11.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax12.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax13.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')

    data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/1251_r.csv')
    data.remove_rows(np.where(data['catflags'] != 0))
    tbjd = data['hjd'] - 2457000
    mag = data['mag']
    flux = 10 ** (- mag / 2.5)
    ax6.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax7.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax8.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax9.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax10.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    ax11.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    ax12.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    ax13.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')

    ax1.set_title('eleanor PCA', loc='left')
    ax2.set_title('eleanor PSF', loc='left')
    ax3.set_title('QLP KSPSAP', loc='left')
    ax4.set_title('Moffat PSF', loc='left')
    ax5.set_title('ePSF', loc='left')
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", direction="in")
    ax2.set_xticklabels([])
    ax2.tick_params(axis="both", direction="in")
    ax3.set_xticklabels([])
    ax3.tick_params(axis="both", direction="in")
    ax4.set_xticklabels([])
    ax4.tick_params(axis="both", direction="in")
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax6.tick_params(axis="both", direction="in")
    ax7.set_xticklabels([])
    ax7.set_yticklabels([])
    ax7.tick_params(axis="both", direction="in")
    ax8.set_xticklabels([])
    ax8.set_yticklabels([])
    ax8.tick_params(axis="both", direction="in")
    ax9.set_xticklabels([])
    ax9.set_yticklabels([])
    ax9.tick_params(axis="both", direction="in")

    ax5.tick_params(axis="both", direction="in")
    ax10.set_yticklabels([])
    ax10.tick_params(axis="both", direction="in")
    ax11.set_ylabel('Normalized Flux')
    ax11.tick_params(axis="both", direction="in")
    ax11.yaxis.set_label_position("right")
    ax11.yaxis.tick_right()
    ax11.set_xticklabels([])
    ax12.set_ylabel('Normalized Flux')
    ax12.tick_params(axis="both", direction="in")
    ax12.yaxis.set_label_position("right")
    ax12.yaxis.tick_right()
    ax12.set_xticklabels([])
    ax13.set_ylabel('Normalized Flux')
    ax13.tick_params(axis="both", direction="in")
    ax13.yaxis.set_label_position("right")
    ax13.yaxis.tick_right()

    ax1.set_ylabel('Normalized Flux')
    ax2.set_ylabel('Normalized Flux')
    ax3.set_ylabel('Normalized Flux')
    ax4.set_ylabel('Normalized Flux')
    ax5.set_ylabel('Normalized Flux')
    # ax1.set_xlabel('TBJD')
    # ax2.set_xlabel('TBJD')
    # ax3.set_xlabel('TBJD')
    # ax4.set_xlabel('TBJD')
    ax5.set_xlabel('TBJD')
    # ax6.set_xlabel('Phase (days)')
    # ax7.set_xlabel('Phase (days)')
    # ax8.set_xlabel('Phase (days)')
    # ax9.set_xlabel('Phase (days)')
    ax10.set_xlabel('Phase (days)')
    ax13.set_xlabel('Phase (days)')

    ax1.set_ylim(0.65, 1.1)
    ax6.set_ylim(0.65, 1.1)
    ax2.set_ylim(0.65, 1.1)
    ax7.set_ylim(0.65, 1.1)
    ax3.set_ylim(0.65, 1.1)
    ax8.set_ylim(0.65, 1.1)
    ax4.set_ylim(0.65, 1.1)
    ax9.set_ylim(0.65, 1.1)
    ax5.set_ylim(0.65, 1.1)
    ax10.set_ylim(0.65, 1.1)
    ax11.set_ylim(0.993, 1.006)
    ax12.set_ylim(0.993, 1.006)
    ax13.set_ylim(0.993, 1.006)

    ax1.plot(time[t3_:], flatten_lc___[t3_:] / np.median(flatten_lc___), '.', c='silver', ms=2)
    ax1.plot(time[t1_:t2_], flatten_lc___[t1_:t2_] / np.median(flatten_lc___), '.', c='silver', ms=1)
    ax2.plot(time[t3_:], flatten_lc__[t3_:], '.', c='silver', ms=1)
    ax2.plot(time[t1_:t2_], flatten_lc__[t1_:t2_], '.', c='silver', ms=1)
    ax4.plot(source.time[t3:], flatten_lc_[t3:], '.', c='silver', ms=1)
    ax4.plot(source.time[t1:t2], flatten_lc_[t1:t2], '.', c='silver', ms=1)
    ax5.plot(source.time[t3:], flatten_lc[t3:], '.', c='silver', ms=1)
    ax5.plot(source.time[t1:t2], flatten_lc[t1:t2], '.', c='silver', ms=1)
    con1 = ConnectionPatch(xyA=(ax6.get_xlim()[1], 1.006), xyB=(ax11.get_xlim()[0], 1.006), coordsA="data",
                           coordsB="data", axesA=ax6,
                           axesB=ax11, color="k")
    con2 = ConnectionPatch(xyA=(ax6.get_xlim()[1], 0.993), xyB=(ax11.get_xlim()[0], 0.993), coordsA="data",
                           coordsB="data", axesA=ax6,
                           axesB=ax11, color="k")
    ax6.add_artist(con1)
    ax6.add_artist(con2)
    con1 = ConnectionPatch(xyA=(ax7.get_xlim()[1], 1.006), xyB=(ax12.get_xlim()[0], 1.006), coordsA="data",
                           coordsB="data", axesA=ax7,
                           axesB=ax12, color="k")
    con2 = ConnectionPatch(xyA=(ax7.get_xlim()[1], 0.993), xyB=(ax12.get_xlim()[0], 0.993), coordsA="data",
                           coordsB="data", axesA=ax7,
                           axesB=ax12, color="k")
    ax7.add_artist(con1)
    ax7.add_artist(con2)
    con1 = ConnectionPatch(xyA=(ax8.get_xlim()[1], 1.006), xyB=(ax13.get_xlim()[0], 1.006), coordsA="data",
                           coordsB="data", axesA=ax8,
                           axesB=ax13, color="k")
    con2 = ConnectionPatch(xyA=(ax8.get_xlim()[1], 0.993), xyB=(ax13.get_xlim()[0], 0.993), coordsA="data",
                           coordsB="data", axesA=ax8,
                           axesB=ax13, color="k")
    ax8.add_artist(con1)
    ax8.add_artist(con2)
    ax13.legend(bbox_to_anchor=(1, -1.7), loc=1)
    # plt.savefig('/mnt/c/users/tehan/desktop/light_curve_comparison_1251.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def figure_2():
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')

    def flatten_lc(source, lightcurve, index, bg_mod=0):
        flatten_lc = flatten(source.time, (lightcurve[index] - bg_mod) / (np.median(lightcurve[index]) - bg_mod),
                             window_length=1,
                             method='biweight',
                             return_trend=False)
        return flatten_lc

    index = [699, 77, 1251, 469, 1585]
    period = [1.01968, 1.9221, 2.2895, 6.126, 6.558]
    lc = np.zeros((len(index), len(source.time)))
    for i in range(len(index)):
        lc[i] = flatten_lc(source, lightcurve, index[i])
        if i == 2:
            print(
                lightcurve[index[i]][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][index[i]]))
            lc[i] = flatten_lc(source, lightcurve, index[i], bg_mod=
            lightcurve[index[i]][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][index[i]]))

    fig = plt.figure(constrained_layout=False, figsize=(10, 9))
    gs = fig.add_gridspec(len(index), 4)
    gs.update(wspace=0.2, hspace=0.6)
    t1 = 530
    t2 = 555
    t3 = 1080
    for i in range(len(index)):
        ax1 = fig.add_subplot(gs[i, 0:2])
        ax2 = fig.add_subplot(gs[i, 2:])
        ax1.plot(source.time, lc[i], '.k', ms=1, zorder=0)
        ax1.scatter(source.time[t3:], lc[i][t3:], marker='x', c='r', s=7, linewidths=0.5)
        ax1.scatter(source.time[t1:t2], lc[i][t1:t2], marker='x', c='r', s=7, linewidths=0.5, label='TESS outliers')
        # ax1.plot(source.time[t2:t2 + 500], lc[i][t2:t2 + 500], '.', c='C0', ms=3)
        ax2.plot(source.time[0:t1] % period[i], lc[i][0:t1], '.k', ms=1)
        ax2.plot(source.time[t2:t3] % period[i], lc[i][t2:t3], '.k', ms=1, label='TESS')
        ax1.tick_params(axis="both", direction="in")
        ax2.set_yticklabels([])
        ax2.tick_params(axis="both", direction="in")

        ylim = ax2.get_ylim()
        ax2.set_ylim((ylim[0] - 0.02, ylim[1] + 0.02))
        ylim = ax2.get_ylim()
        ax1.set_ylim(ylim)
        try:
            data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/{index[i]}_g.csv')
            data.remove_rows(np.where(data['catflags'] != 0))
            tbjd = data['hjd'] - 2457000
            mag = data['mag']
            flux = 10 ** (- mag / 2.5)  # 3.208e-10 *
            ax2_ = ax2.twinx()
            ax2_.plot(tbjd % period[i], flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
            # ax2_.set_ylabel('ZTF mag')
            ax2_.set_ylim(ylim)
            ax2_.get_yaxis().set_visible(False)
            # ax2_.tick_params(axis='y', colors='k')
        except:
            pass
        try:
            data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/{index[i]}_r.csv')
            data.remove_rows(np.where(data['catflags'] != 0))
            tbjd = data['hjd'] - 2457000
            mag = data['mag']
            flux = 10 ** ((4.74 - mag) / 2.5)
            ax2__ = ax2.twinx()
            ax2__.scatter(tbjd % period[i], flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                          label='ZTF r-band')
            ax2__.set_ylim(ylim)
            ax2__.get_yaxis().set_visible(False)
            if i == 2:
                ax2.set_ylim([0.65, 1.1])
                ax2_.set_ylim([0.65, 1.1])
                ax2__.set_ylim([0.65, 1.1])
        except:
            pass
        if i == 4:
            ax1.set_xlabel('TBJD', labelpad=0)
        ax1.set_ylabel('Normalized Flux', labelpad=0)
        ax2.set_xlabel('Phase (days)', labelpad=0)
        ax1.set_title(f'{source.gaia[index[i]]["designation"]}', loc='left')
        ax2.set_title(f'P = {period[i]}' + f' TESS magnitude = {source.gaia[index[i]]["tess_mag"]:.2f}')
    ax1.legend(bbox_to_anchor=(0.9, -.35))
    ax2.legend(bbox_to_anchor=(-.8, -.35))
    ax2_.legend(bbox_to_anchor=(0.5, -.35))
    ax2__.legend(bbox_to_anchor=(.9, -.35))
    # plt.savefig(f'/mnt/c/users/tehan/desktop/EBs.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def figure_3():
    from tqdm import trange
    import os
    from glob import glob
    local_directory = f'/mnt/d/TESS_Sector_17/'
    input_files = glob(f'/mnt/d/TESS_Sector_17/mastDownload/HLSP/*/*.fits')
    input_files1 = glob('/mnt/d/TESS_Sector_17/source/1-1/pca/*')
    input_files2 = glob('/mnt/d/TESS_Sector_17/source/1-1/psf/*')
    mag = []
    mean_diff = []
    for i in trange(len(input_files)):
        with fits.open(input_files[i], mode='denywrite') as hdul:
            quality = hdul[1].data['QUALITY']
            lc = hdul[1].data['KSPSAP_FLUX']
            mag_ = hdul[0].header['TESSMAG']
            scale = 1.5e4 * 10 ** ((10 - mag_) / 2.5)
            index = np.where(quality == 0)
            mag.append(mag_)
            mean_diff.append(scale * np.mean(np.abs(np.diff(lc[index] / np.nanmedian(lc[index])))))
    mag1 = []
    mean_diff1 = []
    for i in trange(len(input_files1)):
        lc = np.load(input_files1[i])
        mag1.append(float(os.path.basename(input_files1[i])[:-4]))
        mean_diff1.append(np.mean(np.abs(np.diff(lc))))
    mag2 = []
    mean_diff2 = []
    for i in trange(len(input_files2)):
        lc = np.load(input_files2[i])
        mag2.append(float(os.path.basename(input_files2[i])[:-4]))
        mean_diff2.append(np.mean(np.abs(np.diff(lc))))
    mean_diff3 = np.load(local_directory + f'mean_diff2_07_07.npy')
    fig = plt.figure(figsize=(5, 5))
    plt.plot(mag, mean_diff, '.', c='C0', ms=3, label='QLP')
    plt.plot(mag2, mean_diff2, '^', c='C2', ms=2, label='eleanor PSF')
    plt.plot(mag1, mean_diff1, '^', c='C1', ms=2, label='eleanor PCA')
    plt.plot(mean_diff3[0], mean_diff3[1], 'D', c='r', ms=1.5, label='TGLC')
    plt.legend(loc=2)
    plt.xlabel('TESS magnitude')
    plt.ylabel(r'Mean Adjacent Difference ($\mathrm{e^-}$/ s)')
    plt.yscale('log')
    plt.savefig('/mnt/c/users/tehan/desktop/MAD.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_4():
    with open(f'/mnt/d/TESS_Sector_17/' + f'source/1-1/source_00_00.pkl', 'rb') as input_:
        source = pickle.load(input_)
    local_bg = np.load('/mnt/c/users/tehan/desktop/local_bg00_00.npy')
    plt.imshow(np.log10(source.flux[0]), cmap='bone', origin='lower')
    plt.scatter(local_bg[0], local_bg[1], s=3 * np.sqrt(abs(local_bg[2])), marker='.', c='C0', label='overestimate')
    plt.scatter(local_bg[0][np.where(local_bg[2] < 0)], local_bg[1][np.where(local_bg[2] < 0)],
                s=3 * np.sqrt(abs(local_bg[2][np.where(local_bg[2] < 0)])), marker='.', c='C1', label='underestimate')
    plt.legend(loc=4)
    plt.title('Local background eliminates vignetting')
    plt.xlim(-0.5, 149.5)
    plt.ylim(-0.5, 149.5)
    plt.xlabel('pixels')
    plt.ylabel('pixels')
    # plt.savefig('/mnt/c/users/tehan/desktop/bg_mod_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_5():
    ccd = '1-2'
    cut_x = 11
    cut_y = 11
    # local_directory = f'/mnt/d/TESS_Sector_17/'
    # with open(local_directory + f'source/{ccd}/source_{cut_x:02d}_{cut_y:02d}.pkl', 'rb') as input_:
    #     source = pickle.load(input_)
    with open(f'/mnt/c/users/tehan/desktop/7654/source_NGC 7654.pkl', 'rb') as input_:
        source = pickle.load(input_)
    # plt.imshow(source.flux[0], origin='lower', norm=colors.LogNorm())
    # plt.scatter(source.gaia['sector_17_x'][:100], source.gaia['sector_17_y'][:100], s=0.5, c='r')
    # plt.scatter(source.gaia['sector_17_x'][6], source.gaia['sector_17_y'][6], s=0.5, c='r')
    # plt.colorbar()
    # plt.show()
    contamination = np.load('/mnt/c/users/tehan/desktop/7654/contamination.npy')
    contamination_8 = np.load('/mnt/c/users/tehan/desktop/7654/contamination_8.npy')
    fig = plt.figure(constrained_layout=False, figsize=(11, 4))
    gs = fig.add_gridspec(1, 31)
    gs.update(wspace=1, hspace=0.1)

    # cmap = plt.get_cmap('cmr.fusion')  # MPL
    cmap = 'RdBu'
    ax1 = fig.add_subplot(gs[0, 0:10], projection=source.wcs)
    ax1.set_title('TESS FFI', pad=10)
    im1 = ax1.imshow(source.flux[0], origin='lower', cmap=cmap, vmin=-5000, vmax=5000)
    ax1.scatter(source.gaia['sector_17_x'][:100], source.gaia['sector_17_y'][:100], s=5, c='r')
    ax1.scatter(source.gaia['sector_17_x'][8], source.gaia['sector_17_y'][8], s=30, c='r', marker='*')
    ax1.coords['pos.eq.ra'].set_axislabel('Right Ascension', minpad=-1)
    ax1.coords['pos.eq.ra'].set_axislabel_position('l')
    ax1.coords['pos.eq.dec'].set_axislabel('Declination')
    ax1.coords['pos.eq.dec'].set_axislabel_position('b')
    ax1.coords.grid(color='k', ls='dotted')
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.tick_params(axis='y', labelleft=True)

    ax2 = fig.add_subplot(gs[0, 10:20], projection=source.wcs)
    ax2.set_title('Simulated background stars', pad=10)
    im2 = ax2.imshow(contamination_8, origin='lower', cmap=cmap, vmin=-5000, vmax=5000)
    ax2.scatter(source.gaia['sector_17_x'][:8], source.gaia['sector_17_y'][:8], s=5, c='r')
    ax2.scatter(source.gaia['sector_17_x'][9:100], source.gaia['sector_17_y'][9:100], s=5, c='r')
    # ax2.set_xticks([20, 25, 30, 35, 40])
    # ax2.set_yticks([20, 25, 30, 35, 40])

    ax2.coords['pos.eq.ra'].set_ticklabel_visible(False)
    ax2.coords['pos.eq.dec'].set_axislabel('Declination')
    ax2.coords['pos.eq.dec'].set_axislabel_position('b')
    ax2.coords.grid(color='k', ls='dotted')
    ax2.tick_params(axis='x', labelbottom=True)
    ax2.tick_params(axis='y', labelleft=True)

    ax3 = fig.add_subplot(gs[0, 20:30], projection=source.wcs)
    ax3.set_title('Decontaminated target star', pad=10)
    im3 = ax3.imshow(source.flux[0] - contamination_8, origin='lower', cmap=cmap, vmin=-5000, vmax=5000)
    ax3.scatter(source.gaia['sector_17_x'][0], source.gaia['sector_17_y'][0], s=5, c='r',
                label='background stars')
    ax3.scatter(source.gaia['sector_17_x'][8], source.gaia['sector_17_y'][8], s=30, c='r', marker='*',
                label='target star')
    ax3.coords['pos.eq.ra'].set_ticklabel_visible(False)
    ax3.coords['pos.eq.dec'].set_axislabel('Declination')
    ax3.coords['pos.eq.dec'].set_axislabel_position('b')
    ax3.coords.grid(color='k', ls='dotted')
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.tick_params(axis='y', labelleft=True)

    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    ax_cb = fig.colorbar(im3, cax=fig.add_subplot(gs[0, 30]), orientation='vertical',
                         boundaries=np.linspace(-1000, 5000, 1000),
                         ticks=[-1000, 0, 1000, 2000, 3000, 4000, 5000], aspect=50, shrink=0.7)
    ax_cb.ax.set_yticklabels(['-1', '0', '1', '2', '3', '4', '5'])
    ax_cb.ax.set_ylabel(r'TESS Flux ($\times 1000$ $\mathrm{e^-}$/ s) ')
    ax3.legend(loc=4, prop={'size': 8})
    plt.setp([ax1, ax2, ax3], xlim=(21.5, 36.5), ylim=(18.5, 33.5))
    plt.savefig('/mnt/c/users/tehan/desktop/remove_contamination.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    figure_5()

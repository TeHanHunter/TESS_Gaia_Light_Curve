import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten
from scipy import ndimage
from astropy.io import ascii
from astropy.io import fits
from tqdm import trange
from matplotlib.patches import ConnectionPatch
from tglc.target_lightcurve import epsf
from tglc.ffi_cut import ffi_cut
from tglc.quick_lc import tglc_lc
import matplotlib.patheffects as pe


def load_eleanor(ld='', tic=1, sector=1):
    eleanor_pca = np.load(ld + f'eleanor/TIC {tic}_{sector}_corr.npy')
    eleanor_psf = np.load(ld + f'eleanor/TIC {tic}_{sector}_psf.npy')
    eleanor_t = eleanor_pca[0]
    eleanor_pca_f = flatten(eleanor_t, eleanor_pca[1] / np.nanmedian(eleanor_pca[1]), window_length=1,
                            method='biweight', return_trend=False)
    try:
        eleanor_psf_f = flatten(eleanor_t, eleanor_psf[1] / np.nanmedian(eleanor_psf[1]), window_length=1,
                                method='biweight', return_trend=False)
    except:
        eleanor_psf_f = np.zeros(len(eleanor_t))
    if sector > 26:
        eleanor_t = np.mean(eleanor_t[:len(eleanor_t) // 3 * 3].reshape(-1, 3), axis=1)
        eleanor_pca_f = np.mean(eleanor_pca_f[:len(eleanor_pca_f) // 3 * 3].reshape(-1, 3), axis=1)
        eleanor_psf_f = np.mean(eleanor_psf_f[:len(eleanor_psf_f) // 3 * 3].reshape(-1, 3), axis=1)
    return eleanor_t, eleanor_pca_f, eleanor_psf_f


def load_qlp(ld='', tic=1, sector=1):
    qlp = fits.open(ld + f'HLSP/hlsp_qlp_tess_ffi_s{sector:04d}-{tic:016d}_tess_v01_llc/' +
                    f'hlsp_qlp_tess_ffi_s{sector:04d}-{tic:016d}_tess_v01_llc.fits', mode='denywrite')
    quality = qlp[1].data['QUALITY']
    index = np.where(quality == 0)
    qlp_t = qlp[1].data['TIME'][index]
    lc = qlp[1].data['KSPSAP_FLUX']
    qlp_f = flatten(qlp_t, lc[index] / np.nanmedian(lc[index]), window_length=1, method='biweight',
                    return_trend=False)
    if sector > 26:
        qlp_t = np.mean(qlp_t[:len(qlp_t) // 3 * 3].reshape(-1, 3), axis=1)
        qlp_f = np.mean(qlp_f[:len(qlp_f) // 3 * 3].reshape(-1, 3), axis=1)
    return qlp_t, qlp_f


def load_ztf(ld='', index=1):
    data = ascii.read(ld + f'ZTF/{index}_g.csv')
    data.remove_rows(np.where(data['catflags'] != 0))
    ztf_g_t = data['hjd'] - 2457000
    mag = data['mag']
    ztf_g_flux = 10 ** (- mag / 2.5)
    try:
        data = ascii.read(ld + f'ZTF/{index}_r.csv')
        data.remove_rows(np.where(data['catflags'] != 0))
        ztf_r_t = data['hjd'] - 2457000
        mag = data['mag']
        ztf_r_flux = 10 ** (- mag / 2.5)
    except:
        return ztf_g_t, ztf_g_flux
    return ztf_g_t, ztf_g_flux, ztf_r_t, ztf_r_flux


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

    fig = plt.figure(constrained_layout=False, figsize=(10, 6))
    gs = fig.add_gridspec(4, 14)
    gs.update(wspace=0.2, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0:5])
    ax2 = fig.add_subplot(gs[1, 0:5])
    # ax3 = fig.add_subplot(gs[2, 0:5])
    ax4 = fig.add_subplot(gs[2, 0:5])
    ax5 = fig.add_subplot(gs[3, 0:5])

    ax6 = fig.add_subplot(gs[0, 5:10])
    ax7 = fig.add_subplot(gs[1, 5:10])
    # ax8 = fig.add_subplot(gs[2, 5:10])
    ax9 = fig.add_subplot(gs[2, 5:10])
    ax10 = fig.add_subplot(gs[3, 5:10])

    ax11 = fig.add_subplot(gs[0, 11:])
    ax12 = fig.add_subplot(gs[1, 11:])
    # ax13 = fig.add_subplot(gs[2, 12:])

    ax1.plot(time, flatten_lc___ / np.median(flatten_lc___), '.k', ms=1)
    ax2.plot(time, flatten_lc__, '.k', ms=1)
    # ax3.plot(qlp_data['TIME'], qlp_data['KSPSAP_FLUX'], '.k', ms=1)
    ax4.plot(source.time, flatten_lc_, '.k', ms=1)
    ax5.plot(source.time, flatten_lc, '.k', ms=1)

    ax6.plot(time[0:t1_] % period / period, flatten_lc___[0:t1_] / np.median(flatten_lc___), '.k', ms=1)
    ax6.plot(time[t2_:t3_] % period / period, flatten_lc___[t2_:t3_] / np.median(flatten_lc___), '.k', ms=1)
    ax7.plot(time[0:t1_] % period / period, flatten_lc__[0:t1_], '.k', ms=1)
    ax7.plot(time[t2_:t3_] % period / period, flatten_lc__[t2_:t3_], '.k', ms=1)
    # ax8.plot(qlp_data['TIME'] % period, qlp_data['KSPSAP_FLUX'], '.k', ms=1)
    ax9.plot(source.time[0:t1] % period / period, flatten_lc_[0:t1], '.k', ms=1)
    ax9.plot(source.time[t2:t3] % period / period, flatten_lc_[t2:t3], '.k', ms=1)
    ax10.plot(source.time[0:t1] % period / period, flatten_lc[0:t1], '.k', ms=1)
    ax10.plot(source.time[t2:t3] % period / period, flatten_lc[t2:t3], '.k', ms=1)
    ax11.plot(time[0:t1_] % period / period, flatten_lc___[0:t1_] / np.median(flatten_lc___), '.k', ms=1, zorder=3)
    ax11.plot(time[t2_:t3_] % period / period, flatten_lc___[t2_:t3_] / np.median(flatten_lc___), '.k', ms=1, zorder=3)
    ax12.plot(time[0:t1_] % period / period, flatten_lc__[0:t1_], '.k', ms=1, zorder=3)
    ax12.plot(time[t2_:t3_] % period / period, flatten_lc__[t2_:t3_], '.k', ms=1, zorder=3, label='TESS')
    # ax13.plot(qlp_data['TIME'] % period/period, qlp_data['KSPSAP_FLUX'], '.k', ms=1, label='TESS', zorder=3)

    data = ascii.read(f'/mnt/d/Astro/Output of SEBIT/eb_candidate_new/ZTF/1251_g.csv')
    data.remove_rows(np.where(data['catflags'] != 0))
    tbjd = data['hjd'] - 2457000
    mag = data['mag']
    flux = 10 ** (- mag / 2.5)  # 3.208e-10 *
    ax6.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax7.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    # ax8.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax9.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax10.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax11.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    ax12.plot(tbjd % period / period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
    # ax13.plot(tbjd % period, flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')

    data = ascii.read(f'/mnt/d/Astro/Output of SEBIT/eb_candidate_new/ZTF/1251_r.csv')
    data.remove_rows(np.where(data['catflags'] != 0))
    tbjd = data['hjd'] - 2457000
    mag = data['mag']
    flux = 10 ** (- mag / 2.5)
    ax6.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax7.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    # ax8.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
    #             label='ZTF r-band')
    ax9.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                label='ZTF r-band')
    ax10.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    ax11.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    ax12.scatter(tbjd % period / period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                 label='ZTF r-band')
    # ax13.scatter(tbjd % period, flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
    #              label='ZTF r-band')

    ax1.set_title('eleanor PCA', loc='left')
    ax2.set_title('eleanor PSF', loc='left')
    # ax3.set_title('QLP KSPSAP', loc='left')
    ax4.set_title('Moffat PSF', loc='left')
    ax5.set_title('effective PSF', loc='left')
    ax1.set_xticklabels([])
    ax1.tick_params(axis="both", direction="in")
    ax2.set_xticklabels([])
    ax2.tick_params(axis="both", direction="in")
    # ax3.set_xticklabels([])
    # ax3.tick_params(axis="both", direction="in")
    ax4.set_xticklabels([])
    ax4.tick_params(axis="both", direction="in")
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax6.tick_params(axis="both", direction="in")
    ax7.set_xticklabels([])
    ax7.set_yticklabels([])
    ax7.tick_params(axis="both", direction="in")
    # ax8.set_xticklabels([])
    # ax8.set_yticklabels([])
    # ax8.tick_params(axis="both", direction="in")
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
    # ax12.set_xticklabels([])
    # ax13.set_ylabel('Normalized Flux')
    # ax13.tick_params(axis="both", direction="in")
    # ax13.yaxis.set_label_position("right")
    # ax13.yaxis.tick_right()

    ax1.set_ylabel('Normalized Flux')
    ax2.set_ylabel('Normalized Flux')
    # ax3.set_ylabel('Normalized Flux')
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
    ax10.set_xlabel('Phase')
    ax12.set_xlabel('Phase')

    ax1.set_ylim(0.65, 1.1)
    ax6.set_ylim(0.65, 1.1)
    ax2.set_ylim(0.65, 1.1)
    ax7.set_ylim(0.65, 1.1)
    # ax3.set_ylim(0.65, 1.1)
    # ax8.set_ylim(0.65, 1.1)
    ax4.set_ylim(0.65, 1.1)
    ax9.set_ylim(0.65, 1.1)
    ax5.set_ylim(0.65, 1.1)
    ax10.set_ylim(0.65, 1.1)
    ax11.set_ylim(0.993, 1.006)
    ax12.set_ylim(0.993, 1.006)
    # ax13.set_ylim(0.993, 1.006)

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
    # con1 = ConnectionPatch(xyA=(ax8.get_xlim()[1], 1.006), xyB=(ax13.get_xlim()[0], 1.006), coordsA="data",
    #                        coordsB="data", axesA=ax8,
    #                        axesB=ax13, color="k")
    # con2 = ConnectionPatch(xyA=(ax8.get_xlim()[1], 0.993), xyB=(ax13.get_xlim()[0], 0.993), coordsA="data",
    #                        coordsB="data", axesA=ax8,
    #                        axesB=ax13, color="k")
    # ax8.add_artist(con1)
    # ax8.add_artist(con2)
    ax12.legend(bbox_to_anchor=(1, -1.7), loc=1)
    plt.savefig('/mnt/c/users/tehan/desktop/light_curve_comparison_1251.png', bbox_inches='tight', dpi=300)
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
            data = ascii.read(f'/mnt/d/Astro/Output of SEBIT/eb_candidate_new/ZTF/{index[i]}_r.csv')
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


def eleanor(tic, local_directory=''):
    sector = 17
    star = eleanor.Source(tic=tic, sector=int(sector))
    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=True, do_pca=True)
    q = data.quality == 0
    np.save(f'{local_directory}TIC{tic}_{sector}_corr.npy', np.array([data.time[q], data.corr_flux[q]]))


def figure_3():
    target = '21.0607 34.4578'
    # target = 'TOI 519'
    local_directory = f'/home/tehan/Documents/tglc/{target}/'
    # os.makedirs(local_directory, exist_ok=True)
    # tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=20,
    #                 get_all_lc=True, first_sector_only=False, sector=17)
    files = glob(f'{local_directory}lc/*.fits')
    print(len(files))
    tic = np.zeros((len(files), 2))
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            try:
                tic[i] = [int(hdul[0].header['TICID']), hdul[0].header['TESSMAG']]
            except:
                pass
    np.save(f'/home/tehan/Documents/tglc/{target}/tic.npy', tic)
    # tic = np.load(f'/home/tehan/Documents/tglc/{target}/tic.npy')

    # 1-1/07_07 # 21.0607 34.4578 # 90
    noise_2015 = ascii.read('/home/tehan/Documents/tglc/prior_mad/noisemodel.dat')
    qlp_file = glob(f'{local_directory}QLP/HLSP/*/*.fits')
    # ele_file = glob(f'{local_directory}lc_eleanor_psf/*.npy')

    mag_qlp = []
    median_diff_qlp = []
    for i in trange(len(qlp_file)):
        with fits.open(qlp_file[i], mode='denywrite') as hdul:
            quality = hdul[1].data['QUALITY']
            lc = hdul[1].data['KSPSAP_FLUX']
            mag_ = hdul[0].header['TESSMAG']
            scale = 1.5e4 * 10 ** ((10 - mag_) / 2.5)
            index = np.where(quality == 0)
            mag_qlp.append(mag_)
            median_diff_qlp.append(np.nanmedian(np.abs(np.diff(lc[index]))))
    mag_ele = []
    median_diff_ele = []
    # diff = []
    # for i in trange(len(ele_file)):
    #     lc = np.load(ele_file[i])[1]
    #     tic_id = int(os.path.basename(ele_file[i]).split(' ')[-1][:-10])
    #     mag_ = tic[np.where(tic[:, 0] == tic_id)[0][0], 1]
    #     mag_ele.append(mag_)
    #     scale = 1.5e4 * 10 ** ((10 - mag_) / 2.5)
    #     median_diff_ele.append(np.nanmedian(np.abs(np.diff(lc))) / scale)
    #     diff.append((scale - np.median(lc))/scale)
    #
    # plt.plot(mag_ele, diff, '.')
    # plt.ylim(-50, 50)
    # plt.show()
    tglc_mag = np.load(f'/home/tehan/Documents/tglc/{target}/mag.npy')
    mag_both = np.load(f'/home/tehan/Documents/tglc/{target}/mag_both.npy')
    MAD_aper = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_aper.npy')
    # AAD_aper = np.load(f'/home/tehan/Documents/tglc/{target}/AAD_aper.npy')
    MAD_psf = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_psf.npy')
    # AAD_psf = np.load(f'/home/tehan/Documents/tglc/{target}/AAD_psf.npy')
    MAD_both = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_both.npy')

    aper_precision = 1.48 * MAD_aper / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - tglc_mag) / 2.5))
    psf_precision = 1.48 * MAD_psf / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - tglc_mag) / 2.5))
    aver_precision = 1.48 * MAD_both / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - mag_both) / 2.5))
    qlp_precision = 1.48 * np.array(median_diff_qlp) / np.sqrt(2)
    ele_precision = 1.48 * np.array(median_diff_ele) / np.sqrt(2)

    fig, ax = plt.subplots(2, 2, sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.1, wspace=0.05),
                           figsize=(10, 5))
    ax[0, 0].plot(mag_both, aver_precision, 'D', c='tomato', ms=1, label='TGLC Weighted', alpha=0.9)
    # ax[0].plot(mag_ele, ele_precision, '^', c='C0', ms=1.5, label='eleanor PSF', alpha=0.4)
    ax[0, 0].plot(mag_qlp, qlp_precision, '^', c='teal', ms=1.5, label='QLP', alpha=0.9)

    # ax[0].plot(tglc_mag, aper_precision, 'D', c='k', ms=1, label='TGLC PSF', alpha=0.8)
    ax[0, 0].plot(noise_2015['col1'], noise_2015['col2'], c='k', ms=1.5, label='Sullivan (2015)', alpha=1)
    # # ax[0].plot(mean_diff_aper[0], aper_precision, 'D', c='r', ms=1, label='TGLC Aper', alpha=0.8)
    ax[0, 0].hlines(y=.1, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')
    ax[0, 0].hlines(y=.01, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')

    leg = ax[0, 0].legend(loc=4, markerscale=4, fontsize=8)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax[0, 0].set_ylabel(r'Estimated Photometric Precision')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylim(1e-4, 1)
    ax[0, 0].set_title('Sparse Field')

    psf_ratio = psf_precision / aver_precision
    psf_tglc_mag = tglc_mag[np.invert(np.isnan(psf_ratio))]
    psf_ratio = psf_ratio[np.invert(np.isnan(psf_ratio))]
    psf_runningmed = ndimage.median_filter(psf_ratio, size=250, mode='nearest')

    aper_ratio = aper_precision / aver_precision
    aper_tglc_mag = tglc_mag[np.invert(np.isnan(aper_ratio))]
    aper_ratio = aper_ratio[np.invert(np.isnan(aper_ratio))]
    aper_runningmed = ndimage.median_filter(aper_ratio, size=300, mode='nearest')

    ax[1, 0].plot(psf_tglc_mag[:-100], psf_ratio[:-100], '.', c='C1', ms=6, alpha=0.15,
                  label='TGLC PSF Precision/TGLC Weighted Precision')
    ax[1, 0].plot(aper_tglc_mag[:-100], aper_ratio[:-100], '.', c='C0', ms=6, alpha=0.15,
                  label='TGLC Aperture Precision/TGLC Weighted Precision')
    ax[1, 0].plot(psf_tglc_mag[:-100], psf_runningmed[:-100], c='C1', label='Median', lw=1.5,
                  path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    ax[1, 0].plot(aper_tglc_mag[:-100], aper_runningmed[:-100], c='C0', label='Median', lw=1.5,
                  path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

    ax[1, 0].hlines(y=1, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')
    # ax[1].set_yscale('log')
    ax[1, 0].set_ylim(0.5, 1.5)
    ax[1, 0].tick_params(axis='y', which='minor', labelleft=False)
    ax[1, 0].set_yticks(ticks=[0.5, 1, 1.5], labels=['0.5', '1', '1.5'])
    # ax[1].set_title('Photometric Precision Ratio')
    ax[1, 0].set_xlabel('TESS magnitude')
    ax[1, 0].set_ylabel('Precision Ratio')
    leg = ax[1, 0].legend(loc=4, markerscale=1, ncol=2, columnspacing=1, fontsize=7.2)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlim(7, 20.5)

    ##############################
    # target = '21.0607 34.4578'
    target = 'TOI 519'
    local_directory = f'/home/tehan/Documents/tglc/{target}/'
    # os.makedirs(local_directory, exist_ok=True)
    # tglc_lc(target=target, local_directory=local_directory, size=90, save_aper=False, limit_mag=20,
    #                 get_all_lc=True, first_sector_only=False, sector=17)
    files = glob(f'{local_directory}lc/*.fits')

    tic = np.zeros((len(files), 2))
    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            try:
                tic[i] = [int(hdul[0].header['TICID']), hdul[0].header['TESSMAG']]
            except:
                pass
    np.save(f'/home/tehan/Documents/tglc/{target}/tic.npy', tic)
    # tic = np.load(f'/home/tehan/Documents/tglc/{target}/tic.npy')

    # 1-1/07_07 # 21.0607 34.4578 # 90
    noise_2015 = ascii.read('/home/tehan/Documents/tglc/prior_mad/noisemodel.dat')
    qlp_file = glob(f'{local_directory}QLP/HLSP/*/*.fits')
    # ele_file = glob(f'{local_directory}lc_eleanor_psf/*.npy')

    mag_qlp = []
    median_diff_qlp = []
    for i in trange(len(qlp_file)):
        with fits.open(qlp_file[i], mode='denywrite') as hdul:
            quality = hdul[1].data['QUALITY']
            lc = hdul[1].data['KSPSAP_FLUX']
            mag_ = hdul[0].header['TESSMAG']
            scale = 1.5e4 * 10 ** ((10 - mag_) / 2.5)
            index = np.where(quality == 0)
            mag_qlp.append(mag_)
            median_diff_qlp.append(np.nanmedian(np.abs(np.diff(lc[index]))))
    tglc_mag = np.load(f'/home/tehan/Documents/tglc/{target}/mag.npy')
    mag_both = np.load(f'/home/tehan/Documents/tglc/{target}/mag_both.npy')
    MAD_aper = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_aper.npy')
    # AAD_aper = np.load(f'/home/tehan/Documents/tglc/{target}/AAD_aper.npy')
    MAD_psf = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_psf.npy')
    # AAD_psf = np.load(f'/home/tehan/Documents/tglc/{target}/AAD_psf.npy')
    MAD_both = np.load(f'/home/tehan/Documents/tglc/{target}/MAD_both.npy')

    aper_precision = 1.48 * MAD_aper / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - tglc_mag) / 2.5))
    psf_precision = 1.48 * MAD_psf / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - tglc_mag) / 2.5))
    aver_precision = 1.48 * MAD_both / (np.sqrt(2) * 1.5e4 * 10 ** ((10 - mag_both) / 2.5))
    qlp_precision = 1.48 * np.array(median_diff_qlp) / np.sqrt(2)
    # ele_precision = 1.48 * np.array(median_diff_ele) / np.sqrt(2)

    # fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.1), figsize=(5, 7))
    ax[0, 1].plot(mag_both, aver_precision, 'D', c='tomato', ms=1, label='TGLC Weighted', alpha=0.9)
    # ax[0].plot(mag_ele, ele_precision, '^', c='C0', ms=1.5, label='eleanor PSF', alpha=0.4)
    ax[0, 1].plot(mag_qlp, qlp_precision, '^', c='teal', ms=1.5, label='QLP', alpha=0.9)

    # ax[0].plot(tglc_mag, aper_precision, 'D', c='k', ms=1, label='TGLC PSF', alpha=0.8)
    ax[0, 1].plot(noise_2015['col1'], noise_2015['col2'], c='k', ms=1.5, label='Sullivan (2015)', alpha=1)
    # # ax[0].plot(mean_diff_aper[0], aper_precision, 'D', c='r', ms=1, label='TGLC Aper', alpha=0.8)
    ax[0, 1].hlines(y=.1, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')
    ax[0, 1].hlines(y=.01, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')

    leg = ax[0, 1].legend(loc=4, markerscale=4, fontsize=8)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # ax[0, 1].set_ylabel(r'Estimated Photometric Precision')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylim(1e-4, 1)
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_title('Crowded Field')

    psf_ratio = psf_precision / aver_precision
    psf_tglc_mag = tglc_mag[np.invert(np.isnan(psf_ratio))]
    psf_ratio = psf_ratio[np.invert(np.isnan(psf_ratio))]
    psf_runningmed = ndimage.median_filter(psf_ratio, size=250, mode='nearest')

    aper_ratio = aper_precision / aver_precision
    aper_tglc_mag = tglc_mag[np.invert(np.isnan(aper_ratio))]
    aper_ratio = aper_ratio[np.invert(np.isnan(aper_ratio))]
    aper_runningmed = ndimage.median_filter(aper_ratio, size=300, mode='nearest')

    ax[1, 1].plot(psf_tglc_mag[:-100], psf_ratio[:-100], '.', c='C1', ms=6, alpha=0.15,
                  label='TGLC PSF Precision/TGLC Weighted Precision')
    ax[1, 1].plot(aper_tglc_mag[:-100], aper_ratio[:-100], '.', c='C0', ms=6, alpha=0.15,
                  label='TGLC Aperture Precision/TGLC Weighted Precision')
    ax[1, 1].plot(psf_tglc_mag[:-100], psf_runningmed[:-100], c='C1', label='Median', lw=1.5,
                  path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    ax[1, 1].plot(aper_tglc_mag[:-100], aper_runningmed[:-100], c='C0', label='Median', lw=1.5,
                  path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

    ax[1, 1].hlines(y=1, xmin=8, xmax=np.max(tglc_mag), colors='k', linestyles='dotted')
    # ax[1].set_yscale('log')
    ax[1, 1].set_ylim(0.5, 1.5)
    ax[1, 1].tick_params(axis='y', which='minor', labelleft=False)
    ax[1, 1].set_yticks(ticks=[0.5, 1, 1.5], labels=['0.5', '1', '1.5'])
    # ax[1].set_title('Photometric Precision Ratio')
    ax[1, 1].set_xlabel('TESS magnitude')
    ax[1, 1].set_yticklabels([])
    leg = ax[1, 1].legend(loc=4, markerscale=1, ncol=2, columnspacing=1, fontsize=7.2)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.xlim(7, 20.5)

    # plt.savefig(f'{local_directory}MAD.png', bbox_inches='tight', dpi=300)
    plt.show()
    # point-to-point scatter


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
    target = 'NGC 7654'
    local_directory = f'/mnt/c/users/tehan/desktop/7654/{target}/'
    # local_directory = f'/home/tehan/data/{target}/'
    os.makedirs(local_directory + 'source/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    # source = ffi_cut(target=target, size=50, local_directory=local_directory, sector=18)
    with open(f'{local_directory}source/source_NGC 7654_sector_18.pkl', 'rb') as input_:
        source = pickle.load(input_)
    # epsf(source, factor=2, sector=source.sector, target=target, power=1.4, local_directory=local_directory,
    #      name=None, limit_mag=15, save_aper=False)
    contamination_8 = np.load('/mnt/c/users/tehan/desktop/7654/contamination_8_.npy').reshape(50, 50)
    fig = plt.figure(constrained_layout=False, figsize=(11, 4))
    gs = fig.add_gridspec(1, 31)
    gs.update(wspace=1, hspace=0.1)

    # cmap = plt.get_cmap('cmr.fusion')  # MPL
    cmap = 'RdBu'
    ax1 = fig.add_subplot(gs[0, 0:10], projection=source.wcs, slices=('y', 'x'))
    ax1.set_title('TESS FFI', pad=10)
    im1 = ax1.imshow(source.flux[0].transpose(), origin='lower', cmap=cmap, vmin=-5000, vmax=5000)
    ax1.scatter(source.gaia['sector_18_y'][:100], source.gaia['sector_18_x'][:100], s=5, c='r',
                label='background stars')
    ax1.scatter(source.gaia['sector_18_y'][8], source.gaia['sector_18_x'][8], s=30, c='r', marker='*',
                label='target star')
    ax1.coords['pos.eq.ra'].set_axislabel('Right Ascension')
    ax1.coords['pos.eq.ra'].set_axislabel_position('b')
    ax1.coords['pos.eq.dec'].set_axislabel('Declination')
    ax1.coords['pos.eq.dec'].set_axislabel_position('l')
    ax1.coords.grid(color='k', ls='dotted')
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.tick_params(axis='y', labelleft=True)

    ax2 = fig.add_subplot(gs[0, 10:20], projection=source.wcs, slices=('y', 'x'))
    ax2.set_title('Simulated background stars', pad=10)
    im2 = ax2.imshow(contamination_8.transpose(), origin='lower', cmap=cmap, vmin=-5000, vmax=5000)
    ax2.scatter(source.gaia['sector_18_y'][:8], source.gaia['sector_18_x'][:8], s=5, c='r')
    ax2.scatter(source.gaia['sector_18_y'][9:100], source.gaia['sector_18_x'][9:100], s=5, c='r')
    # ax2.set_xticks([20, 25, 30, 35, 40])
    # ax2.set_yticks([20, 25, 30, 35, 40])

    ax2.coords['pos.eq.dec'].set_ticklabel_visible(False)
    ax2.coords['pos.eq.ra'].set_axislabel('Right Ascension')
    ax2.coords['pos.eq.ra'].set_axislabel_position('b')
    ax2.coords.grid(color='k', ls='dotted')
    ax2.tick_params(axis='x', labelbottom=True)
    ax2.tick_params(axis='y', labelleft=True)

    ax3 = fig.add_subplot(gs[0, 20:30], projection=source.wcs, slices=('y', 'x'))
    ax3.set_title('Decontaminated target star', pad=10)
    im3 = ax3.imshow(source.flux[0].transpose() - contamination_8.transpose(), origin='lower', cmap=cmap, vmin=-5000,
                     vmax=5000)
    ax3.scatter(source.gaia['sector_18_y'][8], source.gaia['sector_18_x'][8], s=30, c='r', marker='*')
    ax3.coords['pos.eq.dec'].set_ticklabel_visible(False)
    ax3.coords['pos.eq.ra'].set_axislabel('Right Ascension')
    ax3.coords['pos.eq.ra'].set_axislabel_position('b')
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
    ax1.legend(loc=2, prop={'size': 8})
    plt.setp([ax1, ax2, ax3], xlim=(18.5, 33.5), ylim=(21.5, 36.5))
    # plt.savefig('/mnt/c/users/tehan/desktop/remove_contamination_.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def figure_6(mode='psf'):
    type = f'cal_{mode}_flux'
    # local_directory = '/home/tehan/data/exoplanets/'
    local_directory = '/home/tehan/data/known_exoplanet/'
    os.makedirs(local_directory + f'transits/', exist_ok=True)
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    data = ascii.read(local_directory + 'PS_2022.04.17_18.23.57_.csv')
    hosts = list(data['tic_id'])
    # for i in range(len(hosts)):
    #     target = hosts[i]  # Target identifier or coordinates TOI-3714
    #     print(target)
    #     size = 90  # int, suggests big cuts
    #     source = ffi_cut(target=target, size=size, local_directory=local_directory)
    #     for j in range(len(source.sector_table)):
    #         source.select_sector(sector=source.sector_table['sector'][j])
    #         epsf(source, factor=2, sector=source.sector, target=target, local_directory=local_directory,
    #              name='Gaia DR3 '+data['gaia_id'][i].split()[-1], power=1.5, save_aper=True)
    # plt.imshow(source.flux[0])
    # plt.scatter(source.gaia[f'sector_{source.sector_table["sector"][j]}_x'][:100],
    #             source.gaia[f'sector_{source.sector_table["sector"][j]}_y'][:100], c='r', s=5)
    # plt.xlim(-0.5, 89.5)
    # plt.ylim(-0.5, 89.5)
    # plt.title(f'{target}_sector_{source.sector_table["sector"][j]}')
    # plt.show()

    fig = plt.figure(constrained_layout=False, figsize=(10, 10))
    gs = fig.add_gridspec(5, 12)
    gs.update(wspace=0.2, hspace=0.4)
    ###########################################
    index = np.where(data['pl_name'] == 'TOI-674 b')
    # period = float(data['pl_orbper'][index])
    period = 1.977165
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0009*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_09 = hdul[1].data['time'][q]
        f_09 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0010*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_10 = hdul[1].data['time'][q]
        f_10 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0036*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_36 = hdul[1].data['time'][q]
        f_36 = hdul[1].data[type][q]
        t_36 = np.mean(t_36[:len(t_36) // 3 * 3].reshape(-1, 3), axis=1)
        f_36 = np.mean(f_36[:len(f_36) // 3 * 3].reshape(-1, 3), axis=1)
    ax1_1 = fig.add_subplot(gs[0, :3])
    ax1_2 = fig.add_subplot(gs[0, 3:6])
    ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax1_4 = fig.add_subplot(gs[0, 9:12])

    ax1_1.plot(t_09, f_09, '.', c='k', markersize=1)
    ax1_2.plot(t_10, f_10, '.', c='k', markersize=1)
    ax1_3.plot(t_36, f_36, '.', c='k', markersize=1)

    ax1_4.plot(t_09 % period / period - phase_fold_mid, f_09, '.', c='C0', markersize=2, label='9')
    ax1_4.plot(t_10 % period / period - phase_fold_mid, f_10, '.', c='C1', markersize=2, label='10')
    ax1_4.plot(t_36 % period / period - phase_fold_mid, f_36, '.', c='C3', markersize=2, label='36')
    ax1_4.legend(loc=3, fontsize=6, markerscale=1)
    # split
    ax1_1.spines['right'].set_visible(False)
    ax1_2.spines['left'].set_visible(False)
    ax1_2.spines['right'].set_visible(False)
    ax1_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1_1.plot([1, 1], [0, 1], transform=ax1_1.transAxes, **kwargs)
    ax1_2.plot([0, 0], [0, 1], transform=ax1_2.transAxes, **kwargs)
    ax1_2.plot([1, 1], [0, 1], transform=ax1_2.transAxes, **kwargs)
    ax1_3.plot([0, 0], [0, 1], transform=ax1_3.transAxes, **kwargs)
    ax1_2.set_yticklabels([])
    ax1_2.tick_params(axis='y', left=False)
    ax1_3.set_yticklabels([])
    ax1_3.tick_params(axis='y', left=False)
    ax1_4.set_yticklabels([])
    # ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax1_1.set_ylim(0.975, 1.01)
    ax1_2.set_ylim(0.975, 1.01)
    ax1_3.set_ylim(0.975, 1.01)
    ax1_4.set_ylim(0.975, 1.01)
    ax1_4.set_xlim(- 0.03, 0.03)

    ax1_1.set_title('Sector 9')
    ax1_2.set_title('Sector 10')
    ax1_3.set_title('Sector 36')
    ax1_4.set_title('TOI-674 b', {'fontweight': 'semibold'})
    ax1_1.set_ylabel('Normalized Flux')
    # ax1_1.set_xlabel('Time (TBJD)')
    # ax1_2.set_xlabel('Time (TBJD)')
    # ax1_3.set_xlabel('Time (TBJD)')
    # ax1_4.set_xlabel('Phase')
    # ax1_4.text(0.98, 0.1, 'Aper', horizontalalignment='right', transform=ax1_4.transAxes)

    ###########################################
    index = np.where(data['pl_name'] == 'LHS 3844 b')
    period = float(data['pl_orbper'][index])
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6385548541499112448-s0027*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_27 = hdul[1].data['time'][q]
        f_27 = hdul[1].data[type][q]
        t_27 = np.mean(t_27[:len(t_27) // 3 * 3].reshape(-1, 3), axis=1)
        f_27 = np.mean(f_27[:len(f_27) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6385548541499112448-s0028*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_28 = hdul[1].data['time'][q]
        f_28 = hdul[1].data[type][q]
        t_28 = np.mean(t_28[:len(t_28) // 3 * 3].reshape(-1, 3), axis=1)
        f_28 = np.mean(f_28[:len(f_28) // 3 * 3].reshape(-1, 3), axis=1)
    ax2_1 = fig.add_subplot(gs[1, :3])
    ax2_2 = fig.add_subplot(gs[1, 3:6])
    ax2_4 = fig.add_subplot(gs[1, 9:12])

    ax2_1.plot(t_27, f_27, '.', c='k', markersize=1)
    ax2_2.plot(t_28, f_28, '.', c='k', markersize=1)
    ax2_4.plot(t_27 % period / period - phase_fold_mid, f_27, '.', c='C0', markersize=2, label='27')
    ax2_4.plot(t_28 % period / period - phase_fold_mid, f_28, '.', c='C1', markersize=2, label='28')
    ax2_4.legend(loc=3, fontsize=6, markerscale=1)
    # split
    ax2_1.spines['right'].set_visible(False)
    ax2_2.spines['left'].set_visible(False)

    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax2_1.plot([1, 1], [0, 1], transform=ax2_1.transAxes, **kwargs)
    ax2_2.plot([0, 0], [0, 1], transform=ax2_2.transAxes, **kwargs)
    ax2_2.set_yticklabels([])
    ax2_2.tick_params(axis='y', left=False)
    ax2_4.set_yticklabels([])
    # ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax2_1.set_ylim(0.988, 1.007)
    ax2_2.set_ylim(0.988, 1.007)
    ax2_4.set_ylim(0.988, 1.007)
    ax2_4.set_xlim(- 0.1, 0.1)

    ax2_1.set_title('Sector 27')
    ax2_2.set_title('Sector 28')
    ax2_4.set_title('LHS 3844 b', {'fontweight': 'semibold'})
    ax2_1.set_ylabel('Normalized Flux')
    # ax2_1.set_xlabel('Time (TBJD)')
    # ax2_2.set_xlabel('Time (TBJD)')
    # ax2_4.set_xlabel('Phase')
    # ax2_4.text(0.98, 0.1, 'PSF', horizontalalignment='right', transform=ax2_4.transAxes)

    ###########################################
    index = np.where(data['pl_name'] == 'TOI-530 b')
    period = 6.387583
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0006*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_06 = hdul[1].data['time'][q]
        f_06 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0044*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_44 = hdul[1].data['time'][q]
        f_44 = hdul[1].data[type][q]
        t_44 = np.mean(t_44[:len(t_44) // 3 * 3].reshape(-1, 3), axis=1)
        f_44 = np.mean(f_44[:len(f_44) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0045*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_45 = hdul[1].data['time'][q]
        f_45 = hdul[1].data[type][q]
        t_45 = np.mean(t_45[:len(t_45) // 3 * 3].reshape(-1, 3), axis=1)
        f_45 = np.mean(f_45[:len(f_45) // 3 * 3].reshape(-1, 3), axis=1)
    ax3_1 = fig.add_subplot(gs[2, :3])
    ax3_2 = fig.add_subplot(gs[2, 3:6])
    ax3_3 = fig.add_subplot(gs[2, 6:9])
    ax3_4 = fig.add_subplot(gs[2, 9:12])

    ax3_1.plot(t_06, f_06, '.', c='k', markersize=1)
    ax3_2.plot(t_44, f_44, '.', c='k', markersize=1)
    ax3_3.plot(t_45, f_45, '.', c='k', markersize=1)

    ax3_4.plot(t_06 % period / period - phase_fold_mid, f_06, '.', c='C0', markersize=2, label='6')
    ax3_4.plot(t_44 % period / period - phase_fold_mid, f_44, '.', c='C1', markersize=2, label='44')
    ax3_4.plot(t_45 % period / period - phase_fold_mid, f_45, '.', c='C3', markersize=2, label='45')
    ax3_4.legend(loc=3, fontsize=6, markerscale=1)
    # split
    ax3_1.spines['right'].set_visible(False)
    ax3_2.spines['left'].set_visible(False)
    ax3_2.spines['right'].set_visible(False)
    ax3_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax3_1.plot([1, 1], [0, 1], transform=ax3_1.transAxes, **kwargs)
    ax3_2.plot([0, 0], [0, 1], transform=ax3_2.transAxes, **kwargs)
    ax3_2.plot([1, 1], [0, 1], transform=ax3_2.transAxes, **kwargs)
    ax3_3.plot([0, 0], [0, 1], transform=ax3_3.transAxes, **kwargs)
    ax3_2.set_yticklabels([])
    ax3_2.tick_params(axis='y', left=False)
    ax3_3.set_yticklabels([])
    ax3_3.tick_params(axis='y', left=False)
    ax3_4.set_yticklabels([])
    ax3_1.set_xticks([1470, 1480, 1490])
    ax3_1.set_xticklabels([1470, 1480, None])
    # ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax3_1.set_ylim(0.95, 1.03)
    ax3_2.set_ylim(0.95, 1.03)
    ax3_3.set_ylim(0.95, 1.03)
    ax3_4.set_ylim(0.95, 1.03)
    ax3_4.set_xlim(- 0.03, 0.03)

    ax3_1.set_title('Sector 6')
    ax3_2.set_title('Sector 44')
    ax3_3.set_title('Sector 45')
    ax3_4.set_title('TOI-530 b', {'fontweight': 'semibold'})
    ax3_1.set_ylabel('Normalized Flux')
    # ax3_1.set_xlabel('Time (TBJD)')
    # ax3_2.set_xlabel('Time (TBJD)')
    # ax3_3.set_xlabel('Time (TBJD)')
    # ax3_4.set_xlabel('Phase')
    # ax3_4.text(0.98, 0.1, 'Aper', horizontalalignment='right', transform=ax3_4.transAxes)

    ###########################################
    index = np.where(data['pl_name'] == 'TOI-2406 b')
    period = 3.076676
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0003*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_03 = hdul[1].data['time'][q]
        f_03 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0042*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_42 = hdul[1].data['time'][q]
        f_42 = hdul[1].data[type][q]
        t_42 = np.mean(t_42[:len(t_42) // 3 * 3].reshape(-1, 3), axis=1)
        f_42 = np.mean(f_42[:len(f_42) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0043*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_43 = hdul[1].data['time'][q]
        f_43 = hdul[1].data[type][q]
        t_43 = np.mean(t_43[:len(t_43) // 3 * 3].reshape(-1, 3), axis=1)
        f_43 = np.mean(f_43[:len(f_43) // 3 * 3].reshape(-1, 3), axis=1)
    ax4_1 = fig.add_subplot(gs[3, :3])
    ax4_2 = fig.add_subplot(gs[3, 3:6])
    ax4_3 = fig.add_subplot(gs[3, 6:9])
    ax4_4 = fig.add_subplot(gs[3, 9:12])

    ax4_1.plot(t_03, f_03, '.', c='k', markersize=1)
    ax4_2.plot(t_42, f_42, '.', c='k', markersize=1)
    ax4_3.plot(t_43, f_43, '.', c='k', markersize=1)

    ax4_4.plot(t_03 % period / period - phase_fold_mid, f_03, '.', c='C0', markersize=2, label='3')
    ax4_4.plot(t_42 % period / period - phase_fold_mid, f_42, '.', c='C1', markersize=2, label='42')
    ax4_4.plot(t_43 % period / period - phase_fold_mid, f_43, '.', c='C3', markersize=2, label='43')
    ax4_4.legend(loc=3, fontsize=6, markerscale=1)
    # split
    ax4_1.spines['right'].set_visible(False)
    ax4_2.spines['left'].set_visible(False)
    ax4_2.spines['right'].set_visible(False)
    ax4_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax4_1.plot([1, 1], [0, 1], transform=ax4_1.transAxes, **kwargs)
    ax4_2.plot([0, 0], [0, 1], transform=ax4_2.transAxes, **kwargs)
    ax4_2.plot([1, 1], [0, 1], transform=ax4_2.transAxes, **kwargs)
    ax4_3.plot([0, 0], [0, 1], transform=ax4_3.transAxes, **kwargs)
    ax4_2.set_yticklabels([])
    ax4_2.tick_params(axis='y', left=False)
    ax4_3.set_yticklabels([])
    ax4_3.tick_params(axis='y', left=False)
    ax4_4.set_yticklabels([])
    # ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax4_1.set_ylim(0.945, 1.04)
    ax4_2.set_ylim(0.945, 1.04)
    ax4_3.set_ylim(0.945, 1.04)
    ax4_4.set_ylim(0.945, 1.04)
    ax4_4.set_xlim(- 0.04, 0.04)

    ax4_1.set_title('Sector 3')
    ax4_2.set_title('Sector 42')
    ax4_3.set_title('Sector 43')
    ax4_4.set_title('TOI-2406 b', {'fontweight': 'semibold'})
    ax4_1.set_ylabel('Normalized Flux')
    # ax4_1.set_xlabel('Time (TBJD)')
    # ax4_2.set_xlabel('Time (TBJD)')
    # ax4_3.set_xlabel('Time (TBJD)')
    # ax4_4.set_xlabel('Phase')
    # ax4_4.text(0.98, 0.1, 'PSF', horizontalalignment='right', transform=ax4_4.transAxes)

    ###########################################
    index = np.where(data['pl_name'] == 'TOI-519 b')
    period = 1.265232
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0007*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_07 = hdul[1].data['time'][q]
        f_07 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0008*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_08 = hdul[1].data['time'][q]
        f_08 = hdul[1].data[type][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0034*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_34 = hdul[1].data['time'][q]
        f_34 = hdul[1].data[type][q]
        t_34 = np.mean(t_34[:len(t_34) // 3 * 3].reshape(-1, 3), axis=1)
        f_34 = np.mean(f_34[:len(f_34) // 3 * 3].reshape(-1, 3), axis=1)
    ax5_1 = fig.add_subplot(gs[4, :3])
    ax5_2 = fig.add_subplot(gs[4, 3:6])
    ax5_3 = fig.add_subplot(gs[4, 6:9])
    ax5_4 = fig.add_subplot(gs[4, 9:12])

    ax5_1.plot(t_07, f_07, '.', c='k', markersize=1)
    ax5_2.plot(t_08, f_08, '.', c='k', markersize=1)
    ax5_3.plot(t_34, f_34, '.', c='k', markersize=1)

    ax5_4.plot(t_07 % period / period - phase_fold_mid, f_07, '.', c='C0', markersize=2, label='7')
    ax5_4.plot(t_08 % period / period - phase_fold_mid, f_08, '.', c='C1', markersize=2, label='8')
    ax5_4.plot(t_34 % period / period - phase_fold_mid, f_34, '.', c='C3', markersize=2, label='34')
    ax5_4.legend(loc=3, fontsize=6, markerscale=1)
    # split
    ax5_1.spines['right'].set_visible(False)
    ax5_2.spines['left'].set_visible(False)
    ax5_2.spines['right'].set_visible(False)
    ax5_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax5_1.plot([1, 1], [0, 1], transform=ax5_1.transAxes, **kwargs)
    ax5_2.plot([0, 0], [0, 1], transform=ax5_2.transAxes, **kwargs)
    ax5_2.plot([1, 1], [0, 1], transform=ax5_2.transAxes, **kwargs)
    ax5_3.plot([0, 0], [0, 1], transform=ax5_3.transAxes, **kwargs)
    ax5_2.set_yticklabels([])
    ax5_2.tick_params(axis='y', left=False)
    ax5_3.set_yticklabels([])
    ax5_3.tick_params(axis='y', left=False)
    ax5_4.set_yticklabels([])
    ax5_4.set_xticks([-0.04, 0, 0.04])
    ax5_4.set_xticklabels([f'\N{MINUS SIGN}0.04', '0.00', '0.04'])
    # ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax5_1.set_ylim(0.83, 1.05)
    ax5_2.set_ylim(0.83, 1.05)
    ax5_3.set_ylim(0.83, 1.05)
    ax5_4.set_ylim(0.83, 1.05)
    ax5_4.set_xlim(- 0.05, 0.05)

    ax5_1.set_title('Sector 7')
    ax5_2.set_title('Sector 8')
    ax5_3.set_title('Sector 34')
    ax5_4.set_title('TOI-519 b', {'fontweight': 'semibold'})
    ax5_1.set_ylabel('Normalized Flux')
    ax5_1.set_xlabel('Time (TBJD)')
    ax5_2.set_xlabel('Time (TBJD)')
    ax5_3.set_xlabel('Time (TBJD)')
    ax5_4.set_xlabel('Phase')
    # ax5_4.text(0.98, 0.1, 'Aper', horizontalalignment='right', transform=ax5_4.transAxes)

    plt.savefig(f'{local_directory}known_exoplanets_{mode}.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_7():
    local_directory = '/mnt/c/users/tehan/desktop/known_exoplanet/'
    data = ascii.read(local_directory + 'PS_2022.04.17_18.23.57_.csv')
    fig = plt.figure(constrained_layout=False, figsize=(10, 8))
    gs = fig.add_gridspec(5, 12)
    gs.update(wspace=0.3, hspace=0.3)
    color = ['C0', 'C1', 'C3']

    #########################################################################
    # TOI-674
    tic = 158588995

    # load QLP
    qlp_9_t, qlp_9_f = load_qlp(ld=local_directory, tic=tic, sector=9)
    qlp_10_t, qlp_10_f = load_qlp(ld=local_directory, tic=tic, sector=10)
    qlp_36_t, qlp_36_f = load_qlp(ld=local_directory, tic=tic, sector=36)

    # load eleanor
    eleanor_9_t, eleanor_9_f_pca, eleanor_9_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=9)
    eleanor_10_t, eleanor_10_f_pca, eleanor_10_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=10)
    eleanor_36_t, eleanor_36_f_pca, eleanor_36_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=36)

    files = glob(local_directory + 'SPOC/TOI-674/*.fits')
    index = np.where(data['pl_name'] == 'TOI-674 b')
    period = 1.977165
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax1_1 = fig.add_subplot(gs[0, :3])
    ax1_2 = fig.add_subplot(gs[0, 3:6])
    ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax1_4 = fig.add_subplot(gs[0, 9:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax1_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax1_2.plot(eleanor_9_t % period / period - phase_fold_mid, eleanor_9_f_pca, '.', c=color[0], markersize=2,
               label='9')
    ax1_2.plot(eleanor_10_t % period / period - phase_fold_mid, eleanor_10_f_pca, '.', c=color[1], markersize=2,
               label='10')
    ax1_2.plot(eleanor_36_t % period / period - phase_fold_mid, eleanor_36_f_pca, '.', c=color[2], markersize=2,
               label='36')
    ax1_3.plot(eleanor_9_t % period / period - phase_fold_mid, eleanor_9_f_psf, '.', c=color[0], markersize=2,
               label='9')
    ax1_3.plot(eleanor_10_t % period / period - phase_fold_mid, eleanor_10_f_psf, '.', c=color[1], markersize=2,
               label='10')
    ax1_3.plot(eleanor_36_t % period / period - phase_fold_mid, eleanor_36_f_psf, '.', c=color[2], markersize=2,
               label='36')
    ax1_4.plot(qlp_9_t % period / period - phase_fold_mid, qlp_9_f, '.', c=color[0], markersize=2, label='9')
    ax1_4.plot(qlp_10_t % period / period - phase_fold_mid, qlp_10_f, '.', c=color[1], markersize=2, label='10')
    ax1_4.plot(qlp_36_t % period / period - phase_fold_mid, qlp_36_f, '.', c=color[2], markersize=2, label='36')

    ax1_1.legend(loc=3, fontsize=6)
    ax1_2.legend(loc=3, fontsize=6)
    ax1_3.legend(loc=3, fontsize=6)
    ax1_4.legend(loc=3, fontsize=6)
    ax1_1.set_ylim(0.975, 1.01)
    ax1_2.set_ylim(0.975, 1.01)
    ax1_3.set_ylim(0.975, 1.01)
    ax1_4.set_ylim(0.975, 1.01)
    ax1_1.set_xlim(- 0.03, 0.03)
    ax1_2.set_xlim(- 0.03, 0.03)
    ax1_3.set_xlim(- 0.03, 0.03)
    ax1_4.set_xlim(- 0.03, 0.03)
    ax1_2.set_yticklabels([])
    ax1_3.set_yticklabels([])
    ax1_4.set_yticklabels([])

    ax1_1.set_title('SPOC 2-min')
    ax1_2.set_title('eleanor CORR')
    ax1_3.set_title('eleanor PSF')
    ax1_4.set_title('QLP')
    ax1_1.set_ylabel('Normalized Flux')
    ax1_3.text(2.25, 0.5, f'TOI-674 b', horizontalalignment='center',
               verticalalignment='center', transform=ax1_3.transAxes, rotation=270, fontweight='semibold')
    ax1_3.text(2.15, 0.5, 'mag=11.88', horizontalalignment='center',
               verticalalignment='center', transform=ax1_3.transAxes, rotation=270)
    #########################################################################
    # LHS 3844
    tic = 410153553

    # load QLP
    qlp_27_t, qlp_27_f = load_qlp(ld=local_directory, tic=tic, sector=27)
    qlp_28_t, qlp_28_f = load_qlp(ld=local_directory, tic=tic, sector=28)

    # load eleanor
    eleanor_27_t, eleanor_27_f_pca, eleanor_27_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=27)
    eleanor_28_t, eleanor_28_f_pca, eleanor_28_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=28)

    files = glob(local_directory + 'SPOC/LHS 3844/*.fits')
    index = np.where(data['pl_name'] == 'LHS 3844 b')
    period = float(data['pl_orbper'][index])
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax2_1 = fig.add_subplot(gs[1, :3])
    ax2_2 = fig.add_subplot(gs[1, 3:6])
    ax2_3 = fig.add_subplot(gs[1, 6:9])
    ax2_4 = fig.add_subplot(gs[1, 9:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            if hdul[0].header['sector'] == 1:
                continue
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax2_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i - 1],
                       ms=2,
                       label=str(hdul[0].header['sector']))
    ax2_2.plot(eleanor_27_t % period / period - phase_fold_mid, eleanor_27_f_pca, '.', c=color[0], markersize=2,
               label='27')
    ax2_2.plot(eleanor_28_t % period / period - phase_fold_mid, eleanor_28_f_pca, '.', c=color[1], markersize=2,
               label='28')
    ax2_3.plot(eleanor_27_t % period / period - phase_fold_mid, eleanor_27_f_psf, '.', c=color[0], markersize=2,
               label='27')
    ax2_3.plot(eleanor_28_t % period / period - phase_fold_mid, eleanor_28_f_psf, '.', c=color[1], markersize=2,
               label='28')
    ax2_4.plot(qlp_27_t % period / period - phase_fold_mid, qlp_27_f, '.', c=color[0], markersize=2, label='27')
    ax2_4.plot(qlp_28_t % period / period - phase_fold_mid, qlp_28_f, '.', c=color[1], markersize=2, label='28')

    ax2_1.legend(loc=3, fontsize=6)
    ax2_2.legend(loc=3, fontsize=6)
    ax2_3.legend(loc=3, fontsize=6)
    ax2_4.legend(loc=3, fontsize=6)
    ax2_1.set_ylim(0.988, 1.007)
    ax2_2.set_ylim(0.988, 1.007)
    ax2_3.set_ylim(0.988, 1.007)
    ax2_4.set_ylim(0.988, 1.007)
    ax2_1.set_xlim(- 0.07, 0.07)
    ax2_2.set_xlim(- 0.07, 0.07)
    ax2_3.set_xlim(- 0.07, 0.07)
    ax2_4.set_xlim(- 0.07, 0.07)
    ax2_2.set_yticklabels([])
    ax2_3.set_yticklabels([])
    ax2_4.set_yticklabels([])
    ax2_1.set_ylabel('Normalized Flux')
    ax2_3.text(2.25, 0.5, f'LHS 3844 b', horizontalalignment='center',
               verticalalignment='center', transform=ax2_3.transAxes, rotation=270, fontweight='semibold')
    ax2_3.text(2.15, 0.5, 'mag=11.92', horizontalalignment='center',
               verticalalignment='center', transform=ax2_3.transAxes, rotation=270)
    #########################################################################
    # TOI-530
    tic = 387690507

    # load QLP
    qlp_6_t, qlp_6_f = load_qlp(ld=local_directory, tic=tic, sector=6)

    # load eleanor
    eleanor_6_t, eleanor_6_f_pca, eleanor_6_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=6)
    eleanor_44_t, eleanor_44_f_pca, eleanor_44_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=44)
    eleanor_45_t, eleanor_45_f_pca, eleanor_45_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=45)

    files = glob(local_directory + 'SPOC/TOI-530/*.fits')
    index = np.where(data['pl_name'] == 'TOI-530 b')
    period = 6.387583
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax3_1 = fig.add_subplot(gs[2, :3])
    ax3_2 = fig.add_subplot(gs[2, 3:6])
    ax3_3 = fig.add_subplot(gs[2, 6:9])
    ax3_4 = fig.add_subplot(gs[2, 9:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax3_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax3_2.plot(eleanor_6_t % period / period - phase_fold_mid, eleanor_6_f_pca, '.', c=color[0], markersize=2,
               label='6')
    ax3_2.plot(eleanor_44_t % period / period - phase_fold_mid, eleanor_44_f_pca, '.', c=color[1], markersize=2,
               label='44')
    ax3_2.plot(eleanor_45_t % period / period - phase_fold_mid, eleanor_45_f_pca, '.', c=color[2], markersize=2,
               label='45')
    ax3_3.plot(eleanor_6_t % period / period - phase_fold_mid, eleanor_6_f_psf, '.', c=color[0], markersize=2,
               label='6')
    ax3_3.plot(eleanor_44_t % period / period - phase_fold_mid, eleanor_44_f_psf, '.', c=color[1], markersize=2,
               label='44')
    ax3_3.plot(eleanor_45_t % period / period - phase_fold_mid, eleanor_45_f_psf, '.', c=color[2], markersize=2,
               label='45')
    ax3_4.plot(qlp_6_t % period / period - phase_fold_mid, qlp_6_f, '.', c=color[0], markersize=2, label='6')

    ax3_1.legend(loc=3, fontsize=6)
    ax3_2.legend(loc=3, fontsize=6)
    ax3_3.legend(loc=3, fontsize=6)
    ax3_4.legend(loc=3, fontsize=6)
    ax3_1.set_ylim(0.95, 1.03)
    ax3_2.set_ylim(0.95, 1.03)
    ax3_3.set_ylim(0.95, 1.03)
    ax3_4.set_ylim(0.95, 1.03)
    ax3_1.set_xlim(- 0.03, 0.03)
    ax3_2.set_xlim(- 0.03, 0.03)
    ax3_3.set_xlim(- 0.03, 0.03)
    ax3_4.set_xlim(- 0.03, 0.03)
    ax3_2.set_yticklabels([])
    ax3_3.set_yticklabels([])
    ax3_4.set_yticklabels([])
    ax3_1.set_ylabel('Normalized Flux')
    ax3_3.text(2.25, 0.5, 'TOI-530 b', horizontalalignment='center',
               verticalalignment='center', transform=ax3_3.transAxes, rotation=270, fontweight='semibold')
    ax3_3.text(2.15, 0.5, 'mag=13.53', horizontalalignment='center',
               verticalalignment='center', transform=ax3_3.transAxes, rotation=270)
    #########################################################################
    # TOI-2406
    tic = 212957629

    # load QLP
    qlp_30_t, qlp_30_f = load_qlp(ld=local_directory, tic=tic, sector=30)

    # load eleanor
    eleanor_3_t, eleanor_3_f_pca, eleanor_3_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=3)
    eleanor_42_t, eleanor_42_f_pca, eleanor_42_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=42)
    eleanor_43_t, eleanor_43_f_pca, eleanor_43_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=43)

    files = glob(local_directory + 'SPOC/TOI-2406/*.fits')
    index = np.where(data['pl_name'] == 'TOI-2406 b')
    period = 3.076676
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax4_1 = fig.add_subplot(gs[3, :3])
    ax4_2 = fig.add_subplot(gs[3, 3:6])
    ax4_3 = fig.add_subplot(gs[3, 6:9])
    ax4_4 = fig.add_subplot(gs[3, 9:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax4_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i + 1],
                       ms=2,
                       label=str(hdul[0].header['sector']))
    ax4_2.plot(eleanor_3_t % period / period - phase_fold_mid, eleanor_3_f_pca, '.', c=color[0], markersize=2,
               label='3')
    ax4_2.plot(eleanor_42_t % period / period - phase_fold_mid, eleanor_42_f_pca, '.', c=color[1], markersize=2,
               label='42')
    ax4_2.plot(eleanor_43_t % period / period - phase_fold_mid, eleanor_43_f_pca, '.', c=color[2], markersize=2,
               label='43')
    ax4_3.plot(eleanor_3_t % period / period - phase_fold_mid, eleanor_3_f_psf, '.', c=color[0], markersize=2,
               label='3')
    ax4_3.plot(eleanor_42_t % period / period - phase_fold_mid, eleanor_42_f_psf, '.', c=color[1], markersize=2,
               label='42')
    ax4_3.plot(eleanor_43_t % period / period - phase_fold_mid, eleanor_43_f_psf, '.', c=color[2], markersize=2,
               label='43')
    ax4_4.plot(qlp_30_t % period / period - phase_fold_mid, qlp_30_f, '.', c=color[0], markersize=2, label='30')

    ax4_1.legend(loc=3, fontsize=6)
    ax4_2.legend(loc=3, fontsize=6)
    ax4_3.legend(loc=3, fontsize=6)
    ax4_4.legend(loc=3, fontsize=6)
    ax4_1.set_ylim(0.945, 1.04)
    ax4_2.set_ylim(0.945, 1.04)
    ax4_3.set_ylim(0.945, 1.04)
    ax4_4.set_ylim(0.945, 1.04)
    ax4_1.set_xlim(- 0.04, 0.04)
    ax4_2.set_xlim(- 0.04, 0.04)
    ax4_3.set_xlim(- 0.04, 0.04)
    ax4_4.set_xlim(- 0.04, 0.04)
    ax4_1.set_xticks([-0.03, 0, 0.03])
    ax4_1.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_2.set_xticks([-0.03, 0, 0.03])
    ax4_2.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_3.set_xticks([-0.03, 0, 0.03])
    ax4_3.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_4.set_xticks([-0.03, 0, 0.03])
    ax4_4.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_4.set_xlabel('Phase')
    ax4_2.set_yticklabels([])
    ax4_3.set_yticklabels([])
    ax4_4.set_yticklabels([])
    ax4_1.set_ylabel('Normalized Flux')
    ax4_3.text(2.25, 0.5, 'TOI-2406 b', horizontalalignment='center',
               verticalalignment='center', transform=ax4_3.transAxes, rotation=270, fontweight='semibold')
    ax4_3.text(2.15, 0.5, 'mag=14.31', horizontalalignment='center',
               verticalalignment='center', transform=ax4_3.transAxes, rotation=270)

    #########################################################################
    # TOI-519
    tic = 218795833

    # load eleanor
    eleanor_7_t, eleanor_7_f_aper, eleanor_7_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=7)
    eleanor_8_t, eleanor_8_f_aper, eleanor_8_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=8)
    eleanor_34_t, eleanor_34_f_aper, eleanor_34_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=34)

    files = glob(local_directory + 'SPOC/TOI-519/*.fits')
    index = np.where(data['pl_name'] == 'TOI-519 b')
    period = 1.265232
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax5_1 = fig.add_subplot(gs[4, :3])
    ax5_2 = fig.add_subplot(gs[4, 3:6])
    ax5_3 = fig.add_subplot(gs[4, 6:9])
    # ax5_4 = fig.add_subplot(gs[4, 9:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            if hdul[0].header['sector'] == 34:
                i = i + 1
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax5_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax5_2.plot(eleanor_7_t % period / period - phase_fold_mid, eleanor_7_f_aper, '.', c=color[0], markersize=2,
               label='7')
    ax5_2.plot(eleanor_8_t % period / period - phase_fold_mid, eleanor_8_f_aper, '.', c=color[1], markersize=2,
               label='8')
    ax5_2.plot(eleanor_34_t % period / period - phase_fold_mid, eleanor_34_f_aper, '.', c=color[2], markersize=2,
               label='34')
    # ax5_3.plot(eleanor_7_t % period / period - phase_fold_mid, eleanor_7_f_psf, '.', c=color[0], markersize=2,
    #            label='7')
    # ax5_3.plot(eleanor_8_t % period / period - phase_fold_mid, eleanor_8_f_psf, '.', c=color[1], markersize=2,
    #            label='8')
    ax5_3.plot(eleanor_34_t % period / period - phase_fold_mid, eleanor_34_f_psf, '.', c=color[2], markersize=2,
               label='34')

    ax5_1.legend(loc=3, fontsize=6)
    ax5_2.legend(loc=3, fontsize=6)
    ax5_3.legend(loc=3, fontsize=6)
    ax5_1.set_ylim(0.83, 1.05)
    ax5_2.set_ylim(0.83, 1.05)
    ax5_3.set_ylim(0.83, 1.05)
    ax5_1.set_xlim(- 0.05, 0.05)
    ax5_2.set_xlim(- 0.05, 0.05)
    ax5_3.set_xlim(- 0.05, 0.05)
    ax5_1.set_yticks([0.9, 1.0])
    ax5_1.set_yticklabels(['0.90', '1.00'])
    ax5_1.set_xticks([-0.03, 0, 0.03])
    ax5_1.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_2.set_xticks([-0.03, 0, 0.03])
    ax5_2.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_3.set_xticks([-0.03, 0, 0.03])
    ax5_3.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_2.set_yticklabels([])
    ax5_3.set_yticklabels([])
    ax5_1.set_xlabel('Phase')
    ax5_2.set_xlabel('Phase')
    ax5_3.set_xlabel('Phase')
    ax5_1.set_ylabel('Normalized Flux')
    ax5_3.text(2.25, 0.5, 'TOI-519 b', horizontalalignment='center',
               verticalalignment='center', transform=ax5_3.transAxes, rotation=270, fontweight='semibold')
    ax5_3.text(2.15, 0.5, 'mag=14.43', horizontalalignment='center',
               verticalalignment='center', transform=ax5_3.transAxes, rotation=270)
    # ax5_1.set_yticklabels([])

    # plt.savefig('/mnt/c/users/tehan/desktop/known_exoplanets_other.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_8():
    local_directory = '/mnt/c/users/tehan/desktop/NGC 7654/'
    fig = plt.figure(constrained_layout=False, figsize=(10, 7))
    gs = fig.add_gridspec(5, 12)
    gs.update(wspace=0.2, hspace=0.2)
    index = [77, 469, 699, 1251, 1585]
    # TIC 270022476
    tic = 270022476
    with fits.open(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2015669349341459328-s0017_tess_v1_llc.fits',
                   mode='denywrite') as hdul:
        q = hdul[1].data['TGLC_flags'] == 0
        t = hdul[1].data['time'][q]
        f = hdul[1].data['cal_psf_flux'][q]

    eleanor_t, eleanor_f_pca, eleanor_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=17)
    qlp_t, qlp_f = load_qlp(ld=local_directory, tic=tic, sector=17)
    ztf_g_t, ztf_g_flux = load_ztf(ld=local_directory, index=index[0])

    period = 1.9221
    ax1_1 = fig.add_subplot(gs[0, :3])
    ax1_2 = fig.add_subplot(gs[0, 3:6])
    ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax1_4 = fig.add_subplot(gs[0, 9:])
    ax1_1.plot(t % period / period, f, '.', c='k', markersize=1, zorder=3)
    ax1_1.plot(ztf_g_t % period / period, ztf_g_flux / np.median(ztf_g_flux), 'x', color='green', ms=2,
               label='ZTF g-band')
    ax1_2.plot(eleanor_t % period / period, eleanor_f_pca, '.', c='k', markersize=1)
    ax1_3.plot(eleanor_t % period / period, eleanor_f_psf, '.', c='k', markersize=1)
    ax1_4.plot(qlp_t % period / period, qlp_f, '.', c='k', markersize=1)

    ax1_1.set_xticklabels([])
    ax1_2.set_xticklabels([])
    ax1_3.set_xticklabels([])
    ax1_4.set_xticklabels([])
    ax1_2.set_yticklabels([])
    ax1_3.set_yticklabels([])
    ax1_4.set_yticklabels([])
    ax1_1.set_ylim(0.88, 1.03)
    ax1_2.set_ylim(0.88, 1.03)
    ax1_3.set_ylim(0.88, 1.03)
    ax1_4.set_ylim(0.88, 1.03)
    ax1_1.set_title('TGLC PSF')
    ax1_2.set_title('eleanor CORR')
    ax1_3.set_title('eleanor PSF')
    ax1_4.set_title('QLP')
    ax1_1.set_ylabel('Norm Flux')
    ax1_3.text(2.25, 0.5, f'TIC \n{tic}', horizontalalignment='center',
               verticalalignment='center', transform=ax1_3.transAxes, rotation=270, fontweight='semibold')
    ax1_3.text(2.12, 0.5, 'mag=11.52', horizontalalignment='center',
               verticalalignment='center', transform=ax1_3.transAxes, rotation=270)

    # TIC 270140796
    tic = 270140796
    with fits.open(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2015671415229352192-s0017_tess_v1_llc.fits',
                   mode='denywrite') as hdul:
        q = hdul[1].data['TGLC_flags'] == 0
        t = hdul[1].data['time'][q]
        f = hdul[1].data['cal_psf_flux'][q]
    eleanor_t, eleanor_f_pca, eleanor_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=17)
    qlp_t, qlp_f = load_qlp(ld=local_directory, tic=tic, sector=17)
    ztf_g_t, ztf_g_flux, ztf_r_t, ztf_r_flux = load_ztf(ld=local_directory, index=index[1])

    period = 6.126
    ax2_1 = fig.add_subplot(gs[1, :3])
    ax2_2 = fig.add_subplot(gs[1, 3:6])
    ax2_3 = fig.add_subplot(gs[1, 6:9])
    ax2_4 = fig.add_subplot(gs[1, 9:])
    ax2_1.plot(t % period / period, f, '.', c='k', markersize=1, zorder=3)
    ax2_1.plot(ztf_g_t % period / period, ztf_g_flux / np.median(ztf_g_flux), 'x', color='green', ms=2,
               label='ZTF g-band')
    ax2_1.scatter(ztf_r_t % period / period, ztf_r_flux / np.median(ztf_r_flux), facecolors='none',
                  edgecolors='orangered', s=3, label='ZTF r-band')
    ax2_2.plot(eleanor_t % period / period, eleanor_f_pca, '.', c='k', markersize=1)
    ax2_3.plot(eleanor_t % period / period, eleanor_f_psf, '.', c='k', markersize=1)
    ax2_4.plot(qlp_t % period / period, qlp_f, '.', c='k', markersize=1)

    ax2_1.set_xticklabels([])
    ax2_2.set_xticklabels([])
    ax2_3.set_xticklabels([])
    ax2_2.set_yticklabels([])
    ax2_3.set_yticklabels([])
    ax2_4.set_yticklabels([])
    ax2_1.set_ylim(0.80, 1.05)
    ax2_2.set_ylim(0.80, 1.05)
    ax2_3.set_ylim(0.80, 1.05)
    ax2_4.set_ylim(0.80, 1.05)
    ax2_4.set_xlabel('Phase')
    ax2_1.set_ylabel('Norm Flux')
    ax2_3.text(2.25, 0.5, f'TIC \n{tic}', horizontalalignment='center',
               verticalalignment='center', transform=ax2_3.transAxes, rotation=270, fontweight='semibold')
    ax2_3.text(2.12, 0.5, 'mag=13.44', horizontalalignment='center',
               verticalalignment='center', transform=ax2_3.transAxes, rotation=270)

    # TIC 269820902
    tic = 269820902
    with fits.open(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2015648943960251008-s0017_tess_v1_llc.fits',
                   mode='denywrite') as hdul:
        q = hdul[1].data['TGLC_flags'] == 0
        t = hdul[1].data['time'][q]
        f = hdul[1].data['cal_psf_flux'][q]
    eleanor_t, eleanor_f_pca, eleanor_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=17)
    ztf_g_t, ztf_g_flux, ztf_r_t, ztf_r_flux = load_ztf(ld=local_directory, index=index[2])
    period = 1.01968
    ax3_1 = fig.add_subplot(gs[2, :3])
    ax3_2 = fig.add_subplot(gs[2, 3:6])
    ax3_3 = fig.add_subplot(gs[2, 6:9])
    # ax3_4 = fig.add_subplot(gs[2, 9:])
    ax3_1.plot(t % period / period, f, '.', c='k', markersize=1, zorder=3)
    ax3_1.plot(ztf_g_t % period / period, ztf_g_flux / np.median(ztf_g_flux), 'x', color='green', ms=2,
               label='ZTF g-band')
    ax3_1.scatter(ztf_r_t % period / period, ztf_r_flux / np.median(ztf_r_flux), facecolors='none',
                  edgecolors='orangered', s=3, label='ZTF r-band')
    ax3_2.plot(eleanor_t % period / period, eleanor_f_pca, '.', c='k', markersize=1)
    ax3_3.plot(eleanor_t % period / period, eleanor_f_psf, '.', c='k', markersize=1)

    ax3_1.set_xticklabels([])
    ax3_2.set_xticklabels([])
    ax3_3.set_xticklabels([])
    ax3_2.set_yticklabels([])
    ax3_3.set_yticklabels([])
    # ax3_4.set_yticklabels([])
    ax3_1.set_ylim(0.85, 1.05)
    ax3_2.set_ylim(0.85, 1.05)
    ax3_3.set_ylim(0.85, 1.05)
    ax3_1.set_ylabel('Norm Flux')
    ax3_3.text(2.25, 0.5, f'TIC \n{tic}', horizontalalignment='center',
               verticalalignment='center', transform=ax3_3.transAxes, rotation=270, fontweight='semibold')
    ax3_3.text(2.12, 0.5, 'mag=13.90', horizontalalignment='center',
               verticalalignment='center', transform=ax3_3.transAxes, rotation=270)

    # TIC 270023061
    tic = 270023061
    with fits.open(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2015656743621212928-s0017_tess_v1_llc.fits',
                   mode='denywrite') as hdul:
        q = hdul[1].data['TGLC_flags'] == 0
        t = hdul[1].data['time'][q]
        f = hdul[1].data['cal_psf_flux'][q]
    eleanor_t, eleanor_f_pca, eleanor_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=17)
    ztf_g_t, ztf_g_flux, ztf_r_t, ztf_r_flux = load_ztf(ld=local_directory, index=index[3])
    period = 2.2895
    ax4_1 = fig.add_subplot(gs[3, :3])
    ax4_2 = fig.add_subplot(gs[3, 3:6])
    ax4_3 = fig.add_subplot(gs[3, 6:9])
    # ax4_4 = fig.add_subplot(gs[3, 9:])
    ax4_1.plot(t % period / period, f, '.', c='k', markersize=1, zorder=3)
    ax4_1.plot(ztf_g_t % period / period, ztf_g_flux / np.median(ztf_g_flux), 'x', color='green', ms=2,
               label='ZTF g-band')
    ax4_1.scatter(ztf_r_t % period / period, ztf_r_flux / np.median(ztf_r_flux), facecolors='none',
                  edgecolors='orangered', s=3, label='ZTF r-band')
    ax4_2.plot(eleanor_t % period / period, eleanor_f_pca, '.', c='k', markersize=1)
    ax4_3.plot(eleanor_t % period / period, eleanor_f_psf, '.', c='k', markersize=1)

    ax4_1.set_xticklabels([])
    ax4_2.set_xticklabels([])
    ax4_3.set_xticklabels([])
    ax4_2.set_yticklabels([])
    ax4_3.set_yticklabels([])
    # ax4_4.set_yticklabels([])
    ax4_1.set_ylim(0.6, 1.12)
    ax4_2.set_ylim(0.6, 1.12)
    ax4_3.set_ylim(0.6, 1.12)
    ax4_1.set_ylabel('Norm Flux')
    ax4_3.text(2.25, 0.5, f'TIC \n{tic}', horizontalalignment='center',
               verticalalignment='center', transform=ax4_3.transAxes, rotation=270, fontweight='semibold')
    ax4_3.text(2.12, 0.5, 'mag=14.71', horizontalalignment='center',
               verticalalignment='center', transform=ax4_3.transAxes, rotation=270)

    # TIC 269820513
    tic = 269820513
    with fits.open(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2015457010457210752-s0017_tess_v1_llc.fits',
                   mode='denywrite') as hdul:
        q = hdul[1].data['TGLC_flags'] == 0
        t = hdul[1].data['time'][q]
        f = hdul[1].data['cal_psf_flux'][q]
    eleanor_t, eleanor_f_pca, eleanor_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=17)
    ztf_g_t, ztf_g_flux, ztf_r_t, ztf_r_flux = load_ztf(ld=local_directory, index=index[4])
    period = 6.558
    ax5_1 = fig.add_subplot(gs[4, :3])
    ax5_2 = fig.add_subplot(gs[4, 3:6])
    ax5_3 = fig.add_subplot(gs[4, 6:9])
    # ax5_4 = fig.add_subplot(gs[4, 9:])
    ax5_1.plot(t % period / period, f, '.', c='k', markersize=1, zorder=3, label='TESS FFI')
    ax5_1.plot(ztf_g_t % period / period, ztf_g_flux / np.median(ztf_g_flux), 'x', color='green', ms=2,
               label='ZTF g-band')
    ax5_1.scatter(ztf_r_t % period / period, ztf_r_flux / np.median(ztf_r_flux), facecolors='none',
                  edgecolors='orangered', s=3, label='ZTF r-band')
    ax5_2.plot(eleanor_t % period / period, eleanor_f_pca, '.', c='k', markersize=1)
    ax5_3.plot(eleanor_t % period / period, eleanor_f_psf, '.', c='k', markersize=1)

    ax5_2.set_yticklabels([])
    ax5_3.set_yticklabels([])
    # ax5_4.set_yticklabels([])
    ax5_1.set_ylim(0.72, 1.06)
    ax5_2.set_ylim(0.72, 1.06)
    ax5_3.set_ylim(0.72, 1.06)
    ax5_1.set_xlabel('Phase')
    ax5_2.set_xlabel('Phase')
    ax5_3.set_xlabel('Phase')
    ax5_1.set_ylabel('Norm Flux')
    ax5_3.text(2.25, 0.5, f'TIC \n{tic}', horizontalalignment='center',
               verticalalignment='center', transform=ax5_3.transAxes, rotation=270, fontweight='semibold')
    ax5_3.text(2.12, 0.5, 'mag=15.03', horizontalalignment='center',
               verticalalignment='center', transform=ax5_3.transAxes, rotation=270)
    ax5_1.legend(bbox_to_anchor=(3.3, 0), loc=3, markerscale=2)
    # plt.savefig('/mnt/c/users/tehan/desktop/EB_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_9():
    local_directory = '/mnt/c/users/tehan/desktop/known_exoplanet/'
    data = ascii.read(local_directory + 'PS_2022.04.17_18.23.57_.csv')
    fig = plt.figure(constrained_layout=False, figsize=(10, 8))
    gs = fig.add_gridspec(5, 10)
    gs.update(wspace=0.1, hspace=0.3)
    color = ['C0', 'C1', 'C3']

    #########################################################################
    # TOI-674
    tic = 158588995

    # load QLP
    qlp_9_t, qlp_9_f = load_qlp(ld=local_directory, tic=tic, sector=9)
    qlp_10_t, qlp_10_f = load_qlp(ld=local_directory, tic=tic, sector=10)
    qlp_36_t, qlp_36_f = load_qlp(ld=local_directory, tic=tic, sector=36)

    # load eleanor
    eleanor_9_t, eleanor_9_f_pca, eleanor_9_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=9)
    eleanor_10_t, eleanor_10_f_pca, eleanor_10_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=10)
    eleanor_36_t, eleanor_36_f_pca, eleanor_36_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=36)

    # load TGLC
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0009*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_09 = hdul[1].data['time'][q]
        f_psf_09 = hdul[1].data['cal_psf_flux'][q]
        f_aper_09 = hdul[1].data['cal_aper_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0010*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_10 = hdul[1].data['time'][q]
        f_psf_10 = hdul[1].data['cal_psf_flux'][q]
        f_aper_10 = hdul[1].data['cal_aper_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5400949450924312576-s0036*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_36 = hdul[1].data['time'][q]
        f_psf_36 = hdul[1].data['cal_psf_flux'][q]
        f_aper_36 = hdul[1].data['cal_aper_flux'][q]
        t_36 = np.mean(t_36[:len(t_36) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_36 = np.mean(f_psf_36[:len(f_psf_36) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_36 = np.mean(f_aper_36[:len(f_aper_36) // 3 * 3].reshape(-1, 3), axis=1)

    files = glob(local_directory + 'SPOC/TOI-674/*.fits')
    index = np.where(data['pl_name'] == 'TOI-674 b')
    period = 1.977165
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax1_1 = fig.add_subplot(gs[0, :2])
    ax1_2 = fig.add_subplot(gs[0, 2:4])
    # ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax1_4 = fig.add_subplot(gs[0, 4:6])
    ax1_5 = fig.add_subplot(gs[0, 6:8])
    ax1_6 = fig.add_subplot(gs[0, 8:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax1_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax1_2.plot(eleanor_9_t % period / period - phase_fold_mid, eleanor_9_f_pca, '.', c=color[0], markersize=2,
               label='9')
    ax1_2.plot(eleanor_10_t % period / period - phase_fold_mid, eleanor_10_f_pca, '.', c=color[1], markersize=2,
               label='10')
    ax1_2.plot(eleanor_36_t % period / period - phase_fold_mid, eleanor_36_f_pca, '.', c=color[2], markersize=2,
               label='36')
    # ax1_3.plot(eleanor_9_t % period / period - phase_fold_mid, eleanor_9_f_psf, '.', c=color[0], markersize=2,
    #            label='9')
    # ax1_3.plot(eleanor_10_t % period / period - phase_fold_mid, eleanor_10_f_psf, '.', c=color[1], markersize=2,
    #            label='10')
    # ax1_3.plot(eleanor_36_t % period / period - phase_fold_mid, eleanor_36_f_psf, '.', c=color[2], markersize=2,
    #            label='36')
    ax1_4.plot(qlp_9_t % period / period - phase_fold_mid, qlp_9_f, '.', c=color[0], markersize=2, label='9')
    ax1_4.plot(qlp_10_t % period / period - phase_fold_mid, qlp_10_f, '.', c=color[1], markersize=2, label='10')
    ax1_4.plot(qlp_36_t % period / period - phase_fold_mid, qlp_36_f, '.', c=color[2], markersize=2, label='36')

    ax1_5.plot(t_09 % period / period - phase_fold_mid, f_aper_09, '.', c=color[0], markersize=2, label='9')
    ax1_5.plot(t_10 % period / period - phase_fold_mid, f_aper_10, '.', c=color[1], markersize=2, label='10')
    ax1_5.plot(t_36 % period / period - phase_fold_mid, f_aper_36, '.', c=color[2], markersize=2, label='36')

    ax1_6.plot(t_09 % period / period - phase_fold_mid, f_psf_09, '.', c=color[0], markersize=2, label='9')
    ax1_6.plot(t_10 % period / period - phase_fold_mid, f_psf_10, '.', c=color[1], markersize=2, label='10')
    ax1_6.plot(t_36 % period / period - phase_fold_mid, f_psf_36, '.', c=color[2], markersize=2, label='36')

    ax1_1.legend(loc=3, fontsize=6)
    ax1_2.legend(loc=3, fontsize=6)
    # ax1_3.legend(loc=3, fontsize=6)
    ax1_4.legend(loc=3, fontsize=6)
    ax1_5.legend(loc=3, fontsize=6)
    ax1_6.legend(loc=3, fontsize=6)
    ax1_1.set_ylim(0.975, 1.01)
    ax1_2.set_ylim(0.975, 1.01)
    # ax1_3.set_ylim(0.975, 1.01)
    ax1_4.set_ylim(0.975, 1.01)
    ax1_5.set_ylim(0.975, 1.01)
    ax1_6.set_ylim(0.975, 1.01)
    ax1_1.set_xlim(- 0.03, 0.03)
    ax1_2.set_xlim(- 0.03, 0.03)
    # ax1_3.set_xlim(- 0.03, 0.03)
    ax1_4.set_xlim(- 0.03, 0.03)
    ax1_5.set_xlim(- 0.03, 0.03)
    ax1_6.set_xlim(- 0.03, 0.03)
    ax1_2.set_yticklabels([])
    # ax1_3.set_yticklabels([])
    ax1_4.set_yticklabels([])
    ax1_5.set_yticklabels([])
    ax1_6.set_yticklabels([])

    ax1_1.set_title('SPOC 2-min')
    ax1_2.set_title('eleanor CORR')
    # ax1_3.set_title('eleanor PSF')
    ax1_4.set_title('QLP')
    ax1_5.set_title('TGLC aperture', weight='bold')
    ax1_6.set_title('TGLC PSF', weight='bold')
    ax1_1.set_ylabel('Normalized Flux')
    ax1_5.text(2.25, 0.5, f'TOI-674 b', horizontalalignment='center',
               verticalalignment='center', transform=ax1_5.transAxes, rotation=270, fontweight='semibold')
    ax1_5.text(2.15, 0.5, 'mag=11.88', horizontalalignment='center',
               verticalalignment='center', transform=ax1_5.transAxes, rotation=270)

    #########################################################################
    # LHS 3844
    tic = 410153553

    # load QLP
    qlp_27_t, qlp_27_f = load_qlp(ld=local_directory, tic=tic, sector=27)
    qlp_28_t, qlp_28_f = load_qlp(ld=local_directory, tic=tic, sector=28)

    # load eleanor
    eleanor_27_t, eleanor_27_f_pca, eleanor_27_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=27)
    eleanor_28_t, eleanor_28_f_pca, eleanor_28_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=28)

    # load TGLC
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6385548541499112448-s0027*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_27 = hdul[1].data['time'][q]
        f_aper_27 = hdul[1].data['cal_aper_flux'][q]
        f_psf_27 = hdul[1].data['cal_psf_flux'][q]
        t_27 = np.mean(t_27[:len(t_27) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_27 = np.mean(f_aper_27[:len(f_aper_27) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_27 = np.mean(f_psf_27[:len(f_psf_27) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6385548541499112448-s0028*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_28 = hdul[1].data['time'][q]
        f_aper_28 = hdul[1].data['cal_aper_flux'][q]
        f_psf_28 = hdul[1].data['cal_psf_flux'][q]
        t_28 = np.mean(t_28[:len(t_28) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_28 = np.mean(f_aper_28[:len(f_aper_28) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_28 = np.mean(f_psf_28[:len(f_psf_28) // 3 * 3].reshape(-1, 3), axis=1)

    files = glob(local_directory + 'SPOC/LHS 3844/*.fits')
    index = np.where(data['pl_name'] == 'LHS 3844 b')
    period = float(data['pl_orbper'][index])
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax2_1 = fig.add_subplot(gs[1, :2])
    ax2_2 = fig.add_subplot(gs[1, 2:4])
    # ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax2_4 = fig.add_subplot(gs[1, 4:6])
    ax2_5 = fig.add_subplot(gs[1, 6:8])
    ax2_6 = fig.add_subplot(gs[1, 8:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            if hdul[0].header['sector'] == 1:
                continue
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax2_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i - 1],
                       ms=2,
                       label=str(hdul[0].header['sector']))
    ax2_2.plot(eleanor_27_t % period / period - phase_fold_mid, eleanor_27_f_pca, '.', c=color[0], markersize=2,
               label='27')
    ax2_2.plot(eleanor_28_t % period / period - phase_fold_mid, eleanor_28_f_pca, '.', c=color[1], markersize=2,
               label='28')
    # ax2_3.plot(eleanor_27_t % period / period - phase_fold_mid, eleanor_27_f_psf, '.', c=color[0], markersize=2,
    #            label='27')
    # ax2_3.plot(eleanor_28_t % period / period - phase_fold_mid, eleanor_28_f_psf, '.', c=color[1], markersize=2,
    #            label='28')
    ax2_4.plot(qlp_27_t % period / period - phase_fold_mid, qlp_27_f, '.', c=color[0], markersize=2, label='27')
    ax2_4.plot(qlp_28_t % period / period - phase_fold_mid, qlp_28_f, '.', c=color[1], markersize=2, label='28')

    ax2_5.plot(t_27 % period / period - phase_fold_mid, f_aper_27, '.', c=color[0], markersize=2, label='27')
    ax2_5.plot(t_28 % period / period - phase_fold_mid, f_aper_28, '.', c=color[1], markersize=2, label='28')

    ax2_6.plot(t_27 % period / period - phase_fold_mid, f_psf_27, '.', c=color[0], markersize=2, label='27')
    ax2_6.plot(t_28 % period / period - phase_fold_mid, f_psf_28, '.', c=color[1], markersize=2, label='28')
    ax2_1.legend(loc=3, fontsize=6)
    ax2_2.legend(loc=3, fontsize=6)
    # ax2_3.legend(loc=3, fontsize=6)
    ax2_4.legend(loc=3, fontsize=6)
    ax2_5.legend(loc=3, fontsize=6)
    ax2_6.legend(loc=3, fontsize=6)
    ax2_1.set_ylim(0.988, 1.007)
    ax2_2.set_ylim(0.988, 1.007)
    # ax2_3.set_ylim(0.988, 1.007)
    ax2_4.set_ylim(0.988, 1.007)
    ax2_5.set_ylim(0.988, 1.007)
    ax2_6.set_ylim(0.988, 1.007)
    ax2_1.set_xlim(- 0.07, 0.07)
    ax2_2.set_xlim(- 0.07, 0.07)
    # ax2_3.set_xlim(- 0.07, 0.07)
    ax2_4.set_xlim(- 0.07, 0.07)
    ax2_5.set_xlim(- 0.07, 0.07)
    ax2_6.set_xlim(- 0.07, 0.07)
    ax2_2.set_yticklabels([])
    # ax2_3.set_yticklabels([])
    ax2_4.set_yticklabels([])
    ax2_5.set_yticklabels([])
    ax2_6.set_yticklabels([])
    ax2_1.set_ylabel('Normalized Flux')
    ax2_5.text(2.25, 0.5, f'LHS 3844 b', horizontalalignment='center',
               verticalalignment='center', transform=ax2_5.transAxes, rotation=270, fontweight='semibold')
    ax2_5.text(2.15, 0.5, 'mag=11.92', horizontalalignment='center',
               verticalalignment='center', transform=ax2_5.transAxes, rotation=270)
    #########################################################################
    # TOI-530
    tic = 387690507

    # load QLP
    qlp_6_t, qlp_6_f = load_qlp(ld=local_directory, tic=tic, sector=6)

    # load eleanor
    eleanor_6_t, eleanor_6_f_pca, eleanor_6_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=6)
    eleanor_44_t, eleanor_44_f_pca, eleanor_44_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=44)
    eleanor_45_t, eleanor_45_f_pca, eleanor_45_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=45)

    # load TGLC
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0006*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_06 = hdul[1].data['time'][q]
        f_aper_06 = hdul[1].data['cal_aper_flux'][q]
        f_psf_06 = hdul[1].data['cal_psf_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0044*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_44 = hdul[1].data['time'][q]
        f_aper_44 = hdul[1].data['cal_aper_flux'][q]
        f_psf_44 = hdul[1].data['cal_psf_flux'][q]
        t_44 = np.mean(t_44[:len(t_44) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_44 = np.mean(f_aper_44[:len(f_aper_44) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_44 = np.mean(f_psf_44[:len(f_psf_44) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-3353218995355814656-s0045*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_45 = hdul[1].data['time'][q]
        f_aper_45 = hdul[1].data['cal_aper_flux'][q]
        f_psf_45 = hdul[1].data['cal_psf_flux'][q]
        t_45 = np.mean(t_45[:len(t_45) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_45 = np.mean(f_aper_45[:len(f_aper_45) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_45 = np.mean(f_psf_45[:len(f_psf_45) // 3 * 3].reshape(-1, 3), axis=1)

    files = glob(local_directory + 'SPOC/TOI-530/*.fits')
    index = np.where(data['pl_name'] == 'TOI-530 b')
    period = 6.387583
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax3_1 = fig.add_subplot(gs[2, :2])
    ax3_2 = fig.add_subplot(gs[2, 2:4])
    # ax1_3 = fig.add_subplot(gs[0, 6:9])
    ax3_4 = fig.add_subplot(gs[2, 4:6])
    ax3_5 = fig.add_subplot(gs[2, 6:8])
    ax3_6 = fig.add_subplot(gs[2, 8:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax3_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax3_2.plot(eleanor_6_t % period / period - phase_fold_mid, eleanor_6_f_pca, '.', c=color[0], markersize=2,
               label='6')
    ax3_2.plot(eleanor_44_t % period / period - phase_fold_mid, eleanor_44_f_pca, '.', c=color[1], markersize=2,
               label='44')
    ax3_2.plot(eleanor_45_t % period / period - phase_fold_mid, eleanor_45_f_pca, '.', c=color[2], markersize=2,
               label='45')
    # ax3_3.plot(eleanor_6_t % period / period - phase_fold_mid, eleanor_6_f_psf, '.', c=color[0], markersize=2,
    #            label='6')
    # ax3_3.plot(eleanor_44_t % period / period - phase_fold_mid, eleanor_44_f_psf, '.', c=color[1], markersize=2,
    #            label='44')
    ax3_4.plot(qlp_6_t % period / period - phase_fold_mid, qlp_6_f, '.', c=color[0], markersize=2, label='6')

    ax3_5.plot(t_06 % period / period - phase_fold_mid, f_aper_06, '.', c=color[0], markersize=2, label='6')
    ax3_5.plot(t_44 % period / period - phase_fold_mid, f_aper_44, '.', c=color[1], markersize=2, label='44')
    ax3_5.plot(t_45 % period / period - phase_fold_mid, f_aper_45, '.', c=color[2], markersize=2, label='45')

    ax3_6.plot(t_06 % period / period - phase_fold_mid, f_psf_06, '.', c=color[0], markersize=2, label='6')
    ax3_6.plot(t_44 % period / period - phase_fold_mid, f_psf_44, '.', c=color[1], markersize=2, label='44')
    ax3_6.plot(t_45 % period / period - phase_fold_mid, f_psf_45, '.', c=color[2], markersize=2, label='45')

    ax3_1.legend(loc=3, fontsize=6)
    ax3_2.legend(loc=3, fontsize=6)
    # ax3_3.legend(loc=3, fontsize=6)
    ax3_4.legend(loc=3, fontsize=6)
    ax3_5.legend(loc=3, fontsize=6)
    ax3_6.legend(loc=3, fontsize=6)
    ax3_1.set_ylim(0.95, 1.03)
    ax3_2.set_ylim(0.95, 1.03)
    # ax3_3.set_ylim(0.95, 1.03)
    ax3_4.set_ylim(0.95, 1.03)
    ax3_5.set_ylim(0.95, 1.03)
    ax3_6.set_ylim(0.95, 1.03)
    ax3_1.set_xlim(- 0.03, 0.03)
    ax3_2.set_xlim(- 0.03, 0.03)
    # ax3_3.set_xlim(- 0.03, 0.03)
    ax3_4.set_xlim(- 0.03, 0.03)
    ax3_5.set_xlim(- 0.03, 0.03)
    ax3_6.set_xlim(- 0.03, 0.03)
    ax3_2.set_yticklabels([])
    # ax3_3.set_yticklabels([])
    ax3_4.set_yticklabels([])
    ax3_5.set_yticklabels([])
    ax3_6.set_yticklabels([])
    ax3_1.set_ylabel('Normalized Flux')
    ax3_5.text(2.25, 0.5, 'TOI-530 b', horizontalalignment='center',
               verticalalignment='center', transform=ax3_5.transAxes, rotation=270, fontweight='semibold')
    ax3_5.text(2.15, 0.5, 'mag=13.53', horizontalalignment='center',
               verticalalignment='center', transform=ax3_5.transAxes, rotation=270)
    #########################################################################
    # TOI-2406
    tic = 212957629

    # load QLP
    qlp_30_t, qlp_30_f = load_qlp(ld=local_directory, tic=tic, sector=30)

    # load eleanor
    eleanor_3_t, eleanor_3_f_pca, eleanor_3_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=3)
    eleanor_42_t, eleanor_42_f_pca, eleanor_42_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=42)
    eleanor_43_t, eleanor_43_f_pca, eleanor_43_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=43)

    # load TGLC
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0003*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_03 = hdul[1].data['time'][q]
        f_aper_03 = hdul[1].data['cal_aper_flux'][q]
        f_psf_03 = hdul[1].data['cal_psf_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0042*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_42 = hdul[1].data['time'][q]
        f_aper_42 = hdul[1].data['cal_aper_flux'][q]
        f_psf_42 = hdul[1].data['cal_psf_flux'][q]
        t_42 = np.mean(t_42[:len(t_42) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_42 = np.mean(f_aper_42[:len(f_aper_42) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_42 = np.mean(f_psf_42[:len(f_psf_42) // 3 * 3].reshape(-1, 3), axis=1)
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-2528453161326406016-s0043*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_43 = hdul[1].data['time'][q]
        f_aper_43 = hdul[1].data['cal_aper_flux'][q]
        f_psf_43 = hdul[1].data['cal_psf_flux'][q]
        t_43 = np.mean(t_43[:len(t_43) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_43 = np.mean(f_aper_43[:len(f_aper_43) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_43 = np.mean(f_psf_43[:len(f_psf_43) // 3 * 3].reshape(-1, 3), axis=1)

    files = glob(local_directory + 'SPOC/TOI-2406/*.fits')
    index = np.where(data['pl_name'] == 'TOI-2406 b')
    period = 3.076676
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax4_1 = fig.add_subplot(gs[3, :2])
    ax4_2 = fig.add_subplot(gs[3, 2:4])
    # ax4_3 = fig.add_subplot(gs[3, 6:9])
    ax4_4 = fig.add_subplot(gs[3, 4:6])
    ax4_5 = fig.add_subplot(gs[3, 6:8])
    ax4_6 = fig.add_subplot(gs[3, 8:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax4_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i + 1],
                       ms=2,
                       label=str(hdul[0].header['sector']))
    ax4_2.plot(eleanor_3_t % period / period - phase_fold_mid, eleanor_3_f_pca, '.', c=color[0], markersize=2,
               label='3')
    ax4_2.plot(eleanor_42_t % period / period - phase_fold_mid, eleanor_42_f_pca, '.', c=color[1], markersize=2,
               label='42')
    ax4_2.plot(eleanor_43_t % period / period - phase_fold_mid, eleanor_43_f_pca, '.', c=color[2], markersize=2,
               label='43')
    # ax4_3.plot(eleanor_3_t % period / period - phase_fold_mid, eleanor_3_f_psf, '.', c=color[0], markersize=2,
    #            label='3')
    # ax4_3.plot(eleanor_42_t % period / period - phase_fold_mid, eleanor_42_f_psf, '.', c=color[1], markersize=2,
    #            label='42')
    # ax4_3.plot(eleanor_43_t % period / period - phase_fold_mid, eleanor_43_f_psf, '.', c=color[2], markersize=2,
    #            label='43')
    ax4_4.plot(qlp_30_t % period / period - phase_fold_mid, qlp_30_f, '.', c=color[0], markersize=2, label='30')

    ax4_5.plot(t_03 % period / period - phase_fold_mid, f_aper_03, '.', c=color[0], markersize=2,
               label='3')
    ax4_5.plot(t_42 % period / period - phase_fold_mid, f_aper_42, '.', c=color[1], markersize=2,
               label='42')
    ax4_5.plot(t_43 % period / period - phase_fold_mid, f_aper_43, '.', c=color[2], markersize=2,
               label='43')

    ax4_6.plot(t_03 % period / period - phase_fold_mid, f_psf_03, '.', c=color[0], markersize=2,
               label='3')
    ax4_6.plot(t_42 % period / period - phase_fold_mid, f_psf_42, '.', c=color[1], markersize=2,
               label='42')
    ax4_6.plot(t_43 % period / period - phase_fold_mid, f_psf_43, '.', c=color[2], markersize=2,
               label='43')

    ax4_1.legend(loc=3, fontsize=6)
    ax4_2.legend(loc=3, fontsize=6)
    # ax4_3.legend(loc=3, fontsize=6)
    ax4_4.legend(loc=3, fontsize=6)
    ax4_5.legend(loc=3, fontsize=6)
    ax4_6.legend(loc=3, fontsize=6)
    ax4_1.set_ylim(0.945, 1.04)
    ax4_2.set_ylim(0.945, 1.04)
    # ax4_3.set_ylim(0.945, 1.04)
    ax4_4.set_ylim(0.945, 1.04)
    ax4_5.set_ylim(0.945, 1.04)
    ax4_6.set_ylim(0.945, 1.04)
    ax4_1.set_xlim(- 0.04, 0.04)
    ax4_2.set_xlim(- 0.04, 0.04)
    # ax4_3.set_xlim(- 0.04, 0.04)
    ax4_4.set_xlim(- 0.04, 0.04)
    ax4_5.set_xlim(- 0.04, 0.04)
    ax4_6.set_xlim(- 0.04, 0.04)
    ax4_1.set_xticks([-0.03, 0, 0.03])
    ax4_1.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_2.set_xticks([-0.03, 0, 0.03])
    ax4_2.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    # ax4_3.set_xticks([-0.03, 0, 0.03])
    # ax4_3.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_4.set_xticks([-0.03, 0, 0.03])
    ax4_4.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_4.set_xlabel('Phase')
    ax4_5.set_xticks([-0.03, 0, 0.03])
    ax4_5.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_6.set_xticks([-0.03, 0, 0.03])
    ax4_6.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax4_2.set_yticklabels([])
    # ax4_3.set_yticklabels([])
    ax4_4.set_yticklabels([])
    ax4_5.set_yticklabels([])
    ax4_6.set_yticklabels([])
    ax4_1.set_ylabel('Normalized Flux')
    ax4_5.text(2.25, 0.5, 'TOI-2406 b', horizontalalignment='center',
               verticalalignment='center', transform=ax4_5.transAxes, rotation=270, fontweight='semibold')
    ax4_5.text(2.15, 0.5, 'mag=14.31', horizontalalignment='center',
               verticalalignment='center', transform=ax4_5.transAxes, rotation=270)

    #########################################################################
    # TOI-519
    tic = 218795833

    # load eleanor
    eleanor_7_t, eleanor_7_f_aper, eleanor_7_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=7)
    eleanor_8_t, eleanor_8_f_aper, eleanor_8_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=8)
    eleanor_34_t, eleanor_34_f_aper, eleanor_34_f_psf = load_eleanor(ld=local_directory, tic=tic, sector=34)

    # load TGLC
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0007*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_07 = hdul[1].data['time'][q]
        f_aper_07 = hdul[1].data['cal_aper_flux'][q]
        f_psf_07 = hdul[1].data['cal_psf_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0008*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_08 = hdul[1].data['time'][q]
        f_aper_08 = hdul[1].data['cal_aper_flux'][q]
        f_psf_08 = hdul[1].data['cal_psf_flux'][q]
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5707485527450614656-s0034*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_34 = hdul[1].data['time'][q]
        f_aper_34 = hdul[1].data['cal_aper_flux'][q]
        f_psf_34 = hdul[1].data['cal_psf_flux'][q]
        t_34 = np.mean(t_34[:len(t_34) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_34 = np.mean(f_aper_34[:len(f_aper_34) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_34 = np.mean(f_psf_34[:len(f_psf_34) // 3 * 3].reshape(-1, 3), axis=1)

    files = glob(local_directory + 'SPOC/TOI-519/*.fits')
    index = np.where(data['pl_name'] == 'TOI-519 b')
    period = 1.265232
    t_0 = float(data['pl_tranmid'][index])
    phase_fold_mid = (t_0 - 2457000) % period / period
    ax5_1 = fig.add_subplot(gs[4, :2])
    ax5_2 = fig.add_subplot(gs[4, 2:4])
    # ax5_3 = fig.add_subplot(gs[4, 6:9])
    # ax5_4 = fig.add_subplot(gs[4, 9:])
    ax5_5 = fig.add_subplot(gs[4, 6:8])
    ax5_6 = fig.add_subplot(gs[4, 8:])

    for i in range(len(files)):
        with fits.open(files[i], mode='denywrite') as hdul:
            if hdul[0].header['sector'] == 34:
                i = i + 1
            spoc_t = hdul[1].data['TIME']
            spoc_f = hdul[1].data['PDCSAP_FLUX']
            spoc_t = np.mean(spoc_t[:len(spoc_t) // 15 * 15].reshape(-1, 15), axis=1)
            spoc_f = np.mean(spoc_f[:len(spoc_f) // 15 * 15].reshape(-1, 15), axis=1)
            ax5_1.plot(spoc_t % period / period - phase_fold_mid, spoc_f / np.nanmedian(spoc_f), '.', c=color[i], ms=2,
                       label=str(hdul[0].header['sector']))
    ax5_2.plot(eleanor_7_t % period / period - phase_fold_mid, eleanor_7_f_aper, '.', c=color[0], markersize=2,
               label='7')
    ax5_2.plot(eleanor_8_t % period / period - phase_fold_mid, eleanor_8_f_aper, '.', c=color[1], markersize=2,
               label='8')
    ax5_2.plot(eleanor_34_t % period / period - phase_fold_mid, eleanor_34_f_aper, '.', c=color[2], markersize=2,
               label='34')
    # ax5_3.plot(eleanor_7_t % period / period - phase_fold_mid, eleanor_7_f_psf, '.', c=color[0], markersize=2,
    #            label='7')
    # ax5_3.plot(eleanor_8_t % period / period - phase_fold_mid, eleanor_8_f_psf, '.', c=color[1], markersize=2,
    #            label='8')
    # ax5_3.plot(eleanor_34_t % period / period - phase_fold_mid, eleanor_34_f_psf, '.', c=color[2], markersize=2,
    #            label='34')

    ax5_5.plot(t_07 % period / period - phase_fold_mid, f_aper_07, '.', c=color[0], markersize=2,
               label='7')
    ax5_5.plot(t_08 % period / period - phase_fold_mid, f_aper_08, '.', c=color[1], markersize=2,
               label='8')
    ax5_5.plot(t_34 % period / period - phase_fold_mid, f_aper_34, '.', c=color[2], markersize=2,
               label='34')

    ax5_6.plot(t_07 % period / period - phase_fold_mid, f_psf_07, '.', c=color[0], markersize=2,
               label='7')
    ax5_6.plot(t_08 % period / period - phase_fold_mid, f_psf_08, '.', c=color[1], markersize=2,
               label='8')
    ax5_6.plot(t_34 % period / period - phase_fold_mid, f_psf_34, '.', c=color[2], markersize=2,
               label='34')

    ax5_1.legend(loc=3, fontsize=6)
    ax5_2.legend(loc=3, fontsize=6)
    # ax5_3.legend(loc=3, fontsize=6)
    ax5_5.legend(loc=3, fontsize=6)
    ax5_6.legend(loc=3, fontsize=6)
    ax5_1.set_ylim(0.83, 1.05)
    ax5_2.set_ylim(0.83, 1.05)
    # ax5_3.set_ylim(0.83, 1.05)
    ax5_5.set_ylim(0.83, 1.05)
    ax5_6.set_ylim(0.83, 1.05)
    ax5_1.set_xlim(- 0.05, 0.05)
    ax5_2.set_xlim(- 0.05, 0.05)
    # ax5_3.set_xlim(- 0.05, 0.05)
    ax5_5.set_xlim(- 0.05, 0.05)
    ax5_6.set_xlim(- 0.05, 0.05)
    ax5_1.set_yticks([0.9, 1.0])
    ax5_1.set_yticklabels(['0.90', '1.00'])
    ax5_1.set_xticks([-0.03, 0, 0.03])
    ax5_1.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_2.set_xticks([-0.03, 0, 0.03])
    ax5_2.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    # ax5_3.set_xticks([-0.03, 0, 0.03])
    # ax5_3.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_5.set_xticks([-0.03, 0, 0.03])
    ax5_5.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_6.set_xticks([-0.03, 0, 0.03])
    ax5_6.set_xticklabels(['\N{MINUS SIGN}0.03', '0', '0.03'])
    ax5_2.set_yticklabels([])
    # ax5_3.set_yticklabels([])
    ax5_5.set_yticklabels([])
    ax5_6.set_yticklabels([])
    ax5_1.set_xlabel('Phase')
    ax5_2.set_xlabel('Phase')
    # ax5_3.set_xlabel('Phase')
    ax5_5.set_xlabel('Phase')
    ax5_6.set_xlabel('Phase')
    ax5_1.set_ylabel('Normalized Flux')
    ax5_5.text(2.25, 0.5, 'TOI-519 b', horizontalalignment='center',
               verticalalignment='center', transform=ax5_5.transAxes, rotation=270, fontweight='semibold')
    ax5_5.text(2.15, 0.5, 'mag=14.43', horizontalalignment='center',
               verticalalignment='center', transform=ax5_5.transAxes, rotation=270)
    # ax5_1.set_yticklabels([])

    plt.savefig('/mnt/c/users/tehan/desktop/known_exoplanets_all.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_10():
    size = 90
    local_directory = '/home/tehan/data/variables/'
    os.makedirs(local_directory + f'lc/', exist_ok=True)
    os.makedirs(local_directory + f'epsf/', exist_ok=True)
    os.makedirs(local_directory + f'source/', exist_ok=True)
    hosts = [
        ('SX Dor', 'Gaia DR2 4662259606266850944'),  # Molnar: RR Lyrae, hard to disdinguish for eleanor
        ('TIC 177309964', 'Gaia DR2 5260885172921947008'),  # Zhan: Faint rotator
        ('AV Gru', 'Gaia DR2 6512192214932460416')  # Plachy: Cepheid, dim and kind of noisy
    ]
    #####################
    # 3 6 7 8 9 10 17 27 28 34 36 42 43 44 45
    sectors = [2, 11, 38]
    for sector in sectors:
        source = ffi_cut(target=hosts[0][0], size=size, local_directory=local_directory, sector=sector)
        epsf(source, factor=2, sector=source.sector, target=hosts[0][0], local_directory=local_directory,
             name=hosts[0][1], save_aper=True)

    #####################
    sectors = [4, 12, 31]
    for sector in sectors:
        source = ffi_cut(target=hosts[1][0], size=size, local_directory=local_directory, sector=sector)
        epsf(source, factor=2, sector=source.sector, target=hosts[1][0], local_directory=local_directory,
             name=hosts[1][1], save_aper=True)

    #####################
    sectors = [1, 28]
    for sector in sectors:
        source = ffi_cut(target=hosts[2][0], size=size, local_directory=local_directory, sector=sector)
        epsf(source, factor=2, sector=source.sector, target=hosts[2][0], local_directory=local_directory,
             name=hosts[2][1], save_aper=True)

    fig = plt.figure(constrained_layout=False, figsize=(10, 6))
    gs = fig.add_gridspec(8, 9, wspace=0.2, hspace=0, height_ratios=[1, 1, 0.8, 1, 1, 0.8, 1, 1])
    local_directory = '/home/tehan/data/variables/'
    # local_directory = '/mnt/c/users/tehan/desktop/variables/'
    ##########
    period = 0.63150
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-4662259606266850944-s0002*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_02 = hdul[1].data['time'][q]
        f_psf_02 = hdul[1].data['cal_psf_flux'][q]
        f_aper_02 = hdul[1].data['cal_aper_flux'][q]

    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-4662259606266850944-s0011*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_11 = hdul[1].data['time'][q]
        f_psf_11 = hdul[1].data['cal_psf_flux'][q]
        f_aper_11 = hdul[1].data['cal_aper_flux'][q]

    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-4662259606266850944-s0038*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_38 = hdul[1].data['time'][q]
        f_psf_38 = hdul[1].data['cal_psf_flux'][q]
        f_aper_38 = hdul[1].data['cal_aper_flux'][q]
        t_38 = np.mean(t_38[:len(t_38) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_38 = np.mean(f_psf_38[:len(f_psf_38) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_38 = np.mean(f_aper_38[:len(f_aper_38) // 3 * 3].reshape(-1, 3), axis=1)

    ax0_1 = fig.add_subplot(gs[0, :3])
    ax0_2 = fig.add_subplot(gs[0, 3:6])
    ax0_3 = fig.add_subplot(gs[0, 6:9])
    ax1_1 = fig.add_subplot(gs[1, :3])
    ax1_2 = fig.add_subplot(gs[1, 3:6])
    ax1_3 = fig.add_subplot(gs[1, 6:9])

    ax0_1.plot(t_02, f_aper_02, '.', c='k', markersize=1, label='2')
    ax0_2.plot(t_11, f_aper_11, '.', c='k', markersize=1, label='11')
    ax0_3.plot(t_38, f_aper_38, '.', c='k', markersize=1, label='38')

    ax1_1.plot(t_02, f_psf_02, '.', c='k', markersize=1, label='2')
    ax1_2.plot(t_11, f_psf_11, '.', c='k', markersize=1, label='11')
    ax1_3.plot(t_38, f_psf_38, '.', c='k', markersize=1, label='38')

    # split
    low = 0.5
    high = 1.95
    ax0_1.spines['right'].set_visible(False)
    ax0_2.spines['left'].set_visible(False)
    ax0_2.spines['right'].set_visible(False)
    ax0_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax0_1.plot([1, 1], [0, 1], transform=ax0_1.transAxes, **kwargs)
    ax0_2.plot([0, 0], [0, 1], transform=ax0_2.transAxes, **kwargs)
    ax0_2.plot([1, 1], [0, 1], transform=ax0_2.transAxes, **kwargs)
    ax0_3.plot([0, 0], [0, 1], transform=ax0_3.transAxes, **kwargs)
    ax0_1.set_xticklabels([])
    ax0_2.set_yticklabels([])
    ax0_2.set_xticklabels([])
    ax0_2.tick_params(axis='y', left=False)
    ax0_3.set_yticklabels([])
    ax0_3.set_xticklabels([])
    ax0_3.tick_params(axis='y', left=False)
    ax0_1.set_ylim(low, high)
    ax0_2.set_ylim(low, high)
    ax0_3.set_ylim(low, high)
    ax0_1.set_title('Sector 2')
    ax0_2.set_title('Sector 11')
    ax0_3.set_title('Sector 38')
    ax0_1.text(-0.2, 0.5, 'aperture', horizontalalignment='center',
               verticalalignment='center', transform=ax0_1.transAxes, rotation=90)
    ax0_2.text(2.18, 0, f'{hosts[0][0]}', horizontalalignment='center',
               verticalalignment='center', transform=ax0_2.transAxes, rotation=270, fontweight='semibold')
    ax0_2.text(2.1, 0, 'RR Lyrae', horizontalalignment='center',
               verticalalignment='center', transform=ax0_2.transAxes, rotation=270)

    ax1_1.spines['right'].set_visible(False)
    ax1_2.spines['left'].set_visible(False)
    ax1_2.spines['right'].set_visible(False)
    ax1_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1_1.plot([1, 1], [0, 1], transform=ax1_1.transAxes, **kwargs)
    ax1_2.plot([0, 0], [0, 1], transform=ax1_2.transAxes, **kwargs)
    ax1_2.plot([1, 1], [0, 1], transform=ax1_2.transAxes, **kwargs)
    ax1_3.plot([0, 0], [0, 1], transform=ax1_3.transAxes, **kwargs)
    ax1_2.set_yticklabels([])
    ax1_2.tick_params(axis='y', left=False)
    ax1_3.set_yticklabels([])
    ax1_3.tick_params(axis='y', left=False)
    ax1_1.set_ylim(low, high)
    ax1_2.set_ylim(low, high)
    ax1_3.set_ylim(low, high)
    ax1_1.text(-0.2, 0.5, 'PSF', horizontalalignment='center',
               verticalalignment='center', transform=ax1_1.transAxes, rotation=90)

    ##########
    period = 10.881 / 24
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5260885172921947008-s0004*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_04 = hdul[1].data['time'][q]
        f_psf_04 = hdul[1].data['cal_psf_flux'][q]
        f_aper_04 = hdul[1].data['cal_aper_flux'][q]

    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5260885172921947008-s0012*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_12 = hdul[1].data['time'][q]
        f_psf_12 = hdul[1].data['cal_psf_flux'][q]
        f_aper_12 = hdul[1].data['cal_aper_flux'][q]

    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-5260885172921947008-s0031*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_31 = hdul[1].data['time'][q]
        f_psf_31 = hdul[1].data['cal_psf_flux'][q]
        f_aper_31 = hdul[1].data['cal_aper_flux'][q]
        t_31 = np.mean(t_31[:len(t_31) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_31 = np.mean(f_psf_31[:len(f_psf_31) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_31 = np.mean(f_aper_31[:len(f_aper_31) // 3 * 3].reshape(-1, 3), axis=1)

    ax3_1 = fig.add_subplot(gs[3, :3])
    ax3_2 = fig.add_subplot(gs[3, 3:6])
    ax3_3 = fig.add_subplot(gs[3, 6:])

    ax4_1 = fig.add_subplot(gs[4, :3])
    ax4_2 = fig.add_subplot(gs[4, 3:6])
    ax4_3 = fig.add_subplot(gs[4, 6:])

    ax3_1.plot(t_04, f_aper_04, '.', c='k', markersize=1, label='4')
    ax3_2.plot(t_12, f_aper_12, '.', c='k', markersize=1, label='12')
    ax3_3.plot(t_31, f_aper_31, '.', c='k', markersize=1, label='31')

    ax4_1.plot(t_04, f_psf_04, '.', c='k', markersize=1, label='4')
    ax4_2.plot(t_12, f_psf_12, '.', c='k', markersize=1, label='12')
    ax4_3.plot(t_31, f_psf_31, '.', c='k', markersize=1, label='31')

    # split
    low = 0.88
    high = 1.09
    ax3_1.spines['right'].set_visible(False)
    ax3_2.spines['left'].set_visible(False)
    ax3_2.spines['right'].set_visible(False)
    ax3_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax3_1.plot([1, 1], [0, 1], transform=ax3_1.transAxes, **kwargs)
    ax3_2.plot([0, 0], [0, 1], transform=ax3_2.transAxes, **kwargs)
    ax3_2.plot([1, 1], [0, 1], transform=ax3_2.transAxes, **kwargs)
    ax3_3.plot([0, 0], [0, 1], transform=ax3_3.transAxes, **kwargs)
    ax3_1.set_xticklabels([])
    ax3_2.set_yticklabels([])
    ax3_2.set_xticklabels([])
    ax3_2.tick_params(axis='y', left=False)
    ax3_3.set_yticklabels([])
    ax3_3.set_xticklabels([])
    ax3_3.tick_params(axis='y', left=False)
    ax3_1.set_ylim(low, high)
    ax3_2.set_ylim(low, high)
    ax3_3.set_ylim(low, high)
    ax3_1.set_title('Sector 4')
    ax3_2.set_title('Sector 12')
    ax3_3.set_title('Sector 31')
    ax3_1.text(-0.2, 0.5, 'aperture', horizontalalignment='center',
               verticalalignment='center', transform=ax3_1.transAxes, rotation=90)
    ax3_2.text(2.18, 0, f'{hosts[1][0]}', horizontalalignment='center',
               verticalalignment='center', transform=ax3_2.transAxes, rotation=270, fontweight='semibold')
    ax3_2.text(2.1, 0, 'Rotator', horizontalalignment='center',
               verticalalignment='center', transform=ax3_2.transAxes, rotation=270)

    ax4_1.spines['right'].set_visible(False)
    ax4_2.spines['left'].set_visible(False)
    ax4_2.spines['right'].set_visible(False)
    ax4_3.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax4_1.plot([1, 1], [0, 1], transform=ax4_1.transAxes, **kwargs)
    ax4_2.plot([0, 0], [0, 1], transform=ax4_2.transAxes, **kwargs)
    ax4_2.plot([1, 1], [0, 1], transform=ax4_2.transAxes, **kwargs)
    ax4_3.plot([0, 0], [0, 1], transform=ax4_3.transAxes, **kwargs)
    ax4_2.set_yticklabels([])
    ax4_2.tick_params(axis='y', left=False)
    ax4_3.set_yticklabels([])
    ax4_3.tick_params(axis='y', left=False)
    ax4_1.set_ylim(low, high)
    ax4_2.set_ylim(low, high)
    ax4_3.set_ylim(low, high)
    ax4_3.set_xlabel('TBJD')
    ax4_1.text(-0.2, 0.5, 'PSF', horizontalalignment='center',
               verticalalignment='center', transform=ax4_1.transAxes, rotation=90)

    ##########
    period = 1.00581
    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6512192214932460416-s0001*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_01 = hdul[1].data['time'][q]
        f_psf_01 = hdul[1].data['cal_psf_flux'][q]
        # f_psf_01 = f_psf_01 + hdul[1].header['LOC_BG']
        # f_psf_01 = flatten(t_01, f_psf_01 / np.nanmedian(f_psf_01), window_length=1, method='biweight',
        #                    return_trend=False)
        f_aper_01 = hdul[1].data['cal_aper_flux'][q]

    with fits.open(glob(f'{local_directory}lc/hlsp_tglc_tess_ffi_gaiaid-6512192214932460416-s0028*.fits')[0],
                   mode='denywrite') as hdul:
        q = list(hdul[1].data['TESS_flags'] == 0) and list(hdul[1].data['TGLC_flags'] == 0)
        t_28 = hdul[1].data['time'][q]
        f_psf_28 = hdul[1].data['cal_psf_flux'][q]
        f_aper_28 = hdul[1].data['cal_aper_flux'][q]
        t_28 = np.mean(t_28[:len(t_28) // 3 * 3].reshape(-1, 3), axis=1)
        f_psf_28 = np.mean(f_psf_28[:len(f_psf_28) // 3 * 3].reshape(-1, 3), axis=1)
        f_aper_28 = np.mean(f_aper_28[:len(f_aper_28) // 3 * 3].reshape(-1, 3), axis=1)

    ax6_1 = fig.add_subplot(gs[6, :3])
    ax6_2 = fig.add_subplot(gs[6, 3:6])
    ax7_1 = fig.add_subplot(gs[7, :3])
    ax7_2 = fig.add_subplot(gs[7, 3:6])

    ax6_1.plot(t_01, f_aper_01, '.', c='k', markersize=1, label='1')
    ax6_2.plot(t_28, f_aper_28, '.', c='k', markersize=1, label='28')

    ax7_1.plot(t_01, f_psf_01, '.', c='k', markersize=1, label='1')
    ax7_2.plot(t_28, f_psf_28, '.', c='k', markersize=1, label='28')

    # split
    low = 0.45
    high = 2.4
    ax6_1.spines['right'].set_visible(False)
    ax6_2.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax6_1.plot([1, 1], [0, 1], transform=ax6_1.transAxes, **kwargs)
    ax6_2.plot([0, 0], [0, 1], transform=ax6_2.transAxes, **kwargs)
    ax6_1.set_xticklabels([])
    ax6_2.set_yticklabels([])
    ax6_2.set_xticklabels([])
    ax6_2.tick_params(axis='y', left=False)
    ax6_1.set_ylim(low, high)
    ax6_2.set_ylim(low, high)
    ax6_1.set_title('Sector 1')
    ax6_2.set_title('Sector 28')
    ax6_1.text(-0.2, 0.5, 'aperture', horizontalalignment='center',
               verticalalignment='center', transform=ax6_1.transAxes, rotation=90)
    ax6_2.text(2.18, 0, f'{hosts[2][0]}', horizontalalignment='center',
               verticalalignment='center', transform=ax6_2.transAxes, rotation=270, fontweight='semibold')
    ax6_2.text(2.1, 0, 'Cepheid', horizontalalignment='center',
               verticalalignment='center', transform=ax6_2.transAxes, rotation=270)

    ax7_1.spines['right'].set_visible(False)
    ax7_2.spines['left'].set_visible(False)
    d = .7  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax7_1.plot([1, 1], [0, 1], transform=ax7_1.transAxes, **kwargs)
    ax7_2.plot([0, 0], [0, 1], transform=ax7_2.transAxes, **kwargs)
    ax7_2.set_yticklabels([])
    ax7_2.tick_params(axis='y', left=False)
    ax7_1.set_ylim(low, high)
    ax7_2.set_ylim(low, high)
    ax7_1.set_xlabel('TBJD')
    ax7_2.set_xlabel('TBJD')
    ax7_1.text(-0.2, 0.5, 'PSF', horizontalalignment='center',
               verticalalignment='center', transform=ax7_1.transAxes, rotation=90)

    plt.savefig(f'{local_directory}variables.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_11():
    with open(f'/mnt/c/users/tehan/desktop/source_00_03.pkl', 'rb') as input_:
        source = pickle.load(input_)
    fig = plt.figure(constrained_layout=False, figsize=(13, 4))
    gs = fig.add_gridspec(1, 35)
    gs.update(wspace=1, hspace=0.1)
    vmax = 140
    vmin = 100

    ax1 = fig.add_subplot(gs[0, 0:10])
    ax1.imshow(np.nanmedian(source.flux, axis=0), vmin=vmin, vmax=vmax, origin='lower', cmap='viridis')
    ax1.set_title('TESS FFI cutout')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.tick_params(axis='x', bottom=False)
    ax1.tick_params(axis='y', left=False)
    # ax1.set_ylabel('Pixels')
    # ax1.set_xlabel('Pixels')
    # ax1.set_xlim(50,75)
    # ax1.set_ylim(80,105)
    mask = source.mask.data
    mask[source.mask.mask] = 0
    # ax2.set_xlim(50,75)
    # ax2.set_ylim(80,105)

    bg = np.load('/mnt/c/users/tehan/desktop/bg_00_03_sector_2.npy')
    ax2 = fig.add_subplot(gs[0, 10:20])
    im2 = ax2.imshow(bg[:150 ** 2, 0].reshape(150, 150), vmin=vmin, vmax=vmax, origin='lower', cmap='viridis')
    ax2.set_title('Simulated background')
    # ax2.set_xlabel('Pixels')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', bottom=False)
    ax2.tick_params(axis='y', left=False)

    ax_cb = fig.colorbar(im2, cax=fig.add_subplot(gs[0, 20]), orientation='vertical',
                         boundaries=np.linspace(vmin, vmax, 1000),
                         ticks=[100, 110, 120, 130, 140], aspect=50, shrink=0.7)
    ax_cb.ax.set_yticklabels(['100', '110', '120', '130', r'$\geq 140$'])

    # ax_cb.ax.set_ylabel(r'TESS Flux ($\mathrm{e^-}$/ s) ')
    vmin = 0
    vmax = 40
    ax3 = fig.add_subplot(gs[0, 24:34])
    im3 = ax3.imshow(np.nanmedian(source.flux, axis=0) - bg[:150 ** 2, 0].reshape(150, 150), vmin=vmin, vmax=vmax,
                     origin='lower', cmap='viridis')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.tick_params(axis='x', bottom=False)
    ax3.tick_params(axis='y', left=False)
    ax3.set_title('Background removed FFI')
    # ax3.set_xlabel('Pixels')

    ax_cb = fig.colorbar(im3, cax=fig.add_subplot(gs[0, 34]), orientation='vertical',
                         boundaries=np.linspace(vmin, vmax, 1000),
                         ticks=[0, 10, 20, 30, 40], aspect=50, shrink=0.7)
    ax_cb.ax.set_yticklabels(['0', '10', '20', '30', r'$\geq 40$'])
    ax_cb.ax.set_ylabel(r'TESS Flux ($\mathrm{e^-}$/ s) ')
    # plt.savefig('/mnt/c/users/tehan/desktop/cal_bg.png', bbox_inches='tight', dpi=300)
    plt.show()


def figure_12():
    local_directory = '/home/tehan/data/sector0001/'
    camccd = '4-3'
    sector = 1
    fig = plt.figure(constrained_layout=False, figsize=(9, 9))
    gs = fig.add_gridspec(14, 14)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(196):
        cut_x = i // 14
        cut_y = i % 14
        psf = np.load(f'{local_directory}epsf/{camccd}/epsf_{cut_x:02d}_{cut_y:02d}_sector_{sector}_{camccd}.npy')
        cmap = 'bone'
        if np.isnan(psf).any():
            cmap = 'inferno'
        ax = fig.add_subplot(gs[13 - cut_y, cut_x])
        ax.imshow(psf[0, :23 ** 2].reshape(23, 23), cmap=cmap, origin='lower')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', left=False)
    plt.savefig(f'{local_directory}/epsf_examples.png', bbox_inches='tight', dpi=300)


def figure_13():
    files = glob.glob('/mnt/c/users/tehan/desktop/powers/*.npy')
    med_mad = np.load('/mnt/c/users/tehan/desktop/median_mad.npy')
    powers = np.linspace(0.4, 2, 30)
    plt.figure(constrained_layout=False, figsize=(5, 4))
    for i in range(196):
        residual = np.load(files[i])
        plt.plot(powers, residual / np.median(residual), c='C0', alpha=0.1)
    plt.plot(1, 0, c='C0', alpha=1, label='MAD of each cutout')
    plt.plot(powers, med_mad / np.median(med_mad), c='C1', lw=2.5, label='MAD of all cutouts')
    plt.legend()
    plt.xlabel(r'weighting power $l$')
    plt.ylabel('Normalized MAD of residual image')
    plt.ylim(0.86, 1.8)
    plt.savefig(f'/mnt/c/users/tehan/desktop/powers.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    figure_3()

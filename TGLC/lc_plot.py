import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten
from astropy.io import ascii

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    period = 2.2903
    t1_ = 435
    t2_ = 455
    t3_ = 940

    t1 = 530
    t2 = 555
    t3 = 1080

    time = np.load('/mnt/c/users/tehan/desktop/eleanor_time.npy')
    eleanor_aperture = np.load('/mnt/c/users/tehan/desktop/eleanor_aperture_cross_1251.npy')
    eleanor_PSF = np.load('/mnt/c/users/tehan/desktop/eleanor_PSF_1251.npy')
    # moffat = np.load('/mnt/c/users/tehan/desktop/moffat_1251.npy')
    moffat = np.load('/mnt/c/users/tehan/desktop/mod_1251.npy')
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')
    qlp = ascii.read(
        '/mnt/c/users/tehan/desktop/hlsp_qlp_tess_ffi_s0017-0000000270023089_tess_v01_llc.txt')

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
    flatten_lc_ = moffat
    # eleanor gaussian
    flatten_lc__, trend_lc__ = flatten(time, eleanor_PSF / np.median(eleanor_PSF), window_length=1, method='biweight',
                                       return_trend=True)

    # qlp
    flatten_lc___, trend_lc___ = flatten(qlp['Time (BTJD)'], qlp['Normalized SAP_FLUX'], window_length=1,
                                       method='biweight',
                                       return_trend=True)

    fig = plt.figure(constrained_layout=False, figsize=(10, 9))
    gs = fig.add_gridspec(5, 5)
    gs.update(wspace=0.1, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[2, 0:3])
    ax4 = fig.add_subplot(gs[3, 0:3])
    ax5 = fig.add_subplot(gs[4, 0:3])

    ax6 = fig.add_subplot(gs[0, 3:5])
    ax7 = fig.add_subplot(gs[1, 3:5])
    ax8 = fig.add_subplot(gs[2, 3:5])
    ax9 = fig.add_subplot(gs[3, 3:5])
    ax10 = fig.add_subplot(gs[4, 3:5])


    ax1.plot(time, eleanor_aperture / np.median(eleanor_aperture), '.k', ms=2)
    ax2.plot(time, flatten_lc__, '.k', ms=2)
    ax3.plot(qlp['Time (BTJD)'], flatten_lc___, '.k', ms=2)
    ax4.plot(source.time, flatten_lc_, '.k', ms=2)
    ax5.plot(source.time, flatten_lc, '.k', ms=2)

    ax6.plot(time[0:t1_] % period, eleanor_aperture[0:t1_] / np.median(eleanor_aperture), '.k', ms=1)
    ax6.plot(time[t2_:t3_] % period, eleanor_aperture[t2_:t3_] / np.median(eleanor_aperture), '.k', ms=1)
    ax7.plot(time[0:t1_] % period, flatten_lc__[0:t1_], '.k', ms=1)
    ax7.plot(time[t2_:t3_] % period, flatten_lc__[t2_:t3_], '.k', ms=1)
    ax8.plot(qlp['Time (BTJD)'] % period, flatten_lc___, '.k', ms=1)
    ax9.plot(source.time[0:t1] % period, flatten_lc_[0:t1], '.k', ms=1)
    ax9.plot(source.time[t2:t3] % period, flatten_lc_[t2:t3], '.k', ms=1)
    ax10.plot(source.time[0:t1] % period, flatten_lc[0:t1], '.k', ms=1)
    ax10.plot(source.time[t2:t3] % period, flatten_lc[t2:t3], '.k', ms=1)

    ax1.set_title('eleanor aperture', loc='left')
    ax2.set_title('detrended eleanor Gaussian PSF', loc='left')
    ax3.set_title('detrended QLP', loc='left')
    ax4.set_title('detrended mod ePSF', loc='left')
    ax5.set_title('detrended ePSF', loc='left')
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

    ax1.set_ylabel('Normalized Flux')
    ax2.set_ylabel('Normalized Flux')
    ax3.set_ylabel('Normalized Flux')
    ax4.set_ylabel('Normalized Flux')
    ax5.set_ylabel('Normalized Flux')
    # ax1.set_xlabel('TBJD')
    # ax2.set_xlabel('TBJD')
    # ax3.set_xlabel('TBJD')
    ax5.set_xlabel('TBJD')
    # ax5.set_xlabel('Phase (days)')
    # ax6.set_xlabel('Phase (days)')
    # ax7.set_xlabel('Phase (days)')
    ax10.set_xlabel('Phase (days)')

    ax1.set_ylim(0.993, 1.006)
    ax6.set_ylim(0.993, 1.006)
    ax2.set_ylim(0.993, 1.006)
    ax7.set_ylim(0.993, 1.006)
    ax3.set_ylim(0.993, 1.006)
    ax8.set_ylim(0.993, 1.006)
    ax4.set_ylim(0.65, 1.15)
    ax9.set_ylim(0.65, 1.15)
    ax5.set_ylim(0.65, 1.15)
    ax10.set_ylim(0.65, 1.15)

    ax1.plot(time[t3_:], eleanor_aperture[t3_:] / np.median(eleanor_aperture), '.', c='silver', ms=2)
    ax1.plot(time[t1_:t2_], eleanor_aperture[t1_:t2_] / np.median(eleanor_aperture), '.', c='silver', ms=2)
    ax2.plot(time[t3_:], flatten_lc__[t3_:], '.', c='silver', ms=2)
    ax2.plot(time[t1_:t2_], flatten_lc__[t1_:t2_], '.', c='silver', ms=2)
    ax4.plot(source.time[t3:], flatten_lc_[t3:], '.', c='silver', ms=2)
    ax4.plot(source.time[t1:t2], flatten_lc_[t1:t2], '.', c='silver', ms=2)
    ax5.plot(source.time[t3:], flatten_lc[t3:], '.', c='silver', ms=2)
    ax5.plot(source.time[t1:t2], flatten_lc[t1:t2], '.', c='silver', ms=2)

    plt.savefig('/mnt/c/users/tehan/desktop/light_curve_comparison_1251_mod.png', dpi=300)
    plt.show()

    # time_arg = np.argsort(time % period)
    # moffat_time_arg = np.argsort(source.time % period)

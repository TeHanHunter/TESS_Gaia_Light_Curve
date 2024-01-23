import glob, os, re
import numpy as np
import pandas as pd
import lightkurve
import astropy.units as u
import matplotlib.pyplot as plt
from astropy import coordinates
from astroquery.mast import Catalogs
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import transitleastsquares
TESS_JD_OFFSET = 2457000.
import argparse

from .exoplanet_functions import get_lc_batman, i_from_aRs_and_b, rho_star_from_P_and_aRs, calc_transit_duration
from numpy import square, log, ndarray, pi
from math import log as mlog
LOG_TWO_PI = log(2*pi)

def ll_normal_es_py(o,m,e):
    """Normal log likelihood for scalar average standard deviation."""
    npt = o.size
    return -npt*mlog(e) -0.5*npt*LOG_TWO_PI - 0.5*square(o-m).sum()/e**2

def ll_normal_ev_py(o,m,e):
    """Normal log likelihood for varying e"""
    npt = o.size
    return -log(e).sum() -0.5*npt*LOG_TWO_PI - 0.5*square((o-m)/e).sum()


class LCLPF(object):
    """
    An object to fit transits to LK objects

    EXAMPLE:
        t = np.linspace(1,20,2000)
        P = 6.3
        T0 = 0.1#1690.
        rprs = 0.1#np.sqrt(0.005)
        b = 0.1
        rho = 1.
        aRs = exoplanet_functions.aRs_from_rho_and_P(rho,P)
        inc = exoplanet_functions.i_from_aRs_and_b(aRs,b)
        lcm = tess_help.lc_transit(t,P,T0,rprs,i=inc,rho=rho,error=0.001)
        lcm.plot()
        pv_known = [P,T0,rprs,b,aRs]
        pv_known_str = L.pv2str(pv_known)

        pg = lcm.to_periodogram('bls',minimum_period=0.9,duration=0.05,maximum_period=20.,oversample=20)
        P_bls = pg.period_at_max_power.value
        T0_bls = pg.transit_time_at_max_power
        print(P_bls,T0_bls)

        L = tess_help.LCLPF(lcm)
        pinit = [pg.period_at_max_power.value,pg.transit_time_at_max_power,np.sqrt(pg.depth_at_max_power),0.1,10.]
        L.optimize(pinit)
        L.plot_transit(L.min_pv)
        lmm = L.get_transit_lc(L.min_pv)

        fig, ax = plt.subplots()
        pg.plot(view='period',ax=ax)
        ax.axvline(pg.period_at_max_power.value,color='orange',alpha=0.4)

        fig, ax = plt.subplots()
        lcm.fold(L.min_pv[0],L.min_pv[1]).scatter(ax=ax)
        lmm.fold(L.min_pv[0],L.min_pv[1]).plot(ax=ax,color='crimson')
        ax.set_xlim(-0.05,0.05)

        ax.set_title('Deriv: {}\nKnown: {}'.format(L.min_pv_str,pv_known_str))
    """

    def __init__(self, lc, lsq_method, lsq_tol, lsq_maxiter, period_limits, limbdarkcoeff, supersample_factor,
                 exp_time):
        self.lc = lc.remove_nans()
        self.lsq_method = lsq_method
        self.lsq_tol = lsq_tol
        self.lsq_maxiter = lsq_maxiter
        self.period_limits = period_limits
        self.limbdarkcoeff = limbdarkcoeff
        self.supersample_factor = supersample_factor
        self.exp_time = exp_time
        if hasattr(self.lc.time, 'value'):
            if np.min(self.lc.time.value) < TESS_JD_OFFSET:
                self.time = self.lc.time.value
            else:
                self.time = self.lc.time.value - TESS_JD_OFFSET
        else:
            if np.min(self.lc.time) < TESS_JD_OFFSET:
                self.time = self.lc.time
            else:
                self.time = self.lc.time - TESS_JD_OFFSET

        self.mask = np.array([True for i in range(len(self.time))])
        self.time = self.time[self.mask]
        if hasattr(self.lc.flux, 'value'):
            self.flux = (self.lc.flux[self.mask].value).astype(float)
        else:
            self.flux = (self.lc.flux[self.mask]).astype(float)
        if hasattr(self.lc.flux, 'value'):
            self.flux_err = (self.lc.flux_err[self.mask].value).astype(float)
        else:
            self.flux_err = (self.lc.flux_err[self.mask]).astype(float)

    def get_transit(self, pv, time=None):
        if time is None:
            time = self.time
        for thisindex in range(len(pv)):
            if hasattr(pv[thisindex], 'value'):
                pv[thisindex] = pv[thisindex].value
        _P = pv[0]
        _T0 = pv[1]
        _rprs = pv[2]
        _b = pv[3]
        _aRs = pv[4]
        _inc = i_from_aRs_and_b(_aRs, _b)
        return get_lc_batman(time, _T0, _P, _inc, _rprs, _aRs, 0., 0., u=self.limbdarkcoeff,
                             supersample_factor=self.supersample_factor, exp_time=self.exp_time)

    def get_transit_lc(self, pv, time=None, highres=False):
        if highres:
            time = np.linspace(self.time.min(), self.time.max(), len(self.time) * 5)
        elif time is None:
            time = self.time
        f = self.get_transit(pv, time)
        return lightkurve.LightCurve(time=time, flux=f)

    def plot_transit(self, pv, time=None):
        f = self.get_transit(pv, time)
        fig, ax = plt.subplots()
        ax.plot(self.time, self.flux, alpha=0.5, marker='o', lw=0)
        ax.plot(self.time, f, color='crimson', alpha=0.8, lw=1)

    def get_lnlike(self, pv):
        _P = pv[0]
        _T0 = pv[1]
        _rprs = pv[2]
        _b = pv[3]
        _aRs = pv[4]
        pv = np.array(pv)
        if any(pv <= 0) or (_rprs > 1.) or not (0.75 < _P < 80.) or (_b > 1.5):
            return np.inf
        f = self.get_transit(pv)
        return -ll_normal_ev_py(self.flux, f, self.flux_err)

    def optimize(self, pv_init):
        for thisindex in range(len(pv_init)):
            if hasattr(pv_init[thisindex], 'value'):
                pv_init[thisindex] = pv_init[thisindex].value
        mr = minimize(self.get_lnlike, pv_init, method=self.lsq_method, tol=self.lsq_tol,
                      bounds=[self.period_limits, (None, None), (None, None), (None, None), (None, None)],
                      options={'maxiter': self.lsq_maxiter})
        pv_min = mr.x
        print(mr.message)
        lnlike, x = -mr.fun, mr.x
        self.min_pv = mr.x
        self.min_pv_dict = self.pv2dict(mr.x)
        self.min_pv_str = self.pv2str(mr.x)
        self.min_pv_res = self.pv2results(mr.x)
        self.min_pv_lc = self.get_transit_lc(mr.x)
        if len(self.time) < 5000:
            self.min_pv_lc_plot = self.get_transit_lc(mr.x, highres=True)
        else:
            self.min_pv_lc_plot = self.get_transit_lc(mr.x)
        return lnlike, x

    def pv2dict(self, pv):
        return {'pl_orbper': pv[0],
                'pl_tranmid': pv[1],
                'pl_ratror': pv[2],
                'pl_imppar': pv[3],
                'pl_ratdor': pv[4]}

    def pv2results(self, pv):
        inc = i_from_aRs_and_b(pv[4], pv[3])
        rho = rho_star_from_P_and_aRs(pv[0], pv[4])
        return {'pl_orbper': pv[0],
                'pl_tranmid': pv[1],
                'pl_ratror': pv[2],
                'pl_imppar': pv[3],
                'pl_ratdor': pv[4],
                'pl_trandep': 1 - self.get_transit(pv, np.atleast_1d(pv[1]))[0],  # Analytical depth
                'st_dens': rho,
                'pl_orbincl': inc,
                'pl_trandur': calc_transit_duration(pv[0], pv[2], pv[4], np.deg2rad(inc))}  # in Days

    def pv2str(self, pv):
        return 'P={:0.7f}d, T0={:0.5f}, RpRs={:0.5f}, b={:0.3f}, aRs={:0.2f}'.format(*pv)


class TESSBLS(object):
    """
    EXAMPLE:
        LB = tess_help.TESSBLS(lc)
        LB.run_bls()
    """

    def __init__(self, lc, method='L-BFGS-B', tol=1e-6, maxiter=None, period_limits=(None, None),
                 limbdarkcoeff=[0.3, 0.3], supersample_factor=1, exp_time=0):
        self.lc = lc.normalize()
        self.L = LCLPF(self.lc, lsq_method=method, lsq_tol=tol, lsq_maxiter=maxiter, period_limits=period_limits,
                       limbdarkcoeff=limbdarkcoeff, supersample_factor=supersample_factor, exp_time=exp_time)

    def run_bls(self, plot=True, minimum_period=0.5, maximum_period=30., tls=False, oversample=1, frequency_factor=10):
        print('#####')
        print('Running BLS')
        print('MinPeriod={}, MaxPeriod={}'.format(minimum_period, maximum_period))
        if not tls:
            self.pg = self.lc.to_periodogram('bls', minimum_period=minimum_period, duration=0.05,
                                             maximum_period=maximum_period, oversample=oversample,
                                             frequency_factor=frequency_factor)
            self.snr = self.pg.snr
            P_bls = self.pg.period_at_max_power.value
            T0_bls = self.pg.transit_time_at_max_power.value
        else:
            tlsmodel = transitleastsquares.transitleastsquares(t=self.L.time, y=self.L.flux, dy=self.L.flux_err)
            self.pg = tlsmodel.power(oversampling_factor=oversample, duration_grid_step=1.05, period_max=maximum_period,
                                     period_min=minimum_period, transit_depth_min=800 * 1e-6)
            self.snr = self.pg
            P_bls = self.pg.period
            T0_bls = self.pg.T0
            self.pg.period_at_max_power = P_bls
            self.pg.transit_time_at_max_power = self.pg.T0
            self.pg.depth_at_max_power = 1e0 - self.pg.depth
            self.pg.get_transit_mask = transitleastsquares.transit_mask
            self.pg.max_power = max(self.pg.power) / u.day
            self.pg.get_transit_model = lambda period: lightkurve.LightCurve(time=self.pg['model_lightcurve_time'],
                                                                             flux=self.pg['model_lightcurve_model'],
                                                                             label="Transit Model Flux")
        print(P_bls, T0_bls)

        # Optimize
        print('################################')
        print('Optimizing')
        pinit = [P_bls, T0_bls, np.sqrt(self.pg.depth_at_max_power), 0.3, 15.]
        for thisindex in range(len(pinit)):
            if hasattr(pinit[thisindex], 'value'):
                pinit[thisindex] = pinit[thisindex].value
        ln, min_pv = self.L.optimize(pinit)

        if plot:
            print('Plotting')
            self.L.plot_transit(self.L.min_pv)
            lmm = self.L.get_transit_lc(self.L.min_pv)

            fig, ax = plt.subplots()
            self.pg.plot(view='period', ax=ax)
            ax.axvline(P_bls, color='orange', alpha=0.4)
            ax.axvline(self.L.min_pv[0], color='red', alpha=0.4, linestyle='--')

            fig, ax = plt.subplots()
            self.lc.fold(P_bls, T0_bls).scatter(ax=ax, color='blue', alpha=0.1)
            self.lc.fold(self.L.min_pv[0], self.L.min_pv[1]).scatter(ax=ax, color='black', alpha=0.5)
            lmm.fold(self.L.min_pv[0], self.L.min_pv[1]).plot(ax=ax, color='crimson', lw=2)
            # ax.set_xlim(-0.05,0.05)

            ax.set_title('Deriv: {}, ln={}'.format(self.L.min_pv_str, ln))
        return ln, min_pv
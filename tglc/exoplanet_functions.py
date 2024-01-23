import numpy as np
import batman
import astropy.constants as aconst
import astropy.units as u
import os
from uncertainties import ufloat
from uncertainties.umath import *

os.sys.path.append(os.getenv('NEID_ETC'))

def rho_star_from_P_and_aRs(P, aRs, return_cgs=True):
    """
    Density of the star from period of planet P, and aRs from transit observations.

    INPUT:
    - P in days
    - aRs - unitless

    OUTPUT:
    density of the star
    FLAG: return_cgs:
    -- if true: return in g/cm^3
    -- else: return in solar densities

    NOTE:
    - Assumes that eccentricity of the planet is 0. Otherwise not valid
    - Uses equation from Seager et al. 2003

    EXAMPLE:
        rho_star_from_P_and_aRs(0.2803,2.3,return_cgs=True) # Should be 2.92

        P = np.random.normal(0.2803226,0.0000013,size=1000)
        a = np.random.normal(2.31,0.08,size=1000)
        dens = rho_star_from_P_and_aRs(P,a,return_cgs=True)
        df = pd.DataFrame(dens)
        astropylib.mcFunc.calc_medvals2(df)
    """
    per = P * 24. * 60. * 60  # Change to seconds
    rho_sun = aconst.M_sun.value / (4. * np.pi * (aconst.R_sun.value ** 3.) / 3.)  # kg/m^3

    # Density from transit, see equation in Seager et al. 2003
    rho_star = 3. * np.pi * (aRs ** 3.) / (aconst.G.value * (per ** 2.))  # kg/m^3
    if return_cgs:
        return rho_star * (1000. / (100. * 100. * 100.))  # g/cm^3
    else:
        return rho_star / rho_sun


def get_lc_batman(times, t0, P, i, rprs, aRs, e, omega, u, supersample_factor=1, exp_time=0., limbdark="quadratic"):
    """
    Calls BATMAN and returns the transit model

    INPUT:
        times
        t0
        P
        i
        rprs
        aRs
        e
        omega
        u
        supersample_factor=1
        exp_time=0.
        limbdark="quadratic"

    OUTPUT:
        lc - the lightcurve model at *times*

    EXAMPLE:
        P = 2.34
        T0 = 1690.
        rprs = 0.1
        x = lc.time
        i = 90.
        aRs = exoplanet_functions.aRs_from_rho_and_P(15.,P)
        e, w = 0.,0.
        u = [0.3,0.3]
        f = exoplanet_functions.get_lc_batman(x,T0,P,i,rprs,aRs,e,w,u)
    """
    supersample_factor = int(supersample_factor)
    params = batman.TransitParams()
    params.t0 = t0
    params.per = P
    params.inc = i
    params.rp = rprs
    params.a = aRs
    params.ecc = e
    params.w = omega
    # q1, q2 = pv[5], pv[6]
    # self.params.u = astropylib.mcFunc.u1_u2_from_q1_q2(q1,q2) # Kipping 2013 formalism
    params.u = u
    params.limb_dark = limbdark
    params.fp = 0.001
    transitmodel = batman.TransitModel(params, times, transittype='primary',
                                       supersample_factor=supersample_factor,
                                       exp_time=exp_time)
    lc = transitmodel.light_curve(params)
    return lc


def calc_transit_duration(P, RpRs, aRs, i, omega=0, ecc=0):
    """
    A function to calculate the transit duration

    INPUT:
    P - days
    RpRs - ratio of companion to star radius
    aRs - ratio of semi-major axis to star radius
    i - inclination in radians
    omega - argument of pericenter in radians
    ecc - eccentricity in radians

    OUTPUT:
    transit duration in same units as the period

    NOTES:
    See Eq 15 in Winn et al. 2010: https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    sini = np.sin(i)
    esinw = ecc * np.sin(omega)
    b = aRs * np.cos(i) * (1e0 - ecc ** 2) / (1e0 + esinw)
    return (P / np.pi * np.arcsin(np.sqrt((1e0 + np.absolute(RpRs)) ** 2 - b ** 2) / (sini * aRs)) * np.sqrt(
        1e0 - ecc ** 2) / (1e0 + esinw))


def i_from_aRs_and_b(aRs, b):
    return np.rad2deg(np.arccos(b / aRs))  # aRs*np.cos(np.deg2rad(i))


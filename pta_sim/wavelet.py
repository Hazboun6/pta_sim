from __future__ import division

import numpy as np

import glob

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

def create_gw_antenna_pattern(theta, phi, gwtheta, gwphi):

    """

    Function to create pulsar antenna pattern functions as defined

    in Ellis, Siemens, and Creighton (2012).

    :param theta: Polar angle of pulsar location.

    :param phi: Azimuthal angle of pulsar location.

    :param gwtheta: GW polar angle in radians

    :param gwphi: GW azimuthal angle in radians



    :return: (fplus, fcross, cosMu), where fplus and fcross

             are the plus and cross antenna pattern functions

             and cosMu is the cosine of the angle between the

             pulsar and the GW source.

    """



    # use definition from Sesana et al 2010 and Ellis et al 2012

    m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])

    n = np.array([-np.cos(gwtheta)*np.cos(gwphi),

                  -np.cos(gwtheta)*np.sin(gwphi),

                  np.sin(gwtheta)])

    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi),

                      -np.sin(gwtheta)*np.sin(gwphi),

                      -np.cos(gwtheta)])



    phat = np.array([np.sin(theta)*np.cos(phi),

                     np.sin(theta)*np.sin(phi),

                     np.cos(theta)])



    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))

    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))

    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu


def construct_wavelet(t, A, t0, f0, tau, phi0):

    wave = np.zeros(len(t))

    # width of gaussian
    Q = tau*(2*np.pi*f0)

    # get time range
    tind = np.logical_and(t>=t0-4*tau, t<=t0+4*tau)

    wave[tind] = (A / (2*np.pi*f0)) * np.exp(-(2*np.pi*f0*(t[tind]-t0))**2/Q**2) * \
            np.cos(2*np.pi*f0*(t[tind]-t0)+phi0)

    return wave

@signal_base.function
def chrom_wavelet(toas, freqs, log10_A, t0, log10_f0, tau, phi0, exp):


    if exp in [2.0,4.0,4.4]:
        Exp = exp
    else: #Assuming that exp is a uniform prior from -0.5 to 2.5
        if -0.5 <= exp < 0.5:
            Exp = 2.0
        elif 0.5 <= exp < 1.5:
            Exp = 4.0
        elif 1.5 <= exp < 2.5:
            Exp = 4.4

    A = 10**log10_A
    f0 = 10**log10_f0
    w = construct_wavelet(toas, A, t0, f0, tau, phi0)
    res = w*(1400/freqs)**Exp

    return res


@signal_base.function

def gw_wavelet(toas, theta, phi, gwtheta, gwphi, gwpsi, gweps,
                         log10_gwA, gwt0, log10_gwf0, gwtau, gwphi0):

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*gwpsi), np.cos(2*gwpsi)

    # unit vectors to GW source
    m = np.array([-singwphi, cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    res = []
    #for ct, p in enumerate(psr):

        # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    gwA = 10**log10_gwA
    gwf0 = 10**log10_gwf0
    #gwQ = 10**log10_gwQ

    wplus = construct_wavelet(toas, gwA, gwt0, gwf0, gwtau, gwphi0)
    wcross = gweps * construct_wavelet(toas, gwA, gwt0, gwf0, gwtau, gwphi0+3*np.pi/2)

    res= fplus * (wplus*cos2psi - wcross*sin2psi) + fcross * (wplus*sin2psi + wcross*cos2psi)

    return res

@signal_base.function

def gw_wavelet_dropout(toas, theta, phi, gwtheta, gwphi, gwpsi, gweps,
                         log10_gwA, gwt0, log10_gwf0, gwtau, gwphi0,
                         k_drop=0.5,k_threshold=0.5):

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*gwpsi), np.cos(2*gwpsi)

    # unit vectors to GW source
    m = np.array([-singwphi, cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    res = []
    #for ct, p in enumerate(psr):

        # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    gwA = 10**log10_gwA
    gwf0 = 10**log10_gwf0
    #gwQ = 10**log10_gwQ

    wplus = construct_wavelet(toas, gwA, gwt0, gwf0, gwtau, gwphi0)
    wcross = gweps * construct_wavelet(toas, gwA, gwt0, gwf0, gwtau, gwphi0+3*np.pi/2)

    res= fplus * (wplus*cos2psi - wcross*sin2psi) + fcross * (wplus*sin2psi + wcross*cos2psi)

    if k_drop >= k_threshold: k_switch = 1.0
    elif k_drop < k_threshold: k_switch = 0.0

    return k_switch*res

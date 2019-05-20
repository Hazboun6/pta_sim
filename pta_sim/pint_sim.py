#!/usr/bin/env python
# coding: utf-8
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from pint.residuals import resids
import pint.toa as toa
from pint import models

def make_ideal(toas, model, iterations=2):
    '''
    Takes a pint.toas and pint.model object and effectively zeros out the residuals.
    '''
    for ii in range(iterations):
        rs=resids(toas, model)
        toas.adjust_TOAs(TimeDelta(-1.0*rs.time_resids))

def createfourierdesignmatrix_red(toas, nmodes=30, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  pshift=False, modes=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    Parameters
    ----------

    toas : array
        Vector of time series in seconds.

    nmodes : int
        Number of fourier coefficients to use.

    Tspan : float
        Option to us some other Tspan [s]

    logf : bool
        Use log frequency spacing.

    fmin : float
        Lower sampling frequency.

    fmax : float
        Upper sampling frequency.

    pshift : bool
        Option to add random phase shift.

    modes : array
        Option to provide explicit list or array of sampling frequencies.

    Returns
    -------
    F : array
        fourier design matrix, [NTOAs x 2 nfreqs].
    f : arraty
        Sampling frequencies, [2 nfreqs].
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # add random phase shift to basis functions
    ranphase = (np.random.uniform(0.0, 2 * np.pi, nmodes)
                if pshift else np.zeros(nmodes))

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*toas[:,None]*f[None,:] +
                      ranphase[None,:])
    F[:,1::2] = np.cos(2*np.pi*toas[:,None]*f[None,:] +
                       ranphase[None,:])

    return F, Ffreqs

def add_rednoise(TOAs, A, gamma, components=30,
                 seed=None, modes=None, Tspan=None):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f * year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""

    # nobs=len(psr.toas)
    nobs = len(TOAs.table)

    day_in_sec = 86400
    year_in_sec = 365.25*day_in_sec
    fyr = 1 / year_in_sec

    if seed is not None:
        np.random.seed(seed)
    if modes is not None:
        print('Must use linear spacing.')

    toas = np.array(TOAs.table['tdbld'], dtype='float64') * day_in_sec #to sec
    Tspan = toas.max() - toas.min()
    F, freqs = createfourierdesignmatrix_red(toas,Tspan=Tspan,modes=modes)
    prior = A**2 * (freqs/fyr)**(-gamma) / (12 * np.pi**2 * Tspan) * year_in_sec**3
    y = np.sqrt(prior) * np.random.randn(freqs.size)
    dt = np.dot(F,y) * u.s
    TOAs.adjust_TOAs(TimeDelta(dt.to('day')))

def add_dm_rednoise(TOAs, A, gamma, components=30, rf_ref=1400,
                    seed=None, modes=None, Tspan=None, useDM=False):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""

    # nobs=len(psr.toas)
    nobs = len(TOAs.table)
    radio_freqs = TOAs.table['freq']
    if useDM:
        rf_ref = 4.15e3
    chrom = rf_ref**2 / radio_freqs**2
    day_in_sec = 86400
    year_in_sec = 365.25*day_in_sec
    fyr = 1 / year_in_sec

    if seed is not None:
        np.random.seed(seed)

    toas = np.array(TOAs.table['tdbld'], dtype='float64') * day_in_sec #to sec

    Tspan = toas.max() - toas.min()

    F, freqs = createfourierdesignmatrix_red(toas,Tspan=Tspan,modes=modes)
    prior = A**2 * (freqs/fyr)**(-gamma) / (12 * np.pi**2 * Tspan) * year_in_sec**3

    y = np.sqrt(prior) * np.random.randn(freqs.size)
    dt = chrom.quantity.value * np.dot(F,y) * u.s
    TOAs.adjust_TOAs(TimeDelta(dt.to('day')))

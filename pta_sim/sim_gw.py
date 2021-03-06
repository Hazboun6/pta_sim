#!/usr/bin/env python
# coding: utf-8

# Code for running NANOGrav DSA2000 Simulation using the Optimal Statistic
# Libstempo is used to do the simulating

from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
from shutil import copyfile, copy2
import scipy.stats as sci_stats

# sys.path.insert(0,'/home/jeffrey.hazboun/nanograv/dsa2000_simulations/enterprise/')
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

from enterprise_extensions import models, model_utils, dropout
from enterprise_extensions.frequentist import optimal_statistic as OS

import argparse

######## Convenience Functions ##########
day_in_sec = 24.*3600
year_in_sec = 365.25*day_in_sec

def psr_name(x, rhs='.'):
    return x.split('/')[-1].split(rhs)[0]


######## Simulations ############

class Simulation(object):

    def __init__(self, parfiles, timfiles, ephem='DE436',verbose=True):

        #### Make a list of libstempo.tempopulsar objects.
        libs_psrs = []
        for p, t in zip(parfiles, timfiles):

            psr = T2.tempopulsar(p, t, ephem=ephem, maxobs=30000)
            libs_psrs.append(psr)
            if verbose:
                print('\rPSR {0} loaded.'.format(psr_name(p)),
                      flush=True,end='')

        print('')
        self.libs_psrs = libs_psrs
        self.pnames = [p.name for p in libs_psrs]
        self.psrs = None
        self.ephem = ephem
        self.first_toa = np.amin([p.toas().min() for p in libs_psrs])
        self.last_toa = np.amax([p.toas().max() for p in libs_psrs])
        self.toa_cuts = [self.last_toa]
        self.gwb_added = False


    def createGWB(self, A_gwb, gamma_gw= 13./3, seed=None, fit=None):
        """Great GWB using libstempo.toasim"""
        if A_gwb!=0:
            LT.createGWB(self.libs_psrs, A_gwb, gamma_gw, seed=seed)
            # Fit libstempo psr
            if fit is not None:
                for psr in self.libs_psrs:
                    psr.fit(iters=fit)
        else:
            pass
        self.gwb_added = True
        self.seed = seed

    def add_rn(self, rn_psrs, seeds=None):
        """
        Add rednoise to a subset of the pulsars.

        Parameters
        ----------
        rn_psrs : dict
            Dictionary of rednoise parameter entries. The keys are pulsar names
            while each entry is an array/list/tuple of RN amplitude, RN spectral
            index.
        """
        if seeds is None:
            seeds=[None for ky in rn_psrs.keys()]

        for ii, (p, pars) in enumerate(rn_psrs.items()):
            A = pars[0]
            gamma = pars[1]
            pidx = self.pnames.index(p)
            LT.add_rednoise(self.libs_psrs[pidx], A, gamma,
                            components=30, seed=seeds[ii])

    def init_ePulsars(self, **kwarg):
        if not self.gwb_added:
            raise ValueError('Must add GWB signal (createGWB) before '
                             'initializing enterprise.Pulsar objects. If none '
                             'desired set A_gwb=0.')

        psrs = []
        for p in self.libs_psrs:
            psr = Pulsar(p, ephem=self.ephem, **kwarg)

            ### Check first toa ####
            #mn = psr.residuals[:100].mean()
            #std = psr.residuals[:100].std()
            #check_val = mn + 4*std
            #if any([abs(res)>abs(check_val) for res in psr.residuals]):
            #    mask = np.where(abs(psr.residuals)>abs(check_val),False,True)
            #    model_utils.mask_filter(psr,mask)
            # mask = np.ones_like(psr.toas,dtype=bool)
            # mask[0] = False
            # model_utils.mask_filter(psr,mask)

            psrs.append(psr)

        self.psrs=psrs

    def filter_by_mjd(self,end_time):
        """Filter the TOAs by MJD"""
        #### Filter TOAs by time ##########
        if np.amin(self.toa_cuts)<end_time:
            raise ValueError('Can only filter the TOAs to earlier times. Must'
                             ' start again and start with latest cuts first.')
        if self.psrs is None:
            raise ValueError('Enterprise pulsars have not be initialized yet!'
                             ' Use init_ePulsars().')
        idxs = []
        first_toa = np.amin([p.toas.min() for p in self.psrs])/day_in_sec

        for psr in self.psrs:
            psr.filter_data(start_time=self.first_toa, end_time=end_time)
            if psr.toas.size==0:
                idxs.append(self.psrs.index(psr))
            else:
                timespan = (psr.toas[-1]-psr.toas[0])/year_in_sec
                if timespan<3.0:
                    idxs.append(self.psrs.index(psr))

        #### Remove empty pulsars, Reverse to keep correct idx order.

        for ii in reversed(idxs):
            del self.psrs[ii]

        if np.amin(self.toa_cuts)==end_time:
            pass
        else:
            self.toa_cuts.append(end_time)

def model_simple(psrs, psd='powerlaw', efac=False, n_gwbfreqs=30,
                 components=30, freqs=None,
                 vary_gamma=False, upper_limit=False, bayesephem=False,
                 select='backend', red_noise=False, Tspan=None, hd_orf=False,
                 rn_dropout=False, dp_threshold=0.5):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with the most simple model allowable for enterprise:
    per pulsar:
        1. fixed EFAC per backend/receiver system at 1.0
        2. Linear timing model.
        3. Red noise modeled as a power-law with
            30 sampling frequencies. Default=False
    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when performing upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    model = gp_signals.TimingModel()

    #Only White Noise is EFAC set to 1.0
    selection = selections.Selection(selections.by_backend)
    if efac:
        ef = parameter.Uniform(0.1,10.0)
    else:
        ef = parameter.Constant(1.00)

    model += white_signals.MeasurementNoise(efac=ef, selection=selection)

    # common red noise block
    if upper_limit:
        log10_A_gw = parameter.LinearExp(-18,-12)('gw_log10_A')
    else:
        log10_A_gw = parameter.Uniform(-18,-12)('gw_log10_A')

    if vary_gamma:
        gamma_gw = parameter.Uniform(0,7)('gw_gamma')
    else:
        gamma_gw = parameter.Constant(4.33)('gw_gamma')

    pl = signal_base.Function(utils.powerlaw, log10_A=log10_A_gw,
                              gamma=gamma_gw)


    if hd_orf:
        if freqs is None:
            gw = gp_signals.FourierBasisCommonGP(spectrum=pl,
                                                 orf=utils.hd_orf(),
                                                 components=n_gwbfreqs,
                                                 Tspan=Tspan,
                                                 name='gw')
        else:
            gw = gp_signals.FourierBasisCommonGP(spectrum=pl,
                                                 orf=utils.hd_orf(),
                                                 modes=freqs,
                                                 name='gw')
        model += gw
    else:
        if freqs is None:
            crn = gp_signals.FourierBasisGP(spectrum=pl, components=n_gwbfreqs,
                                            Tspan=Tspan, name='gw')
        else:
            crn = gp_signals.FourierBasisGP(spectrum=pl, modes=freqs,
                                            name='gw')
        model += crn

    if red_noise and rn_dropout:
        if amp_prior == 'uniform':
            log10_A = parameter.LinearExp(-20, -11)
        elif amp_prior == 'log-uniform':
            log10_A = parameter.Uniform(-20, -11)
        else:
            log10_A = parameter.Uniform(-20, -11)

        gamma = parameter.Uniform(0, 7)
        k_drop = parameter.Uniform(0, 1)
        if dp_threshold == 6.0:
            dp_threshold = parameter.Uniform(0,1)('k_threshold')
        pl = dropout.dropout_powerlaw(log10_A=log10_A, gamma=gamma,
                                     k_drop=k_drop, k_threshold=dp_threshold)
        rn = gp_signals.FourierBasisGP(pl, components=components,
                                       Tspan=Tspan, name='red_noise')
        model += rn
    elif red_noise:
        # red noise
        model += models.red_noise_block(prior=amp_prior, Tspan=Tspan,
                                        components=components)

    # ephemeris model
    if bayesephem:
        model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # set up PTA
    pta = signal_base.PTA([model(p) for p in psrs])

    return pta

######## Red Noise Parameters ############
#Since there is no red noise we make a dictionary of low RN Values

def get_rn_dict(pta):
    #psr_list = pta
    pass


######## File I/O ###############
def save(outpath):
    pass

# if __name__=='__main__':

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
import cloudpickle
import logging

from enterprise_extensions import models, model_utils, hypermodel, sampler
from enterprise.signals.signal_base import PTA
from enterprise.signals import gp_signals, signal_base, deterministic_signals, parameter, selections, white_signals, utils
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise import constants as const

from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions import blocks
from enterprise_extensions import gp_kernels as gpk
from enterprise_extensions import chromatic as chrom
import la_forge.core as co

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
#Is chain longer than niter?
# longer = chain_length_bool(args.outdir, int(args.niter//10))

# if longer:
#     sys.exit() #Hmmmm what to do here?
# else:
#     pass

"""
List of things that we need to change in this code:
* [x] Decide what to do about the 2 separate DM GPs in B1937
* [x] Use the same input kwarg for every pulsar
* [x] Change the code to edit the output kwarg using Jeremy's script
* [x] Be careful to change what we need to in the various kwarg dictionaries in the script
* [x] See if we can output the size of the basis to file when we output the other runtime stuff.
* [x] Deal with the SW model. Fit across yearly bins at first.
* [x] Figure out B1937+21 chromatic model and Fourier option. Need a switch? 
* [x] Make new WN empirical distributions? 
* [x] Check red noise is correctly being modeled.
* [x] Check that timing models are being modeled correctly.
* [x] Code up a varying chromatic index.
"""

with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta_crn = cloudpickle.load(f)
else:
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    psrname = args.psr

    def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp'):
        """
        Returns chromatic exponential dip (i.e. TOA advance):

        :param tmin, tmax:
            search window for exponential dip time.
        :param idx:
            index of radio frequency dependence (i.e. DM is 2). If this is set
            to 'vary' then the index will vary from 1 - 6
        :param sign:
            set sign of dip: 'positive', 'negative', or 'vary'
        :param name: Name of signal

        :return dmexp:
            chromatic exponential dip waveform.
        """
        t0_dmexp = parameter.Uniform(tmin,tmax)
        log10_Amp_dmexp = parameter.Uniform(-6.1, -5.6)
        log10_tau_dmexp = parameter.Uniform(1.2, 2.0)
        if sign == 'vary':
            sign_param = parameter.Uniform(-1.0, 1.0)
        elif sign == 'positive':
            sign_param = 1.0
        else:
            sign_param = -1.0
        wf = chrom.chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                                   t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                                   sign_param=sign_param, idx=idx)
        dmexp = deterministic_signals.Deterministic(wf, name=name)

        return dmexp

    # adding white-noise, separating out Adv Noise Psrs, and acting on psr objects
    final_psrs = []
    psr_models = []
    ### Add a stand alone SW deter model
    bins = np.loadtxt(args.sw_bins)
    bins *= 24*3600 #Convert to secs

    if args.sw_fit_path is None:
        if args.ACEprior:
            n_earth = chrom.solar_wind.ACE_SWEPAM_Parameter(size=bins.size-1)('n_earth')
        else:
            n_earth = parameter.Uniform(0,30,size=bins.size-1)('n_earth')

        np_earth = parameter.Uniform(-4, -2)('np_4p39')
    else:
        n_earth = parameter.Constant()('n_earth')
        np_earth = parameter.Constant()('np_4p39')
        with open(args.sw_fit_path,'r') as fin:
            sw_vals = json.load(fin)
        noise.update(sw_vals)

    deter_sw = chrom.solar_wind.solar_wind(n_earth=n_earth, n_earth_bins=bins)
    mean_sw = deterministic_signals.Deterministic(deter_sw, name='sw_r2')

    if args.sw_r4p4:
        sw_power = parameter.Constant(4.39)('sw_power_4p39')
        deter_sw_p = chrom.solar_wind.solar_wind_r_to_p(n_earth=np_earth,
                                                        power=sw_power,
                                                        log10_ne=True)
        mean_sw += deterministic_signals.Deterministic(deter_sw_p,
                                                       name='sw_4p39')

    pnames = [p.name for p in pkl_psrs]
    pidx = pnames.index(args.psr)
    new_psr = pkl_psrs[pidx]
    ### Get kwargs dictionary
    kwarg_path = args.model_kwargs_path
    kwarg_path += f'{psrname}_model_kwargs.json'
    with open(kwarg_path, 'r') as fin:
        kwargs = json.load(fin)

    ## Build special DM GP models for B1937
    if psrname == 'B1937+21' and kwargs["dmgp_kernel"]=="nondiag" and kwargs['dm_nondiag_kernel']!='dmx_like':
        # Periodic GP kernel for DM
        log10_sigma = parameter.Uniform(-10, -4.8)
        log10_ell = parameter.Uniform(1, 2.4)
        log10_p = parameter.Uniform(-2, -1)
        log10_gam_p = parameter.Uniform(-2, 2)
        log10_ell_rf = parameter.Uniform(0, 7)
        log10_alpha_wgt = parameter.Uniform(-4, 3)
        if kwargs["dm_nondiag_kernel"] == "periodic_rfband":
            dm_basis = gpk.get_tf_quantization_matrix(df=kwargs['dm_df'], 
                                                      dt=kwargs['dm_dt']*86400,
                                                      dm=True)
            dm_prior = gpk.tf_kernel(log10_sigma=log10_sigma,
                                     log10_ell=log10_ell,
                                     log10_gam_p=log10_gam_p,
                                     log10_p=log10_p,
                                     log10_alpha_wgt=log10_alpha_wgt,
                                     log10_ell2=log10_ell_rf)
        elif kwargs["dm_nondiag_kernel"] == "periodic":
            dm_basis = gpk.linear_interp_basis_dm(dt=kwargs['dm_dt']*86400)
            dm_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                           log10_ell=log10_ell,
                                           log10_gam_p=log10_gam_p,
                                           log10_p=log10_p)
        
        dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp1')
        # Periodic GP kernel for DM
        log10_sigma2 = parameter.Uniform(-4.8, -3)
        log10_ell2 = parameter.Uniform(2.4, 5)
        log10_p2 = parameter.Uniform(-2, 2)
        log10_gam_p2 = parameter.Uniform(-2, 2)
        log10_ell_rf2 = parameter.Uniform(0, 7)
        log10_alpha_wgt2 = parameter.Uniform(-4, 3)
        if kwargs["dm_nondiag_kernel"] == "periodic_rfband":
            dm_basis2 = gpk.get_tf_quantization_matrix(df=kwargs['dm_df'], 
                                                      dt=kwargs['dm_dt']*86400,
                                                      dm=True)
            dm_prior2 = gpk.tf_kernel(log10_sigma=log10_sigma2,
                                      log10_ell=log10_ell2,
                                      log10_gam_p=log10_gam_p2,
                                      log10_p=log10_p2,
                                      log10_alpha_wgt=log10_alpha_wgt2,
                                      log10_ell2=log10_ell_rf2)
        elif kwargs["dm_nondiag_kernel"] == "periodic":
            dm_basis2 = gpk.linear_interp_basis_dm(dt=kwargs['dm_dt']*86400)
            dm_prior2 = gpk.periodic_kernel(log10_sigma=log10_sigma2,
                                            log10_ell=log10_ell2,
                                            log10_gam_p=log10_gam_p2,
                                            log10_p=log10_p2)
        
        
        dmgp2 = gp_signals.BasisGP(dm_prior2, dm_basis2, name='dm_gp2')


        kwargs.update({'white_vary':args.vary_wn,
                       'dm_var': False,
                       'red_var': True,
                       'extra_sigs':dmgp + dmgp2 + mean_sw,
                       'psr_model':True,
                       })
    elif psrname == 'J1713+0747':
        index1 = parameter.Uniform(0, 5)
        index2 = parameter.Uniform(0.9, 1.7)
        
        dip1 = dm_exponential_dip(54740, 54780, idx=index2, sign='negative', name='exp1')
        dip2 = dm_exponential_dip(57506, 57514, idx=index1, sign='negative', name='exp2')
       
        kwargs.update({'white_vary':args.vary_wn,
                       'extra_sigs':mean_sw + dip1 + dip2,
                       'psr_model':True,
                       'red_var': True,
                       })
    ## Treat all other Adv Noise pulsars the same
    else:
        ### Turn SW model off. Add in stand alone SW model and common process. Return model.
        kwargs.update({'white_vary':args.vary_wn,
                       'extra_sigs':mean_sw,
                       'psr_model':True,
                       'red_var': True,
                       })

    if args.gfl:
        kwargs.update({'factorized_like':True,
                       'psd':'spectrum',
                       'gw_components':30,
                       'fact_like_logmin':-14.2,
                       'fact_like_logmax':-1.2,})
    if args.gwb_on:
        kwargs.update({'factorized_like':True,
                       'gw_components':args.n_gwbfreqs,
                       'fact_like_gamma':args.gamma_gw,})

    psr_models.append(model_singlepsr_noise(new_psr, **kwargs))
    final_psrs.append(new_psr)

    models = [m(psr) for psr,m in  zip(final_psrs,psr_models)]
    pta_crn = signal_base.PTA(models)
    pta_crn.set_default_params(noise)


groups = sampler.get_parameter_groups(pta_crn)
groups.extend(sampler.get_psr_groups(pta_crn))
Sampler = sampler.setup_sampler(pta_crn, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr, groups=groups)

Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_empirical_distr, 40)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_prior, 10)
Sampler.addProposalToCycle(Sampler.jp.draw_from_empirical_distr, 120)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_red_prior, 60)
Sampler.addProposalToCycle(Sampler.jp.draw_from_dm_gp_prior, 40)
Sampler.addProposalToCycle(Sampler.jp.draw_from_chrom_gp_prior, 40)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_dmexpcusp_prior, 10)
if psrname == 'J1713+0747':
    Sampler.addProposalToCycle(Sampler.jp.draw_from_par_prior(['exp1','exp2']),30)
#                                                            'np_4p39',
#                                                            'dm_cusp',
#                                                            'dmexp']),
#                                                            30)
def draw_from_sw_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    signal_name = 'sw_r2'

    # draw parameter from signal model
    param = np.random.choice(self.snames[signal_name])
    if param.size:
        idx2 = np.random.randint(0, param.size)
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

    # scalar parameter
    else:
        q[self.pmap[str(param)]] = param.sample()

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

def draw_from_sw4p39_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    signal_name = 'sw_4p39'

    # draw parameter from signal model
    param = np.random.choice(self.snames[signal_name])
    if param.size:
        idx2 = np.random.randint(0, param.size)
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

    # scalar parameter
    else:
        q[self.pmap[str(param)]] = param.sample()

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

def draw_from_gw_gamma_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    # draw parameter from signal model
    signal_name = [par for par in self.pnames
                   if ('gw' in par and 'gamma' in par)][0]
    idx = list(self.pnames).index(signal_name)
    param = self.params[idx]

    q[self.pmap[str(param)]] = np.random.uniform(param.prior._defaults['pmin'], param.prior._defaults['pmax'])

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

if args.sw_fit_path is None:
    sampler.JumpProposal.draw_from_sw_prior = draw_from_sw_prior
    sampler.JumpProposal.draw_from_sw4p39_prior = draw_from_sw4p39_prior
# sampler.JumpProposal.draw_from_gw_gamma_prior = draw_from_gw_gamma_prior

    Sampler.addProposalToCycle(Sampler.jp.draw_from_sw_prior, 25)
    Sampler.addProposalToCycle(Sampler.jp.draw_from_sw4p39_prior, 25)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_gw_gamma_prior, 25)


try:
    achrom_freqs = get_freqs(pta_crn, signal_id='red_noise')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

try:
    chrom_basis = get_freqs(pta_crn, signal_id='dm_gp')
    np.savetxt(args.outdir + 'chrom_basis.txt', chrom_basis, fmt='%.18e')
except:
    pass

if args.initsamp is None:
    x0 = np.hstack(p.sample() for p in pta_crn.params)
else:
    with open(args.initsamp, 'r') as fin:
        init = json.load(fin)
    x0 = np.array([init[k] for k in pta_crn.param_names])

if args.ladderpath is not None:
    ladder = np.loadtxt(args.ladderpath)
else:
    ladder = None

Sampler.sample(x0, args.niter, ladder=ladder, SCAMweight=200, AMweight=100,
               DEweight=200, burn=3000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)

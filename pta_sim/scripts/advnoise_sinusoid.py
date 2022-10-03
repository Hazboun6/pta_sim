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

with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta_crn = cloudpickle.load(f)
else:
    # with open('{0}'.format(args.pickle), "rb") as f:
    #     pkl_psrs = pickle.load(f)

    adv_noise_psr_list = ['B1855+09', #32
                          'B1937+21', #42
                          'J0030+0451',# #1.4 **
                          'J0613-0200',# -25 *
                          'J0645+5158',# 28 *
                          'J1012+5307',#38
                          'J1024-0719', #-16 **
                          'J1455-3330', #-16 **
                          'J1600-3053', #-10 **
                          'J1614-2230', #-1 **
                          'J1640+2224', #44
                          'J1713+0747', #30 *
                          'J1738+0333', #-26 *
                          'J1741+1351', #37
                          'J1744-1134', #12 **
                          'J1909-3744', #15 **
                          'J1910+1256', #35
                          'J2010-1323', #6 **
                          'J2043+1711',#40
                          'J2317+1439'] #17 *

    adv_noise_psr_list = [np.array(adv_noise_psr_list)[args.process]]
    psrname = adv_noise_psr_list[0]

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

    # timing model
    s = gp_signals.MarginalizingTimingModel()

    # intrinsic red noise
    s += blocks.red_noise_block(prior='log-uniform', Tspan=args.tspan, components=30)
    # adding white-noise, separating out Adv Noise Psrs, and acting on psr objects
    final_psrs = []
    psr_models = []
    ### Add a stand alone SW deter model
    bins = np.linspace(53215, 57934, 26)
    bins *= 24*3600 #Convert to secs
    # n_earth = chrom.solar_wind.ACE_SWEPAM_Parameter(size=bins.size-1)('n_earth')
    if args.sw_fit_path is None:
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


    sw_power = parameter.Constant(4.39)('sw_power_4p39')
    deter_sw_p = chrom.solar_wind.solar_wind_r_to_p(n_earth=np_earth,
                                                    power=sw_power,
                                                    log10_ne=True)
    mean_sw += deterministic_signals.Deterministic(deter_sw_p,
                                                   name='sw_4p39')



    #### Nihan's Sine wave
    dataset_tmin = 4597873783.54894 #np.min([psr.toas.min() for psr in psrs])
    @parameter.function
    def sine_wave(toas, flags, A = -9, f = -9, phase = 0.0):
        return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)

    def sine_signal(A, f, phase, name = ""):
        return Deterministic(sine_wave(A = A, f = f, phase = phase), name = name)

    day_seconds = 86400
    sin = sine_signal(A = parameter.Uniform(-9, -4)('common_sin_A'), f = parameter.Uniform(-9, -7)('common_sin_f'), phase = parameter.Uniform(0, 2 * np.pi)('common_sin_phase'))


    ### Get the new pulsar object
    ## Remember that J1713's pickle is something you made yourself ##
    filepath = '/gscratch/gwastro/hazboun/nanograv/noise/noise_model_selection/no_dmx_pickles/'
    filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(psrname)
    with open(filepath,'rb') as fin:
        new_psr=pickle.load(fin)

    ### Get kwargs dictionary
    kwarg_path = args.model_kwargs_path
    kwarg_path += f'{psrname}_model_kwargs.json'
    with open(kwarg_path, 'r') as fin:
        kwargs = json.load(fin)

    ## Build special DM GP models for B1937
    if psrname == 'B1937+21':
        # Periodic GP kernel for DM
        log10_sigma = parameter.Uniform(-10, -4.8)
        log10_ell = parameter.Uniform(1, 2.4)
        log10_p = parameter.Uniform(-2, -1)
        log10_gam_p = parameter.Uniform(-2, 2)
        dm_basis = gpk.linear_interp_basis_dm(dt=3*86400)
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
        dm_basis2 = gpk.linear_interp_basis_dm(dt=3*86400)
        dm_prior2 = gpk.periodic_kernel(log10_sigma=log10_sigma2,
                                       log10_ell=log10_ell2,
                                       log10_gam_p=log10_gam_p2,
                                       log10_p=log10_p2)
        dmgp2 = gp_signals.BasisGP(dm_prior2, dm_basis2, name='dm_gp2')
        ch_log10_sigma = parameter.Uniform(-10, -3.5)
        ch_log10_ell = parameter.Uniform(1, 6)
        chm_basis = gpk.linear_interp_basis_chromatic(dt=3*86400, idx=4)
        chm_prior = gpk.se_dm_kernel(log10_sigma=ch_log10_sigma, log10_ell=ch_log10_ell)
        chromgp = gp_signals.BasisGP(chm_prior, chm_basis, name='chrom_gp')

        kwargs.update({'dm_sw_deter':False,
                       'white_vary':args.vary_wn,
                       'red_var': True,
                       'extra_sigs':sin + dmgp + dmgp2 + chromgp + mean_sw,
                       'psr_model':True,
                       'chrom_df':None,
                       'dm_df':None,
                       'tm_marg':True})
    elif psrname == 'J1713+0747':
        index = parameter.Uniform(0.9, 1.7)
        ppta_dip = dm_exponential_dip(57506, 57514, idx=index, sign='negative', name='exp2')

        kwargs.update({'dm_dt':3,
                       'dm_df':None,
                       'chrom_dt':3,
                       'dm_sw_deter':False,
                       'white_vary':args.vary_wn,
                       'dm_expdip':True,
                       'dmexp_sign': 'negative',
                       'num_dmdips':1,
                       'dm_expdip_idx':[2],
                       'dm_expdip_tmin':[54740],
                       'dm_expdip_tmax':[54780],
                       'dmdip_seqname':['dm_1'],
                       'extra_sigs':sin + mean_sw + ppta_dip,
                       'psr_model':True,
                       'red_var': True,
                       'chrom_df':None,
                       'dm_df':None,
                       'tm_marg':True})
    ## Treat all other Adv Noise pulsars the same
    else:
        ### Turn SW model off. Add in stand alone SW model and common process. Return model.
        kwargs.update({'dm_sw_deter':False,
                       'white_vary':args.vary_wn,
                       'extra_sigs':sin + mean_sw,
                       'psr_model':True,
                       'chrom_df':None,
                       'dm_df':None,
                       'red_var': True,
                       'tm_marg':True})

    if args.gfl:
        kwargs.update({'red_var':False,
                       'factorized_like':True,
                       'psd':'spectrum',
                       'Tspan':args.tspan,
                       'gw_components':30,
                       'fact_like_logmin':-14.2,
                       'fact_like_logmax':-1.2,})
    if args.gwb_on:
        kwargs.update({'factorized_like':True,
                      'Tspan':args.tspan,
                      'gw_components':args.n_gwbfreqs,
                      'fact_like_gamma':args.gamma_gw,})

    psr_models.append(model_singlepsr_noise(new_psr, **kwargs))
    final_psrs.append(new_psr)

    models = [m(psr) for psr,m in  zip(final_psrs,psr_models)]
    pta_crn = signal_base.PTA(models)
    pta_crn.set_default_params(noise)

    # with open(args.pta_pkl,'wb') as fout:
        # cloudpickle.dump(pta_crn,fout)

groups = sampler.get_parameter_groups(pta_crn)
groups.extend(sampler.get_psr_groups(pta_crn))
Sampler = sampler.setup_sampler(pta_crn, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr, groups=groups)

Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_empirical_distr, 70)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_prior, 10)
Sampler.addProposalToCycle(Sampler.jp.draw_from_empirical_distr, 120)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_red_prior, 60)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_dm_gp_prior, 40)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_chrom_gp_prior, 10)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_dmexpcusp_prior, 10)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_par_prior(['n_earth',
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
    achrom_freqs = get_freqs(pta_crn, signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

if args.initsamp is None:
    # noise['gw_log10_A'] = np.log10(2e-15)
    # noise['gw_gamma'] = 4.33333
    # nearth_pars = [p for p in pta_crn.param_names if 'n_earth' in p]
    # for npar in nearth_pars:
    #     noise[npar]=6.0
    # noise['np_4p39']=-2.86
    # x0 = np.array([noise[k] for k in pta_crn.param_names])
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

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


with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta_crn = cloudpickle.load(f)
else:
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    with open('{0}'.format(args.pickle_nodmx), "rb") as f:
        nodmx_psrs = pickle.load(f)

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


    def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp', vary=True):
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
        :param vary: Whether to vary the parameters or use constant values.

        :return dmexp:
            chromatic exponential dip waveform.
        """
        if vary:
            t0_dmexp = parameter.Uniform(tmin, tmax)
            log10_Amp_dmexp = parameter.Uniform(-10, -2)
            log10_tau_dmexp = parameter.Uniform(0, 2.5)
        else:
            t0_dmexp = parameter.Constant()
            log10_Amp_dmexp = parameter.Constant()
            log10_tau_dmexp = parameter.Constant()

        if sign == 'vary' and vary:
            sign_param = parameter.Uniform(-1.0, 1.0)
        elif sign == 'vary' and not vary:
            sign_param = parameter.Constant()
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
    tm = gp_signals.TimingModel()
    # s = gp_signals.MarginalizingTimingModel()
    
    # intrinsic red noise
    s = blocks.red_noise_block(prior='log-uniform', Tspan=args.tspan, components=30)

    Tspan_PTA = args.tspan
    log10_rho = parameter.Uniform(-10,-4,size=30)
    fs = gpp.free_spectrum(log10_rho=log10_rho)
    log10_A = parameter.Constant()
    gamma = parameter.Constant()
    plaw_pr = gpp.powerlaw(log10_A=log10_A,gamma=gamma)
    plaw = gp_signals.FourierBasisGP(plaw_pr,components=30,Tspan=args.tspan)
    rn  = gp_signals.FourierBasisGP(fs,components=30,Tspan=args.tspan, name='excess_noise')

    m = s #plaw + rn

    # adding white-noise, separating out Adv Noise Psrs, and acting on psr objects
    final_psrs = []
    psr_models = []
    ### Add a stand alone SW deter model
    bins = np.arange(53215, 59200, 180)
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

    if args.sw_r4p4:
        sw_power = parameter.Constant(4.39)('sw_power_4p39')
        deter_sw_p = chrom.solar_wind.solar_wind_r_to_p(n_earth=np_earth,
                                                        power=sw_power,
                                                        log10_ne=True)
        mean_sw += deterministic_signals.Deterministic(deter_sw_p,
                                                       name='sw_4p39')

    if args.gwb_on:
        orf = 'hd'
    else:
        orf = None
    cs = blocks.common_red_noise_block(psd=args.psd,
                                        prior='log-uniform',
                                        Tspan=args.tspan,
                                        orf=orf,
                                        components=args.n_gwbfreqs,
                                        gamma_val=args.gamma_gw,
                                        name='gw')
    
    #####
    for psr,psr_nodmx in zip(pkl_psrs,nodmx_psrs):
        # Filter out other Adv Noise Pulsars
        if psr.name in adv_noise_psr_list:
            new_psr = psr_nodmx

            ### Get kwargs dictionary
            kwarg_path = args.model_kwargs_path
            kwarg_path += f'{psr.name}_model_kwargs.json'
            with open(kwarg_path, 'r') as fin:
                kwargs = json.load(fin)


    #         if 'wideband' in kwargs.keys():
    #             kwargs['is_wideband'] = kwargs['wideband']
    #             kwargs.__delitem__('wideband')
            ## Build special DM GP models for B1937
            if psr.name == 'B1937+21':
                # Periodic GP kernel for DM
                log10_sigma = parameter.Constant()
                log10_ell = parameter.Constant()
                log10_p = parameter.Constant()
                log10_gam_p = parameter.Constant()
                dm_basis = gpk.linear_interp_basis_dm(dt=3*86400)
                dm_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                            log10_ell=log10_ell,
                                            log10_gam_p=log10_gam_p,
                                            log10_p=log10_p)
                dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp1')
                # Periodic GP kernel for DM
                log10_sigma2 = parameter.Constant()
                log10_ell2 = parameter.Constant()
                log10_p2 = parameter.Constant()
                log10_gam_p2 = parameter.Constant()
                dm_basis2 = gpk.linear_interp_basis_dm(dt=3*86400)
                dm_prior2 = gpk.periodic_kernel(log10_sigma=log10_sigma2,
                                            log10_ell=log10_ell2,
                                            log10_gam_p=log10_gam_p2,
                                            log10_p=log10_p2)
                dmgp2 = gp_signals.BasisGP(dm_prior2, dm_basis2, name='dm_gp2')
                ch_log10_sigma = parameter.Constant()
                ch_log10_ell = parameter.Constant()
                chm_basis = gpk.linear_interp_basis_chromatic(dt=3*86400, idx=4)
                chm_prior = gpk.se_dm_kernel(log10_sigma=ch_log10_sigma, log10_ell=ch_log10_ell)
                chromgp = gp_signals.BasisGP(chm_prior, chm_basis, name='chrom_gp')

                kwargs.update({'dm_sw_deter':False,
                            'white_vary':args.vary_wn,
                            'red_var': False,
                            'extra_sigs':m + dmgp + dmgp2 + chromgp + mean_sw,
                            'psr_model':True,
                            'chrom_df':None,
                            'dm_df':None,
                            'tm_marg':False,
                            'tm_svd':False,
                            'vary_dm':False,
                            'vary_chrom':False})
            elif psr.name == 'J1713+0747':
                index = parameter.Constant()
                ppta_dip = dm_exponential_dip(57506, 57514, idx=index, sign='negative', name='exp2', vary=False)

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
                            'extra_sigs':m + mean_sw + ppta_dip,
                            'psr_model':True,
                            'red_var': False,
                            'chrom_df':None,
                            'dm_df':None,
                            'tm_marg':False,
                            'tm_svd':False,
                            'vary_dm':False,
                            'vary_chrom':False})
            ## Treat all other Adv Noise pulsars the same
            else:
                ### Turn SW model off. Add in stand alone SW model and common process. Return model.
                kwargs.update({'dm_sw_deter':False,
                               'white_vary':args.vary_wn,
                               'extra_sigs':m + mean_sw,
                               'psr_model':True,
                            'chrom_df':None,
                            'dm_df':None,
                            'red_var': False,
                            'tm_marg':False,
                            'vary_dm':False,
                            'tm_svd':False,
                            'vary_chrom':False})
            
            ### Load the appropriate single_pulsar_model
            psr_models.append(model_singlepsr_noise(new_psr, **kwargs))#(new_psr))
            final_psrs.append(new_psr)
        # Treat all other DMX pulsars in the standard way
        elif not args.adv_noise_psrs_only:
            s2 = s + tm + blocks.white_noise_block(vary=False,
                                                   tnequad=False,
                                                   inc_ecorr=True,
                                                   select='backend')
            psr_models.append(s2)#(psr))
            final_psrs.append(psr)

        print(f'\r{psr.name} Complete.',end='',flush=True)

    crn_models = [(m + cs)(psr) for psr,m in  zip(final_psrs,psr_models)]
    # gw_models = [(m + gw)(psr) for psr,m in  zip(final_psrs,psr_models)]

    pta_crn = signal_base.PTA(crn_models)
    # pta_gw = signal_base.PTA(gw_models)

    # # delta_common=0.,
    # ptas = {0:pta_crn,
    #         1:pta_gw}

    pta_crn.set_default_params(noise)

    if args.mk_ptapkl:
        with open(args.pta_pkl,'wb') as fout:
            cloudpickle.dump(pta_crn,fout)

groups = sampler.get_parameter_groups(pta_crn)
groups.extend(sampler.get_psr_groups(pta_crn))
Sampler = sampler.setup_sampler(pta_crn, outdir=args.outdir, resume=True,
                            empirical_distr = args.emp_distr, groups=groups)
    

   

# Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_empirical_distr, 70)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_prior, 10)
# Sampler.addProposalToCycle(Sampler.jp.draw_from_empirical_distr, 120)
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

# if args.sw_fit_path is None:
    # sampler.JumpProposal.draw_from_sw_prior = draw_from_sw_prior
    # sampler.JumpProposal.draw_from_sw4p39_prior = draw_from_sw4p39_prior

    # Sampler.addProposalToCycle(Sampler.jp.draw_from_sw_prior, 25)
    # Sampler.addProposalToCycle(Sampler.jp.draw_from_sw4p39_prior, 25)

if args.psd == 'spectrum':
    def draw_from_rho_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        parnames = [par.name for par in self.params]
        pname = [pnm for pnm in parnames if 'rho' in pnm][0]

        idx = parnames.index(pname)
        param = self.params[idx]

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

    sampler.JumpProposal.draw_from_rho_prior = draw_from_rho_prior
    Sampler.addProposalToCycle(Sampler.jp.draw_from_rho_prior, 25)

if args.psd =='powerlaw' and args.gamma_gw is None:
    sampler.JumpProposal.draw_from_gw_gamma_prior = draw_from_gw_gamma_prior
    Sampler.addProposalToCycle(Sampler.jp.draw_from_gw_gamma_prior, 25)


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

print('Signal Names', Sampler.jp.snames)

Sampler.sample(x0, args.niter, ladder=ladder, SCAMweight=200, AMweight=100,
               DEweight=200, burn=3000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)

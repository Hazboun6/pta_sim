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
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)

    model_paths = sorted(glob.glob('/Users/hazboun/nanograv_detection/12p5yr/nm_select/final_results/final_model_kwargs/*.json'))

    adv_noise_psr_list = ['B1855+09',
                          'B1937+21',
                          'J0613-0200',
                          'J0645+5158',
                          'J1600-3053',
                          'J1614-2230',
                          'J1640+2224',
                          'J1713+0747',
                          'J1741+1351',
                          'J1744-1134',
                          'J2043+1711',]

    # Set Tspan for RN

    Tspan_PTA = model_utils.get_tspan(pkl_psrs)
    # common red noise block
    cs = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan_PTA,
                                       components=5, gamma_val=4.33, name='gw')
    gw = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan_PTA,
                                       components=5, gamma_val=4.33, name='gw', orf='hd')
    # intrinsic red noise
    s = blocks.red_noise_block(prior='log-uniform', Tspan=Tspan_PTA, components=30)
    # timing model
    s += gp_signals.TimingModel()
    # adding white-noise, separating out Adv Noise Psrs, and acting on psr objects
    final_psrs = []
    psr_models = []
    ### Add a stand alone SW deter model
    n_earth = chrom.solar_wind.ACE_SWEPAM_Parameter()('n_earth')
    deter_sw = chrom.solar_wind.solar_wind(n_earth=n_earth)
    mean_sw = deterministic_signals.Deterministic(deter_sw, name='sw')
    for psr in pkl_psrs:
        # Filter out other Adv Noise Pulsars
        if psr.name in adv_noise_psr_list:
            ### Get the new pulsar object
            ## Remember that J1713's pickle is something you made yourself ##
            filepath = '/gscratch/gwastro/hazboun/nanograv/noise/noise_model_selection/no_dmx_pickles/'
            filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(psr.name)
            with open(filepath,'rb') as fin:
                new_psr=pickle.load(fin)

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
                               'white_vary': False,
                               'extra_sigs':dmgp + dmgp2 + chromgp + mean_sw,
                               'psr_model':True})
            ## Treat all other Adv Noise pulsars the same
            else:
                ### Turn SW model off. Add in stand alone SW model and common process. Return model.
                kwargs.update({'dm_sw_deter':False,
                               'white_vary': False,
                               'extra_sigs':mean_sw,
                               'psr_model':True})
            ### Load the appropriate single_pulsar_model
            psr_models.append(model_singlepsr_noise(new_psr, **kwargs))#(new_psr))
            final_psrs.append(new_psr)
        # Treat all other DMX pulsars in the standard way
        elif not args.adv_noise_psrs_only:
            s2 = s + blocks.white_noise_block(vary=False, inc_ecorr=True, select='backend')
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

    with open(args.pta_pkl,'wb') as fout:
        cloudpickle.dump(pta_crn,fout)

groups = sampler.get_parameter_groups(pta_crn)
groups.extend(sampler.get_psr_groups(pta_crn))
Sampler = sampler.setup_sampler(pta_crn, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr, groups=groups)

Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_empirical_distr, 70)
Sampler.addProposalToCycle(Sampler.jp.draw_from_psr_prior, 10)
Sampler.addProposalToCycle(Sampler.jp.draw_from_empirical_distr, 70)
Sampler.addProposalToCycle(Sampler.jp.draw_from_red_prior, 50)
Sampler.addProposalToCycle(Sampler.jp.draw_from_dm_gp_prior, 40)
Sampler.addProposalToCycle(Sampler.jp.draw_from_chrom_gp_prior, 30)
Sampler.addProposalToCycle(Sampler.jp.draw_from_dmexpcusp_prior, 30)
Sampler.addProposalToCycle(Sampler.jp.draw_from_par_prior(['n_earth']), 30)

try:
    achrom_freqs = get_freqs(pta_crn, signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass


noise['gw_log10_A'] = np.log10(2e-15)
x0 = np.array([noise[k] for k in pta_crn.param_names])

Sampler.sample(x0, args.niter, SCAMweight=50, AMweight=30,
               DEweight=60, burn=300000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

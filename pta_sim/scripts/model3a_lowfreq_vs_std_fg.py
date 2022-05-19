#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
import cloudpickle
import logging

from enterprise_extensions import models, model_utils, hypermodel, sampler
from enterprise.signals.signal_base import PTA
from enterprise.signals import (gp_signals,
                                signal_base,
                                deterministic_signals,
                                parameter,
                                selections,
                                white_signals,
                                utils)
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise import constants as const

from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions import blocks, model_orfs
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



if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        ptas = cloudpickle.load(f)

    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)
else:
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)


    # Set Tspan for RN

    Tspan_PTA = model_utils.get_tspan(pkl_psrs)
    # common red noise block
    fmin = 10.0
    modes, wgts = model_utils.linBinning(Tspan_PTA, 0,
                                         1.0 / fmin / Tspan_PTA,
                                         14, 5)
    # wgts = wgts**2.0

    # timing model
    s = gp_signals.MarginalizingTimingModel()

    s += blocks.white_noise_block(vary=False, inc_ecorr=True, select='backend')

    rn_low = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',
                                    Tspan=Tspan_PTA, modes=modes, wgts=wgts,)
    rn_std = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',
                                    Tspan=Tspan_PTA, components=30)

    gamma_gw = parameter.Constant(4.3333)('gw_gamma')
    log10_Agw = parameter.Uniform(-18, -14)('gw_log10_A')
    plaw_low = gpp.powerlaw_genmodes(log10_A=log10_Agw,
                                     gamma=gamma_gw,
                                     wgts=wgts)
    plaw_std = gpp.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)

    gw_std = gp_signals.FourierBasisCommonGP(plaw_std,
                                             model_orfs.hd_orf(),
                                             components=14,
                                             Tspan=Tspan_PTA,
                                             name='gw')


    gw_low = gp_signals.FourierBasisCommonGP(plaw_low,
                                             model_orfs.hd_orf(),
                                             modes=modes,
                                             name='gw')


    std_models = [(s + rn_std + gw_std)(psr) for psr in pkl_psrs]
    low_models = [(s + rn_low + gw_low)(psr) for psr in pkl_psrs]

    pta_std = signal_base.PTA(std_models)
    pta_low = signal_base.PTA(low_models)

    ptas = {0:pta_std,
            1:pta_low}

    pta_std.set_default_params(noise)
    pta_low.set_default_params(noise)

    with open(args.pta_pkl,'wb') as fout:
        cloudpickle.dump(ptas,fout)

hm = hypermodel.HyperModel(models=ptas)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr = args.emp_distr)


try:
    gw_std_freqs = get_freqs(pta_std, signal_id='gw')
    gw_low_freqs = get_freqs(pta_low, signal_id='gw')
    np.savetxt(args.outdir + 'low_gw_freqs.txt', gw_low_freqs, fmt='%.18e')
    np.savetxt(args.outdir + 'std_gw_freqs.txt', gw_std_freqs, fmt='%.18e')
except:
    pass

model_params = {}
for ii,mod in enumerate(ptas):
    model_params.update({ii : ptas[ii].param_names})

with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params, fout, sort_keys=True, indent=4,
              separators=(',', ': '))

x0 = hm.initial_sample()

sampler.sample(x0, args.niter, SCAMweight=100, AMweight=100,
               DEweight=100, burn=50000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)

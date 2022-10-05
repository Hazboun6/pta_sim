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

with open('{0}'.format(args.pickle), "rb") as f:
    psrs = pickle.load(f)

psr = psrs[args.process]

#### Nihan's Sine wave
dataset_tmin = 4597873783.54894 #np.min([psr.toas.min() for psr in psrs])
@parameter.function
def sine_wave(toas, flags, A = -9, f = -9, phase = 0.0):
    return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)

def sine_signal(A, f, phase, name = ""):
    return deterministic_signals.Deterministic(sine_wave(A = A, f = f, phase = phase), name = name)

day_seconds = 86400
sin = sine_signal(A = parameter.Uniform(-9, -4)('common_sin_A'), f = parameter.Uniform(-9, -7)('common_sin_f'), phase = parameter.Uniform(0, 2 * np.pi)('common_sin_phase'))


### Turn SW model off. Add in stand alone SW model and common process. Return model.
kwargs={'white_vary':args.vary_wn,
        'extra_sigs':sin,
        'red_var': True,
        'tm_marg':True,
        'tnequad':True}

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

pta = model_singlepsr_noise(psr, **kwargs)
pta.set_default_params(noise)

groups = sampler.get_parameter_groups(pta)
groups.extend(sampler.get_psr_groups(pta))
Sampler = sampler.setup_sampler(pta, outdir=args.outdir+f'{psr.name}/', resume=True,
                                empirical_distr = args.emp_distr, groups=groups)

Sampler.addProposalToCycle(Sampler.jp.draw_from_empirical_distr, 120)

try:
    achrom_freqs = get_freqs(pta, signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

x0 = np.hstack([p.sample() for p in pta.params])


Sampler.sample(x0, args.niter, SCAMweight=200, AMweight=100,
               DEweight=200, burn=3000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)

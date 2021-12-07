#!/usr/bin/env python
# coding: utf-8

import numpy as np
import astropy.units as u

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import gp_bases, gp_priors
from enterprise.signals import deterministic_signals
from enterprise import constants as const

import corner, pickle, sys, json
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import models, model_utils, sampler, blocks

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

with open(args.pickle,'rb')as fin:
    psrs = pickle.load(fin)

psr = psrs[args.process]
print(f'Starting {psr.name}.')
with open(args.noisepath,'r') as fin:
    noise = json.load(fin)

if args.tspan is None:
    Tspan = model_utils.get_tspan([psr])
else:
    Tspan = args.tspan

tm = gp_signals.TimingModel()
log10_rho = parameter.Uniform(-10,-4,size=30)
fs = gp_priors.free_spectrum(log10_rho=log10_rho)
wn = blocks.white_noise_block(inc_ecorr=True)

log10_A = parameter.Constant()
gamma = parameter.Constant()
plaw_pr = gp_priors.powerlaw(log10_A=log10_A,gamma=gamma)
plaw = gp_signals.FourierBasisGP(plaw_pr,components=30,Tspan=Tspan)
rn  = gp_signals.FourierBasisGP(fs,components=30,Tspan=Tspan, name='excess_noise')

m = tm + wn + plaw + rn
if args.gwb_on:
    gw_log10_A = parameter.Constant('gw_log10_A')
    gw_gamma = parameter.Constant(4.3333)('gw_gamma')
    gw_pr = gp_priors.powerlaw(log10_A=gw_log10_A,gamma=gw_gamma)
    gwb = gp_signals.FourierBasisGP(gw_pr,components=args.n_gwbfreqs,Tspan=Tspan)
    m += gwb
    
pta = signal_base.PTA(m(psr))
pta.set_default_params(noise)
x0 = np.hstack(p.sample() for p in pta.params)
samp = sampler.setup_sampler(pta,outdir=args.outdir+f'/{psr.name}/',resume=False)

N = args.niter
samp.sample(x0, Niter=N, burn=200000)
print(f'{psr.name} complete.')

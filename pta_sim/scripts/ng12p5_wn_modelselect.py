#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy
import logging
import numpy as np
import scipy.stats as sps

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise import constants as const
from enterprise_extensions import models, model_utils, sampler, dropout

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psrs = pickle.load(fin)

pidx = args.process
psr = psrs[pidx]

backend = selections.Selection(selections.by_backend)
ng_backend = selections.Selection(selections.nanograv_backends)

efac = parameter.Uniform(0.1,5)
equad = parameter.Uniform(-8.5,-5)
ecorr = parameter.Uniform(-8.5,-5)
kdrop = parameter.Uniform(0,1)
kthresh = 0.5

ef = dropout.dropout_MeasurementNoise(efac=efac,
                                      efac_drop=kdrop,
                                      efac_thresh=kthresh,
                                      selection=backend)
eq = dropout.dropout_EquadNoise(log10_equad=equad,
                                equad_drop=kdrop,
                                equad_thresh=kthresh,
                                selection=backend)
ec = dropout.dropout_EcorrBasisModel(log10_ecorr=ecorr,
                                     ecorr_drop=kdrop,
                                     ecorr_thresh=kthresh,
                                     selection=ng_backend)

tm = gp_signals.TimingModel()

log10_A = parameter.Uniform(-18,-12)
gamma = parameter.Uniform(0,7)
plaw = dropout.dropout_powerlaw(log10_A=log10_A,gamma=gamma,k_drop=kdrop,)
rn  = gp_signals.FourierBasisGP(plaw, components=30)

m = tm + ef + eq + ec + rn

pta = signal_base.PTA([m(psr)])

x0 = np.hstack(p.sample() for p in pta.params)

# sampler for N steps
N = args.niter

if psr.name in ['B1937+21','J1713+0747']:
    N *= 2

samp = sampler.setup_sampler(pta,
                             outdir=args.outdir+f'{psr.name}/',
                             resume=True)

samp.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=int(N//4),
               writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

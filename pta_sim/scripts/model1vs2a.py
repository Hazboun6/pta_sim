#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
from shutil import copyfile, copy2

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
from enterprise import constants as const

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import models, model_utils, hypermodel, dropout, blocks
from enterprise_extensions.frequentist import optimal_statistic as OS


import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.sim_gw import Simulation, model_simple
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

#Is chain longer than niter?
longer = chain_length_bool(args.outdir, args.niter)

if longer and os.path.exists(args.core_path):
    sys.end()
elif longer:
    save_core(args.corepath, args.outdir)
    sys.end() #Hmmmm what to do here?
else:
    pass

if args.pickle=='no_pickle':
    #Get par and tim files.
    parfiles = sorted(glob.glob(args.pardir+'*.par'))
    timfiles = sorted(glob.glob(args.timdir+'*.tim'))

    if args.psr_list is not None:
        parfiles = filter_psr_path(parfiles,args.psr_list,rhs='_')
        timfiles = filter_psr_path(timfiles,args.psr_list,rhs='_')

    psrs = []
    for p, t in zip(parfiles, timfiles):
        psr = Pulsar(p, t, ephem=args.ephem)
        psrs.append(psr)

else:
    with open('{0}'.format(args.pickle), "rb") as f:
        psrs = pickle.load(f)

    if args.psr_list is not None:
        idxs = []
        for idx, psr in enumerate(psrs):
            if psr.name not in args.psr_list:
                idxs.append(idx)

        for idx in reversed(idxs):
            del psrs[idx]


pidxs = []
for pidx, psr in enumerate(psrs):
    start_time = psr.toas.min()/(24*3600)
    last_time = psr.toas.max()/(24*3600)
    if (last_time-start_time)/365.25 < args.min_tspan:
        print('PSR {0} baseline less than {1} years. Not being included in analysis'.format(psr.name,args.min_tspan))
        pidxs.append(pidx)

for idx in reversed(pidxs):
    del psrs[idx]


with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if args.tspan is None:
    Tspan = model_utils.get_tspan(psrs)
else:
    Tspan=args.tspan


### Timing Model ###
tm = gp_signals.TimingModel()



### Red Noise ###
# Code for red noise dropout
if args.dropout:
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    k_drop = parameter.Uniform(0, 1)
    if args.dp_thresh == 6.0:
        dp_thresh = parameter.Uniform(0,1)('k_threshold')
    else:
        dp_thresh = args.dp_thresh
    pl = dropout.dropout_powerlaw(log10_A=log10_A, gamma=gamma,
                                  k_drop=k_drop, k_threshold=dp_thresh)
    rn_plaw = gp_signals.FourierBasisGP(pl, components=30,
                                        Tspan=Tspan, name='red_noise')

else:
    rn_plaw = models.red_noise_block(psd='powerlaw', prior='log-uniform',
                                     Tspan=Tspan, components=30,
                                     gamma_val=None)

### GWB ###
gw = models.common_red_noise_block(psd=args.psd, prior='log-uniform',
                                   Tspan=Tspan, gamma_val=13/3., name='gw',
                                   components=args.nfreqs,
                                   delta_val=0.0)
base_model = tm

if args.bayes_ephem:
    base_model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

if args.rn_psrs[0]=='all':
    rn_psrs='all'
else:
    rn_psrs=args.rn_psrs

model1_psrs = []
model2a_psrs = []
if rn_psrs=='all':
    model_1 = base_model + rn_plaw
    model_2a = model_1 + gw
    model_1_ec = model_1 + blocks.white_noise_block(vary=False, inc_ecorr=True)
    model_1 += blocks.white_noise_block(vary=False, inc_ecorr=False)
    model_2a_ec = model_2a + blocks.white_noise_block(vary=False, inc_ecorr=True)
    model_2a += blocks.white_noise_block(vary=False, inc_ecorr=False)

    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not args.wideband:
            model1_psrs.append(model_1_ec(p))
            model2a_psrs.append(model_2a_ec(p))
        else:
            model1_psrs.append(model_1(p))
            model2a_psrs.append(model_2a(p))

elif isinstance(rn_psrs,list):
    model1_psrs = []
    model2a_psrs = []
    model_2a_base = base_model + gw
    model_1 = base_model + rn_plaw
    model_2a = model_1 + gw
    for p in psrs:
        if p.name in rn_psrs:
            model1_psrs.append(model_1(p))
            model2a_psrs.append(model_2a(p))
        else:
            model1_psrs.append(base_model(p))
            model2a_psrs.append(model_2a_base(p))

pta_noise = signal_base.PTA(model1_psrs)
pta_noise.set_default_params(noise)

pta_gw = signal_base.PTA(model2a_psrs)
pta_gw.set_default_params(noise)

ptas = {0:pta_noise,
        1:pta_gw}

if args.model_wts is None:
    model_wts = None
else:
    model_wts = dict(enumerate(args.model_wts))

hm = hypermodel.HyperModel(models=ptas, log_weights=model_wts)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr=args.emp_distr)

model_params = {}
for ii,mod in enumerate(ptas):
    model_params.update({ii : ptas[ii].param_names})

with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params, fout, sort_keys=True, indent=4,
              separators=(',', ': '))

x0 = hm.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=20, DEweight=50,
               burn=300000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

save_core(args.corepath, args.outdir)

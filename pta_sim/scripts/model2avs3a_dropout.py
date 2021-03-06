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
from enterprise_extensions import models, model_utils, hypermodel, dropout
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

with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if args.tspan is None:
    Tspan = model_utils.get_tspan(psrs)
else:
    Tspan=args.tspan

if args.wideband:
    inc_ecorr = False
else:
    inc_ecorr = True

### Timing Model ###
tm = gp_signals.TimingModel()
### White Noise ###
wn = models.white_noise_block(vary=False, inc_ecorr=inc_ecorr)

### Red Noise ###
# Code for red noise dropout
if args.gwb_ul:
    prior = 'uniform'
else:
    prior = 'log-uniform'

if args.dropout:
    if args.gwb_ul:
        log10_A = parameter.LinearExp(-20, -11)
    else:
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
    rn_plaw = models.red_noise_block(psd='powerlaw', prior=prior,
                                     Tspan=Tspan, components=30,
                                     gamma_val=None)

### GWB ###
crn = models.common_red_noise_block(psd='powerlaw', prior=prior,
                                    components=args.n_gwbfreqs,
                                    Tspan=Tspan, gamma_val=13/3., name='gw')

gw = models.common_red_noise_block(psd='powerlaw', prior=prior,
                                   components=args.n_gwbfreqs, orf='hd',
                                   Tspan=Tspan, gamma_val=13/3., name='gw')
base_model = tm + wn

if args.bayes_ephem:
    base_model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

if args.rn_psrs[0]=='all':
    rn_psrs='all'
else:
    rn_psrs=args.rn_psrs

if rn_psrs=='all':
    model_2a = base_model + rn_plaw + crn
    model_3a = base_model + rn_plaw + gw
    model2a_psrs = [model_2a(p) for p in psrs]
    model3a_psrs = [model_3a(p) for p in psrs]
elif isinstance(rn_psrs,list):
    model2a_psrs = []
    model3a_psrs = []
    model_2a_base = base_model + crn
    model_3a_base = base_model + gw
    model_2a = base_model + rn_plaw + crn
    model_3a = base_model + rn_plaw + gw
    for p in psrs:
        if p.name in rn_psrs:
            model2a_psrs.append(model_2a(p))
            model3a_psrs.append(model_3a(p))
        else:
            model2a_psrs.append(model_2a_base(p))
            model3a_psrs.append(model_3a_base(p))

pta_crn = signal_base.PTA(model2a_psrs)
pta_crn.set_default_params(noise)

pta_gw = signal_base.PTA(model3a_psrs)
pta_gw.set_default_params(noise)

ptas = {0:pta_crn,
        1:pta_gw}

if args.emp_distr is None:
    emp_dist = '/home/jeffrey.hazboun/nanograv/Data/pickles/ng11yr_v2_std_plaw_emp_dist.pkl'
else:
    emp_dist = args.emp_distr

hm = hypermodel.HyperModel(models=ptas)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr=emp_dist)

achrom_freqs = get_freqs(ptas[0],signal_id='gw')
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')


x0 = hm.initial_sample()
k_drop_idxs = np.where(['k_drop' in p for p in hm.param_names])
x0[k_drop_idxs] = 1.0
print('Initial Sample: ',x0)
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=args.writeHotChains,)

save_core(args.corepath, args.outdir)

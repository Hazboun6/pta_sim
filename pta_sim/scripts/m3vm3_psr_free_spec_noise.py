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
from enterprise_extensions import models, model_utils
from enterprise_extensions import blocks, sampler, hypermodel
from enterprise_extensions.frequentist import optimal_statistic as OS

sys.path.insert(0,'/Users/hazboun/software_development/pta_sim/')

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.sim_gw import Simulation, model_simple
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

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
            if psr.name in args.psr_list:
                idxs.append(idx)

        for idx in reversed(idxs):
            del psrs[idx]

noise_plaw = {}
with open(args.noisepath, 'r') as fin:
    noise_plaw.update(json.load(fin))

noise_fs = {}
with open(args.noisepath, 'r') as fin:#.replace('plaw','fs')
    noise_fs.update(json.load(fin))


if args.tspan is None:
    Tspan = model_utils.get_tspan(psrs)
else:
    Tspan = args.tspan

if args.wideband:
    inc_ecorr = False
else:
    inc_ecorr = True

### White Noise ###
wn = blocks.white_noise_block(vary=False, inc_ecorr=inc_ecorr)
### Red Noise ###
rn_plaw = blocks.red_noise_block(psd='powerlaw', prior='log-uniform',
                                 Tspan=Tspan, components=30, gamma_val=None)

rn_fs = blocks.red_noise_block(psd='spectrum', prior='log-uniform',
                               Tspan=Tspan, components=30, gamma_val=None)

### GWB ###
gw = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                   Tspan=Tspan, components=args.n_gwbfreqs,
                                   gamma_val=args.gamma_gw, name='gw')

base_model = wn + gw

if args.bayes_ephem:
    base_model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

model_plaw = base_model + rn_plaw
model_fs = base_model + rn_fs

model_list = []
noise = {}
for psr in psrs:
    if psr.name in args.free_spec_psrs:
        model_list.append(model_fs(psr))
        noise.update(noise_fs[psr.name])
    else:
        model_list.append(model_plaw(psr))
        noise.update(noise_plaw[psr.name])

pta_gw_fs = signal_base.PTA(model_list)
pta.set_default_params(noise)


pta_gw = models.model_3a(psrs, psd=args.psd, noisedict=noise,
                         n_gwbfreqs=args.nfreqs, gamma_common=args.gamma_gw,
                         delta_common=0., upper_limit=False,
                         bayesephem=args.bayes_ephem, be_type='setIII',
                         wideband=False, rn_psrs=rn_psrs,
                         pshift=args.pshift)
ptas = {0:pta_gw,
        1:pta_gw_fs}

if args.model_wts is None:
    model_wts = None
else:
    model_wts = dict(enumerate(args.model_wts))

hm = hypermodel.HyperModel(models=ptas, log_weights=model_wts)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr = args.emp_distr)

try:
    achrom_freqs = get_freqs(ptas[0], signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

model_params = {}
for ii,mod in enumerate(ptas):
    model_params.update({ii : ptas[ii].param_names})

with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params, fout, sort_keys=True, indent=4,
              separators=(',', ': '))

x0 = hm.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15,
               DEweight=30, burn=300000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

save_core(args.corepath, args.outdir)

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
from enterprise_extensions.frequentist import optimal_statistic as OS

sys.path.insert(0,'/Users/hazboun/software_development/pta_sim/')

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
    noise_json =json.load(fin)

noise = {}
for p in psrs:
    noise.update(noise_json[p.name])

if args.tspan is None:
    Tspan = model_utils.get_tspan(psrs)
else:
    Tspan=args.tspan

if args.wideband:
    inc_ecorr = False
else:
    inc_ecorr = True

### White Noise ###
wn = models.white_noise_block(vary=False, inc_ecorr=inc_ecorr)
### Red Noise ###
rn_plaw = models.red_noise_block(psd='powerlaw', prior='log-uniform',
                                 Tspan=Tspan, components=30, gamma_val=None)
### GWB ###
gw = models.common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                   Tspan=Tspan, gamma_val=13/3., name='gw')
base_model = wn

if args.bayes_ephem:
    base_model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

model_1 = base_model + rn_plaw
model_2a = model_1 + gw

pta_noise = signal_base.PTA([model_1(p) for p in psrs])
pta_noise.set_default_params(noise)

pta_gw = signal_base.PTA([model_2a(p) for p in psrs])
pta_gw.set_default_params(noise)

ptas = {0:pta_noise,
        1:pta_gw}

hm = model_utils.HyperModel(models=ptas)
sampler = hm.setup_sampler(outdir=args.outdir,resume=True)

achrom_freqs = get_freqs(ptas[0])
# np.save(args.outdir + 'pars.npy', pta.param_names)
# np.save(args.outdir + 'par_model.npy', np.array(pta.params).astype(str))
# np.save(args.outdir + 'signals.npy', list(pta.signals.keys()))
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')


x0 = hm.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15, DEweight=50, )

save_core(args.corepath, args.outdir)

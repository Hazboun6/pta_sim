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
            if psr.name in args.psr_list:
                idxs.append(idx)

        for idx in reversed(idxs):
            del psrs[idx]

noise_plaw = {}
with open(args.noisepath, 'r') as fin:
    noise_plaw.update(json.load(fin))

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
                                   Tspan=Tspan, gamma_val=None, name='gw')
base_model = wn + gw

if args.bayes_ephem:
    base_model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

model_plaw = base_model + rn_plaw

model_list = []
noise = {}
for psr in psrs:
    if psr.name in args.free_spec_psrs:
        model_list.append(model_fs(psr))
        noise.update(noise_fs[psr.name])
    else:
        model_list.append(model_plaw(psr))
        noise.update(noise_plaw[psr.name])

pta = signal_base.PTA(model_list)
pta.set_default_params(noise)

x0 = np.hstack(p.sample() for p in pta.params)
ndim = x0.size

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups

groups = model_utils.get_parameter_groups(pta)
if args.bayes_ephem:
    eph_pars = ['d_jupiter_mass', 'd_neptune_mass', 'd_saturn_mass',
                'd_uranus_mass', 'frame_drift_rate',
                'jup_orb_elements_0', 'jup_orb_elements_1',
                'jup_orb_elements_2', 'jup_orb_elements_3',
                'jup_orb_elements_4', 'jup_orb_elements_5']
    ephem_idx = [pta.param_names.index(par)
                 for par in pta.param_names
                 if par in eph_pars]
    groups.append(ephem_idx)

gw_idx = [pta.param_names.index(par) for par in pta.param_names if 'gw' in par]
groups.append(gw_idx)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups, outDir=args.outdir, resume=True)

achrom_freqs = get_freqs(pta)
np.save(args.outdir + 'pars.npy', pta.param_names)
np.save(args.outdir + 'par_model.npy', np.array(pta.params).astype(str))
np.save(args.outdir + 'signals.npy', list(pta.signals.keys()))
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')

jp = model_utils.JumpProposal(pta, empirical_distr = args.emp_distr)
sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_red_prior, 15)
sampler.addProposalToCycle(jp.draw_from_empirical_distr, 80)
if args.bayes_ephem:
    sampler.addProposalToCycle(jp.draw_from_ephem_prior, 15)

N = int(args.niter)

sampler.sample(x0, Niter=N, SCAMweight=30, AMweight=15,
               DEweight=50, burn=100000)

save_core(args.corepath, args.outdir)

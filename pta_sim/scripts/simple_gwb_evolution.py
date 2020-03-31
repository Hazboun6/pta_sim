#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
from shutil import copyfile, copy2

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

#Get par and tim files.
parfiles = sorted(glob.glob(args.pardir+'*.par'))
timfiles = sorted(glob.glob(args.timdir+'*.tim'))

if args.psr_list is not None:
    parfiles = filter_psr_path(parfiles,args.psr_list,rhs='_')
    timfiles = filter_psr_path(timfiles,args.psr_list,rhs='_')

sim = Simulation(parfiles, timfiles, ephem=args.ephem, verbose=True)

if args.process is None:
    seed_gwb = None
else:
    seed_gwb = 1978 + args.process

for ii,p in enumerate(sim.libs_psrs):
    LT.add_efac(p, seed=seed_gwb+ii)

sim.createGWB(A_gwb=args.A_gwb, gamma_gw=args.gamma_gw,
              seed=seed_gwb, fit=args.fit)

sim.init_ePulsars()

# noise = {}
# with open(args.noisepath, 'r') as fin:
#     noise.update(json.load(fin))
#
# pta = models.model_2a(psrs=sim.psrs, psd='powerlaw', noisedict=noise,
#                       components=30, gamma_common=args.gamma_gw,
#                       upper_limit=True, bayesephem=False)

sim.filter_by_mjd(args.end_time)
pta = model_simple(psrs=sim.psrs, psd='powerlaw', components=30,
                   gamma_common=args.gamma_gw, upper_limit=args.gwb_ul,
                   efac=args.efac, hd_orf=args.hd,
                   bayesephem=False, select='backend', red_noise=args.rednoise,
                   Tspan=args.tspan)

x0 = np.hstack(p.sample() for p in pta.params)
ndim = x0.size

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups

groups = model_utils.get_parameter_groups(pta)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups, outDir=args.outdir, resume=True)

achrom_freqs = get_freqs(pta)
np.save(args.outdir + 'pars.npy', pta.param_names)
np.save(args.outdir + 'par_model.npy', np.array(pta.params).astype(str))
np.save(args.outdir + 'signals.npy', list(pta.signals.keys()))
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')

jp = model_utils.JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_red_prior, 15)
sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 15)
if args.bayes_ephem:
    sampler.addProposalToCycle(jp.draw_from_ephem_prior, 15)

N = int(args.niter)

sampler.sample(x0, Niter=N, SCAMweight=30, AMweight=15,
               DEweight=50, burn=100000)

save_core(args.corepath, args.outdir)

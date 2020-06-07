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
from enterprise_extensions import models, model_utils, sampler
from enterprise_extensions.frequentist import optimal_statistic as OS
from la_forge.core import load_Core

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.sim_gw import Simulation, model_simple
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()


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

if os.path.exists(args.rn_psrs[0]) and len(args.rn_psrs)==1:
    with open(args.rn_psrs[0],'r') as fin:
        rn_psrs = json.load(fin)
    print('Adding RN to the following pulsars.')
    print(list(rn_psrs.keys()))
    sim.add_rn(rn_psrs, seeds=[seed_gwb+314+ii for ii in range(len(rn_psrs))])

sim.createGWB(A_gwb=args.A_gwb, gamma_gw=args.gamma_gw,
              seed=seed_gwb, fit=args.fit)

sim.init_ePulsars()

sim.filter_by_mjd(args.end_time)
pta = model_simple(psrs=sim.psrs, psd='powerlaw', components=30,
                   vary_gamma=args.vary_gamma, upper_limit=args.gwb_ul,
                   efac=args.efac, hd_orf=args.hd, rn_dropout=args.dropout,
                   bayesephem=False, select='backend', red_noise=args.rednoise,
                   Tspan=args.tspan, dp_threshold=args.dp_thresh)

os = OS.OptimalStatistic(sim.psrs, bayesephem=False, gamma_common=4.33,
                         orf='hd', pta=pta)

c0 = load_Core(args.corepath)

np.save(args.outdir+'os_max_{0}_{1}'.format(args.nyears,args.process),
        os.compute_noise_maximized_os(c0.chain))
np.save(args.outdir+'os_marg_{0}_{1}'.format(args.nyears,args.process),
        os.compute_noise_marginalized_os(c0.chain, N=1000))

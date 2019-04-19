#!/usr/bin/env python
# coding: utf-8

# Code for running analyzing simulations using the Optimal Statistic
# Libstempo is used to do the simulating

from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
from shutil import copyfile, copy2
import scipy.stats as sci_stats

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
import enterprise.constants as const

from enterprise_extensions import models, model_utils
from enterprise_extensions.frequentist import optimal_statistic as OS

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.sim_gw import Simulation

args = parse_sim.arguments()

outdir_str = args.outpath.split('/')
outdir = '/'.join(outdir_str[:-1])
if not os.path.exists(outdir):
    os.makedirs(outdir)

#Get par and tim files.
parfiles = sorted(glob.glob(args.pardir+'*.par'))
timfiles = sorted(glob.glob(args.timdir+'*.tim'))

#cuts = [57000., 57450.,
#        57900., 58350.,
#        58800., 59034.,
#        59269., 59503.,
#        59738., 60207.,
#        60676., 61589.,
#        62502., 63415.,
#        64328., 65241.,
#        66154.]
cuts = np.linspace(57000.,66154.,10)
#cuts = [57000, 58800, 60676.,  62502.,  64328.,  66154.]

sim = Simulation(parfiles, timfiles, ephem=args.ephem,verbose=True)

cuts = cuts[::-1]
seed_gwb = args.process

sim.createGWB(A_gwb=args.A_gwb, gamma_gw=args.gamma_gw, seed=seed_gwb)

sim.init_ePulsars()

pta = model_simple(sim.psrs, gamma_common=args.gamma_gw)

OptStat = OS.OptimalStatistic(psrs=sim.psrs, pta=pta,
                              bayesephem=args.bayes_ephem)

par_dict = {'log10_A_gw':np.log10(args.A_gwb)}
xi, rho, sigma, Agwb_sqr, os_sigma = OptStat.compute_os(params=par_dict)

out = [[Agwb_sqr, os_sigma, sim.last_toa, sim.seed]]
np.savetxt(args.outpath, out, fmt='%e, %e, %f, %i')

print('MJD {0} analysis complete'.format(sim.last_toa))

for mjd in cuts:
    sim.filter_by_mjd(mjd)
    pta = model_simple(sim.psrs,gamma_common=args.gamma_gw)
    OptStat = OS.OptimalStatistic(psrs=sim.psrs, pta=pta,
                                  bayesephem=args.bayes_ephem)


    xi, rho, sigma, Agwb_sqr, os_sigma = OptStat.compute_os(params=par_dict)

    out.append([Agwb_sqr, os_sigma, mjd, sim.seed])

    np.savetxt(args.outpath, out, fmt='%e, %e, %f, %i')

    print('MJD {0} analysis complete'.format(mjd))

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import astropy.units as u
import glob, os, json, copy,sys
import scipy.linalg as sl

#Import all of the enterprise scripts
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

import corner, pickle
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import models, model_utils
from pint.residuals import resids
import pint.toa as toa
import matplotlib.pyplot as plt
from astropy import log
from pint import models
from astropy.time import Time, TimeDelta
import astropy.units as u

import pta_sim
import pta_sim.pint_sim as pint_sim
import pta_sim.noise as nutils
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()


from astropy import log
log.setLevel('CRITICAL')

par = args.parpath
tim = args.timpath

ts = toa.get_TOAs(tim,usepickle=False)
m = models.get_model(par)

if args.process is None:
    seed_rn = None
    seed_dmrn = None
    seed_ef = None
    seed_ec = None
    seed_eq = None
else:
    seed_rn = 1978 + args.process
    seed_dmrn = 666 + args.process
    seed_ef = 1492 + args.process
    seed_ec = 1554 + args.process
    seed_eq = 2000 + args.process

pint_sim.make_ideal(ts,m)
pint_sim.add_rednoise(ts, A=args.A_rn, gamma=args.gamma_rn, seed=seed_rn)
pint_sim.add_dm_rednoise(ts, A=args.A_dm, gamma=args.gamma_dm, seed=seed_dmrn)

noise_dict = nutils.load_noise_files(noisepath=args.noisepath)

eq_flags, equads = nutils.get_noise('equad','B1937+21',noise_dict)
ef_flags, efacs = nutils.get_noise('efac','B1937+21',noise_dict)
pint_sim.add_equad(ts,equads,flagid='f',flags=eq_flags,seed=seed_eq)
pint_sim.add_efac(ts,efacs,flagid='f',flags=ef_flags,seed=seed_ef)
ec_flags, ecorrs = nutils.get_noise('ecorr','B1937+21',noise_dict)
ec_flags.append('NICER'), ecorrs.append(0.0)
ec_flags, ecorrs
pint_sim.add_ecorr(ts,ecorrs,flagid='f',flags=ec_flags,seed=seed_ec)

psr = Pulsar(ts,m)

#### Set Up Enterprise Models ########
# Red noise parameter priors
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# White noise parameter priors
efac = parameter.Uniform(0.01, 10.0)
equad = parameter.Uniform(-8.5, -5)
ecorr = parameter.Uniform(-8.5, -5)

#Calculate the time span of the dataset
Tspan = psr.toas.max() - psr.toas.min()

## White Noise Block
# Define new mask and selection to avoid non-channelized NICER TOAs (for ECORR)
def channelized_backends(backend_flags):
    """Selection function to split by channelized backend flags only."""
    flagvals = np.unique(backend_flags)
    ch_b = ['ASP', 'GASP', 'GUPPI', 'PUPPI', 'NUPPI']
    flagvals = filter(lambda x: any(map(lambda y: y in x, ch_b)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}

selection_ch = selections.Selection(channelized_backends)
selection = selections.Selection(selections.by_backend)

ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection_ch)

# red noise (powerlaw with 5 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
basis = utils.createfourierdesignmatrix_red(Tspan=Tspan, nmodes=args.nfreqs,
                                            logf=args.logf)
rn = gp_signals.BasisGP(priorFunction=pl, name='red_noise', basisFunction=basis)

# timing model
tm = gp_signals.TimingModel(use_svd=False)

model = tm + ef + eq + ec + rn

if args.dm_gp_psrs[0] == args.psr:
    dm_basis = utils.createfourierdesignmatrix_dm(Tspan=Tspan,
                                                  nmodes=args.nfreqs,
                                                  logf=args.logf)
    dm_gp = gp_signals.BasisGP(priorFunction=pl,
                               basisFunction=dm_basis,
                               name='dm_gp')
    model += dm_gp

pta = signal_base.PTA(model(psr))

x0 = np.hstack(p.sample() for p in pta.params)
ndim = len(x0)

pta.get_lnlikelihood(x0)

groups = model_utils.get_parameter_groups(pta)
groups.extend([[pta.param_names.index(p) for p in pta.param_names if fl in p]
               for fl in ef_flags])

if args.dm_gp_psrs[0] == args.psr:
    gp_pars = ['{0}_dm_gp_gamma'.format(args.psr),
               '{0}_dm_gp_log10_A'.format(args.psr),
               '{0}_red_noise_gamma'.format(args.psr),
               '{0}_red_noise_log10_A'.format(args.psr)]
    groups.append([pta.param_names.index(p) for p in gp_pars])

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution

# sampler object
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov,
                 groups=groups, outDir=args.outdir, resume=True)

achrom_freqs = get_freqs(pta, 'red_noise')
np.save(args.outdir + 'pars.npy', pta.param_names)
np.save(args.outdir + 'par_model.npy', np.array(pta.params).astype(str))
np.save(args.outdir + 'signals.npy', list(pta.signals.keys()))
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')

jp = model_utils.JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 20)
sampler.addProposalToCycle(jp.draw_from_prior, 20)
# sampler.addProposalToCycle(jp.draw_from_red_prior, 20)

sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15,
               DEweight=50, burn=100000)

save_core(args.corepath, args.outdir, remove=True)

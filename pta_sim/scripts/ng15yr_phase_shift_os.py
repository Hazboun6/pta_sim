#!/usr/bin/env python
# coding: utf-8


import numpy as np
import la_forge.core as co
import pickle, json, copy
import matplotlib.pyplot as plt

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

from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as OS

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

with open(args.pickle,'rb')as fin:
    psrs = pickle.load(fin)

tmin = np.amin([p.toas.min() for p in psrs])
tmax = np.amax([p.toas.max() for p in psrs])
Tspan = tmax - tmin

# Red noise parameter priors
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# GW parameter priors
gw_log10_A = parameter.Uniform(-18, -14)('gw_log10_A')
gw_gamma = parameter.Constant(13./3)('gw_gamma')

# White noise parameter priors
efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()

Nf = args.nfreqs
freqs = np.linspace(1/Tspan,Nf/Tspan,Nf)


# # white noise
selection = selections.Selection(selections.nanograv_backends)

ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, modes=freqs)

# timing model
tm = gp_signals.TimingModel()

# gw (powerlaw with 5 frequencies)

gw_pl = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
gw_pshift = gp_signals.FourierBasisGP(spectrum=gw_pl,
                                      modes=freqs[:args.n_gwbfreqs],
                                      name='gw', pshift=True,
                                      pseed=args.process)

model_pshift = tm + ef + eq + ec + rn + gw_pshift


pta_pshift = signal_base.PTA([model_pshift(p) for p in psrs])
with open(args.noisepath,'r') as fin:
    noise = json.load(fin)


pta_pshift.set_default_params(noise)

os_pshift = OS(psrs=psrs, pta=pta_pshift, orf=args.orf)

c0 = co.load_Core(args.corepath)
chain = c0.chain[c0.burn:,:-4]
pars = c0.params[:-4]
N = args.niter
Ahat_pshift = np.zeros(N)
snr_pshift = np.zeros(N)
for ii in range(N):
    param_dict = {}
    idx = np.random.randint(0,chain.shape[0])
    param_dict = dict(zip(pars,chain[idx,:]))
    _, _, _, Asqr, Sigma = os_pshift.compute_os(params=param_dict)
    Ahat_pshift[ii] = Asqr
    snr_pshift[ii] = Asqr/Sigma

out = [Ahat_pshift.mean(),snr_pshift.mean(),args.process]
np.savetxt(args.outpath, out, fmt='%e, %f, %i')

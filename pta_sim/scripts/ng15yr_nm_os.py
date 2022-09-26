#!/usr/bin/env python
# coding: utf-8


import numpy as np
import la_forge.core as co
import pickle, json, copy, os, sys
import matplotlib.pyplot as plt
import cloudpickle

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

if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta = cloudpickle.load(f)
else:
    raise ValueError(f'{args.pta_pkl} does not exist.')


pta_os = OS(psrs=psrs, pta=pta, orf=args.orf,
               gamma_common=args.gamma_gw)

c0 = co.Core(corepath=args.corepath)
chain = c0.chain[c0.burn:,:-4]
pars = c0.params[:-4]

mlv_idx = np.argmax(c0.chain[c0.burn:,-4])

#If core has "crn" replace...
if 'gw_crn_log10_A' in pars:
    pidx = pars.index('gw_crn_log10_A')
    pars[pidx] = 'gw_log10_A'

if 'gw_crn_gamma' in pars:
    pidx = pars.index('gw_crn_gamma')
    pars[pidx] = 'gw_gamma'

if args.orf is "multiple":
    xi, rho, sig, A, A_err = pta_os.compute_noise_marginalized_multiple_corr_os(chain,param_names=pars)
else:
    xi, rho, sig, A, A_err = pta_os.compute_noise_marginalized_os(chain,param_names=pars)

np.save(args.outdir+f'os_xi_{args.orf}',xi)
np.save(args.outdir+f'os_rho{args.orf}',rho)
np.save(args.outdir+f'os_sig_{args.orf}',sig)
np.save(args.outdir+f'os_A_{args.orf}',A)
np.save(args.outdir+f'os_A_err_{args.orf}',A_err)

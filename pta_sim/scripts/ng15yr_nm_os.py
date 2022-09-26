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


os_pshift = OS(psrs=psrs, pta=pta, orf=args.orf,
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

N = args.niter
# M = args.miter

if os.path.exists(args.outdir+f'os_snr_{args.process}.txt'):
    with open(args.outdir+f'os_snr_{args.process}.txt','r') as file:
        # get the last line and the start
        for line in file:
            pass
        last_entry = line.split('\t')[-1].split('\n')[0]
        if 'Pshift Seed' in last_entry:
            Nstart = 0
        else:
            Nstart = int(float(last_entry)) - args.miter*args.process + 1
else:
    with open(args.outdir+f'os_snr_{args.process}.txt','w') as file:
        file.write('\t'.join(['OS (\hat{A}^2)','SNR','Pshift Seed'])+'\n')
    Nstart = 0


for jj in range(Nstart, N):
    Ahat_pshift = np.zeros(N)
    snr_pshift = np.zeros(N)
    for ii in range(Nstart, N):
        param_dict = {}
        if not args.mlv:
            idx = ii#np.random.randint(0,chain.shape[0])
        else:
            idx = mlv_idx
        param_dict = dict(zip(pars,chain[idx,:]))
        _, _, _, Asqr, Sigma = os_pshift.compute_os(params=param_dict)
        Ahat_pshift[ii] = Asqr
        snr_pshift[ii] = Asqr/Sigma
        # if ii in check:
        #     print(f'{ii/N*100} % complete.')

    out = np.array([np.mean(Ahat_pshift),np.mean(snr_pshift)])
    with open(args.outdir+f'os_snr_{args.process}.txt','a') as file:
        file.write('\t'.join(out.astype(str))+'\n')

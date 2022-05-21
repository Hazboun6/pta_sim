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
        pta_pshift = cloudpickle.load(f)
else:
    tmin = np.amin([p.toas.min() for p in psrs])
    tmax = np.amax([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # Red noise parameter priors
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # GW parameter priors
    if args.gamma_gw is None:
        gw_log10_A = parameter.Uniform(-18, -11)('gw_log10_A')
        gw_gamma = parameter.Uniform(0, 7)('gw_gamma')
    else:
        if np.abs(args.gamma_gw - 4.33) < 0.1:
            gw_log10_A = parameter.Uniform(-18, -14)('gw_log10_A')
        else:
            gw_log10_A = parameter.Uniform(-18, -11)('gw_log10_A')
        gw_gamma = parameter.Constant(args.gamma_gw)('gw_gamma')

    # White noise parameter priors
    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    Nf = args.nfreqs
    freqs = np.linspace(1/Tspan,Nf/Tspan,Nf)


    # # white noise
    selection = selections.Selection(selections.nanograv_backends)

    ef = white_signals.MeasurementNoise(efac=efac,
                                        log10_t2equad=equad,
                                        selection=selection)
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
                                          name='gw_crn', pshift=True,
                                          pseed=parameter.Uniform(0,100000000)('pseed'))#args.process)

    model_pshift = tm + ef + ec + rn + gw_pshift

    pta_pshift = signal_base.PTA([model_pshift(p) for p in psrs])
    with open(args.noisepath,'r') as fin:
        noise = json.load(fin)

    pta_pshift.set_default_params(noise)

    if args.mk_ptapkl:
        with open(args.pta_pkl, "wb") as f:
            cloudpickle.dump(pta_pshift,f)


os_pshift = OS(psrs=psrs, pta=pta_pshift, orf=args.orf,
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
M = args.miter

seed_par = [p for p in pta_pshift.param_names if 'pseed' in p][0]

if os.path.exists(args.outdir+f'os_snr_seed_{args.process}.txt'):
    with open(args.outdir+f'os_snr_seed_{args.process}.txt','r') as file:
        # get the last line and the start
        for line in file:
            pass
        last_entry = line.split('\t')[-1].split('\n')[0]
        if 'Pshift Seed' in last_entry:
            Mstart = 0
        else:
            Mstart = int(float(last_entry)) - args.miter*args.process + 1
else:
    with open(args.outdir+f'os_snr_seed_{args.process}.txt','w') as file:
        file.write('\t'.join(['OS (\hat{A}^2)','SNR','Pshift Seed'])+'\n')
    Mstart = 0

if Mstart == M-1:
    sys.exit()
else:
    for jj in range(Mstart, M):
        Ahat_pshift = np.zeros(N)
        snr_pshift = np.zeros(N)
        for ii in range(N):
            param_dict = {}
            if not args.mlv:
                idx = ii#np.random.randint(0,chain.shape[0])
            else:
                idx = mlv_idx
            param_dict = dict(zip(pars,chain[idx,:]))
            param_dict.update({seed_par:jj+args.miter*args.process})
            _, _, _, Asqr, Sigma = os_pshift.compute_os(params=param_dict)
            Ahat_pshift[ii] = Asqr
            snr_pshift[ii] = Asqr/Sigma
            # if ii in check:
            #     print(f'{ii/N*100} % complete.')

        out = np.array([np.mean(Ahat_pshift),np.mean(snr_pshift),param_dict[seed_par]])
        with open(args.outdir+f'os_snr_seed_{args.process}.txt','a') as file:
            file.write('\t'.join(out.astype(str))+'\n')

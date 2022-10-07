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
from enterprise.signals import gp_signals, gp_priors
from enterprise.signals import deterministic_signals
from enterprise import constants as const

from enterprise_extensions.gp_kernels import periodic_kernel
from enterprise_extensions import sampler

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta = cloudpickle.load(f)
else:
    with open(args.pickle,'rb')as fin:
        psrs = pickle.load(fin)

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
    plaw_rn = gp_signals.FourierBasisGP(spectrum=pl, modes=freqs)

    log10_rho = parameter.Uniform(-10,-4,size=30)
    fs = gp_priors.free_spectrum(log10_rho=log10_rho)
    fs_rn  = gp_signals.FourierBasisGP(spectrum=fs, modes=freqs)

    # timing model
    tm = gp_signals.MarginalizingTimingModel()

    # gw (powerlaw with 5 frequencies)

    gw_pl = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
    gw = gp_signals.FourierBasisGP(spectrum=gw_pl,
                                   modes=freqs[:args.n_gwbfreqs],
                                   name='gw_crn')


    model_plaw = tm + ef + ec + plaw_rn + gw
    model_fs = tm + ef + ec + fs_rn + gw

    with open(args.noisepath,'r') as fin:
        noise = json.load(fin)

    with open(args.model_kwargs_path, 'r') as fin:
        fs_noise = json.load(fin)

    models = []
    for psr in psrs:
        if psr.name in args.free_spec_psrs:
            models.append(model_fs(psr))
            keys = [ky for ky in fs_noise.keys() if (psr.name in ky and 'rho' not in ky)]
            for ky in keys:
                noise.update({ky:fs_noise[ky]})
        else:
            models.append(model_plaw(psr))

    pta = signal_base.PTA(models)
    pta.set_default_params(noise)

    if args.mk_ptapkl:
        with open(args.pta_pkl, "wb") as f:
            cloudpickle.dump(pta,f)

Sampler = sampler.setup_sampler(pta, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr)
def draw_from_gw_gamma_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    # draw parameter from signal model
    signal_name = [par for par in self.pnames
                   if ('gw' in par and 'gamma' in par)][0]
    idx = list(self.pnames).index(signal_name)
    param = self.params[idx]

    q[self.pmap[str(param)]] = np.random.uniform(param.prior._defaults['pmin'], param.prior._defaults['pmax'])

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

if args.gamma_gw is None:
    sampler.JumpProposal.draw_from_gw_gamma_prior = draw_from_gw_gamma_prior
    Sampler.addProposalToCycle(Sampler.jp.draw_from_gw_gamma_prior, 25)

def draw_from_rho_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        parnames = [par.name for par in self.params]
        pname = [pnm for pnm in parnames if rho' in pnm][0]

        idx = parnames.index(pname)
        param = self.params[idx]

        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()


        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

sampler.JumpProposal.draw_from_rho_prior = draw_from_rho_prior
Sampler.addProposalToCycle(Sampler.jp.draw_from_rho_prior, 25)

try:
    achrom_freqs = get_freqs(pta, signal_id='gw_crn')
    np.savetxt(args.outdir + 'achrom_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

x0 = np.hstack([p.sample() for p in pta.params])
Sampler.sample(x0, args.niter, SCAMweight=200, AMweight=100,
               DEweight=200, burn=50000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

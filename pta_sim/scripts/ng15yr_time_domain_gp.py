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

from enterprise_extensions.gp_kernels import periodic_kernel

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
    tm = gp_signals.MarginalizingTimingModel()

    # gw (powerlaw with 5 frequencies)

    gw_pl = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
    gw = gp_signals.FourierBasisGP(spectrum=gw_pl,
                                   modes=freqs[:args.n_gwbfreqs],
                                   name='gw_crn')

    log10_sigma = parameter.Uniform(-10, -3)
    log10_ell = parameter.Uniform(1, 5)
    log10_p = parameter.Uniform(-2, 2)
    log10_gam_p = parameter.Uniform(-2, 2)

    dm_basis = utils.linear_interp_basis(dt=7*const.day)
    qp = periodic_kernel(log10_sigma=log10_sigma,
                         log10_ell=log10_ell,
                         log10_gam_p=log10_gam_p,
                         log10_p=log10_p)

    def by_ao(backend_flags):
    """Selection function to split by backend flags."""
        flagvals = ["ASP", "PUPPI"]
        return {val: backend_flags == val for val in flagvals}

    def by_ao(backend_flags):
    """Selection function to split by backend flags."""
        flagvals = ["GASP", "GUPPI"]
        return {val: backend_flags == val for val in flagvals}

    def by_vla(backend_flags):
    """Selection function to split by backend flags."""
        flagvals = ["YUPPI"]
        return {val: backend_flags == val for val in flagvals}

    selection_ao = selections.Selection(by_ao)
    selection_gbt = selections.Selection(by_gbt)
    selection_vla = selections.Selection(by_vla)

    tdgp_ao = gp_signals.BasisGP(qp, dm_basis, name='ao_periodic',
                                  coefficients=False,
                                  selection=selection_ao)
    tdgp_gbt = gp_signals.BasisGP(qp, dm_basis, name='gbt_periodic',
                                  coefficients=False,
                                  selection=selection_gbt)
    tdgp_vla = gp_signals.BasisGP(qp, dm_basis, name='vla_periodic_',
                                  coefficients=False,
                                  selection=selection_vla)


    model = tm + ef + ec + rn + gw + tdgp_ao + tdgp_gbt + tdgp_vla

    pta = signal_base.PTA([model(p) for p in psrs])

    with open(args.noisepath,'r') as fin:
        noise = json.load(fin)

    pta.set_default_params(noise)

    if args.mk_ptapkl:
        with open(args.pta_pkl, "wb") as f:
            cloudpickle.dump(pta,f)

Sampler = sampler.setup_sampler(pta, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr)

Sampler.addProposalToCycle(Sampler.jp.draw_from_par_prior(['ao_periodic_log10_ell',
                                                           'ao_periodic_log10_sigma',
                                                           'ao_periodic_log10_gam_p',
                                                           'ao_periodic_log10_p',
                                                           'gbt_periodic_log10_ell',
                                                           'gbt_periodic_log10_sigma',
                                                           'gbt_periodic_log10_gam_p',
                                                           'gbt_periodic_log10_p',
                                                           'vla_periodic_log10_ell',
                                                           'vla_periodic_log10_sigma',
                                                           'vla_periodic_log10_gam_p',
                                                           'vla_periodic_log10_p',
                                                           ])
                           , 30)

try:
    achrom_freqs = get_freqs(pta, signal_id='gw_crn')
    np.savetxt(args.outdir + 'achrom_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

x0 = np.hstack(p.sample() for p in pta.params)
Sampler.sample(x0, args.niter, SCAMweight=200, AMweight=100,
               DEweight=200, burn=50000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

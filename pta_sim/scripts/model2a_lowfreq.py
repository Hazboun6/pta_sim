#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
import cloudpickle
import logging

from enterprise_extensions import models, model_utils, hypermodel, sampler
from enterprise.signals.signal_base import PTA
from enterprise.signals import gp_signals, signal_base, deterministic_signals, parameter, selections, white_signals, utils
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise import constants as const

from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions import blocks
from enterprise_extensions import gp_kernels as gpk
from enterprise_extensions import chromatic as chrom
import la_forge.core as co

import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
#Is chain longer than niter?
# longer = chain_length_bool(args.outdir, int(args.niter//10))

# if longer:
#     sys.exit() #Hmmmm what to do here?
# else:
#     pass



if os.path.exists(args.pta_pkl):
    with open(args.pta_pkl, "rb") as f:
        pta_crn = cloudpickle.load(f)

    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)
else:
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)


    # Set Tspan for RN

    Tspan_PTA = model_utils.get_tspan(pkl_psrs)
    # common red noise block
    fmin = 10.0
    modes, wgts = model_utils.linBinning(Tspan_PTA, 1,
                                         1.0 / fmin / Tspan_PTA,
                                         14, 5)
    # wgts = wgts**2.0

    # timing model
    s = gp_signals.MarginalizingTimingModel()

    s += blocks.white_noise_block(vary=False, inc_ecorr=True, select='backend')

    s += blocks.red_noise_block(psd='powerlaw', prior='log-uniform',
                                Tspan=Tspan_PTA, modes=modes, wgts=wgts,)

    log10_rho_gw = parameter.Uniform(-9, -2, size=19)('gw_crn_log10_rho')
    cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)
    s += gp_signals.FourierBasisGP(cpl, modes=modes, name='gw_crn')

    # gw = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan_PTA,
    #                                    components=5, gamma_val=4.33, name='gw', orf='hd')


    crn_models = [s(psr) for psr in pkl_psrs]
    # gw_models = [(m + gw)(psr) for psr,m in  zip(final_psrs,psr_models)]

    pta_crn = signal_base.PTA(crn_models)
    # pta_gw = signal_base.PTA(gw_models)

    # # delta_common=0.,
    # ptas = {0:pta_crn,
    #         1:pta_gw}

    pta_crn.set_default_params(noise)

    with open(args.pta_pkl,'wb') as fout:
        cloudpickle.dump(pta_crn,fout)

groups = sampler.get_parameter_groups(pta_crn)
groups.extend(sampler.get_psr_groups(pta_crn))
Sampler = sampler.setup_sampler(pta_crn, outdir=args.outdir, resume=True,
                                empirical_distr = args.emp_distr, groups=groups)

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

# def draw_from_gw_rho_prior(self, x, iter, beta):
#
#         q = x.copy()
#         lqxy = 0
#
#         signal_name = 'gw_crn'
#
#         # draw parameter from signal model
#         param = np.random.choice(self.snames[signal_name])
#         if param.size:
#             idx2 = np.random.randint(0, param.size)
#             q[self.pmap[str(param)]][idx2] = param.sample()[idx2]
#
#         # scalar parameter
#         else:
#             q[self.pmap[str(param)]] = param.sample()
#
#         # forward-backward jump probability
#         lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
#                 param.get_logpdf(q[self.pmap[str(param)]]))
#
#         return q, float(lqxy)

def draw_from_gw_rho_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        parnames = [par.name for par in self.params]
        pname = [pnm for pnm in parnames
                    if ('gw' in pnm and 'rho' in pnm)][0]

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

sampler.JumpProposal.draw_from_gw_rho_prior = draw_from_gw_rho_prior
Sampler.addProposalToCycle(Sampler.jp.draw_from_gw_rho_prior, 25)


try:
    achrom_freqs = get_freqs(pta_crn, signal_id='gw_crn')
    np.savetxt(args.outdir + 'achrom_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

x0 = np.hstack(p.sample() for p in pta_crn.params)
Sampler.sample(x0, args.niter, SCAMweight=200, AMweight=100,
               DEweight=200, burn=50000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)

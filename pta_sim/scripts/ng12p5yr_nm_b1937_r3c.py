#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy
import logging
import numpy as np
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel
from enterprise.signals import parameter, gp_signals, deterministic_signals
from enterprise.signals import signal_base
from enterprise_extensions import gp_kernels as gpk

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Add to exponential dips for J1713+0747
                #Model, kernel, extra DMGP, Chrom Kernel, Chrom Quad, Index
model_labels = [['A', 'periodic', False, True, 'sq_exp', False, 4],
                ['B', 'periodic', True, True, 'sq_exp', False, 4],
                ['C', 'periodic', False, True, 'sq_exp', True, 4],
                ['D', 'periodic', True, True, 'sq_exp', True, 4],
                ]

ptas = {}
all_kwargs = {}


# Periodic GP kernel for DM
log10_sigma = parameter.Uniform(-4.4, -3)
log10_ell = parameter.Uniform(3, 4)
log10_p = parameter.Uniform(-1, 1)
log10_gam_p = parameter.Uniform(-1.5, 1)

dm_basis = gpk.linear_interp_basis_dm(dt=14*86400)
dm_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                               log10_ell=log10_ell,
                               log10_gam_p=log10_gam_p,
                               log10_p=log10_p)

dmgp2 = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp2')

@signal_base.function
def chromatic_quad(toas, freqs, quad_coeff=np.ones(3)*1e-10, idx=4):
    """
    Basis for chromatic quadratic function.

    :param idx: index of chromatic dependence

    :return ret: normalized quadratic basis matrix [Ntoa, 3]
    """
    t0 = (toas.max() + toas.min()) / 2
    a, b, c = 10**quad_coeff[0], 10**quad_coeff[1], 10**quad_coeff[2]
    quad = (a*(toas-t0)**2 + b*(toas-t0) + c)* (1400/freqs) ** idx

    return quad

quad_coeff = parameter.Uniform(-10, -4, size=3)
deter_chrom = chromatic_quad(quad_coeff=quad_coeff)
chrom_quad = deterministic_signals.Deterministic(deter_chrom,
                                                 name='deter_chrom_quad')

for ii, ent in enumerate(model_labels):
    if ent[2] and ent[5]:
        extra = dmgp2 + chrom_quad
    elif ent[2]:
        extra = dmgp2
    elif ent[5]:
        extra = chrom_quad
    else:
        extra = None

    new_kwargs = {'dm_nondiag_kernel':ent[1],
                  'chrom_gp': ent[3],
                  'chrom_gp_kernel':'nondiag',
                  'chrom_idx':ent[6],
                  'chrom_kernel':ent[4],
                  'chrom_dt':14,
                  'dm_expdip':False,
                  'extra_sigs':extra,
                  }

    kwargs = copy.deepcopy(model_kwargs['5'])
    kwargs.update(new_kwargs)
    ptas[ii] = model_singlepsr_noise(psr, **kwargs)
    all_kwargs[ii] = kwargs

super_model = HyperModel(ptas)

sampler = super_model.setup_sampler(resume=True, outdir=args.outdir,
                                    empirical_distr=args.emp_distr)

model_params = {}
for ky, pta in ptas.items():
    model_params.update({str(ky): pta.param_names})

with open(args.outdir + '/model_params.json', 'w') as fout:
    json.dump(model_params, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

kwargs_out = copy.deepcopy(all_kwargs)
kys = list(kwargs_out.keys())
kwargs_out[kys[0]]['extra_sigs'] = None
kwargs_out[kys[1]]['extra_sigs'] = str('dm_gp2')
kwargs_out[kys[2]]['extra_sigs'] = str('chrom_quad')
kwargs_out[kys[3]]['extra_sigs'] = str('dm_gp2 + chrom_quad')

with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(kwargs_out, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_labels.json', 'w') as fout:
    json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

# sampler for N steps
N = args.niter
x0 = super_model.initial_sample()

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=500000,
               writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

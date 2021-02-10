#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy
import logging
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Add to exponential dips for J1713+0747
                #Model, kernel, dm_dt, chrom_dt, Chrom Kernel, Index
model_labels = [['A', 'sq_exp', 14, 14, True, 'sq_exp', 4],
                ['B', 'sq_exp', 14, 14, True, 'periodic', 4],
                ['A', 'sq_exp', 7, 7, True, 'sq_exp', 4],
                ['B', 'sq_exp', 7, 7, True, 'periodic', 4],
                ['A', 'sq_exp', 3, 3, True, 'sq_exp', 4],
                ['B', 'sq_exp', 3, 3, True, 'periodic', 4],
                ]

ptas = {}
all_kwargs = {}
for ii, ent in enumerate(model_labels):

    new_kwargs = {'dm_nondiag_kernel':ent[1],
                  'chrom_gp': ent[4],
                  'chrom_gp_kernel':'nondiag',
                  'chrom_idx':ent[6],
                  'chrom_kernel':ent[5],
                  'dm_dt':ent[2],
                  'chrom_dt':ent[3],
                  'dm_expdip':False,
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

with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(all_kwargs, fout, sort_keys=True,
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

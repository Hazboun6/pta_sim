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


model_labels = [['A', 'NANOGrav'],
                ['B', '1.13 Dip w/ Scatt GP']
                ['C', 'PPTA']]

ptas = {}
all_kwargs = {}

kwargs = copy.deepcopy(model_kwargs['0'])
ptas[0] = model_singlepsr_noise(psr, **kwargs)
all_kwargs[0] = kwargs

dip_kwargs = {'dm_expdip':True,
              'dmexp_sign': 'negative',
              'num_dmdips':2,
              'dm_expdip_idx':[2,1.15],
              'dm_expdip_tmin':[54700,57450],
              'dm_expdip_tmax':[54850,57560],
              'dmdip_seqname':['dm_1','pr_1']}

# kwargs = copy.deepcopy(model_kwargs['0'])
kwargs.update(dip_kwargs)
ptas[1] = model_singlepsr_noise(psr, **kwargs)
all_kwargs[1] = kwargs

rm_chrom_kwargs = {'chrom_gp':False,}

# kwargs = copy.deepcopy(model_kwargs['0'])
kwargs.update(rm_chrom_kwargs)
ptas[2] = model_singlepsr_noise(psr, **kwargs)
all_kwargs[2] = kwargs


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

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=300000)

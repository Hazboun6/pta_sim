#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle
import logging
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

filepath = '/home/jeffrey.hazboun/nanograv/12p5yr_work/noise_model/'
filepath += 'noise_model_selection/no_dmx_pickles/'
filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(args.psr)
with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Add to exponential dips for J1713+0747
if args.psr == 'J1713+0747':
    dip_kwargs = {'dm_expdip':True,
                  'dmexp_sign': 'negative',
                  'num_dmdips':2,
                  'dm_expdip_tmin':[54700,57450],
                  'dm_expdip_tmax':[54850,57560],
                  'dmdip_seqname':'ism'}
    for dict in model_kwargs.values():
        dict.update(dip_kwargs)

ptas = {}
for ky, kwargs in model_kwargs.items():
    ptas[int(ky)] = model_singlepsr_noise(psr, **kwargs)

model_labels = [['A', 'powerlaw', 'None', True],
                ['B', 'powerlaw', 'None', False],
                ['C', 'powerlaw', 'sq_exp', True],
                ['D', 'powerlaw', 'sq_exp', False],
                ['E', 'powerlaw', 'periodic', True],
                ['F', 'powerlaw', 'periodic', False]]

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
    json.dump(model_kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_labels.json', 'w') as fout:
    json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

# sampler for N steps
N = args.niter
x0 = super_model.initial_sample()

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=300000)

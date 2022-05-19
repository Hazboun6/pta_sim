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
                #Model, kernel, DM1, DM2, Chrom Kernel, Index
model_labels = [['A', 'periodic', True, True, True, 'periodic', 4],
                ['B', 'sq_exp', True, True, True, 'periodic', 4],
                ['C', 'periodic_rfband', True, True, True, 'periodic', 4],
                ['D', 'sq_exp_rfband', True, True, True, 'periodic', 4],
                ]



ptas = {}
all_kwargs = {}
for ii, ent in enumerate(model_labels):
    num_dips = 0
    dm_expdip_tmin = []
    dm_expdip_tmax = []
    dm_expdip_idx = []
    dmdip_seqname = []

    if ent[2]: # Add 1st ISM event, DM
        num_dips +=1
        dm_expdip_tmin.append(54700)
        dm_expdip_tmax.append(54850)
        dm_expdip_idx.append(2)
        dmdip_seqname.append('dm_1')
    if ent[3]: # Add 2nd ISM event, DM
        num_dips +=1
        dm_expdip_tmin.append(57450)
        dm_expdip_tmax.append(57560)
        dm_expdip_idx.append(2)
        dmdip_seqname.append('dm_2')
    # if ent[4]: # Add 1st ISM event, Scattering
    #     num_dips +=1
    #     dm_expdip_tmin.append(54700)
    #     dm_expdip_tmax.append(54850)
    #     dm_expdip_idx.append(4)
    #     dmdip_seqname.append('chrom_1')
    # if ent[5]: # Add 2nd ISM event, Scattering
    #     num_dips +=1
    #     dm_expdip_tmin.append(57450)
    #     dm_expdip_tmax.append(57560)
    #     dm_expdip_idx.append(4)
    #     dmdip_seqname.append('chrom_2')

    dip_kwargs = {'dm_expdip':True,
                  'dmexp_sign': 'negative',
                  'dm_nondiag_kernel':ent[1],
                  'chrom_gp': ent[4],
                  'chrom_gp_kernel':'nondiag',
                  'chrom_idx':ent[6],
                  'chrom_kernel':ent[5],
                  'chrom_dt':7,
                  'num_dmdips':num_dips,
                  'dm_expdip_idx':dm_expdip_idx,
                  'dm_expdip_tmin':dm_expdip_tmin,
                  'dm_expdip_tmax':dm_expdip_tmax,
                  'dmdip_seqname':dmdip_seqname}

    kwargs = copy.deepcopy(model_kwargs['5'])
    kwargs.update(dip_kwargs)
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

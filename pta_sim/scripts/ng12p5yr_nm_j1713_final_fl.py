#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy
import logging
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel
from enterprise_extensions.chromatic import chrom_exp_decay
from enterprise_extensions.blocks import common_red_noise_block, red_noise_block
from enterprise.signals import parameter
from enterprise.signals import deterministic_signals

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)


model_labels = [['A', 'Vary Dip'],]

ptas = {}
all_kwargs = {}

def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp'):
    """
    Returns chromatic exponential dip (i.e. TOA advance):

    :param tmin, tmax:
        search window for exponential dip time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sign:
        set sign of dip: 'positive', 'negative', or 'vary'
    :param name: Name of signal

    :return dmexp:
        chromatic exponential dip waveform.
    """
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(-10, -2)
    log10_tau_dmexp = parameter.Uniform(0, 2.5)
    if sign == 'vary':
        sign_param = parameter.Uniform(-1.0, 1.0)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0
    wf = chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                         t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                         sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp

Tspan = 407576851.48121357
rn = red_noise_block(psd='powerlaw', prior='log-uniform',
                     Tspan=Tspan, components=30, gamma_val=None)

gw = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                            Tspan=Tspan, components=5, gamma_val=4.3333)
sig = rn + gw

index = parameter.Uniform(0, 2)

ppta_dip = dm_exponential_dip(57450, 57560, idx=index, sign='negative', name='exp2')

kwargs = copy.deepcopy(model_kwargs['0'])
kwargs.update({'red_var':False,
               'dm_dt':3,
               'chrom_dt':3,})

dip_kwargs = {'dm_expdip':True,
              'dmexp_sign': 'negative',
              'num_dmdips':1,
              'dm_expdip_idx':[2],
              'dm_expdip_tmin':[54700],
              'dm_expdip_tmax':[54850],
              'dmdip_seqname':['dm_1'],
              'extra_sigs':ppta_dip+rn+gw}

kwargs.update(dip_kwargs)
ptas[0] = model_singlepsr_noise(psr, **kwargs)
all_kwargs[0] = kwargs

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
kwargs_out[kys[1]]['extra_sigs'] = str('vary_idx_dip + fact like')

with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(kwargs_out, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_labels.json', 'w') as fout:
    json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

# sampler for N steps
N = args.niter
x0 = super_model.initial_sample()

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, writeHotChains=args.writeHotChains, burn=300000)

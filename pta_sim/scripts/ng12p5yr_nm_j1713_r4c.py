#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy, os
import logging
import scipy.stats as sps
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel
from enterprise_extensions.blocks import common_red_noise_block, red_noise_block
from enterprise_extensions.chromatic import solar_wind
import ultranest
import ultranest.stepsampler
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Add to exponential dips for J1713+0747
                #Model, kernel, DM1, DM2, Chrom Kernel, Index, GW
model_labels = [['A', 'periodic_rfband', True, True, True, 'periodic', 4, False],
                ['B', 'periodic_rfband', True, True, True, 'periodic', 4, True],
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
    Tspan = 407576851.48121357
    rn = red_noise_block(psd='powerlaw', prior='log-uniform',
                         Tspan=Tspan, components=30, gamma_val=None)
    if ent[7]:
        gw = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                    Tspan=Tspan, components=5, gamma_val=4.3333)
        sig = rn + gw
    else:
        sig = rn

    # Note that I have turned off the RN model in model_single_psr
    # And added a RN model with the Tspan of the PTA
    dip_kwargs = {'dm_expdip':True,
                  'dmexp_sign': 'negative',
                  'dm_nondiag_kernel':ent[1],
                  'chrom_gp': ent[4],
                  'chrom_gp_kernel':'nondiag',
                  'chrom_idx':ent[6],
                  'chrom_kernel':ent[5],
                  'dm_dt':3,
                  'chrom_dt':3,
                  'num_dmdips':num_dips,
                  'dm_expdip_idx':dm_expdip_idx,
                  'dm_expdip_tmin':dm_expdip_tmin,
                  'dm_expdip_tmax':dm_expdip_tmax,
                  'dmdip_seqname':dmdip_seqname,
                  'red_var':False,
                  'extra_sigs':sig}

    kwargs = copy.deepcopy(model_kwargs['5'])
    kwargs.update(dip_kwargs)
    ptas[ii] = model_singlepsr_noise(psr, **kwargs)
    all_kwargs[ii] = kwargs

pta = ptas[1]

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

model_params = {}
for ky, pta in ptas.items():
    model_params.update({str(ky): pta.param_names})

with open(args.outdir + '/model_params.json', 'w') as fout:
    json.dump(model_params, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

kwargs_out = copy.deepcopy(all_kwargs)
kys = list(kwargs_out.keys())
kwargs_out[kys[0]]['extra_sigs'] = str('rn_only')
kwargs_out[kys[1]]['extra_sigs'] = str('gw_rn')

with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(kwargs_out, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_labels.json', 'w') as fout:
    json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

np.savetxt(args.outdir + '/pars.txt', pta.param_names, fmt='%s')
np.savetxt(args.outdir + '/priors.txt',
           list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

class sw_trans():
    def __init__(self):
        self.ppf = solar_wind.ACE_RV.ppf
    def __call__(self, quantile):
        return self.ppf(quantile)

class uniform_trans():
    def __init__(self, pmin, pmax):
        self.width = pmax - pmin
        self.pmin = pmin
    def __call__(self, quantile):
        return quantile * self.width + self.pmin

class normal_trans():
    def __init__(self, mean, std):
        self.rvs = sps.norm(loc=mean,scale=std)
    def __call__(self, quantile):
        return self.rvs.ppf(quantile)

transforms = []
for nm, param in zip(pta.param_names,pta.params):
    if param.type.lowercase()=='uniform':
        pmin = param.prior._defaults['pmin']
        pmax = param.prior._defaults['pmax']
        transforms.append(uniform_trans(pmin,pmax))
    elif param.type.lowercase()=='normal':
        mu = param.prior._defaults['mu']
        sigma = param.prior._defaults['sigma']
        transforms.append(normal_trans(mu,sigma))
    elif param.type.lowercase()=='ace_swepam':
        transforms.append(sw_trans())

def transform(quantile):
    return np.array([t(q) for q,t in zip(quantile,transforms)])

sampler1 = ultranest.ReactiveNestedSampler(
    pta.param_names,
    pta.get_lnlikelihood,
    transform,
    log_dir=args.outdir
)
sampler1.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=8)
ndim = len(pta.params)
sampler1.run(dlogz=0.5 + 0.1 * ndim,
     # update_interval_iter_fraction=0.4 if ndim > 20 else 0.2,
     # max_num_improvement_loops=3,
     min_num_live_points=400)

#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy, sys
import logging
import numpy as np
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel
from enterprise.signals import parameter, gp_signals, deterministic_signals
from enterprise.signals import signal_base
from enterprise_extensions import gp_kernels as gpk
from enterprise_extensions import chromatic as chrom
from enterprise_extensions.blocks import chromatic_noise_block
from enterprise_extensions.blocks import common_red_noise_block, red_noise_block
from enterprise_extensions import sampler as samp

from la_forge.core import Core

import pta_sim.bayes as bys
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

longer = bys.chain_length_bool(args.outdir, int(args.niter/10)-100)

if longer and os.path.exists(args.corepath+f'{psr.name}.core'):
    sys.exit()
elif longer:
    c0 = Core(chaindir=args.outdir)
    c0.set_rn_freqs(freq_path=args.outdir+'/achrom_freqs.txt')
    c0.save(args.corepath+f'{psr.name}.core')
    sys.exit() #Hmmmm what to do here?
else:
    pass

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Binned Solar Wind Model
bins = np.linspace(53215, 57934, 26)
bins *= 24*3600 #Convert to secs
n_earth = parameter.Constant(size=size=bins.size-1)('n_earth')
# n_earth = chrom.solar_wind.ACE_SWEPAM_Parameter(size=bins.size-1)('n_earth')
deter_sw = chrom.solar_wind.solar_wind(n_earth=n_earth, n_earth_bins=bins)
mean_sw = deterministic_signals.Deterministic(deter_sw, name='sw_r2')

np_earth = parameter.Uniform(-4, -2)('np_4p39')
sw_power = parameter.Constant(4.39)('sw_power_4p39')
deter_sw_p = chrom.solar_wind.solar_wind_r_to_p(n_earth=np_earth,
                                                power=sw_power,
                                                log10_ne=True)
mean_sw += deterministic_signals.Deterministic(deter_sw_p,
                                               name='sw_4p39')


Tspan = 407576851.48121357

new_kwargs = {'dm_sw_deter':False,
              'red_var':False,
              'factorized_like':True,
              'psd':'spectrum',
              'Tspan':Tspan,
              'gw_components':30,
              'dm_df':None,
              'chrom_df':None,
              'fact_like_logmin':-14.2,
              'fact_like_logmax':-1.2,
              'extra_sigs':mean_sw,
              }

kwargs = copy.deepcopy(model_kwargs)
kwargs.update(new_kwargs)

# Special pulsars
if psr.name == 'B1937+21':
    # Periodic GP kernel for DM
    log10_sigma = parameter.Uniform(-10, -4.8)
    log10_ell = parameter.Uniform(1, 2.4)
    log10_p = parameter.Uniform(-2, -1)
    log10_gam_p = parameter.Uniform(-2, 2)
    dm_basis = gpk.linear_interp_basis_dm(dt=3*86400)
    dm_prior = gpk.periodic_kernel(log10_sigma=log10_sigma,
                                   log10_ell=log10_ell,
                                   log10_gam_p=log10_gam_p,
                                   log10_p=log10_p)
    dmgp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp1')
    # Periodic GP kernel for DM
    log10_sigma2 = parameter.Uniform(-4.8, -3)
    log10_ell2 = parameter.Uniform(2.4, 5)
    log10_p2 = parameter.Uniform(-2, 2)
    log10_gam_p2 = parameter.Uniform(-2, 2)
    dm_basis2 = gpk.linear_interp_basis_dm(dt=3*86400)
    dm_prior2 = gpk.periodic_kernel(log10_sigma=log10_sigma2,
                                   log10_ell=log10_ell2,
                                   log10_gam_p=log10_gam_p2,
                                   log10_p=log10_p2)
    dmgp2 = gp_signals.BasisGP(dm_prior2, dm_basis2, name='dm_gp2')
    ch_log10_sigma = parameter.Uniform(-10, -3.5)
    ch_log10_ell = parameter.Uniform(1, 6)
    chm_basis = gpk.linear_interp_basis_chromatic(dt=3*86400, idx=4)
    chm_prior = gpk.se_dm_kernel(log10_sigma=ch_log10_sigma, log10_ell=ch_log10_ell)
    chromgp = gp_signals.BasisGP(chm_prior, chm_basis, name='chrom_gp')
    b1937_chrom_model = dmgp + dmgp2 + chromgp + mean_sw
    kwargs.update({extra_sigs:b1937_chrom_model})

# Setup PTA
pta = model_singlepsr_noise(psr, **kwargs)
sampler = samp.setup_sampler(pta=pta,
                             resume=True,
                             outdir=args.outdir,
                             empirical_distr=args.emp_distr)


with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

freqs = bys.get_freqs(pta, signal_id='gw')
np.savetxt(args.outdir+'achrom_freqs.txt', freqs)

# sampler for N steps
N = args.niter
x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=500000,
               writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

c0 = Core(chaindir=Outdir)
c0.set_rn_freqs(freq_path=Outdir+'/achrom_freqs.txt')
c0.save(args.corepath+f'{psr.name}.core')

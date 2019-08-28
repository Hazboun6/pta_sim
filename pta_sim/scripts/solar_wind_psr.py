#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pint.toa as toa
import pint.models as models
import pint.fitter as fit
import pint.residuals as r
import astropy.units as u

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

import corner, pickle, sys, json
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import models, model_utils
from enterprise_extensions.electromagnetic import solar_wind as SW
from astropy import log
import glob
log.setLevel('CRITICAL')

import pta_sim
import pta_sim.bayes
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

if args.pickle=='no_pickle':
    if args.use_pint:
        parfile = args.pardir + args.psr + '*.gls.par'
        timfile = args.timdir + args.psr + '*.tim'

        t=toa.get_TOAs(timfile)
        m=models.get_model(parfile)
        f=fit.GLSFitter(t,m)
        f.fit_toas(maxiter=2)

        psr = Pulsar(t, f.model, ephem='DE436')
    else:
        if args.psr == 'J1713+0747':
            parfile = glob.glob(args.pardir + args.psr + '*.gls.t2.par')[0]
        else:
            parfile = glob.glob(args.pardir + args.psr + '*.gls.par')[0]

        timfile = glob.glob(args.timdir + args.psr + '*.tim')[0]
        psr = Pulsar(parfile, timfile, ephem='DE436')
else:
    with open(args.pickle, 'rb') as fin:
        psrs = pickle.load(fin)
    psr = [p for p in psrs if p.name==args.psr][0]

if args.end_time is None:
    Outdir = args.outdir+'all/{0}/'.format(psr.name)
else:
    start_time = psr.toas.min()/(24*3600)
    if (args.end_time-start_time)/365.25 <= 3.0:
        print('PSR {0} baseline too short for this slice.'.format(p.name))
        sys.end()
    psr.filter_data(start_time=start_time, end_time=args.end_time)
    Outdir = args.outdir+'{0}/{1}/'.format(args.nyears,psr.name)

m = models.white_noise_block(vary=True, inc_ecorr=True)
m += gp_signals.TimingModel(use_svd=False)
m += models.red_noise_block(psd=args.psd, prior='log-uniform',
                            components=args.nfreqs, gamma_val=None)

dm_gp1 = models.dm_noise_block(gp_kernel='diag', psd='powerlaw',
                               prior='log-uniform', Tspan=None,
                               components=30, gamma_val=None,
                               coefficients=False)

dm_gp2 = models.dm_noise_block(gp_kernel='diag', psd='powerlaw',
                               prior='log-uniform', Tspan=None,
                               components=10, gamma_val=None,
                               coefficients=False)

n_earth = SW.ACE_SWEPAM_Parameter()('n_earth')
sw = SW.solar_wind(n_earth=n_earth)
mean_sw = deterministic_signals.Deterministic(sw, name='mean_sw')

log10_A_sw = parameter.Uniform(-10,1)('log10_A_sw')
gamma_sw = parameter.Uniform(-2,1)('gamma_sw')
dm_sw_basis = SW.createfourierdesignmatrix_solar_dm(nmodes=15,Tspan=None)
dm_sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
gp_sw = gp_signals.BasisGP(priorFunction=dm_sw_prior,
                           basisFunction=dm_sw_basis,
                           name='gp_sw')
sw_models = []
sw_models.append(m + dm_gp1) #Model 0, Just DMGP
sw_models.append(m + mean_sw) #Model 1, Just Deterministic SW
sw_models.append(m + dm_gp2 + mean_sw) #Model 2, DMGP + Deter SW
sw_models.append(m + mean_sw + gp_sw) #Model 3, Deter SW + SW GP
sw_models.append(m + SW.solar_wind_block(ACE_prior=True, include_dmgp=True))
#Model 4, All the things

ptas = []
model_params = {}
for ii,mod in enumerate(sw_models):
    ptas.append(signal_base.PTA(mod(psr)))
    model_params.update({ii : ptas[ii].param_names})

super_model = model_utils.HyperModel(ptas)
super_model.setup_sampler(resume=True, outdir=args.outdir)
with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params,fout,sort_keys=True,indent=4,separators=(',', ': '))

x0 = super_model.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15, DEweight=50, )

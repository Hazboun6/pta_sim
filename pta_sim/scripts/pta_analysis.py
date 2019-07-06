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

import corner, pickle, sys
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import models, model_utils

from astropy import log
import glob
log.setLevel('CRITICAL')

import pta_sim
import pta_sim.bayes
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

if args.pickle=='no_pickle':
    if args.use_pint:
        parfiles = glob.glob(args.pardir + '*.gls.par')
        timfiles = glob.glob(args.timdir + '*.tim')
        if len(timfiles)!=len(parfiles):
            raise ValueError('List of parfiles and timfiles not equal!!!')

        psrs = []
        for par,tim in zip(parfiles, timfiles):
            t = toa.get_TOAs(tim)
            m = models.get_model(par)
            f = fit.GLSFitter(t,m)
            f.fit_toas(maxiter=2)
            psr = Pulsar(t, f.model, ephem=args.ephem)
            psrs.append(psr)
    else:
        if args.psr == 'J1713+0747':
            parfile = glob.glob(args.pardir + args.psr + '*.gls.t2.par')[0]
        else:
            parfile = glob.glob(args.pardir + args.psr + '*.gls.par')[0]
        parfiles = glob.glob(args.pardir + '*.gls.par')
        timfiles = glob.glob(args.timdir + '*.tim')
        j1713_tempo_par = [p for p in parfiles
                           if ('J1713+0747' in p) and ('.gls.t2.par' not in p)]
        parfiles.remove(j1713_tempo_par)
        if len(timfiles)!=len(parfiles):
            raise ValueError('List of parfiles and timfiles not equal!!!')
        for par,tim in zip(parfiles, timfiles):
            psr = Pulsar(par, tim, ephem=args.ephem)
            psrs.append(psr)

else:
    with open(args.pickle, 'rb') as fin:
        psrs = pickle.load(fin)


if args.end_time is None:
    Outdir = args.outdir+'all/'
else:
    for psr in psrs:
        start_time = psr.toas.min()/(24*3600)
        if (args.end_time-start_time)/365.25 <= 3.0:
            print('PSR {0} baseline too short for this slice.'.format(psr.name))
            pass
        else:
            psr.filter_data(start_time=start_time, end_time=args.end_time)
    Outdir = args.outdir+'{0}/'.format(args.nyears)

with open(args.noisepath, 'r') as fin:
    noise_dict = json.load(fin)

if args.vary_gamma:
    gamma_gw = None
else:
    gamma_gw = args.gamma_gw
if args.model=='model_2a':
    pta = model_2a(psrs, psd=args.psd,
                   noisedict=noise_dict,
                   components=args.nfreqs,
                   gamma_common=gamma_gw,
                   upper_limit=args.gwb_ul,
                   bayesephem=args.bayes_ephem,
                   wideband=args.wideband,
                   select='backend')
else:
    raise NotImplementedError('Please add this model to the script.')

sampler = model_utils.setup_sampler(pta=pta,
                                    outdir=Outdir,
                                    resume=True)

freqs = get_freqs(pta, signal_id='gw')
np.savetxt(Outdir+'achrom_freqs.txt', freqs)

x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0, Niter=args.niter)

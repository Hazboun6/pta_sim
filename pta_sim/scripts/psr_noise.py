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
            parfile = args.pardir + args.psr + '*.gls.t2.par'
        else:
            parfile = args.pardir + args.psr + '*.gls.par'

        timfile = args.timdir + args.psr + '*.tim'
        psr = Pulsar(parfile, timfile, ephem='DE436')
else:
    with open(args.pickle, 'rb') as fin:
        psrs = pickle.load(fin)
    pnames = [p.name for p in psrs]
    pidx = pnames.index(args.psr)
    psr = psrs[pidx]

if args.end_time is None:
    Outdir = args.outdir+'all/'
else:
    start_time = psr.toas.min()/(24*3600)
    if (args.end_time-start_time)/365.25 <= 3.0:
        print('PSR {0} baseline to short for this slice.'.format(p.name))
        sys.end()
    psr.filter_data(start_time=start_time, end_time=args.end_time)
    Outdir = args.outdir+'{0}/'.format(args.nyears)

pta = models.model_singlepsr_noise(psr, red_var=True,
                                   psd=args.psd,
                                   components=args.nfreqs,
                                   wideband=args.wideband)

sampler = model_utils.setup_sampler(pta=pta,
                                    outdir=Outdir,
                                    resume=True)

x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0, Niter=args.niter)

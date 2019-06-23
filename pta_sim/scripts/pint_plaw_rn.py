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

import corner, pickle
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions import models, model_utils

from astropy import log
import glob
log.setLevel('CRITICAL')

import pta_sim
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

parfile = sorted(glob.glob(args.pardir + args.psr + '*.gls.par'))
timfile = sorted(glob.glob(args.timdir + args.psr + '*.tim'))

t=toa.get_TOAs(timfile)
m=models.get_model(parfile)
f=fit.GLSFitter(t,m)
f.fit_toas(maxiter=3)

psr=Pulsar(t, f.model, ephem='DE436')

pta = models.model_singlepsr_noise(psr,red_var=True)
sampler=model_utils.setup_sampler(pta=pta,
                                  outdir=args.outdir,
                                  resume=True)

x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0,Niter=args.niter)

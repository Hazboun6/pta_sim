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
        parfiles = glob.glob(args.pardir + '*.par')
        timfiles = glob.glob(args.timdir + '*.tim')
        j1713_tempo_par = [p for p in parfiles
                           if ('J1713+0747' in p)
                           and ('.gls.t2.par' not in p)][0]
        parfiles.remove(j1713_tempo_par)
        if len(timfiles)!=len(parfiles):
            raise ValueError('List of parfiles and timfiles not equal!!!')
        for par,tim in zip(parfiles, timfiles):
            psr = Pulsar(par, tim, ephem=args.ephem)
            psrs.append(psr)

else:
    with open(args.pickle, 'rb') as fin:
        psrs = pickle.load(fin)

psr_names = [p.name for p in psrs]

if args.rm_psrs is not None:
    rm_idxs = [psr_names.index(p) for p in args.rm_psrs]
    for idx in reversed(rm_idxs):
        del psrs[idx]

if args.truncate_psr is not None:
    if len(args.truncate_psr)!=len(args.truncate_mjd):
        err_msg = 'List of psrs to truncate and truncation MJDs must be equal!!'
        raise ValueError(err_msg)
    for pname, mjd in zip(args.truncate_psr,args.truncate_mjd):
        pidx = psr_names.index(pname)
        start_time = psrs[pidx].toas.min()/(24*3600)
        psrs[pidx].filter_data(start_time=start_time, end_time=mjd)

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
elif args.model=='model_general':
    pta = model_general(psrs, common_psd='powerlaw',
                        red_psd=args.psd, orf=None,
                        common_components=args.nfreqs,
                        red_components=30,
                        dm_components=30,
                        modes=None, wgts=None,
                        noisedict=noise_dict,
                        tm_svd=False, tm_norm=True,
                        gamma_common=gamma_gw,
                        upper_limit=args.gwb_ul,
                        bayesephem=args.bayes_ephem,
                        wideband=args.wideband,
                        dm_var=True, dm_type='gp',
                        dm_psd='powerlaw', dm_annual=False,
                        white_vary=False, gequad=False, dm_chrom=False,
                        dmchrom_psd='powerlaw', dmchrom_idx=4,
                        red_select=None,
                        red_breakflat=False,
                        red_breakflat_fq=None,
                        coefficients=False,)
else:
    raise NotImplementedError('Please add this model to the script.')

sampler = model_utils.setup_sampler(pta=pta,
                                    outdir=Outdir,
                                    resume=True)

freqs = get_freqs(pta, signal_id='gw')
np.savetxt(Outdir+'achrom_freqs.txt', freqs)

x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0, Niter=args.niter)

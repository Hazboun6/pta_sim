#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
# import pint.toa as toa
# import pint.models as models
# import pint.fitter as fit
# import pint.residuals as r
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

from enterprise_extensions import models, model_utils, sampler

from astropy import log
import glob
log.setLevel('CRITICAL')

from la_forge.core import Core

import pta_sim
import pta_sim.bayes as bys
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

#Is chain longer than niter?


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
            parfile = glob.glob(args.pardir + args.psr + '*.t2.*par')[0]
        else:
            parfile = glob.glob(args.pardir + args.psr + '*.gls.*par')[0]

        timfile = glob.glob(args.timdir + args.psr + '*.tim')[0]
        psr = Pulsar(parfile, timfile, ephem='DE436')
else:
    with open(args.pickle, 'rb') as fin:
        psrs = pickle.load(fin,encoding='latin1')
    if isinstance(psrs,list):
        pnames = [p.name for p in psrs]
        if args.psr is not None:
            pidx = pnames.index(args.psr)
        elif args.process is not None:
            pidx = args.process
        psr = psrs[pidx]
    else:
        psr = psrs

if args.end_time is None:
    Outdir = args.outdir+'/{0}/'.format(psr.name)
else:
    start_time = psr.toas.min()/(24*3600)
    if (args.end_time-start_time)/365.25 <= 3.0:
        print('PSR {0} baseline too short for this slice.'.format(p.name))
        sys.exit()
    psr.filter_data(start_time=start_time, end_time=args.end_time)
    Outdir = args.outdir+'{0}/{1}/'.format(args.nyears,psr.name)

longer = bys.chain_length_bool(Outdir, int(args.niter/10))

if longer and os.path.exists(args.corepath+f'{psr.name}.core'):
    sys.exit()
elif longer:
    c0 = Core(chaindir=Outdir)
    c0.set_rn_freqs(freq_path=Outdir+'/achrom_freqs.txt')
    c0.save(args.corepath+f'{psr.name}.core')
    sys.exit() #Hmmmm what to do here?
else:
    pass

emp_dist_path = args.emp_distr

if args.gwb_bf or args.gwb_ul:
    if args.gwb_bf:
        prior = 'log-uniform'
    elif args.gwb_ul:
        prior = 'uniform'
    m = models.white_noise_block(vary=True, inc_ecorr=True)
    m += gp_signals.TimingModel(use_svd=False)
    m += models.red_noise_block(psd=args.psd, prior=prior,
                                components=args.nfreqs, gamma_val=None)
    m += models.common_red_noise_block(gamma_val=13/3., prior=prior,
                                       psd=args.psd, components=args.nfreqs)
    pta = signal_base.PTA(m(psr))

else:
    if args.gfl:
        vary_rn = False
        extra_sigs = None
    elif args.gfl_lite:
        args.gfl = True
        vary_rn = False
        rn = models.red_noise_block(components=args.nfreqs,
                                    Tspan=args.tspan,
                                    gamma_val=None)
        extra_sigs = rn
    else:
        vary_rn = True
        extra_sigs = None

    pta = models.model_singlepsr_noise(psr, red_var=vary_rn,
                                       psd=args.psd, Tspan=args.tspan,
                                       components=args.nfreqs,
                                       factorized_like=args.gfl,
                                       gw_components=args.n_gwbfreqs,
                                       fact_like_logmin=-14.2,
                                       fact_like_logmax=-1.2,
                                       is_wideband=args.wideband,
                                       extra_sigs=extra_sigs)

    emp_dist_path = args.emp_distr.replace('PSR_NAME',psr.name)

Sampler = sampler.setup_sampler(pta=pta,
                                outdir=Outdir,
                                empirical_distr = emp_dist_path,
                                resume=True)
if args.gfl:
    freqs = bys.get_freqs(pta, signal_id='gw')
else:
    freqs = bys.get_freqs(pta, signal_id='red_noise')

np.savetxt(Outdir+'achrom_freqs.txt', freqs)

x0 = np.hstack(p.sample() for p in pta.params)

Sampler.sample(x0, Niter=args.niter, burn=100000,
               writeHotChains=args.writeHotChains)

c0 = Core(chaindir=Outdir)
c0.set_rn_freqs(freq_path=Outdir+'/achrom_freqs.txt')
c0.save(args.corepath+f'{psr.name}.core')

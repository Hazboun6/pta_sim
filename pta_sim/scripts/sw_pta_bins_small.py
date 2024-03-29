#!/usr/bin/env python
# coding: utf-8
import numpy as np
# import pint.toa as toa
# import pint.models as models
# import pint.fitter as fit
# import pint.residuals as r
import astropy.units as u
import scipy.integrate as spi
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

from enterprise_extensions import models, model_utils, sampler
from enterprise_extensions.chromatic import solar_wind as SW
from enterprise_extensions.chromatic import chromatic as chr
from enterprise_extensions.gp_kernels import linear_interp_basis_dm, se_dm_kernel
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
                           and ('.t2.' not in p)][0]
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

tmin = np.amin([p.toas.min() for p in psrs])
tmax = np.amax([p.toas.max() for p in psrs])
Tspan = tmax-tmin

noise_json = args.noisepath
with open(noise_json, "r") as f:
    noise_dict = json.load(f)

if args.gwb_ul:
    prior = 'uniform'
else:
    prior = 'log-uniform'

model = models.white_noise_block(vary=False, inc_ecorr=True)
model += gp_signals.TimingModel(use_svd=False)
model += models.red_noise_block(psd=args.psd, prior=prior,
                                components=args.nfreqs, gamma_val=None)

if args.gwb_off:
    pass
else:
    if args.hd:
        orf='hd'
    else:
        orf=None
    gw = models.common_red_noise_block(psd=args.psd, prior=prior,
                                       Tspan=Tspan, orf=orf,
                                       gamma_val=args.gamma_gw,
                                       name='gw')
    model += gw


log10_sigma = parameter.Uniform(-10, -4)
log10_ell = parameter.Uniform(1, 4)
dm_basis = linear_interp_basis_dm(dt=15*86400)
dm_prior = se_dm_kernel(log10_sigma=log10_sigma,log10_ell=log10_ell)
dm_gp = gp_signals.BasisGP(dm_prior, dm_basis, name='dm_gp')
dm_block = dm_gp

# Make solar wind signals
print('sw_r2p ',args.sw_r2p)
# if isinstance(args.sw_r2p,(float,int)):
#     args.sw_r2p = [args.sw_r2p]

if args.sw_r2p_ranges is None:
    sw_r2p_ranges = args.sw_r2p
elif len(args.sw_r2p)!=len(args.sw_r2p_ranges):
    raise ValueError('Number of SW powers must match number of prior ranges!! '
                     'Set # nonvarying ')
else:
    sw_r2p_ranges = args.sw_r2p_ranges

for ii, (power, pr_range) in enumerate(zip(args.sw_r2p, sw_r2p_ranges)):
    print(type(power),type(pr_range))
    if len(power)==1 and power[0] == 2.0:
        print('1 ',power,pr_range)
        bins = np.linspace(53215,57388,46)
        bins *= 24*3600 #Convert to secs
        n_earth = SW.ACE_SWEPAM_Parameter(size=bins.size-1)('nE')
        deter_sw = SW.solar_wind(n_earth=n_earth, n_earth_bins=bins)
        dm_block += deterministic_signals.Deterministic(deter_sw,
                                                        name='sw_r2')
    elif len(power)==1:
        print('2 ',power,pr_range)
        n_earth = parameter.Uniform(pr_range[0],
                                    pr_range[1])('nE_{0}'.format(power))
        sw_power = parameter.Constant(power[0])('sw_power_{0}'.format(ii+1))
        log10_ne = True if pr_range[0] < 0 else False
        deter_sw = SW.solar_wind_r_to_p(n_earth=n_earth,
                                                power=sw_power,
                                                log10_ne=log10_ne)
        dm_block += deterministic_signals.Deterministic(deter_sw,
                                                     name='sw_{0}'.format(ii+1))
    elif len(power)>1:
        print('3 ',power,pr_range)
        n_earth = parameter.Uniform(pr_range[0], pr_range[1])('nE_p{0}'.format(ii+1))
        sw_power = parameter.Uniform(power[0], power[1])('sw_power_{0}'.format(ii+1))
        log10_ne = True if pr_range[0] < 0 else False
        deter_sw = SW.solar_wind_r_to_p(n_earth=n_earth,
                                                power=sw_power,
                                                log10_ne=log10_ne)
        dm_block += deterministic_signals.Deterministic(deter_sw,
                                                     name='sw_{0}'.format(ii+1))

if args.bayes_ephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(model='setIII',
                                                        use_epoch_toas=True)
    model += eph

norm_model = model + dm_block
if args.dm_dip:
    psr_models = []
    for p in psrs:
        if p.name == 'J1713+0747':
            dmdip = chr.dm_exponential_dip(tmin=54700,tmax=54900)
            model_j1713 = norm_model + dmdip
            psr_models.append(model_j1713(p))
        else:
            psr_models.append(norm_model(p))
else:
    psr_models = [norm_model(p) for p in psrs]

pta = signal_base.PTA(psr_models)

pta.set_default_params(noise_dict)

x0 = np.hstack(p.sample() for p in pta.params)

ndim = x0.size

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups

groups = sampler.get_parameter_groups(pta)

Sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                 cov, groups=groups,
                 outDir=args.outdir, resume=True)

np.savetxt(args.outdir + '/pars.txt', pta.param_names, fmt='%s')
np.savetxt(args.outdir + '/priors.txt',
           list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

class my_JP(sampler.JumpProposal):
    def __init__(self, pta, snames=None, empirical_distr=None):
        super().__init__(pta, snames=None, empirical_distr=None)

    def draw_from_sw1_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'sw_1'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_sw2_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'sw_2'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_sw3_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'sw_3'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)

    def draw_from_sw4_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'sw_4'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()

        # forward-backward jump probability
        lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                param.get_logpdf(q[self.pmap[str(param)]]))

        return q, float(lqxy)


emp_dist_pkl= args.emp_distr
jp = my_JP(pta, empirical_distr=emp_dist_pkl)
Sampler.addProposalToCycle(jp.draw_from_prior, 15)
# Sampler.addProposalToCycle(jp.draw_from_dm_sw_prior, 20)
Sampler.addProposalToCycle(jp.draw_from_signal_prior, 20)
Sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 35)
Sampler.addProposalToCycle(jp.draw_from_sw1_prior, 60)
if args.bayes_ephem:
    Sampler.addProposalToCycle(jp.draw_from_ephem_prior, 35)
for ii,pow in enumerate(args.sw_r2p):
    draw = getattr(jp,'draw_from_sw{0}_prior'.format(ii+1))
    Sampler.addProposalToCycle(draw, 60)
if args.gwb_off:
    pass
else:
    Sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 20)
Sampler.addProposalToCycle(jp.draw_from_empirical_distr, 60)

N = args.niter
Sampler.sample(x0, Niter=N, SCAMweight=30, AMweight=15,
               writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain,
               DEweight=30, burn=200000)

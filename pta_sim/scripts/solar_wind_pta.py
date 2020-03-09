#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pint.toa as toa
import pint.models as models
import pint.fitter as fit
import pint.residuals as r
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

noise_json = '/home/jeffrey.hazboun/nanograv/Work/solar_wind/ng11yr_sw_noise_dict.json'
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

n_earth = SW.ACE_SWEPAM_Parameter()('n_earth')
sw = SW.solar_wind(n_earth=n_earth)
mean_sw = deterministic_signals.Deterministic(sw, name='mean_sw')

dm_block = dm_gp + mean_sw

if args.sw_r4p4:
    def DM_m(z,b,m):
        return 1/(z**2+b**2)**(m/2)

    def dm_integral(th,Re,m):#quad, trapz
        return spi.quad(DM_m,
                        -Re*np.cos(th),np.inf,
                        args=(Re*np.sin(th),m),epsabs=1.49e-11)[0]

    def psi_m(theta,Rearth,m):
        return np.array([dm_integral(th,Re,m)
                         for th,Re in zip(theta,Rearth)])

    AU_light_sec = const.AU / const.c #1 AU in light seconds
    AU_pc = const.AU / const.pc #1 AU in parsecs (for DM normalization)

    def dm_solar_m(n_m,m,psi):
        return n_m*AU_pc*AU_light_sec**(m-1)*psi


    for p in psrs:
        theta, R_earth = SW.theta_impact(p.planetssb, p.sunssb, p.pos_t)
        p.psi = psi_m(theta,R_earth,4.39)
        p.sol_ang = theta


    @signal_base.function
    def solar_wind_m(toas, freqs, planetssb, pos_t, psi,
                      log10_n_m=-2, m=4.4):
        """
        Construct DM-Solar Model fourier design matrix.

        :param toas: vector of time series in seconds
        :param planetssb: solar system bayrcenter positions
        :param pos_t: pulsar position as 3-vector
        :param freqs: radio frequencies of observations [MHz]
        :param n_m: The electron density from the solar wind at 1 AU. Due
                to a radial denisty of 1/r^m.

        :return dt_DM: Chromatic time delay due to solar wind
        """

        n_m = 10**log10_n_m

        dm_sol_wind = dm_solar_m(n_m,m,psi)#(n_m,theta,R_earth,m)
        dt_DM = (dm_sol_wind) * 4.148808e3 / freqs**2

        return dt_DM

    n_earth_m = parameter.Uniform(-10,-2)('n_m4p4')
    sw_m = solar_wind_m(log10_n_m=n_earth_m,m=4.4)
    mean_sw_m = deterministic_signals.Deterministic(sw_m, name='mean_sw_m4p4')

    dm_block += mean_sw_m

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

np.savetxt(args.outdir + 'pars.txt', pta.param_names, fmt='%s')
np.savetxt(args.outdir + '/priors.txt',
           list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

class my_JP(sampler.JumpProposal):
    def __init__(self, pta, snames=None, empirical_distr=None):
        super().__init__(pta, snames=None, empirical_distr=None)

    def draw_from_mean_sw_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'mean_sw'

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

    def draw_from_mean_sw_m4p4_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'mean_sw_m4p4'

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

emp_dist_pkl='/home/jeffrey.hazboun/nanograv/Work/solar_wind/ng11yr_emp_dist_sw_plaw_rn_be_se_dmgp.pkl'
jp = my_JP(pta, empirical_distr=emp_dist_pkl)
Sampler.addProposalToCycle(jp.draw_from_prior, 15)
# Sampler.addProposalToCycle(jp.draw_from_dm_sw_prior, 20)
Sampler.addProposalToCycle(jp.draw_from_signal_prior, 20)
Sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 35)
Sampler.addProposalToCycle(jp.draw_from_mean_sw_prior, 15)
if args.bayes_ephem:
    Sampler.addProposalToCycle(jp.draw_from_ephem_prior, 35)
if args.sw_r4p4:
    Sampler.addProposalToCycle(jp.draw_from_mean_sw_m4p4_prior, 15)
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

#!/usr/bin/env python
# coding: utf-8
import numpy as np
# import pint.toa as toa
# import pint.models as models
# import pint.fitter as fit
# import pint.residuals as r
import astropy.units as u
import scipy.integrate as spi
import scipy.stats as sps
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

import corner, pickle, sys, json, os
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

import ultranest
import ultranest.stepsampler

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
        if len(pr_range)!=1:
            n_earth = parameter.Uniform(pr_range[0],
                                        pr_range[1])('nE_{0}'.format(ii+1))
        else:
            n_earth = SW.ACE_SWEPAM_Parameter()('nE_{0}'.format(ii+1))
        deter_sw = SW.solar_wind(n_earth=n_earth)
        dm_block += deterministic_signals.Deterministic(deter_sw,
                                                 name='sw_{0}'.format(ii+1))
    elif len(power)==1:
        print('2 ',power,pr_range)
        n_earth = parameter.Uniform(pr_range[0],
                                    pr_range[1])('nE_{0}'.format(ii+1))
        sw_power = parameter.Constant(power[0])('sw_power_{0}'.format(ii+1))
        log10_ne = True if pr_range[0] < 0 else False
        deter_sw = SW.solar_wind_r_to_p(n_earth=n_earth,
                                                power=sw_power,
                                                log10_ne=log10_ne)
        dm_block += deterministic_signals.Deterministic(deter_sw,
                                                     name='sw_{0}'.format(ii+1))
    elif len(power)>1:
        print('3 ',power,pr_range)
        n_earth = parameter.Uniform(pr_range[0], pr_range[1])('nE_{0}'.format(ii+1))
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

if args.sw_pta_gp:
    @signal_base.function
    def solar_wind_perturb(toas, freqs, planetssb, sunssb, pos_t, n_earth_rho=0,
                   n_mean=5, nmodes=20,
                   Tspan=None, logf=False, fmin=None, fmax=None, modes=None):

        """
        Construct DM-Solar Model fourier design matrix.

        :param toas: vector of time series in seconds
        :param planetssb: solar system bayrcenter positions
        :param pos_t: pulsar position as 3-vector
        :param freqs: radio frequencies of observations [MHz]
        :param n_earth_rho: electron density from the solar wind
                    at 1 AU.
        :param n_earth_bins: Number of binned values of n_earth for which to fit or
                    an array or list of bin edges to use for binned n_Earth values.
                    In the latter case the first and last edges must encompass all
                    TOAs and in all cases it must match the size (number of
                    elements) of log10_n_earth.
        :param t_init: Initial time of earliest TOA in entire dataset, including all
                    pulsar.
        :param t_final: Final time of latest TOA in entire dataset, including all
                    pulsar.

        :return dt_DM: DM due to solar wind
        """
        if modes is not None:
            nmodes = len(modes)

        #print(n_earth_rho)
        if n_earth_rho.size!= 2*nmodes:
            raise ValueError('Length of n_earth_rho must match 2 x nmodes.')

        F, Ffreqs = utils.createfourierdesignmatrix_red(toas, nmodes=nmodes, Tspan=Tspan,
                                                        logf=logf, fmin=fmin, fmax=fmax,
                                                        modes=modes)

        n_Earth = np.einsum('ij,j', F, n_earth_rho)#np.repeat(10**n_earth_rho,2))
        theta, R_earth, _, _ = SW.theta_impact(planetssb, sunssb, pos_t)
        dm_sol_wind = SW.dm_solar(1.0, theta, R_earth)
        dt_sw = n_Earth * dm_sol_wind * 4.148808e3 / freqs**2

        return dt_sw

    n_earth_rho = parameter.Normal(0, 0.5, size=60)('n_earth_rho')
    sw_pert = solar_wind_perturb(n_earth_rho=n_earth_rho, Tspan=Tspan, nmodes=30)
    sw_perturb = deterministic_signals.Deterministic(sw_pert, name='sw_perturb')
    model += sw_perturb

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


# set up jump groups by red noise groups

# groups = sampler.get_parameter_groups(pta)
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

np.savetxt(args.outdir + '/pars.txt', pta.param_names, fmt='%s')
np.savetxt(args.outdir + '/priors.txt',
           list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')
class sw_trans():
    def __init__(self):
        self.ppf = SW.ACE_RV.ppf
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
    if param.type.lower()=='uniform':
        pmin = param.prior._defaults['pmin']
        pmax = param.prior._defaults['pmax']
        transforms.append(uniform_trans(pmin,pmax))
    elif param.type.lower()=='normal':
        mu = param.prior._defaults['mu']
        sigma = param.prior._defaults['sigma']
        transforms.append(normal_trans(mu,sigma))
    elif param.type.lower()=='ace_swepam_parameter':
        transforms.append(sw_trans())

def transform(quantile):
    return np.array([t(q) for q,t in zip(quantile,transforms)])

sampler1 = ultranest.ReactiveNestedSampler(
    pta.param_names,
    pta.get_lnlikelihood,
    transform,
    log_dir=args.outdir,
    resume=True,
)
ndim = len(pta.params)
sampler1.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=2*ndim)

sampler1.run(dlogz=0.5 + 0.1 * ndim,
     # update_interval_iter_fraction=0.4 if ndim > 20 else 0.2,
     # max_num_improvement_loops=3,
     min_num_live_points=400)

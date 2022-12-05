#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.interpolate as si
import scipy.stats as sps
import scipy.linalg as sl
import glob, pickle, json, copy


import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky

from enterprise.pulsar import Pulsar as ePulsar

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

with open(args.pickle, 'rb') as fin:
    ePsrs=pickle.load(fin)


tspans = [hsen.get_Tspan([p])/(365.25*24*3600) for p in ePsrs]

psr_list = [p.name for p in ePsrs]

with open(args.noisepath, 'r') as fin:
    noise = json.load(fin)

def quantize_fast(toas, toaerrs, flags=None, dt=0.1):
    r"""
    Function to quantize and average TOAs by observation epoch. Used especially
    for NANOGrav multiband data.

    Pulled from `[3]`_.

    .. _[3]: https://github.com/vallis/libstempo/blob/master/libstempo/toasim.py

    Parameters
    ----------

    times : array
        TOAs for a pulsar.

    flags : array, optional
        Flags for TOAs.

    dt : float
        Coarse graining time [days].
    """
    isort = np.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]
    dt *= (24*3600)
    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(toas[l]) for l in bucket_ind],'d')
    avetoaerrs = np.array([sps.hmean(toaerrs[l]) for l in bucket_ind],'d')
    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])

    U = np.zeros((len(toas),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1

    if flags is not None:
        return avetoas, avetoaerrs, aveflags, U, bucket_ind
    else:
        return avetoas, avetoaerrs, U, bucket_ind


def make_corr(psr,noise):
    N = psr.toaerrs.size
    corr = np.zeros((N,N))
    _, _, fl, _, bi = quantize_fast(psr.toas,psr.toaerrs,
                                     flags=psr.flags['f'],dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl,dtype=float)
    for be in backends:
        mask = np.where(psr.flags['f']==be)
        key_ef = '{0}_{1}_{2}'.format(psr.name,be,'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name,be,'t2equad')
        sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2)
                           + (10**noise[key_eq])**2)
        mask_ec = np.where(fl==be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name,be,'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket),len(bucket)))
         for ii, bucket in enumerate(bi)]

    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr

rn = {}
for ky in psr_list:
    parA = '{0}_red_noise_log10_A'.format(ky)
    parGam = '{0}_red_noise_gamma'.format(ky)
    gam, A = noise[parGam], noise[parA]
    rn[ky]= [10**A, gam]

Tspan = hsen.get_Tspan(ePsrs)

fyr = 1/(365.25*24*3600)

if args.logf:
    freqs = np.logspace(np.log10(2e-10),np.log10(2e-7),300)
else:
    freqs = np.linspace(1/Tspan,args.n_gwbfreqs/Tspan,args.n_gwbfreqs)


ePsr = ePsrs[args.process]
if ePsr.toas.size>20_000:
    thin = 5
else:
    thin = args.thin
corr = make_corr(ePsr,noise=noise)[::thin,::thin]
if args.A_gwb != 0.0:
    if args.gamma_gw is None:
        gamma_gw = 13/3.
    else:
        gamma_gw = args.gamma_gw
    gwb = hsen.red_noise_powerlaw(A=args.A_gwb, gamma=gamma_gw, freqs=freqs)
    corr += hsen.corr_from_psd(freqs=freqs, psd=gwb,
                               toas=ePsr.toas[::thin])
if args.rednoise:
    Amp, gam = rn[ePsr.name]
    plaw = hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)
    corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                               toas=ePsr.toas[::thin])

psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                  toaerrs=ePsr.toaerrs[::thin],
                  phi=ePsr.phi,
                  theta=ePsr.theta,
                  N=corr,
                  designmatrix=ePsr.Mmat[::thin,:])

psr.name = ePsr.name
_ = psr.G

if args.savepsr:
    with open(args.outdir+f'/{args.label}_psr_{ePsr.name}.has', 'wb') as fout:
        pickle.dump(psr,fout)

sp = hsen.Spectrum(psr, freqs=freqs)
sp.name = psr.name
_ = sp.NcalInv
_ = sp.P_n


with open(args.outdir+f'/{args.label}_spec_{ePsr.name}.has', 'wb') as fout:
    pickle.dump(sp,fout)

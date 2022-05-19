#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy

from enterprise_extensions import models, model_utils, hypermodel
from enterprise.signals import signal_base
import pta_sim
import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

#Is chain longer than niter?
longer = chain_length_bool(args.outdir, int(args.niter//10))

if longer:
    sys.exit() #Hmmmm what to do here?
else:
    pass

if args.pickle=='no_pickle':
    #Get par and tim files.
    parfiles = sorted(glob.glob(args.pardir+'*.par'))
    timfiles = sorted(glob.glob(args.timdir+'*.tim'))

    if args.psr_list is not None:
        parfiles = filter_psr_path(parfiles,args.psr_list,rhs='_')
        timfiles = filter_psr_path(timfiles,args.psr_list,rhs='_')

    psrs = []
    for p, t in zip(parfiles, timfiles):
        psr = Pulsar(p, t, ephem=args.ephem)
        psrs.append(psr)

else:
    with open('{0}'.format(args.pickle), "rb") as f:
        psrs = pickle.load(f)

    if args.psr_list is not None:
        idxs = []
        for idx, psr in enumerate(psrs):
            if psr.name not in args.psr_list:
                idxs.append(idx)

        for idx in reversed(idxs):
            del psrs[idx]


with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if args.rn_psrs[0]=='all':
    rn_psrs='all'
else:
    rn_psrs=args.rn_psrs


pta_gw = models.model_3a(psrs, psd='powerlaw', noisedict=noise,
                            n_gwbfreqs=args.nfreqs,
                            gamma_common=13/3., upper_limit=False,
                            bayesephem=args.bayes_ephem, be_type='setIII',
                            wideband=False, rn_psrs=rn_psrs,
                            pshift=args.pshift, pseed=args.process,
                            psr_models=False)

pta_crn = models.model_2a(psrs, psd='powerlaw', noisedict=noise,
                             n_gwbfreqs=args.nfreqs,
                             gamma_common=13/3., upper_limit=False,
                             bayesephem=args.bayes_ephem, be_type='setIII',
                             wideband=False, rn_psrs=rn_psrs,
                             pshift=args.pshift, pseed=args.process,
                             select='backend', psr_models=False)

print(pta_gw.get_lnlikelihood(np.hstack(p.sample() for p in pta_gw.params)))
print(pta_crn.get_lnlikelihood(np.hstack(p.sample() for p in pta_crn.params)))
#
# #Load in T matrix from previous run to check same random phase shifts
# if os.path.exists(args.outdir+'Tmats.npy'):
#     Tmats = np.load(args.outdir+'Tmats.npy')
#     print('Loading Tmat list from earlier incarnation.')
# else: # Save random phase shifted T matrices
#     Tmats = [sc._Fmat for sc in pta_gw._signalcollections]
#     os.makedirs(args.outdir)
#     np.save(args.outdir+'Tmats.npy',Tmats)
#     print('Pulling Tmat from model3a and saving')
#
# # Check mods_2a T matrices against the 3a ones
# for ii, (s2,s3) in enumerate(zip(pta_gw._signalcollections,
#                                  pta_crn._signalcollections)):
#     if s2.psrname != s3.psrname:
#         raise ValueError('Pulsars do not match for T Matrix check')
#     else:
#         if np.all(pta_crn._signalcollections[ii]._Fmat == Tmats[ii]):
#             print('CRN T Matrices match')
#         else:
#             err_msg = 'PSR {0} Stored T Matrix'.format(s3.psrname)
#             err_msg +=' does not match CRN PTA !!'
#             raise ValueError(err_msg)
#
#         if np.all(pta_gw._signalcollections[ii]._Fmat == Tmats[ii]):
#             print('GW T Matrices match')
#         else:
#             err_msg = 'PSR {0} Stored T Matrix'.format(s3.psrname)
#             err_msg +=' does not match GW PTA !!'
#             raise ValueError(err_msg)


ptas = {0:pta_crn,
        1:pta_gw}

hm = hypermodel.HyperModel(models=ptas)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr = args.emp_distr)

achrom_freqs = get_freqs(ptas[0], signal_id='gw')
np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')

model_params = {}
for ii,mod in enumerate(ptas):
    model_params.update({ii : ptas[ii].param_names})

with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params, fout, sort_keys=True, indent=4,
              separators=(',', ': '))

x0 = hm.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15,
               DEweight=30, burn=2000000)

save_core(args.corepath, args.outdir)

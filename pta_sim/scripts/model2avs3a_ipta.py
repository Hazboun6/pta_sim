#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
import cloudpickle

from enterprise_extensions import models, model_utils, hypermodel

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

if os.path.exists(args.outdir+'pta.pkl'):
    with open(args.outdir+'pta.pkl', "rb") as f:
        ptas = cloudpickle.load(f)
else:
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

    psr_names = [p.name for p in psrs]

    if args.rm_psrs is not None:
        rm_idxs = [psr_names.index(p) for p in args.rm_psrs]
        print('Removing the following pulsars:\n {0}'.format(args.rm_psrs))
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
        pidxs = []
        for pidx, psr in enumerate(psrs):
            start_time = psr.toas.min()/(24*3600)
            last_time = psr.toas.max()/(24*3600)
            if (last_time-start_time)/365.25 < args.min_tspan:
                print('PSR {0} baseline less than {1} years. Not being included in analysis'.format(psr.name,args.min_tspan))
                pidxs.append(pidx)

        for idx in reversed(pidxs):
            del psrs[idx]
    else:
        pidxs = []
        for pidx, psr in enumerate(psrs):
            start_time = psr.toas.min()/(24*3600)
            if (args.end_time-start_time)/365.25 <= args.min_tspan:
                print('PSR {0} baseline too short for this slice.'.format(psr.name))
                pidxs.append(pidx)
            else:
                psr.filter_data(start_time=start_time, end_time=args.end_time)

        for idx in reversed(pidxs):
            del psrs[idx]
        Outdir = args.outdir+'{0}/'.format(args.nyears)



    with open(args.noisepath, 'r') as fin:
        noise =json.load(fin)

    if args.rn_psrs[0]=='all':
        rn_psrs='all'
    else:
        rn_psrs=args.rn_psrs

    pta_crn = models.model_general(psrs, tm_var=False,
                      Tspan=None, common_psd=args.psd, red_psd='powerlaw', orf='crn',
                      common_components=args.nfreqs, red_components=30,
                      dm_components=30, noisedict=noise, rn_psrs=rn_psrs,
                      gamma_common=args.gamma_gw, delta_common=0.,
                      upper_limit=False,
                      bayesephem=args.bayes_ephem, be_type='setIII_1980',
                      dm_var=True, dm_type='gp', dm_psd='powerlaw',names=['gw'])

    pta_gw = models.model_general(psrs, tm_var=False,
                      Tspan=None, common_psd=args.psd, red_psd='powerlaw', orf='hd',
                      common_components=args.nfreqs, red_components=30,
                      dm_components=30, noisedict=noise, rn_psrs=rn_psrs,
                      gamma_common=args.gamma_gw, delta_common=0.,
                      upper_limit=False,
                      bayesephem=args.bayes_ephem, be_type='setIII_1980',
                      dm_var=True, dm_type='gp', dm_psd='powerlaw',names=['gw'])

    # delta_common=0.,
    ptas = {0:pta_crn,
            1:pta_gw}

if args.model_wts is None:
    model_wts = None
else:
    model_wts = dict(enumerate(args.model_wts))

hm = hypermodel.HyperModel(models=ptas, log_weights=model_wts)
sampler = hm.setup_sampler(outdir=args.outdir, resume=True,
                           empirical_distr = args.emp_distr)

try:
    achrom_freqs = get_freqs(ptas[0], signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

model_params = {}
for ii,mod in enumerate(ptas):
    model_params.update({ii : ptas[ii].param_names})

with open(args.outdir+'/model_params.json' , 'w') as fout:
    json.dump(model_params, fout, sort_keys=True, indent=4,
              separators=(',', ': '))

x0 = hm.initial_sample()
sampler.sample(x0, args.niter, SCAMweight=30, AMweight=15,
               DEweight=30, burn=300000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain)

save_core(args.corepath, args.outdir)

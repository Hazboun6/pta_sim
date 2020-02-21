#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle
import numpy as np
import la_forge.diagnostics as dg
import la_forge.core as co
import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()

rootdir = '/home/jeffrey.hazboun/nanograv/12p5yr_work/noise_model/'
if args.psr != 'J1918-0642':
    chaindir = rootdir + 'psr_chains/{0}/1st_round/'.format(args.psr)
else:
    chaindir = rootdir + 'psr_chains/{0}/1st_round_rf/'.format(args.psr)

plotdir = rootdir + '1st_round_plots/'
label = 'PSR {}, 1st Round Model Selection'.format(args.psr)
cH=co.HyperModelCore(label=label,
                     chaindir=chaindir)

vals, bins = np.histogram(cH.get_param('nmodel'),
                          bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5])
n_most = np.argmax(vals)

dg.plot_chains(cH,hist=False,pars=['nmodel'],ncols=1,
               title_y=1.05,
               suptitle='PSR {0} Model Selection Integer'.format(args.psr),
               save=plotdir+'{0}_nmodel.pdf'.format(args.psr))

c0 = cH.model_core(n_most)
dg.plot_chains(c0, hist=False, ncols=5, title_y=1.05,
               exclude=['lnprior', 'pt_chain_accept'],
               save=plotdir+'{0}_nmost_param_chains.pdf'.format(args.psr))

std_dir = '/home/joseph.simon/12p5yr_GWB/V3_dataset/chains/'
std_dir += 'SinglePsr_StandardNoise/{0}/singlepsrNoise/'.format(args.psr)
std_core = co.Core(label='Std RN Plaw {0}'.format(args.psr),
                   chaindir=std_dir)

dg.plot_chains([c5,std_core], hist=True, pars=std_core.params[:-4], ncols=4,
               title_y=1.05,linewidth=2,
               legend_labels=['Best Model','Std DMX'],
               plotdir+'{0}_std_noise_compare.pdf'.format(args.psr))

with open(chaindir+'/model_labels.json' , 'r') as fin:
    model_labels= json.load(fin)

dg.noise_flower(cH,
                colLabels=['Model','Achrom', 'DM GP', 'SW GP'],
                cellText=model_labels,
                colWidths=[0.15,0.4,0.25,0.25],
                plot_path=plotdir+'{0}_noise_flower.pdf'.format(args.psr))

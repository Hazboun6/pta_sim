#!/usr/bin/env python
# coding: utf-8

# Code for running enterprise Bayesian analyses on simulated data.

from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

from enterprise_extensions import models, model_utils
# from enterprise_extensions.electromagnetic import solar_wind
from la_forge.core import Core, HyperModelCore

def chain_length_bool(path, N):
    chain_path = path + '/chain_1.txt'
    if os.path.exists(chain_path):
        with open(chain_path) as f:
            Chain_rows = sum(1 for _ in f)
    else:
        Chain_rows = 0

    if Chain_rows >= N:
        return True
    else:
        return False

def save_core(core_path, chaindir, remove=False,
              hyper_model=False, param_dict=None):
    if hyper_model:
        co = HyperModelCore(label=core_path, chaindir=chaindir,
                            param_dict=None)
    else:
        co = Core(label=core_path, chaindir=chaindir)

    try:
        co.set_rn_freqs(freq_path=chaindir+'/achrom_rn_freqs.txt')
    except:
        pass

    co.save(core_path)

    if remove:
        files = os.listdir(chaindir)
        idxs = []
        for f in files:
            if f.endswith('.txt'):
                os.remove(f)
                idxs.append(files.index(f))
            if f.endswith('.npy'):
                os.remove(f)
                idxs.append(files.index(f))
        print('Following files removed:')
        for idx in idxs:
            print(files[idx])
    del co

def get_freqs(pta, signal_id='red_noise'):
    """ Hackish way to get frequency vector."""
    for sig in pta._signalcollections[0]._signals:
        if sig.signal_id == signal_id:
            sig._construct_basis()
            freqs = np.array(sig._labels[''])[::2]
            break
    return freqs

def psr_name(x,rhs='.'):
    return x.split('/')[-1].split(rhs)[0]

def filter_psr_path(path_list,psr_list,rhs='_'):
    return [p for p in path_list if psr_name(p, rhs) in psr_list]

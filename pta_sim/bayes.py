#!/usr/bin/env python
# coding: utf-8

# Code for running enterprise Bayesian analyses on simulated data.

from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

from enterprise_extensions import models, model_utils
# from enterprise_extensions.electromagnetic import solar_wind
from la_forge.core import Core

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

def save_core(core_path,chaindir):
    co = Core(label='',chaindir=chaindir)
    co.set_rn_freqs(chaindir+'/achrom_rn_freqs.txt')
    co.save(core_path)
    del co

def get_freqs(pta):
    """ Hackish way to get frequency vector."""
    for sig in pta._signalcollections[0]._signals:
        if sig.signal_name == 'red noise':
            sig._construct_basis()
            freqs = np.array(sig._labels[''])[::2]
            break
    return freqs

def psr_name(x,rhs='.'):
    return x.split('/')[-1].split(rhs)[0]

def filter_psr_path(path_list,psr_list,rhs='_'):
    return [p for p in path_list if psr_name(p, rhs) in psr_list]

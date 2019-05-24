#!/usr/bin/env python
# coding: utf-8
import numpy as np
import glob, json

from collections import OrderedDict

__all__ = ['load_noise_files',
           'get_noise']

def load_noise_files(noisedir=None, psr_list=None, noisepath=None):
    '''
    Function to load enterprise-style noise dictionaries either from a single
    `json` file or a directory of them with psr names in the file name.

    Parameters
    ----------
    noisedir : str
        Path to directory of pulsar noise files.

    psr_list : list of str
        List of pulsar to choose which files from `noisedir` to load.

    noisepath : str
        Path to single pulsar niose file.

    Returns
    -------
    Dictionary of concatenated noise parameters.
    '''
    noisedict = OrderedDict()
    if noisedir is not None and noisepath is None:
        noisepaths = glob.glob(noisedir+'/*.json')
        noisepaths = sorted([f for f in noisepaths
                             if any([p in f for p in psr_list])])
        for path in noisepaths:
            with open(path,'r') as fin:
                pdict = json.load(fin)
            noisedict.update(pdict)
    elif noisepath is not None and noisedir is None:
        with open(noisepath,'r') as fin:
            noisedict = json.load(fin)
    elif noisepath is None and noisedir is None:
        raise ValueError('Must enter noise directory or noise file path.')

    return noisedict


def get_noise(par, psr, noisedict, return_dict=False):
    '''
    Function to retrieve noise parameters from
    enterprise-style noise dictionary.

    Parameters
    ----------
    par : str,
        Parameter name to search for. Should match string in
        noise dictionary key. Eg. {ecorr, equad, efac, gamma, ...}

    Returns
    -------

    flags : list
        List of flags use for given noise parameter. If no selection used
        then returns parameter name.

    values : list
        List of noise values from dictionary corresponding to flags above.

    Optionally returns a dictionary where flags are the keys.
    '''
    flags = []
    vals = []
    for ky in noisedict.keys():
        if par in ky and psr in ky:
            islog = False
            out_psr = '{0}_'.format(psr)
            if 'log' in ky:
                out_par = '_log10_{0}'.format(par)
                islog = True
            else:
                out_par = '_{0}'.format(par)
            flag = ky.replace(out_psr,'')
            flag = flag.replace(out_par,'')
            flags.append(flag)
            if islog:
                vals.append(10**noisedict[ky])
            else:
                vals.append(noisedict[ky])
        else:
            pass
    if return_dict:
        return dict(zip(flags, vals))
    else:
        return flags, vals

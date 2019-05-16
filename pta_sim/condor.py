#!/usr/bin/env python
# coding: utf-8

# Code for writing scripts for use on Condor


from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

def write_submit(slug, py_script, sub_path, args, err_dir='./error_out/',
                 mpi=False, **sub_kwargs):
    '''
    Function to write condor subimt files.

    Parameters
    ----------

    slug : str
        String, usually including $() variables for condor submit file that
        will be used for naming error and out files.

    py_script : str
        Full path to Python script being run by this submit file.

    err_dir : str
        Directory for error, log and out files.

    mpi : bool
        Whether to do an mpi run. Must set appropriate arguments in `args`
        dictionary.

    args : dict, OrderedDict
        Dictionary of arguments to included with Python script.

    sub_kwargs : dict, OrderedDict
        Dictionary of other values for submit file, including details of mpi runs

    '''
    if os.path.exists(err_dir):
        pass
    else:
        os.makedirs(err_dir)

    out = '{0}/{1}.out'.format(err_dir,slug)
    err = '{0}/{1}.err'.format(err_dir,slug)

    #$(nyears)_$(psrs)_slice
    #$(nyears)_$(psrs)_slice

    if mpi:
        executable = '/usr/bin/mpirun'
        environment = '\"MKL_NUM_THREADS={0}\"'.format(sub_kwargs['request_cpus'])
        arguments = '-np {0} {1}'.format(1,py_script)
        #TODO Make the np call more general.
    else:
        environment = ''
        executable = py_script
        arguments = ''

    for ky in args.keys():
        arguments += ' {0} {1}'.format(ky, args[ky])


    values = OrderedDict({
              'universe':'vanilla',
              'request_cpus' : 1,
              'request_memory' : 3072,
              'environment' : environment,
              'accounting_group' : 'cgca.nanograv',
              'run_as_owner' : 'true',
              'priority' : 10,
              'getenv' : 'True',
              'output' : out,
              'error' : err,
              'executable' : executable,
              'arguments' : arguments,
              'queue' : '',
              })

    values.update(sub_kwargs)

    mpi_keys = ['request_cpus']

    if mpi and any([ky not in sub_kwargs.keys() for ky in mpi_keys]):
        raise ValueError('Must include all appropriate MPI values.')
    elif mpi and all([ky in sub_kwargs.keys() for ky in mpi_keys]):
        pass
    else:
        values.__delitem__('environment')

    with open(sub_path,"w") as output:
        for ky in values.keys():
            if ky != 'queue':
                output.write('{0} = {1} \n'.format(ky, values[ky]))
            else:
                output.write('{0} {1} \n'.format(ky, values[ky]))

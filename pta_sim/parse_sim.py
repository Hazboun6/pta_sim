#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np

parser = argparse.ArgumentParser()

timdir = '/home/jeffrey.hazboun/nanograv/dsa2000_simulations/data_jsh/partim/'
pardir = '/home/jeffrey.hazboun/nanograv/dsa2000_simulations/data_jsh/partim/'
pardir_no_dmx = '/home/jeffrey.hazboun/nanograv/dsa2000_simulations/dsa_partim_181214'

parser.add_argument('--A_dm', dest='A_dm', action='store',
                    type=float, default=0.0,
                    help='Chromatic f^-2 Red Noise Amplitude to Simulate')
parser.add_argument('--A_gwb', dest='A_gwb', action='store',
                    type=float, default=0.0, help='GWB Amplitude to Simulate')
parser.add_argument('--A_rn', dest='A_rn', action='store',
                    type=float, default=0.0,
                    help='Achromatic Red Noise Amplitude to Simulate')
parser.add_argument('--corepath', dest='corepath', action='store', type=str,
                    default='./chain.core', help='Path name (dir and filename) for output.')
parser.add_argument('--outdir', dest='outdir', action='store', type=str,
                    default='./', help='Path directory for output.')
parser.add_argument('--outtimdir', dest='outtimdir', action='store', type=str,
                    default='./fake_tim_files', help='Output Directory for tim files.')
parser.add_argument('--ephem', dest='ephem', action='store',
                    type=str, default='DE436', help='SS Ephemeris')
parser.add_argument('--end_time', dest='end_time', action='store',
                    type=float, default=None, help='End MJD for slicing.')
parser.add_argument('--process', dest='process', action='store',
                    type=int, help='Condor process number')
parser.add_argument('--gamma_dm', dest='gamma_dm', action='store',
                    type=float, default=2.0,
                    help='Chromatic f^-2 Red Noise Spectral Index to Simulate')
parser.add_argument('--gamma_gw', dest='gamma_gw', action='store',
                    type=float, default=13./3, help='GWB spectral index')
parser.add_argument('--gamma_rn', dest='gamma_rn', action='store',
                    type=float, default=5,
                    help='Achromatic Red Noise spectral index to Simulate')
parser.add_argument('--obs', dest='obs', action='store',
                    type=str, default='all', help='Observatory')
parser.add_argument('--pickle', dest='pickle', action='store',
                    type=str, default='no_pickle',
                    help='Option to give a pickled enterprise list of pulsars')
parser.add_argument('--nfreqs', dest='nfreqs', action='store',
                    type=int, default=30, help='Number of Freqs in GW model')

parser.add_argument('--bayes_ephem', dest='bayes_ephem', action='store_true',
                    default=False, help='Option to use Bayes Ephemeris Modeling.')

parser.add_argument('--cuts', dest='cuts', action='append',
                    help='List of cuts at which to calculate OS')
parser.add_argument('--dm_gp_psrs',dest='dm_gp_psrs', action='append',
                    type=str, help='Pulsars to use dm gaussian process '
                                   'modeling for analysis. Append with a '
                                   'flag for each pulsar.')
parser.add_argument('--dm_sin_psrs',dest='dm_sin_psrs', action='append',
                    type=str, default=None,
                    help='Pulsars to use dm sine wave modeling for analysis. '
                         'Append with a flag for each pulsar.')
parser.add_argument('--emp_distr', dest='emp_distr', action='store',
                    type=str, default=None,
                    help='Option to give a pickled enterprise list of pulsars')
parser.add_argument('--free_spec_psrs',dest='free_spec_psrs', action='append',
                    type=str,
                    help='Pulsars for which to use a free spectral '
                         'model for analysis. Append with a flag for each pulsar.')
parser.add_argument('--logf', dest='logf', action='store_true',
                    default=False,
                    help='Option to use log spaced frequencies in GPs.')
parser.add_argument('--niter', dest='niter', action='store',
                    type=int, default=1001000, help='Number of iterations')
parser.add_argument('--noisepath', dest='noisepath', action='store', type=str,
                    default='./os.txt', help='Path to noise file or files.')
parser.add_argument('--outpath', dest='outpath', action='store', type=str,
                    default='./os.txt', help='Path name (dir and filename) for output.')
parser.add_argument('--pardir', dest='pardir', action='store', type=str,
                    default=pardir, help='par file Directory')
parser.add_argument('--parpath', dest='parpath', action='store', type=str,
                    default=None, help='par file path')
parser.add_argument('--pardir_no_dmx', dest='pardir_no_dmx', action='store',
                    type=str, default=pardir_no_dmx, help='Output Directory')
parser.add_argument('--psr', dest='psr', action='store',
                    type=str, default=None,
                    help='Name of an individual pulsar. Used for noise runs.')
parser.add_argument('--psr_list', dest='psr_list', action='append',
                    default=None,
                    help='List pulsar names to use in analysis.')
parser.add_argument('--spac_corr_info', dest='spac_corr_info',
                    action='store_true', default=False,
                    help='Whether to write out the spacial correlation information')
parser.add_argument('--timdir', dest='timdir', action='store', type=str,
                    default=timdir, help='Tim file Directory')
parser.add_argument('--timpath', dest='timpath', action='store', type=str,
                    default=None, help='Tim file path')
parser.add_argument('--tspan', dest='tspan', action='store',
                    type=float, default=None,
                    help='Timespan to use for GP frequencies.')
parser.add_argument('--vary_dm_params', dest='vary_dm_params', action='store_false',
                    default=True, help='Option to vary the DM model parameters.')
parser.add_argument('--vary_gamma', dest='vary_gamma', action='store_true',
                    default=False,
                    help='Option to vary gamma, the spectral index '
                         'on the common red noise process.')
parser.add_argument('--wideband', dest='wideband', action='store_true',
                    default=False, help='Option to use wideband data.')
# parse arguments
args = parser.parse_args()

if args.psr_list is None:
    pass
elif len(args.psr_list)==1:
    args.psr_list = np.loadtxt(args.psr_list[0], dtype='S25').astype('U25')

def arguments():
    return args

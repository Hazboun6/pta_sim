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
parser.add_argument('--adv_noise_psrs_only', dest='adv_noise_psrs_only',
                    action='store_true',
                    default=False,
                    help='Option to use only advanced noise pulsars in analysis.')
parser.add_argument('--bayes_ephem', dest='bayes_ephem', action='store_true',
                    default=False,
                    help='Option to use Bayes Ephemeris Modeling.')
parser.add_argument('--corepath', dest='corepath', action='store', type=str,
                    default='./chain.core',
                    help='Path name (dir and filename) for output.')
parser.add_argument('--cuts', dest='cuts', action='append',
                    help='List of cuts at which to calculate OS')
parser.add_argument('--dm_dip', dest='dm_dip', action='store_true',
                    default=False,
                    help='Option to use a DM exponential dip in J1713.')
parser.add_argument('--dm_gp_psrs',dest='dm_gp_psrs', action='append',
                    type=str, help='Pulsars to use dm gaussian process '
                                   'modeling for analysis. Append with a '
                                   'flag for each pulsar.')
parser.add_argument('--dm_sin_psrs',dest='dm_sin_psrs', action='append',
                    type=str, default=None,
                    help='Pulsars to use dm sine wave modeling for analysis. '
                         'Append with a flag for each pulsar.')
parser.add_argument('--efac', dest='efac', action='store_true',
                    default=False, help='Search EFAC in an analysis.')
parser.add_argument('--dropout', dest='dropout', action='store_true',
                    default=False, help='Use a dropout analysis.')
parser.add_argument('--dp_thresh', dest='dp_thresh', action='store',
                    type=float, default=0.5, help='Dropout threshold, between 0 and 1.')
parser.add_argument('--ephem', dest='ephem', action='store',
                    type=str, default='DE436', help='SS Ephemeris')
parser.add_argument('--emp_distr', dest='emp_distr', action='store',
                    default=None,
                    help='Option to give a pickled enterprise list of pulsars')
parser.add_argument('--end_time', dest='end_time', action='store',
                    type=float, default=None, help='End MJD for slicing.')
parser.add_argument('--fit', dest='fit', action='store',
                    type=int, default=None, help='Number of fits to do after GWB injection')
parser.add_argument('--free_spec_psrs',dest='free_spec_psrs', action='append',
                    type=str,
                    help='Pulsars for which to use a free spectral '
                         'model for analysis. Append with a flag for each pulsar.')
parser.add_argument('--gamma_dm', dest='gamma_dm', action='store',
                    type=float, default=2.0,
                    help='Chromatic f^-2 Red Noise Spectral Index to Simulate')
parser.add_argument('--gamma_gw', dest='gamma_gw', action='store',
                    type=float, default=None, help='GWB spectral index')
parser.add_argument('--gamma_rn', dest='gamma_rn', action='store',
                    type=float, default=None,
                    help='Achromatic Red Noise spectral index to Simulate')
parser.add_argument('--gwb_bf', dest='gwb_bf', action='store_true',
                    default=False, help='Do a GWB detection run.')
parser.add_argument('--gwb_off', dest='gwb_off', action='store_true',
                    default=False, help='Turn off GWB signal analysis.')
parser.add_argument('--gwb_ul', dest='gwb_ul', action='store_true',
                    default=False, help='Do a GWB upper limit analysis.')
parser.add_argument('--hd', dest='hd', action='store_true',
                    default=False, help='Add HD spatial correlations.')
parser.add_argument('--hot_chain', dest='hot_chain', action='store_true',
                    default=False,
                    help='Use a hot chain with parallel tempering')
parser.add_argument('--logf', dest='logf', action='store_true',
                    default=False,
                    help='Option to use log spaced frequencies in GPs.')
parser.add_argument('--min_tspan', dest='min_tspan', action='store',
                    type=float, default=3.0, help='Minimum timspan of pulsars to include.')
parser.add_argument('--model', dest='model', action='store',
                    type=str, default='model_2a', help='NG11yr Model Name')
parser.add_argument('--model_kwargs_path', dest='model_kwargs_path',
                    action='store',
                    type=str, default=None, help='kwargs for model selection')
parser.add_argument('--model_wts', dest='model_wts', action='append',
                    type=float, default=None,
                    help='Log Model weights for model selection. Append one per model.')
parser.add_argument('--nfreqs', dest='nfreqs', action='store',
                    type=int, default=30, help='Number of Freqs')
parser.add_argument('--n_crnfreqs', dest='n_crnfreqs', action='store',
                    type=int, default=30, help='Number of Freqs in CRN model')
parser.add_argument('--n_gwbfreqs', dest='n_gwbfreqs', action='store',
                    type=int, default=30, help='Number of Freqs in GW model')
parser.add_argument('--nyears', dest='nyears', action='store',
                    type=float, default=None,
                    help='Number of years of data to include')
parser.add_argument('--niter', dest='niter', action='store',
                    type=int, default=1001000, help='Number of iterations')
parser.add_argument('--noisepath', dest='noisepath', action='store',
                    default=None, help='Path to noise file or files.')
parser.add_argument('--obs', dest='obs', action='store',
                    type=str, default='all', help='Observatory')
parser.add_argument('--orf', dest='orf', action='store',
                    type=str, default='hd',
                    help='Overlap Reduction Function to use as spatial correlations.')
parser.add_argument('--outdir', dest='outdir', action='store', type=str,
                    default='./', help='Path directory for output.')
parser.add_argument('--outtimdir', dest='outtimdir', action='store', type=str,
                    default='./fake_tim_files',
                    help='Output Directory for tim files.')
parser.add_argument('--outpath', dest='outpath', action='store', type=str,
                    default='./os.txt',
                    help='Path name (dir and filename) for output.')
parser.add_argument('--pardir', dest='pardir', action='store', type=str,
                    default=pardir, help='par file Directory')
parser.add_argument('--parpath', dest='parpath', action='store', type=str,
                    default=None, help='par file path')
parser.add_argument('--pardir_no_dmx', dest='pardir_no_dmx', action='store',
                    type=str, default=pardir_no_dmx, help='Output Directory')
parser.add_argument('--pickle', dest='pickle', action='store',
                    type=str, default='no_pickle',
                    help='Option to give a pickled enterprise list of pulsars')
parser.add_argument('--process', dest='process', action='store',
                    type=int, help='Condor process number')
parser.add_argument('--psd', dest='psd', action='store',
                    type=str, default='powerlaw',
                    help='Name of power spectral density prior fcn.')
parser.add_argument('--psd_crn', dest='psd_crn', action='store',
                    type=str, default='powerlaw',
                    help='Name of CRN power spectral density prior fcn.')
parser.add_argument('--psd_gw', dest='psd_gw', action='store',
                    type=str, default='powerlaw',
                    help='Name of GW power spectral density prior fcn.')
parser.add_argument('--pshift', dest='pshift', action='store_true',
                    default=False, help='Turn on phase shifts.')
parser.add_argument('--psr', dest='psr', action='store',
                    type=str, default=None,
                    help='Name of an individual pulsar. Used for noise runs.')
parser.add_argument('--psr_list', dest='psr_list', action='append',
                    default=None,
                    help='List pulsar names to use in analysis.')
parser.add_argument('--pta_pkl', dest='pta_pkl', action='store',
                    type=str, default='./',
                    help='Path for saving full PTA pickle using cloudpickle.')
parser.add_argument('--rednoise', dest='rednoise',
                    action='store_true', default=False,
                    help='Whether to use a red noise model.')
parser.add_argument('--rm_psrs', dest='rm_psrs', action='append',
                    type=str,
                    default=None, help='Append to list of pulsars to remove.')
parser.add_argument('--rn_psrs',dest='rn_psrs', default='all',
                    type=lambda s: [str(item) for item in s.split(',')],
                    help='List of psrs to use achrom rn models on.')
parser.add_argument('--single_swgp_log10_ell', dest='single_swgp_log10_ell',
                    action='store_true', default=False,
                    help='Whether to use a single swgp log10_ell parameter.')
parser.add_argument('--sky_scramble', dest='sky_scramble',
                    action='store', default=None, type=str,
                    help='Path to sky scramble sky positions.')
parser.add_argument('--spat_corr_info', dest='spat_corr_info',
                    action='store_true', default=False,
                    help='Whether to write out the spatial correlation information')
parser.add_argument('--start_time', dest='start_time', action='store',
                    type=float, default=None, help='Start MJD for slicing. In days.')
parser.add_argument('--sw_gp_mono_gp', dest='sw_gp_mono_gp',
                    action='store_true', default=False,
                    help='Whether to use a full monopole GP across the PTA.')
parser.add_argument('--sw_pta_gp', dest='sw_pta_gp',
                    action='store_true', default=False,
                    help='Whether to use a full GP across the PTA.')
parser.add_argument('--sw_r2p', dest='sw_r2p',
                    type=lambda s: [[float(item) for item in l.split(',')]
                                     for l in s.split(';')],
                    help='Power(s) of solar wind model to use. '
                         'If list of two floats is given these will become a uniform prior.'
                         'Add as string with commas separating range and semicolons'
                         'separating different terms, eg. \"2;3,5;14,18\"')
parser.add_argument('--sw_r2p_ranges', dest='sw_r2p_ranges',
                    type=lambda s: [[float(item) for item in l.split(',')]
                                     for l in s.split(';')],
                    help='Amplitude for extra terms in solar wind.'
                         'Range for log uniform prior provided as two item list.'
                         'Add as string with commas separating range and semicolons'
                         'separating different terms, eg. \"2;-10,-2;-20,-8\"')
parser.add_argument('--sw_r4p4', dest='sw_r4p4',
                    action='store_true', default=False,
                    help='Whether to use the 1/r^4.4 in the sw model.')
parser.add_argument('--timdir', dest='timdir', action='store', type=str,
                    default=timdir, help='Tim file Directory')
parser.add_argument('--timpath', dest='timpath', action='store', type=str,
                    default=None, help='Tim file path')
parser.add_argument('--truncate_psr', dest='truncate_psr', action='append',
                    type=str,
                    default=None, help='Append to list of pulsars to truncate')
parser.add_argument('--truncate_mjd', dest='truncate_mjd', action='append',
                    type=str, default=None,
                    help='Append to list of mjds to truncate.')
parser.add_argument('--tspan', dest='tspan', action='store',
                    type=float, default=None,
                    help='Timespan to use for GP frequencies.')
parser.add_argument('--use_pint', dest='use_pint',
                    action='store_true', default=False,
                    help='Whether to use PINT as the timing package.')
parser.add_argument('--vary_dm_params', dest='vary_dm_params', action='store_false',
                    default=True, help='Option to vary the DM model parameters.')
parser.add_argument('--vary_gamma', dest='vary_gamma', action='store_true',
                    default=False,
                    help='Option to vary gamma, the spectral index '
                         'on the common red noise process.')
parser.add_argument('--wideband', dest='wideband', action='store_true',
                    default=False, help='Option to use wideband data.')
parser.add_argument('--writeHotChains', dest='writeHotChains',
                    action='store_true', default=False,
                    help='Option to use write hot chains with parallel tempering.')
# parse arguments
args = parser.parse_args()

if args.psr_list is None:
    pass
elif len(args.psr_list)==1:
    args.psr_list = np.loadtxt(args.psr_list[0], dtype='S25').astype('U25')

def arguments():
    return args

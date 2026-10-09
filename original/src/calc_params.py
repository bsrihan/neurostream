#! /usr/bin/env python
"""Estimate thresholds and re-reference weights from a real NSX recording.

This is the offline setup step, not the millisecond loop. It reads a
Blackrock ``.ns6`` file and writes a JSON file that ``baseline.ipynb``
loads. The online processor applies those saved numbers. It does not
recompute them.

What it estimates:

- Noise on each electrode, after an optional 250-5000 Hz filter. The
  threshold is a multiple of that root-mean-square voltage. The usual
  multiple is ``-4.5``. An electrode that never varies gets a threshold
  of ``-1e6`` so it never counts as a spike.
- Re-reference weights. A common average uses an equal mix inside each
  electrode group. A linear regression reference fits a mix for each
  electrode from the others in its group. ``-d 60`` uses only the first
  60 seconds, so a long file is not loaded twice.

Usage:
    python calc_params.py -f file.nsx
    python calc_params.py -f file.nsx -o output.json -t -3.5 --reref lrr
    python calc_params.py -f file.nsx -d 60 --no_filter
"""

import argparse
import json
import logging
import numbers
import os
import signal
import sys

import brpylib
import numba
import numpy as np
import scipy.signal
from joblib import Parallel, delayed
from utils import plot_txpanel

###############################################
# Function definitions
###############################################


def common_average_reference(data, group_list):
    """Subtract each group's average from every electrode in that group.

    Electrodes in a group often hear the same noise. Removing the average
    leaves the part that differed across electrodes. ``data`` is shaped
    ``(electrodes, time)`` and is modified in place.
    """
    for g in group_list:
        data[g, :] -= data[g, :].mean(axis=0, keepdims=True)


###############################################
# Initialize script
###############################################

ap = argparse.ArgumentParser(
    description=
    'Calculate voltage thresholds and normalization parameters from NSX file')
ap.add_argument('-f',
                '--nsx_file',
                type=str,
                required=True,
                help='Path to NSX file to process')
ap.add_argument('-o',
                '--output',
                type=str,
                required=False,
                help='Output JSON file path (default: thresh_norm.json)')
ap.add_argument('-n',
                '--nickname',
                type=str,
                required=False,
                default='calc_params',
                help='Nickname for logging (default: calc_params)')
ap.add_argument(
    '-u',
    '--unshuffle_file',
    type=str,
    required=False,
    help='Path to JSON file containing electrode mapping for unshuffling')
ap.add_argument(
    '-t',
    '--thresh_mult',
    type=float,
    required=False,
    default=-4.5,
    help='RMS multiplier for threshold calculation (default: -4.5)')
ap.add_argument('-d',
                '--data_time_s',
                type=float,
                required=False,
                default=None,
                help='Time window in seconds to process (default: all data)')
ap.add_argument('--no_filter', action='store_true', help='Skip filtering step')
ap.add_argument('--causal',
                action='store_true',
                help='Use causal filtering instead of acausal')
ap.add_argument('--reref',
                type=str,
                required=False,
                default='car',
                help='Rereference type (default: car)')
ap.add_argument('--plot_spike_panel',
                action='store_true',
                help='Plot spike panel')
args = ap.parse_args()

NSX_FILE = args.nsx_file
OUTPUT_FILE = args.output if args.output else 'thresh_norm.json'
NAME = args.nickname

loglevel = 'INFO'
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(format=f'[{NAME}] %(levelname)s: %(message)s',
                    level=numeric_level,
                    stream=sys.stdout)


def signal_handler(sig, frame):  # setup the clean exit code with a warning
    logging.info('SIGINT received. Exiting...')
    sys.exit(1)


# place the sigint signal handler
signal.signal(signal.SIGINT, signal_handler)

###############################################
# Load NSX file and set up default parameters
###############################################
try:
    logging.info(f"Loading NSX file: {NSX_FILE}")
    nsx_file = brpylib.NsxFile(NSX_FILE)
    cont_data = nsx_file.getdata()
    logging.info('NSX file loaded successfully.')
except Exception as e:
    logging.error(f"Error loading NSX file: {e}")
    sys.exit(1)

# Set up parameters from command line arguments and defaults
graph_params = {
    'thresh_mult': args.thresh_mult,
    'filter_first': not args.no_filter,
    'causal': args.causal,
    'butter_order': 4,
    'butter_lowercut': 250,
    'butter_uppercut': 5000,
    'samp_freq': 30000,
    'decimate': 1,
    'data_time_s': args.data_time_s,
    'rereference': args.reref,
    'reref_group_sizes': 64,
    'exclude_channels': [],
    'plot_spike_panel': args.plot_spike_panel,
}

###############################################
# Extract data from NSX file
###############################################

# Get data from the first (and typically only) data segment
if len(cont_data['data']) == 0:
    logging.error("No data found in NSX file")
    sys.exit(1)

# Extract the data array (channels x samples)
all_data = cont_data['data'][0].astype(np.float64)

# Get sampling frequency from NSX file
sample_rate = (nsx_file.basic_header['SampleResolution'] /
               nsx_file.basic_header['Period'])
graph_params['samp_freq'] = sample_rate

# Get number of channels and samples
n_channels, n_samples = all_data.shape
ch_per_stream = [n_channels]  # Single stream for NSX files

logging.info(
    f'Loaded {n_channels} channels with {n_samples} samples at {sample_rate} Hz'
)

# Set up unshuffle matrix (identity by default for NSX files)
unshuffle_matrix = np.eye(n_channels, dtype=np.float64)
unshuffle = False

# Check if unshuffle file is provided via command line
if hasattr(args, 'unshuffle_file') and args.unshuffle_file:
    try:
        with open(args.unshuffle_file, 'r') as f:
            unshuffle_dict = json.load(f)
        logging.info(
            f'Array index unshuffle dict loaded from file: {args.unshuffle_file}'
        )
        unshuffle = True

        # Build unshuffling matrix
        unshuffle_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
        electrode_mapping = np.array(unshuffle_dict['electrode_mapping'])
        for chan_in in range(n_channels):
            chan_out = electrode_mapping[chan_in] - 1
            unshuffle_matrix[chan_out, chan_in] = 1
    except Exception as e:
        logging.warning(
            f'Could not load unshuffle file: {e}. Using identity matrix.')
        unshuffle = False

# the RMS multiplier to use to calculate voltage thresholds
thresh_mult = graph_params.get('thresh_mult', -4.5)

# Use all channels by default for NSX files
ch_mask = np.arange(n_channels, dtype=np.uint16)

# whether to filter the data in 'input_stream_name' before calculating thresholds
filter_first = graph_params.get('filter_first', True)

# whether to causally or acausally filter
causal = graph_params.get('causal', False)

# filter order
butter_order = graph_params.get('butter_order', 4)

# filter lower cutoff
butter_lowercut = graph_params.get('butter_lowercut', None)

# filter upper cutoff
butter_uppercut = graph_params.get('butter_uppercut', None)

# sampling frequency
samp_freq = graph_params.get('samp_freq', 30000)

# decimation
decimate = graph_params.get('decimate', 1)

# amount of data to process in seconds
data_time_s = graph_params.get('data_time_s', None)

# re-reference type
reref = graph_params.get('rereference', 'car').lower()

# Set up re-reference groupings for NSX files
if reref is not None:
    reref_sizes = graph_params.get('reref_group_sizes', n_channels)
    if not isinstance(reref_sizes, list):
        if isinstance(reref_sizes, int):
            # Create groups of the specified size
            reref_groups = []
            ch_count = 0
            while ch_count < n_channels:
                group_size = min(reref_sizes, n_channels - ch_count)
                reref_groups.append(
                    np.arange(ch_count, ch_count + group_size).tolist())
                ch_count += group_size
        else:
            logging.error(
                f'reref_group_sizes must be an int or list, got {type(reref_sizes)}'
            )
            sys.exit(1)
    else:
        # Use provided list of group sizes
        reref_groups = []
        ch_count = 0
        for g in reref_sizes:
            if not isinstance(g, int):
                logging.error(
                    f'All group sizes must be integers, got {type(g)}')
                sys.exit(1)
            reref_groups.append(np.arange(ch_count, ch_count + g).tolist())
            ch_count += g
else:
    reref_groups = []

# exclude channels
if 'exclude_channels' in graph_params:
    exclude_ch = graph_params['exclude_channels']
    if not isinstance(exclude_ch, list):
        if isinstance(exclude_ch, int):
            exclude_ch = [exclude_ch]
    for c in exclude_ch:
        if not isinstance(c, int):
            logging.error(
                f'\'exclude_channels\' must be a list of \'int\'s or a single \'int\', but {graph_params["exclude_channels"]} was given. Exiting'
            )
            sys.exit(1)
    for c in exclude_ch:
        for g in reref_groups:
            if c in g:
                g.remove(c)

# keep only masked channels
for g_idx in range(len(reref_groups)):
    reref_groups[g_idx] = list(
        set(reref_groups[g_idx]).intersection(set(ch_mask)))

# spike panel plotting parameters
plot_spike_panel = graph_params.get('plot_spike_panel', False)

# spike panel plotting configuration
if plot_spike_panel:
    # Convert time_s to max_samples if specified
    time_s = graph_params.get('spike_panel_time_s', 10)
    max_samples = None
    if time_s is not None:
        if not isinstance(time_s, numbers.Number) or time_s <= 0:
            logging.error(
                f'\'spike_panel_time_s\' must be a positive number, but it was {time_s}. Exiting'
            )
            sys.exit(1)
        max_samples = int(time_s * samp_freq)
        logging.info(
            f'Spike panel will use first {time_s} seconds ({max_samples} samples) of data'
        )

    spike_panel_params = {
        'pre_cross':
        graph_params.get('spike_pre_cross', 8),
        'post_cross':
        graph_params.get('spike_post_cross', 30),
        'spike_window_samples':
        graph_params.get('spike_window_samples', 90),
        'max_spikes_to_plot':
        graph_params.get('max_spikes_to_plot', 100),
        'ylim':
        graph_params.get('spike_ylim', (-500, 500)),
        'figsize_per_channel':
        graph_params.get('spike_figsize_per_channel', (2, 2)),
        'wb_on_right':
        graph_params.get('spike_wb_on_right', False),
        'save_path':
        graph_params.get('spike_panel_save_path', None),
        'max_samples':
        max_samples
    }
###############################################
# Process data time window
###############################################

# Limit data to specified time window if requested
if data_time_s is not None:
    data_time_samples = int(data_time_s * samp_freq)
    if n_samples < data_time_samples:
        logging.error(
            f'Not enough samples in data to process {data_time_s} seconds '
            f'(only {n_samples} samples available, need {data_time_samples}), exiting'
        )
        sys.exit(1)
    n_samples = data_time_samples
    all_data = all_data[:, :n_samples]

if n_samples == 0:
    logging.info('No samples to process, exiting')
    sys.exit(0)

logging.info(f'Processing {all_data.shape[1]} samples of data')

# unshuffle data
all_data = unshuffle_matrix @ all_data

# check for nonvarying channels
nonvarying_ch = np.argwhere(np.all(all_data[:, 1:] == all_data[:, :-1],
                                   axis=1)).flatten()
if len(nonvarying_ch) > 0:
    logging.warning(
        f'Found nonvarying channels (unshuffled): {nonvarying_ch.tolist()}')

# remove nonvarying channels from rereference groups
for g_idx in range(len(reref_groups)):
    reref_groups[g_idx] = list(
        set(reref_groups[g_idx]).difference(set(nonvarying_ch)))

###############################################
# Filter if requested
###############################################

if filter_first:
    if butter_lowercut and butter_uppercut:
        filt_type = 'bandpass'
        Wn = [butter_lowercut, butter_uppercut]
    elif butter_uppercut:
        filt_type = 'lowpass'
        Wn = butter_uppercut
    elif butter_lowercut:
        filt_type = 'highpass'
        Wn = butter_lowercut
    else:
        logging.error(
            f'Either butter low cutoff or high cutoff must be defined. Exiting'
        )
        sys.exit(1)

    sos = scipy.signal.butter(butter_order,
                              Wn,
                              btype=filt_type,
                              analog=False,
                              output='sos',
                              fs=samp_freq)  # set up a filter

    # initialize the state of the filter
    zi_flat = scipy.signal.sosfilt_zi(sos)
    # so that we have the right number of dimensions
    zi = np.zeros((zi_flat.shape[0], np.sum(ch_per_stream), zi_flat.shape[1]))
    # filter initialization
    for ii in range(np.sum(ch_per_stream)):
        zi[:, ii, :] = zi_flat

    # log the filter info
    causal_str = 'causal' if causal else 'acausal'
    message = (f'Using {butter_order :d} order, '
               f'{Wn} hz {filt_type} {causal_str} filter')
    message += f' with {reref.upper()}' if reref is not None else ''
    logging.info(message)

    if causal:
        all_data = scipy.signal.sosfilt(sos, all_data, axis=1, zi=zi)
    else:
        all_data = scipy.signal.sosfiltfilt(sos, all_data, axis=1)

logging.debug('Finished filtering')
###############################################
# Compute rereferencing parameters
###############################################


@numba.jit('float64[:,:](float64[:,:], float64[:,:])', nopython=True)
def rereference_data(data, reref_params):
    """Apply ``y = (I - P) x`` for one window.

    ``P`` holds the re-reference weights. Same mix as
    ``utils.rereference_data``. This copy is compiled with numba because
    ``calc_params.py`` calls it while fitting weights.
    """
    data = np.ascontiguousarray(data)
    reref_mat = np.eye(reref_params.shape[0]) - reref_params
    reref_data = reref_mat @ data
    return reref_data


def calc_lrr_params_parallel(channel, group, decimate=1):
    """
    Calculate parameters for linear regression reference. This version is
    made to be used with multiprocessing.

    Parameters
    ----------
    channel : int
        Index of the channel for which the reference is being computed
    group : array-like of shape (n_group_channels,)
        List of channels to use for the referencing
    decimate : int, optional
        Factor by which to decimate the data, by default 1 (no decimation)

    Returns
    -------
    channel : int
        Index of the channel for which the reference was computed
    group : array-like of shape (n_ref_channels,)
        List of channels used for the referencing
    params : numpy.ndarray of shape (n_ref_channels,)
        Weights of each channel to use when rereferencing
    """
    grp = np.setdiff1d(group, channel)

    # ignore nonvarying channels
    if channel in nonvarying_ch:
        return channel, grp, np.zeros(len(grp))

    X = all_data[grp, ::decimate].T
    y = all_data[channel, ::decimate].reshape(1, -1)
    params = np.linalg.solve(X.T @ X, X.T @ y.T).T

    return channel, grp, params


reref_params = np.zeros((n_channels, n_channels), dtype=np.float64)
if reref == 'car':
    for g in reref_groups:
        if len(g) > 0:  # Only process non-empty groups
            # np.ix_ fills the whole group block; [g, g] would set only
            # the diagonal and leave the common average out.
            reref_params[np.ix_(g, g)] = 1. / len(g)

elif reref == 'lrr':
    # use single-precision for faster compute
    all_data = all_data.astype(np.float32)
    with Parallel(n_jobs=-1, require='sharedmem') as parallel:
        # loop through the groups and compute LRR for each one
        for g in reref_groups:
            if len(g) > 0:  # Only process non-empty groups
                # compute the LRR parameters for each channel in this group
                tasks = [
                    delayed(calc_lrr_params_parallel)(channel=ch, group=g)
                    for ch in g
                ]
                lrr_params = parallel(tasks)
                # unpack the parallel execution results - assign the LRR parameters
                # to the reref_params array
                for item in lrr_params:
                    ch, grp, output = item
                    reref_params[ch, grp] = output
    all_data = all_data.astype(np.float64)

# Re-reference the data
all_data = rereference_data(all_data, reref_params)
logging.debug('Finished rereferencing')

###############################################
# Compute thresholds
###############################################

thresholds = (thresh_mult *
              np.sqrt(np.mean(np.square(all_data), axis=1))).reshape(-1, 1)
thresholds[nonvarying_ch] = -1e6  # so crossings are never triggered

###############################################
# Generate spike panel plot if requested
###############################################

if plot_spike_panel:
    logging.info('Generating spike panel plots for each channel group')

    # Determine base save path if not specified
    base_save_path = os.path.splitext(OUTPUT_FILE)[0] + '_spike_panel'

    # Ensure directory exists
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)

    # Generate separate plots for each re-referencing group
    for group_idx, group_channels in enumerate(reref_groups):
        if len(group_channels) == 0:
            logging.warning(f'Skipping empty re-referencing group {group_idx}')
            continue

        # Create group-specific title and save path
        group_title = (f'Group {group_idx + 1} '
                       f'(ch {group_channels[0]+1}-{group_channels[-1]+1})')
        group_save_path = f'{base_save_path}_group_{group_idx + 1}.png'

        logging.info(f'Generating spike panel for group {group_idx + 1} '
                     f'with {len(group_channels)} channels...')

        # Generate the plot for this group
        plot_txpanel(
            all_data,
            title=group_title,
            thresh_mult=thresh_mult,
            pre_cross=spike_panel_params['pre_cross'],
            post_cross=spike_panel_params['post_cross'],
            spike_window_samples=spike_panel_params['spike_window_samples'],
            max_spikes_to_plot=spike_panel_params['max_spikes_to_plot'],
            ylim=spike_panel_params['ylim'],
            figsize_per_channel=spike_panel_params['figsize_per_channel'],
            wb_on_right=spike_panel_params['wb_on_right'],
            save_path=group_save_path,
            channels=group_channels,
            max_samples=spike_panel_params['max_samples'])

    logging.info(f'Spike panel plot generation completed. '
                 f'Generated {len(reref_groups)} plots.')

###############################################
# Save parameters to JSON file
###############################################

output_dict = {
    'nsx_file': NSX_FILE,
    'n_channels': n_channels,
    'n_samples': n_samples,
    'sample_rate': sample_rate,
    'thresh_mult': thresh_mult,
    'filter_first': filter_first
}

if filter_first:
    output_dict['causal'] = causal
    output_dict['butter_order'] = butter_order
    output_dict['butter_passband'] = Wn
    output_dict['samp_freq'] = samp_freq

output_dict['rereference'] = reref
if reref:
    output_dict['reref_groups'] = reref_groups

output_dict['thresholds'] = thresholds.reshape(-1).tolist()
output_dict['rereference_parameters'] = reref_params.tolist()
output_dict['channel_unshuffling'] = unshuffle_matrix.tolist()

# Create output directory if it doesn't exist
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the file
logging.info(f'Saving parameters to {OUTPUT_FILE}')
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_dict, f, indent=2, sort_keys=False)

logging.info('Parameter calculation completed successfully!')

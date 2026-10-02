import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def get_filter_func(causal=False, use_fir=True):
    """Return the baseline millisecond filter.

    The forward pass is a Butterworth second-order-section filter
    (``scipy.signal.sosfilt``), causal in time, with delay state carried
    from one millisecond to the next. When ``causal`` is false, a reverse
    pass cancels that phase. ``use_fir`` selects the reverse pass used by
    the notebook: the impulse response of the same Butterworth, applied as
    a valid convolution over the 4 ms look-back. The other reverse pass is
    a second IIR run on the time-reversed buffer.

    Parameters
    ----------
    causal : bool
        When true, return only the forward IIR. When false, return the
        forward-then-reverse filter.
    use_fir : bool
        When true and ``causal`` is false, the reverse pass is an FIR.
    """

    def causal_filter(data, filt_data, sos, zi):
        """
        causal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        """
        filt_data[:, :], zi[:, :] = scipy.signal.sosfilt(sos,
                                                         data,
                                                         axis=1,
                                                         zi=zi)

    def acausal_filter(data,
                       filt_data,
                       rev_buffer,
                       sos,
                       zi,
                       rev_win=None,
                       rev_zi=None):
        """
        acausal filtering

        Parameters
        ----------
        data : array_like
            An N-dimensional input array.
        filt_data : ndarray
            Array to store the output of the digital filter.
        rev_buffer : ndarray
            Array to store the output of the forward IIR filter.
        sos : array_like
            Array of second-order filter coefficients
        zi : array_like
            Initial conditions for the cascaded filter delays
        rev_win : array-like, optional
            Coefficients of the reverse FIR filter
        rev_zi : array-like, optional
            Steady-state conditions of the reverse filter
        """
        # shift the buffer
        n_samp = data.shape[1]
        rev_buffer[:, :-n_samp] = rev_buffer[:, n_samp:]

        # run the forward pass filter
        rev_buffer[:, -n_samp:], zi[:, :] = scipy.signal.sosfilt(sos,
                                                                 data,
                                                                 axis=1,
                                                                 zi=zi)
        # run the backward pass filter
        # 1. pass in the reversed buffer
        # 2. get the last N samples of the filter output
        # 3. reverse the output when saving to filt_data
        if use_fir:
            for ii in range(filt_data.shape[0]):
                filt_data[ii, ::-1] = np.convolve(rev_buffer[ii, ::-1],
                                                  rev_win, 'valid')
        else:
            ic = rev_zi * filt_data[:, -1][None, :, None]
            filt_data[:, ::-1] = scipy.signal.sosfilt(sos,
                                                      rev_buffer[:, ::-1],
                                                      axis=1,
                                                      zi=ic)[0][:, -n_samp:]

    filter_func = causal_filter if causal else acausal_filter

    return filter_func


def build_filter(but_order=4,
                 but_low=250,
                 but_high=5000,
                 acausal_filter_type='fir',
                 causal=False,
                 acausal_filter_lag=120,
                 fs=30000,
                 n_channels=64):
    """Design the spike-band filter and its per-channel delay state.

    Default band is 250-5000 Hz, Butterworth order 4, at ``fs`` samples
    per second. That is the spike band used for threshold crossings
    (Masse et al., 2014, J. Neural Eng.) and for spike-band power
    (Nason et al., 2020, J. Neural Eng.). Both cutoffs set a band-pass.
    One cutoff sets a high-pass or a low-pass.

    The forward filter is always an IIR in second-order sections. Its
    initial state is ``sosfilt_zi`` copied onto every channel, which is
    the steady state for a constant input of 1, not a zero buffer.
    Acausal mode also builds the reverse FIR: the response of that IIR
    to an impulse ``acausal_filter_lag + 1`` samples long. At 30 kHz a
    lag of 120 samples is 4 ms of look-back.

    Parameters
    ----------
    but_order : int
        Butterworth order.
    but_low, but_high : float
        Cutoffs in Hz. Zero or None drops that edge.
    acausal_filter_type : str
        ``"fir"`` (notebook default) or anything else for an IIR reverse pass.
    causal : bool
        Skip the reverse pass when true.
    acausal_filter_lag : int
        Reverse-filter length, in samples, minus one.
    fs : float
        Sampling rate in Hz.
    n_channels : int
        How many copies of the delay state to allocate.
    """
    # determine filter type
    if but_low and but_high:
        filt_type = 'bandpass'
        Wn = [but_low, but_high]
    elif but_high:
        filt_type = 'lowpass'
        Wn = but_high
    elif but_low:
        filt_type = 'highpass'
        Wn = but_low
    else:
        raise ValueError("Must specify 'butter_lowercut' or 'butter_uppercut'")

    # set up filter
    sos = scipy.signal.butter(but_order,
                              Wn,
                              btype=filt_type,
                              analog=False,
                              output='sos',
                              fs=fs)  # set up a filter
    # initialize the state of the filter
    zi_flat = scipy.signal.sosfilt_zi(sos)
    # so that we have the right number of dimensions
    zi = np.zeros((zi_flat.shape[0], n_channels, zi_flat.shape[1]))
    # filter initialization
    for ii in range(n_channels):
        zi[:, ii, :] = zi_flat

    # select the filtering function
    if acausal_filter_type and acausal_filter_type.lower() == 'fir':
        use_fir = True
    else:
        use_fir = False
    filter_func = get_filter_func(causal, use_fir=use_fir)

    # log the filter info
    causal_str = 'causal' if causal else 'acausal'
    message = (f'Loading {but_order :d} order, '
               f'{Wn} hz {filt_type} {causal_str}')
    if causal:
        message += ' IIR filter'
    elif use_fir:
        message += ' IIR-FIR filter'
    else:
        message += ' IIR-IIR filter'
    print(message)

    if not causal:
        if use_fir:
            # FIR filter (backward)
            N = acausal_filter_lag + 1  # length of the filter
            imp = scipy.signal.unit_impulse(N)
            rev_win = scipy.signal.sosfilt(sos, imp)
            # filter initialization
            rev_zi_flat = scipy.signal.lfilter_zi(rev_win, 1.0)
            rev_zi = np.zeros((n_channels, rev_zi_flat.shape[0]))
            for ii in range(n_channels):
                rev_zi[ii, :] = rev_zi_flat
        else:
            rev_win = None
            rev_zi = zi.copy()

    if causal:
        return filter_func, sos, zi
    else:
        return filter_func, sos, zi, rev_win, rev_zi


def rereference_data(data, reref_params):
    """Re-reference one window.

    ``reref_params`` is ``P`` in the mixing ``y = (I - P) x``. A common
    average puts ``1 / n`` in every entry of a group's block. A linear
    regression reference puts the fitted weights there instead. Channels
    outside a group stay zero in ``P``, so they are not mixed. ``data``
    is shaped ``(n_channels, n_samples)`` and is in the same units as the
    recording (microvolts, for an NSX file).
    """
    data = np.ascontiguousarray(data)
    reref_mat = np.eye(reref_params.shape[0]) - reref_params
    reref_data = reref_mat @ data
    return reref_data


def plot_txpanel(filt_data,
                 title=None,
                 thresh_mult=-4.5,
                 thresholds=None,
                 pre_cross=8,
                 post_cross=30,
                 spike_window_samples=90,
                 max_spikes_to_plot=100,
                 ylim=(-250, 250),
                 figsize_per_channel=(2, 2),
                 wb_on_right=False,
                 save_path=None,
                 channels=None,
                 max_samples=None):
    """
    Plot threshold crossings for neural data as a spike panel.
    
    Parameters
    ----------
    filt_data : numpy.ndarray
        Filtered neural data with shape (n_channels, n_samples)
    title : str, optional
        Title for the plot. Defaults to None.
    thresh_mult : float, optional
        Multiplier for RMS to set threshold. Defaults to -4.5.
    thresholds : numpy.ndarray, optional
        Thresholds for each channel. If None, calculated from RMS.
    pre_cross : int, optional
        Number of samples before threshold crossing to plot. Defaults to 8.
    post_cross : int, optional
        Number of samples after threshold crossing to plot. Defaults to 30.
    spike_window_samples : int, optional
        Minimum samples between spikes to avoid duplicates. Defaults to 90.
    max_spikes_to_plot : int, optional
        Maximum number of spikes to plot per channel. Defaults to 100.
    ylim : tuple, optional
        Y-axis limits for plots. Defaults to (-250, 250).
    figsize_per_channel : tuple, optional
        Figure size per channel in inches. Defaults to (2, 2).
    wb_on_right : bool, optional
        Whether to orient the plot so that the wire bundle is on the right.
        Defaults to False.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed. Defaults to None.
    channels : list, optional
        List of channel indices to plot. If None, plots all channels.
        Defaults to None.
    max_samples : int, optional
        Maximum number of samples to use for spike panel. If None, uses all data.
        Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure, numpy.ndarray
        Figure and axes objects
    """
    # Select channels to plot
    if channels is None:
        channels = list(range(filt_data.shape[0]))

    # Filter data to selected channels
    group_data = filt_data[channels, :]

    # Limit data to specified number of samples if requested
    if max_samples is not None and max_samples < group_data.shape[1]:
        group_data = group_data[:, :max_samples]

    if thresholds is None:
        # Calculate RMS and thresholds for selected channels
        rms = np.sqrt(np.mean(np.square(group_data), axis=1))
        thresholds = rms * thresh_mult

    n_channels = len(channels)
    ncols = int(np.sqrt(n_channels))
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(ncols=ncols,
                             nrows=nrows,
                             figsize=(figsize_per_channel[0] * ncols,
                                      figsize_per_channel[1] * nrows),
                             squeeze=False)

    if title is not None:
        axes[0, 0].set_title(title, fontsize=12, loc='left')

    if wb_on_right:
        axes = np.fliplr(axes)
        axes = np.flipud(axes)
        axes = axes.T

    for i in range(n_channels):
        y = group_data[i, :]
        threshold = thresholds[i]

        # Find threshold crossings (negative-going)
        spike_idxs = np.where(np.diff(np.sign(y - threshold)) == -2)[0]

        # Remove spikes that are too close to previous spikes
        if len(spike_idxs) > 1:
            spike_idxs = np.delete(
                spike_idxs,
                np.where(np.diff(spike_idxs) < spike_window_samples)[0] + 1)

        # Limit number of spikes to plot for performance
        if len(spike_idxs) > max_spikes_to_plot:
            spike_idxs = spike_idxs[:max_spikes_to_plot]

        ax = axes.flatten()[i]

        # Plot vertical line at threshold crossing point
        ax.axvline(pre_cross, color='k', alpha=0.25, linestyle='--')

        # Plot spike waveforms
        for spike_idx in spike_idxs:
            start_idx = max(0, spike_idx - pre_cross)
            end_idx = min(len(y), spike_idx + post_cross)
            waveform = y[start_idx:end_idx]
            x_vals = np.arange(len(waveform))

            # Adjust x_vals to center the spike at pre_cross
            if spike_idx >= pre_cross:
                x_vals = x_vals - (spike_idx - start_idx - pre_cross)

            ax.plot(x_vals, waveform, 'b', linewidth=0.5, alpha=0.5)

        # Remove plot ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.axis('off')

        # Set axis limits
        ax.set_ylim(ylim)
        ax.set_xlim([0, pre_cross + post_cross])

        # Add channel number in upper right corner
        ch_text = f'ch {channels[i] + 1}'
        ax.text(0.9,
                0.9,
                ch_text,
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes,
                color='k',
                fontsize=8)

    # Remove empty subplots
    for i in range(n_channels, len(axes.flat)):
        axes.flat[i].remove()

    plt.tight_layout()

    # Save or show the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f'Spike panel plot saved to {save_path}')
    else:
        plt.show()

    return fig, axes

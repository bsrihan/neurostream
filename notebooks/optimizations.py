"""Real-time pipeline optimizations for 30 kHz neural data.

Lossless (same spikes and spike-band power as ``baseline.ipynb``):

- Re-reference each multi-electrode array on its own thread. Arrays are the
  re-referencing groups. Channels in different groups never mix, so the groups
  can run concurrently.
- After re-referencing, filter and extract features for many channels at once.
  Each channel is an independent task. A thread pool runs as many of those
  tasks as there are cores (or one thread per channel, if asked).
- Optional GPU filtering. Apple MPS is preferred, then an integrated CUDA GPU,
  then a discrete CUDA GPU. Those first two share memory with the CPU, so the
  2048-channel buffer does not have to be copied across a bus.
- Spikes are returned as ``(channel, millisecond)`` events and as a CSR matrix,
  which is the form to store or send. The dense raster is still available.

Lossy (opt in with ``decimate=True``):

- Low-pass the raw signal, then keep every other sample (30 kHz -> 15 kHz).
  The cutoff is 0.8 times the new Nyquist (6 kHz), so the spike band still
  passes and energy that would alias is attenuated first. Feature frames stay
  at 1 kHz.
"""

from __future__ import annotations

import contextlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.signal
import scipy.sparse

from utils import build_filter, rereference_data

logger = logging.getLogger(__name__)

# 4 ms acausal tail, in seconds. At 30 kHz this is the notebook's lag of 120
# samples; after 2x decimation it is 60 samples at 15 kHz.
_ACAUSAL_LAG_S = 0.004


def select_gpu_device():
    """Pick a GPU device, preferring unified memory.

    Returns ``(device, kind)`` where ``kind`` is ``"unified"`` or
    ``"discrete"``. Returns ``(None, None)`` when PyTorch is missing or no
    GPU is visible. A CPU-only PyTorch install is not used: the SciPy filter
    is faster there.
    """
    try:
        import torch
    except ImportError:
        return None, None

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "unified"

    if torch.cuda.is_available():
        integrated = False
        try:
            integrated = bool(torch.cuda.get_device_properties(0).is_integrated)
        except (AttributeError, RuntimeError):
            integrated = False
        kind = "unified" if integrated else "discrete"
        return torch.device("cuda"), kind

    return None, None


def spikes_to_events(spikes):
    """Encode a dense ``(channels, milliseconds)`` raster as events.

    Each row is ``(channel, millisecond)`` for a non-zero bin. Most bins are
    zero, so this is the form to write to disk or put on the wire. Packing
    the events is extra work on the real-time path; skip it when the consumer
    wants the dense raster.
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError(
            "spikes must have shape (n_channels, n_milliseconds), "
            f"got {spikes.shape}")
    channels, times = np.nonzero(spikes)
    if channels.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack((channels, times)).astype(np.int32, copy=False)


def spikes_to_sparse(spikes):
    """CSR view of a dense spike raster. Same contents, fewer stored zeros."""
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError(
            "spikes must have shape (n_channels, n_milliseconds), "
            f"got {spikes.shape}")
    return scipy.sparse.csr_matrix(spikes)


def threshold_crossings(filtered, thresholds):
    """Negative-going threshold crossings, matching ``baseline.ipynb``.

    Parameters
    ----------
    filtered : ndarray, shape (n_channels, n_samples)
    thresholds : ndarray, shape (n_channels,) or (n_channels, 1)

    Returns
    -------
    spikes : ndarray, shape (n_channels,), int16
        1 if the channel crossed threshold anywhere in the window.
    crossing_events : ndarray, shape (n_events, 2), int32
        ``(channel, sample_index)`` of each crossing. ``sample_index`` is the
        first sample at or below threshold.
    """
    filtered = np.asarray(filtered)
    thresholds = np.asarray(thresholds).reshape(-1, 1)
    if thresholds.shape[0] != filtered.shape[0]:
        raise ValueError(
            f"{thresholds.shape[0]} thresholds for {filtered.shape[0]} channels"
        )
    below = (filtered[:, 1:] < thresholds) & (filtered[:, :-1] >= thresholds)
    spikes = np.any(below, axis=1).astype(np.int16)
    channels, relative = np.nonzero(below)
    if channels.size == 0:
        events = np.zeros((0, 2), dtype=np.int32)
    else:
        events = np.column_stack((channels, relative + 1)).astype(np.int32,
                                                                  copy=False)
    return spikes, events


def spike_band_power(filtered):
    """Mean log power over the window, in dB. Matches ``baseline.ipynb``."""
    filtered = np.asarray(filtered)
    return (10 * np.log10(np.square(filtered))).mean(axis=1)


class AntiAliasDecimator:
    """Causal low-pass, then keep every other sample.

    ``scipy.signal.decimate`` is not used. The README asks for a low-pass
    followed by dropping alternate samples, and the filter has to stream one
    millisecond at a time. State is carried across calls. An odd trailing
    sample is held until the next call so the kept-sample phase stays aligned
    to the original stream (samples 0, 2, 4, ...).

    From 30 kHz the new Nyquist is 7.5 kHz. The Butterworth cutoff is
    ``cutoff_frac`` times that (6 kHz by default) so the 250-5000 Hz spike
    band is unchanged and content that would fold into it is attenuated.
    """

    def __init__(self,
                 n_channels,
                 sample_rate=30000.0,
                 order=8,
                 cutoff_frac=0.8):
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.n_channels = int(n_channels)
        self.sample_rate = float(sample_rate)
        self.factor = 2
        self.output_rate = self.sample_rate / self.factor
        self.order = int(order)
        self.cutoff_frac = float(cutoff_frac)
        nyquist_out = self.output_rate / 2.0
        self.cutoff = self.cutoff_frac * nyquist_out
        if not (0 < self.cutoff < self.sample_rate / 2.0):
            raise ValueError(
                f"cutoff {self.cutoff} Hz is outside the Nyquist range")

        self.sos = scipy.signal.butter(self.order,
                                       self.cutoff,
                                       btype="lowpass",
                                       analog=False,
                                       output="sos",
                                       fs=self.sample_rate)
        n_sections = self.sos.shape[0]
        # Rest initial conditions. sosfilt_zi assumes a constant input of 1
        # and would put a step into the first samples.
        self.zi = np.zeros((n_sections, self.n_channels, 2), dtype=np.float64)
        self._remainder = np.zeros((self.n_channels, 0), dtype=np.float64)

    def process(self, data):
        """Filter ``data`` of shape ``(n_channels, n_samples)`` and downsample.

        Returns shape ``(n_channels, n_kept)``. ``n_kept`` is half the number
        of samples consumed, which excludes a single held-back odd sample.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(
                f"expected shape ({self.n_channels}, n_samples), got {data.shape}"
            )
        if self._remainder.shape[1]:
            data = np.concatenate((self._remainder, data), axis=1)

        n_samples = data.shape[1]
        n_even = n_samples - (n_samples % self.factor)
        self._remainder = np.ascontiguousarray(data[:, n_even:])
        if n_even == 0:
            return np.zeros((self.n_channels, 0), dtype=np.float64)

        chunk = np.ascontiguousarray(data[:, :n_even])
        filtered, self.zi = scipy.signal.sosfilt(self.sos,
                                                 chunk,
                                                 axis=1,
                                                 zi=self.zi)
        return filtered[:, ::self.factor]


def sosfilt_torch(sos, data, zi):
    """Direct-form-II-transposed SOS filter. Matches ``scipy.signal.sosfilt``.

    ``data`` is ``(n_channels, n_samples)``, ``zi`` is ``(n_sections,
    n_channels, 2)`` and is updated in place. Channels run in parallel; time
    and SOS sections stay sequential because the recurrence requires it.
    """
    y = data
    n_samples = data.shape[1]
    for section in range(sos.shape[0]):
        b0, b1, b2, a0, a1, a2 = (sos[section, k] for k in range(6))
        inv_a0 = 1.0 / a0
        b0 = b0 * inv_a0
        b1 = b1 * inv_a0
        b2 = b2 * inv_a0
        a1 = a1 * inv_a0
        a2 = a2 * inv_a0
        z1 = zi[section, :, 0]
        z2 = zi[section, :, 1]
        out = _empty_like(y)
        for n in range(n_samples):
            xn = y[:, n]
            yn = b0 * xn + z1
            z1_next = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            z1 = z1_next
            out[:, n] = yn
        zi[section, :, 0] = z1
        zi[section, :, 1] = z2
        y = out
    return y, zi


def _empty_like(tensor):
    # Local import so CPU-only callers do not need PyTorch installed.
    import torch
    return torch.empty_like(tensor)


def _blas_limits():
    """Keep BLAS from starting its own threads inside a channel worker."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def _channel_blocks(n_channels, n_workers):
    n_workers = max(1, min(int(n_workers), int(n_channels)))
    edges = np.linspace(0, n_channels, n_workers + 1, dtype=int)
    return [
        slice(int(start), int(stop))
        for start, stop in zip(edges[:-1], edges[1:]) if stop > start
    ]


def _normalize_groups(n_channels, reref_groups):
    if reref_groups is None:
        return [list(range(n_channels))]
    groups = []
    seen = np.zeros(n_channels, dtype=bool)
    for group in reref_groups:
        channels = np.asarray(group, dtype=np.int64).ravel()
        if channels.size == 0:
            continue
        if channels.min() < 0 or channels.max() >= n_channels:
            raise ValueError(
                f"channel index {channels.tolist()} outside 0..{n_channels - 1}"
            )
        if seen[channels].any():
            raise ValueError(f"re-referencing groups overlap at {channels.tolist()}")
        seen[channels] = True
        groups.append(channels.tolist())
    missing = np.flatnonzero(~seen).tolist()
    if missing:
        groups.append(missing)
    if not groups:
        groups = [list(range(n_channels))]
    return groups


def _groups_are_independent(reref_params, groups):
    """True when no re-reference weight crosses between groups."""
    n_channels = reref_params.shape[0]
    for group in groups:
        idx = np.asarray(group, dtype=np.int64)
        outside = np.ones(n_channels, dtype=bool)
        outside[idx] = False
        if not np.any(outside):
            continue
        if np.any(reref_params[np.ix_(idx, np.flatnonzero(outside))]):
            return False
    return True


@dataclass
class WindowResult:
    """One millisecond of pipeline output."""

    rereferenced: np.ndarray
    filtered: np.ndarray
    spikes: np.ndarray
    spike_band_power: np.ndarray
    crossing_events: np.ndarray


@dataclass
class RecordingResult:
    """A whole recording, in the same layout ``baseline.ipynb`` plots.

    ``spike_events`` and ``spikes_sparse`` are the dense ``spikes`` raster
    stored without the zero bins. ``crossing_events`` keeps every sample-level
    threshold crossing at the processing rate (15 kHz when decimated, otherwise
    30 kHz).
    """

    rereferenced: np.ndarray
    filtered: np.ndarray
    spikes: np.ndarray | None
    spike_band_power: np.ndarray
    spike_events: np.ndarray
    spikes_sparse: scipy.sparse.csr_matrix
    crossing_events: np.ndarray
    sample_rate: float
    process_rate: float
    decimated: bool


class OptimizedProcessor:
    """Stream 30 kHz neural data with the README optimizations turned on.

    Call ``process_window`` once per millisecond, or ``process_recording`` on
    an array shaped ``(n_channels, n_samples)``. Re-referencing groups run on
    separate threads. Filtering and feature extraction then run across
    channels. Set ``use_gpu="auto"`` to filter on MPS or CUDA when a device is
    present, ``device`` to force one (including ``"cpu"``), and
    ``decimate=True`` for the lossy 15 kHz path.

    The processor is stateful (filter delays, decimator phase). Use it from
    one thread, and do not overlap ``process_window`` calls.
    """

    def __init__(self,
                 n_channels,
                 reref_params,
                 thresholds,
                 reref_groups=None,
                 sample_rate=30000.0,
                 but_order=4,
                 but_low=250.0,
                 but_high=5000.0,
                 causal=False,
                 acausal_filter_type="fir",
                 acausal_filter_lag=None,
                 per_array_threads=True,
                 per_channel_threads=True,
                 channel_workers=None,
                 use_gpu="auto",
                 device=None,
                 decimate=False,
                 store_dense_spikes=True):
        self.n_channels = int(n_channels)
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.reref_params = np.asarray(reref_params, dtype=np.float64)
        if self.reref_params.shape != (self.n_channels, self.n_channels):
            raise ValueError(
                "reref_params must have shape "
                f"{(self.n_channels, self.n_channels)}, got {self.reref_params.shape}"
            )
        self.thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1, 1)
        if self.thresholds.shape[0] != self.n_channels:
            raise ValueError(
                f"{self.thresholds.shape[0]} thresholds for {self.n_channels} channels"
            )

        self.sample_rate = float(sample_rate)
        self.decimate = bool(decimate)
        self.causal = bool(causal)
        self.store_dense_spikes = bool(store_dense_spikes)
        self.process_rate = (self.sample_rate / 2.0
                             if self.decimate else self.sample_rate)
        self.input_samples_per_window = int(self.sample_rate / 1000.0)
        self.samples_per_window = int(self.process_rate / 1000.0)
        if self.input_samples_per_window < 2 or self.samples_per_window < 2:
            raise ValueError(
                "sample rate must yield at least 2 samples per millisecond "
                f"(got {self.input_samples_per_window} in, "
                f"{self.samples_per_window} processed)")

        groups = _normalize_groups(self.n_channels, reref_groups)
        if not _groups_are_independent(self.reref_params, groups):
            logger.info(
                "re-reference weights cross groups; processing arrays together"
            )
            groups = [list(range(self.n_channels))]
        self.groups = groups

        if acausal_filter_lag is None:
            acausal_filter_lag = int(round(_ACAUSAL_LAG_S * self.process_rate))
        self.acausal_filter_lag = int(acausal_filter_lag)

        built = build_filter(but_order=but_order,
                             but_low=but_low,
                             but_high=but_high,
                             acausal_filter_type=acausal_filter_type,
                             causal=self.causal,
                             acausal_filter_lag=self.acausal_filter_lag,
                             fs=self.process_rate,
                             n_channels=self.n_channels)
        if self.causal:
            self.filter_func, self.sos, self.zi = built
            self.rev_win = None
            self.rev_zi = None
        else:
            (self.filter_func, self.sos, self.zi, self.rev_win,
             self.rev_zi) = built

        buffer_len = self.acausal_filter_lag + self.samples_per_window
        self.rev_buffer = np.full((self.n_channels, buffer_len),
                                  np.nan,
                                  dtype=np.float32)
        self.filt_buffer = np.zeros(
            (self.n_channels, self.samples_per_window), dtype=np.float32)
        self._spikes_win = np.zeros(self.n_channels, dtype=np.int16)
        self._sbp_win = np.zeros(self.n_channels, dtype=np.float64)
        self._crossings = np.zeros(
            (self.n_channels, self.samples_per_window - 1), dtype=bool)
        self._reref_window = np.zeros(
            (self.n_channels, self.samples_per_window), dtype=np.float64)
        self._raw_proc = None

        self._decimator = None
        if self.decimate:
            self._decimator = AntiAliasDecimator(self.n_channels,
                                                 sample_rate=self.sample_rate)

        self.use_gpu = False
        self.gpu_device = None
        self.gpu_memory = None
        self._torch = None
        self._gpu_dtype = None
        self._sos_t = None
        self._zi_t = None
        self._rev_t = None
        self._rev_win_t = None
        self._rev_zi_t = None
        self._prev_last_t = None
        self._resolve_gpu(use_gpu, device)

        if per_channel_threads:
            if channel_workers is None:
                channel_workers = os.cpu_count() or 1
            self.channel_workers = max(1, int(channel_workers))
        else:
            self.channel_workers = 1
        self._channel_blocks = _channel_blocks(self.n_channels,
                                               self.channel_workers)

        self._array_pool = None
        self._channel_pool = None
        if per_array_threads and len(self.groups) > 1:
            self._array_pool = ThreadPoolExecutor(
                max_workers=len(self.groups),
                thread_name_prefix="mea",
            )
        if len(self._channel_blocks) > 1:
            self._channel_pool = ThreadPoolExecutor(
                max_workers=len(self._channel_blocks),
                thread_name_prefix="channel",
            )

        self._closed = False
        logger.info(
            "processor arrays=%d channel_workers=%d gpu=%s decimate=%s",
            len(self.groups),
            len(self._channel_blocks),
            self.gpu_device if self.use_gpu else "off",
            self.decimate,
        )

    def _resolve_gpu(self, use_gpu, device):
        if use_gpu is False or use_gpu is None:
            return
        if device is None:
            device, kind = select_gpu_device()
            if device is None:
                if use_gpu is True:
                    raise RuntimeError(
                        "use_gpu=True but neither CUDA nor MPS is available. "
                        "Install PyTorch and retry, or pass use_gpu=False.")
                logger.info("no GPU available; filtering on CPU")
                return
        else:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "A torch device was requested but PyTorch is not installed."
                ) from exc
            if isinstance(device, str):
                device = torch.device(device)
            if device.type == "mps":
                kind = "unified"
            elif device.type == "cuda":
                kind = select_gpu_device()[1] or "discrete"
            else:
                kind = "cpu"
        import torch
        self._torch = torch
        self.use_gpu = True
        self.gpu_device = device
        self.gpu_memory = kind
        # MPS has no float64. CUDA does, and float64 matches SciPy closely.
        dtype = torch.float32 if device.type == "mps" else torch.float64
        self._gpu_dtype = dtype
        self._sos_t = torch.as_tensor(self.sos, device=device, dtype=dtype)
        self._zi_t = torch.as_tensor(np.array(self.zi),
                                    device=device,
                                    dtype=dtype)
        self._rev_t = torch.full((self.n_channels, self.rev_buffer.shape[1]),
                                 float("nan"),
                                 device=device,
                                 dtype=dtype)
        if self.rev_win is not None:
            self._rev_win_t = torch.as_tensor(self.rev_win,
                                             device=device,
                                             dtype=dtype)
        if self.rev_zi is not None:
            self._rev_zi_t = torch.as_tensor(np.array(self.rev_zi),
                                            device=device,
                                            dtype=dtype)
        self._prev_last_t = torch.zeros(self.n_channels,
                                        device=device,
                                        dtype=dtype)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pool in (self._array_pool, self._channel_pool):
            if pool is not None:
                pool.shutdown(wait=True)
        self._array_pool = None
        self._channel_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            return

    def process_window(self, raw):
        """Process one millisecond of raw samples.

        ``raw`` has shape ``(n_channels, input_samples_per_window)`` at
        ``sample_rate``. When decimating, those samples are low-pass filtered
        and downsampled before re-referencing.
        """
        if self._closed:
            raise RuntimeError("processor is closed")
        raw = np.asarray(raw)
        expected = (self.n_channels, self.input_samples_per_window)
        if raw.shape != expected:
            raise ValueError(f"expected raw shape {expected}, got {raw.shape}")

        if self._decimator is not None:
            processed = self._decimator.process(raw)
            if processed.shape[1] != self.samples_per_window:
                raise RuntimeError(
                    "decimator returned "
                    f"{processed.shape[1]} samples, expected "
                    f"{self.samples_per_window}")
        else:
            processed = raw
        self._raw_proc = processed

        self._map(self._array_pool, self._reref_one, self.groups)
        if self.use_gpu:
            self._gpu_filter()
            self._map(self._channel_pool, self._features_block,
                      self._channel_blocks)
        else:
            self._map(self._channel_pool, self._filter_and_features_block,
                      self._channel_blocks)

        channels, relative = np.nonzero(self._crossings)
        if channels.size == 0:
            crossing_events = np.zeros((0, 2), dtype=np.int32)
        else:
            crossing_events = np.column_stack(
                (channels, relative + 1)).astype(np.int32, copy=False)
        return WindowResult(
            rereferenced=self._reref_window.copy(),
            filtered=self.filt_buffer.copy(),
            spikes=self._spikes_win.copy(),
            spike_band_power=self._sbp_win.copy(),
            crossing_events=crossing_events,
        )

    def process_recording(self, data):
        """Process a recording in consecutive 1 ms windows.

        Samples past the last full millisecond are dropped, same as
        ``baseline.ipynb``.
        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(
                f"expected shape ({self.n_channels}, n_samples), got {data.shape}"
            )
        samples_per_window = self.input_samples_per_window
        n_windows = data.shape[1] // samples_per_window
        n_processed = n_windows * self.samples_per_window

        rereferenced = np.empty((self.n_channels, n_processed),
                                dtype=np.float32)
        filtered = np.empty((self.n_channels, n_processed), dtype=np.float32)
        spike_band = np.empty((self.n_channels, n_windows), dtype=np.float32)
        spikes = (np.empty((self.n_channels, n_windows), dtype=np.int16)
                  if self.store_dense_spikes else None)
        binned_events = []
        crossing_parts = []

        for index in range(n_windows):
            start = index * samples_per_window
            window = self.process_window(data[:, start:start + samples_per_window])
            out_start = index * self.samples_per_window
            out_stop = out_start + self.samples_per_window
            rereferenced[:, out_start:out_stop] = window.rereferenced
            filtered[:, out_start:out_stop] = window.filtered
            spike_band[:, index] = window.spike_band_power
            if spikes is not None:
                spikes[:, index] = window.spikes
            fired = np.flatnonzero(window.spikes)
            if fired.size:
                times = np.full(fired.size, index, dtype=np.int32)
                binned_events.append(np.column_stack((fired, times)))
            if window.crossing_events.size:
                absolute = window.crossing_events.copy()
                absolute[:, 1] += out_start
                crossing_parts.append(absolute)

        if binned_events:
            spike_events = np.vstack(binned_events).astype(np.int32, copy=False)
        else:
            spike_events = np.zeros((0, 2), dtype=np.int32)
        if crossing_parts:
            crossing_events = np.vstack(crossing_parts).astype(np.int32,
                                                              copy=False)
        else:
            crossing_events = np.zeros((0, 2), dtype=np.int32)

        if spikes is not None:
            spikes_sparse = spikes_to_sparse(spikes)
        else:
            spikes_sparse = _events_to_sparse(spike_events, self.n_channels,
                                             n_windows)

        return RecordingResult(
            rereferenced=rereferenced,
            filtered=filtered,
            spikes=spikes,
            spike_band_power=spike_band,
            spike_events=spike_events,
            spikes_sparse=spikes_sparse,
            crossing_events=crossing_events,
            sample_rate=self.sample_rate,
            process_rate=self.process_rate,
            decimated=self.decimate,
        )

    def _map(self, pool, fn, items):
        items = list(items)
        if pool is None or len(items) <= 1:
            for item in items:
                fn(item)
            return
        list(pool.map(fn, items))

    def _reref_one(self, channels):
        with _blas_limits():
            idx = np.asarray(channels, dtype=np.intp)
            self._reref_window[idx] = rereference_data(
                self._raw_proc[idx],
                self.reref_params[np.ix_(idx, idx)],
            )

    def _filter_and_features_block(self, sl):
        with _blas_limits():
            self._apply_filter(sl)
            self._features_block(sl)

    def _apply_filter(self, sl):
        data = self._reref_window[sl]
        filtered = self.filt_buffer[sl]
        if self.causal:
            self.filter_func(data, filtered, self.sos, self.zi[:, sl, :])
            return
        self.filter_func(data, filtered, self.rev_buffer[sl], self.sos,
                         self.zi[:, sl, :], self.rev_win,
                         self._slice_rev_zi(sl))

    def _slice_rev_zi(self, sl):
        if self.rev_zi is None:
            return None
        if self.rev_zi.ndim == 2:
            return self.rev_zi[sl]
        return self.rev_zi[:, sl, :]

    def _features_block(self, sl):
        filtered = self.filt_buffer[sl]
        thresholds = self.thresholds[sl]
        below = ((filtered[:, 1:] < thresholds)
                 & (filtered[:, :-1] >= thresholds))
        self._crossings[sl] = below
        self._spikes_win[sl] = np.any(below, axis=1).astype(np.int16)
        self._sbp_win[sl] = spike_band_power(filtered)

    def _gpu_filter(self):
        torch = self._torch
        data = torch.tensor(np.ascontiguousarray(self._reref_window),
                            device=self.gpu_device,
                            dtype=self._gpu_dtype)
        n_samples = data.shape[1]
        if self.causal:
            filtered, self._zi_t = sosfilt_torch(self._sos_t, data, self._zi_t)
        else:
            tail = self._rev_t[:, n_samples:].clone()
            forward, self._zi_t = sosfilt_torch(self._sos_t, data, self._zi_t)
            self._rev_t[:, :-n_samples] = tail
            self._rev_t[:, -n_samples:] = forward
            if self._rev_win_t is not None:
                filtered = _fir_reverse_torch(self._rev_t, self._rev_win_t)
            else:
                filtered = self._iir_reverse_torch(n_samples)
        cpu = filtered.detach().to("cpu")
        self.filt_buffer[:, :] = cpu.numpy().astype(np.float32, copy=False)

    def _iir_reverse_torch(self, n_samples):
        # Same recurrence as utils.acausal_filter when use_fir is False.
        # rev_zi is scaled by the previous window's last output sample.
        ic = self._rev_zi_t * self._prev_last_t[None, :, None]
        flipped = self._rev_t.flip(1)
        reverse, _ = sosfilt_torch(self._sos_t, flipped, ic)
        filtered = reverse[:, -n_samples:].flip(1)
        self._prev_last_t = filtered[:, -1].detach()
        return filtered


def _fir_reverse_torch(rev_buffer, rev_win):
    """Match ``np.convolve(buffer[::-1], rev_win, 'valid')`` then flip back."""
    import torch
    flipped = rev_buffer.flip(1)
    weight = rev_win.flip(0).view(1, 1, -1)
    valid = torch.nn.functional.conv1d(flipped.unsqueeze(1), weight).squeeze(1)
    return valid.flip(1)


def _events_to_sparse(events, n_channels, n_windows):
    if events.size == 0:
        return scipy.sparse.csr_matrix((n_channels, n_windows), dtype=np.int16)
    values = np.ones(events.shape[0], dtype=np.int16)
    return scipy.sparse.csr_matrix(
        (values, (events[:, 0], events[:, 1])),
        shape=(n_channels, n_windows),
    )

"""Optimizations match the baseline notebook loop, aside from lossy decimation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from optimizations import (  # noqa: E402
    AntiAliasDecimator,
    OptimizedProcessor,
    _groups_are_independent,
    select_gpu_device,
    sosfilt_torch,
    spikes_to_events,
    spikes_to_sparse,
    threshold_crossings,
)
from utils import build_filter, rereference_data  # noqa: E402


def _block_params(n_channels, groups, rng):
    params = np.zeros((n_channels, n_channels), dtype=np.float64)
    for group in groups:
        idx = np.asarray(group)
        block = rng.normal(scale=0.05, size=(idx.size, idx.size))
        block = np.abs(block)
        block /= block.sum(axis=1, keepdims=True)
        params[np.ix_(idx, idx)] = block
    return params


def notebook_loop(raw, reref_params, thresholds, sample_rate=30000.0):
    """The baseline.ipynb millisecond loop, without the NSX file."""
    n_channels = raw.shape[0]
    samples_per_ms = int(sample_rate / 1000.0)
    n_windows = raw.shape[1] // samples_per_ms
    lag = int(round(0.004 * sample_rate))
    filter_func, sos, zi, rev_win, rev_zi = build_filter(
        n_channels=n_channels,
        but_low=250,
        but_high=5000,
        causal=False,
        acausal_filter_type="fir",
        acausal_filter_lag=lag,
        fs=sample_rate,
    )
    spikes = np.zeros((n_channels, n_windows), dtype=np.int16)
    spike_band = np.zeros((n_channels, n_windows), dtype=np.float32)
    n_out = n_windows * samples_per_ms
    filtered = np.zeros((n_channels, n_out), dtype=np.float32)
    rereferenced = np.zeros((n_channels, n_out), dtype=np.float32)
    rev_buffer = np.full((n_channels, lag + samples_per_ms),
                         np.nan,
                         dtype=np.float32)
    filt_buffer = np.zeros((n_channels, samples_per_ms), dtype=np.float32)
    thresholds = np.asarray(thresholds).reshape(-1, 1)
    for index in range(n_windows):
        start = index * samples_per_ms
        stop = start + samples_per_ms
        chunk = raw[:, start:stop]
        reref = rereference_data(chunk, reref_params)
        rereferenced[:, start:stop] = reref
        filter_func(reref, filt_buffer, rev_buffer, sos, zi, rev_win, rev_zi)
        filtered[:, start:stop] = filt_buffer
        crossings = ((filt_buffer[:, 1:] < thresholds)
                     & (filt_buffer[:, :-1] >= thresholds))
        spikes[:, index] = np.any(crossings, axis=1).astype(np.int16)
        spike_band[:, index] = (10 * np.log10(np.square(filt_buffer))).mean(axis=1)
    return rereferenced, filtered, spikes, spike_band


class OptimizationsTest(unittest.TestCase):

    def _recording(self, n_channels=12, n_windows=8, groups=None):
        rng = np.random.default_rng(7)
        if groups is None:
            groups = [list(range(0, n_channels // 2)),
                      list(range(n_channels // 2, n_channels))]
        raw = rng.normal(scale=50.0, size=(n_channels, n_windows * 30))
        params = _block_params(n_channels, groups, rng)
        thresholds = -3.5 * np.full((n_channels, 1), 40.0)
        return raw, params, thresholds, groups

    def _run(self, **kwargs):
        raw, params, thresholds, groups = self._recording()
        with OptimizedProcessor(n_channels=raw.shape[0],
                                reref_params=params,
                                thresholds=thresholds,
                                reref_groups=groups,
                                use_gpu=False,
                                **kwargs) as proc:
            result = proc.process_recording(raw)
        reference = notebook_loop(raw, params, thresholds)
        return result, reference

    def _assert_matches(self, result, reference):
        reref, filtered, spikes, spike_band = reference
        np.testing.assert_allclose(result.rereferenced,
                                   reref,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   equal_nan=True)
        np.testing.assert_allclose(result.filtered,
                                   filtered,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   equal_nan=True)
        np.testing.assert_array_equal(result.spikes, spikes)
        np.testing.assert_allclose(result.spike_band_power,
                                   spike_band,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   equal_nan=True)

    def test_serial_matches_notebook_loop(self):
        result, reference = self._run(per_array_threads=False,
                                      per_channel_threads=False,
                                      decimate=False)
        self._assert_matches(result, reference)

    def test_array_and_channel_threads_match_notebook_loop(self):
        result, reference = self._run(per_array_threads=True,
                                      per_channel_threads=True,
                                      channel_workers=2,
                                      decimate=False)
        self._assert_matches(result, reference)

    def test_one_thread_per_channel_matches_notebook_loop(self):
        raw, params, thresholds, groups = self._recording()
        with OptimizedProcessor(n_channels=raw.shape[0],
                                reref_params=params,
                                thresholds=thresholds,
                                reref_groups=groups,
                                per_array_threads=True,
                                per_channel_threads=True,
                                channel_workers=raw.shape[0],
                                use_gpu=False) as proc:
            self.assertEqual(len(proc._channel_blocks), raw.shape[0])
            result = proc.process_recording(raw)
        self._assert_matches(result, notebook_loop(raw, params, thresholds))

    def test_unsorted_groups_match_notebook_loop(self):
        raw, params, thresholds, groups = self._recording()
        groups = [list(reversed(group)) for group in groups]
        with OptimizedProcessor(n_channels=raw.shape[0],
                                reref_params=params,
                                thresholds=thresholds,
                                reref_groups=groups,
                                use_gpu=False) as proc:
            result = proc.process_recording(raw)
        self._assert_matches(result, notebook_loop(raw, params, thresholds))

    def test_sparse_events_rebuild_the_dense_raster(self):
        result, _ = self._run()
        dense = np.zeros_like(result.spikes)
        if result.spike_events.size:
            dense[result.spike_events[:, 0], result.spike_events[:, 1]] = 1
        np.testing.assert_array_equal(dense, result.spikes)
        np.testing.assert_array_equal(result.spikes_sparse.toarray(),
                                      result.spikes)
        np.testing.assert_array_equal(spikes_to_events(result.spikes),
                                      result.spike_events)
        np.testing.assert_array_equal(
            spikes_to_sparse(result.spikes).toarray(), result.spikes)

    def test_store_dense_spikes_false_keeps_events(self):
        raw, params, thresholds, groups = self._recording()
        with OptimizedProcessor(n_channels=raw.shape[0],
                                reref_params=params,
                                thresholds=thresholds,
                                reref_groups=groups,
                                use_gpu=False,
                                store_dense_spikes=False) as proc:
            result = proc.process_recording(raw)
        self.assertIsNone(result.spikes)
        rebuilt = np.zeros((raw.shape[0], raw.shape[1] // 30), dtype=np.int16)
        if result.spike_events.size:
            rebuilt[result.spike_events[:, 0], result.spike_events[:, 1]] = 1
        np.testing.assert_array_equal(result.spikes_sparse.toarray(), rebuilt)

    def test_overlapping_groups_raise(self):
        raw, params, thresholds, _ = self._recording()
        with self.assertRaises(ValueError):
            OptimizedProcessor(n_channels=raw.shape[0],
                               reref_params=params,
                               thresholds=thresholds,
                               reref_groups=[[0, 1, 2], [2, 3, 4]],
                               use_gpu=False)

    def test_coupled_weights_still_match_full_rereference(self):
        raw, _, thresholds, groups = self._recording()
        rng = np.random.default_rng(3)
        coupled = rng.normal(scale=0.01, size=(raw.shape[0], raw.shape[0]))
        self.assertFalse(_groups_are_independent(coupled, groups))
        with OptimizedProcessor(n_channels=raw.shape[0],
                                reref_params=coupled,
                                thresholds=thresholds,
                                reref_groups=groups,
                                use_gpu=False) as proc:
            self.assertEqual(len(proc.groups), 1)
            result = proc.process_recording(raw)
        self._assert_matches(result,
                             notebook_loop(raw, coupled, thresholds))

    def test_threshold_crossing_sample_index(self):
        filtered = np.array([[0.0, -1.0, -2.0], [1.0, 1.0, 1.0]],
                            dtype=np.float32)
        spikes, events = threshold_crossings(filtered, np.array([-0.5, -0.5]))
        np.testing.assert_array_equal(spikes, [1, 0])
        np.testing.assert_array_equal(events, [[0, 1]])

    def test_decimator_matches_lowpass_then_stride_and_streams(self):
        rng = np.random.default_rng(11)
        raw = rng.normal(size=(3, 60))
        once = AntiAliasDecimator(3, 30000.0)
        batch = once.process(raw)
        filtered, _ = scipy.signal.sosfilt(once.sos,
                                           raw,
                                           axis=1,
                                           zi=np.zeros_like(once.zi))
        np.testing.assert_allclose(batch, filtered[:, ::2])

        streamed = AntiAliasDecimator(3, 30000.0)
        parts = [
            streamed.process(raw[:, :31]),
            streamed.process(raw[:, 31:]),
        ]
        np.testing.assert_allclose(np.concatenate(parts, axis=1), batch)

    def test_decimator_attenuates_above_nyquist(self):
        sample_rate = 30000.0
        n_samples = 3000
        t = np.arange(n_samples) / sample_rate
        low = np.sin(2 * np.pi * 1000 * t)
        high = np.sin(2 * np.pi * 12000 * t)
        low_out = AntiAliasDecimator(1, sample_rate).process(low[None, :])
        high_out = AntiAliasDecimator(1, sample_rate).process(high[None, :])
        # Drop the filter transient.
        low_rms = np.sqrt(np.mean(np.square(low_out[:, 200:])))
        high_rms = np.sqrt(np.mean(np.square(high_out[:, 200:])))
        self.assertGreater(low_rms, 0.5)
        self.assertLess(high_rms, 0.05 * low_rms)
        # Skipping samples with no low-pass leaves the 12 kHz tone large.
        naive = high[::2]
        naive_rms = np.sqrt(np.mean(np.square(naive[200:])))
        self.assertGreater(naive_rms, 10 * high_rms)

    def test_lossy_decimated_pipeline_shape_and_finite_tail(self):
        rng = np.random.default_rng(5)
        n_channels = 8
        n_windows = 12
        groups = [list(range(4)), list(range(4, 8))]
        raw = rng.normal(scale=30.0, size=(n_channels, n_windows * 30))
        params = _block_params(n_channels, groups, rng)
        thresholds = -np.full((n_channels, 1), 80.0)
        with OptimizedProcessor(n_channels=n_channels,
                                reref_params=params,
                                thresholds=thresholds,
                                reref_groups=groups,
                                decimate=True,
                                use_gpu=False) as proc:
            self.assertEqual(proc.process_rate, 15000.0)
            self.assertEqual(proc.samples_per_window, 15)
            self.assertEqual(proc.acausal_filter_lag, 60)
            result = proc.process_recording(raw)
        self.assertTrue(result.decimated)
        self.assertEqual(result.filtered.shape, (n_channels, n_windows * 15))
        self.assertEqual(result.spikes.shape, (n_channels, n_windows))
        self.assertTrue(np.isfinite(result.filtered[:, 200:]).all())
        self.assertTrue(np.isfinite(result.spike_band_power[:, 8:]).all())

    def test_empty_spike_events_have_two_columns(self):
        events = spikes_to_events(np.zeros((4, 5), dtype=np.int16))
        self.assertEqual(events.shape, (0, 2))

    def test_select_gpu_device_without_a_gpu_returns_none(self):
        device, kind = select_gpu_device()
        if device is None:
            self.assertIsNone(kind)
        else:
            self.assertIn(kind, ("unified", "discrete"))


def _torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_torch_available(), "PyTorch is not installed")
class TorchFilterTest(unittest.TestCase):

    def test_sosfilt_torch_matches_scipy_on_cpu(self):
        import torch
        rng = np.random.default_rng(0)
        sos = scipy.signal.butter(4, [250, 5000],
                                  btype="bandpass",
                                  output="sos",
                                  fs=30000)
        data = rng.normal(size=(6, 30))
        zi_flat = scipy.signal.sosfilt_zi(sos)
        zi = np.zeros((zi_flat.shape[0], data.shape[0], zi_flat.shape[1]))
        zi[:, :, :] = zi_flat[:, None, :]
        expected, zi_expected = scipy.signal.sosfilt(sos, data, axis=1, zi=zi)
        y, zf = sosfilt_torch(
            torch.as_tensor(sos, dtype=torch.float64),
            torch.as_tensor(data, dtype=torch.float64),
            torch.as_tensor(zi, dtype=torch.float64),
        )
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(zf.numpy(), zi_expected, rtol=1e-10, atol=1e-10)

    def test_causal_and_iir_reverse_match_scipy(self):
        raw, params, thresholds, groups = OptimizationsTest()._recording(
            n_windows=6)
        for causal, acausal_filter_type in ((True, "fir"), (False, "iir")):
            kwargs = dict(n_channels=raw.shape[0],
                          reref_params=params,
                          thresholds=thresholds,
                          reref_groups=groups,
                          causal=causal,
                          acausal_filter_type=acausal_filter_type,
                          per_array_threads=True,
                          per_channel_threads=True,
                          channel_workers=raw.shape[0])
            with OptimizedProcessor(use_gpu=False, **kwargs) as cpu:
                cpu_result = cpu.process_recording(raw)
            with OptimizedProcessor(use_gpu=True, device="cpu", **kwargs) as gpu:
                gpu_result = gpu.process_recording(raw)
            np.testing.assert_allclose(gpu_result.filtered,
                                       cpu_result.filtered,
                                       rtol=1e-5,
                                       atol=1e-5,
                                       equal_nan=True)
            np.testing.assert_array_equal(gpu_result.spikes, cpu_result.spikes)

    def test_torch_cpu_device_matches_scipy_pipeline(self):
        raw, params, thresholds, groups = OptimizationsTest()._recording(
            n_channels=8, n_windows=6)
        kwargs = dict(n_channels=raw.shape[0],
                      reref_params=params,
                      thresholds=thresholds,
                      reref_groups=groups,
                      per_array_threads=True,
                      per_channel_threads=True,
                      decimate=False)
        with OptimizedProcessor(use_gpu=False, **kwargs) as cpu:
            cpu_result = cpu.process_recording(raw)
        with OptimizedProcessor(use_gpu=True, device="cpu", **kwargs) as gpu:
            self.assertEqual(gpu.gpu_memory, "cpu")
            gpu_result = gpu.process_recording(raw)
        np.testing.assert_allclose(gpu_result.filtered,
                                   cpu_result.filtered,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   equal_nan=True)
        np.testing.assert_allclose(gpu_result.spike_band_power,
                                   cpu_result.spike_band_power,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   equal_nan=True)
        np.testing.assert_array_equal(gpu_result.spikes, cpu_result.spikes)

    def test_gpu_processor_matches_cpu_when_device_present(self):
        device, _ = select_gpu_device()
        if device is None:
            self.skipTest("no CUDA or MPS device")
        raw, params, thresholds, groups = OptimizationsTest()._recording(
            n_channels=8, n_windows=6)
        kwargs = dict(n_channels=raw.shape[0],
                      reref_params=params,
                      thresholds=thresholds,
                      reref_groups=groups,
                      decimate=False)
        with OptimizedProcessor(use_gpu=False, **kwargs) as cpu:
            cpu_result = cpu.process_recording(raw)
        with OptimizedProcessor(use_gpu=True, **kwargs) as gpu:
            gpu_result = gpu.process_recording(raw)
        np.testing.assert_allclose(gpu_result.filtered,
                                   cpu_result.filtered,
                                   rtol=1e-4,
                                   atol=1e-3,
                                   equal_nan=True)
        np.testing.assert_array_equal(gpu_result.spikes, cpu_result.spikes)


if __name__ == "__main__":
    unittest.main()

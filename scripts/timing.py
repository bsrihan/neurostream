"""Timing and agreement helpers behind ``notebooks/rust_results.ipynb``.

Every time here is **milliseconds of compute per second of data**: wall
time divided by the duration of the data processed, times 1000. Under 1000
means the path keeps up with the recording.

Three paths are timed the same way, steady state: the processor is built
beforehand, the output arrays already exist, and only the per-millisecond
work is inside the clock.

- the original loop from the Python project (``run_original_loop``),
- the Python kept path, ``OptimizedProcessor`` (``kept_path_steady``),
- the Rust crate, through the ``neurostream-bench`` binary (``rust_bench``).

The synthetic recording is Gaussian noise with a spike planted every 10 ms
and a flat threshold, the same generator the Python project's
``optimization_results.ipynb`` uses. It stands in for electrode counts the
public recording does not have (it has 128). Timing depends on the array
sizes, not the waveform, so this is fair for speed; it is not neural data.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import time

import numpy as np

from paths import ROOT, add_python_project

SAMPLE_RATE = 30_000
SAMPLES_PER_MS = 30
GROUP_SIZE = 64
LAG_SAMPLES = 120
STAGES = ("reref", "forward", "reverse", "features", "write")
STAGE_LABELS = {"reref": "re-reference", "forward": "forward filter", "reverse": "reverse filter",
                "features": "spikes + power", "write": "write"}


def quiet(fn):
    """Call ``fn`` with stdout swallowed (the Python filter builder prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def load_params(path):
    """``(reref_matrix, thresholds, groups)`` from a ``calc_params.py`` file."""
    p = json.load(open(path))
    reref = np.asarray(p["rereference_parameters"], dtype=np.float64)
    thresholds = np.asarray(p["thresholds"], dtype=np.float64).ravel()
    groups = [list(g) for g in p["reref_groups"]]
    return reref, thresholds, groups, p


def make_recording(n_channels, n_ms, seed=1, spike_every_ms=10, spike_amp=-500.0):
    """Synthetic channel-major recording: ``(raw, reref_params, thresholds, groups)``."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(scale=20.0, size=(n_channels, n_ms * SAMPLES_PER_MS))
    groups = [list(range(s, min(s + GROUP_SIZE, n_channels))) for s in range(0, n_channels, GROUP_SIZE)]
    params = np.zeros((n_channels, n_channels))
    for g in groups:
        params[np.ix_(g, g)] = 1.0 / len(g)
    for ms in range(8, n_ms, spike_every_ms):
        start = ms * SAMPLES_PER_MS + 12
        raw[(ms // spike_every_ms) % n_channels, start:start + 2] = spike_amp
    return raw, params, np.full((n_channels, 1), -120.0), groups


def run_original_loop(raw, params, thresholds):
    """The Python project's first implementation, timed. Returns seconds."""
    add_python_project()
    from utils import build_filter, rereference_data

    n_ch, n_win = raw.shape[0], raw.shape[1] // SAMPLES_PER_MS
    filter_func, sos, zi, rev_win, rev_zi = quiet(lambda: build_filter(
        n_channels=n_ch, but_low=250, but_high=5000, causal=False, acausal_filter_type="fir",
        acausal_filter_lag=LAG_SAMPLES, fs=SAMPLE_RATE))
    spikes = np.zeros((n_ch, n_win), dtype=np.int16)
    sbp = np.zeros((n_ch, n_win), dtype=np.float32)
    rev_buffer = np.full((n_ch, LAG_SAMPLES + SAMPLES_PER_MS), np.nan, dtype=np.float32)
    filt = np.zeros((n_ch, SAMPLES_PER_MS), dtype=np.float32)
    thr = np.asarray(thresholds).reshape(-1, 1)
    t0 = time.perf_counter()
    for i in range(n_win):
        chunk = rereference_data(raw[:, i * SAMPLES_PER_MS:(i + 1) * SAMPLES_PER_MS], params)
        filter_func(chunk, filt, rev_buffer, sos, zi, rev_win, rev_zi)
        spikes[:, i] = np.any((filt[:, 1:] < thr) & (filt[:, :-1] >= thr), axis=1)
        sbp[:, i] = (10 * np.log10(np.square(filt))).mean(axis=1)
    return time.perf_counter() - t0


def original_steady(n_channels, n_ms, repeats=3):
    data, params, thr, _ = make_recording(n_channels, n_ms)
    return float(np.median([run_original_loop(data, params, thr) / (n_ms / 1000.0) * 1000 for _ in range(repeats)]))


def kept_path_processor(n_channels, reref, thresholds, groups):
    """A single-threaded ``OptimizedProcessor`` on the CPU, built quietly."""
    add_python_project()
    from optimizations import OptimizedProcessor
    return quiet(lambda: OptimizedProcessor(n_channels, reref, thresholds, groups, use_gpu=False,
                                            per_array_threads=False, per_channel_threads=False))


def kept_path_steady(n_channels, n_ms, repeats=3, stages=False):
    """Steady-state time of the Python kept path on synthetic data.

    Returns ``(total_ms_per_s, split)``. With ``stages=True`` the loop is
    instrumented and ``split`` holds the median per stage; the instrumented
    sum is a little above the plain total because the clock stops between
    stages. With ``stages=False`` the loop is the processor's own and
    ``split`` is ``None``.
    """
    add_python_project()
    from optimizations import _apply_fir

    data, params, thr, gl = make_recording(n_channels, n_ms + 20)
    proc = kept_path_processor(n_channels, params, thr, gl)
    quiet(lambda: proc.process_recording(data[:, :SAMPLES_PER_MS * 20]))
    data = data[:, SAMPLES_PER_MS * 20:]
    spw = proc.samples_per_window
    filt_out = np.zeros((n_channels, n_ms * spw), dtype=np.float32)
    reref_out = np.zeros_like(filt_out)
    sbp_out = np.zeros((n_channels, n_ms), dtype=np.float32)
    spk_out = np.zeros((n_channels, n_ms), dtype=np.int16)

    def once():
        acc = dict.fromkeys(STAGES, 0.0)
        for i in range(n_ms):
            chunk = data[:, i * SAMPLES_PER_MS:(i + 1) * SAMPLES_PER_MS]
            if not stages:
                proc._compute_window(chunk)
            else:
                t0 = time.perf_counter()
                np.copyto(proc._raw_buf, chunk)
                proc._apply_reref()
                t1 = time.perf_counter()
                buf = proc.rev_buffer
                buf[:, :-spw] = buf[:, spw:]
                proc._forward_sos(proc._reref_window, buf[:, -spw:])
                t2 = time.perf_counter()
                proc._leading_nans = 0
                _apply_fir(buf, proc._rev_win, spw, proc.filt_buffer, proc._fir_band, 0)
                t3 = time.perf_counter()
                proc._features_all()
                t4 = time.perf_counter()
            o = i * spw
            reref_out[:, o:o + spw] = proc._reref_window
            filt_out[:, o:o + spw] = proc.filt_buffer
            sbp_out[:, i] = proc._sbp_win
            spk_out[:, i] = proc._spikes_win
            if stages:
                t5 = time.perf_counter()
                for k, v in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                    acc[k] += v
        return acc

    once()
    neural = n_ms / 1000.0
    totals, splits = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        acc = once()
        totals.append((time.perf_counter() - t0) / neural * 1000)
        splits.append({k: acc[k] / neural * 1000 for k in STAGES})
    proc.close()
    split = {k: float(np.median([s[k] for s in splits])) for k in STAGES} if stages else None
    return float(np.median(totals)), split


def rust_bench(args):
    """Run ``neurostream-bench`` (building it first if needed); one dict per JSON line."""
    bench = ROOT / "target" / "release" / "neurostream-bench"
    if not bench.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True, capture_output=True)
    out = subprocess.run([str(bench)] + [str(a) for a in args], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def steady_state_table(channels=(64, 256, 1024, 2048), n_ms=200, repeats=3):
    """Totals for the four paths and stage splits for the two single-thread ones.

    Returns ``(steady, stages)``: ``steady[path][n_channels]`` in ms per
    second of data, ``stages[(path, n_channels)]`` a dict over ``STAGES``.
    """
    steady = {"original loop": {}, "Python kept path": {}, "Rust, 1 thread": {}, "Rust, 4 lanes": {}}
    stages = {}
    for n in channels:
        steady["original loop"][n] = original_steady(n, n_ms, repeats)
        steady["Python kept path"][n], _ = kept_path_steady(n, n_ms, repeats)
        _, stages[("Python kept path", n)] = kept_path_steady(n, n_ms, repeats, stages=True)
    rows = rust_bench(["synthetic", "--channels", ",".join(map(str, channels)), "--threads", "1,4",
                       "--windows", n_ms, "--repeats", repeats])
    for row in rows:
        name = "Rust, 1 thread" if row["threads"] == 1 else "Rust, 4 lanes"
        steady[name][row["channels"]] = row["total_ms_per_s"]
        if row["stages_ms_per_s"]:
            stages[(name, row["channels"])] = row["stages_ms_per_s"]
    return steady, stages


def agreement(rust_out, py_result):
    """How far apart the Rust and Python kept-path outputs are on one recording."""
    ok = ~np.isnan(py_result.filtered)
    diff = (rust_out["filtered"] - py_result.filtered)[ok]
    return dict(
        spikes_py=int(py_result.spikes.sum()),
        spikes_rust=int(rust_out["spikes"].sum()),
        spike_mismatch=int((rust_out["spikes"] != py_result.spikes).sum()),
        filt_max=float(np.abs(diff).max()),
        filt_scale=float(np.abs(py_result.filtered[ok]).max()),
        sbp_max=float(np.nanmax(np.abs(rust_out["spike_band_power"] - py_result.spike_band_power))),
        filt_diff=diff,
    )


def cpu_label():
    text = ""
    try:
        text = open("/proc/cpuinfo", encoding="utf-8").read()
    except OSError:
        pass
    model = next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name")), "unknown CPU")
    avx = "AVX-512" if "avx512f" in text else "no AVX-512"
    return f"{model}, {os.cpu_count()} cores, {avx}"

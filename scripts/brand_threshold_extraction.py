"""Run brand-nsp's ``thresholdExtraction`` node on a stretch of the recording.

This is the processing path the project README cites. The node is run as
written, as its own process, talking to a local Redis server the way it does
in a BRAND graph: parameters come from a ``supergraph_stream`` entry, raw
samples arrive one millisecond per stream entry, and the node writes one
``crossings`` entry per millisecond with the timestamp of the input window it
describes.

Usage from a notebook::

    from brand_threshold_extraction import run_node
    out = run_node(raw, thresholds, seconds=10)
    out["crossings"]      # (n_channels, n_out) int16
    out["timestamps"]     # (n_out,) first sample index of the input window each column describes
    out["filtered"]       # (n_channels, n_out * 30) int16, the node's filtered output

``raw`` is time-major ``(n_samples, n_channels)`` int16, as ``neurostream.read_ns6``
returns it. The node's output for input window ``w`` carries timestamp
``30 * w``; the kept path reports that window as output window ``w + 4``
(its 4 ms look-ahead), which is the shift ``align`` applies.

Requirements: ``redis-server`` on PATH, the ``redis``, ``pyyaml``, ``sh`` and
``coloredlogs`` Python packages, the ``brand`` package from
https://github.com/brandbci/brand (``pip install brand/lib/python``) and a
clone of https://github.com/brandbci/brand-nsp (cloned on first use).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

BRAND_NSP_URL = "https://github.com/brandbci/brand-nsp"
BRAND_NSP_COMMIT = "4c891da832767615eb47e61e692b23a672392180"
NODE_NAME = "thresholdExtraction"
SAMPLES_PER_WINDOW = 30
LAG_WINDOWS = 4


def ensure_brand_nsp(dest: str | os.PathLike = "/tmp/brand-nsp") -> Path:
    """Clone brand-nsp at the pinned commit if it is not there yet."""
    dest = Path(dest)
    node = dest / "nodes" / NODE_NAME / f"{NODE_NAME}.py"
    if not node.exists():
        subprocess.run(["git", "clone", "--quiet", BRAND_NSP_URL, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", BRAND_NSP_COMMIT], check=True)
    return node


def node_parameters(n_channels: int, thresholds_file: str, thresh_mult: float,
                    output_filtered: bool = True) -> dict:
    """The parameters the node would get from a BRAND graph YAML."""
    return {
        "log": "INFO",
        "thresh_mult": thresh_mult,
        "pack_per_call": 1,
        "output_filtered": output_filtered,
        "acausal_filter": "fir",
        "acausal_filter_lag": LAG_WINDOWS * SAMPLES_PER_WINDOW,
        "enable_CAR": True,
        "CAR_group_sizes": 64,
        "input_name": "nsp_neural",
        "input_samp_per_stream": SAMPLES_PER_WINDOW,
        "input_chan_per_stream": n_channels,
        "input_samp_freq": 30000,
        "butter_order": 4,
        "butter_lowercut": 250,
        "butter_uppercut": 5000,
        "thresholds_file": thresholds_file,
    }


def run_node(raw: np.ndarray, thresholds: np.ndarray, seconds: float | None = None,
             thresh_mult: float = -3.5, port: int = 6380, brand_nsp_dir="/tmp/brand-nsp",
             output_filtered: bool = True, timeout_s: float = 600.0) -> dict:
    """Stream ``raw`` through the node and return what it wrote.

    The node's thresholds are read from ``thresholds`` (one per channel), so
    the comparison is between processing paths, not between threshold
    estimates.
    """
    raw = np.ascontiguousarray(raw)
    if raw.ndim != 2 or raw.dtype != np.int16:
        raise ValueError("raw must be (n_samples, n_channels) int16")
    n_channels = raw.shape[1]
    n_windows = raw.shape[0] // SAMPLES_PER_WINDOW
    if seconds is not None:
        n_windows = min(n_windows, int(round(seconds * 1000)))
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    if thresholds.shape[0] != n_channels:
        raise ValueError(f"{thresholds.shape[0]} thresholds for {n_channels} channels")

    import redis  # noqa: delayed so the module imports without it
    import yaml

    node_path = ensure_brand_nsp(brand_nsp_dir)
    if shutil.which("redis-server") is None:
        raise RuntimeError("redis-server is not on PATH")

    workdir = Path(tempfile.mkdtemp(prefix="brand_te_"))
    thresholds_file = workdir / "thresholds.yaml"
    with open(thresholds_file, "w") as f:
        yaml.safe_dump({"thresholds": [float(t) for t in thresholds]}, f)

    redis_log = open(workdir / "redis.log", "w")
    redis_proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no",
         "--bind", "127.0.0.1", "--loglevel", "warning"],
        stdout=redis_log, stderr=subprocess.STDOUT)
    node_proc = None
    try:
        r = redis.Redis("127.0.0.1", port)
        deadline = time.time() + 20
        while True:
            try:
                r.ping()
                break
            except redis.exceptions.ConnectionError:
                if time.time() > deadline:
                    raise RuntimeError("redis-server did not start")
                time.sleep(0.05)
        r.flushall()

        params = node_parameters(n_channels, str(thresholds_file), thresh_mult, output_filtered)
        supergraph = {"nodes": {NODE_NAME: {"nickname": NODE_NAME, "parameters": params}}}
        r.xadd("supergraph_stream", {"data": json.dumps(supergraph)})

        node_log = open(workdir / "node.log", "w")
        node_proc = subprocess.Popen(
            ["python3", str(node_path), "-n", NODE_NAME, "-i", "127.0.0.1", "-p", str(port)],
            cwd=node_path.parent, stdout=node_log, stderr=subprocess.STDOUT)

        # The node reads only entries added after it starts listening, so
        # wait for it to report its filter before sending data.
        deadline = time.time() + 60
        while True:
            text = (workdir / "node.log").read_text()
            if "Loaded thresholds from" in text and "Loading 4 order" in text:
                break
            if node_proc.poll() is not None:
                raise RuntimeError(f"node exited early:\n{text}")
            if time.time() > deadline:
                raise RuntimeError(f"node did not start:\n{text}")
            time.sleep(0.05)
        time.sleep(1.0)

        t_feed0 = time.perf_counter()
        pipe = r.pipeline(transaction=False)
        for w in range(n_windows):
            s0 = w * SAMPLES_PER_WINDOW
            block = np.ascontiguousarray(raw[s0:s0 + SAMPLES_PER_WINDOW].T)  # (n_channels, 30)
            ts = np.arange(s0, s0 + SAMPLES_PER_WINDOW, dtype=np.uint32)
            pipe.xadd(params["input_name"], {b"samples": block.tobytes(), b"timestamps": ts.tobytes()})
            if w % 500 == 499:
                pipe.execute()
        pipe.execute()
        t_feed = time.perf_counter() - t_feed0

        expected = n_windows - LAG_WINDOWS
        deadline = time.time() + timeout_s
        while r.xlen(NODE_NAME) < expected:
            if node_proc.poll() is not None:
                raise RuntimeError("node exited:\n" + (workdir / "node.log").read_text())
            if time.time() > deadline:
                raise RuntimeError(f"node wrote {r.xlen(NODE_NAME)} of {expected} entries before timeout")
            time.sleep(0.05)

        entries = r.xrange(NODE_NAME, "-", "+")
        n_out = len(entries)
        crossings = np.empty((n_channels, n_out), dtype=np.int16)
        timestamps = np.empty(n_out, dtype=np.int64)
        wall_ns = np.empty(n_out, dtype=np.int64)
        for j, (_, d) in enumerate(entries):
            crossings[:, j] = np.frombuffer(d[b"crossings"], dtype=np.int16)
            timestamps[j] = int(np.frombuffer(d[b"timestamps"], dtype=np.uint32)[0])
            wall_ns[j] = int(np.frombuffer(d[b"ts"], dtype=np.uint64)[0])

        filtered = None
        if output_filtered:
            fentries = r.xrange(f"{NODE_NAME}_filt", "-", "+")
            filtered = np.empty((n_channels, len(fentries) * SAMPLES_PER_WINDOW), dtype=np.int16)
            for j, (_, d) in enumerate(fentries):
                filtered[:, j * SAMPLES_PER_WINDOW:(j + 1) * SAMPLES_PER_WINDOW] = np.frombuffer(
                    d[b"samples"], dtype=np.int16).reshape(n_channels, SAMPLES_PER_WINDOW)

        return {
            "crossings": crossings,
            "timestamps": timestamps,
            "filtered": filtered,
            "n_windows_in": n_windows,
            "feed_seconds": t_feed,
            "node_span_seconds": float(wall_ns[-1] - wall_ns[0]) / 1e9 if n_out > 1 else float("nan"),
            "node_log": (workdir / "node.log").read_text(),
            "parameters": params,
            "brand_nsp_commit": BRAND_NSP_COMMIT,
        }
    finally:
        if node_proc is not None and node_proc.poll() is None:
            node_proc.terminate()
            try:
                node_proc.wait(5)
            except subprocess.TimeoutExpired:
                node_proc.kill()
        redis_proc.terminate()
        try:
            redis_proc.wait(5)
        except subprocess.TimeoutExpired:
            redis_proc.kill()
        redis_log.close()


def align(node_out: dict, kept_spikes: np.ndarray, lag_windows: int = LAG_WINDOWS):
    """Line the node's columns up with the kept path's output windows.

    The node stamps each column with the input window it describes; the kept
    path reports input window ``w`` as output window ``w + lag_windows``.
    Returns ``(tool, ours)`` of equal shape ``(n_channels, n_common)`` plus
    the input window index of each column.
    """
    w_in = node_out["timestamps"] // SAMPLES_PER_WINDOW
    ours_idx = w_in + lag_windows
    keep = ours_idx < kept_spikes.shape[1]
    tool = node_out["crossings"][:, keep]
    ours = kept_spikes[:, ours_idx[keep]]
    return tool, ours, w_in[keep]


def whole_file_fallback(raw: np.ndarray, reref: np.ndarray, thresholds: np.ndarray) -> dict:
    """The standard whole-file version of the same filter.

    Used only when the node cannot be run: re-reference, then
    ``scipy.signal.sosfiltfilt`` over the whole stretch (forward and backward,
    zero phase, no look-ahead limit), then the same crossing rule per
    millisecond. Returns the same keys as ``run_node`` with no waveform and
    no node timing, so the notebook can carry on and say which path it used.
    """
    import scipy.signal

    x = raw.astype(np.float64).T
    x = (np.eye(x.shape[0]) - reref) @ x
    sos = scipy.signal.butter(4, [250, 5000], btype="bandpass", fs=30000, output="sos")
    f = scipy.signal.sosfiltfilt(sos, x, axis=1)
    n_win = f.shape[1] // SAMPLES_PER_WINDOW
    f = f[:, :n_win * SAMPLES_PER_WINDOW].reshape(f.shape[0], n_win, SAMPLES_PER_WINDOW)
    thr = np.asarray(thresholds).ravel()[:, None, None]
    crossings = np.any((f[:, :, 1:] < thr) & (f[:, :, :-1] >= thr), axis=2).astype(np.int16)
    return {"crossings": crossings, "timestamps": np.arange(n_win) * SAMPLES_PER_WINDOW, "filtered": None,
            "n_windows_in": n_win, "feed_seconds": float("nan"), "node_span_seconds": float("nan"),
            "node_log": "", "parameters": {}, "brand_nsp_commit": None,
            "name": "whole-file scipy.signal.sosfiltfilt (fallback)", "is_fallback": True}


def run_tool_or_fallback(raw, reref, thresholds, seconds, thresh_mult=-3.5, **kwargs) -> dict:
    """Run the node; if it cannot run, use the whole-file filter and say so loudly."""
    try:
        out = run_node(raw, thresholds, seconds=seconds, thresh_mult=thresh_mult, **kwargs)
        out["name"] = f"brand-nsp thresholdExtraction ({out['brand_nsp_commit'][:7]})"
        out["is_fallback"] = False
        return out
    except Exception as exc:  # noqa: BLE001 - any failure means the fallback
        print("=" * 78)
        print("THE ESTABLISHED TOOL COULD NOT BE RUN; USING THE WHOLE-FILE sosfiltfilt FALLBACK")
        print(f"reason: {exc}")
        print("=" * 78)
        out = whole_file_fallback(raw[:int(seconds * 30000)], reref, thresholds)
        out["fallback_reason"] = str(exc)
        return out


def lag_scan(node_out: dict, kept_spikes: np.ndarray, max_lag: int = 8):
    """Bins that differ for each shift 0..max_lag ms applied to our output.

    The right shift is the filter delay (4 ms); every other shift compares
    unrelated milliseconds, so the scan is an empirical check of the
    alignment rather than an assumption.
    """
    lags = np.arange(0, max_lag + 1)
    n_tool = node_out["crossings"].shape[1]
    per_lag = []
    for lag in lags:
        n = min(n_tool, kept_spikes.shape[1] - lag)
        per_lag.append(int((node_out["crossings"][:, :n] != kept_spikes[:, lag:lag + n]).sum()))
    return lags, per_lag, int(lags[int(np.argmin(per_lag))])


def disagreements(node_out: dict, kept_spikes: np.ndarray) -> dict:
    """Align, then count. Returns the aligned arrays and the totals."""
    tool, ours, w_in = align(node_out, kept_spikes)
    only_tool = (tool == 1) & (ours == 0)
    only_ours = (tool == 0) & (ours == 1)
    return dict(
        tool=tool, ours=ours, w_in=w_in, only_tool=only_tool, only_ours=only_ours,
        windows=int(tool.size), spikes_tool=int(tool.sum()), spikes_ours=int(ours.sum()),
        both=int(((tool == 1) & (ours == 1)).sum()),
        n_only_tool=int(only_tool.sum()), n_only_ours=int(only_ours.sum()),
        disagree=int(only_tool.sum() + only_ours.sum()),
    )

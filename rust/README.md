# neurostream (Rust)

The millisecond pipeline from [`../original`](../original), written in Rust: re-reference, forward Butterworth filter, 4 ms reverse filter, threshold crossings and spike-band power, once per millisecond, for up to 2048 electrodes at 30 kHz. It produces the same spikes as the Python `OptimizedProcessor` (0 disagreements on the public recording) and the same waveforms to float32 rounding.

The crate builds three ways:

- a Rust library (`neurostream`), used by the tests and the benchmark;
- a benchmark binary, `neurostream-bench`;
- a Python module, also called `neurostream`, built with maturin, so the notebooks in `notebooks/` can call it.

## Layout

```
rust/
  Cargo.toml, pyproject.toml, .cargo/config.toml
  src/
    lib.rs        Processor, Lane, Config; one window or a whole recording
    design.rs     Butterworth design (butter -> lp2bp -> bilinear -> sos), zi, impulse response
    reref.rs      (I - P) x per electrode group, one small matrix multiply each
    filter.rs     forward SOS (state carried between windows); reverse FIR over a ring buffer
    features.rs   negative threshold crossing, mean 10 log10(x^2)
    nsx.rs        Blackrock .ns6 reader (time-major int16, no copies)
    params.rs     the JSON parameter file calc_params.py writes
    python.rs     PyO3 bindings (feature "python")
    bin/bench.rs  neurostream-bench
  notebooks/
    rust_results.ipynb      graphs: stage split, Rust vs Python, threads, agreement
    tool_comparison.ipynb   the kept path against an established tool on the same recording
```

Notebook folders hold only `.ipynb` files.

## Build and run

```bash
# Rust 1.85 or newer (rustup toolchain install stable)
cd rust
cargo test --release            # 13 tests: filter design vs SciPy values, lanes, warm-up, kernels
cargo build --release

# steady-state timing, synthetic data, one JSON line per row
./target/release/neurostream-bench synthetic --channels 64,256,1024,2048 --threads 1,4 --windows 200 --repeats 3
# the public recording (first 10 s)
./target/release/neurostream-bench ns6 --file ../data/NSP1_aligned.ns6 --params ../data/NSP1_aligned_params.json --seconds 10 --threads 1,4

# Python module
pip install maturin
maturin develop --release       # or: maturin build --release -o dist && pip install dist/*.whl
```

`.cargo/config.toml` sets `-C target-cpu=native`. The binary and the wheel are built for the CPU that compiled them; rebuild on the machine you time on.

### From Python

```python
import json, numpy as np, neurostream

P = json.load(open("../../data/NSP1_aligned_params.json"))
raw, rate = neurostream.read_ns6("../../data/NSP1_aligned.ns6", max_samples=300_000)  # (n_samples, n_channels) int16
proc = neurostream.Processor(np.array(P["rereference_parameters"]), np.array(P["thresholds"]).ravel(), P["reref_groups"], threads=1)
out = proc.process_recording(raw)
out["spikes"]            # (n_channels, n_windows) int16, same layout as the Python tree
out["filtered"]          # (n_channels, n_samples) float32, first 4 ms NaN
out["spike_band_power"]  # (n_channels, n_windows) float32
```

`neurostream.filter_design()` returns the `(sos, zi, reverse_window)` the Rust design produces, for checking against SciPy (they agree to 1e-14). `proc.profile_recording(raw)` returns seconds per stage.

## How it is built

**Time-major everywhere.** A window is `n_samples x n_channels` with channels contiguous, which is the order the `.ns6` file stores samples in. The recording goes from disk into the pipeline without a transpose, and the forward filter and the feature loops walk across electrodes in their inner loop, so the compiler emits one instruction per eight channels (AVX-512) without hand-written SIMD.

**Lanes.** A `Processor` is split into `Lane`s. Each lane owns a contiguous range of channels and every buffer and filter state for them. With `threads=1` there is one lane and no thread pool. With more, each millisecond is processed by all lanes at once on a rayon pool; no lane touches another's memory, so there is nothing to lock. Lane boundaries follow re-reference groups, so a lane never needs another lane's channels. Parameter files whose re-reference weights cross groups (the LRR file for the 128-channel recording) fall back to one lane.

**Reverse filter.** The 4 ms look-ahead buffer is a ring, so a new millisecond is written in place instead of shifting the buffer. The backward filter is a band matrix with 121 taps; it is applied with a tiled kernel that keeps 12 outputs for 16 channels in 24 AVX-512 registers and reads each buffer row once per block (`filter.rs`, `avx512::fir_tiled`). There is a portable fallback for CPUs without AVX-512. The forward output is rounded to float32 before it enters the buffer, as in the Python tree, which is why the two agree bit for bit on spikes.

**Spike-band power.** `10 log10(x^2)` is the costliest part of the features stage in Python. Here `log10` is computed in-line from the float exponent and a short series, which vectorizes; NaN, zero and infinity are detected in the same pass and fixed up only for windows that contain them.

**Filter design.** `design.rs` follows SciPy step by step (`butter` -> `lp2bp_zpk` -> `bilinear_zpk` -> `zpk2sos`, pairing `'nearest'`) so the sections, their initial state and the reverse window match `scipy.signal` to 1e-12; the tests hold SciPy's numbers as literals.

## Numbers

Steady state, this machine (4-core Intel Xeon, Sapphire Rapids, AVX-512), milliseconds of compute per second of data; under 1000 is real time. Synthetic data, 200 windows, median of three runs. The Python column is the kept path (`OptimizedProcessor`, single thread) from `original/notebooks/optimization_results.ipynb`.

| channels | Rust, 1 thread | Rust, 4 lanes | Python kept path |
| --- | --- | --- | --- |
| 64 | 18 | 18 (one group, one lane) | 48 |
| 256 | 65 | 42 | 145 |
| 1024 | 352 | 138 | 579 |
| 2048 | 813 | 324 | 1181 |

Single-thread stage split at 1024 channels: re-reference 71, forward 45, reverse 165, features 40, write 4. The reverse filter is the largest stage; it runs near the one-fused-multiply-add-per-cycle rate of this CPU.

The public 128-channel recording, first 10 seconds, LRR parameters: 33 ms per second of data, one thread. Against the Python kept path on the same stretch: 0 of 26,611 spikes differ, re-referenced data identical, filtered waveform within 4e-6 of 1055 (float32 rounding), spike-band power within 1.2e-5 dB.

The graphs are in [`notebooks/rust_results.ipynb`](./notebooks/rust_results.ipynb). The comparison against an established tool is in [`notebooks/tool_comparison.ipynb`](./notebooks/tool_comparison.ipynb).

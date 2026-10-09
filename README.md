# neurostream

Ultra-low latency signal processing for neural data: re-reference, filter, and extract spikes and spike-band power from 2048-channel 30 kHz microelectrode-array recordings, one millisecond at a time.

The repository is split in two trees that share one recording.

| folder | what is in it |
| --- | --- |
| [`original/`](./original) | The Python project. `src/` holds the original millisecond loop (`utils.py`), the parameter script (`calc_params.py`), and the optimized `OptimizedProcessor` (`optimizations.py`, `x86_forward.py`). `notebooks/` holds `baseline.ipynb` and `optimization_results.ipynb`. `tests/` checks that the fast path matches the original loop. |
| [`rust/`](./rust) | The Rust rewrite of the same pipeline, with a Python module built by maturin so the notebooks can call it. `notebooks/rust_results.ipynb` is the graphs: agreement with the Python path, stage split, lanes, real recording. `notebooks/tool_comparison.ipynb` runs one established tool (brand-nsp `thresholdExtraction`) on the same stretch of the recording and counts disagreements. `scripts/` holds the harness that runs that tool through Redis. |
| `data/` | The public recording `NSP1_aligned.ns6` and the parameter files. Git-ignored; `original/scripts/fetch_example_data.sh` downloads it here. |

Notebook folders contain only `.ipynb` files. Code lives in `original/src/`, `rust/src/` and `rust/scripts/`.

## Where things stand

Milliseconds of compute per second of data on a 4-core Intel Xeon (AVX-512); under 1000 is real time. Steady state, synthetic data, single thread unless noted; one run of `rust/notebooks/rust_results.ipynb`.

| electrodes | original loop | Python kept path | Rust, 1 thread | Rust, 4 lanes |
| --- | --- | --- | --- | --- |
| 64 | 195 | 47 | 17 | 17 |
| 256 | 766 | 167 | 70 | 69 |
| 1024 | 4052 | 646 | 362 | 179 |
| 2048 | 13446 | 1435 | 839 | 472 |

All three give the same spikes on the real recording (0 of 26,611 bins differ). Against brand-nsp `thresholdExtraction`, run unmodified through Redis on the same 10 s with shared thresholds and aligned for the 4 ms filter delay: 0 of 29,974 spikes differ, waveform within 1 LSB. That comparison also found that `calc_params.py` was writing diagonal-only CAR weights (now fixed; the LRR file every earlier number used was unaffected).

## Quick start

```bash
# recording (706 MB) and parameters
./original/scripts/fetch_example_data.sh
cd original/src
python calc_params.py -f ../../data/NSP1_aligned.ns6 -o ../../data/NSP1_aligned_params.json -t -3.5 --reref lrr -d 60
python calc_params.py -f ../../data/NSP1_aligned.ns6 -o ../../data/NSP1_aligned_params_car.json -t -3.5 --reref car -d 60
cd ../..

# Python tree
cd original && python -m unittest tests.test_optimizations && cd ..

# Rust tree (see rust/README.md)
cd rust && cargo test --release && cargo build --release && pip install maturin && maturin develop --release
```

Each tree's own README explains what it measures and where the numbers come from.

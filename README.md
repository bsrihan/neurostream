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

All three give the same spikes on the real recording (0 of 26,611 bins differ). Against one established tool, brand-nsp `thresholdExtraction`, run unmodified through Redis on the same stretch with shared thresholds and aligned for the 4 ms filter delay (`rust/notebooks/tool_comparison.ipynb`):

| path | batch or streaming | spikes, 10 s | bins that differ from the tool | wall time (ms per s of data) |
| --- | --- | ---: | ---: | ---: |
| brand-nsp `thresholdExtraction` (`4c891da`) | streaming (1 ms packets through Redis, 4 ms look-ahead) | 29,974 | — | 574 |
| Python kept path (`OptimizedProcessor`) | streaming (1 ms windows, 4 ms look-ahead) | 29,974 | 0 of 1,279,488 | 142 |
| Rust (kept path) | streaming (1 ms windows, 4 ms look-ahead) | 29,974 | 0 of 1,279,488 | 37 |
| *60 s, same three paths* | streaming | 174,453 / 174,453 | 0 of 7,679,488 | tool 537, Rust 41 |

What the tool was allowed to see that the streaming path was not: nothing in the signal path. It received the same 1 ms packets, the same 4 ms look-ahead and the same thresholds; the one thing it had that a live run would not is that all of the data was already sitting in Redis before it started, so its wall time contains no waiting for packets to arrive. The thresholds are the only batch element: both paths were handed the same values, computed offline by `calc_params.py` from a whole-file `sosfiltfilt` of the first 60 s.

- The tool sees the same stretch and the same thresholds: first 10 s of `NSP1_aligned.ns6` fed to both; one thresholds file handed to the node and to the kept path.
- Spike comparisons are aligned for filter delay: shift of 4 ms from the node's own timestamps; a lag scan puts the minimum at 4 ms.
- Wall time is measured on the same machine: every number from one run of the notebook on this host.

That comparison also found that `calc_params.py` was writing diagonal-only CAR weights (now fixed; the LRR file every earlier number used was unaffected).

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

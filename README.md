# neurostream

Ultra-low latency signal processing for neural data: re-reference, filter, and extract spikes and spike-band power from 2048-channel 30 kHz microelectrode-array recordings, one millisecond at a time.

The repository is split in two trees that share one recording.

| folder | what is in it |
| --- | --- |
| [`original/`](./original) | The Python project. `src/` holds the original millisecond loop (`utils.py`), the parameter script (`calc_params.py`), and the optimized `OptimizedProcessor` (`optimizations.py`, `x86_forward.py`). `notebooks/` holds `baseline.ipynb` and `optimization_results.ipynb`. `tests/` checks that the fast path matches the original loop. |
| [`rust/`](./rust) | The Rust rewrite of the same pipeline, with a Python module built by maturin so the notebooks can call it. `notebooks/` holds the results notebook (graphs) and the comparison against an established tool. |
| `data/` | The public recording `NSP1_aligned.ns6` and the parameter files. Git-ignored; `original/scripts/fetch_example_data.sh` downloads it here. |

Notebook folders contain only `.ipynb` files. Code lives in `original/src/` and `rust/src/`.

## Quick start

```bash
# recording (706 MB) and parameters
./original/scripts/fetch_example_data.sh
cd original/src
python calc_params.py -f ../../data/NSP1_aligned.ns6 -o ../../data/NSP1_aligned_params.json -t -3.5 --reref lrr -d 60
cd ../..

# Python tree
cd original && python -m unittest tests.test_optimizations && cd ..

# Rust tree (see rust/README.md)
cd rust && cargo test --release && pip install maturin && maturin develop --release
```

Each tree's own README explains what it measures and where the numbers come from.

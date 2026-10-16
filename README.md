# neurostream

Ultra-low latency signal processing for neural data: re-reference, filter, and extract spikes and spike-band power from 2048-channel 30 kHz microelectrode-array recordings, one millisecond at a time.

This repository holds the **Rust** pipeline. The Python project that this repository used to contain (the original millisecond loop, `calc_params.py`, the optimized `OptimizedProcessor`, and the notebooks that measure them) now lives in its own repository, [bsrihan/neurostream-python](https://github.com/bsrihan/neurostream-python), with its full history. The Rust notebooks compare against it and expect a checkout of it next to this one (see `scripts/paths.py` once it lands).

The recording both projects use is `NSP1_aligned.ns6` (128 electrodes, 30 kHz, public, 706 MB). It is git-ignored; each project's `scripts/fetch_example_data.sh` downloads it to that project's `data/`.

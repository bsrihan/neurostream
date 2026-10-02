# neurostream
Ultra-low latency signal processing library for neural data (work in progress)

**Design requirements**: Re-reference, filter, and extract features from 2048-channel 30 kHz microelectrode array data in real-time with an output rate of up to 1 kHz   
**Constraints**: Runs on desktop PCs and laptops (with or without GPUs) using Linux (preferred) or macOS

## Getting started

### 1. Create the environment

On macOS, or whenever an exact lockfile is not required:

```bash
conda env create -f environment.min.yml
conda activate neurostream
```

`environment.yml` is an exact export from a Linux machine. It pins Linux-only
packages and will not solve on macOS, so prefer it only when reproducing that
environment specifically.

Then register the Jupyter kernel that `baseline.ipynb` asks for:

```bash
python -m ipykernel install --user --name neurostream
```

The notebook pins `kernelspec.name = neurostream` in its metadata, so without
this step Jupyter reports the kernel as missing and non-interactive execution
fails with `NoSuchKernel: neurostream`.

### 2. Download the example recording

```bash
./scripts/fetch_example_data.sh
```

This writes `data/NSP1_aligned.ns6` (~706 MiB), which is git-ignored. The file
holds 128 channels sampled at 30 kHz for about 96 seconds; the 1024 electrodes
in the dataset name are split across several NSPs. `baseline.ipynb` is the
notebook that reads this file. The timing notebook does not. See "Which
recording the numbers used" below.

### 3. Compute thresholds and re-referencing parameters

`calc_params.py` and the notebook both resolve `utils` and their data paths
relative to `notebooks/`, so run them from there:

```bash
cd notebooks
python calc_params.py \
    -f ../data/NSP1_aligned.ns6 \
    -o ../data/NSP1_aligned_params.json \
    -t -3.5 --reref lrr --plot_spike_panel \
    -d 60
```

This measures per-channel noise and writes voltage thresholds and
re-referencing weights to the JSON file. `--plot_spike_panel` also saves spike
panels next to it, which are worth a look before continuing. Takes about a
minute for the example file.

`-d 60` caps the calculation at the first 60 seconds. Without it the entire
recording is loaded as float64 — roughly 3 GB for the example file, before the
temporaries `sosfiltfilt` allocates on top — which is enough to exhaust a 16 GB
machine. A minute of data is also what the thresholding section below
recommends, and `baseline.ipynb` only processes the first 10 seconds anyway.

### 4. Run the pipeline

```bash
jupyter lab baseline.ipynb
```

The notebook reads the recording and the JSON from step 3, processes the data
one millisecond at a time, and plots spikes and spike-band power.

## What this does

A channel is one electrode. Each electrode is sampled 30,000 times a second (30 kHz). The job is to take one millisecond of those voltages and return two things for every electrode: whether a spike happened, and one number for how strong the spike-band activity was.

Real time means the computer finishes that millisecond before the next one arrives. Over a full second of data, that is under 1000 ms of clock time.

The steps, in order:

1. **Re-reference.** Many electrodes pick up the same noise. Subtract a shared estimate so what is left is more local to each electrode. A common average subtracts the group's mean. A linear regression reference subtracts a fitted mix of the other electrodes in the group.
2. **Filter.** Keep 250–5000 Hz, the band where spikes show up. The filter looks 4 ms ahead so it does not slide the spike later in time.
3. **Threshold crossing.** If the filtered voltage falls through that electrode's threshold during the millisecond, count one spike. Extra crossings in the same millisecond do not add more spikes.
4. **Spike-band power.** Square the filtered voltage, take ten times the log, and average over the millisecond. This tracks similar activity without using a threshold.

[baseline.ipynb](./notebooks/baseline.ipynb) runs these steps on the example recording. [notebooks/optimization_results.ipynb](./notebooks/optimization_results.ipynb) times the fast version of the same steps.

The paragraphs below are the same steps with the usual names and the papers they come from.

## Signal Processing

### Re-referencing

The raw data consists of voltage measurements relative to a set of reference electrodes. This data can still have noise that is correlated across channels, so it is often desirable to apply a common-average reference or a linear regression reference. See [re_reference.py](https://github.com/brandbci/brand-nsp/blob/main/nodes/re_reference/re_reference.py) in `brand-nsp` for an example of how this is done.

### Filtering

After re-referencing, the neural data is filtered with either a high-pass or band-pass filter. For neural spiking activity (a.k.a action potentials or threshold crossings), the frequency range of interest is typically 250-5000 Hz. [Masse et al 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4169749/) showed that zero-phase acausal filtering is better for spike detection than causal filtering. When using this acausal filtering approach, we maintain a 4 ms buffer of data and run the backwards pass of the filter over that buffer to cancel out the phase shift caused by the forward pass.

### Thresholding

For each channel, set a threshold that is a multiple of the root mean square (RMS) voltage (typically -4.5 * RMS). This threshold can be set once using a minute of sample data at the start of a recording session or it can be updated with a running window throughout the recording. See [calcThreshNorm.py](https://github.com/brandbci/brand-nsp/blob/main/derivatives/calcThreshNorm/calcThreshNorm.py) for an example of how thresholds are calculated.

### Threshold crossings

Detect times when a channel's voltage drops below its threshold and count those as spikes. Neurons cannot spike faster than 1 kHz, so, if you detect multiple threshold crossings within a 1 millisecond window, only the first one should be counted. Theoretically, you can pick up real spikes that are less than 1 millisecond apart if each spike comes from a different neuron, but this is rare and often ignored in practice. Spike-sorting methods would be able to estimate which signals are coming from which neuron, but they are costly to run in real-time and not needed to get an accurate estimate of neural activity ([Trautmann et al 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7002296/)). See [thresholdExtraction.py](https://github.com/brandbci/brand-nsp/blob/main/nodes/thresholdExtraction/thresholdExtraction.py) in `brand-nsp` for an example of how filtering and spike detection is done.

### Spike-band power

Spike-band power is an alternative to threshold crossings that is meant to capture similar activity without the use of thresholds ([Nason et al 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7982996/)). To extract it, filter the data to the spike band (250-5000 Hz or 300-1000 Hz) and square the result. Then, you can either take the log of the resulting signal or leave it as-is. To downsample the signal from 30 kHz to 1 kHz, take the mean power within each 1 ms window. See [bpExtraction.py](https://github.com/brandbci/brand-nsp/blob/main/nodes/bpExtraction/bpExtraction.py) in `brand-nsp` for an example of how spike-band power extraction is done.

## Optimizations

These are implemented in [`notebooks/optimizations.py`](./notebooks/optimizations.py) and used by the processing loop in [`baseline.ipynb`](./notebooks/baseline.ipynb). Lossless options reproduce the baseline spikes and spike-band power. Decimation is off unless you ask for it, because it changes the waveforms. Before-and-after numbers are in [`notebooks/optimization_results.ipynb`](./notebooks/optimization_results.ipynb).

```python
from optimizations import OptimizedProcessor

with OptimizedProcessor(
        n_channels=n_channels,
        reref_params=reref_params,
        thresholds=thresholds,
        reref_groups=reref_groups,
        sample_rate=sample_rate,
        use_gpu="auto",
        decimate=False,
) as processor:
    result = processor.process_recording(raw)

result.spike_events    # (channel, millisecond) for each spike
result.spikes_sparse   # CSR matrix of the same raster
```

Run that from `notebooks/`, which is also where `baseline.ipynb` imports it.

### Lossless

These keep the same spikes and the same spike-band power as the original loop.

- **Block re-reference.** Each array of electrodes keeps a small mixing matrix and reuses it every millisecond. The code does not rebuild a giant matrix of every electrode against every other electrode. If a weight really does mix two arrays, those electrodes stay one block so the result does not change.
- **Faster forward filter on this kind of CPU.** The forward half of the filter is the same Butterworth SciPy uses. On an Intel/AMD CPU with AVX-512, eight electrodes are filtered in one instruction. The loop uses one thread. Running it on many threads at once slowed the reverse-filter step that comes next, so that version was dropped. Other CPUs, and `use_x86=False`, keep SciPy. On the 4-core Xeon used for the notebook, a processor that is already built finishes 1024 channels in 579 ms per second of data. 2048 channels takes about 1.2 s. On the real recording tiled to those sizes the numbers are 585 ms and 1191 ms. The forward filter was the slowest stage before this change. After it, the reverse filter is the slowest stage, and that stage was left alone.
- **Reverse filter as one multiply.** The backward half of the filter used to be a Python loop, one electrode at a time. It is now one multiply across all electrodes. A thread pool around each millisecond was slower than the original loop, so the CPU path does not start one. `per_array_threads` and `per_channel_threads` are still accepted and do not change that path.
- **GPU, when the machine has one.** Install PyTorch to enable it. `use_gpu="auto"` picks an Apple GPU first, then a GPU that shares the computer's memory, then a separate NVIDIA GPU. Shared memory matters because copying 2048 channels across a bus can cost more than the filter. With no GPU, filtering stays on the CPU and matches the original loop. An Apple GPU uses 32-bit numbers, so the last bits can differ.
- **Spikes stored without the zeros.** Most milliseconds have no spike. `spike_events` is a list of `(electrode, millisecond)` for the bins that fired. `spikes_sparse` is the same list in a compressed matrix. The full grid is still available. `crossing_events` keeps every sample that crossed threshold, not just one per millisecond.

### Lossy

- **Keep every other sample.** `decimate=True` low-pass filters at 6 kHz, then drops every other sample, so 30 kHz becomes 15 kHz. The spike band still gets through. Energy that would fold into that band is reduced first. Spike and power frames stay at one per millisecond. This changes the waveforms, so it is off unless you ask for it.

## Results in two tables

Both tables are printed by the last cell of [notebooks/optimization_results.ipynb](./notebooks/optimization_results.ipynb) ("Results at a glance"). Numbers are from the saved run on a 4-core Intel Xeon with AVX-512, in ms of wall time per second of data; under 1000 is real time.

**What worked.** Each stage timed alone at 1024 electrodes, original method versus replacement. Output is identical.

| stage | original | new | saved | how |
| --- | ---: | ---: | ---: | --- |
| Re-reference | 1299 | 57 | 1241 | one 64×64 block per group, built once, instead of rebuilding the full matrix every millisecond |
| Forward filter | 226 | 60 | 167 | the same Butterworth filter compiled for AVX-512, eight electrodes per instruction |
| Reverse filter | 1963 | 160 | 1802 | one matrix multiply instead of a Python loop calling `np.convolve` per electrode |

Combined, steady state: 64 electrodes 48 ms, 256 → 145 ms, 1024 → 579 ms, 2048 → 1181 ms. 1024 is real time; 2048 is about 1.2× too slow, and the reverse filter is now the largest stage.

**What did not work.** Each idea was implemented, measured against the same work without it, and removed. "Added" is how much slower it made the pipeline.

| attempt | electrodes | without | with | added | why |
| --- | ---: | ---: | ---: | ---: | --- |
| Thread pool around each millisecond | 64 | 211 | 458 | +247 | submitting and collecting pool tasks costs more than a millisecond of work; the interpreter lock keeps NumPy from overlapping |
| One Python thread per electrode | 64 | 211 | 1672 | +1461 | the same overhead, once per electrode |
| Fortran-order output buffer, copied back at the end | 1024 | 1086 | 1283 | +197 | the final copy back walks the whole array with a bad stride and costs what the contiguous writes saved |
| Multi-threaded (OpenMP) forward filter next to the multiply | 1024 | 155 | 13953 | +13798 | two thread pools (OpenMP and BLAS) fight over four cores and the multiply stalls |

## Which recording the numbers used

Two recordings. Both are in [notebooks/optimization_results.ipynb](./notebooks/optimization_results.ipynb).

**The real one.** `data/NSP1_aligned.ns6` is a Blackrock recording from a Utah-array implant: 128 electrodes, 30 kHz, about 96 seconds, about 706 MB. It is not stored in git; step 2 above downloads it. Thresholds are −3.5 times each electrode's RMS and the re-reference is a linear regression fit inside each 64-electrode group, both from `calc_params.py` on the first 60 seconds. The first 10 seconds go through the original loop and the fast processor. On this file the two paths return the same 26,611 spikes, with 0 mismatched bins and 0 difference in the filtered voltage and the spike-band power. `baseline.ipynb` runs the same file and plots the spikes.

**The synthetic one.** The real file has 128 electrodes. The 1024- and 2048-electrode timing rows need more, so the notebook also builds a recording in memory: groups of 64 electrodes, Gaussian noise, and a short negative pulse every 10 ms on one electrode. A second timing table stacks copies of the 128 real electrodes to reach 1024 and 2048, so those rows run on real waveforms with the fitted weights and thresholds repeated per copy.

The notebook says which table each number comes from.
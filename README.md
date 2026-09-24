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
in the dataset name are split across several NSPs.

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

## Signal Processing

This section summarizes the standard signal processing steps used on data from microelectrode arrays. For a code example, see [baseline.ipynb](./notebooks/baseline.ipynb).

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

These are implemented in [`notebooks/optimizations.py`](./notebooks/optimizations.py) and used by the processing loop in [`baseline.ipynb`](./notebooks/baseline.ipynb). Lossless options reproduce the baseline spikes and spike-band power. Decimation is off unless you ask for it, because it changes the waveforms.

```python
from optimizations import OptimizedProcessor

with OptimizedProcessor(
        n_channels=n_channels,
        reref_params=reref_params,
        thresholds=thresholds,
        reref_groups=reref_groups,
        sample_rate=sample_rate,
        per_array_threads=True,
        per_channel_threads=True,
        use_gpu="auto",
        decimate=False,
) as processor:
    result = processor.process_recording(raw)

result.spike_events    # (channel, millisecond) for each spike
result.spikes_sparse   # CSR matrix of the same raster
```

Run that from `notebooks/`, which is also where `baseline.ipynb` imports it.

### Lossless

- **One thread per array.** Each re-referencing group is a multi-electrode array. Weights do not cross groups, so each group is re-referenced on its own thread. If a weight matrix does mix groups, those channels stay in one thread so the result does not change.
- **Filtering and feature extraction across channels.** After re-referencing, channels are independent. Each channel is its own task. The pool size defaults to the number of CPUs so a 1 ms frame stays inside the real-time budget; pass `channel_workers=n_channels` for one thread per channel.
- **GPU filtering, preferring unified memory.** Install PyTorch to enable it. `use_gpu="auto"` selects Apple MPS first, then an integrated CUDA GPU, then a discrete CUDA GPU. MPS and integrated GPUs share memory with the CPU, which is the case the design notes call out. The IIR filter is parallel across channels. With no GPU, or without PyTorch, filtering stays on the CPU and matches the baseline loop. CUDA runs the same recurrence in float64. MPS has no float64, so those results can differ in the last bits.
- **Sparse spikes.** `spike_events` is a `(channel, millisecond)` list and `spikes_sparse` is the CSR matrix of the dense raster. Most bins are zero, so this is the form to store or send. Building that list is extra work on the real-time path; leave `store_dense_spikes=True` (the default) when the next step wants the raster in memory. `crossing_events` keeps every sample-level threshold crossing.

### Lossy

- **Decimate 30 kHz to 15 kHz.** `decimate=True` runs a causal order-8 Butterworth low-pass at 6 kHz (0.8 times the new 7.5 kHz Nyquist), then keeps every other sample. The spike band (250-5000 Hz) still passes, and energy that would alias is attenuated before the drop. Feature frames stay at 1 kHz. The acausal lag stays 4 ms, which is 60 samples at 15 kHz. Thresholds are still applied in volts; they were usually estimated at the original rate.
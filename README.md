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

## Potential Optimizations

### Lossless

- Process data from each multi-electrode array in a separate thread
- After re-referencing, run filtering and feature extraction for each channel in a separate thread
- Use GPUs for filtering, particularly on machines with unified memory
- Encode spikes as events or sparse arrays instead of dense arrays. This would save disk space and bandwidth but probably makes real-time processing slower.

### Lossy

- Decimate the raw signal from 30 kHz to 15 kHz by skipping every other sample (must apply low-pass anti-aliasing filter first)
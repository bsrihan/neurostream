//! Python module `neurostream`, built with maturin.
//!
//! ```python
//! import neurostream
//! raw, rate = neurostream.read_ns6("../data/NSP1_aligned.ns6", max_samples=300_000)
//! proc = neurostream.Processor(reref_params, thresholds, groups, threads=1)
//! out = proc.process_recording(raw)          # raw is (n_samples, n_channels) int16
//! out["spikes"], out["filtered"], out["spike_band_power"]
//! ```
//!
//! Arrays going in are time-major, `(n_samples, n_channels)`, the order the
//! file stores them in. Arrays coming out are channel-major views,
//! `(n_channels, n_samples)` and `(n_channels, n_windows)`, so they line up
//! with the Python tree's outputs without a copy: the module returns the
//! time-major buffer and transposes the view.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::params::Params;
use crate::{Config, Processor, StageTimes};

#[pyclass(name = "Processor", unsendable)]
struct PyProcessor {
    inner: Processor,
}

fn to_time_major<'py>(py: Python<'py>, data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data).reshape([rows, cols])?;
    Ok(arr.getattr("T")?)
}

fn to_time_major_i16<'py>(py: Python<'py>, data: Vec<i16>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data).reshape([rows, cols])?;
    Ok(arr.getattr("T")?)
}

#[pymethods]
impl PyProcessor {
    /// `reref_params` is `(n_channels, n_channels)`, `thresholds` is
    /// `(n_channels,)` or `(n_channels, 1)`, `groups` is a list of lists of
    /// channel indices. `threads` is the number of lanes; 1 is single-threaded.
    #[new]
    #[pyo3(signature = (reref_params, thresholds, groups, sample_rate = 30000.0, threads = 1))]
    fn new(
        reref_params: PyReadonlyArray2<f64>,
        thresholds: PyReadonlyArray1<f64>,
        groups: Vec<Vec<usize>>,
        sample_rate: f64,
        threads: usize,
    ) -> PyResult<PyProcessor> {
        let p = reref_params.as_array();
        let n = p.shape()[0];
        if p.shape()[1] != n {
            return Err(PyValueError::new_err("reref_params must be square"));
        }
        let thr = thresholds.as_array();
        if thr.len() != n {
            return Err(PyValueError::new_err(format!("{} thresholds for {} channels", thr.len(), n)));
        }
        let params = Params {
            n_channels: n,
            sample_rate,
            thresholds: thr.to_vec(),
            rereference_parameters: (0..n).map(|i| (0..n).map(|j| p[[i, j]]).collect()).collect(),
            reref_groups: groups,
            rereference: None,
            thresh_mult: None,
        };
        let inner = Processor::new(&params, Config { sample_rate, threads, ..Config::default() }).map_err(PyValueError::new_err)?;
        Ok(PyProcessor { inner })
    }

    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.n_channels
    }

    #[getter]
    fn lanes(&self) -> usize {
        self.inner.lanes()
    }

    #[getter]
    fn samples_per_window(&self) -> usize {
        self.inner.samples_per_window
    }

    /// Delay from input to output, in samples (the 4 ms look-ahead).
    #[getter]
    fn lag_samples(&self) -> usize {
        self.inner.lag_samples()
    }

    /// Process a time-major recording `(n_samples, n_channels)` of int16 or
    /// float64. Returns a dict of channel-major arrays: `filtered`
    /// `(n_channels, n_samples)` float32, `spikes` `(n_channels, n_windows)`
    /// int16, `spike_band_power` `(n_channels, n_windows)` float32, and
    /// `rereferenced` when `keep_rereferenced` is true.
    #[pyo3(signature = (raw, keep_rereferenced = false))]
    fn process_recording<'py>(&mut self, py: Python<'py>, raw: Bound<'py, PyAny>, keep_rereferenced: bool) -> PyResult<Bound<'py, PyDict>> {
        let result = if let Ok(arr) = raw.extract::<PyReadonlyArray2<i16>>() {
            let shape = arr.as_array().shape().to_vec();
            check_shape(&shape, self.inner.n_channels)?;
            let flat = arr.as_slice().map_err(|_| PyValueError::new_err("raw must be C-contiguous (n_samples, n_channels)"))?;
            py.allow_threads(|| self.inner.process_recording(flat, shape[0], keep_rereferenced))
        } else if let Ok(arr) = raw.extract::<PyReadonlyArray2<f64>>() {
            let shape = arr.as_array().shape().to_vec();
            check_shape(&shape, self.inner.n_channels)?;
            let flat = arr.as_slice().map_err(|_| PyValueError::new_err("raw must be C-contiguous (n_samples, n_channels)"))?;
            py.allow_threads(|| self.inner.process_recording(flat, shape[0], keep_rereferenced))
        } else {
            return Err(PyValueError::new_err("raw must be an int16 or float64 array of shape (n_samples, n_channels)"));
        };
        let n = result.n_channels;
        let n_samples = result.n_windows * result.samples_per_window;
        let out = PyDict::new(py);
        out.set_item("filtered", to_time_major(py, result.filtered, n_samples, n)?)?;
        out.set_item("spikes", to_time_major_i16(py, result.spikes, result.n_windows, n)?)?;
        out.set_item("spike_band_power", to_time_major(py, result.spike_band_power, result.n_windows, n)?)?;
        if keep_rereferenced {
            out.set_item("rereferenced", to_time_major(py, result.rereferenced, n_samples, n)?)?;
        }
        out.set_item("n_windows", result.n_windows)?;
        Ok(out)
    }

    /// Run the recording with the clock stopped between stages (single lane
    /// only). Returns seconds spent in each stage over the whole recording.
    fn profile_recording<'py>(&mut self, py: Python<'py>, raw: PyReadonlyArray2<i16>) -> PyResult<Bound<'py, PyDict>> {
        let shape = raw.as_array().shape().to_vec();
        check_shape(&shape, self.inner.n_channels)?;
        if self.inner.lanes() != 1 {
            return Err(PyValueError::new_err("profile_recording needs threads=1"));
        }
        let flat = raw.as_slice().map_err(|_| PyValueError::new_err("raw must be C-contiguous"))?;
        let n = self.inner.n_channels;
        let spw = self.inner.samples_per_window;
        let n_windows = shape[0] / spw;
        let mut filtered = vec![0.0f32; n_windows * spw * n];
        let mut spikes = vec![0i16; n_windows * n];
        let mut sbp = vec![0.0f32; n_windows * n];
        let times: StageTimes = py.allow_threads(|| self.inner.profile_recording(flat, shape[0], &mut filtered, &mut spikes, &mut sbp));
        let out = PyDict::new(py);
        out.set_item("reref", times.reref)?;
        out.set_item("forward", times.forward)?;
        out.set_item("reverse", times.reverse)?;
        out.set_item("features", times.features)?;
        out.set_item("write", times.write)?;
        Ok(out)
    }
}

fn check_shape(shape: &[usize], n_channels: usize) -> PyResult<()> {
    if shape.len() != 2 || shape[1] != n_channels {
        return Err(PyValueError::new_err(format!("raw must be (n_samples, {n_channels}), got {shape:?}")));
    }
    Ok(())
}

/// Read a Blackrock `.ns6` file. Returns `(samples, sample_rate)` with
/// `samples` time-major `(n_samples, n_channels)` int16, straight from disk.
#[pyfunction]
#[pyo3(signature = (path, max_samples = None))]
fn read_ns6<'py>(py: Python<'py>, path: &str, max_samples: Option<usize>) -> PyResult<(Bound<'py, PyArray2<i16>>, f64)> {
    let rec = py
        .allow_threads(|| crate::nsx::read_samples(std::path::Path::new(path), max_samples))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = rec.header.n_channels;
    let rate = rec.header.sample_rate;
    let arr = rec.samples.into_pyarray(py).reshape([rec.n_samples, n])?;
    Ok((arr, rate))
}

/// Filter coefficients the Rust design produces, for checking against SciPy:
/// `(sos, zi, reverse_window)`.
#[pyfunction]
#[pyo3(signature = (order = 4, low_hz = 250.0, high_hz = 5000.0, sample_rate = 30000.0, lag_s = 0.004))]
fn filter_design<'py>(
    py: Python<'py>,
    order: usize,
    low_hz: f64,
    high_hz: f64,
    sample_rate: f64,
    lag_s: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let sos = crate::design::butter_bandpass(order, low_hz, high_hz, sample_rate);
    let zi = crate::design::sosfilt_zi(&sos);
    let lag = (lag_s * sample_rate).round() as usize;
    let win = crate::design::impulse_response(&sos, lag + 1);
    let sos_flat: Vec<f64> = sos.iter().flat_map(|s| s.iter().copied()).collect();
    let zi_flat: Vec<f64> = zi.iter().flat_map(|z| z.iter().copied()).collect();
    Ok((
        PyArray1::from_vec(py, sos_flat).reshape([sos.len(), 6])?,
        PyArray1::from_vec(py, zi_flat).reshape([zi.len(), 2])?,
        PyArray1::from_vec(py, win),
    ))
}

/// SIMD width this build was compiled for.
#[pyfunction]
fn build_target() -> &'static str {
    crate::build_target()
}

#[pymodule]
fn neurostream(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProcessor>()?;
    m.add_function(wrap_pyfunction!(read_ns6, m)?)?;
    m.add_function(wrap_pyfunction!(filter_design, m)?)?;
    m.add_function(wrap_pyfunction!(build_target, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

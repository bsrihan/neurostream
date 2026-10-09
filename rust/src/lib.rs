//! neurostream in Rust: the millisecond neural pipeline.
//!
//! Each electrode of a microelectrode array is sampled 30,000 times a second.
//! Once per millisecond, for every electrode, this crate returns whether a
//! spike happened and one spike-band power number. The steps are the same as
//! the Python tree in `../original`:
//!
//! 1. **Re-reference** ([`reref`]): subtract the shared part of each group of
//!    electrodes, `y = (I - P) x`, with one small matrix per group.
//! 2. **Forward filter** ([`filter::ForwardSos`]): Butterworth order 4,
//!    250-5000 Hz, state carried from one millisecond to the next.
//! 3. **Reverse filter** ([`filter::ReverseFir`]): the same filter backward
//!    over a 4 ms look-ahead, so spikes are not shifted in time. One matrix
//!    multiply. The output is therefore 4 ms behind the input.
//! 4. **Features** ([`features`]): negative threshold crossing (at most one per
//!    millisecond) and mean `10 log10(x^2)`.
//!
//! Everything is time-major: a window is `n_samples x n_channels` with
//! channels contiguous. That is the order Blackrock files store samples in,
//! so a recording streams straight from disk into the pipeline, and it lets
//! the forward filter and the feature loops walk across electrodes in the
//! inner loop, which the compiler turns into wide SIMD instructions.
//!
//! The processor is split into [`Lane`]s, each owning a contiguous range of
//! channels and all of its buffers and filter state. With one lane the
//! pipeline is single-threaded. With several, each window is processed by all
//! lanes at once on a rayon pool; because every lane owns its memory there is
//! no shared state to lock and no cross-thread writes.

pub mod design;
pub mod features;
pub mod filter;
pub mod nsx;
pub mod params;
pub mod reref;

#[cfg(feature = "python")]
mod python;

use std::time::Instant;

use rayon::prelude::*;

use crate::design::{butter_bandpass, impulse_response, sosfilt_zi};
use crate::filter::{ForwardSos, ReverseFir};
use crate::params::Params;
use crate::reref::Reref;

/// Filter settings. The defaults are the ones every notebook uses.
#[derive(Debug, Clone)]
pub struct Config {
    pub sample_rate: f64,
    pub order: usize,
    pub low_hz: f64,
    pub high_hz: f64,
    /// Look-ahead of the reverse filter, in seconds (4 ms).
    pub lag_s: f64,
    /// Number of lanes / threads. 1 means no thread pool at all.
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { sample_rate: 30000.0, order: 4, low_hz: 250.0, high_hz: 5000.0, lag_s: 0.004, threads: 1 }
    }
}

/// Everything one contiguous range of channels needs, owned outright.
pub struct Lane {
    pub c0: usize,
    pub c1: usize,
    reref: Reref,
    forward: ForwardSos,
    reverse: ReverseFir,
    thresholds: Vec<f64>,
    /// Time-major `n_samples x width` buffers.
    pub raw: Vec<f64>,
    pub rereferenced: Vec<f64>,
    forward_out: Vec<f64>,
    pub filtered: Vec<f32>,
    scratch: Vec<f64>,
    pub spikes: Vec<i16>,
    pub spike_band_power: Vec<f32>,
}

impl Lane {
    fn width(&self) -> usize {
        self.c1 - self.c0
    }

    /// Re-reference, forward filter, reverse filter, features, for one window
    /// already copied into `self.raw`.
    fn run(&mut self, n_samples: usize) {
        let w = self.width();
        self.reref.apply(&self.raw, &mut self.rereferenced, n_samples);
        self.forward_out.copy_from_slice(&self.rereferenced);
        self.forward.apply(&mut self.forward_out, n_samples);
        self.reverse.push(&self.forward_out, 0, w);
        self.reverse.finish_push();
        self.reverse.apply_range(&mut self.filtered, &mut self.scratch, 0, w);
        features::features_range(
            &self.filtered,
            n_samples,
            w,
            &self.thresholds,
            &mut self.spikes,
            &mut self.spike_band_power,
            0,
            w,
        );
    }
}

/// Stage durations in seconds, accumulated over a profiled run.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimes {
    pub reref: f64,
    pub forward: f64,
    pub reverse: f64,
    pub features: f64,
    pub write: f64,
}

/// One recording's outputs, time-major.
#[derive(Debug, Clone)]
pub struct RecordingResult {
    pub n_channels: usize,
    pub n_windows: usize,
    pub samples_per_window: usize,
    /// `n_windows * samples_per_window x n_channels`.
    pub filtered: Vec<f32>,
    pub rereferenced: Vec<f32>,
    /// `n_windows x n_channels`, 0 or 1.
    pub spikes: Vec<i16>,
    pub spike_band_power: Vec<f32>,
}

pub struct Processor {
    pub n_channels: usize,
    pub samples_per_window: usize,
    pub config: Config,
    lanes: Vec<Lane>,
    pool: Option<rayon::ThreadPool>,
}

impl Processor {
    pub fn new(params: &Params, config: Config) -> Result<Processor, String> {
        params.validate()?;
        let n = params.n_channels;
        let spw = (config.sample_rate / 1000.0).round() as usize;
        let lag = (config.lag_s * config.sample_rate).round() as usize;
        let sos = butter_bandpass(config.order, config.low_hz, config.high_hz, config.sample_rate);
        let zi = sosfilt_zi(&sos);
        let win = impulse_response(&sos, lag + 1);

        let whole = Reref::new(params);
        let ranges = lane_ranges(&whole, n, config.threads.max(1));
        let mut lanes = Vec::with_capacity(ranges.len());
        for (c0, c1) in ranges {
            let w = c1 - c0;
            // Blocks inside this lane, re-indexed relative to c0.
            let blocks = whole
                .blocks
                .iter()
                .filter(|b| b.channels[0] >= c0 && b.channels[0] < c1)
                .map(|b| reref::Block {
                    channels: b.channels.iter().map(|c| c - c0).collect(),
                    contiguous: b.contiguous.map(|f| f - c0),
                    mat: b.mat.clone(),
                })
                .collect();
            lanes.push(Lane {
                c0,
                c1,
                reref: Reref { n_channels: w, blocks },
                forward: ForwardSos::new(&sos, &zi, w),
                reverse: ReverseFir::new(&win, spw, w),
                thresholds: params.thresholds[c0..c1].to_vec(),
                raw: vec![0.0; spw * w],
                rereferenced: vec![0.0; spw * w],
                forward_out: vec![0.0; spw * w],
                filtered: vec![0.0; spw * w],
                scratch: vec![0.0; spw * w],
                spikes: vec![0; w],
                spike_band_power: vec![0.0; w],
            });
        }
        let pool = if lanes.len() > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(lanes.len())
                    .build()
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        Ok(Processor { n_channels: n, samples_per_window: spw, config, lanes, pool })
    }

    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Samples of delay between input and output (the 4 ms look-ahead).
    pub fn lag_samples(&self) -> usize {
        self.lanes[0].reverse.lag()
    }

    /// Copy one time-major window (`samples_per_window x n_channels`) into
    /// the lanes' input buffers, converting to `f64`.
    fn scatter<T: Copy + Into<f64>>(&mut self, raw: &[T]) {
        let n = self.n_channels;
        let spw = self.samples_per_window;
        debug_assert_eq!(raw.len(), spw * n);
        for lane in &mut self.lanes {
            let (c0, c1, w) = (lane.c0, lane.c1, lane.width());
            for s in 0..spw {
                let dst = &mut lane.raw[s * w..(s + 1) * w];
                let src = &raw[s * n + c0..s * n + c1];
                for (d, v) in dst.iter_mut().zip(src.iter()) {
                    *d = (*v).into();
                }
            }
        }
    }

    fn run_lanes(&mut self) {
        let spw = self.samples_per_window;
        match &self.pool {
            None => {
                for lane in &mut self.lanes {
                    lane.run(spw);
                }
            }
            Some(pool) => {
                let lanes = &mut self.lanes;
                pool.install(|| lanes.par_iter_mut().for_each(|lane| lane.run(spw)));
            }
        }
    }

    /// Copy this window's outputs into time-major destination slices:
    /// `filtered`/`rereferenced` are `samples_per_window x n_channels`,
    /// `spikes`/`sbp` are `n_channels`.
    fn gather(&self, filtered: &mut [f32], rereferenced: Option<&mut [f32]>, spikes: &mut [i16], sbp: &mut [f32]) {
        let n = self.n_channels;
        let spw = self.samples_per_window;
        let mut rereferenced = rereferenced;
        for lane in &self.lanes {
            let (c0, c1, w) = (lane.c0, lane.c1, lane.width());
            for s in 0..spw {
                filtered[s * n + c0..s * n + c1].copy_from_slice(&lane.filtered[s * w..(s + 1) * w]);
                if let Some(r) = rereferenced.as_deref_mut() {
                    let dst = &mut r[s * n + c0..s * n + c1];
                    for (d, v) in dst.iter_mut().zip(lane.rereferenced[s * w..(s + 1) * w].iter()) {
                        *d = *v as f32;
                    }
                }
            }
            spikes[c0..c1].copy_from_slice(&lane.spikes);
            sbp[c0..c1].copy_from_slice(&lane.spike_band_power);
        }
    }

    /// Process one millisecond. `raw` is time-major `samples_per_window x
    /// n_channels`. Outputs are written to the given slices (see [`Self::gather`]).
    pub fn process_window<T: Copy + Into<f64>>(
        &mut self,
        raw: &[T],
        filtered: &mut [f32],
        rereferenced: Option<&mut [f32]>,
        spikes: &mut [i16],
        sbp: &mut [f32],
    ) {
        self.scatter(raw);
        self.run_lanes();
        self.gather(filtered, rereferenced, spikes, sbp);
    }

    /// Process a whole time-major recording (`n_samples x n_channels`),
    /// dropping any samples after the last full millisecond.
    pub fn process_recording<T: Copy + Into<f64>>(&mut self, raw: &[T], n_samples: usize, keep_rereferenced: bool) -> RecordingResult {
        let n = self.n_channels;
        let spw = self.samples_per_window;
        assert_eq!(raw.len(), n_samples * n, "raw must be n_samples x n_channels");
        let n_windows = n_samples / spw;
        let mut filtered = vec![0.0f32; n_windows * spw * n];
        let mut rereferenced = if keep_rereferenced { vec![0.0f32; n_windows * spw * n] } else { Vec::new() };
        let mut spikes = vec![0i16; n_windows * n];
        let mut sbp = vec![0.0f32; n_windows * n];
        for w in 0..n_windows {
            let window = &raw[w * spw * n..(w + 1) * spw * n];
            let filt = &mut filtered[w * spw * n..(w + 1) * spw * n];
            let reref = if keep_rereferenced { Some(&mut rereferenced[w * spw * n..(w + 1) * spw * n]) } else { None };
            self.scatter(window);
            self.run_lanes();
            self.gather(filt, reref, &mut spikes[w * n..(w + 1) * n], &mut sbp[w * n..(w + 1) * n]);
        }
        RecordingResult { n_channels: n, n_windows, samples_per_window: spw, filtered, rereferenced, spikes, spike_band_power: sbp }
    }

    /// Same loop as [`Self::process_recording`] with the clock stopped between
    /// stages. Single lane only (the stages of several lanes overlap in time).
    /// Writes go to the caller's pre-allocated outputs so the write stage
    /// measures the copy, not first-touch page faults.
    pub fn profile_recording<T: Copy + Into<f64>>(
        &mut self,
        raw: &[T],
        n_samples: usize,
        filtered: &mut [f32],
        spikes: &mut [i16],
        sbp: &mut [f32],
    ) -> StageTimes {
        assert_eq!(self.lanes.len(), 1, "profile_recording needs a single-lane processor");
        let n = self.n_channels;
        let spw = self.samples_per_window;
        let n_windows = n_samples / spw;
        let mut times = StageTimes::default();
        for w in 0..n_windows {
            let window = &raw[w * spw * n..(w + 1) * spw * n];
            self.scatter(window);
            let lane = &mut self.lanes[0];
            let t0 = Instant::now();
            lane.reref.apply(&lane.raw, &mut lane.rereferenced, spw);
            let t1 = Instant::now();
            lane.forward_out.copy_from_slice(&lane.rereferenced);
            lane.forward.apply(&mut lane.forward_out, spw);
            let t2 = Instant::now();
            lane.reverse.push(&lane.forward_out, 0, n);
            lane.reverse.finish_push();
            lane.reverse.apply_range(&mut lane.filtered, &mut lane.scratch, 0, n);
            let t3 = Instant::now();
            features::features_range(&lane.filtered, spw, n, &lane.thresholds, &mut lane.spikes, &mut lane.spike_band_power, 0, n);
            let t4 = Instant::now();
            filtered[w * spw * n..(w + 1) * spw * n].copy_from_slice(&lane.filtered);
            spikes[w * n..(w + 1) * n].copy_from_slice(&lane.spikes);
            sbp[w * n..(w + 1) * n].copy_from_slice(&lane.spike_band_power);
            let t5 = Instant::now();
            times.reref += (t1 - t0).as_secs_f64();
            times.forward += (t2 - t1).as_secs_f64();
            times.reverse += (t3 - t2).as_secs_f64();
            times.features += (t4 - t3).as_secs_f64();
            times.write += (t5 - t4).as_secs_f64();
        }
        times
    }
}

/// Split the channels into `threads` contiguous ranges along re-reference
/// block boundaries. Falls back to one range when a block is not contiguous
/// or there are fewer blocks than threads.
fn lane_ranges(reref: &Reref, n_channels: usize, threads: usize) -> Vec<(usize, usize)> {
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    for b in &reref.blocks {
        match b.contiguous {
            Some(first) => blocks.push((first, first + b.channels.len())),
            None => return vec![(0, n_channels)],
        }
    }
    blocks.sort_unstable();
    if threads <= 1 || blocks.len() < threads {
        return vec![(0, n_channels)];
    }
    // Assign blocks to lanes so each lane carries about the same channel count.
    let target = (n_channels as f64) / threads as f64;
    let mut ranges = Vec::with_capacity(threads);
    let mut start = blocks[0].0;
    let mut count = 0usize;
    for (i, &(b0, b1)) in blocks.iter().enumerate() {
        count += b1 - b0;
        let lanes_left = threads - ranges.len();
        let last_block = i + 1 == blocks.len();
        if last_block || (lanes_left > 1 && count as f64 >= target * 0.9) {
            ranges.push((start, b1));
            if !last_block {
                start = blocks[i + 1].0;
            }
            count = 0;
        }
    }
    if ranges.len() < 2 {
        return vec![(0, n_channels)];
    }
    ranges
}

/// Human-readable description of the SIMD width this build was compiled for.
pub fn build_target() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "x86_64 with AVX-512 (eight f64 per instruction)"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 with AVX2 (four f64 per instruction)"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 with NEON (two f64 per instruction)"
    } else {
        "generic target"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n_channels: usize, n_ms: usize) -> Vec<f64> {
        let spw = 30;
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng % 4001) as f64 - 2000.0) / 50.0
        };
        let mut raw = vec![0.0; n_ms * spw * n_channels];
        for v in raw.iter_mut() {
            *v = next();
        }
        // A sharp negative pulse on one channel every 10 ms.
        let mut ms = 8;
        while ms < n_ms {
            let c = (ms / 10) % n_channels;
            let s = ms * spw + 12;
            raw[s * n_channels + c] = -500.0;
            raw[(s + 1) * n_channels + c] = -500.0;
            ms += 10;
        }
        raw
    }

    #[test]
    fn four_lanes_match_one_lane() {
        let n = 256;
        let n_ms = 40;
        let raw = synthetic(n, n_ms);
        let params = Params::common_average(n, 64, -120.0, 30000.0);
        let mut one = Processor::new(&params, Config { threads: 1, ..Config::default() }).unwrap();
        let mut four = Processor::new(&params, Config { threads: 4, ..Config::default() }).unwrap();
        assert_eq!(four.lanes(), 4);
        let a = one.process_recording(&raw, n_ms * 30, true);
        let b = four.process_recording(&raw, n_ms * 30, true);
        assert_eq!(a.spikes, b.spikes);
        for (x, y) in a.filtered.iter().zip(b.filtered.iter()) {
            assert!((x.is_nan() && y.is_nan()) || x == y);
        }
        assert!(a.spikes.iter().map(|s| *s as i32).sum::<i32>() > 0);
    }

    #[test]
    fn first_four_milliseconds_are_warmup() {
        let n = 8;
        let raw = synthetic(n, 10);
        let params = Params::common_average(n, 8, -120.0, 30000.0);
        let mut proc = Processor::new(&params, Config::default()).unwrap();
        let out = proc.process_recording(&raw, 300, false);
        assert!(out.filtered[..4 * 30 * n].iter().all(|v| v.is_nan()));
        assert!(out.filtered[4 * 30 * n..].iter().all(|v| v.is_finite()));
        assert!(out.spikes[..4 * n].iter().all(|s| *s == 0));
    }
}

//! Per-recording parameters: thresholds, re-reference weights, and groups.
//!
//! These come from the Python tree's `calc_params.py`, which writes a JSON
//! file after looking at the first minute of a recording. Both trees read the
//! same file so they start from identical thresholds and weights.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

/// Thresholds and re-reference weights for one recording.
#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub n_channels: usize,
    pub sample_rate: f64,
    /// One voltage threshold per channel, in the recording's units. Spikes
    /// are negative-going crossings, so these are negative numbers.
    pub thresholds: Vec<f64>,
    /// Row `i` holds the weights of the other channels that are subtracted
    /// from channel `i`: `y = (I - P) x`. Common average puts `1/n` in every
    /// entry of a group's block; linear regression puts fitted weights there.
    pub rereference_parameters: Vec<Vec<f64>>,
    /// Channel indices of each re-reference group (64 electrodes per array).
    pub reref_groups: Vec<Vec<usize>>,
    #[serde(default)]
    pub rereference: Option<String>,
    #[serde(default)]
    pub thresh_mult: Option<f64>,
}

impl Params {
    pub fn from_json_file(path: &Path) -> io::Result<Params> {
        let text = fs::read_to_string(path)?;
        let params: Params = serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        params.validate().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(params)
    }

    /// Common-average parameters for `n_channels` in contiguous groups of
    /// `group_size`, with no file. Used by the synthetic benchmarks.
    pub fn common_average(n_channels: usize, group_size: usize, threshold: f64, sample_rate: f64) -> Params {
        let mut reref = vec![vec![0.0; n_channels]; n_channels];
        let mut groups = Vec::new();
        let mut start = 0;
        while start < n_channels {
            let stop = (start + group_size).min(n_channels);
            let weight = 1.0 / (stop - start) as f64;
            for i in start..stop {
                for j in start..stop {
                    reref[i][j] = weight;
                }
            }
            groups.push((start..stop).collect());
            start = stop;
        }
        Params {
            n_channels,
            sample_rate,
            thresholds: vec![threshold; n_channels],
            rereference_parameters: reref,
            reref_groups: groups,
            rereference: Some("car".to_string()),
            thresh_mult: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_channels;
        if self.thresholds.len() != n {
            return Err(format!("{} thresholds for {} channels", self.thresholds.len(), n));
        }
        if self.rereference_parameters.len() != n || self.rereference_parameters.iter().any(|r| r.len() != n) {
            return Err("rereference_parameters must be n_channels x n_channels".into());
        }
        let mut seen = vec![false; n];
        for g in &self.reref_groups {
            for &c in g {
                if c >= n {
                    return Err(format!("group channel {c} outside 0..{n}"));
                }
                if seen[c] {
                    return Err(format!("channel {c} appears in two groups"));
                }
                seen[c] = true;
            }
        }
        Ok(())
    }
}

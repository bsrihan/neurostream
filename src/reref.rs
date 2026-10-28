//! Re-reference: subtract the shared part of each electrode group.
//!
//! The math is `y = (I - P) x`. Instead of one matrix over every electrode
//! against every other, each group keeps its own small `(I - P)` block, built
//! once. Applying a block to one millisecond is one small matrix multiply.
//! Windows are time-major, `x[s * n_channels + c]`, so a group of contiguous
//! channels is a strided sub-matrix and needs no copy.

use crate::params::Params;

/// One re-reference block: the channels it covers and its `(I - P)` matrix.
#[derive(Debug, Clone)]
pub struct Block {
    pub channels: Vec<usize>,
    /// `Some(first)` when `channels` is the range `first..first + len`.
    pub contiguous: Option<usize>,
    /// Row-major `len x len`, `mat[i * len + j]` multiplies channel `j` into
    /// output channel `i` (both indices inside the block).
    pub mat: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Reref {
    pub n_channels: usize,
    pub blocks: Vec<Block>,
}

impl Reref {
    /// Build the blocks from `params`. Channels missing from every group get
    /// a group of their own. If a weight crosses between groups, the groups
    /// cannot be separated, so one block over all channels is used instead;
    /// the result is the same either way.
    pub fn new(params: &Params) -> Reref {
        let n = params.n_channels;
        let p = &params.rereference_parameters;
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut seen = vec![false; n];
        for g in &params.reref_groups {
            if g.is_empty() {
                continue;
            }
            let mut sorted = g.clone();
            sorted.sort_unstable();
            for &c in &sorted {
                seen[c] = true;
            }
            groups.push(sorted);
        }
        let missing: Vec<usize> = (0..n).filter(|c| !seen[*c]).collect();
        if !missing.is_empty() {
            groups.push(missing);
        }
        if groups.is_empty() || !independent(p, &groups, n) {
            groups = vec![(0..n).collect()];
        }
        let blocks = groups
            .into_iter()
            .map(|channels| {
                let len = channels.len();
                let mut mat = vec![0.0; len * len];
                for (i, &ci) in channels.iter().enumerate() {
                    for (j, &cj) in channels.iter().enumerate() {
                        let identity = if i == j { 1.0 } else { 0.0 };
                        mat[i * len + j] = identity - p[ci][cj];
                    }
                }
                let contiguous = channels.windows(2).all(|w| w[1] == w[0] + 1).then(|| channels[0]);
                Block { channels, contiguous, mat }
            })
            .collect();
        Reref { n_channels: n, blocks }
    }

    /// Re-reference `n_samples` time-major samples from `x` into `y`, for the
    /// blocks in `block_range` only (so threads can split the work).
    pub fn apply_blocks(&self, x: &[f64], y: &mut [f64], n_samples: usize, block_range: std::ops::Range<usize>) {
        let n = self.n_channels;
        debug_assert_eq!(x.len(), n_samples * n);
        debug_assert_eq!(y.len(), n_samples * n);
        for block in &self.blocks[block_range] {
            let g = block.channels.len();
            if let Some(first) = block.contiguous {
                // SAFETY: strides keep every access inside x and y, which are
                // n_samples x n_channels, and first + g <= n_channels.
                unsafe {
                    matrixmultiply::dgemm(
                        n_samples,
                        g,
                        g,
                        1.0,
                        x.as_ptr().add(first),
                        n as isize,
                        1,
                        block.mat.as_ptr(),
                        1,
                        g as isize,
                        0.0,
                        y.as_mut_ptr().add(first),
                        n as isize,
                        1,
                    );
                }
            } else {
                // Scattered channels: gather, multiply, scatter.
                let mut gathered = vec![0.0; n_samples * g];
                for s in 0..n_samples {
                    for (j, &c) in block.channels.iter().enumerate() {
                        gathered[s * g + j] = x[s * n + c];
                    }
                }
                let mut out = vec![0.0; n_samples * g];
                unsafe {
                    matrixmultiply::dgemm(
                        n_samples,
                        g,
                        g,
                        1.0,
                        gathered.as_ptr(),
                        g as isize,
                        1,
                        block.mat.as_ptr(),
                        1,
                        g as isize,
                        0.0,
                        out.as_mut_ptr(),
                        g as isize,
                        1,
                    );
                }
                for s in 0..n_samples {
                    for (i, &c) in block.channels.iter().enumerate() {
                        y[s * n + c] = out[s * g + i];
                    }
                }
            }
        }
    }

    pub fn apply(&self, x: &[f64], y: &mut [f64], n_samples: usize) {
        self.apply_blocks(x, y, n_samples, 0..self.blocks.len());
    }
}

fn independent(p: &[Vec<f64>], groups: &[Vec<usize>], n: usize) -> bool {
    let mut group_of = vec![usize::MAX; n];
    for (gi, g) in groups.iter().enumerate() {
        for &c in g {
            group_of[c] = gi;
        }
    }
    for i in 0..n {
        for j in 0..n {
            if p[i][j] != 0.0 && group_of[i] != group_of[j] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_multiply_equals_full_matrix() {
        let n = 6;
        let params = Params::common_average(n, 3, -50.0, 30000.0);
        let reref = Reref::new(&params);
        assert_eq!(reref.blocks.len(), 2);
        let n_samples = 4;
        let x: Vec<f64> = (0..n_samples * n).map(|i| (i as f64 * 0.37).sin() * 100.0).collect();
        let mut y = vec![0.0; n_samples * n];
        reref.apply(&x, &mut y, n_samples);
        for s in 0..n_samples {
            for i in 0..n {
                let mut full = x[s * n + i];
                for j in 0..n {
                    full -= params.rereference_parameters[i][j] * x[s * n + j];
                }
                assert!((y[s * n + i] - full).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cross_group_weight_falls_back_to_one_block() {
        let mut params = Params::common_average(4, 2, -50.0, 30000.0);
        params.rereference_parameters[0][3] = 0.1;
        let reref = Reref::new(&params);
        assert_eq!(reref.blocks.len(), 1);
        assert_eq!(reref.blocks[0].channels, vec![0, 1, 2, 3]);
    }
}

//! The two halves of the acausal 250-5000 Hz filter.
//!
//! Forward: the Butterworth cascade, one state per section per electrode,
//! run on a time-major window so the inner loop walks across electrodes and
//! the compiler can vectorize it (eight electrodes per AVX-512 instruction).
//!
//! Reverse: the same filter run backward over a 4 ms look-ahead buffer, which
//! cancels the phase shift of the forward pass. Expressed as one matrix
//! multiply `W (30 x 150) @ buffer (150 x channels)`, where `W` is the
//! impulse response laid out as a band. While the buffer still holds its
//! NaN start-up rows, the direct sum is used instead so that NaN only reaches
//! the outputs whose taps touch it, exactly like the Python tree.

use crate::design::Section;

/// Forward Butterworth cascade with per-electrode state.
#[derive(Debug, Clone)]
pub struct ForwardSos {
    /// Per section `[b0, b1, b2, a1, a2]`, normalized by `a0`.
    coeffs: Vec<[f64; 5]>,
    n_channels: usize,
    /// `z1[section * n_channels + c]` and the same for `z2`.
    z1: Vec<f64>,
    z2: Vec<f64>,
}

impl ForwardSos {
    /// `zi` is the per-section starting state, copied onto every electrode.
    pub fn new(sos: &[Section], zi: &[[f64; 2]], n_channels: usize) -> ForwardSos {
        let coeffs = sos
            .iter()
            .map(|s| {
                let inv = 1.0 / s[3];
                [s[0] * inv, s[1] * inv, s[2] * inv, s[4] * inv, s[5] * inv]
            })
            .collect();
        let mut z1 = vec![0.0; sos.len() * n_channels];
        let mut z2 = vec![0.0; sos.len() * n_channels];
        for (section, state) in zi.iter().enumerate() {
            z1[section * n_channels..(section + 1) * n_channels].fill(state[0]);
            z2[section * n_channels..(section + 1) * n_channels].fill(state[1]);
        }
        ForwardSos { coeffs, n_channels, z1, z2 }
    }

    /// Filter `window` (time-major, `n_samples x n_channels`) in place, for
    /// channels `c0..c1` only. The state is carried to the next call.
    pub fn apply_range(&mut self, window: &mut [f64], n_samples: usize, c0: usize, c1: usize) {
        let n = self.n_channels;
        debug_assert_eq!(window.len(), n_samples * n);
        for (section, coeff) in self.coeffs.iter().enumerate() {
            let [b0, b1, b2, a1, a2] = *coeff;
            let z1 = &mut self.z1[section * n + c0..section * n + c1];
            let z2 = &mut self.z2[section * n + c0..section * n + c1];
            for s in 0..n_samples {
                let row = &mut window[s * n + c0..s * n + c1];
                // Direct form II transposed, same recurrence as scipy.signal.sosfilt.
                for ((x, z1), z2) in row.iter_mut().zip(z1.iter_mut()).zip(z2.iter_mut()) {
                    let xn = *x;
                    let yn = b0 * xn + *z1;
                    *z1 = b1 * xn - a1 * yn + *z2;
                    *z2 = b2 * xn - a2 * yn;
                    *x = yn;
                }
            }
        }
    }

    pub fn apply(&mut self, window: &mut [f64], n_samples: usize) {
        self.apply_range(window, n_samples, 0, self.n_channels)
    }
}

/// Reverse FIR over the look-ahead buffer.
#[derive(Debug, Clone)]
pub struct ReverseFir {
    n_channels: usize,
    /// Samples per output window (30 at 30 kHz).
    n_out: usize,
    /// Buffer rows: `lag + n_out` (150 at 30 kHz).
    buffer_len: usize,
    /// Impulse response, `lag + 1` taps.
    win: Vec<f64>,
    /// `win` with `PAD` zeros on each side, for the tiled kernels.
    win_padded: Vec<f64>,
    /// `band[i * buffer_len + r] = win[r - i]` for `0 <= r - i < taps`, else 0.
    band: Vec<f64>,
    /// Time-major `buffer_len x n_channels`. Starts as NaN; real samples
    /// shift in from the bottom, 30 rows per millisecond.
    pub buffer: Vec<f64>,
    /// How many leading rows are still NaN.
    leading_nans: usize,
    /// Physical index of the oldest row.
    head: usize,
}

impl ReverseFir {
    pub fn new(win: &[f64], n_out: usize, n_channels: usize) -> ReverseFir {
        let taps = win.len();
        let buffer_len = taps - 1 + n_out;
        let mut band = vec![0.0; n_out * buffer_len];
        for i in 0..n_out {
            for t in 0..taps {
                band[i * buffer_len + i + t] = win[t];
            }
        }
        let mut win_padded = vec![0.0f64; taps + 2 * PAD];
        win_padded[PAD..PAD + taps].copy_from_slice(win);
        ReverseFir {
            n_channels,
            n_out,
            buffer_len,
            win: win.to_vec(),
            win_padded,
            band,
            buffer: vec![f64::NAN; buffer_len * n_channels],
            leading_nans: buffer_len,
            head: 0,
        }
    }

    pub fn lag(&self) -> usize {
        self.win.len() - 1
    }

    /// Store one window of forward-filtered samples (time-major `n_out x
    /// n_channels`) over the oldest rows of the ring buffer. The forward
    /// output is rounded to `f32` first, because the Python tree keeps this
    /// buffer in float32; both trees then see the same values.
    ///
    /// The buffer is a ring: `head` is the physical index of the oldest row,
    /// and logical row `r` lives at `(head + r) % buffer_len`. Overwriting
    /// the oldest rows replaces the shift-everything-up copy the Python tree
    /// does each millisecond.
    pub fn push(&mut self, forward: &[f64], c0: usize, c1: usize) {
        let n = self.n_channels;
        for s in 0..self.n_out {
            let phys = (self.head + s) % self.buffer_len;
            let dst = &mut self.buffer[phys * n + c0..phys * n + c1];
            let src = &forward[s * n + c0..s * n + c1];
            for (d, v) in dst.iter_mut().zip(src.iter()) {
                *d = (*v as f32) as f64;
            }
        }
    }

    /// Call once per window after every channel range has been pushed.
    pub fn finish_push(&mut self) {
        self.head = (self.head + self.n_out) % self.buffer_len;
        self.leading_nans = self.leading_nans.saturating_sub(self.n_out);
    }

    /// Logical row `r` (0 = oldest) of the buffer, channels `c0..c1`.
    pub fn row(&self, r: usize, c0: usize, c1: usize) -> &[f64] {
        let phys = (self.head + r) % self.buffer_len;
        &self.buffer[phys * self.n_channels + c0..phys * self.n_channels + c1]
    }

    /// Write the filtered window (time-major `n_out x n_channels`, as `f32`)
    /// for channels `c0..c1`. `scratch` must hold `n_out * (c1 - c0)` values.
    ///
    /// Once the buffer is free of NaN rows the work is the band multiply.
    /// `use_gemm` picks the general matrix multiply; otherwise the tiled
    /// kernel below, which knows the band is a sliding window and keeps a
    /// block of outputs in registers.
    pub fn apply_range(&self, out: &mut [f32], scratch: &mut [f64], c0: usize, c1: usize) {
        self.apply_range_with(out, scratch, c0, c1, false)
    }

    pub fn apply_range_with(&self, out: &mut [f32], scratch: &mut [f64], c0: usize, c1: usize, use_gemm: bool) {
        let n = self.n_channels;
        let width = c1 - c0;
        debug_assert!(scratch.len() >= self.n_out * width);
        if self.leading_nans == 0 && use_gemm {
            // The general multiply needs the rows in logical order, so unroll
            // the ring into a temporary first. This path exists for the
            // comparison in the tests and the notebook, not for speed.
            let mut ordered = vec![0.0f64; self.buffer_len * width];
            for r in 0..self.buffer_len {
                ordered[r * width..(r + 1) * width].copy_from_slice(self.row(r, c0, c1));
            }
            // SAFETY: A is n_out x buffer_len (row-major), B is
            // buffer_len x width (row-major), C is n_out x width.
            unsafe {
                matrixmultiply::dgemm(
                    self.n_out,
                    self.buffer_len,
                    width,
                    1.0,
                    self.band.as_ptr(),
                    self.buffer_len as isize,
                    1,
                    ordered.as_ptr(),
                    width as isize,
                    1,
                    0.0,
                    scratch.as_mut_ptr(),
                    width as isize,
                    1,
                );
            }
        } else if self.leading_nans == 0 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                // SAFETY: the build enables AVX-512 for this CPU (target-cpu=native).
                unsafe {
                    avx512::fir_tiled(&self.buffer, n, self.head, self.buffer_len, c0, c1, &self.win, &self.win_padded, self.n_out, scratch)
                };
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            {
                fir_tiled(&self.buffer, n, self.head, self.buffer_len, c0, c1, &self.win, &self.win_padded, self.n_out, scratch);
            }
        } else {
            // Direct sum while NaN rows remain: a dense multiply would turn
            // 0 * NaN into NaN for every output.
            let taps = self.win.len();
            for i in 0..self.n_out {
                let row = &mut scratch[i * width..(i + 1) * width];
                row.fill(0.0);
                for t in 0..taps {
                    let w = self.win[t];
                    let src = self.row(i + t, c0, c1);
                    for (acc, v) in row.iter_mut().zip(src.iter()) {
                        *acc += w * v;
                    }
                }
            }
        }
        for i in 0..self.n_out {
            let dst = &mut out[i * n + c0..i * n + c1];
            let src = &scratch[i * width..(i + 1) * width];
            for (d, v) in dst.iter_mut().zip(src.iter()) {
                *d = *v as f32;
            }
        }
    }
}

/// Zeros added on each side of the tap window so a block of outputs can index
/// it without bounds checks. Sized for the larger (AVX-512) block.
const PAD: usize = 11;
/// Channels per register tile in the portable kernel: two AVX2 registers.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
const TILE: usize = 8;
/// Outputs accumulated at once in the portable kernel: 12 AVX2 accumulator
/// registers, leaving room for the loaded row and the broadcast tap.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
const OUT_BLOCK: usize = 6;

/// The same sliding-window kernel with AVX-512 intrinsics: sixteen channels
/// (two registers) by twelve outputs, 24 accumulators held in the 32 zmm
/// registers. Each buffer row is loaded once per block; each tap is one
/// broadcast; every other instruction is a fused multiply-add.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512 {
    use core::arch::x86_64::*;

    const TILE: usize = 16;
    const OUT_BLOCK: usize = 12;
    const PAD: usize = super::PAD;

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn fir_tiled(
        buffer: &[f64],
        stride: usize,
        head: usize,
        buffer_len: usize,
        c0: usize,
        c1: usize,
        win: &[f64],
        padded: &[f64],
        n_out: usize,
        scratch: &mut [f64],
    ) {
        let taps = win.len();
        let width = c1 - c0;
        debug_assert_eq!(buffer_len, n_out - 1 + taps);
        debug_assert_eq!(padded.len(), taps + 2 * PAD);
        let phys = |r: usize| (head + r) % buffer_len;
        let buf = buffer.as_ptr();
        let pad = padded.as_ptr();

        let mut c = 0;
        while c + TILE <= width {
            let mut i0 = 0;
            while i0 < n_out {
                let i_count = OUT_BLOCK.min(n_out - i0);
                let mut acc0 = [_mm512_setzero_pd(); OUT_BLOCK];
                let mut acc1 = [_mm512_setzero_pd(); OUT_BLOCK];
                let r_end = (i0 + OUT_BLOCK - 1 + taps).min(buffer_len);
                for r in i0..r_end {
                    let at = buf.add(phys(r) * stride + c0 + c);
                    let row0 = _mm512_loadu_pd(at);
                    let row1 = _mm512_loadu_pd(at.add(8));
                    let base = pad.add(r + PAD - i0);
                    for k in 0..OUT_BLOCK {
                        let w = _mm512_set1_pd(*base.sub(k));
                        acc0[k] = _mm512_fmadd_pd(w, row0, acc0[k]);
                        acc1[k] = _mm512_fmadd_pd(w, row1, acc1[k]);
                    }
                }
                for k in 0..i_count {
                    let out = scratch.as_mut_ptr().add((i0 + k) * width + c);
                    _mm512_storeu_pd(out, acc0[k]);
                    _mm512_storeu_pd(out.add(8), acc1[k]);
                }
                i0 += i_count;
            }
            c += TILE;
        }
        // Channels left over after the last full tile: plain sum.
        if c < width {
            for i in 0..n_out {
                for cc in c..width {
                    let mut acc = 0.0;
                    for t in 0..taps {
                        acc += win[t] * buffer[phys(i + t) * stride + c0 + cc];
                    }
                    scratch[i * width + cc] = acc;
                }
            }
        }
    }
}

/// `scratch[i * width + c] = sum_t win[t] * buffer[(i + t) * stride + c0 + c]`.
///
/// The band matrix is a sliding window, so instead of a general multiply
/// each tile of eight channels walks the buffer rows once per block of six
/// outputs, keeping those outputs in registers. Every buffer value is loaded
/// once per block and every tap is a broadcast; the loop is fused-multiply-add
/// bound rather than memory bound.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
#[inline(never)]
fn fir_tiled(buffer: &[f64], stride: usize, head: usize, buffer_len: usize, c0: usize, c1: usize, win: &[f64], padded: &[f64], n_out: usize, scratch: &mut [f64]) {
    let taps = win.len();
    let width = c1 - c0;
    debug_assert_eq!(buffer_len, n_out - 1 + taps);
    let phys = |r: usize| (head + r) % buffer_len;
    // `padded` is the window with zeros on both sides so that
    // `padded[r - i + PAD]` is valid for every (row, output) pair in a block;
    // out-of-range pairs multiply by zero. The buffer holds no NaN or
    // infinity at this point, so a zero tap contributes exactly nothing.
    debug_assert_eq!(padded.len(), taps + 2 * PAD);

    let mut c = 0;
    while c + TILE <= width {
        let mut i0 = 0;
        while i0 < n_out {
            let i_count = OUT_BLOCK.min(n_out - i0);
            let mut acc = [[0.0f64; TILE]; OUT_BLOCK];
            // Rows that feed outputs i0..i0+OUT_BLOCK, clipped to the buffer.
            let r_end = (i0 + OUT_BLOCK - 1 + taps).min(buffer_len);
            for r in i0..r_end {
                let at = phys(r) * stride + c0 + c;
                let row: &[f64; TILE] = buffer[at..at + TILE].try_into().unwrap();
                let base = r + PAD - i0;
                for k in 0..OUT_BLOCK {
                    let w = padded[base - k];
                    let a = &mut acc[k];
                    for lane in 0..TILE {
                        // Explicit fused multiply-add: Rust does not fuse
                        // `a + w * x` on its own.
                        a[lane] = w.mul_add(row[lane], a[lane]);
                    }
                }
            }
            for (k, a) in acc.iter().enumerate().take(i_count) {
                scratch[(i0 + k) * width + c..(i0 + k) * width + c + TILE].copy_from_slice(a);
            }
            i0 += i_count;
        }
        c += TILE;
    }
    // Channels left over after the last full tile: plain sum.
    if c < width {
        for i in 0..n_out {
            for cc in c..width {
                let mut acc = 0.0;
                for t in 0..taps {
                    acc += win[t] * buffer[phys(i + t) * stride + c0 + cc];
                }
                scratch[i * width + cc] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::{butter_bandpass, impulse_response, sosfilt_zi};

    #[test]
    fn tiled_fir_equals_gemm() {
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        let win = impulse_response(&sos, 121);
        for &n in &[8usize, 13, 64] {
            let mut fir = ReverseFir::new(&win, 30, n);
            let mut rng = 99u64;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng % 2000) as f64 - 1000.0
            };
            for _ in 0..6 {
                let forward: Vec<f64> = (0..30 * n).map(|_| next()).collect();
                fir.push(&forward, 0, n);
                fir.finish_push();
            }
            let mut a = vec![0.0f32; 30 * n];
            let mut b = vec![0.0f32; 30 * n];
            let mut scratch = vec![0.0; 30 * n];
            fir.apply_range_with(&mut a, &mut scratch, 0, n, true);
            fir.apply_range_with(&mut b, &mut scratch, 0, n, false);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() <= 1e-3 + 1e-6 * y.abs(), "{x} vs {y} at n={n}");
            }
        }
    }

    #[test]
    fn forward_matches_scalar_reference() {
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        let zi = sosfilt_zi(&sos);
        let n = 3;
        let n_samples = 50;
        let mut window: Vec<f64> = (0..n_samples * n).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
        let expected = {
            // Per-channel scalar run of the same recurrence.
            let mut out = window.clone();
            for c in 0..n {
                let mut state: Vec<[f64; 2]> = zi.clone();
                for s in 0..n_samples {
                    let mut v = window[s * n + c];
                    for (sec, z) in sos.iter().zip(state.iter_mut()) {
                        let y = sec[0] * v + z[0];
                        z[0] = sec[1] * v - sec[4] * y + z[1];
                        z[1] = sec[2] * v - sec[5] * y;
                        v = y;
                    }
                    out[s * n + c] = v;
                }
            }
            out
        };
        let mut forward = ForwardSos::new(&sos, &zi, n);
        // Two calls of 25 samples: the state must carry across.
        let (first, second) = window.split_at_mut(25 * n);
        forward.apply(first, 25);
        forward.apply(second, 25);
        for (g, w) in window.iter().zip(expected.iter()) {
            assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()));
        }
    }

    #[test]
    fn reverse_multiply_equals_direct_sum_after_warmup() {
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        let win = impulse_response(&sos, 121);
        let n = 5;
        let mut fir = ReverseFir::new(&win, 30, n);
        let mut direct_outputs = Vec::new();
        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 2000) as f64 - 1000.0
        };
        let mut out = vec![0.0f32; 30 * n];
        let mut scratch = vec![0.0; 30 * n];
        for window_index in 0..8 {
            let forward: Vec<f64> = (0..30 * n).map(|_| next()).collect();
            fir.push(&forward, 0, n);
            fir.finish_push();
            fir.apply_range(&mut out, &mut scratch, 0, n);
            // Direct reference on the same buffer.
            let mut reference = vec![0.0f64; 30 * n];
            for i in 0..30 {
                for c in 0..n {
                    let mut acc = 0.0;
                    for t in 0..121 {
                        acc += win[t] * fir.row(i + t, c, c + 1)[0];
                    }
                    reference[i * n + c] = acc;
                }
            }
            if window_index < 4 {
                // First four windows: the oldest rows are NaN, and so are the outputs.
                assert!(out.iter().all(|v| v.is_nan()));
            } else {
                for (g, w) in out.iter().zip(reference.iter()) {
                    assert!(((*g as f64) - w).abs() <= 1e-3 + 1e-6 * w.abs(), "{g} vs {w}");
                }
            }
            direct_outputs.push(reference);
        }
    }
}

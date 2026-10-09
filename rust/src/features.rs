//! Per-millisecond features from the filtered window: threshold crossings
//! and spike-band power.
//!
//! A spike is a negative-going crossing: the voltage was at or above the
//! electrode's threshold on one sample and below it on the next. Several
//! crossings in the same millisecond count once. Spike-band power is the mean
//! of `10 * log10(x^2)` over the window, in dB.
//!
//! The logarithm is the expensive part: thirty of them per electrode per
//! millisecond. The C library `log10` is a scalar call, so [`log10_fast`]
//! computes it with integer exponent extraction and a short series, in plain
//! arithmetic the compiler vectorizes across electrodes. It is accurate to
//! about 1e-13 relative, far below the float32 rounding of the Python tree,
//! and falls back to the library for zero, subnormal, infinite, and NaN input.

use std::f64::consts::{LN_10, LN_2, SQRT_2};

/// `log10(x)` for positive, normal `x`, as straight-line arithmetic.
///
/// `x = m * 2^e` with `m` in `[1, 2)`; `m` is pulled toward 1 by halving when
/// it is above `sqrt(2)`, and `ln(m) = 2 * atanh(s)` with `s = (m - 1) / (m + 1)`
/// is summed to the `s^13` term. For `|s| <= 0.172` the dropped tail is below
/// 2e-13.
#[inline(always)]
fn log10_fast(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let series = 1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0 + s2 * (1.0 / 11.0 + s2 * (1.0 / 13.0))))));
    let ln_m = 2.0 * s * series;
    (e as f64 * LN_2 + ln_m) * (1.0 / LN_10)
}

/// Fill `spikes` (0 or 1) and `sbp` for channels `c0..c1` from the time-major
/// filtered window `filt` (`n_samples x n_channels`, `f32`).
#[allow(clippy::too_many_arguments)]
pub fn features_range(
    filt: &[f32],
    n_samples: usize,
    n_channels: usize,
    thresholds: &[f64],
    spikes: &mut [i16],
    sbp: &mut [f32],
    c0: usize,
    c1: usize,
) {
    let n = n_channels;
    let width = c1 - c0;
    for c in c0..c1 {
        spikes[c] = 0;
    }
    for s in 0..n_samples - 1 {
        let now = &filt[s * n + c0..s * n + c1];
        let next = &filt[(s + 1) * n + c0..(s + 1) * n + c1];
        let thr = &thresholds[c0..c1];
        let sp = &mut spikes[c0..c1];
        for i in 0..width {
            // Comparisons in f64 like NumPy does when it compares a float32
            // array with float64 thresholds. NaN compares false, so start-up
            // windows produce no spikes.
            let below = (next[i] as f64) < thr[i] && (now[i] as f64) >= thr[i];
            sp[i] |= below as i16;
        }
    }

    // Spike-band power: accumulate log10(x^2) in f64, average, scale by 10.
    let mut acc = vec![0.0f64; width];
    for s in 0..n_samples {
        let row = &filt[s * n + c0..s * n + c1];
        // `odd` collects, without branching, whether any value in the row is
        // outside the fast path: exact zero (-inf), subnormal, infinity, or
        // NaN (start-up windows). Those have an all-zero or all-one exponent.
        let mut odd = 0u64;
        for (a, v) in acc.iter_mut().zip(row.iter()) {
            let sq = (*v as f64) * (*v as f64);
            let exponent = (sq.to_bits() >> 52) & 0x7ff;
            odd |= (exponent == 0) as u64 | (exponent == 0x7ff) as u64;
            *a += log10_fast(sq);
        }
        if odd != 0 {
            for (a, v) in acc.iter_mut().zip(row.iter()) {
                let sq = (*v as f64) * (*v as f64);
                if !sq.is_normal() {
                    *a += sq.log10() - log10_fast(sq);
                }
            }
        }
    }
    let scale = 10.0 / n_samples as f64;
    for (dst, a) in sbp[c0..c1].iter_mut().zip(acc.iter()) {
        *dst = (a * scale) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log10_is_accurate() {
        let mut x = 1e-30f64;
        while x < 1e30 {
            let want = x.log10();
            let got = log10_fast(x);
            assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{x}: {got} vs {want}");
            x *= 1.37;
        }
    }

    #[test]
    fn one_crossing_per_window_and_log_power() {
        let n = 2;
        let n_samples = 5;
        // channel 0 crosses -10 twice, channel 1 never does.
        let filt: Vec<f32> = vec![
            0.0, 5.0, //
            -20.0, 5.0, //
            0.0, 5.0, //
            -30.0, 5.0, //
            0.0, 5.0,
        ];
        let mut spikes = vec![0i16; n];
        let mut sbp = vec![0f32; n];
        features_range(&filt, n_samples, n, &[-10.0, -10.0], &mut spikes, &mut sbp, 0, n);
        assert_eq!(spikes, vec![1, 0]);
        let expected1 = 10.0 * (25.0f32).log10();
        assert!((sbp[1] - expected1).abs() < 1e-4);
        // Rows with exactly zero voltage have log10(0) = -inf; the mean is -inf.
        assert_eq!(sbp[0], f32::NEG_INFINITY);
    }

    #[test]
    fn nan_window_gives_nan_power_and_no_spike() {
        let n = 1;
        let filt = vec![f32::NAN; 30];
        let mut spikes = vec![1i16; n];
        let mut sbp = vec![0f32; n];
        features_range(&filt, 30, n, &[-10.0], &mut spikes, &mut sbp, 0, n);
        assert_eq!(spikes, vec![0]);
        assert!(sbp[0].is_nan());
    }
}

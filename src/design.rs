//! Filter design: the 250-5000 Hz Butterworth band-pass as second-order
//! sections, the reverse FIR window built from its impulse response, and the
//! per-section starting state SciPy uses.
//!
//! The steps follow `scipy.signal.butter(order, [low, high], btype="bandpass",
//! output="sos", fs=fs)` one for one: analog Butterworth prototype, low-pass to
//! band-pass transform, bilinear transform with frequency pre-warping, then
//! `zpk2sos` with "nearest" pairing. `design_matches_scipy` in the tests checks
//! the coefficients against values printed by SciPy 1.16.

use std::f64::consts::PI;

use num_complex::Complex64;

/// One second-order section `[b0, b1, b2, a0, a1, a2]`, with `a0 == 1`.
pub type Section = [f64; 6];

/// Butterworth band-pass in second-order sections, same layout as SciPy.
///
/// `order` is the prototype order (4 in this project), so the band-pass has
/// `2 * order` poles and `order` sections.
pub fn butter_bandpass(order: usize, low_hz: f64, high_hz: f64, fs: f64) -> Vec<Section> {
    assert!(order >= 1, "filter order must be at least 1");
    assert!(
        0.0 < low_hz && low_hz < high_hz && high_hz < fs / 2.0,
        "cutoffs must satisfy 0 < low < high < fs/2"
    );
    // SciPy normalizes to Nyquist, then designs at an internal rate of 2 Hz.
    let nyq = fs / 2.0;
    let internal_fs = 2.0;
    let warp = |w: f64| 2.0 * internal_fs * (PI * (w / nyq) / internal_fs).tan();
    let (w1, w2) = (warp(low_hz), warp(high_hz));
    let bw = w2 - w1;
    let wo = (w1 * w2).sqrt();

    // Analog prototype: poles on the unit circle in the left half-plane.
    let mut poles: Vec<Complex64> = Vec::with_capacity(order);
    let mut m = -(order as f64) + 1.0;
    while m < order as f64 {
        let angle = PI * m / (2.0 * order as f64);
        poles.push(-Complex64::new(0.0, angle).exp());
        m += 2.0;
    }
    let k = 1.0;

    // lp2bp_zpk: every pole becomes two, and `order` zeros appear at s = 0.
    let degree = order; // no finite zeros in the prototype
    let mut bp_poles = Vec::with_capacity(2 * order);
    let mut plus = Vec::with_capacity(order);
    let mut minus = Vec::with_capacity(order);
    for p in &poles {
        let p_lp = p * (bw / 2.0);
        let root = (p_lp * p_lp - wo * wo).sqrt();
        plus.push(p_lp + root);
        minus.push(p_lp - root);
    }
    bp_poles.extend(plus);
    bp_poles.extend(minus);
    let mut bp_zeros = vec![Complex64::new(0.0, 0.0); degree];
    let k_bp = k * bw.powi(degree as i32);

    // bilinear_zpk at fs = 2: z = (4 + s) / (4 - s); the zeros at infinity
    // land on z = -1.
    let fs2 = 2.0 * internal_fs;
    let degree_z = bp_poles.len() - bp_zeros.len();
    let z_digital: Vec<Complex64> = bp_zeros
        .iter()
        .map(|z| (fs2 + z) / (fs2 - z))
        .chain(std::iter::repeat(Complex64::new(-1.0, 0.0)).take(degree_z))
        .collect();
    let p_digital: Vec<Complex64> = bp_poles.iter().map(|p| (fs2 + p) / (fs2 - p)).collect();
    let num: Complex64 = bp_zeros.iter().map(|z| fs2 - z).product();
    let den: Complex64 = bp_poles.iter().map(|p| fs2 - p).product();
    let k_digital = k_bp * (num / den).re;
    bp_zeros.clear();

    zpk2sos_nearest(&z_digital, &p_digital, k_digital)
}

fn is_real(c: Complex64) -> bool {
    c.im == 0.0
}

/// Keep one member of each complex-conjugate pair (positive imaginary part),
/// followed by the real roots. Mirrors `scipy.signal._cplxreal` for the
/// exactly-conjugate inputs the design above produces.
fn cplxreal(roots: &[Complex64]) -> Vec<Complex64> {
    let tol = 100.0 * f64::EPSILON;
    let mut complex: Vec<Complex64> = Vec::new();
    let mut real: Vec<Complex64> = Vec::new();
    for r in roots {
        if r.im.abs() <= tol * r.norm().max(1.0) {
            real.push(Complex64::new(r.re, 0.0));
        } else if r.im > 0.0 {
            complex.push(*r);
        }
    }
    complex.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap().then(a.im.partial_cmp(&b.im).unwrap()));
    real.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    complex.into_iter().chain(real).collect()
}

fn poly2(r1: Complex64, r2: Complex64) -> [f64; 3] {
    // (x - r1)(x - r2), real parts only; the pairs here are conjugate or real.
    [1.0, -(r1 + r2).re, (r1 * r2).re]
}

/// `scipy.signal.zpk2sos(z, p, k, pairing="nearest")` for the band-pass case:
/// equal numbers of poles and zeros, complex poles in conjugate pairs.
fn zpk2sos_nearest(zeros: &[Complex64], poles: &[Complex64], k: f64) -> Vec<Section> {
    assert_eq!(zeros.len(), poles.len());
    let n_sections = (poles.len() + 1) / 2;
    let mut z = cplxreal(zeros);
    let mut p = cplxreal(poles);

    let mut p_sos: Vec<[Complex64; 2]> = Vec::with_capacity(n_sections);
    let mut z_sos: Vec<[Complex64; 2]> = Vec::with_capacity(n_sections);
    for _ in 0..n_sections {
        // The pole closest to the unit circle is the hardest to realize; it
        // is paired first and placed in the last section.
        let p1_idx = argmin(p.iter().map(|c| (1.0 - c.norm()).abs()));
        let p1 = p.remove(p1_idx);
        let p2;
        let z1;
        let z2;
        if is_real(p1) && p.iter().all(|c| !is_real(*c)) {
            // A lone real pole: pair with the nearest real zero and a zero at
            // the origin. Not reached by the band-pass design.
            let z1_idx = nearest_idx(&z, p1, true);
            z1 = z.remove(z1_idx);
            p2 = Complex64::new(0.0, 0.0);
            z2 = Complex64::new(0.0, 0.0);
        } else {
            let z1_idx = if !is_real(p1) && z.iter().filter(|c| is_real(**c)).count() == 1 {
                nearest_idx(&z, p1, false)
            } else {
                argmin(z.iter().map(|c| (p1 - c).norm()))
            };
            z1 = z.remove(z1_idx);
            if !is_real(p1) {
                p2 = p1.conj();
                if !is_real(z1) {
                    z2 = z1.conj();
                } else {
                    let z2_idx = nearest_idx(&z, p1, true);
                    z2 = z.remove(z2_idx);
                }
            } else {
                // Real pole with other poles left: pair with the nearest real
                // pole, and zeros chosen the same way. Not reached here.
                let p2_idx = nearest_idx(&p, p1, true);
                p2 = p.remove(p2_idx);
                if !is_real(z1) {
                    z2 = z1.conj();
                } else {
                    let z2_idx = nearest_idx(&z, p1, true);
                    z2 = z.remove(z2_idx);
                }
            }
        }
        p_sos.push([p1, p2]);
        z_sos.push([z1, z2]);
    }
    p_sos.reverse();
    z_sos.reverse();

    let mut sos = Vec::with_capacity(n_sections);
    for (si, (zz, pp)) in z_sos.iter().zip(p_sos.iter()).enumerate() {
        let gain = if si == 0 { k } else { 1.0 };
        let b = poly2(zz[0], zz[1]);
        let a = poly2(pp[0], pp[1]);
        sos.push([gain * b[0], gain * b[1], gain * b[2], a[0], a[1], a[2]]);
    }
    sos
}

fn argmin<I: Iterator<Item = f64>>(values: I) -> usize {
    let mut best = 0;
    let mut best_value = f64::INFINITY;
    for (i, v) in values.enumerate() {
        if v < best_value {
            best_value = v;
            best = i;
        }
    }
    best
}

/// Index of the root nearest to `target`, restricted to real or complex roots.
fn nearest_idx(roots: &[Complex64], target: Complex64, want_real: bool) -> usize {
    let mut best = usize::MAX;
    let mut best_value = f64::INFINITY;
    for (i, r) in roots.iter().enumerate() {
        if is_real(*r) != want_real {
            continue;
        }
        let d = (r - target).norm();
        if d < best_value {
            best_value = d;
            best = i;
        }
    }
    assert!(best != usize::MAX, "no root of the requested kind left to pair");
    best
}

/// Starting delay state per section for a constant input of 1, as
/// `scipy.signal.sosfilt_zi`. Both trees start every electrode from this
/// state rather than from zeros.
pub fn sosfilt_zi(sos: &[Section]) -> Vec<[f64; 2]> {
    let mut zi = Vec::with_capacity(sos.len());
    let mut scale = 1.0;
    for s in sos {
        let (b0, b1, b2, a1, a2) = (s[0] / s[3], s[1] / s[3], s[2] / s[3], s[4] / s[3], s[5] / s[3]);
        // lfilter_zi: solve (I - A^T) z = B with B = b[1:] - a[1:] * b[0].
        let bb0 = b1 - a1 * b0;
        let bb1 = b2 - a2 * b0;
        let z0 = (bb0 + bb1) / (1.0 + a1 + a2);
        let z1 = bb1 - a2 * z0;
        zi.push([scale * z0, scale * z1]);
        scale *= (b0 + b1 + b2) / (1.0 + a1 + a2);
    }
    zi
}

/// Impulse response of the cascade, `n_taps` samples long, from a zero state.
/// Used as the reverse FIR window: running it backward over the look-ahead
/// buffer cancels the phase shift of the forward pass.
pub fn impulse_response(sos: &[Section], n_taps: usize) -> Vec<f64> {
    let mut x = vec![0.0; n_taps];
    x[0] = 1.0;
    let mut state = vec![[0.0f64; 2]; sos.len()];
    for sample in x.iter_mut() {
        let mut v = *sample;
        for (s, z) in sos.iter().zip(state.iter_mut()) {
            let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5]);
            let y = b0 * v + z[0];
            z[0] = b1 * v - a1 * y + z[1];
            z[1] = b2 * v - a2 * y;
            v = y;
        }
        *sample = v;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn design_matches_scipy() {
        // scipy.signal.butter(4, [250, 5000], btype="bandpass", output="sos", fs=30000)
        let expected: [Section; 4] = [
            [0.022111075157982225, 0.04422215031596445, 0.022111075157982225, 1.0, -0.6369895274595273, 0.140658453995526],
            [1.0, 2.0, 1.0, 1.0, -0.7799419100625097, 0.533654758975568],
            [1.0, -2.0, 1.0, 1.0, -1.8988925971230104, 0.9019587652362153],
            [1.0, -2.0, 1.0, 1.0, -1.960542441813988, 0.9632954484534988],
        ];
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        assert_eq!(sos.len(), 4);
        for (got, want) in sos.iter().zip(expected.iter()) {
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(close(*g, *w, 1e-12), "got {got:?}, want {want:?}");
            }
        }
    }

    #[test]
    fn zi_matches_scipy() {
        let expected = [
            [0.15348899856546103, -0.0025885597334576933],
            [0.7563202138326413, -0.3217236227167411],
            [-0.9319202875561127, 0.9319202875561099],
            [0.0, 0.0],
        ];
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        let zi = sosfilt_zi(&sos);
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!(close(got[0], want[0], 1e-10) && close(got[1], want[1], 1e-10), "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn impulse_response_matches_scipy() {
        let sos = butter_bandpass(4, 250.0, 5000.0, 30000.0);
        let win = impulse_response(&sos, 121);
        let head = [0.022111075157982225, 0.11666613572204357, 0.25675925669884875, 0.2926878382733656];
        for (g, w) in win.iter().zip(head.iter()) {
            assert!(close(*g, *w, 1e-12));
        }
        assert!(close(win[119], -0.0037702558802604103, 1e-10));
        assert!(close(win[120], -0.003538489632194604, 1e-10));
    }
}

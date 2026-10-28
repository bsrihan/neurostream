//! neurostream in Rust: the millisecond neural pipeline.
//!
//! Each electrode of a microelectrode array is sampled 30,000 times a second.
//! Once per millisecond, for every electrode, this crate returns whether a
//! spike happened and one spike-band power number. The steps are the same as
//! the Python project (bsrihan/neurostream-python):
//!
//! 1. **Re-reference** ([`reref`]): subtract the shared part of each group of
//!    electrodes, `y = (I - P) x`, with one small matrix per group.
//! 2. **Forward filter** (`filter::ForwardSos`): Butterworth order 4,
//!    250-5000 Hz, state carried from one millisecond to the next.
//! 3. **Reverse filter** (`filter::ReverseFir`): the same filter backward
//!    over a 4 ms look-ahead, so spikes are not shifted in time. One matrix
//!    multiply. The output is therefore 4 ms behind the input.
//! 4. **Features** (`features`): negative threshold crossing (at most one per
//!    millisecond) and mean `10 log10(x^2)`.
//!
//! Everything is time-major: a window is `n_samples x n_channels` with
//! channels contiguous. That is the order Blackrock files store samples in,
//! so a recording streams straight from disk into the pipeline, and it lets
//! the forward filter and the feature loops walk across electrodes in the
//! inner loop, which the compiler turns into wide SIMD instructions.
//!
//! The processor is split into `Lane`s, each owning a contiguous range of
//! channels and all of its buffers and filter state. With one lane the
//! pipeline is single-threaded. With several, each window is processed by all
//! lanes at once on a rayon pool; because every lane owns its memory there is
//! no shared state to lock and no cross-thread writes.

pub mod design;
pub mod nsx;
pub mod params;
pub mod reref;

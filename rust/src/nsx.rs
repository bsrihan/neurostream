//! Reader for Blackrock NSx continuous files (`.ns6` at 30 kHz).
//!
//! The format is a fixed basic header ("NEURALCD", version, header length,
//! sampling period, channel count), one 66-byte extended header per channel,
//! then data packets. Each packet is a flag byte, a timestamp (4 bytes for
//! file version 2.x, 8 bytes for 3.x), a sample count, and the samples as
//! little-endian `i16`, one sample of every channel at a time. That
//! sample-major order is exactly the time-major layout the pipeline uses, so
//! the samples are read straight into the processing buffer with no transpose.
//!
//! Values are left in the file's integer units. For the example recording the
//! extended header says 1 LSB = 0.25 uV; the Python tree uses the same raw
//! integers, so the two trees see identical input.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// What the basic header says about the file.
#[derive(Debug, Clone)]
pub struct NsxHeader {
    pub version: (u8, u8),
    pub header_bytes: u32,
    pub label: String,
    /// Samples per second (`time_resolution / period`).
    pub sample_rate: f64,
    pub n_channels: usize,
    pub channel_labels: Vec<String>,
    /// Timestamp size in the data packet header: 4 (v2.x) or 8 (v3.x).
    pub timestamp_bytes: usize,
}

/// A stretch of the recording in time-major order: `samples[s * n_channels + c]`.
#[derive(Debug, Clone)]
pub struct Recording {
    pub header: NsxHeader,
    pub n_samples: usize,
    pub samples: Vec<i16>,
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn u32_at(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

/// Parse the basic and extended headers.
pub fn read_header(path: &Path) -> io::Result<NsxHeader> {
    let mut file = BufReader::new(File::open(path)?);
    let mut basic = [0u8; 314];
    file.read_exact(&mut basic)?;
    if &basic[..8] != b"NEURALCD" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not an NSx 2.x/3.x file (missing NEURALCD)"));
    }
    let version = (basic[8], basic[9]);
    let header_bytes = u32_at(&basic, 10);
    let label = c_string(&basic[14..30]);
    let period = u32_at(&basic, 286) as f64;
    let time_resolution = u32_at(&basic, 290) as f64;
    let n_channels = u32_at(&basic, 310) as usize;
    let mut channel_labels = Vec::with_capacity(n_channels);
    let mut ext = [0u8; 66];
    for _ in 0..n_channels {
        file.read_exact(&mut ext)?;
        if &ext[..2] != b"CC" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected extended header type"));
        }
        channel_labels.push(c_string(&ext[4..20]));
    }
    Ok(NsxHeader {
        version,
        header_bytes,
        label,
        sample_rate: time_resolution / period,
        n_channels,
        channel_labels,
        timestamp_bytes: if version.0 >= 3 { 8 } else { 4 },
    })
}

/// Read up to `max_samples` samples (per channel) from the first data packet.
///
/// The example file is one packet; a file with several packets (paused
/// recording) is read packet by packet until the limit. Gaps between packets
/// are not filled.
pub fn read_samples(path: &Path, max_samples: Option<usize>) -> io::Result<Recording> {
    let header = read_header(path)?;
    let mut file = BufReader::with_capacity(1 << 20, File::open(path)?);
    file.seek(SeekFrom::Start(header.header_bytes as u64))?;
    let n_ch = header.n_channels;
    let limit = max_samples.unwrap_or(usize::MAX);
    let mut samples: Vec<i16> = Vec::new();
    let mut total = 0usize;
    let mut packet_head = vec![0u8; 1 + header.timestamp_bytes + 4];
    loop {
        if total >= limit {
            break;
        }
        match file.read_exact(&mut packet_head) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        if packet_head[0] != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "data packet does not start with 0x01"));
        }
        let n_points = u32_at(&packet_head, 1 + header.timestamp_bytes) as usize;
        let take = n_points.min(limit - total);
        let mut raw = vec![0u8; take * n_ch * 2];
        file.read_exact(&mut raw)?;
        samples.reserve(take * n_ch);
        for chunk in raw.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        total += take;
        if take < n_points {
            // Skip the rest of this packet so a following one starts aligned.
            file.seek(SeekFrom::Current(((n_points - take) * n_ch * 2) as i64))?;
        }
    }
    Ok(Recording { header, n_samples: total, samples })
}

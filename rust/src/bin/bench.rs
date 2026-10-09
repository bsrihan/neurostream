//! Command-line timing of the Rust pipeline, without Python in the loop.
//!
//! Prints one JSON object per line so a notebook can parse it. Times are in
//! milliseconds of wall time per second of data; under 1000 is real time.
//!
//!     neurostream-bench synthetic --channels 64,256,1024,2048 --threads 1,4
//!     neurostream-bench ns6 --file ../data/NSP1_aligned.ns6 \
//!         --params ../data/NSP1_aligned_params.json --seconds 10 --threads 1,4
//!
//! "Steady state" means the processor is already built and has run a few
//! windows before the clock starts, and the output arrays already exist.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use neurostream::params::Params;
use neurostream::{build_target, Config, Processor};

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').filter(|p| !p.is_empty()).map(|p| p.trim().parse().expect("integer list")).collect()
}

fn synthetic(n_channels: usize, n_ms: usize, seed: u64) -> Vec<f64> {
    let spw = 30;
    let mut rng = seed | 1;
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Steady-state total and, for one lane, the stage split. Returns
/// (total ms/s, stage ms/s in order reref, forward, reverse, features, write).
fn steady_state<T: Copy + Into<f64>>(proc: &mut Processor, raw: &[T], n_samples: usize, repeats: usize) -> (f64, Vec<f64>, Vec<String>) {
    let n = proc.n_channels;
    let spw = proc.samples_per_window;
    let n_windows = n_samples / spw;
    let neural_s = n_windows as f64 / 1000.0;
    let mut filtered = vec![0.0f32; n_windows * spw * n];
    let mut spikes = vec![0i16; n_windows * n];
    let mut sbp = vec![0.0f32; n_windows * n];
    // Warm up: filter state, page faults, thread pool.
    for w in 0..n_windows.min(20) {
        let window = &raw[w * spw * n..(w + 1) * spw * n];
        proc.process_window(window, &mut filtered[w * spw * n..(w + 1) * spw * n], None, &mut spikes[w * n..(w + 1) * n], &mut sbp[w * n..(w + 1) * n]);
    }
    let mut totals = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        for w in 0..n_windows {
            let window = &raw[w * spw * n..(w + 1) * spw * n];
            proc.process_window(window, &mut filtered[w * spw * n..(w + 1) * spw * n], None, &mut spikes[w * n..(w + 1) * n], &mut sbp[w * n..(w + 1) * n]);
        }
        totals.push(t0.elapsed().as_secs_f64() / neural_s * 1000.0);
    }
    let mut stages = Vec::new();
    let mut largest = Vec::new();
    if proc.lanes() == 1 {
        let mut rows: Vec<[f64; 5]> = Vec::new();
        for _ in 0..repeats {
            let t = proc.profile_recording(raw, n_samples, &mut filtered, &mut spikes, &mut sbp);
            let row = [t.reref, t.forward, t.reverse, t.features, t.write].map(|x| x / neural_s * 1000.0);
            let names = ["reref", "forward", "reverse", "features", "write"];
            let (mut best, mut best_v) = (0, -1.0);
            for (i, v) in row.iter().enumerate() {
                if *v > best_v {
                    best_v = *v;
                    best = i;
                }
            }
            largest.push(names[best].to_string());
            rows.push(row);
        }
        for i in 0..5 {
            let mut col: Vec<f64> = rows.iter().map(|r| r[i]).collect();
            stages.push(median(&mut col));
        }
    }
    (median(&mut totals), stages, largest)
}

fn print_row(kind: &str, source: &str, n_channels: usize, threads: usize, lanes: usize, total: f64, stages: &[f64], largest: &[String]) {
    let stage_json = if stages.is_empty() {
        "null".to_string()
    } else {
        format!(
            "{{\"reref\":{:.3},\"forward\":{:.3},\"reverse\":{:.3},\"features\":{:.3},\"write\":{:.3}}}",
            stages[0], stages[1], stages[2], stages[3], stages[4]
        )
    };
    let largest_json = format!("[{}]", largest.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(","));
    println!(
        "{{\"kind\":\"{kind}\",\"source\":\"{source}\",\"channels\":{n_channels},\"threads\":{threads},\"lanes\":{lanes},\"total_ms_per_s\":{total:.3},\"stages_ms_per_s\":{stage_json},\"largest_each_repeat\":{largest_json},\"target\":\"{}\"}}",
        build_target()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: neurostream-bench synthetic|ns6 [options]");
        std::process::exit(2);
    }
    let mut channels = vec![64usize, 256, 1024, 2048];
    let mut threads = vec![1usize];
    let mut windows = 200usize;
    let mut repeats = 3usize;
    let mut file: Option<PathBuf> = None;
    let mut params_path: Option<PathBuf> = None;
    let mut seconds = 10usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--channels" => channels = parse_list(&args[i + 1]),
            "--threads" => threads = parse_list(&args[i + 1]),
            "--windows" => windows = args[i + 1].parse().unwrap(),
            "--repeats" => repeats = args[i + 1].parse().unwrap(),
            "--seconds" => seconds = args[i + 1].parse().unwrap(),
            "--file" => file = Some(PathBuf::from(&args[i + 1])),
            "--params" => params_path = Some(PathBuf::from(&args[i + 1])),
            other => {
                eprintln!("unknown option {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }

    match args[1].as_str() {
        "synthetic" => {
            for &n in &channels {
                let raw = synthetic(n, windows, 7);
                let params = Params::common_average(n, 64, -120.0, 30000.0);
                for &t in &threads {
                    let mut proc = Processor::new(&params, Config { threads: t, ..Config::default() }).unwrap();
                    let lanes = proc.lanes();
                    let (total, stages, largest) = steady_state(&mut proc, &raw, windows * 30, repeats);
                    print_row("synthetic", "synthetic", n, t, lanes, total, &stages, &largest);
                }
            }
        }
        "ns6" => {
            let file = file.expect("--file is required");
            let params_path = params_path.expect("--params is required");
            let params = Params::from_json_file(&params_path).expect("params json");
            let rec = neurostream::nsx::read_samples(&file, Some(seconds * 30000)).expect("read ns6");
            let n = rec.header.n_channels;
            assert_eq!(n, params.n_channels);
            for &t in &threads {
                let mut proc = Processor::new(&params, Config { threads: t, ..Config::default() }).unwrap();
                let lanes = proc.lanes();
                let (total, stages, largest) = steady_state(&mut proc, &rec.samples, rec.n_samples, repeats);
                print_row("ns6", &file.display().to_string(), n, t, lanes, total, &stages, &largest);
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}

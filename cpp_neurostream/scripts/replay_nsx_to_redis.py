#!/usr/bin/env python3
"""
Replay NSx file to Redis streams for C++ processor testing.

This script reads a Blackrock NSx file and writes it to Redis streams
in 1ms chunks, mimicking the real-time data flow from cerebusAdapter.

Output Redis stream format (compatible with cerebusAdapter):
- Stream name: "raw_neural_data" (configurable)
- Fields:
  - timestamps: uint64 array (NSP timestamps)
  - samples: int16 array (channels x samples, row-major)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import redis

# Add notebooks directory to path for brpylib
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'notebooks'))
import brpylib


def replay_nsx_to_redis(nsx_path, redis_host='127.0.0.1', redis_port=6379,
                        stream_name='raw_neural_data', max_time_s=None,
                        speed=1.0, clear_stream=True):
    """
    Replay NSx file to Redis stream.

    Parameters
    ----------
    nsx_path : str
        Path to NSx file
    redis_host : str
        Redis server hostname
    redis_port : int
        Redis server port
    stream_name : str
        Name of Redis stream to write to
    max_time_s : float, optional
        Maximum time in seconds to replay (for testing)
    speed : float
        Playback speed multiplier (1.0 = real-time, 0 = as fast as possible)
    clear_stream : bool
        Clear existing stream data before starting
    """
    print(f"[replay_nsx] Loading NSx file: {nsx_path}")
    nsx_file = brpylib.NsxFile(nsx_path)

    # Get basic file info
    sample_rate = (nsx_file.basic_header['SampleResolution'] /
                   nsx_file.basic_header['Period'])
    samples_per_ms = int(sample_rate / 1000)

    print(f"[replay_nsx] Sample rate: {sample_rate} Hz")
    print(f"[replay_nsx] Samples per ms: {samples_per_ms}")

    # Load all data (or limited time window)
    if max_time_s is not None:
        print(f"[replay_nsx] Loading first {max_time_s} seconds...")
        cont_data = nsx_file.getdata(start_time_s=0, data_time_s=max_time_s)
    else:
        print(f"[replay_nsx] Loading all data...")
        cont_data = nsx_file.getdata()

    # Extract data from first segment (assuming single continuous recording)
    if len(cont_data['data']) == 0:
        raise ValueError("No data found in NSx file")

    raw_data = cont_data['data'][0]  # int16, shape (n_channels, n_samples)
    n_channels, n_samples = raw_data.shape

    print(f"[replay_nsx] Loaded {n_channels} channels x {n_samples} samples")
    print(f"[replay_nsx] Duration: {n_samples / sample_rate:.2f} seconds")

    # Get or create timestamps
    start_timestamp = cont_data['data_headers'][0].get('Timestamp', 0)
    if isinstance(start_timestamp, (int, np.integer)):
        # Create synthetic timestamps
        timestamps = np.arange(n_samples, dtype=np.uint64) + start_timestamp
    else:
        timestamps = np.array(start_timestamp, dtype=np.uint64)

    # Connect to Redis
    print(f"[replay_nsx] Connecting to Redis at {redis_host}:{redis_port}")
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

    # Test connection
    try:
        r.ping()
        print("[replay_nsx] Connected to Redis successfully")
    except redis.ConnectionError as e:
        print(f"[replay_nsx] ERROR: Could not connect to Redis: {e}")
        sys.exit(1)

    # Clear stream if requested
    if clear_stream:
        try:
            r.delete(stream_name)
            print(f"[replay_nsx] Cleared existing stream: {stream_name}")
        except:
            pass

    # Calculate number of 1ms chunks
    n_chunks = n_samples // samples_per_ms
    print(f"[replay_nsx] Will write {n_chunks} chunks of {samples_per_ms} samples")

    # Replay data in 1ms chunks
    print(f"[replay_nsx] Starting replay (speed={speed}x)...")
    start_time = time.time()

    for i in range(n_chunks):
        # Extract 1ms chunk
        sample_start = i * samples_per_ms
        sample_end = sample_start + samples_per_ms

        chunk_data = raw_data[:, sample_start:sample_end]
        chunk_timestamps = timestamps[sample_start:sample_end]

        # Prepare Redis entry (same format as cerebusAdapter)
        entry = {
            b'timestamps': chunk_timestamps.tobytes(),
            b'samples': chunk_data.tobytes()
        }

        # Write to Redis stream
        r.xadd(stream_name, entry)

        # Print progress every 100ms
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            progress_pct = 100 * (i + 1) / n_chunks
            print(f"[replay_nsx] Progress: {progress_pct:.1f}% "
                  f"({i+1}/{n_chunks} chunks, {elapsed:.2f}s elapsed)")

        # Sleep to maintain playback speed (if not going as fast as possible)
        if speed > 0:
            time.sleep(0.001 / speed)  # 1ms divided by speed multiplier

    total_time = time.time() - start_time
    print(f"[replay_nsx] Replay complete!")
    print(f"[replay_nsx] Wrote {n_chunks} chunks in {total_time:.2f} seconds")
    print(f"[replay_nsx] Average rate: {n_chunks / total_time:.1f} chunks/s")


def main():
    parser = argparse.ArgumentParser(
        description='Replay NSx file to Redis stream for C++ processor testing'
    )
    parser.add_argument('nsx_file', type=str, help='Path to NSx file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Redis server hostname (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=6379,
                        help='Redis server port (default: 6379)')
    parser.add_argument('--stream', type=str, default='raw_neural_data',
                        help='Redis stream name (default: raw_neural_data)')
    parser.add_argument('-t', '--time', type=float, default=None,
                        help='Maximum time in seconds to replay (for testing)')
    parser.add_argument('-s', '--speed', type=float, default=0,
                        help='Playback speed (0=fast as possible, 1=real-time, default: 0)')
    parser.add_argument('--no-clear', action='store_true',
                        help='Do not clear existing stream before writing')

    args = parser.parse_args()

    try:
        replay_nsx_to_redis(
            args.nsx_file,
            redis_host=args.host,
            redis_port=args.port,
            stream_name=args.stream,
            max_time_s=args.time,
            speed=args.speed,
            clear_stream=not args.no_clear
        )
    except Exception as e:
        print(f"[replay_nsx] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

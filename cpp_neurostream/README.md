# NeuroStream C++ Implementation

High-performance C++ implementation of neural signal processing pipeline.

## Status: Work in Progress

Current implementation focuses on core processing pipeline without Redis/NSx file dependencies.

## Architecture

```
Python (NSx file reader)
    ↓ (binary file)
Data Loader (C++)
    ↓
Re-referencing Engine
    ↓
IIR Filter Bank
    ↓
Feature Extraction
    ↓
Output (spikes + power)
```

## Building

### Dependencies

- CMake 3.15+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- Eigen3 (for matrix operations)

### Install Eigen3

**macOS:**
```bash
brew install eigen
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libeigen3-dev
```

### Build

```bash
cd cpp_neurostream
mkdir -p build
cd build
cmake ..
make -j
```

## Usage

### Prerequisites

1. **Install Redis** (if not already installed):
   ```bash
   # macOS
   brew install redis
   brew services start redis

   # Ubuntu/Linux
   sudo apt install redis-server
   sudo systemctl start redis
   ```

2. **Install hiredis** (C Redis client):
   ```bash
   # macOS
   brew install hiredis

   # Ubuntu/Linux
   sudo apt install libhiredis-dev
   ```

### Step 1: Replay NSx data to Redis

```bash
python scripts/replay_nsx_to_redis.py path/to/file.ns6 \
    --time 10 \
    --speed 0
```

Options:
- `--time 10`: Process only first 10 seconds (for testing)
- `--speed 0`: Go as fast as possible (0=fast, 1=real-time)
- `--stream raw_neural_data`: Redis stream name (default)
- `--host 127.0.0.1`: Redis host (default)
- `--port 6379`: Redis port (default)

### Step 2: Run C++ processor

```bash
./build/neurostream_processor \
    --redis-host 127.0.0.1 \
    --redis-port 6379 \
    --input-stream raw_neural_data \
    --reref-params path/to/reref_params.json \
    --thresholds path/to/thresholds.json
```

### Step 3: Inspect results in Redis

```bash
# View spike events
redis-cli XREAD COUNT 10 STREAMS spike_events 0

# View spike-band power
redis-cli XREAD COUNT 10 STREAMS spike_band_power 0
```

## Project Structure

```
cpp_neurostream/
├── CMakeLists.txt       # Build configuration
├── README.md            # This file
├── include/             # Header files
│   ├── data_structures.hpp  # Core data types
│   ├── data_loader.hpp      # Binary file reader
│   ├── rereference.hpp      # Re-referencing engine
│   ├── filter.hpp           # IIR filters
│   └── features.hpp         # Spike detection & power
├── src/                 # Implementation files
│   ├── main.cpp
│   ├── data_loader.cpp
│   ├── rereference.cpp
│   ├── filter.cpp
│   └── features.cpp
├── scripts/             # Helper scripts
│   └── nsx_to_binary.py     # Python NSx exporter
├── build/               # Build output (generated)
└── data/                # Test data (generated)
```

## Processing Pipeline

### 1. Re-referencing
- **CAR** (Common Average Reference): Subtract mean of channel group
- **LRR** (Linear Regression Reference): Use pre-computed regression weights

### 2. Filtering
- Butterworth bandpass filter (250-5000 Hz)
- 4th order IIR using Second-Order Sections
- Causal filtering (zero-phase can be added later)

### 3. Feature Extraction
- **Spike detection**: Threshold crossings (default: -4.5 × RMS)
- **Spike-band power**: 10×log₁₀(mean(signal²)) per 1ms window

## Performance Goals

- **Throughput**: Process 2048 channels @ 30 kHz
- **Latency**: <2ms per 1ms chunk
- **Speedup**: 10-12× faster than Python

## Validation

Compare output against Python reference implementation:
```bash
python scripts/validate_output.py \
    --python-output python_results/ \
    --cpp-output cpp_results/
```

## TODO

- [ ] Implement re-referencing engine
- [ ] Implement IIR filter
- [ ] Implement feature extraction
- [ ] Create main program with CLI
- [ ] Add validation tests
- [ ] Optimize with SIMD
- [ ] Add multi-threading
- [ ] Profile and benchmark
- [ ] Add Redis input/output (future)
- [ ] Add direct NSx file reading (future)

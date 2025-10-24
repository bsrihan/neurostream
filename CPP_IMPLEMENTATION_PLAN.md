# C++ NeuroStream Implementation Plan

## Overview

C++ implementation of neural signal processing that replaces the Python neurostream pipeline.

**Current Phase**: Starting with simplified architecture - file-based I/O, single-threaded processing to validate correctness against Python implementation before optimizing.

**Future Goal**: Redis-based multi-threaded system for real-time 128-channel processing.

## Revised Architecture (Phase 1: Validation)

### Input/Output Strategy
- **Input**: Redis streams (Python replays NSx file to Redis)
- **Output**: Redis streams (spikes + spike-band power)
- **Rationale**:
  - Use proven infrastructure from brand-nsp ecosystem
  - Same Redis interface for testing and production
  - Easy debugging and data inspection
  - No file format conversion needed

### Processing Pipeline (Single-threaded)

```
Python NSx Replay Script
  ├─ brpylib reads NSx file
  └─ Writes to Redis stream (1ms chunks)
       ↓
RedisDataSource (C++)
       ↓
RereferenceEngine (Eigen)
       ↓
FilterBank (iir1 library)
       ↓
FeatureExtractor
       ↓
RedisDataSink (C++)
  ├─ Spikes stream
  └─ Spike-band power stream
       ↓
Python Validation Script
  └─ Compare with reference output
```

## Future Architecture (Phase 2: Production)

### Core Design Principle
**Filtering-focused parallelism**: Since filtering is the computational bottleneck (~80% of processing time), we dedicate maximum CPU resources to parallel filtering while keeping lightweight operations (re-referencing, feature extraction) in serial threads.

### Threading Model

```
Redis Reader Thread (1)
       ↓
Re-reference Thread (1)
       ↓
Parallel Filter Threads (12)
       ↓
Feature Extraction Thread (1)
       ↓
Redis Writer Thread (1)
```

## Component Specifications

### 1. Main Application (`NeuroStream`)

// TODO: PLan for a budget/ambition of around 4 cores - 
// let's assume we've got ~2 threads per core
// 2 redis, 1 reref, 1 feature, 4 threads for filters

// Later on, make it so they can initialize the C++ service
// via Python (ie, they choose the filter funcs, etc)

// 1. I'll get a version working that reads from Redis, writes out to redis
// 2. Write tests that the out data matches the python - just write both to redis & compare binary data
// 3. Benchmark - we'll get latency results from different configs (mostly around # threads and pipline architecture)
// 4 If ^ promsing - GRPC front (config + grpc service/server)
// 5. Add an API, something like `results = processing_pipeline(data)`, that allows the GRPC service to be called from Python

```cpp
class NeuroStream {
private:
    // Core processing components
    RedisDataSource data_source_;
    RedisDataSink data_sink_;
    RereferenceEngine reref_engine_;
    ThreadedFilterBank filter_bank_;
    FeatureExtractor feature_extractor_;

    // Threading infrastructure
    std::thread redis_reader_;
    std::thread redis_writer_;
    std::thread reref_and_features_;

    // Inter-thread communication
    LockFreeQueue<RawChunk> raw_queue_;
    LockFreeQueue<RerefChunk> reref_queue_;
    LockFreeQueue<FilteredChunk> filtered_queue_;
    LockFreeQueue<FeatureChunk> output_queue_;

    // Configuration
    NeuroConfig config_;
    std::atomic<bool> running_{true};

public:
    void start();
    void stop();
    void run();
};
```

### 2. Data Structures

```cpp
struct RawChunk {
    std::vector<float> data;          // 128 channels × 30 samples
    std::vector<uint64_t> timestamps; // NSP timestamps
    uint64_t redis_timestamp;         // Redis entry timestamp
    int n_channels = 128;
    int n_samples = 30;
};

struct RerefChunk {
    std::vector<float> data;          // Re-referenced data
    uint64_t timestamp;
    int n_channels = 128;
    int n_samples = 30;
};

struct FilteredChunk {
    std::vector<float> data;          // Filtered data (250-5000 Hz)
    uint64_t timestamp;
    int n_channels = 128;
    int n_samples = 30;
};

struct FeatureChunk {
    std::vector<uint8_t> spikes;      // Binary spike detection
    std::vector<float> spike_power;   // Spike-band power
    uint64_t timestamp;
    int n_channels = 128;
};
```

### 3. Redis Interface (`RedisDataSource` & `RedisDataSink`)

```cpp
class RedisDataSource {
private:
    hiredis::RedisClient client_;
    std::string input_stream_;
    std::string last_id_ = "$";

public:
    RedisDataSource(const std::string& host, int port, const std::string& stream);
    RawChunk read_chunk();  // Blocking read from Redis stream
    bool is_connected();
};

class RedisDataSink {
private:
    hiredis::RedisClient client_;
    std::string spike_stream_;
    std::string power_stream_;

public:
    RedisDataSink(const std::string& host, int port,
                  const std::string& spike_stream,
                  const std::string& power_stream);
    void write_features(const FeatureChunk& chunk);
};
```

### 4. Re-referencing Engine (`RereferenceEngine`)

```cpp
class RereferenceEngine {
private:
    Eigen::MatrixXf reref_matrix_;    // Pre-computed re-reference weights
    Eigen::MatrixXf unshuffle_matrix_; // Channel unshuffling matrix

public:
    void load_parameters(const std::string& param_file);
    void apply_car(const float* input, float* output);  // Common average reference
    void apply_lrr(const float* input, float* output);  // Linear regression reference
    RerefChunk process(const RawChunk& input);
};
```

### 5. Filter Bank (`FilterBank`)

**Implementation Strategy:**
- **Forward IIR Pass**: Use **iir1** C++ library (cascaded biquad filters)
  - Handles SOS (Second-Order Sections) format
  - Maintains filter state (`zi`) between chunks
  - Direct replacement for `scipy.signal.sosfilt()`
- **Reverse FIR Pass**: Custom convolution implementation
  - Simple sliding dot product (~10 lines of code)
  - Direct replacement for `np.convolve(..., mode='valid')`
- **Filter Coefficients**: Pre-computed in Python, saved to JSON
  - `sos` - IIR coefficients (from `scipy.signal.butter()`)
  - `zi` - Initial filter state (from `scipy.signal.sosfilt_zi()`)
  - `rev_win` - FIR kernel for reverse pass (from filtering unit impulse)

```cpp
class AcausalFilter {
private:
    // IIR filter using iir1 library
    Iir::Custom::SOSCascade iir_filter_;
    std::vector<float> zi_;  // Filter state

    // FIR convolution for reverse pass
    std::vector<float> rev_win_;  // Reverse filter kernel (120 samples)
    std::vector<float> rev_buffer_;  // Buffer for acausal filtering (150 samples)

    void convolve_valid(const float* signal, int sig_len,
                       const float* kernel, int ker_len,
                       float* output);

public:
    void load_coefficients(const std::string& coeff_file);
    void process(const float* input, float* output, int n_samples);
    void reset_state();
};

class FilterBank {
private:
    std::vector<AcausalFilter> filters_;  // One filter per channel
    int n_channels_;

public:
    FilterBank(int n_channels);
    void load_coefficients(const std::string& coeff_file);
    void process(const RerefChunk& input, FilteredChunk& output);
};
```

WIP
  5. Parallel Filter Bank (ThreadedFilterBank)
```cpp

  class IIRFilter {
  private:
      std::vector<float> sos_coeffs_;   // Second-order sections
      std::vector<float> zi_;           // Filter state

  public:
      void initialize(int order, float low_freq, float high_freq, float fs);
      void process(const float* input, float* output, int n_samples);
      void reset_state();
  };

  class FilterWorker {
  private:
      std::vector<IIRFilter> filters_;  // One filter per assigned channel
      std::vector<int> channel_ids_;    // Which channels this worker handles

  public:
      void assign_channels(const std::vector<int>& channels);
      void process_chunk(const RerefChunk& input, FilteredChunk& output);
  };

  class ThreadedFilterBank {
  private:
      std::vector<FilterWorker> workers_;
      std::vector<std::thread> filter_threads_;
      int num_threads_;

  public:
      ThreadedFilterBank(int num_threads);
      void start_parallel_filtering(LockFreeQueue<RerefChunk>& input_q,
                                   LockFreeQueue<FilteredChunk>& output_q);
      void stop();
  };
```

**Key Design Decisions:**
1. **Why iir1?** It's a proven, header-only C++ library specifically for real-time IIR filtering with state management
2. **Why custom convolution?** It's trivial (10 lines) and has no good library alternative
3. **Why pre-compute coefficients?** Trust scipy's filter design, C++ just applies them

### 6. Feature Extractor (`FeatureExtractor`)

```cpp
class FeatureExtractor {
private:
    std::vector<float> thresholds_;   // Per-channel spike thresholds

public:
    void load_thresholds(const std::string& threshold_file);
    void extract_spikes(const float* filtered_data, uint8_t* spikes);
    void extract_spike_power(const float* filtered_data, float* power);
    FeatureChunk process(const FilteredChunk& input);
};
```

### 7. Configuration (`NeuroConfig`)

```cpp
struct NeuroConfig {
    // Redis configuration
    std::string redis_host = "127.0.0.1";
    int redis_port = 6379;
    std::string input_stream = "raw_neural_data";
    std::string spike_output_stream = "spike_events";
    std::string power_output_stream = "spike_band_power";

    // Threading configuration
    int filter_threads = 12;
    int channels_per_filter_thread = 170;  // 128/12

    // Processing parameters
    int filter_order = 4;
    float filter_low_freq = 250.0f;
    float filter_high_freq = 5000.0f;
    float sample_rate = 30000.0f;
    float threshold_multiplier = -4.5f;

    // Buffer sizes
    int queue_size = 100;  // Buffer 100ms of data

    // File paths
    std::string reref_params_file;
    std::string threshold_params_file;

    static NeuroConfig load_from_yaml(const std::string& config_file);
};
```

## Implementation Phases

### Phase 1: Validation Pipeline (Current - Start Simple!)
**Goal**: Get correct output matching Python, no optimization yet

1. **Python utilities**
   - ✓ Script to save filter coefficients to JSON
   - TODO: Script to replay NSx → Redis streams
   - TODO: Script to validate C++ output vs Python output

2. **Redis interface (C++)**
   - TODO: Implement `RedisDataSource` (read from Redis stream)
   - TODO: Implement `RedisDataSink` (write to Redis streams)
   - Uses hiredis library (already in brand ecosystem)

3. **Core data structures** ✓
   - Implement `RawChunk`, `RerefChunk`, `FilteredChunk`, `FeatureChunk`
   - Redis-compatible data layout

4. **Re-referencing engine** (IN PROGRESS)
   - Load pre-computed matrix from JSON
   - Matrix-vector multiply using Eigen
   - Validate output vs Python

5. **Filter bank** (NEXT)
   - Integrate iir1 library
   - Load SOS coefficients from JSON
   - Implement custom convolution for reverse pass
   - Validate filtered output vs Python

6. **Feature extraction**
   - Threshold crossing detection
   - Spike-band power (log10 of mean squared)
   - Validate features vs Python

7. **Integration and validation**
   - End-to-end processing pipeline
   - Compare Redis outputs with Python reference
   - Fix any numerical differences

### Phase 2: Optimization (Future)
**Goal**: Make it fast

1. **Multi-threading**
   - Parallelize filtering across channels
   - Thread pool implementation
   - Lock-free queues

2. **SIMD optimizations**
   - Vectorize convolution
   - Vectorize power calculation
   - Profile and optimize hot paths

3. **Memory optimization**
   - Memory pools
   - Cache-friendly layouts
   - Reduce allocations

### Phase 3: Production Integration (Future)
**Goal**: Replace Python in real system

1. **Redis integration**
   - Implement `RedisDataSource` using hiredis
   - Implement `RedisDataSink`
   - Test with brand-nsp ecosystem

2. **Direct NSx file reading** (optional)
   - Port brpylib or find C++ library
   - Eliminate Python dependency

3. **Monitoring and diagnostics**
   - Performance metrics
   - Health checks
   - Debugging utilities

## Dependencies

### Required Libraries (Phase 1)
- **hiredis**: C client for Redis (already in brand ecosystem!)
  - Used by cerebusAdapter and other brand-nsp nodes
  - Repository: https://github.com/redis/hiredis
- **Eigen3**: Matrix operations for re-referencing (header-only)
- **iir1**: IIR filter library for `sosfilt` replacement (header-only)
  - Repository: https://github.com/berndporr/iir1
  - Provides cascaded biquad (SOS) filtering with state management
  - Drop-in replacement for scipy.signal.sosfilt()

### Optional Libraries (Phase 2)
- **yaml-cpp**: Configuration file parsing (for now use JSON)
- **spdlog**: High-performance logging (for now use printf)

### Build System
- **CMake**: Build configuration
- **Manual installation**: Download header-only libraries (Eigen, iir1)
- **System package**: Install hiredis via brew/apt

## Performance Targets

### Hardware Assumptions
- 16-core CPU (e.g., Intel Xeon or AMD Ryzen)
- 32GB+ RAM
- SSD storage
- Gigabit network for Redis

### Performance Goals
- **Throughput**: Process 128 channels @ 30 kHz in real-time
- **Latency**: <2ms end-to-end processing latency
- **CPU Usage**: <80% to leave headroom for OS and Redis
- **Memory**: <4GB total memory usage
- **Scalability**: Near-linear scaling with additional filter threads

### Expected Speedup vs Python
- **Filtering**: 10-15x speedup (C++ + parallelization)
- **Re-referencing**: 5-10x speedup (Eigen optimizations)
- **Feature extraction**: 10x speedup (C++ + SIMD)
- **Overall**: 10-12x total speedup

## Testing Strategy

### Unit Tests
- Individual filter correctness
- Matrix operation accuracy
- Redis serialization/deserialization
- Threading synchronization

### Integration Tests
- End-to-end pipeline validation
- Performance benchmarking
- Redis connectivity and failover
- Multi-threaded stress testing

### Validation
- Compare output against Python reference implementation
- Verify real-time performance under load
- Test with actual neural data files

## Key Design Decisions Summary

### Why use Redis from the start?
**Problem**: Need to decouple NSx file parsing from C++ development
**Solution**: Python reads NSx → Redis streams, C++ reads from Redis
**Benefits**:
- Use proven infrastructure from brand-nsp ecosystem
- Same code path for validation and production
- Easy debugging (inspect Redis data)
- Can replay data at any speed
- Zero changes when moving to real-time hardware

### Why use iir1 library instead of implementing filters from scratch?
**Problem**: `scipy.signal.sosfilt()` is complex (cascaded biquad filters with state)
**Solution**: Use proven iir1 library (header-only, designed for real-time DSP)
**Benefit**: Trust existing implementation, save weeks of debugging

### Why implement convolution ourselves?
**Problem**: Need `np.convolve(..., mode='valid')` for reverse FIR pass
**Solution**: Write simple 10-line sliding dot product
**Benefit**: No library overhead, trivial to verify, fast enough

### Why use Eigen for re-referencing?
**Problem**: Need fast matrix-vector multiply
**Solution**: Eigen is industry-standard, header-only, NumPy-like API
**Benefit**: Automatic SIMD optimization, well-tested

### Why pre-compute filter coefficients in Python?
**Problem**: Filter design algorithms are complex
**Solution**: Let scipy compute coefficients once, save to JSON, C++ loads them
**Benefit**: Trust scipy's proven algorithms, C++ just applies coefficients

### Processing Flow
```
Python Replay Script
  → Load NSx file with brpylib
  → Write 1ms chunks to Redis stream
    - Key: "raw_neural_data"
    - Fields: timestamps, samples

C++ Processor (NeuroStream)
  → Read from Redis ("raw_neural_data")
  → Re-reference (Eigen matrix multiply)
  → Filter forward pass (iir1 library)
  → Filter reverse pass (custom convolution)
  → Spike detection (threshold crossing)
  → Spike power (log10 of mean squared)
  → Write to Redis streams
    - "spike_events" (binary spikes)
    - "spike_band_power" (float32 power)

Python Validation Script
  → Read C++ output from Redis
  → Compare with Python reference
  → Verify numerical accuracy
```

**Key Insight**: By using Redis from day 1, Phase 1 (validation) and Phase 3 (production) share the same Redis interface. When you connect to real hardware (cerebusAdapter), the C++ code doesn't change at all!

This plan provides a pragmatic, phased approach: **correctness first, optimization later**.
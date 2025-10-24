#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace neurostream {

/**
 * Configuration parameters for neural signal processing
 */
struct ProcessingConfig {
    // Data dimensions
    int n_channels = 2048;
    int n_samples = 30;  // samples per processing chunk (1ms @ 30kHz)
    float sample_rate = 30000.0f;

    // Re-referencing
    enum class RereferenceType {
        NONE,
        CAR,  // Common Average Reference
        LRR   // Linear Regression Reference
    };
    RereferenceType reref_type = RereferenceType::LRR;
    std::string reref_params_file;

    // Filtering
    int filter_order = 4;
    float filter_low_freq = 250.0f;
    float filter_high_freq = 5000.0f;
    bool use_causal_filter = false;  // false = acausal (zero-phase)

    // Feature extraction
    float threshold_multiplier = -4.5f;
    std::string threshold_file;

    // I/O
    std::string input_data_file;
    std::string input_metadata_file;
    std::string output_spikes_file;
    std::string output_power_file;
};

/**
 * Raw data chunk (input from file)
 */
struct RawChunk {
    std::vector<float> data;           // n_channels x n_samples (row-major)
    std::vector<uint64_t> timestamps;  // n_samples timestamps
    int n_channels;
    int n_samples;

    RawChunk(int channels, int samples)
        : n_channels(channels), n_samples(samples) {
        data.resize(channels * samples);
        timestamps.resize(samples);
    }

    // Access data[channel][sample]
    float& operator()(int channel, int sample) {
        return data[channel * n_samples + sample];
    }

    const float& operator()(int channel, int sample) const {
        return data[channel * n_samples + sample];
    }
};

/**
 * Re-referenced data chunk
 */
struct RerefChunk {
    std::vector<float> data;  // n_channels x n_samples (row-major)
    int n_channels;
    int n_samples;

    RerefChunk(int channels, int samples)
        : n_channels(channels), n_samples(samples) {
        data.resize(channels * samples);
    }

    float& operator()(int channel, int sample) {
        return data[channel * n_samples + sample];
    }

    const float& operator()(int channel, int sample) const {
        return data[channel * n_samples + sample];
    }
};

/**
 * Filtered data chunk
 */
struct FilteredChunk {
    std::vector<float> data;  // n_channels x n_samples (row-major)
    int n_channels;
    int n_samples;

    FilteredChunk(int channels, int samples)
        : n_channels(channels), n_samples(samples) {
        data.resize(channels * samples);
    }

    float& operator()(int channel, int sample) {
        return data[channel * n_samples + sample];
    }

    const float& operator()(int channel, int sample) const {
        return data[channel * n_samples + sample];
    }
};

/**
 * Extracted features (output)
 */
struct FeatureChunk {
    std::vector<uint8_t> spikes;     // n_channels binary spike detection
    std::vector<float> spike_power;  // n_channels spike-band power (dB)
    uint64_t timestamp;
    int n_channels;

    FeatureChunk(int channels)
        : n_channels(channels), timestamp(0) {
        spikes.resize(channels);
        spike_power.resize(channels);
    }
};

} // namespace neurostream

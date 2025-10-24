#pragma once

#include "data_structures.hpp"
#include <vector>
#include <string>

namespace neurostream {

/**
 * Feature extraction: spike detection and spike-band power
 */
class FeatureExtractor {
public:
    FeatureExtractor(int n_channels);

    // Load threshold parameters from JSON file
    void load_thresholds(const std::string& threshold_file);

    // Set thresholds manually
    void set_thresholds(const std::vector<float>& thresholds);

    // Extract features from filtered data
    void process(const FilteredChunk& input, FeatureChunk& output);

private:
    int n_channels_;
    std::vector<float> thresholds_;  // Per-channel spike detection thresholds
    bool thresholds_loaded_;

    // Helper functions
    bool detect_spike(const float* data, int n_samples, float threshold);
    float calculate_spike_power(const float* data, int n_samples);
};

} // namespace neurostream

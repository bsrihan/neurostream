#pragma once

#include "data_structures.hpp"
#include <vector>

namespace neurostream {

/**
 * IIR Butterworth filter implementation using Second-Order Sections (SOS)
 *
 * For simplicity, starting with causal filtering.
 * Acausal (zero-phase) filtering can be added later.
 */
class IIRFilter {
public:
    IIRFilter();

    // Initialize filter coefficients
    void initialize(int order, float low_freq, float high_freq, float sample_rate);

    // Process single channel of data (in-place filtering possible)
    void process(const float* input, float* output, int n_samples);

    // Reset filter state (for new recording)
    void reset_state();

private:
    // Second-order sections coefficients [b0, b1, b2, a0, a1, a2] per section
    std::vector<std::vector<float>> sos_coeffs_;

    // Filter state (two delay elements per SOS section)
    std::vector<float> z_;  // state variables

    int num_sections_;
    bool initialized_;

    // Helper: apply single SOS section
    void apply_sos_section(const float* input, float* output, int n_samples,
                          const std::vector<float>& sos, float& z1, float& z2);
};

/**
 * Bank of filters for processing multiple channels
 */
class FilterBank {
public:
    FilterBank(int n_channels, int order, float low_freq, float high_freq, float sample_rate);

    // Process all channels
    void process(const RerefChunk& input, FilteredChunk& output);

    // Reset all filter states
    void reset_all();

private:
    int n_channels_;
    std::vector<IIRFilter> filters_;  // One filter per channel
};

} // namespace neurostream

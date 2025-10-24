#pragma once

#include "data_structures.hpp"
#include <Eigen/Dense>
#include <string>

namespace neurostream {

/**
 * Re-reference engine using pre-computed weights
 */
class RereferenceEngine {
public:
    RereferenceEngine(int n_channels);

    // Load re-referencing parameters from JSON file
    void load_parameters(const std::string& params_file);

    // Apply re-referencing to a chunk
    void process(const RawChunk& input, RerefChunk& output);

private:
    int n_channels_;
    Eigen::MatrixXf reref_matrix_;  // (n_channels x n_channels)
    bool params_loaded_;
};

} // namespace neurostream

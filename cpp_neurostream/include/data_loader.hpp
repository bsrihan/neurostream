#pragma once

#include "data_structures.hpp"
#include <fstream>
#include <memory>

namespace neurostream {

/**
 * Loads binary neural data exported from Python
 */
class BinaryDataLoader {
public:
    BinaryDataLoader(const std::string& metadata_file, const std::string& data_file);
    ~BinaryDataLoader();

    // Load metadata
    void load_metadata();

    // Read next chunk of data (returns false when no more data)
    bool read_chunk(RawChunk& chunk);

    // Get metadata
    int get_n_channels() const { return n_channels_; }
    int get_n_samples() const { return n_samples_total_; }
    float get_sample_rate() const { return sample_rate_; }

private:
    std::string metadata_file_;
    std::string data_file_;
    std::string timestamps_file_;

    std::ifstream data_stream_;
    std::ifstream timestamps_stream_;

    int n_channels_;
    int n_samples_total_;
    float sample_rate_;
    int samples_read_;
};

} // namespace neurostream

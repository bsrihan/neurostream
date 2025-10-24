#include "data_loader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Simple JSON parsing (for metadata only - just extract values we need)
#include <regex>

namespace neurostream {

BinaryDataLoader::BinaryDataLoader(const std::string& metadata_file,
                                   const std::string& data_file)
    : metadata_file_(metadata_file)
    , data_file_(data_file)
    , n_channels_(0)
    , n_samples_total_(0)
    , sample_rate_(0.0f)
    , samples_read_(0) {
}

BinaryDataLoader::~BinaryDataLoader() {
    if (data_stream_.is_open()) {
        data_stream_.close();
    }
    if (timestamps_stream_.is_open()) {
        timestamps_stream_.close();
    }
}

void BinaryDataLoader::load_metadata() {
    std::ifstream file(metadata_file_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open metadata file: " + metadata_file_);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Simple regex-based JSON parsing (for our specific format)
    std::regex n_channels_regex(R"("n_channels"\s*:\s*(\d+))");
    std::regex n_samples_regex(R"("n_samples"\s*:\s*(\d+))");
    std::regex sample_rate_regex(R"("sample_rate"\s*:\s*([\d.]+))");
    std::regex timestamps_file_regex(R"("timestamps_file"\s*:\s*"([^"]+)")");

    std::smatch match;

    if (std::regex_search(content, match, n_channels_regex)) {
        n_channels_ = std::stoi(match[1]);
    } else {
        throw std::runtime_error("Could not parse n_channels from metadata");
    }

    if (std::regex_search(content, match, n_samples_regex)) {
        n_samples_total_ = std::stoi(match[1]);
    } else {
        throw std::runtime_error("Could not parse n_samples from metadata");
    }

    if (std::regex_search(content, match, sample_rate_regex)) {
        sample_rate_ = std::stof(match[1]);
    } else {
        throw std::runtime_error("Could not parse sample_rate from metadata");
    }

    if (std::regex_search(content, match, timestamps_file_regex)) {
        timestamps_file_ = match[1];
    } else {
        throw std::runtime_error("Could not parse timestamps_file from metadata");
    }

    // Open data files
    data_stream_.open(data_file_, std::ios::binary);
    if (!data_stream_.is_open()) {
        throw std::runtime_error("Failed to open data file: " + data_file_);
    }

    // Construct full path to timestamps file (same directory as metadata)
    size_t last_slash = metadata_file_.find_last_of("/\\");
    std::string dir = (last_slash != std::string::npos)
        ? metadata_file_.substr(0, last_slash + 1)
        : "";
    std::string timestamps_path = dir + timestamps_file_;

    timestamps_stream_.open(timestamps_path, std::ios::binary);
    if (!timestamps_stream_.is_open()) {
        throw std::runtime_error("Failed to open timestamps file: " + timestamps_path);
    }

    std::cout << "[DataLoader] Loaded metadata:" << std::endl;
    std::cout << "  Channels: " << n_channels_ << std::endl;
    std::cout << "  Total samples: " << n_samples_total_ << std::endl;
    std::cout << "  Sample rate: " << sample_rate_ << " Hz" << std::endl;
    std::cout << "  Duration: " << (n_samples_total_ / sample_rate_) << " seconds" << std::endl;

    samples_read_ = 0;
}

bool BinaryDataLoader::read_chunk(RawChunk& chunk) {
    if (samples_read_ >= n_samples_total_) {
        return false;  // No more data
    }

    int samples_to_read = std::min(chunk.n_samples, n_samples_total_ - samples_read_);

    // Read data (channels x samples, row-major = C-style)
    // Binary file is stored as: all samples for ch0, then all samples for ch1, etc.
    size_t bytes_to_read = n_channels_ * samples_to_read * sizeof(float);
    data_stream_.read(reinterpret_cast<char*>(chunk.data.data()), bytes_to_read);

    if (!data_stream_.good() && !data_stream_.eof()) {
        throw std::runtime_error("Error reading from data file");
    }

    // Read timestamps
    size_t timestamp_bytes = samples_to_read * sizeof(uint64_t);
    timestamps_stream_.read(reinterpret_cast<char*>(chunk.timestamps.data()),
                           timestamp_bytes);

    if (!timestamps_stream_.good() && !timestamps_stream_.eof()) {
        throw std::runtime_error("Error reading from timestamps file");
    }

    // Update actual samples read
    chunk.n_samples = samples_to_read;
    samples_read_ += samples_to_read;

    return true;
}

} // namespace neurostream

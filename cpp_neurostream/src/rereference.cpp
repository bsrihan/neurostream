#include "rereference.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <regex>

namespace neurostream {

RereferenceEngine::RereferenceEngine(int n_channels)
    : n_channels_(n_channels)
    , params_loaded_(false) {
    // Initialize identity matrix (no re-referencing by default)
    reref_matrix_ = Eigen::MatrixXf::Identity(n_channels, n_channels);
}

void RereferenceEngine::load_parameters(const std::string& params_file) {
    std::cout << "[Rereference] Loading parameters from: " << params_file << std::endl;

    std::ifstream file(params_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open reref params file: " + params_file);
    }

    // Read entire file
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Find the "rereference_parameters" array in JSON
    // Format: "rereference_parameters": [[val, val, ...], [val, val, ...], ...]
    std::regex start_regex(R"("rereference_parameters"\s*:\s*\[)");
    std::smatch match;

    if (!std::regex_search(content, match, start_regex)) {
        throw std::runtime_error("Could not find 'rereference_parameters' in JSON");
    }

    // Find the position where the array starts
    size_t start_pos = match.position() + match.length();

    // Find matching closing bracket
    int bracket_count = 1;
    size_t end_pos = start_pos;
    while (end_pos < content.length() && bracket_count > 0) {
        if (content[end_pos] == '[') bracket_count++;
        if (content[end_pos] == ']') bracket_count--;
        end_pos++;
    }

    if (bracket_count != 0) {
        throw std::runtime_error("Malformed JSON: unmatched brackets in rereference_parameters");
    }

    // Extract the array content (without outer brackets)
    std::string array_content = content.substr(start_pos, end_pos - start_pos - 1);

    // Parse the matrix values
    // This is a simple parser for nested arrays of numbers
    std::vector<std::vector<float>> matrix_data;
    std::vector<float> current_row;

    std::regex number_regex(R"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
    auto numbers_begin = std::sregex_iterator(array_content.begin(), array_content.end(), number_regex);
    auto numbers_end = std::sregex_iterator();

    int row = 0;
    int col = 0;
    for (std::sregex_iterator i = numbers_begin; i != numbers_end; ++i) {
        std::smatch match = *i;
        float value = std::stof(match.str());
        current_row.push_back(value);
        col++;

        if (col == n_channels_) {
            matrix_data.push_back(current_row);
            current_row.clear();
            col = 0;
            row++;
        }
    }

    if (row != n_channels_) {
        throw std::runtime_error("Reref matrix dimensions don't match n_channels");
    }

    // Convert to Eigen matrix
    reref_matrix_.resize(n_channels_, n_channels_);
    for (int i = 0; i < n_channels_; i++) {
        for (int j = 0; j < n_channels_; j++) {
            reref_matrix_(i, j) = matrix_data[i][j];
        }
    }

    params_loaded_ = true;
    std::cout << "[Rereference] Loaded " << n_channels_ << "x" << n_channels_ << " matrix" << std::endl;
}

void RereferenceEngine::process(const RawChunk& input, RerefChunk& output) {
    if (!params_loaded_) {
        throw std::runtime_error("Re-reference parameters not loaded");
    }

    // Convert input data to Eigen matrix (channels x samples)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input_matrix(input.data.data(), input.n_channels, input.n_samples);

    // Apply re-referencing: output = reref_matrix * input
    // reref_matrix is (n_channels x n_channels)
    // input_matrix is (n_channels x n_samples)
    // result is (n_channels x n_samples)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        output_matrix(output.data.data(), output.n_channels, output.n_samples);

    output_matrix = reref_matrix_ * input_matrix;
}

} // namespace neurostream

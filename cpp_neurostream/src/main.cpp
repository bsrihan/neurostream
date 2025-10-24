#include "redis_io.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "=== NeuroStream C++ Redis Test ===" << std::endl;

    // Parse command line args (simplified for now)
    std::string redis_host = "127.0.0.1";
    int redis_port = 6379;
    std::string input_stream = "raw_neural_data";

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            redis_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            redis_port = std::stoi(argv[++i]);
        } else if (arg == "--stream" && i + 1 < argc) {
            input_stream = argv[++i];
        }
    }

    std::cout << "Connecting to Redis at " << redis_host << ":" << redis_port << std::endl;
    std::cout << "Reading from stream: " << input_stream << std::endl;

    try {
        // Create Redis data source
        neurostream::RedisDataSource source(redis_host, redis_port, input_stream);

        if (!source.is_connected()) {
            std::cerr << "Failed to connect to Redis!" << std::endl;
            return 1;
        }

        std::cout << "Connected successfully!" << std::endl;
        std::cout << "\nReading chunks... (press Ctrl+C to stop)\n" << std::endl;

        // Read and process chunks
        neurostream::RawChunk chunk(2048, 30);  // Default dimensions
        int chunks_read = 0;
        const int max_chunks = 10;  // Only read 10 chunks for testing

        while (chunks_read < max_chunks) {
            if (source.read_chunk(chunk)) {
                chunks_read++;

                std::cout << "=== Chunk " << chunks_read << " ===" << std::endl;
                std::cout << "  Channels: " << chunk.n_channels << std::endl;
                std::cout << "  Samples: " << chunk.n_samples << std::endl;
                std::cout << "  Total data points: " << chunk.data.size() << std::endl;

                // Print first few timestamps
                std::cout << "  First 5 timestamps: ";
                for (int i = 0; i < std::min(5, (int)chunk.timestamps.size()); i++) {
                    std::cout << chunk.timestamps[i] << " ";
                }
                std::cout << std::endl;

                // Print some sample values from first channel
                std::cout << "  First channel, first 10 samples: ";
                for (int i = 0; i < std::min(10, chunk.n_samples); i++) {
                    std::cout << std::fixed << std::setprecision(1)
                              << chunk(0, i) << " ";
                }
                std::cout << std::endl;

                // Calculate some basic stats
                float min_val = chunk.data[0];
                float max_val = chunk.data[0];
                float sum = 0.0f;
                for (float val : chunk.data) {
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    sum += val;
                }
                float mean = sum / chunk.data.size();

                std::cout << "  Data range: [" << min_val << ", " << max_val << "]" << std::endl;
                std::cout << "  Data mean: " << mean << std::endl;
                std::cout << std::endl;

            } else {
                std::cout << "No data available (timeout), waiting..." << std::endl;
            }
        }

        std::cout << "\nSuccessfully read " << chunks_read << " chunks!" << std::endl;
        std::cout << "Redis interface is working correctly." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

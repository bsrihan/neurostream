#pragma once

#include "data_structures.hpp"
#include <string>
#include <memory>

// Forward declare hiredis types to avoid including in header
struct redisContext;
struct redisReply;

namespace neurostream {

/**
 * Redis data source - reads raw neural data from Redis streams
 * Compatible with cerebusAdapter output format
 */
class RedisDataSource {
public:
    RedisDataSource(const std::string& host, int port, const std::string& stream);
    ~RedisDataSource();

    // Read next chunk from Redis stream (blocking)
    bool read_chunk(RawChunk& chunk);

    // Check if connected
    bool is_connected() const;

    // Get stream info
    const std::string& get_stream_name() const { return stream_name_; }

private:
    std::string host_;
    int port_;
    std::string stream_name_;
    std::string last_id_;  // Last read entry ID ("$" for latest)

    redisContext* context_;
    bool connected_;

    void connect();
    void disconnect();
};

/**
 * Redis data sink - writes processed features to Redis streams
 */
class RedisDataSink {
public:
    RedisDataSink(const std::string& host, int port,
                  const std::string& spike_stream,
                  const std::string& power_stream);
    ~RedisDataSink();

    // Write feature chunk to Redis streams
    void write_features(const FeatureChunk& chunk);

    // Check if connected
    bool is_connected() const;

private:
    std::string host_;
    int port_;
    std::string spike_stream_;
    std::string power_stream_;

    redisContext* context_;
    bool connected_;

    void connect();
    void disconnect();
};

} // namespace neurostream

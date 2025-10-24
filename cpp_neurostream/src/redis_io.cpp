#include "redis_io.hpp"
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace neurostream {

//-----------------------------------------------------------------------------
// RedisDataSource
//-----------------------------------------------------------------------------

RedisDataSource::RedisDataSource(const std::string& host, int port,
                                 const std::string& stream)
    : host_(host)
    , port_(port)
    , stream_name_(stream)
    , last_id_("0")  // Start from beginning of stream (use "0" not "$")
    , context_(nullptr)
    , connected_(false) {
    connect();
}

RedisDataSource::~RedisDataSource() {
    disconnect();
}

void RedisDataSource::connect() {
    std::cout << "[RedisDataSource] Connecting to " << host_ << ":" << port_ << std::endl;

    context_ = redisConnect(host_.c_str(), port_);

    if (context_ == nullptr || context_->err) {
        if (context_) {
            std::cerr << "[RedisDataSource] Connection error: " << context_->errstr << std::endl;
            redisFree(context_);
            context_ = nullptr;
        } else {
            std::cerr << "[RedisDataSource] Connection error: can't allocate redis context" << std::endl;
        }
        throw std::runtime_error("Failed to connect to Redis");
    }

    connected_ = true;
    std::cout << "[RedisDataSource] Connected successfully" << std::endl;
}

void RedisDataSource::disconnect() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
    connected_ = false;
}

bool RedisDataSource::is_connected() const {
    return connected_ && context_ != nullptr;
}

bool RedisDataSource::read_chunk(RawChunk& chunk) {
    if (!is_connected()) {
        throw std::runtime_error("Not connected to Redis");
    }

    // Use XREAD to read from stream (blocking with timeout)
    // XREAD BLOCK 1000 STREAMS stream_name last_id
    redisReply* reply = (redisReply*)redisCommand(context_,
        "XREAD BLOCK 1000 COUNT 1 STREAMS %s %s",
        stream_name_.c_str(),
        last_id_.c_str());

    if (reply == nullptr) {
        std::cerr << "[RedisDataSource] Redis command failed" << std::endl;
        return false;
    }

    // Check if we got data
    if (reply->type == REDIS_REPLY_NIL || reply->elements == 0) {
        // Timeout - no data available
        freeReplyObject(reply);
        return false;
    }

    // Parse XREAD response structure:
    // Array of streams -> Array of [stream_name, Array of entries]
    // Each entry is [entry_id, [field1, value1, field2, value2, ...]]

    if (reply->type != REDIS_REPLY_ARRAY || reply->elements != 1) {
        std::cerr << "[RedisDataSource] Unexpected response format" << std::endl;
        freeReplyObject(reply);
        return false;
    }

    redisReply* stream_data = reply->element[0];
    if (stream_data->type != REDIS_REPLY_ARRAY || stream_data->elements != 2) {
        std::cerr << "[RedisDataSource] Unexpected stream format" << std::endl;
        freeReplyObject(reply);
        return false;
    }

    redisReply* entries = stream_data->element[1];
    if (entries->type != REDIS_REPLY_ARRAY || entries->elements == 0) {
        freeReplyObject(reply);
        return false;
    }

    // Get first entry
    redisReply* entry = entries->element[0];
    if (entry->type != REDIS_REPLY_ARRAY || entry->elements != 2) {
        std::cerr << "[RedisDataSource] Unexpected entry format" << std::endl;
        freeReplyObject(reply);
        return false;
    }

    // Update last_id for next read
    last_id_ = std::string(entry->element[0]->str, entry->element[0]->len);

    // Parse fields (array of [key, value, key, value, ...])
    redisReply* fields = entry->element[1];
    if (fields->type != REDIS_REPLY_ARRAY) {
        std::cerr << "[RedisDataSource] Unexpected fields format" << std::endl;
        freeReplyObject(reply);
        return false;
    }

    // Extract timestamps and samples from fields
    const uint8_t* timestamps_data = nullptr;
    size_t timestamps_len = 0;
    const uint8_t* samples_data = nullptr;
    size_t samples_len = 0;

    for (size_t i = 0; i < fields->elements; i += 2) {
        if (i + 1 >= fields->elements) break;

        const char* key = fields->element[i]->str;
        const uint8_t* value = (const uint8_t*)fields->element[i + 1]->str;
        size_t value_len = fields->element[i + 1]->len;

        if (strcmp(key, "timestamps") == 0) {
            timestamps_data = value;
            timestamps_len = value_len;
        } else if (strcmp(key, "samples") == 0) {
            samples_data = value;
            samples_len = value_len;
        }
    }

    if (!timestamps_data || !samples_data) {
        std::cerr << "[RedisDataSource] Missing required fields" << std::endl;
        freeReplyObject(reply);
        return false;
    }

    // Parse data into chunk
    // timestamps: uint64 array
    size_t n_timestamps = timestamps_len / sizeof(uint64_t);
    chunk.timestamps.resize(n_timestamps);
    std::memcpy(chunk.timestamps.data(), timestamps_data, timestamps_len);

    // samples: int16 array (channels x samples, row-major)
    size_t n_values = samples_len / sizeof(int16_t);
    chunk.n_samples = n_timestamps;
    chunk.n_channels = n_values / n_timestamps;

    // Convert int16 to float32
    chunk.data.resize(n_values);
    const int16_t* samples_int16 = (const int16_t*)samples_data;
    for (size_t i = 0; i < n_values; i++) {
        chunk.data[i] = static_cast<float>(samples_int16[i]);
    }

    freeReplyObject(reply);
    return true;
}

//-----------------------------------------------------------------------------
// RedisDataSink
//-----------------------------------------------------------------------------

RedisDataSink::RedisDataSink(const std::string& host, int port,
                             const std::string& spike_stream,
                             const std::string& power_stream)
    : host_(host)
    , port_(port)
    , spike_stream_(spike_stream)
    , power_stream_(power_stream)
    , context_(nullptr)
    , connected_(false) {
    connect();
}

RedisDataSink::~RedisDataSink() {
    disconnect();
}

void RedisDataSink::connect() {
    std::cout << "[RedisDataSink] Connecting to " << host_ << ":" << port_ << std::endl;

    context_ = redisConnect(host_.c_str(), port_);

    if (context_ == nullptr || context_->err) {
        if (context_) {
            std::cerr << "[RedisDataSink] Connection error: " << context_->errstr << std::endl;
            redisFree(context_);
            context_ = nullptr;
        } else {
            std::cerr << "[RedisDataSink] Connection error: can't allocate redis context" << std::endl;
        }
        throw std::runtime_error("Failed to connect to Redis");
    }

    connected_ = true;
    std::cout << "[RedisDataSink] Connected successfully" << std::endl;
}

void RedisDataSink::disconnect() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
    connected_ = false;
}

bool RedisDataSink::is_connected() const {
    return connected_ && context_ != nullptr;
}

void RedisDataSink::write_features(const FeatureChunk& chunk) {
    if (!is_connected()) {
        throw std::runtime_error("Not connected to Redis");
    }

    // Write spikes (binary uint8 array)
    redisReply* spike_reply = (redisReply*)redisCommand(context_,
        "XADD %s * timestamp %llu spikes %b",
        spike_stream_.c_str(),
        (unsigned long long)chunk.timestamp,
        chunk.spikes.data(),
        chunk.spikes.size());

    if (spike_reply == nullptr) {
        std::cerr << "[RedisDataSink] Failed to write spikes" << std::endl;
    } else {
        freeReplyObject(spike_reply);
    }

    // Write spike-band power (float32 array)
    redisReply* power_reply = (redisReply*)redisCommand(context_,
        "XADD %s * timestamp %llu power %b",
        power_stream_.c_str(),
        (unsigned long long)chunk.timestamp,
        chunk.spike_power.data(),
        chunk.spike_power.size() * sizeof(float));

    if (power_reply == nullptr) {
        std::cerr << "[RedisDataSink] Failed to write power" << std::endl;
    } else {
        freeReplyObject(power_reply);
    }
}

} // namespace neurostream

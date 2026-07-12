#pragma once
#include <string>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <fstream>
#include <sstream>
#include <cstring>
#include <filesystem>
#include "world/chunk.hpp"
#include "world/serialization.hpp"

class SaveManager {
public:
    explicit SaveManager(const std::string& worldPath);
    ~SaveManager();

    // Delete copy/move
    SaveManager(const SaveManager&) = delete;
    SaveManager& operator=(const SaveManager&) = delete;
    SaveManager(SaveManager&&) = delete;
    SaveManager& operator=(SaveManager&&) = delete;

    // Save a chunk to disk (async)
    void saveChunk(const Chunk& chunk);

    // Load a chunk from disk (returns nullopt if not found)
    std::optional<Chunk> loadChunk(int chunkX, int chunkZ) const;

    // Save world metadata
    void saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime);

    // Load world metadata
    struct WorldMetadata {
        uint32_t seed;
        Vec3 spawnPos;
        uint64_t worldTime;
    };
    std::optional<WorldMetadata> loadMetadata() const;

    // Flush all pending saves
    void flush();

    // Get world path
    const std::string& getWorldPath() const;

private:
    std::string worldPath_;
    std::string regionsPath_;
    std::string metadataPath_;

    // Save queue state
    std::mutex saveMutex_;
    std::condition_variable saveCondition_;
    std::atomic<bool> running_{true};
    std::thread saveThread_;

    // Queue for async saves
    struct SaveJob {
        std::string regionFile;
        std::vector<uint8_t> compressedData;
        int chunkIndex; // index within region
    };
    std::queue<SaveJob> saveQueue_;

    // Background save thread
    void saveLoop();

    // Compress data with LZ4
    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) const;
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& data, size_t maxDecompressedSize) const;

    // Get region file path for chunk coordinates
    std::string getRegionPath(int chunkX, int chunkZ) const;

    // Get region coordinates from chunk coordinates
    static int getRegionCoord(int chunkCoord);

    // Get chunk index within a region (0-255)
    static int getChunkIndexInRegion(int chunkX, int chunkZ);

    // Ensure directory exists
    bool ensureDirectory(const std::string& path) const;

    // Write compressed data to file
    bool writeFile(const std::string& path, const std::vector<uint8_t>& data) const;

    // Read file to bytes
    std::optional<std::vector<uint8_t>> readFile(const std::string& path) const;
};

#pragma once
#include "world/chunk.hpp"
#include "world/serialization.hpp"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

// ---------------------------------------------------------------------------
// SaveManager — one file per chunk, sharded into 32×32-chunk directories
// (regions/r.X.Z/c.<cx>.<cz>.dat, LZ4-compressed RYCH payload).
//
// Per-chunk files are correct by construction: no offset table, no
// read-modify-write, atomic via write-temp-then-rename. The old packed
// region format stored ONE chunk per region file (writeFile clobbered the
// whole file per save), silently losing every other edited chunk.
//
// Serialization and compression run on the save thread; a queued chunk
// stays visible to loadChunk through the pending map, so unload-then-return
// can never read a stale file while its replacement sits in the queue.
// Chunks handed to saveChunkAsync must be quiescent (unloaded, or the game
// is quitting) — the save thread reads their blocks without locks.
// ---------------------------------------------------------------------------
class SaveManager {
public:
    explicit SaveManager(const std::string& worldPath);
    ~SaveManager();

    // Delete copy/move
    SaveManager(const SaveManager&) = delete;
    SaveManager& operator=(const SaveManager&) = delete;
    SaveManager(SaveManager&&) = delete;
    SaveManager& operator=(SaveManager&&) = delete;

    // Save a chunk to disk (async; copies the chunk)
    void saveChunk(const Chunk& chunk);

    // Save a chunk to disk without copying. The chunk must not be mutated
    // until the write completes (see the class comment).
    void saveChunkAsync(std::shared_ptr<const Chunk> chunk);

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

    // Save queue state (mutable: loadChunk is const but reads the shield)
    mutable std::mutex saveMutex_;
    std::condition_variable saveCondition_;
    std::atomic<bool> running_{true};
    std::thread saveThread_;

    // Counter for in-flight writes (queued + being written)
    std::atomic<int> pendingWrites_{0};

    // Queue for async saves (serialize + compress happen on the save thread)
    struct SaveJob {
        std::shared_ptr<const Chunk> chunk;
    };
    std::queue<SaveJob> saveQueue_;

    // Queued-but-unwritten chunks, keyed by packed coords — the load shield
    // (guarded by saveMutex_)
    std::unordered_map<uint64_t, std::shared_ptr<const Chunk>> pendingChunks_;

    // Background save thread
    void saveLoop();

    // Compress data with LZ4
    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) const;
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& data,
                                    size_t maxDecompressedSize) const;

    // Per-chunk file paths, sharded by 32×32-chunk region directory
    std::string getChunkDir(int chunkX, int chunkZ) const;
    std::string getChunkPath(int chunkX, int chunkZ) const;

    // Get region coordinates from chunk coordinates
    static int getRegionCoord(int chunkCoord);

    static uint64_t packChunkKey(int chunkX, int chunkZ);

    // Ensure directory exists
    bool ensureDirectory(const std::string& path) const;

    // Write data to file atomically (temp file + rename)
    bool writeFile(const std::string& path, const std::vector<uint8_t>& data) const;

    // Read file to bytes
    std::optional<std::vector<uint8_t>> readFile(const std::string& path) const;
};

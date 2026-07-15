#pragma once

#include "world/chunk.hpp"
#include "world/fluid.hpp"
#include "world/serialization.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class SaveManager {
public:
    static constexpr size_t PLAYER_INVENTORY_SLOTS = 9;
    static constexpr size_t MAX_PENDING_SAVE_JOBS = 32'768;

    // Optional deterministic write failures used by persistence regression
    // tests. Production callers leave this unset.
    struct TestHooks {
        std::atomic<size_t> writeFailuresRemaining{0};
        std::atomic<size_t> loadFailuresReported{0};
        std::atomic<bool> pauseWrites{false};
    };

    struct PlayerMetadata {
        float yaw = 0.0f;
        float pitch = 0.0f;
        int health = 20;
        int selectedSlot = 0;
        std::array<BlockType, PLAYER_INVENTORY_SLOTS> inventory = {
            BlockType::STONE,     BlockType::DIRT,   BlockType::GRASS,
            BlockType::LOG,       BlockType::PLANKS, BlockType::SAND,
            BlockType::SANDSTONE, BlockType::GLASS,  BlockType::FLOWER_RED,
        };
    };

    explicit SaveManager(const std::string& worldPath,
                         std::shared_ptr<TestHooks> testHooks = nullptr);
    ~SaveManager();

    SaveManager(const SaveManager&) = delete;
    SaveManager& operator=(const SaveManager&) = delete;
    SaveManager(SaveManager&&) = delete;
    SaveManager& operator=(SaveManager&&) = delete;

    void saveChunk(const Chunk& chunk);
    void saveChunkAsync(std::shared_ptr<const Chunk> chunk);
    std::optional<Chunk> loadChunk(ChunkPos pos) const;
    std::vector<int32_t> savedSections(ColumnPos pos) const;
    std::unordered_map<ColumnPos, std::vector<int32_t>>
    savedSectionsForColumns(std::span<const ColumnPos> columns) const;
    bool saveDeferredFluidFrontiers(const std::vector<FluidBoundaryFrontier>& frontiers);
    std::vector<FluidBoundaryFrontier> loadDeferredFluidFrontiers() const;

    bool saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime);
    bool saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime,
                      const PlayerMetadata& player);

    struct WorldMetadata {
        uint32_t seed = 0;
        Vec3 spawnPos{};
        uint64_t worldTime = 0;
        uint32_t chunkFormatVersion = CHUNK_VERSION;
        uint32_t generatorVersion = 2;
        PlayerMetadata player;
    };
    std::optional<WorldMetadata> loadMetadata() const;

    // Waits for all queued cube writes and reports whether each one reached
    // both its cube file and column manifest.
    bool flush();
    const std::string& getWorldPath() const { return worldPath_; }
    size_t pendingSaveCount() const;
    uint64_t coalescedSaveCount() const { return coalescedSaves_.load(std::memory_order_relaxed); }

private:
    std::string worldPath_;
    std::string regionsPath_;
    std::string metadataPath_;

    mutable std::mutex saveMutex_;
    std::condition_variable saveCondition_;
    std::atomic<bool> running_{true};
    std::thread saveThread_;
    std::atomic<size_t> pendingWrites_{0};
    std::shared_ptr<TestHooks> testHooks_;

    std::deque<ChunkPos> saveQueue_;
    std::unordered_set<ChunkPos> queuedChunks_;
    std::unordered_map<ChunkPos, std::shared_ptr<const Chunk>> pendingChunks_;
    std::unordered_set<ChunkPos> failedChunks_;
    mutable std::unordered_set<ChunkPos> reportedLoadFailures_;
    std::atomic<uint64_t> coalescedSaves_{0};

    struct ColumnManifest {
        std::vector<int32_t> editedSections;
        std::vector<FluidBoundaryFrontier> fluidFrontiers;
    };
    mutable std::mutex manifestMutex_;
    mutable std::mutex manifestWriteMutex_;
    mutable std::unordered_map<ColumnPos, ColumnManifest> manifests_;

    void enqueueChunkSnapshot(std::shared_ptr<const Chunk> chunk);
    void saveLoop();
    std::vector<uint8_t> compress(const std::vector<uint8_t>& data) const;
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& data,
                                    size_t maxDecompressedSize) const;

    std::string getChunkDir(ColumnPos pos) const;
    std::string getChunkPath(ChunkPos pos) const;
    std::string getManifestPath(ColumnPos pos) const;
    static int64_t getRegionCoord(int64_t chunkCoord);
    bool updateManifest(ChunkPos pos) const;
    void loadManifestIndex();
    void recoverOrphanedCubes();
    bool writeManifest(ColumnPos pos, const ColumnManifest& manifest) const;

    bool ensureDirectory(const std::string& path) const;
    bool writeFile(const std::string& path, const std::vector<uint8_t>& data) const;
    bool writeFileWithRetries(const std::string& path, const std::vector<uint8_t>& data) const;
    std::optional<std::vector<uint8_t>> readFile(const std::string& path) const;
    void reportLoadFailureOnce(ChunkPos pos, const char* reason) const;
};

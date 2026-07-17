#pragma once

#include "world/chunk.hpp"
#include "world/fluid.hpp"
#include "world/furnace.hpp"
#include "world/item.hpp"
#include "world/serialization.hpp"
#include "world/world_config.hpp"

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
    static constexpr size_t PLAYER_INVENTORY_SLOTS = 36;
    static constexpr size_t PLAYER_HOTBAR_SLOTS = 9;
    static constexpr size_t MAX_PENDING_SAVE_JOBS = 32'768;
    static constexpr uint32_t CURRENT_GENERATOR_VERSION = 3;
    static constexpr const char* CURRENT_REGIONS_DIRECTORY = "regions-v3";

    // Optional deterministic write failures used by persistence regression
    // tests. Production callers leave this unset.
    struct TestHooks {
        std::atomic<size_t> writeFailuresRemaining{0};
        std::atomic<size_t> loadFailuresReported{0};
        std::atomic<bool> pauseWrites{false};
    };

    // Defaults reproduce the classic starter hotbar for worlds without a
    // player section (fresh dirs and the legacy metadata shape).
    static constexpr std::array<BlockType, PLAYER_HOTBAR_SLOTS> DEFAULT_HOTBAR_BLOCKS = {
        BlockType::STONE,     BlockType::DIRT,   BlockType::GRASS,
        BlockType::LOG,       BlockType::PLANKS, BlockType::SAND,
        BlockType::SANDSTONE, BlockType::GLASS,  BlockType::FLOWER_RED,
    };

    static constexpr std::array<ItemStack, PLAYER_INVENTORY_SLOTS> defaultInventory() {
        std::array<ItemStack, PLAYER_INVENTORY_SLOTS> slots{};
        for (size_t slot = 0; slot < DEFAULT_HOTBAR_BLOCKS.size(); ++slot) {
            slots[slot] = ItemStack{itemFromBlock(DEFAULT_HOTBAR_BLOCKS[slot]), 1, 0};
        }
        return slots;
    }

    struct PlayerMetadata {
        float yaw = 0.0f;
        float pitch = 0.0f;
        int health = 20;
        int hunger = 20;
        int selectedSlot = 0;
        std::array<ItemStack, PLAYER_INVENTORY_SLOTS> inventory = defaultInventory();
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

    struct WorldMetadata {
        uint32_t seed = 0;
        Vec3 spawnPos{};  // respawn anchor, fixed at world creation
        Vec3 playerPos{}; // last saved player position; legacy saves reuse spawnPos
        uint64_t worldTime = 0;
        uint32_t chunkFormatVersion = CHUNK_VERSION;
        uint32_t generatorVersion = CURRENT_GENERATOR_VERSION;
        std::string name;                       // display name; defaults to the directory
        GameMode gameMode = GameMode::CREATIVE; // legacy worlds keep free flight
        GenerationSettings generation;
        uint64_t createdMs = 0; // unix ms; 0 = unknown (legacy)
        uint64_t lastPlayedMs = 0;
        PlayerMetadata player;
    };

    // Writes every field and stamps lastPlayedMs with the current wall clock.
    bool saveMetadata(const WorldMetadata& metadata);
    std::optional<WorldMetadata> loadMetadata() const;
    // Metadata for an arbitrary world file without constructing a SaveManager
    // (world enumeration must not spawn save threads per row).
    static std::optional<WorldMetadata> readMetadataFile(const std::string& path);

    // Stateful blocks (furnaces) persist in one per-world sidecar; missing
    // files read as empty, unknown record types are skipped for forward
    // compatibility.
    bool saveBlockEntities(const FurnaceMap& furnaces);
    FurnaceMap loadBlockEntities() const;

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
    std::string blockEntitiesPath_;

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

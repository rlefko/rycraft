#pragma once
#include "common/thread_pool.hpp"
#include "world/biome.hpp"
#include "world/caves.hpp"
#include "world/chunk.hpp"
#include "world/chunk_pos.hpp"
#include "world/noise.hpp"
#include "world/ores.hpp"
#include "world/structures.hpp"
#include "world/surface.hpp"
#include "world/terrain.hpp"
#include "world/trees.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

class SaveManager;

class World {
public:
    explicit World(uint32_t seed, int viewDistance = 32);
    ~World();

    // Delete copy/move (contains non-copyable ThreadPool and mutex)
    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) = delete;
    World& operator=(World&&) = delete;

    // Get or generate a chunk (thread-safe)
    std::shared_ptr<Chunk> getChunk(int chunkX, int chunkZ);

    // Get block at world position (thread-safe)
    BlockType getBlock(int x, int y, int z);

    // Set block at world position (thread-safe, marks chunk dirty)
    void setBlock(int x, int y, int z, BlockType type);

    // Get terrain height at world position
    double getTerrainHeight(int x, int z) const;

    // Get biome at world position
    Biome getBiome(int x, int z) const;

    // Get all loaded chunks
    std::vector<std::shared_ptr<Chunk>> getLoadedChunks() const;

    // Get chunks that need mesh updates
    std::vector<std::shared_ptr<Chunk>> getDirtyChunks();

    // Mark chunk as meshed
    void markChunkMeshed(int chunkX, int chunkZ);

    // Get seed
    uint32_t getSeed() const;

    // View distance
    int getViewDistance() const;
    void setViewDistance(int distance);

    // Update chunks around player position (synchronous)
    void updatePlayerPosition(int playerX, int playerZ);

    // Unload chunks outside the view distance (called by updatePlayerPosition)
    void unloadDistantChunks();

    // Start async generation of chunks around player
    void generateAroundPlayer(int playerX, int playerZ);

    // Get generation queue status
    size_t getPendingChunkCount() const;

    // Attach the save manager (non-owning). Once set, getChunk and the async
    // generation path try loading a chunk from disk before generating it —
    // without this, saved block edits were written but never read back.
    void setSaveManager(SaveManager* saveManager);

private:
    uint32_t seed_;
    int viewDistance_;

    // Chunk storage keyed by chunk grid coordinate
    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> chunks_;
    mutable std::mutex chunksMutex_;

    // Persistence (non-owning; the engine owns the SaveManager)
    SaveManager* saveManager_ = nullptr;

    // Generation components
    TerrainGenerator terrainGen_;
    BiomeGenerator biomeGen_;
    CaveGenerator caveGen_;
    OreGenerator oreGen_;
    TreeGenerator treeGen_;
    StructureGenerator structureGen_;

    // Player position for chunk loading. hasPlayerChunk_ stays false until
    // the first updatePlayerPosition call, so the spawn area streams in even
    // when the player starts in chunk (0, 0).
    int playerChunkX_ = 0;
    int playerChunkZ_ = 0;
    bool hasPlayerChunk_ = false;

    // Async generation state
    std::shared_ptr<ThreadPool> genPool_; // lazily initialized
    std::unordered_map<ChunkPos, std::future<void>> pendingGenerations_;
    mutable std::mutex pendingMutex_;

    // Generate a single chunk (synchronous)
    void generateChunk(std::shared_ptr<Chunk> chunk);

    // Generate chunk asynchronously
    void generateChunkAsync(int chunkX, int chunkZ);

    // Load the chunk from disk when possible, otherwise generate it
    // (with the flat-fallback policy on generation failure).
    std::shared_ptr<Chunk> loadOrGenerateChunk(int chunkX, int chunkZ);
};

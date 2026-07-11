#pragma once
#include "world/chunk.hpp"
#include "world/terrain.hpp"
#include "world/biome.hpp"
#include "world/surface.hpp"
#include "world/caves.hpp"
#include "world/ores.hpp"
#include "world/trees.hpp"
#include "world/structures.hpp"
#include "world/noise.hpp"
#include "common/thread_pool.hpp"

#include <unordered_map>
#include <mutex>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <cstdint>
#include <future>

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

    // Start async generation of chunks around player
    void generateAroundPlayer(int playerX, int playerZ);

    // Get generation queue status
    size_t getPendingChunkCount() const;

private:
    uint32_t seed_;
    int viewDistance_;

    // Chunk storage — spatial hash
    std::unordered_map<std::string, std::shared_ptr<Chunk>> chunks_;
    mutable std::mutex chunksMutex_;

    // Generation components
    TerrainGenerator terrainGen_;
    BiomeGenerator biomeGen_;
    CaveGenerator caveGen_;
    OreGenerator oreGen_;
    TreeGenerator treeGen_;
    StructureGenerator structureGen_;

    // Player position for chunk loading
    int playerChunkX_ = 0;
    int playerChunkZ_ = 0;

    // Async generation state
    std::shared_ptr<ThreadPool> genPool_;  // lazily initialized
    std::unordered_map<std::string, std::future<void>> pendingGenerations_;
    mutable std::mutex pendingMutex_;

    // Generate a single chunk (synchronous)
    void generateChunk(std::shared_ptr<Chunk> chunk);

    // Generate chunk asynchronously with LOD
    void generateChunkAsync(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ);

    // Chunk key
    static std::string chunkKey(int cx, int cz);
};

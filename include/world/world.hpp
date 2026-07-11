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

#include <unordered_map>
#include <mutex>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <cstdint>

class World {
public:
    explicit World(uint32_t seed, int viewDistance = 32);

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
    std::vector<std::shared_ptr<Chunk>> getLoadedChunks();

    // Get chunks that need mesh updates
    std::vector<std::shared_ptr<Chunk>> getDirtyChunks();

    // Mark chunk as meshed
    void markChunkMeshed(int chunkX, int chunkZ);

    // Get seed
    uint32_t getSeed() const;

    // View distance
    int getViewDistance() const;
    void setViewDistance(int distance);

    // Update chunks around player position
    void updatePlayerPosition(int playerX, int playerZ);

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

    // Generate a single chunk
    void generateChunk(std::shared_ptr<Chunk> chunk);

    // Chunk key
    static std::string chunkKey(int cx, int cz);
};

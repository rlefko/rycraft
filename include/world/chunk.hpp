#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <optional>
#include "common/math.hpp"
#include "world/block_properties.hpp"

// Backwards compatibility alias
constexpr int CHUNK_SIZE = 16;

constexpr int CHUNK_WIDTH = 16;
constexpr int CHUNK_DEPTH = 16;
constexpr int CHUNK_HEIGHT = 256;
constexpr int CHUNK_VOLUME = CHUNK_WIDTH * CHUNK_DEPTH * CHUNK_HEIGHT;

enum class Biome : uint8_t {
    DeepOcean = 0,
    Ocean = 1,
    Plains = 2,
    Forest = 3,
    Taiga = 4,
    Desert = 5,
    ExtremeHills = 6,
    Swamp = 7,
    MushroomIsland = 8,
    IceSpikes = 9,
    Count = 10
};

struct Chunk {
    // Chunk coordinates in world space
    int chunkX;
    int chunkZ;

    // Block data — flat array for cache-friendly access
    std::vector<BlockType> blocks;

    // Biome map — 16x16 per chunk
    std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH> biomes;

    // Height map — highest non-air block per xz position
    std::array<int, CHUNK_WIDTH * CHUNK_DEPTH> heightMap;

    // Meshing state
    bool needsMeshUpdate = false;
    bool meshed = false;
    bool generated = false;

    Chunk(int cx, int cz);

    // Block access
    BlockType getBlock(int localX, int localY, int localZ) const;
    void setBlock(int localX, int localY, int localZ, BlockType type);

    // World coordinate access
    BlockType getBlockWorld(int x, int y, int z) const;
    void setBlockWorld(int x, int y, int z, BlockType type);

    // Chunk coordinate conversion
    static int worldToChunk(int worldCoord);
    static int chunkToWorld(int chunkCoord, int localCoord);

    // Get world position of chunk corner
    Vec3 getWorldPosition() const;

    // Mark for mesh rebuild
    void markDirty();

    // Mark chunk as meshed (clear dirty flag)
    void setMeshed(bool value);

    // Get AABB of this chunk in world space
    AABB getAABB() const;
};

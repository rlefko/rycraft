#pragma once
#include "common/math.hpp"
#include "world/block_properties.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

// Backwards compatibility alias
constexpr int CHUNK_SIZE = 16;

constexpr int CHUNK_WIDTH = 16;
constexpr int CHUNK_DEPTH = 16;
constexpr int CHUNK_HEIGHT = 256;
constexpr int CHUNK_VOLUME = CHUNK_WIDTH * CHUNK_DEPTH * CHUNK_HEIGHT;

// Open-to-sky air below this floods with water during generation. The one
// definition every producer/consumer shares (the shaders repeat the value
// as a literal — MSL can't include this header).
constexpr int SEA_LEVEL = 64;

// Any surface at or above this gets a snow top, and trees stop growing —
// the two rules must agree, so they share the constant.
constexpr int SNOW_LINE = 108;

enum class Biome : uint8_t {
    DEEP_OCEAN = 0,
    OCEAN = 1,
    PLAINS = 2,
    FOREST = 3,
    TAIGA = 4,
    DESERT = 5,
    EXTREME_HILLS = 6,
    SWAMP = 7,
    MUSHROOM_ISLAND = 8,
    ICE_SPIKES = 9,
    // Values are persisted in saves: only append, never renumber.
    BEACH = 10,
    RIVER = 11,
    BIRCH_FOREST = 12,
    FLOWER_FIELD = 13,
    COUNT = 14
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

    // Player edits since the last save — only these chunks persist on
    // unload/quit (regenerated chunks are pure functions of the seed)
    bool modifiedSinceSave = false;

    // Content revision, bumped by every block edit (self + boundary
    // neighbors). The async mesher stamps results with the version it
    // snapshotted; a mismatch on arrival means "upload, then re-mesh" —
    // a newer mesh always beats a hole.
    std::atomic<uint32_t> version{1};

    Chunk(int cx, int cz);

    // The atomic makes Chunk non-copyable by default; copies happen only on
    // single-threaded paths (serialization, the save-queue shield).
    Chunk(const Chunk& other);
    Chunk(Chunk&& other) noexcept;
    Chunk& operator=(const Chunk& other);

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

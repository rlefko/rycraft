#pragma once

#include "common/math.hpp"
#include "world/block_properties.hpp"
#include "world/chunk_pos.hpp"
#include "world/fluid.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

inline constexpr int CHUNK_EDGE = 16;
inline constexpr int CHUNK_SIZE = CHUNK_EDGE;
inline constexpr int CHUNK_WIDTH = CHUNK_EDGE;
inline constexpr int CHUNK_HEIGHT = CHUNK_EDGE;
inline constexpr int CHUNK_DEPTH = CHUNK_EDGE;
inline constexpr int CHUNK_VOLUME = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;

inline constexpr int WORLD_MIN_Y = -128;
inline constexpr int WORLD_MAX_Y = 511;
inline constexpr int WORLD_MIN_CHUNK_Y = WORLD_MIN_Y / CHUNK_EDGE;
inline constexpr int WORLD_MAX_CHUNK_Y = WORLD_MAX_Y / CHUNK_EDGE;
inline constexpr int WORLD_VERTICAL_CHUNKS = WORLD_MAX_CHUNK_Y - WORLD_MIN_CHUNK_Y + 1;
inline constexpr int SEA_LEVEL = 64;
inline constexpr int32_t INCOMPLETE_SKY_PATH_CUTOFF = WORLD_MAX_Y + 2;

constexpr uint8_t packDerivedLight(uint8_t skyLight, uint8_t blockLight) {
    return static_cast<uint8_t>(((skyLight & 0x0FU) << 4U) | (blockLight & 0x0FU));
}

constexpr uint8_t derivedSkyLight(uint8_t packedLight) {
    return static_cast<uint8_t>(packedLight >> 4U);
}

constexpr uint8_t derivedBlockLight(uint8_t packedLight) {
    return static_cast<uint8_t>(packedLight & 0x0FU);
}

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
    BEACH = 10,
    RIVER = 11,
    BIRCH_FOREST = 12,
    FLOWER_FIELD = 13,
    SAVANNA = 14,
    TROPICAL_RAINFOREST = 15,
    TEMPERATE_RAINFOREST = 16,
    SHRUBLAND = 17,
    STEPPE = 18,
    COLD_DESERT = 19,
    BADLANDS = 20,
    TUNDRA = 21,
    ALPINE = 22,
    MANGROVE = 23,
    FROZEN_OCEAN = 24,
    VOLCANIC_BARREN = 25,
    GLACIER = 26,
    MONTANE_GRASSLAND = 27,
    FLOODED_GRASSLAND = 28,
    MEDITERRANEAN_WOODLAND = 29,
    TEMPERATE_CONIFER_FOREST = 30,
    TROPICAL_CONIFER_FOREST = 31,
    TROPICAL_DRY_FOREST = 32,
    COUNT = 33
};

class Chunk {
public:
    int64_t chunkX = 0;
    int32_t chunkY = 0;
    int64_t chunkZ = 0;

    bool needsMeshUpdate = false;
    bool meshed = false;
    bool generated = false;
    bool modifiedSinceSave = false;
    std::atomic<uint32_t> version{1};

    explicit Chunk(ChunkPos pos);
    Chunk(int64_t cx, int64_t cz) : Chunk(ChunkPos{cx, 0, cz}) {}
    Chunk(int64_t cx, int32_t cy, int64_t cz) : Chunk(ChunkPos{cx, cy, cz}) {}
    Chunk(const Chunk& other);
    Chunk(Chunk&& other) noexcept;
    Chunk& operator=(const Chunk& other);

    ChunkPos pos() const { return {chunkX, chunkY, chunkZ}; }

    BlockType getBlock(int localX, int localY, int localZ) const;
    void setBlock(int localX, int localY, int localZ, BlockType type);
    BlockType getBlockWorld(int64_t x, int32_t y, int64_t z) const;
    void setBlockWorld(int64_t x, int32_t y, int64_t z, BlockType type);

    bool isUniform() const { return blocks_.empty(); }
    BlockType uniformBlock() const { return uniformBlock_; }
    const std::vector<BlockType>& denseBlocks() const { return blocks_; }
    std::vector<BlockType> copyBlocks() const;
    void replaceBlocks(std::vector<BlockType> blocks);
    void fill(BlockType type);
    void compactStorage();

    // Skylight and block light are derived and never serialized. The high
    // nibble stores skylight and the low nibble stores block light. A dark
    // cube keeps this storage empty; the first nonzero write materializes one
    // byte per cell.
    uint8_t getPackedLight(int localX, int localY, int localZ) const;
    uint8_t getSkyLight(int localX, int localY, int localZ) const;
    uint8_t getBlockLight(int localX, int localY, int localZ) const;
    void setSkyLight(int localX, int localY, int localZ, uint8_t level);
    void setBlockLight(int localX, int localY, int localZ, uint8_t level);
    bool hasDerivedLight() const { return !packedLight_.empty(); }
    bool hasBlockLight() const;
    const std::vector<uint8_t>& packedLightData() const { return packedLight_; }
    void replacePackedLight(std::vector<uint8_t> light);
    void clearDerivedLight() { packedLight_.clear(); }

    FluidState getFluidState(int localX, int localY, int localZ) const;
    void setFluidState(int localX, int localY, int localZ, FluidState state);
    bool hasExplicitFluidStates() const { return !fluidStates_.empty(); }
    const std::vector<uint8_t>& explicitFluidStates() const { return fluidStates_; }
    void replaceFluidStates(std::vector<uint8_t> states);

    static constexpr int index(int x, int y, int z) {
        return x + z * CHUNK_EDGE + y * CHUNK_EDGE * CHUNK_EDGE;
    }
    static int64_t worldToChunk(int64_t worldCoord) {
        return world_coord::floorDiv(worldCoord, static_cast<int64_t>(CHUNK_EDGE));
    }
    static int32_t worldToChunkY(int32_t worldY) {
        return world_coord::floorDiv(worldY, static_cast<int32_t>(CHUNK_EDGE));
    }
    static int32_t worldToLocal(int64_t worldCoord) {
        return world_coord::floorMod(worldCoord, CHUNK_EDGE);
    }
    static int32_t worldToLocalY(int32_t worldY) {
        return world_coord::floorMod(worldY, CHUNK_EDGE);
    }

    Vec3 getWorldPosition() const;
    void markDirty() { needsMeshUpdate = true; }
    void setMeshed(bool value) {
        meshed = value;
        if (value) needsMeshUpdate = false;
    }
    AABB getAABB() const;

private:
    BlockType uniformBlock_ = BlockType::AIR;
    std::vector<BlockType> blocks_;
    std::vector<uint8_t> fluidStates_;
    std::vector<uint8_t> packedLight_;

    void materialize();
};

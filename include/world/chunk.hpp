#pragma once

#include "common/math.hpp"
#include "world/block_properties.hpp"
#include "world/chunk_pos.hpp"
#include "world/fluid.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <vector>

inline constexpr int CHUNK_EDGE = 16;
inline constexpr int CHUNK_SIZE = CHUNK_EDGE;
inline constexpr int CHUNK_WIDTH = CHUNK_EDGE;
inline constexpr int CHUNK_HEIGHT = CHUNK_EDGE;
inline constexpr int CHUNK_DEPTH = CHUNK_EDGE;
inline constexpr int CHUNK_VOLUME = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;

inline constexpr int WORLD_MIN_Y = -128;
inline constexpr int WORLD_MAX_Y = 1407;
inline constexpr int WORLD_MIN_CHUNK_Y = WORLD_MIN_Y / CHUNK_EDGE;
inline constexpr int WORLD_MAX_CHUNK_Y = WORLD_MAX_Y / CHUNK_EDGE;
inline constexpr int WORLD_VERTICAL_CHUNKS = WORLD_MAX_CHUNK_Y - WORLD_MIN_CHUNK_Y + 1;
inline constexpr int SEA_LEVEL = 64;
inline constexpr int32_t INCOMPLETE_SKY_PATH_CUTOFF = WORLD_MAX_Y + 2;

inline constexpr uint8_t DERIVED_LIGHT_LEVEL_MASK = 0x0FU;
inline constexpr uint8_t MAX_DERIVED_LIGHT_LEVEL = DERIVED_LIGHT_LEVEL_MASK;

constexpr uint8_t packDerivedLight(uint8_t skyLight, uint8_t blockLight) {
    return static_cast<uint8_t>(((skyLight & DERIVED_LIGHT_LEVEL_MASK) << 4U) |
                                (blockLight & DERIVED_LIGHT_LEVEL_MASK));
}

constexpr uint8_t derivedSkyLight(uint8_t packedLight) {
    return static_cast<uint8_t>((packedLight >> 4U) & DERIVED_LIGHT_LEVEL_MASK);
}

constexpr uint8_t derivedBlockLight(uint8_t packedLight) {
    return static_cast<uint8_t>(packedLight & DERIVED_LIGHT_LEVEL_MASK);
}

constexpr float normalizedDerivedLight(uint8_t lightLevel) {
    return static_cast<float>(lightLevel & DERIVED_LIGHT_LEVEL_MASK) /
           static_cast<float>(MAX_DERIVED_LIGHT_LEVEL);
}

inline constexpr uint8_t FULL_SKY_PACKED_LIGHT = packDerivedLight(MAX_DERIVED_LIGHT_LEVEL, 0);
class VerticalSectionMask {
public:
    static constexpr size_t WORD_BITS = 64;
    static constexpr size_t WORD_COUNT =
        (static_cast<size_t>(WORLD_VERTICAL_CHUNKS) + WORD_BITS - 1) / WORD_BITS;

    constexpr void set(int32_t chunkY) {
        if (!validSection(chunkY)) return;
        const size_t index = sectionIndex(chunkY);
        words_[index / WORD_BITS] |= uint64_t{1} << (index % WORD_BITS);
    }

    constexpr void reset(int32_t chunkY) {
        if (!validSection(chunkY)) return;
        const size_t index = sectionIndex(chunkY);
        words_[index / WORD_BITS] &= ~(uint64_t{1} << (index % WORD_BITS));
    }

    constexpr void merge(const VerticalSectionMask& other) {
        for (size_t word = 0; word < WORD_COUNT; ++word)
            words_[word] |= other.words_[word];
    }

    constexpr bool contains(int32_t chunkY) const {
        if (!validSection(chunkY)) return false;
        const size_t index = sectionIndex(chunkY);
        return (words_[index / WORD_BITS] & (uint64_t{1} << (index % WORD_BITS))) != 0;
    }

    constexpr bool containsRange(int32_t firstChunkY, int32_t lastChunkY) const {
        if (firstChunkY > lastChunkY || !validSection(firstChunkY) || !validSection(lastChunkY)) {
            return false;
        }

        const size_t firstIndex = sectionIndex(firstChunkY);
        const size_t lastIndex = sectionIndex(lastChunkY);
        const size_t firstWord = firstIndex / WORD_BITS;
        const size_t lastWord = lastIndex / WORD_BITS;
        for (size_t word = firstWord; word <= lastWord; ++word) {
            const size_t firstBit = word == firstWord ? firstIndex % WORD_BITS : 0;
            const size_t lastBit = word == lastWord ? lastIndex % WORD_BITS : WORD_BITS - 1;
            const uint64_t lowMask = ~uint64_t{0} << firstBit;
            const uint64_t highMask = lastBit == WORD_BITS - 1
                                          ? ~uint64_t{0}
                                          : (uint64_t{1} << (lastBit + 1)) - uint64_t{1};
            const uint64_t required = lowMask & highMask;
            if ((words_[word] & required) != required) return false;
        }
        return true;
    }

    constexpr bool containsAllSetSections(const VerticalSectionMask& required, int32_t firstChunkY,
                                          int32_t lastChunkY = WORLD_MAX_CHUNK_Y) const {
        if (firstChunkY > lastChunkY || lastChunkY < WORLD_MIN_CHUNK_Y ||
            firstChunkY > WORLD_MAX_CHUNK_Y) {
            return true;
        }

        const size_t firstIndex = sectionIndex(std::max(firstChunkY, WORLD_MIN_CHUNK_Y));
        const size_t lastIndex = sectionIndex(std::min(lastChunkY, WORLD_MAX_CHUNK_Y));
        const size_t firstWord = firstIndex / WORD_BITS;
        const size_t lastWord = lastIndex / WORD_BITS;
        for (size_t word = firstWord; word <= lastWord; ++word) {
            const size_t firstBit = word == firstWord ? firstIndex % WORD_BITS : 0;
            const size_t lastBit = word == lastWord ? lastIndex % WORD_BITS : WORD_BITS - 1;
            const uint64_t lowMask = ~uint64_t{0} << firstBit;
            const uint64_t highMask = lastBit == WORD_BITS - 1
                                          ? ~uint64_t{0}
                                          : (uint64_t{1} << (lastBit + 1)) - uint64_t{1};
            const uint64_t requiredBits = required.words_[word] & lowMask & highMask;
            if ((words_[word] & requiredBits) != requiredBits) return false;
        }
        return true;
    }

    constexpr bool empty() const {
        for (uint64_t word : words_) {
            if (word != 0) return false;
        }
        return true;
    }

    constexpr int32_t highestSection() const {
        for (size_t wordIndex = WORD_COUNT; wordIndex > 0; --wordIndex) {
            const uint64_t word = words_[wordIndex - 1];
            if (word == 0) continue;
            const size_t bit = static_cast<size_t>(std::bit_width(word) - 1U);
            return WORLD_MIN_CHUNK_Y + static_cast<int32_t>((wordIndex - 1) * WORD_BITS + bit);
        }
        return WORLD_MIN_CHUNK_Y - 1;
    }

    constexpr bool operator==(const VerticalSectionMask&) const = default;

    template <typename Visitor>
    constexpr size_t visitSetSections(int32_t firstChunkY, int32_t lastChunkY,
                                      Visitor&& visitor) const {
        if (firstChunkY > lastChunkY || lastChunkY < WORLD_MIN_CHUNK_Y ||
            firstChunkY > WORLD_MAX_CHUNK_Y) {
            return 0;
        }
        const size_t firstIndex = sectionIndex(std::max(firstChunkY, WORLD_MIN_CHUNK_Y));
        const size_t lastIndex = sectionIndex(std::min(lastChunkY, WORLD_MAX_CHUNK_Y));
        size_t visited = 0;
        for (size_t wordIndex = firstIndex / WORD_BITS; wordIndex <= lastIndex / WORD_BITS;
             ++wordIndex) {
            const size_t firstBit =
                wordIndex == firstIndex / WORD_BITS ? firstIndex % WORD_BITS : 0;
            const size_t lastBit =
                wordIndex == lastIndex / WORD_BITS ? lastIndex % WORD_BITS : WORD_BITS - 1;
            const uint64_t lowMask = ~uint64_t{0} << firstBit;
            const uint64_t highMask = lastBit == WORD_BITS - 1
                                          ? ~uint64_t{0}
                                          : (uint64_t{1} << (lastBit + 1)) - uint64_t{1};
            uint64_t remaining = words_[wordIndex] & lowMask & highMask;
            while (remaining != 0) {
                const size_t bit = static_cast<size_t>(std::countr_zero(remaining));
                const int32_t section =
                    WORLD_MIN_CHUNK_Y + static_cast<int32_t>(wordIndex * WORD_BITS + bit);
                ++visited;
                if (!visitor(section)) return visited;
                remaining &= remaining - 1;
            }
        }
        return visited;
    }

    template <typename Visitor>
    constexpr size_t visitSetSectionsDescending(int32_t firstChunkY, int32_t lastChunkY,
                                                Visitor&& visitor) const {
        if (firstChunkY > lastChunkY || lastChunkY < WORLD_MIN_CHUNK_Y ||
            firstChunkY > WORLD_MAX_CHUNK_Y) {
            return 0;
        }
        const size_t firstIndex = sectionIndex(std::max(firstChunkY, WORLD_MIN_CHUNK_Y));
        const size_t lastIndex = sectionIndex(std::min(lastChunkY, WORLD_MAX_CHUNK_Y));
        size_t visited = 0;
        for (size_t wordCursor = lastIndex / WORD_BITS + 1; wordCursor > firstIndex / WORD_BITS;
             --wordCursor) {
            const size_t wordIndex = wordCursor - 1;
            const size_t firstBit =
                wordIndex == firstIndex / WORD_BITS ? firstIndex % WORD_BITS : 0;
            const size_t lastBit =
                wordIndex == lastIndex / WORD_BITS ? lastIndex % WORD_BITS : WORD_BITS - 1;
            const uint64_t lowMask = ~uint64_t{0} << firstBit;
            const uint64_t highMask = lastBit == WORD_BITS - 1
                                          ? ~uint64_t{0}
                                          : (uint64_t{1} << (lastBit + 1)) - uint64_t{1};
            uint64_t remaining = words_[wordIndex] & lowMask & highMask;
            while (remaining != 0) {
                const size_t bit = static_cast<size_t>(std::bit_width(remaining) - 1U);
                const int32_t section =
                    WORLD_MIN_CHUNK_Y + static_cast<int32_t>(wordIndex * WORD_BITS + bit);
                ++visited;
                if (!visitor(section)) return visited;
                remaining &= ~(uint64_t{1} << bit);
            }
        }
        return visited;
    }

private:
    static constexpr bool validSection(int32_t chunkY) {
        return chunkY >= WORLD_MIN_CHUNK_Y && chunkY <= WORLD_MAX_CHUNK_Y;
    }

    static constexpr size_t sectionIndex(int32_t chunkY) {
        return static_cast<size_t>(chunkY - WORLD_MIN_CHUNK_Y);
    }

    std::array<uint64_t, WORD_COUNT> words_{};
};

static_assert(WORLD_MIN_CHUNK_Y == -8);
static_assert(WORLD_MAX_CHUNK_Y == 87);
static_assert(WORLD_VERTICAL_CHUNKS == 96);
static_assert(VerticalSectionMask::WORD_COUNT == 2);

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
    // A cube with deferred publication lighting stays unavailable to mesh
    // snapshots until the bounded fixed-point transaction completes.
    bool publicationLightPending = false;
    bool publicationLightQueued = false;
    uint64_t publicationLightQueueToken = 0;
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

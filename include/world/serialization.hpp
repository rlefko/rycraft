#pragma once
#include "world/chunk.hpp"
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// Binary chunk format:
// Header (20 bytes):
//   uint32_t magic = 0x52594348 ("RYCH")
//   uint32_t version = 3
//   int32_t chunkX
//   int32_t chunkZ
//   uint32_t blockCount (should be CHUNK_VOLUME)
// Data:
//   blockCount bytes of BlockType
//   256 bytes of Biome (16x16)
//   512 bytes of int16_t height map (16x16, little-endian)
//
// v1 stored heights as int8_t, which overflowed at height 128 (terrain
// reaches it) and corrupted tree/structure placement on load. v3 keeps the
// v2 layout but marks the worldgen overhaul epoch: v2 worlds were generated
// with a carver that hollowed out nearly all terrain, so they regenerate.
// Old versions deserialize to nullopt, so pre-v3 chunks simply regenerate.

struct ChunkSaveHeader {
    uint32_t magic;
    uint32_t version;
    int32_t chunkX;
    int32_t chunkZ;
    uint32_t blockCount;
};

constexpr uint32_t CHUNK_MAGIC = 0x52594348;
constexpr uint32_t CHUNK_VERSION = 3;

constexpr size_t HEADER_SIZE = sizeof(ChunkSaveHeader);
constexpr size_t BIOME_DATA_SIZE = CHUNK_WIDTH * CHUNK_DEPTH * sizeof(Biome);
constexpr size_t HEIGHT_MAP_SIZE = CHUNK_WIDTH * CHUNK_DEPTH * sizeof(int16_t);

class ChunkSerializer {
public:
    // Serialize chunk to binary data
    static std::vector<uint8_t> serialize(const Chunk& chunk);

    // Deserialize binary data to chunk (returns nullopt on error)
    static std::optional<Chunk> deserialize(const std::span<const uint8_t>& data);

    // Get expected serialized size
    static size_t serializedSize(const Chunk& chunk);
};

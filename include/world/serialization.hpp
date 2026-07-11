#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <optional>
#include "world/chunk.hpp"

// Binary chunk format:
// Header (20 bytes):
//   uint32_t magic = 0x52594348 ("RYCH")
//   uint32_t version = 1
//   int32_t chunkX
//   int32_t chunkZ
//   uint32_t blockCount (should be CHUNK_VOLUME)
// Data:
//   blockCount bytes of BlockType
//   256 bytes of Biome (16x16)
//   256 bytes of int8_t height map (16x16)

struct ChunkSaveHeader {
    uint32_t magic;
    uint32_t version;
    int32_t chunkX;
    int32_t chunkZ;
    uint32_t blockCount;
};

constexpr uint32_t CHUNK_MAGIC = 0x52594348;
constexpr uint32_t CHUNK_VERSION = 1;

constexpr size_t HEADER_SIZE = sizeof(ChunkSaveHeader);
constexpr size_t BIOME_DATA_SIZE = CHUNK_WIDTH * CHUNK_DEPTH * sizeof(Biome);
constexpr size_t HEIGHT_MAP_SIZE = CHUNK_WIDTH * CHUNK_DEPTH * sizeof(int8_t);

class ChunkSerializer {
public:
    // Serialize chunk to binary data
    static std::vector<uint8_t> serialize(const Chunk& chunk);

    // Deserialize binary data to chunk (returns nullopt on error)
    static std::optional<Chunk> deserialize(const std::span<const uint8_t>& data);

    // Get expected serialized size
    static size_t serializedSize(const Chunk& chunk);
};

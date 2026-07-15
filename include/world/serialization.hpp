#pragma once

#include "world/chunk.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#pragma pack(push, 1)
struct ChunkSaveHeader {
    uint32_t magic;
    uint32_t version;
    int64_t chunkX;
    int32_t chunkY;
    uint32_t flags;
    int64_t chunkZ;
    uint32_t blockCount;
    uint32_t fluidStateCount;
    // IEEE CRC-32 over the uncompressed bytes following this header.
    uint32_t payloadChecksum;
};
#pragma pack(pop)

static_assert(sizeof(ChunkSaveHeader) == 44);

inline constexpr uint32_t CHUNK_MAGIC = 0x52594348;
inline constexpr uint32_t CHUNK_VERSION = 4;
inline constexpr uint32_t CHUNK_FLAG_UNIFORM = 1U << 0U;
inline constexpr uint32_t CHUNK_FLAG_FLUID_STATES = 1U << 1U;
inline constexpr size_t HEADER_SIZE = sizeof(ChunkSaveHeader);

enum class ChunkPayloadValidation : uint8_t {
    VALID,
    INCOMPATIBLE,
    CHECKSUM_MISMATCH,
};

class ChunkSerializer {
public:
    static std::vector<uint8_t> serialize(const Chunk& chunk);
    static std::optional<Chunk> deserialize(std::span<const uint8_t> data);
    static size_t serializedSize(const Chunk& chunk);
    static uint32_t payloadChecksum(std::span<const uint8_t> payload);
    static ChunkPayloadValidation validatePayload(std::span<const uint8_t> data);
};

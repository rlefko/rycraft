#include "world/serialization.hpp"

#include <algorithm>
#include <cstring>

std::vector<uint8_t> ChunkSerializer::serialize(const Chunk& chunk) {
    size_t totalSize = serializedSize(chunk);
    std::vector<uint8_t> buffer(totalSize);

    size_t offset = 0;

    // Write header
    ChunkSaveHeader header;
    header.magic = CHUNK_MAGIC;
    header.version = CHUNK_VERSION;
    header.chunkX = static_cast<int32_t>(chunk.chunkX);
    header.chunkZ = static_cast<int32_t>(chunk.chunkZ);
    header.blockCount = static_cast<uint32_t>(chunk.blocks.size());

    std::memcpy(buffer.data() + offset, &header, sizeof(ChunkSaveHeader));
    offset += sizeof(ChunkSaveHeader);

    // Write blocks
    std::memcpy(buffer.data() + offset, chunk.blocks.data(), chunk.blocks.size());
    offset += chunk.blocks.size();

    // Write biomes
    std::memcpy(buffer.data() + offset, chunk.biomes.data(), BIOME_DATA_SIZE);
    offset += BIOME_DATA_SIZE;

    // Write height map as little-endian int16 (heights reach 128+, which
    // overflowed the old int8 format)
    for (size_t i = 0; i < chunk.heightMap.size(); ++i) {
        auto h = static_cast<int16_t>(chunk.heightMap[i]);
        buffer[offset + i * 2] = static_cast<uint8_t>(h & 0xFF);
        buffer[offset + i * 2 + 1] = static_cast<uint8_t>((h >> 8) & 0xFF);
    }
    offset += HEIGHT_MAP_SIZE;

    return buffer;
}

std::optional<Chunk> ChunkSerializer::deserialize(const std::span<const uint8_t>& data) {
    // Guard: minimum size check
    if (data.size() < HEADER_SIZE) {
        return std::nullopt;
    }

    // Parse header
    ChunkSaveHeader header;
    std::memcpy(&header, data.data(), sizeof(ChunkSaveHeader));

    // Validate magic number
    if (header.magic != CHUNK_MAGIC) {
        return std::nullopt;
    }

    // Validate version
    if (header.version != CHUNK_VERSION) {
        return std::nullopt;
    }

    // Validate block count
    if (header.blockCount != static_cast<uint32_t>(CHUNK_VOLUME)) {
        return std::nullopt;
    }

    // Calculate expected total size
    size_t expectedSize = HEADER_SIZE + header.blockCount + BIOME_DATA_SIZE + HEIGHT_MAP_SIZE;
    if (data.size() < expectedSize) {
        return std::nullopt;
    }

    // Parse data
    size_t offset = HEADER_SIZE;

    // Create chunk
    Chunk chunk(header.chunkX, header.chunkZ);

    // Read blocks
    std::memcpy(chunk.blocks.data(), data.data() + offset, header.blockCount);
    offset += header.blockCount;

    // Read biomes
    std::memcpy(chunk.biomes.data(), data.data() + offset, BIOME_DATA_SIZE);
    offset += BIOME_DATA_SIZE;

    // Read height map (little-endian int16)
    for (size_t i = 0; i < chunk.heightMap.size(); ++i) {
        auto h = static_cast<int16_t>(static_cast<uint16_t>(data[offset + i * 2]) |
                                      (static_cast<uint16_t>(data[offset + i * 2 + 1]) << 8));
        chunk.heightMap[i] = h;
    }

    chunk.generated = true;
    chunk.needsMeshUpdate = true;

    return chunk;
}

size_t ChunkSerializer::serializedSize(const Chunk& chunk) {
    return HEADER_SIZE + chunk.blocks.size() + BIOME_DATA_SIZE + HEIGHT_MAP_SIZE;
}

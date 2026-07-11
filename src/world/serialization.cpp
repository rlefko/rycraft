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

    // Write height map (convert int to int8_t)
    for (size_t i = 0; i < chunk.heightMap.size(); ++i) {
        buffer[offset + i] = static_cast<int8_t>(chunk.heightMap[i]);
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

    // Read height map (convert int8_t to int)
    for (size_t i = 0; i < chunk.heightMap.size(); ++i) {
        chunk.heightMap[i] = static_cast<int>(data[offset + i]);
    }

    chunk.generated = true;
    chunk.needsMeshUpdate = true;

    return chunk;
}

size_t ChunkSerializer::serializedSize(const Chunk& chunk) {
    return HEADER_SIZE + chunk.blocks.size() + BIOME_DATA_SIZE + HEIGHT_MAP_SIZE;
}

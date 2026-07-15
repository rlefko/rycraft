#include "world/serialization.hpp"

#include <algorithm>
#include <cstring>

namespace {

bool validBlockByte(uint8_t value) {
    return value < static_cast<uint8_t>(BlockType::COUNT);
}

bool validFluidByte(uint8_t value) {
    return value == 0xFF ||
           (value != FluidState::source().packed() && FluidState::isValidPacked(value));
}

bool readValidLayout(std::span<const uint8_t> data, ChunkSaveHeader& header) {
    if (data.size() < HEADER_SIZE) return false;
    std::memcpy(&header, data.data(), sizeof(header));
    if (header.magic != CHUNK_MAGIC || header.version != CHUNK_VERSION) return false;
    constexpr uint32_t KNOWN_FLAGS = CHUNK_FLAG_UNIFORM | CHUNK_FLAG_FLUID_STATES;
    if ((header.flags & ~KNOWN_FLAGS) != 0) return false;
    if (header.chunkY < WORLD_MIN_CHUNK_Y || header.chunkY > WORLD_MAX_CHUNK_Y) return false;

    const bool uniform = (header.flags & CHUNK_FLAG_UNIFORM) != 0;
    const bool hasFluidStates = (header.flags & CHUNK_FLAG_FLUID_STATES) != 0;
    if (header.blockCount != (uniform ? 1U : static_cast<uint32_t>(CHUNK_VOLUME))) return false;
    if (header.fluidStateCount != (hasFluidStates ? static_cast<uint32_t>(CHUNK_VOLUME) : 0U)) {
        return false;
    }
    return data.size() == HEADER_SIZE + header.blockCount + header.fluidStateCount;
}

} // namespace

std::vector<uint8_t> ChunkSerializer::serialize(const Chunk& chunk) {
    const bool uniform = chunk.isUniform();
    const bool hasFluidStates = chunk.hasExplicitFluidStates();
    ChunkSaveHeader header{
        .magic = CHUNK_MAGIC,
        .version = CHUNK_VERSION,
        .chunkX = chunk.chunkX,
        .chunkY = chunk.chunkY,
        .flags = static_cast<uint32_t>((uniform ? CHUNK_FLAG_UNIFORM : 0U) |
                                       (hasFluidStates ? CHUNK_FLAG_FLUID_STATES : 0U)),
        .chunkZ = chunk.chunkZ,
        .blockCount = uniform ? 1U : static_cast<uint32_t>(CHUNK_VOLUME),
        .fluidStateCount = hasFluidStates ? static_cast<uint32_t>(CHUNK_VOLUME) : 0U,
        .payloadChecksum = 0,
    };

    std::vector<uint8_t> buffer(serializedSize(chunk));
    std::memcpy(buffer.data(), &header, sizeof(header));
    size_t offset = sizeof(header);
    if (uniform) {
        buffer[offset++] = static_cast<uint8_t>(chunk.uniformBlock());
    } else {
        const auto& blocks = chunk.denseBlocks();
        std::memcpy(buffer.data() + offset, blocks.data(), blocks.size());
        offset += blocks.size();
    }
    if (hasFluidStates) {
        const auto& states = chunk.explicitFluidStates();
        std::memcpy(buffer.data() + offset, states.data(), states.size());
    }
    header.payloadChecksum =
        payloadChecksum(std::span<const uint8_t>(buffer).subspan(static_cast<size_t>(HEADER_SIZE)));
    std::memcpy(buffer.data(), &header, sizeof(header));
    return buffer;
}

std::optional<Chunk> ChunkSerializer::deserialize(std::span<const uint8_t> data) {
    if (validatePayload(data) != ChunkPayloadValidation::VALID) return std::nullopt;
    ChunkSaveHeader header{};
    std::memcpy(&header, data.data(), sizeof(header));

    const bool uniform = (header.flags & CHUNK_FLAG_UNIFORM) != 0;
    const bool hasFluidStates = (header.flags & CHUNK_FLAG_FLUID_STATES) != 0;

    size_t offset = HEADER_SIZE;
    Chunk chunk(ChunkPos{header.chunkX, header.chunkY, header.chunkZ});
    if (uniform) {
        if (!validBlockByte(data[offset])) return std::nullopt;
        chunk.fill(static_cast<BlockType>(data[offset++]));
    } else {
        std::vector<BlockType> blocks(CHUNK_VOLUME);
        for (size_t index = 0; index < blocks.size(); ++index) {
            const uint8_t value = data[offset + index];
            if (!validBlockByte(value)) return std::nullopt;
            blocks[index] = static_cast<BlockType>(value);
        }
        offset += blocks.size();
        chunk.replaceBlocks(std::move(blocks));
    }

    if (hasFluidStates) {
        std::vector<uint8_t> states(CHUNK_VOLUME);
        std::memcpy(states.data(), data.data() + offset, states.size());
        if (!std::all_of(states.begin(), states.end(), validFluidByte)) return std::nullopt;
        for (size_t index = 0; index < states.size(); ++index) {
            if (states[index] == 0xFF) continue;
            const BlockType block =
                chunk.isUniform() ? chunk.uniformBlock() : chunk.denseBlocks()[index];
            if (block != BlockType::WATER) return std::nullopt;
        }
        chunk.replaceFluidStates(std::move(states));
    }
    chunk.generated = true;
    chunk.needsMeshUpdate = true;
    return chunk;
}

size_t ChunkSerializer::serializedSize(const Chunk& chunk) {
    const size_t blockBytes = chunk.isUniform() ? 1 : CHUNK_VOLUME;
    const size_t fluidBytes = chunk.hasExplicitFluidStates() ? CHUNK_VOLUME : 0;
    return HEADER_SIZE + blockBytes + fluidBytes;
}

uint32_t ChunkSerializer::payloadChecksum(std::span<const uint8_t> payload) {
    uint32_t checksum = 0xFFFFFFFFU;
    for (uint8_t byte : payload) {
        checksum ^= byte;
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t polynomialMask = 0U - (checksum & 1U);
            checksum = (checksum >> 1U) ^ (0xEDB88320U & polynomialMask);
        }
    }
    return ~checksum;
}

ChunkPayloadValidation ChunkSerializer::validatePayload(std::span<const uint8_t> data) {
    ChunkSaveHeader header{};
    if (!readValidLayout(data, header)) return ChunkPayloadValidation::INCOMPATIBLE;
    const std::span<const uint8_t> payload = data.subspan(HEADER_SIZE);
    return payloadChecksum(payload) == header.payloadChecksum
               ? ChunkPayloadValidation::VALID
               : ChunkPayloadValidation::CHECKSUM_MISMATCH;
}

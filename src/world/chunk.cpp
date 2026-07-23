#include "world/chunk.hpp"

#include <algorithm>
#include <utility>

Chunk::Chunk(ChunkPos chunkPos) : chunkX(chunkPos.x), chunkY(chunkPos.y), chunkZ(chunkPos.z) {}

Chunk::Chunk(const Chunk& other)
    : chunkX(other.chunkX)
    , chunkY(other.chunkY)
    , chunkZ(other.chunkZ)
    , needsMeshUpdate(other.needsMeshUpdate)
    , meshed(other.meshed)
    , generated(other.generated)
    , modifiedSinceSave(other.modifiedSinceSave)
    , publicationLightPending(other.publicationLightPending)
    , publicationLightQueued(other.publicationLightQueued)
    , publicationLightQueueToken(other.publicationLightQueueToken)
    , version(other.version.load(std::memory_order_relaxed))
    , uniformBlock_(other.uniformBlock_)
    , blocks_(other.blocks_)
    , fluidStates_(other.fluidStates_)
    , packedLight_(other.packedLight_) {}

Chunk::Chunk(Chunk&& other) noexcept
    : chunkX(other.chunkX)
    , chunkY(other.chunkY)
    , chunkZ(other.chunkZ)
    , needsMeshUpdate(other.needsMeshUpdate)
    , meshed(other.meshed)
    , generated(other.generated)
    , modifiedSinceSave(other.modifiedSinceSave)
    , publicationLightPending(other.publicationLightPending)
    , publicationLightQueued(other.publicationLightQueued)
    , publicationLightQueueToken(other.publicationLightQueueToken)
    , version(other.version.load(std::memory_order_relaxed))
    , uniformBlock_(other.uniformBlock_)
    , blocks_(std::move(other.blocks_))
    , fluidStates_(std::move(other.fluidStates_))
    , packedLight_(std::move(other.packedLight_)) {}

Chunk& Chunk::operator=(const Chunk& other) {
    if (this == &other) return *this;
    chunkX = other.chunkX;
    chunkY = other.chunkY;
    chunkZ = other.chunkZ;
    needsMeshUpdate = other.needsMeshUpdate;
    meshed = other.meshed;
    generated = other.generated;
    modifiedSinceSave = other.modifiedSinceSave;
    publicationLightPending = other.publicationLightPending;
    publicationLightQueued = other.publicationLightQueued;
    publicationLightQueueToken = other.publicationLightQueueToken;
    version.store(other.version.load(std::memory_order_relaxed), std::memory_order_relaxed);
    uniformBlock_ = other.uniformBlock_;
    blocks_ = other.blocks_;
    fluidStates_ = other.fluidStates_;
    packedLight_ = other.packedLight_;
    return *this;
}

void Chunk::materialize() {
    if (blocks_.empty()) blocks_.assign(CHUNK_VOLUME, uniformBlock_);
}

BlockType Chunk::getBlock(int localX, int localY, int localZ) const {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return BlockType::AIR;
    }
    return blocks_.empty() ? uniformBlock_ : blocks_[index(localX, localY, localZ)];
}

void Chunk::setBlock(int localX, int localY, int localZ, BlockType type) {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return;
    }
    if (blocks_.empty() && type == uniformBlock_) return;
    materialize();
    blocks_[index(localX, localY, localZ)] = type;
}

uint8_t Chunk::getPackedLight(int localX, int localY, int localZ) const {
    if (packedLight_.empty()) return 0;
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return 0;
    }
    return packedLight_[index(localX, localY, localZ)];
}

uint8_t Chunk::getSkyLight(int localX, int localY, int localZ) const {
    return derivedSkyLight(getPackedLight(localX, localY, localZ));
}

uint8_t Chunk::getBlockLight(int localX, int localY, int localZ) const {
    return derivedBlockLight(getPackedLight(localX, localY, localZ));
}

void Chunk::setSkyLight(int localX, int localY, int localZ, uint8_t level) {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return;
    }
    level = std::min(level, MAX_DERIVED_LIGHT_LEVEL);
    if (packedLight_.empty()) {
        if (level == 0) return;
        packedLight_.assign(CHUNK_VOLUME, 0);
    }
    const int cell = index(localX, localY, localZ);
    packedLight_[cell] = packDerivedLight(level, derivedBlockLight(packedLight_[cell]));
}

void Chunk::setBlockLight(int localX, int localY, int localZ, uint8_t level) {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return;
    }
    level = std::min(level, MAX_DERIVED_LIGHT_LEVEL);
    if (packedLight_.empty()) {
        if (level == 0) return; // stay unallocated while fully dark
        packedLight_.assign(CHUNK_VOLUME, 0);
    }
    const int cell = index(localX, localY, localZ);
    packedLight_[cell] = packDerivedLight(derivedSkyLight(packedLight_[cell]), level);
}

bool Chunk::hasBlockLight() const {
    return std::any_of(packedLight_.begin(), packedLight_.end(),
                       [](uint8_t light) { return derivedBlockLight(light) != 0; });
}

void Chunk::replacePackedLight(std::vector<uint8_t> light) {
    if (!light.empty() && light.size() != CHUNK_VOLUME) return;
    if (std::none_of(light.begin(), light.end(), [](uint8_t packed) { return packed != 0; })) {
        light.clear();
    }
    packedLight_ = std::move(light);
}

BlockType Chunk::getBlockWorld(int64_t x, int32_t y, int64_t z) const {
    return getBlock(worldToLocal(x), worldToLocalY(y), worldToLocal(z));
}

void Chunk::setBlockWorld(int64_t x, int32_t y, int64_t z, BlockType type) {
    setBlock(worldToLocal(x), worldToLocalY(y), worldToLocal(z), type);
}

std::vector<BlockType> Chunk::copyBlocks() const {
    if (!blocks_.empty()) return blocks_;
    return std::vector<BlockType>(CHUNK_VOLUME, uniformBlock_);
}

void Chunk::replaceBlocks(std::vector<BlockType> blocks) {
    if (blocks.size() != CHUNK_VOLUME) return;
    blocks_ = std::move(blocks);
    compactStorage();
}

void Chunk::fill(BlockType type) {
    uniformBlock_ = type;
    blocks_.clear();
}

void Chunk::compactStorage() {
    if (blocks_.empty()) return;
    const BlockType first = blocks_.front();
    if (std::all_of(blocks_.begin(), blocks_.end(),
                    [first](BlockType block) { return block == first; })) {
        uniformBlock_ = first;
        blocks_.clear();
    }
}

FluidState Chunk::getFluidState(int localX, int localY, int localZ) const {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE || fluidStates_.empty()) {
        return FluidState::source();
    }
    const uint8_t packed = fluidStates_[index(localX, localY, localZ)];
    return packed == 0xFF ? FluidState::source() : FluidState::fromPacked(packed);
}

void Chunk::setFluidState(int localX, int localY, int localZ, FluidState state) {
    if (localX < 0 || localX >= CHUNK_EDGE || localY < 0 || localY >= CHUNK_EDGE || localZ < 0 ||
        localZ >= CHUNK_EDGE) {
        return;
    }
    if (fluidStates_.empty() && state.isSource()) return;
    if (fluidStates_.empty()) fluidStates_.assign(CHUNK_VOLUME, 0xFF);
    fluidStates_[index(localX, localY, localZ)] = state.isSource() ? 0xFF : state.packed();
}

void Chunk::replaceFluidStates(std::vector<uint8_t> states) {
    if (!states.empty() && states.size() != CHUNK_VOLUME) return;
    fluidStates_ = std::move(states);
}

Vec3 Chunk::getWorldPosition() const {
    return {static_cast<float>(chunkX * CHUNK_EDGE), static_cast<float>(chunkY * CHUNK_EDGE),
            static_cast<float>(chunkZ * CHUNK_EDGE)};
}

AABB Chunk::getAABB() const {
    const Vec3 min = getWorldPosition();
    const Vec3 max{min.x + CHUNK_EDGE, min.y + CHUNK_EDGE, min.z + CHUNK_EDGE};
    return {min, max};
}

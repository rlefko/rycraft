#include "world/chunk.hpp"

#include <algorithm>
#include <stdexcept>

Chunk::Chunk(int cx, int cz)
    : chunkX(cx)
    , chunkZ(cz)
    , blocks(CHUNK_VOLUME, BlockType::AIR)
    , biomes{}
    , heightMap{} {
    biomes.fill(Biome::PLAINS);
    heightMap.fill(0);
}

Chunk::Chunk(const Chunk& other)
    : chunkX(other.chunkX)
    , chunkZ(other.chunkZ)
    , blocks(other.blocks)
    , blockLight(other.blockLight)
    , biomes(other.biomes)
    , heightMap(other.heightMap)
    , needsMeshUpdate(other.needsMeshUpdate)
    , meshed(other.meshed)
    , generated(other.generated)
    , modifiedSinceSave(other.modifiedSinceSave)
    , version(other.version.load()) {}

Chunk::Chunk(Chunk&& other) noexcept
    : chunkX(other.chunkX)
    , chunkZ(other.chunkZ)
    , blocks(std::move(other.blocks))
    , blockLight(std::move(other.blockLight))
    , biomes(other.biomes)
    , heightMap(other.heightMap)
    , needsMeshUpdate(other.needsMeshUpdate)
    , meshed(other.meshed)
    , generated(other.generated)
    , modifiedSinceSave(other.modifiedSinceSave)
    , version(other.version.load()) {}

Chunk& Chunk::operator=(const Chunk& other) {
    chunkX = other.chunkX;
    chunkZ = other.chunkZ;
    blocks = other.blocks;
    blockLight = other.blockLight;
    biomes = other.biomes;
    heightMap = other.heightMap;
    needsMeshUpdate = other.needsMeshUpdate;
    meshed = other.meshed;
    generated = other.generated;
    modifiedSinceSave = other.modifiedSinceSave;
    version.store(other.version.load());
    return *this;
}

// Internal index calculation: Y-major for greedy meshing
static int chunkIndex(int x, int y, int z) {
    return x + z * CHUNK_WIDTH + y * CHUNK_WIDTH * CHUNK_DEPTH;
}

BlockType Chunk::getBlock(int localX, int localY, int localZ) const {
    if (localX < 0 || localX >= CHUNK_WIDTH) return BlockType::AIR;
    if (localY < 0 || localY >= CHUNK_HEIGHT) return BlockType::AIR;
    if (localZ < 0 || localZ >= CHUNK_DEPTH) return BlockType::AIR;
    return blocks[chunkIndex(localX, localY, localZ)];
}

void Chunk::setBlock(int localX, int localY, int localZ, BlockType type) {
    if (localX < 0 || localX >= CHUNK_WIDTH) return;
    if (localY < 0 || localY >= CHUNK_HEIGHT) return;
    if (localZ < 0 || localZ >= CHUNK_DEPTH) return;
    blocks[chunkIndex(localX, localY, localZ)] = type;
    needsMeshUpdate = true;
}

uint8_t Chunk::getBlockLight(int localX, int localY, int localZ) const {
    if (blockLight.empty()) return 0;
    if (localX < 0 || localX >= CHUNK_WIDTH) return 0;
    if (localY < 0 || localY >= CHUNK_HEIGHT) return 0;
    if (localZ < 0 || localZ >= CHUNK_DEPTH) return 0;
    return blockLight[chunkIndex(localX, localY, localZ)];
}

void Chunk::setBlockLight(int localX, int localY, int localZ, uint8_t level) {
    if (localX < 0 || localX >= CHUNK_WIDTH) return;
    if (localY < 0 || localY >= CHUNK_HEIGHT) return;
    if (localZ < 0 || localZ >= CHUNK_DEPTH) return;
    if (blockLight.empty()) {
        if (level == 0) return; // stay unallocated while fully dark
        blockLight.assign(CHUNK_VOLUME, 0);
    }
    blockLight[chunkIndex(localX, localY, localZ)] = level;
}

BlockType Chunk::getBlockWorld(int x, int y, int z) const {
    int localX = x - chunkX * CHUNK_WIDTH;
    int localZ = z - chunkZ * CHUNK_DEPTH;
    return getBlock(localX, y, localZ);
}

void Chunk::setBlockWorld(int x, int y, int z, BlockType type) {
    int localX = x - chunkX * CHUNK_WIDTH;
    int localZ = z - chunkZ * CHUNK_DEPTH;
    setBlock(localX, y, localZ, type);
}

int Chunk::worldToChunk(int worldCoord) {
    // Truncate toward zero for positive, floor for negative
    return (worldCoord >= 0) ? (worldCoord / CHUNK_WIDTH)
                             : ((worldCoord - CHUNK_WIDTH + 1) / CHUNK_WIDTH);
}

int Chunk::chunkToWorld(int chunkCoord, int localCoord) {
    return chunkCoord * CHUNK_WIDTH + localCoord;
}

Vec3 Chunk::getWorldPosition() const {
    return Vec3{static_cast<float>(chunkX * CHUNK_WIDTH), 0.f,
                static_cast<float>(chunkZ * CHUNK_DEPTH)};
}

void Chunk::markDirty() {
    needsMeshUpdate = true;
}

void Chunk::setMeshed(bool value) {
    meshed = value;
    needsMeshUpdate = false;
}

AABB Chunk::getAABB() const {
    Vec3 min{static_cast<float>(chunkX * CHUNK_WIDTH), 0.f,
             static_cast<float>(chunkZ * CHUNK_DEPTH)};
    Vec3 max{static_cast<float>((chunkX + 1) * CHUNK_WIDTH), static_cast<float>(CHUNK_HEIGHT),
             static_cast<float>((chunkZ + 1) * CHUNK_DEPTH)};
    return AABB{min, max};
}

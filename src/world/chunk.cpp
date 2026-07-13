#include "world/chunk.hpp"

#include <algorithm>
#include <stdexcept>

Chunk::Chunk(int cx, int cz)
    : chunkX(cx)
    , chunkZ(cz)
    , blocks(CHUNK_VOLUME, BlockType::AIR)
    , biomes{}
    , heightMap{}
{
    biomes.fill(Biome::PLAINS);
    heightMap.fill(0);
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
    return (worldCoord >= 0) ? (worldCoord / CHUNK_WIDTH) : ((worldCoord - CHUNK_WIDTH + 1) / CHUNK_WIDTH);
}

int Chunk::chunkToWorld(int chunkCoord, int localCoord) {
    return chunkCoord * CHUNK_WIDTH + localCoord;
}

Vec3 Chunk::getWorldPosition() const {
    return Vec3{static_cast<float>(chunkX * CHUNK_WIDTH),
                0.f,
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
    Vec3 min{static_cast<float>(chunkX * CHUNK_WIDTH),
             0.f,
             static_cast<float>(chunkZ * CHUNK_DEPTH)};
    Vec3 max{static_cast<float>((chunkX + 1) * CHUNK_WIDTH),
             static_cast<float>(CHUNK_HEIGHT),
             static_cast<float>((chunkZ + 1) * CHUNK_DEPTH)};
    return AABB{min, max};
}

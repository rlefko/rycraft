#include "world/world.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

World::World(uint32_t seed, int viewDistance)
    : seed_(seed)
    , viewDistance_(viewDistance)
    , terrainGen_(seed)
    , biomeGen_(seed)
    , caveGen_(seed)
    , oreGen_(seed)
    , treeGen_(seed)
    , structureGen_(seed)
{
}

std::string World::chunkKey(int cx, int cz) {
    std::ostringstream oss;
    oss << cx << "_" << cz;
    return oss.str();
}

void World::generateChunk(std::shared_ptr<Chunk> chunk) {
    int worldBaseX = chunk->chunkX * CHUNK_WIDTH;
    int worldBaseZ = chunk->chunkZ * CHUNK_DEPTH;

    // Step 1: Generate terrain heights
    TerrainConfig terrainConfig;
    BiomeConfig biomeConfig;

    std::vector<double> heights(CHUNK_WIDTH * CHUNK_DEPTH);
    std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH> biomes;

    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            int worldX = worldBaseX + x;
            int worldZ = worldBaseZ + z;
            int xzIndex = x + z * CHUNK_WIDTH;

            double height = terrainGen_.getHeight(
                static_cast<double>(worldX),
                static_cast<double>(worldZ),
                terrainConfig
            );
            heights[xzIndex] = height;

            Biome biome = biomeGen_.getBiome(
                static_cast<double>(worldX),
                static_cast<double>(worldZ),
                height,
                biomeConfig
            );
            biomes[xzIndex] = biome;
        }
    }

    // Step 2: Generate surface blocks
    SurfaceGenerator::generateSurface(*chunk, heights, biomes);

    // Step 3: Carve caves
    CaveConfig caveConfig;
    caveGen_.carve(*chunk, caveConfig);

    // Step 4: Generate ore deposits
    OreConfig oreConfig;
    oreGen_.generate(*chunk, oreConfig);

    // Step 5: Generate trees
    treeGen_.generate(*chunk, biomes);

    // Step 6: Generate structures
    structureGen_.generate(*chunk, biomes);

    // Copy biomes to chunk
    chunk->biomes = biomes;

    // Mark chunk as generated
    chunk->generated = true;
    chunk->needsMeshUpdate = true;
}

std::shared_ptr<Chunk> World::getChunk(int chunkX, int chunkZ) {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::string key = chunkKey(chunkX, chunkZ);

    auto it = chunks_.find(key);
    if (it != chunks_.end()) {
        return it->second;
    }

    // Generate new chunk
    auto chunk = std::make_shared<Chunk>(chunkX, chunkZ);
    generateChunk(chunk);
    chunks_[key] = chunk;

    return chunk;
}

BlockType World::getBlock(int x, int y, int z) {
    int cx = Chunk::worldToChunk(x);
    int cz = Chunk::worldToChunk(z);

    std::shared_ptr<Chunk> chunk = getChunk(cx, cz);
    return chunk->getBlockWorld(x, y, z);
}

void World::setBlock(int x, int y, int z, BlockType type) {
    int cx = Chunk::worldToChunk(x);
    int cz = Chunk::worldToChunk(z);

    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::string key = chunkKey(cx, cz);
    auto it = chunks_.find(key);
    if (it != chunks_.end()) {
        it->second->setBlockWorld(x, y, z, type);
    }
}

double World::getTerrainHeight(int x, int z) const {
    return terrainGen_.getHeight(static_cast<double>(x), static_cast<double>(z));
}

Biome World::getBiome(int x, int z) const {
    double height = getTerrainHeight(x, z);
    return biomeGen_.getBiome(static_cast<double>(x), static_cast<double>(z), height);
}

std::vector<std::shared_ptr<Chunk>> World::getLoadedChunks() {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::vector<std::shared_ptr<Chunk>> result;
    result.reserve(chunks_.size());
    for (const auto& pair : chunks_) {
        result.push_back(pair.second);
    }
    return result;
}

std::vector<std::shared_ptr<Chunk>> World::getDirtyChunks() {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::vector<std::shared_ptr<Chunk>> result;
    for (const auto& pair : chunks_) {
        if (pair.second->needsMeshUpdate) {
            result.push_back(pair.second);
        }
    }
    return result;
}

void World::markChunkMeshed(int chunkX, int chunkZ) {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::string key = chunkKey(chunkX, chunkZ);
    auto it = chunks_.find(key);
    if (it != chunks_.end()) {
        it->second->needsMeshUpdate = false;
    }
}

uint32_t World::getSeed() const {
    return seed_;
}

int World::getViewDistance() const {
    return viewDistance_;
}

void World::setViewDistance(int distance) {
    viewDistance_ = std::max(1, distance);
}

void World::updatePlayerPosition(int playerX, int playerZ) {
    int newPlayerChunkX = Chunk::worldToChunk(playerX);
    int newPlayerChunkZ = Chunk::worldToChunk(playerZ);

    if (newPlayerChunkX == playerChunkX_ && newPlayerChunkZ == playerChunkZ_) {
        return;
    }

    playerChunkX_ = newPlayerChunkX;
    playerChunkZ_ = newPlayerChunkZ;

    // Load chunks within view distance
    for (int dz = -viewDistance_; dz <= viewDistance_; ++dz) {
        for (int dx = -viewDistance_; dx <= viewDistance_; ++dx) {
            getChunk(playerChunkX_ + dx, playerChunkZ_ + dz);
        }
    }

    // Unload chunks outside view distance
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.begin();
    while (it != chunks_.end()) {
        int cx = it->second->chunkX;
        int cz = it->second->chunkZ;

        int distX = std::abs(cx - playerChunkX_);
        int distZ = std::abs(cz - playerChunkZ_);

        if (distX > viewDistance_ || distZ > viewDistance_) {
            it = chunks_.erase(it);
        } else {
            ++it;
        }
    }
}

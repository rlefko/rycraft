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

World::~World() = default;

std::string World::chunkKey(int cx, int cz) {
    std::ostringstream oss;
    oss << cx << "_" << cz;
    return oss.str();
}

void World::generateChunk(std::shared_ptr<Chunk> chunk) {
    int worldBaseX = chunk->chunkX * CHUNK_WIDTH;
    int worldBaseZ = chunk->chunkZ * CHUNK_DEPTH;

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

    SurfaceGenerator::generateSurface(*chunk, heights, biomes);

    CaveConfig caveConfig;
    caveGen_.carve(*chunk, caveConfig);

    OreConfig oreConfig;
    oreGen_.generate(*chunk, oreConfig);

    treeGen_.generate(*chunk, biomes);
    structureGen_.generate(*chunk, biomes);

    chunk->biomes = biomes;
    chunk->generated = true;
    chunk->needsMeshUpdate = true;
}

void World::generateChunkAsync(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ) {
    std::string key = chunkKey(chunkX, chunkZ);

    // Guard: avoid generating chunk twice
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        if (pendingGenerations_.find(key) != pendingGenerations_.end()) {
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        if (chunks_.find(key) != chunks_.end()) {
            return;
        }
    }

    int distX = std::abs(chunkX - playerChunkX);
    int distZ = std::abs(chunkZ - playerChunkZ);
    int distance = std::max(distX, distZ);
    bool isCoarse = distance > 32;

    auto pool = genPool_;
    auto future = pool->submit([this, pool, chunkX, chunkZ, key, isCoarse]() {
        try {
            auto chunk = std::make_shared<Chunk>(chunkX, chunkZ);
            generateChunk(chunk);
            (void)isCoarse;
            std::lock_guard<std::mutex> lock(chunksMutex_);
            chunks_[key] = chunk;
        } catch (...) {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            auto chunk = std::make_shared<Chunk>(chunkX, chunkZ);
            chunk->generated = true;
            chunk->needsMeshUpdate = true;
            chunks_[key] = chunk;
        }
    });

    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingGenerations_[key] = std::move(future);
}

std::shared_ptr<Chunk> World::getChunk(int chunkX, int chunkZ) {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    std::string key = chunkKey(chunkX, chunkZ);

    auto it = chunks_.find(key);
    if (it != chunks_.end()) {
        return it->second;
    }

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

    for (int dz = -viewDistance_; dz <= viewDistance_; ++dz) {
        for (int dx = -viewDistance_; dx <= viewDistance_; ++dx) {
            getChunk(playerChunkX_ + dx, playerChunkZ_ + dz);
        }
    }

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

void World::generateAroundPlayer(int playerX, int playerZ) {
    // Lazy initialization of ThreadPool
    if (!genPool_) {
        genPool_ = std::make_shared<ThreadPool>(4);
    }

    int playerChunkX = Chunk::worldToChunk(playerX);
    int playerChunkZ = Chunk::worldToChunk(playerZ);

    // Clean up completed futures
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        auto it = pendingGenerations_.begin();
        while (it != pendingGenerations_.end()) {
            if (!it->second.valid()) {
                it = pendingGenerations_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Collect chunks needing generation
    std::vector<std::pair<int, int>> chunksToGenerate;
    {
        std::lock_guard<std::mutex> lock1(pendingMutex_);
        std::lock_guard<std::mutex> lock2(chunksMutex_);
        for (int dz = -viewDistance_; dz <= viewDistance_; ++dz) {
            for (int dx = -viewDistance_; dx <= viewDistance_; ++dx) {
                int chunkX = playerChunkX + dx;
                int chunkZ = playerChunkZ + dz;
                std::string key = chunkKey(chunkX, chunkZ);
                if (chunks_.find(key) != chunks_.end()) {
                    continue;
                }
                if (pendingGenerations_.find(key) != pendingGenerations_.end()) {
                    continue;
                }
                chunksToGenerate.emplace_back(chunkX, chunkZ);
            }
        }
    }

    // Submit async tasks
    for (const auto& [chunkX, chunkZ] : chunksToGenerate) {
        generateChunkAsync(chunkX, chunkZ, playerChunkX, playerChunkZ);
    }
}

size_t World::getPendingChunkCount() const {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    size_t pending = 0;
    for (const auto& [key, future] : pendingGenerations_) {
        if (future.valid()) {
            ++pending;
        }
    }
    return pending;
}

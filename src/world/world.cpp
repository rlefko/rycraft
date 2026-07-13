#include "world/world.hpp"

#include "common/error.hpp"
#include "world/save_manager.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>

World::World(uint32_t seed, int viewDistance)
    : seed_(seed)
    , viewDistance_(viewDistance)
    , generator_(seed) {}

World::~World() {
    // Wait for in-flight generation tasks: they capture `this` and insert
    // into chunks_, so destroying the World underneath them is a use-after-free.
    std::lock_guard<std::mutex> lock(pendingMutex_);
    for (auto& [key, future] : pendingGenerations_) {
        if (future.valid()) {
            future.wait();
        }
    }
}

void World::generateChunk(std::shared_ptr<Chunk> chunk) {
    generator_.generate(*chunk);
}

void World::generateChunkAsync(int chunkX, int chunkZ) {
    ChunkPos key{chunkX, chunkZ};

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

    auto pool = genPool_;
    auto future = pool->submit([this, pool, chunkX, chunkZ, key]() {
        auto chunk = loadOrGenerateChunk(chunkX, chunkZ);
        std::lock_guard<std::mutex> lock(chunksMutex_);
        // try_emplace: if another path (e.g. a synchronous getChunk) won the
        // race, keep its chunk — it may already hold player edits.
        chunks_.try_emplace(key, std::move(chunk));
    });

    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingGenerations_[key] = std::move(future);
}

// ---------------------------------------------------------------------------
// loadOrGenerateChunk — disk first (persisted edits), generator second, and a
// blank fallback chunk when generation throws (per the error-handling policy).
// Called WITHOUT chunksMutex_ held: generation is far too slow for a lock the
// render thread also takes.
// ---------------------------------------------------------------------------
std::shared_ptr<Chunk> World::loadOrGenerateChunk(int chunkX, int chunkZ) {
    if (saveManager_) {
        if (auto loaded = saveManager_->loadChunk(chunkX, chunkZ)) {
            return std::make_shared<Chunk>(std::move(*loaded));
        }
    }

    try {
        auto chunk = std::make_shared<Chunk>(chunkX, chunkZ);
        auto start = std::chrono::steady_clock::now();
        generateChunk(chunk);
        recordGenMs(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
        return chunk;
    } catch (const std::exception& e) {
        RY_LOG_ERROR(
            (std::string("Chunk generation failed, using blank fallback: ") + e.what()).c_str());
        auto chunk = std::make_shared<Chunk>(chunkX, chunkZ);
        chunk->generated = true;
        chunk->needsMeshUpdate = true;
        return chunk;
    }
}

void World::setSaveManager(SaveManager* saveManager) {
    saveManager_ = saveManager;
}

std::shared_ptr<Chunk> World::getChunk(int chunkX, int chunkZ) {
    ChunkPos key{chunkX, chunkZ};

    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto it = chunks_.find(key);
        if (it != chunks_.end()) {
            return it->second;
        }
    }

    // Load or generate OUTSIDE the lock — generation takes milliseconds and
    // this mutex is also on the render thread's path. A racing thread may
    // duplicate the work; try_emplace keeps exactly one winner.
    auto chunk = loadOrGenerateChunk(chunkX, chunkZ);

    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto [it, inserted] = chunks_.try_emplace(key, std::move(chunk));
    return it->second;
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

    auto it = chunks_.find(ChunkPos{cx, cz});
    if (it != chunks_.end()) {
        it->second->setBlockWorld(x, y, z, type);
        it->second->modifiedSinceSave = true;
    }
}

double World::getTerrainHeight(int x, int z) const {
    return generator_.baseHeightAt(x, z);
}

Biome World::getBiome(int x, int z) const {
    return generator_.biomeAt(x, z);
}

std::vector<std::shared_ptr<Chunk>> World::getLoadedChunks() const {
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

    auto it = chunks_.find(ChunkPos{chunkX, chunkZ});
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
    // Re-stream immediately so a settings change is visible without waiting
    // for the player to cross a chunk boundary
    if (hasPlayerChunk_) {
        generateAroundPlayer(playerChunkX_ * CHUNK_WIDTH, playerChunkZ_ * CHUNK_DEPTH);
        unloadDistantChunks();
    }
}

void World::updatePlayerPosition(int playerX, int playerZ) {
    int newPlayerChunkX = Chunk::worldToChunk(playerX);
    int newPlayerChunkZ = Chunk::worldToChunk(playerZ);

    if (hasPlayerChunk_ && newPlayerChunkX == playerChunkX_ && newPlayerChunkZ == playerChunkZ_) {
        return;
    }

    hasPlayerChunk_ = true;
    playerChunkX_ = newPlayerChunkX;
    playerChunkZ_ = newPlayerChunkZ;

    // Stream the surrounding chunks in on the generation pool; the workers
    // insert finished chunks into chunks_ themselves.
    generateAroundPlayer(playerX, playerZ);

    unloadDistantChunks();
}

void World::unloadDistantChunks() {
    // Collect under the lock, save after releasing it (never do I/O under
    // chunksMutex_ — the render thread takes it every frame)
    std::vector<std::shared_ptr<Chunk>> toSave;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto it = chunks_.begin();
        while (it != chunks_.end()) {
            int cx = it->second->chunkX;
            int cz = it->second->chunkZ;

            int distX = std::abs(cx - playerChunkX_);
            int distZ = std::abs(cz - playerChunkZ_);

            if (distX > viewDistance_ || distZ > viewDistance_) {
                if (it->second->modifiedSinceSave && saveManager_) {
                    it->second->modifiedSinceSave = false;
                    toSave.push_back(it->second);
                }
                it = chunks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto& chunk : toSave) {
        // The chunk left the map: no game thread mutates it anymore, so the
        // save thread may serialize it lock-free
        saveManager_->saveChunkAsync(std::move(chunk));
    }
}

void World::saveModifiedChunks() {
    if (!saveManager_) {
        return;
    }
    std::vector<std::shared_ptr<Chunk>> toSave;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        for (auto& [key, chunk] : chunks_) {
            if (chunk->modifiedSinceSave) {
                chunk->modifiedSinceSave = false;
                toSave.push_back(chunk);
            }
        }
    }
    for (auto& chunk : toSave) {
        saveManager_->saveChunkAsync(std::move(chunk));
    }
}

void World::generateAroundPlayer(int playerX, int playerZ) {
    // Lazy initialization of ThreadPool
    if (!genPool_) {
        genPool_ = std::make_shared<ThreadPool>(4);
    }

    int playerChunkX = Chunk::worldToChunk(playerX);
    int playerChunkZ = Chunk::worldToChunk(playerZ);

    // Clean up completed futures. valid() stays true until get() is called,
    // so poll readiness instead — otherwise this map only ever grows.
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        auto it = pendingGenerations_.begin();
        while (it != pendingGenerations_.end()) {
            if (!it->second.valid() ||
                it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
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
                ChunkPos key{chunkX, chunkZ};
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
        generateChunkAsync(chunkX, chunkZ);
    }
}

void World::recordGenMs(float ms) {
    // CAS loop: several gen workers may finish simultaneously
    uint32_t oldBits = genMsEmaBits_.load(std::memory_order_relaxed);
    for (;;) {
        float oldEma = std::bit_cast<float>(oldBits);
        float newEma = oldEma == 0.f ? ms : oldEma * 0.9f + ms * 0.1f;
        if (genMsEmaBits_.compare_exchange_weak(oldBits, std::bit_cast<uint32_t>(newEma),
                                                std::memory_order_relaxed)) {
            return;
        }
    }
}

float World::averageGenMs() const {
    return std::bit_cast<float>(genMsEmaBits_.load(std::memory_order_relaxed));
}

size_t World::getPendingChunkCount() const {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    size_t pending = 0;
    for (const auto& [key, future] : pendingGenerations_) {
        // A future stays valid() until get(); only count genuinely unfinished work
        if (future.valid() &&
            future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++pending;
        }
    }
    return pending;
}

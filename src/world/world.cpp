#include "world/world.hpp"

#include "common/error.hpp"
#include "world/save_manager.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>

World::World(uint32_t seed, int viewDistance)
    : seed_(seed)
    , viewDistance_(viewDistance)
    , generator_(seed) {}

void sortChunksByDistance(std::vector<ChunkPos>& chunks, int centerChunkX, int centerChunkZ) {
    auto distanceSq = [&](const ChunkPos& p) {
        int64_t dx = p.x - centerChunkX;
        int64_t dz = p.z - centerChunkZ;
        return dx * dx + dz * dz;
    };
    std::sort(chunks.begin(), chunks.end(),
              [&](const ChunkPos& a, const ChunkPos& b) { return distanceSq(a) > distanceSq(b); });
}

World::~World() {
    // Wait for in-flight generation tasks: they capture `this` and insert
    // into chunks_, so destroying the World underneath them is a
    // use-after-free. Waiting happens OUTSIDE pendingMutex_ (a finishing
    // worker pumps the backlog under that mutex — holding it while waiting
    // on the worker's future would deadlock), and loops because a worker
    // may slip one more submission in before it observes shuttingDown_.
    shuttingDown_.store(true);
    for (;;) {
        std::unordered_map<ChunkPos, std::future<void>> pending;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            genBacklog_.clear();
            if (pendingGenerations_.empty()) {
                break;
            }
            pending = std::move(pendingGenerations_);
            pendingGenerations_.clear();
        }
        for (auto& [key, future] : pending) {
            if (future.valid()) {
                future.wait();
            }
        }
    }
}

void World::generateChunk(std::shared_ptr<Chunk> chunk) {
    generator_.generate(*chunk);
}

void World::generateChunkAsync(int chunkX, int chunkZ) {
    ChunkPos key{chunkX, chunkZ};

    if (shuttingDown_.load()) {
        return;
    }

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
        {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            // try_emplace: if another path (e.g. a synchronous getChunk) won
            // the race, keep its chunk — it may already hold player edits.
            chunks_.try_emplace(key, std::move(chunk));
        }
        // A freed window slot pulls the next-nearest backlog chunk in, so
        // streaming continues without waiting for the next tick
        pumpGeneration();
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
        genMs_.record(
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

BlockType World::getBlockIfLoaded(int x, int y, int z) const {
    int cx = Chunk::worldToChunk(x);
    int cz = Chunk::worldToChunk(z);

    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(ChunkPos{cx, cz});
    if (it == chunks_.end()) {
        return BlockType::AIR;
    }
    return it->second->getBlockWorld(x, y, z);
}

void World::setBlock(int x, int y, int z, BlockType type) {
    int cx = Chunk::worldToChunk(x);
    int cz = Chunk::worldToChunk(z);

    std::lock_guard<std::mutex> lock(chunksMutex_);

    auto it = chunks_.find(ChunkPos{cx, cz});
    if (it == chunks_.end()) {
        return;
    }
    it->second->setBlockWorld(x, y, z, type);
    it->second->modifiedSinceSave = true;
    it->second->needsMeshUpdate = true;
    it->second->version.fetch_add(1, std::memory_order_relaxed);

    // Meshes read one block into each face neighbor: an edit on a boundary
    // column changes the neighbor's border faces too
    auto markNeighbor = [&](int ncx, int ncz) {
        auto neighbor = chunks_.find(ChunkPos{ncx, ncz});
        if (neighbor != chunks_.end()) {
            neighbor->second->needsMeshUpdate = true;
            neighbor->second->version.fetch_add(1, std::memory_order_relaxed);
        }
    };
    int lx = x - cx * CHUNK_WIDTH;
    int lz = z - cz * CHUNK_DEPTH;
    if (lx == 0) markNeighbor(cx - 1, cz);
    if (lx == CHUNK_WIDTH - 1) markNeighbor(cx + 1, cz);
    if (lz == 0) markNeighbor(cx, cz - 1);
    if (lz == CHUNK_DEPTH - 1) markNeighbor(cx, cz + 1);
}

// ---------------------------------------------------------------------------
// snapshotForMeshing — one bounded copy under chunksMutex_ (see
// mesh_snapshot.hpp for why this is safe and why the ring exists). This is
// the deliberate exception to "no work under chunksMutex_": ~83 KB of
// memcpy costs microseconds, unlike generation/IO.
// ---------------------------------------------------------------------------
bool World::snapshotForMeshing(ChunkPos pos, MeshSnapshot& out) const {
    std::lock_guard<std::mutex> lock(chunksMutex_);

    auto self = chunks_.find(pos);
    if (self == chunks_.end() || !self->second->generated) {
        return false;
    }
    const Chunk* neighbors[4] = {nullptr, nullptr, nullptr, nullptr};
    const ChunkPos neighborPos[4] = {
        {pos.x - 1, pos.z}, {pos.x + 1, pos.z}, {pos.x, pos.z - 1}, {pos.x, pos.z + 1}};
    for (int i = 0; i < 4; ++i) {
        auto it = chunks_.find(neighborPos[i]);
        if (it == chunks_.end() || !it->second->generated) {
            return false;
        }
        neighbors[i] = it->second.get();
    }

    out.chunkX = pos.x;
    out.chunkZ = pos.z;
    out.version = self->second->version.load(std::memory_order_relaxed);
    out.resize();
    const Chunk& chunk = *self->second;
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            // Interior row: 16 contiguous blocks in both layouts
            std::memcpy(&out.blocks[MeshSnapshot::index(0, y, z)],
                        &chunk.blocks[z * CHUNK_WIDTH + y * CHUNK_WIDTH * CHUNK_DEPTH],
                        CHUNK_WIDTH * sizeof(BlockType));
            // ±X neighbor walls
            out.blocks[MeshSnapshot::index(-1, y, z)] =
                neighbors[0]->getBlock(CHUNK_WIDTH - 1, y, z);
            out.blocks[MeshSnapshot::index(CHUNK_WIDTH, y, z)] = neighbors[1]->getBlock(0, y, z);
        }
        // ±Z neighbor walls
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            out.blocks[MeshSnapshot::index(x, y, -1)] =
                neighbors[2]->getBlock(x, y, CHUNK_DEPTH - 1);
            out.blocks[MeshSnapshot::index(x, y, CHUNK_DEPTH)] = neighbors[3]->getBlock(x, y, 0);
        }
    }
    return true;
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
        // Same chunk: keep the submission window full anyway
        pumpGeneration();
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

            // Generation reaches vd+1; unloading at vd+2 adds hysteresis so
            // strafing across a boundary doesn't churn the frontier ring
            if (distX > viewDistance_ + 2 || distZ > viewDistance_ + 2) {
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

    // Rebuild the backlog from scratch: chunks that left the radius are
    // implicitly cancelled, and everything still missing re-sorts against
    // the new center. Generation reaches one chunk beyond the render
    // radius so visible chunks always have generated neighbors (the
    // neighbor-aware mesher needs them).
    {
        std::lock_guard<std::mutex> lock1(pendingMutex_);
        std::lock_guard<std::mutex> lock2(chunksMutex_);
        genBacklog_.clear();
        int genRadius = viewDistance_ + 1;
        for (int dz = -genRadius; dz <= genRadius; ++dz) {
            for (int dx = -genRadius; dx <= genRadius; ++dx) {
                ChunkPos key{playerChunkX + dx, playerChunkZ + dz};
                if (chunks_.find(key) != chunks_.end()) {
                    continue;
                }
                if (pendingGenerations_.find(key) != pendingGenerations_.end()) {
                    continue;
                }
                genBacklog_.push_back(key);
            }
        }
        sortChunksByDistance(genBacklog_, playerChunkX, playerChunkZ);
    }

    pumpGeneration();
}

void World::pumpGeneration() {
    if (!genPool_ || shuttingDown_.load()) {
        return;
    }

    // Clean up completed futures. valid() stays true until get() is called,
    // so poll readiness instead — otherwise this map only ever grows.
    std::vector<ChunkPos> toSubmit;
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

        while (pendingGenerations_.size() + toSubmit.size() < MAX_INFLIGHT_GEN &&
               !genBacklog_.empty()) {
            toSubmit.push_back(genBacklog_.back());
            genBacklog_.pop_back();
        }
    }

    // Submit outside the lock (generateChunkAsync re-takes it)
    for (const ChunkPos& pos : toSubmit) {
        generateChunkAsync(pos.x, pos.z);
    }
}

float World::averageGenMs() const {
    return genMs_.value();
}

size_t World::getPendingChunkCount() const {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    // Backlogged + genuinely in-flight: consumers (the HUD, the animal
    // spawn gate) care about "is streaming still working", not window size
    size_t pending = genBacklog_.size();
    for (const auto& [key, future] : pendingGenerations_) {
        // A future stays valid() until get(); only count genuinely unfinished work
        if (future.valid() &&
            future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++pending;
        }
    }
    return pending;
}

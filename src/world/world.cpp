#include "world/world.hpp"

#include "common/error.hpp"
#include "common/thread_priority.hpp"
#include "world/light_engine.hpp"
#include "world/save_manager.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <thread>
#include <unordered_set>

namespace {

int64_t distanceSq(ChunkPos pos, int64_t centerX, int32_t centerY, int64_t centerZ) {
    const int64_t dx = pos.x - centerX;
    const int64_t dy = static_cast<int64_t>(pos.y) - centerY;
    const int64_t dz = pos.z - centerZ;
    return dx * dx + dz * dz + dy * dy * 2;
}

bool validChunkY(int32_t y) {
    return y >= WORLD_MIN_CHUNK_Y && y <= WORLD_MAX_CHUNK_Y;
}

uint64_t sectionBit(int32_t chunkY) {
    return uint64_t{1} << static_cast<uint32_t>(chunkY - WORLD_MIN_CHUNK_Y);
}

uint64_t sectionRangeMask(int32_t firstChunkY, int32_t lastChunkY) {
    const uint32_t first = static_cast<uint32_t>(firstChunkY - WORLD_MIN_CHUNK_Y);
    const uint32_t count = static_cast<uint32_t>(lastChunkY - firstChunkY + 1);
    return ((uint64_t{1} << count) - 1U) << first;
}

bool hasBlockLightEmitter(const Chunk& chunk) {
    if (chunk.isUniform()) return blockLightEmission(chunk.uniformBlock()) != 0;
    return std::any_of(chunk.denseBlocks().begin(), chunk.denseBlocks().end(),
                       [](BlockType block) { return blockLightEmission(block) != 0; });
}

FluidBounds cubeBounds(ChunkPos pos) {
    return {
        .minX = pos.x * CHUNK_EDGE,
        .minY = pos.y * CHUNK_EDGE,
        .minZ = pos.z * CHUNK_EDGE,
        .maxX = pos.x * CHUNK_EDGE + CHUNK_EDGE - 1,
        .maxY = pos.y * CHUNK_EDGE + CHUNK_EDGE - 1,
        .maxZ = pos.z * CHUNK_EDGE + CHUNK_EDGE - 1,
    };
}

std::optional<int>
loadedSurfaceHeight(const std::unordered_map<ChunkPos, std::shared_ptr<Chunk>>& chunks,
                    int64_t worldX, int64_t worldZ) {
    const int64_t chunkX = Chunk::worldToChunk(worldX);
    const int64_t chunkZ = Chunk::worldToChunk(worldZ);
    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    bool hasLoadedCube = false;
    for (int32_t chunkY = WORLD_MAX_CHUNK_Y; chunkY >= WORLD_MIN_CHUNK_Y; --chunkY) {
        const auto iterator = chunks.find({chunkX, chunkY, chunkZ});
        if (iterator == chunks.end() || !iterator->second->generated) continue;
        hasLoadedCube = true;
        for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
            if (isOpaque(iterator->second->getBlock(localX, localY, localZ))) {
                return chunkY * CHUNK_EDGE + localY;
            }
        }
    }
    return hasLoadedCube ? std::optional<int>(WORLD_MIN_Y - 1) : std::nullopt;
}

} // namespace

World::World(uint32_t seed, int viewDistance)
    : seed_(seed)
    , viewDistance_(
          std::clamp(viewDistance, MIN_RENDER_DISTANCE_CHUNKS, MAX_RENDER_DISTANCE_CHUNKS))
    , generator_(seed) {
    std::atomic_store_explicit(&loadedSnapshot_,
                               std::make_shared<const std::vector<std::shared_ptr<Chunk>>>(),
                               std::memory_order_release);
    std::atomic_store_explicit(&meshCandidateSnapshot_,
                               std::make_shared<const std::unordered_set<ChunkPos>>(),
                               std::memory_order_release);
}

void sortChunksByDistance(std::vector<ChunkPos>& chunks, int64_t centerChunkX, int32_t centerChunkY,
                          int64_t centerChunkZ) {
    std::sort(chunks.begin(), chunks.end(), [&](ChunkPos a, ChunkPos b) {
        return distanceSq(a, centerChunkX, centerChunkY, centerChunkZ) >
               distanceSq(b, centerChunkX, centerChunkY, centerChunkZ);
    });
}

World::~World() {
    shuttingDown_.store(true);
    latestActiveSetRequestId_.fetch_add(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(activeSetRequestMutex_);
        stopActiveSetThread_ = true;
        pendingActiveSetRequest_.reset();
        activeSetWorkPending_.store(false, std::memory_order_release);
    }
    activeSetRequestCv_.notify_all();
    if (activeSetThread_.joinable()) activeSetThread_.join();
    for (;;) {
        std::unordered_map<ChunkPos, std::future<void>> pending;
        std::unordered_map<ColumnPos, std::future<void>> pendingPlans;
        size_t activePlans = 0;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            genBacklog_.clear();
            genBacklogSet_.clear();
            columnPlanBacklog_.clear();
            planDependents_.clear();
            missingPlanDependencies_.clear();
            activePlans = activeColumnPlanJobs_;
            if (pendingGenerations_.empty() && pendingColumnPlans_.empty() && activePlans == 0) {
                break;
            }
            pending = std::move(pendingGenerations_);
            pendingGenerations_.clear();
            pendingPlans = std::move(pendingColumnPlans_);
            pendingColumnPlans_.clear();
        }
        for (auto& [pos, future] : pending) {
            if (future.valid()) future.wait();
        }
        for (auto& [pos, future] : pendingPlans) {
            if (future.valid()) future.wait();
        }
        if (pending.empty() && pendingPlans.empty() && activePlans != 0) std::this_thread::yield();
    }
}

void World::setSaveManager(SaveManager* saveManager) {
    saveManager_ = saveManager;
    if (!saveManager_) return;
    std::unordered_set<ChunkPos> resumeCubes;
    for (const FluidBoundaryFrontier& frontier : saveManager_->loadDeferredFluidFrontiers()) {
        fluidScheduler_.restoreDeferredFrontier(frontier);
        resumeCubes.insert({Chunk::worldToChunk(frontier.unavailable.x),
                            Chunk::worldToChunkY(frontier.unavailable.y),
                            Chunk::worldToChunk(frontier.unavailable.z)});
    }
    for (ChunkPos pos : resumeCubes) {
        if (isChunkLoaded(pos)) queueFluidResume(pos);
    }
}

void World::generateChunk(const std::shared_ptr<Chunk>& chunk) {
    generator_.generateCube(*chunk);
}

void World::generateChunkAsync(ChunkPos pos) {
    if (shuttingDown_.load()) return;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        if (pendingGenerations_.contains(pos)) return;
    }
    bool alreadyLoaded = false;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        alreadyLoaded = chunks_.contains(pos);
    }
    if (alreadyLoaded) {
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            generationsInFlight_.erase(pos);
        }
        pumpGeneration();
        return;
    }

    auto pool = genPool_;
    auto future = pool->submit([this, pool, pos]() {
        bool loadedFromSave = false;
        auto chunk = loadOrGenerateChunk(pos, &loadedFromSave);
        bool inserted = false;
        {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            if (retainedChunks_.contains(pos) && !shuttingDown_.load()) {
                const auto [it, didInsert] = chunks_.try_emplace(pos, std::move(chunk));
                inserted = didInsert;
                if (inserted) {
                    loadedSectionMasks_[{pos.x, pos.z}] |= sectionBit(pos.y);
                    loadedCubeCount_.fetch_add(1, std::memory_order_relaxed);
                    loadedSnapshotDirty_.store(true, std::memory_order_release);
                    if (loadedFromSave) {
                        refreshSavedSkyCutoffsLocked(pos);
                    } else {
                        if (extendGeneratedSkyCutoffsLocked(*it->second)) {
                            markSkyColumnMeshesDirtyLocked({pos.x, pos.z});
                        }
                    }
                    markHaloNeighborMeshesDirtyLocked(pos);
                    markSkyContinuityBelowLocked({pos.x, pos.z}, pos.y);
                }
            }
        }
        if (inserted) {
            queueFluidResume(pos);
            queueLightReconcileWithNeighbors(pos);
        }
        pumpGeneration();
    });

    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingGenerations_[pos] = std::move(future);
}

void World::generateColumnPlanAsync(ColumnPos pos) {
    auto pool = genPool_;
    auto future = pool->submit([this, pool, pos]() {
        try {
            generator_.getColumnPlan(pos);
        } catch (const std::exception& error) {
            RY_LOG_ERROR((std::string("Column plan generation failed: ") + error.what()).c_str());
        }
        bool notifyActiveSet = false;
        size_t fullRetainedScanEquivalent = 0;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            columnPlansInFlight_.erase(pos);
            if (activeColumnPlanJobs_ > 0) --activeColumnPlanJobs_;
            ++completedPlansSinceRebuild_;
            const bool drained = activeColumnPlanJobs_ == 0 && columnPlanBacklog_.empty();
            if (completedPlansSinceRebuild_ >= COLUMN_PLAN_REBUILD_BATCH || drained) {
                completedPlansSinceRebuild_ = 0;
                notifyActiveSet = true;
            }
            fullRetainedScanEquivalent = retainedCubeCountForStats_;
        }
        completedColumnPlans_.fetch_add(1, std::memory_order_relaxed);
        fullRetainedScanEquivalent_.fetch_add(fullRetainedScanEquivalent,
                                              std::memory_order_relaxed);
        wakePlanDependents(pos);
        if (notifyActiveSet) {
            columnPlansChanged_.store(true, std::memory_order_release);
            activeSetRebuildNotifications_.fetch_add(1, std::memory_order_relaxed);
        }
        pumpGeneration();
    });

    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingColumnPlans_[pos] = std::move(future);
}

std::shared_ptr<Chunk> World::loadOrGenerateChunk(ChunkPos pos, bool* loadedFromSave) {
    if (loadedFromSave) *loadedFromSave = false;
    if (saveManager_) {
        if (auto loaded = saveManager_->loadChunk(pos)) {
            if (loadedFromSave) *loadedFromSave = true;
            auto chunk = std::make_shared<Chunk>(std::move(*loaded));
            LightEngine::computeSelfLight(*chunk);
            return chunk;
        }
    }

    try {
        auto chunk = std::make_shared<Chunk>(pos);
        const auto start = std::chrono::steady_clock::now();
        generateChunk(chunk);
        genMs_.record(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
        LightEngine::computeSelfLight(*chunk);
        return chunk;
    } catch (const std::exception& error) {
        RY_LOG_ERROR(
            (std::string("Cube generation failed, using empty fallback: ") + error.what()).c_str());
        auto chunk = std::make_shared<Chunk>(pos);
        chunk->generated = true;
        chunk->needsMeshUpdate = true;
        return chunk;
    }
}

std::shared_ptr<Chunk> World::getChunk(ChunkPos pos) {
    if (!validChunkY(pos.y)) return nullptr;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto it = chunks_.find(pos);
        if (it != chunks_.end()) return it->second;
    }

    bool loadedFromSave = false;
    auto chunk = loadOrGenerateChunk(pos, &loadedFromSave);
    std::shared_ptr<Chunk> result;
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        const auto [it, didInsert] = chunks_.try_emplace(pos, std::move(chunk));
        inserted = didInsert;
        if (inserted) {
            loadedSectionMasks_[{pos.x, pos.z}] |= sectionBit(pos.y);
            loadedCubeCount_.fetch_add(1, std::memory_order_relaxed);
            loadedSnapshotDirty_.store(true, std::memory_order_release);
            if (loadedFromSave) {
                refreshSavedSkyCutoffsLocked(pos);
            } else {
                if (extendGeneratedSkyCutoffsLocked(*it->second)) {
                    markSkyColumnMeshesDirtyLocked({pos.x, pos.z});
                }
            }
            markHaloNeighborMeshesDirtyLocked(pos);
            markSkyContinuityBelowLocked({pos.x, pos.z}, pos.y);
        }
        result = it->second;
    }
    if (inserted) {
        queueFluidResume(pos);
        queueLightReconcileWithNeighbors(pos);
    }
    return result;
}

BlockType World::getBlock(int64_t x, int32_t y, int64_t z) {
    if (y < WORLD_MIN_Y) return BlockType::BEDROCK;
    if (y > WORLD_MAX_Y) return BlockType::AIR;
    const ChunkPos pos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)};
    return getChunk(pos)->getBlockWorld(x, y, z);
}

BlockType World::getBlockIfLoaded(int64_t x, int32_t y, int64_t z) const {
    return findBlockIfLoaded(x, y, z).value_or(BlockType::AIR);
}

std::optional<BlockType> World::findBlockIfLoaded(int64_t x, int32_t y, int64_t z) const {
    if (y < WORLD_MIN_Y) return BlockType::BEDROCK;
    if (y > WORLD_MAX_Y) return BlockType::AIR;
    const ChunkPos pos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)};
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    if (it == chunks_.end() || !it->second->generated) return std::nullopt;
    return it->second->getBlockWorld(x, y, z);
}

std::optional<int> World::surfaceHeightIfLoaded(int64_t x, int64_t z) const {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    return loadedSurfaceHeight(chunks_, x, z);
}

BlockType World::getCollisionBlockIfLoaded(int64_t x, int32_t y, int64_t z) const {
    return findBlockIfLoaded(x, y, z).value_or(BlockType::BEDROCK);
}

bool World::isChunkLoaded(ChunkPos pos) const {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    return it != chunks_.end() && it->second->generated;
}

bool World::shouldMeshChunk(ChunkPos pos) const {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    return meshCandidateChunks_.contains(pos);
}

void World::setBlock(int64_t x, int32_t y, int64_t z, BlockType type) {
    if (y < WORLD_MIN_Y || y > WORLD_MAX_Y) return;
    const BlockPos position{x, y, z};
    setBlockLoaded(position, type,
                   type == BlockType::WATER ? std::optional(FluidState::source()) : std::nullopt);
    fluidScheduler_.activateBlockChange(position);
}

void World::setBlockLoaded(BlockPos position, BlockType type,
                           std::optional<FluidState> fluidState) {
    if (position.y < WORLD_MIN_Y || position.y > WORLD_MAX_Y) return;
    const ChunkPos pos{Chunk::worldToChunk(position.x), Chunk::worldToChunkY(position.y),
                       Chunk::worldToChunk(position.z)};
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    if (it == chunks_.end()) return;

    const BlockType oldBlock = it->second->getBlockWorld(position.x, position.y, position.z);
    it->second->setBlockWorld(position.x, position.y, position.z, type);
    const int lx = Chunk::worldToLocal(position.x);
    const int ly = Chunk::worldToLocalY(position.y);
    const int lz = Chunk::worldToLocal(position.z);
    it->second->setFluidState(lx, ly, lz, fluidState.value_or(FluidState::source()));
    it->second->modifiedSinceSave = true;
    it->second->needsMeshUpdate = true;
    it->second->version.fetch_add(1, std::memory_order_relaxed);

    if (isOpaque(oldBlock) != isOpaque(type)) {
        const ColumnPos column{pos.x, pos.z};
        skyOverrideChunkColumns_.insert(column);
        refreshSkyCutoffLocked(position.x, position.z);
        markSkyCutoffMeshesDirtyLocked(position.x, position.z);
    }

    const int minimumOffsetX = lx == 0 ? -1 : 0;
    const int maximumOffsetX = lx == CHUNK_EDGE - 1 ? 1 : 0;
    const int minimumOffsetY = ly == 0 ? -1 : 0;
    const int maximumOffsetY = ly == CHUNK_EDGE - 1 ? 1 : 0;
    const int minimumOffsetZ = lz == 0 ? -1 : 0;
    const int maximumOffsetZ = lz == CHUNK_EDGE - 1 ? 1 : 0;
    for (int offsetY = minimumOffsetY; offsetY <= maximumOffsetY; ++offsetY) {
        for (int offsetZ = minimumOffsetZ; offsetZ <= maximumOffsetZ; ++offsetZ) {
            for (int offsetX = minimumOffsetX; offsetX <= maximumOffsetX; ++offsetX) {
                if (offsetX == 0 && offsetY == 0 && offsetZ == 0) continue;
                auto neighbor = chunks_.find({pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ});
                if (neighbor == chunks_.end()) continue;
                neighbor->second->needsMeshUpdate = true;
                neighbor->second->version.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Any edit can add or remove an emitter or open or seal a light path.
    // Reconcile the edited cube and its full 3D neighborhood. Deduplication
    // keeps repeated fluid changes within the per-tick lighting budget.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                queueLightReconcile({pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ});
            }
        }
    }
}

bool World::refreshSkyCutoffLocked(int64_t worldX, int64_t worldZ) {
    const SkyColumnKey key{worldX, worldZ};
    auto updateOverride = [&](std::optional<int32_t> cutoff) {
        const auto existing = skyCutoffOverrides_.find(key);
        if (!cutoff) {
            if (existing == skyCutoffOverrides_.end()) return false;
            skyCutoffOverrides_.erase(existing);
            return true;
        }
        if (existing != skyCutoffOverrides_.end() && existing->second == *cutoff) return false;
        skyCutoffOverrides_.insert_or_assign(key, *cutoff);
        return true;
    };
    const ColumnPos column{Chunk::worldToChunk(worldX), Chunk::worldToChunk(worldZ)};
    const auto plan = generator_.findColumnPlan(column);
    const std::optional<int> top = loadedSurfaceHeight(chunks_, worldX, worldZ);
    if (!top) {
        return updateOverride(std::nullopt);
    }
    const int32_t loadedCutoff = std::clamp(*top + 1, WORLD_MIN_Y, WORLD_MAX_Y + 1);
    if (!plan) {
        return updateOverride(loadedCutoff);
    }

    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    const int32_t plannedCutoff =
        std::clamp(plan->surfaceY(localX, localZ) + 1, WORLD_MIN_Y, WORLD_MAX_Y + 1);
    const int32_t plannedSurfaceY = std::clamp(plannedCutoff - 1, WORLD_MIN_Y, WORLD_MAX_Y);
    const ChunkPos plannedSurfaceCube{column.x, Chunk::worldToChunkY(plannedSurfaceY), column.z};
    const auto surfaceCube = chunks_.find(plannedSurfaceCube);
    const bool plannedSurfaceLoaded =
        surfaceCube != chunks_.end() && surfaceCube->second->generated;

    // A partial vertical column cannot disprove immutable generated terrain
    // that has not loaded yet. It can still add a roof above that terrain. As
    // soon as the planned surface cube arrives, the loaded scan becomes
    // authoritative and can also represent a removed or lowered surface.
    if (!plannedSurfaceLoaded && loadedCutoff <= plannedCutoff) {
        return updateOverride(std::nullopt);
    }
    if (loadedCutoff == plannedCutoff) {
        return updateOverride(std::nullopt);
    }
    return updateOverride(loadedCutoff);
}

bool World::refreshSkyOverrideColumnLocked(ColumnPos column) {
    bool changed = false;
    const int64_t baseX = column.x * CHUNK_EDGE;
    const int64_t baseZ = column.z * CHUNK_EDGE;
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            changed |= refreshSkyCutoffLocked(baseX + localX, baseZ + localZ);
        }
    }
    return changed;
}

bool World::extendGeneratedSkyCutoffsLocked(const Chunk& chunk) {
    const ColumnPos column{chunk.chunkX, chunk.chunkZ};
    const auto plan = generator_.findColumnPlan(column);
    if (!plan) return false;
    const bool uniform = chunk.isUniform();
    if (uniform && !isOpaque(chunk.uniformBlock())) return false;

    bool changed = false;
    bool hasOverride = false;
    const int32_t cubeBaseY = chunk.chunkY * CHUNK_EDGE;
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const int64_t worldX = column.x * CHUNK_EDGE + localX;
            const int64_t worldZ = column.z * CHUNK_EDGE + localZ;
            const SkyColumnKey key{worldX, worldZ};
            const int32_t plannedSurfaceY = plan->surfaceY(localX, localZ);
            const int32_t plannedCutoff = plannedSurfaceY + 1;
            const auto priorOverride = skyCutoffOverrides_.find(key);
            if (priorOverride != skyCutoffOverrides_.end() &&
                Chunk::worldToChunkY(plannedSurfaceY) == chunk.chunkY) {
                // A saved cube can load before its plan and temporarily use
                // its own top. Once the authoritative surface cube arrives,
                // reconcile only those exceptional columns through the full
                // loaded stack. Ordinary generated columns never take this
                // path.
                changed |= refreshSkyCutoffLocked(worldX, worldZ);
                continue;
            }

            int localTop = uniform ? CHUNK_EDGE - 1 : -1;
            if (!uniform) {
                for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
                    if (isOpaque(chunk.getBlock(localX, localY, localZ))) {
                        localTop = localY;
                        break;
                    }
                }
            }
            if (localTop < 0) continue;

            const int32_t candidateCutoff = cubeBaseY + localTop + 1;
            if (candidateCutoff <= plannedCutoff) continue;

            const auto existing = skyCutoffOverrides_.find(key);
            if (existing == skyCutoffOverrides_.end() || candidateCutoff > existing->second) {
                skyCutoffOverrides_.insert_or_assign(key, candidateCutoff);
                changed = true;
            }
            hasOverride = true;
        }
    }
    if (hasOverride) skyOverrideChunkColumns_.insert(column);
    return changed;
}

void World::markColumnMeshesDirtyLocked(ColumnPos column) {
    for (int32_t chunkY = WORLD_MIN_CHUNK_Y; chunkY <= WORLD_MAX_CHUNK_Y; ++chunkY) {
        auto found = chunks_.find({column.x, chunkY, column.z});
        if (found == chunks_.end() || !found->second->generated) continue;
        found->second->needsMeshUpdate = true;
        found->second->version.fetch_add(1, std::memory_order_relaxed);
    }
}

void World::markHaloNeighborMeshesDirtyLocked(ChunkPos pos) {
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                if (offsetX == 0 && offsetY == 0 && offsetZ == 0) continue;
                const ChunkPos neighbor{pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ};
                if (!validChunkY(neighbor.y)) continue;
                const auto found = chunks_.find(neighbor);
                if (found == chunks_.end() || !found->second->generated) continue;
                found->second->needsMeshUpdate = true;
                found->second->version.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

void World::markSkyContinuityBelowLocked(ColumnPos column, int32_t changedSectionY) {
    // Loading or unloading one section can complete or break the vertical
    // proof of sky access for every loaded section below it. Horizontal
    // neighbors also sample this column through their one-block mesh halo.
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ColumnPos affected{column.x + offsetX, column.z + offsetZ};
            for (int32_t chunkY = WORLD_MIN_CHUNK_Y; chunkY < changedSectionY; ++chunkY) {
                const auto found = chunks_.find({affected.x, chunkY, affected.z});
                if (found == chunks_.end() || !found->second->generated) continue;
                found->second->needsMeshUpdate = true;
                found->second->version.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

void World::markSkyCutoffMeshesDirtyLocked(int64_t worldX, int64_t worldZ) {
    const ColumnPos column{Chunk::worldToChunk(worldX), Chunk::worldToChunk(worldZ)};
    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    const int minimumOffsetX = localX == 0 ? -1 : 0;
    const int maximumOffsetX = localX == CHUNK_EDGE - 1 ? 1 : 0;
    const int minimumOffsetZ = localZ == 0 ? -1 : 0;
    const int maximumOffsetZ = localZ == CHUNK_EDGE - 1 ? 1 : 0;
    for (int offsetZ = minimumOffsetZ; offsetZ <= maximumOffsetZ; ++offsetZ) {
        for (int offsetX = minimumOffsetX; offsetX <= maximumOffsetX; ++offsetX) {
            markColumnMeshesDirtyLocked({column.x + offsetX, column.z + offsetZ});
        }
    }
}

void World::markSkyColumnMeshesDirtyLocked(ColumnPos column) {
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            markColumnMeshesDirtyLocked({column.x + offsetX, column.z + offsetZ});
        }
    }
}

void World::refreshSavedSkyCutoffsLocked(ChunkPos pos) {
    const ColumnPos column{pos.x, pos.z};
    skyOverrideChunkColumns_.insert(column);
    if (refreshSkyOverrideColumnLocked(column)) {
        markSkyColumnMeshesDirtyLocked(column);
    }
}

void World::queueLightReconcile(ChunkPos pos) {
    std::lock_guard<std::mutex> lock(lightMutex_);
    if (lightQueued_.insert(pos).second) {
        lightQueue_.push_back(pos);
    }
}

void World::queueFaceNeighbors(ChunkPos pos) {
    queueLightReconcile({pos.x - 1, pos.y, pos.z});
    queueLightReconcile({pos.x + 1, pos.y, pos.z});
    queueLightReconcile({pos.x, pos.y, pos.z - 1});
    queueLightReconcile({pos.x, pos.y, pos.z + 1});
    if (validChunkY(pos.y - 1)) queueLightReconcile({pos.x, pos.y - 1, pos.z});
    if (validChunkY(pos.y + 1)) queueLightReconcile({pos.x, pos.y + 1, pos.z});
}

void World::queueLightReconcileWithNeighbors(ChunkPos pos) {
    queueLightReconcile(pos);
    queueFaceNeighbors(pos);
}

void World::reconcileLight(int budgetCubes) {
    for (int processed = 0; processed < budgetCubes; ++processed) {
        ChunkPos pos;
        {
            std::lock_guard<std::mutex> lock(lightMutex_);
            if (lightQueue_.empty()) {
                return;
            }
            pos = lightQueue_.back();
            lightQueue_.pop_back();
            lightQueued_.erase(pos);
        }

        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto it = chunks_.find(pos);
        if (it == chunks_.end() || !it->second->generated) {
            continue;
        }
        auto neighborPtr = [&](ChunkPos neighborPos) -> Chunk* {
            auto n = chunks_.find(neighborPos);
            return (n != chunks_.end() && n->second->generated) ? n->second.get() : nullptr;
        };
        std::array<Chunk*, 6> faces = {
            neighborPtr({pos.x - 1, pos.y, pos.z}),
            neighborPtr({pos.x + 1, pos.y, pos.z}),
            neighborPtr({pos.x, pos.y, pos.z - 1}),
            neighborPtr({pos.x, pos.y, pos.z + 1}),
            validChunkY(pos.y - 1) ? neighborPtr({pos.x, pos.y - 1, pos.z}) : nullptr,
            validChunkY(pos.y + 1) ? neighborPtr({pos.x, pos.y + 1, pos.z}) : nullptr,
        };

        auto dark = [](const Chunk* chunk) { return !chunk || !chunk->hasBlockLight(); };
        if (!it->second->hasBlockLight() && std::all_of(faces.begin(), faces.end(), dark) &&
            !hasBlockLightEmitter(*it->second)) {
            continue;
        }

        LightEngine::FaceNeighbors neighbors = {faces[0], faces[1], faces[2],
                                                faces[3], faces[4], faces[5]};
        if (LightEngine::floodChunk(*it->second, neighbors)) {
            it->second->needsMeshUpdate = true;
            it->second->version.fetch_add(1, std::memory_order_relaxed);
            // This chunk's border light moved: each face-neighbor both SAMPLES
            // that border (so its border faces must re-mesh even if its own
            // stored light is unchanged, such as a solid wall at the seam) and
            // may pull in more light (so it must re-reconcile).
            for (Chunk* neighbor : faces) {
                if (neighbor) {
                    neighbor->needsMeshUpdate = true;
                    neighbor->version.fetch_add(1, std::memory_order_relaxed);
                }
            }
            queueFaceNeighbors(pos);
        }
    }
}

FluidCell World::readFluidCell(FluidPos position) const {
    if (position.y < WORLD_MIN_Y) {
        return {.loaded = true, .block = BlockType::BEDROCK, .state = FluidState::source()};
    }
    if (position.y > WORLD_MAX_Y) {
        return {.loaded = true, .block = BlockType::AIR, .state = FluidState::source()};
    }
    const ChunkPos pos{Chunk::worldToChunk(position.x), Chunk::worldToChunkY(position.y),
                       Chunk::worldToChunk(position.z)};
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    if (it == chunks_.end() || !it->second->generated) return {};
    const int lx = Chunk::worldToLocal(position.x);
    const int ly = Chunk::worldToLocalY(position.y);
    const int lz = Chunk::worldToLocal(position.z);
    return {
        .loaded = true,
        .block = it->second->getBlock(lx, ly, lz),
        .state = it->second->getFluidState(lx, ly, lz),
    };
}

void World::writeWater(FluidPos position, FluidState state) {
    setBlockLoaded(position, BlockType::WATER, state);
}

void World::removeWater(FluidPos position) {
    FluidCell cell = readFluidCell(position);
    if (cell.isWater()) setBlockLoaded(position, BlockType::AIR, std::nullopt);
}

void World::queueFluidResume(ChunkPos pos) {
    if (fluidScheduler_.deferredCountIn(cubeBounds(pos)) == 0) return;
    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (fluidResumeQueued_.insert(pos).second) {
        fluidResumeQueue_.push_back(pos);
    }
}

size_t World::tickFluids(double elapsedSeconds) {
    std::vector<ChunkPos> candidates;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        const size_t count = std::min(fluidResumeQueue_.size(), MAX_FLUID_RESUME_CUBES_PER_FRAME);
        candidates.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const ChunkPos pos = fluidResumeQueue_.front();
            fluidResumeQueue_.pop_front();
            fluidResumeQueued_.erase(pos);
            candidates.push_back(pos);
        }
    }

    size_t remainingFrontiers = MAX_FLUID_FRONTIER_RESUMES_PER_FRAME;
    for (ChunkPos pos : candidates) {
        if (remainingFrontiers == 0) {
            queueFluidResume(pos);
            continue;
        }
        if (!isChunkLoaded(pos)) continue;

        const FluidBounds bounds = cubeBounds(pos);
        const size_t before = fluidScheduler_.deferredCountIn(bounds);
        if (before == 0) continue;
        const size_t budget =
            std::min({remainingFrontiers, MAX_FLUID_FRONTIER_RESUMES_PER_CUBE, before});
        fluidScheduler_.resumeDeferredIn(bounds, budget);
        remainingFrontiers -= budget;
        if (fluidScheduler_.deferredCountIn(bounds) > 0) {
            queueFluidResume(pos);
        }
    }
    return fluidScheduler_.advance(elapsedSeconds, *this);
}

float World::getFluidHeightIfLoaded(int64_t x, int32_t y, int64_t z) const {
    const FluidCell cell = readFluidCell({x, y, z});
    if (!cell.isWater()) return 0.0f;
    const FluidCell above = readFluidCell({x, y + 1, z});
    return above.isWater() ? 1.0f : fluidSurfaceHeight(cell.state);
}

size_t World::getPendingFluidCount() const {
    return fluidScheduler_.pendingCount() + fluidScheduler_.deferredCount();
}

uint64_t World::getDroppedFluidUpdateCount() const {
    return fluidScheduler_.droppedUpdateCount();
}

uint64_t World::getDroppedFluidFrontierCount() const {
    return fluidScheduler_.droppedFrontierCount();
}

bool World::snapshotForMeshing(ChunkPos pos, MeshSnapshot& out) const {
    out.pos = pos;
    out.clear();

    // Column plans are immutable and live outside the loaded-cube map. Read
    // all 18 by 18 full-column cutoffs before taking chunksMutex_, avoiding
    // thousands of vertical map probes while generation waits on the lock.
    std::array<std::shared_ptr<const ColumnPlan>, 9> columnPlans;
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            auto plan = generator_.findColumnPlan({pos.x + offsetX, pos.z + offsetZ});
            if (!plan) return false;
            columnPlans[static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1)] = std::move(plan);
        }
    }
    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        const int64_t worldZ = pos.z * CHUNK_EDGE + z;
        const int64_t columnZ = Chunk::worldToChunk(worldZ);
        const int localZ = Chunk::worldToLocal(worldZ);
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            const int64_t worldX = pos.x * CHUNK_EDGE + x;
            const int64_t columnX = Chunk::worldToChunk(worldX);
            const int localX = Chunk::worldToLocal(worldX);
            const int planX = static_cast<int>(columnX - pos.x) + 1;
            const int planZ = static_cast<int>(columnZ - pos.z) + 1;
            const auto& plan = columnPlans[static_cast<size_t>(planZ * 3 + planX)];
            const int32_t generatedCutoff = plan->surfaceY(localX, localZ) + 1;
            out.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(x, z)] = generatedCutoff;
            out.skyCutoffY[MeshSnapshot::skyIndex(x, z)] = generatedCutoff;
        }
    }

    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto self = chunks_.find(pos);
        if (self == chunks_.end() || !self->second->generated) return false;

        for (int z = -1; z <= CHUNK_EDGE; ++z) {
            const int64_t worldZ = pos.z * CHUNK_EDGE + z;
            for (int x = -1; x <= CHUNK_EDGE; ++x) {
                const int64_t worldX = pos.x * CHUNK_EDGE + x;
                const auto edited = skyCutoffOverrides_.find({worldX, worldZ});
                if (edited != skyCutoffOverrides_.end()) {
                    out.skyCutoffY[MeshSnapshot::skyIndex(x, z)] = edited->second;
                }

                // A plan tells us where generated terrain reaches the sky, but
                // it cannot make an absent vertical section visible. Until every
                // section from this cube through the cutoff is loaded, use a
                // fully occluded cutoff. This prevents sunlight from crossing an
                // unrendered gap above an underground camera.
                const int64_t columnX = Chunk::worldToChunk(worldX);
                const int64_t columnZ = Chunk::worldToChunk(worldZ);
                const int32_t cutoff = out.skyCutoffY[MeshSnapshot::skyIndex(x, z)];
                const int32_t cutoffBlock = std::clamp(cutoff - 1, WORLD_MIN_Y, WORLD_MAX_Y);
                const int32_t cutoffSection = Chunk::worldToChunkY(cutoffBlock);
                const int32_t lastRequiredSection = std::max(pos.y, cutoffSection);
                const auto loadedMask = loadedSectionMasks_.find({columnX, columnZ});
                const uint64_t requiredMask = sectionRangeMask(pos.y, lastRequiredSection);
                const bool completeSkyPath = loadedMask != loadedSectionMasks_.end() &&
                                             (loadedMask->second & requiredMask) == requiredMask;
                if (!completeSkyPath) {
                    out.skyCutoffY[MeshSnapshot::skyIndex(x, z)] = WORLD_MAX_Y + 1;
                }
            }
        }

        Chunk* neighborhood[3][3][3]{};
        for (int oy = -1; oy <= 1; ++oy) {
            for (int oz = -1; oz <= 1; ++oz) {
                for (int ox = -1; ox <= 1; ++ox) {
                    const ChunkPos neighborPos{pos.x + ox, pos.y + oy, pos.z + oz};
                    if (!validChunkY(neighborPos.y)) continue;
                    auto it = chunks_.find(neighborPos);
                    if (it != chunks_.end() && it->second->generated) {
                        neighborhood[ox + 1][oy + 1][oz + 1] = it->second.get();
                    } else {
                        if (ox == 1 && oy == 0 && oz == 0)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_PLUS_X;
                        else if (ox == -1 && oy == 0 && oz == 0)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_MINUS_X;
                        else if (ox == 0 && oy == 0 && oz == 1)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_PLUS_Z;
                        else if (ox == 0 && oy == 0 && oz == -1)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_MINUS_Z;
                        else if (ox == 0 && oy == 1 && oz == 0)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_PLUS_Y;
                        else if (ox == 0 && oy == -1 && oz == 0)
                            out.missingNeighborFaces |= MeshSnapshot::MISSING_MINUS_Y;
                    }
                }
            }
        }

        out.version = self->second->version.load(std::memory_order_relaxed);
        for (int y = -1; y <= CHUNK_EDGE; ++y) {
            const int oy = y < 0 ? -1 : (y >= CHUNK_EDGE ? 1 : 0);
            const int localY = y < 0 ? CHUNK_EDGE - 1 : (y >= CHUNK_EDGE ? 0 : y);
            for (int z = -1; z <= CHUNK_EDGE; ++z) {
                const int oz = z < 0 ? -1 : (z >= CHUNK_EDGE ? 1 : 0);
                const int localZ = z < 0 ? CHUNK_EDGE - 1 : (z >= CHUNK_EDGE ? 0 : z);
                for (int x = -1; x <= CHUNK_EDGE; ++x) {
                    const int ox = x < 0 ? -1 : (x >= CHUNK_EDGE ? 1 : 0);
                    const int localX = x < 0 ? CHUNK_EDGE - 1 : (x >= CHUNK_EDGE ? 0 : x);
                    Chunk* source = neighborhood[ox + 1][oy + 1][oz + 1];
                    if (source) {
                        const int target = MeshSnapshot::index(x, y, z);
                        out.blocks[target] = source->getBlock(localX, localY, localZ);
                        out.fluidStates[target] =
                            source->getFluidState(localX, localY, localZ).packed();
                        out.blockLight[target] = source->getBlockLight(localX, localY, localZ);
                    } else if (pos.y + oy < WORLD_MIN_CHUNK_Y) {
                        out.blocks[MeshSnapshot::index(x, y, z)] = BlockType::BEDROCK;
                    } else if (validChunkY(pos.y + oy)) {
                        // Preserve the generated terrain silhouette without
                        // presenting a full black cube face above ground. The
                        // missing halo remains conservatively solid below its
                        // planned surface, which keeps cave AO, collision-facing
                        // geometry, and underground light closure intact.
                        const int32_t worldY = pos.y * CHUNK_EDGE + y;
                        const int32_t generatedCutoff = out.generatedSurfaceCutoffAt(x, z);
                        out.blocks[MeshSnapshot::index(x, y, z)] =
                            generatedCutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN ||
                                    worldY < generatedCutoff
                                ? BlockType::BEDROCK
                                : BlockType::AIR;
                    }
                }
            }
        }
    }

    // Material lookup stays outside chunksMutex_. It runs only for a missing
    // lateral halo, touches at most 64 cached plan columns, and gives an
    // aboveground provisional cliff the same top palette as the arriving
    // terrain instead of an unlit bedrock patch.
    const auto cacheSurfaceMaterial = [&](int x, int z) {
        const int64_t worldX = pos.x * CHUNK_EDGE + x;
        const int64_t worldZ = pos.z * CHUNK_EDGE + z;
        out.generatedSurfaceMaterial[MeshSnapshot::skyIndex(x, z)] =
            generator_.surfaceMaterialAt(worldX, worldZ);
    };
    const auto needsLitSurfaceCap = [&](int selfX, int selfZ, int neighborX, int neighborZ) {
        const int32_t selfCutoff = out.generatedSurfaceCutoffAt(selfX, selfZ);
        const int32_t neighborCutoff = out.generatedSurfaceCutoffAt(neighborX, neighborZ);
        if (selfCutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN ||
            neighborCutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN || neighborCutoff <= selfCutoff) {
            return false;
        }
        const int32_t minimumY = std::max(pos.y * CHUNK_EDGE, selfCutoff);
        const int32_t maximumY = std::min(pos.y * CHUNK_EDGE + CHUNK_EDGE - 1, neighborCutoff - 1);
        for (int32_t worldY = minimumY; worldY <= maximumY; ++worldY) {
            if (!isOpaque(out.at(selfX, worldY - pos.y * CHUNK_EDGE, selfZ))) return true;
        }
        return false;
    };
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_PLUS_X) != 0) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            if (needsLitSurfaceCap(CHUNK_EDGE - 1, z, CHUNK_EDGE, z)) {
                cacheSurfaceMaterial(CHUNK_EDGE, z);
            }
        }
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_MINUS_X) != 0) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            if (needsLitSurfaceCap(0, z, -1, z)) cacheSurfaceMaterial(-1, z);
        }
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_PLUS_Z) != 0) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            if (needsLitSurfaceCap(x, CHUNK_EDGE - 1, x, CHUNK_EDGE)) {
                cacheSurfaceMaterial(x, CHUNK_EDGE);
            }
        }
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_MINUS_Z) != 0) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            if (needsLitSurfaceCap(x, 0, x, -1)) cacheSurfaceMaterial(x, -1);
        }
    }

    return true;
}

double World::getTerrainHeight(int64_t x, int64_t z) const {
    return generator_.baseHeightAt(x, z);
}

Biome World::getBiome(int64_t x, int64_t z) const {
    return generator_.biomeAt(x, z);
}

worldgen::SurfaceSample World::sampleSurface(int64_t x, int64_t z) const {
    return generator_.sampleSurface(x, z);
}

std::optional<worldgen::SurfaceSample> World::findSurfaceSample(int64_t x, int64_t z) const {
    const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
    const auto plan = generator_.findColumnPlan(column);
    if (!plan) return std::nullopt;
    return generator_.sampleSurface(x, z);
}

std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> World::getLoadedSnapshot() const {
    return std::atomic_load_explicit(&loadedSnapshot_, std::memory_order_acquire);
}

std::shared_ptr<const std::unordered_set<ChunkPos>> World::getMeshCandidateSnapshot() const {
    return std::atomic_load_explicit(&meshCandidateSnapshot_, std::memory_order_acquire);
}

void World::publishLoadedSnapshot() {
    if (!loadedSnapshotDirty_.exchange(false, std::memory_order_acq_rel)) return;
    auto snapshot = std::make_shared<std::vector<std::shared_ptr<Chunk>>>();
    auto meshCandidates = std::make_shared<std::unordered_set<ChunkPos>>();
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        snapshot->reserve(chunks_.size());
        for (const auto& [pos, chunk] : chunks_)
            snapshot->push_back(chunk);
        *meshCandidates = meshCandidateChunks_;
    }
    std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> immutable = std::move(snapshot);
    std::shared_ptr<const std::unordered_set<ChunkPos>> immutableMeshCandidates =
        std::move(meshCandidates);
    std::atomic_store_explicit(&loadedSnapshot_, std::move(immutable), std::memory_order_release);
    std::atomic_store_explicit(&meshCandidateSnapshot_, std::move(immutableMeshCandidates),
                               std::memory_order_release);
}

std::vector<std::shared_ptr<Chunk>> World::getLoadedChunks() const {
    return *getLoadedSnapshot();
}

std::vector<std::shared_ptr<Chunk>> World::getDirtyChunks() {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    std::vector<std::shared_ptr<Chunk>> result;
    result.reserve(chunks_.size());
    for (const auto& [pos, chunk] : chunks_) {
        if (chunk->needsMeshUpdate) result.push_back(chunk);
    }
    return result;
}

void World::markChunkMeshed(ChunkPos pos) {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    if (it != chunks_.end()) it->second->needsMeshUpdate = false;
}

void World::setViewDistance(int distance) {
    viewDistance_.store(
        std::clamp(distance, MIN_RENDER_DISTANCE_CHUNKS, MAX_RENDER_DISTANCE_CHUNKS),
        std::memory_order_relaxed);
    if (hasPlayerChunk_) {
        columnPlansChanged_.store(false, std::memory_order_release);
        requestActiveSetRebuild(playerChunkX_ * CHUNK_EDGE, playerChunkY_ * CHUNK_EDGE,
                                playerChunkZ_ * CHUNK_EDGE);
        activeSetRebuildCooldownTicks_ = COLUMN_PLAN_REBUILD_COOLDOWN_TICKS;
    }
}

void World::updatePlayerPosition(int64_t playerX, int32_t playerY, int64_t playerZ) {
    reconcileLight(16);

    const int64_t newChunkX = Chunk::worldToChunk(playerX);
    const int32_t newChunkY = Chunk::worldToChunkY(playerY);
    const int64_t newChunkZ = Chunk::worldToChunk(playerZ);
    if (hasPlayerChunk_ && newChunkX == playerChunkX_ && newChunkY == playerChunkY_ &&
        newChunkZ == playerChunkZ_) {
        const bool allowPlanRebuild = activeSetRebuildCooldownTicks_ == 0;
        if (activeSetRebuildCooldownTicks_ > 0) --activeSetRebuildCooldownTicks_;
        if (allowPlanRebuild && columnPlansChanged_.exchange(false, std::memory_order_acq_rel)) {
            requestActiveSetRebuild(playerX, playerY, playerZ);
            activeSetRebuildCooldownTicks_ = COLUMN_PLAN_REBUILD_COOLDOWN_TICKS;
            return;
        }
        pumpGeneration();
        return;
    }

    hasPlayerChunk_ = true;
    playerChunkX_ = newChunkX;
    playerChunkY_ = newChunkY;
    playerChunkZ_ = newChunkZ;
    columnPlansChanged_.store(false, std::memory_order_release);
    requestActiveSetRebuild(playerX, playerY, playerZ);
    activeSetRebuildCooldownTicks_ = COLUMN_PLAN_REBUILD_COOLDOWN_TICKS;
}

bool World::shouldRetain(ChunkPos pos) const {
    return retainedChunks_.contains(pos);
}

void World::queueGenerationLocked(ChunkPos pos) {
    if (generationsInFlight_.contains(pos) || !genBacklogSet_.insert(pos).second) return;
    genBacklog_.push_back(pos);
}

void World::registerPlanDependenciesLocked(ChunkPos pos) {
    const ColumnPos ownColumn{pos.x, pos.z};
    const auto ownPlan = generator_.findColumnPlan(ownColumn);
    if (!ownPlan) {
        missingPlanDependencies_[pos] = 1;
        planDependents_[ownColumn].push_back({pos, activeSetEpoch_, PlanDependencyKind::OWN_PLAN});
        return;
    }
    if (!ownPlan->exposesSection(pos.y)) {
        queueGenerationLocked(pos);
        return;
    }

    uint8_t missing = 0;
    for (int offsetZ = -2; offsetZ <= 2; ++offsetZ) {
        for (int offsetX = -2; offsetX <= 2; ++offsetX) {
            const ColumnPos dependency{pos.x + offsetX, pos.z + offsetZ};
            if (generator_.findColumnPlan(dependency)) continue;
            ++missing;
            planDependents_[dependency].push_back(
                {pos, activeSetEpoch_, PlanDependencyKind::EXPOSED_APRON});
        }
    }
    if (missing == 0) {
        queueGenerationLocked(pos);
    } else {
        missingPlanDependencies_[pos] = missing;
    }
}

void World::wakePlanDependents(ColumnPos completedPlan) {
    std::vector<PlanDependent> dependents;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        auto iterator = planDependents_.find(completedPlan);
        if (iterator == planDependents_.end()) return;
        dependents = std::move(iterator->second);
        planDependents_.erase(iterator);
    }
    planDependentChecks_.fetch_add(dependents.size(), std::memory_order_relaxed);

    // The completed plan and its dependency list are immutable. Resolve the
    // only potentially expensive cache probes before returning to scheduler
    // state, and never acquire the loaded-world lock from a completion.
    const auto ownPlan = generator_.findColumnPlan(completedPlan);
    std::array<ColumnPos, 25> apron{};
    size_t apronCount = 0;
    if (ownPlan) {
        for (int offsetZ = -2; offsetZ <= 2; ++offsetZ) {
            for (int offsetX = -2; offsetX <= 2; ++offsetX) {
                const ColumnPos dependency{completedPlan.x + offsetX, completedPlan.z + offsetZ};
                if (!generator_.findColumnPlan(dependency)) apron[apronCount++] = dependency;
            }
        }
    }

    std::lock_guard<std::mutex> lock(pendingMutex_);
    std::vector<ColumnPos> missingApron;
    missingApron.reserve(apronCount);
    for (size_t index = 0; index < apronCount; ++index) {
        // A neighbor can publish between the cache probe above and this
        // scheduler lock. Recheck here so no cube registers after the only
        // completion notification for that dependency has already passed.
        if (!generator_.findColumnPlan(apron[index])) missingApron.push_back(apron[index]);
    }

    for (const PlanDependent& dependent : dependents) {
        if (dependent.activeSetEpoch != activeSetEpoch_) continue;
        auto missing = missingPlanDependencies_.find(dependent.pos);
        if (missing == missingPlanDependencies_.end()) continue;

        if (dependent.kind == PlanDependencyKind::EXPOSED_APRON) {
            if (missing->second > 1) {
                --missing->second;
            } else {
                missingPlanDependencies_.erase(missing);
                queueGenerationLocked(dependent.pos);
            }
            continue;
        }

        missingPlanDependencies_.erase(missing);
        if (!ownPlan) continue;
        if (!ownPlan->exposesSection(dependent.pos.y)) {
            queueGenerationLocked(dependent.pos);
            continue;
        }
        if (missingApron.empty()) {
            queueGenerationLocked(dependent.pos);
            continue;
        }

        missingPlanDependencies_[dependent.pos] = static_cast<uint8_t>(missingApron.size());
        for (ColumnPos dependency : missingApron) {
            planDependents_[dependency].push_back(
                {dependent.pos, activeSetEpoch_, PlanDependencyKind::EXPOSED_APRON});
        }
    }
}

void World::unloadDistantChunks() {
    std::vector<std::shared_ptr<Chunk>> toSave;
    std::vector<ChunkPos> unloaded;
    std::unordered_set<ColumnPos> unloadedColumns;
    std::unordered_set<ColumnPos> skyColumnsNeedingRefresh;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        for (auto it = chunks_.begin(); it != chunks_.end();) {
            if (!shouldRetain(it->first)) {
                const ChunkPos position = it->first;
                const ColumnPos column{position.x, position.z};
                markHaloNeighborMeshesDirtyLocked(position);
                markSkyContinuityBelowLocked(column, position.y);
                auto sectionMask = loadedSectionMasks_.find(column);
                if (sectionMask != loadedSectionMasks_.end()) {
                    sectionMask->second &= ~sectionBit(position.y);
                    if (sectionMask->second == 0) loadedSectionMasks_.erase(sectionMask);
                }
                unloaded.push_back(position);
                unloadedColumns.insert(column);
                if (skyOverrideChunkColumns_.contains(column)) {
                    bool ownsOverride = false;
                    const auto plan = generator_.findColumnPlan(column);
                    const int64_t baseX = column.x * CHUNK_EDGE;
                    const int64_t baseZ = column.z * CHUNK_EDGE;
                    for (int localZ = 0; localZ < CHUNK_EDGE && !ownsOverride; ++localZ) {
                        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                            const auto cutoff =
                                skyCutoffOverrides_.find({baseX + localX, baseZ + localZ});
                            if (cutoff == skyCutoffOverrides_.end()) continue;
                            const bool ownsLoadedTop =
                                Chunk::worldToChunkY(cutoff->second - 1) == position.y;
                            const bool ownsPlanAuthority =
                                plan && cutoff->second <= plan->surfaceY(localX, localZ) + 1 &&
                                Chunk::worldToChunkY(plan->surfaceY(localX, localZ)) == position.y;
                            if (ownsLoadedTop || ownsPlanAuthority) {
                                ownsOverride = true;
                                break;
                            }
                        }
                    }
                    if (ownsOverride) skyColumnsNeedingRefresh.insert(column);
                }
                if (it->second->modifiedSinceSave && saveManager_) {
                    toSave.push_back(it->second);
                }
                it = chunks_.erase(it);
                loadedCubeCount_.fetch_sub(1, std::memory_order_relaxed);
                loadedSnapshotDirty_.store(true, std::memory_order_release);
            } else {
                ++it;
            }
        }
        for (ColumnPos column : unloadedColumns) {
            if (!skyOverrideChunkColumns_.contains(column)) continue;
            bool columnStillLoaded = false;
            for (int32_t chunkY = WORLD_MIN_CHUNK_Y; chunkY <= WORLD_MAX_CHUNK_Y; ++chunkY) {
                if (chunks_.contains({column.x, chunkY, column.z})) {
                    columnStillLoaded = true;
                    break;
                }
            }
            if (columnStillLoaded) {
                if (skyColumnsNeedingRefresh.contains(column) &&
                    refreshSkyOverrideColumnLocked(column)) {
                    markSkyColumnMeshesDirtyLocked(column);
                }
                continue;
            }
            const int64_t baseX = column.x * CHUNK_EDGE;
            const int64_t baseZ = column.z * CHUNK_EDGE;
            for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
                for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                    skyCutoffOverrides_.erase({baseX + localX, baseZ + localZ});
                }
            }
            skyOverrideChunkColumns_.erase(column);
            markSkyColumnMeshesDirtyLocked(column);
        }
    }
    for (ChunkPos pos : unloaded)
        queueFaceNeighbors(pos);
    for (auto& chunk : toSave)
        saveManager_->saveChunkAsync(std::move(chunk));
}

bool World::saveModifiedChunks() {
    if (!saveManager_) return false;
    std::vector<std::shared_ptr<Chunk>> toSave;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        for (auto& [pos, chunk] : chunks_) {
            if (!chunk->modifiedSinceSave) continue;
            toSave.push_back(chunk);
        }
    }
    for (auto& chunk : toSave)
        saveManager_->saveChunkAsync(std::move(chunk));
    return saveManager_->saveDeferredFluidFrontiers(fluidScheduler_.deferredFrontiers());
}

void World::generateAroundPlayer(int64_t playerX, int32_t playerY, int64_t playerZ) {
    ensureStreamingWorkers();
    columnPlansChanged_.store(false, std::memory_order_release);
    const ActiveSetRequest request{
        .playerX = playerX,
        .playerY = playerY,
        .playerZ = playerZ,
        .viewDistance = getViewDistance(),
        .id = 0,
    };
    std::lock_guard<std::mutex> buildLock(activeSetBuildMutex_);
    (void)rebuildActiveSet(request);
}

void World::ensureStreamingWorkers() {
    std::lock_guard<std::mutex> lock(activeSetRequestMutex_);
    if (!genPool_) {
        genPool_ = std::make_shared<ThreadPool>(4, ThreadPriority::UTILITY);
    }
    if (!activeSetThread_.joinable()) {
        activeSetThread_ = std::thread([this] { activeSetWorkerLoop(); });
    }
}

void World::requestActiveSetRebuild(int64_t playerX, int32_t playerY, int64_t playerZ) {
    ensureStreamingWorkers();
    {
        std::lock_guard<std::mutex> lock(activeSetRequestMutex_);
        if (shuttingDown_.load(std::memory_order_acquire) || stopActiveSetThread_) return;
        if (activeSetWorkPending_.load(std::memory_order_acquire)) {
            activeSetRequestsCoalesced_.fetch_add(1, std::memory_order_relaxed);
        }
        const uint64_t requestId = ++nextActiveSetRequestId_;
        pendingActiveSetRequest_ = {
            .playerX = playerX,
            .playerY = playerY,
            .playerZ = playerZ,
            .viewDistance = getViewDistance(),
            .id = requestId,
        };
        latestActiveSetRequestId_.store(requestId, std::memory_order_release);
        activeSetWorkPending_.store(true, std::memory_order_release);
        activeSetRequests_.fetch_add(1, std::memory_order_relaxed);
    }
    activeSetRequestCv_.notify_one();
}

void World::activeSetWorkerLoop() {
    setCurrentThreadPriority(ThreadPriority::UTILITY);
    for (;;) {
        ActiveSetRequest request;
        {
            std::unique_lock<std::mutex> lock(activeSetRequestMutex_);
            activeSetRequestCv_.wait(lock, [this] {
                return stopActiveSetThread_ || pendingActiveSetRequest_.has_value();
            });
            if (stopActiveSetThread_) return;
            request = *pendingActiveSetRequest_;
            pendingActiveSetRequest_.reset();
        }

        try {
            bool published = false;
            {
                std::lock_guard<std::mutex> buildLock(activeSetBuildMutex_);
                published = rebuildActiveSet(request);
            }
            if (published && !activeSetRequestIsStale(request.id)) {
                unloadDistantChunks();
            }
        } catch (const std::exception& error) {
            RY_LOG_ERROR((std::string("Active-set rebuild failed: ") + error.what()).c_str());
        } catch (...) {
            RY_LOG_ERROR("Active-set rebuild failed with an unknown exception");
        }

        {
            std::lock_guard<std::mutex> lock(activeSetRequestMutex_);
            if (!pendingActiveSetRequest_) {
                activeSetWorkPending_.store(false, std::memory_order_release);
            }
        }
    }
}

bool World::activeSetRequestIsStale(uint64_t requestId) const {
    return shuttingDown_.load(std::memory_order_acquire) ||
           (requestId != 0 &&
            latestActiveSetRequestId_.load(std::memory_order_acquire) != requestId);
}

bool World::rebuildActiveSet(const ActiveSetRequest& request) {
    const auto rebuildStart = std::chrono::steady_clock::now();
    activeSetRebuilds_.fetch_add(1, std::memory_order_relaxed);
    const int64_t centerX = Chunk::worldToChunk(request.playerX);
    const int32_t centerY = Chunk::worldToChunkY(request.playerY);
    const int64_t centerZ = Chunk::worldToChunk(request.playerZ);
    // Exact cubes carry edits, collision, entities, flora, and runtime fluids.
    // The renderer extends the visible horizon with immutable far-terrain LOD
    // tiles, so a 256-chunk view never expands cubic simulation to 4 km.
    const int exactViewDistance = std::min(request.viewDistance, MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
    const int radius = std::max(exactViewDistance + 1, EXPLORATION_RADIUS_CHUNKS);
    constexpr uint8_t EXPLORATION_PRIORITY = 5;
    auto cancelIfStale = [&] {
        if (!activeSetRequestIsStale(request.id)) return false;
        activeSetBuildsCanceled_.fetch_add(1, std::memory_order_relaxed);
        activeSetBuildMs_.record(std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - rebuildStart)
                                     .count());
        return true;
    };

    std::vector<ColumnPos> visibleColumns;
    const size_t visibleSquare =
        static_cast<size_t>(radius * 2 + 1) * static_cast<size_t>(radius * 2 + 1);
    visibleColumns.reserve(visibleSquare);
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz <= radius * radius) {
                visibleColumns.push_back({centerX + dx, centerZ + dz});
            }
        }
    }
    if (cancelIfStale()) return false;
    std::unordered_map<ColumnPos, std::vector<int32_t>> savedSections;
    if (saveManager_) savedSections = saveManager_->savedSectionsForColumns(visibleColumns);
    if (cancelIfStale()) return false;

    std::unordered_set<ChunkPos> wantedSet;
    std::unordered_map<ChunkPos, uint8_t> wantedPriority;
    wantedSet.reserve(
        std::min<size_t>(MAX_LOADED_CUBES * 2, static_cast<size_t>(radius * radius * 8)));
    wantedPriority.reserve(wantedSet.bucket_count());
    auto addWanted = [&](ChunkPos pos, uint8_t priority) {
        if (!validChunkY(pos.y)) return;
        wantedSet.insert(pos);
        auto [iterator, inserted] = wantedPriority.try_emplace(pos, priority);
        if (!inserted) iterator->second = std::max(iterator->second, priority);
    };
    for (ColumnPos column : visibleColumns) {
        const int dx = static_cast<int>(column.x - centerX);
        const int dz = static_cast<int>(column.z - centerZ);
        const int64_t chunkX = column.x;
        const int64_t chunkZ = column.z;
        if (const auto plan = generator_.findColumnPlan(column)) {
            for (int32_t section : plan->exposedSections()) {
                addWanted({chunkX, section, chunkZ}, 4);
            }

            // Each east and south face has one deterministic owner. When
            // exact density surfaces straddle several vertical sections,
            // mesh every section of the higher column so a canyon or
            // cliff cannot disappear between two otherwise valid surface
            // cubes. Neighbor plans arrive through the existing apron;
            // columnPlansChanged_ rebuilds this set after a cold miss.
            auto exposeBoundaryWall = [&](ColumnPos adjacent, bool eastFace) {
                const auto neighbor = generator_.findColumnPlan(adjacent);
                if (!neighbor) return;
                for (int coordinate = 0; coordinate < CHUNK_EDGE; ++coordinate) {
                    const int currentY = eastFace ? plan->surfaceY(CHUNK_EDGE - 1, coordinate)
                                                  : plan->surfaceY(coordinate, CHUNK_EDGE - 1);
                    const int adjacentY = eastFace ? neighbor->surfaceY(0, coordinate)
                                                   : neighbor->surfaceY(coordinate, 0);
                    if (std::abs(currentY - adjacentY) <= CHUNK_EDGE) continue;
                    const ColumnPos higher = currentY > adjacentY ? column : adjacent;
                    const int minimumY = std::min(currentY, adjacentY);
                    const int maximumY = std::max(currentY, adjacentY);
                    const int32_t first = Chunk::worldToChunkY(minimumY);
                    const int32_t last = Chunk::worldToChunkY(maximumY);
                    for (int32_t section = first; section <= last; ++section) {
                        addWanted({higher.x, section, higher.z}, 4);
                    }
                }
            };
            exposeBoundaryWall({chunkX + 1, chunkZ}, true);
            exposeBoundaryWall({chunkX, chunkZ + 1}, false);
        } else {
            addWanted({chunkX, Chunk::worldToChunkY(SEA_LEVEL), chunkZ}, 4);
        }

        if (const auto saved = savedSections.find(column); saved != savedSections.end()) {
            for (int32_t section : saved->second) {
                addWanted({chunkX, section, chunkZ}, 2);
            }
        }

        if (std::abs(dx) <= EXPLORATION_RADIUS_CHUNKS &&
            std::abs(dz) <= EXPLORATION_RADIUS_CHUNKS &&
            dx * dx + dz * dz <= EXPLORATION_RADIUS_CHUNKS * EXPLORATION_RADIUS_CHUNKS) {
            for (int oy = -EXPLORATION_VERTICAL_RADIUS_CUBES;
                 oy <= EXPLORATION_VERTICAL_RADIUS_CUBES; ++oy) {
                const int32_t y = centerY + oy;
                addWanted({chunkX, y, chunkZ}, EXPLORATION_PRIORITY);
            }
        }
    }
    if (cancelIfStale()) return false;

    std::vector<ChunkPos> wanted(wantedSet.begin(), wantedSet.end());
    std::sort(wanted.begin(), wanted.end(), [&](ChunkPos left, ChunkPos right) {
        const uint8_t leftPriority = wantedPriority.at(left);
        const uint8_t rightPriority = wantedPriority.at(right);
        if (leftPriority != rightPriority) return leftPriority < rightPriority;
        return distanceSq(left, centerX, centerY, centerZ) >
               distanceSq(right, centerX, centerY, centerZ);
    });
    if (wanted.size() > MAX_MESH_RESIDENT_CUBES) {
        wanted.erase(wanted.begin(), wanted.begin() + (wanted.size() - MAX_MESH_RESIDENT_CUBES));
        wantedSet.clear();
        wantedSet.insert(wanted.begin(), wanted.end());
    }

    std::unordered_set<ChunkPos> retained = wantedSet;
    std::unordered_map<ChunkPos, uint8_t> haloPriority;
    retained.reserve(wantedSet.size() * 2);
    for (ChunkPos pos : wantedSet) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    ChunkPos neighbor{pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ};
                    if (!validChunkY(neighbor.y)) continue;
                    retained.insert(neighbor);
                    if (wantedSet.contains(neighbor)) continue;
                    auto [iterator, inserted] =
                        haloPriority.try_emplace(neighbor, wantedPriority.at(pos));
                    if (!inserted)
                        iterator->second = std::max(iterator->second, wantedPriority.at(pos));
                }
            }
        }
    }

    if (retained.size() > MAX_LOADED_CUBES) {
        std::vector<ChunkPos> halo;
        halo.reserve(retained.size() - wantedSet.size());
        for (ChunkPos pos : retained) {
            if (!wantedSet.contains(pos)) halo.push_back(pos);
        }
        std::sort(halo.begin(), halo.end(), [&](ChunkPos left, ChunkPos right) {
            const uint8_t leftPriority = haloPriority.at(left);
            const uint8_t rightPriority = haloPriority.at(right);
            if (leftPriority != rightPriority) return leftPriority < rightPriority;
            return distanceSq(left, centerX, centerY, centerZ) >
                   distanceSq(right, centerX, centerY, centerZ);
        });
        const size_t haloBudget = MAX_LOADED_CUBES - wantedSet.size();
        if (halo.size() > haloBudget) {
            halo.erase(halo.begin(), halo.begin() + (halo.size() - haloBudget));
        }
        retained = wantedSet;
        retained.insert(halo.begin(), halo.end());
    }
    if (cancelIfStale()) return false;

    for (auto iterator = wantedSet.begin(); iterator != wantedSet.end();) {
        bool completeHalo = true;
        for (int offsetY = -1; offsetY <= 1 && completeHalo; ++offsetY) {
            for (int offsetZ = -1; offsetZ <= 1 && completeHalo; ++offsetZ) {
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    const ChunkPos neighbor{iterator->x + offsetX, iterator->y + offsetY,
                                            iterator->z + offsetZ};
                    if (validChunkY(neighbor.y) && !retained.contains(neighbor)) {
                        completeHalo = false;
                        break;
                    }
                }
            }
        }
        if (completeHalo) {
            ++iterator;
        } else {
            iterator = wantedSet.erase(iterator);
        }
    }

    std::unordered_set<ColumnPos> planCenters;
    planCenters.reserve(retained.size());
    for (ChunkPos pos : retained)
        planCenters.insert({pos.x, pos.z});

    std::unordered_set<ColumnPos> requestedPlans;
    requestedPlans.reserve(planCenters.size() * 2 + 64);
    for (ColumnPos center : planCenters) {
        for (int apronZ = -2; apronZ <= 2; ++apronZ) {
            for (int apronX = -2; apronX <= 2; ++apronX) {
                requestedPlans.insert({center.x + apronX, center.z + apronZ});
            }
        }
    }
    if (cancelIfStale()) return false;
    planApronCenters_.fetch_add(planCenters.size(), std::memory_order_relaxed);
    planApronExpansionAttempts_.fetch_add(planCenters.size() * 25, std::memory_order_relaxed);
    planApronCubeExpansionEquivalent_.fetch_add(retained.size() * 25, std::memory_order_relaxed);

    std::vector<ChunkPos> loadOrder(retained.begin(), retained.end());
    sortChunksByDistance(loadOrder, centerX, centerY, centerZ);
    std::vector<ColumnPos> planOrder(requestedPlans.begin(), requestedPlans.end());
    std::sort(planOrder.begin(), planOrder.end(), [&](ColumnPos left, ColumnPos right) {
        const int64_t leftX = left.x - centerX;
        const int64_t leftZ = left.z - centerZ;
        const int64_t rightX = right.x - centerX;
        const int64_t rightZ = right.z - centerZ;
        return leftX * leftX + leftZ * leftZ > rightX * rightX + rightZ * rightZ;
    });

    // Snapshot only cubes with work or storage to preserve. The bounded
    // neighborhood probes and ordering then run without either world lock.
    std::vector<ChunkPos> priorRetentionCandidates;
    if (retained.size() < MAX_LOADED_CUBES) {
        std::lock_guard<std::mutex> pendingLock(pendingMutex_);
        std::lock_guard<std::mutex> chunksLock(chunksMutex_);
        priorRetentionCandidates.reserve(retainedChunks_.size());
        for (ChunkPos candidate : retainedChunks_) {
            if (chunks_.contains(candidate) || generationsInFlight_.contains(candidate)) {
                priorRetentionCandidates.push_back(candidate);
            }
        }
    }
    std::vector<ChunkPos> hysteresisCandidates;
    hysteresisCandidates.reserve(priorRetentionCandidates.size());
    for (ChunkPos candidate : priorRetentionCandidates) {
        if (retained.contains(candidate)) continue;
        bool nearCurrentSet = false;
        for (int offsetY = -VERTICAL_UNLOAD_HYSTERESIS_CUBES;
             offsetY <= VERTICAL_UNLOAD_HYSTERESIS_CUBES && !nearCurrentSet; ++offsetY) {
            for (int offsetZ = -HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS;
                 offsetZ <= HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS && !nearCurrentSet; ++offsetZ) {
                for (int offsetX = -HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS;
                     offsetX <= HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS; ++offsetX) {
                    if (offsetX * offsetX + offsetZ * offsetZ >
                        HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS * HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS) {
                        continue;
                    }
                    if (retained.contains({candidate.x + offsetX, candidate.y + offsetY,
                                           candidate.z + offsetZ})) {
                        nearCurrentSet = true;
                        break;
                    }
                }
            }
        }
        if (nearCurrentSet) hysteresisCandidates.push_back(candidate);
    }
    std::sort(hysteresisCandidates.begin(), hysteresisCandidates.end(),
              [&](ChunkPos left, ChunkPos right) {
                  const int64_t leftDistance = distanceSq(left, centerX, centerY, centerZ);
                  const int64_t rightDistance = distanceSq(right, centerX, centerY, centerZ);
                  if (leftDistance != rightDistance) return leftDistance < rightDistance;
                  if (left.x != right.x) return left.x < right.x;
                  if (left.y != right.y) return left.y < right.y;
                  return left.z < right.z;
              });
    const size_t hysteresisBudget = MAX_LOADED_CUBES - retained.size();
    if (hysteresisCandidates.size() > hysteresisBudget) {
        hysteresisCandidates.resize(hysteresisBudget);
    }
    retained.insert(hysteresisCandidates.begin(), hysteresisCandidates.end());
    hysteresisRetainedCubes_.fetch_add(hysteresisCandidates.size(), std::memory_order_relaxed);
    if (cancelIfStale()) return false;

    {
        std::lock_guard<std::mutex> pendingLock(pendingMutex_);
        if (cancelIfStale()) return false;
        std::vector<ChunkPos> dependencyCandidates;
        dependencyCandidates.reserve(loadOrder.size());
        size_t retainedCubeCount = 0;
        {
            std::lock_guard<std::mutex> chunksLock(chunksMutex_);
            retainedChunks_ = std::move(retained);
            retainedCubeCount = retainedChunks_.size();
            if (meshCandidateChunks_ != wantedSet) {
                meshCandidateChunks_ = wantedSet;
                loadedSnapshotDirty_.store(true, std::memory_order_release);
            }
            for (ChunkPos pos : loadOrder) {
                if (!chunks_.contains(pos) && !generationsInFlight_.contains(pos)) {
                    dependencyCandidates.push_back(pos);
                }
            }
        }
        ++activeSetEpoch_;
        retainedCubeCountForStats_ = retainedCubeCount;
        genBacklog_.clear();
        genBacklogSet_.clear();
        columnPlanBacklog_.clear();
        planDependents_.clear();
        missingPlanDependencies_.clear();
        genBacklogSet_.reserve(loadOrder.size());
        planDependents_.reserve(requestedPlans.size());
        missingPlanDependencies_.reserve(loadOrder.size());
        for (ColumnPos pos : planOrder) {
            if (!generator_.findColumnPlan(pos) && !columnPlansInFlight_.contains(pos)) {
                columnPlanBacklog_.push_back(pos);
            }
        }
        for (ChunkPos pos : dependencyCandidates)
            registerPlanDependenciesLocked(pos);
    }
    pumpGeneration();
    activeSetBuildMs_.record(
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - rebuildStart)
            .count());
    return true;
}

void World::pumpGeneration() {
    if (!genPool_ || shuttingDown_.load()) return;
    std::vector<ChunkPos> toSubmit;
    std::vector<ColumnPos> plansToSubmit;
    bool notifyDrainedPlans = false;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        for (auto it = pendingGenerations_.begin(); it != pendingGenerations_.end();) {
            if (!it->second.valid() ||
                it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                generationsInFlight_.erase(it->first);
                it = pendingGenerations_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = pendingColumnPlans_.begin(); it != pendingColumnPlans_.end();) {
            if (!it->second.valid() ||
                it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                it = pendingColumnPlans_.erase(it);
            } else {
                ++it;
            }
        }
        while (activeColumnPlanJobs_ < MAX_COLD_COLUMN_PLANS && !columnPlanBacklog_.empty()) {
            const ColumnPos pos = columnPlanBacklog_.back();
            columnPlanBacklog_.pop_back();
            if (generator_.findColumnPlan(pos) || columnPlansInFlight_.contains(pos)) continue;
            columnPlansInFlight_.insert(pos);
            ++activeColumnPlanJobs_;
            plansToSubmit.push_back(pos);
        }
        while (generationsInFlight_.size() < MAX_INFLIGHT_GEN && !genBacklog_.empty()) {
            const ChunkPos pos = genBacklog_.back();
            genBacklog_.pop_back();
            genBacklogSet_.erase(pos);
            if (!generationsInFlight_.insert(pos).second) continue;
            toSubmit.push_back(pos);
        }
        if (activeColumnPlanJobs_ == 0 && columnPlanBacklog_.empty() &&
            completedPlansSinceRebuild_ != 0) {
            completedPlansSinceRebuild_ = 0;
            notifyDrainedPlans = true;
        }
    }
    if (notifyDrainedPlans) {
        columnPlansChanged_.store(true, std::memory_order_release);
        activeSetRebuildNotifications_.fetch_add(1, std::memory_order_relaxed);
    }
    for (ColumnPos pos : plansToSubmit)
        generateColumnPlanAsync(pos);
    for (ChunkPos pos : toSubmit)
        generateChunkAsync(pos);
}

size_t World::getPendingChunkCount() const {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    return genBacklog_.size() + generationsInFlight_.size() + columnPlanBacklog_.size() +
           activeColumnPlanJobs_ +
           static_cast<size_t>(activeSetWorkPending_.load(std::memory_order_acquire));
}

size_t World::getLoadedChunkCount() const {
    return loadedCubeCount_.load(std::memory_order_relaxed);
}

StreamingWorkStats World::getStreamingWorkStats() const {
    return {
        .activeSetRebuilds = activeSetRebuilds_.load(std::memory_order_relaxed),
        .planApronCenters = planApronCenters_.load(std::memory_order_relaxed),
        .planApronExpansionAttempts = planApronExpansionAttempts_.load(std::memory_order_relaxed),
        .planApronCubeExpansionEquivalent =
            planApronCubeExpansionEquivalent_.load(std::memory_order_relaxed),
        .completedColumnPlans = completedColumnPlans_.load(std::memory_order_relaxed),
        .planDependentChecks = planDependentChecks_.load(std::memory_order_relaxed),
        .fullRetainedScanEquivalent = fullRetainedScanEquivalent_.load(std::memory_order_relaxed),
        .activeSetRebuildNotifications =
            activeSetRebuildNotifications_.load(std::memory_order_relaxed),
        .hysteresisRetainedCubes = hysteresisRetainedCubes_.load(std::memory_order_relaxed),
        .activeSetRequests = activeSetRequests_.load(std::memory_order_relaxed),
        .activeSetRequestsCoalesced = activeSetRequestsCoalesced_.load(std::memory_order_relaxed),
        .activeSetBuildsCanceled = activeSetBuildsCanceled_.load(std::memory_order_relaxed),
        .activeSetBuildMs = activeSetBuildMs_.value(),
    };
}

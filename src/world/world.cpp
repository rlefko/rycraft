#include "world/world.hpp"

#include "common/error.hpp"
#include "common/thread_priority.hpp"
#include "world/furnace.hpp"
#include "world/learned_terrain.hpp"
#include "world/light_engine.hpp"
#include "world/save_manager.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
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

VerticalSectionMask
potentialSkyOccupancySections(const ColumnPlan& plan,
                              const VerticalSectionMask* savedEditedSections = nullptr,
                              std::optional<int32_t> contiguousFromSection = std::nullopt) {
    VerticalSectionMask required;
    for (const int32_t section : plan.exposedSections())
        required.set(section);
    if (savedEditedSections) required.merge(*savedEditedSections);
    if (contiguousFromSection) {
        const int32_t plannedSurfaceSection =
            Chunk::worldToChunkY(std::clamp(plan.maximumSurfaceY(), WORLD_MIN_Y, WORLD_MAX_Y));
        for (int32_t section = std::max(*contiguousFromSection, WORLD_MIN_CHUNK_Y);
             section <= plannedSurfaceSection; ++section) {
            required.set(section);
        }
    }
    return required;
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
                    const std::unordered_map<ColumnPos, VerticalSectionMask>& loadedSectionMasks,
                    int64_t worldX, int64_t worldZ) {
    const int64_t chunkX = Chunk::worldToChunk(worldX);
    const int64_t chunkZ = Chunk::worldToChunk(worldZ);
    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    const auto loadedSections = loadedSectionMasks.find({chunkX, chunkZ});
    if (loadedSections == loadedSectionMasks.end()) return std::nullopt;
    std::optional<int> result;
    loadedSections->second.visitSetSectionsDescending(
        WORLD_MIN_CHUNK_Y, loadedSections->second.highestSection(), [&](int32_t chunkY) {
            const auto iterator = chunks.find({chunkX, chunkY, chunkZ});
            if (iterator == chunks.end() || !iterator->second->generated) return true;
            for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
                if (isOpaque(iterator->second->getBlock(localX, localY, localZ))) {
                    result = chunkY * CHUNK_EDGE + localY;
                    return false;
                }
            }
            return true;
        });
    return result.value_or(WORLD_MIN_Y - 1);
}

std::optional<float> loadedStrikeSurfaceHeight(
    const std::unordered_map<ChunkPos, std::shared_ptr<Chunk>>& chunks,
    const std::unordered_map<ColumnPos, VerticalSectionMask>& loadedSectionMasks, int64_t worldX,
    int64_t worldZ) {
    const int64_t chunkX = Chunk::worldToChunk(worldX);
    const int64_t chunkZ = Chunk::worldToChunk(worldZ);
    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    const auto loadedSections = loadedSectionMasks.find({chunkX, chunkZ});
    if (loadedSections == loadedSectionMasks.end()) return std::nullopt;
    std::optional<float> result;
    loadedSections->second.visitSetSectionsDescending(
        WORLD_MIN_CHUNK_Y, loadedSections->second.highestSection(), [&](int32_t chunkY) {
            const auto iterator = chunks.find({chunkX, chunkY, chunkZ});
            if (iterator == chunks.end() || !iterator->second->generated) return true;
            for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
                const BlockType block = iterator->second->getBlock(localX, localY, localZ);
                const float worldY = static_cast<float>(chunkY * CHUNK_EDGE + localY);
                if (block == BlockType::WATER) {
                    result = worldY + fluidSurfaceHeight(
                                          iterator->second->getFluidState(localX, localY, localZ));
                    return false;
                }
                if (isOpaque(block)) {
                    result = worldY + 1.0F;
                    return false;
                }
            }
            return true;
        });
    return result;
}

void recordLoadedCubeHighWater(std::atomic<size_t>& highWater, size_t candidate) {
    size_t previous = highWater.load(std::memory_order_relaxed);
    while (previous < candidate &&
           !highWater.compare_exchange_weak(previous, candidate, std::memory_order_relaxed)) {
    }
}

std::atomic<uint64_t> nextWorldInstanceId{1};

uint64_t allocateWorldInstanceId() {
    uint64_t id = nextWorldInstanceId.fetch_add(1, std::memory_order_relaxed);
    // Zero is reserved as the renderer's uninitialized identity. Reaching
    // this branch requires creating more than 2^64 Worlds in one process.
    if (id == 0) id = nextWorldInstanceId.fetch_add(1, std::memory_order_relaxed);
    return id;
}

std::optional<FluidState>
canonicalCollisionFluidState(const worldgen::SurfaceSample& surface, int32_t surfaceY,
                             int32_t worldY) noexcept {
    const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(surface);
    const bool explicitFallOwner =
        surface.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL &&
        surface.hydrology.transitionOwnerId != 0;
    const bool explicitFallingLip =
        explicitFallOwner &&
        (surface.hydrology.generatedFluidLevel == 7 ||
         surface.waterSurface <= surface.hydrology.waterfallBottom + 0.125001);
    const bool waterfallOverlay =
        explicitFallingLip && surface.hydrology.waterfall &&
        surface.hydrology.waterfallTop >= surface.hydrology.waterfallBottom + 0.5 &&
        surface.hydrology.waterfallTop >= surface.waterSurface - 0.125;
    if (waterfallOverlay) {
        const int32_t waterfallBottomY =
            std::clamp(static_cast<int32_t>(std::ceil(surface.hydrology.waterfallBottom)) - 1,
                       WORLD_MIN_Y, WORLD_MAX_Y);
        const int32_t waterfallTopY =
            std::clamp(static_cast<int32_t>(std::ceil(surface.hydrology.waterfallTop)) - 1,
                       WORLD_MIN_Y, WORLD_MAX_Y);
        const int32_t fallingStartY = std::max(fluid.fallingStartY, waterfallBottomY);
        if (worldY >= fallingStartY && worldY <= waterfallTopY) {
            return worldY == waterfallTopY ? FluidState::flowing(7) : FluidState::falling(7);
        }
    }
    if (!fluid.wet || worldY <= surfaceY || worldY > fluid.topY) return std::nullopt;
    return worldY == fluid.topY ? fluid.topState : FluidState::source();
}

} // namespace

World::World(uint64_t seed, int viewDistance, size_t loadedCubeLimit, GenerationSettings generation)
    : World(seed, viewDistance, loadedCubeLimit, nullptr, generation) {}

World::World(uint64_t seed, int viewDistance, size_t loadedCubeLimit,
             std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
             GenerationSettings generation)
    : seed_(seed)
    , generation_(generation)
    , instanceId_(allocateWorldInstanceId())
    , generationContext_(std::move(generationContext))
    , viewDistance_(
          std::clamp(viewDistance, MIN_RENDER_DISTANCE_CHUNKS, MAX_RENDER_DISTANCE_CHUNKS))
    , loadedCubeLimit_(std::clamp<size_t>(loadedCubeLimit, 1, MAX_LOADED_CUBES))
    , generator_(seed, generationContext_, generation) {
    std::atomic_store_explicit(&loadedSnapshot_,
                               std::make_shared<const std::vector<std::shared_ptr<Chunk>>>(),
                               std::memory_order_release);
    std::atomic_store_explicit(&meshCandidateSnapshot_,
                               std::make_shared<const std::unordered_set<ChunkPos>>(),
                               std::memory_order_release);
    std::atomic_store_explicit(&exactSurfaceCoverageSnapshot_,
                               std::make_shared<const ExactSurfaceCoverageSnapshot>(),
                               std::memory_order_release);
    std::atomic_store_explicit(&exactCollisionOwnershipSnapshot_,
                               std::make_shared<const ExactCollisionOwnershipSnapshot>(),
                               std::memory_order_release);
    publicationLightQueue_.reserve(loadedCubeLimit_);
}

void World::setSavedChunkProjection(SavedChunkProjection projection) {
    if (!projection) throw std::invalid_argument("Saved chunk projection is incomplete");
    std::lock_guard lock(chunksMutex_);
    if (!chunks_.empty() || loadedCubeCount_.load(std::memory_order_relaxed) != 0) {
        throw std::logic_error("Saved chunk projection must be installed before streaming");
    }
    savedChunkProjection_ = std::move(projection);
}

uint64_t World::applySavedChunkProjection(Chunk& chunk) const {
    if (savedChunkProjection_) return savedChunkProjection_.apply(chunk);
    normalizePersistedFurnaceVisuals(chunk);
    return 0;
}

bool World::savedChunkProjectionIsCurrent(uint64_t revision) const {
    return !savedChunkProjection_ || savedChunkProjection_.currentRevision() == revision;
}

void sortChunksByDistance(std::vector<ChunkPos>& chunks, int64_t centerChunkX, int32_t centerChunkY,
                          int64_t centerChunkZ) {
    std::sort(chunks.begin(), chunks.end(), [&](ChunkPos a, ChunkPos b) {
        return distanceSq(a, centerChunkX, centerChunkY, centerChunkZ) >
               distanceSq(b, centerChunkX, centerChunkY, centerChunkZ);
    });
}

std::unordered_set<ChunkPos>
selectStableMeshCandidates(const std::unordered_map<ChunkPos, uint8_t>& candidatePriorities,
                           const std::unordered_set<ChunkPos>& previousCandidates, ChunkPos center,
                           size_t capacity) {
    struct RankedCandidate {
        ChunkPos position;
        uint8_t priority = 0;
        bool previouslyResident = false;
        uint64_t horizontalDistanceSquared = 0;
        uint64_t verticalDistanceSquared = 0;
        double stableHorizontalDistance = 0.0;
    };

    std::vector<RankedCandidate> ranked;
    ranked.reserve(candidatePriorities.size());
    for (const auto& [position, priority] : candidatePriorities) {
        const bool previouslyResident = previousCandidates.contains(position);
        const int64_t dx = position.x - center.x;
        const int64_t dy = static_cast<int64_t>(position.y) - center.y;
        const int64_t dz = position.z - center.z;
        const uint64_t horizontalDistanceSquared =
            static_cast<uint64_t>(dx * dx + dz * dz);
        const uint64_t verticalDistanceSquared = static_cast<uint64_t>(dy * dy);
        const double horizontalDistance =
            std::sqrt(static_cast<double>(horizontalDistanceSquared));
        ranked.push_back({
            .position = position,
            .priority = priority,
            .previouslyResident = previouslyResident,
            .horizontalDistanceSquared = horizontalDistanceSquared,
            .verticalDistanceSquared = verticalDistanceSquared,
            .stableHorizontalDistance =
                std::max(0.0, horizontalDistance -
                                  (previouslyResident
                                       ? EXACT_MESH_RESIDENCY_HYSTERESIS_CHUNKS
                                       : 0.0)),
        });
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedCandidate& left, const RankedCandidate& right) {
                  if (left.priority != right.priority) return left.priority > right.priority;
                  // Exact surface ownership is a horizontal contract. Keep
                  // every required vertical section in a nearer column ahead
                  // of a flatter but farther column so capacity pressure cannot
                  // strand the near column behind its coarse far parent.
                  if (left.stableHorizontalDistance != right.stableHorizontalDistance) {
                      return left.stableHorizontalDistance < right.stableHorizontalDistance;
                  }
                  if (left.previouslyResident != right.previouslyResident)
                      return left.previouslyResident;
                  if (left.horizontalDistanceSquared != right.horizontalDistanceSquared) {
                      return left.horizontalDistanceSquared < right.horizontalDistanceSquared;
                  }
                  if (left.verticalDistanceSquared != right.verticalDistanceSquared) {
                      return left.verticalDistanceSquared < right.verticalDistanceSquared;
                  }
                  if (left.position.x != right.position.x)
                      return left.position.x < right.position.x;
                  if (left.position.z != right.position.z)
                      return left.position.z < right.position.z;
                  return left.position.y < right.position.y;
              });
    if (ranked.size() > capacity) ranked.resize(capacity);

    std::unordered_set<ChunkPos> selected;
    selected.reserve(ranked.size());
    for (const RankedCandidate& candidate : ranked)
        selected.insert(candidate.position);
    return selected;
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
    if (genPool_) {
        genPool_->shutdown();
        genPool_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        // Every accepted generation task captures this World. ThreadPool::shutdown
        // drains those tasks and joins all workers before this lock is acquired, so
        // no pending-state writer can remain. A completed task's ready future may
        // still be present because the final worker cannot reap its own future.
        // Destroy those shared states without get() so teardown remains nonthrowing.
        const auto allAcceptedTasksCompleted = [](const auto& pending) {
            return std::ranges::all_of(pending, [](const auto& entry) {
                return !entry.second.future.valid() ||
                       entry.second.future.wait_for(std::chrono::seconds(0)) ==
                           std::future_status::ready;
            });
        };
        [[maybe_unused]] const bool acceptedTasksCompleted =
            allAcceptedTasksCompleted(pendingGenerations_) &&
            allAcceptedTasksCompleted(pendingColumnPlans_);
        assert(acceptedTasksCompleted);
        // In-flight sets and counters are admission reservations. They may outlive
        // a follow-on submission rejected after shutdown began, but no accepted
        // task remains after the future invariant above and the joined pool.
        activeColumnPlanJobs_ = 0;
        pendingGenerations_.clear();
        generationsInFlight_.clear();
        pendingColumnPlans_.clear();
        columnPlansInFlight_.clear();
        columnPlanRetries_.clear();
        genBacklog_ = decltype(genBacklog_){};
        genBacklogSet_.clear();
        columnPlanBacklog_.clear();
        planDependents_.clear();
        missingPlanDependencies_.clear();
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

std::optional<std::string> World::generationFailure() const {
    if (generationContext_) {
        const std::optional<worldgen::learned::GenerationFailure> failure =
            generationContext_->failure();
        if (failure) {
            return failure->message.empty()
                       ? std::optional<std::string>{"Generator v4 authority failed"}
                       : std::optional<std::string>{failure->message};
        }
    }
    {
        std::lock_guard lock(generationFailureMutex_);
        if (generationFailure_) return generationFailure_;
    }
    return std::nullopt;
}

bool World::retryGeneration() {
    if (generationContext_) {
        const std::optional<worldgen::learned::GenerationFailure> failure =
            generationContext_->failure();
        if (failure && !generationContext_->clearRetriableFailure()) return false;
    }
    {
        std::lock_guard lock(generationFailureMutex_);
        generationFailure_.reset();
    }
    if (hasPlayerChunk_) {
        columnPlansChanged_.store(false, std::memory_order_release);
        requestActiveSetRebuild(playerChunkX_ * CHUNK_EDGE, playerChunkY_ * CHUNK_EDGE,
                                playerChunkZ_ * CHUNK_EDGE);
    }
    return true;
}

bool World::exactSpawnBandReady(int64_t worldX, int32_t worldY, int64_t worldZ,
                                int radiusChunks) const {
    const int radius = boundedColdStartExactRadiusChunks(radiusChunks);
    const int64_t centerX = Chunk::worldToChunk(worldX);
    const int64_t centerZ = Chunk::worldToChunk(worldZ);
    const int32_t playerSection = Chunk::worldToChunkY(worldY);

    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (!withinExactStreamingRadius(dx, dz, radius)) continue;
            const ColumnPos column{centerX + dx, centerZ + dz};
            const std::shared_ptr<const ColumnPlan> plan = generator_.findColumnPlan(column);
            if (!plan) return false;
            for (const int32_t sectionY : plan->exposedSections()) {
                if (!isChunkLoaded({column.x, sectionY, column.z})) return false;
            }
            for (int offset = -1; offset <= 1; ++offset) {
                const int32_t sectionY = playerSection + offset;
                if (sectionY < WORLD_MIN_CHUNK_Y || sectionY > WORLD_MAX_CHUNK_Y) continue;
                if (!isChunkLoaded({column.x, sectionY, column.z})) return false;
            }
        }
    }
    return true;
}

bool World::playableSpawnCollisionReady(int64_t worldX, int32_t worldY, int64_t worldZ) const {
    const int64_t centerX = Chunk::worldToChunk(worldX);
    const int32_t centerY = Chunk::worldToChunkY(worldY);
    const int64_t centerZ = Chunk::worldToChunk(worldZ);

    std::lock_guard<std::mutex> lock(chunksMutex_);
    for (int offsetZ = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
         offsetZ <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetZ) {
        for (int offsetX = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
             offsetX <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetX) {
            for (int offsetY = -PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES;
                 offsetY <= PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES; ++offsetY) {
                const ChunkPos position{centerX + offsetX, centerY + offsetY,
                                        centerZ + offsetZ};
                if (!validChunkY(position.y)) continue;
                const auto found = chunks_.find(position);
                if (found == chunks_.end() || !found->second || !found->second->generated) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::optional<Vec3> World::safeSpawnFromReadyPlans(int64_t worldX, int64_t worldZ,
                                                   int radiusChunks) const {
    const int radius = boundedColdStartExactRadiusChunks(radiusChunks);
    const int64_t centerChunkX = Chunk::worldToChunk(worldX);
    const int64_t centerChunkZ = Chunk::worldToChunk(worldZ);
    const int64_t minimumX = (centerChunkX - radius) * CHUNK_EDGE;
    const int64_t minimumZ = (centerChunkZ - radius) * CHUNK_EDGE;
    const int64_t maximumX = (centerChunkX + radius + 1) * CHUNK_EDGE - 1;
    const int64_t maximumZ = (centerChunkZ + radius + 1) * CHUNK_EDGE - 1;

    struct Candidate {
        Vec3 position{};
        double score = std::numeric_limits<double>::infinity();
        int64_t x = 0;
        int64_t z = 0;
    };
    std::optional<Candidate> best;
    const auto drySurfaceAt = [this](int64_t x, int64_t z, double referenceHeight) -> bool {
        const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
        const std::shared_ptr<const ColumnPlan> plan = generator_.findColumnPlan(column);
        if (!plan) return false;
        const worldgen::SurfaceSample neighbor =
            plan->sample(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
        if (!std::isfinite(neighbor.terrainHeight) || !std::isfinite(neighbor.waterSurface) ||
            !std::isfinite(neighbor.slope)) {
            return false;
        }
        const bool waterContact =
            neighbor.hydrology.ocean || neighbor.hydrology.lake || neighbor.hydrology.river ||
            neighbor.hydrology.waterfall || neighbor.hydrology.wetland ||
            neighbor.hydrology.delta ||
            neighbor.hydrology.transitionOwnerKind != worldgen::WaterTransitionKind::NONE ||
            neighbor.waterSurface > neighbor.terrainHeight + 0.01;
        return !waterContact && neighbor.slope <= 0.9 &&
               std::abs(neighbor.terrainHeight - referenceHeight) <= 2.0;
    };
    const auto consider = [&](int64_t candidateX, int64_t candidateZ) {
        const ColumnPos column{Chunk::worldToChunk(candidateX), Chunk::worldToChunk(candidateZ)};
        const int offsetX = static_cast<int>(column.x - centerChunkX);
        const int offsetZ = static_cast<int>(column.z - centerChunkZ);
        if (!withinExactStreamingRadius(offsetX, offsetZ, radius)) return;
        const std::shared_ptr<const ColumnPlan> plan = generator_.findColumnPlan(column);
        if (!plan) return;
        const worldgen::SurfaceSample sample =
            plan->sample(Chunk::worldToLocal(candidateX), Chunk::worldToLocal(candidateZ));
        if (!std::isfinite(sample.terrainHeight) || !std::isfinite(sample.waterSurface) ||
            !std::isfinite(sample.slope)) {
            return;
        }
        const bool waterContact =
            sample.hydrology.ocean || sample.hydrology.lake || sample.hydrology.river ||
            sample.hydrology.waterfall || sample.hydrology.wetland || sample.hydrology.delta ||
            sample.hydrology.transitionOwnerKind != worldgen::WaterTransitionKind::NONE ||
            sample.waterSurface > sample.terrainHeight + 0.01;
        if (waterContact || sample.slope > 0.9) return;
        const int32_t feetBlock = static_cast<int32_t>(std::ceil(sample.terrainHeight));
        if (feetBlock < WORLD_MIN_Y + 1 || feetBlock + 1 > WORLD_MAX_Y) return;
        const std::optional<BlockType> support =
            findBlockIfLoaded(candidateX, feetBlock - 1, candidateZ);
        const std::optional<BlockType> feet = findBlockIfLoaded(candidateX, feetBlock, candidateZ);
        const std::optional<BlockType> head =
            findBlockIfLoaded(candidateX, feetBlock + 1, candidateZ);
        const auto breathable = [](BlockType block) { return !isSolid(block) && !isLiquid(block); };
        if (!support || !feet || !head || !isSolid(*support) || !breathable(*feet) ||
            !breathable(*head)) {
            return;
        }
        for (int dz = -2; dz <= 2; dz += 2) {
            for (int dx = -2; dx <= 2; dx += 2) {
                if (!drySurfaceAt(candidateX + dx, candidateZ + dz, sample.terrainHeight)) return;
            }
        }

        const double dx = static_cast<double>(candidateX - worldX);
        const double dz = static_cast<double>(candidateZ - worldZ);
        double score = dx * dx + dz * dz + std::clamp(sample.slope, 0.0, 4.0) * 256.0;
        if (sample.geology.volcanicActivity > 0.75) score += 16'384.0;
        if (!best || score < best->score ||
            (score == best->score &&
             std::pair{candidateX, candidateZ} < std::pair{best->x, best->z})) {
            best = Candidate{
                .position = {static_cast<float>(candidateX) + 0.5F,
                             static_cast<float>(feetBlock) + 0.05F,
                             static_cast<float>(candidateZ) + 0.5F},
                .score = score,
                .x = candidateX,
                .z = candidateZ,
            };
        }
    };

    for (int64_t z = minimumZ; z <= maximumZ; z += 4) {
        for (int64_t x = minimumX; x <= maximumX; x += 4)
            consider(x, z);
    }
    if (!best) return std::nullopt;

    const int64_t refineMinimumX = std::max(minimumX, best->x - 4);
    const int64_t refineMaximumX = std::min(maximumX, best->x + 4);
    const int64_t refineMinimumZ = std::max(minimumZ, best->z - 4);
    const int64_t refineMaximumZ = std::min(maximumZ, best->z + 4);
    for (int64_t z = refineMinimumZ; z <= refineMaximumZ; ++z) {
        for (int64_t x = refineMinimumX; x <= refineMaximumX; ++x)
            consider(x, z);
    }
    return best->position;
}

void World::latchGenerationFailure(std::string message) {
    latchGenerationFailure({
        .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
        .message = std::move(message),
        .retriable = true,
    });
}

void World::latchGenerationFailure(worldgen::learned::GenerationFailure failure) {
    if (generationContext_) {
        generationContext_->latchFailure(std::move(failure));
        return;
    }
    std::lock_guard lock(generationFailureMutex_);
    if (!generationFailure_) {
        generationFailure_ =
            failure.message.empty() ? "World generation failed" : std::move(failure.message);
    }
}

void World::generateChunk(const std::shared_ptr<Chunk>& chunk) {
    generator_.generateCube(*chunk);
}

void World::generateChunkAsync(ChunkPos pos, int64_t priority) {
    if (shuttingDown_.load() || generationFailure().has_value()) return;
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

    ThreadPool* const pool = genPool_.get();
    auto work = [this, pos]() {
        bool stale = false;
        {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            stale = !retainedChunks_.contains(pos) || shuttingDown_.load();
        }
        if (stale) {
            pumpGeneration();
            return;
        }
        bool loadedFromSave = false;
        uint64_t projectionRevision = 0;
        auto chunk = loadOrGenerateChunk(pos, &loadedFromSave, &projectionRevision);
        if (!chunk) {
            pumpGeneration();
            return;
        }
        // Authority lookup remains outside chunksMutex_. Publication lighting
        // then sees immutable plan pointers for every resident cube it can
        // reach and never waits on generation while holding the world lock.
        const LightColumnPlans lightPlans = findLightColumnPlans(pos);
        bool inserted = false;
        constexpr int MAX_PROJECTION_RETRIES = 4;
        for (int attempt = 0; attempt < MAX_PROJECTION_RETRIES; ++attempt) {
            bool retryProjection = false;
            {
                std::lock_guard<std::mutex> lock(chunksMutex_);
                if (chunks_.contains(pos)) {
                    break;
                }
                if (loadedFromSave && !savedChunkProjectionIsCurrent(projectionRevision)) {
                    retryProjection = true;
                } else if (retainedChunks_.contains(pos) && !shuttingDown_.load() &&
                           chunks_.size() < loadedCubeLimit_) {
                    const auto [it, didInsert] = chunks_.try_emplace(pos, std::move(chunk));
                    inserted = didInsert;
                    if (inserted) {
                        VerticalSectionMask& columnMask = loadedSectionMasks_[{pos.x, pos.z}];
                        const VerticalSectionMask previousColumnMask = columnMask;
                        columnMask.set(pos.y);
                        const size_t loaded =
                            loadedCubeCount_.fetch_add(1, std::memory_order_relaxed) + 1;
                        recordLoadedCubeHighWater(loadedCubeHighWater_, loaded);
                        loadedSnapshotDirty_.store(true, std::memory_order_release);
                        SkyCutoffSectionRange skyCutoffChange =
                            loadedFromSave ? refreshSavedSkyCutoffsLocked(pos)
                                           : extendGeneratedSkyCutoffsLocked(*it->second);
                        if (loweredSkyCutoffColumns_.contains({pos.x, pos.z})) {
                            skyCutoffChange.merge(refreshSkyOverrideColumnLocked({pos.x, pos.z}));
                        }
                        markHaloNeighborMeshesDirtyLocked(pos);
                        const LightEngine::FloodResult initialFlood =
                            initializeChunkLightLocked(pos, lightPlans[4].get());
                        const size_t followupFloods = settleChunkPublicationLightLocked(
                            pos, initialFlood, previousColumnMask, lightPlans, skyCutoffChange,
                            PUBLICATION_LIGHT_SYNC_FLOOD_CAP - 1);
                        publicationLightSyncFloods_.fetch_add(1 + followupFloods,
                                                              std::memory_order_relaxed);
                        recordLoadedCubeHighWater(publicationLightMaxSyncFloods_,
                                                  1 + followupFloods);
                    }
                } else if (retainedChunks_.contains(pos) && !shuttingDown_.load()) {
                    loadedCubeAdmissionsRejected_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!retryProjection) break;
            projectionRevision = applySavedChunkProjection(*chunk);
        }
        if (inserted) {
            queueFluidResume(pos);
        }
        pumpGeneration();
    };

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingGenerations_.contains(pos)) return;
    if (const auto lane = exactPriorityByCube_.find(pos); lane != exactPriorityByCube_.end()) {
        const uint64_t squared = exactStreamingCubePriorityDistance(pos, exactPriorityCenter_);
        priority = exactStreamingTaskPriority(activeSetEpoch_, lane->second,
                                              squared);
    }
    ThreadPool::TaskHandle handle;
    std::future<void> future =
        pool->submitTrackedWithPriority(priority, handle, std::move(work));
    pendingGenerations_[pos] = {std::move(future), handle, priority};
}

void World::generateColumnPlanAsync(ColumnPos pos, int64_t priority) {
    ThreadPool* const pool = genPool_.get();
    auto work = [this, pos]() {
        bool requiredAtStart = false;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            requiredAtStart = exactPriorityByPlan_.contains(pos) && !shuttingDown_.load();
        }
        bool planAvailable = false;
        bool authorityDeferred = false;
        if (requiredAtStart) {
            try {
                planAvailable = generator_.getColumnPlan(pos) != nullptr;
            } catch (const worldgen::learned::GenerationFailureException& error) {
                if (error.status() == worldgen::learned::AuthorityStatus::DEFERRED) {
                    authorityDeferred = true;
                } else {
                    RY_LOG_ERROR(
                        (std::string("Column plan generation failed: ") + error.what()).c_str());
                    latchGenerationFailure(error.failure());
                }
            } catch (const std::exception& error) {
                RY_LOG_ERROR(
                    (std::string("Column plan generation failed: ") + error.what()).c_str());
                latchGenerationFailure(error.what());
            } catch (...) {
                constexpr const char* message =
                    "Column plan generation failed with an unknown exception";
                RY_LOG_ERROR(message);
                latchGenerationFailure(message);
            }
        }
        bool notifyActiveSet = false;
        size_t fullRetainedScanEquivalent = 0;
        ColumnPlanCompletionAction completionAction = ColumnPlanCompletionAction::DROP;
        {
            std::lock_guard<std::mutex> lock(pendingMutex_);
            completionAction = columnPlanCompletionAction(
                planAvailable, shuttingDown_.load(), exactPriorityByPlan_.contains(pos),
                columnPlanRetries_.contains(pos));
            if (completionAction == ColumnPlanCompletionAction::PUBLISH) {
                ++completedPlansSinceRebuild_;
            } else if (completionAction == ColumnPlanCompletionAction::REQUEUE) {
                // Keep the retry private until pumpGeneration observes this
                // task's ready future. Releasing the slot or publishing the
                // backlog entry from inside the worker creates a window where
                // another pump reserves a duplicate that the old future then
                // rejects, leaking the reservation forever.
                columnPlanRetries_.insert(pos);
            }
            if (completedPlansSinceRebuild_ >= COLUMN_PLAN_REBUILD_BATCH) {
                completedPlansSinceRebuild_ = 0;
                notifyActiveSet = true;
            }
            fullRetainedScanEquivalent = retainedCubeCountForStats_;
        }
        if (completionAction == ColumnPlanCompletionAction::PUBLISH) {
            completedColumnPlans_.fetch_add(1, std::memory_order_relaxed);
            fullRetainedScanEquivalent_.fetch_add(fullRetainedScanEquivalent,
                                                  std::memory_order_relaxed);
            wakePlanDependents(pos);
        }
        if (notifyActiveSet) {
            columnPlansChanged_.store(true, std::memory_order_release);
            activeSetRebuildNotifications_.fetch_add(1, std::memory_order_relaxed);
        }
        // A deferred learned page stays owned by its dedicated coordinator.
        // The next gameplay or preparation pump reaps this future, releases
        // the worker slot, and publishes the retry without spinning.
        if (!authorityDeferred) pumpGeneration();
    };

    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingColumnPlans_.contains(pos)) return;
    if (const auto lane = exactPriorityByPlan_.find(pos); lane != exactPriorityByPlan_.end()) {
        const int64_t dx = pos.x - exactPriorityCenter_.x;
        const int64_t dz = pos.z - exactPriorityCenter_.z;
        priority = exactStreamingTaskPriority(activeSetEpoch_, lane->second,
                                              static_cast<uint64_t>(dx * dx + dz * dz));
    }
    ThreadPool::TaskHandle handle;
    std::future<void> future =
        pool->submitTrackedWithPriority(priority, handle, std::move(work));
    pendingColumnPlans_[pos] = {std::move(future), handle, priority};
}

std::shared_ptr<Chunk> World::loadOrGenerateChunk(ChunkPos pos, bool* loadedFromSave,
                                                  uint64_t* projectionRevision) {
    if (loadedFromSave) *loadedFromSave = false;
    if (projectionRevision) *projectionRevision = 0;
    if (saveManager_) {
        if (auto loaded = saveManager_->loadChunk(pos)) {
            const uint64_t appliedRevision = applySavedChunkProjection(*loaded);
            if (loadedFromSave) *loadedFromSave = true;
            if (projectionRevision) *projectionRevision = appliedRevision;
            return std::make_shared<Chunk>(std::move(*loaded));
        }
    }

    if (generationFailure().has_value()) return nullptr;

    try {
        auto chunk = std::make_shared<Chunk>(pos);
        const auto start = std::chrono::steady_clock::now();
        generateChunk(chunk);
        genMs_.record(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
        return chunk;
    } catch (const worldgen::learned::GenerationFailureException& error) {
        if (error.status() == worldgen::learned::AuthorityStatus::DEFERRED) return nullptr;
        const std::string message = std::string("Cube generation failed: ") + error.what();
        worldgen::learned::GenerationFailure failure = error.failure();
        if (failure.message.empty()) failure.message = message;
        latchGenerationFailure(std::move(failure));
        RY_LOG_ERROR(message.c_str());
        return nullptr;
    } catch (const std::exception& error) {
        const std::string message = std::string("Cube generation failed: ") + error.what();
        latchGenerationFailure(message);
        RY_LOG_ERROR(message.c_str());
        return nullptr;
    } catch (...) {
        constexpr const char* message = "Cube generation failed with an unknown exception";
        latchGenerationFailure(message);
        RY_LOG_ERROR(message);
        return nullptr;
    }
}

std::shared_ptr<Chunk> World::getChunk(ChunkPos pos) {
    if (!validChunkY(pos.y)) return nullptr;
    ensureSavedSkyAuthority({pos.x, pos.z});
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto it = chunks_.find(pos);
        if (it != chunks_.end()) return it->second;
        if (chunks_.size() >= loadedCubeLimit_) {
            loadedCubeAdmissionsRejected_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    bool loadedFromSave = false;
    uint64_t projectionRevision = 0;
    auto chunk = loadOrGenerateChunk(pos, &loadedFromSave, &projectionRevision);
    if (!chunk) return nullptr;
    const LightColumnPlans lightPlans = findLightColumnPlans(pos);
    std::shared_ptr<Chunk> result;
    bool inserted = false;
    constexpr int MAX_PROJECTION_RETRIES = 4;
    for (int attempt = 0; attempt < MAX_PROJECTION_RETRIES; ++attempt) {
        bool retryProjection = false;
        {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            if (const auto existing = chunks_.find(pos); existing != chunks_.end()) {
                result = existing->second;
            } else if (loadedFromSave && !savedChunkProjectionIsCurrent(projectionRevision)) {
                retryProjection = true;
            } else if (chunks_.size() >= loadedCubeLimit_) {
                loadedCubeAdmissionsRejected_.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            } else {
                const auto [it, didInsert] = chunks_.try_emplace(pos, std::move(chunk));
                inserted = didInsert;
                if (inserted) {
                    VerticalSectionMask& columnMask = loadedSectionMasks_[{pos.x, pos.z}];
                    const VerticalSectionMask previousColumnMask = columnMask;
                    columnMask.set(pos.y);
                    const size_t loaded =
                        loadedCubeCount_.fetch_add(1, std::memory_order_relaxed) + 1;
                    recordLoadedCubeHighWater(loadedCubeHighWater_, loaded);
                    loadedSnapshotDirty_.store(true, std::memory_order_release);
                    SkyCutoffSectionRange skyCutoffChange =
                        loadedFromSave ? refreshSavedSkyCutoffsLocked(pos)
                                       : extendGeneratedSkyCutoffsLocked(*it->second);
                    if (loweredSkyCutoffColumns_.contains({pos.x, pos.z})) {
                        skyCutoffChange.merge(refreshSkyOverrideColumnLocked({pos.x, pos.z}));
                    }
                    markHaloNeighborMeshesDirtyLocked(pos);
                    const LightEngine::FloodResult initialFlood =
                        initializeChunkLightLocked(pos, lightPlans[4].get());
                    const size_t followupFloods = settleChunkPublicationLightLocked(
                        pos, initialFlood, previousColumnMask, lightPlans, skyCutoffChange,
                        PUBLICATION_LIGHT_SYNC_FLOOD_CAP - 1);
                    publicationLightSyncFloods_.fetch_add(1 + followupFloods,
                                                          std::memory_order_relaxed);
                    recordLoadedCubeHighWater(publicationLightMaxSyncFloods_, 1 + followupFloods);
                }
                result = it->second;
            }
        }
        if (!retryProjection) break;
        projectionRevision = applySavedChunkProjection(*chunk);
    }
    if (!result) return nullptr;
    if (inserted) {
        queueFluidResume(pos);
    }
    return result;
}

BlockType World::getBlock(int64_t x, int32_t y, int64_t z) {
    if (y < WORLD_MIN_Y) return BlockType::BEDROCK;
    if (y > WORLD_MAX_Y) return BlockType::AIR;
    const ChunkPos pos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)};
    const std::shared_ptr<Chunk> chunk = getChunk(pos);
    return chunk ? chunk->getBlockWorld(x, y, z) : BlockType::STONE;
}

BlockType World::getBlockIfLoaded(int64_t x, int32_t y, int64_t z) const {
    return findBlockIfLoaded(x, y, z).value_or(BlockType::AIR);
}

uint8_t World::getPackedLightIfLoaded(int64_t x, int32_t y, int64_t z) const {
    if (y < WORLD_MIN_Y || y > WORLD_MAX_Y) return 0;
    const ChunkPos pos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)};
    std::lock_guard<std::mutex> lock(chunksMutex_);
    const auto found = chunks_.find(pos);
    if (found == chunks_.end() || !found->second->generated) return 0;
    return found->second->getPackedLight(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                         Chunk::worldToLocal(z));
}

void World::samplePackedLightsIfLoaded(std::span<const BlockPos> positions,
                                       std::span<uint8_t> output) const {
    std::fill(output.begin(), output.end(), uint8_t{0});
    const size_t count = std::min(positions.size(), output.size());
    if (count == 0) return;
    std::lock_guard<std::mutex> lock(chunksMutex_);
    for (size_t index = 0; index < count; ++index) {
        const BlockPos position = positions[index];
        if (position.y < WORLD_MIN_Y || position.y > WORLD_MAX_Y) continue;
        const ChunkPos chunkPosition{Chunk::worldToChunk(position.x),
                                     Chunk::worldToChunkY(position.y),
                                     Chunk::worldToChunk(position.z)};
        const auto found = chunks_.find(chunkPosition);
        if (found == chunks_.end() || !found->second->generated) continue;
        output[index] = found->second->getPackedLight(Chunk::worldToLocal(position.x),
                                                      Chunk::worldToLocalY(position.y),
                                                      Chunk::worldToLocal(position.z));
    }
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
    return loadedSurfaceHeight(chunks_, loadedSectionMasks_, x, z);
}

std::optional<float> World::strikeSurfaceHeightIfLoaded(int64_t x, int64_t z) const {
    std::lock_guard<std::mutex> lock(chunksMutex_);
    return loadedStrikeSurfaceHeight(chunks_, loadedSectionMasks_, x, z);
}

BlockType World::getCollisionBlockIfLoaded(int64_t x, int32_t y, int64_t z) const {
    if (y < WORLD_MIN_Y) return BlockType::BEDROCK;
    if (y > WORLD_MAX_Y) return BlockType::AIR;

    const ChunkPos section{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                           Chunk::worldToChunk(z)};
    const ColumnPos column{section.x, section.z};
    const std::shared_ptr<const ExactSurfaceCoverageSnapshot> coverage =
        getExactSurfaceCoverageSnapshot();
    const std::shared_ptr<const ExactCollisionOwnershipSnapshot> collision =
        getExactCollisionOwnershipSnapshot();
    const bool exactOwner = coverage && collision && collision->coverageEpoch == coverage->epoch &&
                            collision->owns(section);
    if (exactOwner) {
        return findBlockIfLoaded(x, y, z).value_or(BlockType::BEDROCK);
    }

    const std::shared_ptr<const ColumnPlan> plan = generator_.findColumnPlan(column);
    if (!plan) return BlockType::BEDROCK;
    const int localX = Chunk::worldToLocal(x);
    const int localZ = Chunk::worldToLocal(z);
    const int surfaceY = plan->surfaceY(localX, localZ);
    const worldgen::SurfaceSample surface = plan->sample(localX, localZ);
    if (canonicalCollisionFluidState(surface, surfaceY, y)) return BlockType::WATER;
    if (y <= surfaceY) return BlockType::STONE;
    return BlockType::AIR;
}

float World::getCollisionFluidHeightIfLoaded(int64_t x, int32_t y, int64_t z) const {
    if (y < WORLD_MIN_Y || y > WORLD_MAX_Y) return 0.0F;

    const ChunkPos section{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                           Chunk::worldToChunk(z)};
    const std::shared_ptr<const ExactSurfaceCoverageSnapshot> coverage =
        getExactSurfaceCoverageSnapshot();
    const std::shared_ptr<const ExactCollisionOwnershipSnapshot> collision =
        getExactCollisionOwnershipSnapshot();
    const bool exactOwner = coverage && collision && collision->coverageEpoch == coverage->epoch &&
                            collision->owns(section);
    if (exactOwner) {
        const FluidCell cell = readFluidCell({x, y, z});
        if (!cell.isWater()) return 0.0F;
        return getCollisionBlockIfLoaded(x, y + 1, z) == BlockType::WATER
                   ? 1.0F
                   : fluidSurfaceHeight(cell.state);
    }

    const ColumnPos column{section.x, section.z};
    const std::shared_ptr<const ColumnPlan> plan = generator_.findColumnPlan(column);
    if (!plan) return 0.0F;
    const int localX = Chunk::worldToLocal(x);
    const int localZ = Chunk::worldToLocal(z);
    const worldgen::SurfaceSample surface = plan->sample(localX, localZ);
    const std::optional<FluidState> state =
        canonicalCollisionFluidState(surface, plan->surfaceY(localX, localZ), y);
    if (!state) return 0.0F;
    return getCollisionBlockIfLoaded(x, y + 1, z) == BlockType::WATER
               ? 1.0F
               : fluidSurfaceHeight(*state);
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

bool World::trySetBlock(int64_t x, int32_t y, int64_t z, BlockType type) {
    if (y < WORLD_MIN_Y || y > WORLD_MAX_Y) return false;
    const BlockPos position{x, y, z};
    const FluidCell existing = readFluidCell(position);
    if (!existing.loaded) return false;
    const FluidState requestedFluid = FluidState::source();
    if (existing.block == type && (type != BlockType::WATER || existing.state == requestedFluid)) {
        // Generated water is immutable static authority until gameplay makes
        // a real edit. A no-op placement must not awaken the mutable fluid
        // scheduler and let it reinterpret a canonical river or lake.
        return false;
    }
    if (!setBlockLoaded(position, type,
                        type == BlockType::WATER ? std::optional(requestedFluid) : std::nullopt,
                        LightUrgency::IMMEDIATE)) {
        return false;
    }
    fluidScheduler_.activateBlockChange(position);
    return true;
}

void World::setBlock(int64_t x, int32_t y, int64_t z, BlockType type) {
    static_cast<void>(trySetBlock(x, y, z, type));
}

bool World::setBlockLoaded(BlockPos position, BlockType type, std::optional<FluidState> fluidState,
                           LightUrgency urgency) {
    if (position.y < WORLD_MIN_Y || position.y > WORLD_MAX_Y) return false;
    const ChunkPos pos{Chunk::worldToChunk(position.x), Chunk::worldToChunkY(position.y),
                       Chunk::worldToChunk(position.z)};
    const int lx = Chunk::worldToLocal(position.x);
    const int ly = Chunk::worldToLocalY(position.y);
    const int lz = Chunk::worldToLocal(position.z);

    // The home cube floods first so its fresh borders seed neighbors, then the
    // rest of the affected neighborhood drains synchronously. The 27 cubes of
    // the 3x3x3 share nine XZ columns; prefetch those column plans before
    // chunksMutex_, matching the reconcileLight convention, so neither the
    // immediate floods nor the neighborhood drain calls the generator under the
    // lock. The plan for offset (dx, dz) lives at index (dx + 1) + (dz + 1) * 3.
    std::array<ChunkPos, 4> immediateCubes{};
    std::array<std::shared_ptr<const ColumnPlan>, 9> columnPlans{};
    size_t immediateCount = 0;
    const auto planForOffset = [&columnPlans, pos](ChunkPos cube) -> const ColumnPlan* {
        const int64_t dx = cube.x - pos.x;
        const int64_t dz = cube.z - pos.z;
        return columnPlans[static_cast<size_t>((dx + 1) + (dz + 1) * 3)].get();
    };
    if (urgency == LightUrgency::IMMEDIATE) {
        immediateCubes[immediateCount++] = pos;
        if (lx == 0) immediateCubes[immediateCount++] = {pos.x - 1, pos.y, pos.z};
        if (lx == CHUNK_EDGE - 1) immediateCubes[immediateCount++] = {pos.x + 1, pos.y, pos.z};
        if (lz == 0) immediateCubes[immediateCount++] = {pos.x, pos.y, pos.z - 1};
        if (lz == CHUNK_EDGE - 1) immediateCubes[immediateCount++] = {pos.x, pos.y, pos.z + 1};
        if (ly == 0 && validChunkY(pos.y - 1)) {
            immediateCubes[immediateCount++] = {pos.x, pos.y - 1, pos.z};
        }
        if (ly == CHUNK_EDGE - 1 && validChunkY(pos.y + 1)) {
            immediateCubes[immediateCount++] = {pos.x, pos.y + 1, pos.z};
        }
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                columnPlans[static_cast<size_t>((dx + 1) + (dz + 1) * 3)] =
                    generator_.findColumnPlan({pos.x + dx, pos.z + dz});
            }
        }
    }

    std::lock_guard<std::mutex> lock(chunksMutex_);
    auto it = chunks_.find(pos);
    if (it == chunks_.end() || !it->second->generated) return false;

    const BlockType oldBlock = it->second->getBlockWorld(position.x, position.y, position.z);
    const FluidState requestedFluid = fluidState.value_or(FluidState::source());
    const FluidState oldFluid = it->second->getFluidState(lx, ly, lz);
    if (oldBlock == type && (type != BlockType::WATER || oldFluid == requestedFluid)) {
        return false;
    }
    it->second->setBlockWorld(position.x, position.y, position.z, type);
    it->second->setFluidState(lx, ly, lz, requestedFluid);
    it->second->modifiedSinceSave = true;
    it->second->needsMeshUpdate = true;
    it->second->version.fetch_add(1, std::memory_order_relaxed);
    if (blockEditResetsIndirectLighting(oldBlock, type)) {
        lightingRevision_.fetch_add(1, std::memory_order_release);
    }

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
    const auto floodedSynchronously = [&](ChunkPos queued) {
        for (size_t cube = 0; cube < immediateCount; ++cube) {
            if (immediateCubes[cube] == queued) return true;
        }
        return false;
    };
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const ChunkPos queued{pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ};
                // Cubes the synchronous flood already fixed would only rerun
                // as no-diff floods; their changed borders requeue neighbors
                // on their own.
                if (floodedSynchronously(queued)) continue;
                queueLightReconcile(queued, urgency);
            }
        }
    }

    // Flooding here, under the same chunksMutex_ hold as the block write and
    // version bumps, guarantees no mesh snapshot can pair post-edit blocks
    // with pre-edit packed light. The home cube floods first so border
    // neighbors seed from its fresh levels, then the neighborhood drain
    // converges the rest before the edit returns, so adjacent cubes light and
    // remesh this tick instead of waiting for the next reconcile.
    for (size_t cube = 0; cube < immediateCount; ++cube) {
        reconcileCubeLocked(immediateCubes[cube], planForOffset(immediateCubes[cube]),
                            LightUrgency::IMMEDIATE);
    }
    if (urgency == LightUrgency::IMMEDIATE) {
        drainEditLightNeighborhoodLocked(pos, columnPlans);
    }
    return true;
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
    if (!plan) {
        const std::optional<int> top =
            loadedSurfaceHeight(chunks_, loadedSectionMasks_, worldX, worldZ);
        return updateOverride(
            top ? std::optional<int32_t>(std::clamp(*top + 1, WORLD_MIN_Y, WORLD_MAX_Y + 1))
                : std::nullopt);
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
    const bool editedColumn = skyOverrideChunkColumns_.contains(column);
    const auto latchLoweredAuthority = [&] {
        if (!editedColumn || !plannedSurfaceLoaded) return;
        if (loweredSkyCutoffColumns_.insert(column).second) {
            // The active-set worker will add the bounded contiguous support
            // hull on the next fixed-tick rebuild.
            columnPlansChanged_.store(true, std::memory_order_release);
        }
    };

    const std::optional<int> top =
        loadedSurfaceHeight(chunks_, loadedSectionMasks_, worldX, worldZ);
    if (!top) {
        latchLoweredAuthority();
        return updateOverride(std::nullopt);
    }
    const int32_t loadedCutoff = std::clamp(*top + 1, WORLD_MIN_Y, WORLD_MAX_Y + 1);

    // A partial vertical column cannot disprove immutable generated terrain
    // that has not loaded yet. It can still add a roof above that terrain. As
    // soon as the planned surface cube arrives, the loaded scan becomes
    // authoritative and can also represent a removed or lowered surface.
    if (!plannedSurfaceLoaded && loadedCutoff <= plannedCutoff) {
        return updateOverride(std::nullopt);
    }
    if (loadedCutoff < plannedCutoff) latchLoweredAuthority();
    if (loadedCutoff == plannedCutoff) {
        return updateOverride(std::nullopt);
    }
    return updateOverride(loadedCutoff);
}

World::SkyCutoffSectionRange World::refreshSkyOverrideColumnLocked(ColumnPos column) {
    SkyCutoffSectionRange changed;
    const auto plan = generator_.findColumnPlan(column);
    const int64_t baseX = column.x * CHUNK_EDGE;
    const int64_t baseZ = column.z * CHUNK_EDGE;
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const SkyColumnKey key{baseX + localX, baseZ + localZ};
            const auto prior = skyCutoffOverrides_.find(key);
            const int32_t plannedCutoff = plan ? plan->surfaceY(localX, localZ) + 1 : WORLD_MIN_Y;
            const int32_t previousCutoff =
                prior == skyCutoffOverrides_.end() ? plannedCutoff : prior->second;
            static_cast<void>(refreshSkyCutoffLocked(key.x, key.z));
            const auto current = skyCutoffOverrides_.find(key);
            const int32_t currentCutoff =
                current == skyCutoffOverrides_.end() ? plannedCutoff : current->second;
            changed.include(previousCutoff, currentCutoff);
        }
    }
    return changed;
}

void World::SkyCutoffSectionRange::include(int32_t oldCutoff, int32_t newCutoff) noexcept {
    if (oldCutoff == newCutoff) return;
    const auto sectionFor = [](int32_t cutoff) {
        const int32_t top = std::clamp(cutoff - 1, WORLD_MIN_Y, WORLD_MAX_Y);
        return Chunk::worldToChunkY(top);
    };
    first = std::min(first, std::min(sectionFor(oldCutoff), sectionFor(newCutoff)));
    last = std::max(last, std::max(sectionFor(oldCutoff), sectionFor(newCutoff)));
}

void World::SkyCutoffSectionRange::merge(const SkyCutoffSectionRange& other) noexcept {
    if (!other.changed()) return;
    first = std::min(first, other.first);
    last = std::max(last, other.last);
}

World::SkyCutoffSectionRange World::extendGeneratedSkyCutoffsLocked(const Chunk& chunk) {
    const ColumnPos column{chunk.chunkX, chunk.chunkZ};
    const auto plan = generator_.findColumnPlan(column);
    if (!plan) return {};
    const bool uniform = chunk.isUniform();
    if (uniform && !isOpaque(chunk.uniformBlock())) return {};

    SkyCutoffSectionRange changed;
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
                const int32_t previousCutoff = priorOverride->second;
                static_cast<void>(refreshSkyCutoffLocked(worldX, worldZ));
                const auto refreshed = skyCutoffOverrides_.find(key);
                const int32_t currentCutoff =
                    refreshed == skyCutoffOverrides_.end() ? plannedCutoff : refreshed->second;
                changed.include(previousCutoff, currentCutoff);
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
                const int32_t previousCutoff =
                    existing == skyCutoffOverrides_.end() ? plannedCutoff : existing->second;
                skyCutoffOverrides_.insert_or_assign(key, candidateCutoff);
                changed.include(previousCutoff, candidateCutoff);
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
        queueLightReconcile(found->first);
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
                queueLightReconcile(found->first);
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

void World::queueSavedSkyPublicationLocked(const std::unordered_set<ColumnPos>& changedColumns) {
    if (changedColumns.empty()) return;
    const auto queueAffected = [&](ChunkPos position, Chunk& chunk, bool ownsChangedAuthority) {
        // A known save manifest can open direct sky for an already resident
        // cube. Keep its complete mesh halo unpublished until the same bounded
        // transaction used by cube insertion has rebuilt packed light.
        chunk.needsMeshUpdate = true;
        chunk.version.fetch_add(1, std::memory_order_relaxed);
        if (ownsChangedAuthority) {
            queuePublicationLightLocked(position);
        } else {
            queueLightReconcile(position);
        }
    };
    if (changedColumns.size() == 1) {
        const ColumnPos changed = *changedColumns.begin();
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                for (int32_t chunkY = WORLD_MIN_CHUNK_Y; chunkY <= WORLD_MAX_CHUNK_Y; ++chunkY) {
                    const ChunkPos position{changed.x + offsetX, chunkY, changed.z + offsetZ};
                    const auto found = chunks_.find(position);
                    if (found == chunks_.end() || !found->second->generated) continue;
                    queueAffected(position, *found->second, offsetX == 0 && offsetZ == 0);
                }
            }
        }
        return;
    }

    for (const auto& [position, chunk] : chunks_) {
        if (!chunk || !chunk->generated) continue;
        const bool ownsChangedAuthority = changedColumns.contains({position.x, position.z});
        bool samplesChangedAuthority = ownsChangedAuthority;
        for (int offsetZ = -1; offsetZ <= 1 && !samplesChangedAuthority; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1 && !samplesChangedAuthority; ++offsetX) {
                if (!changedColumns.contains({position.x + offsetX, position.z + offsetZ})) {
                    continue;
                }
                samplesChangedAuthority = true;
            }
        }
        if (!samplesChangedAuthority) continue;
        queueAffected(position, *chunk, ownsChangedAuthority);
    }
}

World::SkyCutoffSectionRange World::refreshSavedSkyCutoffsLocked(ChunkPos pos) {
    const ColumnPos column{pos.x, pos.z};
    skyOverrideChunkColumns_.insert(column);
    const auto plan = generator_.findColumnPlan(column);
    SkyCutoffSectionRange changed;
    const int64_t baseX = column.x * CHUNK_EDGE;
    const int64_t baseZ = column.z * CHUNK_EDGE;
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const SkyColumnKey key{baseX + localX, baseZ + localZ};
            const auto previous = skyCutoffOverrides_.find(key);
            const int32_t plannedCutoff = plan ? plan->surfaceY(localX, localZ) + 1 : WORLD_MIN_Y;
            const int32_t previousCutoff =
                previous == skyCutoffOverrides_.end() ? plannedCutoff : previous->second;
            static_cast<void>(refreshSkyCutoffLocked(key.x, key.z));
            const auto current = skyCutoffOverrides_.find(key);
            const int32_t currentCutoff =
                current == skyCutoffOverrides_.end() ? plannedCutoff : current->second;
            changed.include(previousCutoff, currentCutoff);
        }
    }
    return changed;
}

void World::queueLightReconcile(ChunkPos pos, LightUrgency urgency) {
    std::lock_guard<std::mutex> lock(lightMutex_);
    // A position may sit in both queue tiers; the later pop floods again
    // with no diff, which is cheaper than searching the other queue for it.
    if (urgency == LightUrgency::IMMEDIATE) {
        if (editLightQueued_.insert(pos).second) {
            editLightQueue_.push_back(pos);
        }
        return;
    }
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

World::LightColumnPlans World::findLightColumnPlans(ChunkPos pos) const {
    LightColumnPlans plans{};
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            plans[static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1)] =
                generator_.findColumnPlan({pos.x + offsetX, pos.z + offsetZ});
        }
    }
    return plans;
}

void World::ensureSavedSkyAuthority(ColumnPos column) {
    if (!saveManager_) return;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        if (skyManifestKnownColumns_.contains(column)) return;
    }

    const std::vector<int32_t> savedSections = saveManager_->savedSections(column);
    VerticalSectionMask savedMask;
    for (const int32_t section : savedSections)
        savedMask.set(section);

    std::lock_guard<std::mutex> lock(chunksMutex_);
    if (!skyManifestKnownColumns_.insert(column).second) return;
    if (!savedMask.empty()) savedEditedSectionMasks_.insert_or_assign(column, savedMask);
    queueSavedSkyPublicationLocked({column});
}

LightEngine::SkyLightSeedColumns World::skyLightSeedsLocked(ChunkPos pos,
                                                            const ColumnPlan* plan) const {
    LightEngine::SkyLightSeedColumns result;
    if (!plan) return result;

    const ColumnPos column{pos.x, pos.z};
    const auto loadedMask = loadedSectionMasks_.find(column);
    if (loadedMask == loadedSectionMasks_.end()) return result;

    return skyLightSeedsForMaskLocked(pos, plan, loadedMask->second);
}

LightEngine::SkyLightSeedColumns
World::skyLightSeedsForMaskLocked(ChunkPos pos, const ColumnPlan* plan,
                                  const VerticalSectionMask& loadedMask) const {
    LightEngine::SkyLightSeedColumns result;
    if (!plan) return result;

    const ColumnPos column{pos.x, pos.z};
    if (saveManager_ && !skyManifestKnownColumns_.contains(column)) return result;

    const int64_t baseX = pos.x * CHUNK_EDGE;
    const int64_t baseZ = pos.z * CHUNK_EDGE;
    const auto saved = savedEditedSectionMasks_.find(column);
    const VerticalSectionMask required = potentialSkyOccupancySections(
        *plan, saved == savedEditedSectionMasks_.end() ? nullptr : &saved->second,
        loweredSkyCutoffColumns_.contains(column) ? std::optional(pos.y) : std::nullopt);
    if (!loadedMask.containsAllSetSections(required, pos.y)) return result;
    for (int z = 0; z < CHUNK_EDGE; ++z) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            int32_t cutoff = std::clamp(plan->surfaceY(x, z) + 1, WORLD_MIN_Y, WORLD_MAX_Y + 1);
            if (const auto edited = skyCutoffOverrides_.find({baseX + x, baseZ + z});
                edited != skyCutoffOverrides_.end()) {
                cutoff = edited->second;
            }

            result.set(x, z, cutoff);
        }
    }
    return result;
}

LightEngine::FloodResult World::initializeChunkLightLocked(ChunkPos pos, const ColumnPlan* plan) {
    const auto found = chunks_.find(pos);
    if (found == chunks_.end() || !found->second->generated) return {};
    const auto neighbor = [&](ChunkPos neighborPos) -> const Chunk* {
        const auto candidate = chunks_.find(neighborPos);
        return candidate != chunks_.end() && candidate->second->generated ? candidate->second.get()
                                                                          : nullptr;
    };
    const LightEngine::FaceNeighbors faces = {
        neighbor({pos.x - 1, pos.y, pos.z}),
        neighbor({pos.x + 1, pos.y, pos.z}),
        neighbor({pos.x, pos.y, pos.z - 1}),
        neighbor({pos.x, pos.y, pos.z + 1}),
        validChunkY(pos.y - 1) ? neighbor({pos.x, pos.y - 1, pos.z}) : nullptr,
        validChunkY(pos.y + 1) ? neighbor({pos.x, pos.y + 1, pos.z}) : nullptr,
    };
    return LightEngine::floodChunk(*found->second, faces, skyLightSeedsLocked(pos, plan));
}

void World::queuePublicationLightLocked(ChunkPos pos) {
    const auto found = chunks_.find(pos);
    if (found == chunks_.end() || !found->second->generated) return;
    found->second->publicationLightPending = true;
    if (found->second->publicationLightQueued) return;

    // Consume publication work in camera/flora lane and distance order. Equal
    // priorities retain FIFO order, while a camera jump reprioritizes the
    // unconsumed suffix. Compact only the already reserved storage when its
    // consumed prefix reaches the physical capacity.
    if (publicationLightQueue_.size() >= loadedCubeLimit_) {
        size_t write = 0;
        for (size_t read = publicationLightQueueHead_; read < publicationLightQueue_.size();
             ++read) {
            const PublicationLightQueueEntry queued = publicationLightQueue_[read];
            const auto candidate = chunks_.find(queued.position);
            if (candidate == chunks_.end() || !candidate->second->generated ||
                !candidate->second->publicationLightQueued ||
                candidate->second->publicationLightQueueToken != queued.token) {
                continue;
            }
            publicationLightQueue_[write++] = queued;
        }
        publicationLightQueue_.resize(write);
        publicationLightQueueHead_ = 0;
        publicationLightDeferredQueue_.store(write, std::memory_order_relaxed);
    }
    if (publicationLightQueue_.size() >= loadedCubeLimit_) return;
    found->second->publicationLightQueued = true;
    ++nextPublicationLightQueueToken_;
    if (nextPublicationLightQueueToken_ == 0) ++nextPublicationLightQueueToken_;
    found->second->publicationLightQueueToken = nextPublicationLightQueueToken_;
    const ChunkPos center{
        lightPriorityCenterX_.load(std::memory_order_relaxed),
        lightPriorityCenterY_.load(std::memory_order_relaxed),
        lightPriorityCenterZ_.load(std::memory_order_relaxed),
    };
    const int64_t priority = exactPublicationLightPriority(pos, center);
    const auto first = publicationLightQueue_.begin() +
                       static_cast<std::vector<PublicationLightQueueEntry>::difference_type>(
                           publicationLightQueueHead_);
    const auto insertion = std::find_if(
        first, publicationLightQueue_.end(), [priority](const PublicationLightQueueEntry& queued) {
            return queued.priority < priority;
        });
    publicationLightQueue_.insert(insertion, {pos, nextPublicationLightQueueToken_, priority});
    const size_t queuedCount = publicationLightQueue_.size() - publicationLightQueueHead_;
    publicationLightDeferredQueue_.store(queuedCount, std::memory_order_relaxed);
    publicationLightDeferredCubes_.fetch_add(1, std::memory_order_relaxed);
    recordLoadedCubeHighWater(publicationLightMaxDeferredQueue_, queuedCount);
}

void World::reprioritizePublicationLightLocked(ChunkPos center) {
    const auto first = publicationLightQueue_.begin() +
                       static_cast<std::vector<PublicationLightQueueEntry>::difference_type>(
                           publicationLightQueueHead_);
    for (auto iterator = first; iterator != publicationLightQueue_.end(); ++iterator) {
        iterator->priority = exactPublicationLightPriority(iterator->position, center);
    }
    std::stable_sort(first, publicationLightQueue_.end(),
                     [](const PublicationLightQueueEntry& left,
                        const PublicationLightQueueEntry& right) {
                         return left.priority > right.priority;
                     });
}

size_t World::settleChunkPublicationLightLocked(ChunkPos pos,
                                                const LightEngine::FloodResult& initialFlood,
                                                const VerticalSectionMask& previousColumnMask,
                                                const LightColumnPlans& columnPlans,
                                                const SkyCutoffSectionRange& skyCutoffChange,
                                                size_t floodCap) {
    // A generated cube becomes visible as soon as chunksMutex_ is released.
    // Settle every light change caused by its arrival under this same lock so
    // an existing halo cannot rebuild once with stale light and then visibly
    // relight on a later fixed tick.
    constexpr std::array<ChunkPos, 6> FACE_OFFSETS = {
        ChunkPos{-1, 0, 0}, ChunkPos{1, 0, 0},  ChunkPos{0, 0, -1},
        ChunkPos{0, 0, 1},  ChunkPos{0, -1, 0}, ChunkPos{0, 1, 0},
    };
    const auto planFor = [&](ChunkPos candidate) -> const ColumnPlan* {
        const int64_t offsetX = candidate.x - pos.x;
        const int64_t offsetZ = candidate.z - pos.z;
        if (offsetX < -1 || offsetX > 1 || offsetZ < -1 || offsetZ > 1) return nullptr;
        return columnPlans[static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1)].get();
    };

    struct WorkItem {
        ChunkPos position;
        const ColumnPlan* plan = nullptr;
    };
    std::array<WorkItem, PUBLICATION_LIGHT_SYNC_FLOOD_CAP> work{};
    // The initial cube can expose six faces, and each bounded flood can expose
    // six more unique neighbors. Retain every pending bit we set so the final
    // commit either clears it or gives that cube a durable queue entry.
    std::array<ChunkPos, 1 + 6 + PUBLICATION_LIGHT_SYNC_FLOOD_CAP * 6> touched{};
    size_t workCount = 0;
    size_t touchedCount = 0;
    bool deferred = false;

    const auto rememberTouched = [&](ChunkPos candidate) {
        if (std::find(touched.begin(), touched.begin() + static_cast<ptrdiff_t>(touchedCount),
                      candidate) != touched.begin() + static_cast<ptrdiff_t>(touchedCount)) {
            return;
        }
        if (touchedCount < touched.size()) touched[touchedCount++] = candidate;
    };
    const auto defer = [&](ChunkPos candidate) {
        deferred = true;
        rememberTouched(candidate);
        queuePublicationLightLocked(candidate);
    };
    const auto enqueue = [&](ChunkPos candidate) {
        if (!validChunkY(candidate.y)) return;
        const auto found = chunks_.find(candidate);
        if (found == chunks_.end() || !found->second->generated) return;
        const ColumnPlan* plan = planFor(candidate);
        if (!plan) {
            defer(candidate);
            return;
        }
        const auto queued =
            std::find_if(work.begin(), work.begin() + static_cast<ptrdiff_t>(workCount),
                         [&](const WorkItem& item) { return item.position == candidate; });
        if (queued != work.begin() + static_cast<ptrdiff_t>(workCount)) return;
        if (workCount == work.size()) {
            defer(candidate);
            return;
        }
        found->second->publicationLightPending = true;
        rememberTouched(candidate);
        work[workCount++] = {candidate, plan};
    };

    const auto markMeshDirty = [&](ChunkPos candidate) {
        const auto found = chunks_.find(candidate);
        if (found == chunks_.end() || !found->second->generated) return;
        found->second->needsMeshUpdate = true;
        found->second->version.fetch_add(1, std::memory_order_relaxed);
    };

    const auto enqueueChangedFaces = [&](ChunkPos source, uint8_t changedFaceMask,
                                         bool markNeighborMeshes) {
        for (size_t face = 0; face < FACE_OFFSETS.size(); ++face) {
            if ((changedFaceMask & (1U << face)) == 0) continue;
            const ChunkPos offset = FACE_OFFSETS[face];
            const ChunkPos neighbor{source.x + offset.x, source.y + offset.y, source.z + offset.z};
            if (markNeighborMeshes) markMeshDirty(neighbor);
            enqueue(neighbor);
        }
    };

    // markHaloNeighborMeshesDirtyLocked already dirtied the immediate halo.
    // The changed faces identify which of those cubes must also import the
    // newly published border light.
    enqueueChangedFaces(pos, initialFlood.changedFaceMask, false);

    // Inserting a sparse potential-occupancy section can complete sky
    // authority for any resident section below it even when a large proven
    // empty gap separates them. Compare exact old and new seed authority for
    // every loaded lower section, then relight only those that changed.
    const auto markSkySection = [&](int32_t sectionY) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            if (!validChunkY(sectionY + offsetY)) continue;
            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    markMeshDirty({pos.x + offsetX, sectionY + offsetY, pos.z + offsetZ});
                }
            }
        }
        enqueue({pos.x, sectionY, pos.z});
    };

    const auto currentMask = loadedSectionMasks_.find({pos.x, pos.z});
    const ColumnPlan* ownPlan = columnPlans[4].get();
    if (currentMask != loadedSectionMasks_.end() && ownPlan) {
        previousColumnMask.visitSetSections(WORLD_MIN_CHUNK_Y, pos.y - 1, [&](int32_t sectionY) {
            publicationLightSectionVisits_.fetch_add(1, std::memory_order_relaxed);
            const auto found = chunks_.find({pos.x, sectionY, pos.z});
            if (found == chunks_.end() || !found->second->generated) return true;
            const ChunkPos candidate{pos.x, sectionY, pos.z};
            const auto oldSeeds =
                skyLightSeedsForMaskLocked(candidate, ownPlan, previousColumnMask);
            const auto newSeeds =
                skyLightSeedsForMaskLocked(candidate, ownPlan, currentMask->second);
            if (oldSeeds.cutoffY != newSeeds.cutoffY) markSkySection(sectionY);
            return true;
        });
    }

    // A generated feature or saved edit can move the immutable surface cutoff
    // even when the loaded-section mask did not change a seed's completeness.
    // Recompute every resident section in that column before publication and
    // invalidate every loaded neighboring mesh that samples its cutoff halo.
    if (skyCutoffChange.changed() && currentMask != loadedSectionMasks_.end()) {
        currentMask->second.visitSetSections(
            skyCutoffChange.first, skyCutoffChange.last, [&](int32_t sectionY) {
                publicationLightSectionVisits_.fetch_add(1, std::memory_order_relaxed);
                markSkySection(sectionY);
                return true;
            });
    }

    rememberTouched(pos);
    if (const auto found = chunks_.find(pos); found != chunks_.end()) {
        found->second->publicationLightPending = true;
    }
    size_t processed = 0;
    floodCap = std::min(floodCap, PUBLICATION_LIGHT_SYNC_FLOOD_CAP);
    while (workCount != 0 && processed < floodCap) {
        const WorkItem item = work[--workCount];
        const ChunkPos candidate = item.position;

        const auto found = chunks_.find(candidate);
        const ColumnPlan* plan = item.plan;
        if (found == chunks_.end() || !found->second->generated || !plan) continue;

        const auto neighbor = [&](ChunkPos neighborPos) -> const Chunk* {
            const auto adjacent = chunks_.find(neighborPos);
            return adjacent != chunks_.end() && adjacent->second->generated ? adjacent->second.get()
                                                                            : nullptr;
        };
        const LightEngine::FaceNeighbors faces = {
            neighbor({candidate.x - 1, candidate.y, candidate.z}),
            neighbor({candidate.x + 1, candidate.y, candidate.z}),
            neighbor({candidate.x, candidate.y, candidate.z - 1}),
            neighbor({candidate.x, candidate.y, candidate.z + 1}),
            validChunkY(candidate.y - 1) ? neighbor({candidate.x, candidate.y - 1, candidate.z})
                                         : nullptr,
            validChunkY(candidate.y + 1) ? neighbor({candidate.x, candidate.y + 1, candidate.z})
                                         : nullptr,
        };
        const LightEngine::FloodResult result =
            LightEngine::floodChunk(*found->second, faces, skyLightSeedsLocked(candidate, plan));
        ++processed;
        if (!result.changedState) continue;

        found->second->needsMeshUpdate = true;
        found->second->version.fetch_add(1, std::memory_order_relaxed);
        enqueueChangedFaces(candidate, result.changedFaceMask, true);
    }

    if (workCount != 0) {
        deferred = true;
        for (size_t index = 0; index < workCount; ++index)
            queuePublicationLightLocked(work[index].position);
    }
    if (deferred) {
        for (size_t index = 0; index < touchedCount; ++index)
            queuePublicationLightLocked(touched[index]);
        return processed;
    }
    for (size_t index = 0; index < touchedCount; ++index) {
        const auto found = chunks_.find(touched[index]);
        if (found == chunks_.end()) continue;
        found->second->publicationLightPending = false;
        found->second->publicationLightQueued = false;
    }
    return processed;
}

size_t World::drainPublicationLightLocked(ChunkPos first, const LightColumnPlans& columnPlans,
                                          size_t floodBudget) {
    if (floodBudget == 0) return 0;
    const auto found = chunks_.find(first);
    if (found == chunks_.end() || !found->second->generated ||
        !found->second->publicationLightPending) {
        return 0;
    }
    const LightEngine::FloodResult initial =
        initializeChunkLightLocked(first, columnPlans[4].get());
    VerticalSectionMask currentMask;
    if (const auto mask = loadedSectionMasks_.find({first.x, first.z});
        mask != loadedSectionMasks_.end()) {
        currentMask = mask->second;
    }
    // The initial flood is part of the same bounded publication transaction.
    // Cap its follow-up budget one lower so a large fixed-tick allowance can
    // never turn the 32-flood safety bound into 33 floods.
    const size_t transactionBudget =
        std::min(floodBudget, PUBLICATION_LIGHT_SYNC_FLOOD_CAP);
    const size_t followup = settleChunkPublicationLightLocked(first, initial, currentMask,
                                                              columnPlans, {},
                                                              transactionBudget - 1);
    const size_t total = 1 + followup;
    publicationLightSyncFloods_.fetch_add(total, std::memory_order_relaxed);
    recordLoadedCubeHighWater(publicationLightMaxSyncFloods_, total);
    return total;
}

bool World::reconcileCubeLocked(ChunkPos pos, const ColumnPlan* plan, LightUrgency urgency) {
    auto it = chunks_.find(pos);
    if (it == chunks_.end() || !it->second->generated) {
        return false;
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

    const LightEngine::SkyLightSeedColumns skySeeds = skyLightSeedsLocked(pos, plan);
    const bool hasDirectSky = std::ranges::any_of(skySeeds.cutoffY, [&](int32_t cutoff) {
        return cutoff <= pos.y * CHUNK_EDGE + CHUNK_EDGE - 1;
    });
    auto dark = [](const Chunk* chunk) { return !chunk || !chunk->hasDerivedLight(); };
    if (!it->second->hasDerivedLight() && std::all_of(faces.begin(), faces.end(), dark) &&
        !hasBlockLightEmitter(*it->second) && !hasDirectSky) {
        return false;
    }

    LightEngine::FaceNeighbors neighbors = {faces[0], faces[1], faces[2],
                                            faces[3], faces[4], faces[5]};
    const LightEngine::FloodResult result =
        LightEngine::floodChunk(*it->second, neighbors, skySeeds);
    if (result.changedState) {
        it->second->needsMeshUpdate = true;
        it->second->version.fetch_add(1, std::memory_order_relaxed);
        if (urgency == LightUrgency::IMMEDIATE) {
            lightingRevision_.fetch_add(1, std::memory_order_release);
        }
        // This chunk's border light moved: each face-neighbor both SAMPLES
        // that border (so its border faces must re-mesh even if its own
        // stored light is unchanged, such as a solid wall at the seam) and
        // may pull in more light (so it must re-reconcile). Spread stays in
        // the queue tier it started in so edit-driven propagation cannot
        // sink back into the starving streaming queue.
        constexpr std::array<ChunkPos, 6> OFFSETS = {
            ChunkPos{-1, 0, 0}, ChunkPos{1, 0, 0},  ChunkPos{0, 0, -1},
            ChunkPos{0, 0, 1},  ChunkPos{0, -1, 0}, ChunkPos{0, 1, 0},
        };
        for (size_t face = 0; face < OFFSETS.size(); ++face) {
            if ((result.changedFaceMask & (1U << face)) == 0) continue;
            if (Chunk* neighbor = faces[face]) {
                neighbor->needsMeshUpdate = true;
                neighbor->version.fetch_add(1, std::memory_order_relaxed);
            }
            const ChunkPos offset = OFFSETS[face];
            queueLightReconcile({pos.x + offset.x, pos.y + offset.y, pos.z + offset.z}, urgency);
        }
    }
    return result.changedState;
}

void World::reconcileLight(int budgetCubes) {
    int processed = 0;
    while (processed < budgetCubes) {
        std::optional<ChunkPos> publication;
        {
            std::lock_guard<std::mutex> lock(chunksMutex_);
            while (publicationLightQueueHead_ < publicationLightQueue_.size()) {
                const PublicationLightQueueEntry queued =
                    publicationLightQueue_[publicationLightQueueHead_++];
                publicationLightDeferredQueue_.store(publicationLightQueue_.size() -
                                                         publicationLightQueueHead_,
                                                     std::memory_order_relaxed);
                const auto found = chunks_.find(queued.position);
                if (found == chunks_.end() || !found->second->generated ||
                    !found->second->publicationLightQueued ||
                    found->second->publicationLightQueueToken != queued.token) {
                    continue;
                }
                found->second->publicationLightQueued = false;
                publication = queued.position;
                break;
            }
            if (publicationLightQueueHead_ == publicationLightQueue_.size()) {
                publicationLightQueue_.clear();
                publicationLightQueueHead_ = 0;
            }
        }
        if (publication) {
            const LightColumnPlans plans = findLightColumnPlans(*publication);
            std::lock_guard<std::mutex> lock(chunksMutex_);
            const size_t consumed = drainPublicationLightLocked(
                *publication, plans, static_cast<size_t>(budgetCubes - processed));
            processed += static_cast<int>(std::max<size_t>(consumed, 1));
            continue;
        }

        ChunkPos pos;
        LightUrgency urgency = LightUrgency::DEFERRED;
        {
            std::lock_guard<std::mutex> lock(lightMutex_);
            if (!editLightQueue_.empty()) {
                pos = editLightQueue_.back();
                editLightQueue_.pop_back();
                editLightQueued_.erase(pos);
                urgency = LightUrgency::IMMEDIATE;
            } else if (!lightQueue_.empty()) {
                pos = lightQueue_.back();
                lightQueue_.pop_back();
                lightQueued_.erase(pos);
            } else {
                return;
            }
        }

        // Cache lookup is outside chunksMutex_ and never opens authority.
        const auto lightPlan = generator_.findColumnPlan({pos.x, pos.z});
        std::lock_guard<std::mutex> lock(chunksMutex_);
        reconcileCubeLocked(pos, lightPlan.get(), urgency);
        ++processed;
    }
}

int World::drainEditLightNeighborhoodLocked(
    ChunkPos home, const std::array<std::shared_ptr<const ColumnPlan>, 9>& columnPlans) {
    // Runs with chunksMutex_ held. reconcileCubeLocked re-enqueues a neighbor
    // whenever a shared border changed, so popping the IMMEDIATE queue until it
    // empties floods the whole connected affected set to its fixed point.
    std::vector<ChunkPos> outOfRange;
    int processed = 0;
    while (processed < EDIT_SYNC_LIGHT_FLOOD_CAP) {
        ChunkPos pos;
        {
            std::lock_guard<std::mutex> lock(lightMutex_);
            if (editLightQueue_.empty()) break;
            pos = editLightQueue_.back();
            editLightQueue_.pop_back();
            editLightQueued_.erase(pos);
        }
        const int64_t dx = pos.x - home.x;
        const int64_t dz = pos.z - home.z;
        if (dx < -1 || dx > 1 || dz < -1 || dz > 1) {
            // A single edit's light cannot reach a column two cubes away; if the
            // queue ever holds one (unrelated churn), leave it for the tick
            // drain rather than calling the generator under the lock.
            outOfRange.push_back(pos);
            continue;
        }
        reconcileCubeLocked(pos, columnPlans[static_cast<size_t>((dx + 1) + (dz + 1) * 3)].get(),
                            LightUrgency::IMMEDIATE);
        ++processed;
    }
    if (!outOfRange.empty()) {
        std::lock_guard<std::mutex> lock(lightMutex_);
        for (const ChunkPos& pos : outOfRange) {
            if (editLightQueued_.insert(pos).second) editLightQueue_.push_back(pos);
        }
    }
    return processed;
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
            out.visualSkyCutoffY[MeshSnapshot::skyIndex(x, z)] = generatedCutoff;
        }
    }

    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        auto self = chunks_.find(pos);
        if (self == chunks_.end() || !self->second->generated ||
            self->second->publicationLightPending) {
            return false;
        }

        for (int z = -1; z <= CHUNK_EDGE; ++z) {
            const int64_t worldZ = pos.z * CHUNK_EDGE + z;
            for (int x = -1; x <= CHUNK_EDGE; ++x) {
                const int64_t worldX = pos.x * CHUNK_EDGE + x;
                const auto edited = skyCutoffOverrides_.find({worldX, worldZ});
                if (edited != skyCutoffOverrides_.end()) {
                    out.skyCutoffY[MeshSnapshot::skyIndex(x, z)] = edited->second;
                }
                // Preserve the complete geometric cutoff for water-interface
                // classification before incomplete streaming authority closes
                // the ordinary skylight path below.
                out.visualSkyCutoffY[MeshSnapshot::skyIndex(x, z)] =
                    out.skyCutoffY[MeshSnapshot::skyIndex(x, z)];

                // The plan proves every section outside exposedSections() has
                // no generated feature capable of changing its sky cutoff. The
                // save manifest gives the same proof for gaps between edited
                // sections. Do not publish this cube's first mesh until every
                // sparse potential-occupancy section at or above it is resident
                // and its publication light transaction has settled. Returning
                // false is nonblocking while the renderer keeps its coarse parent.
                const int64_t columnX = Chunk::worldToChunk(worldX);
                const int64_t columnZ = Chunk::worldToChunk(worldZ);
                const ColumnPos skyColumn{columnX, columnZ};
                const size_t planIndex =
                    static_cast<size_t>((columnZ - pos.z + 1) * 3 + (columnX - pos.x + 1));
                const auto& skyPlan = columnPlans[planIndex];
                const auto saved = savedEditedSectionMasks_.find(skyColumn);
                const VerticalSectionMask required = potentialSkyOccupancySections(
                    *skyPlan, saved == savedEditedSectionMasks_.end() ? nullptr : &saved->second,
                    loweredSkyCutoffColumns_.contains(skyColumn) ? std::optional(pos.y)
                                                                 : std::nullopt);
                const auto loadedMask = loadedSectionMasks_.find(skyColumn);
                const bool manifestAuthority =
                    !saveManager_ || skyManifestKnownColumns_.contains(skyColumn);
                const VerticalSectionMask noLoadedSections;
                const bool completeSkyPath =
                    manifestAuthority &&
                    (loadedMask == loadedSectionMasks_.end() ? noLoadedSections
                                                             : loadedMask->second)
                        .containsAllSetSections(required, pos.y);
                if (!completeSkyPath) return false;
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
                        if (it->second->publicationLightPending) return false;
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
        out.derivedSkyLightValid = true;
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
                        out.packedLight[target] = source->getPackedLight(localX, localY, localZ);
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
                        const int target = MeshSnapshot::index(x, y, z);
                        const bool generatedSkyOpen =
                            generatedCutoff != MeshSnapshot::SKY_CUTOFF_UNKNOWN &&
                            worldY >= generatedCutoff;
                        out.blocks[target] = generatedSkyOpen ? BlockType::AIR : BlockType::BEDROCK;
                    } else {
                        out.packedLight[MeshSnapshot::index(x, y, z)] = FULL_SKY_PACKED_LIGHT;
                    }
                }
            }
        }
    }

    // Material lookup stays outside chunksMutex_. A missing face gets one
    // representative arriving-terrain material, then reuses it across its 16
    // transient caps. This bounds cold frontier work to four nonlinear surface
    // samples per mesh instead of 64 while avoiding a dark default material.
    // The real neighbor mesh replaces this provisional face once it arrives.
    const auto cacheFaceMaterial = [&](int sampleX, int sampleZ, auto assign) {
        const int64_t worldX = pos.x * CHUNK_EDGE + sampleX;
        const int64_t worldZ = pos.z * CHUNK_EDGE + sampleZ;
        const BlockType material = generator_.surfaceMaterialAt(worldX, worldZ);
        for (int coordinate = 0; coordinate < CHUNK_EDGE; ++coordinate)
            assign(coordinate, material);
    };
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_PLUS_X) != 0) {
        cacheFaceMaterial(CHUNK_EDGE, CHUNK_EDGE / 2, [&](int z, BlockType material) {
            out.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = material;
        });
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_MINUS_X) != 0) {
        cacheFaceMaterial(-1, CHUNK_EDGE / 2, [&](int z, BlockType material) {
            out.generatedSurfaceMaterial[MeshSnapshot::skyIndex(-1, z)] = material;
        });
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_PLUS_Z) != 0) {
        cacheFaceMaterial(CHUNK_EDGE / 2, CHUNK_EDGE, [&](int x, BlockType material) {
            out.generatedSurfaceMaterial[MeshSnapshot::skyIndex(x, CHUNK_EDGE)] = material;
        });
    }
    if ((out.missingNeighborFaces & MeshSnapshot::MISSING_MINUS_Z) != 0) {
        cacheFaceMaterial(CHUNK_EDGE / 2, -1, [&](int x, BlockType material) {
            out.generatedSurfaceMaterial[MeshSnapshot::skyIndex(x, -1)] = material;
        });
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

std::shared_ptr<const ExactSurfaceCoverageSnapshot> World::getExactSurfaceCoverageSnapshot() const {
    return std::atomic_load_explicit(&exactSurfaceCoverageSnapshot_, std::memory_order_acquire);
}

std::shared_ptr<const ExactCollisionOwnershipSnapshot>
World::getExactCollisionOwnershipSnapshot() const {
    return std::atomic_load_explicit(&exactCollisionOwnershipSnapshot_,
                                     std::memory_order_acquire);
}

bool World::publishExactCollisionOwnership(uint64_t coverageEpoch,
                                           std::span<const ChunkPos> sections) const {
    auto snapshot = std::make_shared<ExactCollisionOwnershipSnapshot>();
    snapshot->coverageEpoch = coverageEpoch;
    snapshot->sections.reserve(sections.size());
    snapshot->sections.insert(sections.begin(), sections.end());

    std::lock_guard lock(exactCollisionOwnershipPublicationMutex_);
    const std::shared_ptr<const ExactSurfaceCoverageSnapshot> coverage =
        getExactSurfaceCoverageSnapshot();
    if (!coverage || coverage->epoch != coverageEpoch) return false;
    std::shared_ptr<const ExactCollisionOwnershipSnapshot> immutable = std::move(snapshot);
    std::atomic_store_explicit(&exactCollisionOwnershipSnapshot_, std::move(immutable),
                               std::memory_order_release);
    return true;
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

void World::setExactStreamingDistance(int distance) {
    const int bounded = std::clamp(distance, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS,
                                   MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
    if (exactStreamingDistance_.exchange(bounded, std::memory_order_acq_rel) == bounded) return;
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
    const auto lane = exactPriorityByCube_.find(pos);
    const uint8_t priorityLane = lane == exactPriorityByCube_.end() ? 0 : lane->second;
    genBacklog_.push(
        {pos, exactStreamingTaskPriority(
                  activeSetEpoch_, priorityLane,
                  exactStreamingCubePriorityDistance(pos, exactPriorityCenter_))});
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
    for (int offsetZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
         offsetZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetZ) {
        for (int offsetX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             offsetX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetX) {
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
    std::array<ColumnPos, EXACT_STREAMING_PLAN_DEPENDENCY_COLUMN_COUNT> apron{};
    size_t apronCount = 0;
    if (ownPlan) {
        for (int offsetZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             offsetZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetZ) {
            for (int offsetX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
                 offsetX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetX) {
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
                    sectionMask->second.reset(position.y);
                    if (sectionMask->second.empty()) loadedSectionMasks_.erase(sectionMask);
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
            const bool columnStillLoaded = loadedSectionMasks_.contains(column);
            if (columnStillLoaded) {
                if (skyColumnsNeedingRefresh.contains(column)) {
                    const SkyCutoffSectionRange changed = refreshSkyOverrideColumnLocked(column);
                    if (changed.changed()) markSkyColumnMeshesDirtyLocked(column);
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
            loweredSkyCutoffColumns_.erase(column);
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
    const ChunkPos lightCenter{Chunk::worldToChunk(playerX), Chunk::worldToChunkY(playerY),
                               Chunk::worldToChunk(playerZ)};
    lightPriorityCenterX_.store(lightCenter.x, std::memory_order_relaxed);
    lightPriorityCenterY_.store(lightCenter.y, std::memory_order_relaxed);
    lightPriorityCenterZ_.store(lightCenter.z, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> chunksLock(chunksMutex_);
        reprioritizePublicationLightLocked(lightCenter);
    }
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
        genPool_ = std::make_unique<ThreadPool>(
            EXACT_GENERATION_WORKER_COUNT, ThreadPriority::UTILITY, EXACT_LATENCY_WORKER_COUNT);
    }
    if (!activeSetThread_.joinable()) {
        activeSetThread_ = std::thread([this] { activeSetWorkerLoop(); });
    }
}

void World::requestActiveSetRebuild(int64_t playerX, int32_t playerY, int64_t playerZ) {
    ensureStreamingWorkers();
    const ChunkPos lightCenter{Chunk::worldToChunk(playerX), Chunk::worldToChunkY(playerY),
                               Chunk::worldToChunk(playerZ)};
    lightPriorityCenterX_.store(lightCenter.x, std::memory_order_relaxed);
    lightPriorityCenterY_.store(lightCenter.y, std::memory_order_relaxed);
    lightPriorityCenterZ_.store(lightCenter.z, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> chunksLock(chunksMutex_);
        reprioritizePublicationLightLocked(lightCenter);
    }
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
    // This thread decides which exact cubes are allowed to generate, mesh,
    // collide, and evict old residency after movement. Treat it as interactive
    // work so distant construction cannot delay publication of the new camera
    // neighborhood before priority queues even have a chance to react.
    setCurrentThreadPriority(ThreadPriority::USER_INITIATED);
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
            {
                std::lock_guard<std::mutex> buildLock(activeSetBuildMutex_);
                (void)rebuildActiveSet(request);
            }
        } catch (const std::exception& error) {
            const std::string message = std::string("Active-set rebuild failed: ") + error.what();
            RY_LOG_ERROR(message.c_str());
            latchGenerationFailure(message);
        } catch (...) {
            constexpr const char* message = "Active-set rebuild failed with an unknown exception";
            RY_LOG_ERROR(message);
            latchGenerationFailure(message);
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
    // tiles, so a 512-chunk view never expands cubic simulation to 8 km.
    const int exactViewDistance =
        std::min({request.viewDistance, MAX_EXACT_CUBIC_DISTANCE_CHUNKS,
                  exactStreamingDistance_.load(std::memory_order_acquire)});
    const int radius = exactStreamingActiveSetRadiusChunks(exactViewDistance);
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
    struct ExplorationSkyPathRequest {
        int32_t firstSection = WORLD_MAX_CHUNK_Y;
        uint8_t priority = EXACT_STREAMING_SURFACE_PRIORITY_LANE;
    };
    std::unordered_map<ColumnPos, ExplorationSkyPathRequest> explorationSkyPaths;
    explorationSkyPaths.reserve(static_cast<size_t>((EXPLORATION_RADIUS_CHUNKS * 2 + 3) *
                                                    (EXPLORATION_RADIUS_CHUNKS * 2 + 3)));
    const int32_t firstExplorationSection =
        std::max(WORLD_MIN_CHUNK_Y, centerY - EXPLORATION_VERTICAL_RADIUS_CUBES);
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (withinExactStreamingRadius(dx, dz, radius)) {
                const ColumnPos visible{centerX + dx, centerZ + dz};
                visibleColumns.push_back(visible);
                if (std::abs(dx) <= EXPLORATION_RADIUS_CHUNKS &&
                    std::abs(dz) <= EXPLORATION_RADIUS_CHUNKS &&
                    withinExactStreamingRadius(dx, dz, EXPLORATION_RADIUS_CHUNKS)) {
                    const uint8_t priority =
                        dx == 0 && dz == 0 ? EXACT_STREAMING_CAMERA_PRIORITY_LANE
                                          : EXACT_STREAMING_EXPLORATION_PRIORITY_LANE;
                    for (int haloZ = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                         haloZ <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++haloZ) {
                        for (int haloX = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                             haloX <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++haloX) {
                            const ColumnPos authority{visible.x + haloX, visible.z + haloZ};
                            auto [entry, inserted] = explorationSkyPaths.try_emplace(
                                authority,
                                ExplorationSkyPathRequest{firstExplorationSection, priority});
                            if (!inserted) {
                                entry->second.firstSection =
                                    std::min(entry->second.firstSection, firstExplorationSection);
                                entry->second.priority = std::max(entry->second.priority, priority);
                            }
                        }
                    }
                }
            }
        }
    }
    if (cancelIfStale()) return false;

    // Saved ceilings are part of vertical sky authority. Query the bounded
    // exploration halo along with the visible exact columns so a deep-camera
    // first mesh cannot wait forever on an unknown neighboring manifest.
    std::vector<ColumnPos> manifestColumns = visibleColumns;
    manifestColumns.reserve(visibleColumns.size() + explorationSkyPaths.size());
    for (const auto& [column, request] : explorationSkyPaths) {
        (void)request;
        manifestColumns.push_back(column);
    }
    std::sort(manifestColumns.begin(), manifestColumns.end(), [](ColumnPos left, ColumnPos right) {
        if (left.x != right.x) return left.x < right.x;
        return left.z < right.z;
    });
    manifestColumns.erase(std::unique(manifestColumns.begin(), manifestColumns.end()),
                          manifestColumns.end());
    std::unordered_map<ColumnPos, std::vector<int32_t>> savedSections;
    if (saveManager_) savedSections = saveManager_->savedSectionsForColumns(manifestColumns);
    if (cancelIfStale()) return false;

    std::unordered_set<ColumnPos> manifestKnownColumns;
    std::unordered_map<ColumnPos, VerticalSectionMask> savedSectionMasks;
    if (saveManager_) {
        manifestKnownColumns.reserve(manifestColumns.size());
        manifestKnownColumns.insert(manifestColumns.begin(), manifestColumns.end());
        savedSectionMasks.reserve(savedSections.size());
        for (const auto& [column, sections] : savedSections) {
            VerticalSectionMask mask;
            for (const int32_t section : sections)
                mask.set(section);
            if (!mask.empty()) savedSectionMasks.emplace(column, mask);
        }
    }
    std::unordered_set<ColumnPos> loweredSkyCutoffColumns;
    {
        std::lock_guard<std::mutex> lock(chunksMutex_);
        loweredSkyCutoffColumns.reserve(loweredSkyCutoffColumns_.size());
        for (ColumnPos column : loweredSkyCutoffColumns_) {
            if (explorationSkyPaths.contains(column)) {
                loweredSkyCutoffColumns.insert(column);
            }
        }
    }

    std::unordered_set<ChunkPos> wantedSet;
    std::unordered_map<ChunkPos, uint8_t> wantedPriority;
    std::unordered_set<ChunkPos> surfaceOwnershipRequirements;
    std::unordered_set<ChunkPos> floraOwnershipRequirements;
    std::vector<ColumnPos> unresolvedSurfaceColumns;
    wantedSet.reserve(
        std::min<size_t>(loadedCubeLimit_ * 2, static_cast<size_t>(radius * radius * 8)));
    wantedPriority.reserve(wantedSet.bucket_count());
    surfaceOwnershipRequirements.reserve(visibleColumns.size() * 2);
    floraOwnershipRequirements.reserve(visibleColumns.size() * 6);
    unresolvedSurfaceColumns.reserve(visibleColumns.size());
    auto addWanted = [&](ChunkPos pos, uint8_t priority) {
        if (!validChunkY(pos.y)) return;
        wantedSet.insert(pos);
        auto [iterator, inserted] = wantedPriority.try_emplace(pos, priority);
        if (!inserted) iterator->second = std::max(iterator->second, priority);
    };
    auto addSurfaceRequirement = [&](ChunkPos pos, uint8_t priority) { addWanted(pos, priority); };
    auto addFloraOwnershipRequirement = [&](ChunkPos pos, uint8_t priority) {
        addSurfaceRequirement(pos, priority);
        if (validChunkY(pos.y)) floraOwnershipRequirements.insert(pos);
    };
    auto addSurfaceOwnershipRequirement = [&](ChunkPos pos, uint8_t priority) {
        addSurfaceRequirement(pos, priority);
        if (validChunkY(pos.y)) surfaceOwnershipRequirements.insert(pos);
    };
    for (ColumnPos column : visibleColumns) {
        const int dx = static_cast<int>(column.x - centerX);
        const int dz = static_cast<int>(column.z - centerZ);
        const int64_t chunkX = column.x;
        const int64_t chunkZ = column.z;
        const uint8_t surfacePriority = exactStreamingSurfacePriorityLane(dx, dz);
        const uint8_t floraPriority = exactStreamingFloraPriorityLane(dx, dz);
        const uint8_t primarySurfacePriority =
            exactStreamingPrimarySurfacePriorityLane(dx, dz);
        if (const auto plan = generator_.findColumnPlan(column)) {
            for (int32_t section : plan->floraOwnershipSections()) {
                addFloraOwnershipRequirement({chunkX, section, chunkZ}, floraPriority);
            }
            for (int32_t section : plan->surfaceOwnershipSections()) {
                addSurfaceOwnershipRequirement({chunkX, section, chunkZ}, surfacePriority);
            }
            const int32_t primarySection =
                Chunk::worldToChunkY(plan->surfaceY(CHUNK_EDGE / 2, CHUNK_EDGE / 2));
            addSurfaceOwnershipRequirement({chunkX, primarySection, chunkZ},
                                           primarySurfacePriority);

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
                        addSurfaceOwnershipRequirement({higher.x, section, higher.z},
                                                       surfacePriority);
                    }
                }
            };
            exposeBoundaryWall({chunkX + 1, chunkZ}, true);
            exposeBoundaryWall({chunkX, chunkZ + 1}, false);
        } else {
            unresolvedSurfaceColumns.push_back(column);
            addSurfaceOwnershipRequirement({chunkX, Chunk::worldToChunkY(SEA_LEVEL), chunkZ},
                                           primarySurfacePriority);
        }

        if (const auto saved = savedSections.find(column); saved != savedSections.end()) {
            for (int32_t section : saved->second) {
                addWanted({chunkX, section, chunkZ}, EXACT_STREAMING_EDITED_PRIORITY_LANE);
            }
        }

        if (std::abs(dx) <= EXPLORATION_RADIUS_CHUNKS &&
            std::abs(dz) <= EXPLORATION_RADIUS_CHUNKS &&
            withinExactStreamingRadius(dx, dz, EXPLORATION_RADIUS_CHUNKS)) {
            for (int oy = -EXPLORATION_VERTICAL_RADIUS_CUBES;
                 oy <= EXPLORATION_VERTICAL_RADIUS_CUBES; ++oy) {
                const int32_t y = centerY + oy;
                addWanted({chunkX, y, chunkZ},
                          dx == 0 && dz == 0 ? EXACT_STREAMING_CAMERA_PRIORITY_LANE
                                            : EXACT_STREAMING_EXPLORATION_PRIORITY_LANE);
            }
        }
    }
    if (cancelIfStale()) return false;

    // Only the six-chunk exploration disk can contain a camera below the
    // generated surface. Retain sparse generated and saved potential-occupancy
    // sections for its one-column horizontal meshing halo as nonmesh support.
    // Proven-empty vertical gaps never materialize. The theoretical square
    // upper bound remains safely below the hard cap even if every section is
    // edited, and it stays independent of the 32-chunk exact view radius.
    constexpr size_t MAX_EXPLORATION_SKY_AUTHORITY_CUBES =
        static_cast<size_t>(EXPLORATION_RADIUS_CHUNKS * 2 +
                            EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS * 2 + 1) *
        static_cast<size_t>(EXPLORATION_RADIUS_CHUNKS * 2 +
                            EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS * 2 + 1) *
        static_cast<size_t>(WORLD_VERTICAL_CHUNKS);
    static_assert(MAX_EXPLORATION_SKY_AUTHORITY_CUBES < MAX_LOADED_CUBES);
    std::unordered_map<ChunkPos, uint8_t> skyAuthorityPriority;
    skyAuthorityPriority.reserve(std::min(MAX_EXPLORATION_SKY_AUTHORITY_CUBES, loadedCubeLimit_));
    for (const auto& [column, request] : explorationSkyPaths) {
        const auto plan = generator_.findColumnPlan(column);
        if (!plan) continue;

        const auto saved = savedSectionMasks.find(column);
        const VerticalSectionMask required = potentialSkyOccupancySections(
            *plan, saved == savedSectionMasks.end() ? nullptr : &saved->second,
            loweredSkyCutoffColumns.contains(column) ? std::optional(request.firstSection)
                                                     : std::nullopt);
        required.visitSetSections(request.firstSection, WORLD_MAX_CHUNK_Y, [&](int32_t section) {
            const ChunkPos support{column.x, section, column.z};
            auto [entry, inserted] = skyAuthorityPriority.try_emplace(support, request.priority);
            if (!inserted) entry->second = std::max(entry->second, request.priority);
            return true;
        });
    }
    // A diagnostic World may set a cap smaller than one complete authority
    // band. Keep its hard cap and fail closed rather than partially retaining a
    // path that could never authorize a mesh.
    if (skyAuthorityPriority.size() >= loadedCubeLimit_) skyAuthorityPriority.clear();

    auto coverageSnapshot = std::make_shared<ExactSurfaceCoverageSnapshot>();
    coverageSnapshot->nominalRadiusChunks = exactViewDistance;
    coverageSnapshot->requiredSections.assign(surfaceOwnershipRequirements.begin(),
                                              surfaceOwnershipRequirements.end());
    coverageSnapshot->floraRequiredSections.assign(floraOwnershipRequirements.begin(),
                                                   floraOwnershipRequirements.end());
    const auto sectionOrder = [](ChunkPos left, ChunkPos right) {
        if (left.x != right.x) return left.x < right.x;
        if (left.z != right.z) return left.z < right.z;
        return left.y < right.y;
    };
    std::sort(coverageSnapshot->requiredSections.begin(), coverageSnapshot->requiredSections.end(),
              sectionOrder);
    std::sort(coverageSnapshot->floraRequiredSections.begin(),
              coverageSnapshot->floraRequiredSections.end(), sectionOrder);
    std::sort(unresolvedSurfaceColumns.begin(), unresolvedSurfaceColumns.end(),
              [](ColumnPos left, ColumnPos right) {
                  if (left.x != right.x) return left.x < right.x;
                  return left.z < right.z;
              });
    unresolvedSurfaceColumns.erase(
        std::unique(unresolvedSurfaceColumns.begin(), unresolvedSurfaceColumns.end()),
        unresolvedSurfaceColumns.end());
    coverageSnapshot->unresolvedColumns = std::move(unresolvedSurfaceColumns);

    std::unordered_set<ChunkPos> previousMeshCandidates;
    {
        std::lock_guard<std::mutex> chunksLock(chunksMutex_);
        previousMeshCandidates = meshCandidateChunks_;
    }
    const size_t meshCandidateCapacity =
        std::min(MAX_MESH_RESIDENT_CUBES, loadedCubeLimit_ - skyAuthorityPriority.size());
    wantedSet = selectStableMeshCandidates(wantedPriority, previousMeshCandidates,
                                           {centerX, centerY, centerZ}, meshCandidateCapacity);

    std::unordered_set<ChunkPos> retained = wantedSet;
    retained.reserve(wantedSet.size() + skyAuthorityPriority.size());
    for (const auto& [support, priority] : skyAuthorityPriority) {
        (void)priority;
        retained.insert(support);
    }

    std::unordered_map<ChunkPos, uint8_t> haloPriority;
    haloPriority.reserve(wantedSet.size() * 2);
    for (ChunkPos pos : wantedSet) {
        for (int offsetY = -EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES;
             offsetY <= EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES; ++offsetY) {
            for (int offsetZ = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                 offsetZ <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++offsetZ) {
                for (int offsetX = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                     offsetX <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++offsetX) {
                    ChunkPos neighbor{pos.x + offsetX, pos.y + offsetY, pos.z + offsetZ};
                    if (!validChunkY(neighbor.y)) continue;
                    if (retained.contains(neighbor)) continue;
                    auto [iterator, inserted] =
                        haloPriority.try_emplace(neighbor, wantedPriority.at(pos));
                    if (!inserted)
                        iterator->second = std::max(iterator->second, wantedPriority.at(pos));
                }
            }
        }
    }
    const size_t haloBudget = loadedCubeLimit_ - retained.size();
    std::unordered_set<ChunkPos> previousMeshHalos;
    previousMeshHalos.reserve(previousMeshCandidates.size() * 2);
    for (ChunkPos position : previousMeshCandidates) {
        for (int offsetY = -EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES;
             offsetY <= EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES; ++offsetY) {
            for (int offsetZ = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                 offsetZ <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++offsetZ) {
                for (int offsetX = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                     offsetX <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++offsetX) {
                    const ChunkPos neighbor{position.x + offsetX, position.y + offsetY,
                                            position.z + offsetZ};
                    if (validChunkY(neighbor.y)) previousMeshHalos.insert(neighbor);
                }
            }
        }
    }
    const std::unordered_set<ChunkPos> selectedHalos = selectStableMeshCandidates(
        haloPriority, previousMeshHalos, {centerX, centerY, centerZ}, haloBudget);
    retained.reserve(retained.size() + selectedHalos.size());
    retained.insert(selectedHalos.begin(), selectedHalos.end());
    if (cancelIfStale()) return false;

    for (auto iterator = wantedSet.begin(); iterator != wantedSet.end();) {
        bool completeHalo = true;
        for (int offsetY = -EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES;
             offsetY <= EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES && completeHalo; ++offsetY) {
            for (int offsetZ = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                 offsetZ <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS && completeHalo;
                 ++offsetZ) {
                for (int offsetX = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                     offsetX <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++offsetX) {
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

    std::unordered_map<ChunkPos, uint8_t> retainedPriority;
    retainedPriority.reserve(retained.size());
    for (ChunkPos position : retained) {
        uint8_t priority = 0;
        if (const auto wanted = wantedPriority.find(position); wanted != wantedPriority.end())
            priority = std::max(priority, wanted->second);
        if (const auto support = skyAuthorityPriority.find(position);
            support != skyAuthorityPriority.end()) {
            priority = std::max(priority, support->second);
        }
        if (const auto halo = haloPriority.find(position); halo != haloPriority.end())
            priority = std::max(priority, halo->second);
        retainedPriority.emplace(position, priority);
    }

    std::unordered_map<ColumnPos, uint8_t> planCenterPriority;
    planCenterPriority.reserve(retained.size());
    for (ChunkPos cube : retained) {
        auto [iterator, inserted] =
            planCenterPriority.try_emplace({cube.x, cube.z}, retainedPriority.at(cube));
        if (!inserted) iterator->second = std::max(iterator->second, retainedPriority.at(cube));
    }

    std::unordered_map<ColumnPos, uint8_t> requestedPlanPriority;
    requestedPlanPriority.reserve(planCenterPriority.size() * 2 + 64);
    for (const auto& [center, priority] : planCenterPriority) {
        for (int apronZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             apronZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++apronZ) {
            for (int apronX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
                 apronX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++apronX) {
                const ColumnPos plan{center.x + apronX, center.z + apronZ};
                auto [iterator, inserted] = requestedPlanPriority.try_emplace(plan, priority);
                if (!inserted) iterator->second = std::max(iterator->second, priority);
            }
        }
    }
    if (cancelIfStale()) return false;
    planApronCenters_.fetch_add(planCenterPriority.size(), std::memory_order_relaxed);
    planApronExpansionAttempts_.fetch_add(planCenterPriority.size() *
                                              EXACT_STREAMING_PLAN_DEPENDENCY_COLUMN_COUNT,
                                          std::memory_order_relaxed);
    planApronCubeExpansionEquivalent_.fetch_add(
        retained.size() * EXACT_STREAMING_PLAN_DEPENDENCY_COLUMN_COUNT, std::memory_order_relaxed);

    std::vector<ChunkPos> loadOrder(retained.begin(), retained.end());
    std::sort(loadOrder.begin(), loadOrder.end(), [&](ChunkPos left, ChunkPos right) {
        const ChunkPos priorityCenter{centerX, centerY, centerZ};
        const uint64_t leftDistance = exactStreamingCubePriorityDistance(left, priorityCenter);
        const uint64_t rightDistance = exactStreamingCubePriorityDistance(right, priorityCenter);
        const int64_t leftPriority = exactStreamingTaskPriority(
            0, retainedPriority.at(left), leftDistance);
        const int64_t rightPriority = exactStreamingTaskPriority(
            0, retainedPriority.at(right), rightDistance);
        if (leftPriority != rightPriority) return leftPriority < rightPriority;
        if (left.x != right.x) return left.x > right.x;
        if (left.z != right.z) return left.z > right.z;
        return left.y > right.y;
    });
    std::vector<ColumnPos> planOrder;
    planOrder.reserve(requestedPlanPriority.size());
    for (const auto& [position, priority] : requestedPlanPriority) {
        (void)priority;
        planOrder.push_back(position);
    }
    std::sort(planOrder.begin(), planOrder.end(), [&](ColumnPos left, ColumnPos right) {
        const int64_t leftX = left.x - centerX;
        const int64_t leftZ = left.z - centerZ;
        const int64_t rightX = right.x - centerX;
        const int64_t rightZ = right.z - centerZ;
        const uint64_t leftDistance = static_cast<uint64_t>(leftX * leftX + leftZ * leftZ);
        const uint64_t rightDistance = static_cast<uint64_t>(rightX * rightX + rightZ * rightZ);
        const int64_t leftPriority =
            exactStreamingTaskPriority(0, requestedPlanPriority.at(left), leftDistance);
        const int64_t rightPriority =
            exactStreamingTaskPriority(0, requestedPlanPriority.at(right), rightDistance);
        if (leftPriority != rightPriority) return leftPriority < rightPriority;
        if (left.x != right.x) return left.x > right.x;
        return left.z > right.z;
    });

    // Snapshot only cubes with work or storage to preserve. The bounded
    // neighborhood probes and ordering then run without either world lock.
    std::vector<ChunkPos> priorRetentionCandidates;
    if (retained.size() < loadedCubeLimit_) {
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
    const size_t hysteresisBudget = loadedCubeLimit_ - retained.size();
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
            std::unordered_set<ColumnPos> changedManifestColumns;
            std::unordered_set<ColumnPos> publishedManifestColumns;
            if (skyManifestKnownColumns_ != manifestKnownColumns ||
                savedEditedSectionMasks_ != savedSectionMasks) {
                changedManifestColumns.reserve(
                    skyManifestKnownColumns_.size() + manifestKnownColumns.size() +
                    savedEditedSectionMasks_.size() + savedSectionMasks.size());
                for (ColumnPos column : skyManifestKnownColumns_) {
                    if (!manifestKnownColumns.contains(column)) {
                        changedManifestColumns.insert(column);
                    }
                }
                for (ColumnPos column : manifestKnownColumns) {
                    if (!skyManifestKnownColumns_.contains(column)) {
                        changedManifestColumns.insert(column);
                    }
                }
                for (const auto& [column, mask] : savedEditedSectionMasks_) {
                    const auto replacement = savedSectionMasks.find(column);
                    if (replacement == savedSectionMasks.end() || replacement->second != mask) {
                        changedManifestColumns.insert(column);
                    }
                }
                for (const auto& [column, mask] : savedSectionMasks) {
                    const auto previous = savedEditedSectionMasks_.find(column);
                    if (previous == savedEditedSectionMasks_.end() || previous->second != mask) {
                        changedManifestColumns.insert(column);
                    }
                }
                publishedManifestColumns.reserve(changedManifestColumns.size());
                for (ColumnPos column : changedManifestColumns) {
                    if (manifestKnownColumns.contains(column)) {
                        publishedManifestColumns.insert(column);
                    }
                }
            }
            skyManifestKnownColumns_ = std::move(manifestKnownColumns);
            savedEditedSectionMasks_ = std::move(savedSectionMasks);
            queueSavedSkyPublicationLocked(publishedManifestColumns);
            if (!changedManifestColumns.empty()) {
                for (const auto& [position, chunk] : chunks_) {
                    if (!chunk || !chunk->generated ||
                        !changedManifestColumns.contains({position.x, position.z}) ||
                        publishedManifestColumns.contains({position.x, position.z})) {
                        continue;
                    }
                    queueLightReconcile(position);
                }
            }
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
        coverageSnapshot->epoch = activeSetEpoch_;
        std::shared_ptr<const ExactSurfaceCoverageSnapshot> immutableCoverage = coverageSnapshot;
        std::atomic_store_explicit(&exactSurfaceCoverageSnapshot_, std::move(immutableCoverage),
                                   std::memory_order_release);
        retainedCubeCountForStats_ = retainedCubeCount;
        genBacklog_ = decltype(genBacklog_){};
        genBacklogSet_.clear();
        columnPlanBacklog_.clear();
        planDependents_.clear();
        missingPlanDependencies_.clear();
        exactPriorityByCube_ = std::move(retainedPriority);
        exactPriorityByPlan_ = std::move(requestedPlanPriority);
        exactPriorityCenter_ = {centerX, centerY, centerZ};
        // Work that remains inside the disk after camera movement keeps its
        // single-flight reservation, but its queued ThreadPool priority must
        // follow the new camera and epoch. Without this update an overlapping
        // cube that just became the camera column could remain buried behind
        // thousands of newer distant submissions until its old task started.
        if (genPool_) {
            for (auto& [position, pending] : pendingGenerations_) {
                const auto lane = exactPriorityByCube_.find(position);
                if (lane == exactPriorityByCube_.end()) {
                    (void)genPool_->cancelQueued(pending.handle);
                    continue;
                }
                const int64_t priority = exactStreamingTaskPriority(
                    activeSetEpoch_, lane->second,
                    exactStreamingCubePriorityDistance(position, exactPriorityCenter_));
                if (priority == pending.priority) continue;
                (void)genPool_->reprioritize(pending.handle, priority);
                pending.priority = priority;
            }
            for (auto& [position, pending] : pendingColumnPlans_) {
                const auto lane = exactPriorityByPlan_.find(position);
                if (lane == exactPriorityByPlan_.end()) {
                    (void)genPool_->cancelQueued(pending.handle);
                    continue;
                }
                const int64_t dx = position.x - exactPriorityCenter_.x;
                const int64_t dz = position.z - exactPriorityCenter_.z;
                const int64_t priority = exactStreamingTaskPriority(
                    activeSetEpoch_, lane->second, static_cast<uint64_t>(dx * dx + dz * dz));
                if (priority == pending.priority) continue;
                (void)genPool_->reprioritize(pending.handle, priority);
                pending.priority = priority;
            }
        }
        genBacklogSet_.reserve(loadOrder.size());
        planDependents_.reserve(exactPriorityByPlan_.size());
        missingPlanDependencies_.reserve(loadOrder.size());
        for (ColumnPos pos : planOrder) {
            if (!generator_.findColumnPlan(pos) && !columnPlansInFlight_.contains(pos)) {
                columnPlanBacklog_.push_back(pos);
            }
        }
        for (ChunkPos pos : dependencyCandidates)
            registerPlanDependenciesLocked(pos);
    }
    // The newly published retention set owns capacity before any replacement
    // job can insert. A camera jump therefore cannot overlap an old full set
    // with one additional generation queue of new cubes.
    unloadDistantChunks();
    pumpGeneration();
    activeSetBuildMs_.record(
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - rebuildStart)
            .count());
    return true;
}

void World::pumpGeneration() {
    if (!genPool_ || shuttingDown_.load()) return;
    std::vector<std::pair<ChunkPos, int64_t>> toSubmit;
    std::vector<std::pair<ColumnPos, int64_t>> plansToSubmit;
    bool notifyDrainedPlans = false;
    {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        const auto planBacklogLess = [&](ColumnPos left, ColumnPos right) {
            const auto priorityFor = [&](ColumnPos position) {
                const auto lane = exactPriorityByPlan_.find(position);
                const uint8_t priorityLane =
                    lane == exactPriorityByPlan_.end() ? 0 : lane->second;
                const int64_t dx = position.x - exactPriorityCenter_.x;
                const int64_t dz = position.z - exactPriorityCenter_.z;
                return exactStreamingTaskPriority(
                    0, priorityLane, static_cast<uint64_t>(dx * dx + dz * dz));
            };
            const int64_t leftPriority = priorityFor(left);
            const int64_t rightPriority = priorityFor(right);
            if (leftPriority != rightPriority) return leftPriority < rightPriority;
            if (left.x != right.x) return left.x > right.x;
            return left.z > right.z;
        };
        const auto insertPlanBacklog = [&](ColumnPos position) {
            const auto insertion = std::lower_bound(columnPlanBacklog_.begin(),
                                                    columnPlanBacklog_.end(), position,
                                                    planBacklogLess);
            columnPlanBacklog_.insert(insertion, position);
        };
        for (auto it = pendingGenerations_.begin(); it != pendingGenerations_.end();) {
            if (!it->second.future.valid() ||
                it->second.future.wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                const ChunkPos completed = it->first;
                if (it->second.future.valid()) {
                    try {
                        it->second.future.get();
                    } catch (const ThreadPool::TaskCanceled&) {
                        // A newer camera epoch evicted this queued task before
                        // its world-generation callable began.
                    } catch (const std::exception& error) {
                        latchGenerationFailure(std::string("Cube worker failed: ") + error.what());
                    } catch (...) {
                        latchGenerationFailure("Cube worker failed with an unknown exception");
                    }
                }
                generationsInFlight_.erase(completed);
                it = pendingGenerations_.erase(it);
                bool stillRequired = false;
                {
                    std::lock_guard<std::mutex> chunksLock(chunksMutex_);
                    stillRequired = retainedChunks_.contains(completed) &&
                                    !chunks_.contains(completed) &&
                                    !generationFailure().has_value();
                }
                if (stillRequired) registerPlanDependenciesLocked(completed);
            } else {
                ++it;
            }
        }
        for (auto it = pendingColumnPlans_.begin(); it != pendingColumnPlans_.end();) {
            if (!it->second.future.valid() ||
                it->second.future.wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                const ColumnPos completed = it->first;
                if (it->second.future.valid()) {
                    try {
                        it->second.future.get();
                    } catch (const ThreadPool::TaskCanceled&) {
                        // A newer camera epoch evicted this queued task before
                        // its ColumnPlan callable began.
                    } catch (const std::exception& error) {
                        latchGenerationFailure(std::string("Column plan worker failed: ") +
                                               error.what());
                    } catch (...) {
                        latchGenerationFailure(
                            "Column plan worker failed with an unknown exception");
                    }
                }
                columnPlansInFlight_.erase(completed);
                if (activeColumnPlanJobs_ > 0) --activeColumnPlanJobs_;
                const bool retryRequested = columnPlanRetries_.erase(completed) != 0;
                const bool requiredNow = exactPriorityByPlan_.contains(completed);
                const bool alreadyQueued =
                    std::ranges::find(columnPlanBacklog_, completed) != columnPlanBacklog_.end();
                const bool planAvailable = generator_.findColumnPlan(completed) != nullptr;
                const bool generationAvailable = generationFailure() == std::nullopt &&
                                                 !shuttingDown_.load();
                const ColumnPlanRetryPublicationAction retryAction =
                    columnPlanRetryPublicationAction(true, retryRequested, requiredNow,
                                                     alreadyQueued, planAvailable,
                                                     generationAvailable);
                if (retryAction == ColumnPlanRetryPublicationAction::REQUEUE) {
                    // Reaping is the first instant at which a replacement can
                    // be submitted without colliding with the old future.
                    // Preserve the same lane-and-distance order as the active
                    // set. Appending a deferred broad retry would otherwise
                    // put it at the high-priority end ahead of camera plans.
                    insertPlanBacklog(completed);
                }
                it = pendingColumnPlans_.erase(it);
            } else {
                ++it;
            }
        }
        const bool generationAvailable = !generationFailure().has_value();
        size_t currentColumnPlanJobs = 0;
        for (ColumnPos position : columnPlansInFlight_) {
            if (exactPriorityByPlan_.contains(position)) ++currentColumnPlanJobs;
        }
        while (generationAvailable && currentColumnPlanJobs < MAX_COLD_COLUMN_PLANS &&
               !columnPlanBacklog_.empty()) {
            const ColumnPos pos = columnPlanBacklog_.back();
            const auto lane = exactPriorityByPlan_.find(pos);
            const uint8_t priorityLane = lane == exactPriorityByPlan_.end() ? 0 : lane->second;
            if (currentColumnPlanJobs >= exactStreamingPlanSubmissionLimit(priorityLane)) break;
            columnPlanBacklog_.pop_back();
            if (generator_.findColumnPlan(pos) || columnPlansInFlight_.contains(pos)) continue;
            columnPlansInFlight_.insert(pos);
            ++activeColumnPlanJobs_;
            ++currentColumnPlanJobs;
            const int64_t dx = pos.x - exactPriorityCenter_.x;
            const int64_t dz = pos.z - exactPriorityCenter_.z;
            plansToSubmit.emplace_back(
                pos, exactStreamingTaskPriority(activeSetEpoch_, priorityLane,
                                                static_cast<uint64_t>(dx * dx + dz * dz)));
        }
        size_t currentGenerationJobs = 0;
        for (ChunkPos position : generationsInFlight_) {
            if (exactPriorityByCube_.contains(position)) ++currentGenerationJobs;
        }
        while (generationAvailable &&
               currentGenerationJobs < EXACT_GENERATION_SUBMISSION_LIMIT &&
               !genBacklog_.empty()) {
            const GenerationBacklogEntry entry = genBacklog_.top();
            const ChunkPos pos = entry.position;
            const auto lane = exactPriorityByCube_.find(pos);
            const uint8_t priorityLane = lane == exactPriorityByCube_.end() ? 0 : lane->second;
            if (currentGenerationJobs >= exactStreamingCubeSubmissionLimit(priorityLane)) {
                break;
            }
            genBacklog_.pop();
            genBacklogSet_.erase(pos);
            if (!generationsInFlight_.insert(pos).second) continue;
            ++currentGenerationJobs;
            toSubmit.emplace_back(pos, entry.priority);
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
    for (const auto& [pos, priority] : plansToSubmit)
        generateColumnPlanAsync(pos, priority);
    for (const auto& [pos, priority] : toSubmit)
        generateChunkAsync(pos, priority);
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
        .loadedCubeAdmissionsRejected =
            loadedCubeAdmissionsRejected_.load(std::memory_order_relaxed),
        .loadedCubeHighWater = loadedCubeHighWater_.load(std::memory_order_relaxed),
        .publicationLightSyncFloods = publicationLightSyncFloods_.load(std::memory_order_relaxed),
        .publicationLightDeferredCubes =
            publicationLightDeferredCubes_.load(std::memory_order_relaxed),
        .publicationLightSectionVisits =
            publicationLightSectionVisits_.load(std::memory_order_relaxed),
        .publicationLightDeferredQueue =
            publicationLightDeferredQueue_.load(std::memory_order_relaxed),
        .publicationLightMaxSyncFloods =
            publicationLightMaxSyncFloods_.load(std::memory_order_relaxed),
        .publicationLightMaxDeferredQueue =
            publicationLightMaxDeferredQueue_.load(std::memory_order_relaxed),
        .activeSetBuildMs = activeSetBuildMs_.value(),
    };
}

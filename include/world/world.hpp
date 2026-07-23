#pragma once

#include "common/ema.hpp"
#include "common/thread_pool.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"
#include "world/fluid.hpp"
#include "world/light_engine.hpp"
#include "world/mesh_snapshot.hpp"
#include "world/view_distance.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class SaveManager;

void sortChunksByDistance(std::vector<ChunkPos>& chunks, int64_t centerChunkX, int32_t centerChunkY,
                          int64_t centerChunkZ);
inline void sortChunksByDistance(std::vector<ChunkPos>& chunks, int64_t centerChunkX,
                                 int64_t centerChunkZ) {
    sortChunksByDistance(chunks, centerChunkX, 0, centerChunkZ);
}

// Water and air have identical derived-light transmission and emission.
// Their runtime fluid boundary changes do not invalidate opaque SSGI history;
// every other material transition retains the conservative reset contract.
constexpr bool blockEditResetsIndirectLighting(BlockType before, BlockType after) noexcept {
    if (before == after) return false;
    const bool beforeWaterOrAir = before == BlockType::WATER || before == BlockType::AIR;
    const bool afterWaterOrAir = after == BlockType::WATER || after == BlockType::AIR;
    return !(beforeWaterOrAir && afterWaterOrAir);
}

// Selects the bounded exact-mesh set in descending priority order. Exact
// ownership is horizontal, so every vertical section in a nearer column ranks
// before a farther column in the same lane. Existing candidates receive only
// a bounded horizontal-distance credit, so small camera movement does not
// churn effectively tied surfaces while materially nearer work can always
// displace stale residency at the hard cap.
std::unordered_set<ChunkPos>
selectStableMeshCandidates(const std::unordered_map<ChunkPos, uint8_t>& candidatePriorities,
                           const std::unordered_set<ChunkPos>& previousCandidates, ChunkPos center,
                           size_t capacity);

inline constexpr size_t MAX_INFLIGHT_GEN = 64;
inline constexpr size_t EXACT_GENERATION_WORKER_COUNT = 6;
inline constexpr size_t EXACT_LATENCY_WORKER_COUNT = 4;
static_assert(EXACT_LATENCY_WORKER_COUNT <= EXACT_GENERATION_WORKER_COUNT);
inline constexpr size_t EXACT_GENERATION_SUBMISSION_LIMIT = EXACT_GENERATION_WORKER_COUNT + 1;
static_assert(EXACT_GENERATION_SUBMISSION_LIMIT <= MAX_INFLIGHT_GEN);

enum class ColumnPlanCompletionAction : uint8_t {
    PUBLISH,
    REQUEUE,
    DROP,
};

constexpr ColumnPlanCompletionAction columnPlanCompletionAction(bool planAvailable,
                                                                bool shuttingDown, bool requiredNow,
                                                                bool alreadyQueued) {
    if (planAvailable) return ColumnPlanCompletionAction::PUBLISH;
    if (!shuttingDown && requiredNow && !alreadyQueued) return ColumnPlanCompletionAction::REQUEUE;
    return ColumnPlanCompletionAction::DROP;
}

enum class ColumnPlanRetryPublicationAction : uint8_t {
    HOLD,
    REQUEUE,
    DROP,
};

// A completed worker has not left the ThreadPool task until its future is
// ready. Retain its in-flight reservation through that interval so no second
// task can collide with the still-published future and leak the reservation.
constexpr ColumnPlanRetryPublicationAction columnPlanRetryPublicationAction(
    bool futureReaped, bool retryRequested, bool requiredNow, bool alreadyQueued,
    bool planAvailable, bool generationAvailable) noexcept {
    if (!futureReaped) return ColumnPlanRetryPublicationAction::HOLD;
    if (retryRequested && requiredNow && !alreadyQueued && !planAvailable &&
        generationAvailable) {
        return ColumnPlanRetryPublicationAction::REQUEUE;
    }
    return ColumnPlanRetryPublicationAction::DROP;
}
inline constexpr size_t EXACT_MESH_WORKER_COUNT = 4;
inline constexpr size_t MAX_LOADED_CUBES = 32768;
inline constexpr size_t MAX_MESH_RESIDENT_CUBES = 16384;
inline constexpr size_t MAX_COLD_COLUMN_PLANS = 4;
inline constexpr int MAX_EXACT_CUBIC_DISTANCE_CHUNKS = 32;
inline constexpr int EXPLORATION_RADIUS_CHUNKS = 6;
// Generator v4 enters only after the camera column and its mandatory mesh
// halo are exact and complete coarse coverage through the configured visible
// horizon is ready. A zero nominal radius still produces an active radius of
// one below, then the full exact disk expands asynchronously after entry.
inline constexpr int COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS = 0;
static_assert(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS <= MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
// A finalized spawn may occupy any block in its center chunk. Collision
// sweeps deliberately treat missing cubes as solid and probe beyond the
// player AABB, so entry separately waits for the complete one-cube halo that
// cold exact streaming already retains. This is a residency guarantee, not a
// wider nominal exact-generation radius.
inline constexpr int PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS = 1;
inline constexpr int PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES = 1;
// Exact mesh residency retains a one-column horizontal halo around the active
// surface set. Each retained cube then requires a five-by-five ColumnPlan
// dependency apron before exact generation may start. Startup authority
// prequeueing uses these same bounds so a plan cannot open a new hydrology
// owner after the world has been created.
inline constexpr int EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS = 1;
inline constexpr int EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES = 1;
inline constexpr int EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS = 2;
static_assert(PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS <=
              EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS);
static_assert(PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES <=
              EXACT_STREAMING_VERTICAL_MESH_HALO_CUBES);
inline constexpr size_t EXACT_STREAMING_PLAN_DEPENDENCY_COLUMN_COUNT =
    static_cast<size_t>(EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS * 2 + 1) *
    static_cast<size_t>(EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS * 2 + 1);

constexpr int boundedColdStartExactRadiusChunks(int requestedRadius) noexcept {
    return std::clamp(requestedRadius, 0, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
}

constexpr bool withinExactStreamingRadius(int dx, int dz, int radius) noexcept {
    return dx * dx + dz * dz <= radius * radius;
}

constexpr int exactStreamingActiveSetRadiusChunks(int exactViewDistance) noexcept {
    return std::clamp(exactViewDistance, 0, MAX_EXACT_CUBIC_DISTANCE_CHUNKS) + 1;
}

// Exact mesh ownership covers the complete active surface set, including the
// one-column movement and collision ring outside the nominal distance. The
// generated halo begins beyond this radius and remains nonmesh support.
constexpr int exactStreamingMeshRadiusChunks(int exactViewDistance) noexcept {
    return exactStreamingActiveSetRadiusChunks(exactViewDistance);
}

constexpr int exactStreamingPlanCoverageRadiusChunks(int exactViewDistance) noexcept {
    return exactStreamingActiveSetRadiusChunks(exactViewDistance) +
           EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS +
           EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
}

// Stable lanes order exact terrain before distance within one active-set
// epoch. Surface ownership beneath the camera must share gameplay priority
// even when creative flight places the camera many vertical sections above
// it. Complete required surfaces through the full exact disk use the protected
// lane so all of a cliff or water column arrives before work outside that
// disk. Flora keeps its smaller independent completion radius because upper
// solid tree geometry remains behind a separate atomic collision handoff.
inline constexpr uint8_t EXACT_STREAMING_SURFACE_PRIORITY_LANE = 3;
inline constexpr uint8_t EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE = 4;
inline constexpr uint8_t EXACT_STREAMING_FLORA_PRIORITY_LANE = 4;
inline constexpr uint8_t EXACT_STREAMING_EDITED_PRIORITY_LANE = 5;
inline constexpr uint8_t EXACT_STREAMING_EXPLORATION_PRIORITY_LANE = 6;
inline constexpr uint8_t EXACT_STREAMING_CAMERA_PRIORITY_LANE = 7;
static_assert(EXACT_STREAMING_SURFACE_PRIORITY_LANE < EXACT_STREAMING_FLORA_PRIORITY_LANE);
static_assert(EXACT_STREAMING_FLORA_PRIORITY_LANE < EXACT_STREAMING_EDITED_PRIORITY_LANE);
inline constexpr int EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS = 16;
inline constexpr int EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS =
    MAX_EXACT_CUBIC_DISTANCE_CHUNKS;

// Submitted work can no longer be reprioritized once a worker begins a cube
// or learned column query. Keep one physical exact worker uncommitted while
// only broad or medium work is available. A new camera epoch can therefore
// begin immediately instead of waiting behind an entire stale worker wave.
constexpr size_t exactStreamingPlanSubmissionLimit(uint8_t lane) noexcept {
    if (lane >= EXACT_STREAMING_CAMERA_PRIORITY_LANE) return MAX_COLD_COLUMN_PLANS;
    if (lane >= EXACT_STREAMING_EXPLORATION_PRIORITY_LANE) return 3;
    return 2;
}

constexpr size_t exactStreamingCubeSubmissionLimit(uint8_t lane) noexcept {
    if (lane >= EXACT_STREAMING_CAMERA_PRIORITY_LANE)
        return EXACT_GENERATION_SUBMISSION_LIMIT;
    if (lane >= EXACT_STREAMING_EXPLORATION_PRIORITY_LANE) return 4;
    return 3;
}

constexpr uint8_t exactStreamingSurfacePriorityLane(int dx, int dz) noexcept {
    if (dx == 0 && dz == 0) return EXACT_STREAMING_CAMERA_PRIORITY_LANE;
    if (withinExactStreamingRadius(dx, dz, EXPLORATION_RADIUS_CHUNKS)) {
        return EXACT_STREAMING_EXPLORATION_PRIORITY_LANE;
    }
    if (withinExactStreamingRadius(dx, dz,
                                   EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS)) {
        return EXACT_STREAMING_EDITED_PRIORITY_LANE;
    }
    return EXACT_STREAMING_SURFACE_PRIORITY_LANE;
}

constexpr uint8_t exactStreamingPrimarySurfacePriorityLane(int dx, int dz) noexcept {
    return std::max(EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE,
                    exactStreamingSurfacePriorityLane(dx, dz));
}

// Near and middle-distance flora completes before broad primary surfaces at
// the edge of the exact disk. Camera and exploration work retain their higher
// collision-sensitive lanes, while farther exact flora remains optional
// behind the independently drawable far attachment.
constexpr uint8_t exactStreamingFloraPriorityLane(int dx, int dz) noexcept {
    const uint8_t surfaceLane = exactStreamingSurfacePriorityLane(dx, dz);
    if (surfaceLane >= EXACT_STREAMING_EXPLORATION_PRIORITY_LANE) return surfaceLane;
    if (!withinExactStreamingRadius(dx, dz,
                                    EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS)) {
        return EXACT_STREAMING_SURFACE_PRIORITY_LANE;
    }
    return EXACT_STREAMING_FLORA_PRIORITY_LANE;
}

inline constexpr int EXPLORATION_VERTICAL_RADIUS_CUBES = 4;
inline constexpr int HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS = 2;
inline constexpr int VERTICAL_UNLOAD_HYSTERESIS_CUBES = 1;
inline constexpr double EXACT_MESH_RESIDENCY_HYSTERESIS_CHUNKS =
    HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS;
inline constexpr size_t COLUMN_PLAN_REBUILD_BATCH = 128;
inline constexpr size_t COLUMN_PLAN_REBUILD_COOLDOWN_TICKS = 4;

// Exact streaming uses stable lanes before distance. A newer active-set epoch
// outranks queued work from an old camera position, the exploration and
// collision band outranks the broad surface disk, and distance orders work
// within a lane. The low 32 bits are sufficient for the bounded exact radius.
inline constexpr int64_t exactStreamingTaskPriority(uint64_t epoch, uint8_t lane,
                                                    uint64_t distanceSquared) {
    constexpr uint64_t EPOCH_MASK = (uint64_t{1} << 27U) - 1U;
    constexpr uint64_t DISTANCE_MASK = (uint64_t{1} << 32U) - 1U;
    const uint64_t boundedDistance = std::min(distanceSquared, DISTANCE_MASK);
    const uint64_t packed = ((epoch & EPOCH_MASK) << 36U) |
                            (static_cast<uint64_t>(lane & 0x0FU) << 32U) |
                            (DISTANCE_MASK - boundedDistance);
    return static_cast<int64_t>(packed);
}

// Required terrain is a horizontal streaming contract. Pack the complete
// bounded vertical separation below one unit of horizontal squared distance,
// so a tall cliff or cave section in a nearer column cannot sit behind a flat
// section at the far edge of the exact disk merely because the camera is
// airborne. The result remains inside the task priority's low 32 bits for
// every retained exact and halo cube.
constexpr uint64_t exactStreamingCubePriorityDistance(ChunkPos position,
                                                      ChunkPos center) noexcept {
    constexpr uint64_t VERTICAL_SPAN =
        uint64_t{2} * WORLD_VERTICAL_CHUNKS * WORLD_VERTICAL_CHUNKS + 1U;
    constexpr uint64_t DISTANCE_MASK = (uint64_t{1} << 32U) - 1U;
    const auto boundedMagnitude = [](int64_t value) -> uint64_t {
        const uint64_t magnitude = value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1U
                                             : static_cast<uint64_t>(value);
        return std::min<uint64_t>(magnitude, 65'535U);
    };
    const uint64_t dx = boundedMagnitude(position.x - center.x);
    const uint64_t dy = boundedMagnitude(static_cast<int64_t>(position.y) - center.y);
    const uint64_t dz = boundedMagnitude(position.z - center.z);
    const uint64_t maximumHorizontal = DISTANCE_MASK / VERTICAL_SPAN;
    const uint64_t horizontalSquared = std::min(dx * dx + dz * dz, maximumHorizontal);
    const uint64_t verticalSquared = std::min(uint64_t{2} * dy * dy, VERTICAL_SPAN - 1U);
    return horizontalSquared * VERTICAL_SPAN + verticalSquared;
}

constexpr int64_t exactPublicationLightPriority(ChunkPos position, ChunkPos center) noexcept {
    const int64_t dx = position.x - center.x;
    const int64_t dy = static_cast<int64_t>(position.y) - center.y;
    const int64_t dz = position.z - center.z;
    const auto withinRadius = [dx, dz](int64_t radius) {
        return dx >= -radius && dx <= radius && dz >= -radius && dz <= radius &&
               dx * dx + dz * dz <= radius * radius;
    };
    uint8_t lane = EXACT_STREAMING_SURFACE_PRIORITY_LANE;
    if (dx == 0 && dz == 0) {
        lane = EXACT_STREAMING_CAMERA_PRIORITY_LANE;
    } else if (withinRadius(EXPLORATION_RADIUS_CHUNKS)) {
        lane = EXACT_STREAMING_EXPLORATION_PRIORITY_LANE;
    } else if (withinRadius(EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS)) {
        lane = EXACT_STREAMING_EDITED_PRIORITY_LANE;
    }
    const auto boundedMagnitude = [](int64_t value) -> uint64_t {
        const uint64_t magnitude = value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1U
                                             : static_cast<uint64_t>(value);
        return std::min<uint64_t>(magnitude, 65'535U);
    };
    const uint64_t boundedX = boundedMagnitude(dx);
    const uint64_t boundedY = boundedMagnitude(dy);
    const uint64_t boundedZ = boundedMagnitude(dz);
    return exactStreamingTaskPriority(
        0, lane, boundedX * boundedX + boundedZ * boundedZ + boundedY * boundedY * 2U);
}

struct StreamingWorkStats {
    uint64_t activeSetRebuilds = 0;
    uint64_t planApronCenters = 0;
    uint64_t planApronExpansionAttempts = 0;
    uint64_t planApronCubeExpansionEquivalent = 0;
    uint64_t completedColumnPlans = 0;
    uint64_t planDependentChecks = 0;
    // Retained-cube visits the previous completion-wide scan would perform.
    uint64_t fullRetainedScanEquivalent = 0;
    uint64_t activeSetRebuildNotifications = 0;
    uint64_t hysteresisRetainedCubes = 0;
    uint64_t activeSetRequests = 0;
    uint64_t activeSetRequestsCoalesced = 0;
    uint64_t activeSetBuildsCanceled = 0;
    uint64_t loadedCubeAdmissionsRejected = 0;
    size_t loadedCubeHighWater = 0;
    uint64_t publicationLightSyncFloods = 0;
    uint64_t publicationLightDeferredCubes = 0;
    uint64_t publicationLightSectionVisits = 0;
    size_t publicationLightDeferredQueue = 0;
    size_t publicationLightMaxSyncFloods = 0;
    size_t publicationLightMaxDeferredQueue = 0;
    float activeSetBuildMs = 0.0F;
};

// Immutable pre-cap description of the exact geometry that the renderer is
// expected to replace. Terrain and water can hand off after the narrow surface
// set is ready. Far canopy remains authoritative until every section that can
// contain exact tree geometry is revision-matched as well.
struct ExactSurfaceCoverageSnapshot {
    uint64_t epoch = 0;
    int nominalRadiusChunks = 0;
    std::vector<ChunkPos> requiredSections;
    std::vector<ChunkPos> floraRequiredSections;
    std::vector<ColumnPos> unresolvedColumns;
};

// Immutable visual-to-collision handoff for exact chunk sections. The
// renderer publishes only sections whose exact geometry is currently drawn.
// Matching the active coverage epoch prevents a delayed frame from assigning
// collision authority to geometry from an older streaming view.
struct ExactCollisionOwnershipSnapshot {
    uint64_t coverageEpoch = 0;
    std::unordered_set<ChunkPos> sections;

    [[nodiscard]] bool owns(ChunkPos section) const noexcept {
        return sections.contains(section);
    }
};
inline constexpr size_t MAX_FLUID_RESUME_CUBES_PER_FRAME = 64;
inline constexpr size_t MAX_FLUID_FRONTIER_RESUMES_PER_FRAME = 256;
inline constexpr size_t MAX_FLUID_FRONTIER_RESUMES_PER_CUBE = 16;

// Optional engine-owned projection applied to a cube loaded from disk before
// the cube, its derived light, or its mesh can be published. `apply` must be
// deterministic, bounded to the supplied cube, nonblocking, and safe on a
// generation worker. It returns the immutable authority revision it read.
// `currentRevision` must be a lock-free read. World retries a stale transform
// before insertion, but never owns the gameplay sidecar behind the callbacks.
struct SavedChunkProjection {
    std::function<uint64_t(Chunk&)> apply;
    std::function<uint64_t()> currentRevision;

    explicit operator bool() const noexcept { return apply && currentRevision; }
};

class World : public FluidWorldAccess {
public:
    explicit World(uint64_t seed, int viewDistance = DEFAULT_RENDER_DISTANCE_CHUNKS,
                   size_t loadedCubeLimit = MAX_LOADED_CUBES, GenerationSettings generation = {});
    World(uint64_t seed, int viewDistance, size_t loadedCubeLimit,
          std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
          GenerationSettings generation = {});
    ~World();

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) = delete;
    World& operator=(World&&) = delete;

    std::shared_ptr<Chunk> getChunk(ChunkPos pos);
    std::shared_ptr<Chunk> getChunk(int64_t chunkX, int32_t chunkY, int64_t chunkZ) {
        return getChunk(ChunkPos{chunkX, chunkY, chunkZ});
    }
    // Compatibility for callers deliberately addressing the cube containing Y=0.
    std::shared_ptr<Chunk> getChunk(int64_t chunkX, int64_t chunkZ) {
        return getChunk(ChunkPos{chunkX, 0, chunkZ});
    }

    // Install only before a SaveManager or streaming request can load cubes.
    // The callbacks must own everything they capture for at least this
    // World's lifetime; capturing EngineState by raw pointer is invalid.
    void setSavedChunkProjection(SavedChunkProjection projection);

    BlockType getBlock(int64_t x, int32_t y, int64_t z);
    // A missing cube is distinct from air for targeting and other queries
    // that must never make absent world data interactive.
    std::optional<BlockType> findBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    BlockType getBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    uint8_t getPackedLightIfLoaded(int64_t x, int32_t y, int64_t z) const;
    // Samples a bounded gameplay-object batch under one simulation-tick map
    // lock. Renderers consume the cached results and never lock World.
    void samplePackedLightsIfLoaded(std::span<const BlockPos> positions,
                                    std::span<uint8_t> output) const;
    // A renderer-published exact section is collision authority, and a missing
    // published cube stays conservatively closed. Until the matching coverage
    // epoch publishes a section, collision follows the immutable ColumnPlan
    // surface and water proxy instead of exposing partially loaded cubes or
    // placing an invisible closed-cube wall over visible far terrain.
    BlockType getCollisionBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    // Exact-owned water uses its mutable runtime level. Unowned sections use
    // the canonical plan water plane so swimming agrees with the visible far
    // surface without forcing or trusting a partially loaded cube.
    float getCollisionFluidHeightIfLoaded(int64_t x, int32_t y, int64_t z) const;
    bool isChunkLoaded(ChunkPos pos) const;
    bool shouldMeshChunk(ChunkPos pos) const;
    // A gameplay edit is accepted only when its cell is resident and the
    // requested block or water state differs from the published cell.
    [[nodiscard]] bool trySetBlock(int64_t x, int32_t y, int64_t z, BlockType type);
    void setBlock(int64_t x, int32_t y, int64_t z, BlockType type);

    FluidCell readFluidCell(FluidPos position) const override;
    void writeWater(FluidPos position, FluidState state) override;
    void removeWater(FluidPos position) override;
    size_t tickFluids(double elapsedSeconds);
    float getFluidHeightIfLoaded(int64_t x, int32_t y, int64_t z) const;
    size_t getPendingFluidCount() const;
    uint64_t getDroppedFluidUpdateCount() const;
    uint64_t getDroppedFluidFrontierCount() const;

    double getTerrainHeight(int64_t x, int64_t z) const;
    Biome getBiome(int64_t x, int64_t z) const;
    worldgen::SurfaceSample sampleSurface(int64_t x, int64_t z) const;
    std::optional<worldgen::SurfaceSample> findSurfaceSample(int64_t x, int64_t z) const;
    // Highest opaque block in the loaded cubic column, or nullopt if no cube
    // in the column is loaded. Weather uses this without forcing generation.
    std::optional<int> surfaceHeightIfLoaded(int64_t x, int64_t z) const;
    // Highest loaded opaque-block top or water surface in the cubic column.
    // Fractional flowing-water levels are preserved without forcing generation.
    std::optional<float> strikeSurfaceHeightIfLoaded(int64_t x, int64_t z) const;
    size_t cachedColumnPlanCount() const { return generator_.cachedColumnPlanCount(); }

    std::vector<std::shared_ptr<Chunk>> getLoadedChunks() const;
    std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> getLoadedSnapshot() const;
    std::shared_ptr<const std::unordered_set<ChunkPos>> getMeshCandidateSnapshot() const;
    std::shared_ptr<const ExactSurfaceCoverageSnapshot> getExactSurfaceCoverageSnapshot() const;
    std::shared_ptr<const ExactCollisionOwnershipSnapshot>
    getExactCollisionOwnershipSnapshot() const;
    // Render publication is logically external residency state, so this is a
    // const operation on World. It copies the supplied set into one immutable
    // snapshot and rejects a stale coverage epoch without disturbing the last
    // valid publication. Public visibility also gives physics tests the exact
    // same handoff semantics as the renderer.
    [[nodiscard]] bool
    publishExactCollisionOwnership(uint64_t coverageEpoch,
                                   std::span<const ChunkPos> sections) const;
    void publishLoadedSnapshot();
    bool snapshotForMeshing(ChunkPos pos, MeshSnapshot& out) const;
    std::vector<std::shared_ptr<Chunk>> getDirtyChunks();
    void markChunkMeshed(ChunkPos pos);
    void markChunkMeshed(int64_t chunkX, int32_t chunkY, int64_t chunkZ) {
        markChunkMeshed(ChunkPos{chunkX, chunkY, chunkZ});
    }
    void markChunkMeshed(int64_t chunkX, int64_t chunkZ) {
        markChunkMeshed(ChunkPos{chunkX, 0, chunkZ});
    }

    uint64_t getSeed() const { return seed_; }
    const GenerationSettings& getGenerationSettings() const { return generation_; }
    // Process-local identity lets renderer-owned residency state distinguish
    // separate Worlds that intentionally share a seed and generator identity.
    uint64_t instanceId() const { return instanceId_; }
    // Monotonic, lock-free invalidation for temporal render histories. It
    // advances after a visible derived-light field changes.
    uint64_t lightingRevision() const { return lightingRevision_.load(std::memory_order_acquire); }
    int getViewDistance() const { return viewDistance_.load(std::memory_order_relaxed); }
    int getExactViewDistance() const {
        return std::min({getViewDistance(), MAX_EXACT_CUBIC_DISTANCE_CHUNKS,
                         exactStreamingDistance_.load(std::memory_order_relaxed)});
    }
    void setViewDistance(int distance);
    // The render-distance horizon remains unchanged. This cap only bounds
    // mutable cubic streaming while a v4 safe spawn is being established.
    void setExactStreamingDistance(int distance);

    void updatePlayerPosition(int64_t playerX, int32_t playerY, int64_t playerZ);
    void updatePlayerPosition(int64_t playerX, int64_t playerZ) {
        updatePlayerPosition(playerX, SEA_LEVEL, playerZ);
    }
    // Reconcile packed skylight and block light across all six cube faces,
    // bounded so a simulation tick cannot stall on a large lighting update.
    void reconcileLight(int budgetCubes);

    void unloadDistantChunks();
    bool saveModifiedChunks();
    void generateAroundPlayer(int64_t playerX, int32_t playerY, int64_t playerZ);
    void generateAroundPlayer(int64_t playerX, int64_t playerZ) {
        generateAroundPlayer(playerX, SEA_LEVEL, playerZ);
    }
    void pumpGeneration();

    size_t getPendingChunkCount() const;
    size_t getLoadedChunkCount() const;
    StreamingWorkStats getStreamingWorkStats() const;
    float averageGenMs() const { return genMs_.value(); }
    void setSaveManager(SaveManager* saveManager);

    const ChunkGenerator& generator() const { return generator_; }
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext() const {
        return generationContext_;
    }
    // Startup safety follows the nominal circular entry footprint. The active
    // set retains one additional chunk for movement, then mesh and plan work
    // add their own bounded dependency halos.
    [[nodiscard]] bool exactSpawnBandReady(int64_t worldX, int32_t worldY, int64_t worldZ,
                                           int radiusChunks = 1) const;
    // Spawn certification proves canonical support and headroom in the
    // nominal column. Gameplay entry additionally requires this already
    // retained local collision neighborhood so the closed missing-cube
    // fallback cannot become an invisible wall around the player.
    [[nodiscard]] bool playableSpawnCollisionReady(int64_t worldX, int32_t worldY,
                                                    int64_t worldZ) const;
    [[nodiscard]] std::optional<Vec3> safeSpawnFromReadyPlans(int64_t worldX, int64_t worldZ,
                                                              int radiusChunks = 1) const;
    std::optional<std::string> generationFailure() const;
    bool retryGeneration();

private:
    uint64_t seed_;
    GenerationSettings generation_;
    uint64_t instanceId_;
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext_;
    mutable std::mutex generationFailureMutex_;
    std::optional<std::string> generationFailure_;
    std::atomic<int> viewDistance_;
    std::atomic<int> exactStreamingDistance_{MAX_EXACT_CUBIC_DISTANCE_CHUNKS};

    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> chunks_;
    // One bit per resident vertical section, guarded by chunksMutex_. Mesh
    // snapshots compare this with sparse generated and saved occupancy
    // authority without repeatedly probing the chunk map while workers wait
    // on the world lock.
    std::unordered_map<ColumnPos, VerticalSectionMask> loadedSectionMasks_;
    mutable std::mutex chunksMutex_;
    std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> loadedSnapshot_;
    std::shared_ptr<const std::unordered_set<ChunkPos>> meshCandidateSnapshot_;
    std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactSurfaceCoverageSnapshot_;
    mutable std::mutex exactCollisionOwnershipPublicationMutex_;
    mutable std::shared_ptr<const ExactCollisionOwnershipSnapshot>
        exactCollisionOwnershipSnapshot_;
    std::atomic<bool> loadedSnapshotDirty_{true};
    std::atomic<size_t> loadedCubeCount_{0};
    const size_t loadedCubeLimit_;
    std::atomic<size_t> loadedCubeHighWater_{0};
    std::atomic<uint64_t> loadedCubeAdmissionsRejected_{0};

    struct SkyColumnKey {
        int64_t x = 0;
        int64_t z = 0;
        constexpr bool operator==(const SkyColumnKey&) const = default;
    };
    struct SkyColumnKeyHash {
        size_t operator()(const SkyColumnKey& key) const noexcept {
            size_t seed = world_coord::mix(0, static_cast<uint64_t>(key.x));
            return world_coord::mix(seed, static_cast<uint64_t>(key.z));
        }
    };
    // Generated density uses immutable plan cutoffs. Loaded structures and
    // edits can extend or replace that top, so only affected block columns
    // retain an override for mesh snapshots.
    std::unordered_map<SkyColumnKey, int32_t, SkyColumnKeyHash> skyCutoffOverrides_;
    std::unordered_set<ColumnPos> skyOverrideChunkColumns_;
    // Removing a planned surface can move the real cutoff into otherwise
    // unexposed density below it. Such a column conservatively retains the
    // contiguous exploration hull until it unloads; untouched columns keep
    // the sparse exposed-section proof.
    std::unordered_set<ColumnPos> loweredSkyCutoffColumns_;
    // Visible save-manifest authority prevents an unloaded edited roof above
    // generated terrain from being treated as proven open sky. Columns in the
    // known set have completed a manifest lookup, normally through the bulk
    // active-set path. The masks retain every edited section because a large
    // gap between two saved cubes is proven empty and must not be generated
    // merely to establish skylight authority.
    std::unordered_set<ColumnPos> skyManifestKnownColumns_;
    std::unordered_map<ColumnPos, VerticalSectionMask> savedEditedSectionMasks_;

    SaveManager* saveManager_ = nullptr; // non-owning; EngineState destroys it after World
    ChunkGenerator generator_;

    int64_t playerChunkX_ = 0;
    int32_t playerChunkY_ = SEA_LEVEL / CHUNK_EDGE;
    int64_t playerChunkZ_ = 0;
    bool hasPlayerChunk_ = false;
    size_t activeSetRebuildCooldownTicks_ = 0; // main simulation tick only

    struct ActiveSetRequest {
        int64_t playerX = 0;
        int32_t playerY = SEA_LEVEL;
        int64_t playerZ = 0;
        int viewDistance = DEFAULT_RENDER_DISTANCE_CHUNKS;
        uint64_t id = 0;
    };
    std::optional<ActiveSetRequest> pendingActiveSetRequest_;
    std::thread activeSetThread_;
    mutable std::mutex activeSetRequestMutex_;
    std::condition_variable activeSetRequestCv_;
    std::mutex activeSetBuildMutex_;
    bool stopActiveSetThread_ = false; // guarded by activeSetRequestMutex_
    std::atomic<bool> activeSetWorkPending_{false};
    std::atomic<uint64_t> latestActiveSetRequestId_{0};
    uint64_t nextActiveSetRequestId_ = 0; // guarded by activeSetRequestMutex_

    std::unique_ptr<ThreadPool> genPool_;
    struct PendingPoolTask {
        std::future<void> future;
        ThreadPool::TaskHandle handle;
        int64_t priority = 0;
    };
    std::unordered_map<ChunkPos, PendingPoolTask> pendingGenerations_;
    std::unordered_set<ChunkPos> generationsInFlight_;
    std::unordered_map<ColumnPos, PendingPoolTask> pendingColumnPlans_;
    std::unordered_set<ColumnPos> columnPlansInFlight_;
    // A deferred worker records its retry here, but remains in flight until
    // pumpGeneration observes the completed future. Publishing the retry any
    // earlier permits a concurrent pump to reserve a second task that the
    // still-present future rejects, permanently leaking a plan-worker slot.
    std::unordered_set<ColumnPos> columnPlanRetries_;
    struct GenerationBacklogEntry {
        ChunkPos position;
        int64_t priority = 0;
    };
    struct GenerationBacklogLater {
        bool operator()(const GenerationBacklogEntry& left,
                        const GenerationBacklogEntry& right) const {
            if (left.priority != right.priority) return left.priority < right.priority;
            if (left.position.x != right.position.x) return left.position.x > right.position.x;
            if (left.position.z != right.position.z) return left.position.z > right.position.z;
            return left.position.y > right.position.y;
        }
    };
    std::priority_queue<GenerationBacklogEntry, std::vector<GenerationBacklogEntry>,
                        GenerationBacklogLater>
        genBacklog_;
    std::unordered_set<ChunkPos> genBacklogSet_;
    std::vector<ColumnPos> columnPlanBacklog_;
    std::unordered_map<ChunkPos, uint8_t> exactPriorityByCube_;
    std::unordered_map<ColumnPos, uint8_t> exactPriorityByPlan_;
    ChunkPos exactPriorityCenter_{};
    enum class PlanDependencyKind : uint8_t { OWN_PLAN, EXPOSED_APRON };
    struct PlanDependent {
        ChunkPos pos;
        uint64_t activeSetEpoch = 0;
        PlanDependencyKind kind = PlanDependencyKind::OWN_PLAN;
    };
    std::unordered_map<ColumnPos, std::vector<PlanDependent>> planDependents_;
    std::unordered_map<ChunkPos, uint8_t> missingPlanDependencies_;
    uint64_t activeSetEpoch_ = 0;                      // guarded by pendingMutex_
    size_t completedPlansSinceRebuild_ = 0;            // guarded by pendingMutex_
    size_t retainedCubeCountForStats_ = 0;             // guarded by pendingMutex_
    std::unordered_set<ChunkPos> retainedChunks_;      // guarded by chunksMutex_
    std::unordered_set<ChunkPos> meshCandidateChunks_; // guarded by chunksMutex_
    std::atomic<bool> shuttingDown_{false};
    std::atomic<bool> columnPlansChanged_{false};
    size_t activeColumnPlanJobs_ = 0; // guarded by pendingMutex_
    mutable std::mutex pendingMutex_;
    std::atomic<uint64_t> activeSetRebuilds_{0};
    std::atomic<uint64_t> planApronCenters_{0};
    std::atomic<uint64_t> planApronExpansionAttempts_{0};
    std::atomic<uint64_t> planApronCubeExpansionEquivalent_{0};
    std::atomic<uint64_t> completedColumnPlans_{0};
    std::atomic<uint64_t> planDependentChecks_{0};
    std::atomic<uint64_t> fullRetainedScanEquivalent_{0};
    std::atomic<uint64_t> activeSetRebuildNotifications_{0};
    std::atomic<uint64_t> hysteresisRetainedCubes_{0};
    std::atomic<uint64_t> activeSetRequests_{0};
    std::atomic<uint64_t> activeSetRequestsCoalesced_{0};
    std::atomic<uint64_t> activeSetBuildsCanceled_{0};
    AtomicEmaMs activeSetBuildMs_;
    AtomicEmaMs genMs_;
    FluidScheduler fluidScheduler_;
    std::deque<ChunkPos> fluidResumeQueue_;          // guarded by pendingMutex_
    std::unordered_set<ChunkPos> fluidResumeQueued_; // guarded by pendingMutex_

    // Cubes whose derived packed lighting may be stale because a neighbor
    // loaded, unloaded, or changed. The queue is deduplicated and drained on
    // the tick thread. lightMutex_ is never held while acquiring chunksMutex_.
    std::vector<ChunkPos> lightQueue_;
    std::unordered_set<ChunkPos> lightQueued_;
    // Deep propagation from player edits drains ahead of streaming churn.
    // The shared queue pops newest-first and an already-queued position keeps
    // its buried slot, so an edit's spread could otherwise starve for seconds
    // while generation keeps queueing fresher entries on top of it.
    std::vector<ChunkPos> editLightQueue_;
    std::unordered_set<ChunkPos> editLightQueued_;
    // Publication lighting uses per-chunk queue bits and storage reserved at
    // World construction. Generation therefore allocates no worklist nodes
    // while holding chunksMutex_. A pending cube cannot enter a mesh snapshot.
    struct PublicationLightQueueEntry {
        ChunkPos position;
        uint64_t token = 0;
        int64_t priority = 0;
    };
    std::vector<PublicationLightQueueEntry> publicationLightQueue_;
    size_t publicationLightQueueHead_ = 0;
    uint64_t nextPublicationLightQueueToken_ = 0;
    mutable std::mutex lightMutex_;
    std::atomic<uint64_t> lightingRevision_{1};
    std::atomic<uint64_t> publicationLightSyncFloods_{0};
    std::atomic<uint64_t> publicationLightDeferredCubes_{0};
    std::atomic<uint64_t> publicationLightSectionVisits_{0};
    std::atomic<size_t> publicationLightDeferredQueue_{0};
    std::atomic<size_t> publicationLightMaxSyncFloods_{0};
    std::atomic<size_t> publicationLightMaxDeferredQueue_{0};
    std::atomic<int64_t> lightPriorityCenterX_{0};
    std::atomic<int32_t> lightPriorityCenterY_{SEA_LEVEL / CHUNK_EDGE};
    std::atomic<int64_t> lightPriorityCenterZ_{0};
    SavedChunkProjection savedChunkProjection_;

    // Fluid updates run up to 1,024 cells per tick, so their relight must
    // stay on the budgeted reconcile queue. Player and furnace edits are a
    // handful per tick and are remeshed by the render thread before the next
    // tick, so they flood synchronously under the same lock as the block write.
    enum class LightUrgency : uint8_t { DEFERRED, IMMEDIATE };

    using LightColumnPlans = std::array<std::shared_ptr<const ColumnPlan>, 9>;

    struct SkyCutoffSectionRange {
        int32_t first = WORLD_MAX_CHUNK_Y + 1;
        int32_t last = WORLD_MIN_CHUNK_Y - 1;

        [[nodiscard]] bool changed() const noexcept { return first <= last; }
        void include(int32_t oldCutoff, int32_t newCutoff) noexcept;
        void merge(const SkyCutoffSectionRange& other) noexcept;
    };

    static constexpr size_t PUBLICATION_LIGHT_SYNC_FLOOD_CAP = 32;

    void queueLightReconcile(ChunkPos pos, LightUrgency urgency = LightUrgency::DEFERRED);
    void queueFaceNeighbors(ChunkPos pos);
    void ensureSavedSkyAuthority(ColumnPos column);
    LightColumnPlans findLightColumnPlans(ChunkPos pos) const;
    LightEngine::SkyLightSeedColumns skyLightSeedsLocked(ChunkPos pos,
                                                         const ColumnPlan* plan) const;
    LightEngine::SkyLightSeedColumns
    skyLightSeedsForMaskLocked(ChunkPos pos, const ColumnPlan* plan,
                               const VerticalSectionMask& loadedMask) const;
    LightEngine::FloodResult initializeChunkLightLocked(ChunkPos pos, const ColumnPlan* plan);
    void queuePublicationLightLocked(ChunkPos pos);
    void reprioritizePublicationLightLocked(ChunkPos center);
    size_t settleChunkPublicationLightLocked(ChunkPos pos,
                                             const LightEngine::FloodResult& initialFlood,
                                             const VerticalSectionMask& previousColumnMask,
                                             const LightColumnPlans& columnPlans,
                                             const SkyCutoffSectionRange& skyCutoffChange,
                                             size_t floodCap);
    size_t drainPublicationLightLocked(ChunkPos first, const LightColumnPlans& columnPlans,
                                       size_t floodBudget);
    bool reconcileCubeLocked(ChunkPos pos, const ColumnPlan* plan, LightUrgency urgency);

    // A single edit's light reaches at most one cube in each direction (max
    // level 15, one step per block, 16-block edge), so its whole affected
    // neighborhood is a subset of the 3x3x3 around the edit. Drain the IMMEDIATE
    // queue to its fixed point synchronously under the block-write lock, using
    // the nine prefetched column plans, so adjacent cubes never wait a tick to
    // relight. The cap is a pathological safety valve; any residue falls back to
    // the per-tick reconcile.
    static constexpr int EDIT_SYNC_LIGHT_FLOOD_CAP = 96;
    int drainEditLightNeighborhoodLocked(
        ChunkPos home, const std::array<std::shared_ptr<const ColumnPlan>, 9>& columnPlans);

    void generateChunk(const std::shared_ptr<Chunk>& chunk);
    void latchGenerationFailure(std::string message);
    void latchGenerationFailure(worldgen::learned::GenerationFailure failure);
    void generateChunkAsync(ChunkPos pos, int64_t priority);
    void generateColumnPlanAsync(ColumnPos pos, int64_t priority);
    std::shared_ptr<Chunk> loadOrGenerateChunk(ChunkPos pos, bool* loadedFromSave = nullptr,
                                               uint64_t* projectionRevision = nullptr);
    uint64_t applySavedChunkProjection(Chunk& chunk) const;
    bool savedChunkProjectionIsCurrent(uint64_t revision) const;
    bool shouldRetain(ChunkPos pos) const;
    bool setBlockLoaded(BlockPos position, BlockType type, std::optional<FluidState> fluidState,
                        LightUrgency urgency = LightUrgency::DEFERRED);
    bool refreshSkyCutoffLocked(int64_t worldX, int64_t worldZ);
    SkyCutoffSectionRange refreshSkyOverrideColumnLocked(ColumnPos column);
    SkyCutoffSectionRange extendGeneratedSkyCutoffsLocked(const Chunk& chunk);
    SkyCutoffSectionRange refreshSavedSkyCutoffsLocked(ChunkPos pos);
    void markColumnMeshesDirtyLocked(ColumnPos column);
    void markHaloNeighborMeshesDirtyLocked(ChunkPos pos);
    void markSkyContinuityBelowLocked(ColumnPos column, int32_t changedSectionY);
    void markSkyCutoffMeshesDirtyLocked(int64_t worldX, int64_t worldZ);
    void markSkyColumnMeshesDirtyLocked(ColumnPos column);
    void queueSavedSkyPublicationLocked(const std::unordered_set<ColumnPos>& changedColumns);
    void queueFluidResume(ChunkPos pos);
    void registerPlanDependenciesLocked(ChunkPos pos);
    void wakePlanDependents(ColumnPos completedPlan);
    void queueGenerationLocked(ChunkPos pos);
    void ensureStreamingWorkers();
    void requestActiveSetRebuild(int64_t playerX, int32_t playerY, int64_t playerZ);
    void activeSetWorkerLoop();
    bool rebuildActiveSet(const ActiveSetRequest& request);
    bool activeSetRequestIsStale(uint64_t requestId) const;
};

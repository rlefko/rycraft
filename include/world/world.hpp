#pragma once

#include "common/ema.hpp"
#include "common/thread_pool.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"
#include "world/fluid.hpp"
#include "world/mesh_snapshot.hpp"
#include "world/view_distance.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
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

// Selects the bounded exact-mesh set in descending priority order. Existing
// candidates win ties before distance so a small camera movement cannot churn
// already-published surface, cliff, or tree meshes at the hard residency cap.
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
inline constexpr size_t EXACT_MESH_WORKER_COUNT = 4;
inline constexpr size_t MAX_LOADED_CUBES = 32768;
inline constexpr size_t MAX_MESH_RESIDENT_CUBES = 16384;
inline constexpr size_t MAX_COLD_COLUMN_PLANS = 2;
inline constexpr int MAX_EXACT_CUBIC_DISTANCE_CHUNKS = 32;
inline constexpr int EXPLORATION_RADIUS_CHUNKS = 6;
inline constexpr int EXPLORATION_VERTICAL_RADIUS_CUBES = 4;
inline constexpr int HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS = 2;
inline constexpr int VERTICAL_UNLOAD_HYSTERESIS_CUBES = 1;
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
    float activeSetBuildMs = 0.0F;
};

// Immutable pre-cap description of the exact surface that the renderer is
// expected to replace. Far terrain uses this to retain its coarse parent
// until every current exact requirement has a revision-matched mesh.
struct ExactSurfaceCoverageSnapshot {
    uint64_t epoch = 0;
    int nominalRadiusChunks = 0;
    std::vector<ChunkPos> requiredSections;
    std::vector<ColumnPos> unresolvedColumns;
};
inline constexpr size_t MAX_FLUID_RESUME_CUBES_PER_FRAME = 64;
inline constexpr size_t MAX_FLUID_FRONTIER_RESUMES_PER_FRAME = 256;
inline constexpr size_t MAX_FLUID_FRONTIER_RESUMES_PER_CUBE = 16;

class World : public FluidWorldAccess {
public:
    explicit World(uint32_t seed, int viewDistance = DEFAULT_RENDER_DISTANCE_CHUNKS,
                   size_t loadedCubeLimit = MAX_LOADED_CUBES, GenerationSettings generation = {});
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

    BlockType getBlock(int64_t x, int32_t y, int64_t z);
    // A missing cube is distinct from air for targeting and other queries
    // that must never make absent world data interactive.
    std::optional<BlockType> findBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    BlockType getBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    BlockType getCollisionBlockIfLoaded(int64_t x, int32_t y, int64_t z) const;
    bool isChunkLoaded(ChunkPos pos) const;
    bool shouldMeshChunk(ChunkPos pos) const;
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
    size_t cachedColumnPlanCount() const { return generator_.cachedColumnPlanCount(); }

    std::vector<std::shared_ptr<Chunk>> getLoadedChunks() const;
    std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> getLoadedSnapshot() const;
    std::shared_ptr<const std::unordered_set<ChunkPos>> getMeshCandidateSnapshot() const;
    std::shared_ptr<const ExactSurfaceCoverageSnapshot> getExactSurfaceCoverageSnapshot() const;
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

    uint32_t getSeed() const { return seed_; }
    const GenerationSettings& getGenerationSettings() const { return generation_; }
    int getViewDistance() const { return viewDistance_.load(std::memory_order_relaxed); }
    int getExactViewDistance() const {
        return std::min(getViewDistance(), MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
    }
    void setViewDistance(int distance);

    void updatePlayerPosition(int64_t playerX, int32_t playerY, int64_t playerZ);
    void updatePlayerPosition(int64_t playerX, int64_t playerZ) {
        updatePlayerPosition(playerX, SEA_LEVEL, playerZ);
    }
    // Pull block light across all six cube faces until quiescent, bounded so a
    // simulation tick cannot stall on a large lighting update.
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

private:
    uint32_t seed_;
    GenerationSettings generation_;
    std::atomic<int> viewDistance_;

    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> chunks_;
    // One bit per supported vertical section, guarded by chunksMutex_. Mesh
    // snapshots use this to prove a contiguous sky path without repeatedly
    // probing the chunk map while workers wait on the world lock.
    std::unordered_map<ColumnPos, uint64_t> loadedSectionMasks_;
    mutable std::mutex chunksMutex_;
    std::shared_ptr<const std::vector<std::shared_ptr<Chunk>>> loadedSnapshot_;
    std::shared_ptr<const std::unordered_set<ChunkPos>> meshCandidateSnapshot_;
    std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactSurfaceCoverageSnapshot_;
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

    SaveManager* saveManager_ = nullptr;
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

    std::shared_ptr<ThreadPool> genPool_;
    std::unordered_map<ChunkPos, std::future<void>> pendingGenerations_;
    std::unordered_set<ChunkPos> generationsInFlight_;
    std::unordered_map<ColumnPos, std::future<void>> pendingColumnPlans_;
    std::unordered_set<ColumnPos> columnPlansInFlight_;
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

    // Cubes whose derived block light may be stale because a neighbor loaded
    // or an edit landed. The queue is deduplicated and drained on the tick
    // thread. lightMutex_ is never held while acquiring chunksMutex_.
    std::vector<ChunkPos> lightQueue_;
    std::unordered_set<ChunkPos> lightQueued_;
    mutable std::mutex lightMutex_;

    void queueLightReconcile(ChunkPos pos);
    void queueFaceNeighbors(ChunkPos pos);
    void queueLightReconcileWithNeighbors(ChunkPos pos);

    void generateChunk(const std::shared_ptr<Chunk>& chunk);
    void generateChunkAsync(ChunkPos pos, int64_t priority);
    void generateColumnPlanAsync(ColumnPos pos, int64_t priority);
    std::shared_ptr<Chunk> loadOrGenerateChunk(ChunkPos pos, bool* loadedFromSave = nullptr);
    bool shouldRetain(ChunkPos pos) const;
    void setBlockLoaded(BlockPos position, BlockType type, std::optional<FluidState> fluidState);
    bool refreshSkyCutoffLocked(int64_t worldX, int64_t worldZ);
    bool refreshSkyOverrideColumnLocked(ColumnPos column);
    bool extendGeneratedSkyCutoffsLocked(const Chunk& chunk);
    void refreshSavedSkyCutoffsLocked(ChunkPos pos);
    void markColumnMeshesDirtyLocked(ColumnPos column);
    void markHaloNeighborMeshesDirtyLocked(ChunkPos pos);
    void markSkyContinuityBelowLocked(ColumnPos column, int32_t changedSectionY);
    void markSkyCutoffMeshesDirtyLocked(int64_t worldX, int64_t worldZ);
    void markSkyColumnMeshesDirtyLocked(ColumnPos column);
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

#pragma once

#include "render/vertex.hpp"
#include "world/features.hpp"
#include "world/macro_generation.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline constexpr int FAR_TERRAIN_TILE_EDGE = 256;
inline constexpr int FAR_TERRAIN_NEAR_CHUNK_RADIUS = 32;
inline constexpr int FAR_TERRAIN_MAX_CHUNK_RADIUS = 256;
inline constexpr float FAR_TERRAIN_SKIRT_DEPTH = 64.0F;
inline constexpr int FAR_TERRAIN_OCCLUDER_PATCH_EDGE = 64;
inline constexpr int FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE = 32;
inline constexpr int FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE = 64;
inline constexpr float FAR_TERRAIN_LOD_TRANSITION_SECONDS = 0.40F;
inline constexpr size_t FAR_TERRAIN_OCCLUDER_PATCH_COUNT =
    (FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE) *
    (FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE);

enum class FarTerrainStep : uint8_t {
    TWO = 2,
    FOUR = 4,
    EIGHT = 8,
    SIXTEEN = 16,
};

constexpr int farTerrainStepSize(FarTerrainStep step) {
    return static_cast<int>(step);
}

// Returns one bit per horizontal FaceNormal when the displayed neighbor is
// resident at a coarser step. Only the finer tile owns the downward skirt.
// Same-LOD and unavailable neighbors need no skirt and must not become walls.
uint32_t
farTerrainSkirtEdgeMask(FarTerrainStep step,
                        const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps);

// Returns no value outside the far-terrain annulus. Ring endpoints are
// inclusive at the lower resolution and exclusive at the upper resolution.
std::optional<FarTerrainStep> farTerrainStepForChunkDistance(double chunkDistance);

// Selects one of the bounded far-terrain tiers using distance plus immutable
// terrain/hydrology complexity measured by the worker-built tile. Complexity
// shifts detailed terrain outward by at most 35 percent; the previous tier
// adds asymmetric thresholds so ordinary camera motion cannot chatter at a
// ring boundary.
std::optional<FarTerrainStep>
farTerrainStepForMetrics(double chunkDistance, float complexity,
                         std::optional<FarTerrainStep> previousStep = std::nullopt);

// A topology replacement fades the currently displayed tile fully into the
// frame fog, swaps at the hidden midpoint, then fades the target back out.
// This keeps the compact 16-byte vertex ABI unchanged and draws only one tier
// at a time. The renderer bounds the number of simultaneous transitions.
struct FarTerrainTransitionSample {
    bool drawTarget = false;
    bool complete = false;
    float fogBlend = 0.0F;
};

FarTerrainTransitionSample sampleFarTerrainTransition(float elapsedSeconds);

struct FarTerrainKey {
    int64_t tileX = 0;
    int64_t tileZ = 0;
    FarTerrainStep step = FarTerrainStep::FOUR;

    constexpr bool operator==(const FarTerrainKey&) const = default;
};

struct FarTerrainKeyHash {
    size_t operator()(const FarTerrainKey& key) const noexcept;
};

struct FarTerrainBounds {
    int64_t minX = 0;
    int64_t maxX = 0;
    int64_t minZ = 0;
    int64_t maxZ = 0;
    float minY = 0.0F;
    float maxY = 0.0F;
};

struct FarTerrainViewTile {
    FarTerrainKey key;
    FarTerrainBounds bounds;
    double distanceSquared = 0.0;
    double distanceChunks = 0.0;
};

// Enumerates the circular far-terrain annulus in nearest-first order. Tiles
// that straddle the exact boundary are retained; the shared fragment handoff
// clips their inner overlap and fades them in across the exact world's halo.
void selectFarTerrainView(double cameraX, double cameraZ, int exactChunkRadius,
                          int visibleChunkRadius, std::vector<FarTerrainViewTile>& output);

// Vertices are tile-local in X/Z and world-relative to Y=0. A future draw
// path can therefore use an exact int64 tile origin before converting to the
// camera-relative float origin already used by cubic chunks.
struct FarTerrainMesh {
    FarTerrainKey key;
    int64_t originX = 0;
    int64_t originZ = 0;
    FarTerrainBounds bounds;
    FarTerrainBounds surfaceBounds;
    std::array<FarTerrainBounds, FAR_TERRAIN_OCCLUDER_PATCH_COUNT> occluderPatches{};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t opaqueIndexCount = 0;
    uint32_t terrainQuadCount = 0;
    uint32_t waterQuadCount = 0;
    uint32_t waterContourTriangleCount = 0;
    uint32_t waterfallQuadCount = 0;
    uint32_t skirtQuadCount = 0;
    uint32_t canopyAnchorCount = 0;
    uint32_t canopyImpostorQuadCount = 0;
    uint32_t mergedTerrainCellCount = 0;
    float complexity = 0.0F;
    uint64_t deterministicHash = 0;

    size_t byteSize() const;
};

struct FarTerrainGeometrySample {
    double terrainHeight = 0.0;
    double waterSurface = SEA_LEVEL;
    double discharge = 0.0;
    double sediment = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    double flowX = 1.0;
    double flowZ = 0.0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool waterfall = false;
    bool waterfallAnchor = false;
    bool delta = false;
};

struct FarTerrainSource {
    using GeometryFunction =
        std::function<FarTerrainGeometrySample(int64_t worldX, int64_t worldZ)>;
    using MaterialFunction = std::function<BlockType(int64_t worldX, int64_t worldZ,
                                                     const FarTerrainGeometrySample& geometry)>;
    using CanopyFunction =
        std::function<std::vector<FarCanopy>(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                             int64_t maximumZ, FarTerrainStep step)>;

    GeometryFunction geometry;
    MaterialFunction material;
    GeometryFunction nearGeometry;
    MaterialFunction nearMaterial;
    CanopyFunction canopies;
};

class FarTerrainMesher {
public:
    using SurfaceSampleFunction =
        std::function<worldgen::SurfaceSample(int64_t worldX, int64_t worldZ)>;

    // Pure CPU build. All world samples use globally aligned int64
    // coordinates. The returned object is immutable to all scheduler clients.
    static std::shared_ptr<const FarTerrainMesh> build(FarTerrainKey key,
                                                       const FarTerrainSource& source);
    static std::shared_ptr<const FarTerrainMesh>
    buildFromSurface(FarTerrainKey key, const SurfaceSampleFunction& sampleSurface);
    static FarTerrainSource surfaceGeometrySource(SurfaceSampleFunction sampleSurface);
    static FarTerrainSource tieredSurfaceGeometrySource(SurfaceSampleFunction exactNearSurface,
                                                        SurfaceSampleFunction coarseSurface);
    static FarTerrainSource
    macroGeometrySource(std::shared_ptr<worldgen::MacroGenerationSampler> sampler);
};

struct FarTerrainResult {
    FarTerrainKey key;
    uint64_t epoch = 0;
    std::shared_ptr<const FarTerrainMesh> mesh;
    bool failed = false;
};

struct FarTerrainSchedulerLimits {
    size_t maxPending = 64;
    size_t maxCompleted = 32;
    size_t maxCacheEntries = 1024;
    size_t maxCacheBytes = 512 * 1024 * 1024;
};

struct FarTerrainSchedulerStats {
    size_t inFlight = 0;
    size_t queued = 0;
    size_t completed = 0;
    size_t cacheEntries = 0;
    size_t cacheBytes = 0;
    uint64_t epoch = 0;
    uint64_t submitted = 0;
    uint64_t built = 0;
    uint64_t canceled = 0;
    uint64_t failed = 0;
    uint64_t cacheHits = 0;
};

struct TerrainHorizonViewpoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Conservative terrain-only occlusion for front-to-back far tiles. A tile
// is rejected only when every azimuth bin intersected by its projected AABB
// has a strictly closer occluder that fully covers the bin and establishes a
// lower-bound horizon above the tile's maximum possible elevation angle.
// Camera-overlapping bounds stay visible to avoid false positives.
class TerrainHorizonCuller {
public:
    static constexpr size_t AZIMUTH_BIN_COUNT = 256;

    explicit TerrainHorizonCuller(TerrainHorizonViewpoint viewpoint = {});

    void reset(TerrainHorizonViewpoint viewpoint);
    bool isOccluded(const FarTerrainBounds& surfaceBounds) const;
    void addOccluder(const FarTerrainBounds& surfaceBounds);
    bool testAndAdd(const FarTerrainBounds& surfaceBounds);

    static double horizontalDistanceSquared(const FarTerrainBounds& bounds,
                                            TerrainHorizonViewpoint viewpoint);

private:
    struct HorizonEntry {
        double farthestDistance = 0.0;
        double minimumElevation = 0.0;
    };

    static constexpr size_t MAX_HORIZONS_PER_BIN = 8;
    TerrainHorizonViewpoint viewpoint_;
    std::array<std::array<HorizonEntry, MAX_HORIZONS_PER_BIN>, AZIMUTH_BIN_COUNT> horizons_{};
    std::array<uint8_t, AZIMUTH_BIN_COUNT> horizonCounts_{};
};

// Bounded scheduler for coordinate-pure far terrain. enqueue(), cache
// lookup, and result draining never construct meshes. advanceEpoch() cancels
// queued work immediately and causes stale worker results to be discarded.
class FarTerrainScheduler {
public:
    static constexpr size_t WORKER_COUNT = 4;

    explicit FarTerrainScheduler(FarTerrainSource source, FarTerrainSchedulerLimits limits = {});
    explicit FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits = {});
    ~FarTerrainScheduler();

    FarTerrainScheduler(const FarTerrainScheduler&) = delete;
    FarTerrainScheduler& operator=(const FarTerrainScheduler&) = delete;

    bool enqueue(FarTerrainKey key);
    // Cancels queued work and drops completed/cache entries that no longer
    // belong to the current view. Running obsolete jobs are discarded before
    // publication, so LOD changes cannot consume the bounded cache or delay
    // current tiles behind stale work.
    void retainWanted(const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted);
    uint64_t advanceEpoch();
    uint64_t currentEpoch() const { return epoch_.load(std::memory_order_acquire); }

    void drainCompleted(std::vector<FarTerrainResult>& output);
    std::shared_ptr<const FarTerrainMesh> findCached(FarTerrainKey key) const;
    void clearCache();
    FarTerrainSchedulerStats stats() const;
    void shutdown();

private:
    struct Job {
        FarTerrainKey key;
        uint64_t epoch = 0;
    };

    struct CacheEntry {
        std::shared_ptr<const FarTerrainMesh> mesh;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
    };

    FarTerrainSource source_;
    FarTerrainSchedulerLimits limits_;
    std::vector<std::thread> workers_;

    mutable std::mutex jobMutex_;
    std::condition_variable jobCv_;
    std::deque<Job> jobs_;
    std::unordered_map<FarTerrainKey, uint64_t, FarTerrainKeyHash> activeKeys_;
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wantedKeys_;
    bool wantedFilterEnabled_ = false;

    mutable std::mutex completedMutex_;
    std::deque<FarTerrainResult> completed_;

    mutable std::mutex cacheMutex_;
    mutable std::unordered_map<FarTerrainKey, CacheEntry, FarTerrainKeyHash> cache_;
    mutable size_t cacheBytes_ = 0;
    mutable uint64_t accessClock_ = 0;

    std::atomic<bool> running_{true};
    std::atomic<uint64_t> epoch_{1};
    std::atomic<size_t> inFlight_{0};
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> built_{0};
    std::atomic<uint64_t> canceled_{0};
    std::atomic<uint64_t> failed_{0};
    mutable std::atomic<uint64_t> cacheHits_{0};

    void workerLoop();
    void finishJob(const Job& job);
    void storeCompleted(FarTerrainResult result);
    void storeCache(std::shared_ptr<const FarTerrainMesh> mesh);
};

#pragma once

#include "render/shader_types.hpp"
#include "render/vertex.hpp"
#include "world/features.hpp"
#include "world/macro_generation.hpp"
#include "world/surface_material.hpp"
#include "world/view_distance.hpp"
#include "world/world_config.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline constexpr int FAR_TERRAIN_TILE_EDGE = static_cast<int>(FAR_TERRAIN_TILE_EDGE_BLOCKS);
inline constexpr int FAR_TERRAIN_NEAR_CHUNK_RADIUS = 32;
inline constexpr int FAR_TERRAIN_MAX_CHUNK_RADIUS = MAX_RENDER_DISTANCE_CHUNKS;
inline constexpr float FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS = 16.0F;
inline constexpr double FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS = 128.0;
inline constexpr double FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS = 224.0;
inline constexpr double FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS = 352.0;
inline constexpr double FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS =
    static_cast<double>(FAR_TERRAIN_MAX_CHUNK_RADIUS);
inline constexpr float FAR_TERRAIN_SKIRT_DEPTH = 64.0F;
inline constexpr int FAR_TERRAIN_OCCLUDER_PATCH_EDGE = 64;
inline constexpr float FAR_TERRAIN_LOD_TRANSITION_SECONDS =
    FAR_TERRAIN_LOD_TRANSITION_SECONDS_VALUE;
inline constexpr float FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS =
    FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS_VALUE;
inline constexpr float FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS = 0.12F;
inline constexpr double FAR_TERRAIN_STEP16_RELIEF_ENVELOPE = 0.86;
inline constexpr double FAR_TERRAIN_STEP32_RELIEF_ENVELOPE = 2.25;
// Maximum gradient carried by the 16- and 8-block relief bands omitted from
// a step-16 sample: 2 pi times the sum of amplitude divided by wavelength.
inline constexpr double FAR_TERRAIN_STEP16_RELIEF_SLOPE_ENVELOPE = 0.45;
inline constexpr double FAR_TERRAIN_STEP32_RELIEF_SLOPE_ENVELOPE = 0.90;
inline constexpr double FAR_TERRAIN_EMITTED_SURFACE_ENVELOPE = 1.25;
inline constexpr size_t FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME = 32;
inline constexpr size_t FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME = 12;
inline constexpr size_t FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME = 32ull * 1024 * 1024;
// A resident parent in the connected coverage prefix may request its selected
// refinement before the complete 8 km parent disk is ready. The scheduler
// reserves the remaining workers and queue capacity for base coverage, so this
// bounded lane cannot starve the frontier.
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT = 4;
// Split the eight utility workers evenly while the connected parent frontier
// is incomplete. Four parents keep the fog edge moving, while four refinements
// can replace emergency geometry and populate progressive middle tiers without
// waiting for the full horizon disk.
inline constexpr size_t FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE = 4;
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME = 4;
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME = 4;
inline constexpr size_t FAR_TERRAIN_NEAR_FALLBACK_TILE_COUNT = 3;
inline constexpr size_t FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS = 64;
inline constexpr size_t FAR_TERRAIN_OCCLUDER_PATCH_COUNT =
    (FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE) *
    (FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE);

enum class FarTerrainStep : uint8_t {
    ONE = 1,
    TWO = 2,
    FOUR = 4,
    EIGHT = 8,
    SIXTEEN = 16,
    THIRTY_TWO = 32,
};

constexpr int farTerrainStepSize(FarTerrainStep step) {
    return static_cast<int>(step);
}

constexpr size_t farTerrainBaseWorkerReservation(size_t workerBudget,
                                                 bool baseWorkQueued) noexcept {
    return baseWorkQueued ? std::min(workerBudget, FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE)
                          : 0;
}

constexpr size_t farTerrainUrgentWorkerLimit(size_t workerBudget, bool baseWorkQueued) noexcept {
    const size_t reserved = farTerrainBaseWorkerReservation(workerBudget, baseWorkQueued);
    return std::min(FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT, workerBudget - reserved);
}

constexpr worldgen::SurfaceFootprint farTerrainSurfaceFootprint(FarTerrainStep step) noexcept {
    switch (step) {
        case FarTerrainStep::ONE:
            return worldgen::SurfaceFootprint::BLOCK_1;
        case FarTerrainStep::TWO:
            return worldgen::SurfaceFootprint::BLOCK_2;
        case FarTerrainStep::FOUR:
            return worldgen::SurfaceFootprint::BLOCK_4;
        case FarTerrainStep::EIGHT:
            return worldgen::SurfaceFootprint::BLOCK_8;
        case FarTerrainStep::SIXTEEN:
            return worldgen::SurfaceFootprint::BLOCK_16;
        case FarTerrainStep::THIRTY_TWO:
            return worldgen::SurfaceFootprint::BLOCK_32;
    }
    return worldgen::SurfaceFootprint::BLOCK_32;
}

constexpr std::optional<FarTerrainStep> farTerrainStepForSize(int step) noexcept {
    switch (step) {
        case 1:
            return FarTerrainStep::ONE;
        case 2:
            return FarTerrainStep::TWO;
        case 4:
            return FarTerrainStep::FOUR;
        case 8:
            return FarTerrainStep::EIGHT;
        case 16:
            return FarTerrainStep::SIXTEEN;
        case 32:
            return FarTerrainStep::THIRTY_TWO;
        default:
            return std::nullopt;
    }
}

inline constexpr FarTerrainStep FAR_TERRAIN_BASE_STEP = FarTerrainStep::THIRTY_TWO;
inline constexpr std::array<FarTerrainStep, 4> FAR_TERRAIN_REFINEMENT_STEPS = {
    FarTerrainStep::SIXTEEN,
    FarTerrainStep::EIGHT,
    FarTerrainStep::FOUR,
    FarTerrainStep::TWO,
};

constexpr bool farTerrainIsBaseStep(FarTerrainStep step) {
    return step == FAR_TERRAIN_BASE_STEP;
}

// Once a replacement is resident, the renderer can move directly to it under
// the bounded topology-swap contract. This keeps cold nearby tiles from
// dwelling on every intermediate emergency tier.
FarTerrainStep farTerrainNextDisplayedStep(FarTerrainStep displayed, FarTerrainStep desired);

struct FarTerrainRefinementOrder {
    std::array<FarTerrainStep, FAR_TERRAIN_REFINEMENT_STEPS.size()> steps{};
    size_t count = 0;
};

// The broad optional lane enumerates refinements coarse-to-fine. The connected
// near lane may seed a faster step-8 fallback first, and display selection may
// skip any completed intermediate tier.
FarTerrainRefinementOrder farTerrainRefinementOrder(FarTerrainStep desired);

using FarTerrainStepMask = uint8_t;

constexpr FarTerrainStepMask farTerrainStepMask(FarTerrainStep step) {
    return static_cast<FarTerrainStepMask>(step);
}

// Picks the finest resident tier at or above the requested footprint. During
// refinement this can only preserve or improve the displayed detail. A
// coarser result is allowed only when distance selection explicitly requests
// coarsening and that exact tier is resident.
FarTerrainStep farTerrainFinestReadyStep(FarTerrainStep displayed, FarTerrainStep desired,
                                         FarTerrainStepMask readySteps);

// First-time display must never regress through a coarse parent when a
// revision-matched selected tier is already GPU resident. The step-32 parent
// remains mandatory for connected coverage, but a cached nearby step-2 mesh
// can become the displayed tier directly after a camera jump or cache reentry.
// No tier is displayable until its coordinate's parent is resident.
std::optional<FarTerrainStep>
farTerrainInitialDisplayedStep(FarTerrainStep desired, FarTerrainStepMask readySteps,
                               FarTerrainStep coarsestAllowed = FAR_TERRAIN_BASE_STEP);

// Coarser parents remain resident for dependency and horizon coverage, but
// cannot become visible when a protected exact-loading tile requires a finer
// temporary tier.
bool farTerrainDisplayedStepAllowed(FarTerrainStep step, FarTerrainStep coarsestAllowed);

// An active replacement owns both GPU tiers until completion. Deferring the
// next upload avoids immediately sweeping and then re-uploading an
// intermediate that is not yet displayed or part of the active pair.
std::optional<FarTerrainStep> farTerrainReadyTransitionTarget(FarTerrainStep displayed,
                                                              FarTerrainStep desired,
                                                              FarTerrainStepMask readySteps,
                                                              bool transitionActive);

// A newly visible step-32 parent near the exact handoff waits briefly for a
// ready desired target. That target may begin immediately; only an optional
// intermediate placeholder is held for this bounded window, including after a
// camera jump reveals an already cached parent.
bool farTerrainDeferNearIntermediate(FarTerrainStep displayed, FarTerrainStep desired,
                                     FarTerrainStep target, float parentAgeSeconds);

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
// terrain/hydrology complexity measured by the worker-built tile. The wide
// Exact cubes provide the one-block tier. The 2-, 4-, 8-, and 16-block far
// bands then taper gradually toward the horizon.
// Complexity shifts detail outward by at most 35 percent, while the previous
// tier adds asymmetric thresholds so ordinary camera motion cannot chatter.
std::optional<FarTerrainStep>
farTerrainStepForMetrics(double chunkDistance, float complexity,
                         std::optional<FarTerrainStep> previousStep = std::nullopt);

// A topology replacement keeps complete terrain meshes on either side of a
// narrow fog-covered swap. Canopies exchange monotonically over the full
// duration, skirts follow the visible terrain topology, and water retains
// source ownership until completion. The renderer bounds simultaneous pairs.
struct FarTerrainTransitionSample {
    bool drawTarget = false;
    bool complete = false;
    float fogBlend = 0.0F;
    float progress = 0.0F;
};

FarTerrainTransitionSample sampleFarTerrainTransition(float elapsedSeconds);

// Pure transition state used by the renderer and regression tests. An active
// replacement remains authoritative until it completes, even if the desired
// distance tier changes while its terrain and canopy exchange is in progress.
struct FarTerrainLodAdvance {
    FarTerrainStep displayed = FAR_TERRAIN_BASE_STEP;
    std::optional<FarTerrainStep> transitionTarget;
    bool completedTransition = false;
};

FarTerrainLodAdvance advanceFarTerrainLod(FarTerrainStep displayed, FarTerrainStep desired,
                                          std::optional<FarTerrainStep> activeTarget = std::nullopt,
                                          float activeElapsedSeconds = 0.0F);

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

// Enumerates every 256-block tile intersecting the circular visible disk in
// nearest-first order. Exact ownership is resolved separately from current
// cubic mesh residency and never removes the coarse parent needed to cover
// cold or capped exact terrain.
void selectFarTerrainView(double cameraX, double cameraZ, int visibleChunkRadius,
                          std::vector<FarTerrainViewTile>& output);

struct FarTerrainCoverageFrontier {
    // Zero disables the temporary frontier because every selected base is
    // resident. A missing tile containing the camera also yields zero, but in
    // that state no far tile is eligible to draw.
    float distanceBlocks = 0.0F;
    double distanceSquaredBlocks = 0.0;
    uint32_t missingBaseTiles = 0;
    bool complete = true;
};

using FarTerrainResidencyFunction = std::function<bool(const FarTerrainKey&)>;

// Finds the nearest absent step-32 parent. Tiles and fragments at or beyond
// this radial frontier stay hidden, so out-of-order worker completion cannot
// expose a disconnected distant island across an empty band.
FarTerrainCoverageFrontier
farTerrainCoverageFrontier(const std::vector<FarTerrainViewTile>& selected,
                           const FarTerrainResidencyFunction& isResident);

// Tests the same radial frontier used by draw submission. Keeping this pure
// prevents CPU eligibility and the shader fade from disagreeing about a
// missing parent tile.
bool farTerrainCoverageDrawEligible(double tileDistanceSquared,
                                    const FarTerrainCoverageFrontier& frontier);

// A selected refinement is eligible while parent coverage is incomplete only
// when it is already part of the connected visible prefix. The camera tile may
// construct its selected target alongside its missing parent, but display
// still requires the parent and cannot reveal a refinement island. Every
// distance-selected tier participates so the horizon acquires a 16, 8, 4, 2
// taper while farther parents are still streaming.
bool farTerrainConnectedRefinementEligible(const FarTerrainViewTile& tile,
                                           float actualExactHandoffBlocks,
                                           const FarTerrainCoverageFrontier& frontier,
                                           bool baseResident, bool cameraTile = false);

// The broad optional refinement lane stays closed until the complete parent
// disk is resident and the current submission scan reached every parent.
// Connected selected targets and bounded near fallbacks use the earlier lane.
bool farTerrainRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier,
                                  bool allBaseCandidatesScanned);

// Scheduler ordering is lane-first, then nearest-view priority. A refinement
// cannot jump ahead of a parent merely because its tile is closer.
bool farTerrainSubmissionBefore(FarTerrainKey first, uint32_t firstViewPriority,
                                FarTerrainKey second, uint32_t secondViewPriority);

// Appends every coarse parent first, then every distance-selected target in
// nearest-first order, then optional emergency intermediates. The second lane
// prevents a nearby cold tile from dwelling on a coarse placeholder while
// distant fallback work consumes the bounded queue.
void buildFarTerrainResidencyOrder(const std::vector<FarTerrainViewTile>& selected,
                                   std::vector<FarTerrainKey>& output);

// Checks the two-lane order without rebuilding it. Stable camera frames can
// retain both the renderer's set and the scheduler's immutable wanted filter.
bool farTerrainResidencyOrderMatches(const std::vector<FarTerrainViewTile>& selected,
                                     std::span<const FarTerrainKey> order);

// Wanted residency is a set contract. Camera motion may reorder the same
// coordinates by a few places without changing which immutable meshes are
// needed. Keeping that distinction out of retainWanted avoids rebuilding
// cache priority tables and taking three scheduler locks on those frames.
bool farTerrainResidencyMembershipMatches(
    const std::vector<FarTerrainViewTile>& selected,
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted);

// Horizontal distance from a camera to one exact chunk column's closed AABB.
// The renderer uses the nearest unresolved or stale exact requirement as the
// fallback terrain and water handoff instead of trusting the configured
// radius.
double farTerrainColumnDistanceSquared(double cameraX, double cameraZ, ColumnPos column);

// Revision equality, rather than GPU allocation, defines exact readiness.
// This intentionally counts a completed empty mesh as ready while rejecting
// a stale nonempty mesh.
bool farTerrainExactSectionReady(uint32_t builtRevision, uint32_t currentRevision);

// Published exact geometry keeps ownership while its immutable mesh remains
// resident. A later chunk revision requests a replacement without revealing
// far terrain beneath the stale but still displayed mesh.
bool farTerrainExactSectionOwnsSurface(bool previouslyPublished, uint32_t builtRevision,
                                       uint32_t currentRevision);

// Keep optional horizon work bounded while generation, exact mesh ownership,
// render-thread uploads, or revision-ready surface coverage still has work.
// Counts make the policy directly testable without coupling far terrain to a
// particular scheduler implementation.
bool farTerrainExactStreamingBusy(size_t pendingChunks, size_t schedulerOwnedMeshes,
                                  size_t consumerPendingMeshes, size_t requiredSections,
                                  size_t readySections, size_t unresolvedColumns);

struct FarTerrainExactHandoff {
    static constexpr size_t COLUMN_MASK_WORD_COUNT = FAR_TERRAIN_EXACT_MASK_WORD_COUNT;
    using ColumnMask = std::array<uint32_t, COLUMN_MASK_WORD_COUNT>;

    struct TileState {
        ColumnPos coordinate{};
        bool hasRequirements = false;
        bool ready = false;
        float limitingDistanceBlocks = std::numeric_limits<float>::infinity();
        ColumnMask requiredColumns{};
        ColumnMask incompleteColumns{};
    };

    static constexpr size_t MAX_TILE_STATES = 64;
    float distanceBlocks = 0.0F;
    size_t requiredSections = 0;
    size_t readySections = 0;
    size_t unresolvedColumns = 0;
    std::array<TileState, MAX_TILE_STATES> tileStates{};
    size_t tileStateCount = 0;
    std::unordered_map<ColumnPos, uint8_t> tileStateIndices;

    bool tileFullyReady(ColumnPos coordinate) const;
    bool tileFullyOwned(ColumnPos coordinate) const;
    bool columnFullyReady(ColumnPos chunkColumn) const;
    ColumnMask readyColumnMask(ColumnPos tileCoordinate) const;
    float distanceBlocksForTile(ColumnPos coordinate, float nominalDistanceBlocks) const;
};

// A tile near the exact handoff needs fine fallback until every one of its
// chunk columns is exact-owned. A fully ready but partially required boundary
// tile still contains far-owned fragments outside the circular exact disk.
bool farTerrainRequiresCoverageParent(double cameraX, double cameraZ, ColumnPos tile,
                                      float nominalDistanceBlocks,
                                      const FarTerrainExactHandoff& handoff);

using FarTerrainExactReadinessFunction = std::function<bool(ChunkPos)>;

// Computes the exact-to-coarse ownership boundary without render state. Each
// missing or stale section and each unresolved column limits the nominal
// radius by the nearest point on its horizontal column AABB.
FarTerrainExactHandoff farTerrainExactHandoff(double cameraX, double cameraZ,
                                              int nominalRadiusChunks,
                                              std::span<const ChunkPos> requiredSections,
                                              std::span<const ColumnPos> unresolvedColumns,
                                              const FarTerrainExactReadinessFunction& isReady);

// Render-thread cache for the immutable exact-coverage topology. Building the
// section-to-column index is proportional to the full pre-cap requirement set
// but happens only when its published epoch changes. Mesh publications then
// update one indexed column, and camera motion visits only incomplete columns.
class FarTerrainExactCoverageCache {
public:
    void rebuild(uint64_t epoch, int nominalRadiusChunks,
                 std::span<const ChunkPos> requiredSections,
                 std::span<const ColumnPos> unresolvedColumns,
                 const FarTerrainExactReadinessFunction& isReady);
    void clear();
    bool matches(uint64_t epoch, int nominalRadiusChunks) const;
    bool setSectionReady(ChunkPos section, bool ready);
    const FarTerrainExactHandoff& sample(double cameraX, double cameraZ);
    size_t lastSampleColumnVisits() const { return lastSampleColumnVisits_; }

private:
    struct ColumnState {
        ColumnPos coordinate{};
        uint16_t requiredSections = 0;
        uint16_t readySections = 0;
        uint8_t tileIndex = UINT8_MAX;
        uint16_t bit = 0;
        size_t incompleteListIndex = std::numeric_limits<size_t>::max();
        bool unresolved = false;
    };

    void setColumnIncomplete(size_t columnIndex, bool incomplete);

    bool valid_ = false;
    uint64_t epoch_ = 0;
    int nominalRadiusChunks_ = 0;
    FarTerrainExactHandoff handoff_;
    std::vector<ColumnState> columns_;
    std::unordered_map<ColumnPos, size_t> columnIndices_;
    std::unordered_map<ChunkPos, size_t> sectionColumns_;
    std::unordered_set<ChunkPos> readySections_;
    std::vector<size_t> incompleteColumns_;
    size_t lastSampleColumnVisits_ = 0;
};

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
    worldgen::WaterBodyId waterBodyId = worldgen::NO_WATER_BODY;
    uint64_t transitionOwnerId = 0;
    double terrainHeight = 0.0;
    double waterSurface = SEA_LEVEL;
    double discharge = 0.0;
    double sediment = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    double flowX = 1.0;
    double flowZ = 0.0;
    worldgen::WaterTransitionKind transitionOwnerKind = worldgen::WaterTransitionKind::NONE;
    uint8_t generatedFluidLevel = 0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool waterfall = false;
    bool waterfallAnchor = false;
    bool delta = false;
};

// One coordinate-pure authority supplies geometry and appearance for each LOD
// footprint. Hydrology ownership and water elevations remain invariant across
// footprints, while terrain detail may be filtered below the footprint's
// Nyquist limit. Bounds conservatively enclose the exact terrain represented
// by the footprint, allowing a coarse coverage parent to stay beneath exact
// terrain during a cold handoff.
struct FarSurfaceSample {
    FarTerrainGeometrySample geometry;
    double footprintMinimumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    double footprintMaximumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    worldgen::surface_material::SurfaceMaterialPalette materialPalette;
};

// Geometry and conservative vertical coverage for one half-open, step-aligned
// far-terrain cell. terrainHeight is the filtered voxel top displayed by this
// tier. The minimum and maximum include every final terrain surface inside the
// cell, including narrow incision and relief that does not reach a sample
// corner. skirtBottom includes the lowest support required beside any adjacent
// LOD parent. Conservative coverage never depresses the displayed terrain.
struct FarTerrainCellBounds {
    double terrainHeight = std::numeric_limits<double>::quiet_NaN();
    double minimumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    double maximumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    double skirtBottom = std::numeric_limits<double>::quiet_NaN();
    // True when analytical channels, lake contours, volcanic water, or the
    // sea-level envelope can intersect the cell even though its coarse
    // corners and center are dry. Coverage parents use this only to request
    // canonical water probes; it never changes displayed terrain height.
    bool waterTopologyPossible = false;
    // Volcanic islands and crater lakes modify canonical hydrology after the
    // basin solve. Only cells carrying this bit need the more expensive final
    // volcanic water callback; every other cell can use direct hydrology.
    bool volcanicWaterPossible = false;
};

struct FarTerrainSource {
    using SampleFunction = std::function<FarSurfaceSample(int64_t worldX, int64_t worldZ,
                                                          worldgen::SurfaceFootprint footprint)>;
    using GridSampleFunction = std::function<void(
        int64_t originX, int64_t originZ, int spacing, int sampleEdge,
        worldgen::SurfaceFootprint footprint, std::span<FarSurfaceSample> output)>;
    using GeometryGridSampleFunction =
        std::function<void(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                           int sampleWidth, int sampleHeight, worldgen::SurfaceFootprint footprint,
                           std::span<FarTerrainGeometrySample> output)>;
    using GeometryPointSampleFunction = std::function<void(
        std::span<const ColumnPos> positions, worldgen::SurfaceFootprint footprint,
        std::span<FarTerrainGeometrySample> output)>;
    // The mesher requests one rectangular cell grid with a one-cell exterior
    // apron. Each output entry owns [x, x + step) by [z, z + step), so both
    // sides of a tile face query the same global cell bounds. Implementations
    // must batch this directly from their immutable surface authority rather
    // than reconstructing the footprint with block-resolution samples.
    using CellBoundsGridFunction = std::function<void(
        int64_t originX, int64_t originZ, int step, int cellWidth, int cellHeight,
        worldgen::SurfaceFootprint footprint, std::span<FarTerrainCellBounds> output)>;
    using CanopyFunction =
        std::function<std::vector<FarCanopy>(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                             int64_t maximumZ, FarTerrainStep step)>;
    using MaterialRankFunction = std::function<double(int64_t worldX, int64_t worldZ)>;

    SampleFunction sample;
    GridSampleFunction sampleGrid;
    GeometryGridSampleFunction geometryGrid;
    // Canonical shoreline refinement requests sparse globally aligned points
    // in one immutable batch. The production generator groups them by basin,
    // avoiding thousands of one-point hydrology calls in a coarse parent.
    GeometryPointSampleFunction geometryPoints;
    // Canonical basin authority without post-hydrology volcanic overlays.
    // Step-32 coverage uses the regular grid for globally aligned half-open
    // water cells and the point counterpart only for bounded interior
    // recovery probes. Both callbacks must return identical authority at the
    // same coordinate.
    GeometryGridSampleFunction waterAuthorityGrid;
    GeometryPointSampleFunction waterAuthorityPoints;
    CellBoundsGridFunction cellBoundsGrid;
    CanopyFunction canopies;
    MaterialRankFunction materialRank;
};

class FarTerrainMesher {
public:
    using SurfaceSampleFunction = std::function<worldgen::SurfaceSample(
        int64_t worldX, int64_t worldZ, worldgen::SurfaceFootprint footprint)>;
    using BlockSurfaceSampleFunction =
        std::function<worldgen::SurfaceSample(int64_t worldX, int64_t worldZ)>;

    // Pure CPU build. All world samples use globally aligned int64
    // coordinates. The returned object is immutable to all scheduler clients.
    static std::shared_ptr<const FarTerrainMesh> build(FarTerrainKey key,
                                                       const FarTerrainSource& source);
    static std::shared_ptr<const FarTerrainMesh>
    buildFromSurface(FarTerrainKey key, const SurfaceSampleFunction& sampleSurface);
    static std::shared_ptr<const FarTerrainMesh>
    buildFromSurface(FarTerrainKey key, const BlockSurfaceSampleFunction& sampleSurface);
    static FarTerrainSource surfaceGeometrySource(SurfaceSampleFunction sampleSurface);
    static FarTerrainSource surfaceGeometrySource(BlockSurfaceSampleFunction sampleSurface);
    static FarTerrainSource tieredSurfaceGeometrySource(BlockSurfaceSampleFunction exactNearSurface,
                                                        BlockSurfaceSampleFunction coarseSurface);
    static FarTerrainSource generatorGeometrySource(std::shared_ptr<ChunkGenerator> generator);
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
    size_t maxCacheEntries = 9280;
    size_t maxCacheBytes = 3ull * 1024 * 1024 * 1024;
    // Residency changes are maintained by utility workers. A pass examines no
    // more than this many cache records and retires no more than the byte
    // budget. One individually oversized record may retire alone so cleanup
    // cannot stall permanently.
    size_t maxMaintenanceEntries = 64;
    size_t maxMaintenanceBytes = 32ull * 1024 * 1024;
};

struct FarTerrainSchedulerStats {
    size_t inFlight = 0;
    size_t activeWorkers = 0;
    size_t activeBaseWorkers = 0;
    size_t reservedBaseWorkers = 0;
    size_t workerBudget = 0;
    size_t queued = 0;
    size_t completed = 0;
    size_t cacheEntries = 0;
    size_t cacheBytes = 0;
    size_t queuedBase = 0;
    size_t queuedRefinement = 0;
    size_t queuedUrgentRefinement = 0;
    size_t activeUrgentRefinement = 0;
    size_t urgentRefinementInFlight = 0;
    size_t completedBase = 0;
    size_t completedRefinement = 0;
    size_t cacheBaseEntries = 0;
    uint64_t epoch = 0;
    uint64_t submitted = 0;
    uint64_t built = 0;
    uint64_t canceled = 0;
    uint64_t failed = 0;
    uint64_t cacheHits = 0;
    uint64_t wantedUpdates = 0;
    uint64_t wantedNoops = 0;
    size_t maintenancePending = 0;
    uint64_t maintenancePasses = 0;
    uint64_t maintenanceScanned = 0;
    uint64_t maintenanceEvicted = 0;
    uint64_t maintenanceBytes = 0;
    size_t maximumMaintenanceScanned = 0;
    size_t maximumMaintenanceBytes = 0;
};

struct FarTerrainRefinementCacheRequest {
    ColumnPos coordinate{};
    FarTerrainStep displayed = FAR_TERRAIN_BASE_STEP;
    FarTerrainStep desired = FAR_TERRAIN_BASE_STEP;
    FarTerrainStepMask residentSteps = 0;
    bool transitionActive = false;
    bool deferIntermediate = false;
    bool requiresFineFallback = false;
    bool requiresBlockScaleFallback = false;
};

// Orders the connected refinement lane so the camera exploration band gets
// block-scale step-2 geometry first and the rest of the protected exact disk
// gets a fast step-8 fallback next. Three ordinary nearby desired-step-2 tiles
// retain the same progressive fallback behavior. Step 4 follows before
// ordinary spatial tier requests. Duplicate keys are omitted.
void buildFarTerrainProgressiveSubmissionOrder(
    std::span<const FarTerrainRefinementCacheRequest> requests, std::vector<FarTerrainKey>& output);

// Leaves at most one nondeferred intermediate request per free transition
// slot. Desired-tier results remain eligible because they are residency-pinned
// even when this flag is set.
size_t
reserveFarTerrainIntermediateTransitionSlots(std::span<FarTerrainRefinementCacheRequest> requests,
                                             size_t activeTransitions);

struct FarTerrainGenerationCacheStats {
    size_t entries = 0;
    size_t bytes = 0;
};

struct TerrainHorizonViewpoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Coverage parents may establish a horizon only after their whole patch has
// crossed the frontier fog band. A source or target participating in an LOD
// exchange is likewise never a solid occluder until the exchange completes.
bool farTerrainCoveragePatchMayOcclude(const FarTerrainBounds& patch,
                                       TerrainHorizonViewpoint viewpoint,
                                       const FarTerrainCoverageFrontier& frontier,
                                       double coverageFadeBlocks, bool lodTransitionActive);

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
    static constexpr size_t WORKER_COUNT = 8;
    static constexpr size_t LATENCY_WORKER_COUNT = 4;

    explicit FarTerrainScheduler(FarTerrainSource source, FarTerrainSchedulerLimits limits = {});
    // The far generator must agree with cubic emission, so it takes the same
    // generation settings the World was built with.
    explicit FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits = {},
                                 GenerationSettings generation = {});
    ~FarTerrainScheduler();

    FarTerrainScheduler(const FarTerrainScheduler&) = delete;
    FarTerrainScheduler& operator=(const FarTerrainScheduler&) = delete;

    bool enqueue(FarTerrainKey key, uint32_t viewPriority = std::numeric_limits<uint32_t>::max());
    // Connected refinements may pass queued base jobs only inside the fixed
    // urgent quota. Four workers remain available to the base lane whenever
    // the full worker budget and queued base work permit it.
    bool enqueueUrgentRefinement(FarTerrainKey key,
                                 uint32_t viewPriority = std::numeric_limits<uint32_t>::max());
    // Lock-free conservative capacity probe for bounded producer scans. A
    // subsequent enqueue may still lose a race to another producer.
    bool hasSubmissionCapacity() const noexcept;
    bool hasUrgentRefinementCapacity() const noexcept;
    // Exact exploration and collision work may temporarily reserve CPU and
    // cold-hydrology capacity. Running jobs finish, then no more than this
    // many far workers may enter mesh construction.
    void setWorkerBudget(size_t budget);
    // Cancels the bounded job and completion queues immediately, then asks a
    // utility worker to update priorities and retire obsolete cache records in
    // bounded passes. Running obsolete jobs are discarded before publication.
    // This call never scans or destroys the large CPU mesh cache.
    bool retainWanted(const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
                      const std::vector<FarTerrainKey>& nearestFirst = {});
    uint64_t advanceEpoch();
    uint64_t currentEpoch() const { return epoch_.load(std::memory_order_acquire); }

    void drainCompleted(std::vector<FarTerrainResult>& output);
    std::shared_ptr<const FarTerrainMesh> findCached(FarTerrainKey key) const;
    void findCachedBatch(std::span<const FarTerrainKey> keys, size_t maximumResults,
                         std::vector<std::shared_ptr<const FarTerrainMesh>>& output) const;
    std::shared_ptr<const FarTerrainMesh>
    findFinestCached(ColumnPos coordinate, FarTerrainStep displayed, FarTerrainStep desired,
                     FarTerrainStepMask residentSteps, bool transitionActive = false) const;
    void findFinestCachedBatch(std::span<const FarTerrainRefinementCacheRequest> requests,
                               size_t maximumResults,
                               std::vector<std::shared_ptr<const FarTerrainMesh>>& output) const;
    void clearCache();
    FarTerrainSchedulerStats stats() const;
    FarTerrainGenerationCacheStats generationCacheStats() const;
    void shutdown();

private:
    struct Job {
        FarTerrainKey key;
        uint64_t epoch = 0;
        uint32_t viewPriority = std::numeric_limits<uint32_t>::max();
        bool urgentRefinement = false;
    };

    struct CacheEntry {
        std::shared_ptr<const FarTerrainMesh> mesh;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
        uint64_t maintenanceToken = 0;
    };

    struct ResidencyMembership {
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash> keys;
        std::vector<FarTerrainKey> nearestFirst;
        uint64_t revision = 0;
    };

    struct CacheMaintenanceItem {
        FarTerrainKey key;
        uint64_t token = 0;
    };

    std::shared_ptr<ChunkGenerator> generator_;
    FarTerrainSource source_;
    FarTerrainSchedulerLimits limits_;
    std::vector<std::thread> workers_;

    mutable std::mutex jobMutex_;
    std::condition_variable jobCv_;
    std::deque<Job> jobs_;
    std::unordered_map<FarTerrainKey, uint64_t, FarTerrainKeyHash> activeKeys_;
    size_t activeWorkerCount_ = 0;
    size_t workerBudget_ = WORKER_COUNT;
    std::shared_ptr<const ResidencyMembership> wantedMembership_;
    std::deque<std::shared_ptr<const ResidencyMembership>> retiredMemberships_;
    uint64_t nextWantedRevision_ = 0;
    bool residencyMaintenanceRequested_ = false;
    bool residencyMaintenanceActive_ = false;

    mutable std::mutex completedMutex_;
    std::deque<FarTerrainResult> completed_;

    mutable std::mutex cacheMutex_;
    mutable std::unordered_map<FarTerrainKey, CacheEntry, FarTerrainKeyHash> cache_;
    mutable std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> cachePriorities_;
    mutable std::unordered_set<FarTerrainKey, FarTerrainKeyHash> pinnedBaseKeys_;
    mutable std::shared_ptr<const ResidencyMembership> cacheMembership_;
    mutable std::deque<CacheMaintenanceItem> cacheMaintenanceQueue_;
    mutable size_t cacheMaintenanceRemaining_ = 0;
    mutable size_t cacheBytes_ = 0;
    mutable uint64_t accessClock_ = 0;
    mutable uint64_t cacheMaintenanceTokenClock_ = 0;

    std::atomic<bool> running_{true};
    std::atomic<uint64_t> epoch_{1};
    std::atomic<size_t> inFlight_{0};
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> built_{0};
    std::atomic<uint64_t> canceled_{0};
    std::atomic<uint64_t> failed_{0};
    mutable std::atomic<uint64_t> cacheHits_{0};
    std::atomic<uint64_t> wantedUpdates_{0};
    std::atomic<uint64_t> wantedNoops_{0};
    std::atomic<size_t> maintenancePendingSnapshot_{0};
    std::atomic<uint64_t> maintenancePasses_{0};
    std::atomic<uint64_t> maintenanceScanned_{0};
    std::atomic<uint64_t> maintenanceEvicted_{0};
    std::atomic<uint64_t> maintenanceBytes_{0};
    std::atomic<size_t> maximumMaintenanceScanned_{0};
    std::atomic<size_t> maximumMaintenanceBytes_{0};
    std::atomic<size_t> queuedBaseCount_{0};
    std::atomic<size_t> queuedRefinementCount_{0};
    std::atomic<size_t> queuedUrgentRefinementCount_{0};
    std::atomic<size_t> urgentRefinementInFlightCount_{0};
    std::atomic<size_t> activeWorkerCountSnapshot_{0};
    std::atomic<size_t> activeBaseWorkerCountSnapshot_{0};
    std::atomic<size_t> activeUrgentRefinementCountSnapshot_{0};
    std::atomic<size_t> workerBudgetSnapshot_{WORKER_COUNT};
    std::atomic<size_t> completedBaseCount_{0};
    std::atomic<size_t> completedRefinementCount_{0};
    std::atomic<size_t> cacheEntryCount_{0};
    std::atomic<size_t> cacheBaseEntryCount_{0};
    std::atomic<size_t> cacheBytesSnapshot_{0};
    size_t cacheBaseEntries_ = 0;
    size_t activeBaseWorkerCount_ = 0;
    size_t activeUrgentRefinementCount_ = 0;

    bool enqueueInternal(FarTerrainKey key, uint32_t viewPriority, bool urgentRefinement);
    Job takeNextJobLocked();
    void workerLoop(bool latencySensitive);
    void finishJob(const Job& job);
    void storeCompleted(FarTerrainResult result);
    void storeCache(std::shared_ptr<const FarTerrainMesh> mesh);
    bool performResidencyMaintenance();
    void requestResidencyMaintenance();
    void publishCacheStatsLocked();
};

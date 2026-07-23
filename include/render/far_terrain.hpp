#pragma once

#include "render/shader_types.hpp"
#include "render/vertex.hpp"
#include "world/features.hpp"
#include "world/learned_terrain.hpp"
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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

inline constexpr int FAR_TERRAIN_TILE_EDGE = static_cast<int>(FAR_TERRAIN_TILE_EDGE_BLOCKS);
inline constexpr int FAR_TERRAIN_NEAR_CHUNK_RADIUS = 32;
inline constexpr int FAR_TERRAIN_MAX_CHUNK_RADIUS = MAX_RENDER_DISTANCE_CHUNKS;
// The protected publication follows the camera at half-tile boundaries. Its
// 2x2 step-1 core is surrounded by Manhattan-distance rings at steps 2, 4, 8,
// and 16. The distance-four exterior meets ordinary step-32 coverage at 2:1
// while using 60 targets instead of a centered 121-tile square.
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES = 2;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_DISTANCE_TILES = 0;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_DISTANCE_TILES = 1;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_DISTANCE_TILES = 2;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_DISTANCE_TILES = 3;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_DISTANCE_TILES = 4;
// Directional preparation starts only in the half-tile overlap where both the
// current and next 2x2 cores contain the camera. This is the maximum useful
// lead and leaves the current canonical core's 128-block cushion unchanged.
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_PREDICTION_LEAD_BLOCKS = FAR_TERRAIN_TILE_EDGE / 2;
inline constexpr size_t FAR_TERRAIN_MAX_PROTECTED_PREDICTION_SUBMISSIONS_PER_FRAME = 4;
// A camera can occupy either core tile and either half of that tile. Six tile
// widths conservatively cover the outer target's complete block extent. The
// legacy radius name remains for the startup integration's static contract.
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_MAX_CAMERA_TILE_OFFSET = 5;
inline constexpr int FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_RADIUS_TILES =
    FAR_TERRAIN_PROTECTED_NEAR_MAX_CAMERA_TILE_OFFSET;
inline constexpr int FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS =
    ((FAR_TERRAIN_PROTECTED_NEAR_MAX_CAMERA_TILE_OFFSET + 1) * FAR_TERRAIN_TILE_EDGE) / CHUNK_EDGE;
static_assert(FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS == 96);
inline constexpr double FAR_TERRAIN_DESIRED_METRIC_REFRESH_BLOCKS = 4.0;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_TILE_COUNT = 4;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_TILE_COUNT = 8;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_TILE_COUNT = 12;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_TILE_COUNT = 16;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_TILE_COUNT = 20;
inline constexpr size_t FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT =
    FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_TILE_COUNT +
    FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_TILE_COUNT +
    FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_TILE_COUNT +
    FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_TILE_COUNT +
    FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_TILE_COUNT;
static_assert(FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT == 60);
// Entry publishes a connected, fully water-capable parent disk that contains
// the complete protected topology. A positive v4 view setting below 96 chunks
// is raised to that structural minimum for both preparation and gameplay, so
// startup cannot wait forever on targets that selection clipped away.
constexpr int farTerrainEntryHorizonViewDistance(int requestedChunkRadius) noexcept {
    if (requestedChunkRadius <= 0) return 0;
    return std::clamp(std::max(requestedChunkRadius, FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS), 0,
                      FAR_TERRAIN_MAX_CHUNK_RADIUS);
}

// Inspector horizon measurements use the same far-terrain interval as normal
// view selection. A shorter radius would not include the far handoff, and a
// larger one is outside the supported 512-chunk horizon contract.
constexpr bool farTerrainHorizonRadiusValid(int chunkRadius) noexcept {
    return chunkRadius >= FAR_TERRAIN_NEAR_CHUNK_RADIUS &&
           chunkRadius <= FAR_TERRAIN_MAX_CHUNK_RADIUS;
}

constexpr bool
farTerrainCameraMovementRequiresRefresh(std::optional<std::pair<double, double>> previousCamera,
                                        double cameraX, double cameraZ,
                                        double refreshDistanceBlocks) noexcept {
    if (!previousCamera || refreshDistanceBlocks <= 0.0) return true;
    const double movementX = cameraX - previousCamera->first;
    const double movementZ = cameraZ - previousCamera->second;
    return movementX * movementX + movementZ * movementZ >=
           refreshDistanceBlocks * refreshDistanceBlocks;
}

constexpr bool
farTerrainSelectionRequiresRefresh(std::optional<std::pair<double, double>> previousCamera,
                                   double cameraX, double cameraZ, int previousViewDistanceChunks,
                                   int viewDistanceChunks) noexcept {
    if (!previousCamera || previousViewDistanceChunks != viewDistanceChunks) return true;
    return farTerrainCameraMovementRequiresRefresh(previousCamera, cameraX, cameraZ,
                                                   static_cast<double>(CHUNK_EDGE));
}

constexpr bool
farTerrainDesiredMetricsRequireRefresh(bool selectionChanged, bool authorityBoundsDirty,
                                       uint32_t previousViewportHeight, uint32_t viewportHeight,
                                       double previousVerticalFovRadians, double verticalFovRadians,
                                       bool previousDrawGeometry, bool drawGeometry) noexcept {
    return selectionChanged || authorityBoundsDirty || previousViewportHeight != viewportHeight ||
           previousVerticalFovRadians != verticalFovRadians || previousDrawGeometry != drawGeometry;
}
inline constexpr float FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS = 16.0F;
// Exact cubic terrain owns the first 32 chunks. The far hierarchy starts at
// step 2 and doubles its sampling footprint with each doubling of distance.
// Step 32 remains coverage-only and is never a settled gameplay selection.
inline constexpr double FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS = 32.0;
inline constexpr double FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS = 64.0;
inline constexpr double FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS = 128.0;
inline constexpr double FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS = 256.0;
inline constexpr double FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS =
    static_cast<double>(FAR_TERRAIN_MAX_CHUNK_RADIUS);
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
// A settled far mesh may coarsen only when its conservative geometric error
// occupies about half a pixel. Separate refine and coarsen thresholds give
// camera, FOV, and relief changes a deterministic no-thrash band.
inline constexpr double FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS = 0.55;
inline constexpr double FAR_TERRAIN_SCREEN_ERROR_COARSEN_PIXELS = 0.45;
inline constexpr double FAR_TERRAIN_SCREEN_ERROR_RELIEF_BLOCKS = 64.0;
// The revision-9 real-model authority delta at the locked seed-42 mountain
// page measured a 23-block p95 and 46-block maximum omitted Decoder residual.
// PREVIEW remains valid coverage, but this bound makes its perceptual debt
// explicit and prioritizes a same-key FINAL replacement while it is visible.
inline constexpr double FAR_TERRAIN_PREVIEW_RESIDUAL_P95_BLOCKS = 23.0;
inline constexpr double FAR_TERRAIN_PREVIEW_RESIDUAL_MAX_BLOCKS = 46.0;
inline constexpr size_t FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME = 32;
inline constexpr size_t FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME = 12;
inline constexpr size_t FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME = 32ull * 1024 * 1024;
// Distant parent coverage may use only the unreserved portion of a frame's
// upload budget. Nearby detail and its transition counterpart retain one
// opportunity after those parents have uploaded.
inline constexpr size_t FAR_TERRAIN_NEAR_REFINEMENT_UPLOAD_RESERVE_BYTES = 8ull * 1024 * 1024;
// Movement prefetch is deliberately smaller than the learned authority's
// 64-request queue. Visible coverage always fills first, then no more than one
// leading page row is admitted through the lowest-priority lane.
inline constexpr size_t FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES = 8;
// Visible and protected FINAL requests must be able to enter the 64-request
// learned-authority coordinator even while a moved horizon is preparing new
// preview pages.
inline constexpr size_t FAR_TERRAIN_RESERVED_FINAL_AUTHORITY_REQUESTS = 16;
// Flora is requested only for displayed surfaces. Its separate production
// generator shares the bounded low-priority preview lane, so nearby visible
// ecology can pass distant speculation without consuming any of the sixteen
// admissions reserved for exact and visible FINAL terrain.
inline constexpr worldgen::learned::AuthorityRequestPriority FAR_TERRAIN_CANOPY_AUTHORITY_PRIORITY =
    worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW;
// Once connected parents extend through the finest settled far band, a small
// urgent lane may prepare exact-handoff FINAL parents and nearby refinements.
// Entry preparation suppresses optional refinements and canopies separately,
// leaving the remaining workers on full-horizon terrain and water coverage.
inline constexpr int FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS = 32;
static_assert(FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS ==
              static_cast<int>(FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS));

// The byte budget is a pacing target, not a permanent rejection threshold.
// One oversized terrain mesh may consume an otherwise empty frame; all later
// uploads wait for the next frame. This prevents a valid dense step-1 result
// from remaining cached forever without ever becoming GPU resident.
constexpr bool farTerrainUploadFitsFrameBudget(size_t uploadedBytes, size_t candidateBytes,
                                               size_t budgetBytes) noexcept {
    return uploadedBytes == 0 ||
           (uploadedBytes <= budgetBytes && candidateBytes <= budgetBytes - uploadedBytes);
}

constexpr bool farTerrainUploadFitsPrioritizedFrameBudget(size_t uploadedBytes,
                                                          size_t candidateBytes, size_t budgetBytes,
                                                          size_t reservedBytes,
                                                          bool mayConsumeReserve) noexcept {
    const size_t effectiveBudget = mayConsumeReserve             ? budgetBytes
                                   : reservedBytes < budgetBytes ? budgetBytes - reservedBytes
                                                                 : size_t{0};
    if (!mayConsumeReserve && candidateBytes > effectiveBudget) return false;
    return farTerrainUploadFitsFrameBudget(uploadedBytes, candidateBytes, effectiveBudget);
}

// SegmentedMegaBuffer retains the source allocation until the GPU has finished
// the transition frame. Admission therefore counts the full target allocation
// in addition to current use. Aggregate admission cannot eliminate slab
// fragmentation, but it prevents predictable capacity exceptions and protects
// parent coverage from optional refinement pressure.
enum class FarTerrainGpuArenaClass : uint8_t {
    REFINEMENT,
    FLORA,
    NEAR_REFINEMENT,
    COVERAGE,
    CRITICAL_COVERAGE,
    CRITICAL_REFINEMENT,
};

bool farTerrainGpuUploadFitsArena(uint64_t vertexUsedBytes, uint64_t indexUsedBytes,
                                  uint64_t vertexCapacityBytes, uint64_t indexCapacityBytes,
                                  uint64_t candidateVertexBytes, uint64_t candidateIndexBytes,
                                  FarTerrainGpuArenaClass admissionClass) noexcept;

// Near-detail reclamation may remove only hidden, optional GPU copies. Every
// surface that currently supplies coverage, participates in a transition, or
// blocks the next protected publication remains resident until its normal
// frame-safe retirement.
constexpr bool farTerrainGpuMayEvictForNear(bool baseCoverage, bool displayed,
                                            bool lodTransitionEndpoint,
                                            bool authorityTransitionEndpoint, bool protectedClosure,
                                            bool exactFallback,
                                            bool nextCriticalRefinement) noexcept {
    return !baseCoverage && !displayed && !lodTransitionEndpoint && !authorityTransitionEndpoint &&
           !protectedClosure && !exactFallback && !nextCriticalRefinement;
}

// A resident parent in the connected coverage prefix may request its selected
// refinement during gameplay before a moved full-horizon disk is ready. The scheduler
// reserves the remaining workers and queue capacity for base coverage, so this
// bounded lane cannot starve the frontier.
// During gameplay, visible refinement outranks speculative expansion of a
// moved horizon. On the 16-core reference machine, twelve workers can advance
// adjacent refinement tiers while four remain reserved for gap-free parent
// coverage.
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT = 12;
inline constexpr size_t FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT = 4;
// Distant parents may occupy only three quarters of the ordinary 64-job
// scheduler. The remaining slots admit protected FINAL parents and visible
// refinements even when cold preview parents are parked on authority pages.
inline constexpr size_t FAR_TERRAIN_RESERVED_URGENT_SCHEDULER_REQUESTS = 16;

constexpr size_t farTerrainUrgentSchedulerReservation(size_t maximumPending) noexcept {
    if (maximumPending <= 1) return 0;
    return std::min(FAR_TERRAIN_RESERVED_URGENT_SCHEDULER_REQUESTS,
                    std::max<size_t>(1, maximumPending / 4));
}

constexpr size_t farTerrainNonurgentBaseAdmissionLimit(size_t maximumPending) noexcept {
    return maximumPending - farTerrainUrgentSchedulerReservation(maximumPending);
}
// A fixed safety floor for callers that intentionally admit urgent work while
// the connected parent frontier is incomplete. Startup admits only the small
// set of required FINAL exact-handoff parents through that lane; optional
// refinements remain closed until gameplay.
inline constexpr size_t FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE = 4;
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME = 12;
inline constexpr size_t FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME = 4;
inline constexpr size_t FAR_TERRAIN_MAX_PROTECTED_BRIDGE_SUBMISSIONS_PER_FRAME = 4;
inline constexpr size_t FAR_TERRAIN_NEAR_FALLBACK_TILE_COUNT = 3;

constexpr size_t farTerrainProtectedFinalSubmissionFloor(size_t budget,
                                                         bool bridgePrerequisiteRequired) noexcept {
    if (!bridgePrerequisiteRequired || budget < 3) return budget;
    return budget - std::min(FAR_TERRAIN_MAX_PROTECTED_BRIDGE_SUBMISSIONS_PER_FRAME, budget / 3);
}
// A protected authority handoff can retain its complete 60-surface legal
// closure plus ordinary adjacent LOD changes while a moving camera stages the
// replacement. Parent promotions use the same bound.
inline constexpr size_t FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS = 256;
// Both renderer submission lanes feed a scheduler with at most 64 outstanding
// jobs. Ranking more candidates cannot change the jobs admitted this frame and
// used to sort and deduplicate the complete 3,336-tile horizon every frame.
inline constexpr size_t FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS = 64;
// Missing distant step-32 coverage retains enough room for the camera-critical
// refinement floor. A missing parent inside the protected closure may use the
// complete arena, while refinements, authority replacements, and optional
// canopy attachments retain enough room for movement to publish another
// bounded batch of critical coverage parents.
inline constexpr uint64_t FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES = 64ull * 1024 * 1024;
inline constexpr uint64_t FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES = 32ull * 1024 * 1024;
// Visible optional flora has one bounded residency floor below the coverage
// reserve. Broad terrain refinements cannot consume it, but coverage parents
// retain authority to use the complete arena when a gap-free horizon needs
// the space.
inline constexpr uint64_t FAR_TERRAIN_GPU_VERTEX_FLORA_RESERVE_BYTES = 64ull * 1024 * 1024;
inline constexpr uint64_t FAR_TERRAIN_GPU_INDEX_FLORA_RESERVE_BYTES = 32ull * 1024 * 1024;
// Nearby terrain may replace optional flora and broad hidden refinements, but
// distant refinements and flora cannot consume the allocation floor needed by
// one urgent step batch plus both sides of its legal transitions.
inline constexpr uint64_t FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES =
    128ull * 1024 * 1024;
inline constexpr uint64_t FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES = 64ull * 1024 * 1024;
inline constexpr size_t FAR_TERRAIN_MAX_RESIDENCY_KEYS = 24576;
inline constexpr int FAR_TERRAIN_TRANSITION_SAMPLE_STEP = 2;
// The renderer ignores this metadata bit. It lets diagnostics distinguish
// genuine transition surfaces from ordinary voxel tops without assigning
// them skirt behavior in either shader path.
inline constexpr uint32_t FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK = 1U << 30U;
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

// The LOD inspector must distinguish exact cubic ownership from far step 1.
// Those representations have the same spatial resolution but different
// publication, clipping, and lighting paths, so sharing a color concealed the
// source of handoff artifacts in captured frames.
constexpr std::array<float, 4>
terrainLodOverlayColor(std::optional<FarTerrainStep> farStep) noexcept {
    if (!farStep) return {0.04F, 0.88F, 1.00F, 0.78F};
    switch (*farStep) {
        case FarTerrainStep::ONE:
            return {0.12F, 0.38F, 1.00F, 0.78F};
        case FarTerrainStep::TWO:
            return {0.10F, 0.95F, 0.28F, 0.78F};
        case FarTerrainStep::FOUR:
            return {0.95F, 0.90F, 0.12F, 0.78F};
        case FarTerrainStep::EIGHT:
            return {1.00F, 0.52F, 0.08F, 0.78F};
        case FarTerrainStep::SIXTEEN:
            return {0.95F, 0.16F, 0.08F, 0.78F};
        case FarTerrainStep::THIRTY_TWO:
            return {0.62F, 0.08F, 0.92F, 0.78F};
    }
    return {};
}

// Preview authority is temporary cold-coverage geometry at any LOD. A preview
// parent may admit preview children immediately, while a final child still
// requires a final parent. Keeping provenance on every mesh makes that rule
// independent of its geometric LOD key.
enum class FarTerrainAuthorityQuality : uint8_t {
    PREVIEW = 0,
    FINAL = 1,
};

constexpr bool farTerrainAuthoritySatisfies(FarTerrainAuthorityQuality candidate,
                                            FarTerrainAuthorityQuality required) noexcept {
    return static_cast<uint8_t>(candidate) >= static_cast<uint8_t>(required);
}

constexpr bool farTerrainAuthorityMayReplace(FarTerrainAuthorityQuality resident,
                                             FarTerrainAuthorityQuality incoming) noexcept {
    return farTerrainAuthoritySatisfies(incoming, resident);
}

constexpr bool farCanopyMatchesSurface(FarTerrainAuthorityQuality ecology,
                                       FarTerrainAuthorityQuality grounding,
                                       FarTerrainAuthorityQuality surface) noexcept {
    // PREVIEW ecology is a drawable provisional attachment. Its roots still
    // have to consume the same terrain authority as the displayed surface,
    // but final climate must not be a prerequisite for visible vegetation.
    (void)ecology;
    return grounding == surface;
}

constexpr bool farCanopyAnchorIdentityCompatible(FarTerrainAuthorityQuality residentEcology,
                                                 uint64_t residentAnchorIdentityHash,
                                                 FarTerrainAuthorityQuality incomingEcology,
                                                 uint64_t incomingAnchorIdentityHash) noexcept {
    return residentEcology != FarTerrainAuthorityQuality::FINAL ||
           incomingEcology != FarTerrainAuthorityQuality::FINAL ||
           residentAnchorIdentityHash == incomingAnchorIdentityHash;
}

constexpr bool farCanopyMayReplace(FarTerrainAuthorityQuality residentEcology,
                                   FarTerrainAuthorityQuality residentGrounding,
                                   FarTerrainAuthorityQuality incomingEcology,
                                   FarTerrainAuthorityQuality incomingGrounding) noexcept {
    const bool advancesGrounding =
        farTerrainAuthoritySatisfies(incomingGrounding, residentGrounding) &&
        incomingGrounding != residentGrounding;
    const bool replacesAtSameGrounding =
        incomingGrounding == residentGrounding &&
        farTerrainAuthorityMayReplace(residentEcology, incomingEcology);
    return advancesGrounding || replacesAtSameGrounding;
}

constexpr bool farTerrainAuthorityAllowsDisplayedStep(FarTerrainAuthorityQuality parent,
                                                      FarTerrainAuthorityQuality child,
                                                      FarTerrainStep step) noexcept {
    return step == FarTerrainStep::THIRTY_TWO || parent == child;
}

constexpr bool farTerrainAuthorityAllowsDisplayedStepDuringParentPromotion(
    FarTerrainAuthorityQuality parentTarget, std::optional<FarTerrainAuthorityQuality> parentSource,
    FarTerrainAuthorityQuality child, FarTerrainStep step) noexcept {
    return farTerrainAuthorityAllowsDisplayedStep(parentTarget, child, step) ||
           (parentSource && farTerrainAuthorityAllowsDisplayedStep(*parentSource, child, step));
}

constexpr bool
farTerrainRefinementRequiresFinalAuthority(FarTerrainAuthorityQuality parentTarget,
                                           std::optional<FarTerrainAuthorityQuality> parentSource,
                                           bool exactHandoffRequired) noexcept {
    return parentTarget == FarTerrainAuthorityQuality::FINAL &&
           (exactHandoffRequired || !parentSource ||
            *parentSource == FarTerrainAuthorityQuality::FINAL);
}

// Once every required surface section in one chunk column has a published
// mesh, that complete column becomes visual authority in one frame. PREVIEW
// quality and an in-flight PREVIEW-to-FINAL replacement cannot conceal its
// exact replacement. Collision uses the same handoff predicate below instead
// of treating every partially loaded cube as authoritative.
struct FarTerrainExactVisualOwnership {
    bool drawExact = false;
    bool clipFar = false;

    constexpr bool operator==(const FarTerrainExactVisualOwnership&) const = default;
};

constexpr FarTerrainExactVisualOwnership
farTerrainExactVisualOwnership(bool sectionRequired, bool columnFullyReady,
                               bool coverageParentDrawable, bool exactGeometryPresent) noexcept {
    return {
        .drawExact = exactGeometryPresent &&
                     (!sectionRequired || columnFullyReady || !coverageParentDrawable),
        .clipFar = coverageParentDrawable && columnFullyReady,
    };
}

// A revision-ready empty section has no draw call but still publishes exact
// air to collision. Required sections wait for their complete column while a
// drawable far parent remains. Without a parent, exact publication is the
// only available representation and therefore takes ownership immediately.
constexpr bool farTerrainExactCollisionOwnsSection(bool sectionRequired, bool columnFullyReady,
                                                   bool coverageParentDrawable,
                                                   bool revisionReady) noexcept {
    return revisionReady && (!sectionRequired || columnFullyReady || !coverageParentDrawable);
}

constexpr int farTerrainStepSize(FarTerrainStep step) {
    return static_cast<int>(step);
}

// Preparation may build the bounded protected FINAL closure without opening
// ordinary visible refinement or optional canopy work. This lets first entry
// publish one authority-compatible near surface instead of briefly exposing a
// PREVIEW parent beneath FINAL exact collision.
constexpr bool
farTerrainFinalStreamingWorkEnabled(bool gameplayScene,
                                    bool protectedEntryPreparation = false) noexcept {
    return gameplayScene || protectedEntryPreparation;
}

constexpr bool farTerrainOptionalStreamingWorkEnabled(bool gameplayScene,
                                                      bool connectedPrefixReady) noexcept {
    return gameplayScene && connectedPrefixReady;
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
inline constexpr std::array<FarTerrainStep, 5> FAR_TERRAIN_REFINEMENT_STEPS = {
    FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT, FarTerrainStep::FOUR,
    FarTerrainStep::TWO,     FarTerrainStep::ONE,
};

constexpr bool farTerrainIsBaseStep(FarTerrainStep step) {
    return step == FAR_TERRAIN_BASE_STEP;
}

// As a last resort, protected near detail may replace a displayed distant
// step-16 child with its already resident step-32 parent. The demotion is
// legal only when it preserves the 2:1 cardinal-neighbor contract and touches
// no transition, protected closure, exact fallback, or next critical key.
constexpr bool farTerrainDisplayedRefinementMayYieldToParentForNear(
    FarTerrainStep displayedStep,
    const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps, bool parentResident,
    bool lodTransitionEndpoint, bool neighborTransition, bool authorityTransitionEndpoint,
    bool protectedClosure, bool exactFallback, bool nextCriticalRefinement) noexcept {
    if (displayedStep != FarTerrainStep::SIXTEEN || !parentResident || lodTransitionEndpoint ||
        neighborTransition || authorityTransitionEndpoint || protectedClosure || exactFallback ||
        nextCriticalRefinement) {
        return false;
    }
    for (const std::optional<FarTerrainStep> neighbor : displayedNeighborSteps) {
        if (!neighbor) continue;
        const int neighborSize = farTerrainStepSize(*neighbor);
        if (farTerrainStepSize(FAR_TERRAIN_BASE_STEP) > neighborSize * 2) return false;
    }
    return true;
}

// Display changes advance one power-of-two tier at a time. This makes the
// neighbor compatibility gate stable while a connected replacement becomes
// visible.
FarTerrainStep farTerrainNextDisplayedStep(FarTerrainStep displayed, FarTerrainStep desired);

struct FarTerrainRefinementOrder {
    std::array<FarTerrainStep, FAR_TERRAIN_REFINEMENT_STEPS.size()> steps{};
    size_t count = 0;
};

// Residency enumeration remains coarse-to-fine. Submission and display
// advance through the next adjacent tier so bounded work always produces a
// replacement that can satisfy the 2:1 neighbor contract.
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

// First-time display preserves the finest completed refinement. Until any
// refinement is resident, the step-32 parent is a safe temporary fallback:
// the exact-ownership shader clips its ready columns, while unresolved
// columns retain continuous coarse coverage. No tier is displayable until
// its coordinate's parent is resident.
std::optional<FarTerrainStep> farTerrainInitialDisplayedStep(FarTerrainStepMask readySteps);

// A coarser tier remains a legal sole fallback only while no resident tier
// satisfies the protected handoff cap. Active transitions retain both tiers
// independently until their monotonic exchange completes.
bool farTerrainDisplayedStepAllowed(FarTerrainStep step, FarTerrainStep coarsestAllowed,
                                    FarTerrainStepMask readySteps);

// A requested protected patch remains one atomic FINAL publication, but its
// PREVIEW bridge tiers are safe to display while that publication is cold.
// FINAL children remain hidden until the complete protected closure commits.
constexpr bool
farTerrainProtectedIntermediateMayDisplay(bool protectedTarget,
                                          FarTerrainAuthorityQuality quality) noexcept {
    return !protectedTarget || quality == FarTerrainAuthorityQuality::PREVIEW;
}

// A protected FINAL child normally follows a drawable provisional bridge.
// When its coverage parent has already reached FINAL first, that PREVIEW
// bridge can no longer be uploaded without mixing authority qualities. Build
// the hidden FINAL child directly instead; the complete closure still becomes
// drawable only through the atomic protected publication below.
constexpr bool farTerrainProtectedFinalTargetMaySubmit(bool targetFinalResident,
                                                       bool provisionalBridgeDisplayed,
                                                       bool finalParentResident) noexcept {
    return !targetFinalResident && (provisionalBridgeDisplayed || finalParentResident);
}

// Keeps only the bridge tiers between the displayed and desired surfaces.
// Once the desired tier is displayed, temporary step-16, step-8, step-4, and
// step-2 meshes can leave GPU residency while the hidden coverage parent
// remains available for gap-free camera movement.
bool farTerrainRetainsProgressiveStep(FarTerrainStep candidate, FarTerrainStep displayed,
                                      FarTerrainStep desired) noexcept;

// An active replacement owns both GPU tiers until completion. Ordinary cold
// work and intentional coarsening advance through adjacent power-of-two tiers.
// The fixed protected near patch bypasses this helper and publishes its whole
// step-1 plus step-2 topology atomically.
std::optional<FarTerrainStep> farTerrainReadyTransitionTarget(FarTerrainStep displayed,
                                                              FarTerrainStep desired,
                                                              FarTerrainStepMask readySteps,
                                                              bool transitionActive);

// Returns the coarsest temporary surface that may be shown while a selected
// target is cold. The absolute step-1 and step-2 bands always cap fallback at
// step 2, including an unresolved exact-overlap tile.
FarTerrainStep farTerrainCoarsestDrawableFallback(FarTerrainStep desired, bool requiresFineFallback,
                                                  bool requiresBlockScaleFallback) noexcept;

// A newly visible step-32 parent near the exact handoff waits briefly before
// beginning its adjacent progression. The bounded window coalesces a camera
// jump without making the coarse parent a long-lived settled surface.
bool farTerrainDeferNearIntermediate(FarTerrainStep displayed, FarTerrainStep desired,
                                     FarTerrainStep target, float parentAgeSeconds);

// Adjacent displayed tiles differ by at most one power-of-two tier. Missing
// neighbors do not constrain a tile because they own no drawable geometry.
bool farTerrainStepCompatibleWithNeighbors(
    FarTerrainStep step,
    const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps) noexcept;

struct FarTerrainTransitionVertex {
    int16_t x = 0;
    int16_t z = 0;
    uint8_t boundaryEdgeMask = 0;
};

// Triangulates one coarse cell whose selected tile edges use the fine
// canonical sample positions. The perimeter follows the same winding as
// terrain tops, and every triangle faces positive Y. Edge bits use the four
// horizontal FaceNormal values.
struct FarTerrainTransitionTopology {
    static constexpr size_t MAX_VERTEX_COUNT = 69;
    static constexpr size_t MAX_INDEX_COUNT = 204;

    std::array<FarTerrainTransitionVertex, MAX_VERTEX_COUNT> vertices{};
    std::array<uint8_t, MAX_INDEX_COUNT> indices{};
    uint16_t vertexCount = 0;
    uint16_t indexCount = 0;
};

FarTerrainTransitionTopology farTerrainTransitionCellTopology(int coarseStep, int fineStep,
                                                              uint32_t boundaryEdgeMask);

// Returns no value outside the far-terrain annulus. Ring endpoints are
// inclusive at the lower resolution and exclusive at the upper resolution.
std::optional<FarTerrainStep> farTerrainStepForChunkDistance(double chunkDistance);

// Selects one of the absolute v4 far-terrain tiers. Exact cubes own the first
// 32 chunks, followed by step 2 through 64, step 4 through 128, step 8 through
// 256, and step 16 through the 512-chunk horizon. Step 32 is coverage-only.
// Outward-only hysteresis keeps ordinary camera motion from chattering.
std::optional<FarTerrainStep>
farTerrainStepForMetrics(double chunkDistance,
                         std::optional<FarTerrainStep> previousStep = std::nullopt);

struct FarTerrainScreenErrorMetrics {
    double distanceBlocks = 0.0;
    double viewportHeightPixels = 0.0;
    double verticalFovRadians = 0.0;
    // Optional viewportHeight / (2 tan(fov / 2)). The renderer fills this
    // once per frame so thousands of tile checks perform no per-tile
    // trigonometry. Tests and other callers may leave it zero.
    double projectionScalePixels = 0.0;
    double tileReliefBlocks = 0.0;
};

// Returns the projected size of one world block at the tile's nearest point.
// Invalid projection inputs return zero so callers retain the absolute v4
// distance contract instead of accidentally requesting unbounded detail.
double farTerrainProjectedBlockPixels(const FarTerrainScreenErrorMetrics& metrics) noexcept;

// Estimates a conservative scheduling error from grid spacing and observed
// tile relief. High-relief tiles retain ridges and silhouettes until their
// omitted scale is below the configured output-pixel threshold.
double
farTerrainProjectedGeometricErrorPixels(FarTerrainStep step,
                                        const FarTerrainScreenErrorMetrics& metrics) noexcept;

// Adds the measured authority-quality error to the sampling error. FINAL has
// no omitted Decoder residual. PREVIEW uses the conservative revision-9
// maximum so scheduling cannot mistake a smoother proxy mesh for canonical
// detail merely because its cells are smaller.
double farTerrainProjectedDisplayErrorPixels(FarTerrainStep step,
                                             FarTerrainAuthorityQuality quality,
                                             const FarTerrainScreenErrorMetrics& metrics) noexcept;

// Screen-space selection may refine beyond the absolute 32/64/128/256/512 chunk
// bands, including step 1 when a coarser far sample remains perceptible. It may
// never select a coarser tier inside those bands. Outward-only hysteresis
// applies to both distance and projected error.
std::optional<FarTerrainStep>
farTerrainStepForScreenMetrics(double chunkDistance, const FarTerrainScreenErrorMetrics& metrics,
                               std::optional<FarTerrainStep> previousStep = std::nullopt);

// A topology replacement keeps complete terrain meshes on either side of a
// narrow fog-covered swap. Canopies exchange monotonically over the full
// duration, while terrain and connected water exchange together at the hidden
// midpoint. The renderer bounds simultaneous pairs.
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

// A FINAL parent promotion must retain its PREVIEW source only while a
// displayed or transitioning PREVIEW refinement still derives from that
// source. The PREVIEW parent itself is not a child dependency. Treating it as
// one makes the promotion wait on its own visible source forever and prevents
// every finer LOD at that coordinate from becoming displayable.
constexpr bool
farTerrainPreviewChildDependsOnParentSource(const FarTerrainKey& parent,
                                            const FarTerrainKey& candidate,
                                            FarTerrainAuthorityQuality candidateQuality) noexcept {
    return farTerrainIsBaseStep(parent.step) && candidate != parent &&
           !farTerrainIsBaseStep(candidate.step) &&
           candidateQuality == FarTerrainAuthorityQuality::PREVIEW;
}

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
    std::optional<FarTerrainScreenErrorMetrics> screenErrorMetrics;
};

enum class FarTerrainProtectedNearRole : uint8_t {
    NONE = 0,
    STEP_ONE_CORE,
    STEP_TWO_RING,
    STEP_FOUR_RING,
    STEP_EIGHT_RING,
    STEP_SIXTEEN_RING,
};

// Only the one role-selected key at a protected coordinate belongs to its
// atomic closure. Other resident LODs at the same coordinate remain optional.
bool farTerrainProtectedNearTargetKey(const std::optional<ColumnPos>& anchor,
                                      FarTerrainKey key) noexcept;

// Full-arena refinement admission is restricted to a FINAL key in the
// currently requested closure. Active, predicted, PREVIEW, parent, and
// alternate-step keys do not receive this exceptional admission class.
bool farTerrainCriticalProtectedRefinement(const std::optional<ColumnPos>& requestedAnchor,
                                           FarTerrainKey key,
                                           FarTerrainAuthorityQuality quality) noexcept;

// Tracks one published protected anchor and at most one requested replacement.
// Camera movement retains both anchors until the complete requested topology
// is resident. The renderer then publishes the replacement atomically and
// releases the old anchor through ordinary LOD transitions.
class FarTerrainProtectedNearHandoff {
public:
    bool request(ColumnPos anchor) noexcept;
    bool commitRequested(bool ready) noexcept;
    void clear() noexcept;

    const std::optional<ColumnPos>& activeCenter() const noexcept { return activeCenter_; }
    const std::optional<ColumnPos>& requestedCenter() const noexcept { return requestedCenter_; }
    std::optional<ColumnPos> statusCenter() const noexcept {
        return requestedCenter_ ? requestedCenter_ : activeCenter_;
    }

private:
    std::optional<ColumnPos> activeCenter_;
    std::optional<ColumnPos> requestedCenter_;
};

// Returns the minimum corner of the protected 2x2 core. Each containing tile
// pairs with the adjacent tile toward its nearer half-tile boundary. Floor
// division and modulo keep the rule continuous and symmetric below zero.
constexpr ColumnPos farTerrainProtectedNearAnchor(int64_t cameraBlockX,
                                                  int64_t cameraBlockZ) noexcept {
    const auto axisAnchor = [](int64_t block) {
        constexpr int64_t EDGE = FAR_TERRAIN_TILE_EDGE;
        constexpr int32_t HALF_EDGE = FAR_TERRAIN_TILE_EDGE / 2;
        int64_t containing = world_coord::floorDiv(block, EDGE);
        if (world_coord::floorMod(block, FAR_TERRAIN_TILE_EDGE) < HALF_EDGE &&
            containing != std::numeric_limits<int64_t>::min()) {
            --containing;
        }
        return containing;
    };
    return {axisAnchor(cameraBlockX), axisAnchor(cameraBlockZ)};
}

// Predicts at most one adjacent protected anchor from recent movement. An
// axis advances only inside the half-tile lead band and only when the future
// core already contains the camera. The result is a CPU prefetch hint. It
// never changes the canonical handoff or weakens its spatial cushion.
constexpr std::optional<ColumnPos>
farTerrainPredictedProtectedNearAnchor(int64_t cameraBlockX, int64_t cameraBlockZ,
                                       int recentMotionX, int recentMotionZ) noexcept {
    if (recentMotionX == 0 && recentMotionZ == 0) return std::nullopt;
    const ColumnPos current = farTerrainProtectedNearAnchor(cameraBlockX, cameraBlockZ);
    const auto predictAxis = [](int64_t block, int motion, int64_t anchor) {
        if (motion == 0) return anchor;
        constexpr __int128 EDGE = FAR_TERRAIN_TILE_EDGE;
        constexpr __int128 HALF_EDGE = FAR_TERRAIN_TILE_EDGE / 2;
        const __int128 canonical = anchor;
        const __int128 boundary =
            motion > 0 ? (canonical + 1) * EDGE + HALF_EDGE : canonical * EDGE + HALF_EDGE;
        const __int128 distance = motion > 0 ? boundary - block : block - boundary;
        if (distance < 0 || distance > FAR_TERRAIN_PROTECTED_NEAR_PREDICTION_LEAD_BLOCKS)
            return anchor;
        const __int128 candidate = canonical + (motion > 0 ? 1 : -1);
        if (candidate < std::numeric_limits<int64_t>::min() ||
            candidate > std::numeric_limits<int64_t>::max()) {
            return anchor;
        }
        const __int128 minimum = candidate * EDGE;
        const __int128 maximum = minimum + FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES * EDGE;
        if (static_cast<__int128>(block) < minimum || static_cast<__int128>(block) >= maximum)
            return anchor;
        return static_cast<int64_t>(candidate);
    };
    const ColumnPos predicted{predictAxis(cameraBlockX, recentMotionX, current.x),
                              predictAxis(cameraBlockZ, recentMotionZ, current.z)};
    return predicted == current ? std::nullopt : std::optional<ColumnPos>{predicted};
}

FarTerrainProtectedNearRole farTerrainProtectedNearRole(ColumnPos anchor,
                                                        ColumnPos coordinate) noexcept;
std::optional<FarTerrainStep>
farTerrainProtectedNearRequiredStep(const FarTerrainProtectedNearHandoff& handoff,
                                    ColumnPos coordinate) noexcept;

constexpr bool farTerrainStepOneResidencyRequired(double chunkDistance) noexcept {
    return chunkDistance >= 0.0 && chunkDistance < FAR_TERRAIN_NEAR_CHUNK_RADIUS;
}

constexpr FarTerrainStep farTerrainResidencyTarget(const FarTerrainViewTile& tile) noexcept {
    return farTerrainStepOneResidencyRequired(tile.distanceChunks) ? FarTerrainStep::ONE
                                                                   : tile.key.step;
}

struct FarTerrainNativeHydrologyDependency {
    int64_t ownerPageX = 0;
    int64_t ownerPageZ = 0;
    worldgen::learned::NativeRect finalTerrainRegion;

    auto operator<=>(const FarTerrainNativeHydrologyDependency&) const = default;
};

// The FINAL step-32 generator path has a smaller learned dependency set than
// the complete PREVIEW topology closure. World support is half-open and covers
// every surface vertex, conservative cell, topology cell, center probe,
// canonical native-water sample, and bounded volcanic refinement position.
struct FarTerrainFinalBaseAuthorityDependencies {
    int64_t minimumWorldX = 0;
    int64_t minimumWorldZ = 0;
    int64_t maximumWorldXExclusive = 0;
    int64_t maximumWorldZExclusive = 0;
    std::vector<worldgen::learned::TerrainPageCoordinate> geometryPages;
    std::vector<FarTerrainNativeHydrologyDependency> nativeHydrology;
    // When one already-required transient hydrology input contains every
    // bilinear terrain sample, FINAL geometry can reuse a deterministic crop
    // of that same quantized grid instead of inferring persistent pages.
    std::optional<worldgen::learned::NativeRect> transientGeometryRegion;
};

// Enumerates every 256-block tile intersecting the circular visible disk in
// nearest-first order. Exact ownership is resolved separately from current
// cubic mesh residency and never removes the coarse parent needed to cover
// cold or capped exact terrain.
void selectFarTerrainView(double cameraX, double cameraZ, int visibleChunkRadius,
                          std::vector<FarTerrainViewTile>& output);

// Returns the deduplicated preview-authority pages required to route every
// selected base parent and its sampling apron. Pages are nearest-first so the
// bounded learned-authority coordinator can warm a visible horizon
// without waiting for individual mesh jobs to discover them repeatedly.
std::vector<worldgen::learned::TerrainPageCoordinate>
farTerrainCoarseAuthorityPages(std::span<const FarTerrainViewTile> selected, double cameraX,
                               double cameraZ);

// Plans only the FINAL learned dependencies actually consumed by one base
// mesh. Unlike farTerrainCoarseAuthorityPages, this does not materialize every
// persistent terrain page beneath a native hydrology owner's 2,048-block
// topology closure. Native owners use their exact transient 517 by 517 input.
FarTerrainFinalBaseAuthorityDependencies
farTerrainFinalBaseAuthorityDependencies(FarTerrainKey key);

// Coalesces the native hydrology inputs shared by a protected near patch into
// deterministic groups of at most 2x2 adjacent owners. Every canonical owner
// remains an exact crop of one returned rectangle; the grouping only removes
// duplicate learned inference across overlapping aprons.
std::vector<worldgen::learned::NativeRect>
farTerrainProtectedFinalTerrainRegions(std::span<const FarTerrainKey> targets);

// Selects a bounded row of authority pages immediately beyond the visible
// closure in the camera's recent direction of travel. The caller updates this
// plan only after moving at least one chunk. These pages are hints only and may
// never delay the visible preview or protected handoff lanes.
std::vector<worldgen::learned::TerrainPageCoordinate> farTerrainSpeculativeAuthorityPages(
    std::span<const worldgen::learned::TerrainPageCoordinate> visiblePages, double previousCameraX,
    double previousCameraZ, double cameraX, double cameraZ);

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

// Enumerates one protected 2x2 step-1 core and its four Manhattan-distance
// rings. Finer targets precede coarser targets and each group is lexicographic,
// making scheduling and diagnostics query-order stable. Selection clips only
// the outer edge when it coincides with the configured horizon.
void buildFarTerrainProtectedNearTargets(ColumnPos anchor,
                                         std::span<const FarTerrainViewTile> selected,
                                         std::vector<FarTerrainKey>& targets);
bool farTerrainProtectedNearTargetsReady(std::span<const FarTerrainKey> targets,
                                         const FarTerrainResidencyFunction& isCompatibleResident);

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

// A connected prefix through the finest settled band may spend the bounded
// urgent lane on nearby protected refinements while the outer horizon keeps
// streaming. Ordinary optional work remains gated separately.
bool farTerrainConnectedRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier) noexcept;

// The broad optional refinement lane stays closed until the complete parent
// disk is resident and the current submission scan reached every parent.
// Connected selected targets and bounded near fallbacks use the earlier lane.
bool farTerrainRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier,
                                  bool allBaseCandidatesScanned);

// Scheduler ordering is lane-first, then nearest-view priority. A refinement
// cannot jump ahead of a parent merely because its tile is closer.
bool farTerrainSubmissionBefore(FarTerrainKey first, uint32_t firstViewPriority,
                                FarTerrainKey second, uint32_t secondViewPriority);

// Appends every coarse parent first. Critical camera, protected, and exact
// fallback coordinates then advance through global adjacent wavefronts before
// the broad fallback repeats the same step-16 through step-1 order. A distant
// bridge can therefore never evict a finer critical result from the CPU cache.
void buildFarTerrainResidencyOrder(const std::vector<FarTerrainViewTile>& selected,
                                   std::vector<FarTerrainKey>& output,
                                   std::span<const ColumnPos> criticalCoordinates = {});

// Builds the independent cache-protection order for camera-critical keys.
// Every required target ranks ahead of every parent, and every parent ranks
// ahead of intermediate bridge copies. This prevents one coordinate's bridge
// lineage from denying an adjacent core surface under exceptional pressure.
void buildFarTerrainCriticalResidencyOrder(std::span<const FarTerrainKey> targets,
                                           std::vector<FarTerrainKey>& output);

// Builds two complete critical classes without interleaving their lineages.
// Current exact and protected ownership therefore ranks ahead of every
// directional prediction, while overlapping immutable keys are retained once.
void buildFarTerrainTieredCriticalResidencyOrder(std::span<const FarTerrainKey> currentTargets,
                                                 std::span<const FarTerrainKey> predictedTargets,
                                                 std::vector<FarTerrainKey>& output);

// Checks the two-lane order without rebuilding it. Stable camera frames can
// retain both the renderer's set and the scheduler's immutable wanted filter.
bool farTerrainResidencyOrderMatches(const std::vector<FarTerrainViewTile>& selected,
                                     std::span<const FarTerrainKey> order,
                                     std::span<const ColumnPos> criticalCoordinates = {});

// Wanted residency is a set contract. Camera motion may reorder the same
// coordinates by a few places without changing which immutable meshes are
// needed. Keeping that distinction out of retainWanted avoids rebuilding
// cache priority tables and taking three scheduler locks on those frames.
bool farTerrainResidencyMembershipMatches(
    const std::vector<FarTerrainViewTile>& selected,
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
    std::span<const FarTerrainKey> additionalKeys = {});

// Retained for protected exact-handoff and diagnostic step-1 selections. The
// ordinary far selector never emits step 1 because exact cubes own that band.
bool buildFarTerrainConnectedNearPatchHandoff(std::span<const FarTerrainViewTile> selected,
                                              const FarTerrainResidencyFunction& isResident,
                                              std::vector<FarTerrainKey>& targets);

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

// A revision-ready exact surface section and its far parent must never draw
// at the same time. A partially published column keeps its parent as the
// gap-free owner, so required exact sections become drawable atomically only
// after the complete column handoff is ready. If the parent is unavailable,
// exact geometry remains the conservative no-gap fallback.
bool farTerrainExactSectionDrawAllowed(bool sectionRequired, bool columnFullyReady,
                                       bool coverageParentDrawable) noexcept;

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

// Lists every selected step-32 parent that still lacks FINAL authority.
// Parents intersecting the exact handoff come first in nearest-first order;
// the return value is the length of that required prefix. Startup waits only
// for this prefix, while the remaining preview horizon can refine in gameplay.
uint32_t buildFarTerrainFinalParentUpgradeOrder(std::span<const FarTerrainViewTile> selected,
                                                double cameraX, double cameraZ,
                                                float nominalExactDistanceBlocks,
                                                const FarTerrainExactHandoff& handoff,
                                                const FarTerrainResidencyFunction& isFinalResident,
                                                std::vector<FarTerrainKey>& output);

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
    bool sectionRequired(ChunkPos section) const;
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
struct FarTerrainWaterTopologySignature {
    uint64_t bodyIdentityHash = 0;
    uint64_t transitionIdentityHash = 0;
    uint64_t connectivityHash = 0;
    uint32_t bodyIdentityCount = 0;
    uint32_t transitionIdentityCount = 0;
    uint32_t connectivityCount = 0;

    auto operator<=>(const FarTerrainWaterTopologySignature&) const = default;
};

enum class FarTerrainBoundaryEdge : uint8_t {
    WEST = 0,
    EAST = 1,
    NORTH = 2,
    SOUTH = 3,
};

// Every far payload records the canonical two-block height samples used by
// the no-skirt transition topology. Opposite edges from adjacent meshes built
// from one authority quality must have identical hashes. This is compact
// enough to retain with GPU residency and lets entry fail closed before a
// mixed or incomplete protected patch becomes visible.
struct FarTerrainSurfaceBoundarySignature {
    static constexpr size_t SAMPLE_COUNT =
        FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1;

    std::array<uint64_t, 4> heightHashes{};
    bool valid = false;

    auto operator<=>(const FarTerrainSurfaceBoundarySignature&) const = default;
};

constexpr bool
farTerrainAuthorityPromotionPreservesWater(const FarTerrainWaterTopologySignature& preview,
                                           const FarTerrainWaterTopologySignature& final) noexcept {
    return preview == final;
}

enum class FarTerrainWaterPromotionAction : uint8_t {
    MATCHED_TOPOLOGY_TRANSITION,
    ATOMIC_TOPOLOGY_SWAP,
};

// PREVIEW and FINAL are separate learned surfaces, so refinement may reveal or
// remove a complete water body. A differing topology is admitted as one
// terrain-and-water exchange at the fog-covered midpoint. Exact ownership is
// not a prerequisite for distant FINAL convergence.
constexpr FarTerrainWaterPromotionAction
farTerrainWaterPromotionAction(const FarTerrainWaterTopologySignature& preview,
                               const FarTerrainWaterTopologySignature& final) noexcept {
    if (farTerrainAuthorityPromotionPreservesWater(preview, final)) {
        return FarTerrainWaterPromotionAction::MATCHED_TOPOLOGY_TRANSITION;
    }
    return FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP;
}

struct FarTerrainMesh {
    FarTerrainKey key;
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL;
    // Protected FINAL publication still requires exact-compatible shared
    // boundaries. Per-column exact visual ownership is controlled separately
    // so a temporary PREVIEW surface can never conceal loaded collision.
    bool exactAuthorityCompatible = true;
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
    uint32_t mergedTerrainCellCount = 0;
    uint32_t transitionTriangleCount = 0;
    // Step-32 startup diagnostics distinguish bounded sparse rectangles from
    // the complete 66-by-66 native-water page. These counters describe source
    // samples requested while building this immutable payload and do not
    // participate in its deterministic geometry hash.
    uint32_t step32WaterGridCallCount = 0;
    uint32_t step32WaterGridSampleCount = 0;
    uint32_t step32WaterPointSampleCount = 0;
    uint32_t step32WaterDenseGridCallCount = 0;
    float complexity = 0.0F;
    uint64_t deterministicHash = 0;
    FarTerrainWaterTopologySignature waterTopology;
    FarTerrainSurfaceBoundarySignature surfaceBoundary;

    size_t byteSize() const;
};

struct FarTerrainProtectedNearSurface {
    FarTerrainKey key;
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    FarTerrainAuthorityQuality parentAuthorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    bool exactAuthorityCompatible = false;
    FarTerrainSurfaceBoundarySignature surfaceBoundary;
};

struct FarTerrainProtectedNearGeometryStatus {
    size_t expectedTargets = 0;
    size_t presentTargets = 0;
    size_t finalTargets = 0;
    size_t finalParents = 0;
    size_t expectedFinalParents = 0;
    size_t exactCompatibleTargets = 0;
    size_t expectedSharedBoundaries = 0;
    size_t matchingSharedBoundaries = 0;
    size_t mismatchedSharedBoundaries = 0;
    size_t incompatibleLodBoundaries = 0;

    [[nodiscard]] bool ready() const noexcept {
        return expectedTargets != 0 && presentTargets == expectedTargets &&
               finalTargets == expectedTargets && finalParents == expectedFinalParents &&
               exactCompatibleTargets == expectedTargets &&
               matchingSharedBoundaries == expectedSharedBoundaries &&
               mismatchedSharedBoundaries == 0 && incompatibleLodBoundaries == 0;
    }
};

// Validates the selected protected representation as one visual publication
// unit. Residency alone is insufficient: every target and parent must consume
// FINAL authority, every shared boundary must match, and every cardinal LOD
// ratio must remain at most 2:1.
FarTerrainProtectedNearGeometryStatus
farTerrainProtectedNearGeometryStatus(ColumnPos anchor,
                                      std::span<const FarTerrainKey> expectedTargets,
                                      std::span<const FarTerrainProtectedNearSurface> surfaces);

// Wall-clock build measurements are diagnostic only and never participate in
// attachment identity. Keeping each phase separate makes a cold authority
// wait distinguishable from CPU-heavy ecology or grounding work in captures.
struct FarCanopyBuildDiagnostics {
    uint64_t canopyCollectionMicroseconds = 0;
    uint64_t floraCollectionMicroseconds = 0;
    uint64_t groundingMicroseconds = 0;
    uint64_t geometryMicroseconds = 0;
    uint64_t totalMicroseconds = 0;
    uint32_t canopyCandidateCount = 0;
    uint32_t floraCandidateCount = 0;
    uint32_t acceptedCanopyCount = 0;
    uint32_t acceptedFloraCount = 0;
    uint32_t occupiedGroundCellCount = 0;
    uint32_t sparseGroundCellCount = 0;
    uint32_t denseGroundGridSampleCount = 0;
    uint32_t transitionGroundSampleCount = 0;

    bool operator==(const FarCanopyBuildDiagnostics&) const = default;
};

// Optional flora geometry is resident independently from the terrain and
// water payload. A missing, canceled, or delayed attachment never changes the
// base tile's drawability or lifetime.
struct FarCanopyAttachment {
    FarTerrainKey key;
    // Ecology authority owns anchor identity, species, and acceptance. It is
    // intentionally independent from the terrain quality used to ground the
    // same anchors while a preview surface remains drawable.
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL;
    FarTerrainAuthorityQuality groundingQuality = FarTerrainAuthorityQuality::FINAL;
    int64_t originX = 0;
    int64_t originZ = 0;
    FarTerrainBounds bounds;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t canopyAnchorCount = 0;
    uint32_t canopyImpostorQuadCount = 0;
    uint32_t floraAnchorCount = 0;
    uint32_t floraImpostorQuadCount = 0;
    uint64_t anchorIdentityHash = 0;
    uint64_t deterministicHash = 0;
    FarCanopyBuildDiagnostics buildDiagnostics;

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
    // A wetland is a shallow, parent-owned standing-water fringe. Its body ID
    // and stage must remain distinct from an arbitrary dry habitat flag at
    // every far level.
    bool wetland = false;
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
// corner. Conservative coverage never emits geometry or depresses the
// displayed terrain.
struct FarTerrainCellBounds {
    double terrainHeight = std::numeric_limits<double>::quiet_NaN();
    double minimumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    double maximumTerrainHeight = std::numeric_limits<double>::quiet_NaN();
    // True when analytical channels, lake or ocean contours, or volcanic water
    // can intersect the cell even though its coarse
    // corners and center are dry. Coverage parents use this only to request
    // canonical water probes; it never changes displayed terrain height.
    bool waterTopologyPossible = false;
    // Volcanic islands and crater lakes modify canonical hydrology after the
    // basin solve. Only cells carrying this bit need the more expensive final
    // volcanic water callback; every other cell can use direct hydrology.
    bool volcanicWaterPossible = false;
    // A routed channel passes close enough to the cell that a native-grid
    // waterfall anchor may lie between its displayed terrain samples. This
    // keeps the anchor scan bounded to channel neighborhoods rather than
    // probing every coarse ocean parent.
    bool waterfallPossible = false;
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
    // Optional terrain-only fast path for densely occupied flora attachments.
    // The result is the exact displayed top of every cell in a square grid,
    // without rebuilding canonical water and conservative coverage bounds.
    using TerrainCellTopGridFunction =
        std::function<void(int64_t originX, int64_t originZ, int step, int cellEdge,
                           worldgen::SurfaceFootprint footprint, std::span<float> output)>;
    // Sparse flora usually occupies only a small fraction of a tile. This
    // callback returns the displayed top for the requested row-major cell
    // indices without forcing a complete 257-by-257 step-one terrain query.
    using TerrainCellTopPointFunction =
        std::function<void(int64_t originX, int64_t originZ, int step, int cellEdge,
                           worldgen::SurfaceFootprint footprint,
                           std::span<const uint32_t> occupiedCells, std::span<float> output)>;
    using CanopyFunction =
        std::function<std::vector<FarCanopy>(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                             int64_t maximumZ, FarTerrainStep step)>;
    using FloraFunction =
        std::function<std::vector<FarFlora>(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                            int64_t maximumZ, FarTerrainStep step)>;
    using MaterialRankFunction = std::function<double(int64_t worldX, int64_t worldZ)>;
    using FinalBaseAuthorityDependenciesFunction =
        std::function<FarTerrainFinalBaseAuthorityDependencies(FarTerrainKey key)>;

    SampleFunction sample;
    GridSampleFunction sampleGrid;
    GeometryGridSampleFunction geometryGrid;
    // Canonical shoreline refinement requests sparse globally aligned points
    // in one immutable batch. The production generator groups them by basin,
    // avoiding thousands of one-point hydrology calls in a coarse parent.
    GeometryPointSampleFunction geometryPoints;
    // Immutable native hydrology evaluated before the expensive emitted-terrain
    // contact pass. These callbacks preserve solved body identity, stage, and
    // shoreline topology for distant water while leaving exact geometryPoints
    // available for the bounded volcanic and near-field exceptions.
    GeometryGridSampleFunction canonicalWaterGrid;
    GeometryPointSampleFunction canonicalWaterPoints;
    // Canonical basin authority without post-hydrology volcanic overlays.
    // Step-32 coverage uses the regular grid for globally aligned half-open
    // water cells and the point counterpart only for bounded interior
    // recovery probes. Both callbacks must return identical authority at the
    // same coordinate.
    GeometryGridSampleFunction waterAuthorityGrid;
    GeometryPointSampleFunction waterAuthorityPoints;
    CellBoundsGridFunction cellBoundsGrid;
    TerrainCellTopGridFunction terrainCellTopGrid;
    TerrainCellTopPointFunction terrainCellTopPoints;
    CanopyFunction canopies;
    FloraFunction flora;
    MaterialRankFunction materialRank;
    // Learned native hydrology supplies a complete 32-block topology proof.
    // The step-32 mesher may then query only ambiguous native-water cells;
    // sources without that proof retain the dense canonical raster.
    bool sparseStep32Water = false;
    // Learned coarse tiers can consume macro terrain and native hydrology
    // directly. The mesher carries their actual footprint into every water
    // callback so block-exact ColumnPlans and ordinary stage tiles remain
    // exclusive to exact terrain.
    bool planFreeCoarseAuthority = false;
    // Present only when FINAL base meshes use learned native hydrology. Test
    // and legacy sources omit this and retain the generic page-only gate.
    FinalBaseAuthorityDependenciesFunction finalBaseAuthorityDependencies;
};

class FarTerrainMesher {
public:
    using SurfaceSampleFunction = std::function<worldgen::SurfaceSample(
        int64_t worldX, int64_t worldZ, worldgen::SurfaceFootprint footprint)>;
    using BlockSurfaceSampleFunction =
        std::function<worldgen::SurfaceSample(int64_t worldX, int64_t worldZ)>;

    // Pure CPU base build. All world samples use globally aligned int64
    // coordinates. The immutable result contains terrain, standing water, and
    // falls only; optional flora is always a separate attachment.
    static std::shared_ptr<const FarTerrainMesh>
    build(FarTerrainKey key, const FarTerrainSource& source,
          FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL);
    static std::shared_ptr<const FarCanopyAttachment> buildCanopyAttachment(
        FarTerrainKey key, const FarTerrainSource& source,
        FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL);
    static std::shared_ptr<const FarCanopyAttachment> buildCanopyAttachment(
        FarTerrainKey key, const FarTerrainSource& ecologySource,
        const FarTerrainSource& groundingSource, FarTerrainAuthorityQuality groundingQuality,
        FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL,
        FarCanopyBuildDiagnostics* diagnostics = nullptr);
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

private:
    static std::shared_ptr<const FarTerrainMesh>
    buildInternal(FarTerrainKey key, const FarTerrainSource& source,
                  FarTerrainAuthorityQuality authorityQuality);
};

struct FarTerrainResult {
    FarTerrainKey key;
    uint64_t epoch = 0;
    std::shared_ptr<const FarTerrainMesh> mesh;
    bool failed = false;
};

struct FarCanopyResult {
    FarTerrainKey key;
    uint64_t epoch = 0;
    std::shared_ptr<const FarCanopyAttachment> attachment;
    bool failed = false;
};

struct FarTerrainSchedulerLimits {
    size_t maxPending = 64;
    size_t maxCompleted = 32;
    // A screen-error-refined 512-chunk selection can retain all six geometric
    // tiers for a bounded subset of its roughly 3,400 tiles. The entry cap
    // covers even the conservative all-tier set with deterministic margin;
    // the independent byte cap remains the real memory bound.
    size_t maxCacheEntries = FAR_TERRAIN_MAX_RESIDENCY_KEYS;
    size_t maxCacheBytes = 3ull * 1024 * 1024 * 1024;
    size_t maxCanopyPending = 64;
    size_t maxCanopyCompleted = 32;
    size_t maxCanopyCacheEntries = FAR_TERRAIN_MAX_RESIDENCY_KEYS;
    size_t maxCanopyCacheBytes = 512ull * 1024 * 1024;
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
    // Base jobs waiting on their known preview-authority closure retain their
    // scheduler slot but do not occupy a terrain worker or re-enter it until
    // a dependency becomes ready.
    size_t parkedBase = 0;
    // A stronger request for a key already running retains that key's single
    // scheduler slot and becomes its deterministic next build.
    size_t terrainFollowups = 0;
    size_t queuedRefinement = 0;
    size_t queuedUrgentRefinement = 0;
    size_t activeUrgentRefinement = 0;
    size_t urgentRefinementInFlight = 0;
    size_t visibleFinalParentInFlight = 0;
    size_t completedBase = 0;
    size_t completedRefinement = 0;
    size_t cacheBaseEntries = 0;
    size_t canopyInFlight = 0;
    size_t activeCanopyWorkers = 0;
    size_t queuedCanopy = 0;
    size_t parkedCanopy = 0;
    size_t completedCanopy = 0;
    size_t canopyCacheEntries = 0;
    size_t canopyCacheBytes = 0;
    uint64_t epoch = 0;
    uint64_t submitted = 0;
    uint64_t built = 0;
    uint64_t canceled = 0;
    uint64_t criticalDisplacements = 0;
    uint64_t failed = 0;
    uint64_t deferred = 0;
    uint64_t step32WaterGridCalls = 0;
    uint64_t step32WaterGridSamples = 0;
    uint64_t step32WaterPointSamples = 0;
    uint64_t step32WaterDenseGridCalls = 0;
    // Worker retries admitted by a newly observable learned-authority result.
    // This remains distinct from render submissions and terminal deferrals.
    uint64_t authorityCompletionResumes = 0;
    // A native-hydrology deferral can complete its shared reconciliation
    // without publishing another learned-authority result. When every
    // producer is idle, one parked FINAL parent is retried as a bounded
    // liveness probe instead of waiting forever on an impossible completion.
    uint64_t quiescentAuthorityResumes = 0;
    uint64_t canopySubmitted = 0;
    uint64_t canopyBuilt = 0;
    uint64_t canopyCanceled = 0;
    uint64_t canopyFailed = 0;
    uint64_t canopyDeferred = 0;
    uint64_t canopyAuthorityCompletionResumes = 0;
    uint64_t canopyCacheHits = 0;
    uint64_t cacheHits = 0;
    uint64_t wantedUpdates = 0;
    uint64_t wantedNoops = 0;
    // Camera-relative critical ordering is intentionally independent of the
    // full horizon membership revision. Routine movement can refresh this
    // bounded lane without copying the complete residency maps.
    uint64_t criticalPriorityUpdates = 0;
    uint64_t criticalPriorityNoops = 0;
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
    bool cameraTile = false;
    bool visible = false;
    bool displayableWavefront = true;
    double projectedErrorPixels = 0.0;
    double distanceSquaredBlocks = std::numeric_limits<double>::infinity();
    // Only the bounded protected closure may bypass intermediate mesh builds.
    // Its complete step-1, 2, 4, 8, and 16 topology publishes atomically.
    bool protectedNearTarget = false;
};

// Orders the connected refinement lane by protected-handoff need and visible
// projected error, then emits the next missing adjacent tier. Expensive fine
// targets cannot occupy the bounded lane before the step-16, step-8, and
// step-4 bridges that can immediately replace a step-32 parent while retaining
// the 2:1 boundary contract. Duplicate and already resident keys are omitted.
void buildFarTerrainProgressiveSubmissionOrder(
    std::span<const FarTerrainRefinementCacheRequest> requests, std::vector<FarTerrainKey>& output,
    size_t maximumResults = std::numeric_limits<size_t>::max());

// Leaves at most one nondeferred intermediate request per free transition
// slot. Desired-tier results remain eligible because they are residency-pinned
// even when this flag is set.
size_t
reserveFarTerrainIntermediateTransitionSlots(std::span<FarTerrainRefinementCacheRequest> requests,
                                             size_t activeTransitions);

// Fixed-bin timing avoids allocating or sorting on the render thread while
// retaining an exact maximum and a conservative cumulative p95 for diagnostics.
class FarTerrainPlannerTimingHistogram {
public:
    static constexpr size_t BIN_COUNT = 256;
    static constexpr double BIN_WIDTH_MILLISECONDS = 0.1;

    void clear() noexcept;
    void record(double milliseconds) noexcept;
    float percentile95Milliseconds() const noexcept;
    float maximumMilliseconds() const noexcept { return maximumMilliseconds_; }
    uint64_t sampleCount() const noexcept { return sampleCount_; }

private:
    std::array<uint64_t, BIN_COUNT> bins_{};
    uint64_t sampleCount_ = 0;
    float maximumMilliseconds_ = 0.0F;
};

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
    // The v4 native router admits at most sixteen independent page builds.
    // Match the 16 physical-core startup target without making far terrain a
    // source of unbounded concurrency; the lower-priority canopy lane stays
    // separate and can be paused during coarse coverage.
    static constexpr size_t WORKER_COUNT = 16;
    static constexpr size_t LATENCY_WORKER_COUNT = WORKER_COUNT / 2;
    static constexpr size_t CANOPY_WORKER_COUNT = 2;

    explicit FarTerrainScheduler(FarTerrainSource source, FarTerrainSchedulerLimits limits = {});
    // Retains a source's normal mesh callbacks while adding the learned
    // preview-authority gate. This keeps deterministic test and diagnostic
    // sources on the same cold-horizon scheduling path as production.
    FarTerrainScheduler(
        FarTerrainSource source,
        std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
        FarTerrainSchedulerLimits limits = {});
    // Seed-owned schedulers must use the same gameplay generation settings as
    // exact cubic emission. This keeps structures and every future generation
    // toggle consistent across exact, preview, and final far geometry.
    explicit FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits = {},
                                 GenerationSettings generation = {});
    FarTerrainScheduler(
        uint64_t worldSeed,
        std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
        FarTerrainSchedulerLimits limits = {}, GenerationSettings generation = {});
    ~FarTerrainScheduler();

    FarTerrainScheduler(const FarTerrainScheduler&) = delete;
    FarTerrainScheduler& operator=(const FarTerrainScheduler&) = delete;

    bool enqueue(FarTerrainKey key, uint32_t viewPriority = std::numeric_limits<uint32_t>::max());
    // A missing step-32 parent in the current camera or protected closure is
    // coverage, but it must not wait behind the ordinary horizon parent cap.
    // This bounded lane prepares the PREVIEW parent at protected priority and
    // may replace a farther queued or authority-parked ordinary parent. It
    // never removes an already resident drawable parent.
    bool enqueueUrgentCoverage(FarTerrainKey key,
                               uint32_t viewPriority = std::numeric_limits<uint32_t>::max());
    // Replaces an already drawable preview step-32 payload with the same
    // geometric LOD sampled from final authority. The renderer performs the
    // GPU replacement atomically before it admits any final child tier.
    bool enqueueFinalBase(FarTerrainKey key,
                          uint32_t viewPriority = std::numeric_limits<uint32_t>::max(),
                          bool protectedHandoff = false);
    // The renderer explicitly requests flora only after a surface is
    // displayed. The optional lane publishes PREVIEW ecology immediately
    // when FINAL ecology is deferred, then replaces it atomically after the
    // final anchors are resident. Either ecology quality may be grounded
    // against the temporary PREVIEW surface or its FINAL replacement without
    // consuming terrain submission capacity.
    bool
    enqueueCanopy(FarTerrainKey key, uint32_t viewPriority = std::numeric_limits<uint32_t>::max(),
                  FarTerrainAuthorityQuality groundingQuality = FarTerrainAuthorityQuality::FINAL);
    // Connected refinements may pass queued base jobs only inside the fixed
    // urgent quota. Four workers are nominally reserved for the base lane
    // whenever the full worker budget and queued base work permit it. An
    // urgent camera-critical refinement bypasses that reservation so an
    // unrelated distant parent cannot preserve near-player visual debt.
    bool enqueueUrgentRefinement(FarTerrainKey key,
                                 uint32_t viewPriority = std::numeric_limits<uint32_t>::max(),
                                 bool cameraNearCritical = false);
    // Protected exact-handoff refinements use FINAL authority without giving
    // up the bounded urgent lane. This prevents a ready preview step from
    // becoming the long-lived visual neighbor of exact final geometry.
    bool enqueueUrgentFinalRefinement(FarTerrainKey key,
                                      uint32_t viewPriority = std::numeric_limits<uint32_t>::max(),
                                      bool cameraNearCritical = false);
    // Upgrades an already useful preview refinement from final authority in a
    // lower-priority lane. Protected final handoff work uses the urgent method
    // above while ordinary final refinement remains optional.
    bool enqueueFinalRefinement(FarTerrainKey key,
                                uint32_t viewPriority = std::numeric_limits<uint32_t>::max(),
                                bool cameraNearCritical = false);
    // Lock-free conservative capacity probe for bounded producer scans. A
    // subsequent enqueue may still lose a race to another producer.
    bool hasSubmissionCapacity() const noexcept;
    bool hasUrgentRefinementCapacity() const noexcept;
    // Exact exploration and collision work may temporarily reserve CPU and
    // cold-hydrology capacity. Running jobs finish, then no more than this
    // many far workers may enter mesh construction.
    void setWorkerBudget(size_t budget);
    // Gameplay enables this while exact or nearer desired-LOD debt exists.
    // Urgent connected refinements then use every admitted far worker and may
    // displace queued ordinary horizon parents. Preparation retains the base
    // reservation needed to establish the first connected prefix.
    void setNearFirstWorkEnabled(bool enabled);
    // Cold entry admits the bounded protected FINAL closure after its
    // connected PREVIEW parent prefix is drawable. Ordinary perceptual FINAL
    // work remains disabled until gameplay.
    void setFinalStreamingWorkEnabled(bool enabled) noexcept;
    // Flora is optional. A zero budget preserves its bounded queue but holds
    // new attachment construction until terrain coverage can spare the CPU
    // and shared generation authority.
    void setCanopyWorkerBudget(size_t budget);
    // The render thread supplies the deduplicated preview pages covering its
    // current base horizon, then pumps only nonblocking page requests. This
    // is intentionally separate from mesh scheduling because authority-page
    // construction has its own bounded coordinator.
    void
    setCoarseAuthorityPrefetchPages(std::vector<worldgen::learned::TerrainPageCoordinate> pages);
    void pumpCoarseAuthorityPrefetch();
    // Movement-ahead hints use the final priority lane and are submitted only
    // after every visible preview page has reached a terminal ready state.
    void setSpeculativeAuthorityPrefetchPages(
        std::vector<worldgen::learned::TerrainPageCoordinate> pages);
    void pumpSpeculativeAuthorityPrefetch();
    // Polls the bounded parked FINAL parent upgrades. This is separate
    // from preview horizon prefetch because the two quality caches complete
    // independently even when their page coordinates are identical.
    void pumpFinalBaseAuthority();
    // Canopy jobs retain their bounded scheduler slot while final ecology is
    // cold. A completed authority request wakes them without render-thread
    // resubmission or rebuilding their already resident terrain surface.
    void pumpCanopyAuthority();
    // Cancels the bounded job and completion queues immediately, refreshes
    // canopy priorities when only the nearest-first view order changed, then
    // asks a utility worker to retire obsolete cache records in bounded passes.
    // Running obsolete jobs are discarded before publication. This call never
    // scans or destroys the large CPU mesh cache. criticalKeys is a distinct
    // highest-priority-first order; its first entry may reclaim cache space
    // from later critical entries, but never the reverse.
    bool retainWanted(const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
                      const std::vector<FarTerrainKey>& nearestFirst = {},
                      std::span<const FarTerrainKey> criticalKeys = {});
    // Refreshes only the bounded camera-critical classification and order.
    // The broad wanted set and its cache-maintenance revision remain intact,
    // so a stable 16-block view refresh does not copy or retire the horizon.
    bool refreshCriticalPriorities(std::span<const FarTerrainKey> criticalKeys);
    // Quiesces work tied to a rejected preparation anchor without shutting down
    // immutable caches or worker threads. In-flight inference may finish in its
    // shared authority cache, but this scheduler no longer polls or publishes it.
    uint64_t advanceProtectedHandoffEpoch();
    uint64_t cancelViewPreparation();
    uint64_t advanceEpoch();
    uint64_t currentEpoch() const { return epoch_.load(std::memory_order_acquire); }

    void drainCompleted(std::vector<FarTerrainResult>& output);
    void drainCanopyCompleted(std::vector<FarCanopyResult>& output);
    std::shared_ptr<const FarTerrainMesh> findCached(FarTerrainKey key) const;
    void findCachedBatch(
        std::span<const FarTerrainKey> keys, size_t maximumResults,
        std::vector<std::shared_ptr<const FarTerrainMesh>>& output,
        FarTerrainAuthorityQuality minimumQuality = FarTerrainAuthorityQuality::PREVIEW) const;
    std::shared_ptr<const FarCanopyAttachment> findCachedCanopy(FarTerrainKey key) const;
    void
    findCachedCanopyBatch(std::span<const FarTerrainKey> keys, size_t maximumResults,
                          std::vector<std::shared_ptr<const FarCanopyAttachment>>& output) const;
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
        // The camera tile and its protected or exact-fallback neighborhood
        // preempt optional distant refinement, regardless of authority lane.
        bool cameraNearCritical = false;
        bool visibleFinalParent = false;
        FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL;
        worldgen::learned::AuthorityRequestPriority authorityPriority =
            worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT;
        uint64_t protectedHandoffEpoch = 0;
        std::vector<worldgen::learned::TerrainPageCoordinate> authorityDependencies;
        std::vector<FarTerrainNativeHydrologyDependency> nativeHydrologyDependencies;
        std::optional<worldgen::learned::NativeRect> transientGeometryDependency;
        uint64_t authorityCompletionAtDispatch = 0;
        uint32_t quiescentAuthorityRetries = 0;
    };

    struct BaseAuthorityWaitSet {
        std::vector<worldgen::learned::TerrainPageCoordinate> pages;
        std::vector<FarTerrainNativeHydrologyDependency> nativeHydrology;
        std::optional<uint64_t> minimumAuthorityCompletion;

        [[nodiscard]] bool empty() const noexcept {
            return pages.empty() && nativeHydrology.empty() && !minimumAuthorityCompletion;
        }
    };

    struct ParkedBaseJob {
        Job job;
        BaseAuthorityWaitSet waitingOn;
    };

    enum class BaseAuthorityPreparation : uint8_t {
        Ready,
        Deferred,
        Failed,
    };

    struct CanopyJob {
        FarTerrainKey key;
        uint64_t epoch = 0;
        uint32_t viewPriority = std::numeric_limits<uint32_t>::max();
        FarTerrainAuthorityQuality groundingQuality = FarTerrainAuthorityQuality::FINAL;
        // PREVIEW publication is a distinct scheduler phase and always sorts
        // ahead of FINAL ecology promotion. This prevents a cold final model
        // request from occupying the only nearby vegetation worker before any
        // drawable attachment exists.
        FarTerrainAuthorityQuality ecologyQuality = FarTerrainAuthorityQuality::FINAL;
        uint64_t minimumAuthorityCompletion = 0;
        // A successful provisional publication lets a later surface-quality
        // request retarget the parked FINAL retry without withdrawing the
        // attachment already visible on the old surface.
        bool provisionalPublished = false;
    };

    struct CacheEntry {
        std::shared_ptr<const FarTerrainMesh> mesh;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
        uint64_t maintenanceToken = 0;
    };

    struct CanopyCacheEntry {
        std::shared_ptr<const FarCanopyAttachment> attachment;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
    };

    struct ResidencyMembership {
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash> keys;
        std::vector<FarTerrainKey> nearestFirst;
        std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> priorities;
        // One rank per horizontal tile, independent of geometric tier. Cache
        // admission and completed-result retention use this before the broad
        // coarse-to-fine wavefront rank so distant step-16 work can never
        // displace a nearer step-2 or step-1 result.
        std::unordered_map<ColumnPos, uint32_t> coordinatePriorities;
        uint64_t revision = 0;
    };

    struct CacheMaintenanceItem {
        FarTerrainKey key;
        uint64_t token = 0;
    };

    std::shared_ptr<ChunkGenerator> generator_;
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext_;
    FarTerrainSource source_;
    // Protected refinements use an otherwise identical generator whose
    // context carries the protected authority priority through mesh-time
    // queries. Base preflight already passes this priority explicitly.
    FarTerrainSource protectedSource_;
    std::shared_ptr<ChunkGenerator> canopyGenerator_;
    FarTerrainSource canopySource_;
    std::shared_ptr<ChunkGenerator> previewGenerator_;
    std::shared_ptr<worldgen::learned::WorldGenerationContext> previewGenerationContext_;
    FarTerrainSource previewSource_;
    // Render-thread owned. The desired vector is reset when the visible
    // membership changes. Submitted pages stay in the outstanding vector
    // until this thread observes their terminal result, including after a
    // camera move. That observation releases completed single-flight records
    // and latches a terminal failure instead of leaving a full authority
    // queue permanently opaque to the startup UI.
    std::vector<worldgen::learned::TerrainPageCoordinate> coarseAuthorityPrefetchPages_;
    size_t coarseAuthorityPrefetchCursor_ = 0;
    std::vector<worldgen::learned::TerrainPageCoordinate> coarseAuthorityPrefetchOutstandingPages_;
    std::vector<worldgen::learned::TerrainPageCoordinate> speculativeAuthorityPrefetchPages_;
    size_t speculativeAuthorityPrefetchCursor_ = 0;
    std::vector<worldgen::learned::TerrainPageCoordinate>
        speculativeAuthorityPrefetchOutstandingPages_;
    FarTerrainSchedulerLimits limits_;
    std::vector<std::thread> workers_;
    std::vector<std::thread> canopyWorkers_;

    mutable std::mutex jobMutex_;
    std::condition_variable jobCv_;
    std::deque<Job> jobs_;
    std::unordered_map<FarTerrainKey, uint64_t, FarTerrainKeyHash> activeKeys_;
    std::unordered_map<FarTerrainKey, Job, FarTerrainKeyHash> activeWorkerJobs_;
    std::unordered_map<FarTerrainKey, Job, FarTerrainKeyHash> terrainFollowupJobs_;
    std::unordered_map<FarTerrainKey, ParkedBaseJob, FarTerrainKeyHash> parkedBaseJobs_;
    std::map<worldgen::learned::TerrainPageCoordinate, std::vector<FarTerrainKey>>
        parkedBaseWaiters_;
    size_t activeWorkerCount_ = 0;
    size_t workerBudget_ = WORKER_COUNT;
    bool nearFirstWorkEnabled_ = false;
    std::shared_ptr<const ResidencyMembership> wantedMembership_;
    std::deque<std::shared_ptr<const ResidencyMembership>> retiredMemberships_;
    // Guarded by jobMutex_. Critical admission has a small independent order.
    // Keeping it outside ResidencyMembership lets camera motion update nearby
    // priorities without copying thousands of broad keys and priority pairs.
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> criticalWantedKeys_;
    std::vector<FarTerrainKey> criticalNearestFirst_;
    std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> criticalPriorities_;
    uint64_t nextWantedRevision_ = 0;
    bool residencyMaintenanceRequested_ = false;
    bool residencyMaintenanceActive_ = false;

    mutable std::mutex canopyJobMutex_;
    std::condition_variable canopyJobCv_;
    std::deque<CanopyJob> canopyJobs_;
    std::unordered_map<FarTerrainKey, CanopyJob, FarTerrainKeyHash> parkedCanopyJobs_;
    std::unordered_map<FarTerrainKey, CanopyJob, FarTerrainKeyHash> canopyFollowupJobs_;
    std::unordered_map<FarTerrainKey, uint64_t, FarTerrainKeyHash> activeCanopyKeys_;
    // Coordinate ranks are refreshed with the current view even when the set
    // of wanted keys is unchanged. Every queued, parked, and newly submitted
    // attachment consumes this one ordering so an old camera position cannot
    // retain a permanently favorable priority.
    std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> canopyViewPriorities_;
    size_t canopyWorkerBudget_ = CANOPY_WORKER_COUNT;
    // These counts are protected by canopyJobMutex_. FINAL work may begin
    // only after the visible PREVIEW phase drains, and at most one FINAL
    // build may run at once so camera movement always has a provisional lane.
    size_t activePreviewCanopyWorkerCount_ = 0;
    size_t activeFinalCanopyWorkerCount_ = 0;
    // Guarded by canopyJobMutex_. A cold visible step-1 attachment may yield
    // only this bounded number of PREVIEW dispatches before it receives a
    // lane, even if camera motion keeps replenishing coarse coverage work.
    size_t coarseCanopyDispatchStreak_ = 0;

    mutable std::mutex completedMutex_;
    std::deque<FarTerrainResult> completed_;

    mutable std::mutex canopyCompletedMutex_;
    std::deque<FarCanopyResult> canopyCompleted_;

    mutable std::mutex cacheMutex_;
    mutable std::unordered_map<FarTerrainKey, CacheEntry, FarTerrainKeyHash> cache_;
    mutable std::shared_ptr<const ResidencyMembership> cacheMembership_;
    mutable std::deque<CacheMaintenanceItem> cacheMaintenanceQueue_;
    mutable size_t cacheMaintenanceRemaining_ = 0;
    mutable size_t cacheBytes_ = 0;
    mutable uint64_t accessClock_ = 0;
    mutable uint64_t cacheMaintenanceTokenClock_ = 0;

    std::atomic<bool> running_{true};
    std::atomic<bool> finalStreamingWorkEnabled_{true};
    std::atomic<uint64_t> epoch_{1};
    std::atomic<uint64_t> protectedHandoffEpoch_{1};
    std::atomic<size_t> inFlight_{0};
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> built_{0};
    std::atomic<uint64_t> canceled_{0};
    std::atomic<uint64_t> criticalDisplacements_{0};
    std::atomic<uint64_t> failed_{0};
    std::atomic<uint64_t> deferred_{0};
    std::atomic<uint64_t> step32WaterGridCalls_{0};
    std::atomic<uint64_t> step32WaterGridSamples_{0};
    std::atomic<uint64_t> step32WaterPointSamples_{0};
    std::atomic<uint64_t> step32WaterDenseGridCalls_{0};
    std::atomic<uint64_t> authorityCompletionResumes_{0};
    std::atomic<uint64_t> quiescentAuthorityResumes_{0};
    mutable std::atomic<uint64_t> cacheHits_{0};
    std::atomic<uint64_t> wantedUpdates_{0};
    std::atomic<uint64_t> wantedNoops_{0};
    std::atomic<uint64_t> criticalPriorityUpdates_{0};
    std::atomic<uint64_t> criticalPriorityNoops_{0};
    std::atomic<size_t> maintenancePendingSnapshot_{0};
    std::atomic<uint64_t> maintenancePasses_{0};
    std::atomic<uint64_t> maintenanceScanned_{0};
    std::atomic<uint64_t> maintenanceEvicted_{0};
    std::atomic<uint64_t> maintenanceBytes_{0};
    std::atomic<size_t> maximumMaintenanceScanned_{0};
    std::atomic<size_t> maximumMaintenanceBytes_{0};
    std::atomic<size_t> queuedBaseCount_{0};
    std::atomic<size_t> parkedBaseCount_{0};
    std::atomic<size_t> terrainFollowupCount_{0};
    std::atomic<size_t> queuedRefinementCount_{0};
    std::atomic<size_t> queuedUrgentRefinementCount_{0};
    std::atomic<size_t> urgentRefinementInFlightCount_{0};
    std::atomic<size_t> visibleFinalParentInFlightCount_{0};
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

    mutable std::mutex canopyCacheMutex_;
    mutable std::unordered_map<FarTerrainKey, CanopyCacheEntry, FarTerrainKeyHash> canopyCache_;
    mutable std::shared_ptr<const ResidencyMembership> canopyCacheMembership_;
    mutable size_t canopyCacheBytes_ = 0;
    mutable uint64_t canopyAccessClock_ = 0;

    std::atomic<size_t> canopyInFlight_{0};
    std::atomic<size_t> activeCanopyWorkerCount_{0};
    std::atomic<size_t> queuedCanopyCount_{0};
    std::atomic<size_t> parkedCanopyCount_{0};
    std::atomic<size_t> completedCanopyCount_{0};
    std::atomic<size_t> canopyCacheEntryCount_{0};
    std::atomic<size_t> canopyCacheBytesSnapshot_{0};
    std::atomic<uint64_t> canopySubmitted_{0};
    std::atomic<uint64_t> canopyBuilt_{0};
    std::atomic<uint64_t> canopyCanceled_{0};
    std::atomic<uint64_t> canopyFailed_{0};
    std::atomic<uint64_t> canopyDeferred_{0};
    std::atomic<uint64_t> canopyAuthorityCompletionResumes_{0};
    mutable std::atomic<uint64_t> canopyCacheHits_{0};

    bool enqueueInternal(FarTerrainKey key, uint32_t viewPriority, bool urgentRefinement,
                         bool cameraNearCritical, FarTerrainAuthorityQuality authorityQuality,
                         worldgen::learned::AuthorityRequestPriority authorityPriority);
    enum class ExistingJobResolution : uint8_t {
        NotFound,
        Unchanged,
        Upgraded,
    };
    static bool jobBefore(const Job& first, const Job& second) noexcept;
    static Job mergeJobRequest(const Job& current, const Job& requested);
    static bool sameJobRequest(const Job& first, const Job& second) noexcept;
    static bool executionUpgradeRequested(const Job& current, const Job& requested) noexcept;
    // Called with jobMutex_ held. Camera-near urgent work may replace a
    // strictly less-important queued or dependency-parked optional job. A
    // critical missing PREVIEW parent may also replace a farther ordinary
    // parent request, but no scheduler action removes resident GPU coverage.
    bool makeRoomForJobLocked(const Job& incoming, bool removeVictim);
    void queueJobLocked(Job job);
    ExistingJobResolution upgradeExistingJobLocked(const Job& requested);
    BaseAuthorityPreparation prepareBaseAuthority(Job& job, BaseAuthorityWaitSet& waitingOn);
    bool parkActiveBaseJob(Job job, BaseAuthorityWaitSet waitingOn);
    void parkBaseJobLocked(Job job, BaseAuthorityWaitSet waitingOn);
    void wakeParkedBaseJobsForReadyPage(worldgen::learned::TerrainPageCoordinate coordinate);
    void wakeParkedBaseJobIfReady(FarTerrainKey key);
    size_t refreshParkedBaseJob(FarTerrainKey key, size_t maximumWork);
    void cancelParkedBaseJob(FarTerrainKey key);
    void removeParkedBaseWaitersLocked(const ParkedBaseJob& job);
    void releaseActiveWorkerLocked(const Job& job);
    Job takeNextJobLocked();
    void workerLoop(bool latencySensitive);
    void canopyWorkerLoop();
    void latchCriticalMeshFailure(const Job& job, std::string message) const;
    static bool canopyJobBefore(const CanopyJob& first, const CanopyJob& second) noexcept;
    bool enqueueCanopyInternal(FarTerrainKey key, uint64_t epoch, uint32_t viewPriority,
                               FarTerrainAuthorityQuality groundingQuality);
    bool parkCanopyJob(CanopyJob job);
    [[nodiscard]] const FarTerrainSource&
    sourceFor(FarTerrainAuthorityQuality authorityQuality,
              worldgen::learned::AuthorityRequestPriority authorityPriority) const;
    void finishJob(const Job& job);
    void finishCanopyJob(const CanopyJob& job, bool allowFollowup = false);
    bool storeCompleted(FarTerrainResult result);
    bool storeCanopyCompleted(FarCanopyResult result);
    void storeCache(std::shared_ptr<const FarTerrainMesh> mesh);
    bool storeCanopyCache(std::shared_ptr<const FarCanopyAttachment> attachment);
    bool performResidencyMaintenance();
    void requestResidencyMaintenance();
    void publishCacheStatsLocked();
    // Requires jobMutex_. Critical publication order is exact-key specific;
    // ordinary work is horizontal-nearest first and only then coarse-to-fine
    // within one tile so its drawable parent remains available.
    bool completedResultBeforeLocked(const FarTerrainResult& first,
                                     const FarTerrainResult& second) const noexcept;
};

#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/math.hpp"
#include "render/block_texture_array.hpp"
#include "render/far_terrain.hpp"
#include "render/frame_ring.hpp"
#include "render/gpu_timer.hpp"
#include "render/graphics_settings.hpp"
#include "render/lod_mesher.hpp"
#include "render/mega_buffer.hpp"
#include "render/mesh_scheduler.hpp"
#include "render/particles.hpp"
#include "render/shader_types.hpp"
#include "render/ui_menu.hpp"
#include "render/vertex.hpp"
#include "world/block_properties.hpp"
#include "world/mesh_snapshot.hpp"

// Forward declarations
class Entity;
struct ItemEntity;
struct Boat;
class EntityRenderer;
class ItemEntityRenderer;
class BoatRenderer;
class World;
class Camera;
class UIOverlay;
class Bloom;
class PostStack;
class ShadowMap;
class Volumetrics;
class ScreenSpaceLighting;
class AtmosphereRenderer;
class CloudRenderer;
class LightningRenderer;
class ParticleSystem;
class WeatherSnapshot;
struct IndirectHistoryState;
struct WeatherSample;
struct ExactSurfaceCoverageSnapshot;

// Modal v4 preparation has no gameplay fixed tick to publish the world's
// immutable render snapshots. Keep that synchronization in the preparation
// path so newly generated exact cubes and their mesh-candidate set become
// observable before handoff readiness is sampled.
void publishV4PreparationWorldSnapshot(World& world);

struct BlockHighlight {
    Vec3 blockPosition;
    BlockSelectionBounds localBounds;
};

// Fixed-tick progression keeps SSGI history. Rewinds and large jumps, such
// as sleeping through the night, invalidate it.
constexpr bool indirectLightingTimeDiscontinuity(bool hasPrevious, uint64_t previous,
                                                 uint64_t current) noexcept {
    if (!hasPrevious) return false;
    if (current < previous) return true;
    return current - previous > 8U;
}

constexpr bool exactLightingPublicationSatisfied(uint32_t builtVersion,
                                                 uint32_t targetVersion) noexcept {
    return builtVersion == targetVersion || static_cast<int32_t>(builtVersion - targetVersion) > 0;
}

constexpr bool exactMeshPublicationInvalidatesHistory(bool trackedLightingPublication,
                                                      bool replacedUploadedMesh,
                                                      uint32_t builtVersion,
                                                      uint32_t targetVersion) noexcept {
    return trackedLightingPublication && replacedUploadedMesh &&
           exactLightingPublicationSatisfied(builtVersion, targetVersion);
}

constexpr uint64_t indirectLightingRevision(uint64_t worldRevision,
                                            uint64_t publicationRevision) noexcept {
    return worldRevision + publicationRevision;
}

inline constexpr int EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS = 16;
inline constexpr int EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS = 32;
inline constexpr uint64_t EXACT_MESH_VERTICAL_PRIORITY_SPAN = uint64_t{1} << 16U;
inline constexpr uint64_t EXACT_SURFACE_MESH_SUBLANE_OFFSET = uint64_t{1} << 31U;
inline constexpr uint64_t EXACT_FLORA_MESH_SUBLANE_OFFSET = uint64_t{1} << 32U;

struct ExactMeshCandidatePriority {
    MeshPriorityLane lane = MeshPriorityLane::BROAD_SURFACE;
    uint64_t distanceSquared = 0;

    bool operator==(const ExactMeshCandidatePriority&) const = default;
};

struct ExactMeshCandidate {
    ExactMeshCandidatePriority priority;
    const Chunk* chunk = nullptr;
};

constexpr bool exactMeshCandidateRanksBefore(ExactMeshCandidatePriority left,
                                             ExactMeshCandidatePriority right) noexcept {
    if (left.lane != right.lane)
        return static_cast<uint8_t>(left.lane) > static_cast<uint8_t>(right.lane);
    return left.distanceSquared < right.distanceSquared;
}

// Camera-column and exploration terrain keep their scheduler lanes. Every
// required surface section in the near and medium exact band shares that
// reserved capacity before optional tree-support sections. This prevents a
// distant completion wave from consuming the mesh and upload queues while
// block-scale terrain beside the player is still represented by a far parent.
constexpr ExactMeshCandidatePriority exactMeshCandidatePriority(int64_t dx, int64_t dy, int64_t dz,
                                                                bool explorationBand,
                                                                bool surfaceRequired,
                                                                bool floraRequired) noexcept {
    const uint64_t horizontalSquared = static_cast<uint64_t>(dx * dx + dz * dz);
    const uint64_t verticalDistance = std::min(static_cast<uint64_t>(dy * dy) * uint64_t{2},
                                               EXACT_MESH_VERTICAL_PRIORITY_SPAN - 1U);
    const uint64_t rawDistance =
        horizontalSquared * EXACT_MESH_VERTICAL_PRIORITY_SPAN + verticalDistance;
    const uint64_t boundedSublaneDistance =
        std::min(rawDistance, EXACT_FLORA_MESH_SUBLANE_OFFSET - 1U);
    if (dx == 0 && dz == 0) return {MeshPriorityLane::CAMERA_COLUMN, boundedSublaneDistance};
    if (explorationBand) return {MeshPriorityLane::CAMERA_BAND, boundedSublaneDistance};
    constexpr uint64_t SURFACE_RADIUS_SQUARED =
        static_cast<uint64_t>(EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS) *
        EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS;
    constexpr uint64_t FLORA_RADIUS_SQUARED =
        static_cast<uint64_t>(EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS) *
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS;
    if (surfaceRequired && horizontalSquared <= SURFACE_RADIUS_SQUARED) {
        return {MeshPriorityLane::CAMERA_BAND,
                EXACT_SURFACE_MESH_SUBLANE_OFFSET + boundedSublaneDistance};
    }
    if (floraRequired && horizontalSquared <= FLORA_RADIUS_SQUARED) {
        return {MeshPriorityLane::CAMERA_BAND,
                EXACT_FLORA_MESH_SUBLANE_OFFSET + boundedSublaneDistance};
    }
    return {MeshPriorityLane::BROAD_SURFACE, rawDistance};
}

// Worker completion order is intentionally unrelated to publication order.
// Reconstruct the terrain-only scheduling priority at the current camera so
// completed meshes inside the camera column and exploration band consume the
// bounded upload budget before distant results. Coordinate ordering makes the
// decision deterministic while a stable sort preserves revision order for the
// same section.
struct ExactMeshUploadPriority {
    ExactMeshCandidatePriority candidate;
    ChunkPos position;

    bool operator==(const ExactMeshUploadPriority&) const = default;
};

constexpr ExactMeshUploadPriority exactMeshUploadPriority(ChunkPos position, ChunkPos camera,
                                                          int explorationRadiusChunks,
                                                          bool surfaceRequired = false,
                                                          bool floraRequired = false) noexcept {
    const int64_t dx = position.x - camera.x;
    const int64_t dy = static_cast<int64_t>(position.y) - camera.y;
    const int64_t dz = position.z - camera.z;
    const int64_t explorationRadius = std::max(explorationRadiusChunks, 0);
    const bool explorationBand = dx * dx + dz * dz <= explorationRadius * explorationRadius;
    return {exactMeshCandidatePriority(dx, dy, dz, explorationBand, surfaceRequired, floraRequired),
            position};
}

constexpr bool exactMeshUploadRanksBefore(ExactMeshUploadPriority left,
                                          ExactMeshUploadPriority right) noexcept {
    if (left.candidate != right.candidate)
        return exactMeshCandidateRanksBefore(left.candidate, right.candidate);
    if (left.position.x != right.position.x) return left.position.x < right.position.x;
    if (left.position.z != right.position.z) return left.position.z < right.position.z;
    return left.position.y < right.position.y;
}

// Capacity reclamation is the inverse of publication priority. The least
// important and horizontally farthest pending entry yields first. Exact
// owners and successful uploads from the current drain are never candidates:
// the latter have committed GPU storage but intentionally wait for the
// post-drain atomic column handoff before becoming exact owners.
constexpr bool exactMeshRegistryVictimEligible(bool exactOwned, bool committedThisDrain) noexcept {
    return !exactOwned && !committedThisDrain;
}

constexpr bool exactMeshEvictionRanksBefore(ExactMeshUploadPriority left,
                                            ExactMeshUploadPriority right) noexcept {
    return exactMeshUploadRanksBefore(right, left);
}

// At the hard registry cap, only a strictly more important current-camera
// request may replace an unowned placeholder. This keeps a delayed distant
// completion from displacing the maximum-detail surface it was supposed to
// yield to.
constexpr bool exactMeshRegistryMayReplace(ExactMeshUploadPriority incoming,
                                           ExactMeshUploadPriority victim) noexcept {
    return exactMeshUploadRanksBefore(incoming, victim);
}

// GPU-side per-chunk mesh allocation tracking. opaqueIndexCount splits the
// one allocation into the opaque section and the water section (see
// MeshOutput). builtVersion/requestedVersion carry the staleness protocol:
// a chunk re-meshes whenever its content version differs from the mesh it
// is drawing, and a version already requested isn't requested twice.
struct ChunkMeshState {
    MegaBuffer::ChunkAllocation alloc;
    uint32_t opaqueIndexCount = 0;
    uint32_t builtVersion = 0;     // 0 = nothing built yet
    uint32_t requestedVersion = 0; // 0 = no build in flight
    bool uploaded = false;
};

// Worker results may complete out of order across renderer drains. Publish
// only a result for the currently loaded revision, and never let it replace a
// resident result from a later serial revision. Revision zero remains the
// "nothing built" sentinel used by ChunkMeshState.
constexpr bool chunkMeshAsyncResultCanReplace(uint32_t builtVersion, uint32_t liveVersion,
                                              uint32_t residentVersion) {
    return builtVersion == liveVersion &&
           (residentVersion == 0 || static_cast<int32_t>(builtVersion - residentVersion) > 0);
}

// A completion owns only the request revision captured when it was queued.
// An older completion must not clear a newer request for the same cube.
constexpr uint32_t chunkMeshRequestAfterCompletion(uint32_t pendingVersion,
                                                   uint32_t completedRequestVersion) {
    return pendingVersion == completedRequestVersion ? 0 : pendingVersion;
}

// A hard-cap admission guard shared by exact mesh upload and scheduling.
// Published exact owners are not eviction candidates. When every resident
// entry owns its section, new work waits instead of growing past the cap.
constexpr bool chunkMeshRegistryCanAdmit(size_t residentCount, size_t capacity,
                                         bool alreadyResident, bool hasUnownedVictim) {
    if (residentCount > capacity) return false;
    return alreadyResident || residentCount < capacity || hasUnownedVictim;
}

// GPU allocation for one immutable 256x256-block far-terrain tile. Far tiles
// use the same compact vertex format and shader as exact cubes, but live in a
// separate bounded arena so horizon streaming can never evict collision-zone
// meshes.
struct FarTerrainMeshState {
    MegaBuffer::ChunkAllocation alloc;
    FarTerrainBounds bounds;
    FarTerrainBounds surfaceBounds;
    std::array<FarTerrainBounds, FAR_TERRAIN_OCCLUDER_PATCH_COUNT> occluderPatches{};
    uint32_t opaqueIndexCount = 0;
    float complexity = 0.0F;
    uint64_t deterministicHash = 0;
    FarTerrainWaterTopologySignature waterTopology;
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL;
    bool uploaded = false;
    FarTerrainSurfaceBoundarySignature surfaceBoundary;
    bool exactAuthorityCompatible = true;
};

struct FarCanopyMeshState {
    std::optional<MegaBuffer::ChunkAllocation> alloc;
    FarTerrainBounds bounds;
    uint64_t deterministicHash = 0;
    uint64_t anchorIdentityHash = 0;
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::FINAL;
    FarTerrainAuthorityQuality groundingQuality = FarTerrainAuthorityQuality::FINAL;
};

constexpr bool farCanopyCastsShadow(bool attachmentPresent, bool allocationPresent,
                                    uint32_t indexCount) noexcept {
    return attachmentPresent && allocationPresent && indexCount != 0U;
}

constexpr uint64_t farCanopyShadowRevision(uint64_t baseRevision, bool attachmentPresent,
                                           FarTerrainAuthorityQuality quality,
                                           uint64_t deterministicHash) noexcept {
    return baseRevision ^ (attachmentPresent ? 0xC4A09E3779B97F4AULL : 0ULL) ^
           (static_cast<uint64_t>(quality) << 56U) ^ deterministicHash;
}

inline FarTerrainBounds farShadowCasterBounds(const FarTerrainBounds& surface,
                                              const std::optional<FarTerrainBounds>& canopy) {
    if (!canopy) return surface;
    FarTerrainBounds bounds = surface;
    bounds.minX = std::min(bounds.minX, canopy->minX);
    bounds.maxX = std::max(bounds.maxX, canopy->maxX);
    bounds.minY = std::min(bounds.minY, canopy->minY);
    bounds.maxY = std::max(bounds.maxY, canopy->maxY);
    bounds.minZ = std::min(bounds.minZ, canopy->minZ);
    bounds.maxZ = std::max(bounds.maxZ, canopy->maxZ);
    return bounds;
}

enum class FarTerrainUploadAction : uint8_t {
    REJECT,
    INSERT_AFTER_UPLOAD,
    REPLACE_AFTER_UPLOAD,
};

// A same-key FINAL mesh can replace PREVIEW only after its complete GPU
// allocation and upload succeed. Until that commit point the existing mesh
// remains both resident and drawable.
constexpr FarTerrainUploadAction
farTerrainUploadAction(std::optional<FarTerrainAuthorityQuality> resident,
                       FarTerrainAuthorityQuality incoming) noexcept {
    if (!resident) return FarTerrainUploadAction::INSERT_AFTER_UPLOAD;
    if (*resident == FarTerrainAuthorityQuality::PREVIEW &&
        incoming == FarTerrainAuthorityQuality::FINAL) {
        return FarTerrainUploadAction::REPLACE_AFTER_UPLOAD;
    }
    return FarTerrainUploadAction::REJECT;
}

constexpr bool farTerrainUploadCommitAllowed(FarTerrainUploadAction action,
                                             bool uploadComplete) noexcept {
    return uploadComplete && action != FarTerrainUploadAction::REJECT;
}

struct FarTerrainLodTransition {
    FarTerrainKey from;
    FarTerrainKey to;
    double startedAtSeconds = 0.0;
};

// Terrain LOD may finish before its independently scheduled flora attachment.
// Keep the last drawable attachment as a bounded per-tile fallback instead of
// forcing terrain to wait or exposing a barren interval. A compatible target
// retires the fallback atomically, so fallback ownership never moves backward.
enum class FarCanopyLodCompletionAction : uint8_t {
    NONE,
    ADOPT_SOURCE,
    RETAIN_FALLBACK,
    RETIRE_FALLBACK,
};

constexpr FarCanopyLodCompletionAction
farCanopyLodCompletionAction(bool fallbackPresent, bool sourceCanopyPresent,
                             bool targetCanopyPresent) noexcept {
    if (targetCanopyPresent)
        return fallbackPresent ? FarCanopyLodCompletionAction::RETIRE_FALLBACK
                               : FarCanopyLodCompletionAction::NONE;
    if (fallbackPresent) return FarCanopyLodCompletionAction::RETAIN_FALLBACK;
    return sourceCanopyPresent ? FarCanopyLodCompletionAction::ADOPT_SOURCE
                               : FarCanopyLodCompletionAction::NONE;
}

constexpr bool farCanopyLodTargetUsesSourceFallback(bool sourceCanopyPresent,
                                                    bool targetCanopyPresent) noexcept {
    return sourceCanopyPresent && !targetCanopyPresent;
}

struct FarTerrainCanopyRefreshRequest {
    FarTerrainKey key;
    // The request always converges on FINAL ecology, but PREVIEW ecology may
    // publish first. Roots must match the currently displayed surface
    // authority until that surface promotes.
    FarTerrainAuthorityQuality groundingQuality = FarTerrainAuthorityQuality::FINAL;
    // Saturating horizontal distance in blocks. Unlike a rotating batch rank,
    // this absolute priority remains stable when another attachment finishes.
    uint32_t viewPriority = 0;
    bool transitionTarget = false;
    double distanceSquaredBlocks = 0.0;

    bool operator==(const FarTerrainCanopyRefreshRequest&) const = default;
};

// Optional flora follows only surfaces that can actually be drawn. Hidden
// coarse parents and speculative bridge tiers must not occupy the bounded
// canopy queue ahead of a nearby displayed surface. PREVIEW ecology is a
// valid provisional publication while FINAL ecology remains deferred, and
// grounding follows the displayed PREVIEW or FINAL surface. A valid empty
// attachment still satisfies a resident surface because residency is
// represented by the state entry, not by a GPU allocation. A provisional
// attachment remains eligible for recovery, but missing drawable attachments
// fill the bounded request batch before provisional FINAL promotions. The
// scheduler automatically follows a successful PREVIEW with FINAL work, so a
// parked promotion cannot leave the rest of the visible horizon barren.
void buildFarTerrainCanopyRefreshBatch(
    const std::unordered_map<ColumnPos, FarTerrainKey>& displayed,
    const std::unordered_map<ColumnPos, FarTerrainLodTransition>& transitions,
    const std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash>& residents,
    const std::unordered_map<FarTerrainKey, FarCanopyMeshState, FarTerrainKeyHash>& attachments,
    double cameraX, double cameraZ, size_t requestBudget,
    std::vector<FarTerrainCanopyRefreshRequest>& output,
    const FarTerrainExactHandoff* exactFloraHandoff = nullptr);

// Exact generation and meshing share the same physical cores as far terrain.
// Exact publication owns every physical core unless a nearer far replacement
// is itself visible debt. In that case eight far workers advance the closest
// desired LODs while exact generation keeps the other cores. Local far debt
// without exact debt uses twelve workers; the broad sixteen-worker horizon
// returns only after both debts clear.
inline constexpr size_t FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET = 8;
inline constexpr size_t FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET = 12;
static_assert(FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET >=
              FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE * 2);
static_assert(FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET < FarTerrainScheduler::WORKER_COUNT);

constexpr size_t farTerrainWorkerBudget(bool exactStreamingDebt, bool localTerrainDebt) noexcept {
    if (exactStreamingDebt)
        return localTerrainDebt ? FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET : size_t{0};
    if (localTerrainDebt) return FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET;
    return FarTerrainScheduler::WORKER_COUNT;
}

constexpr bool farTerrainOrdinaryCoverageWorkEnabled(bool gameplayScene, bool exactStreamingDebt,
                                                     bool localTerrainDebt) noexcept {
    return !gameplayScene || (!exactStreamingDebt && !localTerrainDebt);
}

// Flora is optional relative to terrain, water, and exact collision. Before a
// connected far prefix exists, hard local debt keeps both utility workers
// idle. Once that drawable prefix is complete, one low-priority worker remains
// available for provisional flora even while local or exact publication debt
// persists. This prevents continuously replenished terrain work from starving
// every distant canopy. The second worker remains reserved until all stronger
// work drains.
constexpr size_t farTerrainCanopyWorkerBudget(bool gameplayScene, bool connectedPrefixReady,
                                              bool localTerrainDebt,
                                              bool exactStreamingDebt) noexcept {
    if (!gameplayScene) return 0;
    if (localTerrainDebt) return connectedPrefixReady ? size_t{1} : size_t{0};
    if (exactStreamingDebt) return 1;
    return connectedPrefixReady ? FarTerrainScheduler::CANOPY_WORKER_COUNT : size_t{1};
}

// The flora ownership handoff keeps far attachments visible until exact crown
// sections arrive. Allowing one optional worker is therefore safe only after
// no unfinished exact terrain or flora column remains inside the block-scale
// flora radius. Work farther out stays higher priority in its own exact lanes
// but no longer leaves the already settled foreground and middle distance
// barren.
constexpr bool farTerrainCanopyHasNearExactSurfaceDebt(bool exactStreamingDebt,
                                                       float nearestIncompleteSurfaceBlocks,
                                                       int floraRadiusChunks) noexcept {
    const float protectedRadiusBlocks =
        static_cast<float>(std::max(floraRadiusChunks, 0) * CHUNK_EDGE);
    // The negated greater-than comparison treats NaN and negative diagnostics
    // as unresolved instead of opening optional work on an invalid handoff.
    return exactStreamingDebt && !(nearestIncompleteSurfaceBlocks > protectedRadiusBlocks);
}

constexpr bool farTerrainCanopyHasNearExactPublicationDebt(bool exactStreamingDebt,
                                                           float nearestIncompleteSurfaceBlocks,
                                                           float nearestIncompleteFloraBlocks,
                                                           int floraRadiusChunks) noexcept {
    return farTerrainCanopyHasNearExactSurfaceDebt(
               exactStreamingDebt, nearestIncompleteSurfaceBlocks, floraRadiusChunks) ||
           farTerrainCanopyHasNearExactSurfaceDebt(exactStreamingDebt, nearestIncompleteFloraBlocks,
                                                   floraRadiusChunks);
}

// A tree can span several vertical sections. Terrain-bearing sections may
// publish with their surface column, but an upper trunk or crown section waits
// for the complete flora column. Draw and collision therefore adopt the same
// solid tree geometry on one frame instead of exposing partial tree sections.
constexpr bool exactFloraSectionMayPublish(bool surfaceRequired, bool floraRequired,
                                           bool floraColumnFullyReady) noexcept {
    return !floraRequired || surfaceRequired || floraColumnFullyReady;
}

// A protected handoff defines the coarsest representation that may be used;
// it cannot replace a finer screen-error result near the camera.
constexpr std::optional<FarTerrainStep>
farTerrainProtectedDesiredStep(std::optional<FarTerrainStep> desired,
                               std::optional<FarTerrainStep> protectedStep) noexcept {
    if (!protectedStep) return desired;
    if (!desired || farTerrainStepSize(*protectedStep) < farTerrainStepSize(*desired))
        return protectedStep;
    return desired;
}

// A protected handoff is one atomic GPU publication. Hidden FINAL targets and
// their coverage parents must survive ordinary desired/displayed cleanup until
// the anchor commits or is abandoned. Active and requested anchors are both
// retained because movement keeps the active surface drawable while its
// replacement is assembled.
inline bool
farTerrainProtectedGpuResidencyRequired(FarTerrainKey key,
                                        const std::optional<ColumnPos>& activeAnchor,
                                        const std::optional<ColumnPos>& requestedAnchor) noexcept {
    const ColumnPos coordinate{key.tileX, key.tileZ};
    const auto requiredByAnchor = [&](const std::optional<ColumnPos>& anchor) {
        if (!anchor) return false;
        const FarTerrainProtectedNearRole role = farTerrainProtectedNearRole(*anchor, coordinate);
        if (role == FarTerrainProtectedNearRole::NONE) return false;
        if (farTerrainIsBaseStep(key.step)) return true;
        switch (role) {
            case FarTerrainProtectedNearRole::STEP_ONE_CORE:
                return key.step == FarTerrainStep::ONE;
            case FarTerrainProtectedNearRole::STEP_TWO_RING:
                return key.step == FarTerrainStep::TWO;
            case FarTerrainProtectedNearRole::STEP_FOUR_RING:
                return key.step == FarTerrainStep::FOUR;
            case FarTerrainProtectedNearRole::STEP_EIGHT_RING:
                return key.step == FarTerrainStep::EIGHT;
            case FarTerrainProtectedNearRole::STEP_SIXTEEN_RING:
                return key.step == FarTerrainStep::SIXTEEN;
            case FarTerrainProtectedNearRole::NONE:
                break;
        }
        return false;
    };
    return requiredByAnchor(activeAnchor) || requiredByAnchor(requestedAnchor);
}

struct FarTerrainAuthorityTransition {
    FarTerrainKey key;
    FarTerrainMeshState source;
    std::optional<FarCanopyMeshState> sourceCanopy;
    double startedAtSeconds = 0.0;
    // Requested protected handoffs upload into resident GPU allocations while
    // their PREVIEW sources remain the sole draw owners. One atomic center
    // commit assigns a shared start time and publishes every target together.
    bool published = true;
};

// One chunk's water draw, recorded during the opaque pass and encoded by
// the water pass after the scene resolves.
struct WaterDraw {
    simd_float4 origin;
    simd_float4 overlayColorAndStrength;
    simd_uint4 farMetadata;
    FarTerrainOwnershipUniforms farOwnership;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint64_t vertexOffset;
    uint64_t indexOffset;
    uint32_t indexCount;
    float distSq; // camera distance², for back-to-front ordering
};

// ---------------------------------------------------------------------------
// RenderPipeline owns the integrated Metal frame graph. Weather, atmosphere,
// cloud shadows, and terrain shadows are prepared before the native 4x MSAA
// sky and opaque pass. Screen-space lighting, clouds, lightning, water,
// froxels, and weather particles then composite into resolved HDR before
// exposure, flare, bloom, grading, and UI produce the drawable.
//   3. UI overlay pass onto the drawable
//
// Also owns frustum culling and the on-demand chunk mesh cache.
// ---------------------------------------------------------------------------
class RenderPipeline {
public:
    RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                   uint32_t height, uint64_t worldSeed = 0);

    ~RenderPipeline();

    // Render a single frame.
    // Handles empty world gracefully (sky-only output).
    void render(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const World& world, const Camera& camera,
                uint64_t worldTime = 0, double deltaSeconds = 0.0,
                std::optional<BlockHighlight> highlightedBlock = std::nullopt,
                const UIFrameState& uiFrame = UIFrameState{},
                const std::vector<std::shared_ptr<Entity>>* entities = nullptr,
                const std::vector<ItemEntity>* itemEntities = nullptr,
                const std::vector<Boat>* boats = nullptr,
                std::shared_ptr<const WeatherSnapshot> weatherSnapshot = nullptr,
                const std::vector<LightningEvent>* lightningEvents = nullptr);

    // Menu-only frame when no world session is live: one single-sample pass
    // that clears the drawable to the backdrop color and draws the UI
    // overlay. No HDR, no depth, no world reads.
    void renderMenuOnly(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable,
                        const UIFrameState& uiFrame);

    // Advances exact meshes and the protected FINAL far closure inside the
    // same frame-ring transaction that presents the modal preparation UI.
    // No gameplay geometry, shadows, postprocessing, or canopy is encoded.
    void renderV4Preparation(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable,
                             const UIFrameState& uiFrame, World& world, const Camera& camera);

    // Detach every renderer structure that references or is keyed by the
    // current World: the mesh scheduler (it captures a const World& lazily
    // at first render), pending mesh results, resident cube meshes, exact
    // ownership, and the recorded far-terrain identity so the next session
    // rebuilds even under an equal seed. Must run before the World dies.
    void endWorldSession();

    // Cancels preparation work for a rejected spawn anchor without tearing
    // down exact mesh state or the current World. Already running immutable
    // work may finish, but its old epoch cannot publish or enqueue follow-ups.
    void cancelV4Preparation();

    // Reallocate MSAA and resolve textures for new viewport size.
    void resize(uint32_t width, uint32_t height);

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When zero, the bloom pass is skipped entirely (saves 13 render passes).
    void setBloomIntensity(float intensity);
    float getBloomIntensity() const { return _bloomIntensity; }

    // Push the current video settings; disabled effects skip their passes
    // wholesale on the next frame. Called at init and on every change.
    void setGraphicsSettings(const GraphicsSettings& gfx);

    // Update particle system physics (call each game tick).
    void tickParticles(float dt, const World& world, const Vec3& playerPosition,
                       const WeatherSample& weather);

    // Exponential fog density per block (settings menu).
    void setFogDensity(float density) { _fogDensity = density; }

    // Rain wetness 0-1 from the engine's weather state (darkens albedo and
    // adds a sun sheen in the chunk shader).
    void setWetness(float wetness) { _wetness = wetness; }

    // Write the next presented frame to `path` as a PNG (async, off the
    // render thread). Used by the playtest workflow for headless visual
    // verification, macOS screen-recording permissions don't apply.
    void requestFrameCapture(const std::string& path);

    // Chunk streaming counters for the F3 HUD (render thread only).
    struct ChunkRenderStats {
        float meshMsAvg = 0.f;
        uint32_t meshBuildsLastFrame = 0;
        float megaUsedMB = 0.f;
        float megaCapMB = 0.f;
        uint32_t meshCubeCount = 0;
        uint32_t meshPendingCount = 0;
        uint32_t meshQueueHighWater = 0;
        uint64_t meshCoalescedCount = 0;
        uint64_t meshDroppedStaleCount = 0;
        uint32_t exactSurfaceRequiredCount = 0;
        uint32_t exactSurfaceReadyCount = 0;
        uint32_t exactSurfaceUnresolvedColumnCount = 0;
        float exactSurfaceHandoffBlocks = 0.f;
        uint32_t farWantedTileCount = 0;
        uint32_t farResidentTileCount = 0;
        uint32_t farBaseWantedTileCount = 0;
        uint32_t farBaseResidentTileCount = 0;
        uint32_t farBaseDrawnTileCount = 0;
        uint32_t farBaseMissingTileCount = 0;
        // Cold entry waits for one bounded 60-tile FINAL closure from step 1
        // through step 16, its required FINAL parents, and matching shared
        // boundaries. These counters describe publishable terrain and water;
        // canopy attachments remain optional and nonblocking.
        uint32_t farProtectedNearWantedTileCount = 0;
        uint32_t farProtectedNearResidentTileCount = 0;
        uint32_t farProtectedNearMissingTileCount = 0;
        uint32_t farProtectedNearBoundaryMismatchCount = 0;
        std::array<uint32_t, 5> farProtectedNearTargetCountsByStep{};
        uint32_t farProtectedNearFinalParentCount = 0;
        uint32_t farProtectedNearFinalTargetCount = 0;
        uint32_t farProtectedNearExactCompatibleTargetCount = 0;
        uint32_t farProtectedNearLodMismatchCount = 0;
        uint32_t farProtectedNearAuthorityMismatchCount = 0;
        bool farProtectedNearReady = false;
        int64_t farProtectedNearAnchorTileX = 0;
        int64_t farProtectedNearAnchorTileZ = 0;
        uint64_t farProtectedNearViewEpoch = 0;
        uint64_t farProtectedNearWorldEpoch = 0;
        // The scheduler epoch can advance before a complete closure has been
        // validated. The closure epoch is stamped atomically with the anchor
        // and counts above, so startup cannot retag an older ready closure as
        // current merely because the camera returned to its prior anchor.
        uint64_t farProtectedNearCurrentEpoch = 0;
        uint64_t farProtectedNearClosureEpoch = 0;
        uint64_t exactSurfaceEpoch = 0;
        uint32_t farCriticalWantedTileCount = 0;
        uint32_t farCriticalResidentTileCount = 0;
        uint32_t farCriticalMissingTileCount = 0;
        // Fixed indices are step 1, 2, 4, 8, 16, and 32. Desired counts describe
        // the screen-error selection, resident counts describe uploaded active
        // meshes, displayed counts describe logical tile ownership before
        // culling, and drawn counts describe this frame's visible tile owners.
        std::array<uint32_t, 6> farTierDesiredTileCounts{};
        std::array<uint32_t, 6> farTierResidentMeshCounts{};
        std::array<uint32_t, 6> farTierDisplayedTileCounts{};
        std::array<uint32_t, 6> farTierDrawnTileCounts{};
        std::array<uint32_t, 6> farTierResidentPreviewCounts{};
        std::array<uint32_t, 6> farTierResidentFinalCounts{};
        std::array<uint32_t, 6> farTierDisplayedPreviewCounts{};
        std::array<uint32_t, 6> farTierDisplayedFinalCounts{};
        float farWorstVisibleProjectedErrorPixels = 0.0F;
        int64_t farWorstVisibleTileX = 0;
        int64_t farWorstVisibleTileZ = 0;
        uint8_t farWorstVisibleDesiredStep = 0;
        uint8_t farWorstVisibleDisplayedStep = 0;
        uint8_t farWorstVisibleDisplayedQuality = 0;
        uint32_t farWorstVisiblePreviewResidentMask = 0;
        uint32_t farWorstVisibleFinalResidentMask = 0;
        uint32_t farVisibleProjectedErrorViolationCount = 0;
        uint32_t farVisiblePerceptualFinalRequestCount = 0;
        uint32_t farPendingAuthorityTransitionCount = 0;
        uint32_t farExactHandoffMissingFinalParentCount = 0;
        // These identify the selection that produced the coverage counts.
        // Startup must never accept a complete horizon from a previous spawn.
        int farBaseViewDistanceChunks = 0;
        int64_t farBaseCenterTileX = 0;
        int64_t farBaseCenterTileZ = 0;
        // Changes whenever the far scheduler is rebuilt for a new World.
        // Together with the view epoch, this makes stale coverage counters
        // impossible to use as a startup readiness result.
        uint64_t farBaseWorldEpoch = 0;
        uint64_t farBaseViewEpoch = 0;
        uint32_t farRefinementWantedTileCount = 0;
        uint32_t farRefinementResidentTileCount = 0;
        uint32_t farRefinementDrawnTileCount = 0;
        uint32_t farDrawnTileCount = 0;
        uint32_t farFrustumCulledTileCount = 0;
        uint32_t farOcclusionCulledTileCount = 0;
        uint32_t farPendingTileCount = 0;
        uint32_t farUploadsLastFrame = 0;
        uint32_t farQueuedBaseTileCount = 0;
        uint32_t farQueuedRefinementTileCount = 0;
        uint32_t farActiveBaseWorkerCount = 0;
        uint32_t farReservedBaseWorkerCount = 0;
        uint32_t farActiveUrgentRefinementCount = 0;
        uint64_t farCriticalSchedulerDisplacementCount = 0;
        uint32_t farWorkerBudget = 0;
        uint64_t farStep32WaterGridCalls = 0;
        uint64_t farStep32WaterGridSamples = 0;
        uint64_t farStep32WaterPointSamples = 0;
        uint64_t farStep32WaterDenseGridCalls = 0;
        uint32_t farCachedBaseTileCount = 0;
        uint32_t farCanopyInFlightCount = 0;
        uint32_t farActiveCanopyWorkerCount = 0;
        uint32_t farQueuedCanopyCount = 0;
        uint32_t farParkedCanopyCount = 0;
        uint32_t farCompletedCanopyCount = 0;
        uint32_t farCanopyCacheEntryCount = 0;
        uint64_t farCanopyFailedCount = 0;
        uint64_t farCanopyDeferredCount = 0;
        uint64_t farCanopyAuthorityCompletionResumeCount = 0;
        float farCoverageFrontierBlocks = 0.f;
        float farCacheMB = 0.f;
        float farCanopyCacheMB = 0.f;
        float farMegaUsedMB = 0.f;
        float farPlannerMsLast = 0.f;
        float farPlannerMsP95 = 0.f;
        float farPlannerMsMax = 0.f;
        float farPlannerSelectionMsP95 = 0.f;
        float farPlannerPublicationMsP95 = 0.f;
        float farPlannerResidencyMsP95 = 0.f;
        uint64_t farArenaAdmissionDeniedCount = 0;
        uint64_t farNearArenaReclaimCount = 0;
        uint64_t farNearArenaReclaimedBytes = 0;
    };
    ChunkRenderStats chunkRenderStats() const { return _chunkStats; }
    FarTerrainGenerationCacheStats farGenerationCacheStats() const;

    struct AtmosphericRenderStats {
        uint32_t shadowRefreshMask = 0;
        std::array<uint32_t, SHADOW_CASCADE_COUNT> shadowCasterCounts{};
        std::array<uint64_t, SHADOW_CASCADE_COUNT> shadowRefreshCounts{};
        uint32_t indirectHistoryResetMask = 0;
        bool indirectHistoryValid = false;
        bool cloudHistoryValid = false;
        bool froxelHistoryValid = false;
        uint64_t atmosphereSlowRefreshCount = 0;
        uint64_t atmosphereSkyRefreshCount = 0;
        uint64_t indirectPersistentBytes = 0;
        uint64_t cloudPersistentBytes = 0;
        uint64_t froxelPersistentBytes = 0;
        uint64_t integratedPersistentBytes = 0;
        uint64_t lightningEventId = 0;
        float lunarPhaseEnergy = 0.0F;
        float lunarPhaseCycle = 0.0F;
    };
    AtmosphericRenderStats atmosphericRenderStats() const;

    // Real GPU frame time (EMA over completed command buffers) for the F3
    // HUD and the 60-frame diagnostic log.
    float gpuFrameMs() const { return _gpuTimer->frameMsEma(); }

    // Per-pass GPU breakdown; empty unless RYCRAFT_GPU_COUNTERS is set.
    std::string gpuPassBreakdown() const { return _gpuTimer->passBreakdown(); }

    // Stop the mesh workers (they reference the World). The engine calls
    // this on the quit path BEFORE the world is destroyed; the scheduler's
    // destructor also calls it defensively.
    void shutdownMeshWorkers();

private:
    enum class WorldgenOverlayMode : uint8_t {
        NONE,
        GEOLOGY,
        HYDROLOGY,
        CLIMATE,
        BIOME,
        LOD,
        AUTHORITY,
    };

    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState{};
    id<MTLRenderPipelineState> _coherentResolvePipelineState{};
    id<MTLDepthStencilState> _depthState{};

    // Sky pipeline state (drawn first in the scene pass, behind everything)
    id<MTLRenderPipelineState> _skyPipelineState{};
    id<MTLDepthStencilState> _skyDepthState{};

    // Depth-tested but non-writing state (block highlight)
    id<MTLDepthStencilState> _noDepthWriteState{};

    // Block highlight pipeline state (wireframe lines)
    id<MTLRenderPipelineState> _highlightPipelineState{};
    id<MTLBuffer> _highlightVertexBuffer{};

    // Water reads the resolved opaque depth for refraction and rejection, then
    // hardware depth-tests and writes the nearest interface into media depth.
    id<MTLRenderPipelineState> _waterPipelineState{};
    id<MTLRenderPipelineState> _underwaterOverlayState{};
    std::vector<WaterDraw> _waterDraws; // reused each frame

    // MSAA render targets (memoryless, resolved or discarded at pass end)
    id<MTLTexture> _colorMSAA{};
    id<MTLTexture> _surfaceMSAA{};
    id<MTLTexture> _reactiveMSAA{};
    id<MTLTexture> _resolveDepthKeyMSAA{};
    id<MTLTexture> _depthMSAA{};

    // Single-sample resolve target feeding bloom / the drawable blit
    id<MTLTexture> _colorResolve{};
    id<MTLTexture> _surfaceResolve{};
    id<MTLTexture> _reactiveResolve{};

    // Water pass inputs: the opaque scene's resolved depth, a water-inclusive
    // media depth, and a copy of the resolved color the refraction samples.
    id<MTLTexture> _depthResolve{};
    id<MTLTexture> _mediaDepthResolve{};
    id<MTLTexture> _sceneColorCopy{};

    // Frames-in-flight gate + per-frame constants arena: every uniform block
    // the CPU rewrites per frame sub-allocates from the current slot.
    FrameRing _frameRing;

    // The frame's chunk Uniforms allocation, filled by renderChunks, also
    // bound by the entity renderer and the water pass vertex stage.
    FrameRing::Alloc _frameUniforms;

    // Lazy segmented storage keeps published exact allocations valid while the
    // cold-start radius expands. Exact and far meshes use independent arenas.
    std::unique_ptr<SegmentedMegaBuffer> _megaBuffer;

    // Array texture of procedural block textures.
    std::unique_ptr<BlockTextureArray> _blockTextures;

    // UI overlay for HUD rendering (crosshair, hotbar, menus).
    std::unique_ptr<UIOverlay> _uiOverlay;

    // Bloom post-processing (HDR extract + blur)
    std::unique_ptr<Bloom> _bloom;

    // Final composite: exposure, tonemap, grade, sharpen (always runs)
    std::unique_ptr<PostStack> _postStack;

    // Cascaded sun/moon shadow maps (skipped when shadowQuality is 0)
    std::unique_ptr<ShadowMap> _shadowMap;

    // Unified atmospheric froxel volume.
    std::unique_ptr<Volumetrics> _volumetrics;

    // Near-field GTAO and diffuse screen-space indirect lighting.
    std::unique_ptr<ScreenSpaceLighting> _screenSpaceLighting;

    // Physical atmosphere LUTs and true volumetric cloud layers.
    std::unique_ptr<AtmosphereRenderer> _atmosphere;
    std::unique_ptr<CloudRenderer> _clouds;
    std::unique_ptr<LightningRenderer> _lightning;

    // The shadow sampling block the scene pass binds each frame, the
    // computed cascades when shadows are on, or a zeroed (strength 0) block
    // when off/faded so the chunk fragment reads full sun without branching.
    ShadowUniforms _sceneShadowUniforms{};

    // Weather particle system (rain/snow)
    std::unique_ptr<ParticleSystem> _particles;

    // Animal voxel-model renderer
    std::unique_ptr<EntityRenderer> _entityRenderer;
    std::unique_ptr<ItemEntityRenderer> _itemEntityRenderer;
    std::unique_ptr<BoatRenderer> _boatRenderer;

    // GPU frame/pass timing (per-pass sampling only under RYCRAFT_GPU_COUNTERS)
    std::unique_ptr<GpuFrameTimer> _gpuTimer;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _bloomIntensity;

    // Video settings copy, pushed by the engine (render thread only)
    GraphicsSettings _gfx;

    // Exponential fog density per block
    float _fogDensity = 0.00015f;
    float _wetness = 0.0f;
    uint32_t _shadowRefreshMask = 0;
    std::array<uint32_t, SHADOW_CASCADE_COUNT> _shadowCasterCounts{};
    uint32_t _indirectHistoryResetMask = 0;
    uint64_t _lastLightningEventId = 0;
    float _lunarPhaseEnergy = 0.0F;
    float _lunarPhaseCycle = 0.0F;
    float _directSpecularFactor = 0.0F;
    uint8_t _activeCelestialSource = 0;
    // True while the camera is submerged this frame (set by render()). Gates
    // the rain-wetness sun sheen off: a gloss toward the sun on floors seen
    // through five blocks of water read as a white-out, not rain.
    bool _cameraUnderwater = false;
    // Eased sky exposure of the camera's water column (see render()); 0 in
    // sealed aquifers and under roofed water where sunlight cannot reach.
    float _uwSkyExposure = 1.0f;
    // World Y of the surface of the water body the camera is in (see render()).
    float _uwSurfaceY = 0.0f;
    simd_float4x4 _previousViewProjection = matrix_identity_float4x4;
    std::unique_ptr<IndirectHistoryState> _indirectHistoryState;
    uint64_t _previousWorldTime = 0;
    uint64_t _indirectTimeDiscontinuityRevision = 0;
    uint64_t _exactMaterialPublicationRevision = 0;
    uint64_t _observedWorldLightingRevision = 0;
    std::unordered_map<ChunkPos, uint32_t> _pendingExactLightingPublications;
    bool _exactLightingPublicationBatchPending = false;
    bool _exactLightingPublicationCompleted = false;
    uint64_t _forcedStateRevision = 0;
    uint8_t _previousWeatherPreset = 0;
    bool _hasPreviousWorldTime = false;
    bool _weatherSnapshotWasPresent = false;

    // Frame animation clock driving water waves, caustics, and foliage sway in
    // the scene AND shadow passes, one value per frame so the two can never
    // sample different phases. It accumulates the real frame delta (NOT the
    // day-night worldTime), so animation keeps flowing when the time of day is
    // frozen (captures) or paused and never jumps at the daily rollover. Bounded
    // (wraps at 3600 s) so the float keeps sub-millisecond phase precision.
    float _animTime = 0.0f;
    double _animClock = 0.0;

    // Optional deterministic developer overlay selected by
    // RYCRAFT_WORLDGEN_OVERLAY.
    WorldgenOverlayMode _worldgenOverlayMode = WorldgenOverlayMode::NONE;

    // Drawable dimensions (the scene renders at native resolution)
    uint32_t _displayWidth;
    uint32_t _displayHeight;

    // Pending frame-capture destination (empty when no capture is queued)
    std::string _capturePath;

    // (Re)allocate the MSAA + resolve textures at the current drawable size.
    void allocateSceneTargets();
    void releaseSceneTargets();

    // Encode the drawable readback + async PNG write for requestFrameCapture.
    void encodeFrameCapture(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> frameTexture);

    // ---- Frustum culling ----
    void extractFrustumPlanes(const Mat4& vpMatrix);
    bool isChunkInFrustum(const AABB& chunkAABB) const;
    float _frustumPlanes[6][4];

    // ---- Chunk mesh cache ----
    // An entry with uploaded == false marks a cube whose mesh is empty
    // (all air) so it is not rebuilt every frame.
    std::unordered_map<ChunkPos, ChunkMeshState> _chunkMeshes;

    // Scratch map reused each frame to sweep unloaded meshes and validate
    // asynchronous results against the loaded chunk's current revision.
    std::unordered_map<ChunkPos, const Chunk*> _liveChunksByPosition;

    // HUD counters, written only by the render thread during renderChunks.
    ChunkRenderStats _chunkStats;

    // Reused meshing buffers (render thread only), snapshot + scratch keep
    // their capacity across builds instead of reallocating ~85 KB each time.
    // (These serve the synchronous edit fast path; workers have their own.)
    MeshSnapshot _meshSnapshot;
    MeshScratch _meshScratch;

    // Async meshing: workers build, renderChunks drains + uploads. Both
    // vectors are reused across frames.
    std::unique_ptr<MeshScheduler> _meshScheduler;
    std::vector<MeshResult> _pendingResults;
    std::vector<ExactMeshCandidate> _meshCandidates;
    // Reused publication batch for exact sections whose collision becomes
    // authoritative in the same visual handoff as their revision-ready mesh.
    std::vector<ChunkPos> _exactCollisionOwnedSections;
    std::vector<ChunkPos> _publishedExactCollisionOwnedSections;
    std::optional<uint64_t> _publishedExactCollisionCoverageEpoch;
    // Once a revision-matched exact mesh is published, it retains ownership
    // while that immutable mesh remains resident. A later edit can enqueue a
    // replacement without briefly resurfacing far terrain underneath it.
    std::unordered_set<ChunkPos> _exactOwnedSections;
    // Shadow encoding precedes the color pass, so it replays the prior frame's
    // stable exact ownership. PREVIEW and FINAL far surfaces both consume the
    // same column mask and cannot cast beneath an exact owner.
    std::unordered_set<ChunkPos> _exactShadowOwnedSections;
    FarTerrainExactCoverageCache _farTerrainExactCoverage;
    // Exact terrain and water need only the narrow surface section set. Far
    // canopy remains visible until every optional section that can contain an
    // exact trunk or crown is also revision-ready.
    FarTerrainExactCoverageCache _farTerrainExactFloraCoverage;

    // ---- Far-terrain LOD annulus ----
    // Exact cubic terrain stops at radius 32. Immutable 256x256-block tiles
    // cover the remaining visible annulus with 2, 4, 8, and 16-block sampling.
    // CPU construction, CPU caching, and GPU residency all have independent
    // hard bounds so the 512-chunk horizon remains inside the 64 GB target.
    std::unique_ptr<FarTerrainScheduler> _farTerrainScheduler;
    std::unique_ptr<SegmentedMegaBuffer> _farMegaBuffer;
    std::optional<uint64_t> _farTerrainSeed;
    std::optional<uint64_t> _farTerrainWorldInstanceId;
    std::optional<ColumnPos> _farTerrainCenterTile;
    // Starts conservatively and records the completed prior-frame policy so an
    // early planner pass cannot reopen all sixteen workers before reevaluating
    // current protected and exact publication debt.
    bool _farTerrainLocalTerrainDebt = true;
    std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash> _farTerrainMeshes;
    std::unordered_map<FarTerrainKey, FarCanopyMeshState, FarTerrainKeyHash> _farCanopyAttachments;
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> _farTerrainWanted;
    std::vector<FarTerrainKey> _farTerrainPriorityOrder;
    std::unordered_set<ColumnPos> _farTerrainActiveTiles;
    std::unordered_map<ColumnPos, FarTerrainKey> _farTerrainDesiredByTile;
    std::unordered_map<ColumnPos, FarTerrainKey> _farTerrainDisplayedByTile;
    std::unordered_map<ColumnPos, FarTerrainLodTransition> _farTerrainTransitions;
    // A source attachment can outlive its geometric LOD while the displayed
    // target's independent attachment builds. The active-tile sweep and the
    // existing optional arena eviction policy keep this map bounded.
    std::unordered_map<ColumnPos, FarTerrainKey> _farCanopyLodFallbacks;
    std::unordered_map<FarTerrainKey, FarTerrainAuthorityTransition, FarTerrainKeyHash>
        _farTerrainAuthorityTransitions;
    struct FarShadowDrawPlan {
        ColumnPos coordinate;
        FarTerrainKey key;
        FarTerrainMeshState state;
        simd_uint4 farMetadata{};
        FarTerrainOwnershipUniforms farOwnership{};
        std::optional<FarCanopyMeshState> canopy;
    };
    // Exact copy of the prior color pass's eligible far draw plans. Shadows
    // encode earlier in the next frame, so retaining this authority keeps LOD
    // transitions and disconnected coverage from diverging.
    std::vector<FarShadowDrawPlan> _farShadowDrawPlans;
    // Coordinate-keyed grace state makes the full candidate pass linear. A
    // vector lookup here previously became quadratic while nearby parents
    // arrived together.
    std::unordered_map<ColumnPos, double> _farTerrainNearGraceStartedAt;
    std::vector<FarTerrainResult> _farTerrainResults;
    std::vector<FarCanopyResult> _farCanopyResults;
    // Stable preparation frames reuse the nearest-first circular selection.
    // Desired LOD still updates below from immutable bounds/distances, but
    // avoiding a 3k-tile clear, sort, and allocation keeps the menu responsive
    // while the cold horizon is waiting on generation authority.
    std::optional<std::pair<double, double>> _farTerrainSelectionCamera;
    // Screen error follows camera motion more closely than the expensive
    // horizon selection. A four-block refresh can refine nearby desired tiers
    // without rebuilding residency when the immutable key set is unchanged.
    std::optional<std::pair<double, double>> _farTerrainDesiredMetricsCamera;
    // Updated only after one chunk of movement so sub-block camera jitter
    // cannot continually replace the bounded speculative authority plan.
    std::optional<std::pair<double, double>> _farTerrainSpeculativeCamera;
    int _farTerrainSelectionViewDistance = -1;
    FarTerrainProtectedNearHandoff _farTerrainProtectedNearHandoff;
    uint64_t _farTerrainProtectedNearEpoch = 0;
    int _farTerrainProtectedRecentMotionX = 0;
    int _farTerrainProtectedRecentMotionZ = 0;
    std::optional<ColumnPos> _farTerrainPredictedNearAnchor;
    struct FarTerrainProtectedNearClosureSnapshot {
        uint32_t wantedTileCount = 0;
        uint32_t residentTileCount = 0;
        uint32_t missingTileCount = 0;
        uint32_t boundaryMismatchCount = 0;
        std::array<uint32_t, 5> targetCountsByStep{};
        uint32_t finalParentCount = 0;
        uint32_t finalTargetCount = 0;
        uint32_t exactCompatibleTargetCount = 0;
        uint32_t lodMismatchCount = 0;
        uint32_t authorityMismatchCount = 0;
        bool ready = false;
        ColumnPos anchor;
        uint64_t viewEpoch = 0;
        uint64_t worldEpoch = 0;
        uint64_t protectedEpoch = 0;
    };
    std::optional<FarTerrainProtectedNearClosureSnapshot> _farTerrainProtectedNearClosureSnapshot;
    uint32_t _farTerrainDesiredViewportHeight = 0;
    double _farTerrainDesiredVerticalFovRadians = 0.0;
    bool _farTerrainDesiredDrawGeometry = false;
    bool _farTerrainDesiredMetricsDirty = true;
    uint64_t _farTerrainViewEpoch = 0;
    uint64_t _farTerrainWorldEpoch = 0;
    std::vector<FarTerrainViewTile> _farTerrainCandidates;
    std::vector<FarTerrainKey> _farTerrainCachedBaseRequests;
    std::vector<FarTerrainKey> _farTerrainMissingBaseRequests;
    std::vector<FarTerrainKey> _farTerrainDistantBaseRequests;
    std::vector<FarTerrainKey> _farTerrainFinalBaseRequests;
    std::vector<FarTerrainRefinementCacheRequest> _farTerrainPerceptualFinalRequests;
    std::vector<FarTerrainKey> _farTerrainFinalRefinementRequests;
    std::vector<FarTerrainCanopyRefreshRequest> _farTerrainCanopyRefreshRequests;
    std::vector<FarTerrainKey> _farTerrainCanopyRefreshKeys;
    std::vector<FarTerrainRefinementCacheRequest> _farTerrainUrgentRefinementRequests;
    std::vector<FarTerrainKey> _farTerrainUrgentRefinementKeys;
    std::vector<FarTerrainKey> _farTerrainConnectedNearPatchTargets;
    std::vector<worldgen::learned::NativeRect> _farTerrainProtectedFinalTerrainRegions;
    // Directional closure preparation remains CPU-only until its canonical
    // anchor is requested. These keys never enter desired, display, upload,
    // GPU-critical, or closure-stat state.
    std::vector<FarTerrainKey> _farTerrainPredictedNearPatchTargets;
    std::vector<FarTerrainKey> _farTerrainPredictedCriticalResidencyKeys;
    std::vector<ColumnPos> _farTerrainCriticalResidencyCoordinates;
    std::vector<FarTerrainKey> _farTerrainCriticalResidencyTargets;
    std::vector<ColumnPos> _farTerrainCriticalResidencyCoordinateScratch;
    std::vector<FarTerrainKey> _farTerrainCriticalResidencyTargetScratch;
    // Every parent and adjacent refinement needed to publish the current
    // camera-critical targets. Scheduler cache admission treats this complete
    // lineage as protected, so a distant parent cannot evict the only legal
    // bridge to nearby maximum detail.
    std::vector<FarTerrainKey> _farTerrainCriticalResidencyKeys;
    std::vector<FarTerrainKey> _farTerrainRefinementSubmissionKeys;
    std::vector<std::shared_ptr<const FarTerrainMesh>> _farTerrainCachedMeshes;
    std::vector<std::shared_ptr<const FarCanopyAttachment>> _farTerrainCachedCanopies;
    FarTerrainPlannerTimingHistogram _farTerrainPlannerTimings;
    FarTerrainPlannerTimingHistogram _farTerrainSelectionTimings;
    FarTerrainPlannerTimingHistogram _farTerrainPublicationTimings;
    FarTerrainPlannerTimingHistogram _farTerrainResidencyTimings;
    uint64_t _farTerrainArenaAdmissionDeniedCount = 0;
    uint64_t _farTerrainNearArenaReclaimCount = 0;
    uint64_t _farTerrainNearArenaReclaimedBytes = 0;
    size_t _farTerrainResidentWantedCount = 0;
    size_t _farTerrainResidentRefinementCount = 0;

    void renderFarTerrain(id<MTLRenderCommandEncoder> encoder, const World& world,
                          const Camera& camera, const float fogColor[3], bool drawGeometry = true,
                          int selectedViewDistance = -1,
                          std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactCoverage = {},
                          bool prepareProtectedFinal = false);
    bool updateFarTerrainSelection(const Vec3& cameraPosition, int visibleChunks);
    void resetFarTerrain(const World& world);
    void setExactSectionOwned(ChunkPos position, bool owned);
    void clearExactSectionOwnership();

    // ---- Day/Night Cycle ----
    // sunDirection/sunColor come out as the ACTIVE directional light, the sun
    // by day, the moon (dim cool light) by night, so terrain shading and
    // shadows share one light. The sky keeps the real sun/moon positions for
    // its discs. shadowStrength is the cascade term's weight (0 at the horizon
    // crossing so the sun→moon swap never pops).
    void computeDayNightUniforms(uint64_t worldTime, float sunDirection[3], float sunColor[3],
                                 float ambientColor[3], SkyUniforms& skyUniforms,
                                 float& shadowStrength);

    // ---- Scene pass stages (all encode into the single MSAA scene encoder) ----
    void renderSky(id<MTLRenderCommandEncoder> encoder, const FrameRing::Alloc& skyUniforms);

    void renderChunks(id<MTLRenderCommandEncoder> encoder, const World& world,
                      const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                      const Mat4& viewMatrix, const Mat4& projectionMatrix, const Camera& camera,
                      const float sunDirection[3], const float sunColor[3],
                      const float ambientColor[3], const float fogColor[3],
                      const FoliageWindUniforms& foliageWind, bool drawGeometry = true,
                      bool prepareProtectedFinal = false);

    // Encode the cascade depth passes before the scene pass (no-op when
    // shadowQuality is 0 or strength is 0). lightDirection is the active
    // directional light (sun by day, moon by night). Fills
    // _sceneShadowUniforms for the chunk fragment; reuses loadedChunks so the
    // locked chunk-list copy happens once per frame.
    void renderShadows(id<MTLCommandBuffer> commandBuffer,
                       const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                       const Camera& camera, const float lightDirection[3], float strength,
                       const FoliageWindUniforms& foliageWind,
                       const std::vector<std::shared_ptr<Entity>>* entities,
                       const std::vector<ItemEntity>* itemEntities, const std::vector<Boat>* boats);

    void renderBlockHighlight(id<MTLRenderCommandEncoder> encoder, const BlockHighlight& highlight,
                              const Mat4& viewMatrix, const Mat4& projectionMatrix);

    // Water pass: own encoder after the scene pass resolves. Encodes the
    // recorded _waterDraws plus the underwater overlay when submerged.
    void renderWater(id<MTLCommandBuffer> commandBuffer, const Mat4& viewMatrix,
                     const Mat4& projectionMatrix, const Vec3& cameraPosition,
                     bool cameraUnderwater, const SkyUniforms& skyUniforms,
                     const float directLightDirection[3], const float directLightRadiance[3],
                     const float fogColor[3]);

    void renderUIOverlay(id<MTLRenderCommandEncoder> encoder, const UIFrameState& uiFrame);
};

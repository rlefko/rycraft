#pragma once

#include "common/math.hpp"
#include "world/learned_terrain.hpp"
#include "world/save_manager.hpp"
#include "world/terrain_bootstrap.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

inline constexpr Vec3 GENERATOR_V4_INITIAL_SPAWN{512.f, 100.f, 512.f};

namespace worldgen {
struct HydrologySample;
}

enum class V4WorldOpenStatus : uint8_t {
    Ready,
    BootstrapNotReady,
    InvalidWorldDirectory,
    MissingMetadata,
    IdentityConflict,
    PersistenceFailure,
};

constexpr bool v4WorldOpenFailureRetryable(V4WorldOpenStatus status) noexcept {
    return status == V4WorldOpenStatus::BootstrapNotReady ||
           status == V4WorldOpenStatus::PersistenceFailure;
}

constexpr bool v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus status) noexcept {
    return status == V4WorldOpenStatus::InvalidWorldDirectory ||
           status == V4WorldOpenStatus::MissingMetadata ||
           status == V4WorldOpenStatus::IdentityConflict;
}

struct V4WorldOpenResult {
    V4WorldOpenStatus status = V4WorldOpenStatus::BootstrapNotReady;
    std::unique_ptr<SaveManager> saveManager;
    SaveManager::WorldMetadata metadata;
    // The exact profile selected by the caller, or the default profile when
    // no override was supplied. Identity conflicts fail closed.
    std::filesystem::path profilePath;
    bool usingSeparateProfile = false;
    bool newlyCreated = false;
    bool fresh = false;
    std::string message;

    [[nodiscard]] bool ready() const noexcept {
        return status == V4WorldOpenStatus::Ready && saveManager != nullptr;
    }
};

// An explicit Worlds-screen request to publish a distinct generator v4
// profile. Existing profiles are never reused or modified for this intent,
// even when they have the same seed and generation fingerprint. This lets
// players create more than one world from one immutable generator identity.
struct V4WorldCreationRequest {
    std::string displayName;
    GameMode gameMode = GameMode::SURVIVAL;
    GenerationSettings generation;
    SaveManager::PlayerMetadata player;
};

// Resolves an optional playtest/capture profile override without touching the
// filesystem. Relative values are direct children of applicationSupport;
// absolute values remain exact. openQualifiedV4World performs the containment
// and profile-name validation before any persistence mutation.
std::optional<std::filesystem::path>
resolveV4LaunchProfilePath(const std::filesystem::path& applicationSupport,
                           std::string_view requestedPath);

// Opens or initializes a generator v4 persistence profile only after model
// and runtime qualification. A fresh identity uses rycraft_world_v4. An
// existing profile with another seed or fingerprint fails closed and must be
// resolved through explicit world selection or creation. Legacy
// rycraft_world is never examined or created.
V4WorldOpenResult
openQualifiedV4World(worldgen::bootstrap::TerrainGenerationBootstrap& bootstrap, uint64_t seed,
                     Vec3 initialSpawn = GENERATOR_V4_INITIAL_SPAWN, uint64_t initialWorldTime = 0,
                     std::shared_ptr<SaveManager::TestHooks> persistenceTestHooks = nullptr,
                     std::optional<std::filesystem::path> preferredProfilePath = std::nullopt,
                     std::optional<V4WorldCreationRequest> creationRequest = std::nullopt);

enum class V4SpawnAuthorityPrequeueStatus : uint8_t {
    Ready,
    Deferred,
    Failed,
};

struct V4SpawnAuthorityPrequeueResult {
    // Logical dependency counts. The topology closure can contain refinement
    // coordinates, so their sum is not necessarily the unique page count.
    size_t finalTopologyPageCount = 0;
    size_t finalRefinementPageCount = 0;
    size_t finalPageCount = 0;
    size_t hydrologyOwnerCount = 0;
    size_t preparedHydrologyOwnerCount = 0;
    bool reusedPreparedHydrology = false;
    bool reusedCertifiedDryFootprint = false;
    V4SpawnAuthorityPrequeueStatus status = V4SpawnAuthorityPrequeueStatus::Deferred;
    std::optional<worldgen::learned::GenerationFailure> failure;

    // Repair replacement and qualification are legal only after every exact
    // refinement page is cached and atomically published and every topology
    // owner is prepared semantically, the complete cold exact footprint is
    // installed as a strict dry certificate, or the owner is backed by its
    // complete FINAL page closure. A queued or building dependency remains
    // deferred even when all of its flights were admitted.
    [[nodiscard]] bool ready() const noexcept {
        return status == V4SpawnAuthorityPrequeueStatus::Ready;
    }
    [[nodiscard]] bool allowsWorldConstruction() const noexcept { return ready(); }
    [[nodiscard]] bool deferred() const noexcept {
        return status == V4SpawnAuthorityPrequeueStatus::Deferred;
    }
    [[nodiscard]] bool failed() const noexcept {
        return status == V4SpawnAuthorityPrequeueStatus::Failed;
    }
};

// The coarse learned selector can propose a high, flat point that final
// canonical hydrology later classifies as a lake, river, wetland, or other
// surface water. The screen directly prepares its single native owner, then
// finds a globally four-block-aligned point with a five-by-five flatness
// buffer. The common path installs a strict page-local canonical dry proof.
// An all-positive continental owner can return only a provisional FINAL
// learned proposal; World's radius-zero exact plan remains the mandatory
// canonical admission check. The screen runs on its own worker and therefore
// never executes inference, waits for authority, or routes a page on the
// render thread.
enum class V4SpawnWaterScreenStatus : uint8_t {
    Dry,
    // The candidate has no locally safe canonical dry site. This includes
    // water bodies and dry points that fail the required local buffer.
    Water,
    Deferred,
    Failed,
};

struct V4SpawnWaterScreenResult {
    V4SpawnWaterScreenStatus status = V4SpawnWaterScreenStatus::Deferred;
    std::optional<worldgen::learned::GenerationFailure> failure;
    // Every dry result carries the point that the next startup stage must
    // use. It can move anywhere inside the proposed candidate's original
    // half-open native hydrology owner, but never crosses an owner edge or
    // opens a neighboring owner. World still owns final canonical water,
    // collision, support, and headroom validation.
    std::optional<Vec3> resolvedCandidate;
    // The conservative page-local proof cannot certify an all-positive
    // continental owner because it has no permanent ocean terminal. This
    // rare fallback carries only a positive, flat FINAL learned proposal. It
    // installs no canonical dry certificate and cannot finalize metadata;
    // World's radius-zero exact plan must still reject any lake, wetland,
    // river, unsupported surface, or blocked headroom before entry.
    bool provisionalLearnedDry = false;

    [[nodiscard]] bool dry() const noexcept { return status == V4SpawnWaterScreenStatus::Dry; }
    [[nodiscard]] bool canonicalDryCertificateInstalled() const noexcept {
        return dry() && !provisionalLearnedDry;
    }
    [[nodiscard]] bool requiresExactPlanValidation() const noexcept { return dry(); }
    [[nodiscard]] bool water() const noexcept { return status == V4SpawnWaterScreenStatus::Water; }
    [[nodiscard]] bool deferred() const noexcept {
        return status == V4SpawnWaterScreenStatus::Deferred;
    }
    [[nodiscard]] bool failed() const noexcept {
        return status == V4SpawnWaterScreenStatus::Failed;
    }
};

// A spawn candidate is dry only when its own canonical column has no water
// state. This deliberately treats an active delta or any named transition as
// water even if a coarse classification flag has not yet reached that sample.
// A non-finite stage or bed is likewise unsafe and must not authorize entry.
[[nodiscard]] bool
v4SpawnCandidateHasCanonicalSurfaceWater(const worldgen::HydrologySample& hydrology) noexcept;

// Safe-spawn placement remains pinned to the native four-block raster so it
// cannot invent a separate shoreline.
inline constexpr int V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS = worldgen::learned::MODEL_BLOCK_SCALE;
inline constexpr int V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES = 2;
// Every accepted proposal first validates only its center chunk. A canonical
// result has an installed local dry certificate; a provisional continental
// result has learned terrain only. In both cases the radius-zero exact plan is
// the mandatory water, collision, support, and headroom authority before
// metadata finalization or gameplay entry.
inline constexpr int V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS = 0;

// A cold exact footprint is the union of every 49-by-49 hydrology query made
// by the production active disk, horizontal mesh halo, and ColumnPlan apron.
// Rows are inclusive intervals relative to the center chunk's world origin.
// Keeping the nonrectangular shape explicit prevents startup from certifying
// unrelated corners or omitting the rounded active-set shoulders.
struct V4SpawnExactFootprintRow {
    int64_t zOffset = 0;
    int64_t minimumXOffset = 0;
    int64_t maximumXOffset = -1;

    [[nodiscard]] constexpr size_t sampleCount() const noexcept {
        return maximumXOffset >= minimumXOffset
                   ? static_cast<size_t>(maximumXOffset - minimumXOffset + 1)
                   : 0;
    }

    auto operator<=>(const V4SpawnExactFootprintRow&) const = default;
};

struct V4SpawnAlignedCandidate {
    int64_t worldX = 0;
    int64_t worldZ = 0;

    auto operator<=>(const V4SpawnAlignedCandidate&) const = default;
};

// Learned suitability tiers can contain many equally safe dry points. Prefer
// the point that asks the protected FINAL handoff to open fewer direct native
// hydrology owners, then the point whose protected owners overlap more of the
// already-required exact spawn closure. Exact owner and terrain-page counts
// retain their existing tie-break order after those entry-wide costs.
struct V4SpawnPlacementAuthorityCost {
    size_t protectedDirectOwnerCount = 0;
    size_t protectedExactOverlapCount = 0;
    size_t exactOwnerCount = 0;
    size_t finalRefinementPageCount = 0;

    auto operator<=>(const V4SpawnPlacementAuthorityCost&) const = default;
};

[[nodiscard]] std::optional<V4SpawnPlacementAuthorityCost>
v4SpawnPlacementAuthorityCost(int64_t worldX, int64_t worldZ) noexcept;

[[nodiscard]] constexpr bool
v4SpawnPlacementAuthorityPreferred(const V4SpawnPlacementAuthorityCost& candidate,
                                   const V4SpawnPlacementAuthorityCost& current) noexcept {
    if (candidate.protectedDirectOwnerCount != current.protectedDirectOwnerCount) {
        return candidate.protectedDirectOwnerCount < current.protectedDirectOwnerCount;
    }
    if (candidate.protectedExactOverlapCount != current.protectedExactOverlapCount) {
        return candidate.protectedExactOverlapCount > current.protectedExactOverlapCount;
    }
    if (candidate.exactOwnerCount != current.exactOwnerCount)
        return candidate.exactOwnerCount < current.exactOwnerCount;
    return candidate.finalRefinementPageCount < current.finalRefinementPageCount;
}

// Returns the production cold footprint as sorted row-contiguous intervals.
// An empty result means the streaming constants no longer describe a valid
// bounded footprint and startup must fail closed.
[[nodiscard]] std::vector<V4SpawnExactFootprintRow> v4ColdSpawnExactFootprintRows();

// Expands the production footprint around a center chunk into distinct
// row-major world columns. Overflow returns no value rather than wrapping a
// startup certificate into an unrelated coordinate.
[[nodiscard]] std::optional<std::vector<V4SpawnAlignedCandidate>>
v4ColdSpawnExactFootprintPoints(int64_t centerChunkX, int64_t centerChunkZ);

// Expands only the radius-zero exact safety column's five-by-five ColumnPlan
// dependency apron. Each plan consumes one inclusive 49-by-49 hydrology
// raster, so their union is the exact 113-by-113 block footprint needed to
// prove collision, support, and headroom before ordinary cold streaming is
// released. The result is row-major and contains no duplicates.
[[nodiscard]] std::optional<std::vector<V4SpawnAlignedCandidate>>
v4SpawnCertificationExactFootprintPoints(int64_t centerChunkX, int64_t centerChunkZ);

// Finds center chunks whose local five-by-five native safety buffer is
// certified by a row-major globally aligned four-block lattice mask beginning
// at (`originWorldX`, `originWorldZ`). The wider production footprint may
// contain canonical rivers, lakes, wetlands, or coasts, so the caller may use
// these candidates for a stronger exact certificate or retain the local proof
// as a strict semantic fallback. Each result is the nearest four-block-aligned
// point inside its chunk to `requestedCandidate`. Within that certification
// tier, results preserve 64-block squared-distance locality bands, minimize
// the protected FINAL owner footprint, maximize reuse of already-required
// exact owners, retain the exact-owner and terrain-page tie breaks, then use
// squared distance followed by world Z and X. This pure helper performs no
// authority or persistence work and is shared by startup and focused
// regressions.
[[nodiscard]] std::vector<V4SpawnAlignedCandidate>
v4RankCertifiedDrySpawnCandidates(Vec3 requestedCandidate, int64_t originWorldX,
                                  int64_t originWorldZ, int sampleWidth, int sampleHeight,
                                  std::span<const uint8_t> certified);

// Chooses the nearest locally flat, dry point whose complete five-by-five
// canonical-water buffer is dry. `samples` is a row-major four-block native
// raster beginning at (`originWorldX`, `originWorldZ`). This pure selection
// step is shared by the asynchronous screen and focused regression tests;
// it performs no authority query or persistence work.
[[nodiscard]] std::optional<Vec3>
v4SelectLocalDrySpawnCandidate(Vec3 requestedCandidate, int64_t originWorldX, int64_t originWorldZ,
                               int sampleWidth, int sampleHeight,
                               std::span<const worldgen::HydrologySample> samples) noexcept;

inline constexpr std::chrono::milliseconds V4_SPAWN_WATER_SCREEN_RETRY_INTERVAL{5};
inline constexpr std::chrono::milliseconds V4_SPAWN_WATER_SCREEN_NO_PROGRESS_TIMEOUT{30'000};
// A nonzero queued or active authority/hydrology count proves that the screen
// has admitted bounded work. That work can include one cold Core ML batch, so
// the complete screen receives the same hard five-minute settlement budget as
// startup. The deadline is immutable from request admission and is distinct
// from the short idle watchdog. Neither progress nor repeated deferred replies
// can extend it.
inline constexpr std::chrono::milliseconds V4_SPAWN_WATER_SCREEN_ACTIVE_WORK_TIMEOUT{300'000};

struct V4SpawnWaterScreenTiming {
    // Both values are bounded by V4SpawnWaterScreen before use. The explicit
    // timing object lets deterministic tests exercise the failure policy
    // without turning a no-progress regression into a thirty-second test.
    std::chrono::milliseconds retryInterval = V4_SPAWN_WATER_SCREEN_RETRY_INTERVAL;
    std::chrono::milliseconds noProgressTimeout = V4_SPAWN_WATER_SCREEN_NO_PROGRESS_TIMEOUT;
    std::chrono::milliseconds activeWorkTimeout = V4_SPAWN_WATER_SCREEN_ACTIVE_WORK_TIMEOUT;
};

class V4SpawnWaterScreen {
public:
    explicit V4SpawnWaterScreen(V4SpawnWaterScreenTiming timing = {});
    ~V4SpawnWaterScreen();

    V4SpawnWaterScreen(const V4SpawnWaterScreen&) = delete;
    V4SpawnWaterScreen& operator=(const V4SpawnWaterScreen&) = delete;

    // Starts or polls a bounded canonical-water screen for `candidate`. An
    // all-positive owner can return a provisional learned proposal only after
    // its conservative canonical proof has no valid center.
    // Repeating the same request is idempotent. A new candidate supersedes an
    // unfinished one without waiting for it on the caller's thread. Call
    // reset before retrying a terminal screen failure.
    [[nodiscard]] V4SpawnWaterScreenResult
    screen(const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
           Vec3 candidate);

    // Stops polling a completed or superseded candidate. It does not cancel
    // an already-running immutable hydrology page, which may remain useful in
    // the shared context cache.
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Nonblockingly prepares the complete FINAL hydrology topology and terrain
// refinement closure for explicit repair and qualification. Normal entry does
// not use this wide closure: its one-owner water screen either installs a
// 25-sample local dry certificate or emits an explicitly provisional learned
// proposal, then constructs World for radius-zero exact validation. Wider
// exact and horizon work can begin after that validation. Deferred means a
// repair caller polls on a later frame.
// Ready means every requested page is cached and durably published. PREVIEW
// authority remains reserved for the coarse far horizon and cannot authorize
// spawn water or collision.
V4SpawnAuthorityPrequeueResult prequeueV4SpawnAuthority(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& generationContext,
    int64_t worldX, int64_t worldZ, int radiusChunks = 1);

// A fresh world may begin over learned ocean. This nonblocking probe examines
// the model's actual coarse output in deterministic regions. One coarse cell
// is one 256-pixel authority page: 1,024 blocks or 7.68 kilometers at the
// v4 physical scale. A 16-cell square is therefore the largest page-aligned
// region whose half-edge is 61.44 kilometers, the representable bound below
// the 64-kilometer spawn-search contract. The Minecraft reference's 128-cell
// range assumes one-meter blocks and would cover 491.52 kilometers here.
//
// The probe never materializes preview authority pages. It queries one stable
// maximum-size coarse grid, so an ordinal identifies one candidate without
// being affected by smaller-grid borders or retries. Final learned authority
// must independently approve every returned candidate before exact hydrology
// and collision validate it. Callers advance the ordinal only after a final
// page has no candidate or a later exact safety check rejects it.
inline constexpr double V4_DRY_SPAWN_SEARCH_TARGET_RADIUS_METERS = 64'000.0;
inline constexpr double V4_DRY_SPAWN_COARSE_CELL_EDGE_METERS =
    static_cast<double>(worldgen::learned::AUTHORITY_PAGE_NATIVE_EDGE) *
    worldgen::learned::MODEL_BLOCK_SCALE * worldgen::learned::WORLD_METERS_PER_BLOCK;
inline constexpr uint16_t V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE = 16;
inline constexpr double V4_DRY_SPAWN_SEARCH_REPRESENTABLE_HALF_EDGE_METERS =
    static_cast<double>(V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE / 2) *
    V4_DRY_SPAWN_COARSE_CELL_EDGE_METERS;
static_assert(V4_DRY_SPAWN_SEARCH_REPRESENTABLE_HALF_EDGE_METERS <=
              V4_DRY_SPAWN_SEARCH_TARGET_RADIUS_METERS);
static_assert((static_cast<double>(V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE / 2 + 1) *
               V4_DRY_SPAWN_COARSE_CELL_EDGE_METERS) > V4_DRY_SPAWN_SEARCH_TARGET_RADIUS_METERS);
// The preferred pass requires a one-cell dry border, but the fallback must
// still examine every coarse owner. A narrow island or low coastal plain is a
// valid final-authority proposal even when it cannot satisfy that preference.
// One proposal represents each aligned two-by-two authority-page hydrology
// owner, preventing repeated construction of a rejected owner. An odd search
// origin can intersect nine owners per axis. Canonical water and exact
// collision remain the admission authority.
inline constexpr uint32_t V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES =
    static_cast<uint32_t>(V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE / 2 + 1) *
    static_cast<uint32_t>(V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE / 2 + 1);
// Far terrain remains dormant until canonical water screening has accepted a
// candidate and its FINAL spawn authority is ready. World construction proves
// both conditions, so horizon preparation can overlap exact collision and
// headroom validation. Gameplay entry still requires the finalized safe spawn
// through v4EntryHorizonReady and the engine's separate spawn gates.
constexpr bool v4MayPrepareHorizon(bool canonicalDryCandidateAccepted,
                                   bool finalSpawnAuthorityReady) noexcept {
    return canonicalDryCandidateAccepted && finalSpawnAuthorityReady;
}

// Six far-terrain tile widths cover the radius-five protected closure plus
// the renderer's maximum one-tile frontier fade. Protected diagonal targets
// beyond this radial prefix stay suppressed until their parent frontier
// advances, so they cannot appear as disconnected islands.
inline constexpr int V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS = 96;

constexpr int v4RequiredEntryParentRadiusChunks(int configuredHorizonChunks) noexcept {
    if (configuredHorizonChunks <= 0) return 0;
    return configuredHorizonChunks < V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS
               ? configuredHorizonChunks
               : V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS;
}

// Entry needs a connected coarse terrain-and-water parent frontier through
// the bounded entry radius. The full configured horizon remains selected and
// continues filling after entry. The exact spawn band is already FINAL and
// collision-safe before this gate runs. Protected FINAL far refinement and
// canopies cannot delay or satisfy this gate.
constexpr bool v4EntryConnectedParentReady(int configuredHorizonChunks, int entryHorizonChunks,
                                           float connectedParentRadiusChunks, uint32_t baseWanted,
                                           uint32_t baseResident, uint32_t baseMissing) noexcept {
    if (configuredHorizonChunks <= 0 || entryHorizonChunks <= 0 ||
        entryHorizonChunks != v4RequiredEntryParentRadiusChunks(configuredHorizonChunks) ||
        baseWanted == 0 || static_cast<uint64_t>(baseResident) + baseMissing != baseWanted) {
        return false;
    }
    if (baseMissing == 0) return baseResident == baseWanted;
    if (entryHorizonChunks == configuredHorizonChunks) return false;
    return baseResident > 0 && connectedParentRadiusChunks >= entryHorizonChunks &&
           connectedParentRadiusChunks < configuredHorizonChunks;
}

constexpr bool v4EntryHorizonReady(bool permitted, bool matchesSpawn, uint32_t freshFrames,
                                   int requiredHorizonChunks, int selectedHorizonChunks,
                                   int entryHorizonChunks, float connectedParentRadiusChunks,
                                   uint32_t baseWanted, uint32_t baseResident,
                                   uint32_t baseMissing) noexcept {
    return permitted && matchesSpawn && freshFrames >= 2 && requiredHorizonChunks > 0 &&
           selectedHorizonChunks == requiredHorizonChunks &&
           v4EntryConnectedParentReady(requiredHorizonChunks, entryHorizonChunks,
                                       connectedParentRadiusChunks, baseWanted, baseResident,
                                       baseMissing);
}

// Gameplay entry publishes one atomic FINAL near-player closure. Its
// position-aware two-by-two core is surrounded by Manhattan shells whose
// sample step doubles at every shell. The exterior therefore meets step 32
// with a 2:1 transition, while the player begins inside a complete step 1
// surface. These counts describe target coordinates, not triangles or model
// pages, and are intentionally exact so a partial or duplicated closure
// cannot authorize entry.
inline constexpr std::array<uint32_t, 5> V4_ENTRY_FINAL_TARGETS_BY_STEP{4, 8, 12, 16, 20};
inline constexpr std::array<uint8_t, 5> V4_ENTRY_FINAL_TARGET_STEPS{1, 2, 4, 8, 16};
inline constexpr uint32_t V4_ENTRY_FINAL_TARGET_COUNT = 60;
inline constexpr uint32_t V4_ENTRY_COLLISION_CUBE_COUNT = 27;
static_assert([] {
    uint32_t total = 0;
    for (const uint32_t count : V4_ENTRY_FINAL_TARGETS_BY_STEP)
        total += count;
    return total == V4_ENTRY_FINAL_TARGET_COUNT;
}());

// Preparation progress consumes the renderer's FINAL-compatible count, but a
// retained snapshot belongs to the prior request until its closure epoch has
// been revalidated. Suppressing that stale count keeps the menu consistent
// with the fail-closed gameplay gate.
constexpr uint32_t v4NearEntryFinalCompatibleProgress(uint64_t currentProtectedEpoch,
                                                      uint64_t closureProtectedEpoch,
                                                      uint32_t finalCompatibleCount) noexcept {
    if (currentProtectedEpoch == 0 || currentProtectedEpoch != closureProtectedEpoch) return 0;
    return std::min(finalCompatibleCount, V4_ENTRY_FINAL_TARGET_COUNT);
}

struct V4NearEntryClosureAnchor {
    // The lower-left far-tile coordinate of the position-aware two-by-two
    // step 1 core. The camera's half-tile position determines this anchor.
    int64_t minimumTileX = 0;
    int64_t minimumTileZ = 0;

    auto operator<=>(const V4NearEntryClosureAnchor&) const = default;
};

// This is an aggregate of completed renderer and World state. Every ready
// count includes only an item whose generator identity, FINAL quality, view
// epoch, world epoch, protected-handoff epoch, and anchor match the closure.
// Callers must not count queued uploads, PREVIEW payloads, stale fallback
// geometry, or meshes from another revision. Canopy is deliberately absent:
// it can neither satisfy nor delay terrain, water, collision, or exact-mesh
// publication.
struct V4NearEntryClosureInput {
    uint64_t currentViewEpoch = 0;
    uint64_t closureViewEpoch = 0;
    uint64_t currentWorldEpoch = 0;
    uint64_t closureWorldEpoch = 0;
    uint64_t currentProtectedEpoch = 0;
    uint64_t closureProtectedEpoch = 0;
    V4NearEntryClosureAnchor currentAnchor;
    V4NearEntryClosureAnchor closureAnchor;

    bool connectedPreviewParentPrefixReady = false;
    std::array<uint32_t, 5> finalTargetCountsByStep{};
    uint32_t matchingFinalParentsUploaded = 0;
    uint32_t matchingFinalParentsResident = 0;
    uint32_t matchingFinalChildrenUploaded = 0;
    uint32_t matchingFinalChildrenResident = 0;
    uint32_t exactCompatibleTargets = 0;
    uint32_t lodTransitionMismatches = 0;
    uint32_t authorityTransitionMismatches = 0;

    uint32_t collisionCubesReady = 0;
    uint32_t exactMeshesRequired = 0;
    uint32_t matchingExactMeshesReady = 0;
    uint64_t currentExactMeshRevision = 0;
    uint64_t readyExactMeshRevision = 0;
};

enum class V4NearEntryClosureStatus : uint8_t {
    Ready,
    EpochMismatch,
    AnchorMismatch,
    PreviewParentPrefixIncomplete,
    ProtectedTopologyMismatch,
    FinalParentsIncomplete,
    FinalChildrenIncomplete,
    ExactCompatibilityIncomplete,
    TransitionMismatch,
    CollisionIncomplete,
    ExactMeshesIncomplete,
    ExactMeshRevisionMismatch,
};

// Evaluates the single atomic publication boundary. The ordered result is
// suitable for startup diagnostics, while v4NearEntryClosureReady is the
// fail-closed admission predicate used by gameplay entry.
[[nodiscard]] constexpr V4NearEntryClosureStatus
v4NearEntryClosureStatus(const V4NearEntryClosureInput& input) noexcept {
    if (input.currentViewEpoch != input.closureViewEpoch ||
        input.currentWorldEpoch != input.closureWorldEpoch ||
        input.currentProtectedEpoch != input.closureProtectedEpoch) {
        return V4NearEntryClosureStatus::EpochMismatch;
    }
    if (input.currentAnchor != input.closureAnchor) return V4NearEntryClosureStatus::AnchorMismatch;
    if (!input.connectedPreviewParentPrefixReady)
        return V4NearEntryClosureStatus::PreviewParentPrefixIncomplete;
    if (input.finalTargetCountsByStep != V4_ENTRY_FINAL_TARGETS_BY_STEP)
        return V4NearEntryClosureStatus::ProtectedTopologyMismatch;
    if (input.matchingFinalParentsUploaded != V4_ENTRY_FINAL_TARGET_COUNT ||
        input.matchingFinalParentsResident != V4_ENTRY_FINAL_TARGET_COUNT) {
        return V4NearEntryClosureStatus::FinalParentsIncomplete;
    }
    if (input.matchingFinalChildrenUploaded != V4_ENTRY_FINAL_TARGET_COUNT ||
        input.matchingFinalChildrenResident != V4_ENTRY_FINAL_TARGET_COUNT) {
        return V4NearEntryClosureStatus::FinalChildrenIncomplete;
    }
    if (input.exactCompatibleTargets != V4_ENTRY_FINAL_TARGET_COUNT)
        return V4NearEntryClosureStatus::ExactCompatibilityIncomplete;
    if (input.lodTransitionMismatches != 0 || input.authorityTransitionMismatches != 0)
        return V4NearEntryClosureStatus::TransitionMismatch;
    if (input.collisionCubesReady != V4_ENTRY_COLLISION_CUBE_COUNT)
        return V4NearEntryClosureStatus::CollisionIncomplete;
    if (input.exactMeshesRequired == 0 ||
        input.matchingExactMeshesReady != input.exactMeshesRequired) {
        return V4NearEntryClosureStatus::ExactMeshesIncomplete;
    }
    if (input.readyExactMeshRevision != input.currentExactMeshRevision)
        return V4NearEntryClosureStatus::ExactMeshRevisionMismatch;
    return V4NearEntryClosureStatus::Ready;
}

[[nodiscard]] constexpr bool
v4NearEntryClosureReady(const V4NearEntryClosureInput& input) noexcept {
    return v4NearEntryClosureStatus(input) == V4NearEntryClosureStatus::Ready;
}

// A page-locally certified candidate may authorize horizon preparation while
// exact collision and headroom finish. A learned-only continental proposal
// remains a land-search candidate until radius-zero exact validation accepts
// it, so it cannot report dry land or authorize far work prematurely.
constexpr bool v4CanonicalDrySpawnAccepted(bool candidateActive, bool candidateProvisional,
                                           bool spawnSafetyValidated) noexcept {
    return spawnSafetyValidated || (candidateActive && !candidateProvisional);
}

// Profiles written before the strict dry-spawn revision, or before the safe
// spawn was persisted separately from the player location, must be relocated
// or revalidated once. Ordinary player movement may update `playerPos`, but
// cannot replace the verified safe spawn.
constexpr bool v4SpawnRequiresStrictDryValidation(bool spawnFinalized, uint32_t spawnSafetyRevision,
                                                  bool hasSafeSpawnPos) noexcept {
    return !spawnFinalized || !hasSafeSpawnPos ||
           spawnSafetyRevision != SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
}

// A resumable player position is not spawn authority. In particular, a
// pre-safe-spawn record can point into an ocean far from any dry land inside
// the bounded recovery search. Use a persisted safe spawn as a deterministic
// revalidation anchor when one exists; otherwise restart the search from the
// caller's requested fresh-world location. This keeps a legacy ocean resume
// from trapping its own recovery in another all-ocean search region.
[[nodiscard]] constexpr Vec3 v4DrySpawnSearchOrigin(const std::optional<Vec3>& safeSpawnPos,
                                                    Vec3 requestedSpawn) noexcept {
    return safeSpawnPos ? *safeSpawnPos : requestedSpawn;
}

struct V4DrySpawnRecoverySearch {
    Vec3 primary{};
    // A pre-current safe spawn was accepted under an older validation
    // contract. It remains a useful first recovery location, but cannot be
    // allowed to trap the bounded search if it is actually stale ocean.
    std::optional<Vec3> fallback;
};

// Current-revision profiles do not enter recovery and retain an intentional
// player resume exactly as saved. Strict revalidation tries a legacy safe
// position first for locality, then retries once from the requested fresh
// world anchor if the former has no legal dry candidate.
[[nodiscard]] constexpr V4DrySpawnRecoverySearch
v4DrySpawnRecoverySearch(const std::optional<Vec3>& safeSpawnPos, Vec3 requestedSpawn,
                         bool requiresStrictValidation) noexcept {
    if (!requiresStrictValidation || !safeSpawnPos)
        return {.primary = v4DrySpawnSearchOrigin(safeSpawnPos, requestedSpawn)};
    // The vertical coordinate does not affect the learned page search. Avoid
    // a redundant full retry when only a saved standing height differs.
    if (safeSpawnPos->x == requestedSpawn.x && safeSpawnPos->z == requestedSpawn.z)
        return {.primary = *safeSpawnPos};
    return {.primary = *safeSpawnPos, .fallback = requestedSpawn};
}

worldgen::learned::AuthorityResult<std::optional<Vec3>> findV4DryLandSpawnCandidate(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
    int64_t originWorldX, int64_t originWorldZ, uint32_t ordinal);

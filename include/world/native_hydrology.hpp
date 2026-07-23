#pragma once

#include "world/basin_solver.hpp"
#include "world/hydrology_authority.hpp"
#include "world/learned_terrain.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace worldgen {

inline constexpr int NATIVE_HYDROLOGY_RASTER_SPACING = learned::MODEL_BLOCK_SCALE;
inline constexpr int NATIVE_HYDROLOGY_PAGE_EDGE = hydrology::HYDROLOGY_AUTHORITY_PAGE_BLOCK_EDGE;
// Step-32 far parents are subdivided into these globally aligned compact
// topology cells. This is deliberately coarser than the canonical four-block
// water raster: it identifies only parents that need that raster, while a
// uniform ocean or lake remains a one-quad fast path.
inline constexpr int NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE = 32;
static_assert(NATIVE_HYDROLOGY_RASTER_SPACING == 4);
static_assert(NATIVE_HYDROLOGY_PAGE_EDGE == 2'048);
static_assert(NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE % NATIVE_HYDROLOGY_RASTER_SPACING == 0);
static_assert(NATIVE_HYDROLOGY_PAGE_EDGE % NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE == 0);
inline constexpr double NATIVE_HYDROLOGY_CELL_EDGE_METERS =
    NATIVE_HYDROLOGY_RASTER_SPACING * learned::WORLD_METERS_PER_BLOCK;
// Learned elevation is measured relative to mean sea level. Routing keeps
// that physical datum until it emits game-facing block-height authority.
inline constexpr double NATIVE_HYDROLOGY_SEA_LEVEL_METERS = 0.0;
inline constexpr double NATIVE_HYDROLOGY_CELL_AREA_SQUARE_KILOMETERS =
    NATIVE_HYDROLOGY_CELL_EDGE_METERS * NATIVE_HYDROLOGY_CELL_EDGE_METERS / 1'000'000.0;
// Routed discharge is accumulated as annual runoff millimeters times square
// kilometers. One such unit is 1,000 cubic meters per year. Keep the
// renderable-channel threshold in physical flow units, then convert it once
// so a change in native cell size cannot turn a few pixels of rainfall into a
// river. This only controls river and channel-projection topology: runoff,
// baseflow, recharge, lakes, and wetlands retain their calibrated physical
// values below this display threshold.
inline constexpr double NATIVE_HYDROLOGY_DISCHARGE_UNIT_CUBIC_METERS_PER_YEAR = 1'000.0;
inline constexpr double NATIVE_HYDROLOGY_SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0;
inline constexpr double NATIVE_HYDROLOGY_MINIMUM_CHANNEL_FLOW_LITERS_PER_SECOND = 1.0;
inline constexpr double NATIVE_HYDROLOGY_MINIMUM_CHANNEL_FLOW_CUBIC_METERS_PER_SECOND =
    NATIVE_HYDROLOGY_MINIMUM_CHANNEL_FLOW_LITERS_PER_SECOND / 1'000.0;
inline constexpr double NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE =
    NATIVE_HYDROLOGY_MINIMUM_CHANNEL_FLOW_CUBIC_METERS_PER_SECOND *
    NATIVE_HYDROLOGY_SECONDS_PER_YEAR / NATIVE_HYDROLOGY_DISCHARGE_UNIT_CUBIC_METERS_PER_YEAR;
inline constexpr double NATIVE_HYDROLOGY_PAGE_EDGE_KILOMETERS =
    NATIVE_HYDROLOGY_PAGE_EDGE * learned::WORLD_METERS_PER_BLOCK / 1'000.0;
inline constexpr double NATIVE_HYDROLOGY_PAGE_AREA_SQUARE_KILOMETERS =
    NATIVE_HYDROLOGY_PAGE_EDGE_KILOMETERS * NATIVE_HYDROLOGY_PAGE_EDGE_KILOMETERS;
// Small standalone routers retain one complete bounded reconciliation set:
// the owner, four cardinal depression neighbors, and a possible diagonal
// point-handoff page. The v4 world context opts into the larger visible-
// horizon budget below.
inline constexpr size_t NATIVE_HYDROLOGY_CACHE_BYTE_BUDGET = 192U * 1024U * 1024U;
// A 512-chunk horizon spans roughly 71 native hydrology owners. Retaining
// that immutable working set prevents far terrain from repeatedly decoding
// the same persisted pages while building its 3,000-plus parent tiles. This
// is a strict LRU budget, not an allocation made at startup.
inline constexpr size_t NATIVE_HYDROLOGY_VISIBLE_HORIZON_CACHE_BYTE_BUDGET = 1'536U * 1024U * 1024U;
inline constexpr size_t NATIVE_HYDROLOGY_MAX_PAGE_BYTES = 32U * 1024U * 1024U;
inline constexpr size_t NATIVE_HYDROLOGY_MAX_BUILD_BYTES = 64U * 1024U * 1024U;
// Quantized ordinary-stage tiles are derived from immutable page authority and
// retained in a separate per-page LRU. Keeping this cache bounded prevents a
// sparse river scan from silently expanding the persisted-page working set.
inline constexpr size_t NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET = 1U * 1024U * 1024U;
// A 2,048-block route owns substantial temporary Priority-Flood state. Keep
// enough independent pages in flight to use the available Apple Silicon CPU
// cores while bounding the worst-case scratch reservation to one GiB.
inline constexpr size_t NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS = 16;
inline constexpr size_t NATIVE_HYDROLOGY_PARALLEL_BUILD_MEMORY_BUDGET = 1U * 1024U * 1024U * 1024U;
static_assert(NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS * NATIVE_HYDROLOGY_MAX_BUILD_BYTES <=
              NATIVE_HYDROLOGY_PARALLEL_BUILD_MEMORY_BUDGET);
inline constexpr int NATIVE_HYDROLOGY_HANDOFF_BLOCKS = 8;
inline constexpr size_t NATIVE_HYDROLOGY_MAX_HANDOFF_PAGES = 6;
// One cold exact footprint currently contains 40,609 distinct columns. Keep
// dry proof and installation batches independently bounded above that finite
// contract so a caller cannot turn a page-local proof into an unbounded
// allocation or traversal.
inline constexpr size_t NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES = 65'536U;
// A spill-summary traversal is compact, but resolving an unbounded component
// would make one query an accidental whole-world walk. Sixty-four owner pages
// cover 131.072 kilometers along a one-page-wide chain. Reaching the bound
// fails closed instead of publishing a partial identity or stage.
inline constexpr size_t NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES = 64;
inline constexpr size_t NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_NODES = 256;
inline constexpr uint16_t NATIVE_HYDROLOGY_PAYLOAD_SCHEMA_VERSION = 6;

namespace native_hydrology_detail {

// Returns the clipped source-to-receiver coordinate used for ordinary stage
// and backwater reconstruction. This stays monotone across a bowed channel's
// complete ribbon even where nearest-centerline arclength has a medial-axis
// discontinuity. Non-finite or degenerate geometry returns zero.
[[nodiscard]] double receiverAxisProgress(double worldX, double worldZ, double sourceX,
                                          double sourceZ, double receiverX,
                                          double receiverZ) noexcept;

// An explicit fall owns the published surface across its complete transition
// footprint, even when the same column is categorically part of a standing
// lake or ocean. Its receiving stage must therefore override that standing
// body's otherwise selected surface.
[[nodiscard]] double explicitFallPublishedWaterSurface(bool waterfall, double waterfallBottom,
                                                       double ordinaryWaterSurface) noexcept;

} // namespace native_hydrology_detail

// The ocean is one global body for an immutable world identity. Final learned
// pages may refine the preview shoreline while retaining this preview-stable
// ownership value.
WaterBodyId nativeOceanWaterBodyId(uint64_t worldSeed) noexcept;

// Returns the lexicographically ordered learned pages covering one native
// hydrology owner page and its two-cell raster apron. Startup can enqueue this
// finite set before routing begins without asking hierarchy reconciliation to
// perform learned queries.
std::vector<learned::TerrainPageCoordinate>
nativeHydrologyRequiredAuthorityPages(int64_t ownerPageX, int64_t ownerPageZ);

// Returns the exact model-native FINAL rectangle consumed while constructing
// one 2,048-block owner, including the canonical two-cell routing apron. The
// half-open 517 by 517 result is suitable for the transient inference lane and
// is overflow-checked for signed owner coordinates.
learned::NativeRect nativeHydrologyFinalTerrainRegion(int64_t ownerPageX, int64_t ownerPageZ);

// Plans the page-wide FINAL topology and bounded FINAL terrain refinement
// required by an exact block rectangle. Coordinates are half-open, and both
// result vectors are lexicographically ordered and deduplicated. Callers
// coalesce overlapping coordinates before submitting them because an
// owner-page topology closure normally contains the exact refinement pages.
struct NativeHydrologyAuthorityRequirements {
    std::vector<learned::TerrainPageCoordinate> finalTopologyPages;
    std::vector<learned::TerrainPageCoordinate> finalRefinementPages;

    // Counts unique FINAL requests, not overlapping logical dependencies.
    [[nodiscard]] size_t totalPageCount() const noexcept {
        size_t topology = 0;
        size_t refinement = 0;
        size_t total = 0;
        while (topology < finalTopologyPages.size() && refinement < finalRefinementPages.size()) {
            const learned::TerrainPageCoordinate& topologyCoordinate = finalTopologyPages[topology];
            const learned::TerrainPageCoordinate& refinementCoordinate =
                finalRefinementPages[refinement];
            if (topologyCoordinate < refinementCoordinate) {
                ++topology;
            } else if (refinementCoordinate < topologyCoordinate) {
                ++refinement;
            } else {
                ++topology;
                ++refinement;
            }
            ++total;
        }
        return total + (finalTopologyPages.size() - topology) +
               (finalRefinementPages.size() - refinement);
    }
};

NativeHydrologyAuthorityRequirements nativeHydrologyAuthorityRequirementsForWorldRect(
    int64_t minimumX, int64_t minimumZ, int64_t maximumXExclusive, int64_t maximumZExclusive);

struct NativeHydrologyPosition {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const NativeHydrologyPosition&) const = default;
};

// Opaque evidence that every integer position in one bounded footprint passed
// the router's immutable page-local dry proof. Only the issuing router can
// construct or install this evidence. Macro generation may inspect the raw
// samples to apply its final learned-terrain adaptation before installation.
class NativeHydrologyDryFootprintCertificate {
public:
    [[nodiscard]] std::span<const NativeHydrologyPosition> positions() const noexcept {
        return positions_;
    }
    [[nodiscard]] std::span<const BasinSample> samples() const noexcept { return samples_; }
    [[nodiscard]] size_t size() const noexcept { return positions_.size(); }

private:
    friend class NativeHydrologyRouter;

    const void* issuer_ = nullptr;
    std::vector<NativeHydrologyPosition> positions_;
    std::vector<BasinSample> samples_;
};

struct NativeHydrologyInput {
    // Model-native physical elevation in meters relative to mean sea level.
    // The router must not receive a block-height conversion here: Priority-
    // Flood, lake classification, and D-infinity all retain this resolution.
    double elevationMeters = 0.0;
    BasinHydroclimateSample climate;
};

using NativeHydrologyInputFunction =
    std::function<void(std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>)>;

// Conservative topology evidence for one half-open 32-by-32 block cell.
// `waterTopologyPossible` is true only where a hidden shore, lake, routed
// channel, or waterfall can affect the cell. It intentionally remains false
// for a uniformly standing open-water cell so the far renderer does not turn
// the entire horizon into a dense raster.
struct NativeHydrologyTopologyCell {
    bool waterTopologyPossible = false;
    bool waterfallPossible = false;
};

struct NativeHydrologyCacheMetrics {
    size_t entries = 0;
    size_t bytes = 0;
    size_t peakBuildBytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t builds = 0;
    // A missing learned rectangle is normal single-flight backpressure. It
    // parks the caller for a later retry and is not a failed hydrology build.
    uint64_t deferredBuilds = 0;
    uint64_t failures = 0;
    uint64_t persistedLoads = 0;
    uint64_t persistedWrites = 0;
    uint64_t persistedRepairs = 0;
    size_t lastPersistedPayloadBytes = 0;
    uint64_t lastWarmLoadNanoseconds = 0;
    // A grid can contain many samples owned by the same immutable page. These
    // are reused within one query without taking the shared cache lock again.
    uint64_t batchPageReuses = 0;
    // Number of exact block columns admitted to strict page-local dry proofs
    // after same-owner and handoff validation. Startup diagnostics use this to
    // catch accidental owner-wide rescans when the proposed chunk's complete
    // exact footprint already proves dry.
    uint64_t dryCertificateSamples = 0;
    size_t activeBuilds = 0;
    size_t peakConcurrentBuilds = 0;
    uint64_t buildAdmissionWaits = 0;
    size_t reconciliationEntries = 0;
    size_t reconciliationBytes = 0;
    uint64_t reconciliationHits = 0;
    uint64_t reconciliationMisses = 0;
    size_t connectedWetlandEntries = 0;
    size_t seaBackwaterEntries = 0;
    size_t openDepressionEntries = 0;
    size_t openDepressionBytes = 0;
    uint64_t openDepressionHits = 0;
    uint64_t openDepressionMisses = 0;
    uint64_t openDepressionBuilds = 0;
    uint64_t openDepressionFailures = 0;
    size_t ordinaryStageTileEntries = 0;
    size_t ordinaryStageTileBytes = 0;
    size_t ordinaryStageTilePeakPageBytes = 0;
    uint64_t ordinaryStageTileHits = 0;
    uint64_t ordinaryStageTileMisses = 0;
    uint64_t ordinaryStageTileBuilds = 0;
    uint64_t ordinaryStageTileFailures = 0;
    uint64_t ordinaryStageTileBuildNanoseconds = 0;
    uint64_t ordinaryStageTileExpandedBuilds = 0;
    uint64_t ordinaryStageTileBuildWaits = 0;
    // Samples intentionally kept on raw routed stage/topology because their
    // grid spacing is coarse or they came from an explicit coarse point
    // batch. This distinguishes the LOD bypass from an exact query that
    // happened not to intersect a river.
    uint64_t ordinaryStageCoarseGridSamples = 0;
};

// Quality-independent coarse anchors preserve preview and final identity for
// the ordinary one-component case. The router reconciles proven opposing edge
// portals through a deterministic tiled hierarchy derived from immutable page
// summaries. The traversal fails closed at its explicit bound instead of
// publishing a partial global identity.
// Context clones share one immutable registry even though each quality owns a
// separate page cache and persistence namespace.
class NativeHydrologyIdentityRegistry {
public:
    explicit NativeHydrologyIdentityRegistry(uint64_t worldSeed) noexcept;

    // Uses a 256-block coarse anchor for one unambiguous local component.
    // This is stable when a preview or final refinement shifts a single basin
    // within that anchor.
    [[nodiscard]] WaterBodyId localLakeBodyId(int64_t rootX, int64_t rootZ) const noexcept;
    // Uses the exact local component anchor only when two disconnected
    // components collide in the same coarse anchor. It prevents an alias but
    // does not claim cross-quality or cross-page topology reconciliation.
    [[nodiscard]] WaterBodyId disambiguatedLocalLakeBodyId(int64_t rootX,
                                                           int64_t rootZ) const noexcept;
    [[nodiscard]] uint64_t seed() const noexcept;

private:
    uint64_t seed_ = 0;
};

// Compact page-owned inputs to the deterministic tiled spill reducer. These
// are also used by diagnostics and synthetic qualification tests, so the
// hierarchy can be verified without constructing a 2,048-block raster for
// every page in a long basin.
struct NativeHydrologySpillNodeSummary {
    int64_t pageX = 0;
    int64_t pageZ = 0;
    WaterBodyId localBodyId = NO_WATER_BODY;
    int64_t localAnchorX = 0;
    int64_t localAnchorZ = 0;
    double localStage = 0.0;
    double coreAreaSquareKilometers = 0.0;
    double coreVolumeCubicMeters = 0.0;
    double coreRunoffMmSquareKilometers = 0.0;
    BasinOutlet naturalOutlet = BasinOutlet::NONE;
    int64_t naturalOutletX = 0;
    int64_t naturalOutletZ = 0;
    double naturalOutletStage = 0.0;

    auto operator<=>(const NativeHydrologySpillNodeSummary&) const = default;
};

struct NativeHydrologySpillPortalSummary {
    int64_t firstPageX = 0;
    int64_t firstPageZ = 0;
    WaterBodyId firstLocalBodyId = NO_WATER_BODY;
    int64_t secondPageX = 0;
    int64_t secondPageZ = 0;
    WaterBodyId secondLocalBodyId = NO_WATER_BODY;
    double minimumWetStage = 0.0;
    double compatibleStage = 0.0;
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const NativeHydrologySpillPortalSummary&) const = default;
};

struct NativeHydrologySpillResolution {
    WaterBodyId canonicalBodyId = NO_WATER_BODY;
    double stage = 0.0;
    double areaSquareKilometers = 0.0;
    double volumeCubicMeters = 0.0;
    double runoffMmSquareKilometers = 0.0;
    BasinOutlet outlet = BasinOutlet::NONE;
    int64_t outletX = 0;
    int64_t outletZ = 0;
    size_t pageCount = 0;

    auto operator<=>(const NativeHydrologySpillResolution&) const = default;
};

std::optional<NativeHydrologySpillResolution>
resolveNativeHydrologySpillSummaries(int64_t sourcePageX, int64_t sourcePageZ,
                                     WaterBodyId sourceLocalBodyId,
                                     std::span<const NativeHydrologySpillNodeSummary> nodes,
                                     std::span<const NativeHydrologySpillPortalSummary> portals);

// Routes learned elevation on one globally aligned 2,048-block page at the
// model's native four-block spacing. The compact page is separate from the
// legacy 16-block BasinSolver and never changes its constants or cache keys.
class NativeHydrologyRouter {
public:
    explicit NativeHydrologyRouter(uint64_t worldSeed,
                                   size_t cacheByteBudget = NATIVE_HYDROLOGY_CACHE_BYTE_BUDGET);
    NativeHydrologyRouter(uint64_t worldSeed,
                          std::shared_ptr<hydrology::HydrologyAuthorityStore> authorityStore,
                          size_t cacheByteBudget = NATIVE_HYDROLOGY_CACHE_BYTE_BUDGET);
    NativeHydrologyRouter(uint64_t worldSeed,
                          std::shared_ptr<hydrology::HydrologyAuthorityStore> authorityStore,
                          std::shared_ptr<const NativeHydrologyIdentityRegistry> identityRegistry,
                          size_t cacheByteBudget = NATIVE_HYDROLOGY_CACHE_BYTE_BUDGET);
    ~NativeHydrologyRouter();

    NativeHydrologyRouter(const NativeHydrologyRouter&) = delete;
    NativeHydrologyRouter& operator=(const NativeHydrologyRouter&) = delete;
    NativeHydrologyRouter(NativeHydrologyRouter&&) noexcept;
    NativeHydrologyRouter& operator=(NativeHydrologyRouter&&) noexcept;

    BasinSample sample(double x, double z, const NativeHydrologyInputFunction& input,
                       bool* certifiedDryHit = nullptr,
                       learned::AuthorityRequestPriority priority =
                           learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    // Builds or loads exactly one immutable owner page. This does not perform
    // handoff, open-depression, wetland, or sea-backwater reconciliation and
    // therefore is not evidence that the owner's canonical semantics have
    // been prepared for arbitrary queries.
    void prepareOwner(int64_t ownerPageX, int64_t ownerPageZ,
                      const NativeHydrologyInputFunction& input,
                      learned::AuthorityRequestPriority priority =
                          learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    // Attempts a conservative dry-locality proof from one immutable owner.
    // Certified samples are canonical for surface-water absence without
    // opening neighboring pages. Uncertified output samples remain default
    // initialized and callers must use the ordinary sampling path for them.
    // Every point in the batch must share one owner and remain strictly more
    // than the handoff distance from its edge; otherwise every mask byte is
    // zero and no page is prepared.
    void certifyDryPoints(std::span<const BasinSamplePosition> positions,
                          const NativeHydrologyInputFunction& input, std::span<BasinSample> output,
                          std::span<uint8_t> certified,
                          learned::AuthorityRequestPriority priority =
                              learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    // Produces opaque all-or-nothing evidence for an integer-coordinate
    // footprint. A partial proof returns nullopt and leaves any installed
    // footprint unchanged.
    [[nodiscard]] std::optional<NativeHydrologyDryFootprintCertificate>
    certifyDryFootprint(std::span<const BasinSamplePosition> positions,
                        const NativeHydrologyInputFunction& input,
                        learned::AuthorityRequestPriority priority =
                            learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    // Atomically replaces the one installed startup footprint. The evidence
    // must have been issued by this router and remains bounded by the dry
    // certificate limit.
    [[nodiscard]] bool
    replaceCertifiedDryFootprint(const NativeHydrologyDryFootprintCertificate& certificate) const;
    [[nodiscard]] bool
    certifiedDryFootprintContains(std::span<const BasinSamplePosition> positions) const;
    void clearCertifiedDryFootprint() const;
    void sampleGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth,
                    int sampleHeight, const NativeHydrologyInputFunction& input,
                    std::span<BasinSample> output, std::span<uint8_t> certifiedDryHits = {},
                    learned::AuthorityRequestPriority priority =
                        learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    void samplePoints(std::span<const BasinSamplePosition> positions,
                      const NativeHydrologyInputFunction& input, std::span<BasinSample> output,
                      std::span<uint8_t> certifiedDryHits = {},
                      learned::AuthorityRequestPriority priority =
                          learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;
    // Samples arbitrary positions from their immutable native pages without
    // constructing block-exact ordinary-stage reconciliation tiles. Coarse
    // preview consumers retain routed wet/dry topology while batching every
    // requested root through one page-retaining query.
    void sampleCoarsePoints(std::span<const BasinSamplePosition> positions,
                            const NativeHydrologyInputFunction& input,
                            std::span<BasinSample> output, std::span<uint8_t> certifiedDryHits = {},
                            learned::AuthorityRequestPriority priority =
                                learned::AuthorityRequestPriority::COARSE_PREVIEW) const;
    // Samples a regular grid of globally aligned 32-block half-open cells.
    // The result is a compact reduction of already routed native-page state,
    // not a water raster or a scalar channel-projection pass.
    void sampleTopologyGrid(int64_t originX, int64_t originZ, int cellWidth, int cellHeight,
                            const NativeHydrologyInputFunction& input,
                            std::span<NativeHydrologyTopologyCell> output,
                            std::span<uint8_t> certifiedDryHits = {},
                            learned::AuthorityRequestPriority priority =
                                learned::AuthorityRequestPriority::EXPLORATION_EXACT) const;

    [[nodiscard]] NativeHydrologyCacheMetrics cacheMetrics() const;
    void clear() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace worldgen

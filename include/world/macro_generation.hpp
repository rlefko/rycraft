#pragma once

#include "common/counter_rng.hpp"
#include "world/alpine_morphology.hpp"
#include "world/basin_solver.hpp"
#include "world/chunk.hpp"
#include "world/noise.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace worldgen {

struct Vector2d {
    double x = 0.0;
    double z = 0.0;
};

inline constexpr int64_t HOTSPOT_LATTICE_EDGE = 16'384;

struct HotspotChainPrimitive {
    Vector2d sourcePlateVelocity;
    Vector2d direction;
    double sourceX = 0.0;
    double sourceZ = 0.0;
    double length = 0.0;
    bool active = false;
};

enum class CrustType : uint8_t {
    OCEANIC,
    CONTINENTAL,
};

enum class PlateBoundary : uint8_t {
    NONE,
    CONVERGENT,
    DIVERGENT,
    TRANSFORM,
};

enum class RockType : uint8_t {
    GRANITE,
    BASALT,
    LIMESTONE,
    SANDSTONE,
    VOLCANIC,
};

// The horizontal support represented by a surface sample. Exact cubic
// emission uses BLOCK_1, while far terrain requests progressively wider
// footprints so detail below its Nyquist limit is removed instead of
// aliasing into a different large-scale shape.
enum class SurfaceFootprint : uint8_t {
    BLOCK_1 = 1,
    BLOCK_2 = 2,
    BLOCK_4 = 4,
    BLOCK_8 = 8,
    BLOCK_16 = 16,
    BLOCK_32 = 32,
};

constexpr int surfaceFootprintWidth(SurfaceFootprint footprint) noexcept {
    return static_cast<int>(footprint);
}

// Exact columns and every filtered far footprint use the same post-erosion
// detail scales. Keeping these shared prevents the handoff from changing
// terrain amplitude while preserving extra dry-land texture and a supported
// sea-floor cap below source water.
inline constexpr double DRY_RELIEF_DETAIL_SCALE = 1.5;
inline constexpr double OCEAN_FLOOR_DETAIL_SCALE = 0.25;

inline constexpr int MACRO_CONTROL_TILE_EDGE = 64;
inline constexpr int MACRO_CONTROL_SPACING = 8;
inline constexpr int MACRO_CONTROL_CORE_EDGE = MACRO_CONTROL_TILE_EDGE / MACRO_CONTROL_SPACING + 1;
inline constexpr int MACRO_CONTROL_GRID_EDGE = MACRO_CONTROL_CORE_EDGE + 2;
inline constexpr size_t MACRO_CONTROL_SAMPLE_COUNT =
    MACRO_CONTROL_GRID_EDGE * MACRO_CONTROL_GRID_EDGE;
inline constexpr size_t MACRO_CONTROL_CACHE_CAPACITY = 1'024;
inline constexpr size_t MACRO_CONTROL_CACHE_BYTE_BUDGET = 128ull * 1024 * 1024;

// Climate varies at substantially longer wavelengths than the two-block far
// tier. A small, globally aligned control lattice lets every far footprint
// share a filtered field without repeating the bounded 4,096-block moisture
// integration at every mesh vertex.
inline constexpr int FAR_CLIMATE_CONTROL_TILE_EDGE = 256;
inline constexpr int FAR_CLIMATE_CONTROL_SPACING = 128;
inline constexpr int FAR_CLIMATE_CONTROL_CORE_EDGE =
    FAR_CLIMATE_CONTROL_TILE_EDGE / FAR_CLIMATE_CONTROL_SPACING + 1;
inline constexpr int FAR_CLIMATE_CONTROL_GRID_EDGE = FAR_CLIMATE_CONTROL_CORE_EDGE + 2;
inline constexpr size_t FAR_CLIMATE_CONTROL_SAMPLE_COUNT =
    FAR_CLIMATE_CONTROL_GRID_EDGE * FAR_CLIMATE_CONTROL_GRID_EDGE;
inline constexpr size_t FAR_CLIMATE_CONTROL_CACHE_CAPACITY = 1'024;
inline constexpr size_t FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET = 8ull * 1024 * 1024;

struct MacroControlCacheMetrics {
    size_t entries = 0;
    size_t bytes = 0;
    size_t capacity = 0;
    size_t byteBudget = 0;
    size_t activeBuilds = 0;
    size_t peakBuilds = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t builds = 0;
    uint64_t evictions = 0;
    uint64_t singleFlightWaits = 0;
};

struct LithologyBlend {
    RockType primary = RockType::GRANITE;
    RockType secondary = RockType::GRANITE;
    // Normalized secondary contribution. The primary remains dominant, so
    // this value is always in [0, 0.5].
    double transition = 0.0;
    // Positive distance into the primary facies. Zero is the implicit rock
    // contact reconstructed from the warped plate sites.
    double contactDistance = 0.0;
};

struct GeologySample {
    uint64_t plateId = 0;
    CrustType crust = CrustType::CONTINENTAL;
    PlateBoundary boundary = PlateBoundary::NONE;
    RockType rock = RockType::GRANITE;
    LithologyBlend lithology;
    // Continuous multi-facies resistance used by erosion and alpine relief.
    // Unlike the diagnostic primary and secondary rock IDs, this scalar does
    // not jump where two equally distant secondary sites exchange rank.
    double erosionResistance = 0.92;
    Vector2d plateVelocity;
    double continentalFraction = 0.0;
    double crustAge = 0.0;
    double crustThickness = 0.0;
    double crustDensity = 0.0;
    double distanceToBoundary = 0.0;
    double uplift = 0.0;
    double rift = 0.0;
    double faultStrength = 0.0;
    double hotspotInfluence = 0.0;
    double volcanicActivity = 0.0;
};

struct HydrologySample {
    WaterBodyId waterBodyId = NO_WATER_BODY;
    uint8_t generatedFluidLevel = 0;
    WaterTransitionKind transitionOwnerKind = WaterTransitionKind::NONE;
    uint64_t transitionOwnerId = 0;
    Vector2d flowDirection;
    double surfaceElevation = 0.0;
    double waterSurface = 0.0;
    double discharge = 0.0;
    double sediment = 0.0;
    double channelDistance = 0.0;
    double channelWidth = 0.0;
    double channelDepth = 0.0;
    double channelGradient = 0.0;
    double erosionDepth = 0.0;
    double lakeDepth = 0.0;
    double lakeShoreDistance = -1.0e9;
    double shoreWaterSurface = 0.0;
    double lakeBankTarget = 0.0;
    double lakeBankInfluence = 0.0;
    double lakeAreaSquareKilometers = 0.0;
    double lakeVolumeCubicMeters = 0.0;
    double lakeRunoffMmSquareKilometers = 0.0;
    double lakeLossMm = 0.0;
    double lakeOverflowMmSquareKilometers = 0.0;
    double lakeSpillSurface = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    uint8_t streamOrder = 0;
    uint8_t distributaryCount = 0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool lakeBank = false;
    bool channelBank = false;
    bool endorheic = false;
    bool waterfall = false;
    bool waterfallAnchor = false;
    bool delta = false;
};

struct ClimateFields {
    Vector2d wind;
    double temperatureC = 0.0;
    double annualPrecipitationMm = 0.0;
    double potentialEvapotranspirationMm = 0.0;
    double aridity = 0.0;
    double relativeHumidity = 0.0;
};

struct SoilSample {
    double moisture = 0.0;
    double fertility = 0.0;
    double drainage = 0.0;
    double waterTable = 0.0;
};

inline constexpr size_t MACRO_BIOME_COUNT = static_cast<size_t>(Biome::COUNT);

struct BiomeSuitability {
    // Suitability is a bounded ranking field, not accumulated geometry. Single
    // precision keeps the nine-sample ColumnPlan compact while retaining far
    // more resolution than biome transitions need.
    std::array<float, MACRO_BIOME_COUNT> scores{};
};

struct BiomeBlend {
    Biome primary = Biome::PLAINS;
    Biome secondary = Biome::PLAINS;
    double transition = 0.0;
};

// Returns the normalized contribution retained in the public two-biome
// blend. Consumers use this instead of switching directly on the primary ID
// when a continuous density or suitability is available.
double biomeBlendWeight(const BiomeBlend& blend, Biome biome) noexcept;

// Isotropic multiscale rank used to dither discrete materials and species
// across continuous blends. Rotated smooth fields preserve natural patches
// without exposing world-aligned ownership cells. The result is
// coordinate-pure and lies strictly between zero and one.
double multiscaleDitherThreshold(const CounterRng& random, uint64_t stream, int64_t x, int64_t z,
                                 uint32_t index = 0) noexcept;

// Continuous local-water contribution shared by climate recomputation and
// generated crater overlays. River influence extends through nearby banks,
// and lake or ocean influence tapers with water depth.
double climateWaterInfluence(const HydrologySample& hydrology) noexcept;

enum class Ecotope : uint32_t {
    NONE = 0,
    RIVERBANK = 1U << 0,
    FLOODPLAIN = 1U << 1,
    DELTA = 1U << 2,
    LAKESHORE = 1U << 3,
    COAST = 1U << 4,
    CLIFF = 1U << 5,
    SCREE = 1U << 6,
    CANYON = 1U << 7,
    GEOTHERMAL = 1U << 8,
    CAVE = 1U << 9,
    AQUIFER = 1U << 10,
    VALLEY = 1U << 11,
    FOOTHILL = 1U << 12,
    MONTANE = 1U << 13,
    SUBALPINE = 1U << 14,
    ALPINE_ZONE = 1U << 15,
    SNOWFIELD = 1U << 16,
    GLACIER = 1U << 17,
    EXPOSED_PEAK = 1U << 18,
    ALL = (1U << 19) - 1U,
};

constexpr Ecotope operator|(Ecotope lhs, Ecotope rhs) {
    return static_cast<Ecotope>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr Ecotope& operator|=(Ecotope& lhs, Ecotope rhs) {
    lhs = lhs | rhs;
    return lhs;
}

constexpr bool hasEcotope(Ecotope value, Ecotope flag) {
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

struct SurfaceSample {
    GeologySample geology;
    HydrologySample hydrology;
    ClimateFields climate;
    SoilSample soil;
    BiomeSuitability suitability;
    BiomeBlend biome;
    Ecotope ecotopes = Ecotope::NONE;
    double terrainHeight = 0.0;
    double waterSurface = 0.0;
    double slope = 0.0;
};

// Coordinate-pure finished fluid state for one generated surface column.
// Standing water owns an implicit source top. Routed channels quantize their
// exposed analytical cross section to Java-style levels, while full covered
// voxels remain sources and a waterfall begins above its receiving top cell.
struct GeneratedFluidColumn {
    FluidState topState = FluidState::source();
    int topY = WORLD_MIN_Y;
    int fallingStartY = WORLD_MAX_Y + 1;
    double visibleSurface = 0.0;
    bool wet = false;
    bool standing = false;
};

GeneratedFluidColumn generatedFluidColumn(const SurfaceSample& surface) noexcept;

// Immutable view of one cached C2 macro-control tile. Exact ColumnPlans pin
// this lightweight handle so climate, soil, suitability, slope, and
// continuous geological signals use the same reconstruction as far terrain
// without retaining a callback into the generator.
class MacroControlView {
public:
    MacroControlView() = default;

    explicit operator bool() const noexcept { return impl_ != nullptr; }
    void reconstructContinuous(double x, double z, SurfaceSample& destination) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;

    explicit MacroControlView(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

    friend class MacroGenerationSampler;
};

static_assert(sizeof(MacroControlView) <= 2 * sizeof(void*));

// Coordinate-pure macro terrain fields. All searches and integrations have
// compile-time bounds, so a sample cannot trigger unbounded world traversal.
class MacroGenerationSampler {
public:
    explicit MacroGenerationSampler(
        uint64_t worldSeed, size_t macroControlCacheCapacity = MACRO_CONTROL_CACHE_CAPACITY,
        size_t macroControlCacheByteBudget = MACRO_CONTROL_CACHE_BYTE_BUDGET,
        size_t farClimateControlCacheCapacity = FAR_CLIMATE_CONTROL_CACHE_CAPACITY,
        size_t farClimateControlCacheByteBudget = FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    ~MacroGenerationSampler();

    MacroGenerationSampler(const MacroGenerationSampler&) = delete;
    MacroGenerationSampler& operator=(const MacroGenerationSampler&) = delete;

    HotspotChainPrimitive hotspotChain(int64_t cellX, int64_t cellZ) const;
    GeologySample sampleGeology(double x, double z) const;
    HydrologySample sampleHydrology(double x, double z) const;
    ClimateFields sampleClimate(double x, double z, double terrainHeight) const;
    SoilSample sampleSoil(double x, double z, const GeologySample& geology,
                          const HydrologySample& hydrology, const ClimateFields& climate) const;
    BiomeSuitability biomeSuitability(const GeologySample& geology,
                                      const HydrologySample& hydrology,
                                      const ClimateFields& climate, const SoilSample& soil,
                                      double terrainHeight, double slope) const;
    static BiomeBlend selectBiome(const BiomeSuitability& suitability);
    static double ecotopeInfluence(const SurfaceSample& surface, Ecotope ecotope);
    static Ecotope classifyEcotopes(const SurfaceSample& surface);
    SurfaceSample sampleSurface(double x, double z,
                                SurfaceFootprint footprint = SurfaceFootprint::BLOCK_1) const;
    MacroControlView controlView(ColumnPos chunkColumn) const;
    // Samples a regular global grid in row-major order. One cache lookup is
    // retained across all points that share a 64-block control tile.
    void sampleSurfaceGrid(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                           SurfaceFootprint footprint, std::span<SurfaceSample> output) const;
    void sampleGeometryGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                            int sampleWidth, int sampleHeight, SurfaceFootprint footprint,
                            std::span<SurfaceSample> output) const;
    // Geometry-only counterpart for sparse globally anchored far features and
    // shoreline probes. Basin ownership is solved once for the complete batch.
    void sampleGeometryPoints(std::span<const ColumnPos> positions, SurfaceFootprint footprint,
                              std::span<SurfaceSample> output) const;
    void sampleHydrologyGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                             int sampleWidth, int sampleHeight,
                             std::span<HydrologySample> output) const;
    void sampleHydrologyPoints(std::span<const ColumnPos> positions,
                               std::span<HydrologySample> output) const;
    // Reconstructs final macro fields at arbitrary integer positions while
    // batching their basin lookups. This is the coordinate-pure point
    // counterpart to sampleSurfaceGrid for sparse feature anchors.
    void sampleSurfacePoints(std::span<const ColumnPos> positions, SurfaceFootprint footprint,
                             std::span<SurfaceSample> output) const;
    MacroControlCacheMetrics macroControlCacheMetrics() const;
    MacroControlCacheMetrics farClimateControlCacheMetrics() const;
    void clearMacroControlCache() const;
    BasinCacheMetrics basinCacheMetrics() const { return basinSolver_.cacheMetrics(); }
    void clearBasinCache() const { basinSolver_.clear(); }

    // Relief before channel incision. This is useful for density construction
    // and deterministic diagnostics.
    double preliminaryElevation(double x, double z) const;
    double sampleProvisionalRainfall(double x, double z) const;
    // Coordinate-pure detail band added to dry post-hydrology terrain. Column
    // plans retain the resulting block-resolution height while reconstructing
    // climate and ecology from their shared continuous control tile.
    double reliefDetail(double x, double z, SurfaceFootprint footprint) const;

private:
    friend class MacroControlView;

    struct PlateSite;
    struct DrainageNode;
    struct MacroControlTile;
    struct FarClimateControlTile;
    class MacroControlTileCache;
    class FarClimateControlTileCache;

    PlateSite plateSite(int64_t cellX, int64_t cellZ) const;
    Vector2d plateVelocityAt(double x, double z) const;
    DrainageNode drainageNode(int64_t cellX, int64_t cellZ) const;
    DrainageNode downstreamNode(const DrainageNode& node) const;
    HydrologySample sampleHydrologyFallback(double x, double z) const;
    HydrologySample sampleClimateHydrology(double x, double z) const;
    ClimateFields sampleClimateWithHydrology(double x, double z, double terrainHeight,
                                             bool canonicalHydrology,
                                             const HydrologySample* localHydrology = nullptr) const;
    double provisionalRainfall(double x, double z, double elevation) const;
    double provisionalPotentialEvapotranspiration(double x, double z, double elevation) const;
    double terrainSlope(double x, double z) const;
    double alpineSurfaceDetail(double x, double z, SurfaceFootprint footprint,
                               const GeologySample& geology, const HydrologySample& hydrology,
                               const ClimateFields& climate) const;
    void applyAlpineSurfaceDetail(double x, double z, SurfaceFootprint footprint,
                                  SurfaceSample& surface) const;
    SurfaceSample sampleSurfaceDirect(double x, double z, SurfaceFootprint footprint) const;
    std::shared_ptr<const MacroControlTile> controlTile(int64_t tileX, int64_t tileZ) const;
    SurfaceSample sampleSurfaceFromControlTile(const MacroControlTile& tile, double x, double z,
                                               SurfaceFootprint footprint,
                                               const HydrologySample* hydrology = nullptr) const;
    std::shared_ptr<const FarClimateControlTile> farClimateControlTile(int64_t tileX,
                                                                       int64_t tileZ) const;
    SurfaceSample
    sampleSurfaceFromFarClimateControlTile(const FarClimateControlTile& tile, double x, double z,
                                           SurfaceFootprint footprint,
                                           const HydrologySample* hydrology = nullptr) const;

    uint64_t cacheTag_ = 0;
    CounterRng random_;
    AlpineMorphologySampler alpineMorphology_;
    mutable BasinSolver basinSolver_;
    mutable std::unique_ptr<MacroControlTileCache> macroControlCache_;
    mutable std::unique_ptr<FarClimateControlTileCache> farClimateControlCache_;
    SimplexNoise continentalNoise_;
    SimplexNoise warpXNoise_;
    SimplexNoise warpZNoise_;
    SimplexNoise reliefNoise_;
    SimplexNoise pressureNoise_;
    SimplexNoise insolationNoise_;
    SimplexNoise soilNoise_;
    SimplexNoise climateDetailNoise_;
};

} // namespace worldgen

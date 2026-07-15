#pragma once

#include "common/counter_rng.hpp"
#include "world/basin_solver.hpp"
#include "world/chunk.hpp"
#include "world/noise.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

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

struct GeologySample {
    uint64_t plateId = 0;
    CrustType crust = CrustType::CONTINENTAL;
    PlateBoundary boundary = PlateBoundary::NONE;
    RockType rock = RockType::GRANITE;
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
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    uint8_t streamOrder = 0;
    uint8_t distributaryCount = 0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool lakeBank = false;
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

// Hierarchical base-four rank used to dither discrete materials and species
// across a continuous biome blend. Nearby columns share coarse rank digits,
// while the one-block digit preserves the requested transition proportion.
// The result is coordinate-pure and lies strictly between zero and one.
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

// Coordinate-pure macro terrain fields. All searches and integrations have
// compile-time bounds, so a sample cannot trigger unbounded world traversal.
class MacroGenerationSampler {
public:
    explicit MacroGenerationSampler(uint64_t worldSeed);

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
    SurfaceSample sampleSurface(double x, double z) const;
    BasinCacheMetrics basinCacheMetrics() const { return basinSolver_.cacheMetrics(); }
    void clearBasinCache() const { basinSolver_.clear(); }

    // Relief before channel incision. This is useful for density construction
    // and deterministic diagnostics.
    double preliminaryElevation(double x, double z) const;

private:
    struct PlateSite;
    struct DrainageNode;

    PlateSite plateSite(int64_t cellX, int64_t cellZ) const;
    Vector2d plateVelocityAt(double x, double z) const;
    DrainageNode drainageNode(int64_t cellX, int64_t cellZ) const;
    DrainageNode downstreamNode(const DrainageNode& node) const;
    HydrologySample sampleHydrologyFallback(double x, double z) const;
    double provisionalRainfall(double x, double z, double elevation) const;
    double terrainSlope(double x, double z) const;

    uint64_t cacheTag_ = 0;
    CounterRng random_;
    mutable BasinSolver basinSolver_;
    SimplexNoise continentalNoise_;
    SimplexNoise warpXNoise_;
    SimplexNoise warpZNoise_;
    SimplexNoise reliefNoise_;
    SimplexNoise soilNoise_;
};

} // namespace worldgen

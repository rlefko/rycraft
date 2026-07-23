#include "world/chunk_generator.hpp"
#include "world/learned_terrain.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <queue>
#include <utility>

namespace {

constexpr int LAVA_LEVEL = -96;
constexpr int HOTSPOT_QUERY_RADIUS = 1;
constexpr int64_t VOLCANIC_ARC_CELL_EDGE = 1024;
constexpr int64_t AQUIFER_CELL_EDGE = 64;
constexpr int AQUIFER_CELL_HEIGHT = 32;
constexpr int COHERENT_CRATER_SURFACE_DEPTH = 8;
constexpr size_t HOTSPOT_PRIMITIVE_CACHE_CAPACITY = 1024;
constexpr size_t ARC_PRIMITIVE_CACHE_CAPACITY = 2048;
constexpr int COLUMN_PLAN_CONSTRUCTION_APRON_EDGE = COLUMN_PLAN_LATTICE_EDGE + 2;
constexpr size_t COLUMN_PLAN_CONSTRUCTION_APRON_SAMPLES =
    COLUMN_PLAN_CONSTRUCTION_APRON_EDGE * COLUMN_PLAN_CONSTRUCTION_APRON_EDGE;
constexpr int COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE = COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE;
constexpr size_t COLUMN_PLAN_CONSTRUCTION_AUTHORITY_SAMPLES =
    COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE * COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE;
constexpr int LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE = CHUNK_EDGE + 1;
constexpr size_t LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_SAMPLES =
    LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE * LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE;
constexpr int COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE = CHUNK_EDGE + 1;
constexpr size_t COLUMN_PLAN_CONSTRUCTION_GEOLOGY_SAMPLES =
    COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE * COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE;
constexpr size_t COLUMN_PLAN_CONSTRUCTION_PERIMETER_SAMPLES =
    COLUMN_PLAN_CONSTRUCTION_APRON_SAMPLES - COLUMN_PLAN_LATTICE_SAMPLES;
static_assert(COLUMN_PLAN_CONSTRUCTION_PERIMETER_SAMPLES == 16);
static_assert(feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH <= COLUMN_PLAN_LATTICE_SPACING);

int densityLatticeLevelCountForSpacing(int verticalSpacing) {
    return (WORLD_MAX_Y - WORLD_MIN_Y + 1) / verticalSpacing + 1;
}

int maximumEvaluatedDensityLevel(double yCap, int verticalSpacing) {
    const int latticeLevels = densityLatticeLevelCountForSpacing(verticalSpacing);
    const int maximumLatticeY = WORLD_MIN_Y + (latticeLevels - 1) * verticalSpacing;
    if (std::isnan(yCap) || yCap < WORLD_MIN_Y) return -1;
    if (yCap >= maximumLatticeY) return latticeLevels - 1;
    return static_cast<int>((yCap - WORLD_MIN_Y) / verticalSpacing);
}

constexpr uint64_t HOTSPOT_PROPERTIES_STREAM = 0x1202;
constexpr uint64_t VOLCANIC_ARC_STREAM = 0x564F4C4341524331ULL;
constexpr uint64_t BEDROCK_STREAM = 0x424544524F434B31ULL;
constexpr uint64_t AQUIFER_STREAM = 0x4151554946455231ULL;
constexpr uint64_t STRATA_STREAM = 0x5354524154413031ULL;
constexpr uint64_t CRATER_WATER_BODY_SALT = 0x4352415445524C4BULL;
using ColumnPlanConstructionSurfaceApron =
    std::array<worldgen::SurfaceSample, COLUMN_PLAN_CONSTRUCTION_APRON_SAMPLES>;
using ColumnPlanConstructionHydrology =
    std::array<worldgen::HydrologySample, COLUMN_PLAN_CONSTRUCTION_AUTHORITY_SAMPLES>;
using LegacyColumnPlanConstructionHydrology =
    std::array<worldgen::HydrologySample, LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_SAMPLES>;
using ColumnPlanConstructionGeology =
    std::array<worldgen::GeologySample, COLUMN_PLAN_CONSTRUCTION_GEOLOGY_SAMPLES>;
using ColumnPlanConstructionPerimeter = std::array<double, COLUMN_PLAN_CONSTRUCTION_APRON_SAMPLES>;

// The cache consumes its callbacks synchronously and retains only the finished
// immutable plan. Keep the fixed construction grids behind lazy pointers so a
// cache hit pays neither their roughly 128 KiB transient footprint nor any
// basin work. A cold single-flight producer fills each grid once when the
// corresponding constructor phase first asks for it.
struct ColumnPlanConstructionState {
    std::unique_ptr<ColumnPlanConstructionSurfaceApron> surfaceApron;
    std::unique_ptr<ColumnPlanConstructionHydrology> hydrology;
    std::unique_ptr<ColumnPlanConstructionGeology> geology;
    std::unique_ptr<ColumnPlanConstructionPerimeter> perimeterHeight;
};

uint64_t nextGeneratorInstanceToken() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

struct ThreadVolcanoPrimitiveCache {
    const void* owner = nullptr;
    uint64_t ownerToken = 0;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> hotspotCells;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> arcCells;
};

struct ThreadWaterContactCache {
    const void* owner = nullptr;
    uint64_t ownerToken = 0;
    std::unordered_map<ColumnPos, worldgen::SurfaceSample> surfaces;
};

ThreadVolcanoPrimitiveCache& threadVolcanoPrimitiveCache(const void* owner, uint64_t ownerToken) {
    thread_local ThreadVolcanoPrimitiveCache cache;
    if (cache.owner != owner || cache.ownerToken != ownerToken) {
        cache.owner = owner;
        cache.ownerToken = ownerToken;
        cache.hotspotCells.clear();
        cache.arcCells.clear();
    }
    return cache;
}

ThreadWaterContactCache& threadWaterContactCache(const void* owner, uint64_t ownerToken) {
    thread_local ThreadWaterContactCache cache;
    if (cache.owner != owner || cache.ownerToken != ownerToken || cache.surfaces.size() > 16'384) {
        cache.owner = owner;
        cache.ownerToken = ownerToken;
        cache.surfaces.clear();
    }
    return cache;
}

int64_t latticeFloor(int64_t value) {
    return world_coord::floorDiv(value, static_cast<int64_t>(LATTICE_XZ)) * LATTICE_XZ;
}

bool hasSurfaceWater(const worldgen::HydrologySample& hydrology) {
    return hydrology.ocean || hydrology.river || hydrology.lake || hydrology.wetland;
}

bool requiresLegacyExactShoreSupport(const worldgen::HydrologySample& hydrology) {
    constexpr double SHORE_SUPPORT_REACH = 56.0;
    return !hasSurfaceWater(hydrology) && !hydrology.channelBank && !hydrology.lakeBank &&
           hydrology.shoreWaterSurface > SEA_LEVEL && hydrology.lakeShoreDistance <= 0.0 &&
           hydrology.lakeShoreDistance > -SHORE_SUPPORT_REACH &&
           hydrology.surfaceElevation + 2.5 >= hydrology.shoreWaterSurface;
}

bool hasCanonicalWaterFeature(const worldgen::HydrologySample& hydrology) {
    return hasSurfaceWater(hydrology) || hydrology.wetland || hydrology.delta ||
           hydrology.waterfall ||
           hydrology.transitionOwnerKind != worldgen::WaterTransitionKind::NONE ||
           hydrology.transitionOwnerId != 0;
}

double canonicalColumnPlanContactDistance(double distance) {
    constexpr double SCALE = 8.0;
    const double scaled = std::round(distance * SCALE);
    const auto encoded = static_cast<int16_t>(
        std::clamp(scaled, static_cast<double>(std::numeric_limits<int16_t>::min()),
                   static_cast<double>(std::numeric_limits<int16_t>::max())));
    return static_cast<double>(encoded) / SCALE;
}

double canonicalColumnPlanLithologyTransition(double transition) {
    constexpr double SCALE = 131070.0;
    const auto encoded =
        static_cast<uint16_t>(std::round(std::clamp(transition, 0.0, 0.5) * SCALE));
    return static_cast<double>(encoded) / SCALE;
}

double canonicalColumnPlanShoreDistance(double distance) {
    if (!std::isfinite(distance) || distance <= -4095.0) return -1.0e9;
    const auto encoded = static_cast<int16_t>(std::clamp(
        std::lround(distance * 8.0), static_cast<long>(std::numeric_limits<int16_t>::min() + 1),
        static_cast<long>(std::numeric_limits<int16_t>::max())));
    return static_cast<double>(encoded) / 8.0;
}

worldgen::Vector2d canonicalColumnPlanFlow(worldgen::Vector2d flow) {
    const double magnitude = std::hypot(flow.x, flow.z);
    if (magnitude < 1.0e-12) {
        flow = {1.0, 0.0};
    } else {
        flow.x /= magnitude;
        flow.z /= magnitude;
    }
    const auto canonicalComponent = [](double component) {
        constexpr double COMPONENT_SCALE = static_cast<double>(std::numeric_limits<int16_t>::max());
        const auto encoded =
            static_cast<int16_t>(std::lround(std::clamp(component, -1.0, 1.0) * COMPONENT_SCALE));
        return static_cast<double>(encoded) / COMPONENT_SCALE;
    };
    return {canonicalComponent(flow.x), canonicalComponent(flow.z)};
}

void canonicalizeColumnPlanGeology(worldgen::GeologySample& geology) {
    geology.lithology.transition =
        canonicalColumnPlanLithologyTransition(geology.lithology.transition);
    geology.lithology.contactDistance =
        canonicalColumnPlanContactDistance(geology.lithology.contactDistance);
}

void canonicalizeColumnPlanHydrology(worldgen::HydrologySample& hydrology) {
    hydrology.surfaceElevation = static_cast<float>(hydrology.surfaceElevation);
    hydrology.waterSurface = static_cast<float>(hydrology.waterSurface);
    hydrology.lakeShoreDistance = canonicalColumnPlanShoreDistance(hydrology.lakeShoreDistance);
    hydrology.lakeBankInfluence = static_cast<double>(std::lround(
                                      std::clamp(hydrology.lakeBankInfluence, 0.0, 1.0) * 255.0)) /
                                  255.0;
    if (hydrology.waterfall) {
        hydrology.flowDirection = canonicalColumnPlanFlow(hydrology.flowDirection);
        hydrology.waterfallTop = static_cast<float>(hydrology.waterfallTop);
        hydrology.waterfallBottom = static_cast<float>(hydrology.waterfallBottom);
        hydrology.waterfallWidth = static_cast<float>(hydrology.waterfallWidth);
    }
}

double smoothstep(double edge0, double edge1, double value) {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    const double amount = std::clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
    return amount * amount * (3.0 - 2.0 * amount);
}

double smootherstep(double edge0, double edge1, double value) {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    const double amount = std::clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
    return amount * amount * amount * (amount * (amount * 6.0 - 15.0) + 10.0);
}

double volcanoProfile(const VolcanoPrimitive& volcano, double radialDistance) {
    const double normalizedRadius = radialDistance / std::max(1.0, volcano.radius);
    return volcano.shield ? std::pow(std::max(0.0, 1.0 - normalizedRadius * normalizedRadius), 2.0)
                          : std::pow(std::max(0.0, 1.0 - normalizedRadius), 1.28);
}

double craterFactor(const VolcanoPrimitive& volcano, double radialDistance) {
    return radialDistance < volcano.craterRadius
               ? 1.0 - smoothstep(0.12, 1.0, radialDistance / volcano.craterRadius)
               : 0.0;
}

double craterProfileHeight(const VolcanoPrimitive& volcano, double radialDistance) {
    return volcano.craterDatumElevation +
           volcano.coneHeight * volcanoProfile(volcano, radialDistance) -
           volcano.craterDepth * craterFactor(volcano, radialDistance);
}

double craterWarpedDistance(const VolcanoPrimitive& volcano, double offsetX, double offsetZ) {
    const double radialDistance = std::hypot(offsetX, offsetZ);
    if (radialDistance < 1.0e-9) return 0.0;

    const double angle = std::atan2(offsetZ, offsetX);
    const double idPhase =
        static_cast<double>((volcano.id >> 17U) & 0xFFFFU) / 65535.0 * 2.0 * std::numbers::pi;
    const double radialPhase = radialDistance / std::max(24.0, volcano.craterRadius * 0.72);
    const double scale =
        std::clamp(1.0 + std::sin(angle * 3.0 + volcano.tubePhase) * 0.052 +
                       std::sin(angle * 5.0 + idPhase) * 0.027 +
                       std::sin(angle * 2.0 + radialPhase + idPhase * 0.61) * 0.018,
                   0.88, 1.12);
    return radialDistance / scale;
}

void configureCraterLake(VolcanoPrimitive& volcano) {
    volcano.craterLakeRadius = 0.0;
    volcano.craterLakeSurface = 0.0;
    volcano.craterRimElevation = 0.0;
    volcano.craterRimWidth = std::clamp(volcano.craterRadius * 0.22, 12.0, 24.0);
    if (!volcano.craterLake || volcano.craterRadius <= 2.0 || volcano.craterDepth <= 2.0) {
        volcano.craterLake = false;
        return;
    }

    // The rim is an absolute local volcanic profile rather than an additive
    // depression draped over arbitrary macro relief. Sampling the complete
    // analytical ring makes the enclosure invariant explicit and leaves a
    // full block of freeboard before a lake is accepted.
    constexpr int RIM_VALIDATION_SAMPLES = 96;
    double validatedRim = std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < RIM_VALIDATION_SAMPLES; ++sample) {
        const double angle =
            static_cast<double>(sample) / RIM_VALIDATION_SAMPLES * 2.0 * std::numbers::pi;
        double innerRadius = volcano.craterRadius * 0.75;
        double outerRadius = volcano.craterRadius * 1.25;
        for (int iteration = 0; iteration < 32; ++iteration) {
            const double radius = (innerRadius + outerRadius) * 0.5;
            const double warped =
                craterWarpedDistance(volcano, std::cos(angle) * radius, std::sin(angle) * radius);
            if (warped < volcano.craterRadius) {
                innerRadius = radius;
            } else {
                outerRadius = radius;
            }
        }
        const double ringRadius = (innerRadius + outerRadius) * 0.5;
        const double ringHeight =
            craterProfileHeight(volcano, craterWarpedDistance(volcano, std::cos(angle) * ringRadius,
                                                              std::sin(angle) * ringRadius));
        validatedRim = std::min(validatedRim, ringHeight);
    }
    const double desiredLakeSurface =
        volcano.craterDatumElevation + volcano.coneHeight - volcano.craterDepth * 0.42;
    const double centerFloor = craterProfileHeight(volcano, 0.0);
    volcano.craterRimElevation = validatedRim;
    volcano.craterLakeSurface = std::min(desiredLakeSurface, validatedRim - 1.0);
    if (!std::isfinite(volcano.craterLakeSurface) || volcano.craterLakeSurface <= SEA_LEVEL + 1.0 ||
        volcano.craterLakeSurface <= centerFloor + 1.5) {
        volcano.craterLake = false;
        volcano.craterLakeSurface = 0.0;
        return;
    }

    double wetRadius = 0.0;
    double dryRadius = volcano.craterRadius;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double midpoint = (wetRadius + dryRadius) * 0.5;
        if (craterProfileHeight(volcano, midpoint) < volcano.craterLakeSurface) {
            wetRadius = midpoint;
        } else {
            dryRadius = midpoint;
        }
    }
    volcano.craterLakeRadius = wetRadius;
    if (volcano.craterLakeRadius <= 2.0 || volcano.craterLakeRadius >= volcano.craterRadius - 1.0) {
        volcano.craterLake = false;
        volcano.craterLakeRadius = 0.0;
        volcano.craterLakeSurface = 0.0;
    }
}

double localWaterClimateInfluence(const worldgen::HydrologySample& hydrology) {
    return worldgen::climateWaterInfluence(hydrology);
}

void clearSurfaceWater(worldgen::HydrologySample& hydrology) {
    hydrology.ocean = false;
    hydrology.river = false;
    hydrology.lake = false;
    hydrology.wetland = false;
    hydrology.endorheic = false;
    hydrology.waterfall = false;
    hydrology.waterfallAnchor = false;
    hydrology.delta = false;
    hydrology.estuary = false;
    hydrology.brackish = false;
    hydrology.waterSurface = 0.0;
    hydrology.waterfallTop = 0.0;
    hydrology.waterfallBottom = 0.0;
    hydrology.waterfallWidth = 0.0;
    hydrology.lakeDepth = 0.0;
    hydrology.lakeShoreDistance = -1.0e9;
    hydrology.shoreWaterSurface = 0.0;
    hydrology.lakeBank = false;
    hydrology.lakeBankTarget = 0.0;
    hydrology.lakeBankInfluence = 0.0;
    hydrology.lakeAreaSquareKilometers = 0.0;
    hydrology.lakeVolumeCubicMeters = 0.0;
    hydrology.lakeRunoffMmSquareKilometers = 0.0;
    hydrology.lakeLossMm = 0.0;
    hydrology.lakeOverflowMmSquareKilometers = 0.0;
    hydrology.lakeSpillSurface = 0.0;
    hydrology.streamOrder = 0;
    hydrology.distributaryCount = 0;
    hydrology.waterBodyId = worldgen::NO_WATER_BODY;
    hydrology.generatedFluidLevel = 0;
    hydrology.transitionOwnerKind = worldgen::WaterTransitionKind::NONE;
    hydrology.transitionOwnerId = 0;
}

void carveSolvedWaterBed(worldgen::HydrologySample& hydrology, int& surfaceY) {
    for (int iteration = 0; iteration < 4; ++iteration) {
        if (!hasSurfaceWater(hydrology)) return;
        const int waterTopY = std::clamp(static_cast<int>(std::ceil(hydrology.waterSurface)) - 1,
                                         WORLD_MIN_Y, WORLD_MAX_Y);
        const int plannedFloorY = std::clamp(
            static_cast<int>(std::ceil(hydrology.surfaceElevation)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
        const int carved =
            std::min(surfaceY, std::min(plannedFloorY, std::max(WORLD_MIN_Y, waterTopY - 1)));
        if (carved == surfaceY) return;
        surfaceY = carved;
    }
}

void reconcileEmittedSurfaceY(worldgen::HydrologySample& hydrology, int& surfaceY,
                              bool learnedAuthority) {
    if (learnedAuthority) {
        carveSolvedWaterBed(hydrology, surfaceY);
    } else if (hasSurfaceWater(hydrology)) {
        const int waterTopY = std::clamp(static_cast<int>(std::ceil(hydrology.waterSurface)) - 1,
                                         WORLD_MIN_Y, WORLD_MAX_Y);
        const int plannedFloorY = std::clamp(
            static_cast<int>(std::ceil(hydrology.surfaceElevation)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
        surfaceY = std::min(plannedFloorY, std::max(WORLD_MIN_Y, waterTopY - 1));
    } else if (hydrology.channelBank || hydrology.lakeBank) {
        const int plannedBankY = std::clamp(
            static_cast<int>(std::ceil(hydrology.surfaceElevation)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
        surfaceY = std::max(surfaceY, plannedBankY);
    } else if (hydrology.surfaceElevation >= SEA_LEVEL && surfaceY < SEA_LEVEL - 1) {
        surfaceY = SEA_LEVEL - 1;
    }

    if (!learnedAuthority && hydrology.waterfall &&
        hydrology.waterfallTop >= hydrology.waterfallBottom + 0.5) {
        const int receivingWaterY = std::clamp(
            static_cast<int>(std::ceil(hydrology.waterfallBottom)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
        surfaceY = std::min(surfaceY, std::max(WORLD_MIN_Y, receivingWaterY - 1));
    }
}

void applyEmittedSurfaceTopology(worldgen::SurfaceSample& surface, double emittedTop,
                                 bool learnedAuthority) {
    surface.emittedTerrainHeight = static_cast<float>(emittedTop);
    if (learnedAuthority) {
        // Learned v4 keeps its continuous terrain and routed hydrology as the
        // canonical physical authority. Voxel emission is a separate mesh,
        // collision, and fluid contact concern, not an ecology input.
        return;
    }
    surface.terrainHeight = emittedTop;
    surface.hydrology.surfaceElevation = emittedTop;
    if (surface.hydrology.channelBank) {
        clearSurfaceWater(surface.hydrology);
        surface.hydrology.channelBank = true;
    } else if (!hasSurfaceWater(surface.hydrology) && !surface.hydrology.lakeBank) {
        // Water ownership is solved by the continuous hydrology field. Voxel
        // density may shape a dry coast down to the sea plane, but it cannot
        // create an isolated ocean cell that no drainage body owns.
        clearSurfaceWater(surface.hydrology);
    } else if (surface.hydrology.ocean && emittedTop >= SEA_LEVEL) {
        clearSurfaceWater(surface.hydrology);
    }
    if (surface.hydrology.lake) {
        surface.hydrology.lakeDepth = std::max(0.0, surface.hydrology.waterSurface - emittedTop);
    }
    surface.waterSurface = surface.hydrology.waterSurface;
}

bool legalExplicitFall(const worldgen::SurfaceSample& surface) {
    return surface.hydrology.waterfall &&
           surface.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL &&
           surface.hydrology.transitionOwnerId != 0 &&
           surface.hydrology.waterfallTop >= surface.hydrology.waterfallBottom + 0.5;
}

void ensureExplicitFallOwnership(worldgen::SurfaceSample& surface, int64_t x, int64_t z) {
    if (!surface.hydrology.waterfall ||
        surface.hydrology.waterfallTop < surface.hydrology.waterfallBottom + 0.5 ||
        legalExplicitFall(surface)) {
        return;
    }
    uint64_t owner = hash64(0x4E41'5449'5645'464CULL ^ static_cast<uint64_t>(x));
    owner = hash64(owner ^ static_cast<uint64_t>(z));
    owner = hash64(owner ^ surface.hydrology.waterBodyId);
    if (owner == 0) owner = 1;
    surface.hydrology.transitionOwnerKind = worldgen::WaterTransitionKind::EXPLICIT_FALL;
    surface.hydrology.transitionOwnerId = owner;
}

void applyExplicitWetContact(worldgen::SurfaceSample& surface, int64_t x, int64_t z,
                             const worldgen::SurfaceSample& receiver, int offsetX, int offsetZ) {
    const worldgen::GeneratedFluidColumn sourceFluid = worldgen::generatedFluidColumn(surface);
    const worldgen::GeneratedFluidColumn receiverFluid = worldgen::generatedFluidColumn(receiver);
    if (!sourceFluid.wet || !receiverFluid.wet || legalExplicitFall(surface) ||
        legalExplicitFall(receiver) ||
        sourceFluid.visibleSurface <= receiverFluid.visibleSurface + 0.125001) {
        return;
    }

    uint64_t owner = hash64(0x5745545F434F4E54ULL ^ static_cast<uint64_t>(x));
    owner = hash64(owner ^ static_cast<uint64_t>(z));
    owner = hash64(owner ^ static_cast<uint64_t>(static_cast<int64_t>(offsetX)));
    owner = hash64(owner ^ static_cast<uint64_t>(static_cast<int64_t>(offsetZ)));
    if (owner == 0) owner = 1;

    const int fallingTopY = static_cast<int>(std::ceil(sourceFluid.visibleSurface)) - 1;
    surface.waterSurface = static_cast<double>(fallingTopY) + 0.125;
    surface.terrainHeight = std::min(surface.terrainHeight, receiverFluid.visibleSurface - 0.125);
    surface.emittedTerrainHeight = static_cast<float>(surface.terrainHeight);
    surface.hydrology.surfaceElevation = surface.terrainHeight;
    surface.hydrology.channelDepth =
        std::max(surface.hydrology.channelDepth, surface.waterSurface - surface.terrainHeight);
    surface.hydrology.channelGradient =
        std::max(surface.hydrology.channelGradient,
                 sourceFluid.visibleSurface - receiverFluid.visibleSurface);
    surface.hydrology.ocean = false;
    surface.hydrology.river = true;
    surface.hydrology.lake = false;
    surface.hydrology.lakeBank = false;
    surface.hydrology.channelBank = false;
    surface.hydrology.endorheic = false;
    surface.hydrology.waterfall = true;
    surface.hydrology.waterfallAnchor = true;
    surface.hydrology.lakeDepth = 0.0;
    surface.hydrology.lakeAreaSquareKilometers = 0.0;
    surface.hydrology.lakeVolumeCubicMeters = 0.0;
    surface.hydrology.lakeRunoffMmSquareKilometers = 0.0;
    surface.hydrology.lakeLossMm = 0.0;
    surface.hydrology.lakeOverflowMmSquareKilometers = 0.0;
    surface.hydrology.lakeSpillSurface = 0.0;
    surface.hydrology.waterBodyId = worldgen::NO_WATER_BODY;
    surface.hydrology.generatedFluidLevel = 7;
    surface.hydrology.transitionOwnerKind = worldgen::WaterTransitionKind::EXPLICIT_FALL;
    surface.hydrology.transitionOwnerId = owner;
    surface.hydrology.waterfallTop = sourceFluid.visibleSurface;
    surface.hydrology.waterfallBottom = receiverFluid.visibleSurface;
    surface.hydrology.waterfallWidth = 1.0;
    surface.hydrology.waterSurface = surface.waterSurface;
}

template <typename SampleUnfinalized>
worldgen::SurfaceSample finalizeCoordinateWetContact(int64_t x, int64_t z,
                                                     worldgen::SurfaceSample result,
                                                     SampleUnfinalized&& sampleUnfinalized,
                                                     bool repairStandingRiver = false) {
    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};
    ensureExplicitFallOwnership(result, x, z);

    const worldgen::GeneratedFluidColumn repairedFluid = worldgen::generatedFluidColumn(result);
    if (!repairedFluid.wet) return result;
    if (legalExplicitFall(result)) return result;
    const bool partialRiver = result.hydrology.river && result.hydrology.generatedFluidLevel > 0;
    const bool mayExposeWetContact = result.hydrology.lake || result.hydrology.waterfall ||
                                     partialRiver ||
                                     (repairStandingRiver && result.hydrology.river);
    if (!mayExposeWetContact) return result;

    struct Receiver {
        worldgen::SurfaceSample sample;
        int offsetX = 0;
        int offsetZ = 0;
        double visibleSurface = 0.0;
        bool found = false;
    };
    Receiver receiver;
    const double sourceVisibleSurface = worldgen::generatedFluidColumn(result).visibleSurface;
    constexpr double MAXIMUM_UNTAGGED_VISIBLE_STEP = 0.125001;
    for (const auto [offsetX, offsetZ] : CARDINAL) {
        worldgen::SurfaceSample neighbor = sampleUnfinalized(x + offsetX, z + offsetZ);
        ensureExplicitFallOwnership(neighbor, x + offsetX, z + offsetZ);
        const worldgen::GeneratedFluidColumn neighborFluid =
            worldgen::generatedFluidColumn(neighbor);
        if (!neighborFluid.wet || legalExplicitFall(neighbor) ||
            sourceVisibleSurface <= neighborFluid.visibleSurface + MAXIMUM_UNTAGGED_VISIBLE_STEP) {
            continue;
        }
        const auto candidateRank = std::tuple{neighborFluid.visibleSurface, offsetX, offsetZ};
        const auto retainedRank =
            std::tuple{receiver.visibleSurface, receiver.offsetX, receiver.offsetZ};
        if (!receiver.found || candidateRank < retainedRank) {
            receiver = {
                .sample = std::move(neighbor),
                .offsetX = offsetX,
                .offsetZ = offsetZ,
                .visibleSurface = neighborFluid.visibleSurface,
                .found = true,
            };
        }
    }
    if (receiver.found) {
        applyExplicitWetContact(result, x, z, receiver.sample, receiver.offsetX, receiver.offsetZ);
    }
    return result;
}

bool applyCraterLake(worldgen::SurfaceSample& surface, const VolcanicColumnSample& volcanism,
                     bool applyOuterProfile, bool learnedAuthority) {
    if (!volcanism.craterLake || volcanism.craterLakeRadius <= 0.0 ||
        volcanism.craterRimElevation <= volcanism.craterLakeSurface) {
        return false;
    }

    if (volcanism.craterProfileDistance <= volcanism.craterRadius) {
        surface.terrainHeight = volcanism.craterTerrainTarget;
    } else if (applyOuterProfile && volcanism.craterProfileInfluence > 0.0) {
        surface.terrainHeight += (volcanism.craterTerrainTarget - surface.terrainHeight) *
                                 volcanism.craterProfileInfluence;
    }
    const double minimumTerrain = learnedAuthority ? static_cast<double>(WORLD_MIN_Y) : -112.0;
    const double maximumTerrain = learnedAuthority ? static_cast<double>(WORLD_MAX_Y) : 480.0;
    surface.terrainHeight = std::clamp(surface.terrainHeight, minimumTerrain, maximumTerrain);
    surface.hydrology.surfaceElevation = surface.terrainHeight;

    const double shoreDistance = volcanism.craterLakeRadius - volcanism.craterProfileDistance;
    const bool insideCaldera = volcanism.craterProfileDistance <= volcanism.craterRadius;
    const bool wet =
        shoreDistance > 0.0 && volcanism.craterLakeSurface > surface.terrainHeight + 0.01;
    if (!wet) {
        if (!insideCaldera) return false;
        clearSurfaceWater(surface.hydrology);
        surface.hydrology.lakeShoreDistance = std::min(0.0, shoreDistance);
        surface.hydrology.shoreWaterSurface = volcanism.craterLakeSurface;
        if (shoreDistance > -volcanism.craterRimWidth) {
            surface.hydrology.lakeBank = true;
            surface.hydrology.lakeBankInfluence =
                1.0 - smootherstep(0.0, volcanism.craterRimWidth, -shoreDistance);
            // The absolute crater profile already supplies its physical rim.
            // Keep the bank fields as descriptive ecology and material data;
            // raising a dry column to the lake stage here would manufacture a
            // retaining wall after the water body had already been solved.
            surface.hydrology.lakeBankTarget = surface.terrainHeight;
        }
        surface.waterSurface = 0.0;
        return false;
    }
    clearSurfaceWater(surface.hydrology);
    surface.hydrology.lake = true;
    surface.hydrology.endorheic = true;
    surface.hydrology.waterBodyId = volcanism.craterWaterBodyId;
    surface.hydrology.lakeShoreDistance = shoreDistance;
    surface.hydrology.shoreWaterSurface = volcanism.craterLakeSurface;
    surface.hydrology.lakeDepth = volcanism.craterLakeSurface - surface.terrainHeight;
    surface.hydrology.waterSurface = volcanism.craterLakeSurface;
    surface.waterSurface = volcanism.craterLakeSurface;
    return true;
}

double emittedVolcanicHeight(const worldgen::SurfaceSample& macroSurface,
                             const VolcanicColumnSample& volcanism, bool learnedAuthority) {
    double heightAdjustment = volcanism.heightAdjustment;
    if (macroSurface.geology.crust == worldgen::CrustType::OCEANIC &&
        macroSurface.hydrology.ocean && volcanism.strongestProfile > 0.38) {
        const double islandLift =
            (static_cast<double>(SEA_LEVEL + 6) - macroSurface.terrainHeight) *
            volcanism.strongestProfile;
        heightAdjustment = std::max(heightAdjustment, std::min(18.0, islandLift));
    }
    const double minimumTerrain = learnedAuthority ? static_cast<double>(WORLD_MIN_Y) : -112.0;
    const double maximumTerrain = learnedAuthority ? static_cast<double>(WORLD_MAX_Y) : 480.0;
    return std::clamp(macroSurface.terrainHeight + std::clamp(heightAdjustment, -18.0, 18.0),
                      minimumTerrain, maximumTerrain);
}

Biome ditheredBiome(const worldgen::SurfaceSample& surface, const CounterRng& random, int64_t x,
                    int64_t z) {
    return worldgen::surface_material::materialBiome(surface, random, x, z);
}

bool isAlluvialDeposit(const worldgen::SurfaceSample& surface, const CounterRng& random, int64_t x,
                       int64_t z) {
    const double influence = std::max(
        {worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::FLOODPLAIN),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::LAKESHORE),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::DELTA)});
    const double coverage =
        std::clamp(influence * (0.18 + surface.soil.moisture * 0.52), 0.0, 0.72);
    return worldgen::multiscaleDitherThreshold(random, worldgen::surface_material::DITHER_STREAM, x,
                                               z, 1) < coverage;
}

worldgen::surface_material::VolcanicSignals materialSignals(const VolcanicColumnSample& volcanism) {
    return {
        .basaltField = volcanism.basaltField,
        .craterFactor = volcanism.craterFactor,
        .conduitExposure = volcanism.centerDistance <= volcanism.conduitRadius * 2.5,
    };
}

struct StrataColumn {
    worldgen::GeologySample geology;
    double upperOffset = 0.0;
    double lowerOffset = 0.0;
    double thicknessScale = 1.0;
    double thicknessUndulation = 0.0;
    double deformationAmplitude = 0.0;
    double phaseSine = 0.0;
    double phaseCosine = 1.0;
    double secondaryPhaseSine = 0.0;
    double secondaryPhaseCosine = 1.0;
    int unconformityY = 0;
    double intrusionStrength = 0.0;
    double intrusionDistance = 1.0;
    double intrusionCenterY = 0.0;
    double intrusionHalfHeight = 0.0;
    double lensStrength = 0.0;
    double lensCenterY = 0.0;
    double lensHalfHeight = 0.0;
};

struct VerticalStrataWaves {
    static constexpr size_t SAMPLE_COUNT = WORLD_MAX_Y - WORLD_MIN_Y + 1;

    std::array<double, SAMPLE_COUNT> sine83{};
    std::array<double, SAMPLE_COUNT> cosine83{};
    std::array<double, SAMPLE_COUNT> sine137{};
    std::array<double, SAMPLE_COUNT> cosine137{};
    std::array<double, SAMPLE_COUNT> sine211{};
    std::array<double, SAMPLE_COUNT> cosine211{};

    VerticalStrataWaves() {
        for (int y = WORLD_MIN_Y; y <= WORLD_MAX_Y; ++y) {
            const size_t index = static_cast<size_t>(y - WORLD_MIN_Y);
            const double phase83 = static_cast<double>(y) * (2.0 * std::numbers::pi / 83.0);
            const double phase137 = static_cast<double>(y) * (2.0 * std::numbers::pi / 137.0);
            const double phase211 = static_cast<double>(y) * (2.0 * std::numbers::pi / 211.0);
            sine83[index] = std::sin(phase83);
            cosine83[index] = std::cos(phase83);
            sine137[index] = std::sin(phase137);
            cosine137[index] = std::cos(phase137);
            sine211[index] = std::sin(phase211);
            cosine211[index] = std::cos(phase211);
        }
    }
};

const VerticalStrataWaves& verticalStrataWaves() {
    static const VerticalStrataWaves waves;
    return waves;
}

double counterSimplex2D(const CounterRng& random, uint64_t stream, double x, double z,
                        uint32_t index) {
    struct Gradient {
        double x;
        double z;
    };
    constexpr double F2 = 0.36602540378443864676;
    constexpr double G2 = 0.21132486540518711775;
    constexpr std::array<Gradient, 16> GRADIENTS = {{
        {1.0, 0.0},
        {0.9238795325, 0.3826834324},
        {0.7071067812, 0.7071067812},
        {0.3826834324, 0.9238795325},
        {0.0, 1.0},
        {-0.3826834324, 0.9238795325},
        {-0.7071067812, 0.7071067812},
        {-0.9238795325, 0.3826834324},
        {-1.0, 0.0},
        {-0.9238795325, -0.3826834324},
        {-0.7071067812, -0.7071067812},
        {-0.3826834324, -0.9238795325},
        {0.0, -1.0},
        {0.3826834324, -0.9238795325},
        {0.7071067812, -0.7071067812},
        {0.9238795325, -0.3826834324},
    }};
    const auto boundedFloorToInt64 = [](double value) {
        // A floating-point to integer conversion outside the destination
        // range is undefined. Leave room for the simplex corner offsets as
        // well, so even diagnostic samples at extreme coordinates stay
        // deterministic instead of overflowing cell + 1 below.
        constexpr double MIN_EXCLUSIVE = -0x1.0p63;
        constexpr double MAX_EXCLUSIVE = 0x1.0p63;
        if (!std::isfinite(value)) {
            return std::signbit(value) ? std::numeric_limits<int64_t>::min() + 2
                                       : std::numeric_limits<int64_t>::max() - 2;
        }
        if (value <= MIN_EXCLUSIVE) return std::numeric_limits<int64_t>::min() + 2;
        if (value >= MAX_EXCLUSIVE) return std::numeric_limits<int64_t>::max() - 2;
        return static_cast<int64_t>(std::floor(value));
    };
    const double skew = (x + z) * F2;
    const int64_t cellX = boundedFloorToInt64(x + skew);
    const int64_t cellZ = boundedFloorToInt64(z + skew);
    const double unskew = (static_cast<double>(cellX) + static_cast<double>(cellZ)) * G2;
    const double x0 = x - (static_cast<double>(cellX) - unskew);
    const double z0 = z - (static_cast<double>(cellZ) - unskew);
    const int64_t xStep = x0 > z0 ? 1 : 0;
    const int64_t zStep = x0 > z0 ? 0 : 1;
    const std::array<double, 3> offsetsX = {x0, x0 - static_cast<double>(xStep) + G2,
                                            x0 - 1.0 + 2.0 * G2};
    const std::array<double, 3> offsetsZ = {z0, z0 - static_cast<double>(zStep) + G2,
                                            z0 - 1.0 + 2.0 * G2};
    const std::array<int64_t, 3> latticeX = {cellX, cellX + xStep, cellX + 1};
    const std::array<int64_t, 3> latticeZ = {cellZ, cellZ + zStep, cellZ + 1};
    double value = 0.0;
    for (size_t corner = 0; corner < offsetsX.size(); ++corner) {
        double attenuation =
            0.5 - offsetsX[corner] * offsetsX[corner] - offsetsZ[corner] * offsetsZ[corner];
        if (attenuation <= 0.0) continue;
        const uint32_t hash = random.u32(stream, latticeX[corner], 0, latticeZ[corner],
                                         index + static_cast<uint32_t>(corner));
        const Gradient gradient = GRADIENTS[hash & 15U];
        attenuation *= attenuation;
        value += attenuation * attenuation *
                 (gradient.x * offsetsX[corner] + gradient.z * offsetsZ[corner]);
    }
    return value * 70.0;
}

double strataField(const CounterRng& random, uint64_t stream, double x, double z, double scale,
                   double angle, uint32_t index) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double sampleX = (x * cosine - z * sine) / scale;
    const double sampleZ = (x * sine + z * cosine) / scale;
    return counterSimplex2D(random, stream, sampleX, sampleZ, index);
}

StrataColumn strataColumnFor(const worldgen::GeologySample& geology, const CounterRng& random,
                             int64_t x, int64_t z, int surfaceY) {
    // A scalar implicit field keeps bedding continuous through column and
    // macro boundaries. Geological contacts change its material response,
    // not the coordinate system itself, so a plate ID cannot expose the
    // column-plan lattice as a vertical wall.
    const double direction = random.uniform01(STRATA_STREAM, 0, 0, 0, 0) * std::numbers::pi * 2.0;
    const double dip = 0.018 + random.uniform01(STRATA_STREAM, 0, 0, 0, 1) * 0.032;
    const double along =
        static_cast<double>(x) * std::cos(direction) + static_cast<double>(z) * std::sin(direction);
    const double broadFold =
        strataField(random, STRATA_STREAM ^ 0xB0AD'F01DULL, x, z, 1850.0, 0.489, 0) * 8.5;
    const double foldedBeds =
        strataField(random, STRATA_STREAM ^ 0xF01D'BED5ULL, x, z, 620.0, 1.237, 1) * 3.8;
    const double bedIrregularity =
        strataField(random, STRATA_STREAM ^ 0x0B3D'1AA1ULL, x, z, 61.0, 0.716, 18) * 2.2 +
        strataField(random, STRATA_STREAM ^ 0x0B3D'1AA2ULL, x, z, 23.0, 2.672, 19) * 0.8;
    const double faultField = std::clamp(
        broadFold / 8.5 * 0.55 + foldedBeds / 3.8 * 0.25 + bedIrregularity / 3.0 * 0.20, -1.0, 1.0);
    const bool taggedFault =
        geology.boundary == worldgen::PlateBoundary::TRANSFORM && geology.faultStrength > 0.25;
    const double faultOffset =
        (taggedFault ? std::copysign(1.0, faultField) : std::tanh(faultField * 1.8)) *
        geology.faultStrength * (taggedFault ? 11.0 : 3.0);
    const double thicknessField =
        std::clamp(broadFold / 8.5 * 0.55 + foldedBeds / 3.8 * 0.45, -1.0, 1.0);
    const double unconformityField = std::clamp(
        broadFold / 8.5 * 0.45 - faultField * 0.35 + bedIrregularity / 3.0 * 0.20, -1.0, 1.0);
    const double bodyField =
        strataField(random, STRATA_STREAM ^ 0x1E45'B3D5ULL, x, z, 240.0, 1.421, 7);
    const double intrusion =
        strataField(random, STRATA_STREAM ^ 0x1A72'0510ULL, x, z, 360.0, 0.296, 6);
    const double curvedDike = intrusion + bedIrregularity * 0.10 + foldedBeds * 0.025;
    const double lens = bodyField;
    const double lowerDirection = direction + 0.39;
    const double lowerAlong = static_cast<double>(x) * std::cos(lowerDirection) +
                              static_cast<double>(z) * std::sin(lowerDirection);
    const double lowerFold = broadFold * 0.32 - foldedBeds * 0.40 + bedIrregularity * 0.60;
    const double deformation =
        std::clamp(foldedBeds / 3.8 * 0.65 + bedIrregularity / 3.0 * 0.35, -1.0, 1.0);
    const double deformationPhase = std::clamp(lowerFold / 9.0, -1.0, 1.0) * std::numbers::pi;
    const double bodyHeight = std::clamp(lens * 0.68 + intrusion * 0.32, -1.0, 1.0);
    worldgen::GeologySample emittedGeology = geology;
    if (emittedGeology.rock != geology.lithology.primary &&
        emittedGeology.rock != geology.lithology.secondary) {
        emittedGeology.rock = geology.lithology.primary;
    }
    // Do not wrap the scalar phase. Variable bed thickness and the 71-layer
    // intrusion period make every finite modulus visible as a planar reset.
    // Double precision retains sub-block phase at the practical travel limit.
    const double upperOffset = along * dip + broadFold + foldedBeds + bedIrregularity + faultOffset;
    const double lowerOffset = lowerAlong * dip * 0.76 + lowerFold + faultOffset * 0.65 +
                               unconformityField * 8.0 + bedIrregularity * 0.72;
    return {
        .geology = emittedGeology,
        .upperOffset = upperOffset,
        .lowerOffset = lowerOffset,
        .thicknessScale = std::clamp(1.0 + thicknessField * 0.22, 0.74, 1.28),
        .thicknessUndulation = thicknessField * 0.12,
        .deformationAmplitude = 0.45 + std::abs(deformation) * 2.1,
        .phaseSine = std::sin(deformationPhase),
        .phaseCosine = std::cos(deformationPhase),
        .secondaryPhaseSine = std::sin(deformationPhase * -0.47),
        .secondaryPhaseCosine = std::cos(deformationPhase * -0.47),
        .unconformityY =
            surfaceY - static_cast<int>(std::lround(38.0 + (unconformityField + 1.0) * 15.0)),
        .intrusionStrength =
            intrusion * 0.5 + geology.volcanicActivity * 0.75 + geology.uplift * 0.22,
        .intrusionDistance = std::abs(curvedDike),
        .intrusionCenterY = -18.0 + bodyHeight * 128.0,
        .intrusionHalfHeight = 14.0 + std::abs(intrusion) * 28.0,
        .lensStrength = lens,
        .lensCenterY = -24.0 + bodyHeight * 104.0,
        .lensHalfHeight = 1.4 + std::abs(lens) * 3.6,
    };
}

BlockType strataBlockFor(const StrataColumn& strata, int y, int depth) {
    const worldgen::GeologySample& geology = strata.geology;
    const VerticalStrataWaves& waves = verticalStrataWaves();
    const size_t verticalIndex =
        static_cast<size_t>(std::clamp(y, WORLD_MIN_Y, WORLD_MAX_Y) - WORLD_MIN_Y);
    const double offset = y > strata.unconformityY ? strata.upperOffset : strata.lowerOffset;
    const double verticalDeformation =
        (waves.sine83[verticalIndex] * strata.phaseCosine +
         waves.cosine83[verticalIndex] * strata.phaseSine) *
            strata.deformationAmplitude +
        (waves.sine211[verticalIndex] * strata.secondaryPhaseCosine +
         waves.cosine211[verticalIndex] * strata.secondaryPhaseSine) *
            strata.deformationAmplitude * 0.38;
    const double thickness =
        strata.thicknessScale *
        std::clamp(1.0 + strata.thicknessUndulation *
                             (waves.sine137[verticalIndex] * strata.phaseCosine +
                              waves.cosine137[verticalIndex] * strata.phaseSine),
                   0.72, 1.32);
    // Thickness stretches vertical distance from the local unconformity. It
    // must not scale the unbounded regional phase, or a tiny thickness change
    // becomes a large horizontal jump after long-distance travel.
    const double adjustedY = static_cast<double>(strata.unconformityY) +
                             (static_cast<double>(y - strata.unconformityY) / thickness);
    const int64_t layerCoordinate =
        static_cast<int64_t>(std::floor(adjustedY + offset + verticalDeformation));
    const bool intrusiveHost = geology.volcanicActivity > 0.14 || geology.uplift > 0.42 ||
                               geology.rock == worldgen::RockType::VOLCANIC;
    const bool dike = intrusiveHost && strata.intrusionDistance < 0.095;
    const bool pluton =
        intrusiveHost && strata.intrusionStrength > 0.78 &&
        std::abs(static_cast<double>(y) - strata.intrusionCenterY) < strata.intrusionHalfHeight;
    const bool continentalArc = geology.crust == worldgen::CrustType::CONTINENTAL &&
                                geology.boundary == worldgen::PlateBoundary::CONVERGENT &&
                                geology.uplift > 0.55;
    const int intrusionBand = world_coord::floorMod(layerCoordinate, int32_t{71});
    const bool sill = (strata.intrusionStrength > 0.72 || continentalArc) &&
                      intrusionBand < (continentalArc ? 6 : 2);
    if (depth > 8 && (dike || pluton || sill)) {
        return geology.volcanicActivity > 0.72 ? BlockType::BASALT : BlockType::ANDESITE;
    }
    const bool lens =
        std::abs(strata.lensStrength) > 0.62 &&
        std::abs(static_cast<double>(y) - strata.lensCenterY) <= strata.lensHalfHeight;
    if (depth > 16 && lens) {
        return strata.lensStrength > 0.0 ? BlockType::LIMESTONE : BlockType::SANDSTONE;
    }
    switch (geology.rock) {
        case worldgen::RockType::GRANITE: {
            const int layer = world_coord::floorMod(layerCoordinate, int32_t{48});
            const bool arcIntrusion = geology.crust == worldgen::CrustType::CONTINENTAL &&
                                      geology.boundary == worldgen::PlateBoundary::CONVERGENT &&
                                      geology.uplift > 0.18;
            return arcIntrusion && layer < 6 ? BlockType::ANDESITE : BlockType::STONE;
        }
        case worldgen::RockType::BASALT: {
            const int layer = world_coord::floorMod(layerCoordinate, int32_t{18});
            const bool alteredCap = geology.crustAge > 0.68 && depth < 40;
            return alteredCap && layer < 3 ? BlockType::ANDESITE : BlockType::BASALT;
        }
        case worldgen::RockType::LIMESTONE: {
            const int layer = world_coord::floorMod(layerCoordinate, int32_t{20});
            if (layer < 2) return BlockType::CLAY;
            return layer < 6 ? BlockType::SANDSTONE : BlockType::LIMESTONE;
        }
        case worldgen::RockType::SANDSTONE: {
            const int layer = world_coord::floorMod(layerCoordinate, int32_t{22});
            if (layer < 3) return BlockType::CLAY;
            return layer < 6 ? BlockType::SILT : BlockType::SANDSTONE;
        }
        case worldgen::RockType::VOLCANIC: {
            const int layer = world_coord::floorMod(layerCoordinate, int32_t{40});
            const bool continentalArc = geology.crust == worldgen::CrustType::CONTINENTAL &&
                                        geology.boundary == worldgen::PlateBoundary::CONVERGENT;
            if (depth > 12 && geology.volcanicActivity > 0.76 && layer == 0) {
                return BlockType::OBSIDIAN;
            }
            if (layer < 8) return BlockType::BASALT;
            return continentalArc ? BlockType::ANDESITE : BlockType::BASALT;
        }
    }
}

enum class AquiferVoxel : uint8_t {
    NONE,
    SHELL,
    WATER,
};

AquiferVoxel aquiferVoxelAt(const CounterRng& random, int64_t x, int y, int64_t z,
                            const worldgen::SurfaceSample& surface) {
    if (!worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::AQUIFER) ||
        y > static_cast<int>(std::floor(surface.soil.waterTable)) ||
        y > static_cast<int>(std::floor(surface.terrainHeight)) - 12 || y < WORLD_MIN_Y + 6) {
        return AquiferVoxel::NONE;
    }

    const int64_t cellX = world_coord::floorDiv(x, AQUIFER_CELL_EDGE);
    const int64_t cellZ = world_coord::floorDiv(z, AQUIFER_CELL_EDGE);
    const int32_t cellY = world_coord::floorDiv(y, AQUIFER_CELL_HEIGHT);
    const double activation = random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 0);
    const double threshold = 0.08 + surface.soil.moisture * 0.12;
    if (activation >= threshold) return AquiferVoxel::NONE;

    const double centerX = (static_cast<double>(cellX) + 0.38 +
                            random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 1) * 0.24) *
                           AQUIFER_CELL_EDGE;
    const double centerY = (static_cast<double>(cellY) + 0.35 +
                            random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 2) * 0.30) *
                           AQUIFER_CELL_HEIGHT;
    const double centerZ = (static_cast<double>(cellZ) + 0.38 +
                            random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 3) * 0.24) *
                           AQUIFER_CELL_EDGE;
    if (centerY > surface.soil.waterTable - 4.0 || centerY > surface.terrainHeight - 16.0) {
        return AquiferVoxel::NONE;
    }

    const double radiusX = 8.0 + random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 4) * 7.0;
    const double radiusY = 3.0 + random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 5) * 4.0;
    const double radiusZ = 8.0 + random.uniform01(AQUIFER_STREAM, cellX, cellY, cellZ, 6) * 7.0;
    const double dx = (static_cast<double>(x) + 0.5 - centerX) / radiusX;
    const double dy = (static_cast<double>(y) + 0.5 - centerY) / radiusY;
    const double dz = (static_cast<double>(z) + 0.5 - centerZ) / radiusZ;
    const double ellipsoid = dx * dx + dy * dy + dz * dz;
    if (ellipsoid <= 0.66) return AquiferVoxel::WATER;
    if (ellipsoid <= 1.0) return AquiferVoxel::SHELL;
    return AquiferVoxel::NONE;
}

double surfaceDetailAmplitude(const worldgen::SurfaceSample& surface, double slopeEnvelope = 0.0) {
    if (hasSurfaceWater(surface.hydrology)) return 0.75;
    return std::clamp(2.0 + (surface.slope + slopeEnvelope) * 2.5 + surface.geology.uplift * 7.0 +
                          surface.geology.faultStrength * 6.0 +
                          surface.geology.volcanicActivity * 4.0,
                      1.5, 14.0);
}

int boundedLearnedSurfaceY(const worldgen::SurfaceSample& surface, int surfaceY,
                           bool learnedAuthority) {
    if (!learnedAuthority) return surfaceY;
    const double amplitude = surfaceDetailAmplitude(surface);
    int minimum = static_cast<int>(std::floor(surface.terrainHeight - amplitude));
    if (!hasSurfaceWater(surface.hydrology) &&
        surface.terrainHeight <= SEA_LEVEL + amplitude + 1.0) {
        // Post-hydrology detail may texture a learned coast, but it cannot
        // move a final dry column below the sea plane and create a depression
        // that the neighboring ocean must later fill or wall off.
        minimum = std::max(minimum, SEA_LEVEL - 1);
    }
    const int maximum = static_cast<int>(std::ceil(surface.terrainHeight + amplitude));
    return std::clamp(surfaceY, std::max(WORLD_MIN_Y, minimum), std::min(WORLD_MAX_Y, maximum));
}

ColumnShape shapeFromSurface(const worldgen::SurfaceSample& surface, double caveEntrance,
                             bool learnedAuthority) {
    ColumnShape result;
    result.climate.continentalness = surface.geology.continentalFraction * 2.0 - 1.0;
    result.climate.erosion =
        std::clamp(1.0 - surface.geology.uplift * 1.4 - surface.geology.faultStrength, -1.0, 1.0);
    result.climate.ridges =
        std::clamp(surface.geology.uplift + surface.geology.faultStrength * 0.5, -1.0, 1.0);
    result.climate.temperature = std::clamp(surface.climate.temperatureC / 30.0, -1.0, 1.0);
    result.climate.humidity =
        std::clamp(surface.climate.annualPrecipitationMm / 1600.0 - 1.0, -1.0, 1.0);
    result.height = surface.terrainHeight;
    const bool waterCovered = hasSurfaceWater(surface.hydrology);
    // V4's slope-gated residual is already part of this final macro surface.
    // Reapplying the legacy density detail here would displace a coast, divide,
    // or water bed after its clearance gates were checked. Caves remain
    // bounded below the surface by the density field's near-surface sealing.
    result.detailAmp = learnedAuthority ? 0.0 : surfaceDetailAmplitude(surface);
    result.entrance = waterCovered ? -1.0 : caveEntrance;
    result.riverCut = surface.hydrology.erosionDepth;
    result.ravineEdge = 0.0;
    result.ravineFloor = waterCovered ? SEA_LEVEL - 1.0 : result.height;
    return result;
}

uint32_t boundedSubsystemSeed(uint64_t worldSeed) {
    return static_cast<uint32_t>(worldSeed) ^ static_cast<uint32_t>(worldSeed >> 32U);
}

template <typename Sample, typename GridSampler>
void sampleStageFreePointRuns(std::span<const ColumnPos> positions, std::span<Sample> output,
                              GridSampler&& sampleGrid) {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("coarse point sample output has the wrong size");
    }
    if (positions.empty()) return;

    std::vector<size_t> order(positions.size());
    for (size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::ranges::sort(order, [&](size_t first, size_t second) {
        const ColumnPos a = positions[first];
        const ColumnPos b = positions[second];
        if (a.z != b.z) return a.z < b.z;
        if (a.x != b.x) return a.x < b.x;
        return first < second;
    });

    struct UniquePoint {
        ColumnPos position;
        std::vector<size_t> destinations;
    };
    std::vector<UniquePoint> unique;
    unique.reserve(positions.size());
    for (const size_t index : order) {
        if (unique.empty() || unique.back().position != positions[index]) {
            unique.push_back({positions[index], {index}});
        } else {
            unique.back().destinations.push_back(index);
        }
    }

    size_t first = 0;
    while (first < unique.size()) {
        const int64_t row = unique[first].position.z;
        size_t last = first + 1;
        int64_t spacing = 2;
        if (last < unique.size() && unique[last].position.z == row) {
            const __int128 candidate = static_cast<__int128>(unique[last].position.x) -
                                       unique[first].position.x;
            if (candidate <= std::numeric_limits<int>::max()) {
                spacing = static_cast<int64_t>(candidate);
                ++last;
                while (last < unique.size() && unique[last].position.z == row &&
                       static_cast<__int128>(unique[last].position.x) -
                               unique[last - 1].position.x ==
                           spacing) {
                    ++last;
                }
            }
        }
        const size_t count = last - first;
        std::vector<Sample> run(count);
        sampleGrid(unique[first].position.x, row, static_cast<int>(spacing), 2,
                   static_cast<int>(count), 1, run);
        for (size_t offset = 0; offset < count; ++offset) {
            for (const size_t destination : unique[first + offset].destinations)
                output[destination] = run[offset];
        }
        first = last;
    }
}

} // namespace

ChunkGenerator::ChunkGenerator(uint64_t worldSeed, GenerationSettings generation)
    : ChunkGenerator(worldSeed, nullptr, generation) {}

ChunkGenerator::ChunkGenerator(
    uint64_t worldSeed,
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
    GenerationSettings generation)
    : seed_(worldSeed)
    , scratchToken_(nextGeneratorInstanceToken())
    , random_(worldSeed)
    , learnedAuthority_(generationContext != nullptr)
    , previewAuthority_(generationContext && generationContext->quality() ==
                                                 worldgen::learned::AuthorityQuality::PREVIEW)
    , macroSampler_(worldSeed, std::move(generationContext))
    , columnPlanCache_()
    , climate_(boundedSubsystemSeed(worldSeed))
    , density_(boundedSubsystemSeed(worldSeed))
    , ores_(boundedSubsystemSeed(worldSeed))
    , structures_(boundedSubsystemSeed(worldSeed), generation.structures)
    , features_(boundedSubsystemSeed(worldSeed)) {}

void ChunkGenerator::clearMacroCaches() const {
    columnPlanCache_.clear();
    farWaterPlanCache_.clear();
    macroSampler_.clearBasinCache();
    macroSampler_.clearMacroControlCache();
    scratchToken_.store(nextGeneratorInstanceToken(), std::memory_order_relaxed);
}

const ColumnShape& ChunkGenerator::latticeShape(int64_t lx, int64_t lz, GenScratch& scratch) const {
    const ColumnPos key{lx, lz};
    auto it = scratch.shapes.find(key);
    if (it != scratch.shapes.end()) return it->second;
    const worldgen::SurfaceSample surface = surfaceSampleAt(lx, lz, scratch);
    const double caveEntrance =
        climate_.caveEntrance(static_cast<double>(lx), static_cast<double>(lz));
    ColumnShape shape = shapeFromSurface(surface, caveEntrance, learnedAuthority_);
    return scratch.shapes.emplace(key, shape).first->second;
}

std::shared_ptr<const ColumnPlan> ChunkGenerator::getColumnPlan(ColumnPos chunkColumn) const {
    return constructColumnPlan(chunkColumn, true);
}

std::shared_ptr<const ColumnPlan>
ChunkGenerator::constructColumnPlan(ColumnPos chunkColumn, bool retainInCache,
                                    ColumnPlanCache* alternateCache) const {
    columnPlanConstructionRequestCount_.fetch_add(1, std::memory_order_relaxed);
    const int64_t baseX = chunkColumn.x * CHUNK_EDGE;
    const int64_t baseZ = chunkColumn.z * CHUNK_EDGE;
    const int64_t apronOriginX = baseX - COLUMN_PLAN_LATTICE_SPACING;
    const int64_t apronOriginZ = baseZ - COLUMN_PLAN_LATTICE_SPACING;
    ColumnPlanConstructionState construction;

    const auto surfaceApron = [&]() -> const ColumnPlanConstructionSurfaceApron& {
        if (!construction.surfaceApron) {
            auto samples = std::make_unique<ColumnPlanConstructionSurfaceApron>();
            macroSampler_.sampleSurfaceGrid(apronOriginX, apronOriginZ, COLUMN_PLAN_LATTICE_SPACING,
                                            COLUMN_PLAN_CONSTRUCTION_APRON_EDGE,
                                            worldgen::SurfaceFootprint::BLOCK_1, *samples);
            GenScratch& scratch = threadScratch();
            for (int sampleZ = 0; sampleZ < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE; ++sampleZ) {
                for (int sampleX = 0; sampleX < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE; ++sampleX) {
                    const size_t index = static_cast<size_t>(
                        sampleZ * COLUMN_PLAN_CONSTRUCTION_APRON_EDGE + sampleX);
                    const int64_t worldX =
                        apronOriginX + static_cast<int64_t>(sampleX) * COLUMN_PLAN_LATTICE_SPACING;
                    const int64_t worldZ =
                        apronOriginZ + static_cast<int64_t>(sampleZ) * COLUMN_PLAN_LATTICE_SPACING;
                    (*samples)[index] =
                        applyVolcanism(worldX, worldZ, std::move((*samples)[index]), scratch);
                }
            }
            construction.surfaceApron = std::move(samples);
        }
        return *construction.surfaceApron;
    };

    const auto canonicalHydrology = [&]() -> const ColumnPlanConstructionHydrology& {
        if (!construction.hydrology) {
            auto samples = std::make_unique<ColumnPlanConstructionHydrology>();
            if (!learnedAuthority_) {
                // The v3 plan remains byte-compatible with its original
                // 17-by-17 hydrology query. Only learned pages require an
                // apron to resolve canonical contacts across page edges.
                const auto applyLegacyTerrainDetail = [&](worldgen::HydrologySample& hydrology,
                                                          int64_t worldX, int64_t worldZ) {
                    if (hydrology.ocean) {
                        hydrology.surfaceElevation = std::min(
                            hydrology.surfaceElevation +
                                macroSampler_.reliefDetail(static_cast<double>(worldX),
                                                           static_cast<double>(worldZ),
                                                           worldgen::SurfaceFootprint::BLOCK_1) *
                                    worldgen::OCEAN_FLOOR_DETAIL_SCALE,
                            static_cast<double>(SEA_LEVEL) - 0.5);
                    } else if (!hasSurfaceWater(hydrology) && !hydrology.channelBank &&
                               !hydrology.lakeBank) {
                        const double channelClearance =
                            hydrology.channelDistance - hydrology.channelWidth * 2.5;
                        hydrology.surfaceElevation +=
                            macroSampler_.reliefDetail(static_cast<double>(worldX),
                                                       static_cast<double>(worldZ),
                                                       worldgen::SurfaceFootprint::BLOCK_1) *
                            smoothstep(0.0, 32.0, channelClearance) *
                            worldgen::DRY_RELIEF_DETAIL_SCALE;
                    }
                };
                LegacyColumnPlanConstructionHydrology legacySamples{};
                macroSampler_.sampleHydrologyGrid(
                    baseX, baseZ, 1, 1, LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE,
                    LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE, legacySamples);
                for (int localZ = 0; localZ < LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE;
                     ++localZ) {
                    for (int localX = 0; localX < LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE;
                         ++localX) {
                        worldgen::HydrologySample& hydrology = (*samples)[static_cast<size_t>(
                            (localZ + COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS) *
                                COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE +
                            localX + COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS)];
                        hydrology = legacySamples[static_cast<size_t>(
                            localZ * LEGACY_COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE + localX)];
                        if (requiresLegacyExactShoreSupport(hydrology)) {
                            hydrology.channelBank = true;
                        }
                        applyLegacyTerrainDetail(hydrology, baseX + localX, baseZ + localZ);
                    }
                }
            } else {
                macroSampler_.sampleHydrologyGrid(baseX - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS,
                                                  baseZ - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS, 1, 1,
                                                  COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE,
                                                  COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE,
                                                  *samples);
            }
            construction.hydrology = std::move(samples);
        }
        return *construction.hydrology;
    };

    const auto canonicalGeology = [&]() -> const ColumnPlanConstructionGeology& {
        if (!construction.geology) {
            auto samples = std::make_unique<ColumnPlanConstructionGeology>();
            for (int localZ = 0; localZ < COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE; ++localZ) {
                for (int localX = 0; localX < COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE; ++localX) {
                    (*samples)[static_cast<size_t>(localZ * COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE +
                                                   localX)] =
                        macroSampler_.sampleGeology(static_cast<double>(baseX + localX),
                                                    static_cast<double>(baseZ + localZ));
                }
            }
            construction.geology = std::move(samples);
        }
        return *construction.geology;
    };

    const auto perimeterHeight = [&]() -> const ColumnPlanConstructionPerimeter& {
        if (!construction.perimeterHeight) {
            auto heights = std::make_unique<ColumnPlanConstructionPerimeter>();
            std::array<ColumnPos, COLUMN_PLAN_CONSTRUCTION_PERIMETER_SAMPLES> positions{};
            std::array<size_t, COLUMN_PLAN_CONSTRUCTION_PERIMETER_SAMPLES> apronIndices{};
            size_t perimeterIndex = 0;
            for (int sampleZ = 0; sampleZ < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE; ++sampleZ) {
                for (int sampleX = 0; sampleX < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE; ++sampleX) {
                    const bool retained =
                        sampleX > 0 && sampleX < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE - 1 &&
                        sampleZ > 0 && sampleZ < COLUMN_PLAN_CONSTRUCTION_APRON_EDGE - 1;
                    if (retained) continue;
                    const size_t apronIndex = static_cast<size_t>(
                        sampleZ * COLUMN_PLAN_CONSTRUCTION_APRON_EDGE + sampleX);
                    positions[perimeterIndex] = {
                        apronOriginX + static_cast<int64_t>(sampleX) * COLUMN_PLAN_LATTICE_SPACING,
                        apronOriginZ + static_cast<int64_t>(sampleZ) * COLUMN_PLAN_LATTICE_SPACING,
                    };
                    apronIndices[perimeterIndex] = apronIndex;
                    ++perimeterIndex;
                }
            }

            std::array<worldgen::HydrologySample, COLUMN_PLAN_CONSTRUCTION_PERIMETER_SAMPLES>
                hydrology{};
            macroSampler_.sampleHydrologyPoints(positions, hydrology);
            GenScratch& scratch = threadScratch();
            for (size_t index = 0; index < positions.size(); ++index) {
                const ColumnPos position = positions[index];
                worldgen::SurfaceSample surface;
                surface.geology = macroSampler_.sampleGeology(static_cast<double>(position.x),
                                                              static_cast<double>(position.z));
                surface.hydrology = hydrology[index];
                surface.terrainHeight = surface.hydrology.surfaceElevation;
                applyVolcanicGeometry(
                    surface, volcanismAt(position.x, position.z, surface.geology, scratch));
                (*heights)[apronIndices[index]] = surface.terrainHeight;
            }
            construction.perimeterHeight = std::move(heights);
        }
        return *construction.perimeterHeight;
    };

    const ColumnPlanSurfaceSampler sampleSurface = [&](int64_t x, int64_t z) {
        const int sampleX = static_cast<int>((x - apronOriginX) / COLUMN_PLAN_LATTICE_SPACING);
        const int sampleZ = static_cast<int>((z - apronOriginZ) / COLUMN_PLAN_LATTICE_SPACING);
        return surfaceApron()[static_cast<size_t>(sampleZ * COLUMN_PLAN_CONSTRUCTION_APRON_EDGE +
                                                  sampleX)];
    };
    const ColumnPlanHeightSampler sampleHeight = [&](int64_t x, int64_t z) {
        const int sampleX = static_cast<int>((x - apronOriginX) / COLUMN_PLAN_LATTICE_SPACING);
        const int sampleZ = static_cast<int>((z - apronOriginZ) / COLUMN_PLAN_LATTICE_SPACING);
        return perimeterHeight()[static_cast<size_t>(sampleZ * COLUMN_PLAN_CONSTRUCTION_APRON_EDGE +
                                                     sampleX)];
    };
    const ColumnPlanSurfaceGridSampler sampleExactSurface = [this](const ColumnPlan& plan) {
        return exactSurfaceGrid(plan);
    };
    const ColumnPlanHydrologySampler sampleHydrology = [&](int64_t x, int64_t z) {
        const int localX = static_cast<int>(x - baseX);
        const int localZ = static_cast<int>(z - baseZ);
        return canonicalHydrology()[static_cast<size_t>(
            (localZ + COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS) *
                COLUMN_PLAN_CONSTRUCTION_AUTHORITY_EDGE +
            localX + COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS)];
    };
    const ColumnPlanGeologySampler sampleGeology = [&](int64_t x, int64_t z) {
        const int localX = static_cast<int>(x - baseX);
        const int localZ = static_cast<int>(z - baseZ);
        return canonicalGeology()[static_cast<size_t>(
            localZ * COLUMN_PLAN_CONSTRUCTION_GEOLOGY_EDGE + localX)];
    };
    worldgen::MacroControlView controls = macroSampler_.controlView(chunkColumn);
    if (!retainInCache && alternateCache == nullptr) {
        return std::make_shared<ColumnPlan>(chunkColumn, sampleSurface, sampleHeight,
                                            sampleExactSurface, sampleHydrology, sampleGeology,
                                            std::move(controls));
    }
    ColumnPlanCache& targetCache = alternateCache != nullptr ? *alternateCache : columnPlanCache_;
    return targetCache.getOrCreate(chunkColumn, sampleSurface, sampleHeight, sampleExactSurface,
                                   sampleHydrology, sampleGeology, std::move(controls));
}

std::shared_ptr<const ColumnPlan> ChunkGenerator::findColumnPlan(ColumnPos chunkColumn) const {
    return columnPlanCache_.find(chunkColumn);
}

worldgen::SurfaceSample ChunkGenerator::surfaceSampleFromPlan(int64_t x, int64_t z,
                                                              const ColumnPlan& plan,
                                                              GenScratch& scratch) const {
    const int localX = static_cast<int>(x - plan.chunkColumn().x * CHUNK_EDGE);
    const int localZ = static_cast<int>(z - plan.chunkColumn().z * CHUNK_EDGE);
    worldgen::SurfaceSample result = plan.sample(localX, localZ);
    if (learnedAuthority_) {
        // ColumnPlan keeps exact categorical geology compact, while its
        // retained eight-block lattice supplies the legacy continuous fields.
        // The latter must not leak into v4 ecology or density decisions: the
        // learned authority has no control-grid geology phase. Restore the
        // coordinate-pure full geology sample before any downstream consumer
        // observes the exact column.
        result.geology =
            macroSampler_.sampleGeology(static_cast<double>(x), static_cast<double>(z));
        // ColumnPlan restores the canonical block-resolution hydrology floor
        // after reconstructing its compact lattice. Apply the bounded dry
        // residual at the final coordinate as well, rather than interpolating
        // it through that lattice or silently omitting it from exact terrain.
        const double previousTerrainHeight = result.terrainHeight;
        macroSampler_.applyV4SurfaceDetail(static_cast<double>(x), static_cast<double>(z),
                                           worldgen::SurfaceFootprint::BLOCK_1, result);
        const double terrainHeightDelta = result.terrainHeight - previousTerrainHeight;
        if (std::abs(terrainHeightDelta) > 1.0e-12) {
            // The compact plan reconstructed learned climate at the canonical
            // routing floor. Bring its derived values to the same final dry
            // elevation used by direct and far samples without issuing one
            // scalar learned-authority lookup per generated column.
            result.climate.temperatureC += terrainHeightDelta *
                                           worldgen::learned::WORLD_METERS_PER_BLOCK *
                                           result.climate.lapseRateCPerMeter;
            result.climate.potentialEvapotranspirationMm =
                std::clamp(300.0 + std::max(-8.0, result.climate.temperatureC) * 31.0 +
                               result.climate.temperatureVariabilityC * 8.0,
                           120.0, 1'800.0);
            result.climate.relativeHumidity = std::clamp(
                result.climate.annualPrecipitationMm /
                    std::max(1.0,
                             result.climate.annualPrecipitationMm +
                                 result.climate.potentialEvapotranspirationMm *
                                     (0.75 + result.climate.precipitationCoefficientOfVariation)),
                0.0, 1.0);
            result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                                     std::max(1.0, result.climate.annualPrecipitationMm);
        }
    }
    result = applyVolcanism(x, z, std::move(result), scratch);
    // Suitability is nonlinear in final terrain, hydrology, and climate.
    // Recompute it after block-resolution terrain and volcanic deformation
    // instead of preserving the control lattice's interpolated ranking.
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    return refreshDependentSurface(x, z, std::move(result), volcanism);
}

worldgen::SurfaceSample ChunkGenerator::surfaceSampleAt(int64_t x, int64_t z,
                                                        GenScratch& scratch) const {
    const auto sampleUnfinalized = [&](int64_t sampleX, int64_t sampleZ) {
        const ColumnPos chunkColumn{
            world_coord::floorDiv(sampleX, static_cast<int64_t>(CHUNK_EDGE)),
            world_coord::floorDiv(sampleZ, static_cast<int64_t>(CHUNK_EDGE)),
        };
        auto found = scratch.columnPlans.find(chunkColumn);
        if (found == scratch.columnPlans.end()) {
            found = scratch.columnPlans.emplace(chunkColumn, getColumnPlan(chunkColumn)).first;
        }
        worldgen::SurfaceSample sample =
            surfaceSampleFromPlan(sampleX, sampleZ, *found->second, scratch);
        applyEmittedSurfaceTopology(
            sample,
            static_cast<double>(found->second->surfaceY(Chunk::worldToLocal(sampleX),
                                                        Chunk::worldToLocal(sampleZ)) +
                                1),
            learnedAuthority_);
        return sample;
    };
    worldgen::SurfaceSample result = sampleUnfinalized(x, z);
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    return refreshDependentSurface(x, z, std::move(result), volcanism);
}

worldgen::SurfaceSample ChunkGenerator::sampleSurface(int64_t x, int64_t z) const {
    return surfaceSampleAt(x, z, threadScratch());
}

worldgen::SurfaceSample ChunkGenerator::sampleSurface(int64_t x, int64_t z,
                                                      worldgen::SurfaceFootprint footprint) const {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_1) return sampleSurface(x, z);
    return sampleFarSurface(x, z, footprint);
}

worldgen::SurfaceSample ChunkGenerator::sampleFarGeometrySurface(int64_t x, int64_t z) const {
    return sampleFarGeometrySurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
}

worldgen::SurfaceSample
ChunkGenerator::sampleFarGeometrySurface(int64_t x, int64_t z,
                                         worldgen::SurfaceFootprint footprint) const {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_1) {
        const std::array<ColumnPos, 1> positions = {ColumnPos{x, z}};
        std::array<worldgen::SurfaceSample, 1> output{};
        macroSampler_.sampleGeometryPoints(positions, footprint, output);
        GenScratch& scratch = threadScratch();
        const VolcanicColumnSample& volcanism = volcanismAt(x, z, output.front().geology, scratch);
        applyVolcanicGeometry(output.front(), volcanism);
        return output.front();
    }
    worldgen::SurfaceSample result;
    result.geology = macroSampler_.sampleGeology(static_cast<double>(x), static_cast<double>(z));
    const worldgen::SurfaceSample filtered =
        macroSampler_.sampleSurface(static_cast<double>(x), static_cast<double>(z), footprint);
    result.hydrology = filtered.hydrology;
    result.terrainHeight = filtered.terrainHeight;
    result.waterSurface = filtered.waterSurface;
    GenScratch& scratch = threadScratch();
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    applyVolcanicGeometry(result, volcanism);
    return result;
}

void ChunkGenerator::sampleFarGeometryGrid(int64_t originX, int64_t originZ, int spacingX,
                                           int spacingZ, int sampleWidth, int sampleHeight,
                                           worldgen::SurfaceFootprint footprint,
                                           std::span<worldgen::SurfaceSample> output) const {
    macroSampler_.sampleGeometryGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                     sampleHeight, footprint, output);
    GenScratch& scratch = threadScratch();
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacingX;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacingZ;
            const VolcanicColumnSample& volcanism =
                volcanismAt(worldX, worldZ, output[index].geology, scratch);
            applyVolcanicGeometry(output[index], volcanism);
        }
    }
    if (!learnedAuthority_ && footprint == worldgen::SurfaceFootprint::BLOCK_1 && spacingX == 1 &&
        spacingZ == 1) {
        // Legacy macro raster routes can terminate at the global ocean after
        // a multi-block stage drop. Preserve both wet bodies and represent
        // the discontinuity as an explicit fall, rather than publishing an
        // untagged vertical water face in the geometry-only far grid.
        const std::vector<worldgen::SurfaceSample> unfinalized(output.begin(), output.end());
        const auto sampleUnfinalized = [&](int64_t x, int64_t z) {
            const int64_t localX = x - originX;
            const int64_t localZ = z - originZ;
            if (localX < 0 || localZ < 0 || localX >= sampleWidth || localZ >= sampleHeight) {
                return worldgen::SurfaceSample{};
            }
            return unfinalized[static_cast<size_t>(localZ * sampleWidth + localX)];
        };
        for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
                output[index] =
                    finalizeCoordinateWetContact(originX + sampleX, originZ + sampleZ,
                                                 std::move(output[index]), sampleUnfinalized, true);
            }
        }
    }
}

void ChunkGenerator::sampleFarGeometryPoints(std::span<const ColumnPos> positions,
                                             worldgen::SurfaceFootprint footprint,
                                             std::span<worldgen::SurfaceSample> output) const {
    macroSampler_.sampleGeometryPoints(positions, footprint, output);
    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < positions.size(); ++index) {
        const VolcanicColumnSample& volcanism =
            volcanismAt(positions[index].x, positions[index].z, output[index].geology, scratch);
        applyVolcanicGeometry(output[index], volcanism);
    }
}

void ChunkGenerator::sampleNativeHydrologyGeometryGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<worldgen::SurfaceSample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid native hydrology geometry grid");
    }
    std::vector<worldgen::HydrologySample> hydrology(output.size());
    macroSampler_.sampleHydrologyGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                      sampleHeight, hydrology);
    GenScratch& scratch = threadScratch();
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacingX;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacingZ;
            worldgen::SurfaceSample surface;
            surface.geology = macroSampler_.sampleGeology(static_cast<double>(worldX),
                                                          static_cast<double>(worldZ));
            surface.hydrology = hydrology[index];
            surface.terrainHeight = surface.hydrology.surfaceElevation;
            surface.waterSurface = surface.hydrology.waterSurface;
            applyVolcanicGeometry(surface, volcanismAt(worldX, worldZ, surface.geology, scratch));
            output[index] = std::move(surface);
        }
    }
}

void ChunkGenerator::sampleNativeHydrologyGeometryPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid native hydrology geometry points");
    }
    if (positions.empty()) return;
    std::vector<worldgen::HydrologySample> hydrology(output.size());
    macroSampler_.sampleHydrologyPoints(positions, hydrology);
    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        worldgen::SurfaceSample surface;
        surface.geology = macroSampler_.sampleGeology(static_cast<double>(position.x),
                                                      static_cast<double>(position.z));
        surface.hydrology = hydrology[index];
        surface.terrainHeight = surface.hydrology.surfaceElevation;
        surface.waterSurface = surface.hydrology.waterSurface;
        applyVolcanicGeometry(surface,
                              volcanismAt(position.x, position.z, surface.geology, scratch));
        output[index] = std::move(surface);
    }
}

void ChunkGenerator::sampleNativeHydrologyAuthorityGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<worldgen::HydrologySample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid native hydrology authority grid");
    }
    macroSampler_.sampleHydrologyGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                      sampleHeight, output);
}

void ChunkGenerator::sampleNativeHydrologyAuthorityPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::HydrologySample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid native hydrology authority points");
    }
    macroSampler_.sampleHydrologyPoints(positions, output);
}

void ChunkGenerator::sampleCoarseNativeHydrologyAuthorityGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<worldgen::HydrologySample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid coarse native hydrology authority grid");
    }
    if (spacingX == 1 && spacingZ == 1) {
        // Preserve every requested block coordinate while keeping each router
        // call on a two-block lattice. Four parity grids exactly partition the
        // original rectangle, including negative origins and odd dimensions.
        for (int parityZ = 0; parityZ < 2; ++parityZ) {
            const int height = (sampleHeight - parityZ + 1) / 2;
            if (height <= 0) continue;
            for (int parityX = 0; parityX < 2; ++parityX) {
                const int width = (sampleWidth - parityX + 1) / 2;
                if (width <= 0) continue;
                std::vector<worldgen::HydrologySample> parity(static_cast<size_t>(width * height));
                macroSampler_.sampleHydrologyGrid(originX + parityX, originZ + parityZ, 2, 2, width,
                                                  height, parity);
                for (int z = 0; z < height; ++z) {
                    for (int x = 0; x < width; ++x) {
                        output[static_cast<size_t>((z * 2 + parityZ) * sampleWidth + x * 2 +
                                                   parityX)] =
                            parity[static_cast<size_t>(z * width + x)];
                    }
                }
            }
        }
        return;
    }
    macroSampler_.sampleHydrologyGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                      sampleHeight, output);
}

void ChunkGenerator::sampleCoarseNativeHydrologyAuthorityPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::HydrologySample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid coarse native hydrology authority points");
    }
    macroSampler_.sampleCoarseHydrologyPoints(positions, output);
}

void ChunkGenerator::sampleCoarseNativeHydrologyGeometryGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<worldgen::SurfaceSample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid coarse native hydrology geometry grid");
    }
    std::vector<worldgen::HydrologySample> hydrology(output.size());
    sampleCoarseNativeHydrologyAuthorityGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                             sampleHeight, hydrology);
    GenScratch& scratch = threadScratch();
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacingX;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacingZ;
            worldgen::SurfaceSample surface;
            surface.geology = macroSampler_.sampleGeology(static_cast<double>(worldX),
                                                          static_cast<double>(worldZ));
            surface.hydrology = hydrology[index];
            surface.terrainHeight = surface.hydrology.surfaceElevation;
            surface.waterSurface = surface.hydrology.waterSurface;
            applyVolcanicGeometry(surface, volcanismAt(worldX, worldZ, surface.geology, scratch));
            output[index] = std::move(surface);
        }
    }
}

void ChunkGenerator::sampleCoarseNativeHydrologyGeometryPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::SurfaceSample> output) const {
    sampleStageFreePointRuns<worldgen::SurfaceSample>(
        positions, output,
        [this](int64_t originX, int64_t originZ, int spacingX, int spacingZ, int width, int height,
               std::span<worldgen::SurfaceSample> destination) {
            sampleCoarseNativeHydrologyGeometryGrid(originX, originZ, spacingX, spacingZ, width,
                                                    height, destination);
        });
}

void ChunkGenerator::sampleNativeHydrologyTopologyGrid(
    int64_t originX, int64_t originZ, int cellWidth, int cellHeight,
    std::span<worldgen::NativeHydrologyTopologyCell> output) const {
    if (cellWidth <= 0 || cellHeight <= 0 ||
        output.size() != static_cast<size_t>(cellWidth) * static_cast<size_t>(cellHeight)) {
        throw std::invalid_argument("invalid native hydrology topology grid");
    }
    macroSampler_.sampleNativeHydrologyTopologyGrid(originX, originZ, cellWidth, cellHeight,
                                                    output);
}

void ChunkGenerator::markVolcanicWaterCandidates(int64_t originX, int64_t originZ, int step,
                                                 int cellWidth, int cellHeight,
                                                 std::span<uint8_t> output) const {
    if (step <= 0 || cellWidth <= 0 || cellHeight <= 0 ||
        output.size() != static_cast<size_t>(cellWidth) * static_cast<size_t>(cellHeight)) {
        throw std::invalid_argument("invalid volcanic far-water candidate grid");
    }
    std::fill(output.begin(), output.end(), uint8_t{0});

    // V4 volcanic relief is already part of the model-native elevation passed
    // into the canonical router. Its compact topology grid therefore sees a
    // caldera lake even when every step-32 display sample misses it. Running
    // the legacy analytical crater search here would duplicate that proof,
    // request unrelated crater datums, and promote cells to an exact callback
    // that cannot change v4 terrain or water.
    if (learnedAuthority_) return;

    const auto coordinateAt = [](int64_t origin, int index, int spacing) {
        const __int128 value =
            static_cast<__int128>(origin) + static_cast<__int128>(index) * spacing;
        if (value < std::numeric_limits<int64_t>::min() ||
            value > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("volcanic far-water candidate coordinate exceeds int64 range");
        }
        return static_cast<int64_t>(value);
    };
    const int64_t maximumX = coordinateAt(originX, cellWidth, step);
    const int64_t maximumZ = coordinateAt(originZ, cellHeight, step);
    const auto circleTouches = [](const VolcanoPrimitive& volcano, double radius, int64_t minimumX,
                                  int64_t minimumZ, int64_t maximumXExclusive,
                                  int64_t maximumZExclusive) {
        // Volcanic columns sample at block centers and crater warping can
        // stretch the radial shore by 12 percent. Expand the half-open cell
        // slightly so this conservative filter cannot omit a legal lake cell.
        const double closestX = std::clamp(volcano.centerX, static_cast<double>(minimumX) - 1.0,
                                           static_cast<double>(maximumXExclusive) + 1.0);
        const double closestZ = std::clamp(volcano.centerZ, static_cast<double>(minimumZ) - 1.0,
                                           static_cast<double>(maximumZExclusive) + 1.0);
        return std::hypot(volcano.centerX - closestX, volcano.centerZ - closestZ) <= radius;
    };

    GenScratch& scratch = threadScratch();
    const auto markPrimitive = [&](const VolcanoPrimitive& primitive) {
        if (!primitive.craterLake || primitive.craterRadius <= 2.0 ||
            !circleTouches(primitive, primitive.craterRadius * 1.12 + 1.0, originX, originZ,
                           maximumX, maximumZ)) {
            return;
        }
        const VolcanoPrimitive& volcano = resolvedVolcano(primitive, scratch);
        if (!volcano.craterLake || volcano.craterLakeRadius <= 2.0) return;
        const double wetRadius = volcano.craterLakeRadius * 1.12 + 1.0;
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            const int64_t minimumCellZ = coordinateAt(originZ, cellZ, step);
            const int64_t maximumCellZ = coordinateAt(originZ, cellZ + 1, step);
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t minimumCellX = coordinateAt(originX, cellX, step);
                const int64_t maximumCellX = coordinateAt(originX, cellX + 1, step);
                if (!circleTouches(volcano, wetRadius, minimumCellX, minimumCellZ, maximumCellX,
                                   maximumCellZ)) {
                    continue;
                }
                output[static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth) +
                       static_cast<size_t>(cellX)] = 1;
            }
        }
    };
    const auto visitPrimitiveCells = [&](int64_t cellEdge, auto&& visitor) {
        const int64_t firstX = world_coord::floorDiv(originX, cellEdge);
        const int64_t lastX = world_coord::floorDiv(maximumX - 1, cellEdge);
        const int64_t firstZ = world_coord::floorDiv(originZ, cellEdge);
        const int64_t lastZ = world_coord::floorDiv(maximumZ - 1, cellEdge);
        for (int64_t cellZ = firstZ - 1; cellZ <= lastZ + 1; ++cellZ) {
            for (int64_t cellX = firstX - 1; cellX <= lastX + 1; ++cellX) {
                visitor(cellX, cellZ);
            }
        }
    };
    visitPrimitiveCells(worldgen::HOTSPOT_LATTICE_EDGE, [&](int64_t cellX, int64_t cellZ) {
        for (const VolcanoPrimitive& primitive : volcanoesForCell(cellX, cellZ, scratch))
            markPrimitive(primitive);
    });
    visitPrimitiveCells(VOLCANIC_ARC_CELL_EDGE, [&](int64_t cellX, int64_t cellZ) {
        for (const VolcanoPrimitive& primitive : volcanicArcForCell(cellX, cellZ, scratch))
            markPrimitive(primitive);
    });
}

worldgen::SurfaceSample ChunkGenerator::sampleExactGeometrySurface(int64_t x, int64_t z) const {
    return sampleExactSurface(x, z);
}

double ChunkGenerator::emittedSurfaceDetailAmplitude(const worldgen::SurfaceSample& surface,
                                                     double slopeEnvelope) {
    return surfaceDetailAmplitude(surface, std::max(0.0, slopeEnvelope));
}

worldgen::SurfaceSample ChunkGenerator::sampleExactSurface(int64_t x, int64_t z) const {
    return surfaceSampleAt(x, z, threadScratch());
}

void ChunkGenerator::sampleExactSurfaceGrid(int64_t originX, int64_t originZ, int spacing,
                                            int sampleEdge,
                                            std::span<worldgen::SurfaceSample> output) const {
    if (spacing <= 0 || sampleEdge <= 0 ||
        output.size() != static_cast<size_t>(sampleEdge * sampleEdge)) {
        throw std::invalid_argument("invalid exact surface grid");
    }
    if (learnedAuthority_ && spacing == 1) {
        samplePlanFreeExactSurfaceGrid(originX, originZ, sampleEdge, output);
        return;
    }
    GenScratch& scratch = threadScratch();
    for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
            output[static_cast<size_t>(sampleZ * sampleEdge + sampleX)] =
                surfaceSampleAt(worldX, worldZ, scratch);
        }
    }
}

void ChunkGenerator::samplePlanFreeExactSurfaceGrid(
    int64_t originX, int64_t originZ, int sampleEdge,
    std::span<worldgen::SurfaceSample> output) const {
    if (!learnedAuthority_ || sampleEdge <= 0 ||
        output.size() != static_cast<size_t>(sampleEdge * sampleEdge)) {
        throw std::invalid_argument("invalid plan-free exact surface grid");
    }

    // One regular macro and hydrology batch supplies every requested column.
    // The density field then retains only the globally aligned lattice columns
    // that bound this square. This is the dense counterpart to the sparse
    // habitat evaluator and avoids both per-column hash tables and 16-block
    // ColumnPlan construction.
    macroSampler_.sampleSurfaceGrid(originX, originZ, 1, sampleEdge,
                                    worldgen::SurfaceFootprint::BLOCK_1, output);
    GenScratch& scratch = threadScratch();
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const size_t index = static_cast<size_t>(z * sampleEdge + x);
            output[index] =
                applyVolcanism(originX + x, originZ + z, std::move(output[index]), scratch);
        }
    }

    const auto checkedLastCoordinate = [sampleEdge](int64_t origin) {
        const __int128 last = static_cast<__int128>(origin) + sampleEdge - 1;
        if (last > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("plan-free exact grid exceeds int64 coordinates");
        }
        return static_cast<int64_t>(last);
    };
    const int64_t lastX = checkedLastCoordinate(originX);
    const int64_t lastZ = checkedLastCoordinate(originZ);
    const int64_t latticeOriginX = latticeFloor(originX);
    const int64_t latticeOriginZ = latticeFloor(originZ);
    const int64_t latticeLastX = latticeFloor(lastX) + LATTICE_XZ;
    const int64_t latticeLastZ = latticeFloor(lastZ) + LATTICE_XZ;
    const int latticeWidth = static_cast<int>((latticeLastX - latticeOriginX) / LATTICE_XZ + 1);
    const int latticeHeight = static_cast<int>((latticeLastZ - latticeOriginZ) / LATTICE_XZ + 1);

    std::vector<ColumnPos> latticePositions;
    latticePositions.reserve(static_cast<size_t>(latticeWidth * latticeHeight));
    for (int z = 0; z < latticeHeight; ++z) {
        for (int x = 0; x < latticeWidth; ++x) {
            latticePositions.push_back({latticeOriginX + static_cast<int64_t>(x) * LATTICE_XZ,
                                        latticeOriginZ + static_cast<int64_t>(z) * LATTICE_XZ});
        }
    }
    std::vector<worldgen::SurfaceSample> latticeSurfaces(latticePositions.size());
    macroSampler_.sampleSurfacePoints(latticePositions, worldgen::SurfaceFootprint::BLOCK_1,
                                      latticeSurfaces);
    for (size_t index = 0; index < latticeSurfaces.size(); ++index) {
        latticeSurfaces[index] =
            applyVolcanism(latticePositions[index].x, latticePositions[index].z,
                           std::move(latticeSurfaces[index]), scratch);
    }

    struct DensityColumn {
        ColumnShape shape;
        DensityColumnContext context;
        std::vector<double> values;
    };
    const int verticalSpacing = densityLatticeVerticalSpacing();
    const int levelCount = densityLatticeLevelCount();
    std::vector<DensityColumn> densityColumns;
    densityColumns.reserve(latticeSurfaces.size());
    for (size_t index = 0; index < latticeSurfaces.size(); ++index) {
        const ColumnPos position = latticePositions[index];
        const worldgen::SurfaceSample& surface = latticeSurfaces[index];
        ColumnShape shape = shapeFromSurface(
            surface,
            climate_.caveEntrance(static_cast<double>(position.x), static_cast<double>(position.z)),
            learnedAuthority_);
        const int maximumLevel =
            maximumEvaluatedDensityLevel(shape.height + shape.detailAmp + 5.0, verticalSpacing);
        densityColumns.push_back({
            .shape = std::move(shape),
            .context = density_.columnContext(static_cast<double>(position.x),
                                              static_cast<double>(position.z), surface.geology),
            .values = std::vector<double>(static_cast<size_t>(std::max(0, maximumLevel + 1)),
                                          std::numeric_limits<double>::quiet_NaN()),
        });
    }
    const auto latticeIndex = [&](int64_t x, int64_t z) {
        const size_t localX = static_cast<size_t>((x - latticeOriginX) / LATTICE_XZ);
        const size_t localZ = static_cast<size_t>((z - latticeOriginZ) / LATTICE_XZ);
        return localZ * static_cast<size_t>(latticeWidth) + localX;
    };
    const auto densityValue = [&](size_t index, int level) {
        DensityColumn& column = densityColumns[index];
        const size_t bounded = static_cast<size_t>(std::clamp(level, 0, levelCount - 1));
        if (bounded >= column.values.size()) return -DENSITY_CAP;
        double& value = column.values[bounded];
        if (std::isnan(value)) {
            const ColumnPos position = latticePositions[index];
            const double y =
                static_cast<double>(WORLD_MIN_Y + static_cast<int>(bounded) * verticalSpacing);
            value = evaluateDensity(static_cast<double>(position.x), y,
                                    static_cast<double>(position.z), column.shape, column.context);
        }
        return value;
    };

    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const size_t outputIndex = static_cast<size_t>(z * sampleEdge + x);
            const int64_t worldX = originX + x;
            const int64_t worldZ = originZ + z;
            worldgen::SurfaceSample& surface = output[outputIndex];
            const ColumnShape rootShape = shapeFromSurface(
                surface,
                climate_.caveEntrance(static_cast<double>(worldX), static_cast<double>(worldZ)),
                learnedAuthority_);
            const int start = std::clamp(
                static_cast<int>(std::ceil(rootShape.height + rootShape.detailAmp + 8.0)),
                WORLD_MIN_Y + 2, WORLD_MAX_Y);
            const int64_t latticeX = latticeFloor(worldX);
            const int64_t latticeZ = latticeFloor(worldZ);
            const double fx = static_cast<double>(worldX - latticeX) / LATTICE_XZ;
            const double fz = static_cast<double>(worldZ - latticeZ) / LATTICE_XZ;
            const std::array<size_t, 4> corners{
                latticeIndex(latticeX, latticeZ),
                latticeIndex(latticeX + LATTICE_XZ, latticeZ),
                latticeIndex(latticeX, latticeZ + LATTICE_XZ),
                latticeIndex(latticeX + LATTICE_XZ, latticeZ + LATTICE_XZ),
            };
            int surfaceY = WORLD_MIN_Y + 1;
            const int startLevel = (start - WORLD_MIN_Y) / verticalSpacing;
            for (int level = startLevel; level >= 0; --level) {
                const int baseY = WORLD_MIN_Y + level * verticalSpacing;
                const int minimumY = std::max(WORLD_MIN_Y + 2, baseY);
                const int maximumY = std::min(start, baseY + verticalSpacing - 1);
                if (minimumY > maximumY) continue;
                const double below = bilerpDensity(
                    densityValue(corners[0], level), densityValue(corners[1], level),
                    densityValue(corners[2], level), densityValue(corners[3], level), fx, fz);
                const double above = bilerpDensity(densityValue(corners[0], level + 1),
                                                   densityValue(corners[1], level + 1),
                                                   densityValue(corners[2], level + 1),
                                                   densityValue(corners[3], level + 1), fx, fz);
                const auto interpolated = [&](int y) {
                    return lerpDensity(below, above,
                                       static_cast<double>(y - baseY) / verticalSpacing);
                };
                if (interpolated(maximumY) > 0.0) {
                    surfaceY = maximumY;
                    break;
                }
                if (maximumY == minimumY || interpolated(minimumY) <= 0.0) continue;
                for (int y = maximumY - 1; y >= minimumY; --y) {
                    if (interpolated(y) > 0.0) {
                        surfaceY = y;
                        break;
                    }
                }
                if (surfaceY >= minimumY) break;
            }

            const VolcanicColumnSample& volcanism =
                volcanismAt(worldX, worldZ, surface.geology, scratch);
            surfaceY = boundedLearnedSurfaceY(surface, surfaceY, learnedAuthority_);
            if (!learnedAuthority_ && !hasCanonicalWaterFeature(surface.hydrology) &&
                volcanism.craterLake &&
                volcanism.craterProfileInfluence > 1.0e-4) {
                surfaceY = std::clamp(static_cast<int>(std::ceil(surface.terrainHeight)) - 1,
                                      WORLD_MIN_Y, WORLD_MAX_Y);
            }
            reconcileEmittedSurfaceY(surface.hydrology, surfaceY, learnedAuthority_);
            applyEmittedSurfaceTopology(surface, static_cast<double>(surfaceY + 1),
                                        learnedAuthority_);
            surface = refreshDependentSurface(worldX, worldZ, std::move(surface), volcanism);
        }
    }
}

worldgen::SurfaceSample ChunkGenerator::sampleFarSurface(int64_t x, int64_t z) const {
    return sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
}

worldgen::SurfaceSample
ChunkGenerator::sampleFarSurface(int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) const {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_1) return sampleSurface(x, z);
    worldgen::SurfaceSample result = applyVolcanism(
        x, z,
        macroSampler_.sampleSurface(static_cast<double>(x), static_cast<double>(z), footprint),
        threadScratch());

    // v3's filtered macro sample remains its own geometry and water contract.
    // Only v4 has a separately persisted canonical water authority that must
    // override every coarse terrain footprint. Applying that override to v3
    // paired a block-resolution lake stage with a filtered lake bed and left
    // stale depth metadata at far LODs.
    if (!learnedAuthority_) return result;

    const worldgen::SurfaceSample exact = sampleExactSurface(x, z);
    result.hydrology = exact.hydrology;
    result.waterSurface = exact.waterSurface;
    if (hasSurfaceWater(exact.hydrology)) {
        result.terrainHeight = std::min(result.terrainHeight, exact.terrainHeight);
        result.hydrology.surfaceElevation = result.terrainHeight;
    }
    if (result.hydrology.lake) {
        result.hydrology.lakeDepth =
            std::max(0.0, result.hydrology.waterSurface - result.terrainHeight);
    }
    return result;
}

void ChunkGenerator::sampleCanonicalWaterSurfacePoints(
    std::span<const ColumnPos> positions, std::span<worldgen::SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("canonical water point output has the wrong size");
    }
    // Far water needs the same emitted terrain contact as exact cubes, but a
    // tile-wide query must not construct and retain hundreds of complete
    // 16-by-16 column plans. The sparse evaluator computes only the density
    // levels crossed by each requested coordinate and finalizes the same
    // canonical water authority in one deterministic batch.
    if (positions.empty()) return;

    // V4's native router already owns every legal waterfall transition. It
    // therefore needs only requested positions here, rather than the four
    // synthetic wet-contact probes formerly used to invent a local fall.
    if (learnedAuthority_) {
        sampleFarHabitatPoints(positions, output);
        return;
    }

    // Keep the legacy v3 batch geometry unchanged. Its retained neighbor set
    // is part of the existing sampling behavior even though v4 no longer uses
    // it to synthesize water transitions.
    std::unordered_map<ColumnPos, size_t> sampleIndices;
    sampleIndices.reserve(positions.size() * 5);
    std::vector<ColumnPos> samplePositions;
    samplePositions.reserve(positions.size() * 5);
    const auto retain = [&](ColumnPos position) {
        const auto [found, inserted] = sampleIndices.emplace(position, samplePositions.size());
        if (inserted) samplePositions.push_back(position);
        return found->second;
    };
    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};
    for (const ColumnPos position : positions) {
        retain(position);
        for (const auto [offsetX, offsetZ] : CARDINAL)
            retain({position.x + offsetX, position.z + offsetZ});
    }

    std::vector<worldgen::SurfaceSample> samples(samplePositions.size());
    sampleFarHabitatPoints(samplePositions, samples);
    for (size_t index = 0; index < positions.size(); ++index)
        output[index] = samples[sampleIndices.at(positions[index])];
}

worldgen::HydrologySample ChunkGenerator::sampleGeneratedWaterAuthority(int64_t x,
                                                                        int64_t z) const {
    if (!learnedAuthority_)
        return macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
    return sampleExactSurface(x, z).hydrology;
}

void ChunkGenerator::sampleGeneratedWaterAuthorityGrid(
    int64_t originX, int64_t originZ, int spacing, int sampleEdge,
    std::span<worldgen::HydrologySample> output) const {
    if (spacing <= 0 || sampleEdge <= 0 ||
        output.size() != static_cast<size_t>(sampleEdge * sampleEdge)) {
        throw std::invalid_argument("invalid generated water authority grid");
    }
    if (!learnedAuthority_) {
        macroSampler_.sampleHydrologyGrid(originX, originZ, spacing, spacing, sampleEdge,
                                          sampleEdge, output);
        return;
    }
    std::vector<ColumnPos> positions;
    positions.reserve(output.size());
    for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
            positions.push_back({originX + static_cast<int64_t>(sampleX) * spacing,
                                 originZ + static_cast<int64_t>(sampleZ) * spacing});
        }
    }
    sampleGeneratedWaterAuthorityPoints(positions, output);
}

void ChunkGenerator::sampleGeneratedWaterAuthorityPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::HydrologySample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("generated water authority point output has the wrong size");
    }
    if (positions.empty()) return;
    if (!learnedAuthority_) {
        macroSampler_.sampleHydrologyPoints(positions, output);
        return;
    }

    std::unordered_map<ColumnPos, size_t> sampleIndices;
    sampleIndices.reserve(positions.size() * 13);
    std::vector<ColumnPos> samplePositions;
    samplePositions.reserve(positions.size() * 13);
    const auto retain = [&](ColumnPos position) {
        const auto [found, inserted] = sampleIndices.emplace(position, samplePositions.size());
        if (inserted) samplePositions.push_back(position);
        return found->second;
    };
    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};
    std::vector<size_t> requestedIndices;
    requestedIndices.reserve(positions.size());
    for (const ColumnPos position : positions) {
        requestedIndices.push_back(retain(position));
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const ColumnPos neighbor{position.x + offsetX, position.z + offsetZ};
            retain(neighbor);
            for (const auto [secondOffsetX, secondOffsetZ] : CARDINAL) {
                retain({neighbor.x + secondOffsetX, neighbor.z + secondOffsetZ});
            }
        }
    }

    // Most far-grid points are dry macro terrain and cannot change water
    // ownership after voxel emission. Sample the complete coordinate-pure
    // authority first, then pay the lazy density cost for every requested wet
    // column, its banks, and dry columns close enough to open below the sea
    // plane. A standing or routed water cell must expose the same emitted bed
    // elevation as exact terrain, even when its current depth is greater than
    // one block. Otherwise a distant mesh can retain a correct stage above a
    // stale, fractional bed.
    std::vector<worldgen::SurfaceSample> samples(samplePositions.size());
    std::vector<worldgen::HydrologySample> macroHydrology(samplePositions.size());
    macroSampler_.sampleHydrologyPoints(samplePositions, macroHydrology);
    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < samples.size(); ++index) {
        const ColumnPos position = samplePositions[index];
        worldgen::SurfaceSample& sample = samples[index];
        sample.geology = macroSampler_.sampleGeology(static_cast<double>(position.x),
                                                     static_cast<double>(position.z));
        sample.hydrology = macroHydrology[index];
        sample.terrainHeight = sample.hydrology.surfaceElevation;
        sample.waterSurface = sample.hydrology.waterSurface;
        applyVolcanicGeometry(sample, volcanismAt(position.x, position.z, sample.geology, scratch));
    }
    constexpr double MAXIMUM_DRY_SURFACE_DETAIL = 14.0;
    std::vector<bool> exactMask(samplePositions.size(), false);
    for (size_t requestIndex = 0; requestIndex < positions.size(); ++requestIndex) {
        const size_t index = requestedIndices[requestIndex];
        const ColumnPos position = positions[requestIndex];
        const worldgen::SurfaceSample& sample = samples[index];
        const bool wet = hasSurfaceWater(sample.hydrology);
        bool breachedLakeBoundary = false;
        if (sample.hydrology.lake) {
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const worldgen::SurfaceSample& neighbor =
                    samples[sampleIndices.at({position.x + offsetX, position.z + offsetZ})];
                const bool differentBody =
                    !neighbor.hydrology.lake ||
                    neighbor.hydrology.waterBodyId != sample.hydrology.waterBodyId;
                if (differentBody && sample.waterSurface > neighbor.terrainHeight + 1.000001) {
                    breachedLakeBoundary = true;
                    break;
                }
            }
        }
        bool partialDropContact = false;
        if (wet && sample.hydrology.river && sample.hydrology.generatedFluidLevel > 0) {
            const double visibleSurface = worldgen::generatedFluidColumn(sample).visibleSurface;
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const worldgen::SurfaceSample& neighbor =
                    samples[sampleIndices.at({position.x + offsetX, position.z + offsetZ})];
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                if (neighborFluid.wet && !legalExplicitFall(neighbor) &&
                    visibleSurface > neighborFluid.visibleSurface + 1.000001) {
                    partialDropContact = true;
                    break;
                }
            }
        }
        const bool drySeaContact =
            !wet && sample.terrainHeight <= SEA_LEVEL + MAXIMUM_DRY_SURFACE_DETAIL + 1.0;
        const bool topologySensitive =
            wet || sample.hydrology.channelBank || sample.hydrology.lakeBank || drySeaContact;
        if (!topologySensitive) continue;
        exactMask[index] = true;
        if (breachedLakeBoundary || partialDropContact) {
            for (const auto [offsetX, offsetZ] : CARDINAL)
                exactMask[sampleIndices.at({position.x + offsetX, position.z + offsetZ})] = true;
        } else if (drySeaContact) {
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const size_t neighborIndex =
                    sampleIndices.at({position.x + offsetX, position.z + offsetZ});
                const worldgen::SurfaceSample& neighbor = samples[neighborIndex];
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                const double normalFlow = offsetX != 0
                                              ? std::abs(neighbor.hydrology.flowDirection.x)
                                              : std::abs(neighbor.hydrology.flowDirection.z);
                const double tangentialFlow = offsetX != 0
                                                  ? std::abs(neighbor.hydrology.flowDirection.z)
                                                  : std::abs(neighbor.hydrology.flowDirection.x);
                if (neighborFluid.wet && neighbor.hydrology.river &&
                    normalFlow + 1.0e-6 < tangentialFlow &&
                    neighborFluid.visibleSurface > sample.terrainHeight + 1.000001) {
                    exactMask[neighborIndex] = true;
                }
            }
        }
    }

    ThreadWaterContactCache& contactCache = threadWaterContactCache(this, scratch.ownerToken);
    std::vector<ColumnPos> missingExactPositions;
    std::vector<size_t> missingExactIndices;
    missingExactPositions.reserve(samplePositions.size() / 4);
    missingExactIndices.reserve(samplePositions.size() / 4);
    for (size_t index = 0; index < exactMask.size(); ++index) {
        if (!exactMask[index]) continue;
        const auto cached = contactCache.surfaces.find(samplePositions[index]);
        if (cached != contactCache.surfaces.end()) {
            samples[index] = cached->second;
            continue;
        }
        missingExactPositions.push_back(samplePositions[index]);
        missingExactIndices.push_back(index);
    }
    if (!missingExactPositions.empty()) {
        std::vector<worldgen::SurfaceSample> exactSamples(missingExactPositions.size());
        sampleFarHabitatPoints(missingExactPositions, exactSamples);
        for (size_t index = 0; index < exactSamples.size(); ++index)
            samples[missingExactIndices[index]] = exactSamples[index];
        for (size_t index = 0; index < exactSamples.size(); ++index)
            contactCache.surfaces.insert_or_assign(missingExactPositions[index],
                                                   std::move(exactSamples[index]));
    }

    for (size_t index = 0; index < output.size(); ++index) {
        const ColumnPos position = positions[index];
        output[index] = samples[sampleIndices.at(position)].hydrology;
    }
}

void ChunkGenerator::sampleGeneratedWaterGeometryGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<worldgen::SurfaceSample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid generated water geometry grid");
    }
    std::vector<ColumnPos> positions;
    positions.reserve(output.size());
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            positions.push_back({originX + static_cast<int64_t>(sampleX) * spacingX,
                                 originZ + static_cast<int64_t>(sampleZ) * spacingZ});
        }
    }
    sampleCanonicalWaterSurfacePoints(positions, output);
}

void ChunkGenerator::sampleGeneratedWaterGeometryPoints(
    std::span<const ColumnPos> positions, std::span<worldgen::SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("generated water geometry point output has the wrong size");
    }
    sampleCanonicalWaterSurfacePoints(positions, output);
}

void ChunkGenerator::sampleFarSurfacePoints(std::span<const ColumnPos> positions,
                                            worldgen::SurfaceFootprint footprint,
                                            std::span<worldgen::SurfaceSample> output) const {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_1) {
        if (positions.size() != output.size()) {
            throw std::invalid_argument("surface point sample output has the wrong size");
        }
        std::unordered_map<ColumnPos, std::shared_ptr<const ColumnPlan>> retainedPlans;
        retainedPlans.reserve(positions.size() * 2);
        GenScratch& scratch = threadScratch();
        const auto sampleUnfinalized = [&](int64_t x, int64_t z) {
            const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
            auto [plan, inserted] = retainedPlans.try_emplace(column);
            if (inserted) plan->second = constructColumnPlan(column, false);
            worldgen::SurfaceSample result = surfaceSampleFromPlan(x, z, *plan->second, scratch);
            applyEmittedSurfaceTopology(
                result,
                static_cast<double>(
                    plan->second->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z)) + 1),
                learnedAuthority_);
            return result;
        };
        for (size_t index = 0; index < positions.size(); ++index) {
            const ColumnPos position = positions[index];
            worldgen::SurfaceSample result = sampleUnfinalized(position.x, position.z);
            const VolcanicColumnSample& volcanism =
                volcanismAt(position.x, position.z, result.geology, scratch);
            output[index] =
                refreshDependentSurface(position.x, position.z, std::move(result), volcanism);
        }
        return;
    }
    macroSampler_.sampleSurfacePoints(positions, footprint, output);
    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < positions.size(); ++index) {
        output[index] = applyVolcanism(positions[index].x, positions[index].z,
                                       std::move(output[index]), scratch);
    }
}

void ChunkGenerator::sampleFarEcologyPoints(std::span<const ColumnPos> positions, int lodStep,
                                            std::span<worldgen::SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("far ecology point output has the wrong size");
    }
    if (!previewAuthority_ || lodStep < 4) {
        sampleFarHabitatPoints(positions, output);
        return;
    }
    if (positions.empty()) return;

    const worldgen::SurfaceFootprint footprint =
        lodStep <= 4    ? worldgen::SurfaceFootprint::BLOCK_4
        : lodStep <= 8  ? worldgen::SurfaceFootprint::BLOCK_8
        : lodStep <= 16 ? worldgen::SurfaceFootprint::BLOCK_16
                        : worldgen::SurfaceFootprint::BLOCK_32;
    macroSampler_.samplePreviewEcologyPoints(positions, footprint, output);

    // Every candidate shares one stage-free point batch. The router retains
    // immutable native pages across the complete call, preserving exact root
    // wet/dry classification without per-row passes or ordinary-stage tiles.
    std::vector<worldgen::HydrologySample> coarseHydrology(positions.size());
    sampleCoarseNativeHydrologyAuthorityPoints(positions, coarseHydrology);

    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        worldgen::SurfaceSample& surface = output[index];
        surface.hydrology = coarseHydrology[index];
        surface.waterSurface = surface.hydrology.waterSurface;
        surface.slope = std::max(0.0, surface.hydrology.terrainSlope);
        if (hasCanonicalWaterFeature(surface.hydrology)) {
            surface.terrainHeight = std::min(surface.terrainHeight,
                                             surface.hydrology.surfaceElevation);
        }
        surface.hydrology.surfaceElevation = surface.terrainHeight;
        const VolcanicColumnSample& volcanism =
            volcanismAt(position.x, position.z, surface.geology, scratch);
        surface = applyVolcanism(position.x, position.z, std::move(surface), scratch);
        surface = refreshDependentSurface(position.x, position.z, std::move(surface), volcanism);
    }
}

void ChunkGenerator::sampleFarHabitatPoints(std::span<const ColumnPos> positions,
                                            std::span<worldgen::SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("habitat point sample output has the wrong size");
    }
    if (positions.empty()) return;
    const int verticalDensitySpacing = densityLatticeVerticalSpacing();
    const int densityLatticeLevels = densityLatticeLevelCount();

    // Sparse tree roots need the emitted block top, but constructing a full
    // 16 by 16 ColumnPlan for every occupied candidate column stalls coarse
    // parent publication. Batch the coordinate-pure macro authority for the
    // roots and their four density-lattice corners, then evaluate only the
    // vertical density samples that a requested root actually crosses.
    std::unordered_map<ColumnPos, size_t> sampleIndices;
    sampleIndices.reserve(positions.size() * 5);
    std::vector<ColumnPos> samplePositions;
    samplePositions.reserve(positions.size() * 5);
    std::unordered_map<ColumnPos, size_t> hydrologyIndices;
    hydrologyIndices.reserve(positions.size() * 9);
    std::vector<ColumnPos> hydrologyPositions;
    hydrologyPositions.reserve(positions.size() * 9);
    const auto retainHydrology = [&](ColumnPos position) {
        const auto [found, inserted] =
            hydrologyIndices.emplace(position, hydrologyPositions.size());
        if (inserted) hydrologyPositions.push_back(position);
        return found->second;
    };
    const auto retainSample = [&](ColumnPos position) {
        const auto [found, inserted] = sampleIndices.emplace(position, samplePositions.size());
        if (inserted) {
            samplePositions.push_back(position);
            retainHydrology(position);
        }
        return found->second;
    };

    struct RootDensityCorners {
        size_t surface = 0;
        std::array<size_t, 4> lattice{};
        std::array<size_t, 4> hydrologyLattice{};
    };
    std::vector<RootDensityCorners> roots;
    roots.reserve(positions.size());
    for (const ColumnPos position : positions) {
        const int64_t latticeX = latticeFloor(position.x);
        const int64_t latticeZ = latticeFloor(position.z);
        const int64_t hydrologyLatticeX =
            world_coord::floorDiv(position.x, static_cast<int64_t>(COLUMN_PLAN_LATTICE_SPACING)) *
            COLUMN_PLAN_LATTICE_SPACING;
        const int64_t hydrologyLatticeZ =
            world_coord::floorDiv(position.z, static_cast<int64_t>(COLUMN_PLAN_LATTICE_SPACING)) *
            COLUMN_PLAN_LATTICE_SPACING;
        roots.push_back({
            .surface = retainSample(position),
            .lattice =
                {
                    retainSample({latticeX, latticeZ}),
                    retainSample({latticeX + LATTICE_XZ, latticeZ}),
                    retainSample({latticeX, latticeZ + LATTICE_XZ}),
                    retainSample({latticeX + LATTICE_XZ, latticeZ + LATTICE_XZ}),
                },
            .hydrologyLattice =
                {
                    retainHydrology({hydrologyLatticeX, hydrologyLatticeZ}),
                    retainHydrology(
                        {hydrologyLatticeX + COLUMN_PLAN_LATTICE_SPACING, hydrologyLatticeZ}),
                    retainHydrology(
                        {hydrologyLatticeX, hydrologyLatticeZ + COLUMN_PLAN_LATTICE_SPACING}),
                    retainHydrology({hydrologyLatticeX + COLUMN_PLAN_LATTICE_SPACING,
                                     hydrologyLatticeZ + COLUMN_PLAN_LATTICE_SPACING}),
                },
        });
    }

    std::vector<worldgen::SurfaceSample> samples(samplePositions.size());
    macroSampler_.sampleSurfacePoints(samplePositions, worldgen::SurfaceFootprint::BLOCK_1,
                                      samples);
    const std::vector<worldgen::SurfaceSample> macroSamples = samples;
    std::vector<worldgen::HydrologySample> canonicalHydrology(hydrologyPositions.size());
    macroSampler_.sampleHydrologyPoints(hydrologyPositions, canonicalHydrology);
    GenScratch& scratch = threadScratch();
    for (size_t index = 0; index < samples.size(); ++index) {
        const ColumnPos position = samplePositions[index];
        worldgen::SurfaceSample& surface = samples[index];
        surface.hydrology = canonicalHydrology[hydrologyIndices.at(position)];
        if (!learnedAuthority_ && requiresLegacyExactShoreSupport(surface.hydrology)) {
            surface.hydrology.channelBank = true;
        }
        surface.terrainHeight = surface.hydrology.surfaceElevation;
        if (!learnedAuthority_ && surface.hydrology.ocean) {
            surface.terrainHeight =
                std::min(surface.terrainHeight +
                             macroSampler_.reliefDetail(static_cast<double>(position.x),
                                                        static_cast<double>(position.z),
                                                        worldgen::SurfaceFootprint::BLOCK_1) *
                                 worldgen::OCEAN_FLOOR_DETAIL_SCALE,
                         static_cast<double>(SEA_LEVEL) - 0.5);
        } else if (!learnedAuthority_ && !hasSurfaceWater(surface.hydrology) &&
                   !surface.hydrology.channelBank && !surface.hydrology.lakeBank) {
            const double channelClearance =
                surface.hydrology.channelDistance - surface.hydrology.channelWidth * 2.5;
            surface.terrainHeight +=
                macroSampler_.reliefDetail(static_cast<double>(position.x),
                                           static_cast<double>(position.z),
                                           worldgen::SurfaceFootprint::BLOCK_1) *
                smoothstep(0.0, 32.0, channelClearance) * worldgen::DRY_RELIEF_DETAIL_SCALE;
        }
        surface.hydrology.surfaceElevation = surface.terrainHeight;
        surface.waterSurface = surface.hydrology.waterSurface;
        if (learnedAuthority_) {
            // Sparse habitat density must use the same coordinate-pure dry
            // residual as exact columns and far geometry. Resetting to the
            // canonical routing floor above is required for authoritative
            // water, but omitting this post-routing step moved dry roots and
            // left their retained climate adapted to a different elevation.
            macroSampler_.applyV4SurfaceDetail(static_cast<double>(position.x),
                                               static_cast<double>(position.z),
                                               worldgen::SurfaceFootprint::BLOCK_1, surface);
        }
        surface = applyVolcanism(position.x, position.z, std::move(surface), scratch);
    }

    std::unordered_map<ColumnPos, worldgen::MacroControlView> continuousViews;
    continuousViews.reserve(positions.size());
    const auto reconstructContinuous = [&](ColumnPos position, worldgen::SurfaceSample& surface) {
        const ColumnPos chunkColumn{Chunk::worldToChunk(position.x),
                                    Chunk::worldToChunk(position.z)};
        auto [view, inserted] = continuousViews.try_emplace(chunkColumn);
        if (inserted) view->second = macroSampler_.controlView(chunkColumn);
        view->second.reconstructContinuous(static_cast<double>(position.x),
                                           static_cast<double>(position.z), surface);
    };

    struct DensityLatticeColumn {
        ColumnShape shape;
        DensityColumnContext context;
        std::vector<double> values;
    };
    std::vector<std::optional<DensityLatticeColumn>> densityColumns(samples.size());
    const auto densityValue = [&](size_t sampleIndex, int level) {
        std::optional<DensityLatticeColumn>& retained = densityColumns[sampleIndex];
        if (!retained.has_value()) {
            const ColumnPos position = samplePositions[sampleIndex];
            const worldgen::SurfaceSample& surface = samples[sampleIndex];
            retained.emplace();
            retained->shape =
                shapeFromSurface(surface,
                                 climate_.caveEntrance(static_cast<double>(position.x),
                                                       static_cast<double>(position.z)),
                                 learnedAuthority_);
            retained->context = density_.columnContext(
                static_cast<double>(position.x), static_cast<double>(position.z), surface.geology);
            const double yCap = retained->shape.height + retained->shape.detailAmp + 5.0;
            const int maximumEvaluatedLevel =
                maximumEvaluatedDensityLevel(yCap, verticalDensitySpacing);
            retained->values.assign(static_cast<size_t>(maximumEvaluatedLevel + 1),
                                    std::numeric_limits<double>::quiet_NaN());
        }
        DensityLatticeColumn& column = *retained;
        const size_t boundedLevel =
            static_cast<size_t>(std::clamp(level, 0, densityLatticeLevels - 1));
        if (boundedLevel >= column.values.size()) return -DENSITY_CAP;
        if (std::isnan(column.values[boundedLevel])) {
            const ColumnPos position = samplePositions[sampleIndex];
            const double y = static_cast<double>(WORLD_MIN_Y + static_cast<int>(boundedLevel) *
                                                                   verticalDensitySpacing);
            column.values[boundedLevel] =
                evaluateDensity(static_cast<double>(position.x), y, static_cast<double>(position.z),
                                column.shape, column.context);
        }
        return column.values[boundedLevel];
    };

    const auto habitatSurface = [&](ColumnPos position, const RootDensityCorners& root) {
        const int64_t latticeX =
            world_coord::floorDiv(position.x, static_cast<int64_t>(COLUMN_PLAN_LATTICE_SPACING)) *
            COLUMN_PLAN_LATTICE_SPACING;
        const int64_t latticeZ =
            world_coord::floorDiv(position.z, static_cast<int64_t>(COLUMN_PLAN_LATTICE_SPACING)) *
            COLUMN_PLAN_LATTICE_SPACING;
        const double fx = static_cast<double>(position.x - latticeX) / COLUMN_PLAN_LATTICE_SPACING;
        const double fz = static_cast<double>(position.z - latticeZ) / COLUMN_PLAN_LATTICE_SPACING;
        const double smoothFx = smootherstep(0.0, 1.0, fx);
        const double smoothFz = smootherstep(0.0, 1.0, fz);
        const int nearestX = fx <= 0.5 ? 0 : 1;
        const int nearestZ = fz <= 0.5 ? 0 : 1;
        const size_t nearest = root.hydrologyLattice[static_cast<size_t>(nearestZ * 2 + nearestX)];
        worldgen::HydrologySample hydrology;
        if (!learnedAuthority_) {
            hydrology = canonicalHydrology[nearest];
            const auto interpolate = [&](auto getter) {
                return bilerpDensity(getter(canonicalHydrology[root.hydrologyLattice[0]]),
                                     getter(canonicalHydrology[root.hydrologyLattice[1]]),
                                     getter(canonicalHydrology[root.hydrologyLattice[2]]),
                                     getter(canonicalHydrology[root.hydrologyLattice[3]]), smoothFx,
                                     smoothFz);
            };
            hydrology.flowDirection = {
                interpolate([](const auto& value) { return value.flowDirection.x; }),
                interpolate([](const auto& value) { return value.flowDirection.z; }),
            };
            const double flowMagnitude =
                std::hypot(hydrology.flowDirection.x, hydrology.flowDirection.z);
            if (flowMagnitude < 1.0e-12) {
                hydrology.flowDirection = {1.0, 0.0};
            } else {
                hydrology.flowDirection.x /= flowMagnitude;
                hydrology.flowDirection.z /= flowMagnitude;
            }
            hydrology.discharge = interpolate([](const auto& value) { return value.discharge; });
            hydrology.sediment = interpolate([](const auto& value) { return value.sediment; });
            hydrology.channelDistance =
                interpolate([](const auto& value) { return value.channelDistance; });
            hydrology.channelWidth =
                interpolate([](const auto& value) { return value.channelWidth; });
            hydrology.channelDepth =
                interpolate([](const auto& value) { return value.channelDepth; });
            hydrology.channelGradient =
                interpolate([](const auto& value) { return value.channelGradient; });
            hydrology.erosionDepth =
                interpolate([](const auto& value) { return value.erosionDepth; });
        }

        worldgen::HydrologySample canonical = canonicalHydrology[hydrologyIndices.at(position)];
        if (!learnedAuthority_ && requiresLegacyExactShoreSupport(canonical)) {
            canonical.channelBank = true;
        }
        if (!learnedAuthority_ && canonical.ocean) {
            canonical.surfaceElevation =
                std::min(canonical.surfaceElevation +
                             macroSampler_.reliefDetail(static_cast<double>(position.x),
                                                        static_cast<double>(position.z),
                                                        worldgen::SurfaceFootprint::BLOCK_1) *
                                 worldgen::OCEAN_FLOOR_DETAIL_SCALE,
                         static_cast<double>(SEA_LEVEL) - 0.5);
        } else if (!learnedAuthority_ && !hasSurfaceWater(canonical) && !canonical.channelBank &&
                   !canonical.lakeBank) {
            const double channelClearance =
                canonical.channelDistance - canonical.channelWidth * 2.5;
            canonical.surfaceElevation +=
                macroSampler_.reliefDetail(static_cast<double>(position.x),
                                           static_cast<double>(position.z),
                                           worldgen::SurfaceFootprint::BLOCK_1) *
                smoothstep(0.0, 32.0, channelClearance) * worldgen::DRY_RELIEF_DETAIL_SCALE;
        }
        const worldgen::Vector2d directFlow = canonical.flowDirection;
        canonicalizeColumnPlanHydrology(canonical);
        if (learnedAuthority_) {
            // The block-exact v4 hydrology batch already contains every
            // routing field. Do not reintroduce the retired eight-block
            // interpolation while preparing sparse canopy habitat.
            hydrology = canonical;
            const double directFlowMagnitude = std::hypot(directFlow.x, directFlow.z);
            if (directFlowMagnitude < 1.0e-12) {
                hydrology.flowDirection = {};
            } else {
                hydrology.flowDirection = canonicalColumnPlanFlow(directFlow);
                const double canonicalFlowMagnitude =
                    std::hypot(hydrology.flowDirection.x, hydrology.flowDirection.z);
                hydrology.flowDirection.x /= canonicalFlowMagnitude;
                hydrology.flowDirection.z /= canonicalFlowMagnitude;
            }
            hydrology.discharge = static_cast<float>(hydrology.discharge);
            hydrology.sediment = static_cast<float>(hydrology.sediment);
            hydrology.channelDistance = static_cast<float>(hydrology.channelDistance);
            hydrology.channelWidth = static_cast<float>(hydrology.channelWidth);
            hydrology.channelDepth = static_cast<float>(hydrology.channelDepth);
            hydrology.channelGradient = static_cast<float>(hydrology.channelGradient);
            hydrology.erosionDepth = static_cast<float>(hydrology.erosionDepth);
            hydrology.baseflow = static_cast<float>(hydrology.baseflow);
            hydrology.precipitationSeasonality =
                static_cast<float>(hydrology.precipitationSeasonality);
            hydrology.groundwaterRechargeMm = static_cast<float>(hydrology.groundwaterRechargeMm);
            hydrology.groundwaterHead = static_cast<float>(hydrology.groundwaterHead);
            hydrology.hydroperiod = static_cast<float>(hydrology.hydroperiod);
            hydrology.terrainSlope = static_cast<float>(hydrology.terrainSlope);
            hydrology.shoreWaterSurface = canonical.waterSurface;
            hydrology.lake = !canonical.ocean && canonical.lake &&
                             canonical.waterSurface > canonical.surfaceElevation + 0.05;
            hydrology.river = !hydrology.ocean && !hydrology.lake && canonical.river;
            hydrology.wetland = !hydrology.ocean && !hydrology.lake && !hydrology.river &&
                                canonical.wetland &&
                                canonical.waterSurface > canonical.surfaceElevation + 0.01;
            hydrology.endorheic = hydrology.lake && canonical.endorheic;
        } else {
            hydrology.generatedFluidLevel = canonical.generatedFluidLevel;
            hydrology.transitionOwnerKind = canonical.transitionOwnerKind;
            hydrology.transitionOwnerId = canonical.transitionOwnerId;
            hydrology.waterBodyId = canonical.waterBodyId;
            hydrology.lakeShoreDistance = canonical.lakeShoreDistance;
            hydrology.shoreWaterSurface = canonical.waterSurface;
            hydrology.ocean = canonical.ocean;
            hydrology.lake = !canonical.ocean && canonical.lake &&
                             canonical.waterSurface > canonical.surfaceElevation + 0.05;
            hydrology.river = !hydrology.ocean && !hydrology.lake && canonical.river;
            // The legacy macro sampler can retain a wetland classification after
            // local relief has lifted the final bank above its water stage. Match
            // ColumnPlan's block-resolution ownership predicate so far flora
            // never accepts a wetland habitat that exact cube generation resolves
            // as dry terrain.
            hydrology.wetland = !hydrology.ocean && !hydrology.lake && !hydrology.river &&
                                canonical.wetland &&
                                canonical.waterSurface > canonical.surfaceElevation + 0.01;
            hydrology.channelBank = canonical.channelBank;
            hydrology.lakeBank = canonical.lakeBank;
            hydrology.lakeBankInfluence = canonical.lakeBankInfluence;
            hydrology.endorheic = hydrology.lake && canonical.endorheic;
            hydrology.waterfall = canonical.waterfall;
            hydrology.waterfallAnchor = canonical.waterfallAnchor;
            hydrology.waterfallTop = canonical.waterfallTop;
            hydrology.waterfallBottom = canonical.waterfallBottom;
            hydrology.waterfallWidth = canonical.waterfallWidth;
            if (hydrology.waterfall) hydrology.flowDirection = canonical.flowDirection;
            hydrology.delta = canonical.delta;
            if (hydrology.river || hydrology.delta) {
                for (size_t sampleIndex : root.hydrologyLattice) {
                    hydrology.streamOrder = std::max(hydrology.streamOrder,
                                                     canonicalHydrology[sampleIndex].streamOrder);
                    hydrology.distributaryCount =
                        std::max(hydrology.distributaryCount,
                                 canonicalHydrology[sampleIndex].distributaryCount);
                }
            } else {
                hydrology.streamOrder = 0;
                hydrology.distributaryCount = 0;
            }
            hydrology.waterSurface = canonical.waterSurface;
            hydrology.terrainSlope = canonical.terrainSlope;
            hydrology.groundwaterHead = canonical.groundwaterHead;
            hydrology.hydroperiod = canonical.hydroperiod;
            hydrology.groundwaterRechargeMm = canonical.groundwaterRechargeMm;
        }

        worldgen::SurfaceSample result = macroSamples[root.surface];
        result.hydrology = hydrology;
        result.terrainHeight = canonical.surfaceElevation;
        result.hydrology.surfaceElevation = result.terrainHeight;
        result.hydrology.lakeBankTarget = result.hydrology.lakeBank ? result.terrainHeight : 0.0;
        result.hydrology.lakeDepth =
            result.hydrology.lake ? result.hydrology.waterSurface - result.terrainHeight : 0.0;
        result.waterSurface = result.hydrology.waterSurface;
        if (learnedAuthority_) {
            // Canonicalization above deliberately restores the routed floor.
            // Reapply the bounded dry residual before learned climate is
            // reconstructed so exact columns, far geometry, and habitat
            // placement observe one final physical surface.
            macroSampler_.applyV4SurfaceDetail(static_cast<double>(position.x),
                                               static_cast<double>(position.z),
                                               worldgen::SurfaceFootprint::BLOCK_1, result);
        }
        // ColumnPlan reconstructs continuous climate after replacing its
        // lattice hydrology with block authority. Repeat that ordering here
        // so a density-only root cannot cross a species temperature limit.
        reconstructContinuous(position, result);
        canonicalizeColumnPlanGeology(result.geology);
        return applyVolcanism(position.x, position.z, std::move(result), scratch);
    };

    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        const RootDensityCorners& root = roots[index];
        worldgen::SurfaceSample result = habitatSurface(position, root);
        const ColumnShape rootShape = shapeFromSurface(
            result,
            climate_.caveEntrance(static_cast<double>(position.x), static_cast<double>(position.z)),
            learnedAuthority_);
        const int start =
            std::clamp(static_cast<int>(std::ceil(rootShape.height + rootShape.detailAmp + 8.0)),
                       WORLD_MIN_Y + 2, WORLD_MAX_Y);
        const double fx = static_cast<double>(position.x - latticeFloor(position.x)) / LATTICE_XZ;
        const double fz = static_cast<double>(position.z - latticeFloor(position.z)) / LATTICE_XZ;
        int surfaceY = WORLD_MIN_Y + 1;
        const int startLevel = (start - WORLD_MIN_Y) / verticalDensitySpacing;
        for (int level = startLevel; level >= 0; --level) {
            const int levelBaseY = WORLD_MIN_Y + level * verticalDensitySpacing;
            const int minimumY = std::max(WORLD_MIN_Y + 2, levelBaseY);
            const int maximumY = std::min(start, levelBaseY + verticalDensitySpacing - 1);
            if (minimumY > maximumY) continue;
            const double below = bilerpDensity(
                densityValue(root.lattice[0], level), densityValue(root.lattice[1], level),
                densityValue(root.lattice[2], level), densityValue(root.lattice[3], level), fx, fz);
            const double above = bilerpDensity(densityValue(root.lattice[0], level + 1),
                                               densityValue(root.lattice[1], level + 1),
                                               densityValue(root.lattice[2], level + 1),
                                               densityValue(root.lattice[3], level + 1), fx, fz);
            const auto interpolatedDensity = [&](int y) {
                const double fy = static_cast<double>(y - levelBaseY) / verticalDensitySpacing;
                return lerpDensity(below, above, fy);
            };
            if (interpolatedDensity(maximumY) > 0.0) {
                surfaceY = maximumY;
                break;
            }
            // Density is linear between vertical lattice planes. If both
            // endpoints of this integer interval are empty, every intervening
            // block is empty as well and no per-block scan is needed.
            if (maximumY == minimumY || interpolatedDensity(minimumY) <= 0.0) continue;
            for (int y = maximumY - 1; y >= minimumY; --y) {
                if (interpolatedDensity(y) > 0.0) {
                    surfaceY = y;
                    break;
                }
            }
            if (surfaceY >= minimumY) break;
        }

        const VolcanicColumnSample& volcanism =
            volcanismAt(position.x, position.z, result.geology, scratch);
        surfaceY = boundedLearnedSurfaceY(result, surfaceY, learnedAuthority_);
        if (!learnedAuthority_ && volcanism.craterLake &&
            volcanism.craterProfileInfluence > 1.0e-4) {
            surfaceY = std::clamp(static_cast<int>(std::ceil(result.terrainHeight)) - 1,
                                  WORLD_MIN_Y, WORLD_MAX_Y);
        }
        reconcileEmittedSurfaceY(result.hydrology, surfaceY, learnedAuthority_);

        applyEmittedSurfaceTopology(result, static_cast<double>(surfaceY + 1), learnedAuthority_);
        output[index] =
            refreshDependentSurface(position.x, position.z, std::move(result), volcanism);
    }
}

void ChunkGenerator::sampleFarSurfaceGrid(int64_t originX, int64_t originZ, int spacing,
                                          int sampleEdge, worldgen::SurfaceFootprint footprint,
                                          std::span<worldgen::SurfaceSample> output) const {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_1) {
        sampleExactSurfaceGrid(originX, originZ, spacing, sampleEdge, output);
        return;
    }
    macroSampler_.sampleSurfaceGrid(originX, originZ, spacing, sampleEdge, footprint, output);
    GenScratch& scratch = threadScratch();
    for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * sampleEdge + sampleX);
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
            output[index] = applyVolcanism(worldX, worldZ, std::move(output[index]), scratch);
        }
    }
}

void ChunkGenerator::sampleWeatherClimateGrid(int64_t originX, int64_t originZ, int spacing,
                                              int sampleEdge,
                                              std::span<worldgen::SurfaceSample> output) const {
    macroSampler_.sampleClimateGrid(originX, originZ, spacing, sampleEdge, output);
}

BlockType ChunkGenerator::surfaceMaterialAt(int64_t x, int64_t z) const {
    GenScratch& scratch = threadScratch();
    const worldgen::SurfaceSample surface = surfaceSampleAt(x, z, scratch);
    const ColumnPos chunkColumn{
        world_coord::floorDiv(x, static_cast<int64_t>(CHUNK_EDGE)),
        world_coord::floorDiv(z, static_cast<int64_t>(CHUNK_EDGE)),
    };
    const auto found = scratch.columnPlans.find(chunkColumn);
    if (found == scratch.columnPlans.end()) {
        throw std::logic_error("surface material sample did not retain its column plan");
    }
    const int surfaceY = found->second->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
    const int waterTopY =
        std::clamp(static_cast<int>(std::ceil(surface.waterSurface)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
    const bool submerged = hasSurfaceWater(surface.hydrology) && surfaceY < waterTopY;
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
    const auto palette = worldgen::surface_material::materialPalette(
        surface, materialSignals(volcanism),
        worldgen::surface_material::frozen(surface, surface.biome.primary), submerged,
        isAlluvialDeposit(surface, random_, x, z));
    return worldgen::surface_material::selectMaterial(
        palette, worldgen::multiscaleDitherThreshold(
                     random_, worldgen::surface_material::DITHER_STREAM, x, z));
}

worldgen::surface_material::SurfaceMaterialPalette
ChunkGenerator::surfaceMaterialPaletteAt(int64_t x, int64_t z) const {
    GenScratch& scratch = threadScratch();
    const worldgen::SurfaceSample surface = surfaceSampleAt(x, z, scratch);
    const ColumnPos chunkColumn{
        world_coord::floorDiv(x, static_cast<int64_t>(CHUNK_EDGE)),
        world_coord::floorDiv(z, static_cast<int64_t>(CHUNK_EDGE)),
    };
    const auto found = scratch.columnPlans.find(chunkColumn);
    if (found == scratch.columnPlans.end()) {
        throw std::logic_error("surface material palette did not retain its column plan");
    }
    const int surfaceY = found->second->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
    const int waterTopY =
        std::clamp(static_cast<int>(std::ceil(surface.waterSurface)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
    const bool submerged = hasSurfaceWater(surface.hydrology) && surfaceY < waterTopY;
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
    return worldgen::surface_material::materialPalette(
        surface, materialSignals(volcanism),
        worldgen::surface_material::frozen(surface, surface.biome.primary), submerged,
        isAlluvialDeposit(surface, random_, x, z));
}

BlockType ChunkGenerator::farSurfaceMaterialAt(int64_t x, int64_t z) const {
    const worldgen::SurfaceSample surface = sampleFarSurface(x, z);
    return farSurfaceMaterialAt(x, z, surface);
}

BlockType ChunkGenerator::farSurfaceMaterialAt(int64_t x, int64_t z,
                                               const worldgen::SurfaceSample& surface) const {
    GenScratch& scratch = threadScratch();
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
    const Biome biome = ditheredBiome(surface, random_, x, z);
    return worldgen::surface_material::surface(surface, biome, materialSignals(volcanism),
                                               worldgen::surface_material::frozen(surface, biome),
                                               worldgen::surface_material::submerged(surface),
                                               isAlluvialDeposit(surface, random_, x, z));
}

worldgen::surface_material::SurfaceMaterialPalette
ChunkGenerator::farSurfaceMaterialPaletteAt(int64_t x, int64_t z,
                                            worldgen::SurfaceFootprint footprint) const {
    const worldgen::SurfaceSample surface = sampleFarSurface(x, z, footprint);
    return farSurfaceMaterialPaletteAt(x, z, surface);
}

worldgen::surface_material::SurfaceMaterialPalette
ChunkGenerator::farSurfaceMaterialPaletteAt(int64_t x, int64_t z,
                                            const worldgen::SurfaceSample& surface) const {
    GenScratch& scratch = threadScratch();
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
    return worldgen::surface_material::materialPalette(
        surface, materialSignals(volcanism),
        worldgen::surface_material::frozen(surface, surface.biome.primary),
        worldgen::surface_material::submerged(surface), isAlluvialDeposit(surface, random_, x, z));
}

double ChunkGenerator::farSurfaceMaterialRankAt(int64_t x, int64_t z) const {
    return worldgen::multiscaleDitherThreshold(random_, worldgen::surface_material::DITHER_STREAM,
                                               x, z);
}

double ChunkGenerator::sampleFarTerrainHeight(int64_t x, int64_t z) const {
    if (learnedAuthority_) {
        return macroSampler_
            .sampleSurface(static_cast<double>(x), static_cast<double>(z),
                           worldgen::SurfaceFootprint::BLOCK_16)
            .terrainHeight;
    }
    worldgen::SurfaceSample macroSurface;
    macroSurface.geology =
        macroSampler_.sampleGeology(static_cast<double>(x), static_cast<double>(z));
    macroSurface.hydrology =
        macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
    macroSurface.terrainHeight = macroSurface.hydrology.surfaceElevation;
    GenScratch& scratch = threadScratch();
    applyVolcanicGeometry(macroSurface, volcanismAt(x, z, macroSurface.geology, scratch));
    return macroSurface.terrainHeight;
}

std::vector<VolcanoPrimitive> ChunkGenerator::hotspotVolcanoesForCell(int64_t cellX,
                                                                      int64_t cellZ) const {
    GenScratch& scratch = threadScratch();
    const auto& volcanoes = volcanoesForCell(cellX, cellZ, scratch);
    std::vector<VolcanoPrimitive> resolved;
    resolved.reserve(volcanoes.size());
    for (const VolcanoPrimitive& volcano : volcanoes)
        resolved.push_back(resolvedVolcano(volcano, scratch));
    return resolved;
}

std::vector<FarCanopy> ChunkGenerator::collectFarCanopies(int64_t minimumX, int64_t minimumZ,
                                                          int64_t maximumX,
                                                          int64_t maximumZ) const {
    GenScratch& scratch = threadScratch();
    prepareScratch(scratch);
    return features_.collectFarCanopies(minimumX, minimumZ, maximumX, maximumZ, *this, structures_,
                                        scratch);
}

std::vector<FarCanopy> ChunkGenerator::collectFarCanopiesForLod(int64_t minimumX, int64_t minimumZ,
                                                                int64_t maximumX, int64_t maximumZ,
                                                                int lodStep) const {
    if (lodStep <= 2) {
        GenScratch& scratch = threadScratch();
        prepareScratch(scratch);
        // Far attachments always re-ground every accepted root against the
        // displayed terrain cell. The batched anchor sampler already resolves
        // the exact habitat surface, so rebuilding a complete ColumnPlan for
        // every step-one tree only changes an absolute Y value that the
        // mesher immediately discards.
        return features_.collectFarCanopyAnchors(minimumX, minimumZ, maximumX, maximumZ, *this,
                                                 structures_, scratch);
    }
    return features_.collectFarCanopyClusters(minimumX, minimumZ, maximumX, maximumZ, lodStep,
                                              *this);
}

std::vector<FarFlora> ChunkGenerator::collectFarFloraForLod(int64_t minimumX, int64_t minimumZ,
                                                            int64_t maximumX, int64_t maximumZ,
                                                            int lodStep) const {
    return features_.collectFarFlora(minimumX, minimumZ, maximumX, maximumZ, lodStep, *this);
}

const std::vector<VolcanoPrimitive>& ChunkGenerator::volcanoesForCell(int64_t cellX, int64_t cellZ,
                                                                      GenScratch& scratch) const {
    const ColumnPos key{cellX, cellZ};
    auto found = scratch.volcanoCells.find(key);
    if (found != scratch.volcanoCells.end()) return found->second;

    ThreadVolcanoPrimitiveCache& cache = threadVolcanoPrimitiveCache(this, scratch.ownerToken);
    if (const auto cached = cache.hotspotCells.find(key); cached != cache.hotspotCells.end()) {
        return scratch.volcanoCells.emplace(key, cached->second).first->second;
    }

    std::vector<VolcanoPrimitive> volcanoes;
    const worldgen::HotspotChainPrimitive chain = macroSampler_.hotspotChain(cellX, cellZ);
    if (chain.active) {
        const double hotspotX = chain.sourceX;
        const double hotspotZ = chain.sourceZ;
        const double chainLength = chain.length;
        const worldgen::Vector2d chainDirection = chain.direction;
        const worldgen::Vector2d transverse{-chainDirection.z, chainDirection.x};
        const int count = random_.uniformInt(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ, 2, 4, 7);
        volcanoes.reserve(static_cast<size_t>(count));

        for (int index = 0; index < count; ++index) {
            const uint32_t propertyIndex = static_cast<uint32_t>(16 + index * 12);
            const double chainFraction = std::clamp(
                (static_cast<double>(index) +
                 random_.signedUnit(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ, propertyIndex) *
                     0.10) /
                    std::max(1, count - 1),
                0.0, 1.0);
            const double transverseOffset =
                random_.signedUnit(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ, propertyIndex + 1) *
                150.0;
            const double centerX = hotspotX + chainDirection.x * chainLength * chainFraction +
                                   transverse.x * transverseOffset;
            const double centerZ = hotspotZ + chainDirection.z * chainLength * chainFraction +
                                   transverse.z * transverseOffset;
            const bool shield = random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ,
                                                  propertyIndex + 2) < (index == 0 ? 0.72 : 0.48);
            const double sizeRoll =
                random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ, propertyIndex + 3);
            const double radius = shield ? 520.0 + sizeRoll * 430.0 : 260.0 + sizeRoll * 260.0;
            const double coneHeight = shield ? 11.0 + sizeRoll * 6.0 : 13.0 + sizeRoll * 5.0;
            const double craterRadius = shield ? 52.0 + sizeRoll * 58.0 : 22.0 + sizeRoll * 30.0;
            // Broad shield calderas need enough relief for a visibly deep
            // lake after preserving the rain-fed stage and full rim
            // freeboard. The analytical profile still owns both floor and
            // rim, so this deepens coherent basins without affecting marshes
            // or ordinary hydrology depressions.
            const double craterDepth = shield ? 30.0 + sizeRoll * 22.0 : 12.0 + sizeRoll * 10.0;
            // Legacy worlds retain their historical eager datum. V4 learns
            // it only after the primitive passes the bounded influence test
            // in volcanismAt, so an irrelevant hotspot chain cannot expand a
            // far-horizon authority request by kilometers.
            const bool resolveDatumNow = !learnedAuthority_;
            const double centerElevation =
                resolveDatumNow ? macroSampler_.preliminaryElevation(centerX, centerZ) : 0.0;
            const bool craterLake = random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ,
                                                      propertyIndex + 4) < 0.48;
            const uint64_t id =
                random_.u64(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ, propertyIndex + 5);
            VolcanoPrimitive volcano{
                .centerX = centerX,
                .centerZ = centerZ,
                .radius = radius,
                .coneHeight = coneHeight,
                .craterRadius = craterRadius,
                .craterDepth = craterDepth,
                .craterDatumElevation = centerElevation,
                .craterDatumResolved = resolveDatumNow,
                .tubePhase = random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ,
                                               propertyIndex + 6) *
                             2.0 * std::numbers::pi,
                .conduitRadius = 2.2 + random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ,
                                                         propertyIndex + 7) *
                                           2.0,
                .id = id,
                .shield = shield,
                .craterLake = craterLake,
                .lavaBearing = random_.uniform01(HOTSPOT_PROPERTIES_STREAM, cellX, 0, cellZ,
                                                 propertyIndex + 8) < 0.58,
            };
            if (resolveDatumNow) configureCraterLake(volcano);
            volcanoes.push_back(volcano);
        }
    }
    if (cache.hotspotCells.size() >= HOTSPOT_PRIMITIVE_CACHE_CAPACITY) cache.hotspotCells.clear();
    cache.hotspotCells.emplace(key, volcanoes);
    return scratch.volcanoCells.emplace(key, std::move(volcanoes)).first->second;
}

const std::vector<VolcanoPrimitive>&
ChunkGenerator::volcanicArcForCell(int64_t cellX, int64_t cellZ, GenScratch& scratch) const {
    const ColumnPos key{cellX, cellZ};
    auto found = scratch.volcanicArcCells.find(key);
    if (found != scratch.volcanicArcCells.end()) return found->second;

    ThreadVolcanoPrimitiveCache& cache = threadVolcanoPrimitiveCache(this, scratch.ownerToken);
    if (const auto cached = cache.arcCells.find(key); cached != cache.arcCells.end()) {
        return scratch.volcanicArcCells.emplace(key, cached->second).first->second;
    }

    std::vector<VolcanoPrimitive> volcanoes;
    const double centerX = (static_cast<double>(cellX) + 0.28 +
                            random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 0) * 0.44) *
                           VOLCANIC_ARC_CELL_EDGE;
    const double centerZ = (static_cast<double>(cellZ) + 0.28 +
                            random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 1) * 0.44) *
                           VOLCANIC_ARC_CELL_EDGE;
    const worldgen::GeologySample geology = macroSampler_.sampleGeology(centerX, centerZ);
    const double acceptance = 0.22 + geology.volcanicActivity * 0.54;
    if (geology.boundary == worldgen::PlateBoundary::CONVERGENT &&
        geology.volcanicActivity > 0.16 &&
        random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 2) < acceptance) {
        const double sizeRoll = random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 3);
        const double coneHeight = 12.0 + sizeRoll * 6.0;
        const double craterDepth = 8.0 + sizeRoll * 7.0;
        const bool resolveDatumNow = !learnedAuthority_;
        const double centerElevation =
            resolveDatumNow ? macroSampler_.preliminaryElevation(centerX, centerZ) : 0.0;
        VolcanoPrimitive volcano{
            .centerX = centerX,
            .centerZ = centerZ,
            .radius = 230.0 + sizeRoll * 210.0,
            .coneHeight = coneHeight,
            .craterRadius = 21.0 + sizeRoll * 26.0,
            .craterDepth = craterDepth,
            .craterDatumElevation = centerElevation,
            .craterDatumResolved = resolveDatumNow,
            .tubePhase =
                random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 4) * 2.0 * std::numbers::pi,
            .conduitRadius = 2.3 + random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 5) * 1.7,
            .id = random_.u64(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 6),
            .shield = false,
            .craterLake = random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 7) < 0.28,
            .lavaBearing = random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 8) < 0.72,
        };
        if (resolveDatumNow) configureCraterLake(volcano);
        volcanoes.push_back(volcano);
    }
    if (cache.arcCells.size() >= ARC_PRIMITIVE_CACHE_CAPACITY) cache.arcCells.clear();
    cache.arcCells.emplace(key, volcanoes);
    return scratch.volcanicArcCells.emplace(key, std::move(volcanoes)).first->second;
}

const VolcanoPrimitive& ChunkGenerator::resolvedVolcano(const VolcanoPrimitive& volcano,
                                                        GenScratch& scratch) const {
    if (volcano.craterDatumResolved) return volcano;
    const auto found = scratch.resolvedVolcanoes.try_emplace(&volcano, volcano).first;
    VolcanoPrimitive& resolved = found->second;
    if (!resolved.craterDatumResolved) {
        // Learned authority can defer while the datum page is prepared. Do
        // not treat the cache entry created before that exception as a fully
        // resolved volcano on the retry. Keeping the fallible query before
        // any state mutation also leaves the entry safe for an identical
        // later attempt.
        const double datum = macroSampler_.preliminaryElevation(resolved.centerX, resolved.centerZ);
        resolved.craterDatumElevation = datum;
        resolved.craterDatumResolved = true;
        configureCraterLake(resolved);
    }
    return resolved;
}

const VolcanicColumnSample& ChunkGenerator::volcanismAt(int64_t x, int64_t z,
                                                        const worldgen::GeologySample& geology,
                                                        GenScratch& scratch) const {
    const ColumnPos key{x, z};
    auto found = scratch.volcanicColumns.find(key);
    if (found != scratch.volcanicColumns.end()) return found->second;

    VolcanicColumnSample result;
    double strongestProfile = 0.0;
    double selectedCraterScore = std::numeric_limits<double>::infinity();
    bool selectedCraterWet = false;
    auto accumulate = [&](const VolcanoPrimitive& primitive) {
        const double offsetX = static_cast<double>(x) + 0.5 - primitive.centerX;
        const double offsetZ = static_cast<double>(z) + 0.5 - primitive.centerZ;
        const double distance = std::hypot(offsetX, offsetZ);
        if (distance > primitive.radius * 1.20) return;
        // V4 consumes the same primitive as a bounded material and cubic
        // geology signal only. Its terrain profile already entered native
        // routing, and no production v4 consumer needs the legacy analytical
        // lake datum. Avoid a second learned-authority query at the cone
        // center while sampling a nearby column.
        const VolcanoPrimitive& volcano =
            learnedAuthority_ ? primitive : resolvedVolcano(primitive, scratch);

        auto radialShape = [&](double radialDistance) {
            const double profile = volcanoProfile(volcano, radialDistance);
            const double crater = craterFactor(volcano, radialDistance);
            return std::array{profile, crater,
                              volcano.coneHeight * profile - volcano.craterDepth * crater};
        };
        const double profileDistance = craterWarpedDistance(volcano, offsetX, offsetZ);
        const auto [profile, crater, heightAdjustment] = radialShape(profileDistance);
        if (profile > strongestProfile) {
            strongestProfile = profile;
            result.heightAdjustment = heightAdjustment;
            result.strongestProfile = profile;
            result.strongestRadius = volcano.radius;
            const double innerHeight = radialShape(std::max(0.0, profileDistance - 0.5))[2];
            const double outerHeight = radialShape(profileDistance + 0.5)[2];
            result.slopeContribution = std::abs(outerHeight - innerHeight);
            result.craterFactor = crater;
            result.centerDistance = distance;
            result.conduitRadius = volcano.conduitRadius;
            result.conduitDepth = volcano.shield ? 88.0 : 118.0;
            result.conduitLavaBearing = volcano.lavaBearing;
        }

        if (!learnedAuthority_) {
            const double craterScore = profileDistance / std::max(1.0, volcano.craterRadius);
            // The absolute crater datum must taper back to the ordinary volcanic
            // surface over the complete edifice. Ending that authority at the
            // narrow rim apron drops the datum in one column and creates a cliff
            // around an otherwise continuous caldera.
            const double craterAuthorityLimit = volcano.radius;
            const bool candidateCraterWet = profileDistance < volcano.craterLakeRadius;
            if (volcano.craterLake && profileDistance <= craterAuthorityLimit &&
                ((candidateCraterWet && !selectedCraterWet) ||
                 (candidateCraterWet == selectedCraterWet &&
                  craterScore < selectedCraterScore))) {
                selectedCraterScore = craterScore;
                selectedCraterWet = candidateCraterWet;
                result.craterFactor = crater;
                result.craterRadius = volcano.craterRadius;
                result.craterProfileDistance = profileDistance;
                result.craterTerrainTarget = craterProfileHeight(volcano, profileDistance);
                result.craterProfileInfluence =
                    1.0 - smootherstep(volcano.craterRadius, craterAuthorityLimit, profileDistance);
                result.craterLakeRadius = volcano.craterLakeRadius;
                result.craterLakeSurface = volcano.craterLakeSurface;
                result.craterWaterBodyId = hash64(volcano.id ^ CRATER_WATER_BODY_SALT);
                if (result.craterWaterBodyId == worldgen::NO_WATER_BODY) {
                    result.craterWaterBodyId = 1;
                }
                result.craterRimElevation = volcano.craterRimElevation;
                result.craterRimWidth = volcano.craterRimWidth;
                result.craterLake = true;
            }
        }

        const double angle = std::atan2(offsetZ, offsetX);
        const double lobe = 0.88 + std::sin(angle * 5.0 + volcano.tubePhase) * 0.14;
        const double fieldStrength = 1.0 - smoothstep(0.54, 1.08, distance / volcano.radius / lobe);
        result.basaltField = std::max(result.basaltField, fieldStrength);

        const int tubeCount = volcano.shield ? 4 : 3;
        for (int tube = 0; tube < tubeCount; ++tube) {
            const double tubeAngle = volcano.tubePhase + 2.0 * std::numbers::pi * tube / tubeCount;
            const double directionX = std::cos(tubeAngle);
            const double directionZ = std::sin(tubeAngle);
            const double along = offsetX * directionX + offsetZ * directionZ;
            if (along < volcano.craterRadius * 1.25 || along > volcano.radius * 0.82) continue;
            const double cross = -offsetX * directionZ + offsetZ * directionX;
            const double curve =
                std::sin(along / 74.0 + volcano.tubePhase * 1.7) * (volcano.shield ? 8.0 : 5.0);
            const double lateralDistance = std::abs(cross - curve);
            const double radiusRoll =
                static_cast<double>((volcano.id >> ((tube * 11) & 47)) & 0xFFU) / 255.0;
            const double tubeRadius = 2.25 + radiusRoll * 1.35;
            if (lateralDistance >= result.tubeDistance) continue;
            result.tubeDistance = lateralDistance;
            result.tubeRadius = tubeRadius;
            result.tubeCenterOffset = -10.0 - along * (volcano.shield ? 0.025 : 0.045) +
                                      std::sin(along / 39.0 + volcano.tubePhase) * 1.8;
            result.tubeLavaBearing = volcano.lavaBearing;
        }
    };

    const int64_t baseCellX = world_coord::floorDiv(x, worldgen::HOTSPOT_LATTICE_EDGE);
    const int64_t baseCellZ = world_coord::floorDiv(z, worldgen::HOTSPOT_LATTICE_EDGE);
    for (int cellOffsetZ = -HOTSPOT_QUERY_RADIUS; cellOffsetZ <= HOTSPOT_QUERY_RADIUS;
         ++cellOffsetZ) {
        for (int cellOffsetX = -HOTSPOT_QUERY_RADIUS; cellOffsetX <= HOTSPOT_QUERY_RADIUS;
             ++cellOffsetX) {
            const auto& volcanoes =
                volcanoesForCell(baseCellX + cellOffsetX, baseCellZ + cellOffsetZ, scratch);
            for (const VolcanoPrimitive& volcano : volcanoes)
                accumulate(volcano);
        }
    }

    if (geology.distanceToBoundary < 1800.0) {
        const int64_t arcCellX = world_coord::floorDiv(x, VOLCANIC_ARC_CELL_EDGE);
        const int64_t arcCellZ = world_coord::floorDiv(z, VOLCANIC_ARC_CELL_EDGE);
        for (int cellOffsetZ = -1; cellOffsetZ <= 1; ++cellOffsetZ) {
            for (int cellOffsetX = -1; cellOffsetX <= 1; ++cellOffsetX) {
                const auto& volcanoes =
                    volcanicArcForCell(arcCellX + cellOffsetX, arcCellZ + cellOffsetZ, scratch);
                for (const VolcanoPrimitive& volcano : volcanoes)
                    accumulate(volcano);
            }
        }
    }

    result.heightAdjustment = std::clamp(result.heightAdjustment, -18.0, 18.0);
    return scratch.volcanicColumns.emplace(key, result).first->second;
}

worldgen::SurfaceSample
ChunkGenerator::refreshDependentSurface(int64_t x, int64_t z, worldgen::SurfaceSample result,
                                        const VolcanicColumnSample& volcanism) const {
    result.hydrology.lakeDepth =
        result.hydrology.lake ? std::max(0.0, result.hydrology.waterSurface - result.terrainHeight)
                              : 0.0;
    result.waterSurface = result.hydrology.waterSurface;
    if (!learnedAuthority_) {
        const double windSpeed = std::hypot(result.climate.wind.x, result.climate.wind.z);
        result.climate.potentialEvapotranspirationMm = std::clamp(
            300.0 + std::max(-8.0, result.climate.temperatureC) * 31.0 + windSpeed * 170.0, 120.0,
            1800.0);
        result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                                 std::max(1.0, result.climate.annualPrecipitationMm);
    }
    result.soil = macroSampler_.sampleSoil(static_cast<double>(x), static_cast<double>(z),
                                           result.geology, result.hydrology, result.climate);
    result.suitability =
        macroSampler_.biomeSuitability(result.geology, result.hydrology, result.climate,
                                       result.soil, result.terrainHeight, result.slope);
    result.biome = worldgen::MacroGenerationSampler::selectBiome(result.suitability);
    result.ecotopes = worldgen::MacroGenerationSampler::classifyEcotopes(result);
    if (volcanism.basaltField > 0.18) result.ecotopes |= worldgen::Ecotope::GEOTHERMAL;
    if (density_.supportsCaveEcotope(static_cast<double>(x), static_cast<double>(z),
                                     result.terrainHeight, result.geology)) {
        result.ecotopes |= worldgen::Ecotope::CAVE;
    }
    return result;
}

void ChunkGenerator::applyVolcanicGeometry(worldgen::SurfaceSample& result,
                                           const VolcanicColumnSample& volcanism) const {
    if (learnedAuthority_) {
        // V4 cones and calderas already entered the native routing elevation.
        // Applying the legacy deformation here would double their relief and
        // move beds after canonical water was solved.
        return;
    }
    const double solvedTerrain = result.terrainHeight;
    const bool solvedOcean = result.hydrology.ocean;
    const bool preserveCanonicalWater =
        learnedAuthority_ && hasCanonicalWaterFeature(result.hydrology);
    const bool canonicalRoutedWater =
        result.hydrology.river && result.hydrology.transitionOwnerId != 0;
    double deformedTerrain = emittedVolcanicHeight(result, volcanism, learnedAuthority_);
    if (preserveCanonicalWater) {
        // V4 hydrology is solved from immutable learned terrain pages. Until
        // volcanic primitives are included by that pre-hydrology authority,
        // postprocessing may describe volcanic materials but cannot deform a
        // canonical wet column or replace any of its topology.
        deformedTerrain = solvedTerrain;
    } else if (canonicalRoutedWater || result.hydrology.lake) {
        deformedTerrain = std::min(deformedTerrain, solvedTerrain);
    } else if (result.hydrology.channelBank || result.hydrology.lakeBank) {
        deformedTerrain = std::max(deformedTerrain, solvedTerrain);
    }
    result.terrainHeight = deformedTerrain;
    result.hydrology.surfaceElevation = result.terrainHeight;

    if (preserveCanonicalWater) {
        // Preserve every canonical water field exactly. This covers oceans,
        // rivers, lakes, wetlands, deltas, estuaries represented by combined
        // coastal flags, and explicit transition ownership.
    } else if (!learnedAuthority_ && applyCraterLake(result, volcanism, true, false)) {
        // The context-free v3 path retains its analytical crater-lake overlay.
        // V4 exposes water only through canonical native hydrology.
    } else if (solvedOcean && result.terrainHeight < SEA_LEVEL && !result.hydrology.lake &&
               !canonicalRoutedWater) {
        const bool coastalDelta = result.hydrology.delta;
        const uint8_t deltaOrder = result.hydrology.streamOrder;
        const uint8_t distributaryCount = result.hydrology.distributaryCount;
        const bool outletWaterfall =
            result.hydrology.waterfall && result.hydrology.waterfallTop >= SEA_LEVEL + 0.5;
        const double waterfallTop = result.hydrology.waterfallTop;
        const double waterfallBottom = result.hydrology.waterfallBottom;
        const double waterfallWidth = result.hydrology.waterfallWidth;
        const bool waterfallAnchor = result.hydrology.waterfallAnchor;
        clearSurfaceWater(result.hydrology);
        result.hydrology.ocean = true;
        result.hydrology.waterSurface = SEA_LEVEL;
        if (outletWaterfall) {
            result.hydrology.waterfall = true;
            result.hydrology.waterfallTop = waterfallTop;
            result.hydrology.waterfallBottom = waterfallBottom;
            result.hydrology.waterfallWidth = waterfallWidth;
            result.hydrology.waterfallAnchor = waterfallAnchor;
        }
        if (coastalDelta) {
            result.hydrology.delta = true;
            result.hydrology.streamOrder = deltaOrder;
            result.hydrology.distributaryCount = distributaryCount;
        }
    } else if (result.hydrology.ocean ||
               ((result.hydrology.river || result.hydrology.lake) &&
                result.terrainHeight >= result.hydrology.waterSurface - 0.01)) {
        clearSurfaceWater(result.hydrology);
    } else if (!hasSurfaceWater(result.hydrology) && !result.hydrology.lakeBank) {
        // Dry basin and shoreline samples may retain a descriptive candidate
        // water level after their categorical ownership is rejected. Exact
        // emission already clears that stale level. Apply the same contract
        // before far samples are published so every LOD sees one canonical
        // dry-water authority instead of constructing a detached surface.
        clearSurfaceWater(result.hydrology);
    }
    if (result.hydrology.lake) {
        result.hydrology.lakeDepth = result.hydrology.waterSurface - result.terrainHeight;
    }
    result.waterSurface = result.hydrology.waterSurface;
}

worldgen::SurfaceSample ChunkGenerator::applyVolcanism(int64_t x, int64_t z,
                                                       worldgen::SurfaceSample result,
                                                       GenScratch& scratch) const {
    if (learnedAuthority_) {
        const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
        // Material and habitat signals remain bounded consumers of the same
        // pre-routed primitive field. They do not mutate terrain or stage a
        // second time after hydrology.
        if (volcanism.basaltField > 0.18) result.ecotopes |= worldgen::Ecotope::GEOTHERMAL;
        if (density_.supportsCaveEcotope(static_cast<double>(x), static_cast<double>(z),
                                         result.terrainHeight, result.geology)) {
            result.ecotopes |= worldgen::Ecotope::CAVE;
        }
        return result;
    }
    const double macroHeight = result.terrainHeight;
    const worldgen::HydrologySample macroHydrology = result.hydrology;
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    applyVolcanicGeometry(result, volcanism);
    const double heightAdjustment = result.terrainHeight - macroHeight;

    const double broadIslandSlope =
        std::abs(heightAdjustment) / std::max(1.0, volcanism.strongestRadius);
    result.slope =
        std::clamp(result.slope + volcanism.slopeContribution + broadIslandSlope, 0.0, 16.0);

    const double oldClimateHeight = std::max(0.0, macroHeight - SEA_LEVEL);
    const double newClimateHeight = std::max(0.0, result.terrainHeight - SEA_LEVEL);
    const double climateHeightDelta = newClimateHeight - oldClimateHeight;
    if (learnedAuthority_) {
        result.climate.temperatureC += climateHeightDelta *
                                       worldgen::learned::WORLD_METERS_PER_BLOCK *
                                       result.climate.lapseRateCPerMeter;
    } else {
        // Legacy v3 retains its established eight-meter lapse conversion.
        result.climate.temperatureC += climateHeightDelta *
                                       worldgen::LEGACY_WORLD_METERS_PER_BLOCK *
                                       worldgen::LEGACY_LAPSE_RATE_C_PER_METER;
    }

    // The macro climate already integrated the bounded upwind path. Apply the
    // exact local water change introduced by the discrete cone, then refresh
    // every climate value that depends on temperature or available moisture.
    const double localWaterDelta =
        localWaterClimateInfluence(result.hydrology) - localWaterClimateInfluence(macroHydrology);
    result.climate.temperatureC += (13.0 - result.climate.temperatureC) * localWaterDelta * 0.08;
    result.climate.annualPrecipitationMm = std::clamp(
        result.climate.annualPrecipitationMm * (1.0 + localWaterDelta * 0.08), 60.0, 3600.0);
    result.climate.relativeHumidity =
        std::clamp(result.climate.relativeHumidity + localWaterDelta * 0.08, 0.0, 1.0);
    const bool dependenciesChanged = std::abs(heightAdjustment) > 1.0e-6 ||
                                     volcanism.slopeContribution > 1.0e-6 ||
                                     std::abs(localWaterDelta) > 1.0e-9;
    if (dependenciesChanged) {
        return refreshDependentSurface(x, z, std::move(result), volcanism);
    }
    result.ecotopes = worldgen::MacroGenerationSampler::classifyEcotopes(result);
    if (volcanism.basaltField > 0.18) result.ecotopes |= worldgen::Ecotope::GEOTHERMAL;
    if (density_.supportsCaveEcotope(static_cast<double>(x), static_cast<double>(z),
                                     result.terrainHeight, result.geology)) {
        result.ecotopes |= worldgen::Ecotope::CAVE;
    }
    return result;
}

worldgen::SurfaceSample ChunkGenerator::emittedSurfaceAt(int64_t x, int64_t z,
                                                         GenScratch& scratch) const {
    return surfaceSampleAt(x, z, scratch);
}

ColumnPlanSurfaceGrid ChunkGenerator::exactSurfaceGrid(const ColumnPlan& plan) const {
    struct DensityColumn {
        DensityColumnContext context;
        std::vector<double> values;
    };
    const int64_t baseX = plan.chunkColumn().x * CHUNK_EDGE;
    const int64_t baseZ = plan.chunkColumn().z * CHUNK_EDGE;
    const int verticalDensitySpacing = densityLatticeVerticalSpacing();
    const int densityLatticeLevels = densityLatticeLevelCount();
    GenScratch scratch;
    scratch.reset(this, scratchToken_.load(std::memory_order_relaxed));
    std::unordered_map<ColumnPos, ColumnShape> shapes;
    std::unordered_map<ColumnPos, DensityColumn> densityColumns;
    shapes.reserve(25);
    densityColumns.reserve(25);

    auto shapeAt = [&](int64_t x, int64_t z) -> const ColumnShape& {
        const ColumnPos key{x, z};
        auto found = shapes.find(key);
        if (found != shapes.end()) return found->second;
        const worldgen::SurfaceSample surface = surfaceSampleFromPlan(x, z, plan, scratch);
        const double caveEntrance =
            climate_.caveEntrance(static_cast<double>(x), static_cast<double>(z));
        return shapes.emplace(key, shapeFromSurface(surface, caveEntrance, learnedAuthority_))
            .first->second;
    };
    auto densityColumnAt = [&](int64_t x, int64_t z) -> DensityColumn& {
        const ColumnPos key{x, z};
        auto found = densityColumns.find(key);
        if (found != densityColumns.end()) return found->second;
        const worldgen::SurfaceSample surface = surfaceSampleFromPlan(x, z, plan, scratch);
        const ColumnShape& shape = shapeAt(x, z);
        const DensityColumnContext context =
            density_.columnContext(static_cast<double>(x), static_cast<double>(z), surface.geology);
        const double yCap = shape.height + shape.detailAmp + 5.0;
        const int maximumEvaluatedLevel =
            maximumEvaluatedDensityLevel(yCap, verticalDensitySpacing);
        DensityColumn column{
            .context = context,
            .values = std::vector<double>(static_cast<size_t>(maximumEvaluatedLevel + 1),
                                          std::numeric_limits<double>::quiet_NaN()),
        };
        return densityColumns.emplace(key, std::move(column)).first->second;
    };
    auto densityValueAt = [&](int64_t x, int64_t z, int level) {
        DensityColumn& column = densityColumnAt(x, z);
        const size_t boundedLevel =
            static_cast<size_t>(std::clamp(level, 0, densityLatticeLevels - 1));
        if (boundedLevel >= column.values.size()) return -DENSITY_CAP;
        double& value = column.values[boundedLevel];
        if (std::isnan(value)) {
            const double y = static_cast<double>(WORLD_MIN_Y + static_cast<int>(boundedLevel) *
                                                                   verticalDensitySpacing);
            value = evaluateDensity(static_cast<double>(x), y, static_cast<double>(z),
                                    shapeAt(x, z), column.context);
        }
        return value;
    };
    auto densityAt = [&](int64_t x, int y, int64_t z) {
        const int64_t x0 = latticeFloor(x);
        const int64_t z0 = latticeFloor(z);
        const double fx = static_cast<double>(x - x0) / LATTICE_XZ;
        const double fz = static_cast<double>(z - z0) / LATTICE_XZ;
        const int level =
            std::clamp((y - WORLD_MIN_Y) / verticalDensitySpacing, 0, densityLatticeLevels - 2);
        const double fy = static_cast<double>(y - (WORLD_MIN_Y + level * verticalDensitySpacing)) /
                          verticalDensitySpacing;
        const double below =
            bilerpDensity(densityValueAt(x0, z0, level), densityValueAt(x0 + LATTICE_XZ, z0, level),
                          densityValueAt(x0, z0 + LATTICE_XZ, level),
                          densityValueAt(x0 + LATTICE_XZ, z0 + LATTICE_XZ, level), fx, fz);
        const double above = bilerpDensity(
            densityValueAt(x0, z0, level + 1), densityValueAt(x0 + LATTICE_XZ, z0, level + 1),
            densityValueAt(x0, z0 + LATTICE_XZ, level + 1),
            densityValueAt(x0 + LATTICE_XZ, z0 + LATTICE_XZ, level + 1), fx, fz);
        return lerpDensity(below, above, fy);
    };

    ColumnPlanSurfaceGrid result{};
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const int64_t x = baseX + localX;
            const int64_t z = baseZ + localZ;
            const ColumnShape& shape = shapeAt(x, z);
            const int start =
                std::clamp(static_cast<int>(std::ceil(shape.height + shape.detailAmp + 8.0)),
                           WORLD_MIN_Y + 2, WORLD_MAX_Y);
            int surfaceY = WORLD_MIN_Y + 1;
            for (int y = start; y > WORLD_MIN_Y + 1; --y) {
                if (densityAt(x, y, z) > 0.0) {
                    surfaceY = y;
                    break;
                }
            }
            worldgen::SurfaceSample surface = surfaceSampleFromPlan(x, z, plan, scratch);
            const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
            surfaceY = boundedLearnedSurfaceY(surface, surfaceY, learnedAuthority_);
            if (!learnedAuthority_ && volcanism.craterLake &&
                volcanism.craterProfileInfluence > 1.0e-4) {
                surfaceY = std::clamp(static_cast<int>(std::ceil(surface.terrainHeight)) - 1,
                                      WORLD_MIN_Y, WORLD_MAX_Y);
            }
            reconcileEmittedSurfaceY(surface.hydrology, surfaceY, learnedAuthority_);
            result[static_cast<size_t>(localZ * CHUNK_EDGE + localX)] =
                static_cast<int16_t>(surfaceY);
        }
    }
    return result;
}

double ChunkGenerator::evaluateDensity(double x, double y, double z, const ColumnShape& shape,
                                       const DensityColumnContext& context) const {
    densityEvaluationCount_.fetch_add(1, std::memory_order_relaxed);
    return density_.density(x, y, z, shape, context);
}

double ChunkGenerator::latticeDensityAt(int64_t lx, int64_t lz, int level,
                                        GenScratch& scratch) const {
    const int verticalDensitySpacing = densityLatticeVerticalSpacing();
    const int densityLatticeLevels = densityLatticeLevelCount();
    const ColumnPos key{lx, lz};
    auto it = scratch.densityColumns.find(key);
    if (it == scratch.densityColumns.end()) {
        const ColumnShape& shape = latticeShape(lx, lz, scratch);
        const worldgen::GeologySample geology = surfaceSampleAt(lx, lz, scratch).geology;
        const double yCap = shape.height + shape.detailAmp + 5.0;
        const int maximumEvaluatedLevel =
            maximumEvaluatedDensityLevel(yCap, verticalDensitySpacing);
        LazyDensityColumn column{
            .context =
                density_.columnContext(static_cast<double>(lx), static_cast<double>(lz), geology),
            .values = std::vector<double>(static_cast<size_t>(maximumEvaluatedLevel + 1),
                                          std::numeric_limits<double>::quiet_NaN()),
        };
        it = scratch.densityColumns.emplace(key, std::move(column)).first;
    }
    LazyDensityColumn& column = it->second;
    const size_t boundedLevel = static_cast<size_t>(std::clamp(level, 0, densityLatticeLevels - 1));
    if (boundedLevel >= column.values.size()) return -DENSITY_CAP;
    double& value = column.values[boundedLevel];
    if (std::isnan(value)) {
        const double y = static_cast<double>(WORLD_MIN_Y + static_cast<int>(boundedLevel) *
                                                               verticalDensitySpacing);
        value = evaluateDensity(static_cast<double>(lx), y, static_cast<double>(lz),
                                latticeShape(lx, lz, scratch), column.context);
    }
    return value;
}

ColumnShape ChunkGenerator::columnShapeAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    const worldgen::SurfaceSample surface = surfaceSampleAt(x, z, scratch);
    const double caveEntrance =
        climate_.caveEntrance(static_cast<double>(x), static_cast<double>(z));
    return shapeFromSurface(surface, caveEntrance, learnedAuthority_);
}

double ChunkGenerator::interpolatedDensity(int64_t x, int y, int64_t z, GenScratch& scratch) const {
    const int verticalDensitySpacing = densityLatticeVerticalSpacing();
    const int densityLatticeLevels = densityLatticeLevelCount();
    const int64_t lx0 = latticeFloor(x);
    const int64_t lz0 = latticeFloor(z);
    const double fx = static_cast<double>(x - lx0) / LATTICE_XZ;
    const double fz = static_cast<double>(z - lz0) / LATTICE_XZ;
    const int level =
        std::clamp((y - WORLD_MIN_Y) / verticalDensitySpacing, 0, densityLatticeLevels - 2);
    const double fy = static_cast<double>(y - (WORLD_MIN_Y + level * verticalDensitySpacing)) /
                      verticalDensitySpacing;
    const double below =
        bilerpDensity(latticeDensityAt(lx0, lz0, level, scratch),
                      latticeDensityAt(lx0 + LATTICE_XZ, lz0, level, scratch),
                      latticeDensityAt(lx0, lz0 + LATTICE_XZ, level, scratch),
                      latticeDensityAt(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, level, scratch), fx, fz);
    const double above = bilerpDensity(
        latticeDensityAt(lx0, lz0, level + 1, scratch),
        latticeDensityAt(lx0 + LATTICE_XZ, lz0, level + 1, scratch),
        latticeDensityAt(lx0, lz0 + LATTICE_XZ, level + 1, scratch),
        latticeDensityAt(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, level + 1, scratch), fx, fz);
    return lerpDensity(below, above, fy);
}

double ChunkGenerator::baseHeightAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    const worldgen::SurfaceSample surface = emittedSurfaceAt(x, z, scratch);
    return learnedAuthority_ ? worldgen::geometryTerrainHeight(surface) : surface.terrainHeight;
}

Biome ChunkGenerator::biomeAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    return emittedSurfaceAt(x, z, scratch).biome.primary;
}

int ChunkGenerator::surfaceYAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
    auto found = scratch.columnPlans.find(column);
    if (found == scratch.columnPlans.end()) {
        found = scratch.columnPlans.emplace(column, getColumnPlan(column)).first;
    }
    return found->second->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
}

GenScratch& ChunkGenerator::threadScratch() const {
    thread_local GenScratch scratch;
    const uint64_t token = scratchToken_.load(std::memory_order_relaxed);
    if (scratch.owner != this || scratch.ownerToken != token) {
        scratch.reset(this, token);
    } else {
        if (scratch.shapes.size() > 4096) scratch.shapes.clear();
        if (scratch.densityColumns.size() > 4096) scratch.densityColumns.clear();
        if (scratch.columnPlans.size() > 128) scratch.columnPlans.clear();
        if (scratch.volcanicColumns.size() > 4096) scratch.volcanicColumns.clear();
        if (scratch.volcanoCells.size() > 1024) {
            scratch.volcanoCells.clear();
            scratch.resolvedVolcanoes.clear();
        }
        if (scratch.volcanicArcCells.size() > 2048) {
            scratch.volcanicArcCells.clear();
            scratch.resolvedVolcanoes.clear();
        }
        if (scratch.structurePlacements.size() > 4096) scratch.structurePlacements.clear();
    }
    return scratch;
}

void ChunkGenerator::prepareScratch(GenScratch& scratch) const {
    const uint64_t token = scratchToken_.load(std::memory_order_relaxed);
    if (scratch.owner != this || scratch.ownerToken != token) {
        scratch.reset(this, token);
    }
}

double ChunkGenerator::baseHeightAt(int64_t x, int64_t z) const {
    return baseHeightAt(x, z, threadScratch());
}

Biome ChunkGenerator::biomeAt(int64_t x, int64_t z) const {
    return biomeAt(x, z, threadScratch());
}

int ChunkGenerator::surfaceYAt(int64_t x, int64_t z) const {
    return surfaceYAt(x, z, threadScratch());
}

void ChunkGenerator::fillColumn(Chunk& chunk, int lx, int lz,
                                const worldgen::SurfaceSample& surface, int surfaceY,
                                GenScratch& scratch) const {
    const int64_t wx = chunk.chunkX * CHUNK_EDGE + lx;
    const int64_t wz = chunk.chunkZ * CHUNK_EDGE + lz;
    const int baseY = chunk.chunkY * CHUNK_EDGE;
    const uint64_t columnHash =
        random_.u64(worldgen::surface_material::DITHER_STREAM, wx, 0, wz, 1);
    const int subsoilDepth = 2 + static_cast<int>(columnHash % 3);
    const Biome materialBiome = ditheredBiome(surface, random_, wx, wz);
    const bool alluvialDeposit = isAlluvialDeposit(surface, random_, wx, wz);
    const bool frozen = worldgen::surface_material::frozen(surface, surface.biome.primary);
    const VolcanicColumnSample& volcanism = volcanismAt(wx, wz, surface.geology, scratch);
    const worldgen::surface_material::VolcanicSignals volcanicSignals = materialSignals(volcanism);
    worldgen::SurfaceSample emittedFluidSurface = surface;
    emittedFluidSurface.terrainHeight = static_cast<double>(surfaceY + 1);
    const worldgen::GeneratedFluidColumn fluidColumn =
        worldgen::generatedFluidColumn(emittedFluidSurface);
    const bool surfaceWater = fluidColumn.wet;
    const bool supportedRapidSill = !learnedAuthority_ && surface.hydrology.generatedFluidLevel > 0;
    const bool preserveCanonicalWater =
        learnedAuthority_ && hasCanonicalWaterFeature(surface.hydrology);
    const bool legacySealedHydrologySupport =
        !learnedAuthority_ && (surface.hydrology.channelBank || surface.hydrology.lakeBank ||
                               surface.hydrology.waterfall);
    const bool coherentCraterProfile =
        !learnedAuthority_ && !preserveCanonicalWater && volcanism.craterLake &&
        volcanism.craterProfileInfluence > 1.0e-4;
    double waterSurface = hasSurfaceWater(surface.hydrology)
                              ? surface.waterSurface
                              : -std::numeric_limits<double>::infinity();
    const bool explicitFallOwner =
        surface.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL &&
        surface.hydrology.transitionOwnerId != 0;
    const bool explicitFallingLip =
        explicitFallOwner && (surface.hydrology.generatedFluidLevel == 7 ||
                              waterSurface <= surface.hydrology.waterfallBottom + 0.125001);
    const bool waterfallOverlay =
        explicitFallingLip && surface.hydrology.waterfall &&
        surface.hydrology.waterfallTop >= surface.hydrology.waterfallBottom + 0.5 &&
        surface.hydrology.waterfallTop >= waterSurface - 0.125;
    const int waterTopY = surfaceWater ? fluidColumn.topY : WORLD_MIN_Y;
    const int waterfallBottomY =
        waterfallOverlay
            ? std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterfallBottom)) - 1,
                         WORLD_MIN_Y, WORLD_MAX_Y)
            : WORLD_MIN_Y;
    const int waterfallTopY =
        waterfallOverlay
            ? std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterfallTop)) - 1,
                         WORLD_MIN_Y, WORLD_MAX_Y)
            : WORLD_MIN_Y;
    const int fallingStartY =
        waterfallOverlay ? std::max(fluidColumn.fallingStartY, waterfallBottomY) : WORLD_MAX_Y + 1;
    const bool submerged = surfaceWater && surfaceY < waterTopY;
    std::optional<BlockType> selectedSurfaceBlock;
    const auto surfaceBlock = [&] {
        if (!selectedSurfaceBlock.has_value()) {
            const worldgen::surface_material::SurfaceMaterialPalette palette =
                worldgen::surface_material::materialPalette(surface, volcanicSignals, frozen,
                                                            submerged, alluvialDeposit);
            selectedSurfaceBlock = worldgen::surface_material::selectMaterial(
                palette, worldgen::multiscaleDitherThreshold(
                             random_, worldgen::surface_material::DITHER_STREAM, wx, wz));
        }
        return *selectedSurfaceBlock;
    };
    std::optional<StrataColumn> strata;

    for (int ly = 0; ly < CHUNK_EDGE; ++ly) {
        const int wy = baseY + ly;
        BlockType block = BlockType::AIR;
        if (wy <= WORLD_MIN_Y + 1 ||
            (wy == WORLD_MIN_Y + 2 && (random_.u32(BEDROCK_STREAM, wx, wy, wz) & 1U) != 0U)) {
            block = BlockType::BEDROCK;
        } else if (waterfallOverlay && wy >= fallingStartY && wy <= waterfallTopY) {
            block = BlockType::WATER;
        } else if (surfaceWater && wy > surfaceY && wy <= waterTopY) {
            block =
                frozen && !waterfallOverlay && wy == waterTopY ? BlockType::ICE : BlockType::WATER;
        } else if ((supportedRapidSill || coherentCraterProfile) && wy <= surfaceY &&
                   wy >= surfaceY -
                             (coherentCraterProfile ? COHERENT_CRATER_SURFACE_DEPTH : CHUNK_EDGE)) {
            const int depth = surfaceY - wy;
            block = depth == 0 ? surfaceBlock()
                               : worldgen::surface_material::subsurface(
                                     surface, materialBiome, volcanicSignals,
                                     coherentCraterProfile && submerged, alluvialDeposit);
        } else if (submerged && wy == surfaceY) {
            block = surfaceBlock();
        } else if (wy == surfaceY ||
                   (wy < surfaceY && (legacySealedHydrologySupport ||
                                      interpolatedDensity(wx, wy, wz, scratch) > 0.0))) {
            const int depth = surfaceY - wy;
            if (depth == 0) {
                block = surfaceBlock();
            } else if (depth > 0 && depth <= subsoilDepth) {
                block = worldgen::surface_material::subsurface(
                    surface, materialBiome, volcanicSignals, submerged, alluvialDeposit);
            } else {
                if (!strata.has_value()) {
                    strata.emplace(strataColumnFor(surface.geology, random_, wx, wz, surfaceY));
                }
                block = strataBlockFor(*strata, wy, depth);
            }
        } else if (wy <= LAVA_LEVEL && wy < surfaceY - 8) {
            block = BlockType::LAVA;
        }

        if (!legacySealedHydrologySupport && block != BlockType::BEDROCK && wy < surfaceY - 8) {
            switch (aquiferVoxelAt(random_, wx, wy, wz, surface)) {
                case AquiferVoxel::NONE:
                    break;
                case AquiferVoxel::SHELL:
                    block = surface.geology.rock == worldgen::RockType::LIMESTONE
                                ? BlockType::LIMESTONE
                                : BlockType::CLAY;
                    break;
                case AquiferVoxel::WATER:
                    block = BlockType::WATER;
                    break;
            }
        }

        if (!legacySealedHydrologySupport && !preserveCanonicalWater &&
            block != BlockType::BEDROCK && volcanism.tubeDistance < volcanism.tubeRadius &&
            wy < surfaceY - 5) {
            const double tubeCenterY = surface.terrainHeight + volcanism.tubeCenterOffset;
            const double verticalDistance = std::abs((static_cast<double>(wy) + 0.5) - tubeCenterY);
            const double tubeCrossSection =
                std::sqrt(std::max(0.0, volcanism.tubeRadius * volcanism.tubeRadius -
                                            volcanism.tubeDistance * volcanism.tubeDistance));
            if (verticalDistance <= tubeCrossSection) {
                const double relativeY = static_cast<double>(wy) + 0.5 - tubeCenterY;
                if (volcanism.tubeLavaBearing && relativeY < -tubeCrossSection * 0.45) {
                    block = BlockType::LAVA;
                } else if (relativeY < -tubeCrossSection * 0.72) {
                    block = BlockType::BASALT;
                } else {
                    block = BlockType::AIR;
                }
            }
        }

        const double conduitBottom = surface.terrainHeight - volcanism.conduitDepth;
        if (!legacySealedHydrologySupport && !preserveCanonicalWater &&
            block != BlockType::BEDROCK && volcanism.centerDistance <= volcanism.conduitRadius &&
            wy >= static_cast<int>(std::floor(conduitBottom)) && wy < surfaceY - 4) {
            const double innerRadius = volcanism.conduitRadius * 0.58;
            block = volcanism.centerDistance <= innerRadius && volcanism.conduitLavaBearing
                        ? BlockType::LAVA
                        : BlockType::OBSIDIAN;
        }
        chunk.setBlock(lx, ly, lz, block);
        const bool generatedFallCell =
            waterfallOverlay && wy >= fallingStartY && wy <= waterfallTopY;
        if (block == BlockType::WATER && generatedFallCell) {
            const bool horizontalLip = wy == waterfallTopY;
            chunk.setFluidState(lx, ly, lz,
                                horizontalLip ? FluidState::flowing(7) : FluidState::falling(7));
        } else if (block == BlockType::WATER && surfaceWater && wy == waterTopY &&
                   !fluidColumn.topState.isSource()) {
            chunk.setFluidState(lx, ly, lz, fluidColumn.topState);
        }
    }
}

void ChunkGenerator::generate(Chunk& chunk) const {
    lastCubeDensityEvaluationCount_.store(0, std::memory_order_relaxed);
    chunk.replaceFluidStates({});
    if (chunk.chunkY < WORLD_MIN_CHUNK_Y || chunk.chunkY > WORLD_MAX_CHUNK_Y) {
        chunk.fill(chunk.chunkY < WORLD_MIN_CHUNK_Y ? BlockType::BEDROCK : BlockType::AIR);
        chunk.generated = true;
        chunk.needsMeshUpdate = true;
        return;
    }

    const ColumnPos chunkColumn{chunk.chunkX, chunk.chunkZ};
    const std::shared_ptr<const ColumnPlan> plan = getColumnPlan(chunkColumn);
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;

    if (!plan->exposesSection(chunk.chunkY) && baseY > plan->maximumSurfaceY() + 24) {
        chunk.fill(BlockType::AIR);
        chunk.generated = true;
        chunk.needsMeshUpdate = true;
        return;
    }

    GenScratch scratch;
    scratch.reset(this, scratchToken_.load(std::memory_order_relaxed));
    scratch.columnPlans.emplace(chunkColumn, plan);
    const uint64_t densityBeforeCube = densityEvaluationCount();

    for (int lz = 0; lz < CHUNK_EDGE; ++lz) {
        for (int lx = 0; lx < CHUNK_EDGE; ++lx) {
            const int64_t wx = baseX + lx;
            const int64_t wz = baseZ + lz;
            const worldgen::SurfaceSample surface = surfaceSampleAt(wx, wz, scratch);
            fillColumn(chunk, lx, lz, surface, plan->surfaceY(lx, lz), scratch);
        }
    }
    lastCubeDensityEvaluationCount_.store(densityEvaluationCount() - densityBeforeCube,
                                          std::memory_order_relaxed);

    ores_.place(chunk, learnedAuthority_ ? OrePlacer::HostPolicy::NATURAL_ROCK
                                         : OrePlacer::HostPolicy::LEGACY_STONE_ONLY);
    if (plan->exposesSection(chunk.chunkY)) {
        structures_.place(chunk, *this, scratch);
        features_.placeTrees(chunk, *this, structures_, scratch);
        features_.placeFlora(chunk, *this, scratch);
    }
    chunk.compactStorage();
    chunk.generated = true;
    chunk.needsMeshUpdate = true;
}

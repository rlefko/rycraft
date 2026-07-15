#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace {

constexpr int LAVA_LEVEL = -96;
constexpr int LATTICE_LEVELS = (WORLD_MAX_Y - WORLD_MIN_Y + 1) / LATTICE_Y + 1;
constexpr int HOTSPOT_QUERY_RADIUS = 1;
constexpr int64_t VOLCANIC_ARC_CELL_EDGE = 1024;
constexpr int64_t AQUIFER_CELL_EDGE = 64;
constexpr int AQUIFER_CELL_HEIGHT = 32;
constexpr int COHERENT_CRATER_SURFACE_DEPTH = 8;

constexpr uint64_t HOTSPOT_PROPERTIES_STREAM = 0x1202;
constexpr uint64_t VOLCANIC_ARC_STREAM = 0x564F4C4341524331ULL;
constexpr uint64_t BEDROCK_STREAM = 0x424544524F434B31ULL;
constexpr uint64_t AQUIFER_STREAM = 0x4151554946455231ULL;
constexpr uint64_t STRATA_STREAM = 0x5354524154413031ULL;

uint64_t nextGeneratorInstanceToken() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

int64_t latticeFloor(int64_t value) {
    return world_coord::floorDiv(value, static_cast<int64_t>(LATTICE_XZ)) * LATTICE_XZ;
}

bool hasSurfaceWater(const worldgen::HydrologySample& hydrology) {
    return hydrology.ocean || hydrology.river || hydrology.lake;
}

bool isSupportedRiverBank(const worldgen::HydrologySample& hydrology) {
    return !hasSurfaceWater(hydrology) && hydrology.channelWidth > 0.0 &&
           hydrology.channelDistance <=
               hydrology.channelWidth * 0.55 + worldgen::BASIN_RASTER_SPACING &&
           hydrology.surfaceElevation + 0.01 >= hydrology.waterSurface;
}

bool isSupportedLakeBank(const worldgen::HydrologySample& hydrology) {
    return !hasSurfaceWater(hydrology) && hydrology.lakeBank &&
           hydrology.lakeBankInfluence > 1.0e-4 && hydrology.shoreWaterSurface > 0.0;
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
    hydrology.endorheic = false;
    hydrology.waterfall = false;
    hydrology.waterfallAnchor = false;
    hydrology.delta = false;
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
    hydrology.streamOrder = 0;
    hydrology.distributaryCount = 0;
}

bool applyCraterLake(worldgen::SurfaceSample& surface, const VolcanicColumnSample& volcanism,
                     bool applyOuterProfile) {
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
    surface.terrainHeight = std::clamp(surface.terrainHeight, -112.0, 480.0);
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
            surface.hydrology.lakeBankTarget =
                std::max(surface.terrainHeight, std::ceil(volcanism.craterLakeSurface));
            surface.terrainHeight =
                std::max(surface.terrainHeight, surface.hydrology.lakeBankTarget);
            surface.hydrology.surfaceElevation = surface.terrainHeight;
        }
        surface.waterSurface = 0.0;
        return false;
    }
    clearSurfaceWater(surface.hydrology);
    surface.hydrology.lake = true;
    surface.hydrology.endorheic = true;
    surface.hydrology.lakeShoreDistance = shoreDistance;
    surface.hydrology.shoreWaterSurface = volcanism.craterLakeSurface;
    surface.hydrology.lakeDepth = volcanism.craterLakeSurface - surface.terrainHeight;
    surface.hydrology.waterSurface = volcanism.craterLakeSurface;
    surface.waterSurface = volcanism.craterLakeSurface;
    return true;
}

double emittedVolcanicHeight(const worldgen::SurfaceSample& macroSurface,
                             const VolcanicColumnSample& volcanism) {
    double heightAdjustment = volcanism.heightAdjustment;
    if (macroSurface.geology.crust == worldgen::CrustType::OCEANIC &&
        macroSurface.hydrology.ocean && volcanism.strongestProfile > 0.38) {
        const double islandLift =
            (static_cast<double>(SEA_LEVEL + 6) - macroSurface.terrainHeight) *
            volcanism.strongestProfile;
        heightAdjustment = std::max(heightAdjustment, std::min(18.0, islandLift));
    }
    return std::clamp(macroSurface.terrainHeight + std::clamp(heightAdjustment, -18.0, 18.0),
                      -112.0, 480.0);
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
    double verticalOffset = 0.0;
};

StrataColumn strataColumnFor(const worldgen::GeologySample& geology, const CounterRng& random,
                             int64_t x, int64_t z) {
    const uint64_t plateStream = STRATA_STREAM ^ geology.plateId;
    const double direction = random.uniform01(plateStream, 0, 0, 0, 0) * std::numbers::pi * 2.0;
    const double dip = 0.018 + random.uniform01(plateStream, 0, 0, 0, 1) * 0.052;
    const double phase = random.uniform01(plateStream, 0, 0, 0, 2) * 96.0;
    const double foldWavelength = 720.0 + random.uniform01(plateStream, 0, 0, 0, 3) * 1360.0;
    const double foldPhase = random.uniform01(plateStream, 0, 0, 0, 4) * std::numbers::pi * 2.0;
    const double along =
        static_cast<double>(x) * std::cos(direction) + static_cast<double>(z) * std::sin(direction);
    const double across = -static_cast<double>(x) * std::sin(direction) +
                          static_cast<double>(z) * std::cos(direction);
    const double fold =
        std::sin(across / foldWavelength * std::numbers::pi * 2.0 + foldPhase) * 3.5;
    // 7,920 is the least common multiple of every layer period below. The
    // periodic reduction keeps large coordinates numerically stable without
    // introducing a visible reset into any material sequence.
    const double offset = std::remainder(phase + along * dip + fold, 7920.0);
    return {geology, offset};
}

BlockType strataBlockFor(const StrataColumn& strata, int y, int depth) {
    const worldgen::GeologySample& geology = strata.geology;
    const int layerCoordinate = static_cast<int>(std::floor(y + strata.verticalOffset));
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

ColumnShape shapeFromSurface(const worldgen::SurfaceSample& surface,
                             const ColumnShape& legacyShape) {
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
    result.detailAmp = waterCovered
                           ? 0.75
                           : std::clamp(2.0 + surface.slope * 2.5 + surface.geology.uplift * 7.0 +
                                            surface.geology.faultStrength * 6.0 +
                                            surface.geology.volcanicActivity * 4.0,
                                        1.5, 14.0);
    result.entrance = waterCovered ? -1.0 : legacyShape.entrance;
    result.riverCut = surface.hydrology.erosionDepth;
    result.ravineEdge = 0.0;
    result.ravineFloor = waterCovered ? SEA_LEVEL - 1.0 : result.height;
    return result;
}

} // namespace

ChunkGenerator::ChunkGenerator(uint32_t worldSeed)
    : seed_(worldSeed)
    , scratchToken_(nextGeneratorInstanceToken())
    , random_(worldSeed)
    , macroSampler_(worldSeed)
    , columnPlanCache_()
    , climate_(worldSeed)
    , density_(worldSeed)
    , ores_(worldSeed)
    , structures_(worldSeed)
    , features_(worldSeed) {}

void ChunkGenerator::clearMacroCaches() const {
    columnPlanCache_.clear();
    macroSampler_.clearBasinCache();
    scratchToken_.store(nextGeneratorInstanceToken(), std::memory_order_relaxed);
}

const ColumnShape& ChunkGenerator::latticeShape(int64_t lx, int64_t lz, GenScratch& scratch) const {
    const ColumnPos key{lx, lz};
    auto it = scratch.shapes.find(key);
    if (it != scratch.shapes.end()) return it->second;
    const worldgen::SurfaceSample surface = surfaceSampleAt(lx, lz, scratch);
    const ColumnShape legacy =
        climate_.shapeColumn(static_cast<double>(lx), static_cast<double>(lz));
    ColumnShape shape = shapeFromSurface(surface, legacy);
    return scratch.shapes.emplace(key, shape).first->second;
}

std::shared_ptr<const ColumnPlan> ChunkGenerator::getColumnPlan(ColumnPos chunkColumn) const {
    return columnPlanCache_.getOrCreate(
        chunkColumn, [this](int64_t x, int64_t z) { return sampleFarSurface(x, z); },
        [this](int64_t x, int64_t z) { return sampleFarTerrainHeight(x, z); },
        [this](const ColumnPlan& plan) { return exactSurfaceGrid(plan); },
        [this](int64_t x, int64_t z) {
            return macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
        });
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
    if (world_coord::floorMod(x, COLUMN_PLAN_LATTICE_SPACING) == 0 &&
        world_coord::floorMod(z, COLUMN_PLAN_LATTICE_SPACING) == 0) {
        return result;
    }
    if (std::abs(result.hydrology.lakeShoreDistance) <= 24.0) {
        const worldgen::HydrologySample exactHydrology =
            macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
        result.hydrology.lakeShoreDistance = exactHydrology.lakeShoreDistance;
        result.hydrology.shoreWaterSurface = exactHydrology.shoreWaterSurface;
        result.hydrology.lakeBankTarget = exactHydrology.lakeBankTarget;
        result.hydrology.lakeBankInfluence = exactHydrology.lakeBankInfluence;
        result.hydrology.lakeBank = exactHydrology.lakeBank;
        if (exactHydrology.lakeBank) {
            result.terrainHeight = std::max(result.terrainHeight, exactHydrology.surfaceElevation);
            result.hydrology.surfaceElevation = result.terrainHeight;
        }
    }
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    if (result.hydrology.waterfall) {
        const worldgen::HydrologySample exactHydrology =
            macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
        result.hydrology.waterfall = exactHydrology.waterfall;
        result.hydrology.waterfallAnchor = exactHydrology.waterfallAnchor;
        result.hydrology.waterfallTop = exactHydrology.waterfallTop;
        result.hydrology.waterfallBottom = exactHydrology.waterfallBottom;
        result.hydrology.waterfallWidth = exactHydrology.waterfallWidth;
        if (exactHydrology.waterfall) {
            result.hydrology.waterSurface = exactHydrology.waterSurface;
            result.hydrology.flowDirection = exactHydrology.flowDirection;
            result.waterSurface = exactHydrology.waterSurface;
        }
    }
    const double priorWaterInfluence = localWaterClimateInfluence(result.hydrology);
    if (applyCraterLake(result, volcanism, false)) {
        const double localWaterDelta =
            localWaterClimateInfluence(result.hydrology) - priorWaterInfluence;
        result.climate.temperatureC +=
            (13.0 - result.climate.temperatureC) * localWaterDelta * 0.08;
        result.climate.annualPrecipitationMm = std::clamp(
            result.climate.annualPrecipitationMm * (1.0 + localWaterDelta * 0.08), 60.0, 3600.0);
        result.climate.relativeHumidity =
            std::clamp(result.climate.relativeHumidity + localWaterDelta * 0.08, 0.0, 1.0);
    }
    return refreshDependentSurface(x, z, std::move(result), volcanism);
}

worldgen::SurfaceSample ChunkGenerator::surfaceSampleAt(int64_t x, int64_t z,
                                                        GenScratch& scratch) const {
    const ColumnPos chunkColumn{
        world_coord::floorDiv(x, static_cast<int64_t>(CHUNK_EDGE)),
        world_coord::floorDiv(z, static_cast<int64_t>(CHUNK_EDGE)),
    };
    auto found = scratch.columnPlans.find(chunkColumn);
    if (found == scratch.columnPlans.end()) {
        found = scratch.columnPlans.emplace(chunkColumn, getColumnPlan(chunkColumn)).first;
    }
    return surfaceSampleFromPlan(x, z, *found->second, scratch);
}

worldgen::SurfaceSample ChunkGenerator::sampleSurface(int64_t x, int64_t z) const {
    return surfaceSampleAt(x, z, threadScratch());
}

worldgen::SurfaceSample ChunkGenerator::sampleFarGeometrySurface(int64_t x, int64_t z) const {
    worldgen::SurfaceSample result;
    result.geology = macroSampler_.sampleGeology(static_cast<double>(x), static_cast<double>(z));
    result.hydrology =
        macroSampler_.sampleHydrology(static_cast<double>(x), static_cast<double>(z));
    result.terrainHeight = result.hydrology.surfaceElevation;
    result.waterSurface = result.hydrology.waterSurface;
    GenScratch& scratch = threadScratch();
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, result.geology, scratch);
    applyVolcanicGeometry(result, volcanism);
    return result;
}

worldgen::SurfaceSample ChunkGenerator::sampleExactGeometrySurface(int64_t x, int64_t z) const {
    return sampleExactSurface(x, z);
}

worldgen::SurfaceSample ChunkGenerator::sampleExactSurface(int64_t x, int64_t z) const {
    GenScratch& scratch = threadScratch();
    worldgen::SurfaceSample result = surfaceSampleAt(x, z, scratch);
    const ColumnPos chunkColumn{
        world_coord::floorDiv(x, static_cast<int64_t>(CHUNK_EDGE)),
        world_coord::floorDiv(z, static_cast<int64_t>(CHUNK_EDGE)),
    };
    const auto found = scratch.columnPlans.find(chunkColumn);
    if (found == scratch.columnPlans.end()) {
        throw std::logic_error("exact surface sample did not retain its column plan");
    }
    const double emittedTop = static_cast<double>(
        found->second->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z)) + 1);
    result.terrainHeight = emittedTop;
    result.hydrology.surfaceElevation = emittedTop;
    if (result.hydrology.lake) {
        result.hydrology.lakeDepth = std::max(0.0, result.hydrology.waterSurface - emittedTop);
        result.waterSurface = result.hydrology.waterSurface;
    }
    return result;
}

worldgen::SurfaceSample ChunkGenerator::sampleFarSurface(int64_t x, int64_t z) const {
    return applyVolcanism(
        x, z, macroSampler_.sampleSurface(static_cast<double>(x), static_cast<double>(z)),
        threadScratch());
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
    const Biome biome = ditheredBiome(surface, random_, x, z);
    return worldgen::surface_material::surface(surface, biome, materialSignals(volcanism),
                                               worldgen::surface_material::frozen(surface, biome),
                                               submerged,
                                               isAlluvialDeposit(surface, random_, x, z));
}

BlockType ChunkGenerator::farSurfaceMaterialAt(int64_t x, int64_t z) const {
    GenScratch& scratch = threadScratch();
    const worldgen::SurfaceSample surface = sampleFarSurface(x, z);
    const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
    const Biome biome = ditheredBiome(surface, random_, x, z);
    return worldgen::surface_material::surface(surface, biome, materialSignals(volcanism),
                                               worldgen::surface_material::frozen(surface, biome),
                                               worldgen::surface_material::submerged(surface),
                                               isAlluvialDeposit(surface, random_, x, z));
}

double ChunkGenerator::sampleFarTerrainHeight(int64_t x, int64_t z) const {
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
    const auto& volcanoes = volcanoesForCell(cellX, cellZ, threadScratch());
    return {volcanoes.begin(), volcanoes.end()};
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
    if (lodStep <= 4) {
        return collectFarCanopies(minimumX, minimumZ, maximumX, maximumZ);
    }
    return features_.collectFarCanopyClusters(minimumX, minimumZ, maximumX, maximumZ, lodStep,
                                              *this);
}

const std::vector<VolcanoPrimitive>& ChunkGenerator::volcanoesForCell(int64_t cellX, int64_t cellZ,
                                                                      GenScratch& scratch) const {
    const ColumnPos key{cellX, cellZ};
    auto found = scratch.volcanoCells.find(key);
    if (found != scratch.volcanoCells.end()) return found->second;

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
            const double craterDepth = shield ? 15.0 + sizeRoll * 10.0 : 8.0 + sizeRoll * 8.0;
            const double centerElevation = macroSampler_.preliminaryElevation(centerX, centerZ);
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
            configureCraterLake(volcano);
            volcanoes.push_back(volcano);
        }
    }
    return scratch.volcanoCells.emplace(key, std::move(volcanoes)).first->second;
}

const std::vector<VolcanoPrimitive>&
ChunkGenerator::volcanicArcForCell(int64_t cellX, int64_t cellZ, GenScratch& scratch) const {
    const ColumnPos key{cellX, cellZ};
    auto found = scratch.volcanicArcCells.find(key);
    if (found != scratch.volcanicArcCells.end()) return found->second;

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
        const double centerElevation = macroSampler_.preliminaryElevation(centerX, centerZ);
        VolcanoPrimitive volcano{
            .centerX = centerX,
            .centerZ = centerZ,
            .radius = 230.0 + sizeRoll * 210.0,
            .coneHeight = coneHeight,
            .craterRadius = 21.0 + sizeRoll * 26.0,
            .craterDepth = craterDepth,
            .craterDatumElevation = centerElevation,
            .tubePhase =
                random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 4) * 2.0 * std::numbers::pi,
            .conduitRadius = 2.3 + random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 5) * 1.7,
            .id = random_.u64(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 6),
            .shield = false,
            .craterLake = random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 7) < 0.28,
            .lavaBearing = random_.uniform01(VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 8) < 0.72,
        };
        configureCraterLake(volcano);
        volcanoes.push_back(volcano);
    }
    return scratch.volcanicArcCells.emplace(key, std::move(volcanoes)).first->second;
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
    auto accumulate = [&](const VolcanoPrimitive& volcano) {
        const double offsetX = static_cast<double>(x) + 0.5 - volcano.centerX;
        const double offsetZ = static_cast<double>(z) + 0.5 - volcano.centerZ;
        const double distance = std::hypot(offsetX, offsetZ);
        if (distance > volcano.radius * 1.20) return;

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

        const double craterScore = profileDistance / std::max(1.0, volcano.craterRadius);
        if (volcano.craterLake && distance < volcano.radius && craterScore < selectedCraterScore) {
            selectedCraterScore = craterScore;
            result.craterFactor = crater;
            result.craterRadius = volcano.craterRadius;
            result.craterProfileDistance = profileDistance;
            result.craterTerrainTarget = craterProfileHeight(volcano, profileDistance);
            result.craterProfileInfluence =
                1.0 - smootherstep(volcano.craterRadius, volcano.radius, distance);
            result.craterLakeRadius = volcano.craterLakeRadius;
            result.craterLakeSurface = volcano.craterLakeSurface;
            result.craterRimElevation = volcano.craterRimElevation;
            result.craterRimWidth = volcano.craterRimWidth;
            result.craterLake = true;
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
    const double windSpeed = std::hypot(result.climate.wind.x, result.climate.wind.z);
    result.climate.potentialEvapotranspirationMm =
        std::clamp(300.0 + std::max(-8.0, result.climate.temperatureC) * 31.0 + windSpeed * 170.0,
                   120.0, 1800.0);
    result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                             std::max(1.0, result.climate.annualPrecipitationMm);
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
    result.terrainHeight = emittedVolcanicHeight(result, volcanism);
    result.hydrology.surfaceElevation = result.terrainHeight;

    if (applyCraterLake(result, volcanism, true)) {
        // The crater water is emitted at its analytical spill surface.
    } else if (result.terrainHeight < SEA_LEVEL &&
               (!result.hydrology.river || result.hydrology.delta) && !result.hydrology.lake) {
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
    }
    if (result.hydrology.lake) {
        result.hydrology.lakeDepth = result.hydrology.waterSurface - result.terrainHeight;
    }
    result.waterSurface = result.hydrology.waterSurface;
}

worldgen::SurfaceSample ChunkGenerator::applyVolcanism(int64_t x, int64_t z,
                                                       worldgen::SurfaceSample result,
                                                       GenScratch& scratch) const {
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
    result.climate.temperatureC -= (newClimateHeight - oldClimateHeight) * 8.0 * 0.0065;

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
    return refreshDependentSurface(x, z, std::move(result), volcanism);
}

worldgen::SurfaceSample ChunkGenerator::emittedSurfaceAt(int64_t x, int64_t z,
                                                         GenScratch& scratch) const {
    return surfaceSampleAt(x, z, scratch);
}

ColumnPlanSurfaceGrid ChunkGenerator::exactSurfaceGrid(const ColumnPlan& plan) const {
    using DensityColumn = std::array<double, LATTICE_LEVELS>;
    const int64_t baseX = plan.chunkColumn().x * CHUNK_EDGE;
    const int64_t baseZ = plan.chunkColumn().z * CHUNK_EDGE;
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
        const ColumnShape legacy =
            climate_.shapeColumn(static_cast<double>(x), static_cast<double>(z));
        return shapes.emplace(key, shapeFromSurface(surface, legacy)).first->second;
    };
    auto densityColumnAt = [&](int64_t x, int64_t z) -> const DensityColumn& {
        const ColumnPos key{x, z};
        auto found = densityColumns.find(key);
        if (found != densityColumns.end()) return found->second;
        const worldgen::SurfaceSample surface = surfaceSampleFromPlan(x, z, plan, scratch);
        const ColumnShape& shape = shapeAt(x, z);
        const DensityColumnContext context =
            density_.columnContext(static_cast<double>(x), static_cast<double>(z), surface.geology);
        DensityColumn column{};
        const double yCap = shape.height + shape.detailAmp + 5.0;
        for (int level = 0; level < LATTICE_LEVELS; ++level) {
            const double y = static_cast<double>(WORLD_MIN_Y + level * LATTICE_Y);
            column[static_cast<size_t>(level)] =
                y > yCap ? -DENSITY_CAP
                         : density_.density(static_cast<double>(x), y, static_cast<double>(z),
                                            shape, context);
        }
        return densityColumns.emplace(key, std::move(column)).first->second;
    };
    auto densityAt = [&](int64_t x, int y, int64_t z) {
        const int64_t x0 = latticeFloor(x);
        const int64_t z0 = latticeFloor(z);
        const DensityColumn& c00 = densityColumnAt(x0, z0);
        const DensityColumn& c10 = densityColumnAt(x0 + LATTICE_XZ, z0);
        const DensityColumn& c01 = densityColumnAt(x0, z0 + LATTICE_XZ);
        const DensityColumn& c11 = densityColumnAt(x0 + LATTICE_XZ, z0 + LATTICE_XZ);
        const double fx = static_cast<double>(x - x0) / LATTICE_XZ;
        const double fz = static_cast<double>(z - z0) / LATTICE_XZ;
        const int level = (y - WORLD_MIN_Y) / LATTICE_Y;
        const double fy = static_cast<double>(y - (WORLD_MIN_Y + level * LATTICE_Y)) / LATTICE_Y;
        const double below =
            bilerpDensity(c00[static_cast<size_t>(level)], c10[static_cast<size_t>(level)],
                          c01[static_cast<size_t>(level)], c11[static_cast<size_t>(level)], fx, fz);
        const double above = bilerpDensity(
            c00[static_cast<size_t>(level + 1)], c10[static_cast<size_t>(level + 1)],
            c01[static_cast<size_t>(level + 1)], c11[static_cast<size_t>(level + 1)], fx, fz);
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
            const worldgen::SurfaceSample surface = surfaceSampleFromPlan(x, z, plan, scratch);
            const VolcanicColumnSample& volcanism = volcanismAt(x, z, surface.geology, scratch);
            if (volcanism.craterLake && volcanism.craterProfileInfluence > 1.0e-4) {
                surfaceY = std::clamp(static_cast<int>(std::ceil(surface.terrainHeight)) - 1,
                                      WORLD_MIN_Y, WORLD_MAX_Y);
            }
            if (surface.hydrology.lake) {
                const int waterTopY =
                    std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterSurface)) - 1,
                               WORLD_MIN_Y, WORLD_MAX_Y);
                const int plannedFloorY =
                    std::clamp(static_cast<int>(std::ceil(surface.hydrology.surfaceElevation)) - 1,
                               WORLD_MIN_Y, WORLD_MAX_Y);
                surfaceY = std::max(surfaceY, plannedFloorY);
                surfaceY = std::min(surfaceY, std::max(WORLD_MIN_Y, waterTopY - 1));
            }
            if (isSupportedLakeBank(surface.hydrology)) {
                const int bankTopY =
                    std::clamp(static_cast<int>(std::ceil(surface.hydrology.lakeBankTarget)) - 1,
                               WORLD_MIN_Y, WORLD_MAX_Y);
                surfaceY = std::max(surfaceY, bankTopY);
            }
            if (isSupportedRiverBank(surface.hydrology)) {
                const int bankTopY =
                    std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterSurface)) - 1,
                               WORLD_MIN_Y, WORLD_MAX_Y);
                surfaceY = std::max(surfaceY, bankTopY);
            }
            if (surface.hydrology.waterfall &&
                surface.hydrology.waterfallTop >= surface.hydrology.waterfallBottom + 0.5) {
                const int receivingWaterY =
                    std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterfallBottom)) - 1,
                               WORLD_MIN_Y, WORLD_MAX_Y);
                surfaceY = std::min(surfaceY, std::max(WORLD_MIN_Y, receivingWaterY - 1));
            }
            result[static_cast<size_t>(localZ * CHUNK_EDGE + localX)] =
                static_cast<int16_t>(surfaceY);
        }
    }
    return result;
}

const std::vector<double>& ChunkGenerator::latticeDensityColumn(int64_t lx, int64_t lz,
                                                                GenScratch& scratch) const {
    const ColumnPos key{lx, lz};
    auto it = scratch.densityColumns.find(key);
    if (it != scratch.densityColumns.end()) return it->second;

    const ColumnShape& shape = latticeShape(lx, lz, scratch);
    const worldgen::GeologySample geology = surfaceSampleAt(lx, lz, scratch).geology;
    const DensityColumnContext densityContext =
        density_.columnContext(static_cast<double>(lx), static_cast<double>(lz), geology);
    std::vector<double> column(LATTICE_LEVELS);
    const double yCap = shape.height + shape.detailAmp + 5.0;
    for (int level = 0; level < LATTICE_LEVELS; ++level) {
        const double y = static_cast<double>(WORLD_MIN_Y + level * LATTICE_Y);
        column[level] = y > yCap ? -DENSITY_CAP
                                 : density_.density(static_cast<double>(lx), y,
                                                    static_cast<double>(lz), shape, densityContext);
    }
    return scratch.densityColumns.emplace(key, std::move(column)).first->second;
}

ColumnShape ChunkGenerator::columnShapeAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    const worldgen::SurfaceSample surface = surfaceSampleAt(x, z, scratch);
    const ColumnShape legacy = climate_.shapeColumn(static_cast<double>(x), static_cast<double>(z));
    return shapeFromSurface(surface, legacy);
}

double ChunkGenerator::interpolatedDensity(int64_t x, int y, int64_t z, GenScratch& scratch) const {
    const int64_t lx0 = latticeFloor(x);
    const int64_t lz0 = latticeFloor(z);
    const auto& c00 = latticeDensityColumn(lx0, lz0, scratch);
    const auto& c10 = latticeDensityColumn(lx0 + LATTICE_XZ, lz0, scratch);
    const auto& c01 = latticeDensityColumn(lx0, lz0 + LATTICE_XZ, scratch);
    const auto& c11 = latticeDensityColumn(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, scratch);
    const double fx = static_cast<double>(x - lx0) / LATTICE_XZ;
    const double fz = static_cast<double>(z - lz0) / LATTICE_XZ;
    const int level = (y - WORLD_MIN_Y) / LATTICE_Y;
    const double fy = static_cast<double>(y - (WORLD_MIN_Y + level * LATTICE_Y)) / LATTICE_Y;
    const double below = bilerpDensity(c00[level], c10[level], c01[level], c11[level], fx, fz);
    const double above =
        bilerpDensity(c00[level + 1], c10[level + 1], c01[level + 1], c11[level + 1], fx, fz);
    return lerpDensity(below, above, fy);
}

double ChunkGenerator::baseHeightAt(int64_t x, int64_t z, GenScratch& scratch) const {
    prepareScratch(scratch);
    return emittedSurfaceAt(x, z, scratch).terrainHeight;
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
    if (scratch.owner != this || scratch.ownerToken != token || scratch.shapes.size() > 4096 ||
        scratch.columnPlans.size() > 128 || scratch.volcanicColumns.size() > 4096 ||
        scratch.volcanoCells.size() > 1024 || scratch.volcanicArcCells.size() > 2048) {
        scratch.reset(this, token);
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
    const bool frozen = worldgen::surface_material::frozen(surface, materialBiome);
    const VolcanicColumnSample& volcanism = volcanismAt(wx, wz, surface.geology, scratch);
    const worldgen::surface_material::VolcanicSignals volcanicSignals = materialSignals(volcanism);
    const bool surfaceWater = hasSurfaceWater(surface.hydrology);
    const bool supportedRiverBank = isSupportedRiverBank(surface.hydrology);
    const bool supportedLakeBank = isSupportedLakeBank(surface.hydrology);
    const bool coherentCraterProfile =
        volcanism.craterLake && volcanism.craterProfileInfluence > 1.0e-4;
    double waterSurface = hasSurfaceWater(surface.hydrology)
                              ? surface.waterSurface
                              : -std::numeric_limits<double>::infinity();
    const bool waterfallOverlay =
        surface.hydrology.waterfall &&
        surface.hydrology.waterfallTop >= surface.hydrology.waterfallBottom + 0.5 &&
        surface.hydrology.waterfallTop >= waterSurface + 0.5;
    const int waterTopY = surfaceWater ? std::clamp(static_cast<int>(std::ceil(waterSurface)) - 1,
                                                    WORLD_MIN_Y, WORLD_MAX_Y)
                                       : WORLD_MIN_Y;
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
    const int fallingStartY = std::max(waterTopY, waterfallBottomY);
    const bool submerged = surfaceWater && surfaceY < waterTopY;
    const StrataColumn strata = strataColumnFor(surface.geology, random_, wx, wz);

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
        } else if ((supportedRiverBank || supportedLakeBank || coherentCraterProfile) &&
                   wy <= surfaceY &&
                   wy >= surfaceY -
                             (coherentCraterProfile ? COHERENT_CRATER_SURFACE_DEPTH : CHUNK_EDGE)) {
            const int depth = surfaceY - wy;
            block = depth == 0 ? worldgen::surface_material::surface(
                                     surface, materialBiome, volcanicSignals, frozen,
                                     coherentCraterProfile && submerged, alluvialDeposit)
                               : worldgen::surface_material::subsurface(
                                     surface, materialBiome, volcanicSignals,
                                     coherentCraterProfile && submerged, alluvialDeposit);
        } else if (submerged && wy == surfaceY) {
            block = worldgen::surface_material::surface(surface, materialBiome, volcanicSignals,
                                                        frozen, true, alluvialDeposit);
        } else if (interpolatedDensity(wx, wy, wz, scratch) > 0.0) {
            const int depth = surfaceY - wy;
            if (depth == 0) {
                block = worldgen::surface_material::surface(surface, materialBiome, volcanicSignals,
                                                            frozen, submerged, alluvialDeposit);
            } else if (depth > 0 && depth <= subsoilDepth) {
                block = worldgen::surface_material::subsurface(
                    surface, materialBiome, volcanicSignals, submerged, alluvialDeposit);
            } else {
                block = strataBlockFor(strata, wy, depth);
            }
        } else if (wy <= LAVA_LEVEL && wy < surfaceY - 8) {
            block = BlockType::LAVA;
        }

        if (block != BlockType::BEDROCK && wy < surfaceY - 8) {
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

        if (block != BlockType::BEDROCK && volcanism.tubeDistance < volcanism.tubeRadius &&
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
        if (block != BlockType::BEDROCK && volcanism.centerDistance <= volcanism.conduitRadius &&
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
            chunk.setFluidState(lx, ly, lz, FluidState::falling());
        }
    }
}

void ChunkGenerator::generate(Chunk& chunk) const {
    chunk.replaceFluidStates({});
    if (chunk.chunkY < WORLD_MIN_CHUNK_Y || chunk.chunkY > WORLD_MAX_CHUNK_Y) {
        chunk.fill(chunk.chunkY < WORLD_MIN_CHUNK_Y ? BlockType::BEDROCK : BlockType::AIR);
        chunk.generated = true;
        chunk.needsMeshUpdate = true;
        return;
    }

    GenScratch scratch;
    scratch.reset(this, scratchToken_.load(std::memory_order_relaxed));
    const ColumnPos chunkColumn{chunk.chunkX, chunk.chunkZ};
    const std::shared_ptr<const ColumnPlan> plan = getColumnPlan(chunkColumn);
    scratch.columnPlans.emplace(chunkColumn, plan);
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;

    if (!plan->exposesSection(chunk.chunkY) && baseY > plan->maximumSurfaceY() + 24) {
        chunk.fill(BlockType::AIR);
        chunk.generated = true;
        chunk.needsMeshUpdate = true;
        return;
    }

    for (int lz = 0; lz < CHUNK_EDGE; ++lz) {
        for (int lx = 0; lx < CHUNK_EDGE; ++lx) {
            const int64_t wx = baseX + lx;
            const int64_t wz = baseZ + lz;
            const worldgen::SurfaceSample surface = surfaceSampleAt(wx, wz, scratch);
            fillColumn(chunk, lx, lz, surface, plan->surfaceY(lx, lz), scratch);
        }
    }

    ores_.place(chunk);
    if (plan->exposesSection(chunk.chunkY)) {
        structures_.place(chunk, *this, scratch);
        features_.placeTrees(chunk, *this, structures_, scratch);
        features_.placeFlora(chunk, *this, scratch);
    }
    chunk.compactStorage();
    chunk.generated = true;
    chunk.needsMeshUpdate = true;
}

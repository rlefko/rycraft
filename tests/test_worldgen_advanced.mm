#include <catch2/catch_all.hpp>

#include "common/random.hpp"
#include "world/block_properties.hpp"
#include "world/chunk_generator.hpp"
#include "world/column_plan.hpp"
#include "world/macro_generation.hpp"
#include "world/save_manager.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

uint64_t advancedCubeHash(const Chunk& cube) {
    uint64_t hash = hash64(static_cast<uint64_t>(cube.chunkX));
    hash ^= hash64(static_cast<uint64_t>(static_cast<uint32_t>(cube.chunkY)));
    hash ^= hash64(static_cast<uint64_t>(cube.chunkZ));
    for (int y = 0; y < CHUNK_EDGE; ++y) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            for (int x = 0; x < CHUNK_EDGE; ++x) {
                const uint64_t address = static_cast<uint64_t>(Chunk::index(x, y, z));
                const uint64_t block = static_cast<uint8_t>(cube.getBlock(x, y, z));
                const uint64_t fluid = cube.getFluidState(x, y, z).packed();
                hash = hash64(hash ^ (address << 16U) ^ (block << 8U) ^ fluid);
            }
        }
    }
    return hash;
}

using CubeHashes = std::map<std::tuple<int64_t, int32_t, int64_t>, uint64_t>;

CubeHashes generateHashes(const ChunkGenerator& generator, const std::vector<ChunkPos>& positions) {
    CubeHashes result;
    for (ChunkPos position : positions) {
        Chunk cube(position);
        generator.generateCube(cube);
        result.emplace(std::tuple{position.x, position.y, position.z}, advancedCubeHash(cube));
    }
    return result;
}

void requireExactSurface(const worldgen::SurfaceSample& actual,
                         const worldgen::SurfaceSample& expected) {
    REQUIRE(actual.geology.plateId == expected.geology.plateId);
    REQUIRE(actual.geology.crust == expected.geology.crust);
    REQUIRE(actual.geology.boundary == expected.geology.boundary);
    REQUIRE(actual.geology.rock == expected.geology.rock);
    REQUIRE(actual.geology.plateVelocity.x == expected.geology.plateVelocity.x);
    REQUIRE(actual.geology.plateVelocity.z == expected.geology.plateVelocity.z);
    REQUIRE(actual.geology.continentalFraction == expected.geology.continentalFraction);
    REQUIRE(actual.geology.crustAge == expected.geology.crustAge);
    REQUIRE(actual.geology.crustThickness == expected.geology.crustThickness);
    REQUIRE(actual.geology.crustDensity == expected.geology.crustDensity);
    REQUIRE(actual.geology.distanceToBoundary == expected.geology.distanceToBoundary);
    REQUIRE(actual.geology.uplift == expected.geology.uplift);
    REQUIRE(actual.geology.rift == expected.geology.rift);
    REQUIRE(actual.geology.faultStrength == expected.geology.faultStrength);
    REQUIRE(actual.geology.hotspotInfluence == expected.geology.hotspotInfluence);
    REQUIRE(actual.geology.volcanicActivity == expected.geology.volcanicActivity);

    REQUIRE(actual.hydrology.flowDirection.x == expected.hydrology.flowDirection.x);
    REQUIRE(actual.hydrology.flowDirection.z == expected.hydrology.flowDirection.z);
    REQUIRE(actual.hydrology.surfaceElevation == expected.hydrology.surfaceElevation);
    REQUIRE(actual.hydrology.terrainSlope == expected.hydrology.terrainSlope);
    REQUIRE(actual.hydrology.waterSurface == expected.hydrology.waterSurface);
    REQUIRE(actual.hydrology.discharge == expected.hydrology.discharge);
    REQUIRE(actual.hydrology.baseflow == expected.hydrology.baseflow);
    REQUIRE(actual.hydrology.precipitationSeasonality ==
            expected.hydrology.precipitationSeasonality);
    REQUIRE(actual.hydrology.groundwaterRechargeMm == expected.hydrology.groundwaterRechargeMm);
    REQUIRE(actual.hydrology.groundwaterHead == expected.hydrology.groundwaterHead);
    REQUIRE(actual.hydrology.hydroperiod == expected.hydrology.hydroperiod);
    REQUIRE(actual.hydrology.sediment == expected.hydrology.sediment);
    REQUIRE(actual.hydrology.channelDistance == expected.hydrology.channelDistance);
    REQUIRE(actual.hydrology.channelWidth == expected.hydrology.channelWidth);
    REQUIRE(actual.hydrology.channelDepth == expected.hydrology.channelDepth);
    REQUIRE(actual.hydrology.channelGradient == expected.hydrology.channelGradient);
    REQUIRE(actual.hydrology.erosionDepth == expected.hydrology.erosionDepth);
    REQUIRE(actual.hydrology.lakeDepth == expected.hydrology.lakeDepth);
    REQUIRE(actual.hydrology.lakeShoreDistance == expected.hydrology.lakeShoreDistance);
    REQUIRE(actual.hydrology.shoreWaterSurface == expected.hydrology.shoreWaterSurface);
    REQUIRE(actual.hydrology.lakeBankTarget == expected.hydrology.lakeBankTarget);
    REQUIRE(actual.hydrology.lakeBankInfluence == expected.hydrology.lakeBankInfluence);
    REQUIRE(actual.hydrology.waterfallTop == expected.hydrology.waterfallTop);
    REQUIRE(actual.hydrology.waterfallBottom == expected.hydrology.waterfallBottom);
    REQUIRE(actual.hydrology.waterfallWidth == expected.hydrology.waterfallWidth);
    REQUIRE(actual.hydrology.waterBodyId == expected.hydrology.waterBodyId);
    REQUIRE(actual.hydrology.generatedFluidLevel == expected.hydrology.generatedFluidLevel);
    REQUIRE(actual.hydrology.transitionOwnerKind == expected.hydrology.transitionOwnerKind);
    REQUIRE(actual.hydrology.transitionOwnerId == expected.hydrology.transitionOwnerId);
    REQUIRE(actual.hydrology.streamOrder == expected.hydrology.streamOrder);
    REQUIRE(actual.hydrology.distributaryCount == expected.hydrology.distributaryCount);
    REQUIRE(actual.hydrology.ocean == expected.hydrology.ocean);
    REQUIRE(actual.hydrology.river == expected.hydrology.river);
    REQUIRE(actual.hydrology.lake == expected.hydrology.lake);
    REQUIRE(actual.hydrology.lakeBank == expected.hydrology.lakeBank);
    REQUIRE(actual.hydrology.endorheic == expected.hydrology.endorheic);
    REQUIRE(actual.hydrology.waterfall == expected.hydrology.waterfall);
    REQUIRE(actual.hydrology.waterfallAnchor == expected.hydrology.waterfallAnchor);
    REQUIRE(actual.hydrology.delta == expected.hydrology.delta);
    REQUIRE(actual.hydrology.estuary == expected.hydrology.estuary);
    REQUIRE(actual.hydrology.brackish == expected.hydrology.brackish);
    REQUIRE(actual.hydrology.perennial == expected.hydrology.perennial);
    REQUIRE(actual.hydrology.ephemeral == expected.hydrology.ephemeral);
    REQUIRE(actual.hydrology.wetland == expected.hydrology.wetland);

    REQUIRE(actual.climate.wind.x == expected.climate.wind.x);
    REQUIRE(actual.climate.wind.z == expected.climate.wind.z);
    REQUIRE(actual.climate.temperatureC == expected.climate.temperatureC);
    REQUIRE(actual.climate.annualPrecipitationMm == expected.climate.annualPrecipitationMm);
    REQUIRE(actual.climate.potentialEvapotranspirationMm ==
            expected.climate.potentialEvapotranspirationMm);
    REQUIRE(actual.climate.aridity == expected.climate.aridity);
    REQUIRE(actual.climate.relativeHumidity == expected.climate.relativeHumidity);
    REQUIRE(actual.soil.moisture == expected.soil.moisture);
    REQUIRE(actual.soil.fertility == expected.soil.fertility);
    REQUIRE(actual.soil.drainage == expected.soil.drainage);
    REQUIRE(actual.soil.waterTable == expected.soil.waterTable);
    REQUIRE(actual.suitability.scores == expected.suitability.scores);
    REQUIRE(actual.biome.primary == expected.biome.primary);
    REQUIRE(actual.biome.secondary == expected.biome.secondary);
    REQUIRE(actual.biome.transition == expected.biome.transition);
    REQUIRE(actual.ecotopes == expected.ecotopes);
    REQUIRE(actual.terrainHeight == expected.terrainHeight);
    REQUIRE(actual.waterSurface == expected.waterSurface);
    REQUIRE(actual.slope == expected.slope);
}

class ComplianceTempDir {
public:
    explicit ComplianceTempDir(const char* label) {
        static std::atomic<uint64_t> sequence{0};
        path_ = std::filesystem::temp_directory_path() /
                (std::string("rycraft_") + label + "_" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~ComplianceTempDir() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

worldgen::BiomeBlend selectSyntheticBiome(worldgen::MacroGenerationSampler& sampler, Biome target) {
    worldgen::GeologySample geology;
    worldgen::HydrologySample hydrology;
    worldgen::ClimateFields climate;
    worldgen::SoilSample soil;
    soil.fertility = 0.10;
    soil.moisture = 0.20;
    double terrainHeight = 80.0;
    double slope = 0.10;

    auto setClimate = [&](double temperature, double precipitation) {
        climate.temperatureC = temperature;
        climate.annualPrecipitationMm = precipitation;
        climate.potentialEvapotranspirationMm = 600.0;
        climate.aridity = climate.potentialEvapotranspirationMm / precipitation;
    };

    switch (target) {
        case Biome::DESERT:
            setClimate(30.0, 100.0);
            soil.moisture = 0.05;
            break;
        case Biome::FOREST:
            setClimate(16.0, 1250.0);
            soil.fertility = 0.92;
            soil.moisture = 0.62;
            break;
        case Biome::TAIGA:
            setClimate(1.0, 850.0);
            soil.moisture = 0.52;
            break;
        case Biome::SAVANNA:
            setClimate(27.0, 640.0);
            break;
        case Biome::TROPICAL_RAINFOREST:
            setClimate(28.0, 2850.0);
            break;
        case Biome::TEMPERATE_RAINFOREST:
            setClimate(12.0, 2500.0);
            break;
        case Biome::SHRUBLAND:
            setClimate(14.0, 500.0);
            break;
        case Biome::STEPPE:
            setClimate(7.0, 380.0);
            break;
        case Biome::COLD_DESERT:
            setClimate(0.0, 130.0);
            break;
        case Biome::BADLANDS:
            setClimate(22.0, 260.0);
            geology.rock = worldgen::RockType::SANDSTONE;
            break;
        case Biome::TUNDRA:
            setClimate(-5.0, 350.0);
            break;
        case Biome::ALPINE:
            setClimate(-1.0, 600.0);
            terrainHeight = 210.0;
            slope = 1.10;
            break;
        case Biome::MANGROVE:
            setClimate(27.0, 2300.0);
            hydrology.surfaceElevation = 65.0;
            terrainHeight = 65.0;
            soil.fertility = 0.70;
            soil.moisture = 0.90;
            break;
        case Biome::FROZEN_OCEAN:
            setClimate(-8.0, 500.0);
            hydrology.ocean = true;
            hydrology.surfaceElevation = 50.0;
            terrainHeight = 50.0;
            break;
        case Biome::VOLCANIC_BARREN:
            setClimate(5.0, 800.0);
            geology.volcanicActivity = 1.0;
            terrainHeight = 100.0;
            slope = 1.50;
            break;
        case Biome::GLACIER:
            setClimate(-18.0, 1100.0);
            terrainHeight = 250.0;
            slope = 0.40;
            break;
        case Biome::MONTANE_GRASSLAND:
            setClimate(6.0, 720.0);
            terrainHeight = 145.0;
            slope = 0.25;
            soil.moisture = 0.55;
            soil.fertility = 0.28;
            break;
        case Biome::FLOODED_GRASSLAND:
            setClimate(22.0, 1450.0);
            hydrology.channelWidth = 20.0;
            hydrology.channelDistance = 2.0;
            soil.moisture = 0.95;
            soil.fertility = 0.72;
            slope = 0.05;
            break;
        case Biome::MEDITERRANEAN_WOODLAND:
            setClimate(18.0, 550.0);
            soil.moisture = 0.32;
            soil.fertility = 0.28;
            break;
        case Biome::TEMPERATE_CONIFER_FOREST:
            setClimate(6.0, 1450.0);
            soil.moisture = 0.72;
            soil.fertility = 0.34;
            break;
        case Biome::TROPICAL_CONIFER_FOREST:
            setClimate(22.0, 1450.0);
            terrainHeight = 118.0;
            soil.drainage = 0.90;
            soil.moisture = 0.56;
            soil.fertility = 0.30;
            break;
        case Biome::TROPICAL_DRY_FOREST:
            setClimate(27.0, 850.0);
            soil.moisture = 0.38;
            soil.fertility = 0.42;
            break;
        default:
            FAIL("Synthetic fixture requested for a biome outside the advanced set");
    }

    return worldgen::MacroGenerationSampler::selectBiome(
        sampler.biomeSuitability(geology, hydrology, climate, soil, terrainHeight, slope));
}

double basinFixtureElevation(double x, double z) {
    return 92.0 + x * 0.0007 + z * 0.0004 + std::sin(x / 430.0) * 8.0 + std::cos(z / 370.0) * 6.0;
}

double basinFixtureRainfall(double x, double z, double height) {
    return 780.0 + std::sin((x + z) / 800.0) * 120.0 - (height - 80.0) * 0.3;
}

double basinFixtureResistance(double x, double z) {
    return 0.7 + std::sin(x / 600.0) * std::cos(z / 500.0) * 0.2;
}

void requireBasinSamplesEqual(const worldgen::BasinSample& actual,
                              const worldgen::BasinSample& expected) {
    REQUIRE(actual.flowX == expected.flowX);
    REQUIRE(actual.flowZ == expected.flowZ);
    REQUIRE(actual.surfaceElevation == expected.surfaceElevation);
    REQUIRE(actual.terrainSlope == expected.terrainSlope);
    REQUIRE(actual.waterSurface == expected.waterSurface);
    REQUIRE(actual.discharge == expected.discharge);
    REQUIRE(actual.baseflow == expected.baseflow);
    REQUIRE(actual.precipitationSeasonality == expected.precipitationSeasonality);
    REQUIRE(actual.groundwaterRechargeMm == expected.groundwaterRechargeMm);
    REQUIRE(actual.groundwaterHead == expected.groundwaterHead);
    REQUIRE(actual.hydroperiod == expected.hydroperiod);
    REQUIRE(actual.sediment == expected.sediment);
    REQUIRE(actual.channelDistance == expected.channelDistance);
    REQUIRE(actual.channelWidth == expected.channelWidth);
    REQUIRE(actual.channelDepth == expected.channelDepth);
    REQUIRE(actual.channelGradient == expected.channelGradient);
    REQUIRE(actual.erosionDepth == expected.erosionDepth);
    REQUIRE(actual.lakeDepth == expected.lakeDepth);
    REQUIRE(actual.lakeShoreDistance == expected.lakeShoreDistance);
    REQUIRE(actual.shoreWaterSurface == expected.shoreWaterSurface);
    REQUIRE(actual.lakeBankTarget == expected.lakeBankTarget);
    REQUIRE(actual.lakeBankInfluence == expected.lakeBankInfluence);
    REQUIRE(actual.lakeAreaSquareKilometers == expected.lakeAreaSquareKilometers);
    REQUIRE(actual.lakeVolumeCubicMeters == expected.lakeVolumeCubicMeters);
    REQUIRE(actual.lakeRunoffMmSquareKilometers == expected.lakeRunoffMmSquareKilometers);
    REQUIRE(actual.lakeLossMm == expected.lakeLossMm);
    REQUIRE(actual.lakeOverflowMmSquareKilometers == expected.lakeOverflowMmSquareKilometers);
    REQUIRE(actual.lakeSpillSurface == expected.lakeSpillSurface);
    REQUIRE(actual.waterfallTop == expected.waterfallTop);
    REQUIRE(actual.waterfallBottom == expected.waterfallBottom);
    REQUIRE(actual.waterfallWidth == expected.waterfallWidth);
    REQUIRE(actual.waterBodyId == expected.waterBodyId);
    REQUIRE(actual.generatedFluidLevel == expected.generatedFluidLevel);
    REQUIRE(actual.transitionOwnerKind == expected.transitionOwnerKind);
    REQUIRE(actual.transitionOwnerId == expected.transitionOwnerId);
    REQUIRE(actual.streamOrder == expected.streamOrder);
    REQUIRE(actual.distributaryCount == expected.distributaryCount);
    REQUIRE(actual.outlet == expected.outlet);
    REQUIRE(actual.outletX == expected.outletX);
    REQUIRE(actual.outletZ == expected.outletZ);
    REQUIRE(actual.ocean == expected.ocean);
    REQUIRE(actual.river == expected.river);
    REQUIRE(actual.lake == expected.lake);
    REQUIRE(actual.lakeBank == expected.lakeBank);
    REQUIRE(actual.endorheic == expected.endorheic);
    REQUIRE(actual.waterfall == expected.waterfall);
    REQUIRE(actual.waterfallAnchor == expected.waterfallAnchor);
    REQUIRE(actual.delta == expected.delta);
    REQUIRE(actual.estuary == expected.estuary);
    REQUIRE(actual.brackish == expected.brackish);
    REQUIRE(actual.perennial == expected.perennial);
    REQUIRE(actual.ephemeral == expected.ephemeral);
    REQUIRE(actual.wetland == expected.wetland);
    REQUIRE(actual.valid == expected.valid);
}

bool isWetSurface(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.ocean || sample.hydrology.river || sample.hydrology.lake ||
           sample.hydrology.waterfall;
}

bool isFloodTolerantLog(BlockType block) {
    return block == BlockType::MANGROVE_LOG || block == BlockType::WILLOW_LOG;
}

int maximumRootedWaterDepth(BlockType block) {
    return block == BlockType::MANGROVE_LOG ? 3 : 2;
}

bool basinSamplesExactlyEqual(const worldgen::BasinSample& actual,
                              const worldgen::BasinSample& expected) {
    return actual.flowX == expected.flowX && actual.flowZ == expected.flowZ &&
           actual.surfaceElevation == expected.surfaceElevation &&
           actual.waterSurface == expected.waterSurface && actual.discharge == expected.discharge &&
           actual.baseflow == expected.baseflow &&
           actual.precipitationSeasonality == expected.precipitationSeasonality &&
           actual.groundwaterRechargeMm == expected.groundwaterRechargeMm &&
           actual.groundwaterHead == expected.groundwaterHead &&
           actual.hydroperiod == expected.hydroperiod && actual.sediment == expected.sediment &&
           actual.channelDistance == expected.channelDistance &&
           actual.channelWidth == expected.channelWidth &&
           actual.channelDepth == expected.channelDepth &&
           actual.channelGradient == expected.channelGradient &&
           actual.erosionDepth == expected.erosionDepth && actual.lakeDepth == expected.lakeDepth &&
           actual.lakeShoreDistance == expected.lakeShoreDistance &&
           actual.shoreWaterSurface == expected.shoreWaterSurface &&
           actual.lakeBankTarget == expected.lakeBankTarget &&
           actual.lakeBankInfluence == expected.lakeBankInfluence &&
           actual.lakeAreaSquareKilometers == expected.lakeAreaSquareKilometers &&
           actual.lakeVolumeCubicMeters == expected.lakeVolumeCubicMeters &&
           actual.lakeRunoffMmSquareKilometers == expected.lakeRunoffMmSquareKilometers &&
           actual.lakeLossMm == expected.lakeLossMm &&
           actual.lakeOverflowMmSquareKilometers == expected.lakeOverflowMmSquareKilometers &&
           actual.lakeSpillSurface == expected.lakeSpillSurface &&
           actual.waterfallTop == expected.waterfallTop &&
           actual.waterfallBottom == expected.waterfallBottom &&
           actual.waterfallWidth == expected.waterfallWidth &&
           actual.generatedFluidLevel == expected.generatedFluidLevel &&
           actual.transitionOwnerKind == expected.transitionOwnerKind &&
           actual.transitionOwnerId == expected.transitionOwnerId &&
           actual.waterBodyId == expected.waterBodyId &&
           actual.streamOrder == expected.streamOrder &&
           actual.distributaryCount == expected.distributaryCount &&
           actual.outlet == expected.outlet && actual.outletX == expected.outletX &&
           actual.outletZ == expected.outletZ && actual.ocean == expected.ocean &&
           actual.river == expected.river && actual.lake == expected.lake &&
           actual.lakeBank == expected.lakeBank && actual.endorheic == expected.endorheic &&
           actual.waterfall == expected.waterfall &&
           actual.waterfallAnchor == expected.waterfallAnchor && actual.delta == expected.delta &&
           actual.estuary == expected.estuary && actual.brackish == expected.brackish &&
           actual.perennial == expected.perennial && actual.ephemeral == expected.ephemeral &&
           actual.wetland == expected.wetland && actual.valid == expected.valid;
}

struct WaterContinuityAudit {
    size_t wetSamples = 0;
    size_t adjacentWetSamples = 0;
    size_t unsupportedWetSamples = 0;
    size_t invalidWaterSamples = 0;
    size_t zeroLakeIdentities = 0;
    size_t adjacentDistinctLakeBodies = 0;
    double maximumNonWaterfallStep = 0.0;
    double maximumNonWaterfallVisibleStep = 0.0;
    double maximumSameLakeStep = 0.0;
    double maximumWetFloorStep = 0.0;
    size_t longCardinalFloorRuns = 0;
    int64_t worstX = 0;
    int64_t worstZ = 0;
    int worstDx = 0;
    int worstDz = 0;
    std::map<worldgen::WaterBodyId, ColumnPos> lakeRepresentatives;
};

WaterContinuityAudit auditWaterContinuity(ChunkGenerator& generator, int64_t centerX,
                                          int64_t centerZ, int radius, bool exactGeometry = false) {
    const auto taggedSharpTransition = [](const worldgen::SurfaceSample& sample) {
        return sample.hydrology.waterfall ||
               sample.hydrology.transitionOwnerKind ==
                   worldgen::WaterTransitionKind::EXPLICIT_FALL ||
               (sample.hydrology.transitionOwnerKind != worldgen::WaterTransitionKind::NONE &&
                sample.hydrology.channelGradient >= 0.5);
    };
    const int edge = radius * 2 + 1;
    std::vector<worldgen::SurfaceSample> samples(static_cast<size_t>(edge) * edge);
    if (exactGeometry) {
        for (int sampleZ = 0; sampleZ < edge; ++sampleZ) {
            for (int sampleX = 0; sampleX < edge; ++sampleX) {
                samples[static_cast<size_t>(sampleZ) * edge + sampleX] =
                    generator.sampleExactGeometrySurface(centerX - radius + sampleX,
                                                         centerZ - radius + sampleZ);
            }
        }
    } else {
        generator.sampleFarSurfaceGrid(centerX - radius, centerZ - radius, 1, edge,
                                       worldgen::SurfaceFootprint::BLOCK_1, samples);
    }

    WaterContinuityAudit audit;
    for (int sampleZ = 0; sampleZ < edge; ++sampleZ) {
        for (int sampleX = 0; sampleX < edge; ++sampleX) {
            const worldgen::SurfaceSample& sample =
                samples[static_cast<size_t>(sampleZ) * edge + sampleX];
            const bool wet = isWetSurface(sample);
            if (wet) {
                ++audit.wetSamples;
                if (!std::isfinite(sample.terrainHeight) || !std::isfinite(sample.waterSurface)) {
                    ++audit.invalidWaterSamples;
                }
                if (sample.waterSurface <= sample.terrainHeight)
                    ++audit.unsupportedWetSamples;
                if (sample.hydrology.lake) {
                    if (sample.hydrology.waterBodyId == worldgen::NO_WATER_BODY) {
                        ++audit.zeroLakeIdentities;
                    } else {
                        audit.lakeRepresentatives.try_emplace(
                            sample.hydrology.waterBodyId,
                            ColumnPos{centerX - radius + sampleX, centerZ - radius + sampleZ});
                    }
                }
            }

            for (const auto [dx, dz] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (sampleX + dx >= edge || sampleZ + dz >= edge)
                    continue;
                const worldgen::SurfaceSample& neighbor =
                    samples[static_cast<size_t>(sampleZ + dz) * edge + sampleX + dx];
                if (!wet || !isWetSurface(neighbor))
                    continue;
                ++audit.adjacentWetSamples;
                const double step = std::abs(sample.waterSurface - neighbor.waterSurface);
                if (!taggedSharpTransition(sample) && !taggedSharpTransition(neighbor) &&
                    step > audit.maximumNonWaterfallStep) {
                    audit.maximumNonWaterfallStep = step;
                    audit.worstX = centerX - radius + sampleX;
                    audit.worstZ = centerZ - radius + sampleZ;
                    audit.worstDx = dx;
                    audit.worstDz = dz;
                }
                if (!taggedSharpTransition(sample) && !taggedSharpTransition(neighbor)) {
                    const double visibleStep =
                        std::abs(worldgen::generatedFluidColumn(sample).visibleSurface -
                                 worldgen::generatedFluidColumn(neighbor).visibleSurface);
                    audit.maximumNonWaterfallVisibleStep =
                        std::max(audit.maximumNonWaterfallVisibleStep, visibleStep);
                    audit.maximumWetFloorStep =
                        std::max(audit.maximumWetFloorStep,
                                 std::abs(sample.terrainHeight - neighbor.terrainHeight));
                }
                if (sample.hydrology.lake && neighbor.hydrology.lake) {
                    if (sample.hydrology.waterBodyId != neighbor.hydrology.waterBodyId) {
                        ++audit.adjacentDistinctLakeBodies;
                    } else {
                        audit.maximumSameLakeStep = std::max(audit.maximumSameLakeStep, step);
                    }
                }
            }
        }
    }

    const auto discontinuousFloor = [&](int firstX, int firstZ, int secondX, int secondZ) {
        const worldgen::SurfaceSample& first = samples[static_cast<size_t>(firstZ) * edge + firstX];
        const worldgen::SurfaceSample& second =
            samples[static_cast<size_t>(secondZ) * edge + secondX];
        return isWetSurface(first) && isWetSurface(second) && !taggedSharpTransition(first) &&
               !taggedSharpTransition(second) &&
               std::abs(first.terrainHeight - second.terrainHeight) > 1.001;
    };
    for (int faceX = 1; faceX < edge; ++faceX) {
        int run = 0;
        for (int sampleZ = 0; sampleZ < edge; ++sampleZ) {
            if (discontinuousFloor(faceX - 1, sampleZ, faceX, sampleZ)) {
                ++run;
            } else {
                audit.longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        audit.longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }
    for (int faceZ = 1; faceZ < edge; ++faceZ) {
        int run = 0;
        for (int sampleX = 0; sampleX < edge; ++sampleX) {
            if (discontinuousFloor(sampleX, faceZ - 1, sampleX, faceZ)) {
                ++run;
            } else {
                audit.longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        audit.longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }
    return audit;
}

void requireSameWaterAuthority(const worldgen::SurfaceSample& actual,
                               const worldgen::SurfaceSample& expected) {
    REQUIRE(actual.hydrology.ocean == expected.hydrology.ocean);
    REQUIRE(actual.hydrology.river == expected.hydrology.river);
    REQUIRE(actual.hydrology.lake == expected.hydrology.lake);
    REQUIRE(actual.hydrology.endorheic == expected.hydrology.endorheic);
    REQUIRE(actual.hydrology.waterfall == expected.hydrology.waterfall);
    REQUIRE(actual.hydrology.waterfallAnchor == expected.hydrology.waterfallAnchor);
    REQUIRE(actual.hydrology.delta == expected.hydrology.delta);
    REQUIRE(actual.hydrology.waterBodyId == expected.hydrology.waterBodyId);
    REQUIRE(actual.hydrology.transitionOwnerKind == expected.hydrology.transitionOwnerKind);
    REQUIRE(actual.hydrology.transitionOwnerId == expected.hydrology.transitionOwnerId);
    REQUIRE(actual.waterSurface == Catch::Approx(expected.waterSurface).margin(1.0e-9));
    REQUIRE(actual.hydrology.waterfallTop ==
            Catch::Approx(expected.hydrology.waterfallTop).margin(1.0e-9));
    REQUIRE(actual.hydrology.waterfallBottom ==
            Catch::Approx(expected.hydrology.waterfallBottom).margin(1.0e-9));
}

} // namespace

TEST_CASE("Tectonic fields are bounded and expose every plate boundary class",
          "[worldgen][advanced][geology]") {
    worldgen::MacroGenerationSampler sampler(42);
    std::array<bool, 4> boundaries{};
    std::array<bool, 2> crusts{};
    bool foundHotspot = false;

    for (int64_t z = -32'768; z <= 32'768; z += 1024) {
        for (int64_t x = -32'768; x <= 32'768; x += 1024) {
            const worldgen::GeologySample sample = sampler.sampleGeology(x, z);
            const double speed = std::hypot(sample.plateVelocity.x, sample.plateVelocity.z);
            const bool bounded =
                std::isfinite(speed) && speed >= 0.25 && speed <= 1.25 && sample.crustAge >= 0.0 &&
                sample.crustAge < 1.0 && sample.distanceToBoundary >= 0.0 &&
                sample.continentalFraction >= 0.0 && sample.continentalFraction <= 1.0 &&
                sample.uplift >= 0.0 && sample.uplift <= 1.0 && sample.rift >= 0.0 &&
                sample.rift <= 1.0 && sample.faultStrength >= 0.0 && sample.faultStrength <= 1.0 &&
                sample.hotspotInfluence >= 0.0 && sample.hotspotInfluence <= 1.0 &&
                sample.volcanicActivity >= 0.0 && sample.volcanicActivity <= 1.0;
            if (!bounded) {
                INFO("geology sample at " << x << ", " << z);
                FAIL("tectonic sample left its finite physical bounds");
            }

            boundaries[static_cast<size_t>(sample.boundary)] = true;
            crusts[static_cast<size_t>(sample.crust)] = true;
            foundHotspot = foundHotspot || sample.hotspotInfluence > 0.50;
            if (sample.boundary == worldgen::PlateBoundary::CONVERGENT) {
                if (sample.uplift <= 0.0)
                    FAIL("convergent boundary had no uplift");
            } else if (sample.boundary == worldgen::PlateBoundary::DIVERGENT) {
                if (sample.rift <= 0.0)
                    FAIL("divergent boundary had no rift signal");
            } else if (sample.boundary == worldgen::PlateBoundary::TRANSFORM) {
                if (sample.faultStrength <= 0.0)
                    FAIL("transform boundary had no fault signal");
            }
        }
    }

    REQUIRE(std::ranges::all_of(boundaries, [](bool found) { return found; }));
    REQUIRE(std::ranges::all_of(crusts, [](bool found) { return found; }));
    REQUIRE(foundHotspot);
}

TEST_CASE("Basin solutions stitch river portals and retain flat lake surfaces",
          "[worldgen][advanced][hydrology]") {
    worldgen::MacroGenerationSampler sampler(42);

    const worldgen::HydrologySample lake = sampler.sampleHydrology(-8348.0, 2281.0);
    REQUIRE(lake.lake);
    REQUIRE_FALSE(lake.ocean);
    REQUIRE(lake.lakeDepth > 0.0);
    REQUIRE(lake.waterSurface > lake.surfaceElevation);
    size_t lakeSamples = 0;
    for (int dz = -32; dz <= 32; dz += 8) {
        for (int dx = -32; dx <= 32; dx += 8) {
            const worldgen::HydrologySample nearby =
                sampler.sampleHydrology(-8348.0 + dx, 2281.0 + dz);
            if (!nearby.lake)
                continue;
            ++lakeSamples;
            REQUIRE(nearby.waterSurface == Catch::Approx(lake.waterSurface).margin(1.0e-6));
            REQUIRE(nearby.lakeDepth > 0.0);
        }
    }
    REQUIRE(lakeSamples >= 16);

    constexpr double TERMINAL_X = -21'637.0;
    constexpr double TERMINAL_Z = -62'665.0;
    const worldgen::HydrologySample terminal = sampler.sampleHydrology(TERMINAL_X, TERMINAL_Z);
    REQUIRE(terminal.lake);
    REQUIRE(terminal.endorheic);
    REQUIRE_FALSE(terminal.ocean);
    REQUIRE_FALSE(terminal.river);
    REQUIRE(terminal.waterSurface > SEA_LEVEL + 1.0);
    REQUIRE(terminal.lakeDepth > 3.0);
    REQUIRE(terminal.waterSurface - terminal.surfaceElevation ==
            Catch::Approx(terminal.lakeDepth).margin(1.0e-6));
    for (int dz = -24; dz <= 24; dz += 8) {
        for (int dx = -24; dx <= 24; dx += 8) {
            const worldgen::HydrologySample neighbor =
                sampler.sampleHydrology(TERMINAL_X + dx, TERMINAL_Z + dz);
            REQUIRE(neighbor.lake);
            REQUIRE(neighbor.endorheic);
            REQUIRE(neighbor.lakeDepth > 0.0);
            REQUIRE(neighbor.waterSurface == terminal.waterSurface);
        }
    }

    constexpr double BOUNDARY_X = -16'384.0;
    constexpr double PORTAL_Z = -15'177.0;
    constexpr double EPSILON = 1.0e-4;
    const worldgen::HydrologySample west = sampler.sampleHydrology(BOUNDARY_X - EPSILON, PORTAL_Z);
    const worldgen::HydrologySample east = sampler.sampleHydrology(BOUNDARY_X + EPSILON, PORTAL_Z);
    REQUIRE(west.river);
    REQUIRE(east.river);
    REQUIRE(west.streamOrder == east.streamOrder);
    REQUIRE(west.surfaceElevation == Catch::Approx(east.surfaceElevation).margin(0.001));
    REQUIRE(west.waterSurface == Catch::Approx(east.waterSurface).margin(0.001));
    REQUIRE(west.discharge == Catch::Approx(east.discharge).margin(0.1));
    REQUIRE(west.channelWidth == Catch::Approx(east.channelWidth).margin(0.001));
}

TEST_CASE("Lake bathymetry preserves the solved water balance after shoreline shaping",
          "[worldgen][advanced][hydrology][lake][equilibrium][determinism]") {
    constexpr int64_t CATCHMENT_X = 3;
    constexpr int64_t CATCHMENT_Z = 2;
    constexpr int64_t SAMPLE_X =
        static_cast<int64_t>(worldgen::BASIN_CATCHMENT_EDGE) * CATCHMENT_X + 773;
    constexpr int64_t SAMPLE_Z =
        static_cast<int64_t>(worldgen::BASIN_CATCHMENT_EDGE) * CATCHMENT_Z + 911;
    constexpr int64_t APRON = 32;
    constexpr int GRID_EDGE = 133;
    constexpr double RASTER_CELL_SQUARE_KILOMETERS =
        worldgen::BASIN_RASTER_SPACING * worldgen::BASIN_RASTER_SPACING / 1'000'000.0;
    constexpr double RASTER_CELL_SQUARE_METERS =
        worldgen::BASIN_RASTER_SPACING * worldgen::BASIN_RASTER_SPACING;

    worldgen::BasinSolver solver(42);
    const worldgen::BasinSample center = solver.sample(
        SAMPLE_X, SAMPLE_Z, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(center.outlet == worldgen::BasinOutlet::ENDORHEIC);
    REQUIRE(std::trunc(center.outletX) == center.outletX);
    REQUIRE(std::trunc(center.outletZ) == center.outletZ);

    const worldgen::BasinSample lake =
        solver.sample(center.outletX, center.outletZ, basinFixtureElevation, basinFixtureRainfall,
                      basinFixtureResistance);
    CAPTURE(center.outletX, center.outletZ, lake.waterBodyId, lake.lakeAreaSquareKilometers,
            lake.lakeVolumeCubicMeters, lake.lakeRunoffMmSquareKilometers, lake.lakeLossMm,
            lake.lakeOverflowMmSquareKilometers, lake.lakeSpillSurface);
    REQUIRE(lake.lake);
    REQUIRE(lake.endorheic);
    REQUIRE(lake.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(lake.lakeAreaSquareKilometers > 0.0);
    REQUIRE(lake.lakeVolumeCubicMeters > 0.0);
    REQUIRE(lake.lakeRunoffMmSquareKilometers > 0.0);
    REQUIRE(lake.lakeLossMm > 0.0);
    REQUIRE(lake.lakeOverflowMmSquareKilometers == 0.0);
    REQUIRE(lake.lakeSpillSurface >= lake.waterSurface);

    const int64_t gridOriginX =
        CATCHMENT_X * static_cast<int64_t>(worldgen::BASIN_CATCHMENT_EDGE) - APRON;
    const int64_t gridOriginZ =
        CATCHMENT_Z * static_cast<int64_t>(worldgen::BASIN_CATCHMENT_EDGE) - APRON;
    std::vector<worldgen::BasinSample> grid(GRID_EDGE * GRID_EDGE);
    solver.sampleGrid(gridOriginX, gridOriginZ, static_cast<int>(worldgen::BASIN_RASTER_SPACING),
                      static_cast<int>(worldgen::BASIN_RASTER_SPACING), GRID_EDGE, GRID_EDGE,
                      basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance, grid);

    size_t wetCells = 0;
    double shapedVolumeCubicMeters = 0.0;
    for (const worldgen::BasinSample& sample : grid) {
        if (!sample.lake || sample.waterBodyId != lake.waterBodyId)
            continue;
        ++wetCells;
        shapedVolumeCubicMeters += sample.lakeDepth * RASTER_CELL_SQUARE_METERS;
        REQUIRE(sample.waterSurface == lake.waterSurface);
        REQUIRE(sample.lakeAreaSquareKilometers == lake.lakeAreaSquareKilometers);
        REQUIRE(sample.lakeVolumeCubicMeters == lake.lakeVolumeCubicMeters);
        REQUIRE(sample.lakeRunoffMmSquareKilometers == lake.lakeRunoffMmSquareKilometers);
        REQUIRE(sample.lakeLossMm == lake.lakeLossMm);
        REQUIRE(sample.lakeOverflowMmSquareKilometers == lake.lakeOverflowMmSquareKilometers);
        REQUIRE(sample.lakeSpillSurface == lake.lakeSpillSurface);
    }
    REQUIRE(wetCells > 0);
    const double shapedAreaSquareKilometers =
        static_cast<double>(wetCells) * RASTER_CELL_SQUARE_KILOMETERS;
    const double areaTolerance =
        std::max(RASTER_CELL_SQUARE_KILOMETERS, lake.lakeAreaSquareKilometers * 0.05);
    CAPTURE(wetCells, shapedAreaSquareKilometers, shapedVolumeCubicMeters, areaTolerance);
    REQUIRE(std::abs(shapedAreaSquareKilometers - lake.lakeAreaSquareKilometers) <= areaTolerance);
    REQUIRE(shapedVolumeCubicMeters == Catch::Approx(lake.lakeVolumeCubicMeters).epsilon(0.05));

    solver.clear();
    const worldgen::BasinSample unrelated = solver.sample(
        -7'419.0, -7'281.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(unrelated.valid);
    const std::array<worldgen::BasinSamplePosition, 2> reversedPositions = {
        worldgen::BasinSamplePosition{static_cast<double>(static_cast<int64_t>(center.outletX)),
                                      static_cast<double>(static_cast<int64_t>(center.outletZ))},
        worldgen::BasinSamplePosition{SAMPLE_X, SAMPLE_Z},
    };
    std::array<worldgen::BasinSample, 2> rebuilt{};
    solver.samplePoints(reversedPositions, basinFixtureElevation, basinFixtureRainfall,
                        basinFixtureResistance, rebuilt);
    requireBasinSamplesEqual(rebuilt[0], lake);
    requireBasinSamplesEqual(rebuilt[1], center);
}

TEST_CASE("Learned hydroclimate controls runoff baseflow and groundwater authority",
          "[worldgen][advanced][hydrology][climate][groundwater]") {
    constexpr double SAMPLE_X = worldgen::BASIN_CATCHMENT_EDGE * 3.0 + 773.0;
    constexpr double SAMPLE_Z = worldgen::BASIN_CATCHMENT_EDGE * 2.0 + 911.0;
    const auto humidClimate = [](double, double, double) {
        return worldgen::BasinHydroclimateSample{
            .meanTemperatureC = 16.0,
            .temperatureVariabilityC = 4.0,
            .annualPrecipitationMm = 1'600.0,
            .precipitationCoefficientOfVariation = 0.12,
            .lapseRateCPerMeter = -0.0065,
            .potentialEvapotranspirationMm = 550.0,
        };
    };
    const auto aridClimate = [](double, double, double) {
        return worldgen::BasinHydroclimateSample{
            .meanTemperatureC = 28.0,
            .temperatureVariabilityC = 18.0,
            .annualPrecipitationMm = 220.0,
            .precipitationCoefficientOfVariation = 1.10,
            .lapseRateCPerMeter = -0.0045,
            .potentialEvapotranspirationMm = 1'900.0,
        };
    };

    worldgen::BasinSolver humidSolver(42);
    worldgen::BasinSolver aridSolver(42);
    const worldgen::BasinSample humid =
        humidSolver.sample(SAMPLE_X, SAMPLE_Z, basinFixtureElevation, basinFixtureRainfall,
                           basinFixtureResistance, {}, humidClimate);
    const worldgen::BasinSample arid =
        aridSolver.sample(SAMPLE_X, SAMPLE_Z, basinFixtureElevation, basinFixtureRainfall,
                          basinFixtureResistance, {}, aridClimate);

    REQUIRE(humid.valid);
    REQUIRE(arid.valid);
    REQUIRE(humid.discharge > arid.discharge);
    REQUIRE(humid.baseflow > arid.baseflow);
    REQUIRE(humid.groundwaterRechargeMm > arid.groundwaterRechargeMm);
    REQUIRE(humid.precipitationSeasonality < arid.precipitationSeasonality);
    for (const worldgen::BasinSample* sample : {&humid, &arid}) {
        REQUIRE(sample->baseflow >= 0.0);
        REQUIRE(sample->baseflow <= sample->discharge);
        REQUIRE(sample->precipitationSeasonality >= 0.0);
        REQUIRE(sample->precipitationSeasonality <= 1.0);
        REQUIRE(sample->hydroperiod >= 0.0);
        REQUIRE(sample->hydroperiod <= 1.0);
        REQUIRE(std::isfinite(sample->groundwaterHead));
    }
}

TEST_CASE("Pinned open water windows retain continuous supported levels",
          "[worldgen][advanced][hydrology][water][continuity][regression]") {
    struct Fixture {
        uint64_t seed;
        int64_t centerX;
        int64_t centerZ;
        int radius;
        double maximumStep;
        size_t minimumWetSamples;
        size_t minimumLakeBodies;
    };
    constexpr std::array<Fixture, 4> fixtures = {
        Fixture{42, -557, 379, 256, 0.20, 40'000, 0},
        Fixture{764891, 23'870, -110'590, 192, 0.125001, 70'000, 2},
        Fixture{764891, 22'010, -112'014, 64, 0.85, 2'500, 0},
        Fixture{764891, 22'017, -111'442, 96, 0.35, 10'000, 0},
    };

    for (const Fixture& fixture : fixtures) {
        ChunkGenerator generator(fixture.seed);
        const WaterContinuityAudit audit =
            auditWaterContinuity(generator, fixture.centerX, fixture.centerZ, fixture.radius);
        const worldgen::BasinCacheMetrics basinMetrics = generator.basinCacheMetrics();
        const worldgen::SurfaceSample worst = generator.sampleFarSurface(
            audit.worstX, audit.worstZ, worldgen::SurfaceFootprint::BLOCK_1);
        const worldgen::SurfaceSample worstNeighbor =
            generator.sampleFarSurface(audit.worstX + audit.worstDx, audit.worstZ + audit.worstDz,
                                       worldgen::SurfaceFootprint::BLOCK_1);
        CAPTURE(fixture.seed, fixture.centerX, fixture.centerZ, fixture.radius, audit.wetSamples,
                audit.adjacentWetSamples, audit.unsupportedWetSamples, audit.invalidWaterSamples,
                audit.zeroLakeIdentities, audit.adjacentDistinctLakeBodies,
                audit.maximumNonWaterfallStep, audit.maximumNonWaterfallVisibleStep,
                audit.maximumSameLakeStep, audit.worstX, audit.worstZ, audit.worstDx, audit.worstDz,
                audit.lakeRepresentatives.size(), basinMetrics.builds, basinMetrics.failures,
                basinMetrics.fallbackBuilds, worst.terrainHeight, worst.waterSurface,
                worst.hydrology.river, worst.hydrology.ocean, worst.hydrology.lake,
                worst.hydrology.waterfall, worst.hydrology.transitionOwnerKind,
                worst.hydrology.transitionOwnerId, worst.hydrology.generatedFluidLevel,
                worst.hydrology.channelDistance, worst.hydrology.channelWidth,
                worst.hydrology.channelGradient, worst.hydrology.lakeShoreDistance,
                worst.hydrology.shoreWaterSurface, worst.hydrology.waterfallTop,
                worst.hydrology.waterfallBottom, worstNeighbor.terrainHeight,
                worstNeighbor.waterSurface, worstNeighbor.hydrology.river,
                worstNeighbor.hydrology.ocean, worstNeighbor.hydrology.lake,
                worstNeighbor.hydrology.waterfall, worstNeighbor.hydrology.transitionOwnerKind,
                worstNeighbor.hydrology.transitionOwnerId,
                worstNeighbor.hydrology.generatedFluidLevel,
                worstNeighbor.hydrology.channelDistance, worstNeighbor.hydrology.channelWidth,
                worstNeighbor.hydrology.channelGradient, worstNeighbor.hydrology.lakeShoreDistance,
                worstNeighbor.hydrology.shoreWaterSurface, worstNeighbor.hydrology.waterfallTop,
                worstNeighbor.hydrology.waterfallBottom);
        REQUIRE(audit.wetSamples >= fixture.minimumWetSamples);
        REQUIRE(audit.adjacentWetSamples > audit.wetSamples);
        REQUIRE(audit.invalidWaterSamples == 0);
        REQUIRE(audit.unsupportedWetSamples == 0);
        REQUIRE(audit.zeroLakeIdentities == 0);
        REQUIRE(audit.adjacentDistinctLakeBodies == 0);
        REQUIRE(audit.maximumNonWaterfallVisibleStep <= fixture.maximumStep);
        REQUIRE(audit.maximumSameLakeStep <= 0.001);
        REQUIRE(audit.lakeRepresentatives.size() >= fixture.minimumLakeBodies);
        REQUIRE(basinMetrics.failures == 0);
        REQUIRE(basinMetrics.fallbackBuilds == 0);

        if (fixture.minimumLakeBodies >= 2) {
            for (const auto& [body, position] : audit.lakeRepresentatives) {
                const worldgen::SurfaceSample sample =
                    generator.sampleExactGeometrySurface(position.x, position.z);
                CAPTURE(body, position.x, position.z, sample.terrainHeight, sample.waterSurface);
                REQUIRE(sample.hydrology.lake);
                REQUIRE(sample.hydrology.waterBodyId == body);
                REQUIRE(sample.waterSurface > sample.terrainHeight);
            }
        }
    }
}

TEST_CASE("Generated channel beds and visible water remain continuous in the second scene",
          "[worldgen][advanced][hydrology][water][continuity][bed][regression]") {
    constexpr uint64_t SEED = 42;
    constexpr int64_t CENTER_X = 576;
    constexpr int64_t CENTER_Z = -1'528;
    constexpr int RADIUS = 160;

    ChunkGenerator generator(SEED);
    const WaterContinuityAudit audit =
        auditWaterContinuity(generator, CENTER_X, CENTER_Z, RADIUS, true);
    const worldgen::SurfaceSample worst =
        generator.sampleExactGeometrySurface(audit.worstX, audit.worstZ);
    const worldgen::SurfaceSample worstNeighbor = generator.sampleExactGeometrySurface(
        audit.worstX + audit.worstDx, audit.worstZ + audit.worstDz);
    const worldgen::HydrologySample directWorst =
        generator.sampleGeneratedWaterAuthority(audit.worstX, audit.worstZ);
    const worldgen::HydrologySample directWorstNeighbor = generator.sampleGeneratedWaterAuthority(
        audit.worstX + audit.worstDx, audit.worstZ + audit.worstDz);
    CAPTURE(audit.wetSamples, audit.adjacentWetSamples, audit.unsupportedWetSamples,
            audit.invalidWaterSamples, audit.maximumNonWaterfallStep,
            audit.maximumNonWaterfallVisibleStep, audit.maximumWetFloorStep,
            audit.longCardinalFloorRuns, audit.worstX, audit.worstZ, audit.worstDx, audit.worstDz,
            worst.terrainHeight, worst.waterSurface, worst.hydrology.river, worst.hydrology.ocean,
            worst.hydrology.lake, worst.hydrology.waterfall, worst.hydrology.transitionOwnerKind,
            worst.hydrology.transitionOwnerId, worst.hydrology.generatedFluidLevel,
            worst.hydrology.channelBank, worstNeighbor.terrainHeight, worstNeighbor.waterSurface,
            worstNeighbor.hydrology.river, worstNeighbor.hydrology.ocean,
            worstNeighbor.hydrology.lake, worstNeighbor.hydrology.waterfall,
            worstNeighbor.hydrology.transitionOwnerKind, worstNeighbor.hydrology.transitionOwnerId,
            worstNeighbor.hydrology.generatedFluidLevel, worstNeighbor.hydrology.channelBank,
            directWorst.surfaceElevation, directWorst.waterSurface, directWorst.river,
            directWorst.ocean, directWorst.waterfall, directWorst.transitionOwnerKind,
            directWorst.transitionOwnerId, directWorstNeighbor.surfaceElevation,
            directWorstNeighbor.waterSurface, directWorstNeighbor.river, directWorstNeighbor.ocean,
            directWorstNeighbor.waterfall, directWorstNeighbor.transitionOwnerKind,
            directWorstNeighbor.transitionOwnerId);
    REQUIRE(audit.wetSamples >= 23'500);
    REQUIRE(audit.adjacentWetSamples > audit.wetSamples);
    REQUIRE(audit.unsupportedWetSamples == 0);
    REQUIRE(audit.invalidWaterSamples == 0);
    REQUIRE(audit.maximumNonWaterfallStep <= 0.20);
    REQUIRE(audit.maximumNonWaterfallVisibleStep <= 0.125001);
    REQUIRE(audit.longCardinalFloorRuns == 0);
}

TEST_CASE("Coastal channels cross catchment faces without wet banks or raster steps",
          "[worldgen][advanced][hydrology][water][coast][seam][regression]") {
    constexpr uint64_t SEED = 42;
    constexpr int64_t CENTER_X = 13;
    constexpr int64_t CENTER_Z = -1'419;
    constexpr int RADIUS = 96;
    constexpr int EDGE = RADIUS * 2 + 1;

    ChunkGenerator generator(SEED);
    std::vector<worldgen::SurfaceSample> samples(static_cast<size_t>(EDGE * EDGE));
    generator.sampleFarSurfaceGrid(CENTER_X - RADIUS, CENTER_Z - RADIUS, 1, EDGE,
                                   worldgen::SurfaceFootprint::BLOCK_1, samples);
    const auto index = [](int sampleX, int sampleZ) {
        return static_cast<size_t>(sampleZ * EDGE + sampleX);
    };
    const auto wet = [](const worldgen::SurfaceSample& sample) {
        return isWetSurface(sample) && sample.waterSurface > sample.terrainHeight + 0.05;
    };
    const auto discontinuousFloor = [&](int firstX, int firstZ, int secondX, int secondZ) {
        const worldgen::SurfaceSample& first = samples[index(firstX, firstZ)];
        const worldgen::SurfaceSample& second = samples[index(secondX, secondZ)];
        return wet(first) && wet(second) && !first.hydrology.waterfall &&
               !second.hydrology.waterfall &&
               std::abs(first.terrainHeight - second.terrainHeight) > 1.001;
    };

    size_t wetSamples = 0;
    size_t wetBanks = 0;
    size_t stageDiscontinuities = 0;
    size_t longCardinalFloorRuns = 0;
    size_t catchmentFaceComparisons = 0;
    double maximumStageStep = 0.0;
    double maximumFloorStep = 0.0;
    int maximumFloorFirstX = 0;
    int maximumFloorFirstZ = 0;
    double maximumCatchmentFaceStep = 0.0;
    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            const worldgen::SurfaceSample& sample = samples[index(sampleX, sampleZ)];
            if (!wet(sample))
                continue;
            ++wetSamples;
            wetBanks += sample.hydrology.lakeBank ? 1U : 0U;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (sampleX + offsetX >= EDGE || sampleZ + offsetZ >= EDGE)
                    continue;
                const worldgen::SurfaceSample& neighbor =
                    samples[index(sampleX + offsetX, sampleZ + offsetZ)];
                if (!wet(neighbor) || sample.hydrology.waterfall || neighbor.hydrology.waterfall) {
                    continue;
                }
                const double step = std::abs(sample.waterSurface - neighbor.waterSurface);
                maximumStageStep = std::max(maximumStageStep, step);
                stageDiscontinuities += step > 0.125001 ? 1U : 0U;
                const double floorStep = std::abs(sample.terrainHeight - neighbor.terrainHeight);
                if (floorStep > maximumFloorStep) {
                    maximumFloorStep = floorStep;
                    maximumFloorFirstX = sampleX;
                    maximumFloorFirstZ = sampleZ;
                }
            }
        }
    }

    for (int faceX = 1; faceX < EDGE; ++faceX) {
        int run = 0;
        for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
            if (discontinuousFloor(faceX - 1, sampleZ, faceX, sampleZ)) {
                ++run;
            } else {
                longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }
    for (int faceZ = 1; faceZ < EDGE; ++faceZ) {
        int run = 0;
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            if (discontinuousFloor(sampleX, faceZ - 1, sampleX, faceZ)) {
                ++run;
            } else {
                longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }

    const int seamX = static_cast<int>(-CENTER_X + RADIUS);
    REQUIRE(seamX > 0);
    REQUIRE(seamX + 1 < EDGE);
    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int firstX : {seamX - 1, seamX}) {
            const worldgen::SurfaceSample& first = samples[index(firstX, sampleZ)];
            const worldgen::SurfaceSample& second = samples[index(firstX + 1, sampleZ)];
            if (!wet(first) || !wet(second) || first.hydrology.waterfall ||
                second.hydrology.waterfall) {
                continue;
            }
            ++catchmentFaceComparisons;
            maximumCatchmentFaceStep = std::max(maximumCatchmentFaceStep,
                                                std::abs(first.waterSurface - second.waterSurface));
        }
    }

    const worldgen::SurfaceSample& maximumFloorFirst =
        samples[index(maximumFloorFirstX, maximumFloorFirstZ)];
    CAPTURE(wetSamples, wetBanks, stageDiscontinuities, maximumStageStep, maximumFloorStep,
            maximumFloorFirstX, maximumFloorFirstZ, maximumFloorFirst.terrainHeight,
            maximumFloorFirst.waterSurface, maximumFloorFirst.hydrology.river,
            maximumFloorFirst.hydrology.ocean, maximumFloorFirst.hydrology.transitionOwnerKind,
            maximumFloorFirst.hydrology.transitionOwnerId, longCardinalFloorRuns,
            catchmentFaceComparisons, maximumCatchmentFaceStep);
    REQUIRE(wetSamples >= 17'000);
    REQUIRE(wetBanks == 0);
    REQUIRE(stageDiscontinuities == 0);
    REQUIRE(maximumStageStep <= 0.125001);
    REQUIRE(longCardinalFloorRuns == 0);
    REQUIRE(catchmentFaceComparisons >= 128);
    REQUIRE(maximumCatchmentFaceStep <= 0.125001);

    constexpr std::array<ColumnPos, 8> REBUILD_PROBES = {
        ColumnPos{-1, -1'432}, ColumnPos{0, -1'432},  ColumnPos{1, -1'432}, ColumnPos{14, -1'443},
        ColumnPos{15, -1'442}, ColumnPos{16, -1'444}, ColumnPos{0, -1'330}, ColumnPos{1, -1'330},
    };
    std::array<worldgen::SurfaceSample, REBUILD_PROBES.size()> expected{};
    for (size_t probe = 0; probe < REBUILD_PROBES.size(); ++probe) {
        expected[probe] = generator.sampleFarSurface(
            REBUILD_PROBES[probe].x, REBUILD_PROBES[probe].z, worldgen::SurfaceFootprint::BLOCK_1);
    }
    generator.clearMacroCaches();
    for (size_t probe = REBUILD_PROBES.size(); probe-- > 0;) {
        const worldgen::SurfaceSample rebuilt = generator.sampleFarSurface(
            REBUILD_PROBES[probe].x, REBUILD_PROBES[probe].z, worldgen::SurfaceFootprint::BLOCK_1);
        requireExactSurface(rebuilt, expected[probe]);
    }
}

TEST_CASE("Pinned steep channels emit stable partial-height rapid water",
          "[worldgen][advanced][hydrology][river][rapid][fluid][runtime][regression]") {
    constexpr uint64_t SEED = 764891;
    constexpr int64_t MINIMUM_X = 24'881;
    constexpr int64_t MAXIMUM_X = 24'890;
    constexpr int64_t MINIMUM_Z = -109'733;
    constexpr int64_t MAXIMUM_Z = -109'723;
    constexpr int WIDTH = static_cast<int>(MAXIMUM_X - MINIMUM_X + 1);
    constexpr int DEPTH = static_cast<int>(MAXIMUM_Z - MINIMUM_Z + 1);
    constexpr int MAXIMUM_SETTLE_TICKS = 400;

    struct RapidColumn {
        double analyticalSurface = 0.0;
        double visibleSurface = 0.0;
        worldgen::Vector2d flowDirection{};
        int topY = WORLD_MIN_Y;
        uint8_t level = 0;
        bool rapid = false;
    };

    ChunkGenerator generator(SEED);
    std::array<RapidColumn, WIDTH * DEPTH> columns{};
    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    const auto columnIndex = [](int64_t x, int64_t z) {
        return static_cast<size_t>(z - MINIMUM_Z) * WIDTH + static_cast<size_t>(x - MINIMUM_X);
    };
    const auto generatedCell = [&](int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return std::pair{
            found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                    Chunk::worldToLocal(z)),
            found->second->getFluidState(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                         Chunk::worldToLocal(z)),
        };
    };

    size_t rapidColumns = 0;
    size_t sourceAnchors = 0;
    size_t validPredecessors = 0;
    std::array<bool, 8> flowingLevels{};
    int minimumTopY = WORLD_MAX_Y;
    int maximumTopY = WORLD_MIN_Y;
    for (int64_t z = MINIMUM_Z; z <= MAXIMUM_Z; ++z) {
        for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
            const worldgen::SurfaceSample far =
                generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
            if (!far.hydrology.river || far.hydrology.waterfall ||
                far.hydrology.channelGradient < 0.125)
                continue;

            const ColumnPos chunkColumn{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
            const std::shared_ptr<const ColumnPlan> plan = generator.getColumnPlan(chunkColumn);
            const worldgen::SurfaceSample planned =
                plan->sample(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
            const worldgen::SurfaceSample exact = generator.sampleExactGeometrySurface(x, z);
            const worldgen::GeneratedFluidColumn farFluid = worldgen::generatedFluidColumn(far);
            const worldgen::GeneratedFluidColumn plannedFluid =
                worldgen::generatedFluidColumn(planned);
            const worldgen::GeneratedFluidColumn exactFluid = worldgen::generatedFluidColumn(exact);

            CAPTURE(x, z, far.waterSurface, far.terrainHeight, far.hydrology.channelGradient,
                    far.hydrology.generatedFluidLevel, planned.hydrology.generatedFluidLevel,
                    exact.hydrology.generatedFluidLevel, farFluid.topY, farFluid.visibleSurface);
            REQUIRE(farFluid.wet);
            REQUIRE(plannedFluid.wet);
            REQUIRE(exactFluid.wet);
            REQUIRE(far.hydrology.generatedFluidLevel <= 7);
            REQUIRE(planned.hydrology.generatedFluidLevel == far.hydrology.generatedFluidLevel);
            REQUIRE(exact.hydrology.generatedFluidLevel == far.hydrology.generatedFluidLevel);
            REQUIRE(planned.waterSurface == Catch::Approx(far.waterSurface).margin(1.0e-9));
            REQUIRE(exact.waterSurface == Catch::Approx(far.waterSurface).margin(1.0e-9));
            REQUIRE(plannedFluid.topY == farFluid.topY);
            REQUIRE(exactFluid.topY == farFluid.topY);
            REQUIRE(plannedFluid.topState == farFluid.topState);
            REQUIRE(exactFluid.topState == farFluid.topState);
            REQUIRE(plannedFluid.visibleSurface == farFluid.visibleSurface);
            REQUIRE(exactFluid.visibleSurface == farFluid.visibleSurface);
            REQUIRE_FALSE(farFluid.topState.isFalling());
            if (far.hydrology.generatedFluidLevel == 0) {
                REQUIRE(farFluid.topState.isSource());
                ++sourceAnchors;
            } else {
                REQUIRE_FALSE(farFluid.topState.isSource());
                REQUIRE(farFluid.topState.level() == far.hydrology.generatedFluidLevel);
                flowingLevels[far.hydrology.generatedFluidLevel] = true;
            }
            REQUIRE(std::abs(farFluid.visibleSurface - far.waterSurface) <= 0.125001);

            const auto [topBlock, topState] = generatedCell(x, farFluid.topY, z);
            REQUIRE(topBlock == BlockType::WATER);
            REQUIRE(topState == farFluid.topState);
            const auto [belowBlock, belowState] = generatedCell(x, farFluid.topY - 1, z);
            REQUIRE(
                (isSolid(belowBlock) || (belowBlock == BlockType::WATER && belowState.isSource())));

            if (!farFluid.topState.isSource()) {
                bool foundPredecessor = false;
                std::array<double, 4> predecessorSurfaces{};
                std::array<uint8_t, 4> predecessorLevels{};
                std::array<worldgen::WaterTransitionKind, 4> predecessorOwnerKinds{};
                std::array<uint64_t, 4> predecessorOwnerIds{};
                size_t predecessorIndex = 0;
                for (const auto [offsetX, offsetZ] :
                     {std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}}) {
                    const worldgen::SurfaceSample neighbor = generator.sampleFarSurface(
                        x + offsetX, z + offsetZ, worldgen::SurfaceFootprint::BLOCK_1);
                    const worldgen::GeneratedFluidColumn neighborFluid =
                        worldgen::generatedFluidColumn(neighbor);
                    predecessorSurfaces[predecessorIndex] = neighborFluid.visibleSurface;
                    predecessorLevels[predecessorIndex] = neighbor.hydrology.generatedFluidLevel;
                    predecessorOwnerKinds[predecessorIndex] =
                        neighbor.hydrology.transitionOwnerKind;
                    predecessorOwnerIds[predecessorIndex] = neighbor.hydrology.transitionOwnerId;
                    ++predecessorIndex;
                    if (!neighborFluid.wet || neighborFluid.topY != farFluid.topY ||
                        neighborFluid.topState.isFalling()) {
                        continue;
                    }
                    const bool sourcePredecessor =
                        farFluid.topState.level() == 1 && neighborFluid.topState.isSource();
                    const bool flowingPredecessor =
                        farFluid.topState.level() > 1 && !neighborFluid.topState.isSource() &&
                        neighborFluid.topState.level() + 1 == farFluid.topState.level();
                    if (!sourcePredecessor && !flowingPredecessor)
                        continue;
                    const auto [predecessorBlock, predecessorState] =
                        generatedCell(x + offsetX, farFluid.topY, z + offsetZ);
                    REQUIRE(predecessorBlock == BlockType::WATER);
                    REQUIRE(predecessorState == neighborFluid.topState);
                    foundPredecessor = true;
                    break;
                }
                CAPTURE(farFluid.topY, farFluid.topState.level(), far.hydrology.transitionOwnerKind,
                        far.hydrology.transitionOwnerId, predecessorSurfaces, predecessorLevels,
                        predecessorOwnerKinds, predecessorOwnerIds);
                REQUIRE(foundPredecessor);
                ++validPredecessors;
            }

            RapidColumn& column = columns[columnIndex(x, z)];
            column.analyticalSurface = far.waterSurface;
            column.visibleSurface = farFluid.visibleSurface;
            column.flowDirection = far.hydrology.flowDirection;
            column.topY = farFluid.topY;
            column.level = far.hydrology.generatedFluidLevel;
            column.rapid = true;
            minimumTopY = std::min(minimumTopY, column.topY);
            maximumTopY = std::max(maximumTopY, column.topY);
            ++rapidColumns;
        }
    }
    // Invalid partial states now become source anchors before publication, so
    // this fixture retains only the subset with a complete predecessor chain.
    REQUIRE(rapidColumns >= 32);
    REQUIRE(sourceAnchors > 0);
    REQUIRE(validPredecessors == rapidColumns - sourceAnchors);
    for (uint8_t level = 1; level <= 7; ++level) {
        CAPTURE(level);
        REQUIRE(flowingLevels[level]);
    }

    constexpr int64_t PINNED_OUTLET_X = 23'725;
    constexpr int64_t PINNED_OUTLET_Z = -110'560;
    const worldgen::SurfaceSample pinnedOutlet = generator.sampleFarSurface(
        PINNED_OUTLET_X, PINNED_OUTLET_Z, worldgen::SurfaceFootprint::BLOCK_1);
    REQUIRE(pinnedOutlet.hydrology.river);
    REQUIRE(pinnedOutlet.hydrology.flowDirection.x > 0.25);
    REQUIRE(pinnedOutlet.hydrology.flowDirection.z < -0.25);
    const worldgen::SurfaceSample routedReceiver = generator.sampleFarSurface(
        PINNED_OUTLET_X + 1, PINNED_OUTLET_Z - 1, worldgen::SurfaceFootprint::BLOCK_1);
    CAPTURE(pinnedOutlet.waterSurface, routedReceiver.waterSurface, routedReceiver.hydrology.ocean,
            routedReceiver.hydrology.river, routedReceiver.hydrology.lake,
            routedReceiver.hydrology.waterfall, routedReceiver.hydrology.channelDistance,
            routedReceiver.hydrology.channelWidth);
    REQUIRE(isWetSurface(routedReceiver));
    REQUIRE(routedReceiver.waterSurface <= pinnedOutlet.waterSurface + 1.0e-6);

    double maximumCardinalVisibleStep = 0.0;
    size_t downstreamComparisons = 0;
    for (int64_t z = MINIMUM_Z; z <= MAXIMUM_Z; ++z) {
        for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
            const RapidColumn& column = columns[columnIndex(x, z)];
            if (!column.rapid)
                continue;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                const int64_t neighborX = x + offsetX;
                const int64_t neighborZ = z + offsetZ;
                if (neighborX > MAXIMUM_X || neighborZ > MAXIMUM_Z)
                    continue;
                const RapidColumn& neighbor = columns[columnIndex(neighborX, neighborZ)];
                if (!neighbor.rapid)
                    continue;
                maximumCardinalVisibleStep =
                    std::max(maximumCardinalVisibleStep,
                             std::abs(column.visibleSurface - neighbor.visibleSurface));
            }

            const int downstreamX =
                column.flowDirection.x > 0.25 ? 1 : (column.flowDirection.x < -0.25 ? -1 : 0);
            const int downstreamZ =
                column.flowDirection.z > 0.25 ? 1 : (column.flowDirection.z < -0.25 ? -1 : 0);
            const int64_t neighborX = x + downstreamX;
            const int64_t neighborZ = z + downstreamZ;
            if ((downstreamX == 0 && downstreamZ == 0) || neighborX < MINIMUM_X ||
                neighborX > MAXIMUM_X || neighborZ < MINIMUM_Z || neighborZ > MAXIMUM_Z) {
                continue;
            }
            const RapidColumn& downstream = columns[columnIndex(neighborX, neighborZ)];
            if (!downstream.rapid)
                continue;
            CAPTURE(x, z, neighborX, neighborZ, column.analyticalSurface,
                    downstream.analyticalSurface, column.visibleSurface, downstream.visibleSurface,
                    column.level, downstream.level, column.flowDirection.x, column.flowDirection.z);
            REQUIRE(downstream.analyticalSurface <= column.analyticalSurface + 1.0e-6);
            REQUIRE(downstream.visibleSurface <= column.visibleSurface + 1.0e-6);
            ++downstreamComparisons;
        }
    }
    CAPTURE(maximumCardinalVisibleStep, downstreamComparisons);
    REQUIRE(maximumCardinalVisibleStep <= 0.250001);
    REQUIRE(downstreamComparisons >= 24);

    World world(SEED, 4);
    const int64_t minimumChunkX = Chunk::worldToChunk(MINIMUM_X) - 1;
    const int64_t maximumChunkX = Chunk::worldToChunk(MAXIMUM_X) + 1;
    const int64_t minimumChunkZ = Chunk::worldToChunk(MINIMUM_Z) - 1;
    const int64_t maximumChunkZ = Chunk::worldToChunk(MAXIMUM_Z) + 1;
    const int32_t minimumChunkY = Chunk::worldToChunkY(minimumTopY) - 1;
    const int32_t maximumChunkY = Chunk::worldToChunkY(maximumTopY) + 1;
    for (int32_t chunkY = minimumChunkY; chunkY <= maximumChunkY; ++chunkY) {
        for (int64_t chunkZ = minimumChunkZ; chunkZ <= maximumChunkZ; ++chunkZ) {
            for (int64_t chunkX = minimumChunkX; chunkX <= maximumChunkX; ++chunkX)
                world.getChunk({chunkX, chunkY, chunkZ});
        }
    }
    REQUIRE(world.getPendingFluidCount() == 0);

    constexpr int SNAPSHOT_MARGIN = 2;
    const int snapshotMinimumY = minimumTopY - SNAPSHOT_MARGIN;
    const int snapshotMaximumY = maximumTopY + SNAPSHOT_MARGIN;
    const auto snapshot = [&] {
        std::vector<uint16_t> result;
        result.reserve(static_cast<size_t>(WIDTH + SNAPSHOT_MARGIN * 2) *
                       static_cast<size_t>(DEPTH + SNAPSHOT_MARGIN * 2) *
                       static_cast<size_t>(snapshotMaximumY - snapshotMinimumY + 1));
        for (int y = snapshotMinimumY; y <= snapshotMaximumY; ++y) {
            for (int64_t z = MINIMUM_Z - SNAPSHOT_MARGIN; z <= MAXIMUM_Z + SNAPSHOT_MARGIN; ++z) {
                for (int64_t x = MINIMUM_X - SNAPSHOT_MARGIN; x <= MAXIMUM_X + SNAPSHOT_MARGIN;
                     ++x) {
                    const FluidCell cell = world.readFluidCell({x, y, z});
                    REQUIRE(cell.loaded);
                    result.push_back(static_cast<uint16_t>(static_cast<uint16_t>(cell.block) << 8U |
                                                           cell.state.packed()));
                }
            }
        }
        return result;
    };

    const std::vector<uint16_t> before = snapshot();
    for (int64_t z = MINIMUM_Z; z <= MAXIMUM_Z; ++z) {
        for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
            const RapidColumn& column = columns[columnIndex(x, z)];
            if (!column.rapid)
                continue;
            const FluidCell generated = world.readFluidCell({x, column.topY, z});
            REQUIRE(generated.isWater());
            if (column.level == 0) {
                REQUIRE(generated.state.isSource());
            } else {
                REQUIRE_FALSE(generated.state.isSource());
                REQUIRE(generated.state.level() == column.level);
            }
            const FluidCell airAbove = world.readFluidCell({x, column.topY + 1, z});
            REQUIRE(airAbove.loaded);
            REQUIRE(airAbove.block == BlockType::AIR);
            world.setBlock(x, column.topY + 1, z, BlockType::AIR);
        }
    }

    int elapsedTicks = 0;
    while (elapsedTicks < 40) {
        world.tickFluids(1.0 / static_cast<double>(FLUID_TICKS_PER_SECOND));
        ++elapsedTicks;
    }
    const std::vector<uint16_t> afterFortyTicks = snapshot();
    const auto firstEarlyChange =
        std::mismatch(before.begin(), before.end(), afterFortyTicks.begin());
    if (firstEarlyChange.first != before.end()) {
        constexpr size_t SNAPSHOT_WIDTH = WIDTH + SNAPSHOT_MARGIN * 2;
        constexpr size_t SNAPSHOT_DEPTH = DEPTH + SNAPSHOT_MARGIN * 2;
        const size_t linear =
            static_cast<size_t>(std::distance(before.begin(), firstEarlyChange.first));
        const int changedY =
            snapshotMinimumY + static_cast<int>(linear / (SNAPSHOT_WIDTH * SNAPSHOT_DEPTH));
        const size_t horizontal = linear % (SNAPSHOT_WIDTH * SNAPSHOT_DEPTH);
        const int64_t changedZ =
            MINIMUM_Z - SNAPSHOT_MARGIN + static_cast<int64_t>(horizontal / SNAPSHOT_WIDTH);
        const int64_t changedX =
            MINIMUM_X - SNAPSHOT_MARGIN + static_cast<int64_t>(horizontal % SNAPSHOT_WIDTH);
        const auto encodedCell = [&](int64_t x, int y, int64_t z) {
            const FluidCell cell = world.readFluidCell({x, y, z});
            return static_cast<uint16_t>(static_cast<uint16_t>(cell.block) << 8U |
                                         cell.state.packed());
        };
        const uint16_t changedCenter = encodedCell(changedX, changedY, changedZ);
        const uint16_t changedDown = encodedCell(changedX, changedY - 1, changedZ);
        const uint16_t changedUp = encodedCell(changedX, changedY + 1, changedZ);
        const uint16_t changedWest = encodedCell(changedX - 1, changedY, changedZ);
        const uint16_t changedEast = encodedCell(changedX + 1, changedY, changedZ);
        const uint16_t changedNorth = encodedCell(changedX, changedY, changedZ - 1);
        const uint16_t changedSouth = encodedCell(changedX, changedY, changedZ + 1);
        CAPTURE(changedX, changedY, changedZ, *firstEarlyChange.first, *firstEarlyChange.second,
                changedCenter, changedDown, changedUp, changedWest, changedEast, changedNorth,
                changedSouth, world.getPendingFluidCount());
        CHECK(firstEarlyChange.first == before.end());
    }
    while (world.getPendingFluidCount() > 0 && elapsedTicks < MAXIMUM_SETTLE_TICKS) {
        world.tickFluids(1.0 / static_cast<double>(FLUID_TICKS_PER_SECOND));
        ++elapsedTicks;
    }
    CAPTURE(elapsedTicks, world.getPendingFluidCount());
    REQUIRE(elapsedTicks >= 40);
    REQUIRE(elapsedTicks < MAXIMUM_SETTLE_TICKS);
    REQUIRE(world.getPendingFluidCount() == 0);
    REQUIRE(snapshot() == before);
}

TEST_CASE("Exact and multiresolution samples share one water authority",
          "[worldgen][advanced][hydrology][water][lod][determinism]") {
    struct Fixture {
        uint64_t seed;
        int64_t x;
        int64_t z;
    };
    constexpr std::array<Fixture, 8> fixtures = {
        Fixture{42, -557, 379},
        Fixture{42, -395, 393},
        Fixture{42, -360, 416},
        Fixture{42, -8'192, 3'056},
        Fixture{42, -8'240, 3'088},
        Fixture{764891, 23'029, -111'486},
        Fixture{764891, 21'992, -112'016},
        Fixture{764891, 22'017, -111'442},
    };
    constexpr std::array<worldgen::SurfaceFootprint, 5> footprints = {
        worldgen::SurfaceFootprint::BLOCK_1,  worldgen::SurfaceFootprint::BLOCK_2,
        worldgen::SurfaceFootprint::BLOCK_4,  worldgen::SurfaceFootprint::BLOCK_8,
        worldgen::SurfaceFootprint::BLOCK_16,
    };

    std::map<uint64_t, std::unique_ptr<ChunkGenerator>> generators;
    for (const Fixture& fixture : fixtures) {
        auto [found, inserted] = generators.try_emplace(fixture.seed);
        if (inserted)
            found->second = std::make_unique<ChunkGenerator>(fixture.seed);
        ChunkGenerator& generator = *found->second;
        const worldgen::SurfaceSample canonical =
            generator.sampleFarSurface(fixture.x, fixture.z, worldgen::SurfaceFootprint::BLOCK_1);
        CAPTURE(fixture.seed, fixture.x, fixture.z, canonical.terrainHeight, canonical.waterSurface,
                canonical.hydrology.ocean, canonical.hydrology.river, canonical.hydrology.lake,
                canonical.hydrology.waterfall, canonical.hydrology.waterBodyId);
        requireSameWaterAuthority(generator.sampleSurface(fixture.x, fixture.z), canonical);
        requireSameWaterAuthority(generator.sampleExactGeometrySurface(fixture.x, fixture.z),
                                  canonical);
        for (worldgen::SurfaceFootprint footprint : footprints) {
            const worldgen::SurfaceSample sampled =
                generator.sampleFarSurface(fixture.x, fixture.z, footprint);
            requireSameWaterAuthority(sampled, canonical);
            if (isWetSurface(sampled))
                REQUIRE(sampled.waterSurface > sampled.terrainHeight);
            if (sampled.hydrology.lake) {
                REQUIRE(sampled.hydrology.waterBodyId != worldgen::NO_WATER_BODY);
                REQUIRE(sampled.hydrology.lakeDepth ==
                        Catch::Approx(sampled.waterSurface - sampled.terrainHeight).margin(1.0e-6));
            }
        }
    }
}

TEST_CASE("Seed forty two emits contiguous settled water from block authority",
          "[worldgen][advanced][hydrology][water][emission][regression]") {
    struct Window {
        int64_t centerX;
        int64_t centerZ;
        int radius;
    };
    constexpr std::array<Window, 2> windows = {
        Window{-557, 379, 48},
        Window{-395, 405, 48},
    };
    ChunkGenerator generator(42);
    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    const auto blockAt = [&](int64_t x, int y, int64_t z) -> std::pair<BlockType, FluidState> {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return {found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                        Chunk::worldToLocal(z)),
                found->second->getFluidState(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                             Chunk::worldToLocal(z))};
    };

    for (const Window& window : windows) {
        const int edge = window.radius * 2 + 1;
        std::vector<uint8_t> expectedWet(static_cast<size_t>(edge) * edge);
        std::vector<uint8_t> emittedWet(static_cast<size_t>(edge) * edge);
        std::array<size_t, 8> expectedXPhase{};
        std::array<size_t, 8> expectedZPhase{};
        std::array<size_t, 8> emittedXPhase{};
        std::array<size_t, 8> emittedZPhase{};
        size_t topologyMismatches = 0;
        size_t levelMismatches = 0;
        size_t occupancyMismatches = 0;
        size_t incorrectFluidColumns = 0;
        int64_t firstOccupancyMismatchX = std::numeric_limits<int64_t>::min();
        int64_t firstOccupancyMismatchZ = std::numeric_limits<int64_t>::min();
        int firstOccupancyMismatchSurfaceY = WORLD_MIN_Y;
        int firstOccupancyMismatchWaterY = WORLD_MIN_Y;
        BlockType firstOccupancyMismatchBlock = BlockType::AIR;

        for (int localZ = 0; localZ < edge; ++localZ) {
            for (int localX = 0; localX < edge; ++localX) {
                const int64_t x = window.centerX - window.radius + localX;
                const int64_t z = window.centerZ - window.radius + localZ;
                const worldgen::SurfaceSample block =
                    generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
                const worldgen::SurfaceSample plan = generator.sampleSurface(x, z);
                const bool blockWet = isWetSurface(block);
                if (block.hydrology.ocean != plan.hydrology.ocean ||
                    block.hydrology.river != plan.hydrology.river ||
                    block.hydrology.lake != plan.hydrology.lake ||
                    block.hydrology.waterfall != plan.hydrology.waterfall ||
                    block.hydrology.waterBodyId != plan.hydrology.waterBodyId) {
                    ++topologyMismatches;
                }
                if (blockWet && std::abs(block.waterSurface - plan.waterSurface) > 1.0e-4) {
                    ++levelMismatches;
                }

                const int surfaceY = generator.surfaceYAt(x, z);
                const int waterTopY = std::clamp(
                    static_cast<int>(std::ceil(block.waterSurface)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
                const bool shouldEmit = blockWet && surfaceY < waterTopY;
                const BlockType topBlock = blockAt(x, waterTopY, z).first;
                const bool topIsWater = topBlock == BlockType::WATER;
                const bool topIsRootedTree = isFloodTolerantLog(topBlock);
                const bool topIsOccupied = topIsWater || topIsRootedTree;
                const size_t index = static_cast<size_t>(localZ) * edge + localX;
                expectedWet[index] = static_cast<uint8_t>(shouldEmit);
                emittedWet[index] = static_cast<uint8_t>(topIsOccupied);
                const size_t xPhase = static_cast<size_t>(world_coord::floorMod(x, int64_t{8}));
                const size_t zPhase = static_cast<size_t>(world_coord::floorMod(z, int64_t{8}));
                if (shouldEmit) {
                    ++expectedXPhase[xPhase];
                    ++expectedZPhase[zPhase];
                }
                if (topIsOccupied) {
                    ++emittedXPhase[xPhase];
                    ++emittedZPhase[zPhase];
                }
                if (shouldEmit != topIsOccupied) {
                    ++occupancyMismatches;
                    if (firstOccupancyMismatchX == std::numeric_limits<int64_t>::min()) {
                        firstOccupancyMismatchX = x;
                        firstOccupancyMismatchZ = z;
                        firstOccupancyMismatchSurfaceY = surfaceY;
                        firstOccupancyMismatchWaterY = waterTopY;
                        firstOccupancyMismatchBlock = topBlock;
                    }
                }
                if (shouldEmit) {
                    bool completeColumn = true;
                    const worldgen::GeneratedFluidColumn generatedFluid =
                        worldgen::generatedFluidColumn(block);
                    for (int y = surfaceY + 1; y <= waterTopY && completeColumn; ++y) {
                        const auto [columnBlock, columnFluid] = blockAt(x, y, z);
                        const FluidState expectedFluid =
                            y == waterTopY ? generatedFluid.topState : FluidState::source();
                        completeColumn =
                            isFloodTolerantLog(columnBlock) ||
                            (columnBlock == BlockType::WATER && columnFluid == expectedFluid);
                    }
                    if (!completeColumn) {
                        ++incorrectFluidColumns;
                    }
                }
            }
        }

        size_t isolatedWetColumns = 0;
        std::optional<ColumnPos> firstIsolatedWetColumn;
        for (int z = 1; z + 1 < edge; ++z) {
            for (int x = 1; x + 1 < edge; ++x) {
                const size_t index = static_cast<size_t>(z) * edge + x;
                if (emittedWet[index] == 0)
                    continue;
                bool hasWetNeighbor = false;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dz == 0)
                            continue;
                        hasWetNeighbor =
                            hasWetNeighbor ||
                            emittedWet[static_cast<size_t>(z + dz) * edge + x + dx] != 0;
                    }
                }
                if (!hasWetNeighbor) {
                    ++isolatedWetColumns;
                    if (!firstIsolatedWetColumn.has_value()) {
                        firstIsolatedWetColumn = ColumnPos{
                            window.centerX - window.radius + x,
                            window.centerZ - window.radius + z,
                        };
                    }
                }
            }
        }

        const int64_t firstIsolatedX = firstIsolatedWetColumn.has_value()
                                           ? firstIsolatedWetColumn->x
                                           : std::numeric_limits<int64_t>::min();
        const int64_t firstIsolatedZ = firstIsolatedWetColumn.has_value()
                                           ? firstIsolatedWetColumn->z
                                           : std::numeric_limits<int64_t>::min();
        CAPTURE(window.centerX, window.centerZ, window.radius, topologyMismatches, levelMismatches,
                occupancyMismatches, incorrectFluidColumns, isolatedWetColumns, expectedXPhase,
                expectedZPhase, emittedXPhase, emittedZPhase, firstIsolatedX, firstIsolatedZ,
                firstOccupancyMismatchX, firstOccupancyMismatchZ, firstOccupancyMismatchSurfaceY,
                firstOccupancyMismatchWaterY, firstOccupancyMismatchBlock);
        REQUIRE(topologyMismatches == 0);
        REQUIRE(levelMismatches == 0);
        REQUIRE(occupancyMismatches == 0);
        REQUIRE(incorrectFluidColumns == 0);
        REQUIRE(isolatedWetColumns == 0);
        for (size_t phase = 0; phase < 8; ++phase) {
            REQUIRE(expectedXPhase[phase] > 0);
            REQUIRE(expectedZPhase[phase] > 0);
            REQUIRE(emittedXPhase[phase] == expectedXPhase[phase]);
            REQUIRE(emittedZPhase[phase] == expectedZPhase[phase]);
        }
    }
}

TEST_CASE("Lake identity and shoreline contours cross catchment and page faces",
          "[worldgen][advanced][hydrology][lake][shore][seam]") {
    worldgen::MacroGenerationSampler sampler(42);
    constexpr double CATCHMENT_FACE_X = -8'192.0;
    constexpr double LAKE_Z = 3'056.0;
    constexpr double PAGE_FACE_Z = 3'072.0;
    constexpr double EPSILON = 1.0e-4;

    const worldgen::HydrologySample west =
        sampler.sampleHydrology(CATCHMENT_FACE_X - EPSILON, LAKE_Z);
    const worldgen::HydrologySample east =
        sampler.sampleHydrology(CATCHMENT_FACE_X + EPSILON, LAKE_Z);
    CAPTURE(west.surfaceElevation, east.surfaceElevation, west.lakeDepth, east.lakeDepth,
            west.lakeShoreDistance, east.lakeShoreDistance, west.channelDistance,
            east.channelDistance);
    REQUIRE(west.lake);
    REQUIRE(east.lake);
    REQUIRE(west.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(east.waterBodyId == west.waterBodyId);
    REQUIRE(east.waterSurface == Catch::Approx(west.waterSurface).margin(1.0e-6));
    REQUIRE(east.surfaceElevation == Catch::Approx(west.surfaceElevation).margin(0.001));
    REQUIRE(east.lakeShoreDistance == Catch::Approx(west.lakeShoreDistance).margin(0.001));

    std::optional<double> contourX;
    for (double x = -8'320.0; x <= -8'160.0; x += 1.0) {
        const worldgen::HydrologySample north = sampler.sampleHydrology(x, PAGE_FACE_Z - 0.25);
        const worldgen::HydrologySample south = sampler.sampleHydrology(x, PAGE_FACE_Z + 0.25);
        if (north.shoreWaterSurface <= 0.0 || south.shoreWaterSurface <= 0.0)
            continue;
        if (std::abs(north.lakeShoreDistance) > 8.0 || std::abs(south.lakeShoreDistance) > 8.0) {
            continue;
        }
        contourX = x;
        break;
    }
    REQUIRE(contourX.has_value());

    const worldgen::HydrologySample northOuter =
        sampler.sampleHydrology(*contourX, PAGE_FACE_Z - 0.25);
    const worldgen::HydrologySample northInner =
        sampler.sampleHydrology(*contourX, PAGE_FACE_Z - EPSILON);
    const worldgen::HydrologySample southInner =
        sampler.sampleHydrology(*contourX, PAGE_FACE_Z + EPSILON);
    const worldgen::HydrologySample southOuter =
        sampler.sampleHydrology(*contourX, PAGE_FACE_Z + 0.25);
    REQUIRE(northInner.shoreWaterSurface ==
            Catch::Approx(southInner.shoreWaterSurface).margin(1.0e-6));
    REQUIRE(northInner.lakeShoreDistance ==
            Catch::Approx(southInner.lakeShoreDistance).margin(0.001));

    const double northDerivative =
        (northInner.lakeShoreDistance - northOuter.lakeShoreDistance) / (0.25 - EPSILON);
    const double southDerivative =
        (southOuter.lakeShoreDistance - southInner.lakeShoreDistance) / (0.25 - EPSILON);
    REQUIRE(northDerivative == Catch::Approx(southDerivative).margin(0.05));
}

TEST_CASE("Lake membership keeps one supported body level through cubic emission",
          "[worldgen][advanced][hydrology][lake][support]") {
    constexpr int64_t LAKE_X = -8'348;
    constexpr int64_t LAKE_Z = 2'281;
    worldgen::MacroGenerationSampler macro(42);
    ChunkGenerator generator(42);

    const worldgen::HydrologySample lakeMacro =
        macro.sampleHydrology(static_cast<double>(LAKE_X), static_cast<double>(LAKE_Z));
    const worldgen::SurfaceSample lakePlan = generator.sampleSurface(LAKE_X, LAKE_Z);
    const worldgen::SurfaceSample lakeExact = generator.sampleExactSurface(LAKE_X, LAKE_Z);
    REQUIRE(lakeMacro.lake);
    REQUIRE(lakePlan.hydrology.lake);
    REQUIRE(lakeExact.hydrology.lake);
    REQUIRE(lakeMacro.waterSurface - lakeMacro.surfaceElevation ==
            Catch::Approx(lakeMacro.lakeDepth).margin(1.0e-6));
    REQUIRE(lakePlan.waterSurface - lakePlan.terrainHeight ==
            Catch::Approx(lakePlan.hydrology.lakeDepth).margin(1.0e-6));
    REQUIRE(lakeExact.waterSurface - lakeExact.terrainHeight ==
            Catch::Approx(lakeExact.hydrology.lakeDepth).margin(1.0e-6));

    const int firstWaterY = static_cast<int>(lakeExact.terrainHeight);
    const int lastWaterY = static_cast<int>(std::ceil(lakeExact.waterSurface)) - 1;
    REQUIRE(firstWaterY <= lastWaterY);
    for (int worldY = firstWaterY; worldY <= lastWaterY; ++worldY) {
        Chunk cube(ChunkPos{Chunk::worldToChunk(LAKE_X), Chunk::worldToChunkY(worldY),
                            Chunk::worldToChunk(LAKE_Z)});
        generator.generateCube(cube);
        REQUIRE(cube.getBlock(Chunk::worldToLocal(LAKE_X), Chunk::worldToLocalY(worldY),
                              Chunk::worldToLocal(LAKE_Z)) == BlockType::WATER);
    }

    size_t supportedLakeSamples = 0;
    for (int dz = -96; dz <= 96; dz += 4) {
        for (int dx = -96; dx <= 96; dx += 4) {
            const worldgen::SurfaceSample sample =
                generator.sampleSurface(LAKE_X + dx, LAKE_Z + dz);
            if (!sample.hydrology.lake)
                continue;
            ++supportedLakeSamples;
            REQUIRE(sample.waterSurface > sample.terrainHeight);
            REQUIRE(sample.waterSurface - sample.terrainHeight ==
                    Catch::Approx(sample.hydrology.lakeDepth).margin(1.0e-5));
        }
    }
    REQUIRE(supportedLakeSamples > 64);

    const worldgen::HydrologySample throughLake = macro.sampleHydrology(-8'272.0, 3'056.0);
    const worldgen::HydrologySample outlet = macro.sampleHydrology(-8'240.0, 3'088.0);
    const worldgen::HydrologySample downstream = macro.sampleHydrology(-8'192.0, 3'136.0);
    REQUIRE(throughLake.lake);
    REQUIRE_FALSE(throughLake.endorheic);
    REQUIRE(outlet.ocean);
    REQUIRE_FALSE(outlet.river);
    REQUIRE(downstream.ocean);
    REQUIRE_FALSE(downstream.river);
    REQUIRE(outlet.waterfall);
    REQUIRE(outlet.surfaceElevation < outlet.waterSurface);
    REQUIRE(downstream.surfaceElevation < downstream.waterSurface);
    REQUIRE(outlet.waterSurface <= throughLake.waterSurface);
    REQUIRE(downstream.waterSurface <= outlet.waterSurface);
}

TEST_CASE("Canonical lake samples stitch across negative column faces",
          "[worldgen][advanced][hydrology][lake][seam]") {
    ChunkGenerator generator(42);
    constexpr int64_t FACE_X = -8'256;
    constexpr int64_t BASE_Z = 2'288;
    const ColumnPos eastColumn{Chunk::worldToChunk(FACE_X), Chunk::worldToChunk(BASE_Z)};
    const ColumnPos westColumn{eastColumn.x - 1, eastColumn.z};
    const auto west = generator.getColumnPlan(westColumn);
    const auto east = generator.getColumnPlan(eastColumn);

    for (int localZ = 1; localZ < CHUNK_EDGE; ++localZ) {
        const worldgen::SurfaceSample westFace = west->sample(CHUNK_EDGE, localZ);
        const worldgen::SurfaceSample eastFace = east->sample(0, localZ);
        REQUIRE(westFace.hydrology.lake == eastFace.hydrology.lake);
        REQUIRE(westFace.hydrology.endorheic == eastFace.hydrology.endorheic);
        REQUIRE(westFace.waterSurface == Catch::Approx(eastFace.waterSurface).margin(1.0e-5));
        REQUIRE(westFace.terrainHeight == Catch::Approx(eastFace.terrainHeight).margin(1.0e-5));
        REQUIRE(westFace.hydrology.lakeDepth ==
                Catch::Approx(eastFace.hydrology.lakeDepth).margin(1.0e-5));
    }
}

TEST_CASE("Elevated lake shorelines follow supported terrain contours",
          "[worldgen][advanced][hydrology][lake][shore][support][regression]") {
    constexpr int64_t MIN_X = -8'288;
    constexpr int64_t MAX_X = -8'064;
    constexpr int64_t MIN_Z = 3'176;
    constexpr int64_t MAX_Z = 3'336;
    worldgen::MacroGenerationSampler macro(42);
    ChunkGenerator generator(42);
    const double lakeLevel = macro.sampleHydrology(-8'272.0, 3'056.0).waterSurface;

    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    const auto blockAt = [&](int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return std::pair{found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                                 Chunk::worldToLocal(z)),
                         found->second->getFluidState(Chunk::worldToLocal(x),
                                                      Chunk::worldToLocalY(y),
                                                      Chunk::worldToLocal(z))};
    };

    std::vector<std::pair<int64_t, int64_t>> shoreline;
    for (int64_t z = MIN_Z; z <= MAX_Z; ++z) {
        bool previousLake = macro.sampleHydrology(MIN_X, z).lake;
        for (int64_t x = MIN_X + 1; x <= MAX_X; ++x) {
            const worldgen::HydrologySample current = macro.sampleHydrology(x, z);
            if (previousLake && !current.lake && !current.river && !current.waterfall) {
                const int64_t wetX = x - 1;
                bool outletTransition = false;
                for (int dz = -8; dz <= 8 && !outletTransition; dz += 4) {
                    for (int dx = -8; dx <= 8; dx += 4) {
                        const worldgen::HydrologySample nearby =
                            macro.sampleHydrology(x + dx, z + dz);
                        if (nearby.river || nearby.waterfall) {
                            outletTransition = true;
                            break;
                        }
                    }
                }
                if (outletTransition)
                    break;
                const worldgen::HydrologySample wet = macro.sampleHydrology(wetX, z);
                const worldgen::SurfaceSample plannedWet = generator.sampleSurface(wetX, z);
                const worldgen::SurfaceSample exactWet = generator.sampleExactSurface(wetX, z);
                const worldgen::SurfaceSample exactDry = generator.sampleExactSurface(x, z);
                INFO("shore at " << x << ", " << z);
                REQUIRE(wet.lake);
                REQUIRE(wet.waterSurface == Catch::Approx(lakeLevel).margin(1.0e-4));
                REQUIRE(wet.lakeShoreDistance > 0.0);
                REQUIRE_FALSE(current.lake);
                REQUIRE(current.lakeBank);
                REQUIRE(current.lakeBankInfluence > 0.0);
                REQUIRE(current.lakeShoreDistance <= 0.0);
                REQUIRE(current.shoreWaterSurface ==
                        Catch::Approx(wet.waterSurface).margin(1.0e-4));
                REQUIRE(exactWet.hydrology.lake);
                REQUIRE_FALSE(exactDry.hydrology.lake);

                const int waterTopY = static_cast<int>(std::ceil(wet.waterSurface)) - 1;
                const auto [wetBlock, wetFluid] = blockAt(wetX, waterTopY, z);
                const auto [dryBlock, dryFluid] = blockAt(x, waterTopY, z);
                CAPTURE(wetX, wet.surfaceElevation, wet.waterSurface, plannedWet.hydrology.lake,
                        plannedWet.hydrology.lakeBank, plannedWet.terrainHeight,
                        plannedWet.waterSurface, exactWet.terrainHeight,
                        generator.surfaceYAt(wetX, z), waterTopY, wetBlock);
                (void)dryFluid;
                if (isFloodTolerantLog(wetBlock)) {
                    const int surfaceY = generator.surfaceYAt(wetX, z);
                    REQUIRE(waterTopY - surfaceY <= maximumRootedWaterDepth(wetBlock));
                    REQUIRE(worldgen::surface_material::supportsTreeRooting(
                        blockAt(wetX, surfaceY, z).first));
                    for (int y = surfaceY + 1; y <= waterTopY; ++y)
                        REQUIRE(blockAt(wetX, y, z).first == wetBlock);
                } else {
                    REQUIRE(wetBlock == BlockType::WATER);
                    REQUIRE(wetFluid.isSource());
                }
                REQUIRE(isSolid(dryBlock));
                REQUIRE(exactDry.terrainHeight > waterTopY);

                const worldgen::SurfaceSample nextDry = generator.sampleExactSurface(x + 1, z);
                REQUIRE(std::abs(nextDry.terrainHeight - exactDry.terrainHeight) <= 2.0);
                shoreline.emplace_back(z, x);
                break;
            }
            previousLake = current.lake;
        }
    }

    REQUIRE(shoreline.size() > 120);
    std::set<int64_t> distinctShoreX;
    int64_t minimumShoreX = std::numeric_limits<int64_t>::max();
    int64_t maximumShoreX = std::numeric_limits<int64_t>::min();
    size_t longestStraightRun = 0;
    size_t straightRun = 0;
    int64_t previousZ = std::numeric_limits<int64_t>::min();
    int64_t previousX = std::numeric_limits<int64_t>::min();
    for (const auto [z, x] : shoreline) {
        distinctShoreX.insert(x);
        minimumShoreX = std::min(minimumShoreX, x);
        maximumShoreX = std::max(maximumShoreX, x);
        straightRun = z == previousZ + 1 && x == previousX ? straightRun + 1 : 1;
        longestStraightRun = std::max(longestStraightRun, straightRun);
        previousZ = z;
        previousX = x;
    }
    REQUIRE(distinctShoreX.size() >= 8);
    REQUIRE(maximumShoreX - minimumShoreX >= 12);
    REQUIRE(longestStraightRun <= 24);

    const worldgen::HydrologySample seamBank = macro.sampleHydrology(-8'192.0, 3'040.0);
    REQUIRE_FALSE(seamBank.lakeBank);
    REQUIRE(seamBank.lakeBankInfluence == 0.0);

    double maximumProminence = 0.0;
    size_t dryProminenceSamples = 0;
    const auto isExposedDryTerrain = [](const worldgen::SurfaceSample& sample) {
        return !sample.hydrology.ocean && !sample.hydrology.lake && !sample.hydrology.river &&
               !sample.hydrology.waterfall;
    };
    // Cover the complete dry receiving rim rather than only the former
    // catchment-local contour. The wider window strengthens the prominence
    // check after the canonical lake crosses that storage face.
    for (int64_t z = 3'096; z <= 3'176; z += 4) {
        for (int64_t x = -8'240; x <= -8'152; x += 4) {
            const worldgen::SurfaceSample center = generator.sampleExactSurface(x, z);
            if (!isExposedDryTerrain(center))
                continue;
            double perimeterTotal = 0.0;
            size_t perimeterCount = 0;
            bool exposedPerimeter = true;
            for (int dz = -8; dz <= 8; dz += 4) {
                for (int dx = -8; dx <= 8; dx += 4) {
                    if (std::abs(dx) != 8 && std::abs(dz) != 8)
                        continue;
                    const worldgen::SurfaceSample perimeter =
                        generator.sampleExactSurface(x + dx, z + dz);
                    if (!isExposedDryTerrain(perimeter)) {
                        exposedPerimeter = false;
                        break;
                    }
                    perimeterTotal += perimeter.terrainHeight;
                    ++perimeterCount;
                }
                if (!exposedPerimeter)
                    break;
            }
            if (!exposedPerimeter)
                continue;
            ++dryProminenceSamples;
            maximumProminence =
                std::max(maximumProminence, center.terrainHeight - perimeterTotal / perimeterCount);
        }
    }
    REQUIRE(dryProminenceSamples > 16);
    REQUIRE(maximumProminence <= 6.0);
}

TEST_CASE("Lake outlets emit narrow finished falls into their receiving water",
          "[worldgen][advanced][hydrology][lake][waterfall][support][determinism]") {
    constexpr int64_t LAKE_OUTLET_X = -8'272;
    constexpr int64_t LAKE_OUTLET_Z = 3'056;
    constexpr int64_t FALL_X = -8'240;
    constexpr int64_t FALL_LIP_X = -8'241;
    constexpr int64_t FALL_Z = 3'088;
    worldgen::MacroGenerationSampler macro(42);

    const worldgen::HydrologySample lake = macro.sampleHydrology(LAKE_OUTLET_X, LAKE_OUTLET_Z);
    const worldgen::HydrologySample fall = macro.sampleHydrology(FALL_X, FALL_Z);
    REQUIRE(lake.lake);
    REQUIRE_FALSE(lake.endorheic);
    REQUIRE(fall.ocean);
    REQUIRE_FALSE(fall.river);
    REQUIRE(fall.waterfall);
    REQUIRE(fall.waterfallAnchor);
    REQUIRE(fall.streamOrder >= lake.streamOrder);
    REQUIRE(fall.discharge >= lake.discharge);
    REQUIRE(fall.waterSurface > fall.surfaceElevation);
    REQUIRE(fall.waterSurface < lake.waterSurface);
    REQUIRE(fall.waterfallBottom == Catch::Approx(fall.waterSurface));
    REQUIRE(fall.waterfallTop == Catch::Approx(lake.waterSurface).margin(1.0e-4));
    REQUIRE(fall.waterfallTop - fall.waterfallBottom >= 2.5);
    REQUIRE(fall.waterfallWidth >= 4.0);
    REQUIRE_FALSE(fall.lakeBank);
    REQUIRE(fall.waterBodyId == worldgen::NO_WATER_BODY);

    constexpr int64_t CORRIDOR_ORIGIN_X = -8'296;
    constexpr int64_t CORRIDOR_ORIGIN_Z = 3'032;
    constexpr int CORRIDOR_EDGE = 81;
    std::vector<uint8_t> corridorWet(CORRIDOR_EDGE * CORRIDOR_EDGE);
    for (int z = 0; z < CORRIDOR_EDGE; ++z) {
        for (int x = 0; x < CORRIDOR_EDGE; ++x) {
            const worldgen::HydrologySample sample =
                macro.sampleHydrology(CORRIDOR_ORIGIN_X + x, CORRIDOR_ORIGIN_Z + z);
            corridorWet[static_cast<size_t>(z * CORRIDOR_EDGE + x)] = static_cast<uint8_t>(
                sample.ocean || sample.river || sample.lake || sample.waterfall);
        }
    }
    const auto corridorIndex = [](int64_t x, int64_t z) {
        return static_cast<int>((z - CORRIDOR_ORIGIN_Z) * CORRIDOR_EDGE + (x - CORRIDOR_ORIGIN_X));
    };
    const int start = corridorIndex(LAKE_OUTLET_X, LAKE_OUTLET_Z);
    const int target = corridorIndex(FALL_X, FALL_Z);
    REQUIRE(corridorWet[static_cast<size_t>(start)] != 0);
    REQUIRE(corridorWet[static_cast<size_t>(target)] != 0);
    std::vector<uint8_t> visited(CORRIDOR_EDGE * CORRIDOR_EDGE);
    std::vector<int> frontier{start};
    visited[static_cast<size_t>(start)] = 1;
    while (!frontier.empty() && visited[static_cast<size_t>(target)] == 0) {
        const int current = frontier.back();
        frontier.pop_back();
        const int x = current % CORRIDOR_EDGE;
        const int z = current / CORRIDOR_EDGE;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dz == 0)
                    continue;
                const int neighborX = x + dx;
                const int neighborZ = z + dz;
                if (neighborX < 0 || neighborX >= CORRIDOR_EDGE || neighborZ < 0 ||
                    neighborZ >= CORRIDOR_EDGE) {
                    continue;
                }
                const int neighbor = neighborZ * CORRIDOR_EDGE + neighborX;
                if (corridorWet[static_cast<size_t>(neighbor)] == 0 ||
                    visited[static_cast<size_t>(neighbor)] != 0) {
                    continue;
                }
                visited[static_cast<size_t>(neighbor)] = 1;
                frontier.push_back(neighbor);
            }
        }
    }
    REQUIRE(visited[static_cast<size_t>(target)] != 0);

    constexpr double DOWNSTREAM_PORTAL_X = -8'192.0;
    constexpr double DOWNSTREAM_PORTAL_Z = 3'136.0;
    constexpr double PORTAL_EPSILON = 1.0e-4;
    const worldgen::HydrologySample portalWest =
        macro.sampleHydrology(DOWNSTREAM_PORTAL_X - PORTAL_EPSILON, DOWNSTREAM_PORTAL_Z);
    const worldgen::HydrologySample portalEast =
        macro.sampleHydrology(DOWNSTREAM_PORTAL_X + PORTAL_EPSILON, DOWNSTREAM_PORTAL_Z);
    REQUIRE(portalWest.ocean);
    REQUIRE(portalEast.ocean);
    REQUIRE_FALSE(portalWest.river);
    REQUIRE_FALSE(portalEast.river);
    REQUIRE(portalWest.surfaceElevation ==
            Catch::Approx(portalEast.surfaceElevation).margin(0.001));
    REQUIRE(portalWest.waterSurface == Catch::Approx(portalEast.waterSurface).margin(0.001));
    REQUIRE(portalWest.channelWidth == Catch::Approx(portalEast.channelWidth).margin(0.001));
    REQUIRE(portalWest.channelDepth == Catch::Approx(portalEast.channelDepth).margin(0.001));
    REQUIRE(portalWest.waterSurface >= SEA_LEVEL);
    REQUIRE(portalWest.waterSurface == SEA_LEVEL);
    REQUIRE(portalWest.waterSurface <= fall.waterfallBottom + 0.02);
    REQUIRE(portalWest.waterSurface > portalWest.surfaceElevation);
    REQUIRE(portalWest.waterSurface - portalWest.surfaceElevation + 0.05 >=
            portalWest.channelDepth);
    REQUIRE(portalWest.waterSurface - portalWest.surfaceElevation <= 16.0);

    macro.clearBasinCache();
    const worldgen::HydrologySample rebuilt = macro.sampleHydrology(FALL_X, FALL_Z);
    REQUIRE(rebuilt.waterSurface == fall.waterSurface);
    REQUIRE(rebuilt.waterfallTop == fall.waterfallTop);
    REQUIRE(rebuilt.waterfallBottom == fall.waterfallBottom);
    REQUIRE(rebuilt.waterfallWidth == fall.waterfallWidth);
    REQUIRE(rebuilt.waterfallAnchor == fall.waterfallAnchor);

    ChunkGenerator generator(42);
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(FALL_X, FALL_Z);
    const worldgen::SurfaceSample far = generator.sampleFarSurface(FALL_X, FALL_Z);
    REQUIRE(exact.hydrology.waterfall);
    REQUIRE(exact.hydrology.waterfallAnchor);
    REQUIRE(exact.waterSurface == Catch::Approx(fall.waterSurface));
    REQUIRE(exact.hydrology.waterfallTop == Catch::Approx(fall.waterfallTop));
    REQUIRE(far.hydrology.waterfall);
    REQUIRE(far.hydrology.waterfallAnchor);
    REQUIRE(far.waterSurface == Catch::Approx(fall.waterSurface));
    REQUIRE(far.hydrology.waterfallTop == Catch::Approx(fall.waterfallTop));

    std::array<std::unique_ptr<Chunk>, 3> cubes;
    for (int index = 0; index < 3; ++index) {
        const int worldY = 62 + index * CHUNK_EDGE;
        cubes[index] = std::make_unique<Chunk>(ChunkPos{Chunk::worldToChunk(FALL_LIP_X),
                                                        Chunk::worldToChunkY(worldY),
                                                        Chunk::worldToChunk(FALL_Z)});
        generator.generateCube(*cubes[index]);
    }
    auto cubeAt = [&](int worldY) -> Chunk& {
        return *cubes[static_cast<size_t>(Chunk::worldToChunkY(worldY) - Chunk::worldToChunkY(62))];
    };
    const int localX = Chunk::worldToLocal(FALL_LIP_X);
    const int localZ = Chunk::worldToLocal(FALL_Z);
    const worldgen::SurfaceSample exactLip = generator.sampleExactSurface(FALL_LIP_X, FALL_Z);
    REQUIRE(exactLip.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL);
    REQUIRE(exactLip.hydrology.generatedFluidLevel == 7);
    const worldgen::GeneratedFluidColumn generatedFall = worldgen::generatedFluidColumn(exactLip);
    const int firstFallingY = generatedFall.fallingStartY;
    const int lastFallingY = static_cast<int>(std::ceil(fall.waterfallTop)) - 1;
    for (int worldY = firstFallingY; worldY <= lastFallingY; ++worldY) {
        Chunk& cube = cubeAt(worldY);
        REQUIRE(cube.getBlock(localX, Chunk::worldToLocalY(worldY), localZ) == BlockType::WATER);
        const FluidState state = cube.getFluidState(localX, Chunk::worldToLocalY(worldY), localZ);
        REQUIRE(state.level() == 7);
        REQUIRE(state.isFalling() == (worldY < lastFallingY));
    }
    Chunk& receivingCube = cubeAt(firstFallingY - 1);
    REQUIRE(receivingCube.getBlock(localX, Chunk::worldToLocalY(firstFallingY - 1), localZ) ==
            BlockType::WATER);
    REQUIRE(receivingCube.getFluidState(localX, Chunk::worldToLocalY(firstFallingY - 1), localZ)
                .isSource());
    REQUIRE(
        cubeAt(lastFallingY + 1).getBlock(localX, Chunk::worldToLocalY(lastFallingY + 1), localZ) !=
        BlockType::WATER);

    const double flowLength = std::hypot(fall.flowDirection.x, fall.flowDirection.z);
    REQUIRE(flowLength > 0.0);
    double minimumAlong = std::numeric_limits<double>::max();
    double maximumAlong = std::numeric_limits<double>::lowest();
    int minimumAlongDx = 0;
    int minimumAlongDz = 0;
    int maximumAlongDx = 0;
    int maximumAlongDz = 0;
    worldgen::HydrologySample minimumAlongSample;
    worldgen::HydrologySample maximumAlongSample;
    size_t footprintSamples = 0;
    for (int dz = -16; dz <= 16; ++dz) {
        for (int dx = -16; dx <= 16; ++dx) {
            const worldgen::HydrologySample sample =
                macro.sampleHydrology(FALL_X + dx, FALL_Z + dz);
            if (!sample.waterfall || sample.waterfallTop < sample.waterfallBottom + 0.5)
                continue;
            const double along =
                (dx * fall.flowDirection.x + dz * fall.flowDirection.z) / flowLength;
            if (along < minimumAlong) {
                minimumAlong = along;
                minimumAlongDx = dx;
                minimumAlongDz = dz;
                minimumAlongSample = sample;
            }
            if (along > maximumAlong) {
                maximumAlong = along;
                maximumAlongDx = dx;
                maximumAlongDz = dz;
                maximumAlongSample = sample;
            }
            ++footprintSamples;
        }
    }
    CAPTURE(minimumAlong, maximumAlong, minimumAlongDx, minimumAlongDz, maximumAlongDx,
            maximumAlongDz, minimumAlongSample.waterfallTop, minimumAlongSample.waterfallBottom,
            minimumAlongSample.transitionOwnerKind, minimumAlongSample.transitionOwnerId,
            maximumAlongSample.waterfallTop, maximumAlongSample.waterfallBottom,
            maximumAlongSample.transitionOwnerKind, maximumAlongSample.transitionOwnerId);
    REQUIRE(footprintSamples > 0);
    REQUIRE(maximumAlong - minimumAlong <= 3.5);
    const int outsideOffset = static_cast<int>(std::ceil(fall.waterfallWidth + 4.0));
    const int64_t outsideX =
        FALL_X +
        static_cast<int64_t>(std::llround(-fall.flowDirection.z / flowLength * outsideOffset));
    const int64_t outsideZ =
        FALL_Z +
        static_cast<int64_t>(std::llround(fall.flowDirection.x / flowLength * outsideOffset));
    const worldgen::SurfaceSample outside = generator.sampleExactSurface(outsideX, outsideZ);
    REQUIRE_FALSE(outside.hydrology.waterfall);
    if (outside.waterSurface > SEA_LEVEL) {
        REQUIRE((outside.hydrology.lake || outside.hydrology.river || outside.hydrology.lakeBank));
    }
    Chunk outsideUpper(ChunkPos{Chunk::worldToChunk(outsideX), Chunk::worldToChunkY(lastFallingY),
                                Chunk::worldToChunk(outsideZ)});
    generator.generateCube(outsideUpper);
    const int outsideLocalX = Chunk::worldToLocal(outsideX);
    const int outsideLocalY = Chunk::worldToLocalY(lastFallingY);
    const int outsideLocalZ = Chunk::worldToLocal(outsideZ);
    if (outsideUpper.getBlock(outsideLocalX, outsideLocalY, outsideLocalZ) == BlockType::WATER) {
        REQUIRE_FALSE(
            outsideUpper.getFluidState(outsideLocalX, outsideLocalY, outsideLocalZ).isFalling());
    }
}

TEST_CASE("Bounded basin routes terminate at named outlets across every seam orientation",
          "[worldgen][advanced][hydrology][determinism]") {
    worldgen::BasinSolver solver(42);
    struct Probe {
        double x;
        double z;
        worldgen::BasinSample sample;
    };
    std::vector<Probe> forward;
    size_t sharedPortals = 0;
    size_t endorheicSinks = 0;
    size_t verticalPortals = 0;
    size_t horizontalPortals = 0;
    size_t negativePortals = 0;
    constexpr double EPSILON = 1.0e-4;

    for (int cellZ = -3; cellZ <= 3; ++cellZ) {
        for (int cellX = -3; cellX <= 3; ++cellX) {
            const double x = cellX * worldgen::BASIN_CATCHMENT_EDGE + 773.0;
            const double z = cellZ * worldgen::BASIN_CATCHMENT_EDGE + 911.0;
            const worldgen::BasinSample sample = solver.sample(
                x, z, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
            INFO("catchment " << cellX << ", " << cellZ);
            REQUIRE(sample.valid);
            REQUIRE(sample.outlet != worldgen::BasinOutlet::NONE);
            forward.push_back({x, z, sample});

            if (sample.outlet == worldgen::BasinOutlet::ENDORHEIC) {
                ++endorheicSinks;
                const double west = cellX * worldgen::BASIN_CATCHMENT_EDGE;
                const double north = cellZ * worldgen::BASIN_CATCHMENT_EDGE;
                REQUIRE(sample.outletX > west);
                REQUIRE(sample.outletX < west + worldgen::BASIN_CATCHMENT_EDGE);
                REQUIRE(sample.outletZ > north);
                REQUIRE(sample.outletZ < north + worldgen::BASIN_CATCHMENT_EDGE);
                continue;
            }

            REQUIRE(sample.outlet == worldgen::BasinOutlet::SHARED_PORTAL);
            ++sharedPortals;
            const double west = cellX * worldgen::BASIN_CATCHMENT_EDGE;
            const double east = west + worldgen::BASIN_CATCHMENT_EDGE;
            const double north = cellZ * worldgen::BASIN_CATCHMENT_EDGE;
            const double south = north + worldgen::BASIN_CATCHMENT_EDGE;
            const bool vertical = std::abs(sample.outletX - west) < EPSILON ||
                                  std::abs(sample.outletX - east) < EPSILON;
            const bool horizontal = std::abs(sample.outletZ - north) < EPSILON ||
                                    std::abs(sample.outletZ - south) < EPSILON;
            REQUIRE(vertical != horizontal);
            verticalPortals += vertical ? 1U : 0U;
            horizontalPortals += horizontal ? 1U : 0U;
            negativePortals += sample.outletX < 0.0 || sample.outletZ < 0.0 ? 1U : 0U;

            const worldgen::BasinSample first =
                solver.sample(sample.outletX - (vertical ? EPSILON : 0.0),
                              sample.outletZ - (horizontal ? EPSILON : 0.0), basinFixtureElevation,
                              basinFixtureRainfall, basinFixtureResistance);
            const worldgen::BasinSample second =
                solver.sample(sample.outletX + (vertical ? EPSILON : 0.0),
                              sample.outletZ + (horizontal ? EPSILON : 0.0), basinFixtureElevation,
                              basinFixtureRainfall, basinFixtureResistance);
            REQUIRE(first.outlet != worldgen::BasinOutlet::NONE);
            REQUIRE(second.outlet != worldgen::BasinOutlet::NONE);
            REQUIRE(first.surfaceElevation == Catch::Approx(second.surfaceElevation).margin(0.01));
            REQUIRE(first.waterSurface == Catch::Approx(second.waterSurface).margin(0.01));
            REQUIRE(first.discharge == Catch::Approx(second.discharge).margin(0.1));
            REQUIRE(first.sediment == Catch::Approx(second.sediment).margin(0.1));
            REQUIRE(first.channelWidth == Catch::Approx(second.channelWidth).margin(0.01));
            REQUIRE(first.erosionDepth == Catch::Approx(second.erosionDepth).margin(0.01));
            REQUIRE(first.streamOrder == second.streamOrder);
        }
    }

    REQUIRE(sharedPortals > 20);
    REQUIRE(endorheicSinks > 3);
    REQUIRE(verticalPortals > 5);
    REQUIRE(horizontalPortals > 5);
    REQUIRE(negativePortals > 5);
    const worldgen::BasinCacheMetrics warm = solver.cacheMetrics();
    REQUIRE(warm.failures == 0);
    REQUIRE(warm.bytes <= worldgen::BASIN_CACHE_BYTE_BUDGET);

    solver.clear();
    for (auto iterator = forward.rbegin(); iterator != forward.rend(); ++iterator) {
        const worldgen::BasinSample rebuilt =
            solver.sample(iterator->x, iterator->z, basinFixtureElevation, basinFixtureRainfall,
                          basinFixtureResistance);
        REQUIRE(rebuilt.outlet == iterator->sample.outlet);
        REQUIRE(rebuilt.outletX == iterator->sample.outletX);
        REQUIRE(rebuilt.outletZ == iterator->sample.outletZ);
        REQUIRE(rebuilt.surfaceElevation == iterator->sample.surfaceElevation);
        REQUIRE(rebuilt.waterSurface == iterator->sample.waterSurface);
        REQUIRE(rebuilt.discharge == iterator->sample.discharge);
        REQUIRE(rebuilt.sediment == iterator->sample.sediment);
    }
    REQUIRE(solver.cacheMetrics().failures == 0);

    worldgen::BasinSolver oceanSolver(42);
    const auto oceanElevation = [](double x, double z) {
        return 54.0 + std::sin(x / 300.0) + std::cos(z / 270.0);
    };
    const worldgen::BasinSample ocean = oceanSolver.sample(
        -3072.0, 1536.0, oceanElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(ocean.valid);
    REQUIRE(ocean.ocean);
    REQUIRE(ocean.outlet == worldgen::BasinOutlet::OCEAN);
    REQUIRE(oceanSolver.cacheMetrics().failures == 0);

    worldgen::BasinSolver coastSolver(42);
    const auto coastElevation = [](double x, double z) {
        return x >= worldgen::BASIN_CATCHMENT_EDGE
                   ? 52.0 + std::sin(z / 500.0)
                   : 96.0 + std::sin(x / 700.0) + std::cos(z / 650.0);
    };
    const worldgen::BasinSample coast = coastSolver.sample(
        773.0, 911.0, coastElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(coast.valid);
    REQUIRE(coast.outlet == worldgen::BasinOutlet::SHARED_PORTAL);
    REQUIRE(coast.outletX == worldgen::BASIN_CATCHMENT_EDGE);
    const bool verticalCoast = coast.outletX == worldgen::BASIN_CATCHMENT_EDGE;
    const worldgen::BasinSample landMouth =
        coastSolver.sample(coast.outletX - (verticalCoast ? EPSILON : 0.0),
                           coast.outletZ - (verticalCoast ? 0.0 : EPSILON), coastElevation,
                           basinFixtureRainfall, basinFixtureResistance);
    const worldgen::BasinSample oceanMouth =
        coastSolver.sample(coast.outletX + (verticalCoast ? EPSILON : 0.0),
                           coast.outletZ + (verticalCoast ? 0.0 : EPSILON), coastElevation,
                           basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(landMouth.waterSurface == Catch::Approx(SEA_LEVEL).margin(1.0e-6));
    REQUIRE(oceanMouth.waterSurface == Catch::Approx(SEA_LEVEL).margin(1.0e-6));
    REQUIRE(oceanMouth.ocean);
    REQUIRE(coastSolver.cacheMetrics().failures == 0);
}

TEST_CASE("Catchment guides bend without breaking their shared portal",
          "[worldgen][advanced][hydrology][meander][seam]") {
    worldgen::BasinSolver solver(42);
    const auto eastwardSlope = [](double x, double z) { return 300.0 - x * 0.015 + z * 0.00001; };
    const auto heavyRain = [](double, double, double) { return 1'000.0; };
    const auto resistantRock = [](double, double) { return 1.0; };
    constexpr int GRID_INTERVALS =
        static_cast<int>(worldgen::BASIN_CATCHMENT_EDGE / worldgen::BASIN_RASTER_SPACING);
    constexpr int GRID_EDGE = GRID_INTERVALS + 1;
    std::array<bool, GRID_EDGE * GRID_EDGE> channelCells{};
    size_t activeChannelCells = 0;

    for (int gridZ = 1; gridZ < GRID_INTERVALS; ++gridZ) {
        for (int gridX = 1; gridX < GRID_INTERVALS; ++gridX) {
            const worldgen::BasinSample sample = solver.sample(
                gridX * worldgen::BASIN_RASTER_SPACING, gridZ * worldgen::BASIN_RASTER_SPACING,
                eastwardSlope, heavyRain, resistantRock);
            const bool active =
                sample.streamOrder >= 2 && sample.river && sample.channelDistance <= 8.0;
            channelCells[static_cast<size_t>(gridZ * GRID_EDGE + gridX)] = active;
            activeChannelCells += active ? 1U : 0U;
        }
    }
    REQUIRE(activeChannelCells > 32);

    constexpr std::array<std::pair<int, int>, 4> DIRECTIONS{
        std::pair{1, 0},
        std::pair{0, 1},
        std::pair{1, 1},
        std::pair{1, -1},
    };
    for (const auto& [directionX, directionZ] : DIRECTIONS) {
        int longestRun = 0;
        for (int gridZ = 1; gridZ < GRID_INTERVALS; ++gridZ) {
            for (int gridX = 1; gridX < GRID_INTERVALS; ++gridX) {
                const auto activeAt = [&](int x, int z) {
                    return x > 0 && x < GRID_INTERVALS && z > 0 && z < GRID_INTERVALS &&
                           channelCells[static_cast<size_t>(z * GRID_EDGE + x)];
                };
                if (!activeAt(gridX, gridZ) || activeAt(gridX - directionX, gridZ - directionZ)) {
                    continue;
                }
                int run = 0;
                for (int x = gridX, z = gridZ; activeAt(x, z); x += directionX, z += directionZ) {
                    ++run;
                }
                longestRun = std::max(longestRun, run);
            }
        }
        INFO("direction " << directionX << ", " << directionZ);
        REQUIRE(longestRun <= 24);
    }

    const worldgen::BasinSample catchment =
        solver.sample(773.0, 911.0, eastwardSlope, heavyRain, resistantRock);
    REQUIRE(catchment.outlet == worldgen::BasinOutlet::SHARED_PORTAL);
    REQUIRE(catchment.outletX == worldgen::BASIN_CATCHMENT_EDGE);
    constexpr double PORTAL_EPSILON = 1.0e-4;
    const worldgen::BasinSample west =
        solver.sample(catchment.outletX - PORTAL_EPSILON, catchment.outletZ, eastwardSlope,
                      heavyRain, resistantRock);
    const worldgen::BasinSample east =
        solver.sample(catchment.outletX + PORTAL_EPSILON, catchment.outletZ, eastwardSlope,
                      heavyRain, resistantRock);
    REQUIRE(west.river);
    REQUIRE(east.river);
    REQUIRE(west.surfaceElevation == Catch::Approx(east.surfaceElevation).margin(0.001));
    REQUIRE(west.waterSurface == Catch::Approx(east.waterSurface).margin(0.001));
    REQUIRE(west.discharge == Catch::Approx(east.discharge).margin(0.1));
    REQUIRE(west.channelWidth == Catch::Approx(east.channelWidth).margin(0.001));
    REQUIRE(west.channelDepth == Catch::Approx(east.channelDepth).margin(0.001));
    REQUIRE(west.channelDistance == Catch::Approx(east.channelDistance).margin(1.0e-6));
    REQUIRE(west.flowX == Catch::Approx(east.flowX).margin(0.001));
    REQUIRE(west.flowZ == Catch::Approx(east.flowZ).margin(0.001));
    REQUIRE(west.streamOrder == east.streamOrder);

    double previousWater = std::numeric_limits<double>::infinity();
    for (double along = -24.0; along <= 48.0; along += 6.0) {
        const worldgen::BasinSample channel = solver.sample(
            catchment.outletX + west.flowX * along, catchment.outletZ + west.flowZ * along,
            eastwardSlope, heavyRain, resistantRock);
        INFO("portal guide offset " << along);
        REQUIRE(channel.river);
        REQUIRE(channel.waterSurface <= previousWater + 1.0e-4);
        previousWater = channel.waterSurface;
    }
    REQUIRE(solver.cacheMetrics().failures == 0);
}

TEST_CASE("Dry basin samples do not construct shoreline pages",
          "[worldgen][advanced][hydrology][cache][performance]") {
    worldgen::BasinSolver solver(42);
    const worldgen::BasinSample dry = solver.sample(773.0, 911.0, basinFixtureElevation,
                                                    basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(dry.valid);
    REQUIRE_FALSE(dry.lake);
    const worldgen::BasinCacheMetrics metrics = solver.cacheMetrics();
    REQUIRE(metrics.builds == 1);
    REQUIRE(metrics.entries == 1);
    REQUIRE(metrics.erosionEpochs == 4);
    REQUIRE(metrics.erosionReroutes == 4);
    REQUIRE(metrics.erosionReceiverChanges > 0);
    REQUIRE(metrics.failures == 0);
    REQUIRE(metrics.fallbackBuilds == 0);
    REQUIRE(metrics.shorelineBuilds == 0);
    REQUIRE(metrics.shorelineEntries == 0);
    REQUIRE(metrics.shorelineBytes == 0);
}

TEST_CASE("Normalized lake fringes rebuild signed shoreline authority",
          "[worldgen][advanced][hydrology][shoreline][regression][determinism]") {
    ChunkGenerator generator(42);
    constexpr ColumnPos dryPosition{-26'336, 30'496};
    constexpr std::array<ColumnPos, 3> wetNeighbors{{
        {-26'352, 30'496},
        {-26'336, 30'512},
        {-26'320, 30'496},
    }};

    const worldgen::HydrologySample dry =
        generator.sampleGeneratedWaterAuthority(dryPosition.x, dryPosition.z);
    REQUIRE_FALSE(dry.lake);
    REQUIRE(dry.waterBodyId == worldgen::NO_WATER_BODY);
    REQUIRE(dry.lakeShoreDistance < 0.0);
    REQUIRE(dry.shoreWaterSurface > 0.0);

    worldgen::WaterBodyId body = worldgen::NO_WATER_BODY;
    for (const ColumnPos position : wetNeighbors) {
        const worldgen::HydrologySample wet =
            generator.sampleGeneratedWaterAuthority(position.x, position.z);
        REQUIRE(wet.lake);
        REQUIRE(wet.lakeShoreDistance > 0.0);
        REQUIRE(wet.waterBodyId != worldgen::NO_WATER_BODY);
        if (body == worldgen::NO_WATER_BODY)
            body = wet.waterBodyId;
        REQUIRE(wet.waterBodyId == body);
        REQUIRE(wet.shoreWaterSurface == dry.shoreWaterSurface);
    }

    worldgen::BasinCacheMetrics metrics = generator.basinCacheMetrics();
    REQUIRE(metrics.failures == 0);
    REQUIRE(metrics.fallbackBuilds == 0);

    generator.clearMacroCaches();
    const worldgen::HydrologySample rebuiltDry =
        generator.sampleGeneratedWaterAuthority(dryPosition.x, dryPosition.z);
    const worldgen::HydrologySample rebuiltWet =
        generator.sampleGeneratedWaterAuthority(wetNeighbors.front().x, wetNeighbors.front().z);
    REQUIRE_FALSE(rebuiltDry.lake);
    REQUIRE(rebuiltDry.lakeShoreDistance == dry.lakeShoreDistance);
    REQUIRE(rebuiltDry.shoreWaterSurface == dry.shoreWaterSurface);
    REQUIRE(rebuiltWet.lake);
    REQUIRE(rebuiltWet.waterBodyId == body);
    metrics = generator.basinCacheMetrics();
    REQUIRE(metrics.failures == 0);
    REQUIRE(metrics.fallbackBuilds == 0);
}

TEST_CASE("Thread-local hydrology hits invalidate after clear and eviction",
          "[worldgen][advanced][hydrology][cache][concurrency][determinism]") {
    worldgen::MacroGenerationSampler macro(42);
    std::mutex mutex;
    std::condition_variable changed;
    bool firstComplete = false;
    bool rebuildAllowed = false;
    auto worker = std::async(std::launch::async, [&] {
        const worldgen::HydrologySample first = macro.sampleHydrology(-8'235.0, 2'976.0);
        {
            std::lock_guard lock(mutex);
            firstComplete = true;
        }
        changed.notify_all();
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return rebuildAllowed; });
        }
        return std::pair{first, macro.sampleHydrology(-8'235.0, 2'976.0)};
    });

    {
        std::unique_lock lock(mutex);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds(10), [&] { return firstComplete; }));
    }
    const worldgen::BasinCacheMetrics beforeClear = macro.basinCacheMetrics();
    REQUIRE(beforeClear.builds == 2);
    REQUIRE(beforeClear.shorelineBuilds == 1);
    macro.clearBasinCache();
    {
        std::lock_guard lock(mutex);
        rebuildAllowed = true;
    }
    changed.notify_all();
    const auto [first, rebuilt] = worker.get();
    REQUIRE(rebuilt.waterBodyId == first.waterBodyId);
    REQUIRE(rebuilt.surfaceElevation == first.surfaceElevation);
    REQUIRE(rebuilt.waterSurface == first.waterSurface);
    REQUIRE(rebuilt.lakeShoreDistance == first.lakeShoreDistance);
    const worldgen::BasinCacheMetrics afterClear = macro.basinCacheMetrics();
    REQUIRE(afterClear.builds == 4);
    REQUIRE(afterClear.shorelineBuilds == 2);

    worldgen::BasinSolver evictionSolver(42, 1);
    const worldgen::BasinSample original = evictionSolver.sample(
        773.0, 911.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    const worldgen::BasinSample other =
        evictionSolver.sample(worldgen::BASIN_CATCHMENT_EDGE + 773.0, 911.0, basinFixtureElevation,
                              basinFixtureRainfall, basinFixtureResistance);
    const worldgen::BasinSample afterEviction = evictionSolver.sample(
        773.0, 911.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(original.valid);
    REQUIRE(other.valid);
    REQUIRE(afterEviction.surfaceElevation == original.surfaceElevation);
    REQUIRE(afterEviction.waterSurface == original.waterSurface);
    REQUIRE(evictionSolver.cacheMetrics().builds == 3);
    REQUIRE(evictionSolver.cacheMetrics().entries == 1);
}

TEST_CASE("A clear overlapping a cold build cannot publish a current fast hit",
          "[worldgen][advanced][hydrology][cache][concurrency][regression]") {
    worldgen::BasinSolver solver(42);
    std::mutex mutex;
    std::condition_variable changed;
    bool buildEntered = false;
    bool releaseBuild = false;
    bool firstComplete = false;
    bool secondAllowed = false;
    std::atomic<bool> blocked{false};
    const auto elevation = [&](double x, double z) {
        if (!blocked.exchange(true, std::memory_order_relaxed)) {
            std::unique_lock lock(mutex);
            buildEntered = true;
            changed.notify_all();
            changed.wait(lock, [&] { return releaseBuild; });
        }
        return basinFixtureElevation(x, z);
    };

    auto worker = std::async(std::launch::async, [&] {
        const worldgen::BasinSample first =
            solver.sample(773.0, 911.0, elevation, basinFixtureRainfall, basinFixtureResistance);
        {
            std::lock_guard lock(mutex);
            firstComplete = true;
        }
        changed.notify_all();
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return secondAllowed; });
        }
        const worldgen::BasinSample second =
            solver.sample(773.0, 911.0, elevation, basinFixtureRainfall, basinFixtureResistance);
        return std::pair{first, second};
    });

    {
        std::unique_lock lock(mutex);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds(10), [&] { return buildEntered; }));
    }
    solver.clear();
    {
        std::lock_guard lock(mutex);
        releaseBuild = true;
    }
    changed.notify_all();
    {
        std::unique_lock lock(mutex);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds(10), [&] { return firstComplete; }));
    }
    REQUIRE(solver.cacheMetrics().builds == 0);
    REQUIRE(solver.cacheMetrics().entries == 0);
    {
        std::lock_guard lock(mutex);
        secondAllowed = true;
    }
    changed.notify_all();
    REQUIRE(worker.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    const auto [first, rebuilt] = worker.get();
    requireBasinSamplesEqual(rebuilt, first);
    REQUIRE(solver.cacheMetrics().builds == 1);
    REQUIRE(solver.cacheMetrics().entries == 1);
    REQUIRE(solver.cacheMetrics().failures == 0);
}

TEST_CASE("Scalar and grid sampling retain basins through concurrent one byte eviction",
          "[worldgen][advanced][hydrology][cache][concurrency][regression]") {
    constexpr int GRID_EDGE = 5;
    constexpr int64_t GRID_X = 4'840;
    constexpr int64_t GRID_Z = 696;
    std::array<worldgen::BasinSample, GRID_EDGE * GRID_EDGE> expectedGrid{};
    worldgen::BasinSolver reference(42);
    const worldgen::BasinSample expectedLake = reference.sample(
        4'848.0, 704.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    const worldgen::BasinSample expectedDry = reference.sample(
        773.0, 911.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    reference.sampleGrid(GRID_X, GRID_Z, 4, 4, GRID_EDGE, GRID_EDGE, basinFixtureElevation,
                         basinFixtureRainfall, basinFixtureResistance, expectedGrid);

    worldgen::BasinSolver solver(42, 1);
    std::mutex mutex;
    std::condition_variable changed;
    bool buildEntered = false;
    bool releaseBuild = false;
    std::atomic<bool> blocked{false};
    std::atomic<bool> keepClearing{true};
    const auto elevation = [&](double x, double z) {
        if (!blocked.exchange(true, std::memory_order_relaxed)) {
            std::unique_lock lock(mutex);
            buildEntered = true;
            changed.notify_all();
            changed.wait(lock, [&] { return releaseBuild; });
        }
        return basinFixtureElevation(x, z);
    };
    const auto exercise = [&] {
        for (int iteration = 0; iteration < 3; ++iteration) {
            const worldgen::BasinSample lake = solver.sample(
                4'848.0, 704.0, elevation, basinFixtureRainfall, basinFixtureResistance);
            const worldgen::BasinSample dry = solver.sample(
                773.0, 911.0, elevation, basinFixtureRainfall, basinFixtureResistance);
            if (!basinSamplesExactlyEqual(lake, expectedLake) ||
                !basinSamplesExactlyEqual(dry, expectedDry)) {
                return false;
            }
            std::array<worldgen::BasinSample, GRID_EDGE * GRID_EDGE> actualGrid{};
            solver.sampleGrid(GRID_X, GRID_Z, 4, 4, GRID_EDGE, GRID_EDGE, elevation,
                              basinFixtureRainfall, basinFixtureResistance, actualGrid);
            for (size_t index = 0; index < actualGrid.size(); ++index) {
                if (!basinSamplesExactlyEqual(actualGrid[index], expectedGrid[index]))
                    return false;
            }
        }
        return true;
    };

    std::array<std::future<bool>, 2> workers = {
        std::async(std::launch::async, exercise),
        std::async(std::launch::async, exercise),
    };
    {
        std::unique_lock lock(mutex);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds(10), [&] { return buildEntered; }));
    }
    solver.clear();
    auto clearer = std::async(std::launch::async, [&] {
        size_t clears = 0;
        while (keepClearing.load(std::memory_order_acquire)) {
            solver.clear();
            ++clears;
            std::this_thread::yield();
        }
        return clears;
    });
    {
        std::lock_guard lock(mutex);
        releaseBuild = true;
    }
    changed.notify_all();

    bool deterministic = true;
    for (auto& worker : workers) {
        REQUIRE(worker.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        deterministic = worker.get() && deterministic;
    }
    keepClearing.store(false, std::memory_order_release);
    REQUIRE(clearer.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    const size_t clears = clearer.get();
    CAPTURE(clears, solver.cacheMetrics().builds, solver.cacheMetrics().entries,
            solver.cacheMetrics().failures, solver.cacheMetrics().shorelineFailures);
    REQUIRE(clears > 0);
    REQUIRE(deterministic);
    REQUIRE(solver.cacheMetrics().failures == 0);
    REQUIRE(solver.cacheMetrics().shorelineFailures == 0);
}

TEST_CASE("Cold basin construction admits only two producers across solver instances",
          "[worldgen][advanced][hydrology][concurrency]") {
    std::array<std::unique_ptr<worldgen::BasinSolver>, 4> solvers;
    for (auto& solver : solvers)
        solver = std::make_unique<worldgen::BasinSolver>(42);
    const worldgen::BasinCacheMetrics before = solvers.front()->cacheMetrics();
    std::mutex probeMutex;
    std::condition_variable probeChanged;
    size_t readyRequests = 0;
    size_t enteredBuilds = 0;
    bool startRequests = false;
    bool releaseBuilds = false;

    const auto blockedElevation = [&](double x, double z) {
        static thread_local bool announced = false;
        if (!announced) {
            announced = true;
            std::unique_lock lock(probeMutex);
            ++enteredBuilds;
            probeChanged.notify_all();
            probeChanged.wait(lock, [&releaseBuilds] { return releaseBuilds; });
        }
        return basinFixtureElevation(x, z);
    };

    std::array<std::future<worldgen::BasinSample>, 4> requests;
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index] = std::async(std::launch::async, [&, index] {
            {
                std::unique_lock lock(probeMutex);
                ++readyRequests;
                probeChanged.notify_all();
                probeChanged.wait(lock, [&startRequests] { return startRequests; });
            }
            const double x = static_cast<double>(index) * worldgen::BASIN_CATCHMENT_EDGE + 773.0;
            return solvers[index]->sample(x, 911.0, blockedElevation, basinFixtureRainfall,
                                          basinFixtureResistance);
        });
    }

    bool allRequestsReady = false;
    {
        std::unique_lock lock(probeMutex);
        allRequestsReady = probeChanged.wait_for(lock, std::chrono::seconds(10),
                                                 [&] { return readyRequests == requests.size(); });
        startRequests = true;
    }
    probeChanged.notify_all();

    bool admittedFirstPair = false;
    bool admittedThirdBuild = false;
    {
        std::unique_lock lock(probeMutex);
        admittedFirstPair = probeChanged.wait_for(lock, std::chrono::seconds(10),
                                                  [&enteredBuilds] { return enteredBuilds >= 2; });
        admittedThirdBuild = probeChanged.wait_for(lock, std::chrono::milliseconds(250),
                                                   [&enteredBuilds] { return enteredBuilds > 2; });
        releaseBuilds = true;
    }
    probeChanged.notify_all();

    std::array<worldgen::BasinSample, 4> concurrentSamples;
    for (size_t index = 0; index < requests.size(); ++index) {
        concurrentSamples[index] = requests[index].get();
        REQUIRE(concurrentSamples[index].valid);
    }

    size_t completedBuilds = 0;
    size_t failedBuilds = 0;
    for (const auto& solver : solvers) {
        const worldgen::BasinCacheMetrics metrics = solver->cacheMetrics();
        completedBuilds += metrics.builds;
        failedBuilds += metrics.failures;
    }
    const worldgen::BasinCacheMetrics after = solvers.front()->cacheMetrics();
    REQUIRE(allRequestsReady);
    REQUIRE(admittedFirstPair);
    REQUIRE_FALSE(admittedThirdBuild);
    REQUIRE(completedBuilds == requests.size());
    REQUIRE(failedBuilds == 0);
    REQUIRE(after.activeColdBuilds == 0);
    REQUIRE(after.peakColdBuilds == worldgen::MAX_CONCURRENT_COLD_BASIN_BUILDS);
    REQUIRE(after.throttledBuilds >= before.throttledBuilds + 2);

    for (size_t remaining = concurrentSamples.size(); remaining > 0; --remaining) {
        const size_t index = remaining - 1;
        solvers[index]->clear();
        const double x = static_cast<double>(index) * worldgen::BASIN_CATCHMENT_EDGE + 773.0;
        const worldgen::BasinSample rebuilt = solvers[index]->sample(
            x, 911.0, blockedElevation, basinFixtureRainfall, basinFixtureResistance);
        REQUIRE(rebuilt.surfaceElevation == concurrentSamples[index].surfaceElevation);
        REQUIRE(rebuilt.waterSurface == concurrentSamples[index].waterSurface);
        REQUIRE(rebuilt.discharge == concurrentSamples[index].discharge);
        REQUIRE(rebuilt.outlet == concurrentSamples[index].outlet);
        REQUIRE(rebuilt.outletX == concurrentSamples[index].outletX);
        REQUIRE(rebuilt.outletZ == concurrentSamples[index].outletZ);
    }
}

TEST_CASE("Catchment routing uses a strict drainage order without flat-land cycles",
          "[worldgen][advanced][hydrology][acyclic]") {
    constexpr std::array<std::pair<ColumnPos, double>, 4> elevations{
        std::pair{ColumnPos{-183, -200}, 100.000},
        std::pair{ColumnPos{-183, -199}, 100.006},
        std::pair{ColumnPos{-182, -199}, 100.012},
        std::pair{ColumnPos{-182, -200}, 100.018},
    };
    auto elevation = [&](double x, double z) {
        const ColumnPos cell{
            static_cast<int64_t>(std::floor(x / worldgen::BASIN_CATCHMENT_EDGE)),
            static_cast<int64_t>(std::floor(z / worldgen::BASIN_CATCHMENT_EDGE)),
        };
        for (const auto& [candidate, height] : elevations) {
            if (candidate == cell)
                return height;
        }
        return 1000.0;
    };
    const auto rainfall = [](double, double, double) { return 1.0; };
    const auto resistance = [](double, double) { return 0.5; };
    worldgen::BasinSolver solver(42);

    for (const auto& [start, ignoredHeight] : elevations) {
        (void)ignoredHeight;
        ColumnPos cell = start;
        std::vector<ColumnPos> visited;
        bool terminated = false;
        for (int step = 0; step < 16; ++step) {
            REQUIRE(std::find(visited.begin(), visited.end(), cell) == visited.end());
            visited.push_back(cell);
            const double x = static_cast<double>(cell.x) * worldgen::BASIN_CATCHMENT_EDGE + 773.0;
            const double z = static_cast<double>(cell.z) * worldgen::BASIN_CATCHMENT_EDGE + 911.0;
            const worldgen::BasinSample sample =
                solver.sample(x, z, elevation, rainfall, resistance);
            REQUIRE(sample.valid);
            if (sample.outlet == worldgen::BasinOutlet::OCEAN ||
                sample.outlet == worldgen::BasinOutlet::ENDORHEIC) {
                terminated = true;
                break;
            }
            REQUIRE(sample.outlet == worldgen::BasinOutlet::SHARED_PORTAL);
            const double west = static_cast<double>(cell.x) * worldgen::BASIN_CATCHMENT_EDGE;
            const double east = west + worldgen::BASIN_CATCHMENT_EDGE;
            const double north = static_cast<double>(cell.z) * worldgen::BASIN_CATCHMENT_EDGE;
            const double south = north + worldgen::BASIN_CATCHMENT_EDGE;
            if (std::abs(sample.outletX - west) < 1.0e-6) {
                --cell.x;
            } else if (std::abs(sample.outletX - east) < 1.0e-6) {
                ++cell.x;
            } else if (std::abs(sample.outletZ - north) < 1.0e-6) {
                --cell.z;
            } else {
                REQUIRE(std::abs(sample.outletZ - south) < 1.0e-6);
                ++cell.z;
            }
        }
        REQUIRE(terminated);
    }
}

TEST_CASE("Sediment-rich rivers form deterministic deltas at terminal lake surfaces",
          "[worldgen][advanced][hydrology][delta][lake]") {
    worldgen::BasinSolver solver(42);
    const worldgen::BasinSample delta = solver.sample(4848.0, 704.0, basinFixtureElevation,
                                                      basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(delta.valid);
    REQUIRE(delta.delta);
    REQUIRE(delta.lake);
    REQUIRE(delta.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(delta.endorheic);
    REQUIRE_FALSE(delta.ocean);
    REQUIRE(delta.outlet == worldgen::BasinOutlet::ENDORHEIC);
    REQUIRE(delta.distributaryCount >= 2);
    REQUIRE(delta.distributaryCount <= 4);
    REQUIRE(delta.waterSurface > delta.surfaceElevation);
    REQUIRE(delta.sediment > 0.0);

    const worldgen::BasinSample sameBody = solver.sample(
        4840.0, 712.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(sameBody.lake);
    REQUIRE(sameBody.waterBodyId == delta.waterBodyId);

    solver.clear();
    const worldgen::BasinSample rebuilt = solver.sample(
        4848.0, 704.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(rebuilt.delta);
    REQUIRE(rebuilt.lake);
    REQUIRE(rebuilt.waterBodyId == delta.waterBodyId);
    REQUIRE(rebuilt.waterSurface == delta.waterSurface);
    REQUIRE(rebuilt.surfaceElevation == delta.surfaceElevation);
    REQUIRE(rebuilt.distributaryCount == delta.distributaryCount);
    REQUIRE(solver.cacheMetrics().failures == 0);
}

TEST_CASE("Sparse basin batches exactly match canonical shoreline pages",
          "[worldgen][advanced][hydrology][bulk][shoreline][determinism]") {
    constexpr int SAMPLE_EDGE = 17;
    constexpr int SPACING = 4;
    constexpr int64_t ORIGIN_X = 4'800;
    constexpr int64_t ORIGIN_Z = 656;
    std::array<worldgen::BasinSample, SAMPLE_EDGE * SAMPLE_EDGE> batched{};

    worldgen::BasinSolver batchSolver(42);
    batchSolver.sampleGrid(ORIGIN_X, ORIGIN_Z, SPACING, SPACING, SAMPLE_EDGE, SAMPLE_EDGE,
                           basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance,
                           batched);
    const worldgen::BasinCacheMetrics batchMetrics = batchSolver.cacheMetrics();
    REQUIRE(batchMetrics.failures == 0);
    REQUIRE(batchMetrics.shorelineBuilds == 0);
    REQUIRE(batchMetrics.shorelineEntries == 0);

    worldgen::BasinSolver pointSolver(42);
    bool foundLake = false;
    bool foundDelta = false;
    bool foundShorelineBand = false;
    for (int sampleZ = 0; sampleZ < SAMPLE_EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < SAMPLE_EDGE; ++sampleX) {
            const int64_t x = ORIGIN_X + static_cast<int64_t>(sampleX) * SPACING;
            const int64_t z = ORIGIN_Z + static_cast<int64_t>(sampleZ) * SPACING;
            const worldgen::BasinSample canonical = pointSolver.sample(
                static_cast<double>(x), static_cast<double>(z), basinFixtureElevation,
                basinFixtureRainfall, basinFixtureResistance);
            CAPTURE(x, z);
            INFO("batch body="
                 << batched[static_cast<size_t>(sampleZ * SAMPLE_EDGE + sampleX)].waterBodyId
                 << " endorheic="
                 << batched[static_cast<size_t>(sampleZ * SAMPLE_EDGE + sampleX)].endorheic
                 << " point body=" << canonical.waterBodyId
                 << " endorheic=" << canonical.endorheic);
            requireBasinSamplesEqual(batched[static_cast<size_t>(sampleZ * SAMPLE_EDGE + sampleX)],
                                     canonical);
            foundLake = foundLake || canonical.lake;
            foundDelta = foundDelta || canonical.delta;
            foundShorelineBand = foundShorelineBand || std::abs(canonical.lakeShoreDistance) <= 8.0;
        }
    }
    REQUIRE(foundLake);
    REQUIRE(foundDelta);
    REQUIRE(foundShorelineBand);
    REQUIRE(pointSolver.cacheMetrics().shorelineBuilds > 0);
}

TEST_CASE("Fractional basin point batches preserve scalar samples in every order",
          "[worldgen][advanced][hydrology][bulk][fractional][determinism]") {
    constexpr std::array<worldgen::BasinSamplePosition, 6> POSITIONS = {
        worldgen::BasinSamplePosition{4'800.25, 656.75},
        worldgen::BasinSamplePosition{4'848.5, 704.25},
        worldgen::BasinSamplePosition{4'863.75, 711.125},
        worldgen::BasinSamplePosition{-0.25, 2'047.75},
        worldgen::BasinSamplePosition{-2'048.125, -0.375},
        worldgen::BasinSamplePosition{773.5, 911.25},
    };

    worldgen::BasinSolver scalarSolver(42);
    std::array<worldgen::BasinSample, POSITIONS.size()> expected{};
    for (size_t index = 0; index < POSITIONS.size(); ++index) {
        expected[index] =
            scalarSolver.sample(POSITIONS[index].x, POSITIONS[index].z, basinFixtureElevation,
                                basinFixtureRainfall, basinFixtureResistance);
    }

    worldgen::BasinSolver batchSolver(42);
    std::array<worldgen::BasinSample, POSITIONS.size()> forward{};
    batchSolver.samplePoints(POSITIONS, basinFixtureElevation, basinFixtureRainfall,
                             basinFixtureResistance, forward);
    for (size_t index = 0; index < POSITIONS.size(); ++index) {
        CAPTURE(index, POSITIONS[index].x, POSITIONS[index].z);
        requireBasinSamplesEqual(forward[index], expected[index]);
    }
    REQUIRE(batchSolver.cacheMetrics().scalarSampleCalls == 0);
    REQUIRE(batchSolver.cacheMetrics().failures == 0);

    std::array<worldgen::BasinSamplePosition, POSITIONS.size()> reversePositions = POSITIONS;
    std::ranges::reverse(reversePositions);
    std::array<worldgen::BasinSample, POSITIONS.size()> reverse{};
    batchSolver.clear();
    batchSolver.samplePoints(reversePositions, basinFixtureElevation, basinFixtureRainfall,
                             basinFixtureResistance, reverse);
    for (size_t index = 0; index < POSITIONS.size(); ++index) {
        requireBasinSamplesEqual(reverse[index], expected[POSITIONS.size() - index - 1]);
    }
}

TEST_CASE("Final climate applies the documented lapse rate and bounded water balance",
          "[worldgen][advanced][climate]") {
    worldgen::MacroGenerationSampler sampler(42);
    const worldgen::ClimateFields seaLevel = sampler.sampleClimate(3264.0, 480.0, SEA_LEVEL);
    const worldgen::ClimateFields elevated = sampler.sampleClimate(3264.0, 480.0, 189.0);

    REQUIRE(seaLevel.temperatureC - elevated.temperatureC == Catch::Approx(6.5).margin(1.0e-10));
    REQUIRE(seaLevel.annualPrecipitationMm >= 60.0);
    REQUIRE(seaLevel.annualPrecipitationMm <= 3600.0);
    REQUIRE(seaLevel.potentialEvapotranspirationMm >= 120.0);
    REQUIRE(seaLevel.potentialEvapotranspirationMm <= 1800.0);
    REQUIRE(seaLevel.relativeHumidity >= 0.0);
    REQUIRE(seaLevel.relativeHumidity <= 1.0);
    REQUIRE(seaLevel.aridity ==
            Catch::Approx(seaLevel.potentialEvapotranspirationMm / seaLevel.annualPrecipitationMm));
    const double windSpeed = std::hypot(seaLevel.wind.x, seaLevel.wind.z);
    REQUIRE(windSpeed >= 0.45);
    REQUIRE(windSpeed <= 1.0);
}

TEST_CASE("Every appended climate biome is a reachable primary suitability",
          "[worldgen][advanced][biome]") {
    worldgen::MacroGenerationSampler sampler(42);
    constexpr std::array<Biome, 19> advancedBiomes = {
        Biome::SAVANNA,
        Biome::TROPICAL_RAINFOREST,
        Biome::TEMPERATE_RAINFOREST,
        Biome::SHRUBLAND,
        Biome::STEPPE,
        Biome::COLD_DESERT,
        Biome::BADLANDS,
        Biome::TUNDRA,
        Biome::ALPINE,
        Biome::MANGROVE,
        Biome::FROZEN_OCEAN,
        Biome::VOLCANIC_BARREN,
        Biome::GLACIER,
        Biome::MONTANE_GRASSLAND,
        Biome::FLOODED_GRASSLAND,
        Biome::MEDITERRANEAN_WOODLAND,
        Biome::TEMPERATE_CONIFER_FOREST,
        Biome::TROPICAL_CONIFER_FOREST,
        Biome::TROPICAL_DRY_FOREST,
    };

    for (Biome biome : advancedBiomes) {
        INFO("biome " << static_cast<int>(biome));
        const worldgen::BiomeBlend blend = selectSyntheticBiome(sampler, biome);
        REQUIRE(blend.primary == biome);
        REQUIRE(blend.transition >= 0.0);
        REQUIRE(blend.transition <= 0.5);
    }
}

TEST_CASE("Every One Earth terrestrial class has a reachable biome representative",
          "[worldgen][advanced][biome][one-earth][reachability]") {
    worldgen::MacroGenerationSampler sampler(42);
    constexpr std::array<Biome, 14> representatives = {
        Biome::DESERT,
        Biome::MONTANE_GRASSLAND,
        Biome::STEPPE,
        Biome::SAVANNA,
        Biome::FLOODED_GRASSLAND,
        Biome::MANGROVE,
        Biome::MEDITERRANEAN_WOODLAND,
        Biome::FOREST,
        Biome::TEMPERATE_CONIFER_FOREST,
        Biome::TROPICAL_CONIFER_FOREST,
        Biome::TROPICAL_DRY_FOREST,
        Biome::TROPICAL_RAINFOREST,
        Biome::TAIGA,
        Biome::TUNDRA,
    };

    for (const Biome biome : representatives) {
        INFO("One Earth biome " << static_cast<int>(biome));
        REQUIRE(selectSyntheticBiome(sampler, biome).primary == biome);
    }
}

TEST_CASE("Appended One Earth classes occur in coordinate-generated climate fields",
          "[worldgen][advanced][biome][one-earth][natural-reachability]") {
    struct BiomeLandmark {
        Biome biome;
        int64_t x;
        int64_t z;
    };
    constexpr std::array<BiomeLandmark, 6> landmarks = {
        BiomeLandmark{Biome::MONTANE_GRASSLAND, 27'037, -129},
        BiomeLandmark{Biome::FLOODED_GRASSLAND, -9'003, 21'417},
        BiomeLandmark{Biome::MEDITERRANEAN_WOODLAND, -4'686, 22'170},
        BiomeLandmark{Biome::TEMPERATE_CONIFER_FOREST, -27'971, 29'064},
        BiomeLandmark{Biome::TROPICAL_CONIFER_FOREST, 5'845, 8'750},
        BiomeLandmark{Biome::TROPICAL_DRY_FOREST, 23'552, 51'200},
    };

    ChunkGenerator generator(42);
    for (const BiomeLandmark& landmark : landmarks) {
        const worldgen::SurfaceSample surface = generator.sampleFarSurface(landmark.x, landmark.z);
        INFO("natural biome " << static_cast<int>(landmark.biome) << " at " << landmark.x << ','
                              << landmark.z);
        REQUIRE(worldgen::biomeBlendWeight(surface.biome, landmark.biome) >= 0.15);
    }
}

TEST_CASE("Far surface samples exactly match aligned cubic plan samples",
          "[worldgen][advanced][lod][determinism]") {
    ChunkGenerator generator(42);
    constexpr std::array<ColumnPos, 7> positions = {
        ColumnPos{0, 0},
        ColumnPos{8, 16},
        ColumnPos{-8, -16},
        ColumnPos{-24, 40},
        ColumnPos{2048, -4096},
        ColumnPos{-10'064, -23'056},
        ColumnPos{-14'200, 27'192},
    };

    for (ColumnPos position : positions) {
        INFO("aligned surface at " << position.x << ", " << position.z);
        REQUIRE(world_coord::floorMod(position.x, 8) == 0);
        REQUIRE(world_coord::floorMod(position.z, 8) == 0);
        requireExactSurface(generator.sampleFarSurface(position.x, position.z),
                            generator.sampleSurface(position.x, position.z));
    }
}

TEST_CASE("Golden cubes match between one worker and four cold workers",
          "[worldgen][advanced][determinism][concurrency]") {
    const std::vector<ChunkPos> route = {
        {0, 4, 0},    {1, 4, 0},   {-1, 4, -1},   {16, 5, -8},
        {-24, 2, 19}, {64, 8, 64}, {-96, -2, 48}, {128, 12, -128},
    };
    ChunkGenerator generator(42);
    generator.clearMacroCaches();
    const CubeHashes sequential = generateHashes(generator, route);

    uint64_t routeHash = 0;
    for (const auto& [position, hash] : sequential)
        routeHash = hash64(routeHash ^ hash);
    REQUIRE(routeHash == 0xc1a9be6c13ab0193ULL);

    generator.clearMacroCaches();
    std::array<std::future<CubeHashes>, 4> workers;
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::async(std::launch::async, [&, worker] {
            std::vector<ChunkPos> assigned;
            for (size_t index = worker; index < route.size(); index += workers.size()) {
                assigned.push_back(route[route.size() - 1 - index]);
            }
            return generateHashes(generator, assigned);
        });
    }

    CubeHashes concurrent;
    for (auto& worker : workers) {
        CubeHashes partial = worker.get();
        concurrent.insert(partial.begin(), partial.end());
    }
    REQUIRE(concurrent == sequential);
}

TEST_CASE("A stitched two by two by two cube volume matches direct world sampling",
          "[worldgen][advanced][cubic][determinism]") {
    ChunkGenerator generator(42);
    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> stitched;
    for (int32_t chunkY : {3, 4}) {
        for (int64_t chunkZ : {-1, 0}) {
            for (int64_t chunkX : {-1, 0}) {
                const ChunkPos position{chunkX, chunkY, chunkZ};
                auto cube = std::make_shared<Chunk>(position);
                generator.generateCube(*cube);
                stitched.emplace(position, std::move(cube));
            }
        }
    }

    World world(42);
    for (const auto& [position, cube] : stitched)
        world.getChunk(position);
    size_t compared = 0;
    bool allLoaded = true;
    bool blocksMatch = true;
    bool fluidsMatch = true;
    for (int32_t y = 3 * CHUNK_EDGE; y < 5 * CHUNK_EDGE; ++y) {
        for (int64_t z = -CHUNK_EDGE; z < CHUNK_EDGE; ++z) {
            for (int64_t x = -CHUNK_EDGE; x < CHUNK_EDGE; ++x) {
                const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                        Chunk::worldToChunk(z)};
                const std::shared_ptr<Chunk>& expected = stitched.at(position);
                const int localX = Chunk::worldToLocal(x);
                const int localY = Chunk::worldToLocalY(y);
                const int localZ = Chunk::worldToLocal(z);
                blocksMatch = blocksMatch && world.getBlockIfLoaded(x, y, z) ==
                                                 expected->getBlock(localX, localY, localZ);
                const FluidCell fluid = world.readFluidCell({x, y, z});
                allLoaded = allLoaded && fluid.loaded;
                fluidsMatch =
                    fluidsMatch && fluid.state == expected->getFluidState(localX, localY, localZ);
                ++compared;
            }
        }
    }
    REQUIRE(compared == 8U * CHUNK_VOLUME);
    REQUIRE(allLoaded);
    REQUIRE(blocksMatch);
    REQUIRE(fluidsMatch);
}

TEST_CASE("Globally anchored trees cross a cubic face without order seams",
          "[worldgen][advanced][flora][determinism]") {
    // A birch canopy rooted at (-27282, 74, -17086) crosses this X face.
    constexpr ChunkPos FIRST{-1706, 5, -1068};
    constexpr ChunkPos SECOND{-1705, 5, -1068};
    ChunkGenerator forward(42);
    Chunk forwardFirst(FIRST);
    Chunk forwardSecond(SECOND);
    forward.generateCube(forwardFirst);
    forward.generateCube(forwardSecond);

    const auto countBirchLeaves = [](const Chunk& cube) {
        const std::vector<BlockType> blocks = cube.copyBlocks();
        return std::count(blocks.begin(), blocks.end(), BlockType::BIRCH_LEAVES);
    };
    REQUIRE(countBirchLeaves(forwardFirst) > 0);
    REQUIRE(countBirchLeaves(forwardSecond) > 0);

    ChunkGenerator reverse(42);
    Chunk reverseSecond(SECOND);
    Chunk reverseFirst(FIRST);
    reverse.generateCube(reverseSecond);
    reverse.generateCube(reverseFirst);
    REQUIRE(reverseFirst.copyBlocks() == forwardFirst.copyBlocks());
    REQUIRE(reverseSecond.copyBlocks() == forwardSecond.copyBlocks());
    REQUIRE(reverseFirst.explicitFluidStates() == forwardFirst.explicitFluidStates());
    REQUIRE(reverseSecond.explicitFluidStates() == forwardSecond.explicitFluidStates());
}

TEST_CASE("Batched sparse final surfaces match scalar roots after cache eviction",
          "[worldgen][advanced][surface][batch][volcano][determinism][performance]") {
    constexpr std::array<ColumnPos, 8> POSITIONS = {{
        {-161, 17},
        {-64, -65},
        {-1, -1},
        {0, 0},
        {257, -513},
        {23'029, -111'486},
        {-518'872, -384'992},
        {8'193, -8'193},
    }};
    ChunkGenerator batched(764891);
    std::array<worldgen::SurfaceSample, POSITIONS.size()> first{};
    const uint64_t scalarCallsBefore = batched.basinCacheMetrics().scalarSampleCalls;
    batched.sampleFarSurfacePoints(POSITIONS, worldgen::SurfaceFootprint::BLOCK_1, first);
    REQUIRE(batched.cachedColumnPlanCount() == 0);
    REQUIRE(batched.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    ChunkGenerator scalar(764891);
    for (size_t index = 0; index < POSITIONS.size(); ++index) {
        const ColumnPos position = POSITIONS[index];
        const worldgen::SurfaceSample direct =
            scalar.sampleFarSurface(position.x, position.z, worldgen::SurfaceFootprint::BLOCK_1);
        REQUIRE(first[index].geology.plateId == direct.geology.plateId);
        REQUIRE(first[index].geology.crust == direct.geology.crust);
        REQUIRE(first[index].geology.boundary == direct.geology.boundary);
        REQUIRE(first[index].geology.rock == direct.geology.rock);
        REQUIRE(first[index].terrainHeight == direct.terrainHeight);
        requireSameWaterAuthority(first[index], direct);
        REQUIRE(std::isfinite(first[index].slope));
        REQUIRE(std::isfinite(first[index].climate.temperatureC));
        REQUIRE(std::isfinite(first[index].climate.annualPrecipitationMm));
        REQUIRE(first[index].biome.primary != Biome::COUNT);
        REQUIRE(first[index].biome.secondary != Biome::COUNT);
    }
    REQUIRE(first[5].hydrology.lake);
    REQUIRE(first[5].hydrology.endorheic);

    batched.clearMacroCaches();
    std::array<worldgen::SurfaceSample, POSITIONS.size()> rebuilt{};
    batched.sampleFarSurfacePoints(POSITIONS, worldgen::SurfaceFootprint::BLOCK_1, rebuilt);
    for (size_t index = 0; index < POSITIONS.size(); ++index)
        requireExactSurface(rebuilt[index], first[index]);

    std::array<ColumnPos, POSITIONS.size()> reverse = POSITIONS;
    std::ranges::reverse(reverse);
    std::array<worldgen::SurfaceSample, POSITIONS.size()> reversed{};
    batched.sampleFarSurfacePoints(reverse, worldgen::SurfaceFootprint::BLOCK_1, reversed);
    for (size_t index = 0; index < POSITIONS.size(); ++index)
        requireExactSurface(reversed[index], first[POSITIONS.size() - index - 1]);
}

TEST_CASE("Cold column plans batch every basin construction sample",
          "[worldgen][advanced][column-plan][batch][determinism][performance][regression]") {
    enum class Feature : uint8_t {
        NONE,
        LAKE,
        WATERFALL,
    };
    struct Fixture {
        int64_t x;
        int64_t z;
        Feature feature = Feature::NONE;
    };
    struct Snapshot {
        std::vector<worldgen::SurfaceSample> samples;
        ColumnPlanSurfaceGrid surfaceY{};
        std::vector<int32_t> exposedSections;
        int minimumSurfaceY = 0;
        int maximumSurfaceY = 0;
    };

    const auto capture = [](const ColumnPlan& plan) {
        Snapshot result;
        result.samples.reserve((CHUNK_EDGE + 1) * (CHUNK_EDGE + 1));
        for (int localZ = 0; localZ <= CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX <= CHUNK_EDGE; ++localX)
                result.samples.push_back(plan.sample(localX, localZ));
        }
        for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                result.surfaceY[static_cast<size_t>(localZ * CHUNK_EDGE + localX)] =
                    static_cast<int16_t>(plan.surfaceY(localX, localZ));
            }
        }
        result.exposedSections.assign(plan.exposedSections().begin(), plan.exposedSections().end());
        result.minimumSurfaceY = plan.minimumSurfaceY();
        result.maximumSurfaceY = plan.maximumSurfaceY();
        return result;
    };
    const auto requireSameSnapshot = [](const Snapshot& actual, const Snapshot& expected) {
        REQUIRE(actual.surfaceY == expected.surfaceY);
        REQUIRE(actual.exposedSections == expected.exposedSections);
        REQUIRE(actual.minimumSurfaceY == expected.minimumSurfaceY);
        REQUIRE(actual.maximumSurfaceY == expected.maximumSurfaceY);
        REQUIRE(actual.samples.size() == expected.samples.size());
        for (size_t index = 0; index < actual.samples.size(); ++index) {
            const int localX = static_cast<int>(index % (CHUNK_EDGE + 1));
            const int localZ = static_cast<int>(index / (CHUNK_EDGE + 1));
            const worldgen::SurfaceSample& sample = actual.samples[index];
            const worldgen::SurfaceSample& reference = expected.samples[index];
            CAPTURE(localX, localZ);
            REQUIRE(sample.geology.plateId == reference.geology.plateId);
            REQUIRE(sample.geology.crust == reference.geology.crust);
            REQUIRE(sample.geology.boundary == reference.geology.boundary);
            REQUIRE(sample.geology.rock == reference.geology.rock);
            REQUIRE(sample.geology.lithology.primary == reference.geology.lithology.primary);
            REQUIRE(sample.geology.lithology.secondary == reference.geology.lithology.secondary);
            REQUIRE(sample.geology.lithology.transition == reference.geology.lithology.transition);
            REQUIRE(sample.geology.lithology.contactDistance ==
                    reference.geology.lithology.contactDistance);
            REQUIRE(sample.hydrology.surfaceElevation == reference.hydrology.surfaceElevation);
            REQUIRE(sample.hydrology.waterSurface == reference.hydrology.waterSurface);
            REQUIRE(sample.hydrology.waterBodyId == reference.hydrology.waterBodyId);
            REQUIRE(sample.hydrology.generatedFluidLevel ==
                    reference.hydrology.generatedFluidLevel);
            REQUIRE(sample.hydrology.transitionOwnerKind ==
                    reference.hydrology.transitionOwnerKind);
            REQUIRE(sample.hydrology.transitionOwnerId == reference.hydrology.transitionOwnerId);
            REQUIRE(sample.hydrology.ocean == reference.hydrology.ocean);
            REQUIRE(sample.hydrology.river == reference.hydrology.river);
            REQUIRE(sample.hydrology.lake == reference.hydrology.lake);
            REQUIRE(sample.hydrology.lakeBank == reference.hydrology.lakeBank);
            REQUIRE(sample.hydrology.channelBank == reference.hydrology.channelBank);
            REQUIRE(sample.hydrology.waterfall == reference.hydrology.waterfall);
            REQUIRE(sample.hydrology.waterfallAnchor == reference.hydrology.waterfallAnchor);
            REQUIRE(sample.hydrology.delta == reference.hydrology.delta);
            REQUIRE(sample.hydrology.estuary == reference.hydrology.estuary);
            REQUIRE(sample.hydrology.brackish == reference.hydrology.brackish);
            REQUIRE(sample.hydrology.waterfallTop == reference.hydrology.waterfallTop);
            REQUIRE(sample.hydrology.waterfallBottom == reference.hydrology.waterfallBottom);
            REQUIRE(sample.hydrology.waterfallWidth == reference.hydrology.waterfallWidth);
            if ((localX % COLUMN_PLAN_LATTICE_SPACING) == 0 &&
                (localZ % COLUMN_PLAN_LATTICE_SPACING) == 0) {
                requireExactSurface(sample, reference);
            }
        }
    };
    const auto exercise = [&](uint32_t seed, std::span<const Fixture> fixtures) {
        ChunkGenerator generator(seed);
        std::vector<Snapshot> expected;
        expected.reserve(fixtures.size());
        const uint64_t scalarCallsBefore = generator.basinCacheMetrics().scalarSampleCalls;
        for (const Fixture fixture : fixtures) {
            const ColumnPos column{Chunk::worldToChunk(fixture.x), Chunk::worldToChunk(fixture.z)};
            const std::shared_ptr<const ColumnPlan> plan = generator.getColumnPlan(column);
            REQUIRE(plan);
            REQUIRE(generator.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
            const std::shared_ptr<const ColumnPlan> cached = generator.getColumnPlan(column);
            REQUIRE(cached == plan);
            REQUIRE(generator.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
            const worldgen::SurfaceSample feature = generator.sampleSurface(fixture.x, fixture.z);
            REQUIRE(generator.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
            CAPTURE(seed, fixture.x, fixture.z);
            if (fixture.feature == Feature::LAKE)
                REQUIRE(feature.hydrology.lake);
            if (fixture.feature == Feature::WATERFALL)
                REQUIRE(feature.hydrology.waterfall);
            expected.push_back(capture(*plan));
        }

        generator.clearMacroCaches();
        const uint64_t scalarCallsAfterClear = generator.basinCacheMetrics().scalarSampleCalls;
        for (size_t reverseIndex = fixtures.size(); reverseIndex-- > 0;) {
            const Fixture fixture = fixtures[reverseIndex];
            const ColumnPos column{Chunk::worldToChunk(fixture.x), Chunk::worldToChunk(fixture.z)};
            const std::shared_ptr<const ColumnPlan> rebuilt = generator.getColumnPlan(column);
            REQUIRE(generator.basinCacheMetrics().scalarSampleCalls == scalarCallsAfterClear);
            requireSameSnapshot(capture(*rebuilt), expected[reverseIndex]);
        }
    };

    constexpr std::array<Fixture, 4> SEED_42_FIXTURES = {{
        {40, 56, Feature::NONE},
        {-24, -40, Feature::NONE},
        {-8'235, 2'976, Feature::LAKE},
        {-8'240, 3'088, Feature::WATERFALL},
    }};
    constexpr std::array<Fixture, 1> SEED_764891_FIXTURES = {{
        {23'029, -111'486, Feature::LAKE},
    }};
    exercise(42, SEED_42_FIXTURES);
    exercise(764891, SEED_764891_FIXTURES);
}

TEST_CASE("Coarse canopy candidates retain scalar surface decisions after batching",
          "[worldgen][advanced][flora][canopy][batch][determinism][performance]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    constexpr int64_t CELL_EDGE = 64;
    constexpr int CANDIDATE_COUNT = 6;
    constexpr uint64_t STREAM = 0x46415243414E4F50ULL;
    const int64_t minimumCellX = world_coord::floorDiv(MINIMUM_X, CELL_EDGE);
    const int64_t maximumCellX = world_coord::floorDiv(MAXIMUM_X - 1, CELL_EDGE);
    const int64_t minimumCellZ = world_coord::floorDiv(MINIMUM_Z, CELL_EDGE);
    const int64_t maximumCellZ = world_coord::floorDiv(MAXIMUM_Z - 1, CELL_EDGE);
    CounterRng random(42);
    std::vector<ColumnPos> positions;
    positions.reserve(16 * CANDIDATE_COUNT);
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            for (int slot = 0; slot < CANDIDATE_COUNT; ++slot) {
                const int offsetX =
                    12 + (slot % 3) * 20 + random.uniformInt(STREAM, cellX, slot, cellZ, 0, -4, 4);
                const int offsetZ =
                    18 + (slot / 3) * 28 + random.uniformInt(STREAM, cellX, slot, cellZ, 1, -5, 5);
                positions.push_back({cellX * CELL_EDGE + offsetX, cellZ * CELL_EDGE + offsetZ});
            }
        }
    }
    REQUIRE(positions.size() == 96);

    ChunkGenerator batched(42);
    std::vector<worldgen::SurfaceSample> surfaces(positions.size());
    batched.sampleFarSurfacePoints(positions, worldgen::SurfaceFootprint::BLOCK_1, surfaces);
    ChunkGenerator scalar(42);
    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        requireExactSurface(
            surfaces[index],
            scalar.sampleFarSurface(position.x, position.z, worldgen::SurfaceFootprint::BLOCK_1));
    }

    const uint64_t scalarCallsBefore = batched.basinCacheMetrics().scalarSampleCalls;
    const std::vector<FarCanopy> first =
        batched.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 32);
    REQUIRE_FALSE(first.empty());
    REQUIRE(batched.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    const auto requireExactHabitat = [](ChunkGenerator& generator, const FarCanopy& canopy) {
        const worldgen::SurfaceSample surface =
            generator.sampleFarSurface(canopy.x, canopy.z, worldgen::SurfaceFootprint::BLOCK_1);
        const int sampledGroundY = generator.surfaceYAt(canopy.x, canopy.z);
        const int habitatGroundY = sampledGroundY;
        const auto habitat =
            feature_generation::evaluateTreeHabitat(canopy.species, surface, habitatGroundY);
        CAPTURE(canopy.x, canopy.z, static_cast<int>(canopy.species), sampledGroundY,
                habitatGroundY, habitat.waterDepthBlocks);
        REQUIRE(habitat.allowed);
        REQUIRE(worldgen::surface_material::supportsTreeRooting(
            generator.farSurfaceMaterialAt(canopy.x, canopy.z, surface)));
        REQUIRE(canopy.baseY == sampledGroundY + 1);
        if (habitat.submerged) {
            REQUIRE((canopy.species == feature_generation::TreeSpecies::MANGROVE ||
                     canopy.species == feature_generation::TreeSpecies::WILLOW));
            const int maximumDepth =
                canopy.species == feature_generation::TreeSpecies::MANGROVE ? 3 : 2;
            REQUIRE(habitat.waterDepthBlocks <= maximumDepth);
        }
    };
    for (const FarCanopy& canopy : first)
        requireExactHabitat(scalar, canopy);

    // The reported seed-forty-two camera window includes irregular banks and
    // shallow water where filtered terrain previously admitted dry-land tree
    // silhouettes. Coarse canopies must retain block-resolution ecology there.
    const std::vector<FarCanopy> reported =
        batched.collectFarCanopiesForLod(-813, 123, -301, 635, 32);
    REQUIRE_FALSE(reported.empty());
    for (const FarCanopy& canopy : reported)
        requireExactHabitat(scalar, canopy);

    batched.clearMacroCaches();
    REQUIRE(batched.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 32) ==
            first);
}

TEST_CASE("Flood tolerant trees stay grounded through shallow generated water",
          "[worldgen][advanced][flora][water][grounding][determinism]") {
    constexpr int64_t MINIMUM_X = -10'048;
    constexpr int64_t MINIMUM_Z = 4'992;
    constexpr int64_t MAXIMUM_X = -9'536;
    constexpr int64_t MAXIMUM_Z = 5'504;
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> canopies =
        generator.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z);
    const auto selected =
        std::find_if(canopies.begin(), canopies.end(), [&](const FarCanopy& canopy) {
            if (canopy.species != feature_generation::TreeSpecies::MANGROVE &&
                canopy.species != feature_generation::TreeSpecies::WILLOW) {
                return false;
            }
            const worldgen::SurfaceSample root = generator.sampleSurface(canopy.x, canopy.z);
            const int ground = generator.surfaceYAt(canopy.x, canopy.z);
            return (root.hydrology.ocean || root.hydrology.river || root.hydrology.lake) &&
                   static_cast<int>(std::ceil(root.waterSurface)) - 1 > ground;
        });
    REQUIRE(selected != canopies.end());
    const int64_t rootX = selected->x;
    const int64_t rootZ = selected->z;
    const int maximumDepth = selected->species == feature_generation::TreeSpecies::MANGROVE ? 3 : 2;

    const worldgen::SurfaceSample surface = generator.sampleSurface(rootX, rootZ);
    const int groundY = generator.surfaceYAt(rootX, rootZ);
    const int waterTopY = static_cast<int>(std::ceil(surface.waterSurface)) - 1;
    REQUIRE((surface.hydrology.river || surface.hydrology.lake));
    REQUIRE(waterTopY > groundY);
    REQUIRE(waterTopY - groundY <= maximumDepth);
    REQUIRE(selected->baseY == groundY + 1);

    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> cubes;
    const auto blockAt = [&](int y) {
        const ChunkPos position{Chunk::worldToChunk(rootX), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(rootZ)};
        auto [found, inserted] = cubes.try_emplace(position);
        if (inserted) {
            found->second = std::make_shared<Chunk>(position);
            generator.generateCube(*found->second);
        }
        return found->second->getBlock(Chunk::worldToLocal(rootX), Chunk::worldToLocalY(y),
                                       Chunk::worldToLocal(rootZ));
    };
    REQUIRE(worldgen::surface_material::supportsTreeRooting(blockAt(groundY)));
    for (int y = groundY + 1; y <= waterTopY; ++y) {
        CAPTURE(y);
        REQUIRE(blockAt(y) == selected->logBlock);
    }

    size_t exactSubmergedCrowns = 0;
    for (const FarCanopy& canopy : canopies) {
        const worldgen::SurfaceSample exactSurface = generator.sampleSurface(canopy.x, canopy.z);
        const int exactGroundY = generator.surfaceYAt(canopy.x, canopy.z);
        const int exactWaterTopY = static_cast<int>(std::ceil(exactSurface.waterSurface)) - 1;
        const bool submerged = (exactSurface.hydrology.ocean || exactSurface.hydrology.river ||
                                exactSurface.hydrology.lake) &&
                               exactGroundY < exactWaterTopY;
        if (!submerged)
            continue;
        ++exactSubmergedCrowns;
        CAPTURE(canopy.x, canopy.z, static_cast<int>(canopy.species),
                static_cast<int>(canopy.logBlock), exactGroundY, exactWaterTopY);
        REQUIRE((canopy.logBlock == BlockType::MANGROVE_LOG ||
                 canopy.logBlock == BlockType::WILLOW_LOG));
        REQUIRE(canopy.baseY == exactGroundY + 1);
        REQUIRE(exactWaterTopY - exactGroundY <= 3);
    }
    REQUIRE(exactSubmergedCrowns > 0);

    // Step two uses the same accepted roots and species as exact cube
    // emission. Only the vertical anchor is re-grounded by its displayed
    // reduced voxel in the far mesher.
    const std::vector<FarCanopy> stepTwo =
        generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2);
    REQUIRE(stepTwo.size() == canopies.size());
    for (const FarCanopy& exact : canopies) {
        const auto nearFar =
            std::find_if(stepTwo.begin(), stepTwo.end(), [&](const FarCanopy& canopy) {
                return canopy.anchorId == exact.anchorId;
            });
        REQUIRE(nearFar != stepTwo.end());
        REQUIRE_FALSE(nearFar->aggregate);
        REQUIRE(nearFar->x == exact.x);
        REQUIRE(nearFar->z == exact.z);
        REQUIRE(nearFar->logBlock == exact.logBlock);
        REQUIRE(nearFar->leafBlock == exact.leafBlock);
    }

    generator.clearMacroCaches();
    REQUIRE(generator.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z) == canopies);
    REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2) ==
            stepTwo);
}

TEST_CASE("Near canopy roots satisfy final block-scale habitat and water support",
          "[worldgen][advanced][flora][water][volcano][substrate][lod][regression]") {
    struct Window {
        uint32_t seed;
        int64_t minimumX;
        int64_t minimumZ;
        int64_t maximumX;
        int64_t maximumZ;
    };
    constexpr std::array WINDOWS = {
        Window{42, -10'048, 4'992, -9'536, 5'504},
        Window{42, -27'136, -16'896, -26'624, -16'384},
        Window{42, -258, -1'916, 254, -1'404},
        Window{42, -813, 123, -301, 635},
        Window{764891, 22'773, -111'742, 23'285, -111'230},
    };
    const auto maximumFloodDepth = [](feature_generation::TreeSpecies species) {
        if (species == feature_generation::TreeSpecies::MANGROVE)
            return 3;
        if (species == feature_generation::TreeSpecies::WILLOW)
            return 2;
        return 0;
    };

    size_t evaluated = 0;
    size_t evaluatedSubmerged = 0;
    size_t populatedWindows = 0;
    bool determinismChecked = false;
    for (const Window window : WINDOWS) {
        ChunkGenerator generator(window.seed);
        const std::vector<FarCanopy> canopies = generator.collectFarCanopiesForLod(
            window.minimumX, window.minimumZ, window.maximumX, window.maximumZ, 2);
        CAPTURE(window.seed, window.minimumX, window.minimumZ);
        if (!canopies.empty())
            ++populatedWindows;
        for (const FarCanopy& canopy : canopies) {
            const worldgen::SurfaceSample surface =
                generator.sampleFarSurface(canopy.x, canopy.z, worldgen::SurfaceFootprint::BLOCK_1);
            const int groundY = generator.surfaceYAt(canopy.x, canopy.z);
            const BlockType substrate = generator.farSurfaceMaterialAt(canopy.x, canopy.z, surface);
            CAPTURE(window.seed, canopy.x, canopy.z, static_cast<int>(canopy.species), groundY,
                    static_cast<int>(substrate));
            REQUIRE(worldgen::surface_material::supportsTreeRooting(substrate));
            const auto habitat =
                feature_generation::evaluateTreeHabitat(canopy.species, surface, groundY);
            REQUIRE(habitat.allowed);
            if (habitat.submerged) {
                ++evaluatedSubmerged;
                REQUIRE((canopy.species == feature_generation::TreeSpecies::MANGROVE ||
                         canopy.species == feature_generation::TreeSpecies::WILLOW));
                REQUIRE(habitat.waterDepthBlocks <= maximumFloodDepth(canopy.species));

                const worldgen::SurfaceSample exact = generator.sampleSurface(canopy.x, canopy.z);
                const int exactGroundY = generator.surfaceYAt(canopy.x, canopy.z);
                const int exactWaterTopY = static_cast<int>(std::ceil(exact.waterSurface)) - 1;
                const int exactDepth = std::max(0, exactWaterTopY - exactGroundY);
                REQUIRE(exactDepth <= maximumFloodDepth(canopy.species));
            }
            ++evaluated;
        }

        // Regenerating after a cache eviction proves determinism; the first
        // populated window exercises that property, so the rest skip the
        // redundant recompute to keep the matrix inside its CI budget.
        if (!determinismChecked && !canopies.empty()) {
            generator.clearMacroCaches();
            REQUIRE(generator.collectFarCanopiesForLod(window.minimumX, window.minimumZ,
                                                       window.maximumX, window.maximumZ,
                                                       2) == canopies);
            determinismChecked = true;
        }
    }
    REQUIRE(determinismChecked);
    REQUIRE(evaluated >= 24);
    REQUIRE(evaluatedSubmerged > 0);
    REQUIRE(populatedWindows >= 3);

    ChunkGenerator crater(764891);
    const worldgen::SurfaceSample craterCenter =
        crater.sampleFarSurface(23'029, -111'486, worldgen::SurfaceFootprint::BLOCK_1);
    REQUIRE(craterCenter.hydrology.lake);
    REQUIRE(craterCenter.hydrology.waterBodyId != worldgen::NO_WATER_BODY);
    const std::vector<FarCanopy> craterCanopies =
        crater.collectFarCanopiesForLod(22'901, -111'614, 23'158, -111'357, 2);
    for (const FarCanopy& canopy : craterCanopies) {
        const worldgen::SurfaceSample root =
            crater.sampleFarSurface(canopy.x, canopy.z, worldgen::SurfaceFootprint::BLOCK_1);
        if (root.hydrology.waterBodyId != craterCenter.hydrology.waterBodyId ||
            !worldgen::surface_material::submerged(root)) {
            continue;
        }
        const auto habitat = feature_generation::evaluateTreeHabitat(
            canopy.species, root, crater.surfaceYAt(canopy.x, canopy.z));
        REQUIRE(habitat.allowed);
        REQUIRE(habitat.waterDepthBlocks <= maximumFloodDepth(canopy.species));
    }
}

TEST_CASE("Far canopies reuse accepted tree anchors across negative half-open bounds",
          "[worldgen][advanced][flora][far-canopy][determinism]") {
    constexpr int64_t MINIMUM_X = -160;
    constexpr int64_t MINIMUM_Z = 16;
    constexpr int64_t MAXIMUM_X = -32;
    constexpr int64_t MAXIMUM_Z = 160;
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> first =
        generator.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z);
    REQUIRE_FALSE(first.empty());

    generator.sampleSurface(4096, -8192);
    generator.clearMacroCaches();
    const std::vector<FarCanopy> rebuilt =
        generator.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z);
    REQUIRE(rebuilt == first);
    REQUIRE(generator.collectFarCanopies(MAXIMUM_X, MAXIMUM_Z, MINIMUM_X, MINIMUM_Z).empty());

    ChunkGenerator reordered(42);
    reordered.collectFarCanopies(2048, -1024, 2112, -960);
    REQUIRE(reordered.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z) == first);

    std::unordered_set<uint64_t> ids;
    for (const FarCanopy& canopy : first) {
        REQUIRE(canopy.x < 0);
        REQUIRE(canopy.baseY == generator.surfaceYAt(canopy.x, canopy.z) + 1);
        REQUIRE(canopy.canopyMinimumY <= canopy.canopyMaximumY);
        REQUIRE(canopy.canopyMaximumY <= canopy.topY);
        if (canopy.species == feature_generation::TreeSpecies::FALLEN_LOG) {
            REQUIRE(canopy.canopyRadius == 0);
            REQUIRE(canopy.leafBlock == BlockType::AIR);
            REQUIRE(canopy.logBlock != BlockType::AIR);
            REQUIRE(canopy.formExtent >= 4);
        } else {
            REQUIRE(canopy.canopyRadius > 0);
            REQUIRE(canopy.leafBlock != BlockType::AIR);
        }
        REQUIRE(ids.insert(canopy.anchorId).second);
    }

    const auto crossing = std::find_if(first.begin(), first.end(), [](const FarCanopy& canopy) {
        return canopy.canopyOffsetX == 0 && canopy.canopyRadius >= 2;
    });
    REQUIRE(crossing != first.end());
    const int64_t crossingMinimumX = crossing->x + 1;
    const std::vector<FarCanopy> expanded =
        generator.collectFarCanopies(crossingMinimumX, crossing->z - crossing->canopyRadius,
                                     crossing->x + 2, crossing->z + crossing->canopyRadius + 1);
    REQUIRE(std::find_if(expanded.begin(), expanded.end(), [&](const FarCanopy& canopy) {
                return canopy.anchorId == crossing->anchorId && canopy.x < crossingMinimumX;
            }) != expanded.end());
}

TEST_CASE("Step-two canopies match exact accepted anchors across signed bounds",
          "[worldgen][advanced][flora][far-canopy][lod][performance][determinism]") {
    const auto requireSameSilhouette = [](const FarCanopy& exact, const FarCanopy& stepTwo) {
        REQUIRE_FALSE(exact.aggregate);
        REQUIRE_FALSE(stepTwo.aggregate);
        REQUIRE(stepTwo.anchorId == exact.anchorId);
        REQUIRE(stepTwo.x == exact.x);
        REQUIRE(stepTwo.z == exact.z);
        REQUIRE(stepTwo.logBlock == exact.logBlock);
        REQUIRE(stepTwo.leafBlock == exact.leafBlock);
        REQUIRE(stepTwo.canopyOffsetX == exact.canopyOffsetX);
        REQUIRE(stepTwo.canopyOffsetZ == exact.canopyOffsetZ);
        REQUIRE(stepTwo.canopyRadius == exact.canopyRadius);
        REQUIRE(stepTwo.species == exact.species);
        REQUIRE(stepTwo.formX == exact.formX);
        REQUIRE(stepTwo.formZ == exact.formZ);
        REQUIRE(stepTwo.formExtent == exact.formExtent);
        REQUIRE(stepTwo.topY - stepTwo.baseY == exact.topY - exact.baseY);
        REQUIRE(stepTwo.canopyMinimumY - stepTwo.baseY == exact.canopyMinimumY - exact.baseY);
        REQUIRE(stepTwo.canopyMaximumY - stepTwo.baseY == exact.canopyMaximumY - exact.baseY);
    };
    const auto verifyWindow = [&](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                  int64_t maximumZ) {
        ChunkGenerator generator(42);
        const uint64_t scalarSamplesBefore = generator.basinCacheMetrics().scalarSampleCalls;
        const std::vector<FarCanopy> stepTwo =
            generator.collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ, 2);
        REQUIRE_FALSE(stepTwo.empty());
        REQUIRE(generator.cachedColumnPlanCount() == 0);
        REQUIRE(generator.basinCacheMetrics().scalarSampleCalls == scalarSamplesBefore);

        const std::vector<FarCanopy> exact =
            generator.collectFarCanopies(minimumX, minimumZ, maximumX, maximumZ);
        REQUIRE(exact.size() == stepTwo.size());
        for (const FarCanopy& exactCanopy : exact) {
            const auto nearFar =
                std::find_if(stepTwo.begin(), stepTwo.end(), [&](const FarCanopy& canopy) {
                    return canopy.anchorId == exactCanopy.anchorId;
                });
            REQUIRE(nearFar != stepTwo.end());
            requireSameSilhouette(exactCanopy, *nearFar);
        }

        generator.clearMacroCaches();
        REQUIRE(generator.collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ, 2) ==
                stepTwo);
    };

    verifyWindow(-160, 16, -32, 144);
    verifyWindow(256, 256, 384, 384);
}

TEST_CASE("Exact and step-two trees share pure structure exclusions",
          "[worldgen][advanced][flora][structure][lod][determinism][regression]") {
    constexpr uint32_t SEED = 112233;
    constexpr std::array<ColumnPos, 4> REGIONS = {ColumnPos{-7, 11}, ColumnPos{-1, -1},
                                                  ColumnPos{0, 0}, ColumnPos{3, -4}};
    ChunkGenerator generator(SEED);
    StructurePlacer structures(SEED);
    GenScratch scratch;
    scratch.reset(&generator);

    size_t surroundingCanopies = 0;
    for (const ColumnPos region : REGIONS) {
        const StructurePlacement placement =
            structures.regionPlacement(region.x, region.z, generator, scratch);
        constexpr int64_t WINDOW_RADIUS = 48;
        const int64_t minimumX = placement.anchorX - WINDOW_RADIUS;
        const int64_t minimumZ = placement.anchorZ - WINDOW_RADIUS;
        const int64_t maximumX = placement.anchorX + WINDOW_RADIUS + 1;
        const int64_t maximumZ = placement.anchorZ + WINDOW_RADIUS + 1;
        const std::vector<FarCanopy> exact =
            generator.collectFarCanopies(minimumX, minimumZ, maximumX, maximumZ);
        const std::vector<FarCanopy> stepTwo =
            generator.collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ, 2);
        REQUIRE(stepTwo.size() == exact.size());
        for (const FarCanopy& canopy : stepTwo) {
            CAPTURE(region.x, region.z, placement.anchorX, placement.anchorZ, canopy.x, canopy.z);
            REQUIRE((std::abs(canopy.x - placement.anchorX) > placement.halfX + 1 ||
                     std::abs(canopy.z - placement.anchorZ) > placement.halfZ + 1));
            const auto matching = std::ranges::find_if(exact, [&](const FarCanopy& candidate) {
                return candidate.anchorId == canopy.anchorId;
            });
            REQUIRE(matching != exact.end());
            REQUIRE(matching->x == canopy.x);
            REQUIRE(matching->z == canopy.z);
            REQUIRE(matching->species == canopy.species);
            REQUIRE(matching->formX == canopy.formX);
            REQUIRE(matching->formZ == canopy.formZ);
            REQUIRE(matching->formExtent == canopy.formExtent);
        }
        surroundingCanopies += stepTwo.size();
    }
    REQUIRE(surroundingCanopies > 0);
}

TEST_CASE("Near anchors and distant forest cover are deterministic and globally anchored",
          "[worldgen][advanced][flora][far-canopy][lod][determinism]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    ChunkGenerator generator(42);

    constexpr std::array LOD_STEPS = {2, 4, 8, 16, 32};
    std::array<std::vector<FarCanopy>, LOD_STEPS.size()> tiers;
    for (size_t tierIndex = 0; tierIndex < LOD_STEPS.size(); ++tierIndex) {
        const int lodStep = LOD_STEPS[tierIndex];
        const std::vector<FarCanopy> first =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, lodStep);
        REQUIRE_FALSE(first.empty());
        generator.sampleSurface(4096 + lodStep, -8192 - lodStep);
        generator.clearMacroCaches();
        REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z,
                                                   lodStep) == first);

        std::unordered_set<uint64_t> ids;
        for (const FarCanopy& canopy : first) {
            REQUIRE(canopy.aggregate == (lodStep >= 4));
            if (canopy.aggregate) {
                REQUIRE(canopy.x >= MINIMUM_X);
                REQUIRE(canopy.x < MAXIMUM_X);
                REQUIRE(canopy.z >= MINIMUM_Z);
                REQUIRE(canopy.z < MAXIMUM_Z);
            }
            REQUIRE(canopy.canopyMinimumY <= canopy.canopyMaximumY);
            if (canopy.species == feature_generation::TreeSpecies::FALLEN_LOG) {
                REQUIRE_FALSE(canopy.aggregate);
                REQUIRE(canopy.canopyRadius == 0);
                REQUIRE(canopy.logBlock != BlockType::AIR);
                REQUIRE(canopy.leafBlock == BlockType::AIR);
                REQUIRE(canopy.formExtent >= 4);
            } else {
                REQUIRE(canopy.canopyRadius > 0);
                REQUIRE(canopy.leafBlock != BlockType::AIR);
            }
            REQUIRE(ids.insert(canopy.anchorId).second);
        }

        if (lodStep == 4) {
            std::vector<ColumnPos> positions;
            positions.reserve(first.size());
            for (const FarCanopy& canopy : first)
                positions.push_back({canopy.x, canopy.z});
            std::vector<worldgen::SurfaceSample> habitats(positions.size());
            generator.sampleFarHabitatPoints(positions, habitats);
            for (size_t index = 0; index < first.size(); ++index) {
                const FarCanopy& canopy = first[index];
                const worldgen::SurfaceSample& habitatSurface = habitats[index];
                const int groundY =
                    static_cast<int>(std::ceil(worldgen::geometryTerrainHeight(habitatSurface))) -
                    1;
                const BlockType substrate =
                    generator.farSurfaceMaterialAt(canopy.x, canopy.z, habitatSurface);
                const auto habitat = feature_generation::evaluateTreeHabitat(
                    canopy.species, habitatSurface, groundY);
                CAPTURE(canopy.x, canopy.z, static_cast<int>(canopy.species), groundY,
                        static_cast<int>(substrate));
                REQUIRE_FALSE(worldgen::surface_material::submerged(habitatSurface));
                REQUIRE_FALSE(habitat.submerged);
                REQUIRE(habitat.allowed);
                REQUIRE(worldgen::surface_material::supportsTreeRooting(substrate));
            }
        }

        const int64_t splitX = MINIMUM_X + 128;
        const std::vector<FarCanopy> west =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, splitX, MAXIMUM_Z, lodStep);
        const std::vector<FarCanopy> east =
            generator.collectFarCanopiesForLod(splitX, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, lodStep);
        std::unordered_set<uint64_t> splitIds;
        for (const FarCanopy& canopy : west)
            splitIds.insert(canopy.anchorId);
        for (const FarCanopy& canopy : east)
            splitIds.insert(canopy.anchorId);
        REQUIRE(splitIds == ids);
        tiers[tierIndex] = first;
    }

    // Step two is the exact accepted-anchor distribution. Aggregated step-four
    // and more distant tiers form their own stable, nested hierarchy.
    REQUIRE(tiers[1].size() <= tiers[0].size());
    REQUIRE(tiers[1].size() * 5 >= tiers[0].size());
    for (size_t tierIndex = 2; tierIndex < tiers.size(); ++tierIndex) {
        REQUIRE(tiers[tierIndex].size() <= tiers[tierIndex - 1].size());
        for (const FarCanopy& farther : tiers[tierIndex]) {
            const auto nearer = std::find_if(
                tiers[tierIndex - 1].begin(), tiers[tierIndex - 1].end(),
                [&](const FarCanopy& canopy) { return canopy.anchorId == farther.anchorId; });
            REQUIRE(nearer != tiers[tierIndex - 1].end());
            REQUIRE(*nearer == farther);
        }
    }
}

TEST_CASE("Far canopy aggregation represents exact candidate opportunities",
          "[worldgen][advanced][flora][far-canopy][density][regression]") {
    constexpr double EXACT_ACCEPTANCE = 0.055;
    constexpr double REPRESENTED_EXACT_OPPORTUNITIES = 64.0 / 6.0;
    const double expected = 1.0 - std::pow(1.0 - EXACT_ACCEPTANCE, REPRESENTED_EXACT_OPPORTUNITIES);
    const double actual = feature_generation::farCanopyAggregateAcceptance(EXACT_ACCEPTANCE);

    REQUIRE(feature_generation::farCanopyAggregateAcceptance(0.0) == 0.0);
    REQUIRE(feature_generation::farCanopyAggregateAcceptance(1.0) == 1.0);
    REQUIRE(actual == Catch::Approx(expected));
    REQUIRE(actual > 0.44);
    REQUIRE(actual < 0.46);
    REQUIRE(feature_generation::farCanopyAggregateAcceptance(0.10) > actual);
}

TEST_CASE("Every far canopy corresponds to emitted tree material",
          "[worldgen][advanced][flora][far-canopy]") {
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> discovery = generator.collectFarCanopies(-160, 16, -32, 160);
    REQUIRE_FALSE(discovery.empty());
    const FarCanopy& selected = discovery.front();
    const std::vector<FarCanopy> canopies = generator.collectFarCanopies(
        selected.x - 8, selected.z - 8, selected.x + 9, selected.z + 9);
    REQUIRE_FALSE(canopies.empty());

    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> cubes;
    auto emittedBlock = [&](int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        auto [found, inserted] = cubes.try_emplace(position);
        if (inserted) {
            found->second = std::make_shared<Chunk>(position);
            generator.generateCube(*found->second);
        }
        return found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                       Chunk::worldToLocal(z));
    };

    std::unordered_set<uint64_t> ids;
    for (const FarCanopy& canopy : canopies) {
        REQUIRE(ids.insert(canopy.anchorId).second);
        if (canopy.logBlock != BlockType::AIR) {
            CAPTURE(canopy.x, canopy.z, canopy.baseY, static_cast<int>(canopy.species),
                    static_cast<int>(canopy.logBlock), canopy.anchorId);
            REQUIRE(emittedBlock(canopy.x, canopy.baseY, canopy.z) == canopy.logBlock);
        }

        if (canopy.leafBlock != BlockType::AIR) {
            const int64_t centerX = canopy.x + canopy.canopyOffsetX;
            const int64_t centerZ = canopy.z + canopy.canopyOffsetZ;
            bool foundLeaf = false;
            for (int y = canopy.canopyMinimumY; y <= canopy.canopyMaximumY && !foundLeaf; ++y) {
                for (int64_t z = centerZ - canopy.canopyRadius;
                     z <= centerZ + canopy.canopyRadius && !foundLeaf; ++z) {
                    for (int64_t x = centerX - canopy.canopyRadius;
                         x <= centerX + canopy.canopyRadius; ++x) {
                        if (emittedBlock(x, y, z) == canopy.leafBlock) {
                            foundLeaf = true;
                            break;
                        }
                    }
                }
            }
            REQUIRE(foundLeaf);
        } else {
            REQUIRE(canopy.species == feature_generation::TreeSpecies::FALLEN_LOG);
            const int64_t endX = canopy.x + canopy.formX * (canopy.formExtent - 1);
            const int64_t endZ = canopy.z + canopy.formZ * (canopy.formExtent - 1);
            REQUIRE(emittedBlock(endX, canopy.baseY, endZ) == canopy.logBlock);
        }
    }
}

TEST_CASE("Generated waterfalls carry finished falling states without runtime settling",
          "[worldgen][advanced][hydrology][fluid]") {
    ChunkGenerator generator(42);
    struct FallingLip {
        int64_t x = 0;
        int64_t z = 0;
        worldgen::SurfaceSample surface;
        worldgen::GeneratedFluidColumn fluid;
    };
    std::optional<FallingLip> lip;
    for (int64_t z = 3'078; z <= 3'098 && !lip.has_value(); ++z) {
        for (int64_t x = -8'252; x <= -8'236; ++x) {
            const worldgen::SurfaceSample surface =
                generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
            if (!surface.hydrology.waterfall ||
                surface.hydrology.transitionOwnerKind !=
                    worldgen::WaterTransitionKind::EXPLICIT_FALL ||
                surface.hydrology.generatedFluidLevel != 7) {
                continue;
            }
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(surface);
            if (!fluid.wet || fluid.topState.isFalling() || fluid.topState.level() != 7)
                continue;
            lip = FallingLip{.x = x, .z = z, .surface = surface, .fluid = fluid};
            break;
        }
    }
    REQUIRE(lip.has_value());

    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    const auto generatedCell = [&](int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return std::pair{
            found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                    Chunk::worldToLocal(z)),
            found->second->getFluidState(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                         Chunk::worldToLocal(z)),
        };
    };

    const auto [lipBlock, lipState] = generatedCell(lip->x, lip->fluid.topY, lip->z);
    REQUIRE(lipBlock == BlockType::WATER);
    REQUIRE_FALSE(lipState.isFalling());
    REQUIRE(lipState.level() == 7);

    bool foundPredecessor = false;
    for (const auto [offsetX, offsetZ] :
         {std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}}) {
        const worldgen::SurfaceSample predecessor = generator.sampleFarSurface(
            lip->x + offsetX, lip->z + offsetZ, worldgen::SurfaceFootprint::BLOCK_1);
        const worldgen::GeneratedFluidColumn predecessorFluid =
            worldgen::generatedFluidColumn(predecessor);
        if (!predecessorFluid.wet || predecessorFluid.topY != lip->fluid.topY ||
            predecessorFluid.topState.isFalling() || predecessorFluid.topState.level() != 6 ||
            predecessor.hydrology.transitionOwnerKind !=
                worldgen::WaterTransitionKind::EXPLICIT_FALL ||
            predecessor.hydrology.transitionOwnerId != lip->surface.hydrology.transitionOwnerId) {
            continue;
        }
        const auto [predecessorBlock, predecessorState] =
            generatedCell(lip->x + offsetX, predecessorFluid.topY, lip->z + offsetZ);
        REQUIRE(predecessorBlock == BlockType::WATER);
        REQUIRE(predecessorState == predecessorFluid.topState);
        foundPredecessor = true;
        break;
    }
    REQUIRE(foundPredecessor);

    std::set<std::tuple<int64_t, int32_t, int64_t>> fallingCubes;
    int fallingBottomY = lip->fluid.topY - 1;
    size_t fallingCells = 0;
    while (fallingBottomY >= WORLD_MIN_Y) {
        const auto [block, state] = generatedCell(lip->x, fallingBottomY, lip->z);
        if (block != BlockType::WATER || !state.isFalling() || state.level() != 7)
            break;
        fallingCubes.emplace(Chunk::worldToChunk(lip->x), Chunk::worldToChunkY(fallingBottomY),
                             Chunk::worldToChunk(lip->z));
        ++fallingCells;
        --fallingBottomY;
    }
    CAPTURE(lip->x, lip->z, lip->fluid.topY, fallingBottomY, fallingCells);
    REQUIRE(fallingCells > 0);
    const auto [receiverBlock, receiverState] = generatedCell(lip->x, fallingBottomY, lip->z);
    REQUIRE(receiverBlock == BlockType::WATER);
    REQUIRE(receiverState.isSource());
    for (const auto& cubeKey : fallingCubes) {
        REQUIRE(cubes.at(cubeKey)->hasExplicitFluidStates());
    }
}

TEST_CASE("Generated outlet throats remain settled after nearby block activation",
          "[worldgen][advanced][hydrology][fluid][runtime][regression]") {
    constexpr int64_t MINIMUM_X = -8'252;
    constexpr int64_t MAXIMUM_X = -8'236;
    constexpr int64_t MINIMUM_Z = 3'078;
    constexpr int64_t MAXIMUM_Z = 3'098;
    constexpr int RECEIVER_Y = SEA_LEVEL - 1;
    constexpr int WATER_TOP_Y = 81;
    constexpr int SNAPSHOT_TOP_Y = WATER_TOP_Y + 1;
    constexpr int MAXIMUM_SETTLE_TICKS = 400;

    World world(42, 4);
    const ChunkPos center{Chunk::worldToChunk(-8'240), Chunk::worldToChunkY(WATER_TOP_Y),
                          Chunk::worldToChunk(3'088)};
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetY = -2; offsetY <= 1; ++offsetY) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    REQUIRE(world.getPendingFluidCount() == 0);

    const auto snapshot = [&] {
        std::vector<uint16_t> result;
        result.reserve(static_cast<size_t>(MAXIMUM_X - MINIMUM_X + 1) *
                       static_cast<size_t>(MAXIMUM_Z - MINIMUM_Z + 1) *
                       static_cast<size_t>(SNAPSHOT_TOP_Y - RECEIVER_Y + 1));
        for (int y = RECEIVER_Y; y <= SNAPSHOT_TOP_Y; ++y) {
            for (int64_t z = MINIMUM_Z; z <= MAXIMUM_Z; ++z) {
                for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
                    const FluidCell cell = world.readFluidCell({x, y, z});
                    REQUIRE(cell.loaded);
                    result.push_back(static_cast<uint16_t>(static_cast<uint16_t>(cell.block) << 8U |
                                                           cell.state.packed()));
                }
            }
        }
        return result;
    };

    std::optional<int64_t> sourceX;
    std::array<uint16_t, MAXIMUM_X - MINIMUM_X + 1> topRow{};
    std::array<double, MAXIMUM_X - MINIMUM_X + 1> analyticalSurfaces{};
    std::array<uint8_t, MAXIMUM_X - MINIMUM_X + 1> analyticalLevels{};
    std::array<worldgen::WaterTransitionKind, MAXIMUM_X - MINIMUM_X + 1> analyticalOwnerKinds{};
    std::array<uint64_t, MAXIMUM_X - MINIMUM_X + 1> analyticalOwnerIds{};
    std::array<double, MAXIMUM_X - MINIMUM_X + 1> analyticalFlowX{};
    std::array<double, MAXIMUM_X - MINIMUM_X + 1> analyticalFlowZ{};
    for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
        const size_t index = static_cast<size_t>(x - MINIMUM_X);
        const FluidCell cell = world.readFluidCell({x, WATER_TOP_Y, 3'088});
        topRow[index] =
            static_cast<uint16_t>(static_cast<uint16_t>(cell.block) << 8U | cell.state.packed());
        const worldgen::SurfaceSample surface =
            world.generator().sampleFarSurface(x, 3'088, worldgen::SurfaceFootprint::BLOCK_1);
        analyticalSurfaces[index] = surface.waterSurface;
        analyticalLevels[index] = surface.hydrology.generatedFluidLevel;
        analyticalOwnerKinds[index] = surface.hydrology.transitionOwnerKind;
        analyticalOwnerIds[index] = surface.hydrology.transitionOwnerId;
        analyticalFlowX[index] = surface.hydrology.flowDirection.x;
        analyticalFlowZ[index] = surface.hydrology.flowDirection.z;
    }
    for (int64_t candidate = MINIMUM_X; candidate + 7 <= MAXIMUM_X; ++candidate) {
        const FluidCell source = world.readFluidCell({candidate, WATER_TOP_Y, 3'088});
        if (!source.isWater() || !source.state.isSource())
            continue;
        bool completeGradient = true;
        for (uint8_t level = 1; level <= 7; ++level) {
            const FluidCell flow = world.readFluidCell({candidate + level, WATER_TOP_Y, 3'088});
            completeGradient = completeGradient && flow.isWater() && !flow.state.isFalling() &&
                               flow.state.level() == level;
        }
        if (completeGradient) {
            sourceX = candidate;
            break;
        }
    }
    CAPTURE(topRow, analyticalSurfaces, analyticalLevels, analyticalOwnerKinds, analyticalOwnerIds,
            analyticalFlowX, analyticalFlowZ);
    REQUIRE(sourceX.has_value());
    const int64_t lipX = *sourceX + 7;

    int fallingCells = 0;
    int receiverTopY = WATER_TOP_Y - 1;
    while (receiverTopY >= RECEIVER_Y) {
        const FluidCell cell = world.readFluidCell({lipX, receiverTopY, 3'088});
        if (!cell.isWater() || !cell.state.isFalling() || cell.state.level() != 7)
            break;
        ++fallingCells;
        --receiverTopY;
    }
    CAPTURE(sourceX, lipX, fallingCells, receiverTopY);
    REQUIRE(fallingCells > 0);
    const FluidCell receiver = world.readFluidCell({lipX, receiverTopY, 3'088});
    REQUIRE(receiver.isWater());
    REQUIRE(receiver.state.isSource());

    std::vector<uint8_t> wetColumns(static_cast<size_t>(MAXIMUM_X - MINIMUM_X + 1) *
                                    static_cast<size_t>(MAXIMUM_Z - MINIMUM_Z + 1));
    const auto wetIndex = [](int64_t x, int64_t z) {
        return static_cast<size_t>(z - MINIMUM_Z) * static_cast<size_t>(MAXIMUM_X - MINIMUM_X + 1) +
               static_cast<size_t>(x - MINIMUM_X);
    };
    for (int64_t z = MINIMUM_Z; z <= MAXIMUM_Z; ++z) {
        for (int64_t x = MINIMUM_X; x <= MAXIMUM_X; ++x) {
            for (int y = RECEIVER_Y; y <= WATER_TOP_Y; ++y) {
                if (world.readFluidCell({x, y, z}).isWater()) {
                    wetColumns[wetIndex(x, z)] = 1;
                    break;
                }
            }
        }
    }
    size_t isolatedWetColumns = 0;
    for (int64_t z = MINIMUM_Z + 1; z < MAXIMUM_Z; ++z) {
        for (int64_t x = MINIMUM_X + 1; x < MAXIMUM_X; ++x) {
            if (wetColumns[wetIndex(x, z)] == 0)
                continue;
            bool hasWetNeighbor = false;
            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                    if (offsetX == 0 && offsetZ == 0)
                        continue;
                    hasWetNeighbor =
                        hasWetNeighbor || wetColumns[wetIndex(x + offsetX, z + offsetZ)] != 0;
                }
            }
            isolatedWetColumns += !hasWetNeighbor;
        }
    }
    CAPTURE(isolatedWetColumns);
    REQUIRE(isolatedWetColumns == 0);

    const std::vector<uint16_t> before = snapshot();
    // Include the first downstream column so an incorrectly detached curtain
    // cannot escape activation merely because it is not under the level-seven
    // lip discovered above.
    for (int64_t x = *sourceX; x <= lipX + 1; ++x) {
        const FluidCell airAbove = world.readFluidCell({x, SNAPSHOT_TOP_Y, 3'088});
        REQUIRE(airAbove.loaded);
        REQUIRE(airAbove.block == BlockType::AIR);
        world.setBlock(x, SNAPSHOT_TOP_Y, 3'088, BlockType::AIR);
    }

    int elapsedTicks = 0;
    while (elapsedTicks < 40 ||
           (world.getPendingFluidCount() > 0 && elapsedTicks < MAXIMUM_SETTLE_TICKS)) {
        world.tickFluids(1.0 / static_cast<double>(FLUID_TICKS_PER_SECOND));
        ++elapsedTicks;
    }
    CAPTURE(elapsedTicks, world.getPendingFluidCount(), sourceX, lipX, fallingCells, receiverTopY,
            isolatedWetColumns);
    REQUIRE(elapsedTicks >= 40);
    REQUIRE(elapsedTicks < MAXIMUM_SETTLE_TICKS);
    REQUIRE(world.getPendingFluidCount() == 0);
    const std::vector<uint16_t> after = snapshot();
    const auto changed = std::mismatch(before.begin(), before.end(), after.begin());
    int64_t changedX = 0;
    int64_t changedZ = 0;
    int changedY = 0;
    uint16_t changedBefore = 0;
    uint16_t changedAfter = 0;
    if (changed.first != before.end()) {
        const size_t linear = static_cast<size_t>(std::distance(before.begin(), changed.first));
        constexpr size_t WIDTH = static_cast<size_t>(MAXIMUM_X - MINIMUM_X + 1);
        constexpr size_t DEPTH = static_cast<size_t>(MAXIMUM_Z - MINIMUM_Z + 1);
        changedY = RECEIVER_Y + static_cast<int>(linear / (WIDTH * DEPTH));
        const size_t horizontal = linear % (WIDTH * DEPTH);
        changedZ = MINIMUM_Z + static_cast<int64_t>(horizontal / WIDTH);
        changedX = MINIMUM_X + static_cast<int64_t>(horizontal % WIDTH);
        changedBefore = *changed.first;
        changedAfter = *changed.second;
    }
    CAPTURE(changedX, changedY, changedZ, changedBefore, changedAfter);
    REQUIRE(changed.first == before.end());
}

TEST_CASE("Incised rivers stay supported across cube faces without implicit water walls",
          "[worldgen][advanced][hydrology][river][waterfall][seam][regression]") {
    constexpr std::array<std::pair<int64_t, int64_t>, 4> RIVER_PROBES{{
        {-12'801, 2'759},
        {-12'800, 2'759},
        {-12'801, 2'760},
        {-12'800, 2'760},
    }};
    ChunkGenerator generator(42);

    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    auto blockAt = [&](int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return std::pair{found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                                 Chunk::worldToLocal(z)),
                         found->second->getFluidState(Chunk::worldToLocal(x),
                                                      Chunk::worldToLocalY(y),
                                                      Chunk::worldToLocal(z))};
    };

    for (const auto [x, z] : RIVER_PROBES) {
        const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);
        REQUIRE(surface.hydrology.river);
        REQUIRE_FALSE(surface.hydrology.ocean);
        REQUIRE_FALSE(surface.hydrology.waterfall);
        REQUIRE(surface.hydrology.erosionDepth >= 8.0);
        REQUIRE(surface.waterSurface > surface.terrainHeight);

        const int floorY = static_cast<int>(std::llround(surface.terrainHeight)) - 1;
        const int waterTopY = static_cast<int>(std::ceil(surface.waterSurface)) - 1;
        REQUIRE(isSolid(blockAt(x, floorY, z).first));
        for (int y = floorY + 1; y <= waterTopY; ++y) {
            const auto [block, fluid] = blockAt(x, y, z);
            REQUIRE(block == BlockType::WATER);
            REQUIRE_FALSE(fluid.isFalling());
        }
    }

    constexpr int64_t BANK_X = -12'864;
    constexpr int64_t BANK_Z = 2'695;
    const worldgen::SurfaceSample supportedBank = generator.sampleExactSurface(BANK_X, BANK_Z);
    REQUIRE_FALSE(supportedBank.hydrology.river);
    REQUIRE_FALSE(supportedBank.hydrology.ocean);
    REQUIRE(supportedBank.terrainHeight + 0.01 >= supportedBank.waterSurface);
    const int formerWaterTop = static_cast<int>(std::ceil(supportedBank.waterSurface)) - 1;
    REQUIRE(blockAt(BANK_X, formerWaterTop, BANK_Z).first != BlockType::WATER);
}

TEST_CASE("World regenerates a corrupt cubic save byte identically",
          "[worldgen][advanced][save][determinism]") {
    ComplianceTempDir directory("corrupt_cube");
    const std::filesystem::path region =
        directory.path() / SaveManager::CURRENT_REGIONS_DIRECTORY / "r.0.0";
    std::filesystem::create_directories(region);
    {
        std::ofstream corrupt(region / "c.1.4.1.dat", std::ios::binary | std::ios::trunc);
        const std::array<uint8_t, 7> bytes = {0x52, 0x59, 0x43, 0x48, 0xFF, 0x00, 0x01};
        corrupt.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
    }

    SaveManager saves(directory.path().string());
    World world(42);
    world.setSaveManager(&saves);
    const std::shared_ptr<Chunk> rebuilt = world.getChunk({1, 4, 1});

    ChunkGenerator generator(42);
    Chunk expected(ChunkPos{1, 4, 1});
    generator.generateCube(expected);
    REQUIRE(rebuilt->copyBlocks() == expected.copyBlocks());
    REQUIRE(rebuilt->explicitFluidStates() == expected.explicitFluidStates());
    REQUIRE(rebuilt->generated);
}

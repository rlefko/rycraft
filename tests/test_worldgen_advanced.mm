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
    REQUIRE(actual.hydrology.waterSurface == expected.hydrology.waterSurface);
    REQUIRE(actual.hydrology.discharge == expected.hydrology.discharge);
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

    constexpr double TERMINAL_X = 22'600.0;
    constexpr double TERMINAL_Z = -34'528.0;
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
    constexpr double PORTAL_Z = -15'088.0;
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
    const worldgen::HydrologySample outlet = macro.sampleHydrology(-8'256.0, 3'072.0);
    const worldgen::HydrologySample downstream = macro.sampleHydrology(-8'192.0, 3'136.0);
    REQUIRE(throughLake.lake);
    REQUIRE_FALSE(throughLake.endorheic);
    REQUIRE(outlet.river);
    REQUIRE(downstream.river);
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
    constexpr int64_t MAX_X = -8'192;
    constexpr int64_t MIN_Z = 2'976;
    constexpr int64_t MAX_Z = 3'136;
    constexpr double LAKE_LEVEL = 81.14503479;
    worldgen::MacroGenerationSampler macro(42);
    ChunkGenerator generator(42);

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
                const worldgen::SurfaceSample exactWet = generator.sampleExactSurface(wetX, z);
                const worldgen::SurfaceSample exactDry = generator.sampleExactSurface(x, z);
                INFO("shore at " << x << ", " << z);
                REQUIRE(wet.lake);
                REQUIRE(wet.waterSurface == Catch::Approx(LAKE_LEVEL).margin(1.0e-4));
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
                (void)dryFluid;
                REQUIRE(wetBlock == BlockType::WATER);
                REQUIRE(wetFluid.isSource());
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
    REQUIRE(longestStraightRun <= 48);

    const worldgen::HydrologySample seamBank = macro.sampleHydrology(-8'192.0, 3'040.0);
    REQUIRE_FALSE(seamBank.lakeBank);
    REQUIRE(seamBank.lakeBankInfluence == 0.0);

    double maximumProminence = 0.0;
    for (int64_t z = 3'104; z <= 3'160; z += 4) {
        for (int64_t x = -8'232; x <= -8'168; x += 4) {
            const double height = generator.sampleExactSurface(x, z).terrainHeight;
            double perimeterTotal = 0.0;
            size_t perimeterCount = 0;
            for (int dz = -8; dz <= 8; dz += 4) {
                for (int dx = -8; dx <= 8; dx += 4) {
                    if (std::abs(dx) != 8 && std::abs(dz) != 8)
                        continue;
                    perimeterTotal += generator.sampleExactSurface(x + dx, z + dz).terrainHeight;
                    ++perimeterCount;
                }
            }
            maximumProminence =
                std::max(maximumProminence, height - perimeterTotal / perimeterCount);
        }
    }
    REQUIRE(maximumProminence <= 6.0);
}

TEST_CASE("Lake outlets emit narrow finished falls into their receiving water",
          "[worldgen][advanced][hydrology][lake][waterfall][support][determinism]") {
    constexpr int64_t LAKE_OUTLET_X = -8'272;
    constexpr int64_t LAKE_OUTLET_Z = 3'056;
    constexpr int64_t FALL_X = -8'256;
    constexpr int64_t FALL_Z = 3'072;
    worldgen::MacroGenerationSampler macro(42);

    const worldgen::HydrologySample lake = macro.sampleHydrology(LAKE_OUTLET_X, LAKE_OUTLET_Z);
    const worldgen::HydrologySample fall = macro.sampleHydrology(FALL_X, FALL_Z);
    REQUIRE(lake.lake);
    REQUIRE_FALSE(lake.endorheic);
    REQUIRE_FALSE(fall.ocean);
    REQUIRE(fall.river);
    REQUIRE(fall.waterfall);
    REQUIRE(fall.waterfallAnchor);
    REQUIRE(fall.streamOrder == lake.streamOrder);
    REQUIRE(fall.discharge >= lake.discharge);
    REQUIRE(fall.waterSurface > fall.surfaceElevation);
    REQUIRE(fall.waterSurface < lake.waterSurface);
    REQUIRE(fall.waterfallBottom == Catch::Approx(fall.waterSurface));
    REQUIRE(fall.waterfallTop == Catch::Approx(lake.waterSurface).margin(1.0e-4));
    REQUIRE(fall.waterfallTop - fall.waterfallBottom >= 2.5);
    REQUIRE(fall.waterfallWidth >= 4.0);

    constexpr double DOWNSTREAM_PORTAL_X = -8'192.0;
    constexpr double DOWNSTREAM_PORTAL_Z = 3'136.0;
    constexpr double PORTAL_EPSILON = 1.0e-4;
    const worldgen::HydrologySample portalWest =
        macro.sampleHydrology(DOWNSTREAM_PORTAL_X - PORTAL_EPSILON, DOWNSTREAM_PORTAL_Z);
    const worldgen::HydrologySample portalEast =
        macro.sampleHydrology(DOWNSTREAM_PORTAL_X + PORTAL_EPSILON, DOWNSTREAM_PORTAL_Z);
    REQUIRE(portalWest.river);
    REQUIRE(portalEast.river);
    REQUIRE(portalWest.surfaceElevation ==
            Catch::Approx(portalEast.surfaceElevation).margin(0.001));
    REQUIRE(portalWest.waterSurface == Catch::Approx(portalEast.waterSurface).margin(0.001));
    REQUIRE(portalWest.channelWidth == Catch::Approx(portalEast.channelWidth).margin(0.001));
    REQUIRE(portalWest.channelDepth == Catch::Approx(portalEast.channelDepth).margin(0.001));
    REQUIRE(portalWest.waterSurface >= SEA_LEVEL);
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
        cubes[index] = std::make_unique<Chunk>(ChunkPos{Chunk::worldToChunk(FALL_X),
                                                        Chunk::worldToChunkY(worldY),
                                                        Chunk::worldToChunk(FALL_Z)});
        generator.generateCube(*cubes[index]);
    }
    auto cubeAt = [&](int worldY) -> Chunk& {
        return *cubes[static_cast<size_t>(Chunk::worldToChunkY(worldY) - Chunk::worldToChunkY(62))];
    };
    const int localX = Chunk::worldToLocal(FALL_X);
    const int localZ = Chunk::worldToLocal(FALL_Z);
    const int firstFallingY = static_cast<int>(std::ceil(fall.waterfallBottom)) - 1;
    const int lastFallingY = static_cast<int>(std::ceil(fall.waterfallTop)) - 1;
    for (int worldY = firstFallingY; worldY <= lastFallingY; ++worldY) {
        Chunk& cube = cubeAt(worldY);
        REQUIRE(cube.getBlock(localX, Chunk::worldToLocalY(worldY), localZ) == BlockType::WATER);
        REQUIRE(cube.getFluidState(localX, Chunk::worldToLocalY(worldY), localZ).isFalling());
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
    size_t footprintSamples = 0;
    for (int dz = -16; dz <= 16; ++dz) {
        for (int dx = -16; dx <= 16; ++dx) {
            const worldgen::HydrologySample sample =
                macro.sampleHydrology(FALL_X + dx, FALL_Z + dz);
            if (!sample.waterfall || sample.waterfallTop < sample.waterfallBottom + 0.5)
                continue;
            const double along =
                (dx * fall.flowDirection.x + dz * fall.flowDirection.z) / flowLength;
            minimumAlong = std::min(minimumAlong, along);
            maximumAlong = std::max(maximumAlong, along);
            ++footprintSamples;
        }
    }
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
    if (outside.waterSurface > SEA_LEVEL)
        REQUIRE(outside.hydrology.lake);
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
            const bool active = sample.streamOrder >= 2 && sample.channelDistance < 0.01;
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
    REQUIRE(solver.cacheMetrics().failures == 0);
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
    REQUIRE(delta.endorheic);
    REQUIRE_FALSE(delta.ocean);
    REQUIRE(delta.outlet == worldgen::BasinOutlet::ENDORHEIC);
    REQUIRE(delta.distributaryCount >= 2);
    REQUIRE(delta.distributaryCount <= 4);
    REQUIRE(delta.waterSurface > delta.surfaceElevation);
    REQUIRE(delta.sediment > 0.0);

    solver.clear();
    const worldgen::BasinSample rebuilt = solver.sample(
        4848.0, 704.0, basinFixtureElevation, basinFixtureRainfall, basinFixtureResistance);
    REQUIRE(rebuilt.delta);
    REQUIRE(rebuilt.lake);
    REQUIRE(rebuilt.waterSurface == delta.waterSurface);
    REQUIRE(rebuilt.surfaceElevation == delta.surfaceElevation);
    REQUIRE(rebuilt.distributaryCount == delta.distributaryCount);
    REQUIRE(solver.cacheMetrics().failures == 0);
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
        BiomeLandmark{Biome::MONTANE_GRASSLAND, 26'909, -129},
        BiomeLandmark{Biome::FLOODED_GRASSLAND, -9'003, 21'417},
        BiomeLandmark{Biome::MEDITERRANEAN_WOODLAND, -5'422, 22'586},
        BiomeLandmark{Biome::TEMPERATE_CONIFER_FOREST, -26'355, 29'672},
        BiomeLandmark{Biome::TROPICAL_CONIFER_FOREST, 9'557, 8'126},
        BiomeLandmark{Biome::TROPICAL_DRY_FOREST, 13'138, 9'295},
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
    REQUIRE(routeHash == 0xfbc446b613e77103ULL);

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
    // A large oak rooted at (-27297, 76, -17021) owns both its trunk and the
    // branch crossing the X face between these cubes.
    constexpr ChunkPos FIRST{-1707, 5, -1064};
    constexpr ChunkPos SECOND{-1706, 5, -1064};
    ChunkGenerator forward(42);
    Chunk forwardFirst(FIRST);
    Chunk forwardSecond(SECOND);
    forward.generateCube(forwardFirst);
    forward.generateCube(forwardSecond);

    REQUIRE(forwardFirst.getBlock(15, 3, 3) == BlockType::LOG);
    REQUIRE(forwardSecond.getBlock(0, 3, 3) == BlockType::LOG);

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
        REQUIRE(canopy.canopyRadius > 0);
        REQUIRE(canopy.leafBlock != BlockType::AIR);
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

TEST_CASE("Distant forest cover is deterministic and globally anchored",
          "[worldgen][advanced][flora][far-canopy][lod][determinism]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    ChunkGenerator generator(42);

    REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 4) ==
            generator.collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z));

    for (const int lodStep : {8, 16}) {
        const std::vector<FarCanopy> first =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, lodStep);
        REQUIRE_FALSE(first.empty());
        generator.sampleSurface(4096 + lodStep, -8192 - lodStep);
        generator.clearMacroCaches();
        REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z,
                                                   lodStep) == first);

        std::unordered_set<uint64_t> ids;
        for (const FarCanopy& canopy : first) {
            REQUIRE(canopy.aggregate);
            REQUIRE(canopy.x >= MINIMUM_X);
            REQUIRE(canopy.x < MAXIMUM_X);
            REQUIRE(canopy.z >= MINIMUM_Z);
            REQUIRE(canopy.z < MAXIMUM_Z);
            REQUIRE(canopy.canopyMinimumY <= canopy.canopyMaximumY);
            REQUIRE(canopy.canopyRadius > 0);
            REQUIRE(canopy.logBlock != BlockType::AIR);
            REQUIRE(canopy.leafBlock != BlockType::AIR);
            REQUIRE(ids.insert(canopy.anchorId).second);
        }

        const int64_t splitX = MINIMUM_X + 128;
        const std::vector<FarCanopy> west =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, splitX, MAXIMUM_Z, lodStep);
        const std::vector<FarCanopy> east =
            generator.collectFarCanopiesForLod(splitX, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, lodStep);
        std::unordered_set<uint64_t> splitIds;
        for (const FarCanopy& canopy : west)
            REQUIRE(splitIds.insert(canopy.anchorId).second);
        for (const FarCanopy& canopy : east)
            REQUIRE(splitIds.insert(canopy.anchorId).second);
        REQUIRE(splitIds == ids);
    }
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
            REQUIRE(emittedBlock(canopy.x, canopy.baseY, canopy.z) == canopy.logBlock);
        }

        const int64_t centerX = canopy.x + canopy.canopyOffsetX;
        const int64_t centerZ = canopy.z + canopy.canopyOffsetZ;
        bool foundLeaf = false;
        for (int y = canopy.canopyMinimumY; y <= canopy.canopyMaximumY && !foundLeaf; ++y) {
            for (int64_t z = centerZ - canopy.canopyRadius;
                 z <= centerZ + canopy.canopyRadius && !foundLeaf; ++z) {
                for (int64_t x = centerX - canopy.canopyRadius; x <= centerX + canopy.canopyRadius;
                     ++x) {
                    if (emittedBlock(x, y, z) == canopy.leafBlock) {
                        foundLeaf = true;
                        break;
                    }
                }
            }
        }
        REQUIRE(foundLeaf);
    }
}

TEST_CASE("Generated waterfalls carry finished falling states without runtime settling",
          "[worldgen][advanced][hydrology][fluid]") {
    ChunkGenerator generator(42);
    Chunk waterfall(ChunkPos{Chunk::worldToChunk(-8'256), Chunk::worldToChunkY(74),
                             Chunk::worldToChunk(3'072)});
    generator.generateCube(waterfall);

    size_t fallingCells = 0;
    for (int y = 0; y < CHUNK_EDGE; ++y) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            for (int x = 0; x < CHUNK_EDGE; ++x) {
                const FluidState state = waterfall.getFluidState(x, y, z);
                if (!state.isFalling())
                    continue;
                ++fallingCells;
                REQUIRE(waterfall.getBlock(x, y, z) == BlockType::WATER);
            }
        }
    }
    REQUIRE(fallingCells > 0);
    REQUIRE(waterfall.hasExplicitFluidStates());
}

TEST_CASE("Incised rivers stay supported across cube faces without implicit water walls",
          "[worldgen][advanced][hydrology][river][waterfall][seam][regression]") {
    constexpr std::array<std::pair<int64_t, int64_t>, 4> RIVER_PROBES{{
        {-12'289, 2'653},
        {-12'288, 2'653},
        {-12'289, 2'654},
        {-12'288, 2'654},
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

    constexpr int64_t BANK_X = -12'352;
    constexpr int64_t BANK_Z = 2'653;
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
    const std::filesystem::path region = directory.path() / "regions" / "r.0.0";
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

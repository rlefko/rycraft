#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <future>
#include <limits>
#include <span>
#include <vector>

namespace {

void requireSurfaceSamplesEqual(const worldgen::SurfaceSample& actual,
                                const worldgen::SurfaceSample& expected) {
    REQUIRE(actual.geology.plateId == expected.geology.plateId);
    REQUIRE(actual.geology.crust == expected.geology.crust);
    REQUIRE(actual.geology.boundary == expected.geology.boundary);
    REQUIRE(actual.geology.rock == expected.geology.rock);
    REQUIRE(actual.geology.lithology.primary == expected.geology.lithology.primary);
    REQUIRE(actual.geology.lithology.secondary == expected.geology.lithology.secondary);
    REQUIRE(actual.geology.lithology.transition == expected.geology.lithology.transition);
    REQUIRE(actual.geology.lithology.contactDistance == expected.geology.lithology.contactDistance);
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

    REQUIRE(actual.hydrology.waterBodyId == expected.hydrology.waterBodyId);
    REQUIRE(actual.hydrology.flowDirection.x == expected.hydrology.flowDirection.x);
    REQUIRE(actual.hydrology.flowDirection.z == expected.hydrology.flowDirection.z);
    REQUIRE(actual.hydrology.surfaceElevation == expected.hydrology.surfaceElevation);
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

uint64_t surfaceHash(const worldgen::SurfaceSample& sample) {
    uint64_t hash = 0xCBF29CE484222325ULL;
    const auto add = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 0x100000001B3ULL;
    };
    const auto addDouble = [&add](double value) { add(std::bit_cast<uint64_t>(value)); };
    add(sample.geology.plateId);
    add(static_cast<uint64_t>(sample.geology.crust));
    add(static_cast<uint64_t>(sample.geology.boundary));
    add(static_cast<uint64_t>(sample.geology.lithology.primary));
    add(static_cast<uint64_t>(sample.geology.lithology.secondary));
    addDouble(sample.geology.lithology.transition);
    addDouble(sample.geology.lithology.contactDistance);
    add(sample.hydrology.waterBodyId);
    addDouble(sample.hydrology.surfaceElevation);
    addDouble(sample.hydrology.waterSurface);
    addDouble(sample.hydrology.channelDistance);
    addDouble(sample.hydrology.discharge);
    add(static_cast<uint64_t>(sample.hydrology.ocean));
    add(static_cast<uint64_t>(sample.hydrology.river));
    add(static_cast<uint64_t>(sample.hydrology.lake));
    addDouble(sample.climate.temperatureC);
    addDouble(sample.climate.annualPrecipitationMm);
    addDouble(sample.soil.moisture);
    addDouble(sample.soil.fertility);
    for (float score : sample.suitability.scores)
        add(std::bit_cast<uint32_t>(score));
    add(static_cast<uint64_t>(sample.biome.primary));
    add(static_cast<uint64_t>(sample.biome.secondary));
    addDouble(sample.biome.transition);
    add(static_cast<uint64_t>(sample.ecotopes));
    addDouble(sample.terrainHeight);
    addDouble(sample.waterSurface);
    addDouble(sample.slope);
    return hash;
}

double splineProbe(const worldgen::MacroGenerationSampler& sampler, double x, double z) {
    return sampler.sampleSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1).soil.fertility;
}

double farClimateSplineProbe(const worldgen::MacroGenerationSampler& sampler, double x, double z) {
    return sampler.sampleSurface(x, z, worldgen::SurfaceFootprint::BLOCK_16)
               .climate.annualPrecipitationMm /
           3'600.0;
}

void requireC2Face(const worldgen::MacroGenerationSampler& sampler, double x, double z,
                   bool xFace) {
    constexpr double STEP = 0.25;
    const auto sample = [&](double offset) {
        return xFace ? splineProbe(sampler, x + offset, z) : splineProbe(sampler, x, z + offset);
    };
    const double below2 = sample(-2.0 * STEP);
    const double below = sample(-STEP);
    const double center = sample(0.0);
    const double above = sample(STEP);
    const double above2 = sample(2.0 * STEP);

    const double leftValue = 2.0 * below - below2;
    const double rightValue = 2.0 * above - above2;
    const double leftFirst = (center - below) / STEP;
    const double rightFirst = (above - center) / STEP;
    const double leftSecond = (center - 2.0 * below + below2) / (STEP * STEP);
    const double rightSecond = (above2 - 2.0 * above + center) / (STEP * STEP);
    REQUIRE(leftValue == Catch::Approx(center).margin(0.001));
    REQUIRE(rightValue == Catch::Approx(center).margin(0.001));
    REQUIRE(leftFirst == Catch::Approx(rightFirst).margin(0.005));
    REQUIRE(leftSecond == Catch::Approx(rightSecond).margin(0.005));
}

void requireFarClimateC2Face(const worldgen::MacroGenerationSampler& sampler, double x, double z,
                             bool xFace) {
    constexpr double STEP = 0.25;
    const auto sample = [&](double offset) {
        return xFace ? farClimateSplineProbe(sampler, x + offset, z)
                     : farClimateSplineProbe(sampler, x, z + offset);
    };
    const double below2 = sample(-2.0 * STEP);
    const double below = sample(-STEP);
    const double center = sample(0.0);
    const double above = sample(STEP);
    const double above2 = sample(2.0 * STEP);

    const double leftValue = 2.0 * below - below2;
    const double rightValue = 2.0 * above - above2;
    const double leftFirst = (center - below) / STEP;
    const double rightFirst = (above - center) / STEP;
    const double leftSecond = (center - 2.0 * below + below2) / (STEP * STEP);
    const double rightSecond = (above2 - 2.0 * above + center) / (STEP * STEP);
    REQUIRE(leftValue == Catch::Approx(center).margin(0.001));
    REQUIRE(rightValue == Catch::Approx(center).margin(0.001));
    REQUIRE(leftFirst == Catch::Approx(rightFirst).margin(0.005));
    REQUIRE(leftSecond == Catch::Approx(rightSecond).margin(0.005));
}

double controlViewProbe(const worldgen::MacroControlView& view, double x, double z) {
    worldgen::SurfaceSample sample;
    sample.terrainHeight = 96.0;
    view.reconstructContinuous(x, z, sample);
    return sample.soil.fertility;
}

void requireControlViewC2Face(const worldgen::MacroControlView& belowView,
                              const worldgen::MacroControlView& aboveView, double x, double z,
                              bool xFace) {
    constexpr double STEP = 0.25;
    const auto below = [&](double offset) {
        return xFace ? controlViewProbe(belowView, x + offset, z)
                     : controlViewProbe(belowView, x, z + offset);
    };
    const auto above = [&](double offset) {
        return xFace ? controlViewProbe(aboveView, x + offset, z)
                     : controlViewProbe(aboveView, x, z + offset);
    };
    const double below2 = below(-2.0 * STEP);
    const double below1 = below(-STEP);
    const double centerBelow = below(0.0);
    const double centerAbove = above(0.0);
    const double above1 = above(STEP);
    const double above2 = above(2.0 * STEP);

    REQUIRE(centerBelow == centerAbove);
    const double leftFirst = (centerBelow - below1) / STEP;
    const double rightFirst = (above1 - centerAbove) / STEP;
    const double leftSecond = (centerBelow - 2.0 * below1 + below2) / (STEP * STEP);
    const double rightSecond = (above2 - 2.0 * above1 + centerAbove) / (STEP * STEP);
    REQUIRE(leftFirst == Catch::Approx(rightFirst).margin(0.005));
    REQUIRE(leftSecond == Catch::Approx(rightSecond).margin(0.005));
}

void requireSingleFlightCache(worldgen::SurfaceFootprint footprint, bool farClimate) {
    worldgen::MacroGenerationSampler sampler(42);
    std::promise<void> release;
    const std::shared_future<void> gate = release.get_future().share();
    std::array<std::future<worldgen::SurfaceSample>, 4> futures;
    for (auto& future : futures) {
        future = std::async(std::launch::async, [&sampler, gate, footprint] {
            gate.wait();
            return sampler.sampleSurface(11.0, 13.0, footprint);
        });
    }
    release.set_value();
    const worldgen::SurfaceSample expected = futures.front().get();
    for (size_t index = 1; index < futures.size(); ++index)
        requireSurfaceSamplesEqual(futures[index].get(), expected);

    const auto selectedMetrics = [&] {
        return farClimate ? sampler.farClimateControlCacheMetrics()
                          : sampler.macroControlCacheMetrics();
    };
    const worldgen::MacroControlCacheMetrics built = selectedMetrics();
    REQUIRE(built.capacity == (farClimate ? worldgen::FAR_CLIMATE_CONTROL_CACHE_CAPACITY
                                          : worldgen::MACRO_CONTROL_CACHE_CAPACITY));
    REQUIRE(built.byteBudget == (farClimate ? worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET
                                            : worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET));
    REQUIRE(built.entries == 1);
    REQUIRE(built.bytes <= built.byteBudget);
    REQUIRE(built.misses == 1);
    REQUIRE(built.builds == 1);
    REQUIRE(built.hits == futures.size() - 1);
    REQUIRE(built.singleFlightWaits >= 1);
    REQUIRE(built.activeBuilds == 0);
    REQUIRE(built.peakBuilds == 1);
    REQUIRE((farClimate ? sampler.macroControlCacheMetrics().entries
                        : sampler.farClimateControlCacheMetrics().entries) == 0);

    sampler.clearMacroControlCache();
    const worldgen::MacroControlCacheMetrics cleared = selectedMetrics();
    REQUIRE(cleared.entries == 0);
    REQUIRE(cleared.bytes == 0);
    requireSurfaceSamplesEqual(sampler.sampleSurface(11.0, 13.0, footprint), expected);
    REQUIRE(selectedMetrics().builds == 2);
}

void requireUncacheableControlTileIsSingleFlight(worldgen::SurfaceFootprint footprint,
                                                 bool farClimate) {
    constexpr size_t CALLER_COUNT = 4;
    worldgen::MacroGenerationSampler sampler(
        42, farClimate ? worldgen::MACRO_CONTROL_CACHE_CAPACITY : 8,
        farClimate ? worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET : 1,
        farClimate ? 8 : worldgen::FAR_CLIMATE_CONTROL_CACHE_CAPACITY,
        farClimate ? 1 : worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    std::promise<void> release;
    const std::shared_future<void> gate = release.get_future().share();
    std::array<std::future<worldgen::SurfaceSample>, CALLER_COUNT> futures;
    for (auto& future : futures) {
        future = std::async(std::launch::async, [&sampler, gate, footprint] {
            gate.wait();
            return sampler.sampleSurface(11.0, 13.0, footprint);
        });
    }
    release.set_value();
    const worldgen::SurfaceSample expected = futures.front().get();
    for (size_t index = 1; index < futures.size(); ++index)
        requireSurfaceSamplesEqual(futures[index].get(), expected);

    const auto selectedMetrics = [&] {
        return farClimate ? sampler.farClimateControlCacheMetrics()
                          : sampler.macroControlCacheMetrics();
    };
    const worldgen::MacroControlCacheMetrics shared = selectedMetrics();
    REQUIRE(shared.byteBudget == 1);
    REQUIRE(shared.entries == 0);
    REQUIRE(shared.bytes == 0);
    REQUIRE(shared.activeBuilds == 0);
    REQUIRE(shared.peakBuilds == 1);
    REQUIRE(shared.builds == shared.misses);
    REQUIRE(shared.hits + shared.misses == CALLER_COUNT);
    REQUIRE(shared.hits >= 1);
    REQUIRE(shared.singleFlightWaits == shared.hits);
    REQUIRE(shared.builds < CALLER_COUNT);

    requireSurfaceSamplesEqual(sampler.sampleSurface(11.0, 13.0, footprint), expected);
    const worldgen::MacroControlCacheMetrics rebuilt = selectedMetrics();
    REQUIRE(rebuilt.byteBudget == 1);
    REQUIRE(rebuilt.entries == 0);
    REQUIRE(rebuilt.bytes == 0);
    REQUIRE(rebuilt.builds == shared.builds + 1);
    REQUIRE(rebuilt.misses == shared.misses + 1);
}

} // namespace

TEST_CASE("Macro control grids exactly match point sampling across tile faces",
          "[worldgen][macro-control][bulk][determinism][seam]") {
    worldgen::MacroGenerationSampler sampler(764891);
    constexpr int SAMPLE_EDGE = 3;
    constexpr int SPACING = 8;
    constexpr int64_t ORIGIN = -72;
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> grid;

    for (const worldgen::SurfaceFootprint footprint : {
             worldgen::SurfaceFootprint::BLOCK_1,
             worldgen::SurfaceFootprint::BLOCK_2,
             worldgen::SurfaceFootprint::BLOCK_4,
             worldgen::SurfaceFootprint::BLOCK_8,
             worldgen::SurfaceFootprint::BLOCK_16,
         }) {
        sampler.sampleSurfaceGrid(ORIGIN, ORIGIN, SPACING, SAMPLE_EDGE, footprint, grid);
        for (int sampleZ = 0; sampleZ < SAMPLE_EDGE; ++sampleZ) {
            for (int sampleX = 0; sampleX < SAMPLE_EDGE; ++sampleX) {
                const double x = static_cast<double>(ORIGIN + sampleX * SPACING);
                const double z = static_cast<double>(ORIGIN + sampleZ * SPACING);
                CAPTURE(static_cast<int>(footprint), x, z);
                requireSurfaceSamplesEqual(
                    grid[static_cast<size_t>(sampleZ * SAMPLE_EDGE + sampleX)],
                    sampler.sampleSurface(x, z, footprint));
            }
        }
    }
}

TEST_CASE("Macro control reconstruction is C2 across positive and negative tile faces",
          "[worldgen][macro-control][spline][c2][seam]") {
    worldgen::MacroGenerationSampler sampler(42);
    requireC2Face(sampler, 64.0, 21.25, true);
    requireC2Face(sampler, -64.0, -19.5, true);
    requireC2Face(sampler, 17.75, 64.0, false);
    requireC2Face(sampler, -23.25, -64.0, false);
}

TEST_CASE("Exact control views preserve C2 derivatives across storage and tile faces",
          "[worldgen][macro-control][column-plan][spline][c2][seam]") {
    STATIC_REQUIRE(sizeof(worldgen::MacroControlView) <= 2 * sizeof(void*));
    worldgen::MacroGenerationSampler sampler(42);

    const worldgen::MacroControlView center = sampler.controlView({0, 0});
    requireControlViewC2Face(center, center, 8.0, 21.25, true);
    requireControlViewC2Face(center, center, 17.75, 8.0, false);

    const worldgen::MacroControlView west = sampler.controlView({3, 0});
    const worldgen::MacroControlView east = sampler.controlView({4, 0});
    requireControlViewC2Face(west, east, 64.0, 21.25, true);

    const worldgen::MacroControlView north = sampler.controlView({-1, -1});
    const worldgen::MacroControlView south = sampler.controlView({-1, 0});
    requireControlViewC2Face(north, south, -23.25, 0.0, false);
}

TEST_CASE("Far climate reconstruction is C2 across positive and negative tile faces",
          "[worldgen][macro-control][far-climate][spline][c2][seam]") {
    worldgen::MacroGenerationSampler sampler(42);
    requireFarClimateC2Face(sampler, 256.0, 21.25, true);
    requireFarClimateC2Face(sampler, -256.0, -19.5, true);
    requireFarClimateC2Face(sampler, 17.75, 256.0, false);
    requireFarClimateC2Face(sampler, -23.25, -256.0, false);
}

TEST_CASE("Far climate grids bound control and shoreline construction work",
          "[worldgen][macro-control][far-climate][bulk][cache][work-limit]") {
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_TILE_EDGE == 256);
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_SPACING == 128);
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_SAMPLE_COUNT == 25);
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_CACHE_CAPACITY == 4'096);
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET == 32ull * 1024 * 1024);

    worldgen::MacroGenerationSampler sampler(764891);
    constexpr int SAMPLE_EDGE = 17;
    constexpr int SPACING = 16;
    constexpr int64_t ORIGIN_X = 23'040;
    constexpr int64_t ORIGIN_Z = -111'872;
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> grid{};
    sampler.sampleSurfaceGrid(ORIGIN_X, ORIGIN_Z, SPACING, SAMPLE_EDGE,
                              worldgen::SurfaceFootprint::BLOCK_16, grid);

    const worldgen::MacroControlCacheMetrics farCache = sampler.farClimateControlCacheMetrics();
    REQUIRE(farCache.capacity == worldgen::FAR_CLIMATE_CONTROL_CACHE_CAPACITY);
    REQUIRE(farCache.byteBudget == worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    REQUIRE(farCache.entries == 4);
    REQUIRE(farCache.builds == 4);
    REQUIRE(farCache.activeBuilds == 0);
    REQUIRE(farCache.bytes <= farCache.byteBudget);
    REQUIRE(sampler.macroControlCacheMetrics().entries == 0);
    REQUIRE(sampler.basinCacheMetrics().shorelineEntries == 0);
    REQUIRE(sampler.basinCacheMetrics().shorelineBuilds == 0);
}

TEST_CASE("Far climate cache eviction preserves order-independent surfaces",
          "[worldgen][macro-control][far-climate][cache][lru][determinism]") {
    constexpr std::array<ColumnPos, 3> COORDINATES = {ColumnPos{11, 13}, ColumnPos{267, 9},
                                                      ColumnPos{523, 17}};
    worldgen::MacroGenerationSampler forward(42, worldgen::MACRO_CONTROL_CACHE_CAPACITY,
                                             worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET, 2,
                                             worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    std::array<uint64_t, COORDINATES.size()> hashes{};
    for (size_t index = 0; index < COORDINATES.size(); ++index) {
        hashes[index] = surfaceHash(forward.sampleSurface(static_cast<double>(COORDINATES[index].x),
                                                          static_cast<double>(COORDINATES[index].z),
                                                          worldgen::SurfaceFootprint::BLOCK_16));
    }
    const worldgen::MacroControlCacheMetrics evicted = forward.farClimateControlCacheMetrics();
    REQUIRE(evicted.entries == 2);
    REQUIRE(evicted.evictions == 1);
    REQUIRE(evicted.bytes <= evicted.byteBudget);
    REQUIRE(surfaceHash(forward.sampleSurface(11.0, 13.0, worldgen::SurfaceFootprint::BLOCK_16)) ==
            hashes[0]);
    REQUIRE(forward.farClimateControlCacheMetrics().builds == 4);

    worldgen::MacroGenerationSampler reverse(42, worldgen::MACRO_CONTROL_CACHE_CAPACITY,
                                             worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET, 2,
                                             worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    for (size_t reverseIndex = COORDINATES.size(); reverseIndex-- > 0;) {
        const ColumnPos coordinate = COORDINATES[reverseIndex];
        CAPTURE(reverseIndex, coordinate.x, coordinate.z);
        REQUIRE(surfaceHash(reverse.sampleSurface(
                    static_cast<double>(coordinate.x), static_cast<double>(coordinate.z),
                    worldgen::SurfaceFootprint::BLOCK_16)) == hashes[reverseIndex]);
    }
}

TEST_CASE("Main macro control cache is single-flight bounded and clearable",
          "[worldgen][macro-control][cache][single-flight][concurrency]") {
    STATIC_REQUIRE(worldgen::MACRO_CONTROL_CACHE_CAPACITY == 1'024);
    STATIC_REQUIRE(worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET == 128ull * 1024 * 1024);
    requireSingleFlightCache(worldgen::SurfaceFootprint::BLOCK_1, false);
}

TEST_CASE("Far climate control cache is single-flight bounded and clearable",
          "[worldgen][macro-control][far-climate][cache][single-flight][concurrency]") {
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_CACHE_CAPACITY == 4'096);
    STATIC_REQUIRE(worldgen::FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET == 32ull * 1024 * 1024);
    requireSingleFlightCache(worldgen::SurfaceFootprint::BLOCK_16, true);
}

TEST_CASE("Uncacheable control tiles remain single-flight without exceeding their byte budget",
          "[worldgen][macro-control][cache][single-flight][concurrency][work-limit]") {
    requireUncacheableControlTileIsSingleFlight(worldgen::SurfaceFootprint::BLOCK_1, false);
    requireUncacheableControlTileIsSingleFlight(worldgen::SurfaceFootprint::BLOCK_16, true);
}

TEST_CASE("Macro control eviction preserves order-independent surface hashes",
          "[worldgen][macro-control][cache][lru][determinism]") {
    constexpr std::array<ColumnPos, 3> COORDINATES = {ColumnPos{11, 13}, ColumnPos{75, 9},
                                                      ColumnPos{139, 17}};
    worldgen::MacroGenerationSampler forward(42, 2, worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET);
    std::array<uint64_t, COORDINATES.size()> hashes{};
    for (size_t index = 0; index < COORDINATES.size(); ++index) {
        hashes[index] = surfaceHash(forward.sampleSurface(static_cast<double>(COORDINATES[index].x),
                                                          static_cast<double>(COORDINATES[index].z),
                                                          worldgen::SurfaceFootprint::BLOCK_1));
    }
    const worldgen::MacroControlCacheMetrics evicted = forward.macroControlCacheMetrics();
    REQUIRE(evicted.entries == 2);
    REQUIRE(evicted.evictions == 1);
    REQUIRE(evicted.bytes <= evicted.byteBudget);
    REQUIRE(surfaceHash(forward.sampleSurface(11.0, 13.0, worldgen::SurfaceFootprint::BLOCK_1)) ==
            hashes[0]);
    REQUIRE(forward.macroControlCacheMetrics().builds == 4);

    worldgen::MacroGenerationSampler reverse(42, 2, worldgen::MACRO_CONTROL_CACHE_BYTE_BUDGET);
    for (size_t reverseIndex = COORDINATES.size(); reverseIndex-- > 0;) {
        const ColumnPos coordinate = COORDINATES[reverseIndex];
        CAPTURE(reverseIndex, coordinate.x, coordinate.z);
        REQUIRE(surfaceHash(reverse.sampleSurface(
                    static_cast<double>(coordinate.x), static_cast<double>(coordinate.z),
                    worldgen::SurfaceFootprint::BLOCK_1)) == hashes[reverseIndex]);
    }
}

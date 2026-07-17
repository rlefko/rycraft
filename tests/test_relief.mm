#include <catch2/catch_test_macros.hpp>

#include "world/chunk_generator.hpp"
#include "world/macro_generation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

TEST_CASE("Tectonic relief is deterministic continuous and vertically bounded",
          "[worldgen][geology][relief][continuity][determinism]") {
    worldgen::MacroGenerationSampler sampler(42);
    constexpr std::array<std::pair<int64_t, int64_t>, 8> FIXTURES = {{
        {-81'920, 126'976},
        {-75'008, 88'064},
        {-22'528, 2'816},
        {77'568, 45'824},
        {-23'296, -16'704},
        {-8'576, -3'520},
        {0, 0},
        {-1, -1},
    }};
    constexpr std::array<std::pair<int64_t, int64_t>, 4> CARDINAL = {{
        {1, 0},
        {-1, 0},
        {0, 1},
        {0, -1},
    }};

    for (const auto [x, z] : FIXTURES) {
        const double elevation = sampler.preliminaryElevation(x, z);
        REQUIRE(std::isfinite(elevation));
        REQUIRE(elevation >= -112.0);
        REQUIRE(elevation <= 480.0);
        REQUIRE(sampler.preliminaryElevation(x, z) == elevation);
        for (const auto [dx, dz] : CARDINAL) {
            const double neighbor = sampler.preliminaryElevation(x + dx, z + dz);
            REQUIRE(std::abs(neighbor - elevation) < 4.0);
        }
    }

    ChunkGenerator generator(42);
    for (const auto [x, z] : FIXTURES) {
        const worldgen::SurfaceSample emitted = generator.sampleFarSurface(x, z);
        REQUIRE(std::isfinite(emitted.terrainHeight));
        REQUIRE(emitted.terrainHeight >= -112.0);
        REQUIRE(emitted.terrainHeight <= 480.0);
    }
}

TEST_CASE("Plate triple points retain continuous boundary motion fields",
          "[worldgen][geology][relief][plate][seam]") {
    worldgen::MacroGenerationSampler sampler(42);
    const worldgen::GeologySample west = sampler.sampleGeology(-8'576.0, -3'520.0);
    const worldgen::GeologySample east = sampler.sampleGeology(-8'575.0, -3'520.0);

    REQUIRE(west.plateId != east.plateId);
    REQUIRE(std::abs(west.uplift - east.uplift) < 0.01);
    REQUIRE(std::abs(west.rift - east.rift) < 0.01);
    REQUIRE(std::abs(west.faultStrength - east.faultStrength) < 0.01);
    REQUIRE(std::abs(west.continentalFraction - east.continentalFraction) < 0.01);
    REQUIRE(std::abs(sampler.preliminaryElevation(-8'576.0, -3'520.0) -
                     sampler.preliminaryElevation(-8'575.0, -3'520.0)) < 1.0);

    const worldgen::GeologySample first = sampler.sampleGeology(-23'296.0, -16'704.0);
    const worldgen::GeologySample second = sampler.sampleGeology(-23'294.0, -16'704.0);
    REQUIRE(std::abs(first.uplift - second.uplift) < 0.01);
    REQUIRE(std::abs(first.rift - second.rift) < 0.01);
    REQUIRE(std::abs(first.faultStrength - second.faultStrength) < 0.01);
    REQUIRE(std::abs(sampler.preliminaryElevation(-23'296.0, -16'704.0) -
                     sampler.preliminaryElevation(-23'294.0, -16'704.0)) < 1.0);
}

TEST_CASE("Continuous tectonic signals form broad elevated massifs",
          "[worldgen][geology][relief][mountain][hotspot]") {
    worldgen::MacroGenerationSampler sampler(42);

    const worldgen::GeologySample uplift = sampler.sampleGeology(-81'920.0, 126'976.0);
    const worldgen::GeologySample fault = sampler.sampleGeology(-81'760.0, 88'480.0);
    const worldgen::GeologySample hotspot = sampler.sampleGeology(-22'528.0, 2'816.0);
    const worldgen::GeologySample rift = sampler.sampleGeology(82'080.0, 50'272.0);
    REQUIRE(uplift.uplift > 0.90);
    REQUIRE(fault.faultStrength > 0.80);
    REQUIRE(hotspot.hotspotInfluence > 0.90);
    REQUIRE(rift.rift > 0.50);
    REQUIRE(sampler.preliminaryElevation(-81'920.0, 126'976.0) > 300.0);
    REQUIRE(sampler.preliminaryElevation(-81'760.0, 88'480.0) > 105.0);
    REQUIRE(sampler.preliminaryElevation(-22'528.0, 2'816.0) > 190.0);
    REQUIRE(sampler.preliminaryElevation(82'080.0, 50'272.0) > 105.0);

    ChunkGenerator generator(42);
    REQUIRE(generator.sampleFarSurface(-81'920, 126'976).terrainHeight > 300.0);

    std::size_t elevatedSamples = 0;
    for (int dz = -512; dz <= 512; dz += 256) {
        for (int dx = -512; dx <= 512; dx += 256) {
            elevatedSamples += sampler.preliminaryElevation(-81'920 + dx, 126'976 + dz) > 180.0;
        }
    }
    REQUIRE(elevatedSamples >= 15);
}

TEST_CASE("High discharge carves a supported broad valley deterministically",
          "[worldgen][hydrology][relief][erosion][determinism]") {
    worldgen::MacroGenerationSampler sampler(42);
    const worldgen::HydrologySample valley = sampler.sampleHydrology(4'000.0, 5'280.0);
    const worldgen::HydrologySample shoulder = sampler.sampleHydrology(3'872.0, 5'152.0);
    const worldgen::HydrologySample lowFlow = sampler.sampleHydrology(77'568.0, 45'824.0);

    REQUIRE(valley.river);
    REQUIRE_FALSE(valley.ocean);
    REQUIRE(valley.discharge > 18'000.0);
    REQUIRE(valley.erosionDepth > 20.0);
    REQUIRE(valley.waterSurface > valley.surfaceElevation);
    REQUIRE(shoulder.surfaceElevation > valley.surfaceElevation + 25.0);
    REQUIRE(shoulder.erosionDepth < 1.0);
    REQUIRE(valley.erosionDepth > lowFlow.erosionDepth + 8.0);
    REQUIRE(valley.surfaceElevation >= -112.0);
    REQUIRE(valley.surfaceElevation <= 480.0);

    sampler.clearBasinCache();
    const worldgen::HydrologySample rebuilt = sampler.sampleHydrology(4'000.0, 5'280.0);
    REQUIRE(rebuilt.surfaceElevation == valley.surfaceElevation);
    REQUIRE(rebuilt.waterSurface == valley.waterSurface);
    REQUIRE(rebuilt.discharge == valley.discharge);
    REQUIRE(rebuilt.erosionDepth == valley.erosionDepth);
    REQUIRE(rebuilt.streamOrder == valley.streamOrder);
}

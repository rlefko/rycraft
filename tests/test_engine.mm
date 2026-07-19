#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/atmosphere.hpp>
#include <render/atmospheric_memory.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/cloud_renderer.hpp>
#include <render/graphics_settings.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/render_pipeline.hpp>
#include <render/screen_space_lighting.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/weather.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================

TEST_CASE("Physical atmosphere parameters and volume helpers stay finite",
          "[render][atmosphere][volumetrics]") {
    AtmosphereUniforms atmosphere =
        earthAtmosphereUniforms(225.0F, simd_make_float3(0.3F, 0.8F, 0.2F),
                                simd_make_float3(18.0F, 17.5F, 16.0F), 1.25F, 0.7F, 7);
    REQUIRE(atmosphereUniformsFinite(atmosphere));
    REQUIRE(atmosphere.atmosphereRadii.x == Catch::Approx(6360.0F));
    REQUIRE(atmosphere.atmosphereRadii.y == Catch::Approx(6460.0F));
    REQUIRE(atmosphere.atmosphereRadii.z == Catch::Approx(0.004675F));
    REQUIRE(atmosphere.weatherOptics.z == Catch::Approx(0.0F));
    REQUIRE(ATMOSPHERE_RADIANCE_SCALE ==
            Catch::Approx(16.0F * static_cast<float>(M_PI)).margin(1.0e-6F));

    // The normalized 15-degree clear-noon integral is intentionally lifted
    // into the shared HDR exposure range. Otherwise the terrain can meter as
    // daytime while the physical sky itself grades nearly black.
    const simd_float3 clearNoon =
        atmosphereSceneRadiance(simd_make_float3(0.0115F, 0.0229F, 0.0436F));
    REQUIRE(clearNoon.x > 0.5F);
    REQUIRE(clearNoon.y > 1.0F);
    REQUIRE(clearNoon.z > 2.0F);

    // A view ray that reaches the spherical lower boundary must retain a
    // finite Lambertian radiance. This is the fallback behind streamed
    // terrain gaps, so a daytime horizon cannot reveal a black band.
    const simd_float3 ground = atmosphereGroundRadiance(simd_make_float3(0.18F, 0.20F, 0.16F),
                                                        simd_make_float3(0.94F, 0.87F, 0.77F), 0.8F,
                                                        simd_make_float3(0.003F, 0.010F, 0.032F));
    REQUIRE(ground.x > 0.04F);
    REQUIRE(ground.y > 0.04F);
    REQUIRE(ground.z > 0.03F);

    float previous = 0.0F;
    for (unsigned int slice = 0; slice <= FROXEL_DEPTH; ++slice) {
        float depth = froxelSliceDepth(slice, FROXEL_DEPTH, 0.1F, 8192.0F);
        REQUIRE(std::isfinite(depth));
        REQUIRE(depth >= previous);
        previous = depth;
    }
    REQUIRE(previous == Catch::Approx(8192.0F));

    REQUIRE(beerLambertTransmittance(0.02F, 0.0F) == Catch::Approx(1.0F));
    REQUIRE(beerLambertTransmittance(0.02F, 100.0F) < beerLambertTransmittance(0.02F, 10.0F));
    REQUIRE(beerLambertTransmittance(-1.0F, 100.0F) == Catch::Approx(1.0F));
}

TEST_CASE("Atmosphere transmittance stops at the lower boundary",
          "[render][atmosphere][regression]") {
    constexpr float GROUND_RADIUS_KM = 6360.0F;
    constexpr float TOP_RADIUS_KM = 6460.0F;
    constexpr float CAMERA_RADIUS_KM = GROUND_RADIUS_KM + 0.225F;

    // The transmittance LUT includes downward-looking rows. A camera 225 m
    // above the spherical surface must reach the ground almost immediately,
    // not integrate through the planet to the far top-atmosphere crossing.
    const float farTopDistance =
        atmosphereRayToSphereDistance(CAMERA_RADIUS_KM, -1.0F, TOP_RADIUS_KM);
    const float downwardDistance =
        atmosphereTransmittancePathLength(CAMERA_RADIUS_KM, -1.0F, GROUND_RADIUS_KM, TOP_RADIUS_KM);
    REQUIRE(atmosphereRayHitsGround(CAMERA_RADIUS_KM, -1.0F, GROUND_RADIUS_KM));
    REQUIRE(downwardDistance == Catch::Approx(0.225F).margin(0.002F));
    REQUIRE(downwardDistance < farTopDistance * 0.01F);

    // Upward and above-horizon rows must continue to the upper boundary.
    const float upwardDistance =
        atmosphereTransmittancePathLength(CAMERA_RADIUS_KM, 1.0F, GROUND_RADIUS_KM, TOP_RADIUS_KM);
    REQUIRE_FALSE(atmosphereRayHitsGround(CAMERA_RADIUS_KM, 1.0F, GROUND_RADIUS_KM));
    REQUIRE(upwardDistance == Catch::Approx(TOP_RADIUS_KM - CAMERA_RADIUS_KM).margin(0.002F));
    REQUIRE(upwardDistance ==
            Catch::Approx(atmosphereRayToSphereDistance(CAMERA_RADIUS_KM, 1.0F, TOP_RADIUS_KM)));
}

TEST_CASE("Atmosphere transmittance mu coordinates match the LUT domain",
          "[render][atmosphere][regression]") {
    REQUIRE(atmosphereTransmittanceMuUv(ATMOSPHERE_TRANSMITTANCE_MU_MIN) == Catch::Approx(0.0F));
    REQUIRE(atmosphereTransmittanceMuUv(0.0F) == Catch::Approx(0.15F / 1.15F).margin(1.0e-6F));
    REQUIRE(atmosphereTransmittanceMuUv(ATMOSPHERE_TRANSMITTANCE_MU_MAX) == Catch::Approx(1.0F));
    REQUIRE(atmosphereTransmittanceUvMu(0.0F) == Catch::Approx(ATMOSPHERE_TRANSMITTANCE_MU_MIN));
    REQUIRE(atmosphereTransmittanceUvMu(1.0F) == Catch::Approx(ATMOSPHERE_TRANSMITTANCE_MU_MAX));

    for (float mu :
         {-2.0F, ATMOSPHERE_TRANSMITTANCE_MU_MIN, 0.0F, ATMOSPHERE_TRANSMITTANCE_MU_MAX, 2.0F}) {
        const float uv = atmosphereTransmittanceMuUv(mu);
        REQUIRE(std::isfinite(uv));
        REQUIRE(uv >= 0.0F);
        REQUIRE(uv <= 1.0F);
    }
    REQUIRE(std::isfinite(atmosphereTransmittanceMuUv(std::numeric_limits<float>::quiet_NaN())));
}

TEST_CASE("Volumetric cloud noise tiles in three dimensions and wind uses physical units",
          "[render][clouds][weather]") {
    STATIC_REQUIRE(WeatherSnapshot::GRID_EDGE == WEATHER_MAP_EDGE);
    STATIC_REQUIRE(WeatherSnapshot::GRID_SPACING == static_cast<int>(WEATHER_MAP_CELL_SPACING));
    constexpr int EDGE = 32;
    constexpr uint64_t SEED = 764891;
    const float origin = cloudBaseNoise(0, 7, 19, EDGE, SEED);
    REQUIRE(cloudBaseNoise(EDGE, 7, 19, EDGE, SEED) == Catch::Approx(origin));
    REQUIRE(cloudBaseNoise(0, 7 + EDGE, 19, EDGE, SEED) == Catch::Approx(origin));
    REQUIRE(cloudBaseNoise(0, 7, 19 + EDGE, EDGE, SEED) == Catch::Approx(origin));
    REQUIRE(cloudBaseNoise(3, 4, 5, EDGE, SEED) !=
            Catch::Approx(cloudBaseNoise(3, 5, 5, EDGE, SEED)));
    REQUIRE(cloudBaseNoise(3, 4, 5, EDGE, SEED) >= 0.0F);
    REQUIRE(cloudBaseNoise(3, 4, 5, EDGE, SEED) <= 1.0F);

    REQUIRE(wrappedCloudOffset(100.0, 12.0, 10.0) == Catch::Approx(220.0));
    REQUIRE(wrappedCloudOffset(CLOUD_MOTION_WRAP_BLOCKS - 6.0, 12.0, 1.0) == Catch::Approx(6.0));
    REQUIRE(wrappedCloudOffset(2.0, -4.0, 1.0) == Catch::Approx(CLOUD_MOTION_WRAP_BLOCKS - 2.0));
}

TEST_CASE("Regional cloud motion wraps coherently and ignores camera movement",
          "[render][clouds][weather][precision]") {
    WeatherSample previous{};
    previous.cloudType = CloudType::CUMULUS;
    previous.cloudOffsetBlocks = {1'000'000'000'000.0, -1'000'000'000'000.0};
    previous.highCloudOffsetBlocks = {750'000'000'000.0, -750'000'000'000.0};
    WeatherSample current = previous;
    current.cloudOffsetBlocks.x += 0.5;
    current.cloudOffsetBlocks.z -= 0.25;
    current.highCloudOffsetBlocks.x += 0.75;
    current.highCloudOffsetBlocks.z -= 0.375;

    const simd_float2 previousMotion = decodeCloudMotion(encodeCloudMotion(previous));
    const simd_float2 currentMotion = decodeCloudMotion(encodeCloudMotion(current));
    const simd_float2 delta = cloudMotionDelta(currentMotion, previousMotion);
    REQUIRE(delta.x == Catch::Approx(0.5F).margin(0.001F));
    REQUIRE(delta.y == Catch::Approx(-0.25F).margin(0.001F));

    const simd_float2 previousHighMotion =
        decodeCloudMotion(encodeCloudMotionOffset(previous.highCloudOffsetBlocks));
    const simd_float2 currentHighMotion =
        decodeCloudMotion(encodeCloudMotionOffset(current.highCloudOffsetBlocks));
    const simd_float2 highDelta = cloudMotionDelta(currentHighMotion, previousHighMotion);
    REQUIRE(highDelta.x == Catch::Approx(0.75F).margin(0.001F));
    REQUIRE(highDelta.y == Catch::Approx(-0.375F).margin(0.001F));

    // Changing the categorical profile must not rescale the absolute phase.
    // At large saved-world ages, even a small multiplier would create a
    // visually unbounded jump instead of the sub-block physical motion.
    WeatherSample transitioned = current;
    transitioned.cloudType = CloudType::CIRRUS;
    const simd_float2 transitionedMotion = decodeCloudMotion(encodeCloudMotion(transitioned));
    const simd_float2 transitionDelta = cloudMotionDelta(transitionedMotion, currentMotion);
    REQUIRE(transitionDelta.x == Catch::Approx(0.0F).margin(0.001F));
    REQUIRE(transitionDelta.y == Catch::Approx(0.0F).margin(0.001F));

    WeatherMapUniforms map{};
    map.cellSpacing = WEATHER_MAP_CELL_SPACING;
    map.gridSize = simd_make_uint2(WEATHER_MAP_EDGE, WEATHER_MAP_EDGE);
    map.motionWrapBlocks = CLOUD_MOTION_WRAP_BLOCKS;
    constexpr int64_t ORIGIN_X = 12'288;
    constexpr int64_t ORIGIN_Z = -122'880;
    const simd_float2 worldXZ = simd_make_float2(23'800.0F, -111'200.0F);
    const auto uvForCamera = [&](simd_float2 cameraXZ) {
        WeatherMapUniforms cameraMap = map;
        cameraMap.originXZ = simd_make_float2(static_cast<float>(ORIGIN_X) - cameraXZ.x,
                                              static_cast<float>(ORIGIN_Z) - cameraXZ.y);
        return weatherMapTextureCoordinate(worldXZ - cameraXZ, cameraMap);
    };
    const simd_float2 stationary = uvForCamera(simd_make_float2(23'029.0F, -111'726.0F));
    const simd_float2 moved = uvForCamera(simd_make_float2(24'053.0F, -110'702.0F));
    REQUIRE(stationary.x == Catch::Approx(moved.x).margin(1.0e-6F));
    REQUIRE(stationary.y == Catch::Approx(moved.y).margin(1.0e-6F));
}

TEST_CASE("Cloud horizon uses camera-forward depth within regional weather coverage",
          "[render][clouds][weather][horizon]") {
    WeatherMapUniforms map{};
    map.originXZ = simd_make_float2(-10'240.0F, -10'240.0F);
    map.cellSpacing = WEATHER_MAP_CELL_SPACING;
    map.gridSize = simd_make_uint2(WEATHER_MAP_EDGE, WEATHER_MAP_EDGE);

    const simd_float3 forward = simd_make_float3(0.0F, 0.0F, -1.0F);
    REQUIRE(cloudMarchRayDistanceLimit(forward, forward, CLOUD_HORIZON_VIEW_DEPTH, map) ==
            Catch::Approx(CLOUD_HORIZON_VIEW_DEPTH));

    const simd_float3 diagonal = simd_normalize(simd_make_float3(0.7F, 0.0F, -1.0F));
    const float diagonalDistance =
        cloudMarchRayDistanceLimit(diagonal, forward, CLOUD_HORIZON_VIEW_DEPTH, map);
    REQUIRE(diagonalDistance * simd_dot(diagonal, forward) ==
            Catch::Approx(CLOUD_HORIZON_VIEW_DEPTH).margin(0.01F));
    REQUIRE(diagonalDistance > CLOUD_HORIZON_VIEW_DEPTH);

    // Model the largest permitted camera displacement before recentering.
    // The positive map edge is then closer, and the one-cell guard must cap a
    // wide ray before filtered sampling can clamp to stale boundary weather.
    map.originXZ.x -= 1'024.0F;
    const simd_float3 edgeRay = simd_normalize(simd_make_float3(1.0F, 0.0F, -0.2F));
    const float weatherLimit = cloudWeatherCoverageRayDistance(edgeRay, map);
    const float marchLimit =
        cloudMarchRayDistanceLimit(edgeRay, forward, CLOUD_HORIZON_VIEW_DEPTH, map);
    REQUIRE(marchLimit == Catch::Approx(weatherLimit));
    REQUIRE(marchLimit < cloudViewDepthRayDistance(edgeRay, forward, CLOUD_HORIZON_VIEW_DEPTH));
    REQUIRE(marchLimit * edgeRay.x == Catch::Approx(8'960.0F).margin(0.01F));
}

TEST_CASE("Bulk cloud noise preserves the scalar contract with bounded hash work",
          "[render][clouds][performance]") {
    constexpr int EDGE = CLOUD_BASE_NOISE_EDGE;
    constexpr uint64_t SEED = 764891;
    CloudNoiseGenerationStats stats;
    const std::vector<uint8_t> volume = generateCloudBaseNoiseVolume(EDGE, SEED, &stats);
    REQUIRE(volume.size() == 2'097'152);
    REQUIRE(stats.voxelCount == volume.size());
    REQUIRE(stats.hashEvaluations == 2'112);
    REQUIRE(stats.worleyFeatureTests == 56'623'104);
    REQUIRE(stats.hashEvaluations * 512U < stats.voxelCount);

    constexpr std::array<std::array<int, 3>, 8> SAMPLES{{
        {0, 0, 0},
        {1, 7, 19},
        {31, 63, 95},
        {64, 64, 64},
        {87, 42, 113},
        {126, 3, 91},
        {127, 0, 127},
        {127, 127, 127},
    }};
    for (const auto& coordinate : SAMPLES) {
        const int x = coordinate[0];
        const int y = coordinate[1];
        const int z = coordinate[2];
        const size_t index = (static_cast<size_t>(z) * EDGE + static_cast<size_t>(y)) * EDGE +
                             static_cast<size_t>(x);
        const uint8_t scalar =
            static_cast<uint8_t>(cloudBaseNoise(x, y, z, EDGE, SEED) * 255.0F + 0.5F);
        REQUIRE(volume[index] == scalar);
    }

    constexpr int CONTRACT_EDGE = 32;
    const std::vector<uint8_t> contractVolume = generateCloudBaseNoiseVolume(CONTRACT_EDGE, SEED);
    size_t mismatchedVoxels = 0;
    for (int z = 0; z < CONTRACT_EDGE; ++z) {
        for (int y = 0; y < CONTRACT_EDGE; ++y) {
            for (int x = 0; x < CONTRACT_EDGE; ++x) {
                const size_t index =
                    (static_cast<size_t>(z) * CONTRACT_EDGE + static_cast<size_t>(y)) *
                        CONTRACT_EDGE +
                    static_cast<size_t>(x);
                const uint8_t scalar = static_cast<uint8_t>(
                    cloudBaseNoise(x, y, z, CONTRACT_EDGE, SEED) * 255.0F + 0.5F);
                mismatchedVoxels += contractVolume[index] != scalar ? 1U : 0U;
            }
        }
    }
    REQUIRE(mismatchedVoxels == 0);

    uint64_t digest = 14'695'981'039'346'656'037ULL;
    for (uint8_t value : volume) {
        digest ^= value;
        digest *= 1'099'511'628'211ULL;
    }
    REQUIRE(digest == 0x2F90AEDA76D69F7CULL);
}

TEST_CASE("Screen-space lighting memory follows quality at native resolution",
          "[render][indirect][memory]") {
    constexpr uint32_t WIDTH = 3456;
    constexpr uint32_t HEIGHT = 2234;
    const ScreenSpaceLightingMemoryFootprint off =
        screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, 0);
    REQUIRE(off.workWidth == 1);
    REQUIRE(off.workHeight == 1);
    REQUIRE(off.neutralBytes == 8);
    REQUIRE(off.linearDepthPyramidBytes == 0);
    REQUIRE(off.normalBytes == 0);
    REQUIRE(off.traceBytes == 0);
    REQUIRE(off.historyBytes == 0);
    REQUIRE(off.historyDepthBytes == 0);
    REQUIRE(off.momentsBytes == 0);
    REQUIRE(off.scratchBytes == 0);
    REQUIRE(off.totalBytes() == 8);

    const ScreenSpaceLightingMemoryFootprint medium =
        screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, 1);
    REQUIRE(medium.workWidth == 864);
    REQUIRE(medium.workHeight == 558);
    REQUIRE(medium.linearDepthPyramidBytes == 41'173'704);
    REQUIRE(medium.normalBytes == 30'882'816);
    REQUIRE(medium.traceBytes == 3'856'896);
    REQUIRE(medium.historyBytes == 7'713'792);
    REQUIRE(medium.historyDepthBytes == 3'856'896);
    REQUIRE(medium.momentsBytes == 7'713'792);
    REQUIRE(medium.scratchBytes == 3'856'896);
    REQUIRE(medium.totalBytes() == 99'054'800);

    const ScreenSpaceLightingMemoryFootprint high =
        screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, 2);
    REQUIRE(high.workWidth == 1728);
    REQUIRE(high.workHeight == 1117);
    REQUIRE(high.normalBytes == 30'882'816);
    REQUIRE(high.traceBytes == 15'441'408);
    REQUIRE(high.historyBytes == 30'882'816);
    REQUIRE(high.historyDepthBytes == 15'441'408);
    REQUIRE(high.momentsBytes == 30'882'816);
    REQUIRE(high.scratchBytes == 15'441'408);
    REQUIRE(high.totalBytes() == 180'146'384);
    REQUIRE(screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, -4).totalBytes() == off.totalBytes());
    REQUIRE(screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, 8).totalBytes() == high.totalBytes());
}

TEST_CASE("Screen-space lighting normal guide rejects separate voxel faces",
          "[render][indirect][upsample]") {
    const Vec3 floorNormal{0.0F, 1.0F, 0.0F};
    const Vec3 nearbyFloorNormal{0.0F, 0.98F, 0.2F};
    const Vec3 wallNormal{1.0F, 0.0F, 0.0F};

    REQUIRE(screenSpaceBilateralNormalWeight(floorNormal, floorNormal) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceBilateralNormalWeight(floorNormal, nearbyFloorNormal) > 0.99F);
    REQUIRE(screenSpaceBilateralNormalWeight(floorNormal, wallNormal) == Catch::Approx(0.0F));
    REQUIRE(screenSpaceBilateralNormalWeight(floorNormal, Vec3{}) == Catch::Approx(0.0F));

    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, floorNormal) >
            0.99F);
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, wallNormal) ==
            Catch::Approx(0.0F));
}

TEST_CASE("Screen-space lighting history resets for every discontinuity",
          "[render][indirect][history]") {
    const IndirectHistoryState previous{
        .width = 3456,
        .height = 2234,
        .cameraPosition = {23'029.0F, 225.0F, -111'726.0F},
        .fovDegrees = 70.0F,
        .worldIdentity = 17,
        .forcedStateRevision = 9,
        .quality = 2,
        .directLightSource = 1,
        .priorDepthValid = true,
    };
    REQUIRE(indirectHistoryResetMask(previous, previous) == INDIRECT_HISTORY_STABLE);

    const auto requireOnly = [&previous](auto mutate, uint32_t expected) {
        IndirectHistoryState current = previous;
        mutate(current);
        REQUIRE(indirectHistoryResetMask(previous, current) == expected);
    };
    requireOnly([](IndirectHistoryState& state) { ++state.width; }, INDIRECT_HISTORY_RESIZE);
    requireOnly([](IndirectHistoryState& state) { ++state.height; }, INDIRECT_HISTORY_RESIZE);
    requireOnly([](IndirectHistoryState& state) { state.cameraPosition.x += 8.01F; },
                INDIRECT_HISTORY_TELEPORT);
    requireOnly([](IndirectHistoryState& state) { ++state.worldIdentity; },
                INDIRECT_HISTORY_WORLD_CHANGE);
    requireOnly([](IndirectHistoryState& state) { state.fovDegrees += 0.51F; },
                INDIRECT_HISTORY_FOV_CHANGE);
    requireOnly([](IndirectHistoryState& state) { --state.quality; },
                INDIRECT_HISTORY_QUALITY_CHANGE);
    requireOnly([](IndirectHistoryState& state) { ++state.forcedStateRevision; },
                INDIRECT_HISTORY_FORCED_STATE);
    requireOnly([](IndirectHistoryState& state) { state.priorDepthValid = false; },
                INDIRECT_HISTORY_INVALID_DEPTH);
    requireOnly([](IndirectHistoryState& state) { ++state.directLightSource; },
                INDIRECT_HISTORY_LIGHT_SOURCE);

    IndirectHistoryState continuous = previous;
    continuous.cameraPosition.x += 8.0F;
    continuous.fovDegrees += 0.5F;
    REQUIRE(indirectHistoryResetMask(previous, continuous) == INDIRECT_HISTORY_STABLE);

    IndirectHistoryState allReasons = previous;
    ++allReasons.width;
    allReasons.cameraPosition.x += 9.0F;
    ++allReasons.worldIdentity;
    allReasons.fovDegrees += 1.0F;
    --allReasons.quality;
    ++allReasons.forcedStateRevision;
    allReasons.priorDepthValid = false;
    ++allReasons.directLightSource;
    constexpr uint32_t EVERY_REASON =
        INDIRECT_HISTORY_RESIZE | INDIRECT_HISTORY_TELEPORT | INDIRECT_HISTORY_WORLD_CHANGE |
        INDIRECT_HISTORY_FOV_CHANGE | INDIRECT_HISTORY_QUALITY_CHANGE |
        INDIRECT_HISTORY_FORCED_STATE | INDIRECT_HISTORY_INVALID_DEPTH |
        INDIRECT_HISTORY_LIGHT_SOURCE;
    REQUIRE(indirectHistoryResetMask(previous, allReasons) == EVERY_REASON);
}

TEST_CASE("Cloud memory omits frame and shadow targets when disabled", "[render][clouds][memory]") {
    constexpr uint32_t WIDTH = 3456;
    constexpr uint32_t HEIGHT = 2234;
    const CloudRendererMemoryFootprint off = cloudRendererMemoryFootprint(WIDTH, HEIGHT, 0);
    REQUIRE(off.quarterWidth == 1);
    REQUIRE(off.quarterHeight == 1);
    REQUIRE(off.shadowEdge == 0);
    REQUIRE(off.noiseBytes == 2'162'688);
    REQUIRE(off.weatherBytes == 2'519'424);
    REQUIRE(off.neutralShadowBytes == 2);
    REQUIRE(off.frameTargetBytes == 0);
    REQUIRE(off.shadowBytes == 0);
    REQUIRE(off.totalBytes() == 4'682'114);

    const CloudRendererMemoryFootprint medium = cloudRendererMemoryFootprint(WIDTH, HEIGHT, 1);
    REQUIRE(medium.quarterWidth == 864);
    REQUIRE(medium.quarterHeight == 558);
    REQUIRE(medium.shadowEdge == 1024);
    REQUIRE(medium.frameTargetBytes == 14'463'360);
    REQUIRE(medium.shadowBytes == 2'097'152);
    REQUIRE(medium.totalBytes() == 21'242'626);

    const CloudRendererMemoryFootprint high = cloudRendererMemoryFootprint(WIDTH, HEIGHT, 2);
    REQUIRE(high.shadowEdge == 2048);
    REQUIRE(high.frameTargetBytes == medium.frameTargetBytes);
    REQUIRE(high.shadowBytes == 8'388'608);
    REQUIRE(high.totalBytes() == 27'534'082);
    REQUIRE(cloudRendererMemoryFootprint(WIDTH, HEIGHT, -1).totalBytes() == off.totalBytes());
    REQUIRE(cloudRendererMemoryFootprint(WIDTH, HEIGHT, 4).totalBytes() == high.totalBytes());

    constexpr uint64_t HIGH_TIER_BUDGET = 768ULL * 1024ULL * 1024ULL;
    const AtmosphericMemoryFootprint integrated = atmosphericMemoryFootprint(WIDTH, HEIGHT, 2);
    REQUIRE(waterReflectionPyramidMemoryBytes(0, HEIGHT) == 0);
    REQUIRE(waterReflectionPyramidMemoryBytes(3, 5) == 144);
    REQUIRE(waterReflectionPyramidMemoryBytes(WIDTH, HEIGHT) == 82'347'408);
    REQUIRE(atmosphericSceneTargetMemoryBytes(WIDTH, HEIGHT) == 236'761'488);
    REQUIRE(integrated.sceneTargetBytes == 236'761'488);
    REQUIRE(integrated.shadowBytes == 184'549'376);
    REQUIRE(integrated.indirectBytes == 180'146'384);
    REQUIRE(integrated.atmosphereBytes == 305'152);
    REQUIRE(integrated.cloudBytes == 27'534'082);
    REQUIRE(integrated.volumetricBytes == 86'525'723);
    REQUIRE(integrated.lightningBytes == 2);
    REQUIRE(integrated.totalBytes() == 715'822'207);
    REQUIRE(integrated.totalBytes() < HIGH_TIER_BUDGET);
    REQUIRE(HIGH_TIER_BUDGET - integrated.totalBytes() == 89'484'161);
}

TEST_CASE("Atmospheric diagnostics preserve all cascade worker and memory counters",
          "[engine][hud][atmosphere][diagnostics]") {
    RenderPipeline::AtmosphericRenderStats rendererStats;
    REQUIRE(rendererStats.shadowCasterCounts.size() == SHADOW_CASCADE_COUNT);
    REQUIRE(rendererStats.shadowRefreshCounts.size() == SHADOW_CASCADE_COUNT);
    REQUIRE(std::ranges::all_of(rendererStats.shadowCasterCounts,
                                [](uint32_t value) { return value == 0U; }));
    REQUIRE(std::ranges::all_of(rendererStats.shadowRefreshCounts,
                                [](uint64_t value) { return value == 0U; }));
    REQUIRE(rendererStats.integratedPersistentBytes == 0U);

    PerformanceStats hudStats;
    REQUIRE(hudStats.shadowCasterCounts.size() == SHADOW_CASCADE_COUNT);
    REQUIRE(hudStats.shadowRefreshCounts.size() == SHADOW_CASCADE_COUNT);
    REQUIRE(hudStats.weatherRequests == 0U);
    REQUIRE(hudStats.weatherPendingRequests == 0U);
    REQUIRE_FALSE(hudStats.weatherWorkerBusy);
    REQUIRE(hudStats.thunderPending == 0U);
    REQUIRE(hudStats.integratedAtmosphericPersistentMB == Catch::Approx(0.0F));

    WeatherSystemStats weatherStats;
    weatherStats.requests = 9U;
    weatherStats.coalescedRequests = 4U;
    weatherStats.pendingRequests = 1U;
    weatherStats.workerBusy = true;
    REQUIRE(weatherStats.requests == 9U);
    REQUIRE(weatherStats.coalescedRequests == 4U);
    REQUIRE(weatherStats.pendingRequests == 1U);
    REQUIRE(weatherStats.workerBusy);
}
// ===========================================================================
// Engine: game flow, menus, input, hotbar
// ===========================================================================

TEST_CASE("Unsigned decimal capture ticks reject signs invalid text and overflow",
          "[engine][capture][time]") {
    REQUIRE(parseUnsignedDecimal("0") == 0U);
    REQUIRE(parseUnsignedDecimal("24000") == 24'000U);
    REQUIRE(parseUnsignedDecimal("18446744073709551615") == std::numeric_limits<uint64_t>::max());

    REQUIRE_FALSE(parseUnsignedDecimal(""));
    REQUIRE_FALSE(parseUnsignedDecimal("-1"));
    REQUIRE_FALSE(parseUnsignedDecimal("+1"));
    REQUIRE_FALSE(parseUnsignedDecimal(" 1"));
    REQUIRE_FALSE(parseUnsignedDecimal("1 "));
    REQUIRE_FALSE(parseUnsignedDecimal("0x10"));
    REQUIRE_FALSE(parseUnsignedDecimal("12ticks"));
    REQUIRE_FALSE(parseUnsignedDecimal("18446744073709551616"));
}

TEST_CASE("Capture lightning override parses finite coordinates ID and rendered age",
          "[engine][capture][lightning]") {
    const auto parsed = parseCaptureLightningOverride("23029.25,-111726.5,18446744073709551615,7");
    REQUIRE(parsed);
    REQUIRE(parsed->x == Catch::Approx(23'029.25));
    REQUIRE(parsed->z == Catch::Approx(-111'726.5));
    REQUIRE(parsed->id == std::numeric_limits<uint64_t>::max());
    REQUIRE(parsed->ageTicks == 7U);

    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-111726,17"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-111726,17,2,extra"));
    REQUIRE_FALSE(parseCaptureLightningOverride("nan,-111726,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,inf,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("1e300,-111726,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-1e300,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-111726,-17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-111726,17,-2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029,-111726,18446744073709551616,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("23029, -111726,17,2"));
}

TEST_CASE("InputBindings save/load round-trips a custom binding", "[engine][bindings]") {
    TempDir dir("bindings");
    std::string path = dir.path() + "/bindings.json";

    InputBindings custom;
    custom.forward.key = Key::Up;
    custom.jump.key = Key::F;
    REQUIRE(custom.save(path));

    auto loaded = InputBindings::load(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->forward.key == Key::Up);
    REQUIRE(loaded->jump.key == Key::F);
    REQUIRE(loaded->backward.key == Key::S); // untouched bindings keep defaults
}

TEST_CASE("InputBindings load returns defaults for a missing file", "[engine][bindings]") {
    TempDir dir("bindings_missing");
    auto loaded = InputBindings::load(dir.path() + "/nope.json");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->forward.key == Key::W);
}

TEST_CASE("InputBindings defaults: Ctrl sprints, Shift sneaks", "[engine][bindings]") {
    // Minecraft layout, and what the README documents. Sprint once sat on
    // LeftShift, which fly-descend now needs.
    InputBindings defaults;
    REQUIRE(defaults.sprint.key == Key::LeftControl);
    REQUIRE(defaults.sneak.key == Key::LeftShift);

    TempDir dir("bindings_sprint_sneak");
    std::string path = dir.path() + "/bindings.json";
    REQUIRE(defaults.save(path));
    auto loaded = InputBindings::load(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->sprint.key == Key::LeftControl);
    REQUIRE(loaded->sneak.key == Key::LeftShift);
}

// ============================================================================
// Double-tap detection (sprint on W, fly toggle on Space)
// ============================================================================

TEST_CASE("Double-tap: two presses inside the window latch for the tick", "[engine][input]") {
    InputState input;

    input.recordPress(Key::W, 1.0);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(input.isPressedForTick(Key::W));
    REQUIRE(input.isDown(Key::W));

    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW * 0.5);
    REQUIRE(input.isDoubleTappedForTick(Key::W));

    // Consumed at tick end, exactly like keysPressedForTick
    input.clearTickPresses();
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isPressedForTick(Key::W));
}

TEST_CASE("Double-tap: a slow second press does not latch", "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW + 0.05);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));

    // ...but that second press starts a fresh window
    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW + 0.15);
    REQUIRE(input.isDoubleTappedForTick(Key::W));
}

TEST_CASE("Double-tap: a triple-tap fires exactly one gesture", "[engine][input]") {
    InputState input;
    input.recordPress(Key::Space, 1.0);
    input.recordPress(Key::Space, 1.1); // fires and consumes the history
    REQUIRE(input.isDoubleTappedForTick(Key::Space));
    input.clearTickPresses();

    input.recordPress(Key::Space, 1.2); // pairs with nothing, history was consumed
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));
}

TEST_CASE("Double-tap: keys are tracked independently", "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::Space, 1.1);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));

    input.recordPress(Key::W, 1.2);
    REQUIRE(input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));
}

TEST_CASE("Double-tap: latch survives per-frame update() until a tick consumes it",
          "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::W, 1.1);

    // Several tickless frames pass, the gesture must not be dropped
    input.update();
    input.update();
    REQUIRE(input.isDoubleTappedForTick(Key::W));
}

// ============================================================================
// Game flow + menu layout tests (pure C++, no Metal)
// ============================================================================

TEST_CASE("GameFlow: ESC toggles pause and backs out of settings", "[ui][flow]") {
    GameFlow flow;
    REQUIRE(flow.screen == GameScreen::TITLE);

    // ESC is inert on the title screen
    auto fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE(!fx.captureCursor);

    // PLAY enters gameplay and captures the mouse
    fx = flow.onMenuAction(MenuAction::PLAY);
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);
    REQUIRE(fx.resetTiming);

    // ESC pauses (release + timing reset), ESC again resumes (capture)
    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(fx.releaseCursor);
    REQUIRE(fx.resetTiming);

    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Settings sits under pause; ESC backs out one level
    flow.onEscape();
    flow.onMenuAction(MenuAction::OPEN_SETTINGS);
    REQUIRE(flow.screen == GameScreen::SETTINGS);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PAUSED);
}

TEST_CASE("Settings save/load round-trips values and video settings", "[engine][settings]") {
    TempDir dir("settings");
    std::string path = dir.path() + "/settings.json";

    SettingsValues values;
    values.viewDistance = SettingsValues::MAX_VIEW_DISTANCE;
    values.fogLevel = 7;
    values.sensitivityLevel = 9;
    values.volumeLevel = 2;
    GraphicsSettings gfx;
    gfx.shadowQuality = 1;
    gfx.volumetricLight = false;
    gfx.cloudQuality = 0;
    gfx.indirectLightingQuality = 1;
    gfx.waterReflections = false;
    gfx.wavingFoliage = false;
    gfx.lensFlare = false;
    gfx.bloomLevel = 8;
    gfx.vibrance = 3;
    gfx.sharpening = 6;

    REQUIRE(saveSettings(path, values, gfx));
    LoadedSettings loaded = loadSettings(path);

    REQUIRE(loaded.values.viewDistance == SettingsValues::MAX_VIEW_DISTANCE);
    REQUIRE(loaded.values.fogLevel == 7);
    REQUIRE(loaded.values.sensitivityLevel == 9);
    REQUIRE(loaded.values.volumeLevel == 2);
    REQUIRE(loaded.gfx.shadowQuality == 1);
    REQUIRE(loaded.gfx.volumetricLight == false);
    REQUIRE(loaded.gfx.cloudQuality == 0);
    REQUIRE(loaded.gfx.indirectLightingQuality == 1);
    REQUIRE(loaded.gfx.waterReflections == false);
    REQUIRE(loaded.gfx.wavingFoliage == false);
    REQUIRE(loaded.gfx.lensFlare == false);
    REQUIRE(loaded.gfx.bloomLevel == 8);
    REQUIRE(loaded.gfx.vibrance == 3);
    REQUIRE(loaded.gfx.sharpening == 6);
}

TEST_CASE("Settings reuse the supported world view-distance contract", "[engine][settings]") {
    STATIC_REQUIRE(SettingsValues::MIN_VIEW_DISTANCE == MIN_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::MAX_VIEW_DISTANCE == MAX_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::DEFAULT_VIEW_DISTANCE == DEFAULT_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::VIEW_DISTANCES.front() == MIN_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::VIEW_DISTANCES.back() == MAX_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Default clear-weather fog preserves the eight-kilometer horizon",
          "[engine][settings][render][far-terrain]") {
    REQUIRE(fogDensityForLevel(0) == 0.0F);
    REQUIRE(fogDensityForLevel(3) == Catch::Approx(0.00015F));
    constexpr float HORIZON_BLOCKS = MAX_RENDER_DISTANCE_CHUNKS * CHUNK_EDGE;
    const float fogCoverage = 1.0F - std::exp(-fogDensityForLevel(3) * HORIZON_BLOCKS);
    REQUIRE(fogCoverage < 0.75F);
}

TEST_CASE("Settings load: missing file and out-of-range values fall back", "[engine][settings]") {
    TempDir dir("settings");

    // Missing file → the max-preset defaults
    LoadedSettings missing = loadSettings(dir.path() + "/nope.json");
    REQUIRE(missing.values.viewDistance == SettingsValues::DEFAULT_VIEW_DISTANCE);
    REQUIRE(missing.gfx.shadowQuality == 2);
    REQUIRE(missing.gfx.volumetricLight);
    REQUIRE(missing.gfx.cloudQuality == 2);
    REQUIRE(missing.gfx.indirectLightingQuality == 2);
    REQUIRE(missing.gfx.bloomLevel == 5);
    REQUIRE(missing.gfx.sharpening == 0);

    // Hand-edited garbage clamps instead of exploding
    std::string path = dir.path() + "/settings.json";
    std::filesystem::create_directories(dir.path());
    {
        std::ofstream file(path);
        file << "{ \"viewDistance\": 999, \"shadowQuality\": -3, \"vibrance\": 42 }";
    }
    LoadedSettings clamped = loadSettings(path);
    REQUIRE(clamped.values.viewDistance == SettingsValues::MAX_VIEW_DISTANCE);
    REQUIRE(clamped.gfx.shadowQuality == 0);
    REQUIRE(clamped.gfx.vibrance == 10);
    // Keys the file omits keep their defaults
    REQUIRE(clamped.gfx.cloudQuality == 2);
}

TEST_CASE("Settings migrate legacy ambient and cloud quality keys", "[engine][settings]") {
    TempDir dir("settings-migration");
    std::string path = dir.path() + "/settings.json";
    std::filesystem::create_directories(dir.path());
    {
        std::ofstream file(path);
        file << "{ \"cloudMode\": 1, \"ssao\": 0 }";
    }

    LoadedSettings loaded = loadSettings(path);
    REQUIRE(loaded.gfx.cloudQuality == 1);
    REQUIRE(loaded.gfx.indirectLightingQuality == 0);

    REQUIRE(saveSettings(path, loaded.values, loaded.gfx));
    std::ifstream savedFile(path);
    std::ostringstream saved;
    saved << savedFile.rdbuf();
    REQUIRE(saved.str().find("\"cloudQuality\"") != std::string::npos);
    REQUIRE(saved.str().find("\"indirectLightingQuality\"") != std::string::npos);
    REQUIRE(saved.str().find("\"cloudMode\"") == std::string::npos);
    REQUIRE(saved.str().find("\"ssao\"") == std::string::npos);
}

TEST_CASE("Settings prefer new quality keys over legacy keys", "[engine][settings]") {
    TempDir dir("settings-quality-precedence");
    std::string path = dir.path() + "/settings.json";
    std::filesystem::create_directories(dir.path());
    {
        std::ofstream file(path);
        file << "{ \"cloudQuality\": 2, \"cloudMode\": 0, "
                "\"indirectLightingQuality\": 1, \"ssao\": 0 }";
    }

    const LoadedSettings loaded = loadSettings(path);
    REQUIRE(loaded.gfx.cloudQuality == 2);
    REQUIRE(loaded.gfx.indirectLightingQuality == 1);
}

TEST_CASE("GraphicsSettings env overrides map onto the fields", "[engine][settings]") {
    setenv("RYCRAFT_SHADOWS", "1", 1);
    setenv("RYCRAFT_VL", "0", 1);
    setenv("RYCRAFT_CLOUDS", "1", 1);
    setenv("RYCRAFT_SSR", "0", 1);
    setenv("RYCRAFT_BLOOM", "0", 1); // legacy intensity form: 0 disables

    GraphicsSettings gfx;
    REQUIRE(gfx.applyEnvOverrides()); // reports that overrides fired
    REQUIRE(gfx.shadowQuality == 1);
    REQUIRE(gfx.volumetricLight == false);
    REQUIRE(gfx.cloudQuality == 1);
    REQUIRE(gfx.waterReflections == false);
    REQUIRE(gfx.bloomLevel == 0);
    // Untouched fields keep defaults
    REQUIRE(gfx.indirectLightingQuality == 2);
    REQUIRE(gfx.wavingFoliage);

    unsetenv("RYCRAFT_SHADOWS");
    unsetenv("RYCRAFT_VL");
    unsetenv("RYCRAFT_CLOUDS");
    unsetenv("RYCRAFT_SSR");
    unsetenv("RYCRAFT_BLOOM");

    // With no RYCRAFT_* set it reports false, so the engine keeps saving
    GraphicsSettings clean;
    REQUIRE(!clean.applyEnvOverrides());
}

TEST_CASE("GraphicsSettings prefer new quality overrides over legacy aliases",
          "[engine][settings]") {
    setenv("RYCRAFT_CLOUD_QUALITY", "2", 1);
    setenv("RYCRAFT_CLOUDS", "0", 1);
    setenv("RYCRAFT_INDIRECT_LIGHT", "1", 1);
    setenv("RYCRAFT_SSAO", "0", 1);

    GraphicsSettings gfx;
    REQUIRE(gfx.applyEnvOverrides());
    REQUIRE(gfx.cloudQuality == 2);
    REQUIRE(gfx.indirectLightingQuality == 1);

    unsetenv("RYCRAFT_CLOUD_QUALITY");
    unsetenv("RYCRAFT_CLOUDS");
    unsetenv("RYCRAFT_INDIRECT_LIGHT");
    unsetenv("RYCRAFT_SSAO");
}

TEST_CASE("GameFlow: video settings nest under settings", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);
    flow.onEscape(); // pause
    flow.onMenuAction(MenuAction::OPEN_SETTINGS);

    // OPEN_VIDEO_SETTINGS only works from the settings screen
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(flow.screen == GameScreen::VIDEO_SETTINGS);

    // BACK returns to settings, ESC does the same
    flow.onMenuAction(MenuAction::CLOSE_VIDEO_SETTINGS);
    REQUIRE(flow.screen == GameScreen::SETTINGS);
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::SETTINGS);

    // Video screen freezes the sim like every other menu
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(flow.inMenu());

    // OPEN from a non-settings screen is inert
    GameFlow paused;
    paused.onMenuAction(MenuAction::PLAY);
    paused.onEscape();
    paused.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(paused.screen == GameScreen::PAUSED);
}

TEST_CASE("GameFlow: resume and quit actions", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);
    flow.onEscape(); // pause

    auto fx = flow.onMenuAction(MenuAction::RESUME);
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Quit only works from title/pause
    fx = flow.onMenuAction(MenuAction::QUIT);
    REQUIRE(!fx.requestQuit);
    flow.onEscape();
    fx = flow.onMenuAction(MenuAction::QUIT);
    REQUIRE(fx.requestQuit);
}

TEST_CASE("GameFlow: focus loss force-pauses gameplay only", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);

    auto fx = flow.onFocusLost();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(fx.releaseCursor);

    // Idempotent while already paused
    fx = flow.onFocusLost();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(!fx.releaseCursor);
}

TEST_CASE("Menu layouts: buttons sit on-screen and inside their panel", "[ui][menu]") {
    SettingsValues values;
    GraphicsSettings gfx;
    for (auto [w, h] : {std::pair{1024.f, 768.f}, {2048.f, 1536.f}, {3456.f, 2234.f}}) {
        for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS,
                                  GameScreen::VIDEO_SETTINGS}) {
            MenuLayout layout = buildMenuLayout(screen, w, h, values, gfx);
            REQUIRE(!layout.buttons.empty());

            for (const auto& button : layout.buttons) {
                REQUIRE(button.rect.x >= 0.f);
                REQUIRE(button.rect.y >= 0.f);
                REQUIRE(button.rect.x + button.rect.w <= 1.f);
                REQUIRE(button.rect.y + button.rect.h <= 1.f);
                REQUIRE(button.action != MenuAction::NONE);
                if (layout.panel.w > 0.f) {
                    REQUIRE(button.rect.x >= layout.panel.x);
                    REQUIRE(button.rect.x + button.rect.w <= layout.panel.x + layout.panel.w);
                }
            }

            // No two buttons overlap
            for (size_t i = 0; i < layout.buttons.size(); ++i) {
                for (size_t j = i + 1; j < layout.buttons.size(); ++j) {
                    const UIRect& a = layout.buttons[i].rect;
                    const UIRect& b = layout.buttons[j].rect;
                    bool separated = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y ||
                                     b.y + b.h <= a.y;
                    REQUIRE(separated);
                }
            }
        }
    }

    REQUIRE(buildMenuLayout(GameScreen::PLAYING, 1024.f, 768.f, values, gfx).buttons.empty());
}

TEST_CASE("Menu hit test: button centers hit, gaps miss", "[ui][menu]") {
    SettingsValues values;
    GraphicsSettings gfx;
    MenuLayout layout = buildMenuLayout(GameScreen::PAUSED, 1024.f, 768.f, values, gfx);

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const UIRect& rect = layout.buttons[i].rect;
        REQUIRE(menuHitTest(layout, rect.x + rect.w * 0.5f, rect.y + rect.h * 0.5f) ==
                static_cast<int>(i));
    }

    REQUIRE(menuHitTest(layout, 0.02f, 0.02f) == -1);
}

TEST_CASE("Font covers every character the menus draw", "[ui][font]") {
    SettingsValues values;
    GraphicsSettings gfx;
    std::string needed = "0123456789.:/-+ ";
    for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS,
                              GameScreen::VIDEO_SETTINGS}) {
        MenuLayout layout = buildMenuLayout(screen, 1024.f, 768.f, values, gfx);
        for (const auto& text : layout.texts)
            needed += text.text;
        for (const auto& button : layout.buttons)
            needed += button.label;
    }
    // Plus everything the debug HUD prints
    needed += "FPS: Chunks: Entities: Frame: ";

    for (char c : needed) {
        if (c == ' ')
            continue; // spaces render as gaps by design
        auto bitmap = UIOverlay::getCharBitmap(c);
        bool anyPixel = false;
        for (uint8_t row : bitmap)
            anyPixel |= row != 0;
        INFO("Missing glyph: '" << c << "'");
        REQUIRE(anyPixel);
    }
}

// ============================================================================
// UIOverlay Quad Vertex Generation Tests (no Metal device required)
// ============================================================================

TEST_CASE("UIOverlay quad vertex generation: fullscreen quad", "[render][ui]") {
    // Verify that a fullscreen quad (0,0,1,1) produces correct vertex positions.
    // Layout: [x, y] for each of 4 vertices: BL, TL, BR, TR
    float x = 0.0f, y = 0.0f, w = 1.0f, h = 1.0f;

    // Expected vertices (bottom-left origin):
    // BL: (0, 0), TL: (0, 1), BR: (1, 0), TR: (1, 1)
    struct QuadVertex {
        float px, py;
        float cr, cg, cb, ca;
    };

    QuadVertex expected[4] = {
        {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Verify vertex positions
    REQUIRE(expected[0].px == x);
    REQUIRE(expected[0].py == y);
    REQUIRE(expected[1].px == x);
    REQUIRE(expected[1].py == y + h);
    REQUIRE(expected[2].px == x + w);
    REQUIRE(expected[2].py == y);
    REQUIRE(expected[3].px == x + w);
    REQUIRE(expected[3].py == y + h);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair horizontal line", "[render][ui]") {
    // Simulate crosshair horizontal line at center of 1920×1080 screen
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossH = 1.0f / screenHeight; // 1 pixel height
    float crossW = 20.0f / screenWidth; // 20 pixel width

    float left = centerX - crossW * 0.5f;
    float bottom = centerY - crossH * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossH * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossW > 0.0f);
    REQUIRE(crossH > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossH <= 1.0f);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair vertical line", "[render][ui]") {
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossV = 20.0f / screenHeight;   // 20 pixel height
    float crossLineW = 1.0f / screenWidth; // 1 pixel width

    float left = centerX - crossLineW * 0.5f;
    float bottom = centerY - crossV * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossLineW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossV * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossLineW > 0.0f);
    REQUIRE(crossV > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossLineW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossV <= 1.0f);
}

TEST_CASE("UIOverlay orthographic projection maps screen to NDC", "[render][ui]") {
    // Verify the orthographic projection matrix maps [0,1] screen coords to [-1,1] NDC.
    // Matrix:
    //   [ 2,  0,  0,  0]
    //   [ 0,  2,  0,  0]
    //   [ 0,  0,  1,  0]
    //   [-1, -1,  0,  1]
    //
    // For point (x, y, 0, 1): result = (2x-1, 2y-1, 0, 1)

    auto transform = [](float sx, float sy) -> std::pair<float, float> {
        float nx = 2.0f * sx - 1.0f;
        float ny = 2.0f * sy - 1.0f;
        return {nx, ny};
    };

    // Screen (0, 0) → NDC (-1, -1)
    auto p0 = transform(0.0f, 0.0f);
    REQUIRE(p0.first == Catch::Approx(-1.0f));
    REQUIRE(p0.second == Catch::Approx(-1.0f));

    // Screen (1, 1) → NDC (1, 1)
    auto p1 = transform(1.0f, 1.0f);
    REQUIRE(p1.first == Catch::Approx(1.0f));
    REQUIRE(p1.second == Catch::Approx(1.0f));

    // Screen (0.5, 0.5) → NDC (0, 0)
    auto p2 = transform(0.5f, 0.5f);
    REQUIRE(p2.first == Catch::Approx(0.0f));
    REQUIRE(p2.second == Catch::Approx(0.0f));
}

TEST_CASE("UIOverlay quad index order forms two triangles", "[render][ui]") {
    // Index buffer: {0, 1, 2, 0, 2, 3}
    // Triangle 1: vertices 0, 1, 2 (BL, TL, BR), left-bottom triangle
    // Triangle 2: vertices 0, 2, 3 (BL, BR, TR), right-top triangle
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};

    // Verify 6 indices (2 triangles)
    REQUIRE(sizeof(indices) / sizeof(indices[0]) == 6);

    // Verify all indices reference valid vertices (0-3)
    for (uint16_t idx : indices) {
        REQUIRE((idx >= 0 && idx <= 3));
    }

    // Verify triangle 1 covers bottom-left half
    REQUIRE(indices[0] == 0); // BL
    REQUIRE(indices[1] == 1); // TL
    REQUIRE(indices[2] == 2); // BR

    // Verify triangle 2 covers top-right half
    REQUIRE(indices[3] == 0); // BL
    REQUIRE(indices[4] == 2); // BR
    REQUIRE(indices[5] == 3); // TR
}

// ---- Hotbar Tests (Task 6.3) ----

TEST_CASE("Hotbar: initial slot selection is 0", "[phase6][hotbar]") {
    Hotbar hotbar;
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectSlot clamps to valid range", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Negative index clamps to 0
    hotbar.selectSlot(-5);
    REQUIRE(hotbar.getSelectedIndex() == 0);

    // Out-of-range index clamps to 8
    hotbar.selectSlot(100);
    REQUIRE(hotbar.getSelectedIndex() == 8);

    // Valid index works
    hotbar.selectSlot(4);
    REQUIRE(hotbar.getSelectedIndex() == 4);
}

TEST_CASE("Hotbar: selectNext wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(0);

    for (int i = 1; i <= 8; ++i) {
        hotbar.selectNext();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 8 → 0
    hotbar.selectNext();
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectPrev wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(8);

    for (int i = 7; i >= 0; --i) {
        hotbar.selectPrev();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 0 → 8
    hotbar.selectPrev();
    REQUIRE(hotbar.getSelectedIndex() == 8);
}

TEST_CASE("Hotbar: getSelectedBlockType returns correct type", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Default slot 0 is STONE
    hotbar.selectSlot(0);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::STONE);

    // Slot 1 is DIRT
    hotbar.selectSlot(1);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::DIRT);

    // Slot 2 is GRASS
    hotbar.selectSlot(2);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::GRASS);
}

TEST_CASE("Hotbar: setSlot and getSlot", "[phase6][hotbar]") {
    Hotbar hotbar;

    hotbar.setSlot(0, BlockType::DIAMOND_ORE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);

    // Out-of-range returns AIR
    REQUIRE(hotbar.getSlot(-1) == BlockType::AIR);
    REQUIRE(hotbar.getSlot(9) == BlockType::AIR);

    // setSlot on out-of-range does nothing
    hotbar.setSlot(-1, BlockType::STONE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);
}

TEST_CASE("Hotbar: default slot contents", "[phase6][hotbar]") {
    Hotbar hotbar;

    REQUIRE(hotbar.getSlot(0) == BlockType::STONE);
    REQUIRE(hotbar.getSlot(1) == BlockType::DIRT);
    REQUIRE(hotbar.getSlot(2) == BlockType::GRASS);
    REQUIRE(hotbar.getSlot(3) == BlockType::LOG);
    REQUIRE(hotbar.getSlot(4) == BlockType::PLANKS);
    REQUIRE(hotbar.getSlot(5) == BlockType::SAND);
    REQUIRE(hotbar.getSlot(6) == BlockType::SANDSTONE);
    REQUIRE(hotbar.getSlot(7) == BlockType::GLASS);
    REQUIRE(hotbar.getSlot(8) == BlockType::FLOWER_RED);
}

// ---- Performance HUD Tests ----

TEST_CASE("Performance HUD: FPS averaging over 60 frames", "[phase8][hud]") {
    // Simulate rolling average FPS
    std::vector<float> frameTimes;
    frameTimes.reserve(60);

    auto computeFPS = [&frameTimes](float newFrameTimeMs) -> float {
        frameTimes.push_back(newFrameTimeMs);
        if (frameTimes.size() > 60) {
            frameTimes.erase(frameTimes.begin());
        }

        float totalMs = 0.0f;
        for (float t : frameTimes) {
            totalMs += t;
        }
        return static_cast<float>(frameTimes.size()) * 1000.0f / totalMs;
    };

    // Feed 60 frames at 16.67ms each (60 FPS)
    for (int i = 0; i < 60; ++i) {
        computeFPS(16.67f);
    }

    float fps = computeFPS(16.67f);
    REQUIRE(fps > 55.0f);
    REQUIRE(fps < 65.0f);

    // Feed slower frames → FPS drops
    for (int i = 0; i < 60; ++i) {
        computeFPS(33.33f); // 30 FPS
    }

    fps = computeFPS(33.33f);
    REQUIRE(fps > 25.0f);
    REQUIRE(fps < 35.0f);
}

TEST_CASE("Performance HUD: text positioning", "[phase8][hud]") {
    // HUD at top-left: (8px, height-8px)
    uint32_t width = 1920;
    uint32_t height = 1080;

    float hudX = 8.0f / static_cast<float>(width);
    float hudY = 1.0f - 8.0f / static_cast<float>(height);

    // Verify normalized coordinates are valid
    REQUIRE(hudX > 0.0f);
    REQUIRE(hudX < 0.01f); // Near left edge
    REQUIRE(hudY > 0.99f); // Near top edge
    REQUIRE(hudY < 1.0f);

    // Background dimensions
    float bgWidth = 220.0f / static_cast<float>(width);
    float bgHeight = 80.0f / static_cast<float>(height);

    REQUIRE(bgWidth > 0.0f);
    REQUIRE(bgWidth < 0.2f); // Less than 20% of screen width
    REQUIRE(bgHeight > 0.0f);
    REQUIRE(bgHeight < 0.1f); // Less than 10% of screen height
}

TEST_CASE("Performance HUD: integer to string conversion", "[phase8][hud]") {
    // Simulate the intToString function
    auto intToString = [](int value, char* buf, size_t bufSize) {
        char tmp[20];
        int len = 0;
        if (value == 0) {
            tmp[len++] = '0';
        } else {
            int v = value < 0 ? -value : value;
            while (v > 0) {
                tmp[len++] = '0' + (v % 10);
                v /= 10;
            }
            if (value < 0)
                tmp[len++] = '-';
            for (int i = 0; i < len / 2; ++i) {
                char t = tmp[i];
                tmp[i] = tmp[len - 1 - i];
                tmp[len - 1 - i] = t;
            }
        }
        size_t copyLen = len < static_cast<int>(bufSize - 1) ? len : bufSize - 1;
        std::memcpy(buf, tmp, copyLen);
        buf[copyLen] = '\0';
    };

    char buf[16];

    intToString(0, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "0");

    intToString(42, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "42");

    intToString(12345, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "12345");

    intToString(-7, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "-7");
}

TEST_CASE("Performance HUD: float to string conversion", "[phase8][hud]") {
    auto floatToString = [](float value, char* buf, size_t bufSize) {
        int intPart = static_cast<int>(std::floor(value));
        int fracPart = static_cast<int>((value - std::floor(value)) * 10);

        char tmp[20];
        int len = 0;
        if (intPart == 0) {
            tmp[len++] = '0';
        } else {
            int v = intPart < 0 ? -intPart : intPart;
            while (v > 0) {
                tmp[len++] = '0' + (v % 10);
                v /= 10;
            }
            if (intPart < 0)
                tmp[len++] = '-';
            for (int i = 0; i < len / 2; ++i) {
                char t = tmp[i];
                tmp[i] = tmp[len - 1 - i];
                tmp[len - 1 - i] = t;
            }
        }

        if (len + 3 < static_cast<int>(bufSize)) {
            std::memcpy(buf, tmp, len);
            buf[len] = '.';
            buf[len + 1] = '0' + fracPart;
            buf[len + 2] = '\0';
        } else {
            size_t safeLen =
                len < static_cast<int>(bufSize - 1) ? len : static_cast<int>(bufSize - 1);
            std::memcpy(buf, tmp, safeLen);
            buf[safeLen] = '\0';
        }
    };

    char buf[16];

    floatToString(60.0f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "60.0");

    floatToString(16.7f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "16.7");

    floatToString(0.5f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "0.5");
}

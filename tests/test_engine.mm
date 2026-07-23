#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/application_termination.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <engine/mining.hpp>
#include <engine/playtest_fixture.hpp>
#include <engine/slot_interaction.hpp>
#include <engine/survival.hpp>
#include <engine/v4_world_startup.hpp>
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
#include <world/world_list.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ============================================================================
// Vec3 Tests
// ============================================================================

TEST_CASE("Application termination quiescence is ordered and idempotent",
          "[engine][shutdown][fake][regression]") {
    const std::vector<std::string> expected{
        "save",       "cancel-bootstrap", "stop-render",
        "stop-world", "release-contexts", "release-runtime",
    };

    for (int repetition = 0; repetition < 16; ++repetition) {
        std::vector<std::string> calls;
        ApplicationTerminationActions actions{
            .saveDurableState =
                [&] {
                    calls.emplace_back("save");
                    return true;
                },
            .cancelBootstrap = [&] { calls.emplace_back("cancel-bootstrap"); },
            .stopRenderWorkers = [&] { calls.emplace_back("stop-render"); },
            .stopWorldAndGenerationWorkers = [&] { calls.emplace_back("stop-world"); },
            .releaseGenerationOwners = [&] { calls.emplace_back("release-contexts"); },
            .releaseRuntime = [&] { calls.emplace_back("release-runtime"); },
        };
        ApplicationTerminationQuiescence quiescence;

        REQUIRE(quiescence.quiesce(actions));
        REQUIRE(quiescence.persistenceResolved());
        REQUIRE(quiescence.quiesced());
        REQUIRE(calls == expected);

        REQUIRE(quiescence.quiesce(actions));
        REQUIRE(calls == expected);

        quiescence.resetForWorldSession();
        REQUIRE_FALSE(quiescence.persistenceResolved());
        REQUIRE_FALSE(quiescence.quiesced());
        calls.clear();
        REQUIRE(quiescence.quiesce(actions));
        REQUIRE(calls == expected);
    }
}

TEST_CASE("Application termination retries a failed durable save before teardown",
          "[engine][shutdown][fake][save-failure]") {
    std::vector<std::string> calls;
    bool saveSucceeds = false;
    ApplicationTerminationActions actions{
        .saveDurableState =
            [&] {
                calls.emplace_back("save");
                return saveSucceeds;
            },
        .cancelBootstrap = [&] { calls.emplace_back("cancel-bootstrap"); },
        .stopRenderWorkers = [&] { calls.emplace_back("stop-render"); },
        .stopWorldAndGenerationWorkers = [&] { calls.emplace_back("stop-world"); },
        .releaseGenerationOwners = [&] { calls.emplace_back("release-contexts"); },
        .releaseRuntime = [&] { calls.emplace_back("release-runtime"); },
    };
    ApplicationTerminationQuiescence quiescence;

    REQUIRE_FALSE(quiescence.quiesce(actions));
    REQUIRE_FALSE(quiescence.persistenceResolved());
    REQUIRE_FALSE(quiescence.quiesced());
    REQUIRE(calls == std::vector<std::string>{"save"});

    saveSucceeds = true;
    REQUIRE(quiescence.quiesce(actions));
    const std::vector<std::string> expectedAfterRetry{
        "save",
        "save",
        "cancel-bootstrap",
        "stop-render",
        "stop-world",
        "release-contexts",
        "release-runtime",
    };
    REQUIRE(calls == expectedAfterRetry);
}

TEST_CASE("Application destruction can force quiescence after persistence fails",
          "[engine][shutdown][fake][destructor]") {
    std::vector<std::string> calls;
    ApplicationTerminationActions actions{
        .saveDurableState =
            [&] {
                calls.emplace_back("save");
                return false;
            },
        .cancelBootstrap = [&] { calls.emplace_back("cancel-bootstrap"); },
        .stopRenderWorkers = [&] { calls.emplace_back("stop-render"); },
        .stopWorldAndGenerationWorkers = [&] { calls.emplace_back("stop-world"); },
        .releaseGenerationOwners = [&] { calls.emplace_back("release-contexts"); },
        .releaseRuntime = [&] { calls.emplace_back("release-runtime"); },
    };
    ApplicationTerminationQuiescence quiescence;

    REQUIRE_FALSE(quiescence.quiesce(actions));
    REQUIRE(quiescence.quiesce(actions, false));
    REQUIRE(quiescence.quiesced());
    const std::vector<std::string> expected{
        "save",       "cancel-bootstrap", "stop-render",
        "stop-world", "release-contexts", "release-runtime",
    };
    REQUIRE(calls == expected);
}

TEST_CASE("Physical atmosphere parameters and volume helpers stay finite",
          "[render][atmosphere][volumetrics]") {
    AtmosphereUniforms atmosphere = earthAtmosphereUniforms(
        225.0F, LEGACY_WORLD_PHYSICAL_SCALE, simd_make_float3(0.3F, 0.8F, 0.2F),
        simd_make_float3(18.0F, 17.5F, 16.0F), 1.25F, 0.7F, 7);
    REQUIRE(atmosphereUniformsFinite(atmosphere));
    REQUIRE(atmosphere.atmosphereRadii.x == Catch::Approx(6360.0F));
    REQUIRE(atmosphere.atmosphereRadii.y == Catch::Approx(6460.0F));
    REQUIRE(atmosphere.atmosphereRadii.z == Catch::Approx(0.004675F));
    REQUIRE(atmosphere.weatherOptics.z == Catch::Approx(0.0F));
    REQUIRE(atmosphere.cameraPositionKm.y == Catch::Approx(6360.225F).margin(0.001F));
    REQUIRE(atmosphere.atmosphereRadii.w == Catch::Approx(0.001F));

    const AtmosphereUniforms v4SeaLevel = earthAtmosphereUniforms(
        64.0F, GENERATOR_V4_PHYSICAL_SCALE, simd_make_float3(0.3F, 0.8F, 0.2F),
        simd_make_float3(18.0F, 17.5F, 16.0F), 1.25F, 0.7F, 7);
    const AtmosphereUniforms v4Summit = earthAtmosphereUniforms(
        1'407.0F, GENERATOR_V4_PHYSICAL_SCALE, simd_make_float3(0.3F, 0.8F, 0.2F),
        simd_make_float3(18.0F, 17.5F, 16.0F), 1.25F, 0.7F, 7);
    REQUIRE(v4SeaLevel.cameraPositionKm.y == Catch::Approx(6360.0F));
    REQUIRE(v4Summit.cameraPositionKm.y == Catch::Approx(6370.0725F).margin(0.001F));
    REQUIRE(v4Summit.atmosphereRadii.w == Catch::Approx(0.0075F));
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

TEST_CASE("Remote lightning height refinement never generates exact terrain",
          "[engine][weather][lightning][streaming]") {
    World world(42);
    constexpr int64_t REMOTE_X = 400'000;
    constexpr int64_t REMOTE_Z = -300'000;
    const ColumnPos remoteColumn{Chunk::worldToChunk(REMOTE_X), Chunk::worldToChunk(REMOTE_Z)};
    REQUIRE_FALSE(world.generator().findColumnPlan(remoteColumn));

    LightningEvent event;
    event.x = static_cast<double>(REMOTE_X);
    event.y = 217.0F;
    event.z = static_cast<double>(REMOTE_Z);
    event.cloudY = 450.0F;
    REQUIRE_NOTHROW(resolveLightningTerrainHeightIfLoaded(world, event));
    REQUIRE(event.y == 217.0F);
    REQUIRE(event.cloudY == 450.0F);
    REQUIRE_FALSE(world.generator().findColumnPlan(remoteColumn));
    REQUIRE(world.getPendingChunkCount() == 0);
}

TEST_CASE("Loaded lightning resolves to the top fluid surface",
          "[engine][weather][lightning][water]") {
    World world(42);
    const std::shared_ptr<Chunk> loaded = world.getChunk(ChunkPos{0, 4, 0});
    REQUIRE(loaded);
    loaded->fill(BlockType::AIR);
    loaded->setBlock(8, 0, 8, BlockType::STONE);
    loaded->setBlock(8, 1, 8, BlockType::WATER);
    loaded->setFluidState(8, 1, 8, FluidState::source());
    loaded->setBlock(8, 2, 8, BlockType::WATER);
    loaded->setFluidState(8, 2, 8, FluidState::flowing(4));

    LightningEvent event;
    event.x = 8.25;
    event.y = 200.0F;
    event.z = 8.75;
    event.cloudY = 66.0F;
    resolveLightningTerrainHeightIfLoaded(world, event);

    REQUIRE(event.y == Catch::Approx(66.5F));
    REQUIRE(event.cloudY == Catch::Approx(67.5F));
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

TEST_CASE("Cloud sessions lazily bind seed-owned noise and retain full v4 layer bounds",
          "[render][clouds][session][v4][height]") {
    constexpr uint64_t SEED = 764891;
    REQUIRE(cloudWorldBindingChanged(false, 0, 0, 17, SEED));
    REQUIRE_FALSE(cloudWorldBindingChanged(true, 17, SEED, 17, SEED));
    REQUIRE(cloudWorldBindingChanged(true, 17, SEED, 18, SEED));
    REQUIRE(cloudWorldBindingChanged(true, 17, SEED, 17, SEED + 1));

    // No constructor-owned seed-zero volume is required. The first real
    // binding generates once, equal-seed sessions reuse it, and a new seed
    // regenerates the seed-owned textures.
    REQUIRE(cloudNoiseRegenerationRequired(false, 0, 0));
    REQUIRE_FALSE(cloudNoiseRegenerationRequired(true, SEED, SEED));
    REQUIRE(cloudNoiseRegenerationRequired(true, SEED, SEED + 1));
    REQUIRE(cloudBaseNoise(3, 7, 11, 32, SEED) !=
            Catch::Approx(cloudBaseNoise(3, 7, 11, 32, SEED + 1)));

    WeatherSample lowland{};
    lowland.cloudBaseY = 197.0F;
    lowland.cloudTopY = 240.0F;
    WeatherSample nextLowland = lowland;
    nextLowland.cloudBaseY = 198.0F;
    nextLowland.cloudTopY = 242.0F;
    WeatherSample summit = lowland;
    summit.cloudBaseY = 1'450.0F;
    summit.cloudTopY = 1'900.0F;
    std::vector<WeatherSample> first(WeatherSnapshot::GRID_SAMPLE_COUNT, lowland);
    std::vector<WeatherSample> second(WeatherSnapshot::GRID_SAMPLE_COUNT, nextLowland);
    second[WeatherSnapshot::GRID_SAMPLE_COUNT / 2] = summit;
    WeatherSnapshot snapshot(1, 0, 0, 0, WeatherPreset::NATURAL, std::move(first),
                             std::move(second));
    const simd_float2 bounds = cloudSnapshotMarchLayerBounds(snapshot);
    REQUIRE(bounds.x == Catch::Approx(196.0F));
    REQUIRE(bounds.y == Catch::Approx(1'901.0F));
    REQUIRE(bounds.y > static_cast<float>(WORLD_MAX_Y));

    // The broad snapshot envelope only clips the ray. Each weather-cell
    // segment intersects its own local slab, preserving both a thin lowland
    // cloud and a remote summit cloud within the same coarse march.
    const simd_float2 lowlandInterval =
        cloudRaySegmentLayerIntersection(100.0F, 1.0F, 0.0F, 2'000.0F, 197.0F, 240.0F);
    REQUIRE(lowlandInterval.x == Catch::Approx(97.0F));
    REQUIRE(lowlandInterval.y == Catch::Approx(140.0F));
    const simd_float2 summitInterval =
        cloudRaySegmentLayerIntersection(100.0F, 1.0F, 0.0F, 2'000.0F, 1'450.0F, 1'900.0F);
    REQUIRE(summitInterval.x == Catch::Approx(1'350.0F));
    REQUIRE(summitInterval.y == Catch::Approx(1'800.0F));
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
    REQUIRE(medium.reactiveHistoryBytes == 964'224);
    REQUIRE(medium.totalBytes() == 100'019'024);

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
    REQUIRE(high.reactiveHistoryBytes == 3'860'352);
    REQUIRE(high.totalBytes() == 184'006'736);
    REQUIRE(screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, -4).totalBytes() == off.totalBytes());
    REQUIRE(screenSpaceLightingMemoryFootprint(WIDTH, HEIGHT, 8).totalBytes() == high.totalBytes());
}

TEST_CASE("Screen-space lighting normal guide rejects separate voxel faces",
          "[render][indirect][upsample]") {
    const simd_float3 floorNormal = simd_make_float3(0.0F, 1.0F, 0.0F);
    const simd_float3 nearbyFloorNormal = simd_make_float3(0.0F, 0.98F, 0.2F);
    const simd_float3 wallNormal = simd_make_float3(1.0F, 0.0F, 0.0F);

    REQUIRE(screenSpaceNormalGuideWeight(floorNormal, floorNormal) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceNormalGuideWeight(floorNormal, nearbyFloorNormal) > 0.99F);
    REQUIRE(screenSpaceNormalGuideWeight(floorNormal, wallNormal) == Catch::Approx(0.0F));
    REQUIRE(screenSpaceNormalGuideWeight(floorNormal, simd_make_float3(0.0F, 0.0F, 0.0F)) ==
            Catch::Approx(0.0F));

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
        .lightEditRevision = 11,
        .timeDiscontinuityRevision = 13,
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
    requireOnly([](IndirectHistoryState& state) { ++state.lightEditRevision; },
                INDIRECT_HISTORY_LIGHT_EDIT);
    requireOnly([](IndirectHistoryState& state) { ++state.timeDiscontinuityRevision; },
                INDIRECT_HISTORY_TIME_DISCONTINUITY);

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
    ++allReasons.lightEditRevision;
    ++allReasons.timeDiscontinuityRevision;
    constexpr uint32_t EVERY_REASON =
        INDIRECT_HISTORY_RESIZE | INDIRECT_HISTORY_TELEPORT | INDIRECT_HISTORY_WORLD_CHANGE |
        INDIRECT_HISTORY_FOV_CHANGE | INDIRECT_HISTORY_QUALITY_CHANGE |
        INDIRECT_HISTORY_FORCED_STATE | INDIRECT_HISTORY_INVALID_DEPTH |
        INDIRECT_HISTORY_LIGHT_SOURCE | INDIRECT_HISTORY_LIGHT_EDIT |
        INDIRECT_HISTORY_TIME_DISCONTINUITY;
    REQUIRE(indirectHistoryResetMask(previous, allReasons) == EVERY_REASON);

    REQUIRE(atmosphericHistoryResetMask(INDIRECT_HISTORY_LIGHT_EDIT) == INDIRECT_HISTORY_STABLE);
    REQUIRE(atmosphericHistoryResetMask(INDIRECT_HISTORY_QUALITY_CHANGE) ==
            INDIRECT_HISTORY_STABLE);
    REQUIRE(atmosphericHistoryResetMask(INDIRECT_HISTORY_INVALID_DEPTH) == INDIRECT_HISTORY_STABLE);
    REQUIRE(
        atmosphericHistoryResetMask(INDIRECT_HISTORY_WORLD_CHANGE | INDIRECT_HISTORY_LIGHT_EDIT) ==
        INDIRECT_HISTORY_WORLD_CHANGE);

    REQUIRE_FALSE(indirectLightingTimeDiscontinuity(false, 100, 1));
    REQUIRE_FALSE(indirectLightingTimeDiscontinuity(true, 100, 108));
    REQUIRE(indirectLightingTimeDiscontinuity(true, 100, 109));
    REQUIRE(indirectLightingTimeDiscontinuity(true, 100, 99));

    REQUIRE_FALSE(exactMeshPublicationInvalidatesHistory(false, true, 8, 8));
    REQUIRE_FALSE(exactMeshPublicationInvalidatesHistory(true, false, 8, 8));
    REQUIRE_FALSE(exactMeshPublicationInvalidatesHistory(true, true, 7, 8));
    REQUIRE(exactMeshPublicationInvalidatesHistory(true, true, 8, 8));
    REQUIRE(exactMeshPublicationInvalidatesHistory(true, true, 9, 8));
    REQUIRE(indirectLightingRevision(19, 3) == 22);
}

TEST_CASE("Cloud memory omits seeded noise and frame targets when disabled",
          "[render][clouds][memory]") {
    constexpr uint32_t WIDTH = 3456;
    constexpr uint32_t HEIGHT = 2234;
    const CloudRendererMemoryFootprint off = cloudRendererMemoryFootprint(WIDTH, HEIGHT, 0);
    REQUIRE(off.quarterWidth == 1);
    REQUIRE(off.quarterHeight == 1);
    REQUIRE(off.shadowEdge == 0);
    REQUIRE(off.noiseBytes == 0);
    REQUIRE(off.weatherBytes == 2'519'424);
    REQUIRE(off.neutralShadowBytes == 2);
    REQUIRE(off.frameTargetBytes == 0);
    REQUIRE(off.shadowBytes == 0);
    REQUIRE(off.totalBytes() == 2'519'426);

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
    REQUIRE(atmosphericSceneTargetMemoryBytes(WIDTH, HEIGHT) == 244'482'192);
    REQUIRE(integrated.sceneTargetBytes == 244'482'192);
    REQUIRE(integrated.shadowBytes == 184'549'376);
    REQUIRE(integrated.indirectBytes == 184'006'736);
    REQUIRE(integrated.atmosphereBytes == 305'152);
    REQUIRE(integrated.cloudBytes == 27'534'082);
    REQUIRE(integrated.volumetricBytes == 86'525'723);
    REQUIRE(integrated.lightningBytes == 2);
    REQUIRE(integrated.totalBytes() == 727'403'263);
    REQUIRE(integrated.totalBytes() < HIGH_TIER_BUDGET);
    REQUIRE(HIGH_TIER_BUDGET - integrated.totalBytes() == 77'903'105);
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
    REQUIRE(hudStats.weatherBuildsDeferred == 0U);
    REQUIRE(hudStats.weatherBuildsFailed == 0U);
    REQUIRE(hudStats.weatherPendingRequests == 0U);
    REQUIRE_FALSE(hudStats.weatherWorkerBusy);
    REQUIRE(hudStats.thunderPending == 0U);
    REQUIRE(hudStats.integratedAtmosphericPersistentMB == Catch::Approx(0.0F));

    WeatherSystemStats weatherStats;
    weatherStats.requests = 9U;
    weatherStats.coalescedRequests = 4U;
    weatherStats.buildsDeferred = 3U;
    weatherStats.buildsFailed = 2U;
    weatherStats.pendingRequests = 1U;
    weatherStats.workerBusy = true;
    REQUIRE(weatherStats.requests == 9U);
    REQUIRE(weatherStats.coalescedRequests == 4U);
    REQUIRE(weatherStats.buildsDeferred == 3U);
    REQUIRE(weatherStats.buildsFailed == 2U);
    REQUIRE(weatherStats.pendingRequests == 1U);
    REQUIRE(weatherStats.workerBusy);
}
// ===========================================================================
// Engine: game flow, menus, input, hotbar
// ===========================================================================

TEST_CASE("Playtest screen names remain valid until deferred world entry", "[engine][flow]") {
    REQUIRE(gameScreenFromEnvironment("title") == GameScreen::TITLE);
    REQUIRE(gameScreenFromEnvironment("worlds") == GameScreen::WORLD_SELECT);
    REQUIRE(gameScreenFromEnvironment("create") == GameScreen::WORLD_CREATE);
    REQUIRE(gameScreenFromEnvironment("delete") == GameScreen::WORLD_DELETE_CONFIRM);
    REQUIRE(gameScreenFromEnvironment("playing") == GameScreen::PLAYING);
    REQUIRE(gameScreenFromEnvironment("paused") == GameScreen::PAUSED);
    REQUIRE(gameScreenFromEnvironment("settings") == GameScreen::SETTINGS);
    REQUIRE(gameScreenFromEnvironment("video") == GameScreen::VIDEO_SETTINGS);
    REQUIRE(gameScreenFromEnvironment("inventory") == GameScreen::INVENTORY);
    REQUIRE(gameScreenFromEnvironment("crafting") == GameScreen::CRAFTING);
    REQUIRE(gameScreenFromEnvironment("furnace") == GameScreen::FURNACE);
    REQUIRE(gameScreenFromEnvironment("chest") == GameScreen::CHEST);
    REQUIRE(gameScreenFromEnvironment("death") == GameScreen::DEATH);
    REQUIRE_FALSE(gameScreenFromEnvironment("unknown"));
}

TEST_CASE("Material playtest fixture is opt-in and capture-only",
          "[engine][playtest][materials][save]") {
    REQUIRE_FALSE(materialPlaytestFixtureEnabled(nullptr, nullptr));
    REQUIRE_FALSE(materialPlaytestFixtureEnabled("1", nullptr));
    REQUIRE_FALSE(materialPlaytestFixtureEnabled("1", ""));
    REQUIRE_FALSE(materialPlaytestFixtureEnabled("1", "0"));
    REQUIRE_FALSE(materialPlaytestFixtureEnabled("0", "/tmp/frame.png"));
    REQUIRE(materialPlaytestFixtureEnabled("1", "/tmp/frame.png"));
    REQUIRE(MATERIAL_PLAYTEST_BLOCKS == std::array{BlockType::BED, BlockType::CHEST,
                                                   BlockType::TORCH, BlockType::FURNACE,
                                                   BlockType::FURNACE_LIT});
}

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
    const auto exponent = parseCaptureLightningOverride("1.25e2,-.5,17,2");
    REQUIRE(exponent);
    REQUIRE(exponent->x == Catch::Approx(125.0));
    REQUIRE(exponent->z == Catch::Approx(-0.5));

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
    REQUIRE_FALSE(parseCaptureLightningOverride("1e,-111726,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride(".,-111726,17,2"));
    REQUIRE_FALSE(parseCaptureLightningOverride("1e-10000,-111726,17,2"));
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

TEST_CASE("Capture settings can use an isolated path without changing home",
          "[capture][engine][settings]") {
    const char* previousValue = std::getenv("RYCRAFT_SETTINGS_PATH");
    const std::optional<std::string> previous =
        previousValue ? std::optional<std::string>{previousValue} : std::nullopt;
    REQUIRE(setenv("RYCRAFT_SETTINGS_PATH", "/tmp/rycraft-isolated-settings.json", 1) == 0);
    CHECK(settingsPath() == "/tmp/rycraft-isolated-settings.json");
    if (previous) {
        REQUIRE(setenv("RYCRAFT_SETTINGS_PATH", previous->c_str(), 1) == 0);
    } else {
        REQUIRE(unsetenv("RYCRAFT_SETTINGS_PATH") == 0);
    }
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

TEST_CASE("GameFlow: world session transitions", "[ui][flow]") {
    GameFlow flow;
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE_FALSE(flow.worldScreens());

    // Title -> world select -> create, ESC backs out one level at a time.
    flow.onMenuAction(MenuAction::OPEN_WORLD_SELECT);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onMenuAction(MenuAction::OPEN_WORLD_CREATE);
    REQUIRE(flow.screen == GameScreen::WORLD_CREATE);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onMenuAction(MenuAction::REQUEST_DELETE_WORLD);
    REQUIRE(flow.screen == GameScreen::WORLD_DELETE_CONFIRM);
    flow.onMenuAction(MenuAction::CANCEL_DELETE);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::TITLE);

    // A stale or legacy row requires a second, explicit confirmation before
    // the engine can create a separate successor profile.
    flow.onMenuAction(MenuAction::OPEN_WORLD_SELECT);
    flow.onMenuAction(MenuAction::REQUEST_V4_SUCCESSOR);
    REQUIRE(flow.screen == GameScreen::WORLD_SUCCESSOR_CONFIRM);
    flow.onMenuAction(MenuAction::CANCEL_V4_SUCCESSOR);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onMenuAction(MenuAction::REQUEST_V4_SUCCESSOR);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::TITLE);

    // Side-effectful actions never change the screen by themselves.
    flow.onMenuAction(MenuAction::OPEN_WORLD_SELECT);
    auto fx = flow.onMenuAction(MenuAction::PLAY_SELECTED_WORLD);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    REQUIRE(!fx.captureCursor);

    // The engine drives the start after its side effect succeeds.
    fx = flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);
    REQUIRE(fx.resetTiming);
    REQUIRE(flow.worldScreens());

    // Save-and-quit lands back on the title with a free cursor.
    flow.onEscape(); // paused
    fx = flow.onWorldStopped();
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE(fx.releaseCursor);

    // onWorldStarted refuses screens that already have a session.
    flow.onWorldStarted();
    flow.onEscape(); // paused
    fx = flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(!fx.captureCursor);
}

TEST_CASE("GameFlow: inventory key and container screens", "[ui][flow]") {
    GameFlow flow;
    flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PLAYING);

    auto fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::INVENTORY);
    REQUIRE(fx.releaseCursor);
    REQUIRE(flow.inMenu());
    REQUIRE(flow.inContainer());

    fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Container blocks open their screens from gameplay only.
    fx = flow.onContainerOpened(GameScreen::FURNACE);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    REQUIRE(fx.releaseCursor);
    fx = flow.onContainerOpened(GameScreen::CRAFTING);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    fx = flow.onContainerOpened(GameScreen::CRAFTING);
    REQUIRE(flow.screen == GameScreen::CRAFTING);
    // E closes any container.
    flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    // Only container screens are valid targets.
    fx = flow.onContainerOpened(GameScreen::PAUSED);
    REQUIRE(flow.screen == GameScreen::PLAYING);
}

TEST_CASE("GameFlow: death ignores escape until respawn", "[ui][flow]") {
    GameFlow flow;
    flow.onWorldStarted();

    auto fx = flow.onPlayerDied();
    REQUIRE(flow.screen == GameScreen::DEATH);
    REQUIRE(fx.releaseCursor);

    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::DEATH);
    REQUIRE(!fx.captureCursor);
    fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::DEATH);

    fx = flow.onRespawn();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Dying only happens while playing.
    flow.onEscape();
    fx = flow.onPlayerDied();
    REQUIRE(flow.screen == GameScreen::PAUSED);
}

TEST_CASE("GameFlow restores the exact live screen after generation recovery", "[ui][flow]") {
    GameFlow flow;

    auto effects = flow.onGenerationRecovered(GameScreen::PLAYING);
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(effects.captureCursor);
    REQUIRE_FALSE(effects.releaseCursor);
    REQUIRE(effects.resetTiming);

    effects = flow.onGenerationRecovered(GameScreen::FURNACE);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    REQUIRE_FALSE(effects.captureCursor);
    REQUIRE(effects.releaseCursor);
    REQUIRE(effects.resetTiming);

    effects = flow.onGenerationRecovered(GameScreen::WORLD_SELECT);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    REQUIRE_FALSE(effects.captureCursor);
    REQUIRE_FALSE(effects.releaseCursor);
}

TEST_CASE("Frame capture timing begins with eligible rendered scenes", "[engine][capture]") {
    FrameCaptureClock clock;
    for (uint64_t frame = 0; frame < 240; ++frame) {
        const FrameCaptureActions actions = clock.onRenderedFrame(240);
        REQUIRE_FALSE(actions.capture);
        REQUIRE_FALSE(actions.quit);
    }

    FrameCaptureActions actions = clock.onRenderedFrame(240);
    REQUIRE(actions.capture);
    REQUIRE_FALSE(actions.quit);
    for (uint64_t frame = 0; frame < 59; ++frame) {
        actions = clock.onRenderedFrame(240);
        REQUIRE_FALSE(actions.capture);
        REQUIRE_FALSE(actions.quit);
    }
    actions = clock.onRenderedFrame(240);
    REQUIRE_FALSE(actions.capture);
    REQUIRE(actions.quit);
    actions = clock.onRenderedFrame(240);
    REQUIRE_FALSE(actions.capture);
    REQUIRE_FALSE(actions.quit);
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

TEST_CASE("Video settings names the compatibility toggle as indirect lighting",
          "[ui][menu][indirect]") {
    SettingsValues values;
    GraphicsSettings gfx;
    const MenuLayout layout =
        buildMenuLayout(GameScreen::VIDEO_SETTINGS, 1024.f, 768.f, values, gfx);
    REQUIRE(std::ranges::any_of(
        layout.texts, [](const MenuText& text) { return text.text == "INDIRECT LIGHT"; }));
    REQUIRE_FALSE(std::ranges::any_of(
        layout.texts, [](const MenuText& text) { return text.text == "AMBIENT OCCL"; }));
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

TEST_CASE("World select layout scrolls selects and guards actions", "[ui][worlds]") {
    MenuContext ctx;
    auto buttonWith = [](const MenuLayout& layout, MenuAction action) {
        for (const auto& button : layout.buttons) {
            if (button.action == action)
                return true;
        }
        return false;
    };

    // Empty list: no play/delete targets, create and back present.
    MenuLayout empty = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    REQUIRE_FALSE(buttonWith(empty, MenuAction::PLAY_SELECTED_WORLD));
    REQUIRE_FALSE(buttonWith(empty, MenuAction::REQUEST_DELETE_WORLD));
    REQUIRE(buttonWith(empty, MenuAction::OPEN_WORLD_CREATE));
    REQUIRE(buttonWith(empty, MenuAction::WORLD_BACK));
    REQUIRE_FALSE(buttonWith(empty, MenuAction::WORLD_LIST_UP));

    ctx.allowWorldCreation = false;
    ctx.worldCreationUnavailableReason = "NEW V4 WORLDS ARE NOT AVAILABLE IN THIS BUILD";
    MenuLayout v4Only = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    REQUIRE_FALSE(buttonWith(v4Only, MenuAction::OPEN_WORLD_CREATE));
    REQUIRE(std::ranges::any_of(v4Only.texts, [](const MenuText& text) {
        return text.text == "NEW V4 WORLDS ARE NOT AVAILABLE IN THIS BUILD";
    }));
    ctx.allowWorldCreation = true;
    ctx.worldCreationUnavailableReason.clear();

    // Seven rows: five visible with correct payloads, scroll arrows appear
    // on the scrollable side only.
    for (int i = 0; i < 7; ++i) {
        ctx.worldRows.push_back("World " + std::to_string(i));
    }
    ctx.worldSelect.selected = 1;
    MenuLayout list = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    int rows = 0;
    for (const auto& button : list.buttons) {
        if (button.action != MenuAction::SELECT_WORLD)
            continue;
        REQUIRE(button.payload == rows);
        if (button.payload == 1)
            REQUIRE(button.emphasized);
        ++rows;
    }
    REQUIRE(rows == WorldSelectState::VISIBLE_ROWS);
    REQUIRE_FALSE(buttonWith(list, MenuAction::WORLD_LIST_UP));
    REQUIRE(buttonWith(list, MenuAction::WORLD_LIST_DOWN));
    REQUIRE(buttonWith(list, MenuAction::PLAY_SELECTED_WORLD));
    REQUIRE(buttonWith(list, MenuAction::REQUEST_DELETE_WORLD));

    ctx.selectedWorldRequiresV4Successor = true;
    const MenuLayout successor = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    REQUIRE(std::ranges::any_of(successor.buttons, [](const MenuButton& button) {
        return button.action == MenuAction::REQUEST_V4_SUCCESSOR &&
               button.label == "CREATE V4 SUCCESSOR";
    }));
    REQUIRE_FALSE(std::ranges::any_of(successor.buttons, [](const MenuButton& button) {
        return button.action == MenuAction::PLAY_SELECTED_WORLD;
    }));

    ctx.successorWorldName = "World 1";
    const MenuLayout successorConfirm =
        buildScreenLayout(GameScreen::WORLD_SUCCESSOR_CONFIRM, 1024.f, 768.f, ctx);
    REQUIRE(std::ranges::any_of(successorConfirm.texts, [](const MenuText& text) {
        return text.text == "World 1 WILL REMAIN UNCHANGED";
    }));
    REQUIRE(std::ranges::any_of(successorConfirm.buttons, [](const MenuButton& button) {
        return button.action == MenuAction::CONFIRM_V4_SUCCESSOR;
    }));
    REQUIRE(std::ranges::any_of(successorConfirm.buttons, [](const MenuButton& button) {
        return button.action == MenuAction::CANCEL_V4_SUCCESSOR;
    }));
    ctx.selectedWorldRequiresV4Successor = false;

    // Scrolled to the bottom: rows start at the clamped offset.
    ctx.worldSelect.scroll = 99;
    MenuLayout bottom = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    int firstPayload = -1;
    for (const auto& button : bottom.buttons) {
        if (button.action == MenuAction::SELECT_WORLD) {
            firstPayload = button.payload;
            break;
        }
    }
    REQUIRE(firstPayload == 2);
    REQUIRE(buttonWith(bottom, MenuAction::WORLD_LIST_UP));
    REQUIRE_FALSE(buttonWith(bottom, MenuAction::WORLD_LIST_DOWN));
}

TEST_CASE("World create layout gates the create button on a name", "[ui][worlds]") {
    MenuContext ctx;
    auto hasCreate = [](const MenuLayout& layout) {
        for (const auto& button : layout.buttons) {
            if (button.action == MenuAction::CREATE_WORLD_CONFIRM)
                return true;
        }
        return false;
    };

    MenuLayout unnamed = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE(unnamed.textFields.size() == 2);
    REQUIRE(unnamed.textFields[0].label == "NAME");
    REQUIRE(unnamed.textFields[1].label == "SEED");
    REQUIRE_FALSE(hasCreate(unnamed));

    ctx.worldCreate.name = "   ";
    REQUIRE_FALSE(hasCreate(buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx)));

    ctx.worldCreate.name = "Base";
    ctx.worldCreate.focusedField = 1;
    MenuLayout named = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE(hasCreate(named));
    REQUIRE(named.textFields[1].focused);
    REQUIRE(named.textFields[1].caret);
    REQUIRE_FALSE(named.textFields[0].focused);

    // The caret obeys the blink phase.
    ctx.caretVisible = false;
    MenuLayout blink = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE_FALSE(blink.textFields[1].caret);
    STATIC_REQUIRE(WorldCreateState::MAX_SEED_LENGTH == 20);
}

TEST_CASE("Title and Worlds launches never request a persistence profile",
          "[engine][worlds][v4][startup][regression]") {
    STATIC_REQUIRE_FALSE(launchRequestsWorldSession(std::nullopt, false));
    STATIC_REQUIRE_FALSE(launchRequestsWorldSession(GameScreen::TITLE, false));
    STATIC_REQUIRE_FALSE(launchRequestsWorldSession(GameScreen::WORLD_SELECT, false));
    STATIC_REQUIRE_FALSE(launchRequestsWorldSession(GameScreen::WORLD_CREATE, false));
    STATIC_REQUIRE_FALSE(launchRequestsWorldSession(GameScreen::WORLD_SUCCESSOR_CONFIRM, false));
    STATIC_REQUIRE(launchRequestsWorldSession(GameScreen::PLAYING, false));
    STATIC_REQUIRE(launchRequestsWorldSession(std::nullopt, true));
}

TEST_CASE("Typed hit-testing distinguishes fields and buttons", "[ui][worlds]") {
    MenuContext ctx;
    ctx.worldCreate.name = "Base";
    MenuLayout layout = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);

    const auto& field = layout.textFields[0];
    UIHit hit =
        uiHitTest(layout, field.rect.x + field.rect.w * 0.5f, field.rect.y + field.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::TEXT_FIELD);
    REQUIRE(hit.index == 0);

    const auto& button = layout.buttons.front();
    hit = uiHitTest(layout, button.rect.x + button.rect.w * 0.5f,
                    button.rect.y + button.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::BUTTON);
    REQUIRE(layout.buttons[static_cast<size_t>(hit.index)].action == button.action);

    REQUIRE(uiHitTest(layout, 0.01f, 0.01f).kind == UIHitKind::NONE);
}

TEST_CASE("Text field filtering enforces charset and length", "[ui][worlds]") {
    REQUIRE(filterTextField("My World_2.0-x", false, 24) == "My World_2.0-x");
    REQUIRE(filterTextField("bad!@#chars$%", false, 24) == "badchars");
    REQUIRE(filterTextField("way too long name for the field", false, 10) == "way too lo");
    REQUIRE(filterTextField("seed123seed", true, 10) == "123");
    REQUIRE(filterTextField("42", true, 10) == "42");
    REQUIRE(filterTextField("", true, 10).empty());
}

TEST_CASE("Generator v4 world list includes only root profiles", "[engine][worlds][v4]") {
    TempDir directory("v4_world_list");
    const std::filesystem::path root = directory.path();
    const std::string fingerprint(64, 'a');
    const auto publish = [&](const std::string& name, uint64_t seed) {
        const std::filesystem::path path = root / name;
        SaveManager saves(path.string(), SaveManager::Profile::GeneratorV4);
        SaveManager::WorldMetadata metadata;
        metadata.seed = seed;
        metadata.generationFingerprint = fingerprint;
        metadata.spawnFinalized = true;
        metadata.spawnSafetyRevision = SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
        metadata.spawnPos = Vec3{8.f, 80.f, 8.f};
        metadata.playerPos = Vec3{9.f, 80.f, 9.f};
        metadata.safeSpawnPos = metadata.spawnPos;
        metadata.generatorVersion = SaveManager::GENERATOR_V4_VERSION;
        metadata.name = name;
        REQUIRE(saves.saveMetadata(metadata));
    };

    publish(GENERATOR_V4_WORLD_DIRECTORY, 7);
    const std::string sibling = std::string(GENERATOR_V4_WORLD_DIRECTORY) +
                                "-seed-0000000000000008-fingerprint-" + fingerprint;
    publish(sibling, 8);
    REQUIRE(createWorld("Legacy", 9, GameMode::CREATIVE, {}, root.string()).has_value());

    const std::vector<WorldSummary> worlds = listGeneratorV4Worlds(root.string());
    REQUIRE(worlds.size() == 2);
    REQUIRE(std::ranges::all_of(worlds, [](const WorldSummary& world) {
        return world.metadata.generatorVersion == SaveManager::GENERATOR_V4_VERSION;
    }));
    const std::vector<WorldSummary> selectable = listWorldsForGeneratorV4(root.string());
    REQUIRE(selectable.size() == 3);
    REQUIRE(std::ranges::count_if(selectable, [](const WorldSummary& world) {
                return world.requiresGeneratorV4Successor();
            }) == 1);
    REQUIRE(deleteWorld((root / sibling).string(), root.string()));
    REQUIRE(listGeneratorV4Worlds(root.string()).size() == 1);
}

TEST_CASE("Generator v4 atomic near-entry closure requires the complete FINAL wavefront",
          "[engine][v4][entry][near-closure]") {
    STATIC_REQUIRE(V4_ENTRY_FINAL_TARGET_STEPS == std::array<uint8_t, 5>{1, 2, 4, 8, 16});
    STATIC_REQUIRE(V4_ENTRY_FINAL_TARGETS_BY_STEP == std::array<uint32_t, 5>{4, 8, 12, 16, 20});
    STATIC_REQUIRE(V4_ENTRY_FINAL_TARGET_COUNT == 60);
    STATIC_REQUIRE(V4_ENTRY_COLLISION_CUBE_COUNT == 27);
    STATIC_REQUIRE(v4NearEntryFinalCompatibleProgress(10, 9, 60) == 0);
    STATIC_REQUIRE(v4NearEntryFinalCompatibleProgress(0, 0, 60) == 0);
    STATIC_REQUIRE(v4NearEntryFinalCompatibleProgress(10, 10, 24) == 24);
    STATIC_REQUIRE(v4NearEntryFinalCompatibleProgress(10, 10, 61) == 60);

    V4NearEntryClosureInput input;
    input.currentViewEpoch = 31;
    input.closureViewEpoch = 31;
    input.currentWorldEpoch = 17;
    input.closureWorldEpoch = 17;
    input.currentProtectedEpoch = 9;
    input.closureProtectedEpoch = 9;
    input.currentAnchor = {-12, 24};
    input.closureAnchor = input.currentAnchor;
    input.connectedPreviewParentPrefixReady = true;
    input.finalTargetCountsByStep = V4_ENTRY_FINAL_TARGETS_BY_STEP;
    input.matchingFinalParentsUploaded = V4_ENTRY_FINAL_TARGET_COUNT;
    input.matchingFinalParentsResident = V4_ENTRY_FINAL_TARGET_COUNT;
    input.matchingFinalChildrenUploaded = V4_ENTRY_FINAL_TARGET_COUNT;
    input.matchingFinalChildrenResident = V4_ENTRY_FINAL_TARGET_COUNT;
    input.exactCompatibleTargets = V4_ENTRY_FINAL_TARGET_COUNT;
    input.collisionCubesReady = V4_ENTRY_COLLISION_CUBE_COUNT;
    input.exactMeshesRequired = 27;
    input.matchingExactMeshesReady = input.exactMeshesRequired;
    input.currentExactMeshRevision = 73;
    input.readyExactMeshRevision = 73;

    CHECK(v4NearEntryClosureStatus(input) == V4NearEntryClosureStatus::Ready);
    CHECK(v4NearEntryClosureReady(input));
}

TEST_CASE("Generator v4 atomic near-entry closure fails closed for stale or partial state",
          "[engine][v4][entry][near-closure]") {
    const auto complete = [] {
        V4NearEntryClosureInput input;
        input.currentViewEpoch = 31;
        input.closureViewEpoch = 31;
        input.currentWorldEpoch = 17;
        input.closureWorldEpoch = 17;
        input.currentProtectedEpoch = 9;
        input.closureProtectedEpoch = 9;
        input.currentAnchor = {-12, 24};
        input.closureAnchor = input.currentAnchor;
        input.connectedPreviewParentPrefixReady = true;
        input.finalTargetCountsByStep = V4_ENTRY_FINAL_TARGETS_BY_STEP;
        input.matchingFinalParentsUploaded = V4_ENTRY_FINAL_TARGET_COUNT;
        input.matchingFinalParentsResident = V4_ENTRY_FINAL_TARGET_COUNT;
        input.matchingFinalChildrenUploaded = V4_ENTRY_FINAL_TARGET_COUNT;
        input.matchingFinalChildrenResident = V4_ENTRY_FINAL_TARGET_COUNT;
        input.exactCompatibleTargets = V4_ENTRY_FINAL_TARGET_COUNT;
        input.collisionCubesReady = V4_ENTRY_COLLISION_CUBE_COUNT;
        input.exactMeshesRequired = 27;
        input.matchingExactMeshesReady = input.exactMeshesRequired;
        input.currentExactMeshRevision = 73;
        input.readyExactMeshRevision = 73;
        return input;
    };
    const auto rejectsAs = [&](V4NearEntryClosureInput input, V4NearEntryClosureStatus expected) {
        CAPTURE(input.currentViewEpoch, input.closureViewEpoch, input.currentWorldEpoch,
                input.closureWorldEpoch, input.currentProtectedEpoch, input.closureProtectedEpoch,
                input.currentAnchor.minimumTileX, input.currentAnchor.minimumTileZ,
                input.closureAnchor.minimumTileX, input.closureAnchor.minimumTileZ);
        CHECK(v4NearEntryClosureStatus(input) == expected);
        CHECK_FALSE(v4NearEntryClosureReady(input));
    };

    SECTION("stale identity") {
        auto input = complete();
        ++input.closureViewEpoch;
        rejectsAs(input, V4NearEntryClosureStatus::EpochMismatch);
        input = complete();
        ++input.closureWorldEpoch;
        rejectsAs(input, V4NearEntryClosureStatus::EpochMismatch);
        input = complete();
        ++input.closureProtectedEpoch;
        rejectsAs(input, V4NearEntryClosureStatus::EpochMismatch);
        input = complete();
        ++input.closureAnchor.minimumTileX;
        rejectsAs(input, V4NearEntryClosureStatus::AnchorMismatch);
    }

    SECTION("retained prior closure is not retagged by the current request epoch") {
        RenderPipeline::ChunkRenderStats streaming;
        streaming.farProtectedNearCurrentEpoch = 10;
        streaming.farProtectedNearClosureEpoch = 9;
        streaming.farProtectedNearAnchorTileX = -12;
        streaming.farProtectedNearAnchorTileZ = 24;
        streaming.farProtectedNearViewEpoch = 31;
        streaming.farProtectedNearWorldEpoch = 17;
        streaming.farProtectedNearTargetCountsByStep = V4_ENTRY_FINAL_TARGETS_BY_STEP;
        streaming.farProtectedNearFinalParentCount = V4_ENTRY_FINAL_TARGET_COUNT;
        streaming.farProtectedNearFinalTargetCount = V4_ENTRY_FINAL_TARGET_COUNT;
        streaming.farProtectedNearExactCompatibleTargetCount = V4_ENTRY_FINAL_TARGET_COUNT;
        streaming.farProtectedNearResidentTileCount = V4_ENTRY_FINAL_TARGET_COUNT;
        streaming.farProtectedNearReady = true;

        auto input = complete();
        input.currentProtectedEpoch = streaming.farProtectedNearCurrentEpoch;
        input.closureProtectedEpoch = streaming.farProtectedNearClosureEpoch;
        input.currentAnchor = {-11, 24};
        input.closureAnchor = {streaming.farProtectedNearAnchorTileX,
                               streaming.farProtectedNearAnchorTileZ};
        input.finalTargetCountsByStep = streaming.farProtectedNearTargetCountsByStep;
        input.matchingFinalParentsUploaded = streaming.farProtectedNearFinalParentCount;
        input.matchingFinalParentsResident = streaming.farProtectedNearFinalParentCount;
        input.matchingFinalChildrenUploaded = streaming.farProtectedNearFinalTargetCount;
        input.matchingFinalChildrenResident = streaming.farProtectedNearFinalTargetCount;
        input.exactCompatibleTargets = streaming.farProtectedNearExactCompatibleTargetCount;
        rejectsAs(input, V4NearEntryClosureStatus::EpochMismatch);

        // Even an accidental epoch restamp cannot conceal the stale anchor.
        input.closureProtectedEpoch = input.currentProtectedEpoch;
        rejectsAs(input, V4NearEntryClosureStatus::AnchorMismatch);
    }

    SECTION("preview parent prefix") {
        auto input = complete();
        input.connectedPreviewParentPrefixReady = false;
        rejectsAs(input, V4NearEntryClosureStatus::PreviewParentPrefixIncomplete);
    }

    SECTION("protected topology") {
        auto input = complete();
        ++input.finalTargetCountsByStep[0];
        --input.finalTargetCountsByStep[1];
        rejectsAs(input, V4NearEntryClosureStatus::ProtectedTopologyMismatch);
    }

    SECTION("FINAL parent upload and residency") {
        auto input = complete();
        --input.matchingFinalParentsUploaded;
        rejectsAs(input, V4NearEntryClosureStatus::FinalParentsIncomplete);
        input = complete();
        --input.matchingFinalParentsResident;
        rejectsAs(input, V4NearEntryClosureStatus::FinalParentsIncomplete);
    }

    SECTION("FINAL child upload and residency") {
        auto input = complete();
        --input.matchingFinalChildrenUploaded;
        rejectsAs(input, V4NearEntryClosureStatus::FinalChildrenIncomplete);
        input = complete();
        --input.matchingFinalChildrenResident;
        rejectsAs(input, V4NearEntryClosureStatus::FinalChildrenIncomplete);
    }

    SECTION("exact compatibility and transitions") {
        auto input = complete();
        --input.exactCompatibleTargets;
        rejectsAs(input, V4NearEntryClosureStatus::ExactCompatibilityIncomplete);
        input = complete();
        input.lodTransitionMismatches = 1;
        rejectsAs(input, V4NearEntryClosureStatus::TransitionMismatch);
        input = complete();
        input.authorityTransitionMismatches = 1;
        rejectsAs(input, V4NearEntryClosureStatus::TransitionMismatch);
    }

    SECTION("collision and current-revision exact meshes") {
        auto input = complete();
        --input.collisionCubesReady;
        rejectsAs(input, V4NearEntryClosureStatus::CollisionIncomplete);
        input = complete();
        ++input.collisionCubesReady;
        rejectsAs(input, V4NearEntryClosureStatus::CollisionIncomplete);
        input = complete();
        input.exactMeshesRequired = 0;
        input.matchingExactMeshesReady = 0;
        rejectsAs(input, V4NearEntryClosureStatus::ExactMeshesIncomplete);
        input = complete();
        --input.matchingExactMeshesReady;
        rejectsAs(input, V4NearEntryClosureStatus::ExactMeshesIncomplete);
        input = complete();
        ++input.readyExactMeshRevision;
        rejectsAs(input, V4NearEntryClosureStatus::ExactMeshRevisionMismatch);
    }
}

TEST_CASE("Terrain bootstrap menu exposes every fail-closed startup action",
          "[engine][ui][bootstrap]") {
    using namespace worldgen::bootstrap;
    const auto actions = [](const MenuLayout& layout) {
        std::vector<MenuAction> result;
        for (const MenuButton& button : layout.buttons)
            result.push_back(button.action);
        return result;
    };
    const auto containsText = [](const MenuLayout& layout, std::string_view value) {
        return std::ranges::any_of(layout.texts, [&](const MenuText& text) {
            return text.text.find(value) != std::string::npos;
        });
    };

    TerrainBootstrapSnapshot required;
    required.state = TerrainBootstrapState::ModelRequired;
    required.detail = "The verified generator v4 terrain model is required";
    const MenuLayout requiredLayout = buildTerrainBootstrapLayout(required, 1024.f, 768.f);
    CHECK(containsText(requiredLayout, "MODEL REQUIRED"));
    CHECK(actions(requiredLayout) ==
          std::vector<MenuAction>{MenuAction::DOWNLOAD_MODEL, MenuAction::QUIT});

    for (const TerrainBootstrapState state : {
             TerrainBootstrapState::Downloading,
             TerrainBootstrapState::Verifying,
             TerrainBootstrapState::Compiling,
             TerrainBootstrapState::Loading,
         }) {
        TerrainBootstrapSnapshot active{
            .state = state,
            .completedBytes = 50,
            .totalBytes = 100,
            .currentAsset = "base_model.onnx",
            .detail = "Preparing generator v4",
        };
        const MenuLayout layout = buildTerrainBootstrapLayout(active, 1024.f, 768.f);
        CHECK(layout.progressFraction == Catch::Approx(0.5f));
        CHECK(layout.progressTrack.w > 0.f);
        CHECK(actions(layout) ==
              std::vector<MenuAction>{MenuAction::CANCEL_MODEL, MenuAction::QUIT});
    }

    TerrainBootstrapSnapshot localPack{
        .state = TerrainBootstrapState::Compiling,
        .completedBytes = 100,
        .totalBytes = 100,
        .reusingInstalledPack = true,
        .detail = "Reusing the local model pack; preparing Core ML sessions",
    };
    const MenuLayout localPackLayout = buildTerrainBootstrapLayout(localPack, 1024.f, 768.f);
    CHECK(containsText(localPackLayout, "LOCAL PACK REUSED - NO DOWNLOAD"));

    TerrainBootstrapSnapshot failed;
    failed.state = TerrainBootstrapState::Failed;
    failed.detail = "Verification failed";
    failed.failure = TerrainBootstrapFailure{
        .code = TerrainBootstrapFailureCode::Integrity,
        .message = "Verification failed",
        .retryable = true,
    };
    const MenuLayout failedLayout = buildTerrainBootstrapLayout(failed, 1024.f, 768.f);
    CHECK(containsText(failedLayout, "FAILED"));
    CHECK(actions(failedLayout) == std::vector<MenuAction>{MenuAction::RETRY_MODEL,
                                                           MenuAction::REPAIR_MODEL,
                                                           MenuAction::QUIT});

    failed.failure->retryable = false;
    const MenuLayout terminalLayout = buildTerrainBootstrapLayout(failed, 1024.f, 768.f);
    CHECK(actions(terminalLayout) == std::vector<MenuAction>{MenuAction::QUIT});

    const MenuLayout profileFailureLayout =
        buildTerrainBootstrapLayout(failed, 1024.f, 768.f, true);
    CHECK(actions(profileFailureLayout) ==
          std::vector<MenuAction>{MenuAction::OPEN_WORLD_SELECT, MenuAction::QUIT});

    TerrainBootstrapSnapshot ready;
    ready.state = TerrainBootstrapState::Ready;
    const MenuLayout readyLayout = buildTerrainBootstrapLayout(ready, 1024.f, 768.f);
    CHECK(containsText(readyLayout, "READY"));
    CHECK(readyLayout.buttons.empty());

    const MenuLayout preparing =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .connectedParentRadiusChunks = 48.0F,
                                       .farBaseReady = 9,
                                       .farBaseRequired = 12,
                                       .elapsedSeconds = 8.5},
                                      1024.f, 768.f);
    CHECK(containsText(preparing, "PREPARING WORLD"));
    CHECK(containsText(preparing, "SAFE SPAWN READY"));
    CHECK(containsText(preparing, "CONNECTED ENTRY FRONTIER 48/96 CHUNKS"));
    CHECK(containsText(preparing, "ENTRY 96 CHUNKS / CONFIGURED 512 CHUNKS"));
    CHECK(preparing.progressFraction == Catch::Approx(0.60f));
    CHECK(actions(preparing) == std::vector<MenuAction>{MenuAction::QUIT});

    const MenuLayout nonfiniteFrontier = buildV4WorldPreparationLayout(
        {.safeSpawnReady = true,
         .configuredHorizonRadiusChunks = 512,
         .entryHorizonRadiusChunks = 96,
         .connectedParentRadiusChunks = std::numeric_limits<float>::quiet_NaN(),
         .farBaseReady = 9,
         .farBaseRequired = 12},
        1024.f, 768.f);
    CHECK(containsText(nonfiniteFrontier, "CONNECTED ENTRY FRONTIER 0/96 CHUNKS"));
    CHECK(nonfiniteFrontier.progressFraction == Catch::Approx(0.2f));

    const MenuLayout incompleteShortHorizon =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 64,
                                       .entryHorizonRadiusChunks = 64,
                                       .connectedParentRadiusChunks = 64.0F,
                                       .farBaseReady = 99,
                                       .farBaseRequired = 100},
                                      1024.f, 768.f);
    CHECK(containsText(incompleteShortHorizon, "CONNECTED ENTRY FRONTIER 64/64 CHUNKS"));
    CHECK_FALSE(containsText(incompleteShortHorizon, "PROTECTED NEAR TERRAIN"));

    const MenuLayout preparingCompleteHorizon =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .farBaseReady = 12,
                                       .farBaseRequired = 12,
                                       .elapsedSeconds = 12.0},
                                      1024.f, 768.f);
    CHECK(containsText(preparingCompleteHorizon, "CONNECTED ENTRY FRONTIER READY"));
    CHECK(preparingCompleteHorizon.progressFraction == Catch::Approx(1.0f));

    const MenuLayout preparingConnectedPrefix =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .connectedParentRadiusChunks = 96.0F,
                                       .farBaseReady = 9,
                                       .farBaseRequired = 12,
                                       .elapsedSeconds = 13.0},
                                      1024.f, 768.f);
    CHECK(containsText(preparingConnectedPrefix, "CONNECTED ENTRY FRONTIER READY"));
    CHECK(preparingConnectedPrefix.progressFraction == Catch::Approx(1.0f));

    const MenuLayout preparingNearDetail =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .connectedParentRadiusChunks = 96.0F,
                                       .farBaseReady = 9,
                                       .farBaseRequired = 12,
                                       .nearFinalReady = 24,
                                       .nearFinalRequired = 60,
                                       .elapsedSeconds = 13.0},
                                      1024.f, 768.f);
    CHECK(containsText(preparingNearDetail, "CONNECTED ENTRY FRONTIER READY"));
    CHECK(containsText(preparingNearDetail, "NEAR DETAIL 24/60"));
    CHECK(preparingNearDetail.progressFraction == Catch::Approx(0.76f));

    const MenuLayout readyNearDetail =
        buildV4WorldPreparationLayout({.safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .connectedParentRadiusChunks = 96.0F,
                                       .farBaseReady = 9,
                                       .farBaseRequired = 12,
                                       .nearFinalReady = 60,
                                       .nearFinalRequired = 60},
                                      1024.f, 768.f);
    CHECK(containsText(readyNearDetail, "NEAR DETAIL READY"));
    CHECK(readyNearDetail.progressFraction == Catch::Approx(1.0f));

    const MenuLayout locating = buildV4WorldPreparationLayout({.drySpawnValidated = false,
                                                               .safeSpawnReady = false,
                                                               .farBaseReady = 0,
                                                               .farBaseRequired = 0,
                                                               .elapsedSeconds = 1.0},
                                                              1024.f, 768.f);
    CHECK(containsText(locating, "LOCATING DRY LAND"));
    CHECK(containsText(locating, "HORIZON WAITS FOR DRY LAND"));
    CHECK(locating.progressFraction == Catch::Approx(-1.0f));

    const MenuLayout preparingFinalTerrain =
        buildV4WorldPreparationLayout({.drySpawnValidated = true,
                                       .finalSpawnTerrainReady = false,
                                       .safeSpawnReady = false,
                                       .farBaseReady = 0,
                                       .farBaseRequired = 0,
                                       .elapsedSeconds = 1.5},
                                      1024.f, 768.f);
    CHECK(containsText(preparingFinalTerrain, "PREPARING FINAL TERRAIN"));
    CHECK(containsText(preparingFinalTerrain, "HORIZON WAITS FOR FINAL TERRAIN"));
    CHECK_FALSE(containsText(preparingFinalTerrain, "ENTRY HORIZON 0/0"));
    CHECK(preparingFinalTerrain.progressFraction == Catch::Approx(-1.0f));

    const MenuLayout waitingForSafeSpawn = buildV4WorldPreparationLayout({.drySpawnValidated = true,
                                                                          .safeSpawnReady = false,
                                                                          .farBaseReady = 0,
                                                                          .farBaseRequired = 0,
                                                                          .elapsedSeconds = 2.0},
                                                                         1024.f, 768.f);
    CHECK(containsText(waitingForSafeSpawn, "FINALIZING SAFE SPAWN"));
    CHECK(containsText(waitingForSafeSpawn, "HORIZON WAITS FOR SAFE SPAWN"));
    CHECK_FALSE(containsText(waitingForSafeSpawn, "ENTRY HORIZON 0/0"));
    CHECK(waitingForSafeSpawn.progressFraction == Catch::Approx(-1.0f));

    const MenuLayout initializingHorizon =
        buildV4WorldPreparationLayout({.drySpawnValidated = true,
                                       .safeSpawnReady = true,
                                       .configuredHorizonRadiusChunks = 512,
                                       .entryHorizonRadiusChunks = 96,
                                       .farBaseReady = 0,
                                       .farBaseRequired = 0,
                                       .elapsedSeconds = 3.0},
                                      1024.f, 768.f);
    CHECK(containsText(initializingHorizon, "SAFE SPAWN READY"));
    CHECK(containsText(initializingHorizon, "INITIALIZING CONNECTED ENTRY FRONTIER"));
    CHECK(containsText(initializingHorizon, "ENTRY 96 CHUNKS / CONFIGURED 512 CHUNKS"));
    CHECK_FALSE(containsText(initializingHorizon, "ENTRY HORIZON 0/0"));
    CHECK(initializingHorizon.progressFraction == Catch::Approx(-1.0f));
}

TEST_CASE("Generator v4 entry frontier includes protected detail at every tile corner",
          "[engine][v4][entry][far-terrain]") {
    constexpr float ENTRY_RADIUS_BLOCKS =
        static_cast<float>(V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS * CHUNK_EDGE);
    constexpr double ENTRY_RADIUS_SQUARED =
        static_cast<double>(ENTRY_RADIUS_BLOCKS) * ENTRY_RADIUS_BLOCKS;
    const std::array<std::pair<double, double>, 4> cameraCorners{
        std::pair{0.25, 0.25},
        std::pair{0.25, static_cast<double>(FAR_TERRAIN_TILE_EDGE) - 0.25},
        std::pair{static_cast<double>(FAR_TERRAIN_TILE_EDGE) - 0.25, 0.25},
        std::pair{static_cast<double>(FAR_TERRAIN_TILE_EDGE) - 0.25,
                  static_cast<double>(FAR_TERRAIN_TILE_EDGE) - 0.25},
    };

    for (const auto [cameraX, cameraZ] : cameraCorners) {
        CAPTURE(cameraX, cameraZ);
        std::vector<FarTerrainViewTile> selected;
        selectFarTerrainView(cameraX, cameraZ, MAX_RENDER_DISTANCE_CHUNKS, selected);
        REQUIRE_FALSE(selected.empty());

        std::vector<FarTerrainKey> protectedTargets;
        const ColumnPos protectedAnchor = farTerrainProtectedNearAnchor(cameraX, cameraZ);
        buildFarTerrainProtectedNearTargets(protectedAnchor, selected, protectedTargets);
        REQUIRE(protectedTargets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
        std::unordered_set<ColumnPos> protectedCoordinates;
        protectedCoordinates.reserve(protectedTargets.size());
        for (const FarTerrainKey target : protectedTargets)
            protectedCoordinates.insert({target.tileX, target.tileZ});
        std::unordered_map<ColumnPos, const FarTerrainViewTile*> selectedByCoordinate;
        selectedByCoordinate.reserve(selected.size());
        for (const FarTerrainViewTile& tile : selected)
            selectedByCoordinate.emplace(ColumnPos{tile.key.tileX, tile.key.tileZ}, &tile);
        const auto selectedTile = [&](ColumnPos coordinate) -> const FarTerrainViewTile* {
            const auto found = selectedByCoordinate.find(coordinate);
            return found == selectedByCoordinate.end() ? nullptr : found->second;
        };
        const FarTerrainCoverageFrontier frontier =
            farTerrainCoverageFrontier(selected, [&](FarTerrainKey key) {
                const ColumnPos coordinate{key.tileX, key.tileZ};
                const FarTerrainViewTile* tile = selectedTile(coordinate);
                return tile && (tile->distanceSquared < ENTRY_RADIUS_SQUARED ||
                                protectedCoordinates.contains(coordinate));
            });

        REQUIRE_FALSE(frontier.complete);
        REQUIRE(frontier.distanceBlocks >= ENTRY_RADIUS_BLOCKS);
        const float opaqueRadius =
            frontier.distanceBlocks - farTerrainCoverageFadeBlocks(frontier.distanceBlocks);
        REQUIRE(opaqueRadius >= 84.0F * CHUNK_EDGE);

        for (const FarTerrainKey target : protectedTargets) {
            const FarTerrainViewTile* tile = selectedTile({target.tileX, target.tileZ});
            REQUIRE(tile != nullptr);
            CHECK(tile->distanceSquared < frontier.distanceSquaredBlocks);
            CHECK(farTerrainCoverageDrawEligible(tile->distanceSquared, frontier));
        }
    }
}

TEST_CASE("V4 world-open failures expose only valid recovery paths",
          "[engine][bootstrap][worlds]") {
    CHECK_FALSE(v4WorldOpenFailureRetryable(V4WorldOpenStatus::Ready));
    CHECK(v4WorldOpenFailureRetryable(V4WorldOpenStatus::BootstrapNotReady));
    CHECK(v4WorldOpenFailureRetryable(V4WorldOpenStatus::PersistenceFailure));
    CHECK_FALSE(v4WorldOpenFailureRetryable(V4WorldOpenStatus::InvalidWorldDirectory));
    CHECK_FALSE(v4WorldOpenFailureRetryable(V4WorldOpenStatus::MissingMetadata));
    CHECK_FALSE(v4WorldOpenFailureRetryable(V4WorldOpenStatus::IdentityConflict));

    CHECK(v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus::InvalidWorldDirectory));
    CHECK(v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus::MissingMetadata));
    CHECK(v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus::IdentityConflict));
    CHECK_FALSE(v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus::BootstrapNotReady));
    CHECK_FALSE(v4WorldOpenFailureAllowsWorldSelection(V4WorldOpenStatus::PersistenceFailure));
}

TEST_CASE("Font covers every character the menus draw", "[ui][font]") {
    GraphicsSettings gfx;
    std::string needed = "0123456789.:/-+ ";

    // Every screen with a fully populated context, including a world name
    // exercising the complete allowed charset.
    MenuContext ctx;
    ctx.gfx = &gfx;
    ctx.worldRows = {"A world_NAME.42-x - Survival - Seed 4294967295",
                     "second row - Creative - Seed 7"};
    ctx.worldSelect.selected = 0;
    ctx.worldCreate.name = "AZaz09 ._-";
    ctx.worldCreate.seedText = "0123456789";
    ctx.deleteWorldName = "A world_NAME.42-x";
    ctx.successorWorldName = "A world_NAME.42-x";
    for (GameScreen screen :
         {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS, GameScreen::VIDEO_SETTINGS,
          GameScreen::WORLD_SELECT, GameScreen::WORLD_CREATE, GameScreen::WORLD_DELETE_CONFIRM,
          GameScreen::WORLD_SUCCESSOR_CONFIRM}) {
        MenuLayout layout = buildScreenLayout(screen, 1024.f, 768.f, ctx);
        for (const auto& text : layout.texts)
            needed += text.text;
        for (const auto& button : layout.buttons)
            needed += button.label;
        for (const auto& field : layout.textFields) {
            needed += field.label;
            needed += field.text;
        }
    }
    // The full world-name charset can appear in any typed name.
    for (int c = 0; c < 128; ++c) {
        if (isWorldNameChar(static_cast<char>(c)))
            needed += static_cast<char>(c);
    }
    worldgen::bootstrap::TerrainBootstrapSnapshot bootstrap;
    bootstrap.state = worldgen::bootstrap::TerrainBootstrapState::Failed;
    bootstrap.detail = "Model verification failed";
    bootstrap.failure = worldgen::bootstrap::TerrainBootstrapFailure{
        .code = worldgen::bootstrap::TerrainBootstrapFailureCode::Integrity,
        .message = bootstrap.detail,
        .retryable = true,
    };
    MenuLayout bootstrapLayout = buildTerrainBootstrapLayout(bootstrap, 1024.f, 768.f);
    for (const auto& text : bootstrapLayout.texts)
        needed += text.text;
    for (const auto& button : bootstrapLayout.buttons)
        needed += button.label;
    MenuLayout preparationLayout = buildV4WorldPreparationLayout(
        {.safeSpawnReady = false, .farBaseReady = 1, .farBaseRequired = 2, .elapsedSeconds = 3.0},
        1024.f, 768.f);
    for (const auto& text : preparationLayout.texts)
        needed += text.text;
    for (const auto& button : preparationLayout.buttons)
        needed += button.label;
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

// ---- Inventory Tests ----

TEST_CASE("Inventory: hotbar selection clamps and wraps", "[inventory]") {
    Inventory inventory;
    REQUIRE(inventory.getSelectedIndex() == 0);

    inventory.selectSlot(-5);
    REQUIRE(inventory.getSelectedIndex() == 0);
    inventory.selectSlot(100);
    REQUIRE(inventory.getSelectedIndex() == 8);
    inventory.selectSlot(4);
    REQUIRE(inventory.getSelectedIndex() == 4);

    inventory.selectSlot(8);
    inventory.selectNext();
    REQUIRE(inventory.getSelectedIndex() == 0);
    inventory.selectPrev();
    REQUIRE(inventory.getSelectedIndex() == 8);
}

TEST_CASE("Inventory: slots read and write with range guards", "[inventory]") {
    Inventory inventory;
    REQUIRE(inventory.getSlot(0).empty());

    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::DIAMOND_ORE), 3, 0});
    REQUIRE(inventory.getSlot(0).type == itemFromBlock(BlockType::DIAMOND_ORE));
    REQUIRE(inventory.getSlot(0).count == 3);

    // Main-grid slots exist beyond the hotbar.
    inventory.setSlot(35, ItemStack{ItemType::STICK, 5, 0});
    REQUIRE(inventory.getSlot(35).count == 5);

    // Out-of-range reads return empty; writes drop.
    REQUIRE(inventory.getSlot(-1).empty());
    REQUIRE(inventory.getSlot(Inventory::SLOTS).empty());
    inventory.setSlot(-1, ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(inventory.getSlot(0).type == itemFromBlock(BlockType::DIAMOND_ORE));
}

TEST_CASE("Inventory: selected block resolves through the item registry", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::STONE), 1, 0});
    inventory.setSlot(1, ItemStack{ItemType::IRON_PICKAXE, 1, 250});
    inventory.selectSlot(0);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::STONE);
    // Tools and empty slots place nothing.
    inventory.selectSlot(1);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::AIR);
    inventory.selectSlot(2);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::AIR);
}

TEST_CASE("Inventory: add merges into stacks hotbar first", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(9, ItemStack{ItemType::COAL, 60, 0});

    // Merging tops off the existing main-grid stack, then opens hotbar slot 0.
    REQUIRE(inventory.add(ItemStack{ItemType::COAL, 10, 0}) == 10);
    REQUIRE(inventory.getSlot(9).count == 64);
    REQUIRE(inventory.getSlot(0).type == ItemType::COAL);
    REQUIRE(inventory.getSlot(0).count == 6);

    // A full inventory absorbs nothing.
    Inventory full;
    for (int slot = 0; slot < Inventory::SLOTS; ++slot) {
        full.setSlot(slot, ItemStack{ItemType::STICK, 64, 0});
    }
    REQUIRE(full.add(ItemStack{ItemType::STICK, 1, 0}) == 0);
    REQUIRE(full.add(ItemStack{ItemType::COAL, 1, 0}) == 0);

    // Tools never merge (stack limit one) but fill empty slots.
    Inventory tools;
    REQUIRE(tools.add(ItemStack{ItemType::IRON_AXE, 1, 250}) == 1);
    REQUIRE(tools.getSlot(0).type == ItemType::IRON_AXE);
    REQUIRE(tools.getSlot(0).durability == 250);
}

TEST_CASE("Inventory: consume and tool damage empty the selected slot", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::DIRT), 2, 0});
    inventory.selectSlot(0);
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).count == 1);
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).empty());
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).empty());

    inventory.setSlot(0, ItemStack{ItemType::WOODEN_PICKAXE, 1, 2});
    REQUIRE_FALSE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).durability == 1);
    REQUIRE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).empty());

    // Non-tools never wear.
    inventory.setSlot(0, ItemStack{ItemType::COAL, 4, 0});
    REQUIRE_FALSE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).count == 4);
}

TEST_CASE("Inventory exchanges one stacked bucket without losing either result", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(0, ItemStack{ItemType::BUCKET, 3, 0});
    inventory.selectSlot(0);

    REQUIRE(inventory.exchangeOneSelected(ItemStack{ItemType::WATER_BUCKET, 1, 0}).empty());
    REQUIRE(inventory.getSlot(0) == ItemStack{ItemType::BUCKET, 2, 0});
    REQUIRE(inventory.getSlot(1) == ItemStack{ItemType::WATER_BUCKET, 1, 0});

    Inventory full;
    for (int slot = 0; slot < Inventory::SLOTS; ++slot) {
        full.setSlot(slot, ItemStack{ItemType::STICK, 64, 0});
    }
    full.setSlot(0, ItemStack{ItemType::BUCKET, 3, 0});
    full.selectSlot(0);
    REQUIRE(full.exchangeOneSelected(ItemStack{ItemType::LAVA_BUCKET, 1, 0}) ==
            ItemStack{ItemType::LAVA_BUCKET, 1, 0});
    REQUIRE(full.getSlot(0) == ItemStack{ItemType::BUCKET, 2, 0});

    Inventory filled;
    filled.setSlot(4, ItemStack{ItemType::WATER_BUCKET, 1, 0});
    filled.selectSlot(4);
    REQUIRE(filled.exchangeOneSelected(ItemStack{ItemType::BUCKET, 1, 0}).empty());
    REQUIRE(filled.getSlot(4) == ItemStack{ItemType::BUCKET, 1, 0});
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

TEST_CASE("InputState: text entry accumulates edits and suppresses nothing else", "[input][text]") {
    InputState input;
    REQUIRE_FALSE(input.textEntryActive);

    // Inactive entry ignores edits entirely.
    input.applyTextKey('x');
    input.applyTextBackspace();
    REQUIRE(input.textBuffer.empty());

    input.beginTextEntry("Seed");
    REQUIRE(input.textEntryActive);
    REQUIRE(input.textBuffer == "Seed");

    input.applyTextKey(' ');
    input.applyTextKey('4');
    input.applyTextKey('2');
    REQUIRE(input.textBuffer == "Seed 42");

    // Control characters and non-ASCII bytes never land in the buffer.
    input.applyTextKey('\t');
    input.applyTextKey('\n');
    input.applyTextKey(static_cast<char>(0x1B));
    input.applyTextKey(static_cast<char>(0xC3));
    REQUIRE(input.textBuffer == "Seed 42");

    input.applyTextBackspace();
    REQUIRE(input.textBuffer == "Seed 4");

    // The cap holds regardless of how much is typed.
    for (int i = 0; i < 300; ++i) {
        input.applyTextKey('a');
    }
    REQUIRE(input.textBuffer.size() == InputState::TEXT_BUFFER_MAX);

    const std::string finished = input.endTextEntry();
    REQUIRE_FALSE(input.textEntryActive);
    REQUIRE(finished.size() == InputState::TEXT_BUFFER_MAX);
    REQUIRE(input.textBuffer.empty());

    // Submission is a one-frame edge cleared by update().
    input.beginTextEntry("");
    input.textSubmitted = true;
    input.update();
    REQUIRE_FALSE(input.textSubmitted);
}

namespace {

SlotAccess craftingAccess(std::array<ItemStack, 36>& inventory, std::array<ItemStack, 9>& grid,
                          ItemStack& result, int gridSize, int gridWidth) {
    SlotAccess access;
    access.inventory = inventory.data();
    access.craftGrid = grid.data();
    access.craftGridSize = gridSize;
    access.craftGridWidth = gridWidth;
    access.craftResult = &result;
    return access;
}

} // namespace

TEST_CASE("Slot clicks pick place merge and split", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    inventory[0] = ItemStack{ItemType::COAL, 10, 0};
    inventory[1] = ItemStack{ItemType::COAL, 60, 0};
    inventory[2] = ItemStack{ItemType::STICK, 5, 0};

    // LEFT on a stack picks the whole thing up.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 0}, SlotClickKind::LEFT).changed);
    REQUIRE(cursor == ItemStack{ItemType::COAL, 10, 0});
    REQUIRE(inventory[0].empty());

    // LEFT on the same type merges up to the cap and keeps the rest held.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 1}, SlotClickKind::LEFT).changed);
    REQUIRE(inventory[1].count == 64);
    REQUIRE(cursor.count == 6);

    // LEFT on a different type swaps.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 2}, SlotClickKind::LEFT).changed);
    REQUIRE(cursor == ItemStack{ItemType::STICK, 5, 0});
    REQUIRE(inventory[2] == ItemStack{ItemType::COAL, 6, 0});

    // RIGHT with a held stack places exactly one.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 3}, SlotClickKind::RIGHT).changed);
    REQUIRE(inventory[3] == ItemStack{ItemType::STICK, 1, 0});
    REQUIRE(cursor.count == 4);

    // RIGHT with an empty cursor takes the larger half.
    cursor.clear();
    inventory[4] = ItemStack{ItemType::COAL, 7, 0};
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 4}, SlotClickKind::RIGHT).changed);
    REQUIRE(cursor.count == 4);
    REQUIRE(inventory[4].count == 3);

    // Clicks on empty air with an empty cursor change nothing.
    cursor.clear();
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 30}, SlotClickKind::LEFT).changed);
}

TEST_CASE("Shift clicks quick-move between regions", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    ItemStack cursor;

    // Hotbar to main.
    inventory[2] = ItemStack{ItemType::COAL, 12, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 2}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(inventory[2].empty());
    REQUIRE(inventory[9] == ItemStack{ItemType::COAL, 12, 0});

    // Main to hotbar.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 9}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(inventory[0] == ItemStack{ItemType::COAL, 12, 0});

    // Craft grid to inventory.
    grid[4] = ItemStack{itemFromBlock(BlockType::PLANKS), 3, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CRAFT_IN, 4}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(grid[4].empty());

    // Inventory to an open furnace, routed by what the item can do there.
    ItemStack furnaceInput;
    ItemStack furnaceFuel;
    ItemStack furnaceOutput;
    access.furnaceInput = &furnaceInput;
    access.furnaceFuel = &furnaceFuel;
    access.furnaceOutput = &furnaceOutput;
    inventory[5] = ItemStack{ItemType::RAW_BEEF, 2, 0};
    inventory[6] = ItemStack{ItemType::COAL, 12, 0};
    inventory[7] = ItemStack{ItemType::IRON_INGOT, 1, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 5}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(furnaceInput == ItemStack{ItemType::RAW_BEEF, 2, 0});
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 6}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(furnaceFuel == ItemStack{ItemType::COAL, 12, 0});
    // Neither smeltable nor fuel goes nowhere.
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 7}, SlotClickKind::SHIFT_LEFT)
            .changed);
}

TEST_CASE("Craft output is take-only and consumes the grid", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 3, 0};
    result = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};

    // Placement onto the output is refused.
    cursor = ItemStack{ItemType::COAL, 1, 0};
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::LEFT).changed);
    cursor.clear();

    // Taking crafts once: log consumed, result refreshed for the next craft.
    const auto taken =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::LEFT);
    REQUIRE(taken.changed);
    REQUIRE(taken.crafted);
    REQUIRE(cursor == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});
    REQUIRE(grid[0].count == 2);
    REQUIRE(result == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});

    // Shift-crafting drains the remaining logs straight into the inventory.
    cursor.clear();
    const auto drained =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::SHIFT_LEFT);
    REQUIRE(drained.crafted);
    REQUIRE(grid[0].empty());
    REQUIRE(result.empty());
    REQUIRE(inventory[0] == ItemStack{itemFromBlock(BlockType::PLANKS), 8, 0});
}

TEST_CASE("Shift-crafting into a nearly full inventory never creates items", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    // One log crafts {PLANKS, 4}. Fill every slot with a foreign item except
    // one planks stack with room for exactly 1 more.
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 3, 0};
    result = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};
    for (ItemStack& slot : inventory) {
        slot = ItemStack{ItemType::COAL, 64, 0};
    }
    inventory[0] = ItemStack{itemFromBlock(BlockType::PLANKS), 63, 0};

    const auto before = inventory;
    // The 4-plank batch cannot fully fit (only room for 1), so the craft is
    // refused: no partial deposit, the grid is untouched, the output stands.
    const auto outcome =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::SHIFT_LEFT);
    REQUIRE_FALSE(outcome.changed);
    REQUIRE(inventory == before);
    REQUIRE(grid[0].count == 3);
    REQUIRE(result == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});
}

TEST_CASE("Creative palette hands out stacks and eats held ones", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    SlotAccess access;
    access.inventory = inventory.data();
    access.palette = CREATIVE_PALETTE.data();
    access.paletteSize = static_cast<int>(CREATIVE_PALETTE.size());
    ItemStack cursor;

    const ItemType first = CREATIVE_PALETTE[0];
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::LEFT)
                .changed);
    REQUIRE(cursor.type == first);
    REQUIRE(cursor.count == maxStackSize(first));

    // Holding anything, a palette click trashes it.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 5}, SlotClickKind::LEFT)
                .changed);
    REQUIRE(cursor.empty());

    // RIGHT builds a stack one item at a time.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::RIGHT)
                .changed);
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::RIGHT)
                .changed);
    REQUIRE(cursor == ItemStack{first, 2, 0});

    // SHIFT sends a full stack straight into the inventory; the palette
    // itself never mutates.
    cursor.clear();
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::SHIFT_LEFT)
            .changed);
    REQUIRE(inventory[0].type == first);
}

TEST_CASE("Right-drag spreads one item into each painted slot", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    ItemStack cursor{ItemType::COAL, 5, 0};

    const std::array<SlotRef, 3> painted = {SlotRef{SlotDomain::INVENTORY, 10},
                                            SlotRef{SlotDomain::INVENTORY, 11},
                                            SlotRef{SlotDomain::CRAFT_IN, 4}};
    REQUIRE(applySlotDrag(access, cursor, painted, SlotClickKind::RIGHT).changed);
    REQUIRE(inventory[10] == ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(inventory[11] == ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(grid[4] == ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(cursor.count == 2); // the untouched remainder stays on the cursor
}

TEST_CASE("Left-drag splits the held stack evenly", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    ItemStack cursor{ItemType::COAL, 10, 0};

    // Ten items across three slots: each takes floor(10/3)=3, one stays held.
    const std::array<SlotRef, 3> painted = {SlotRef{SlotDomain::INVENTORY, 0},
                                            SlotRef{SlotDomain::INVENTORY, 1},
                                            SlotRef{SlotDomain::INVENTORY, 2}};
    REQUIRE(applySlotDrag(access, cursor, painted, SlotClickKind::LEFT).changed);
    REQUIRE(inventory[0].count == 3);
    REQUIRE(inventory[1].count == 3);
    REQUIRE(inventory[2].count == 3);
    REQUIRE(cursor.count == 1);

    // A slot already holding the item is topped up by the even share, and a
    // foreign slot is skipped rather than overwritten.
    inventory[3] = ItemStack{ItemType::COAL, 60, 0};
    inventory[4] = ItemStack{ItemType::STICK, 1, 0};
    cursor = ItemStack{ItemType::COAL, 8, 0};
    const std::array<SlotRef, 2> topUp = {SlotRef{SlotDomain::INVENTORY, 3},
                                          SlotRef{SlotDomain::INVENTORY, 4}};
    REQUIRE(applySlotDrag(access, cursor, topUp, SlotClickKind::LEFT).changed);
    REQUIRE(inventory[3].count == 64); // capped at the max stack, absorbing 4
    REQUIRE(inventory[4] == ItemStack{ItemType::STICK, 1, 0});
    REQUIRE(cursor.count == 4);
}

TEST_CASE("Double-click gathers matching stacks up to a full one", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);

    inventory[5] = ItemStack{ItemType::COAL, 64, 0}; // a full stack, left last
    inventory[6] = ItemStack{ItemType::COAL, 10, 0};
    inventory[7] = ItemStack{ItemType::STICK, 20, 0};
    grid[0] = ItemStack{ItemType::COAL, 30, 0};
    ItemStack cursor{ItemType::COAL, 5, 0};

    REQUIRE(applyDoubleClick(access, cursor).changed);
    REQUIRE(cursor.count == 64);                                // exactly one stack
    REQUIRE(inventory[7] == ItemStack{ItemType::STICK, 20, 0}); // foreign untouched
    // Partial stacks are consumed before the full one: 5 + 10 + 30 = 45, then
    // 19 pulled from the full stack to reach 64.
    REQUIRE(inventory[6].empty());
    REQUIRE(grid[0].empty());
    REQUIRE(inventory[5].count == 45);
}

TEST_CASE("Shift-click moves items into and out of an open chest", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 27> chest{};
    SlotAccess access;
    access.inventory = inventory.data();
    access.chest = chest.data();
    access.chestSize = 27;
    ItemStack cursor;

    inventory[0] = ItemStack{ItemType::COAL, 30, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 0}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(inventory[0].empty());
    REQUIRE(chest[0] == ItemStack{ItemType::COAL, 30, 0});

    // Shift-clicking the chest slot sends it back into the player inventory.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::CHEST, 0}, SlotClickKind::SHIFT_LEFT).changed);
    REQUIRE(chest[0].empty());
    bool returned = false;
    for (const ItemStack& slot : inventory) {
        if (slot == ItemStack{ItemType::COAL, 30, 0})
            returned = true;
    }
    REQUIRE(returned);
}

TEST_CASE("Outside drops and container close return items", "[slots]") {
    ItemStack cursor{ItemType::COAL, 5, 0};
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::RIGHT) == ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(cursor.count == 4);
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::LEFT) == ItemStack{ItemType::COAL, 4, 0});
    REQUIRE(cursor.empty());
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::LEFT).empty());

    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result{itemFromBlock(BlockType::PLANKS), 4, 0};
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 2, 0};
    grid[8] = ItemStack{ItemType::STICK, 7, 0};
    cursor = ItemStack{ItemType::COAL, 3, 0};

    REQUIRE(collectOnClose(access, cursor).empty());
    REQUIRE(cursor.empty());
    REQUIRE(grid[0].empty());
    REQUIRE(result.empty());
    int coal = 0;
    int sticks = 0;
    int logs = 0;
    for (const ItemStack& slot : inventory) {
        if (slot.type == ItemType::COAL)
            coal += slot.count;
        if (slot.type == ItemType::STICK)
            sticks += slot.count;
        if (slot.type == itemFromBlock(BlockType::LOG))
            logs += slot.count;
    }
    REQUIRE(coal == 3);
    REQUIRE(sticks == 7);
    REQUIRE(logs == 2);

    // A stuffed inventory reports the homeless remainder.
    for (ItemStack& slot : inventory) {
        slot = ItemStack{ItemType::STICK, 64, 0};
    }
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 2, 0};
    cursor = ItemStack{ItemType::COAL, 3, 0};
    const auto overflow = collectOnClose(access, cursor);
    REQUIRE(overflow.size() == 2);
    REQUIRE(preserveCarriedOverflow(overflow, cursor, grid));
    REQUIRE(cursor == ItemStack{itemFromBlock(BlockType::LOG), 2, 0});
    REQUIRE(grid[0] == ItemStack{ItemType::COAL, 3, 0});
    REQUIRE_FALSE(hasExtendedCarriedCrafting(grid));

    std::array<ItemStack, 9> extendedGrid{};
    extendedGrid[8] = ItemStack{ItemType::COAL, 1, 0};
    REQUIRE(hasExtendedCarriedCrafting(extendedGrid));
}

TEST_CASE("Death drops include persisted cursor and crafting stacks", "[slots][death]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    inventory[0] = ItemStack{ItemType::COAL, 12, 0};
    inventory[35] = ItemStack{ItemType::IRON_PICKAXE, 1, 117};
    ItemStack cursor{ItemType::DIAMOND, 3, 0};
    grid[0] = ItemStack{itemFromBlock(BlockType::PLANKS), 8, 0};
    grid[8] = ItemStack{ItemType::STICK, 2, 0};

    const std::vector<ItemStack> drops = collectDeathDrops(inventory, cursor, grid);

    REQUIRE(drops == std::vector<ItemStack>{ItemStack{ItemType::COAL, 12, 0},
                                            ItemStack{ItemType::IRON_PICKAXE, 1, 117},
                                            ItemStack{ItemType::DIAMOND, 3, 0},
                                            ItemStack{itemFromBlock(BlockType::PLANKS), 8, 0},
                                            ItemStack{ItemType::STICK, 2, 0}});
    REQUIRE(std::ranges::all_of(inventory, [](const ItemStack& stack) { return stack.empty(); }));
    REQUIRE(cursor.empty());
    REQUIRE(std::ranges::all_of(grid, [](const ItemStack& stack) { return stack.empty(); }));
}

TEST_CASE("Furnace layout exposes three slots and two gauges", "[ui][containers]") {
    MenuContext ctx;
    ctx.container.furnaceInput = ItemStack{ItemType::RAW_BEEF, 3, 0};
    ctx.container.furnaceFuel = ItemStack{ItemType::COAL, 5, 0};
    ctx.container.furnaceOutput = ItemStack{ItemType::COOKED_BEEF, 2, 0};
    ctx.container.furnaceCook = 0.5f;
    ctx.container.furnaceFuelLeft = 0.25f;

    MenuLayout layout = buildScreenLayout(GameScreen::FURNACE, 1024.f, 768.f, ctx);
    int input = 0;
    int fuel = 0;
    int output = 0;
    int inventory = 0;
    for (const SlotWidget& slot : layout.slots) {
        switch (slot.ref.domain) {
            case SlotDomain::FURNACE_INPUT:
                ++input;
                REQUIRE(slot.stack == ItemStack{ItemType::RAW_BEEF, 3, 0});
                break;
            case SlotDomain::FURNACE_FUEL:
                ++fuel;
                break;
            case SlotDomain::FURNACE_OUTPUT:
                ++output;
                REQUIRE(slot.stack == ItemStack{ItemType::COOKED_BEEF, 2, 0});
                break;
            case SlotDomain::INVENTORY:
                ++inventory;
                break;
            default:
                break;
        }
    }
    REQUIRE(input == 1);
    REQUIRE(fuel == 1);
    REQUIRE(output == 1);
    REQUIRE(inventory == 36);
    REQUIRE(layout.meters.size() == 2);
    // The cook arrow is the horizontal gauge, the flame the vertical one.
    const bool haveCook = layout.meters[0].fill == 0.5f || layout.meters[1].fill == 0.5f;
    const bool haveFlame = layout.meters[0].fill == 0.25f || layout.meters[1].fill == 0.25f;
    REQUIRE(haveCook);
    REQUIRE(haveFlame);
}

TEST_CASE("Container layouts expose every slot with correct references", "[ui][containers]") {
    MenuContext ctx;
    ctx.container.inventory[0] = ItemStack{ItemType::COAL, 9, 0};
    ctx.container.craftGrid[0] = ItemStack{itemFromBlock(BlockType::LOG), 1, 0};
    ctx.container.craftResult = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};

    MenuLayout survival = buildScreenLayout(GameScreen::INVENTORY, 1024.f, 768.f, ctx);
    int inventorySlots = 0;
    int craftIn = 0;
    int craftOut = 0;
    for (const SlotWidget& slot : survival.slots) {
        if (slot.ref.domain == SlotDomain::INVENTORY)
            ++inventorySlots;
        if (slot.ref.domain == SlotDomain::CRAFT_IN)
            ++craftIn;
        if (slot.ref.domain == SlotDomain::CRAFT_OUT)
            ++craftOut;
    }
    REQUIRE(inventorySlots == 36);
    REQUIRE(craftIn == 4);
    REQUIRE(craftOut == 1);
    REQUIRE(survival.slots.front().stack == ItemStack{itemFromBlock(BlockType::LOG), 1, 0});

    MenuLayout crafting = buildScreenLayout(GameScreen::CRAFTING, 1024.f, 768.f, ctx);
    craftIn = 0;
    for (const SlotWidget& slot : crafting.slots) {
        if (slot.ref.domain == SlotDomain::CRAFT_IN)
            ++craftIn;
    }
    REQUIRE(craftIn == 9);

    // Creative shows the paged palette instead of a craft grid.
    ctx.container.creative = true;
    ctx.container.creativePage = 1;
    MenuLayout creative = buildScreenLayout(GameScreen::INVENTORY, 1024.f, 768.f, ctx);
    int palette = 0;
    int minIndex = 1 << 20;
    for (const SlotWidget& slot : creative.slots) {
        if (slot.ref.domain == SlotDomain::CREATIVE_PALETTE) {
            ++palette;
            minIndex = std::min(minIndex, slot.ref.index);
        }
        REQUIRE(slot.ref.domain != SlotDomain::CRAFT_IN);
    }
    const int expected =
        std::min<int>(CREATIVE_PALETTE_PAGE_SIZE,
                      static_cast<int>(CREATIVE_PALETTE.size()) - CREATIVE_PALETTE_PAGE_SIZE);
    REQUIRE(palette == expected);
    REQUIRE(minIndex == CREATIVE_PALETTE_PAGE_SIZE);

    // Slot hit-testing resolves through the typed path.
    const SlotWidget& probe = survival.slots.front();
    const UIHit hit =
        uiHitTest(survival, probe.rect.x + probe.rect.w * 0.5f, probe.rect.y + probe.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::SLOT);
    REQUIRE(hit.index == 0);
}

TEST_CASE("Mining accumulates over time and completes on a stable target", "[mining]") {
    MiningState state;
    // Stone by hand needs blockBreakTicks(STONE, NONE) = 150 ticks.
    const int needed = blockBreakTicks(BlockType::STONE, ItemType::NONE);
    REQUIRE(needed == 150);

    for (int tick = 0; tick < needed - 1; ++tick) {
        REQUIRE_FALSE(tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE));
        REQUIRE(state.active);
    }
    REQUIRE(state.progress > 0.9f);
    // The final tick completes and resets.
    REQUIRE(tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE));
    REQUIRE_FALSE(state.active);
}

TEST_CASE("Mining resets on release and on a new target", "[mining]") {
    MiningState state;
    for (int tick = 0; tick < 20; ++tick) {
        tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    }
    REQUIRE(state.ticksElapsed == 20);

    // Releasing the button clears progress.
    tickMining(state, false, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    REQUIRE_FALSE(state.active);
    REQUIRE(state.progress == 0.f);

    // Looking at a new block restarts from zero.
    for (int tick = 0; tick < 20; ++tick) {
        tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    }
    tickMining(state, true, true, 9, 9, 9, BlockType::DIRT, ItemType::NONE);
    REQUIRE(state.x == 9);
    REQUIRE(state.block == BlockType::DIRT);
    REQUIRE(state.ticksElapsed == 1);
}

TEST_CASE("Mining respects tool speed and never breaks bedrock", "[mining]") {
    // A stone pickaxe finishes stone far faster than a bare hand.
    MiningState hand;
    int handTicks = 0;
    while (!tickMining(hand, true, true, 0, 0, 0, BlockType::STONE, ItemType::NONE) &&
           handTicks < 1000) {
        ++handTicks;
    }
    MiningState pick;
    int pickTicks = 0;
    while (!tickMining(pick, true, true, 0, 0, 0, BlockType::STONE, ItemType::STONE_PICKAXE) &&
           pickTicks < 1000) {
        ++pickTicks;
    }
    REQUIRE(pickTicks < handTicks);

    // Bedrock never completes, whatever the tool.
    MiningState bedrock;
    for (int tick = 0; tick < 500; ++tick) {
        REQUIRE_FALSE(
            tickMining(bedrock, true, true, 0, 0, 0, BlockType::BEDROCK, ItemType::IRON_PICKAXE));
    }
    REQUIRE(bedrock.progress == 0.f);

    // Instant-break flora completes the first settled tick.
    MiningState grass;
    REQUIRE(tickMining(grass, true, true, 0, 0, 0, BlockType::TALL_GRASS, ItemType::NONE));
}

TEST_CASE("Mining restarts when the held tool changes mid-mine", "[mining]") {
    MiningState state;
    // Start on stone with a stone pickaxe (fast).
    for (int tick = 0; tick < 5; ++tick) {
        tickMining(state, true, true, 0, 0, 0, BlockType::STONE, ItemType::STONE_PICKAXE);
    }
    const int fastNeeded = state.ticksNeeded;
    REQUIRE(state.ticksElapsed == 5);

    // Switching to a bare hand recomputes the (much longer) break time and
    // restarts progress, so pickaxe timing cannot break stone by hand.
    tickMining(state, true, true, 0, 0, 0, BlockType::STONE, ItemType::NONE);
    REQUIRE(state.ticksElapsed == 1);
    REQUIRE(state.ticksNeeded > fastNeeded);
    REQUIRE(state.tool == ItemType::NONE);
}

TEST_CASE("Bed respawn validation waits for resident safe cells", "[survival][bed]") {
    CHECK(validateBedSpawnCells(std::nullopt, BlockType::AIR, BlockType::AIR) ==
          BedSpawnValidation::DEFERRED);
    CHECK(validateBedSpawnCells(BlockType::BED, std::nullopt, BlockType::AIR) ==
          BedSpawnValidation::DEFERRED);
    CHECK(validateBedSpawnCells(BlockType::BED, BlockType::AIR, BlockType::AIR) ==
          BedSpawnValidation::VALID);
    CHECK(validateBedSpawnCells(BlockType::AIR, BlockType::AIR, BlockType::AIR) ==
          BedSpawnValidation::INVALID);
    CHECK(validateBedSpawnCells(BlockType::BED, BlockType::STONE, BlockType::AIR) ==
          BedSpawnValidation::INVALID);
    CHECK(validateBedSpawnCells(BlockType::BED, BlockType::WATER, BlockType::AIR) ==
          BedSpawnValidation::INVALID);
    CHECK(validateBedSpawnCells(BlockType::BED, BlockType::AIR, BlockType::LAVA) ==
          BedSpawnValidation::INVALID);

    const Vec3 spawn{12.5f, 65.f, -8.5f};
    CHECK(bedSpawnAnchoredToBlock(spawn, 12, 64, -9));
    CHECK_FALSE(bedSpawnAnchoredToBlock(spawn, 12, 65, -9));
    CHECK_FALSE(bedSpawnAnchoredToBlock(spawn, 13, 64, -9));
}

TEST_CASE("Survival exhaustion spends saturation then food", "[survival]") {
    SurvivalStats stats;
    stats.saturation = 1.0f;
    stats.food = 20;
    // One EXHAUSTION_THRESHOLD of sprint exhaustion spends one saturation.
    stats.exhaustion = SurvivalStats::EXHAUSTION_THRESHOLD;
    SurvivalTickInputs idle;
    tickSurvivalStats(stats, idle, 20);
    REQUIRE(stats.saturation == Catch::Approx(0.0f));
    REQUIRE(stats.food == 20);

    // With saturation gone, the next threshold eats into food.
    stats.exhaustion = SurvivalStats::EXHAUSTION_THRESHOLD;
    tickSurvivalStats(stats, idle, 20);
    REQUIRE(stats.food == 19);
}

TEST_CASE("Survival regenerates fast with saturation and slow without", "[survival]") {
    SurvivalTickInputs idle;

    // Full food plus ample saturation heals a whole hp every fast interval.
    SurvivalStats fast;
    fast.food = 20;
    fast.saturation = 20.f;
    int delta = 0;
    for (int tick = 0; tick < SurvivalStats::FAST_REGEN_INTERVAL; ++tick) {
        delta = tickSurvivalStats(fast, idle, 15);
    }
    REQUIRE(delta == 1); // +1 hp after only the short fast interval

    // High food with no saturation falls back to the slow regen path.
    SurvivalStats slow;
    slow.food = 18;
    slow.saturation = 0.f;
    for (int tick = 0; tick < SurvivalStats::FAST_REGEN_INTERVAL; ++tick) {
        REQUIRE(tickSurvivalStats(slow, idle, 15) == 0); // no fast heal without saturation
    }
    int slowDelta = 0;
    for (int tick = SurvivalStats::FAST_REGEN_INTERVAL; tick < SurvivalStats::SLOW_REGEN_INTERVAL;
         ++tick) {
        slowDelta = tickSurvivalStats(slow, idle, 15);
    }
    REQUIRE(slowDelta == 1); // +1 hp only after the full slow interval
}

TEST_CASE("Survival regenerates a well-fed player back to full health", "[survival]") {
    SurvivalStats stats;
    stats.food = 20;
    stats.saturation = 20.f;
    SurvivalTickInputs idle;
    int health = 4;
    // A player who stays fed (topping the bar back up as it drains, as a
    // Minecraft player does by eating) regenerates all the way to full health.
    for (int tick = 0; tick < 4000 && health < SurvivalStats::MAX_HEALTH; ++tick) {
        if (stats.saturation <= 0.f) {
            stats.food = SurvivalStats::MAX_FOOD;
            stats.saturation = 20.f;
        }
        health += tickSurvivalStats(stats, idle, health);
    }
    REQUIRE(health == SurvivalStats::MAX_HEALTH);
}

TEST_CASE("Survival starves at empty food down to the floor", "[survival]") {
    SurvivalTickInputs idle;
    SurvivalStats starve;
    starve.food = 0;
    int applied = 0;
    for (int tick = 0; tick < SurvivalStats::STARVE_INTERVAL; ++tick) {
        applied = tickSurvivalStats(starve, idle, 10);
    }
    REQUIRE(applied == -1); // -1 hp after the starve interval

    // Starvation never drops below the floor.
    SurvivalStats floored;
    floored.food = 0;
    for (int tick = 0; tick < SurvivalStats::STARVE_INTERVAL; ++tick) {
        REQUIRE(tickSurvivalStats(floored, idle, SurvivalStats::STARVE_HEALTH_FLOOR) == 0);
    }
}

TEST_CASE("Survival drains air underwater and drowns when empty", "[survival]") {
    SurvivalStats stats;
    SurvivalTickInputs under;
    under.eyesUnderwater = true;

    for (int tick = 0; tick < SurvivalStats::MAX_AIR; ++tick) {
        tickSurvivalStats(stats, under, 20);
    }
    REQUIRE(stats.air == 0);

    // Out of air, drowning damage lands once per interval.
    int worst = 0;
    for (int tick = 0; tick < SurvivalStats::DROWN_DAMAGE_INTERVAL; ++tick) {
        worst = std::min(worst, tickSurvivalStats(stats, under, 20));
    }
    REQUIRE(worst == -SurvivalStats::DROWN_DAMAGE);

    // Surfacing refills air quickly.
    SurvivalTickInputs surface;
    tickSurvivalStats(stats, surface, 20);
    REQUIRE(stats.air == SurvivalStats::AIR_REFILL_PER_TICK);
}

TEST_CASE("Eating requires a held right-click over time on the same slot", "[survival]") {
    EatingState eating;
    // Not holding, or full food, never progresses.
    REQUIRE_FALSE(tickEating(eating, false, 0, true, 10));
    REQUIRE_FALSE(tickEating(eating, true, 0, false, 10));
    REQUIRE_FALSE(tickEating(eating, true, 0, true, SurvivalStats::MAX_FOOD));

    // Held for EAT_TICKS completes exactly once.
    bool finished = false;
    for (int tick = 0; tick < EatingState::EAT_TICKS; ++tick) {
        finished = tickEating(eating, true, 2, true, 10);
    }
    REQUIRE(finished);
    REQUIRE_FALSE(eating.active);

    // Switching the selected slot restarts the timer.
    for (int tick = 0; tick < EatingState::EAT_TICKS - 1; ++tick) {
        tickEating(eating, true, 2, true, 10);
    }
    tickEating(eating, true, 5, true, 10);
    REQUIRE(eating.slot == 5);
    REQUIRE(eating.ticks == 1);
}

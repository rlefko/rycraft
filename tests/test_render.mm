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
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/far_terrain.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/mesh_scheduler.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/light_engine.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Rendering: meshing, textures, shared GPU layouts
// ===========================================================================

// ============================================================================
// Vertex Format Tests
// ============================================================================

TEST_CASE("Vertex size is 16 bytes", "[render][vertex]") {
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Vertex alignment is 16 bytes", "[render][vertex]") {
    REQUIRE(alignof(Vertex) == 16);
}

TEST_CASE("Vertex fields have expected sizes", "[render][vertex]") {
    REQUIRE(sizeof(float16_t) == 2);
    REQUIRE(sizeof(uint8_t) == 1);
    REQUIRE(sizeof(uint32_t) == 4);
}

namespace {

FarTerrainSource farTerrainTestSource() {
    FarTerrainSource source;
    source.geometry = [](int64_t x, int64_t z) {
        FarTerrainGeometrySample sample;
        const int64_t variation = world_coord::floorMod(x + z * 3, 29);
        sample.terrainHeight = 72.0 + static_cast<double>(variation) * 0.25;
        sample.waterSurface = SEA_LEVEL;
        return sample;
    };
    source.material = [](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        return world_coord::floorMod(x / 64 + z / 64, 2) == 0 ? BlockType::GRASS : BlockType::STONE;
    };
    return source;
}

std::map<int, float> farTerrainEdge(const FarTerrainMesh& mesh, bool eastEdge) {
    std::map<int, float> result;
    const float edgeX = eastEdge ? static_cast<float>(FAR_TERRAIN_TILE_EDGE) : 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
            unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::WATER) ||
            static_cast<float>(vertex.px) != edgeX) {
            continue;
        }
        result[static_cast<int>(static_cast<float>(vertex.pz))] = static_cast<float>(vertex.py);
    }
    return result;
}

} // namespace

TEST_CASE("Far terrain chooses globally specified LOD rings", "[render][far-terrain]") {
    REQUIRE_FALSE(farTerrainStepForChunkDistance(31.999).has_value());
    REQUIRE(farTerrainStepForChunkDistance(32.0) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(47.999) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(48.0) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(71.999) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(72.0) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(135.999) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(136.0) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForChunkDistance(255.999) == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(farTerrainStepForChunkDistance(256.0).has_value());
}

TEST_CASE("Far terrain samples exact material regions on a shared global lattice",
          "[render][far-terrain][material][lod][seam]") {
    std::set<std::pair<int64_t, int64_t>> sampledExactMaterials;
    std::set<std::pair<int64_t, int64_t>> sampledCoarseMaterials;
    FarTerrainSource source;
    source.geometry = [](int64_t, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 72.0;
        return sample;
    };
    source.nearGeometry = source.geometry;
    source.material = [&](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        REQUIRE(world_coord::floorMod(x, int64_t{FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE}) == 0);
        REQUIRE(world_coord::floorMod(z, int64_t{FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE}) == 0);
        sampledCoarseMaterials.emplace(x, z);
        return BlockType::STONE;
    };
    source.nearMaterial = [&](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        REQUIRE(world_coord::floorMod(x, int64_t{FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE}) == 0);
        REQUIRE(world_coord::floorMod(z, int64_t{FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE}) == 0);
        sampledExactMaterials.emplace(x, z);
        const int64_t cellX =
            world_coord::floorDiv(x, int64_t{FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE});
        const int64_t cellZ =
            world_coord::floorDiv(z, int64_t{FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE});
        return world_coord::floorMod(cellX + cellZ, int64_t{2}) == 0 ? BlockType::LIMESTONE
                                                                     : BlockType::ANDESITE;
    };

    const auto mesh = FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::TWO}, source);
    REQUIRE(sampledExactMaterials.size() == 64);
    REQUIRE(sampledCoarseMaterials.empty());
    bool sawLimestone = false;
    bool sawAndesite = false;
    for (const Vertex& vertex : mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        const uint8_t texture = unpackTextureLayer(vertex.faceAttr);
        sawLimestone = sawLimestone || texture == static_cast<uint8_t>(BlockType::LIMESTONE);
        sawAndesite = sawAndesite || texture == static_cast<uint8_t>(BlockType::ANDESITE);
    }
    REQUIRE(sawLimestone);
    REQUIRE(sawAndesite);

    sampledExactMaterials.clear();
    source.nearMaterial = [&](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        REQUIRE(world_coord::floorMod(x, int64_t{FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE}) == 0);
        REQUIRE(world_coord::floorMod(z, int64_t{FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE}) == 0);
        sampledExactMaterials.emplace(x, z);
        return BlockType::LIMESTONE;
    };
    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::FOUR}, source);
    REQUIRE(sampledExactMaterials.size() == 16);
    REQUIRE(sampledCoarseMaterials.empty());

    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::SIXTEEN}, source);
    REQUIRE(sampledCoarseMaterials.size() == 16);
    REQUIRE(sampledExactMaterials.size() == 16);
}

TEST_CASE("Fine scheduler terrain uses the exact submerged cube material",
          "[render][far-terrain][scheduler][material][water][lod][seam][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(42, limits);
    constexpr FarTerrainKey KEY{-54, 3, FarTerrainStep::FOUR};
    REQUIRE(scheduler.enqueue(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    scheduler.shutdown();
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);

    // The 64-block material cell rooted at world (-13,760, 832) is a lake
    // threshold where macro relief is dry but the exact emitted cube is
    // submerged clay. Inspect the open interior so adjacent material cells
    // cannot contribute a shared boundary vertex.
    bool foundExactClay = false;
    bool foundCoarseSilt = false;
    for (const Vertex& vertex : completed.front().mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y || vertex.px <= 64 ||
            vertex.px >= 128 || vertex.pz <= 64 || vertex.pz >= 128) {
            continue;
        }
        const BlockType material = static_cast<BlockType>(unpackTextureLayer(vertex.faceAttr));
        foundExactClay = foundExactClay || material == BlockType::CLAY;
        foundCoarseSilt = foundCoarseSilt || material == BlockType::SILT;
    }
    REQUIRE(foundExactClay);
    REQUIRE_FALSE(foundCoarseSilt);
}

TEST_CASE("Far terrain LOD responds to complexity with stable hysteresis",
          "[render][far-terrain][selection]") {
    REQUIRE(farTerrainStepForMetrics(80.0, 0.0F) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(80.0, 1.0F) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(150.0, 0.0F) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(150.0, 1.0F) == FarTerrainStep::EIGHT);

    REQUIRE(farTerrainStepForMetrics(49.0, 0.0F, FarTerrainStep::TWO) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(49.0, 0.0F, FarTerrainStep::FOUR) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(70.0, 0.0F, FarTerrainStep::FOUR) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(70.0, 0.0F, FarTerrainStep::EIGHT) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(40.0, 0.0F, FarTerrainStep::EIGHT) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(140.0, 0.0F, FarTerrainStep::SIXTEEN) ==
            FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(120.0, 0.0F, FarTerrainStep::SIXTEEN) ==
            FarTerrainStep::EIGHT);
}

TEST_CASE("Far terrain topology replacement is fog-continuous and ABI-neutral",
          "[render][far-terrain][transition]") {
    const auto start = sampleFarTerrainTransition(0.0F);
    const auto quarter = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.25F);
    const auto midpoint = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.5F);
    const auto threeQuarter =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.75F);
    const auto complete = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS);

    REQUIRE_FALSE(start.drawTarget);
    REQUIRE(start.fogBlend == 0.0F);
    REQUIRE_FALSE(quarter.drawTarget);
    REQUIRE(quarter.fogBlend == Catch::Approx(0.5F));
    REQUIRE(midpoint.drawTarget);
    REQUIRE(midpoint.fogBlend == 1.0F);
    REQUIRE(threeQuarter.drawTarget);
    REQUIRE(threeQuarter.fogBlend == Catch::Approx(0.5F));
    REQUIRE(complete.drawTarget);
    REQUIRE(complete.complete);
    REQUIRE(complete.fogBlend == 0.0F);
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Far terrain water and canopy share exact handoff coverage",
          "[render][far-terrain][shader][transition]") {
    STATIC_REQUIRE(FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS == CHUNK_EDGE);
    constexpr float EXACT_RADIUS = FAR_TERRAIN_NEAR_CHUNK_RADIUS * CHUNK_EDGE;
    constexpr float MIDPOINT = EXACT_RADIUS + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS * 0.5F;
    constexpr float OUTSIDE = EXACT_RADIUS + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS;

    auto requireSharedHandoff = [] {
        REQUIRE(farTerrainHandoffCoverage(EXACT_RADIUS - 0.001F, EXACT_RADIUS) == 0.0F);
        REQUIRE_FALSE(farTerrainHandoffVisible(EXACT_RADIUS - 0.001F, EXACT_RADIUS, 0.0F));
        REQUIRE_FALSE(farTerrainHandoffVisible(EXACT_RADIUS, EXACT_RADIUS, 0.0F));
        REQUIRE(farTerrainHandoffCoverage(MIDPOINT, EXACT_RADIUS) == Catch::Approx(0.5F));
        REQUIRE(farTerrainHandoffVisible(MIDPOINT, EXACT_RADIUS, 0.49F));
        REQUIRE_FALSE(farTerrainHandoffVisible(MIDPOINT, EXACT_RADIUS, 0.51F));
        REQUIRE(farTerrainHandoffCoverage(OUTSIDE, EXACT_RADIUS) == 1.0F);
        REQUIRE(farTerrainHandoffVisible(OUTSIDE, EXACT_RADIUS, 1.0F));
    };

    SECTION("standing water") {
        requireSharedHandoff();
    }
    SECTION("far canopies") {
        requireSharedHandoff();
    }

    // Exact draws carry radius zero and must never be clipped by the shared
    // shader path. Far LOD selection changes topology, not handoff coverage.
    REQUIRE(farTerrainHandoffVisible(0.0F, 0.0F, 1.0F));
    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        CAPTURE(step);
        REQUIRE(farTerrainHandoffVisible(OUTSIDE, EXACT_RADIUS, 0.99F));
    }
}

TEST_CASE("Far terrain skirts stay hidden throughout the exact handoff",
          "[render][far-terrain][shader][transition][skirt][regression]") {
    constexpr float EXACT_RADIUS = FAR_TERRAIN_NEAR_CHUNK_RADIUS * CHUNK_EDGE;
    constexpr float MIDPOINT = EXACT_RADIUS + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS * 0.5F;
    constexpr float OUTSIDE = EXACT_RADIUS + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS;

    REQUIRE_FALSE(farTerrainSkirtVisible(EXACT_RADIUS, EXACT_RADIUS));
    REQUIRE_FALSE(farTerrainSkirtVisible(MIDPOINT, EXACT_RADIUS));
    REQUIRE(farTerrainSkirtVisible(OUTSIDE, EXACT_RADIUS));
    REQUIRE(farTerrainSkirtVisible(0.0F, 0.0F));

    std::array<std::optional<FarTerrainStep>, 4> neighbors = {
        FarTerrainStep::FOUR, FarTerrainStep::TWO, std::nullopt, FarTerrainStep::EIGHT};
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::TWO, neighbors) ==
            ((1U << static_cast<uint8_t>(FaceNormal::PLUS_X)) |
             (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z))));
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::FOUR, neighbors) ==
            (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z)));
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::SIXTEEN, neighbors) == 0U);

    FarTerrainSource source;
    source.geometry = [](int64_t, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 80.0;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::STONE;
    };
    const auto mesh = FarTerrainMesher::build({-98, -39, FarTerrainStep::TWO}, source);
    const auto markedVertices =
        std::count_if(mesh->vertices.begin(), mesh->vertices.end(), [](const Vertex& vertex) {
            return (vertex.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) != 0U;
        });
    REQUIRE(markedVertices == static_cast<std::ptrdiff_t>(mesh->skirtQuadCount * 4));
    REQUIRE(std::none_of(mesh->vertices.begin(), mesh->vertices.end(), [](const Vertex& vertex) {
        return unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
               (vertex.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) != 0U;
    }));
}

TEST_CASE("Far terrain view selection is circular ordered and negative-coordinate safe",
          "[render][far-terrain][selection]") {
    constexpr double cameraX = -320.5;
    constexpr double cameraZ = -511.25;
    constexpr int exactRadius = 32;
    constexpr int visibleRadius = 256;
    const double exactSquared = std::pow(exactRadius * CHUNK_EDGE, 2.0);
    const double visibleSquared = std::pow(visibleRadius * CHUNK_EDGE, 2.0);

    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(cameraX, cameraZ, exactRadius, visibleRadius, selected);
    REQUIRE_FALSE(selected.empty());

    std::array<bool, 4> reachedStep{};
    bool sawNegativeTile = false;
    bool sawExactBoundaryOverlap = false;
    double previousDistance = -1.0;
    for (const FarTerrainViewTile& tile : selected) {
        REQUIRE(tile.distanceSquared >= previousDistance);
        previousDistance = tile.distanceSquared;
        sawNegativeTile = sawNegativeTile || tile.key.tileX < 0 || tile.key.tileZ < 0;
        reachedStep[tile.key.step == FarTerrainStep::TWO     ? 0
                    : tile.key.step == FarTerrainStep::FOUR  ? 1
                    : tile.key.step == FarTerrainStep::EIGHT ? 2
                                                             : 3] = true;

        double nearestSquared = 0.0;
        if (cameraX < tile.bounds.minX)
            nearestSquared += std::pow(tile.bounds.minX - cameraX, 2.0);
        if (cameraX > tile.bounds.maxX)
            nearestSquared += std::pow(cameraX - tile.bounds.maxX, 2.0);
        if (cameraZ < tile.bounds.minZ)
            nearestSquared += std::pow(tile.bounds.minZ - cameraZ, 2.0);
        if (cameraZ > tile.bounds.maxZ)
            nearestSquared += std::pow(cameraZ - tile.bounds.maxZ, 2.0);
        REQUIRE(nearestSquared < visibleSquared);
        sawExactBoundaryOverlap = sawExactBoundaryOverlap || nearestSquared < exactSquared;

        double farthestSquared = 0.0;
        for (const int64_t x : {tile.bounds.minX, tile.bounds.maxX}) {
            for (const int64_t z : {tile.bounds.minZ, tile.bounds.maxZ}) {
                farthestSquared =
                    std::max(farthestSquared, std::pow(static_cast<double>(x) - cameraX, 2.0) +
                                                  std::pow(static_cast<double>(z) - cameraZ, 2.0));
            }
        }
        REQUIRE(farthestSquared > exactSquared);
    }

    REQUIRE(sawNegativeTile);
    REQUIRE(sawExactBoundaryOverlap);
    REQUIRE(
        std::all_of(reachedStep.begin(), reachedStep.end(), [](bool reached) { return reached; }));
}

TEST_CASE("The first far LOD shares the exact emitted cubic surface",
          "[render][far-terrain][seam][exact]") {
    ChunkGenerator generator(42);
    constexpr int64_t WORLD_X = 16;
    constexpr int64_t WORLD_Z = 84;
    const worldgen::SurfaceSample provisional = generator.sampleSurface(WORLD_X, WORLD_Z);
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(WORLD_X, WORLD_Z);
    REQUIRE(std::abs(provisional.terrainHeight - exact.terrainHeight) > 20.0);
    REQUIRE(exact.terrainHeight == generator.surfaceYAt(WORLD_X, WORLD_Z) + 1.0);

    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });
    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    bool foundSharedVertex = false;
    for (const Vertex& vertex : mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
            unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::WATER) ||
            static_cast<float>(vertex.px) != static_cast<float>(WORLD_X) ||
            static_cast<float>(vertex.pz) != static_cast<float>(WORLD_Z)) {
            continue;
        }
        REQUIRE(static_cast<float>(vertex.py) == static_cast<float>(exact.terrainHeight));
        foundSharedVertex = true;
    }
    REQUIRE(foundSharedVertex);
}

TEST_CASE("Far generated water uses the exact source-block surface plane",
          "[render][far-terrain][water][seam][exact]") {
    ChunkGenerator generator(42);
    constexpr int64_t LAKE_X = -8352;
    constexpr int64_t LAKE_Z = 2160;
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(LAKE_X, LAKE_Z);
    REQUIRE(exact.hydrology.lake);
    const FarTerrainSource source = FarTerrainMesher::surfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); });
    const FarTerrainGeometrySample geometry = source.geometry(LAKE_X, LAKE_Z);
    REQUIRE(geometry.lake);
    REQUIRE(geometry.waterSurface == std::ceil(exact.waterSurface) - 0.125);
    REQUIRE(geometry.waterSurface ==
            std::ceil(exact.waterSurface) - 1.0 + fluidSurfaceHeight(FluidState::source()));
}

TEST_CASE("Canonical lake contours agree through every far LOD",
          "[render][far-terrain][water][lake][seam][lod]") {
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });
    constexpr FarTerrainKey BASE_KEY{-33, 8, FarTerrainStep::TWO};
    constexpr int64_t WORLD_Z = 2'288;
    constexpr int64_t WET_X = -8'256;
    constexpr int64_t SCAN_END_X = -8'224;
    constexpr float LOCAL_Z = 240.0F;
    constexpr float MINIMUM_LOCAL_X = 192.0F;
    constexpr float MAXIMUM_LOCAL_X = 224.0F;
    constexpr int64_t TILE_ORIGIN_X = BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE;

    bool foundWetReference = false;
    int64_t firstDryReferenceX = std::numeric_limits<int64_t>::max();
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const FarTerrainGeometrySample sample = source.nearGeometry(x, WORLD_Z);
        if (sample.lake) {
            foundWetReference = true;
        } else if (foundWetReference) {
            firstDryReferenceX = x;
            break;
        }
    }
    REQUIRE(foundWetReference);
    REQUIRE(firstDryReferenceX != std::numeric_limits<int64_t>::max());
    const float expectedEdgeX = static_cast<float>(firstDryReferenceX - TILE_ORIGIN_X) - 0.5F;
    const float expectedWaterY =
        static_cast<float>(source.nearGeometry(firstDryReferenceX - 1, WORLD_Z).waterSurface);

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const int spacing = farTerrainStepSize(step);
        REQUIRE(world_coord::floorMod(WET_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
        REQUIRE(world_coord::floorMod(SCAN_END_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
    }
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const FarTerrainGeometrySample exact = source.nearGeometry(x, WORLD_Z);
        const FarTerrainGeometrySample coarse = source.geometry(x, WORLD_Z);
        REQUIRE(exact.lake == coarse.lake);
        REQUIRE(exact.waterSurface == Catch::Approx(coarse.waterSurface).margin(1.0e-5));
    }

    const auto rightmostWaterAtCut = [=](const FarTerrainMesh& mesh) {
        float rightmost = -std::numeric_limits<float>::infinity();
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<Vec3, 3> triangle{};
            bool waterTriangle = true;
            for (size_t corner = 0; corner < triangle.size(); ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                waterTriangle =
                    waterTriangle && unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                    unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::WATER) &&
                    static_cast<float>(vertex.py) == expectedWaterY;
                triangle[corner] = {static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                                    static_cast<float>(vertex.pz)};
            }
            if (!waterTriangle)
                continue;
            for (size_t edge = 0; edge < triangle.size(); ++edge) {
                const Vec3& first = triangle[edge];
                const Vec3& second = triangle[(edge + 1) % triangle.size()];
                if (first.z == LOCAL_Z && second.z == LOCAL_Z) {
                    for (const float x : {first.x, second.x}) {
                        if (x >= MINIMUM_LOCAL_X && x <= MAXIMUM_LOCAL_X) {
                            rightmost = std::max(rightmost, x);
                        }
                    }
                    continue;
                }
                if ((first.z < LOCAL_Z && second.z < LOCAL_Z) ||
                    (first.z > LOCAL_Z && second.z > LOCAL_Z) || first.z == second.z) {
                    continue;
                }
                const float amount = (LOCAL_Z - first.z) / (second.z - first.z);
                if (amount < 0.0F || amount > 1.0F)
                    continue;
                const float x = first.x + (second.x - first.x) * amount;
                if (x >= MINIMUM_LOCAL_X && x <= MAXIMUM_LOCAL_X) {
                    rightmost = std::max(rightmost, x);
                }
            }
        }
        return rightmost;
    };

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(rightmostWaterAtCut(*mesh) == Catch::Approx(expectedEdgeX).margin(0.51));
    }
}

TEST_CASE("Far lake outlets use narrow explicit falling prisms at every LOD",
          "[render][far-terrain][water][lake][waterfall][seam][lod]") {
    constexpr int64_t FALL_X = -8'256;
    constexpr int64_t FALL_Z = 3'072;
    constexpr FarTerrainKey BASE_KEY{-33, 12, FarTerrainStep::TWO};
    constexpr float LOCAL_FALL_X = 192.0F;
    constexpr float LOCAL_FALL_Z = 0.0F;
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });

    const FarTerrainGeometrySample near = source.nearGeometry(FALL_X, FALL_Z);
    const FarTerrainGeometrySample coarse = source.geometry(FALL_X, FALL_Z);
    for (const FarTerrainGeometrySample* sample : {&near, &coarse}) {
        REQUIRE_FALSE(sample->ocean);
        REQUIRE(sample->river);
        REQUIRE(sample->waterfall);
        REQUIRE(sample->waterfallAnchor);
        REQUIRE(sample->waterSurface == Catch::Approx(72.875));
        REQUIRE(sample->waterfallBottom == Catch::Approx(73.0));
        REQUIRE(sample->waterfallTop == Catch::Approx(81.14503479).margin(1.0e-4));
        REQUIRE(sample->waterfallWidth >= 4.0);
    }

    const double flowLength = std::hypot(coarse.flowX, coarse.flowZ);
    REQUIRE(flowLength > 0.0);
    const double flowX = coarse.flowX / flowLength;
    const double flowZ = coarse.flowZ / flowLength;
    const int outsideOffset = static_cast<int>(std::ceil(coarse.waterfallWidth + 4.0));
    const int64_t outsideX = FALL_X + static_cast<int64_t>(std::llround(-flowZ * outsideOffset));
    const int64_t outsideZ = FALL_Z + static_cast<int64_t>(std::llround(flowX * outsideOffset));
    const FarTerrainGeometrySample outside = source.geometry(outsideX, outsideZ);
    REQUIRE_FALSE(outside.waterfall);
    if (outside.waterSurface > SEA_LEVEL)
        REQUIRE(outside.lake);

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(mesh->waterfallQuadCount >= 5);
        float minimumFallingY = std::numeric_limits<float>::max();
        float maximumFallingY = std::numeric_limits<float>::lowest();
        double minimumAlong = std::numeric_limits<double>::max();
        double maximumAlong = std::numeric_limits<double>::lowest();
        size_t pinnedVertices = 0;
        size_t verticalVertices = 0;
        for (const Vertex& vertex : mesh->vertices) {
            if (!unpackFluidFalling(vertex.faceAttr))
                continue;
            const float localX = static_cast<float>(vertex.px);
            const float localZ = static_cast<float>(vertex.pz);
            if (std::abs(localX - LOCAL_FALL_X) > 12.0F ||
                std::abs(localZ - LOCAL_FALL_Z) > 12.0F) {
                continue;
            }
            const double offsetX = localX - LOCAL_FALL_X;
            const double offsetZ = localZ - LOCAL_FALL_Z;
            const double along = offsetX * flowX + offsetZ * flowZ;
            const double cross = -offsetX * flowZ + offsetZ * flowX;
            minimumAlong = std::min(minimumAlong, along);
            maximumAlong = std::max(maximumAlong, along);
            REQUIRE(std::abs(cross) <= coarse.waterfallWidth * 0.5 + 0.75);
            minimumFallingY = std::min(minimumFallingY, static_cast<float>(vertex.py));
            maximumFallingY = std::max(maximumFallingY, static_cast<float>(vertex.py));
            if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
                ++verticalVertices;
            ++pinnedVertices;
        }
        REQUIRE(pinnedVertices == 20);
        REQUIRE(verticalVertices == 16);
        REQUIRE(maximumAlong - minimumAlong <= 3.1);
        REQUIRE(minimumFallingY <= near.waterSurface);
        REQUIRE(maximumFallingY == Catch::Approx(std::ceil(near.waterfallTop)));

        const auto rebuilt =
            FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(rebuilt->deterministicHash == mesh->deterministicHash);
    }
}

TEST_CASE("Far terrain retains deterministic forests through every LOD",
          "[render][far-terrain][canopy][lod]") {
    FarTerrainSource source;
    source.geometry = [](int64_t, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 64.0;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::GRASS;
    };
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        std::vector<FarCanopy> canopies;
        for (uint64_t index = 0; index < 4; ++index) {
            FarCanopy canopy;
            canopy.x = 24 + static_cast<int64_t>(index) * 48;
            canopy.z = 96;
            canopy.baseY = 64;
            canopy.topY = 70;
            canopy.canopyMinimumY = 67;
            canopy.canopyMaximumY = 72;
            canopy.canopyRadius = 2;
            canopy.logBlock = BlockType::LOG;
            canopy.leafBlock = BlockType::LEAVES;
            canopy.anchorId = index;
            canopies.push_back(canopy);
        }
        FarCanopy neighboringCanopy = canopies.front();
        neighboringCanopy.x = -1;
        neighboringCanopy.anchorId = 5;
        canopies.push_back(neighboringCanopy);
        return canopies;
    };

    const auto two = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const auto four = FarTerrainMesher::build({0, 0, FarTerrainStep::FOUR}, source);
    const auto eight = FarTerrainMesher::build({0, 0, FarTerrainStep::EIGHT}, source);
    const auto sixteen = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);
    REQUIRE(two->canopyAnchorCount == 4);
    REQUIRE(four->canopyAnchorCount == two->canopyAnchorCount);
    REQUIRE(eight->canopyAnchorCount == 3);
    REQUIRE(sixteen->canopyAnchorCount == 2);

    for (const auto& mesh : {two, four, eight, sixteen}) {
        REQUIRE(mesh->canopyImpostorQuadCount == mesh->canopyAnchorCount * 9);
        const size_t canopyVertexCount = static_cast<size_t>(mesh->canopyImpostorQuadCount) * 4;
        REQUIRE(
            std::count_if(mesh->vertices.begin(), mesh->vertices.end(), [](const Vertex& vertex) {
                return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U;
            }) == static_cast<std::ptrdiff_t>(canopyVertexCount));
        REQUIRE(mesh->surfaceBounds.maxY == 73.0F);
    }
    REQUIRE(two->deterministicHash ==
            FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source)->deterministicHash);
}

TEST_CASE("Far terrain greedily merges flat terrain and water", "[render][far-terrain]") {
    FarTerrainSource source;
    source.geometry = [](int64_t, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 40.0;
        sample.waterSurface = 64.0;
        sample.ocean = true;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::SAND;
    };
    const auto mesh = FarTerrainMesher::build(FarTerrainKey{-2, 3, FarTerrainStep::FOUR}, source);
    REQUIRE(sizeof(Vertex) == 16);
    REQUIRE(mesh->originX == -512);
    REQUIRE(mesh->originZ == 768);
    REQUIRE(mesh->terrainQuadCount == 1);
    REQUIRE(mesh->waterQuadCount == 1);
    REQUIRE(mesh->skirtQuadCount == 256);
    REQUIRE(mesh->mergedTerrainCellCount == 4096);
    REQUIRE(mesh->opaqueIndexCount == (mesh->terrainQuadCount + mesh->skirtQuadCount) * 6);
    REQUIRE(mesh->surfaceBounds.minY == 40.0F);
    REQUIRE(mesh->surfaceBounds.maxY == 64.0F);
    REQUIRE(mesh->bounds.minY == -24.0F);
    REQUIRE(mesh->bounds.maxY == 64.0F);
    REQUIRE(mesh->bounds.minX == -512);
    REQUIRE(mesh->bounds.maxX == -256);
    for (const FarTerrainBounds& patch : mesh->occluderPatches) {
        REQUIRE(patch.maxX - patch.minX == FAR_TERRAIN_OCCLUDER_PATCH_EDGE);
        REQUIRE(patch.maxZ - patch.minZ == FAR_TERRAIN_OCCLUDER_PATCH_EDGE);
        REQUIRE(patch.minY == 40.0F);
        REQUIRE(patch.maxY == 40.0F);
    }

    std::array<int, 6> faceCounts{};
    for (uint32_t indexOffset = 0; indexOffset < mesh->opaqueIndexCount; indexOffset += 6) {
        const Vertex& a = mesh->vertices[mesh->indices[indexOffset]];
        const Vertex& b = mesh->vertices[mesh->indices[indexOffset + 1]];
        const Vertex& c = mesh->vertices[mesh->indices[indexOffset + 2]];
        const float abX = static_cast<float>(b.px) - static_cast<float>(a.px);
        const float abY = static_cast<float>(b.py) - static_cast<float>(a.py);
        const float abZ = static_cast<float>(b.pz) - static_cast<float>(a.pz);
        const float acX = static_cast<float>(c.px) - static_cast<float>(a.px);
        const float acY = static_cast<float>(c.py) - static_cast<float>(a.py);
        const float acZ = static_cast<float>(c.pz) - static_cast<float>(a.pz);
        const Vec3 normal{abY * acZ - abZ * acY, abZ * acX - abX * acZ, abX * acY - abY * acX};
        const FaceNormal face = unpackFace(a.faceAttr);
        ++faceCounts[static_cast<size_t>(face)];
        switch (face) {
            case FaceNormal::PLUS_X:
                REQUIRE(normal.x > 0.0F);
                break;
            case FaceNormal::MINUS_X:
                REQUIRE(normal.x < 0.0F);
                break;
            case FaceNormal::PLUS_Z:
                REQUIRE(normal.z > 0.0F);
                break;
            case FaceNormal::MINUS_Z:
                REQUIRE(normal.z < 0.0F);
                break;
            case FaceNormal::PLUS_Y:
                REQUIRE(normal.y > 0.0F);
                break;
            default:
                FAIL("unexpected far terrain opaque face");
        }
    }
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_Y)] == 1);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_X)] == 64);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::MINUS_X)] == 64);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_Z)] == 64);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::MINUS_Z)] == 64);
}

TEST_CASE("Far terrain water follows deterministic shoreline contours",
          "[render][far-terrain][water][seam]") {
    FarTerrainSource source;
    source.geometry = [](int64_t x, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 40.0;
        sample.waterSurface = 64.0;
        sample.ocean = x < FAR_TERRAIN_TILE_EDGE / 2;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::SAND;
    };

    const FarTerrainKey key{0, 0, FarTerrainStep::SIXTEEN};
    const auto first = FarTerrainMesher::build(key, source);
    const auto second = FarTerrainMesher::build(key, source);
    REQUIRE(first->deterministicHash == second->deterministicHash);
    REQUIRE(first->indices == second->indices);
    REQUIRE(first->waterContourTriangleCount > 0);
    REQUIRE(first->waterQuadCount > 0);
    REQUIRE(first->complexity == 1.0F);

    float easternmostWater = 0.0F;
    bool sawContourVertex = false;
    for (size_t offset = first->opaqueIndexCount; offset < first->indices.size(); offset += 3) {
        std::array<Vec3, 3> triangle{};
        for (size_t corner = 0; corner < triangle.size(); ++corner) {
            const Vertex& vertex = first->vertices[first->indices[offset + corner]];
            REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
            const float x = static_cast<float>(vertex.px);
            easternmostWater = std::max(easternmostWater, x);
            sawContourVertex =
                sawContourVertex || (x > FAR_TERRAIN_TILE_EDGE / 2 - farTerrainStepSize(key.step) &&
                                     x < FAR_TERRAIN_TILE_EDGE / 2);
            triangle[corner] =
                Vec3{x, static_cast<float>(vertex.py), static_cast<float>(vertex.pz)};
        }
        REQUIRE((triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).y > 0.0F);
    }
    REQUIRE(sawContourVertex);
    REQUIRE(easternmostWater < FAR_TERRAIN_TILE_EDGE / 2);
}

TEST_CASE("Coarse lake contours stop at the supported shoreline",
          "[render][far-terrain][water][lake][support]") {
    FarTerrainSource source;
    source.geometry = [](int64_t x, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = x < 100 ? 74.0 : 84.0;
        sample.waterSurface = x < 100 ? 80.0 : SEA_LEVEL;
        sample.lake = x < 100;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::CLAY;
    };

    const auto mesh = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}, source);
    float easternmostWater = 0.0F;
    for (size_t offset = mesh->opaqueIndexCount; offset < mesh->indices.size(); ++offset) {
        const Vertex& vertex = mesh->vertices[mesh->indices[offset]];
        easternmostWater = std::max(easternmostWater, static_cast<float>(vertex.px));
        REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
    }
    REQUIRE(easternmostWater >= 99.0F);
    REQUIRE(easternmostWater <= 99.5F);
}

TEST_CASE("Far terrain shoreline contours stitch across tile faces",
          "[render][far-terrain][water][seam]") {
    FarTerrainSource source;
    source.geometry = [](int64_t x, int64_t z) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 40.0;
        sample.waterSurface = 64.0;
        sample.ocean = z < x - FAR_TERRAIN_TILE_EDGE / 2;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::SAND;
    };
    const auto left = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto right =
        FarTerrainMesher::build(FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}, source);

    const auto edgeCoverage = [](const FarTerrainMesh& mesh, float edgeX) {
        std::vector<std::pair<float, float>> intervals;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<float, 3> edgeZ{};
            size_t edgeVertexCount = 0;
            for (size_t corner = 0; corner < 3; ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                if (static_cast<float>(vertex.px) == edgeX) {
                    edgeZ[edgeVertexCount++] = static_cast<float>(vertex.pz);
                }
            }
            if (edgeVertexCount >= 2) {
                const auto [minimum, maximum] =
                    std::minmax_element(edgeZ.begin(), edgeZ.begin() + edgeVertexCount);
                if (*minimum < *maximum)
                    intervals.emplace_back(*minimum, *maximum);
            }
        }
        std::sort(intervals.begin(), intervals.end());
        std::vector<std::pair<float, float>> merged;
        for (const auto interval : intervals) {
            if (merged.empty() || interval.first > merged.back().second) {
                merged.push_back(interval);
            } else {
                merged.back().second = std::max(merged.back().second, interval.second);
            }
        }
        return merged;
    };
    const auto leftEdge = edgeCoverage(*left, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto rightEdge = edgeCoverage(*right, 0.0F);
    REQUIRE_FALSE(leftEdge.empty());
    REQUIRE(leftEdge == rightEdge);
}

TEST_CASE("Narrow rivers retain identical coverage across mixed LOD tile faces",
          "[render][far-terrain][water][seam][lod]") {
    FarTerrainSource source;
    source.geometry = [](int64_t, int64_t z) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 40.0;
        sample.waterSurface = 64.0;
        sample.river = z >= 5 && z <= 7;
        return sample;
    };
    source.material = [](int64_t, int64_t, const FarTerrainGeometrySample&) {
        return BlockType::CLAY;
    };
    const auto fine = FarTerrainMesher::build({0, 0, FarTerrainStep::FOUR}, source);
    const auto coarse = FarTerrainMesher::build({1, 0, FarTerrainStep::SIXTEEN}, source);

    const auto edgeCoverage = [](const FarTerrainMesh& mesh, float edgeX) {
        std::vector<std::pair<float, float>> intervals;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<float, 3> edgeZ{};
            size_t count = 0;
            for (size_t corner = 0; corner < 3; ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                if (static_cast<float>(vertex.px) == edgeX) {
                    edgeZ[count++] = static_cast<float>(vertex.pz);
                }
            }
            if (count < 2)
                continue;
            const auto [minimum, maximum] =
                std::minmax_element(edgeZ.begin(), edgeZ.begin() + count);
            if (*minimum < *maximum)
                intervals.emplace_back(*minimum, *maximum);
        }
        std::sort(intervals.begin(), intervals.end());
        return intervals;
    };

    const auto fineEdge = edgeCoverage(*fine, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto coarseEdge = edgeCoverage(*coarse, 0.0F);
    REQUIRE_FALSE(fineEdge.empty());
    REQUIRE(fineEdge == coarseEdge);
    REQUIRE(fine->complexity == 1.0F);
    REQUIRE(coarse->complexity == 1.0F);
}

TEST_CASE("Far terrain meshes are order independent and stitch tile edges",
          "[render][far-terrain][determinism]") {
    const FarTerrainSource source = farTerrainTestSource();
    const FarTerrainKey leftKey{-1, -2, FarTerrainStep::FOUR};
    const FarTerrainKey rightKey{0, -2, FarTerrainStep::FOUR};
    const auto leftFirst = FarTerrainMesher::build(leftKey, source);
    const auto rightSecond = FarTerrainMesher::build(rightKey, source);
    const auto rightFirst = FarTerrainMesher::build(rightKey, source);
    const auto leftSecond = FarTerrainMesher::build(leftKey, source);
    REQUIRE(leftFirst->deterministicHash == leftSecond->deterministicHash);
    REQUIRE(rightFirst->deterministicHash == rightSecond->deterministicHash);
    REQUIRE(leftFirst->vertices.size() == leftSecond->vertices.size());
    REQUIRE(leftFirst->indices == leftSecond->indices);

    const std::map<int, float> leftEdge = farTerrainEdge(*leftFirst, true);
    const std::map<int, float> rightEdge = farTerrainEdge(*rightFirst, false);
    REQUIRE_FALSE(leftEdge.empty());
    REQUIRE(leftEdge == rightEdge);
}

TEST_CASE("Far terrain LOD edges share aligned samples and carry downward skirts",
          "[render][far-terrain][seam]") {
    const FarTerrainSource source = farTerrainTestSource();
    const auto fine = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::FOUR}, source);
    const auto coarse =
        FarTerrainMesher::build(FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}, source);
    const std::map<int, float> fineEdge = farTerrainEdge(*fine, true);
    const std::map<int, float> coarseEdge = farTerrainEdge(*coarse, false);
    for (const auto& [z, height] : coarseEdge) {
        REQUIRE(fineEdge.contains(z));
        REQUIRE(fineEdge.at(z) == height);
    }
    REQUIRE(fine->skirtQuadCount == 256);
    REQUIRE(coarse->skirtQuadCount == 64);
    REQUIRE(fine->bounds.minY >= static_cast<float>(WORLD_MIN_Y));
    REQUIRE(fine->bounds.minY <= fine->surfaceBounds.minY - FAR_TERRAIN_SKIRT_DEPTH + 0.01F);
}

TEST_CASE("Terrain horizon culling is conservative", "[render][far-terrain][occlusion]") {
    TerrainHorizonCuller culler({0.0, 64.0, 0.0});
    const FarTerrainBounds uniformRidge{100, 200, -100, 100, 200.0F, 220.0F};
    const FarTerrainBounds hiddenLowland{400, 500, -50, 50, 20.0F, 80.0F};
    REQUIRE_FALSE(culler.testAndAdd(uniformRidge));
    REQUIRE(culler.isOccluded(hiddenLowland));

    const FarTerrainBounds tallPeak{400, 500, -50, 50, 20.0F, 320.0F};
    REQUIRE_FALSE(culler.isOccluded(tallPeak));

    culler.reset({0.0, 64.0, 0.0});
    const FarTerrainBounds peakWithLowValleys{100, 200, -100, 100, 0.0F, 300.0F};
    culler.addOccluder(peakWithLowValleys);
    REQUIRE_FALSE(culler.isOccluded(hiddenLowland));

    culler.reset({0.0, 64.0, 0.0});
    const FarTerrainBounds narrowRidge{100, 200, -5, 5, 200.0F, 220.0F};
    culler.addOccluder(narrowRidge);
    REQUIRE_FALSE(culler.isOccluded(hiddenLowland));
    REQUIRE_FALSE(culler.isOccluded(FarTerrainBounds{-10, 10, -10, 10, 0.0F, 500.0F}));

    SECTION("terrain below a high camera uses conservative distance extrema") {
        culler.reset({0.0, 480.0, 0.0});
        const FarTerrainBounds lowNearRidge{100, 500, -100, 100, 0.0F, 100.0F};
        const FarTerrainBounds lowFarCandidate{600, 700, -50, 50, -128.0F, -100.0F};
        culler.addOccluder(lowNearRidge);
        REQUIRE_FALSE(culler.isOccluded(lowFarCandidate));
    }

    SECTION("terrain below a deep-world camera remains visible without full coverage") {
        culler.reset({0.0, 480.0, 0.0});
        const FarTerrainBounds narrowLowRidge{100, 200, -4, 4, -100.0F, 0.0F};
        const FarTerrainBounds wideLowCandidate{400, 500, -100, 100, -120.0F, -20.0F};
        culler.addOccluder(narrowLowRidge);
        REQUIRE_FALSE(culler.isOccluded(wideLowCandidate));
    }

    SECTION("candidate fringe bins must also have a valid occluder") {
        culler.reset({0.0, 64.0, 0.0});
        const FarTerrainBounds binAlignedRidge{108, 208, -8, 8, 200.0F, 220.0F};
        const FarTerrainBounds partialFringeCandidate{400, 500, -32, 32, 20.0F, 80.0F};
        culler.addOccluder(binAlignedRidge);
        REQUIRE_FALSE(culler.isOccluded(partialFringeCandidate));
    }

    SECTION("a farther horizon never hides nearer terrain") {
        culler.reset({0.0, 64.0, 0.0});
        const FarTerrainBounds farRidge{400, 500, -100, 100, 260.0F, 300.0F};
        const FarTerrainBounds nearCandidate{100, 200, -50, 50, 20.0F, 80.0F};
        culler.addOccluder(farRidge);
        REQUIRE_FALSE(culler.isOccluded(nearCandidate));
    }
}

TEST_CASE("Far terrain scheduler bounds work and never builds on the caller",
          "[render][far-terrain][scheduler]") {
    const std::thread::id caller = std::this_thread::get_id();
    std::mutex threadMutex;
    std::condition_variable threadCv;
    std::set<std::thread::id> workerThreads;
    bool workersReleased = false;
    bool workerGateTimedOut = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto geometry = source.geometry;
    source.geometry = [&](int64_t x, int64_t z) {
        {
            std::unique_lock lock(threadMutex);
            workerThreads.insert(std::this_thread::get_id());
            if (workerThreads.size() == FarTerrainScheduler::WORKER_COUNT) {
                workersReleased = true;
                threadCv.notify_all();
            } else if (!workersReleased && !threadCv.wait_for(lock, std::chrono::seconds(2),
                                                              [&] { return workersReleased; })) {
                workerGateTimedOut = true;
                workersReleased = true;
                threadCv.notify_all();
            }
        }
        return geometry(x, z);
    };
    FarTerrainSchedulerLimits limits;
    constexpr int JOB_COUNT = static_cast<int>(FarTerrainScheduler::WORKER_COUNT * 2);
    limits.maxPending = JOB_COUNT;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 8 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    for (int index = 0; index < JOB_COUNT; ++index) {
        REQUIRE(scheduler.enqueue({index, 0, FarTerrainStep::SIXTEEN}));
    }
    REQUIRE_FALSE(scheduler.enqueue({JOB_COUNT, 0, FarTerrainStep::SIXTEEN}));
    REQUIRE_FALSE(scheduler.enqueue({0, 0, FarTerrainStep::SIXTEEN}));
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const FarTerrainSchedulerStats stats = scheduler.stats();
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.built == JOB_COUNT);
    REQUIRE(stats.completed == 2);
    REQUIRE(stats.cacheEntries <= 2);
    REQUIRE(stats.cacheBytes <= limits.maxCacheBytes);
    {
        std::lock_guard lock(threadMutex);
        REQUIRE_FALSE(workerGateTimedOut);
        REQUIRE(workerThreads.size() == FarTerrainScheduler::WORKER_COUNT);
        REQUIRE_FALSE(workerThreads.contains(caller));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 2);
    for (const FarTerrainResult& result : completed) {
        REQUIRE_FALSE(result.failed);
        REQUIRE(result.mesh);
        REQUIRE(result.epoch == scheduler.currentEpoch());
    }
}

TEST_CASE("Far terrain scheduler discards canceled epochs",
          "[render][far-terrain][scheduler][cancellation]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool entered = false;
    bool released = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto geometry = source.geometry;
    source.geometry = [&](int64_t x, int64_t z) {
        {
            std::unique_lock lock(gateMutex);
            if (!entered) {
                entered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return released; });
            }
        }
        return geometry(x, z);
    };
    FarTerrainScheduler scheduler(source);
    REQUIRE(scheduler.enqueue({0, 0, FarTerrainStep::SIXTEEN}));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; }));
    }
    const uint64_t newEpoch = scheduler.advanceEpoch();
    {
        std::lock_guard lock(gateMutex);
        released = true;
    }
    gateCv.notify_all();
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.empty());
    REQUIRE(scheduler.currentEpoch() == newEpoch);
    REQUIRE(scheduler.stats().canceled >= 1);
}

TEST_CASE("Far terrain scheduler retains only the current view",
          "[render][far-terrain][scheduler][cancellation][cache]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    const std::array<FarTerrainKey, 4> keys{{
        {0, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::SIXTEEN},
        {2, 0, FarTerrainStep::SIXTEEN},
        {3, 0, FarTerrainStep::SIXTEEN},
    }};
    for (const FarTerrainKey& key : keys)
        REQUIRE(scheduler.enqueue(key));
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == keys.size());

    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted{keys[1], keys[3]};
    scheduler.retainWanted(wanted);
    const FarTerrainSchedulerStats retained = scheduler.stats();
    REQUIRE(retained.cacheEntries == wanted.size());
    REQUIRE(retained.completed == wanted.size());
    REQUIRE(scheduler.findCached(keys[1]));
    REQUIRE(scheduler.findCached(keys[3]));
    REQUIRE_FALSE(scheduler.findCached(keys[0]));
    REQUIRE_FALSE(scheduler.findCached(keys[2]));
    REQUIRE_FALSE(scheduler.enqueue(keys[0]));
}

TEST_CASE("Far terrain scheduler cancels obsolete view work",
          "[render][far-terrain][scheduler][cancellation][priority]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    size_t enteredWorkers = 0;
    bool released = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto geometry = source.geometry;
    source.geometry = [&](int64_t x, int64_t z) {
        {
            std::unique_lock lock(gateMutex);
            ++enteredWorkers;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return released; });
        }
        return geometry(x, z);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    std::array<FarTerrainKey, 8> keys{};
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = {static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN};
        REQUIRE(scheduler.enqueue(keys[index]));
    }

    bool allWorkersEntered = false;
    {
        std::unique_lock lock(gateMutex);
        allWorkersEntered = gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            return enteredWorkers >= FarTerrainScheduler::WORKER_COUNT;
        });
    }
    scheduler.retainWanted({keys[0]});
    {
        std::lock_guard lock(gateMutex);
        released = true;
    }
    gateCv.notify_all();
    REQUIRE(allWorkersEntered);

    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const FarTerrainSchedulerStats stats = scheduler.stats();
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.built == 1);
    REQUIRE(stats.canceled == keys.size() - 1);
    REQUIRE(stats.cacheEntries == 1);
    REQUIRE(stats.completed == 1);
    REQUIRE(scheduler.findCached(keys[0]));
}

// ============================================================================
// Greedy Mesher Tests
// ============================================================================

TEST_CASE("Mesher: empty chunk produces no geometry", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // All AIR — no solid blocks

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    REQUIRE(output.vertices.empty());
    REQUIRE(output.indices.empty());
    REQUIRE(output.vertices.capacity() == 0);
    REQUIRE(output.indices.capacity() == 0);
}

TEST_CASE("Mesher: single block produces 6 faces", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 6 faces × 4 vertices = 24 vertices
    REQUIRE(output.vertices.size() == 24);
    // 6 faces × 2 triangles × 3 indices = 36 indices
    REQUIRE(output.indices.size() == 36);
}

TEST_CASE("Mesher: opaque faces use outward winding in all six directions",
          "[render][mesher][winding]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.opaqueIndexCount == 36);

    const auto vertexPosition = [](const Vertex& vertex) {
        return Vec3{static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                    static_cast<float>(vertex.pz)};
    };
    const auto expectedNormal = [](FaceNormal face) {
        switch (face) {
            case FaceNormal::PLUS_X:
                return Vec3{1.f, 0.f, 0.f};
            case FaceNormal::MINUS_X:
                return Vec3{-1.f, 0.f, 0.f};
            case FaceNormal::PLUS_Z:
                return Vec3{0.f, 0.f, 1.f};
            case FaceNormal::MINUS_Z:
                return Vec3{0.f, 0.f, -1.f};
            case FaceNormal::PLUS_Y:
                return Vec3{0.f, 1.f, 0.f};
            case FaceNormal::MINUS_Y:
                return Vec3{0.f, -1.f, 0.f};
            case FaceNormal::CROSS:
                return Vec3{};
        }
        return Vec3{};
    };

    std::array<bool, 6> found{};
    for (size_t offset = 0; offset < output.opaqueIndexCount; offset += 6) {
        const Vertex& first = output.vertices[output.indices[offset]];
        const Vertex& second = output.vertices[output.indices[offset + 1]];
        const Vertex& third = output.vertices[output.indices[offset + 2]];
        const FaceNormal face = unpackFace(first.faceAttr);
        REQUIRE(face != FaceNormal::CROSS);
        const Vec3 normal = (vertexPosition(second) - vertexPosition(first))
                                .cross(vertexPosition(third) - vertexPosition(first));
        REQUIRE(normal.dot(expectedNormal(face)) > 0.f);
        found[static_cast<size_t>(face)] = true;
    }
    for (bool faceFound : found)
        REQUIRE(faceFound);
}

TEST_CASE("Mesher: a solid cuboid greedily reduces every opaque direction",
          "[render][mesher][greedy]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    for (int y = 3; y < 10; ++y) {
        for (int z = 4; z < 10; ++z) {
            for (int x = 2; x < 7; ++x) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
        }
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.opaqueIndexCount == 36);

    std::array<size_t, 6> verticesPerFace{};
    for (const Vertex& vertex : output.vertices) {
        const FaceNormal face = unpackFace(vertex.faceAttr);
        REQUIRE(face != FaceNormal::CROSS);
        ++verticesPerFace[static_cast<size_t>(face)];
    }
    for (size_t count : verticesPerFace)
        REQUIRE(count == 4);
}

TEST_CASE("Mesher: reused scratch produces byte-identical output",
          "[render][mesher][determinism]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    for (int z = 3; z < 12; ++z) {
        for (int x = 2; x < 11; ++x) {
            const int height = 4 + (x * 3 + z * 5) % 6;
            for (int y = 2; y < height; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] =
                    y + 1 == height ? BlockType::GRASS : BlockType::STONE;
            }
        }
    }
    snapshot.blocks[MeshSnapshot::index(1, 10, 1)] = BlockType::TALL_GRASS;
    snapshot.blocks[MeshSnapshot::index(13, 8, 13)] = BlockType::LILY_PAD;
    snapshot.blocks[MeshSnapshot::index(14, 3, 14)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(14, 3, 14)] = FluidState::flowing(5).packed();

    MeshScratch scratch;
    const MeshOutput first = LODMesher::buildMesh(snapshot, scratch);
    scratch.faceKeys.fill(0xFFFFU);
    scratch.skyHeight.fill(0xFFU);
    const MeshOutput second = LODMesher::buildMesh(snapshot, scratch);

    REQUIRE(first.opaqueIndexCount == second.opaqueIndexCount);
    REQUIRE(first.indices == second.indices);
    REQUIRE(first.vertices.size() == second.vertices.size());
    REQUIRE(std::memcmp(first.vertices.data(), second.vertices.data(),
                        first.vertices.size() * sizeof(Vertex)) == 0);
}

TEST_CASE("Mesher: 2x2 flat merges top face", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 2x2 square of STONE at local y=8
    chunk.setBlock(0, 8, 0, BlockType::STONE);
    chunk.setBlock(1, 8, 0, BlockType::STONE);
    chunk.setBlock(0, 8, 1, BlockType::STONE);
    chunk.setBlock(1, 8, 1, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Without greedy merge: 4 top faces = 16 vertices
    // With greedy merge: 1 top face = 4 vertices
    // Total expected:
    //   Top (+Y): 1 merged quad = 4 vertices, 6 indices
    //   Bottom (-Y): 1 merged quad = 4 vertices, 6 indices
    //   +X face (right side of x=1 column): 2 quads (z=0 and z=1) = 8 vertices, 12 indices
    //   -X face (left side of x=0 column): 2 quads = 8 vertices, 12 indices
    //   +Z face (front of z=1 row): 2 quads = 8 vertices, 12 indices
    //   -Z face (back of z=0 row): 2 quads = 8 vertices, 12 indices
    // Total: 40 vertices, 60 indices
    //
    // But +X and -X faces can also merge vertically since all 4 blocks are at same Y
    // +X: blocks at (1,8,0) and (1,8,1) both have +X exposed with the same type
    //   They're adjacent in Z direction, so they merge into 1 quad: 4 vertices, 6 indices
    // Same for -X, +Z, -Z
    //
    // Total: 6 faces × 4 vertices = 24 vertices, 6 × 6 = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify the top face is a single quad (first 4 vertices)
    // All 4 top-face vertices should decode to FaceNormal::PLUS_Y
    bool foundTopQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        if (unpackFace(output.vertices[i].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 1].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 2].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 3].faceAttr) == FaceNormal::PLUS_Y) {
            foundTopQuad = true;
            break;
        }
    }
    REQUIRE(foundTopQuad);
}

TEST_CASE("Mesher: flora emits a contained cross of two quads", "[render][mesher][flora]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::GRASS);
    chunk.setBlock(8, 9, 8, BlockType::TALL_GRASS);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Grass cube: 6 quads. Flora shares four vertices per diagonal while
    // indexing both windings so it remains visible with back-face culling.
    REQUIRE(output.vertices.size() == 32);
    REQUIRE(output.indices.size() == 60);

    int crossVerts = 0;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::CROSS)
            continue;
        ++crossVerts;
        REQUIRE(unpackTextureLayer(v.faceAttr) == static_cast<uint8_t>(BlockType::TALL_GRASS));
        // The coordinate-hashed pose remains contained in its anchor cell and
        // spans the full cell height.
        float px = static_cast<float>(v.px);
        float py = static_cast<float>(v.py);
        float pz = static_cast<float>(v.pz);
        REQUIRE(px >= 8.0f);
        REQUIRE(px <= 9.0f);
        REQUIRE(pz >= 8.0f);
        REQUIRE(pz <= 9.0f);
        REQUIRE((py == 9.f || py == 10.f));
    }
    REQUIRE(crossVerts == 8);

    for (size_t offset : {size_t{36}, size_t{48}}) {
        REQUIRE(output.indices[offset + 6] == output.indices[offset]);
        REQUIRE(output.indices[offset + 7] == output.indices[offset + 2]);
        REQUIRE(output.indices[offset + 8] == output.indices[offset + 1]);
        REQUIRE(output.indices[offset + 9] == output.indices[offset + 3]);
        REQUIRE(output.indices[offset + 10] == output.indices[offset + 5]);
        REQUIRE(output.indices[offset + 11] == output.indices[offset + 4]);
    }
}

TEST_CASE("Mesher: dense flora poses vary deterministically across the world lattice",
          "[render][mesher][flora][determinism]") {
    // This is the cubic column containing the seed-42 dense-flora regression
    // scene. Populate every surface cell so repeated poses would form the
    // conspicuous rows seen in the aerial capture.
    Chunk chunk(ChunkPos{-1553, 4, -601});
    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            chunk.setBlock(x, 8, z, BlockType::GRASS);
            chunk.setBlock(x, 9, z, BlockType::TALL_GRASS);
        }
    }

    LODMesher mesher;
    const MeshOutput first = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    const MeshOutput second = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    REQUIRE(first.indices == second.indices);
    REQUIRE(first.vertices.size() == second.vertices.size());
    REQUIRE(std::memcmp(first.vertices.data(), second.vertices.data(),
                        first.vertices.size() * sizeof(Vertex)) == 0);

    std::vector<const Vertex*> floraVertices;
    for (const Vertex& vertex : first.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::CROSS) {
            floraVertices.push_back(&vertex);
        }
    }
    REQUIRE(floraVertices.size() == CHUNK_WIDTH * CHUNK_DEPTH * 8);

    std::set<std::array<int, 4>> poses;
    for (size_t plant = 0; plant < floraVertices.size() / 8; ++plant) {
        const Vertex& firstBottom = *floraVertices[plant * 8];
        const Vertex& secondBottom = *floraVertices[plant * 8 + 1];
        const int localX = static_cast<int>(plant % CHUNK_WIDTH);
        const int localZ = static_cast<int>(plant / CHUNK_WIDTH);
        const float centerX =
            (static_cast<float>(firstBottom.px) + static_cast<float>(secondBottom.px)) * 0.5F;
        const float centerZ =
            (static_cast<float>(firstBottom.pz) + static_cast<float>(secondBottom.pz)) * 0.5F;
        poses.insert({
            static_cast<int>(std::lround((centerX - static_cast<float>(localX) - 0.5F) * 32.0F)),
            static_cast<int>(std::lround((centerZ - static_cast<float>(localZ) - 0.5F) * 32.0F)),
            static_cast<int>(std::lround(
                (static_cast<float>(secondBottom.px) - static_cast<float>(firstBottom.px)) * 8.0F)),
            static_cast<int>(std::lround(
                (static_cast<float>(secondBottom.pz) - static_cast<float>(firstBottom.pz)) * 8.0F)),
        });
    }
    REQUIRE(poses.size() >= 12);
}

TEST_CASE("Mesher: flat flora emits explicit front and back winding",
          "[render][mesher][flora][winding]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LILY_PAD);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    REQUIRE(output.vertices.size() == 4);
    REQUIRE(output.opaqueIndexCount == 12);
    REQUIRE(output.indices == std::vector<uint32_t>{0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2});
}

TEST_CASE("Mesher: flora does not break greedy merging of the ground", "[render][mesher][flora]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 2x2 grass floor with one flower on top: the floor's +Y face must still
    // merge into a single quad (flora neither occludes nor casts shade)
    for (int z = 0; z < 2; ++z)
        for (int x = 0; x < 2; ++x)
            chunk.setBlock(x, 8, z, BlockType::GRASS);
    chunk.setBlock(0, 9, 0, BlockType::FLOWER_RED);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 2x2 slab = 24 vertices (all faces merged) + 8 flora vertices
    REQUIRE(output.vertices.size() == 32);
}

TEST_CASE("Mesher: water surfaces land in the water section", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // Stone floor with one water block on top: the water's top face (under
    // air) and four sides are water-section; the floor's faces are opaque.
    chunk.setBlock(8, 4, 8, BlockType::STONE);
    chunk.setBlock(8, 5, 8, BlockType::WATER);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Opaque: stone cube 6 faces (water doesn't hide the +Y face) = 36 idx.
    // Water: top + 4 sides = 5 quads = 30 indices (bottom hidden by stone).
    REQUIRE(output.opaqueIndexCount == 36);
    REQUIRE(output.indices.size() == 66);

    // The water top surface sits 0.125 below the cell top (fp16-exact)
    bool foundDroppedTop = false;
    for (const Vertex& v : output.vertices) {
        if (static_cast<float>(v.py) == 5.875f)
            foundDroppedTop = true;
    }
    REQUIRE(foundDroppedTop);
}

TEST_CASE("Mesher: interior water-water faces are culled", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 2x2x2 water cube on a stone slab
    for (int z = 4; z < 6; ++z)
        for (int x = 4; x < 6; ++x) {
            chunk.setBlock(x, 3, z, BlockType::STONE);
            chunk.setBlock(x, 4, z, BlockType::WATER);
            chunk.setBlock(x, 5, z, BlockType::WATER);
        }

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Water section: greedy-merged top (1 quad) + 4 merged side walls
    // (2 wide × 2 tall each → 1 quad per direction) = 5 quads = 30 indices
    uint32_t waterIndexCount =
        static_cast<uint32_t>(output.indices.size()) - output.opaqueIndexCount;
    REQUIRE(waterIndexCount == 30);
}

TEST_CASE("Snapshot mesher uses runtime water levels and falling metadata",
          "[render][mesher][water]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::flowing(7).packed();

    MeshScratch scratch;
    MeshOutput shallow = LODMesher::buildMesh(snapshot, scratch);
    bool foundShallowTop = false;
    for (const Vertex& vertex : shallow.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
            static_cast<float>(vertex.py) == 8.125f) {
            foundShallowTop = true;
        }
    }
    REQUIRE(foundShallowTop);

    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::falling(3).packed();
    MeshOutput falling = LODMesher::buildMesh(snapshot, scratch);
    bool foundFallingFace = false;
    for (const Vertex& vertex : falling.vertices) {
        if (unpackFluidFalling(vertex.faceAttr))
            foundFallingFace = true;
    }
    REQUIRE(foundFallingFace);
}

TEST_CASE("Snapshot water sides are exclusive to falling columns",
          "[render][mesher][water][shoreline]") {
    constexpr std::array<std::pair<int, int>, 4> horizontalEdges{{
        {0, 8},
        {CHUNK_EDGE - 1, 8},
        {8, 0},
        {8, CHUNK_EDGE - 1},
    }};
    MeshScratch scratch;
    for (const auto& [x, z] : horizontalEdges) {
        MeshSnapshot source;
        source.clear();
        source.blocks[MeshSnapshot::index(x, 7, z)] = BlockType::STONE;
        source.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::WATER;
        const MeshOutput output = LODMesher::buildMesh(source, scratch);
        for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
            REQUIRE(unpackFace(output.vertices[output.indices[offset]].faceAttr) ==
                    FaceNormal::PLUS_Y);
        }
    }

    MeshSnapshot waterfall;
    waterfall.clear();
    waterfall.blocks[MeshSnapshot::index(8, 7, 8)] = BlockType::STONE;
    waterfall.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::WATER;
    waterfall.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::falling(3).packed();
    const MeshOutput falling = LODMesher::buildMesh(waterfall, scratch);
    std::array<bool, 4> sideFound{};
    for (size_t offset = falling.opaqueIndexCount; offset < falling.indices.size(); ++offset) {
        const FaceNormal face = unpackFace(falling.vertices[falling.indices[offset]].faceAttr);
        if (face == FaceNormal::MINUS_X)
            sideFound[0] = true;
        if (face == FaceNormal::PLUS_X)
            sideFound[1] = true;
        if (face == FaceNormal::MINUS_Z)
            sideFound[2] = true;
        if (face == FaceNormal::PLUS_Z)
            sideFound[3] = true;
    }
    REQUIRE(std::all_of(sideFound.begin(), sideFound.end(), [](bool found) { return found; }));
}

TEST_CASE("Generated incised rivers mesh continuously across exact cube faces",
          "[render][mesher][water][river][seam][regression]") {
    ChunkGenerator generator(42);
    constexpr int64_t RIVER_X = -12'289;
    constexpr int64_t RIVER_Z = 2'653;
    const std::array<worldgen::SurfaceSample, 4> riverSamples = {
        generator.sampleExactSurface(RIVER_X, RIVER_Z),
        generator.sampleExactSurface(RIVER_X + 1, RIVER_Z),
        generator.sampleExactSurface(RIVER_X, RIVER_Z + 1),
        generator.sampleExactSurface(RIVER_X + 1, RIVER_Z + 1),
    };
    const int WATER_Y = static_cast<int>(std::ceil(riverSamples.front().waterSurface)) - 1;
    for (const worldgen::SurfaceSample& sample : riverSamples) {
        REQUIRE(sample.hydrology.river);
        REQUIRE_FALSE(sample.hydrology.lake);
        REQUIRE_FALSE(sample.hydrology.waterfall);
        REQUIRE(sample.waterSurface > sample.terrainHeight);
        REQUIRE(static_cast<int>(std::ceil(sample.waterSurface)) - 1 == WATER_Y);
    }
    const ChunkPos center{Chunk::worldToChunk(RIVER_X), Chunk::worldToChunkY(WATER_Y),
                          Chunk::worldToChunk(RIVER_Z)};
    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    auto cubeAt = [&](ChunkPos position) -> Chunk& {
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return *found->second;
    };

    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = center;
    for (int y = -1; y <= CHUNK_EDGE; ++y) {
        for (int z = -1; z <= CHUNK_EDGE; ++z) {
            for (int x = -1; x <= CHUNK_EDGE; ++x) {
                const int64_t worldX = center.x * CHUNK_EDGE + x;
                const int worldY = center.y * CHUNK_EDGE + y;
                const int64_t worldZ = center.z * CHUNK_EDGE + z;
                Chunk& cube = cubeAt({Chunk::worldToChunk(worldX), Chunk::worldToChunkY(worldY),
                                      Chunk::worldToChunk(worldZ)});
                const int index = MeshSnapshot::index(x, y, z);
                snapshot.blocks[index] =
                    cube.getBlock(Chunk::worldToLocal(worldX), Chunk::worldToLocalY(worldY),
                                  Chunk::worldToLocal(worldZ));
                snapshot.fluidStates[index] =
                    cube.getFluidState(Chunk::worldToLocal(worldX), Chunk::worldToLocalY(worldY),
                                       Chunk::worldToLocal(worldZ))
                        .packed();
            }
        }
    }

    const int WATER_LOCAL_Y = Chunk::worldToLocalY(WATER_Y);
    constexpr int RIVER_LOCAL_Z = static_cast<int>(RIVER_Z - 2'640);
    REQUIRE(snapshot.at(15, WATER_LOCAL_Y, RIVER_LOCAL_Z) == BlockType::WATER);
    REQUIRE(snapshot.at(16, WATER_LOCAL_Y, RIVER_LOCAL_Z) == BlockType::WATER);
    REQUIRE(snapshot.at(15, WATER_LOCAL_Y, RIVER_LOCAL_Z + 1) == BlockType::WATER);
    REQUIRE(snapshot.at(16, WATER_LOCAL_Y, RIVER_LOCAL_Z + 1) == BlockType::WATER);
    REQUIRE_FALSE(snapshot.fluidAt(15, WATER_LOCAL_Y, RIVER_LOCAL_Z).isFalling());
    REQUIRE_FALSE(snapshot.fluidAt(16, WATER_LOCAL_Y, RIVER_LOCAL_Z).isFalling());

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.indices.size() > output.opaqueIndexCount);
    bool foundSurfaceAtSharedFace = false;
    for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
        const Vertex& vertex = output.vertices[output.indices[offset]];
        REQUIRE_FALSE(unpackFluidFalling(vertex.faceAttr));
        const FaceNormal face = unpackFace(vertex.faceAttr);
        if (face == FaceNormal::PLUS_Y && vertex.px >= 15 && vertex.pz >= RIVER_LOCAL_Z &&
            vertex.pz <= RIVER_LOCAL_Z + 1) {
            foundSurfaceAtSharedFace = true;
        }
        REQUIRE_FALSE((face == FaceNormal::PLUS_X && vertex.px == CHUNK_EDGE &&
                       vertex.pz >= RIVER_LOCAL_Z && vertex.pz <= RIVER_LOCAL_Z + 1));
    }
    REQUIRE(foundSurfaceAtSharedFace);
}

TEST_CASE("Mesher: lava renders as an opaque cube section", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 0, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 6 faces, all in the opaque section; nothing in the water section
    REQUIRE(output.opaqueIndexCount == 36);
    REQUIRE(output.indices.size() == 36);
}

// ============================================================================
// Neighbor-aware (snapshot) meshing — chunk border correctness
// ============================================================================

TEST_CASE("Snapshot mesher: boundary faces follow real neighbor blocks",
          "[render][mesher][border]") {
    MeshSnapshot snapshot;
    snapshot.resize();
    // One stone block on the +X border of the chunk
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;

    // Case 1: neighbor cell across the border is AIR → the +X boundary face
    // must exist (the old mesher skipped the boundary layer: a hole)
    {
        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 24); // full cube
        bool plusXAt16 = false;
        for (const Vertex& v : output.vertices) {
            if (unpackFace(v.faceAttr) == FaceNormal::PLUS_X && static_cast<float>(v.px) == 16.f)
                plusXAt16 = true;
        }
        REQUIRE(plusXAt16);
    }

    // Case 2: neighbor cell solid → the boundary face is culled (the old
    // mesher's -X pass always emitted a hidden wall from the other side)
    {
        snapshot.blocks[MeshSnapshot::index(16, 8, 8)] = BlockType::STONE;
        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 20); // cube minus the shared face
        for (const Vertex& v : output.vertices) {
            REQUIRE(!(unpackFace(v.faceAttr) == FaceNormal::PLUS_X &&
                      static_cast<float>(v.px) == 16.f));
        }
    }
}

TEST_CASE("Snapshot mesher: -X border wall culled against a solid neighbor",
          "[render][mesher][border]") {
    MeshSnapshot snapshot;
    snapshot.resize();
    snapshot.blocks[MeshSnapshot::index(0, 8, 8)] = BlockType::STONE;
    // Solid neighbor wall behind it (x = -1)
    snapshot.blocks[MeshSnapshot::index(-1, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;
    MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.vertices.size() == 20);
    for (const Vertex& v : output.vertices) {
        REQUIRE(
            !(unpackFace(v.faceAttr) == FaceNormal::MINUS_X && static_cast<float>(v.px) == 0.f));
    }
}

TEST_CASE("Snapshot mesher emits exposed faces on all six cube boundaries",
          "[render][mesher][border]") {
    struct BoundaryCase {
        int x;
        int y;
        int z;
        FaceNormal exposedFace;
    };
    constexpr std::array<BoundaryCase, 6> cases{{
        {0, 8, 8, FaceNormal::MINUS_X},
        {15, 8, 8, FaceNormal::PLUS_X},
        {8, 0, 8, FaceNormal::MINUS_Y},
        {8, 15, 8, FaceNormal::PLUS_Y},
        {8, 8, 0, FaceNormal::MINUS_Z},
        {8, 8, 15, FaceNormal::PLUS_Z},
    }};

    MeshScratch scratch;
    for (const BoundaryCase& test : cases) {
        MeshSnapshot snapshot;
        snapshot.clear();
        snapshot.blocks[MeshSnapshot::index(test.x, test.y, test.z)] = BlockType::STONE;
        const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 24);
        REQUIRE(std::any_of(output.vertices.begin(), output.vertices.end(), [&](const Vertex& v) {
            return unpackFace(v.faceAttr) == test.exposedFace;
        }));
    }
}

TEST_CASE("Snapshot mesher culls shared faces on all six cube boundaries",
          "[render][mesher][border]") {
    struct BoundaryCase {
        int blockX;
        int blockY;
        int blockZ;
        int neighborX;
        int neighborY;
        int neighborZ;
        FaceNormal hiddenFace;
    };
    constexpr std::array<BoundaryCase, 6> cases{{
        {0, 8, 8, -1, 8, 8, FaceNormal::MINUS_X},
        {15, 8, 8, 16, 8, 8, FaceNormal::PLUS_X},
        {8, 0, 8, 8, -1, 8, FaceNormal::MINUS_Y},
        {8, 15, 8, 8, 16, 8, FaceNormal::PLUS_Y},
        {8, 8, 0, 8, 8, -1, FaceNormal::MINUS_Z},
        {8, 8, 15, 8, 8, 16, FaceNormal::PLUS_Z},
    }};

    MeshScratch scratch;
    for (const BoundaryCase& test : cases) {
        MeshSnapshot snapshot;
        snapshot.clear();
        snapshot.blocks[MeshSnapshot::index(test.blockX, test.blockY, test.blockZ)] =
            BlockType::STONE;
        snapshot.blocks[MeshSnapshot::index(test.neighborX, test.neighborY, test.neighborZ)] =
            BlockType::STONE;

        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 20);
        for (const Vertex& vertex : output.vertices) {
            REQUIRE(unpackFace(vertex.faceAttr) != test.hiddenFace);
        }
    }
}

TEST_CASE("MegaBuffer free-list coalescing is bounds-safe and lossless", "[render][megabuffer]") {
    using Region = std::pair<uint64_t, uint64_t>;

    // Regression: a single-entry list made the old compaction write one
    // element past the vector's end — slow heap corruption that surfaced as
    // buzzing audio and malloc traps minutes into a session.
    std::vector<Region> single = {{256, 512}};
    MegaBuffer::coalesceFreeList(single);
    REQUIRE(single == std::vector<Region>{{256, 512}});

    // Adjacent regions merge (any input order)…
    std::vector<Region> adjacent = {{768, 256}, {256, 512}};
    MegaBuffer::coalesceFreeList(adjacent);
    REQUIRE(adjacent == std::vector<Region>{{256, 768}});

    // …gaps survive, and the LAST region is kept (the old code erased it)
    std::vector<Region> gapped = {{0, 256}, {512, 256}, {2048, 256}};
    MegaBuffer::coalesceFreeList(gapped);
    REQUIRE(gapped == std::vector<Region>{{0, 256}, {512, 256}, {2048, 256}});

    // Chain of three merges into one
    std::vector<Region> chain = {{512, 256}, {0, 512}, {768, 1024}};
    MegaBuffer::coalesceFreeList(chain);
    REQUIRE(chain == std::vector<Region>{{0, 1792}});

    std::vector<Region> empty;
    MegaBuffer::coalesceFreeList(empty);
    REQUIRE(empty.empty());
}

TEST_CASE("MeshScheduler: builds off-thread with version stamps", "[render][scheduler]") {
    World world(42, 2);
    constexpr ChunkPos center{0, 4, 0};
    world.getChunk(center);
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                if (offsetX == 0 && offsetY == 0 && offsetZ == 0)
                    continue;
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    for (int z = 7; z <= 9; ++z) {
        for (int y = 71; y <= 73; ++y) {
            for (int x = 7; x <= 9; ++x) {
                world.setBlock(x, y, z, BlockType::AIR);
            }
        }
    }
    world.setBlock(8, 72, 8, BlockType::STONE);

    MeshScheduler scheduler(world, 1);
    REQUIRE(scheduler.enqueue(center));

    std::vector<MeshResult> results;
    for (int i = 0; i < 500 && results.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        scheduler.drainCompleted(results);
    }
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].pos == center);
    REQUIRE(results[0].snapshotOk);
    REQUIRE(results[0].builtVersion == world.getChunk(center)->version.load());
    REQUIRE(!results[0].mesh.vertices.empty());

    // A chunk without generated neighbors reports the failed snapshot
    // instead of blocking (the renderer retries once the frontier catches up)
    REQUIRE(scheduler.enqueue(ChunkPos{40, 4, 40}));
    results.clear();
    for (int i = 0; i < 500 && results.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        scheduler.drainCompleted(results);
    }
    REQUIRE(results.size() == 1);
    REQUIRE(!results[0].snapshotOk);

    // Shutdown is idempotent and refuses further work
    scheduler.shutdown();
    scheduler.shutdown();
    REQUIRE(!scheduler.enqueue(center));
}

TEST_CASE("World snapshotForMeshing seals missing neighbors until the real halo arrives",
          "[world][mesher][border][streaming]") {
    World world(4242, 2);
    MeshSnapshot snapshot;
    constexpr ChunkPos center{0, 4, 0};

    // Nothing generated yet
    REQUIRE(!world.snapshotForMeshing(center, snapshot));

    // Plans are immutable prerequisites. An absent cube follows its generated
    // terrain silhouette: conservatively solid below the surface and air
    // above it, instead of presenting a full dark face while streaming.
    world.getChunk(center);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({center.x + offsetX, center.z + offsetZ}));
        }
    }
    REQUIRE(world.snapshotForMeshing(center, snapshot));
    REQUIRE(snapshot.missingNeighborFaces == 0x3FU);
    const int32_t probeWorldY = center.y * CHUNK_EDGE + 8;
    const int32_t generatedCutoff = snapshot.generatedSurfaceCutoffAt(CHUNK_EDGE, 8);
    REQUIRE(generatedCutoff != MeshSnapshot::SKY_CUTOFF_UNKNOWN);
    REQUIRE(snapshot.at(CHUNK_EDGE, 8, 8) ==
            (probeWorldY < generatedCutoff ? BlockType::BEDROCK : BlockType::AIR));
    world.markChunkMeshed(center);
    REQUIRE_FALSE(world.getChunk(center)->needsMeshUpdate);
    REQUIRE(world.getChunk({1, 4, 0}));
    REQUIRE(world.getChunk(center)->needsMeshUpdate);

    // Edge and corner cells affect ambient occlusion and fluid corner heights,
    // so a complete 3x3x3 halo replaces every conservative placeholder.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    auto plusX = world.getChunk({1, 4, 0});
    plusX->setBlock(0, 7, 8, BlockType::WATER);
    plusX->setFluidState(0, 7, 8, FluidState::flowing(5));
    plusX->setBlockLight(0, 7, 8, 9);
    REQUIRE(world.snapshotForMeshing(center, snapshot));
    REQUIRE(snapshot.missingNeighborFaces == 0);

    // Every padded face carries its neighbor's real border cells.
    auto minusX = world.getChunk({-1, 4, 0});
    auto plusY = world.getChunk({0, 5, 0});
    auto minusY = world.getChunk({0, 3, 0});
    auto plusZ = world.getChunk({0, 4, 1});
    auto minusZ = world.getChunk({0, 4, -1});
    for (int coordinate = 0; coordinate < CHUNK_EDGE; coordinate += 5) {
        REQUIRE(snapshot.at(CHUNK_EDGE, coordinate, 8) == plusX->getBlock(0, coordinate, 8));
        REQUIRE(snapshot.at(-1, coordinate, 8) == minusX->getBlock(CHUNK_EDGE - 1, coordinate, 8));
        REQUIRE(snapshot.at(coordinate, CHUNK_EDGE, 8) == plusY->getBlock(coordinate, 0, 8));
        REQUIRE(snapshot.at(coordinate, -1, 8) == minusY->getBlock(coordinate, CHUNK_EDGE - 1, 8));
        REQUIRE(snapshot.at(coordinate, 8, CHUNK_EDGE) == plusZ->getBlock(coordinate, 8, 0));
        REQUIRE(snapshot.at(coordinate, 8, -1) == minusZ->getBlock(coordinate, 8, CHUNK_EDGE - 1));
    }
    REQUIRE(snapshot.at(CHUNK_EDGE, 7, 8) == BlockType::WATER);
    REQUIRE(snapshot.fluidAt(CHUNK_EDGE, 7, 8) == FluidState::flowing(5));
    REQUIRE(snapshot.lightAt(CHUNK_EDGE, 7, 8) == 9);
    REQUIRE(snapshot.skyCutoffAt(8, 8) != MeshSnapshot::SKY_CUTOFF_UNKNOWN);
}

TEST_CASE("Missing surface halos stay lit while underground openings remain dark",
          "[world][mesher][border][streaming][surface][underground]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    // The loaded column ends at world Y=67 while the arriving uphill column
    // continues through Y=71. Its four-block exposed silhouette should use a
    // normally lit terrain material. The four blocks below the local surface
    // still represent a cave opening and must remain sealed and dark.
    for (int z = 0; z < CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = BlockType::GRASS;
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t darkVertices = 0;
    size_t litSurfaceVertices = 0;
    float highestCapY = 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        highestCapY = std::max(highestCapY, static_cast<float>(vertex.py));
        if (unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::BEDROCK)) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
            ++darkVertices;
        } else if (unpackTextureLayer(vertex.faceAttr) == TEXTURE_LAYER_GRASS_SIDE) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
            ++litSurfaceVertices;
        }
    }
    REQUIRE(darkVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(litSurfaceVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(highestCapY == 8.0F);
    REQUIRE(darkVertices + litSurfaceVertices < CHUNK_EDGE * CHUNK_EDGE * 4);
}

TEST_CASE("World setBlock marks boundary neighbors for remeshing", "[world][mesher][border]") {
    World world(7, 2);
    constexpr int32_t sectionY = 6;
    world.getChunk({0, sectionY, 0});
    world.getChunk({-1, sectionY, 0});
    world.getChunk({0, sectionY, -1});
    auto self = world.getChunk({0, sectionY, 0});
    auto negX = world.getChunk({-1, sectionY, 0});
    auto negZ = world.getChunk({0, sectionY, -1});

    self->needsMeshUpdate = false;
    negX->needsMeshUpdate = false;
    negZ->needsMeshUpdate = false;

    // Interior edit: only the chunk itself
    world.setBlock(8, 100, 8, BlockType::STONE);
    REQUIRE(self->needsMeshUpdate);
    REQUIRE(!negX->needsMeshUpdate);

    // Boundary edit at local x == 0: the -X neighbor re-meshes too
    self->needsMeshUpdate = false;
    world.setBlock(0, 100, 8, BlockType::STONE);
    REQUIRE(self->needsMeshUpdate);
    REQUIRE(negX->needsMeshUpdate);
    REQUIRE(!negZ->needsMeshUpdate);
}

TEST_CASE("Mesher: flora is skipped at coarse LODs", "[render][mesher][flora]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    for (int z = 0; z < CHUNK_DEPTH; ++z)
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            chunk.setBlock(x, 8, z, BlockType::GRASS);
            chunk.setBlock(x, 9, z, BlockType::TALL_GRASS);
        }

    LODMesher mesher;
    MeshOutput medium = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::MEDIUM));
    for (const Vertex& v : medium.vertices) {
        REQUIRE(unpackFace(v.faceAttr) != FaceNormal::CROSS);
        REQUIRE(unpackTextureLayer(v.faceAttr) != static_cast<uint8_t>(BlockType::TALL_GRASS));
    }
}

TEST_CASE("Mesher: vertical column merges side faces", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 4-block tall column of STONE at local y=6 through y=9
    chunk.setBlock(8, 6, 8, BlockType::STONE);
    chunk.setBlock(8, 7, 8, BlockType::STONE);
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    chunk.setBlock(8, 9, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Top and bottom each form one quad. Every side forms one quad spanning
    // local y=6 through y=10, for 24 vertices and 36 indices total.

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify side faces span the full column height
    // Each side quad should span four local blocks.
    bool foundSideQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        uint8_t ni = static_cast<uint8_t>(unpackFace(output.vertices[i].faceAttr));
        // Check side faces (face indices 0-3)
        if (ni <= 3 && static_cast<uint8_t>(unpackFace(output.vertices[i + 1].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 2].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 3].faceAttr)) == ni) {
            // Check that the quad spans 4 units in Y
            float minY = std::min({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            float maxY = std::max({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            if (maxY - minY >= 3.5f) { // height=4, account for float16 precision
                foundSideQuad = true;
                break;
            }
        }
    }
    REQUIRE(foundSideQuad);
}

TEST_CASE("Mesher: produces mesh without side effects", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    chunk.needsMeshUpdate = true;
    REQUIRE(chunk.needsMeshUpdate == true);

    LODMesher mesher;
    MeshOutput mesh = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // buildMesh is pure — it does not modify the chunk
    REQUIRE(chunk.needsMeshUpdate == true);
    REQUIRE(mesh.vertices.size() > 0u);
    REQUIRE(mesh.indices.size() > 0u);

    // Caller is responsible for marking the chunk as meshed
    chunk.setMeshed(true);
    chunk.needsMeshUpdate = false;
    REQUIRE(chunk.meshed == true);
    REQUIRE(chunk.needsMeshUpdate == false);
}

// ============================================================================
// Block texture mapping tests (no Metal device required)
// ============================================================================

TEST_CASE("Block textures: every block type maps to a valid layer", "[render][textures]") {
    for (int t = 0; t < static_cast<int>(BlockType::COUNT); ++t) {
        for (int f = 0; f < 6; ++f) {
            uint8_t layer = textureLayerFor(static_cast<BlockType>(t), static_cast<FaceNormal>(f));
            REQUIRE(layer < TEXTURE_LAYER_COUNT);
        }
    }
}

TEST_CASE("Block textures: grass uses per-face layers", "[render][textures]") {
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_Y) ==
            static_cast<uint8_t>(BlockType::GRASS));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Y) ==
            static_cast<uint8_t>(BlockType::DIRT));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_X) == TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Z) == TEXTURE_LAYER_GRASS_SIDE);
}

TEST_CASE("Block textures: face attr pack/unpack round-trips", "[render][textures]") {
    for (int f = 0; f < 6; ++f) {
        for (uint8_t layer :
             {uint8_t{0}, uint8_t{7}, TEXTURE_LAYER_GRASS_SIDE, TEXTURE_LAYER_WHITE}) {
            for (uint8_t light : {uint8_t{0}, uint8_t{4}, uint8_t{15}}) {
                for (uint8_t ao : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
                    for (uint8_t blockLight : {uint8_t{0}, uint8_t{9}, uint8_t{15}}) {
                        for (bool emissive : {false, true}) {
                            for (uint8_t sway : {uint8_t{0}, uint8_t{1}, uint8_t{2}}) {
                                uint32_t attr = packFaceAttr(static_cast<FaceNormal>(f), layer,
                                                             light, ao, blockLight, emissive, sway);
                                REQUIRE(unpackFace(attr) == static_cast<FaceNormal>(f));
                                REQUIRE(unpackTextureLayer(attr) == layer);
                                REQUIRE(unpackSkyLight(attr) == light);
                                REQUIRE(unpackCornerAO(attr) == ao);
                                REQUIRE(unpackBlockLight(attr) == blockLight);
                                REQUIRE(unpackEmissive(attr) == emissive);
                                REQUIRE(unpackSway(attr) == sway);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Block textures: fluid metadata does not overlap shared face attributes",
          "[render][textures][water]") {
    constexpr uint8_t skyLight = 7;
    constexpr uint8_t blockLight = 11;
    const uint32_t attr = packFluidFaceAttr(FaceNormal::PLUS_Z, skyLight, 5, true, blockLight);

    REQUIRE(unpackFace(attr) == FaceNormal::PLUS_Z);
    REQUIRE(unpackTextureLayer(attr) == static_cast<uint8_t>(BlockType::WATER));
    REQUIRE(unpackSkyLight(attr) == skyLight);
    REQUIRE(unpackCornerAO(attr) == 3);
    REQUIRE(unpackBlockLight(attr) == blockLight);
    REQUIRE_FALSE(unpackEmissive(attr));
    REQUIRE(unpackSway(attr) == 0);
    REQUIRE(unpackFluidDirection(attr) == 5);
    REQUIRE(unpackFluidFalling(attr));
    REQUIRE((attr & 0x00FFFFFFU) == packFaceAttr(FaceNormal::PLUS_Z,
                                                 static_cast<uint8_t>(BlockType::WATER), skyLight,
                                                 3, blockLight));
}

TEST_CASE("Block definitions expose exhaustive lighting and sway traits",
          "[world][blocks][light]") {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockType type = static_cast<BlockType>(index);
        const BlockDefinition& definition = blockDefinition(type);
        REQUIRE(blockLightEmission(type) == definition.lightEmission);
        REQUIRE(isEmissive(type) == definition.emissive);
        REQUIRE(swayClass(type) == definition.sway);
        REQUIRE(definition.lightEmission <= 15);
        REQUIRE(definition.sway <= 2);
        REQUIRE(definition.emissive == (definition.lightEmission > 0));
    }
    REQUIRE(blockLightEmission(BlockType::LAVA) == 15);
    REQUIRE(isEmissive(BlockType::LAVA));
    REQUIRE(swayClass(BlockType::ACACIA_LEAVES) == 2);
    REQUIRE(swayClass(BlockType::FLOWER_BLUE) == 1);
    REQUIRE(swayClass(BlockType::SUCCULENT) == 0);
}

TEST_CASE("Mesher: tags sway class for flora and leaves", "[render][mesher][sway]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(4, 8, 4, BlockType::STONE);
    chunk.setBlock(4, 9, 4, BlockType::TALL_GRASS);
    chunk.setBlock(8, 8, 8, BlockType::LEAVES);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool sawFlora = false, sawLeaves = false, sawStatic = false;
    for (const Vertex& v : output.vertices) {
        uint8_t layer = unpackTextureLayer(v.faceAttr);
        if (unpackFace(v.faceAttr) == FaceNormal::CROSS) {
            REQUIRE(unpackSway(v.faceAttr) == 1); // flora bends from the root
            sawFlora = true;
        } else if (layer == static_cast<uint8_t>(BlockType::LEAVES)) {
            REQUIRE(unpackSway(v.faceAttr) == 2); // canopy drifts whole-block
            sawLeaves = true;
        } else if (layer == static_cast<uint8_t>(BlockType::STONE)) {
            REQUIRE(unpackSway(v.faceAttr) == 0); // terrain never sways
            sawStatic = true;
        }
    }
    REQUIRE(sawFlora);
    REQUIRE(sawLeaves);
    REQUIRE(sawStatic);
}

TEST_CASE("Mesher: bakes lava block light and the emissive flag", "[render][mesher][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA);   // light source
    chunk.setBlock(10, 8, 8, BlockType::STONE); // a wall two blocks away
    LightEngine::computeSelfLight(chunk);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool foundLitStoneFace = false;
    bool foundEmissiveLava = false;
    for (const Vertex& v : output.vertices) {
        FaceNormal face = unpackFace(v.faceAttr);
        float x = static_cast<float>(v.px);
        // The stone's -X face (plane x = 10) samples the lit air at x = 9.
        if (face == FaceNormal::MINUS_X && x > 9.9f && x < 10.1f) {
            REQUIRE(unpackBlockLight(v.faceAttr) > 0);
            foundLitStoneFace = true;
        }
        if (unpackEmissive(v.faceAttr)) {
            foundEmissiveLava = true; // only lava sets the emissive bit
        }
    }
    REQUIRE(foundLitStoneFace);
    REQUIRE(foundEmissiveLava);
}

TEST_CASE("LightEngine: block light spills through all six cubic faces", "[world][light][cubic]") {
    struct FaceCase {
        size_t neighborIndex;
        std::array<int, 3> neighborCell;
        std::array<int, 3> borderCell;
        std::array<int, 3> inwardCell;
    };
    constexpr std::array<FaceCase, 6> cases{{
        {0, {15, 8, 8}, {0, 8, 8}, {1, 8, 8}},
        {1, {0, 8, 8}, {15, 8, 8}, {14, 8, 8}},
        {2, {8, 8, 15}, {8, 8, 0}, {8, 8, 1}},
        {3, {8, 8, 0}, {8, 8, 15}, {8, 8, 14}},
        {4, {8, 15, 8}, {8, 0, 8}, {8, 1, 8}},
        {5, {8, 0, 8}, {8, 15, 8}, {8, 14, 8}},
    }};

    for (const FaceCase& test : cases) {
        Chunk self(ChunkPos{0, 0, 0});
        Chunk neighbor(ChunkPos{0, 0, 0});
        neighbor.setBlockLight(test.neighborCell[0], test.neighborCell[1], test.neighborCell[2],
                               10);
        LightEngine::FaceNeighbors neighbors{};
        neighbors[test.neighborIndex] = &neighbor;

        REQUIRE(LightEngine::floodChunk(self, neighbors));
        REQUIRE(self.getBlockLight(test.borderCell[0], test.borderCell[1], test.borderCell[2]) ==
                9);
        REQUIRE(self.getBlockLight(test.inwardCell[0], test.inwardCell[1], test.inwardCell[2]) ==
                8);
    }
}

TEST_CASE("Snapshot mesher samples block light across a cubic halo",
          "[render][mesher][light][border]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    snapshot.blockLight[MeshSnapshot::index(16, 8, 8)] = 12;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundLitBoundary = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 12);
            foundLitBoundary = true;
        }
    }
    REQUIRE(foundLitBoundary);
}

TEST_CASE("Mesher: baked corner AO darkens enclosed voxel corners", "[render][mesher][ao]") {
    // An L-shaped nook: a floor with two walls meeting at a corner. The floor
    // vertex tucked into the inner corner sees occluders on both sides and the
    // diagonal, so its baked AO is the lowest; a vertex out on the open floor
    // stays fully open (AO 3).
    Chunk chunk(ChunkPos{0, 4, 0});
    for (int x = 4; x <= 9; ++x)
        for (int z = 4; z <= 9; ++z)
            chunk.setBlock(x, 4, z, BlockType::STONE); // floor slab
    for (int z = 4; z <= 9; ++z)
        chunk.setBlock(4, 5, z, BlockType::STONE); // wall along -X edge
    for (int x = 4; x <= 9; ++x)
        chunk.setBlock(x, 5, 4, BlockType::STONE); // wall along -Z edge

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    uint8_t innerCornerAO = 3;
    uint8_t maxFloorAO = 0;
    bool foundInner = false;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        float x = static_cast<float>(v.px);
        float y = static_cast<float>(v.py);
        float z = static_cast<float>(v.pz);
        if (y < 4.5f || y > 5.5f)
            continue; // only the floor top at y = 5
        maxFloorAO = std::max(maxFloorAO, unpackCornerAO(v.faceAttr));
        // The concave corner vertex sits where the two walls meet (5,5,5)
        if (x > 4.9f && x < 5.1f && z > 4.9f && z < 5.1f) {
            innerCornerAO = std::min(innerCornerAO, unpackCornerAO(v.faceAttr));
            foundInner = true;
        }
    }
    REQUIRE(foundInner);
    REQUIRE(maxFloorAO == 3);    // open floor away from the walls stays lit
    REQUIRE(innerCornerAO == 0); // two walls + diagonal bury the tucked corner
}

TEST_CASE("Mesher: corner AO follows physical vertices on all six faces",
          "[render][mesher][ao][winding]") {
    struct FaceBasis {
        FaceNormal face;
        std::array<int, 3> normal;
        int tangentA;
        int tangentB;
    };
    constexpr std::array<FaceBasis, 6> faces{{
        {FaceNormal::PLUS_X, {1, 0, 0}, 1, 2},
        {FaceNormal::MINUS_X, {-1, 0, 0}, 1, 2},
        {FaceNormal::PLUS_Y, {0, 1, 0}, 0, 2},
        {FaceNormal::MINUS_Y, {0, -1, 0}, 0, 2},
        {FaceNormal::PLUS_Z, {0, 0, 1}, 0, 1},
        {FaceNormal::MINUS_Z, {0, 0, -1}, 0, 1},
    }};
    constexpr std::array<std::array<int, 2>, 4> cornerSigns{{
        {-1, -1},
        {-1, 1},
        {1, 1},
        {1, -1},
    }};

    MeshScratch scratch;
    for (const FaceBasis& basis : faces) {
        for (const auto& signs : cornerSigns) {
            MeshSnapshot snapshot;
            snapshot.clear();
            constexpr std::array<int, 3> center{8, 8, 8};
            snapshot.blocks[MeshSnapshot::index(center[0], center[1], center[2])] =
                BlockType::STONE;

            std::array<int, 3> exposure = center;
            for (int axis = 0; axis < 3; ++axis)
                exposure[axis] += basis.normal[axis];
            std::array<int, 3> sideA = exposure;
            std::array<int, 3> sideB = exposure;
            sideA[basis.tangentA] += signs[0];
            sideB[basis.tangentB] += signs[1];
            snapshot.blocks[MeshSnapshot::index(sideA[0], sideA[1], sideA[2])] = BlockType::STONE;
            snapshot.blocks[MeshSnapshot::index(sideB[0], sideB[1], sideB[2])] = BlockType::STONE;

            std::array<float, 3> target{8.0F, 8.0F, 8.0F};
            for (int axis = 0; axis < 3; ++axis) {
                if (basis.normal[axis] > 0)
                    target[axis] = 9.0F;
            }
            target[basis.tangentA] = signs[0] > 0 ? 9.0F : 8.0F;
            target[basis.tangentB] = signs[1] > 0 ? 9.0F : 8.0F;

            const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
            bool foundCorner = false;
            for (const Vertex& vertex : output.vertices) {
                if (unpackFace(vertex.faceAttr) == basis.face &&
                    static_cast<float>(vertex.px) == target[0] &&
                    static_cast<float>(vertex.py) == target[1] &&
                    static_cast<float>(vertex.pz) == target[2]) {
                    REQUIRE(unpackCornerAO(vertex.faceAttr) == 0);
                    foundCorner = true;
                }
            }
            REQUIRE(foundCorner);
        }
    }
}

TEST_CASE("Mesher: asymmetric AO triangulates across the brighter diagonal",
          "[render][mesher][ao]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(7, 9, 8)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(8, 9, 7)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundFace = false;
    for (size_t offset = 0; offset + 5 < output.opaqueIndexCount; offset += 6) {
        const Vertex& first = output.vertices[output.indices[offset]];
        const Vertex& second = output.vertices[output.indices[offset + 1]];
        const Vertex& third = output.vertices[output.indices[offset + 2]];
        const Vertex& fourth = output.vertices[output.indices[offset + 5]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            static_cast<float>(first.py) != 9.0F || static_cast<float>(second.py) != 9.0F ||
            static_cast<float>(third.py) != 9.0F || static_cast<float>(fourth.py) != 9.0F) {
            continue;
        }
        const uint8_t chosen = unpackCornerAO(first.faceAttr) + unpackCornerAO(third.faceAttr);
        const uint8_t alternate = unpackCornerAO(second.faceAttr) + unpackCornerAO(fourth.faceAttr);
        REQUIRE(chosen > alternate);
        foundFace = true;
    }
    REQUIRE(foundFace);
}

TEST_CASE("Mesher: opaque cover reduces skylight; non-opaque leaves do not", "[render][mesher]") {
    // Only OPAQUE blocks block the sky. A stone slab overhead shades the
    // ground below; a leaf canopy does not (its real cast shadow handles that,
    // and a column skylight shadow would double up under every tree).
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(4, 4, 8, BlockType::STONE);  // ground under stone cover
    chunk.setBlock(4, 8, 8, BlockType::STONE);  // opaque cover
    chunk.setBlock(12, 4, 8, BlockType::STONE); // ground under a leaf canopy
    chunk.setBlock(12, 8, 8, BlockType::LEAVES);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool foundShadedUnderStone = false;
    bool foundLitUnderLeaves = false;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        float x = static_cast<float>(v.px);
        float y = static_cast<float>(v.py);
        if (y > 4.5f && y < 5.5f && x > 4.4f && x < 5.6f) {
            REQUIRE(unpackSkyLight(v.faceAttr) < 15); // under opaque stone → shaded
            foundShadedUnderStone = true;
        }
        if (y > 4.5f && y < 5.5f && x > 12.4f && x < 13.6f) {
            REQUIRE(unpackSkyLight(v.faceAttr) == 15); // under leaves → still open
            foundLitUnderLeaves = true;
        }
    }
    REQUIRE(foundShadedUnderStone);
    REQUIRE(foundLitUnderLeaves);
}

TEST_CASE("Snapshot mesher uses a global sky cutoff above the cubic halo",
          "[render][mesher][light][skylight]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.blocks[MeshSnapshot::index(8, 4, 8)] = BlockType::STONE;
    snapshot.skyCutoffY[MeshSnapshot::skyIndex(8, 8)] = 96;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundShadedTop = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
            static_cast<float>(vertex.py) == 5.0F && static_cast<float>(vertex.px) >= 8.0F &&
            static_cast<float>(vertex.px) <= 9.0F && static_cast<float>(vertex.pz) >= 8.0F &&
            static_cast<float>(vertex.pz) <= 9.0F) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
            foundShadedTop = true;
        }
    }
    REQUIRE(foundShadedTop);
}

TEST_CASE("Underground skylight stays dark across unloaded vertical sections",
          "[world][mesher][light][skylight][streaming]") {
    World world(42, 4);
    constexpr int64_t worldX = 8;
    constexpr int64_t worldZ = 8;
    const int surfaceY = world.generator().surfaceYAt(worldX, worldZ);
    const int32_t surfaceSection = Chunk::worldToChunkY(surfaceY);
    const int32_t targetSection = surfaceSection - 4;
    REQUIRE(targetSection >= WORLD_MIN_CHUNK_Y);
    const ChunkPos target{0, targetSection, 0};

    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                REQUIRE(
                    world.getChunk({target.x + offsetX, target.y + offsetY, target.z + offsetZ}));
            }
        }
    }
    REQUIRE(world.getChunk({0, surfaceSection, 0}));

    MeshSnapshot separated;
    REQUIRE(world.snapshotForMeshing(target, separated));
    REQUIRE(separated.skyCutoffAt(8, 8) == WORLD_MAX_Y + 1);

    for (int32_t section = targetSection + 2; section <= surfaceSection; ++section) {
        REQUIRE(world.getChunk({0, section, 0}));
    }
    MeshSnapshot connected;
    REQUIRE(world.snapshotForMeshing(target, connected));
    REQUIRE(connected.skyCutoffAt(8, 8) < WORLD_MAX_Y + 1);
}

TEST_CASE("Generated opaque features extend the exact density sky cutoff",
          "[render][mesher][light][skylight][feature]") {
    World world(42, 4);
    constexpr ChunkPos target{-1707, 4, -1064};
    constexpr int64_t worldX = -27297;
    constexpr int64_t worldZ = -17021;
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                REQUIRE(
                    world.getChunk({target.x + offsetX, target.y + offsetY, target.z + offsetZ}));
            }
        }
    }

    const auto plan = world.generator().getColumnPlan({target.x, target.z});
    const int plannedSurface =
        plan->surfaceY(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ));
    const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(worldX, worldZ);
    REQUIRE(loadedTop);
    REQUIRE(*loadedTop > plannedSurface);
    REQUIRE(world.getBlockIfLoaded(worldX, *loadedTop, worldZ) == BlockType::LOG);

    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing(target, snapshot));
    REQUIRE(snapshot.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ))] == *loadedTop + 1);
}

TEST_CASE("Mesh skylight cutoffs follow opaque edits above the cubic halo",
          "[render][mesher][light][skylight][edit]") {
    World world(42, 4);
    constexpr int64_t WORLD_X = 0;
    constexpr int64_t WORLD_Z = 8;
    const int surfaceY = world.generator().surfaceYAt(WORLD_X, WORLD_Z);
    const ChunkPos surfaceCube{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(surfaceY),
                               Chunk::worldToChunk(WORLD_Z)};
    const int roofY = std::min((surfaceCube.y + 2) * CHUNK_EDGE + CHUNK_EDGE / 2, WORLD_MAX_Y - 1);
    REQUIRE(Chunk::worldToChunkY(roofY) > surfaceCube.y + 1);

    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const ChunkPos neighbor{surfaceCube.x + offsetX, surfaceCube.y + offsetY,
                                        surfaceCube.z + offsetZ};
                if (neighbor.y >= WORLD_MIN_CHUNK_Y && neighbor.y <= WORLD_MAX_CHUNK_Y) {
                    REQUIRE(world.getChunk(neighbor));
                }
            }
        }
    }
    REQUIRE(world.getChunk(Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(roofY),
                           Chunk::worldToChunk(WORLD_Z)));

    const ChunkPos negativeXSurfaceCube{surfaceCube.x - 1, surfaceCube.y, surfaceCube.z};
    world.markChunkMeshed(surfaceCube);
    world.markChunkMeshed(negativeXSurfaceCube);
    world.setBlock(WORLD_X, roofY, WORLD_Z, BlockType::STONE);
    REQUIRE(world.getChunk(surfaceCube)->needsMeshUpdate);
    REQUIRE(world.getChunk(negativeXSurfaceCube)->needsMeshUpdate);
    MeshSnapshot covered;
    REQUIRE(world.snapshotForMeshing(surfaceCube, covered));
    REQUIRE(covered.skyCutoffY[MeshSnapshot::skyIndex(Chunk::worldToLocal(WORLD_X),
                                                      Chunk::worldToLocal(WORLD_Z))] == roofY + 1);

    world.markChunkMeshed(surfaceCube);
    world.markChunkMeshed(negativeXSurfaceCube);
    world.setBlock(WORLD_X, roofY, WORLD_Z, BlockType::AIR);
    REQUIRE(world.getChunk(surfaceCube)->needsMeshUpdate);
    REQUIRE(world.getChunk(negativeXSurfaceCube)->needsMeshUpdate);
    const std::optional<int> restoredTop = world.surfaceHeightIfLoaded(WORLD_X, WORLD_Z);
    REQUIRE(restoredTop);
    MeshSnapshot opened;
    REQUIRE(world.snapshotForMeshing(surfaceCube, opened));
    REQUIRE(opened.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z))] == *restoredTop + 1);
}

TEST_CASE("Saved deep edits do not replace an unloaded generated sky cutoff",
          "[render][mesher][light][skylight][save]") {
    TempDir directory("saved_skylight_load_order");
    SaveManager saves(directory.path());
    constexpr uint32_t SEED = 42;
    constexpr int64_t WORLD_X = 8;
    constexpr int64_t WORLD_Z = 8;
    ChunkGenerator generator(SEED);
    const int surfaceY = generator.surfaceYAt(WORLD_X, WORLD_Z);
    const ChunkPos surfaceCube{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(surfaceY),
                               Chunk::worldToChunk(WORLD_Z)};
    const ChunkPos deepCube{surfaceCube.x, surfaceCube.y - 3, surfaceCube.z};
    REQUIRE(deepCube.y >= WORLD_MIN_CHUNK_Y);

    Chunk saved(deepCube);
    generator.generateCube(saved);
    saved.setBlock(Chunk::worldToLocal(WORLD_X), CHUNK_EDGE / 2, Chunk::worldToLocal(WORLD_Z),
                   BlockType::DIAMOND_ORE);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    World world(SEED, 4);
    world.setSaveManager(&saves);
    REQUIRE(world.getChunk(deepCube));
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const ChunkPos neighbor{surfaceCube.x + offsetX, surfaceCube.y + offsetY,
                                        surfaceCube.z + offsetZ};
                if (neighbor.y >= WORLD_MIN_CHUNK_Y && neighbor.y <= WORLD_MAX_CHUNK_Y) {
                    REQUIRE(world.getChunk(neighbor));
                }
            }
        }
    }

    const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(WORLD_X, WORLD_Z);
    REQUIRE(loadedTop);
    REQUIRE(*loadedTop > (deepCube.y + 1) * CHUNK_EDGE);
    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing(surfaceCube, snapshot));
    REQUIRE(snapshot.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z))] == *loadedTop + 1);
}

namespace {

struct alignas(4) TexturePixel {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
};

static_assert(sizeof(TexturePixel) == 4);

std::vector<TexturePixel> readBlockTextureMip(id<MTLTexture> texture, uint8_t layer,
                                              uint32_t mipLevel) {
    const uint32_t edge = BlockTextureArray::TILE_SIZE >> mipLevel;
    std::vector<TexturePixel> pixels(edge * edge);
    [texture getBytes:pixels.data()
          bytesPerRow:edge * sizeof(TexturePixel)
        bytesPerImage:edge * edge * sizeof(TexturePixel)
           fromRegion:MTLRegionMake2D(0, 0, edge, edge)
          mipmapLevel:mipLevel
                slice:layer];
    return pixels;
}

uint64_t blockTextureHash(const BlockTextureArray& textures) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;
    for (uint8_t layer = 0; layer < TEXTURE_LAYER_COUNT; ++layer) {
        for (uint32_t mipLevel = 0; mipLevel < BlockTextureArray::MIP_LEVEL_COUNT; ++mipLevel) {
            const std::vector<TexturePixel> pixels =
                readBlockTextureMip(textures.texture(), layer, mipLevel);
            for (const TexturePixel& pixel : pixels) {
                const auto* bytes = reinterpret_cast<const uint8_t*>(&pixel);
                for (uint32_t component = 0; component < sizeof(TexturePixel); ++component) {
                    hash ^= bytes[component];
                    hash *= FNV_PRIME;
                }
            }
        }
    }
    return hash;
}

uint32_t coveredTextureTexels(const std::vector<TexturePixel>& pixels) {
    return static_cast<uint32_t>(std::count_if(pixels.begin(), pixels.end(),
                                               [](TexturePixel pixel) { return pixel.a >= 128; }));
}

} // namespace

TEST_CASE("Block textures: extra layers extend past the block types", "[render][textures]") {
    REQUIRE(TEXTURE_LAYER_GRASS_SIDE == static_cast<uint8_t>(BlockType::COUNT));
    REQUIRE(TEXTURE_LAYER_COUNT > TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(BlockTextureArray::TILE_SIZE == 16);
    REQUIRE(BlockTextureArray::MIP_LEVEL_COUNT == 5);
}

TEST_CASE("Block textures upload a complete deterministic mip chain", "[render][textures][mip]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    BlockTextureArray first(device);
    BlockTextureArray second(device);
    REQUIRE(first.texture().mipmapLevelCount == BlockTextureArray::MIP_LEVEL_COUNT);
    REQUIRE(first.texture().arrayLength == TEXTURE_LAYER_COUNT);

    const uint64_t firstHash = blockTextureHash(first);
    REQUIRE(blockTextureHash(second) == firstHash);
    REQUIRE(firstHash == 0x3c3f105249a0d97eULL);
}

TEST_CASE("Block texture mips preserve alpha-tested flora coverage", "[render][textures][mip]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    BlockTextureArray textures(device);

    constexpr std::array CUTOUT_LAYERS{BlockType::TALL_GRASS, BlockType::LILY_PAD,
                                       BlockType::LEAVES};
    constexpr uint32_t BASE_TEXELS = BlockTextureArray::TILE_SIZE * BlockTextureArray::TILE_SIZE;
    for (BlockType block : CUTOUT_LAYERS) {
        const auto base = readBlockTextureMip(textures.texture(), static_cast<uint8_t>(block), 0);
        const uint32_t baseCovered = coveredTextureTexels(base);
        REQUIRE(baseCovered > 0);
        REQUIRE(baseCovered < BASE_TEXELS);

        for (uint32_t mipLevel = 1; mipLevel < BlockTextureArray::MIP_LEVEL_COUNT; ++mipLevel) {
            const auto mip =
                readBlockTextureMip(textures.texture(), static_cast<uint8_t>(block), mipLevel);
            uint32_t expectedCovered = static_cast<uint32_t>(
                (static_cast<uint64_t>(baseCovered) * mip.size() + BASE_TEXELS / 2) / BASE_TEXELS);
            expectedCovered = std::max(expectedCovered, 1U);
            REQUIRE(coveredTextureTexels(mip) == expectedCovered);
            for (TexturePixel pixel : mip) {
                if (pixel.a >= 128)
                    REQUIRE(pixel.g > 0);
            }
        }
    }
}

// ============================================================================
// MegaBuffer Constant Tests (no Metal device required)
// ============================================================================

TEST_CASE("MegaBuffer alignment is power of 2", "[render][megabuffer]") {
    uint64_t align = MegaBuffer::ALIGNMENT;
    REQUIRE(align > 0);
    REQUIRE((align & (align - 1)) == 0); // Power of 2 check
    REQUIRE(align == 256);
}

TEST_CASE("MegaBuffer alignUp rounds up to alignment", "[render][megabuffer]") {
    // Test alignment behavior with known values
    // alignUp(x) = (x + 255) & ~255
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    REQUIRE(alignUp(0) == 0);
    REQUIRE(alignUp(1) == 256);
    REQUIRE(alignUp(256) == 256);
    REQUIRE(alignUp(257) == 512);
    REQUIRE(alignUp(512) == 512);
    REQUIRE(alignUp(1000) == 1024);
    REQUIRE(alignUp(16 * sizeof(Vertex)) == 256); // 256 bytes, already aligned
    REQUIRE(alignUp(17 * sizeof(Vertex)) == 512); // 272 bytes, rounds to 512
}

TEST_CASE("MegaBuffer vertex allocation size calculation", "[render][megabuffer]") {
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    // 100 vertices × 16 bytes = 1600 bytes → aligned to 1792 (7 × 256)
    uint64_t vertexBytes = alignUp(100 * sizeof(Vertex));
    REQUIRE(vertexBytes >= 1600);
    REQUIRE(vertexBytes % MegaBuffer::ALIGNMENT == 0);

    // 1000 indices × 4 bytes = 4000 bytes → aligned to 4096 (16 × 256)
    uint64_t indexBytes = alignUp(1000 * sizeof(uint32_t));
    REQUIRE(indexBytes >= 4000);
    REQUIRE(indexBytes % MegaBuffer::ALIGNMENT == 0);
}

// ---- Day/Night Cycle Tests (Task 6.4-6.5) ----

TEST_CASE("Day/night cycle: sun position at noon", "[phase6][daynight]") {
    // At noon: worldTime = 6000 (25% of 24000)
    // orbitalAngle = 0.25 * 2*PI = PI/2
    // sunDirection = (cos(PI/2), sin(PI/2), 0.3) = (0, 1, 0.3)
    uint64_t worldTime = 6000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    // At noon: cos(PI/2) ≈ 0, sin(PI/2) = 1
    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at sunset", "[phase6][daynight]") {
    // At sunset: worldTime = 12000 (50% of 24000)
    // orbitalAngle = 0.5 * 2*PI = PI
    // sunDirection = (cos(PI), sin(PI), 0.3) = (-1, 0, 0.3)
    uint64_t worldTime = 12000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(-1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at midnight", "[phase6][daynight]") {
    // At midnight: worldTime = 18000 (75% of 24000)
    // orbitalAngle = 0.75 * 2*PI = 3PI/2
    // sunDirection = (cos(3PI/2), sin(3PI/2), 0.3) = (0, -1, 0.3)
    uint64_t worldTime = 18000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(-1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at dawn", "[phase6][daynight]") {
    // At dawn: worldTime = 0 (or 24000)
    // orbitalAngle = 0
    // sunDirection = (cos(0), sin(0), 0.3) = (1, 0, 0.3)
    uint64_t worldTime = 0;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: world time wraps at day boundary", "[phase6][daynight]") {
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // worldTime = 48000 (2 days) should wrap to same position as 0
    uint64_t worldTime = 48000;
    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    REQUIRE(dayFraction == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun elevation drives ambient brightness", "[phase6][daynight]") {
    // Test that sun elevation at noon produces higher ambient than at midnight
    auto computeAmbient = [](uint64_t worldTime) -> float {
        static constexpr uint64_t TICKS_PER_DAY = 24000;
        float dayFraction =
            static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
        float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);
        float sunElevation = std::sin(orbitalAngle);

        float ambientDay = 0.35f;
        float ambientNight = 0.1f;
        float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
        return ambientNight + (ambientDay - ambientNight) * ambientT;
    };

    float ambientNoon = computeAmbient(6000);
    float ambientMidnight = computeAmbient(18000);

    // Noon ambient should be higher than midnight
    REQUIRE(ambientNoon > ambientMidnight);
    REQUIRE(ambientNoon == Catch::Approx(0.35f).margin(0.01f));
    REQUIRE(ambientMidnight == Catch::Approx(0.1f).margin(0.01f));
}

// ============================================================================
// Phase 8: Post-Processing, Audio, Performance HUD Tests
// ============================================================================

// ---- Bloom Tests ----

// Uchimura "Gran Turismo" tonemap replicated from post.metal (the composite
// owns tonemapping now — ACES lived in the deleted bloom composite). Pins the
// curve's contract: black stays black, a linear mid section preserves the
// vibrant look, highlights compress, and it never decreases.
static float uchimuraToneMap(float x) {
    const float P = 1.0f, a = 1.0f, m = 0.22f, l = 0.4f, c = 1.33f, b = 0.0f;
    const float l0 = ((P - m) * l) / a;
    const float S0 = m + l0;
    const float S1 = m + a * l0;
    const float C2 = (a * P) / (P - S1);
    const float CP = -C2 / P;
    float w0 =
        1.0f - (x <= 0.0f ? 0.0f : (x >= m ? 1.0f : (x / m) * (x / m) * (3.0f - 2.0f * x / m)));
    float w2 = (x >= m + l0) ? 1.0f : 0.0f;
    float w1 = 1.0f - w0 - w2;
    float T = m * std::pow(x / m, c) + b;
    float L = m + a * (x - m);
    float S = P - (P - S1) * std::exp(CP * (x - S0));
    return T * w0 + L * w1 + S * w2;
}

TEST_CASE("Post: Uchimura tone mapping curve", "[hdr][post]") {
    // Black in, black out
    REQUIRE(uchimuraToneMap(0.0f) == Catch::Approx(0.0f).margin(0.001f));

    // The linear mid keeps mid-tones near identity (the vibrant look)
    float atMid = uchimuraToneMap(0.5f);
    REQUIRE(atMid > 0.35f);
    REQUIRE(atMid < 0.65f);

    // HDR highlights compress below the display max
    REQUIRE(uchimuraToneMap(4.0f) < 1.0f);
    REQUIRE(uchimuraToneMap(8.0f) < 1.0f);

    // Monotonically increasing across the range
    REQUIRE(uchimuraToneMap(0.2f) < uchimuraToneMap(0.5f));
    REQUIRE(uchimuraToneMap(0.5f) < uchimuraToneMap(1.0f));
    REQUIRE(uchimuraToneMap(1.0f) < uchimuraToneMap(2.0f));
}

TEST_CASE("Post: vibrance boosts low-saturation colors more than saturated ones", "[hdr][post]") {
    auto luma = [](float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; };
    // Vibrance boost factor from post.metal: vibrance * (1 - saturation)
    auto satBoost = [](float mx, float mn, float vibrance) {
        return vibrance * (1.0f - std::clamp(mx - mn, 0.0f, 1.0f));
    };
    const float vibrance = 0.5f;
    // A near-gray pixel (low saturation) gets a larger boost than a vivid one
    float grayBoost = satBoost(0.55f, 0.45f, vibrance); // sat 0.1
    float vividBoost = satBoost(0.9f, 0.1f, vibrance);  // sat 0.8
    REQUIRE(grayBoost > vividBoost);
    // Fully saturated → no boost
    REQUIRE(satBoost(1.0f, 0.0f, vibrance) == Catch::Approx(0.0f));
    (void)luma;
}

TEST_CASE("Bloom: extract threshold — bright pixels pass, dark pixels blocked", "[phase8][bloom]") {
    auto softThreshold = [](float luminance, float threshold) -> float {
        float low = threshold - 0.5f;
        float high = threshold + 0.5f;
        if (luminance <= low)
            return 0.0f;
        if (luminance >= high)
            return 1.0f;
        return (luminance - low) / (high - low);
    };

    // Dark pixel (luminance 0.2) with threshold 1.0 → blocked
    REQUIRE(softThreshold(0.2f, 1.0f) == Catch::Approx(0.0f));

    // Bright pixel (luminance 1.5) with threshold 1.0 → passes
    REQUIRE(softThreshold(1.5f, 1.0f) == Catch::Approx(1.0f));

    // Edge pixel (luminance 1.0) with threshold 1.0 → 0.5
    REQUIRE(softThreshold(1.0f, 1.0f) == Catch::Approx(0.5f));

    // Very bright pixel (luminance 3.0) → passes fully
    REQUIRE(softThreshold(3.0f, 1.0f) == Catch::Approx(1.0f));
}

TEST_CASE("Bloom: blur kernel weights are positive and symmetric", "[phase8][bloom]") {
    // 8-tap Kawase blur weights (normalized in shader by dividing by sum)
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // All weights are positive
    for (int i = 0; i < 8; ++i) {
        REQUIRE(weights[i] > 0.0f);
    }

    // Symmetric: first and last match, inner pairs match
    REQUIRE(weights[0] == weights[7]);
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);

    // Sum is used for normalization in shader
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += weights[i];
    }
    REQUIRE(sum > 0.0f); // Non-zero for valid normalization
}

TEST_CASE("Bloom: blur kernel is symmetric", "[phase8][bloom]") {
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // First and last should match
    REQUIRE(weights[0] == weights[7]);
    // Inner pairs should match
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);
}

// ---- Fog Tests ----

TEST_CASE("Fog: exponential fog factor at various distances", "[phase8][fog]") {
    float density = 0.0003f;

    auto fogFactor = [](float distance, float density) -> float {
        return 1.0f - std::exp(-density * distance);
    };

    // At distance 0: no fog
    REQUIRE(fogFactor(0.0f, density) == Catch::Approx(0.0f).epsilon(0.0001f));

    // At distance 100: slight fog
    float f100 = fogFactor(100.0f, density);
    REQUIRE(f100 > 0.0f);
    REQUIRE(f100 < 0.5f);

    // At distance 1000: significant fog
    float f1000 = fogFactor(1000.0f, density);
    REQUIRE(f1000 > 0.2f);
    REQUIRE(f1000 < 0.5f);

    // At distance 5000: very foggy
    float f5000 = fogFactor(5000.0f, density);
    REQUIRE(f5000 > 0.7f);

    // Fog factor increases monotonically with distance
    REQUIRE(fogFactor(100.0f, density) < fogFactor(500.0f, density));
    REQUIRE(fogFactor(500.0f, density) < fogFactor(1000.0f, density));
}

TEST_CASE("Fog: fog color mixing", "[phase8][fog]") {
    struct F3 {
        float x, y, z;
    };

    auto mixFog = [](float fogFactor, F3 fogColor, F3 litColor) -> F3 {
        // fogFactor: 0 = fully fogged, 1 = fully lit
        // mix(fogColor, litColor, fogFactor) = fogColor*(1-fogFactor) + litColor*fogFactor
        return {
            fogColor.x * (1.0f - fogFactor) + litColor.x * fogFactor,
            fogColor.y * (1.0f - fogFactor) + litColor.y * fogFactor,
            fogColor.z * (1.0f - fogFactor) + litColor.z * fogFactor,
        };
    };

    F3 fogColor{0.5f, 0.7f, 0.8f}; // Sky-like
    F3 litColor{0.3f, 0.3f, 0.3f}; // Dark stone

    // No fog (factor=1): fully lit
    auto noFog = mixFog(1.0f, fogColor, litColor);
    REQUIRE(noFog.x == Catch::Approx(litColor.x));

    // Full fog (factor=0): fully fogged
    auto fullFog = mixFog(0.0f, fogColor, litColor);
    REQUIRE(fullFog.x == Catch::Approx(fogColor.x));

    // Half fog: blend
    auto halfFog = mixFog(0.5f, fogColor, litColor);
    REQUIRE(halfFog.x == Catch::Approx((fogColor.x + litColor.x) * 0.5f));
}

// ---- Cloud Tests ----

TEST_CASE("Clouds: noise threshold for cloud generation", "[phase8][clouds]") {
    // Cloud threshold: 0.4 — noise values above this render as clouds
    float threshold = 0.4f;

    auto cloudMask = [](float noise, float threshold) -> float {
        float low = threshold - 0.1f;
        float high = threshold + 0.1f;
        if (noise <= low)
            return 0.0f;
        if (noise >= high)
            return 1.0f;
        return (noise - low) / (high - low);
    };

    // Low noise → no cloud
    REQUIRE(cloudMask(0.2f, threshold) == Catch::Approx(0.0f));

    // High noise → full cloud
    REQUIRE(cloudMask(0.6f, threshold) == Catch::Approx(1.0f));

    // At threshold → partial cloud
    REQUIRE(cloudMask(0.4f, threshold) == Catch::Approx(0.5f));
}

TEST_CASE("Clouds: wind offset calculation", "[phase8][clouds]") {
    // Wind speed: 0.02 blocks/tick
    float windSpeed = 0.02f;

    auto windOffset = [](uint64_t worldTime, float windSpeed) -> float {
        return static_cast<float>(worldTime) * windSpeed;
    };

    // At time 0: no offset
    REQUIRE(windOffset(0, windSpeed) == Catch::Approx(0.0f));

    // At time 1000: offset = 20
    REQUIRE(windOffset(1000, windSpeed) == Catch::Approx(20.0f));

    // At time 5000: offset = 100
    REQUIRE(windOffset(5000, windSpeed) == Catch::Approx(100.0f));

    // Monotonically increasing
    REQUIRE(windOffset(100, windSpeed) < windOffset(200, windSpeed));
}

TEST_CASE("Clouds: cloud altitude constant", "[phase8][clouds]") {
    // Cloud layer at Y=192
    static constexpr float CLOUD_ALTITUDE = 192.0f;
    REQUIRE(CLOUD_ALTITUDE > 0.0f);
    REQUIRE(CLOUD_ALTITUDE <= static_cast<float>(WORLD_MAX_Y));
}

// ---- Shared shader struct layout pins ----
// shader_types.hpp is compiled by BOTH clang++ and the Metal compiler; simd
// types have the same layout in each. These pins catch accidental drift
// (reordered fields, ad-hoc padding) that previously corrupted fog, camera
// position, sky colors, and particle data.

TEST_CASE("Shader types: Uniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(Uniforms) == 304);
    REQUIRE(offsetof(Uniforms, sunDirection) == 192);
    REQUIRE(offsetof(Uniforms, fogColor) == 240);
    REQUIRE(offsetof(Uniforms, fogDensity) == 256);
    REQUIRE(offsetof(Uniforms, cameraPosition) == 272);
    REQUIRE(offsetof(Uniforms, time) == 288);
    REQUIRE(offsetof(Uniforms, swayStrength) == 292);
    REQUIRE(offsetof(Uniforms, wetness) == 296);
    REQUIRE(alignof(Uniforms) == 16);
    REQUIRE(sizeof(ChunkOrigin) == 48);
    REQUIRE(offsetof(ChunkOrigin, farMetadata) == 32);
}

TEST_CASE("Shader types: ShadowUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ShadowPassUniforms) == 80);
    REQUIRE(offsetof(ShadowPassUniforms, time) == 64);
    REQUIRE(offsetof(ShadowPassUniforms, swayStrength) == 68);
    REQUIRE(sizeof(ShadowUniforms) == 224);
    REQUIRE(offsetof(ShadowUniforms, cascadeSplitDist) == 192);
    REQUIRE(offsetof(ShadowUniforms, shadowParams) == 208);
    REQUIRE(SHADOW_CASCADE_COUNT == 3);
}

TEST_CASE("Mat4 orthographic maps near->0 and far->1 (Metal depth)", "[common][math]") {
    Mat4 ortho = Mat4::orthographic(-10.f, 10.f, -10.f, 10.f, 0.f, 100.f);
    // A point at view-space z = -near (0) maps to NDC z = 0
    Vec3 nearPt = ortho.transformVec3({0.f, 0.f, 0.f});
    REQUIRE(nearPt.z == Catch::Approx(0.f).margin(1e-5));
    // A point at view-space z = -far maps to NDC z = 1
    Vec3 farPt = ortho.transformVec3({0.f, 0.f, -100.f});
    REQUIRE(farPt.z == Catch::Approx(1.f).margin(1e-5));
    // x/y map the ortho extents to [-1, 1]
    REQUIRE(ortho.transformVec3({10.f, 0.f, -1.f}).x == Catch::Approx(1.f).margin(1e-5));
    REQUIRE(ortho.transformVec3({-10.f, 0.f, -1.f}).x == Catch::Approx(-1.f).margin(1e-5));
}

TEST_CASE("Shader types: SkyUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(SkyUniforms) == 144);
    REQUIRE(offsetof(SkyUniforms, sunDirection) == 48);
    REQUIRE(offsetof(SkyUniforms, moonDirection) == 64);
    REQUIRE(offsetof(SkyUniforms, zenithColor) == 96);
    REQUIRE(offsetof(SkyUniforms, tanHalfFov) == 128);
    REQUIRE(offsetof(SkyUniforms, sunIntensity) == 136);
    REQUIRE(offsetof(SkyUniforms, starStrength) == 140);
}

TEST_CASE("Shader types: WaterUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(WaterUniforms) == 256);
    REQUIRE(offsetof(WaterUniforms, cameraRelativeViewProjection) == 64);
    REQUIRE(offsetof(WaterUniforms, zenithColor) == 128);
    REQUIRE(offsetof(WaterUniforms, resolution) == 224);
    REQUIRE(offsetof(WaterUniforms, fogDensity) == 232);
    REQUIRE(offsetof(WaterUniforms, time) == 236);
    REQUIRE(offsetof(WaterUniforms, cameraUnderwater) == 240);
    REQUIRE(offsetof(WaterUniforms, ssrStrength) == 244);
    REQUIRE(offsetof(WaterUniforms, skyExposure) == 248);
}

TEST_CASE("Camera-relative water depth stays continuous at large world coordinates",
          "[render][water][precision][seam]") {
    const Vec3 camera{23029.0F, 380.0F, -111486.0F};
    const Mat4 view =
        Mat4::lookAt(camera, Vec3{23050.0F, 307.0F, -111460.0F}, Vec3{0.0F, 1.0F, 0.0F});
    const Mat4 projection =
        Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F, 16.0F / 9.0F, 0.1F, 1000.0F);

    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    std::memcpy(&viewMatrix, view.data.data(), sizeof(viewMatrix));
    std::memcpy(&projectionMatrix, projection.data.data(), sizeof(projectionMatrix));
    viewMatrix.columns[3] = simd_make_float4(0.0F, 0.0F, 0.0F, 1.0F);
    const simd_float4x4 viewProjection = simd_mul(projectionMatrix, viewMatrix);
    const simd_float4x4 inverseViewProjection = simd_inverse(viewProjection);

    auto reconstruct = [&](simd_float3 relative) {
        const simd_float4 clip =
            simd_mul(viewProjection, simd_make_float4(relative.x, relative.y, relative.z, 1.0F));
        const simd_float3 ndc = clip.xyz / clip.w;
        const simd_float2 uv = simd_make_float2(ndc.x * 0.5F + 0.5F, 0.5F - ndc.y * 0.5F);
        const simd_float4 reconstructedClip =
            simd_make_float4(uv.x * 2.0F - 1.0F, 1.0F - uv.y * 2.0F, ndc.z, 1.0F);
        const simd_float4 reconstructed = simd_mul(inverseViewProjection, reconstructedClip);
        return reconstructed.xyz / reconstructed.w;
    };

    const simd_float3 ray = simd_normalize(simd_make_float3(21.0F, -73.0F, 26.0F));
    const simd_float3 waterSurface = ray * 75.0F;
    const simd_float3 lakeFloor = ray * 78.0F;
    const simd_float3 reconstructedSurface = reconstruct(waterSurface);
    const simd_float3 reconstructedFloor = reconstruct(lakeFloor);

    REQUIRE(simd_length(reconstructedSurface - waterSurface) < 1.0e-2F);
    REQUIRE(simd_length(reconstructedFloor - lakeFloor) < 1.0e-2F);
    REQUIRE(simd_length(reconstructedFloor - reconstructedSurface) ==
            Catch::Approx(3.0F).margin(1.0e-2F));

    // The same reconstruction remains stable on both sides of a cubic chunk
    // face even though the absolute Z coordinate is more than 100,000 blocks
    // from the origin.
    for (float absoluteX : {23039.999F, 23040.001F}) {
        const simd_float3 relative =
            simd_make_float3(absoluteX - camera.x, 307.875F - camera.y, -111470.0F - camera.z);
        REQUIRE(simd_length(reconstruct(relative) - relative) < 1.0e-2F);
    }
}

TEST_CASE("Shader types: CloudUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(CloudUniforms) == 112);
    REQUIRE(offsetof(CloudUniforms, sunDirection) == 64);
    REQUIRE(offsetof(CloudUniforms, tanHalfFov) == 80);
    REQUIRE(offsetof(CloudUniforms, cloudThreshold) == 100);
    REQUIRE(offsetof(CloudUniforms, volumetric) == 104);
    REQUIRE(offsetof(CloudUniforms, sunElevation) == 108);
}

TEST_CASE("Shader types: GPUParticle layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(GPUParticle) == 48);
    REQUIRE(offsetof(GPUParticle, velocity) == 16);
    REQUIRE(offsetof(GPUParticle, lifetime) == 32);
    REQUIRE(offsetof(GPUParticle, type) == 36);
}

TEST_CASE("Shader types: ParticleUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ParticleUniforms) == 144);
    REQUIRE(offsetof(ParticleUniforms, cameraPosition) == 128);
}

TEST_CASE("Shader types: BloomUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(BloomUniforms) == 32);
    REQUIRE(offsetof(BloomUniforms, texelSize) == 8);
    REQUIRE(offsetof(BloomUniforms, threshold) == 16);
    REQUIRE(offsetof(BloomUniforms, blurRadius) == 24);
}

TEST_CASE("Shader types: PostUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(PostUniforms) == 40);
    REQUIRE(offsetof(PostUniforms, resolution) == 0);
    REQUIRE(offsetof(PostUniforms, exposure) == 8);
    REQUIRE(offsetof(PostUniforms, bloomIntensity) == 12);
    REQUIRE(offsetof(PostUniforms, vibrance) == 16);
    REQUIRE(offsetof(PostUniforms, sharpening) == 20);
    REQUIRE(offsetof(PostUniforms, frameIndex) == 24);
    REQUIRE(offsetof(PostUniforms, flareStrength) == 28);
    REQUIRE(offsetof(PostUniforms, sunScreenUV) == 32);
    REQUIRE(sizeof(FlareState) == 4);
}

TEST_CASE("Shader types: SsaoUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(SsaoUniforms) == 160);
    REQUIRE(offsetof(SsaoUniforms, invProjection) == 64);
    REQUIRE(offsetof(SsaoUniforms, resolution) == 128);
    REQUIRE(offsetof(SsaoUniforms, radius) == 136);
    REQUIRE(offsetof(SsaoUniforms, strength) == 140);
    REQUIRE(offsetof(SsaoUniforms, bias) == 144);
    REQUIRE(offsetof(SsaoUniforms, frameIndex) == 148);
}

TEST_CASE("Shader types: VolumetricUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(VolumetricUniforms) == 144);
    REQUIRE(offsetof(VolumetricUniforms, cameraPosition) == 64);
    REQUIRE(offsetof(VolumetricUniforms, sunDirection) == 80);
    REQUIRE(offsetof(VolumetricUniforms, sunColor) == 96);
    REQUIRE(offsetof(VolumetricUniforms, stepCount) == 112);
    REQUIRE(offsetof(VolumetricUniforms, underwater) == 128);
    REQUIRE(offsetof(VolumetricUniforms, frameIndex) == 132);
}

TEST_CASE("Shader types: ExposureState + ExposureParams layout match MSL",
          "[render][shader-types]") {
    REQUIRE(sizeof(ExposureState) == 8);
    REQUIRE(offsetof(ExposureState, smoothedLogLum) == 0);
    REQUIRE(offsetof(ExposureState, exposure) == 4);

    REQUIRE(sizeof(ExposureParams) == 32);
    REQUIRE(offsetof(ExposureParams, keyValue) == 0);
    REQUIRE(offsetof(ExposureParams, adaptationRate) == 4);
    REQUIRE(offsetof(ExposureParams, minLogLum) == 8);
    REQUIRE(offsetof(ExposureParams, maxLogLum) == 12);
    REQUIRE(offsetof(ExposureParams, sampleGrid) == 16);
    REQUIRE(offsetof(ExposureParams, minExposure) == 24);
    REQUIRE(offsetof(ExposureParams, maxExposure) == 28);
}

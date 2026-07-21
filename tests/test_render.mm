#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/celestial.hpp>
#include <render/far_terrain.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/mesh_scheduler.hpp>
#include <render/metal_ownership.hpp>
#include <render/pixel_formats.hpp>
#include <render/render_pipeline.hpp>
#include <render/screen_space_lighting.hpp>
#include <render/shader_types.hpp>
#include <render/shadow_map.hpp>
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
#include <world/weather.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Rendering: meshing, textures, shared GPU layouts
// ===========================================================================

namespace {
bool metalOwnershipProbeDeallocated = false;
}

@interface MetalOwnershipProbe : NSObject
@end

@implementation MetalOwnershipProbe
- (void)dealloc {
    metalOwnershipProbeDeallocated = true;
#if !__has_feature(objc_arc)
    [super dealloc];
#endif
}
@end

TEST_CASE("Metal ownership reset releases under ARC and manual reference counting",
          "[render][metal][ownership]") {
    @autoreleasepool {
        metalOwnershipProbeDeallocated = false;
        MetalOwnershipProbe* probe = [[MetalOwnershipProbe alloc] init];
        REQUIRE(probe != nil);

        resetMetalObject(probe);

        REQUIRE(probe == nil);
        REQUIRE(metalOwnershipProbeDeallocated);
    }
}

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

TEST_CASE("Opaque scene pipelines share the HDR surface attachment contract",
          "[render][pipeline]") {
    @autoreleasepool {
        auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        PixelFormats::configureScenePassPipeline(descriptor);

        REQUIRE(descriptor.colorAttachments[0].pixelFormat == PixelFormats::SCENE_HDR);
        REQUIRE(descriptor.colorAttachments[1].pixelFormat == PixelFormats::SURFACE);
        REQUIRE(descriptor.depthAttachmentPixelFormat == PixelFormats::SCENE_DEPTH);
        REQUIRE(descriptor.rasterSampleCount == PixelFormats::SCENE_SAMPLE_COUNT);
    }
}

namespace {

worldgen::surface_material::SurfaceMaterialPalette testMaterialPalette(BlockType material) {
    worldgen::surface_material::SurfaceMaterialPalette palette;
    palette.count = 1;
    palette.entries[0] = {.material = material, .weight = 255};
    return palette;
}

bool sameMaterialPalette(const worldgen::surface_material::SurfaceMaterialPalette& first,
                         const worldgen::surface_material::SurfaceMaterialPalette& second) {
    if (first.count != second.count)
        return false;
    return std::equal(first.entries.begin(), first.entries.begin() + first.count,
                      second.entries.begin(), [](const auto& lhs, const auto& rhs) {
                          return lhs.material == rhs.material && lhs.weight == rhs.weight;
                      });
}

using TestFarGeometryFunction =
    std::function<FarTerrainGeometrySample(int64_t worldX, int64_t worldZ)>;
using TestFarMaterialFunction = std::function<BlockType(int64_t worldX, int64_t worldZ,
                                                        const FarTerrainGeometrySample& geometry)>;

FarTerrainSource testFarTerrainSource(TestFarGeometryFunction geometry,
                                      TestFarMaterialFunction material) {
    FarTerrainSource source;
    source.sample = [geometry = std::move(geometry), material = std::move(material)](
                        int64_t x, int64_t z, worldgen::SurfaceFootprint) {
        FarTerrainGeometrySample surface = geometry(x, z);
        if (surface.lake && surface.waterBodyId == worldgen::NO_WATER_BODY) {
            surface.waterBodyId = 0x5445'5354'4C41'4B45ULL;
        }
        return FarSurfaceSample{
            .geometry = surface,
            .footprintMinimumTerrainHeight = surface.terrainHeight,
            .footprintMaximumTerrainHeight = surface.terrainHeight,
            .materialPalette = testMaterialPalette(material(x, z, surface)),
        };
    };
    return source;
}

FarTerrainGeometrySample
testFarGeometry(const FarTerrainSource& source, int64_t x, int64_t z,
                worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1) {
    return source.sample(x, z, footprint).geometry;
}

FarTerrainSource farTerrainTestSource() {
    return testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            const int64_t variation = world_coord::floorMod(x + z * 3, 29);
            sample.terrainHeight = 72.0 + static_cast<double>(variation) * 0.25;
            sample.waterSurface = SEA_LEVEL;
            return sample;
        },
        [](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
            return world_coord::floorMod(x / 64 + z / 64, 2) == 0 ? BlockType::GRASS
                                                                  : BlockType::STONE;
        });
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

bool farTerrainTopsAreVoxelFlat(const FarTerrainMesh& mesh) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        const float height = static_cast<float>(first.py);
        if (height != std::round(height))
            return false;
        for (const uint32_t corner : {offset + 1, offset + 2, offset + 5}) {
            if (static_cast<float>(mesh.vertices[mesh.indices[corner]].py) != height)
                return false;
        }
    }
    return true;
}

bool farTerrainUsesVoxelFaces(const FarTerrainMesh& mesh, int step) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if ((first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        const FaceNormal face = unpackFace(first.faceAttr);
        std::array<float, 4> xs{};
        std::array<float, 4> ys{};
        std::array<float, 4> zs{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset + QUAD_CORNERS[corner]]];
            xs[corner] = static_cast<float>(vertex.px);
            ys[corner] = static_cast<float>(vertex.py);
            zs[corner] = static_cast<float>(vertex.pz);
            if (ys[corner] != std::round(ys[corner]))
                return false;
        }
        const auto allEqual = [](const auto& values) {
            return std::all_of(values.begin() + 1, values.end(),
                               [&](float value) { return value == values.front(); });
        };
        if (face == FaceNormal::PLUS_Y) {
            if (!allEqual(ys))
                return false;
            for (float x : xs) {
                if (world_coord::floorMod(static_cast<int64_t>(std::llround(x)),
                                          static_cast<int64_t>(step)) != 0) {
                    return false;
                }
            }
            for (float z : zs) {
                if (world_coord::floorMod(static_cast<int64_t>(std::llround(z)),
                                          static_cast<int64_t>(step)) != 0) {
                    return false;
                }
            }
        } else if (face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X) {
            if (!allEqual(xs))
                return false;
        } else if (face == FaceNormal::PLUS_Z || face == FaceNormal::MINUS_Z) {
            if (!allEqual(zs))
                return false;
        } else {
            return false;
        }
    }
    return true;
}

float expectedVoxelCellHeight(const FarTerrainSource& source, int64_t worldX, int64_t worldZ,
                              FarTerrainStep step) {
    const int width = farTerrainStepSize(step);
    const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(step);
    if (step == FarTerrainStep::ONE) {
        const FarSurfaceSample sample = source.sample(worldX, worldZ, footprint);
        return static_cast<float>(std::floor(sample.geometry.terrainHeight + 0.5));
    }
    double height = 0.0;
    for (const auto [dx, dz] :
         std::array<std::pair<int, int>, 4>{{{0, 0}, {width, 0}, {width, width}, {0, width}}}) {
        const FarSurfaceSample sample = source.sample(worldX + dx, worldZ + dz, footprint);
        height += sample.geometry.terrainHeight;
    }
    return static_cast<float>(std::ceil(height / 4.0));
}

std::optional<float> farTerrainHeightAt(const FarTerrainMesh& mesh, float x, float z) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        std::array<float, 4> xs{};
        std::array<float, 4> zs{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset + QUAD_CORNERS[corner]]];
            xs[corner] = static_cast<float>(vertex.px);
            zs[corner] = static_cast<float>(vertex.pz);
        }
        const auto [minimumX, maximumX] = std::minmax_element(xs.begin(), xs.end());
        const auto [minimumZ, maximumZ] = std::minmax_element(zs.begin(), zs.end());
        if (x >= *minimumX && x <= *maximumX && z >= *minimumZ && z <= *maximumZ)
            return static_cast<float>(first.py);
    }
    return std::nullopt;
}

std::vector<float> farTerrainHeightRaster(const FarTerrainMesh& mesh, int spacing) {
    const int edge = FAR_TERRAIN_TILE_EDGE / spacing;
    std::vector<float> result(static_cast<size_t>(edge * edge),
                              std::numeric_limits<float>::quiet_NaN());
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        std::array<float, 4> xs{};
        std::array<float, 4> zs{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset + QUAD_CORNERS[corner]]];
            xs[corner] = static_cast<float>(vertex.px);
            zs[corner] = static_cast<float>(vertex.pz);
        }
        const auto [minimumX, maximumX] = std::minmax_element(xs.begin(), xs.end());
        const auto [minimumZ, maximumZ] = std::minmax_element(zs.begin(), zs.end());
        const int firstX = std::clamp(static_cast<int>(std::floor(*minimumX / spacing)), 0, edge);
        const int lastX = std::clamp(static_cast<int>(std::ceil(*maximumX / spacing)), 0, edge);
        const int firstZ = std::clamp(static_cast<int>(std::floor(*minimumZ / spacing)), 0, edge);
        const int lastZ = std::clamp(static_cast<int>(std::ceil(*maximumZ / spacing)), 0, edge);
        for (int z = firstZ; z < lastZ; ++z) {
            for (int x = firstX; x < lastX; ++x) {
                result[static_cast<size_t>(z * edge + x)] = static_cast<float>(first.py);
            }
        }
    }
    return result;
}

std::optional<float> farWaterTopHeightAt(const FarTerrainMesh& mesh, float x, float z) {
    const auto signedArea = [](float ax, float az, float bx, float bz, float px, float pz) {
        return (px - bx) * (az - bz) - (ax - bx) * (pz - bz);
    };
    for (size_t offset = mesh.opaqueIndexCount; offset + 2 < mesh.indices.size(); offset += 3) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
        const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
        const float firstSign =
            signedArea(static_cast<float>(first.px), static_cast<float>(first.pz),
                       static_cast<float>(second.px), static_cast<float>(second.pz), x, z);
        const float secondSign =
            signedArea(static_cast<float>(second.px), static_cast<float>(second.pz),
                       static_cast<float>(third.px), static_cast<float>(third.pz), x, z);
        const float thirdSign =
            signedArea(static_cast<float>(third.px), static_cast<float>(third.pz),
                       static_cast<float>(first.px), static_cast<float>(first.pz), x, z);
        constexpr float EPSILON = 0.001F;
        const bool hasNegative =
            firstSign < -EPSILON || secondSign < -EPSILON || thirdSign < -EPSILON;
        const bool hasPositive = firstSign > EPSILON || secondSign > EPSILON || thirdSign > EPSILON;
        if (!(hasNegative && hasPositive))
            return static_cast<float>(first.py);
    }
    return std::nullopt;
}

bool farWaterTopCovers(const FarTerrainMesh& mesh, float x, float z) {
    return farWaterTopHeightAt(mesh, x, z).has_value();
}

bool hasTerrainBoundaryRiser(const FarTerrainMesh& mesh, FaceNormal face, float fixedCoordinate,
                             float alongStart, float alongEnd, float bottom, float top) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != face ||
            (first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        std::array<float, 4> fixed{};
        std::array<float, 4> along{};
        std::array<float, 4> heights{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset + QUAD_CORNERS[corner]]];
            const bool xFace = face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X;
            fixed[corner] = xFace ? static_cast<float>(vertex.px) : static_cast<float>(vertex.pz);
            along[corner] = xFace ? static_cast<float>(vertex.pz) : static_cast<float>(vertex.px);
            heights[corner] = static_cast<float>(vertex.py);
        }
        const auto [minimumAlong, maximumAlong] = std::minmax_element(along.begin(), along.end());
        const auto [minimumY, maximumY] = std::minmax_element(heights.begin(), heights.end());
        if (std::all_of(fixed.begin(), fixed.end(),
                        [&](float value) { return value == fixedCoordinate; }) &&
            *minimumAlong == alongStart && *maximumAlong == alongEnd && *minimumY == bottom &&
            *maximumY == top) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("Far terrain chooses globally specified LOD rings", "[render][far-terrain]") {
    REQUIRE_FALSE(farTerrainStepForChunkDistance(31.999).has_value());
    REQUIRE(farTerrainStepForChunkDistance(32.0) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(63.999) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(127.999) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(128.0) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(223.999) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(224.0) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(351.999) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(352.0) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForChunkDistance(511.999) == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(farTerrainStepForChunkDistance(512.0).has_value());
    REQUIRE(FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS - FAR_TERRAIN_NEAR_CHUNK_RADIUS == 96.0);
    REQUIRE(FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS - FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS == 96.0);
    REQUIRE(FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS - FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS == 128.0);
    REQUIRE(FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS - FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS == 160.0);
    REQUIRE(FAR_TERRAIN_MAX_CHUNK_RADIUS == FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_CHUNK_RADIUS == MAX_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Far terrain tiers map explicitly to surface footprints",
          "[render][far-terrain][lod][sampling][contract]") {
    constexpr std::array steps = {
        FarTerrainStep::ONE,   FarTerrainStep::TWO,     FarTerrainStep::FOUR,
        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    constexpr std::array footprints = {
        worldgen::SurfaceFootprint::BLOCK_1,  worldgen::SurfaceFootprint::BLOCK_2,
        worldgen::SurfaceFootprint::BLOCK_4,  worldgen::SurfaceFootprint::BLOCK_8,
        worldgen::SurfaceFootprint::BLOCK_16, worldgen::SurfaceFootprint::BLOCK_32,
    };
    for (size_t index = 0; index < steps.size(); ++index) {
        CAPTURE(index);
        REQUIRE(farTerrainSurfaceFootprint(steps[index]) == footprints[index]);
        REQUIRE(farTerrainStepForSize(worldgen::surfaceFootprintWidth(footprints[index])) ==
                steps[index]);
    }
    REQUIRE_FALSE(farTerrainStepForSize(3).has_value());
}

TEST_CASE("Far terrain samples one material palette per active LOD cell",
          "[render][far-terrain][material][lod][seam]") {
    std::array<std::set<std::pair<int64_t, int64_t>>, 6> sampledFootprints;
    FarTerrainSource source;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        sampledFootprints[static_cast<size_t>(std::countr_zero(static_cast<unsigned int>(
                              worldgen::surfaceFootprintWidth(footprint))))]
            .emplace(x, z);
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 72.0;
        const int64_t cellX = world_coord::floorDiv(
            x, static_cast<int64_t>(worldgen::surfaceFootprintWidth(footprint)));
        const int64_t cellZ = world_coord::floorDiv(
            z, static_cast<int64_t>(worldgen::surfaceFootprintWidth(footprint)));
        const BlockType material = world_coord::floorMod(cellX + cellZ, int64_t{2}) == 0
                                       ? BlockType::LIMESTONE
                                       : BlockType::ANDESITE;
        return FarSurfaceSample{
            .geometry = sample,
            .footprintMinimumTerrainHeight = sample.terrainHeight,
            .footprintMaximumTerrainHeight = sample.terrainHeight,
            .materialPalette = testMaterialPalette(material),
        };
    };

    const auto mesh = FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::TWO}, source);
    const auto& stepTwoSamples = sampledFootprints[1];
    // Boundary risers inspect one neighboring control ring, but the active
    // cell grid must still contain every footprint sample exactly once.
    REQUIRE(stepTwoSamples.size() >= 129 * 129);
    for (int z = 0; z <= 128; ++z) {
        for (int x = 0; x <= 128; ++x)
            REQUIRE(stepTwoSamples.contains({-256 + x * 2, -256 + z * 2}));
    }
    REQUIRE(std::any_of(stepTwoSamples.begin(), stepTwoSamples.end(), [](const auto& coordinate) {
        return world_coord::floorMod(coordinate.first, int64_t{32}) != 0 ||
               world_coord::floorMod(coordinate.second, int64_t{32}) != 0;
    }));
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

    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::FOUR}, source);
    REQUIRE(sampledFootprints[2].size() >= 65 * 65);

    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::SIXTEEN}, source);
    REQUIRE(sampledFootprints[4].size() >= 17 * 17);
}

TEST_CASE("Fine scheduler terrain follows its filtered footprint material palette",
          "[render][far-terrain][scheduler][material][water][lod][seam][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    // Canopy ownership has its own scheduler coverage. Keep this palette
    // regression focused on terrain so cold tree-habitat basins cannot consume
    // its bounded completion window.
    source.canopies = {};
    FarTerrainScheduler scheduler(std::move(source), limits);
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

    // This patch is a lake threshold where block-resolution density and the
    // step-four filtered surface disagree about submersion. The far sample's
    // footprint owns geometry and its palette together, so derive expected
    // materials from that contract instead of pinning a former palette entry.
    std::set<BlockType> expectedMaterials;
    bool sawSubmerged = false;
    bool sawDry = false;
    bool sawFilteredSubmersionDifference = false;
    constexpr int64_t ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    constexpr int STEP = static_cast<int>(FarTerrainStep::FOUR);
    for (int localZ = 64; localZ < 128; localZ += STEP) {
        for (int localX = 64; localX < 128; localX += STEP) {
            const int64_t worldX = ORIGIN_X + localX;
            const int64_t worldZ = ORIGIN_Z + localZ;
            const worldgen::SurfaceSample exact = generator->sampleExactSurface(worldX, worldZ);
            worldgen::SurfaceSample canonical =
                generator->sampleFarSurface(worldX, worldZ, worldgen::SurfaceFootprint::BLOCK_1);
            canonical.terrainHeight = exact.terrainHeight;
            canonical.hydrology.surfaceElevation = exact.terrainHeight;
            const worldgen::SurfaceSample filtered =
                generator->sampleFarSurface(worldX, worldZ, worldgen::SurfaceFootprint::BLOCK_4);
            const bool filteredSubmerged = worldgen::surface_material::submerged(filtered);
            sawSubmerged = sawSubmerged || filteredSubmerged;
            sawDry = sawDry || !filteredSubmerged;
            sawFilteredSubmersionDifference =
                sawFilteredSubmersionDifference ||
                filteredSubmerged != worldgen::surface_material::submerged(canonical);
            const auto palette = generator->farSurfaceMaterialPaletteAt(worldX, worldZ, filtered);
            expectedMaterials.insert(worldgen::surface_material::selectMaterial(
                palette,
                generator->farSurfaceMaterialRankAt(worldX + STEP / 2, worldZ + STEP / 2)));
        }
    }

    std::set<BlockType> observedMaterials;
    for (const Vertex& vertex : completed.front().mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y || vertex.px <= 64 ||
            vertex.px >= 128 || vertex.pz <= 64 || vertex.pz >= 128) {
            continue;
        }
        const BlockType material = static_cast<BlockType>(unpackTextureLayer(vertex.faceAttr));
        if (material != BlockType::WATER)
            observedMaterials.insert(material);
    }
    REQUIRE(sawSubmerged);
    REQUIRE(sawDry);
    REQUIRE(sawFilteredSubmersionDifference);
    REQUIRE(expectedMaterials.size() >= 2);
    // Canopy top faces can share this horizontal window, so the mesh may
    // contain additional leaf materials alongside every terrain material.
    REQUIRE(std::ranges::includes(observedMaterials, expectedMaterials));
}

TEST_CASE("Coverage parents separate filtered voxel tops from conservative bounds",
          "[render][far-terrain][coverage][lod][bounds]") {
    FarTerrainSource source;
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint footprint) {
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = 80.0;
        const bool coverageParent = worldgen::surfaceFootprintWidth(footprint) >= 16;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = coverageParent ? 72.0 : 80.0,
            .footprintMaximumTerrainHeight = coverageParent ? 91.0 : 80.0,
            .materialPalette = testMaterialPalette(BlockType::STONE),
        };
    };

    const auto exact = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const auto coverage = FarTerrainMesher::build({0, 0, FAR_TERRAIN_BASE_STEP}, source);
    const auto terrainHeights = [](const FarTerrainMesh& mesh) {
        std::set<float> result;
        for (const Vertex& vertex : mesh.vertices) {
            if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::WATER)) {
                result.insert(static_cast<float>(vertex.py));
            }
        }
        return result;
    };
    REQUIRE(terrainHeights(*exact) == std::set<float>{80.0F});
    REQUIRE(terrainHeights(*coverage) == std::set<float>{80.0F});
    REQUIRE(coverage->surfaceBounds.minY == 72.0F);
    REQUIRE(exact->surfaceBounds.maxY == 80.0F);
    REQUIRE(coverage->surfaceBounds.maxY == 91.0F);
}

TEST_CASE("Batched far cell bounds cover subcell relief and shorelines at every LOD",
          "[render][far-terrain][coverage][lod][bounds][water][seam][regression]") {
    constexpr std::array STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    for (const FarTerrainStep terrainStep : STEPS) {
        const int step = farTerrainStepSize(terrainStep);
        const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
        constexpr FarTerrainKey BASE_KEY{-2, -3, FarTerrainStep::TWO};
        const FarTerrainKey key{BASE_KEY.tileX, BASE_KEY.tileZ, terrainStep};
        const int64_t originX = key.tileX * FAR_TERRAIN_TILE_EDGE;
        const int64_t originZ = key.tileZ * FAR_TERRAIN_TILE_EDGE;
        const int64_t shorelineX = originX + FAR_TERRAIN_TILE_EDGE / 2 + step / 2;
        const int peakCellX = cellEdge * 3 / 4;
        const int peakCellZ = cellEdge * 3 / 4;
        const int trenchCellX = 0;
        const int trenchCellZ = cellEdge / 4;
        const int64_t peakWorldX = originX + static_cast<int64_t>(peakCellX * step);
        const int64_t peakWorldZ = originZ + static_cast<int64_t>(peakCellZ * step);
        const int64_t trenchWorldX = originX;
        const int64_t trenchWorldZ = originZ + static_cast<int64_t>(trenchCellZ * step);
        int callbackCalls = 0;
        int64_t callbackOriginX = 0;
        int64_t callbackOriginZ = 0;
        int callbackWidth = 0;
        int callbackHeight = 0;
        worldgen::SurfaceFootprint callbackFootprint = worldgen::SurfaceFootprint::BLOCK_1;
        FarTerrainSource source = testFarTerrainSource(
            [shorelineX](int64_t x, int64_t) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = x < shorelineX ? 40.0 : 96.0;
                sample.waterSurface = SEA_LEVEL;
                sample.lake = x < shorelineX;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample& sample) {
                return sample.lake ? BlockType::CLAY : BlockType::STONE;
            });
        source.cellBoundsGrid = [&](int64_t gridOriginX, int64_t gridOriginZ, int gridStep,
                                    int cellWidth, int cellHeight,
                                    worldgen::SurfaceFootprint footprint,
                                    std::span<FarTerrainCellBounds> output) {
            ++callbackCalls;
            callbackOriginX = gridOriginX;
            callbackOriginZ = gridOriginZ;
            callbackWidth = cellWidth;
            callbackHeight = cellHeight;
            callbackFootprint = footprint;
            if (gridStep != step || output.size() != static_cast<size_t>(cellWidth * cellHeight)) {
                throw std::invalid_argument("unexpected test cell bounds grid");
            }
            for (int z = 0; z < cellHeight; ++z) {
                for (int x = 0; x < cellWidth; ++x) {
                    const int64_t worldX = gridOriginX + static_cast<int64_t>(x * gridStep);
                    const int64_t worldZ = gridOriginZ + static_cast<int64_t>(z * gridStep);
                    const int64_t maximumX = worldX + gridStep;
                    double minimum = maximumX <= shorelineX ? 40.0 : 96.0;
                    double maximum = minimum;
                    if (worldX < shorelineX && maximumX > shorelineX) {
                        minimum = 40.0;
                        maximum = 96.0;
                    }
                    double skirtBottom = minimum - FAR_TERRAIN_SKIRT_DEPTH;
                    if (worldX == peakWorldX && worldZ == peakWorldZ) {
                        minimum = 88.9;
                        maximum = 221.25;
                        skirtBottom = 4.5;
                    }
                    if (worldX == trenchWorldX && worldZ == trenchWorldZ) {
                        minimum = -33.2;
                        maximum = 102.1;
                        skirtBottom = -110.4;
                    }
                    output[static_cast<size_t>(z * cellWidth + x)] = {
                        .terrainHeight = minimum,
                        .minimumTerrainHeight = minimum,
                        .maximumTerrainHeight = maximum,
                        .skirtBottom = skirtBottom,
                    };
                }
            }
        };

        const auto mesh = FarTerrainMesher::build(key, source);
        CAPTURE(step);
        REQUIRE(callbackCalls == 1);
        REQUIRE(callbackOriginX == originX - step);
        REQUIRE(callbackOriginZ == originZ - step);
        REQUIRE(callbackWidth == cellEdge + 2);
        REQUIRE(callbackHeight == cellEdge + 2);
        REQUIRE(worldgen::surfaceFootprintWidth(callbackFootprint) == step);
        REQUIRE(farTerrainHeightAt(*mesh, static_cast<float>(peakCellX * step + step / 2),
                                   static_cast<float>(peakCellZ * step + step / 2)) == 89.0F);
        REQUIRE(farTerrainHeightAt(*mesh, static_cast<float>(trenchCellX * step + step / 2),
                                   static_cast<float>(trenchCellZ * step + step / 2)) == -33.0F);
        REQUIRE(mesh->surfaceBounds.minY == -34.0F);
        REQUIRE(mesh->surfaceBounds.maxY == 222.0F);
        REQUIRE(mesh->bounds.minY == -111.0F);
        REQUIRE(mesh->bounds.maxY < mesh->surfaceBounds.maxY);
        REQUIRE(
            farWaterTopCovers(*mesh, FAR_TERRAIN_TILE_EDGE * 0.25F, FAR_TERRAIN_TILE_EDGE * 0.5F));
        REQUIRE_FALSE(farWaterTopCovers(*mesh, static_cast<float>(peakCellX * step + step / 2),
                                        static_cast<float>(peakCellZ * step + step / 2)));

        constexpr int PATCHES_PER_EDGE = FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int peakPatchX = peakCellX * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int peakPatchZ = peakCellZ * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const FarTerrainBounds& peakPatch =
            mesh->occluderPatches[static_cast<size_t>(peakPatchZ * PATCHES_PER_EDGE + peakPatchX)];
        REQUIRE(peakPatch.maxY == 222.0F);
        const int trenchPatchX = trenchCellX * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int trenchPatchZ = trenchCellZ * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const FarTerrainBounds& trenchPatch = mesh->occluderPatches[static_cast<size_t>(
            trenchPatchZ * PATCHES_PER_EDGE + trenchPatchX)];
        REQUIRE(trenchPatch.minY == -34.0F);
    }
}

TEST_CASE("Far cell bounds stitch negative tile faces independent of build order",
          "[render][far-terrain][coverage][lod][bounds][seam][determinism][regression]") {
    constexpr std::array STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    for (const FarTerrainStep terrainStep : STEPS) {
        const int step = farTerrainStepSize(terrainStep);
        size_t boundsCalls = 0;
        FarTerrainSource source = testFarTerrainSource(
            [](int64_t, int64_t) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = 90.0;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::ANDESITE; });
        source.cellBoundsGrid = [&](int64_t originX, int64_t originZ, int gridStep, int cellWidth,
                                    int cellHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainCellBounds> output) {
            ++boundsCalls;
            for (int z = 0; z < cellHeight; ++z) {
                for (int x = 0; x < cellWidth; ++x) {
                    const int64_t worldX = originX + static_cast<int64_t>(x * gridStep);
                    const int64_t worldZ = originZ + static_cast<int64_t>(z * gridStep);
                    const int64_t cellX = world_coord::floorDiv(worldX, int64_t{gridStep});
                    const int64_t cellZ = world_coord::floorDiv(worldZ, int64_t{gridStep});
                    const double minimum =
                        70.0 + world_coord::floorMod(cellX + cellZ * 3, int64_t{9});
                    output[static_cast<size_t>(z * cellWidth + x)] = {
                        .terrainHeight = minimum,
                        .minimumTerrainHeight = minimum,
                        .maximumTerrainHeight = minimum + 0.25,
                        .skirtBottom = minimum - 72.0,
                    };
                }
            }
        };
        const FarTerrainKey leftKey{-1, -1, terrainStep};
        const FarTerrainKey rightKey{0, -1, terrainStep};
        const auto rightFirst = FarTerrainMesher::build(rightKey, source);
        const auto leftSecond = FarTerrainMesher::build(leftKey, source);
        const auto leftFirst = FarTerrainMesher::build(leftKey, source);
        const auto rightSecond = FarTerrainMesher::build(rightKey, source);
        CAPTURE(step);
        REQUIRE(boundsCalls == 4);
        REQUIRE(leftFirst->deterministicHash == leftSecond->deterministicHash);
        REQUIRE(rightFirst->deterministicHash == rightSecond->deterministicHash);
        REQUIRE(leftFirst->surfaceBounds.minY == leftSecond->surfaceBounds.minY);
        REQUIRE(leftFirst->surfaceBounds.maxY == leftSecond->surfaceBounds.maxY);
        REQUIRE(rightFirst->surfaceBounds.minY == rightSecond->surfaceBounds.minY);
        REQUIRE(rightFirst->surfaceBounds.maxY == rightSecond->surfaceBounds.maxY);

        bool sawOwnedRiser = false;
        for (int cellZ = 0; cellZ < FAR_TERRAIN_TILE_EDGE / step; ++cellZ) {
            const int64_t worldZ = -FAR_TERRAIN_TILE_EDGE + static_cast<int64_t>(cellZ * step);
            const int64_t globalZ = world_coord::floorDiv(worldZ, int64_t{step});
            const float leftHeight = static_cast<float>(
                70 + world_coord::floorMod(int64_t{-1} + globalZ * 3, int64_t{9}));
            const float rightHeight = static_cast<float>(
                70 + world_coord::floorMod(int64_t{0} + globalZ * 3, int64_t{9}));
            if (leftHeight > rightHeight) {
                sawOwnedRiser =
                    sawOwnedRiser || hasTerrainBoundaryRiser(
                                         *leftFirst, FaceNormal::PLUS_X, FAR_TERRAIN_TILE_EDGE,
                                         cellZ * step, (cellZ + 1) * step, rightHeight, leftHeight);
            } else if (rightHeight > leftHeight) {
                sawOwnedRiser =
                    sawOwnedRiser ||
                    hasTerrainBoundaryRiser(*rightFirst, FaceNormal::MINUS_X, 0.0F, cellZ * step,
                                            (cellZ + 1) * step, leftHeight, rightHeight);
            }
        }
        REQUIRE(sawOwnedRiser);
    }
}

TEST_CASE("Far cell bounds remain stable after scheduler cache eviction",
          "[render][far-terrain][coverage][bounds][scheduler][cache][determinism][regression]") {
    auto boundsCalls = std::make_shared<std::atomic<size_t>>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 80.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::BASALT; });
    source.cellBoundsGrid = [boundsCalls](int64_t originX, int64_t originZ, int step, int cellWidth,
                                          int cellHeight, worldgen::SurfaceFootprint,
                                          std::span<FarTerrainCellBounds> output) {
        boundsCalls->fetch_add(1, std::memory_order_relaxed);
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t worldX = originX + static_cast<int64_t>(x * step);
                const int64_t worldZ = originZ + static_cast<int64_t>(z * step);
                const int64_t rank =
                    world_coord::floorMod(world_coord::floorDiv(worldX, int64_t{step}) * 5 +
                                              world_coord::floorDiv(worldZ, int64_t{step}) * 7,
                                          int64_t{13});
                const double minimum = 64.0 + rank;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = minimum,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = minimum + 8.25,
                    .skirtBottom = minimum - 70.0,
                };
            }
        }
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    constexpr FarTerrainKey LEFT{-1, -1, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey RIGHT{0, -1, FarTerrainStep::EIGHT};
    const auto buildPass = [&](std::array<FarTerrainKey, 2> keys) {
        for (const FarTerrainKey key : keys)
            REQUIRE(scheduler.enqueue(key));
        std::vector<FarTerrainResult> results;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (results.size() < keys.size() && std::chrono::steady_clock::now() < deadline) {
            scheduler.drainCompleted(results);
            if (results.size() < keys.size())
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        REQUIRE(results.size() == keys.size());
        std::unordered_map<FarTerrainKey, std::shared_ptr<const FarTerrainMesh>, FarTerrainKeyHash>
            meshes;
        for (const FarTerrainResult& result : results) {
            REQUIRE_FALSE(result.failed);
            REQUIRE(result.mesh);
            meshes.emplace(result.key, result.mesh);
        }
        return meshes;
    };

    const auto first = buildPass({LEFT, RIGHT});
    REQUIRE(scheduler.findCached(LEFT));
    REQUIRE(scheduler.findCached(RIGHT));
    scheduler.clearCache();
    REQUIRE_FALSE(scheduler.findCached(LEFT));
    REQUIRE_FALSE(scheduler.findCached(RIGHT));
    const auto second = buildPass({RIGHT, LEFT});
    scheduler.shutdown();
    for (const FarTerrainKey key : {LEFT, RIGHT}) {
        REQUIRE(first.at(key)->deterministicHash == second.at(key)->deterministicHash);
        REQUIRE(first.at(key)->surfaceBounds.minY == second.at(key)->surfaceBounds.minY);
        REQUIRE(first.at(key)->surfaceBounds.maxY == second.at(key)->surfaceBounds.maxY);
        REQUIRE(first.at(key)->bounds.minY == second.at(key)->bounds.minY);
        REQUIRE(first.at(key)->indices == second.at(key)->indices);
    }
    REQUIRE(boundsCalls->load(std::memory_order_relaxed) == 4);
}

TEST_CASE("Production far terrain exposes batched conservative cell bounds",
          "[render][far-terrain][coverage][bounds][production][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(static_cast<bool>(source.cellBoundsGrid));
    REQUIRE(static_cast<bool>(source.sampleGrid));
    std::array<FarSurfaceSample, 9> samples;
    source.sampleGrid(-32, -32, 32, 3, worldgen::SurfaceFootprint::BLOCK_32, samples);
    for (const FarSurfaceSample& sample : samples) {
        REQUIRE(std::isfinite(sample.footprintMinimumTerrainHeight));
        REQUIRE(std::isfinite(sample.footprintMaximumTerrainHeight));
        REQUIRE(sample.footprintMinimumTerrainHeight <= sample.geometry.terrainHeight);
        REQUIRE(sample.footprintMaximumTerrainHeight >= sample.geometry.terrainHeight);
    }
    std::array<FarTerrainCellBounds, 16> bounds;
    source.cellBoundsGrid(-64, -64, 32, 4, 4, worldgen::SurfaceFootprint::BLOCK_32, bounds);
    for (const FarTerrainCellBounds& cell : bounds) {
        REQUIRE(std::isfinite(cell.terrainHeight));
        REQUIRE(std::isfinite(cell.minimumTerrainHeight));
        REQUIRE(std::isfinite(cell.maximumTerrainHeight));
        REQUIRE(std::isfinite(cell.skirtBottom));
        REQUIRE(cell.minimumTerrainHeight <= cell.maximumTerrainHeight);
        REQUIRE(cell.minimumTerrainHeight <= cell.terrainHeight);
        REQUIRE(cell.terrainHeight <= cell.maximumTerrainHeight);
        REQUIRE(cell.skirtBottom <= cell.minimumTerrainHeight);
    }
}

TEST_CASE("Production cell bounds enclose interior emitted terrain and water floors",
          "[render][far-terrain][coverage][bounds][worldgen][hydrology][regression]") {
    struct Fixture {
        uint32_t seed;
        int64_t x;
        int64_t z;
        const char* name;
    };
    constexpr std::array FIXTURES = {
        Fixture{42, -12'289, 2'649, "negative river boundary"},
        Fixture{42, -8'240, 3'088, "waterfall receiver"},
        Fixture{764891, 23'029, -111'486, "caldera interior"},
    };
    constexpr int STEP = 16;
    for (const Fixture& fixture : FIXTURES) {
        auto generator = std::make_shared<ChunkGenerator>(fixture.seed);
        const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        const int64_t originX = world_coord::floorDiv(fixture.x, int64_t{STEP}) * STEP;
        const int64_t originZ = world_coord::floorDiv(fixture.z, int64_t{STEP}) * STEP;
        std::array<FarTerrainCellBounds, 1> bounds{};
        source.cellBoundsGrid(originX, originZ, STEP, 1, 1, worldgen::SurfaceFootprint::BLOCK_16,
                              bounds);
        double exactMinimum = std::numeric_limits<double>::max();
        double exactMaximum = std::numeric_limits<double>::lowest();
        bool sawGeneratedWater = false;
        bool sawWaterfall = false;
        for (int z = 0; z < STEP; ++z) {
            for (int x = 0; x < STEP; ++x) {
                const worldgen::SurfaceSample exact =
                    generator->sampleExactSurface(originX + x, originZ + z);
                exactMinimum = std::min(exactMinimum, exact.terrainHeight);
                exactMaximum = std::max(exactMaximum, exact.terrainHeight);
                sawGeneratedWater = sawGeneratedWater || exact.hydrology.ocean ||
                                    exact.hydrology.river || exact.hydrology.lake;
                sawWaterfall = sawWaterfall || exact.hydrology.waterfall;
            }
        }
        CAPTURE(fixture.name, originX, originZ, exactMinimum, exactMaximum,
                bounds.front().minimumTerrainHeight, bounds.front().maximumTerrainHeight);
        REQUIRE(bounds.front().minimumTerrainHeight <= exactMinimum);
        REQUIRE(bounds.front().maximumTerrainHeight >= exactMaximum);
        if (std::string_view(fixture.name) == "waterfall receiver") {
            REQUIRE(sawGeneratedWater);
            REQUIRE(sawWaterfall);
        }
    }
}

TEST_CASE("Production bounds retain negative step thirty-two parents after cache eviction",
          "[render][far-terrain][coverage][bounds][negative][determinism][cache]") {
    constexpr int64_t ORIGIN_X = -66;
    constexpr int64_t ORIGIN_Z = -34;
    constexpr int CELL_EDGE = 4;
    const auto sampleFineBounds = [](const std::shared_ptr<ChunkGenerator>& generator) {
        const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        std::array<FarTerrainCellBounds, CELL_EDGE * CELL_EDGE> bounds{};
        source.cellBoundsGrid(ORIGIN_X, ORIGIN_Z, 2, CELL_EDGE, CELL_EDGE,
                              worldgen::SurfaceFootprint::BLOCK_2, bounds);
        return bounds;
    };
    auto firstGenerator = std::make_shared<ChunkGenerator>(42);
    const auto first = sampleFineBounds(firstGenerator);
    firstGenerator->clearMacroCaches();
    auto rebuiltGenerator = std::make_shared<ChunkGenerator>(42);
    const auto rebuilt = sampleFineBounds(rebuiltGenerator);
    for (size_t index = 0; index < first.size(); ++index) {
        CAPTURE(index);
        REQUIRE(first[index].terrainHeight == rebuilt[index].terrainHeight);
        REQUIRE(first[index].minimumTerrainHeight == rebuilt[index].minimumTerrainHeight);
        REQUIRE(first[index].maximumTerrainHeight == rebuilt[index].maximumTerrainHeight);
        REQUIRE(first[index].skirtBottom == rebuilt[index].skirtBottom);
        REQUIRE(first[index].minimumTerrainHeight <= first[index].terrainHeight);
        REQUIRE(first[index].terrainHeight <= first[index].maximumTerrainHeight);
        REQUIRE(first[index].skirtBottom <= first[index].minimumTerrainHeight);
    }

    const FarTerrainSource coarseSource =
        FarTerrainMesher::generatorGeometrySource(rebuiltGenerator);
    std::array<FarTerrainCellBounds, 16> parents{};
    coarseSource.cellBoundsGrid(-128, -96, 32, 4, 4, worldgen::SurfaceFootprint::BLOCK_32, parents);
    const double adjacentParentMinimum =
        std::ranges::min(parents, {}, &FarTerrainCellBounds::minimumTerrainHeight)
            .minimumTerrainHeight;
    for (const FarTerrainCellBounds& cell : rebuilt) {
        REQUIRE(cell.skirtBottom <= adjacentParentMinimum);
    }
}

TEST_CASE("Production cell top authority stitches overlapping negative query aprons",
          "[render][far-terrain][coverage][bounds][negative][seam][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int STEP = 2;
    constexpr int EDGE = 4;
    std::array<FarTerrainCellBounds, EDGE * EDGE> crossing{};
    std::array<FarTerrainCellBounds, EDGE * EDGE> nonnegative{};
    source.cellBoundsGrid(-4, -4, STEP, EDGE, EDGE, worldgen::SurfaceFootprint::BLOCK_2, crossing);
    source.cellBoundsGrid(0, -4, STEP, EDGE, EDGE, worldgen::SurfaceFootprint::BLOCK_2,
                          nonnegative);
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < 2; ++x) {
            const FarTerrainCellBounds& left = crossing[static_cast<size_t>(z * EDGE + x + 2)];
            const FarTerrainCellBounds& right = nonnegative[static_cast<size_t>(z * EDGE + x)];
            CAPTURE(x, z);
            REQUIRE(left.terrainHeight == right.terrainHeight);
            REQUIRE(left.minimumTerrainHeight == right.minimumTerrainHeight);
            REQUIRE(left.maximumTerrainHeight == right.maximumTerrainHeight);
            REQUIRE(left.skirtBottom == right.skirtBottom);
        }
    }
}

TEST_CASE("Partially faded coverage and LOD parents never establish an occluder",
          "[render][far-terrain][coverage][occlusion][lod][transition][regression]") {
    FarTerrainCoverageFrontier frontier;
    frontier.complete = false;
    frontier.distanceBlocks = 1024.0F;
    frontier.distanceSquaredBlocks = 1024.0 * 1024.0;
    frontier.missingBaseTiles = 1;
    const TerrainHorizonViewpoint viewpoint{};
    constexpr double FADE_BLOCKS = 256.0;
    const FarTerrainBounds opaquePatch{600, 700, -16, 16, 40.0F, 80.0F};
    const FarTerrainBounds fadingPatch{700, 800, -16, 16, 40.0F, 80.0F};
    REQUIRE(
        farTerrainCoveragePatchMayOcclude(opaquePatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(opaquePatch, viewpoint, frontier, FADE_BLOCKS, true));
    frontier.complete = true;
    REQUIRE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, true));
}

TEST_CASE("Cross-tile canopy crowns expand horizontal surface bounds",
          "[render][far-terrain][canopy][bounds][frustum][seam][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    constexpr FarTerrainKey KEY{1, -2, FarTerrainStep::FOUR};
    constexpr int64_t ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    source.canopies = [](int64_t minimumX, int64_t minimumZ, int64_t, int64_t, FarTerrainStep) {
        FarCanopy canopy;
        canopy.x = minimumX + 1;
        canopy.z = minimumZ + FAR_TERRAIN_TILE_EDGE / 2;
        canopy.baseY = 64;
        canopy.topY = 75;
        canopy.canopyMinimumY = 67;
        canopy.canopyMaximumY = 75;
        canopy.canopyRadius = 8;
        canopy.logBlock = BlockType::LOG;
        canopy.leafBlock = BlockType::LEAVES;
        canopy.anchorId = 17;
        return std::vector<FarCanopy>{canopy};
    };

    const auto mesh = FarTerrainMesher::build(KEY, source);
    REQUIRE(mesh->canopyAnchorCount == 1);
    REQUIRE(mesh->surfaceBounds.minX < ORIGIN_X);
    REQUIRE(mesh->surfaceBounds.maxX == ORIGIN_X + FAR_TERRAIN_TILE_EDGE);
    REQUIRE(mesh->surfaceBounds.minZ == ORIGIN_Z);
    REQUIRE(mesh->surfaceBounds.maxZ == ORIGIN_Z + FAR_TERRAIN_TILE_EDGE);
    // Tile-local vertices still draw from the canonical tile origin. Only the
    // conservative frustum bounds expand around the crossing crown.
    REQUIRE(mesh->bounds.minX == ORIGIN_X);
    REQUIRE(mesh->bounds.maxX == ORIGIN_X + FAR_TERRAIN_TILE_EDGE);
}

TEST_CASE("Dynamic fine skirts cover a lower adjacent step-thirty-two parent",
          "[render][far-terrain][coverage][lod][bounds][skirt][seam][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < 0 ? 20.0 : 100.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t, int step, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t worldX = originX + static_cast<int64_t>(x * step);
                const double minimum = worldX < 0 ? 20.0 : 100.0;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = minimum,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = minimum,
                    .skirtBottom = worldX < 0 ? -44.0 : 10.0,
                };
            }
        }
    };

    const auto parent = FarTerrainMesher::build({-1, 0, FarTerrainStep::THIRTY_TWO}, source);
    const auto fine = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    REQUIRE(parent->surfaceBounds.minY == 20.0F);
    REQUIRE(fine->surfaceBounds.maxY == 100.0F);
    REQUIRE(100.0F - FAR_TERRAIN_SKIRT_DEPTH > parent->surfaceBounds.maxY);
    REQUIRE(fine->bounds.minY == 10.0F);
    bool sawCoveringWestSkirt = false;
    for (uint32_t offset = 0; offset + 5 < fine->opaqueIndexCount; offset += 6) {
        const Vertex& first = fine->vertices[fine->indices[offset]];
        if ((first.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) == 0U ||
            unpackFace(first.faceAttr) != FaceNormal::MINUS_X) {
            continue;
        }
        std::array<float, 4> heights{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            heights[corner] =
                static_cast<float>(fine->vertices[fine->indices[offset + QUAD_CORNERS[corner]]].py);
        }
        sawCoveringWestSkirt =
            sawCoveringWestSkirt ||
            *std::min_element(heights.begin(), heights.end()) <= parent->surfaceBounds.minY;
    }
    REQUIRE(sawCoveringWestSkirt);
}

TEST_CASE("Far terrain LOD responds to complexity with stable hysteresis",
          "[render][far-terrain][selection]") {
    REQUIRE(farTerrainStepForMetrics(100.0, 0.0F) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(100.0, 1.0F) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(80.0, 1.0F) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(300.0, 0.0F) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(300.0, 1.0F) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(450.0, 0.0F) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(450.0, 1.0F) == FarTerrainStep::EIGHT);

    REQUIRE(farTerrainStepForMetrics(68.0, 0.0F, FarTerrainStep::ONE) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(68.0, 0.0F, FarTerrainStep::TWO) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(120.0, 0.0F, FarTerrainStep::FOUR) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(220.0, 0.0F, FarTerrainStep::EIGHT) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(70.0, 0.0F, FarTerrainStep::EIGHT) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(400.0, 0.0F, FarTerrainStep::THIRTY_TWO) ==
            FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(350.0, 0.0F, FarTerrainStep::THIRTY_TWO) ==
            FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(210.0, 0.0F, FarTerrainStep::SIXTEEN) == FarTerrainStep::FOUR);
}

TEST_CASE("Far terrain topology swaps atomically beneath a narrow fog pulse",
          "[render][far-terrain][transition]") {
    const auto start = sampleFarTerrainTransition(0.0F);
    const auto quarter = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.25F);
    const auto midpoint = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.5F);
    const auto threeQuarter =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.75F);
    const auto complete = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS);

    REQUIRE_FALSE(start.drawTarget);
    REQUIRE(start.fogBlend == 0.0F);
    REQUIRE(start.progress == 0.0F);
    REQUIRE_FALSE(quarter.drawTarget);
    REQUIRE(quarter.fogBlend == 0.0F);
    REQUIRE(quarter.progress == Catch::Approx(0.15625F));
    REQUIRE(midpoint.drawTarget);
    REQUIRE(midpoint.fogBlend == 0.0F);
    REQUIRE(midpoint.progress == Catch::Approx(0.5F));
    REQUIRE(threeQuarter.drawTarget);
    REQUIRE(threeQuarter.fogBlend == 0.0F);
    REQUIRE(threeQuarter.progress == Catch::Approx(0.84375F));
    REQUIRE(complete.drawTarget);
    REQUIRE(complete.complete);
    REQUIRE(complete.fogBlend == 0.0F);
    REQUIRE(complete.progress == 1.0F);

    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    REQUIRE(farTerrainLodTerrainVisible(0.0F, SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(0.0F, TARGET));
    REQUIRE(farTerrainLodTerrainVisible(std::nextafter(0.5F, 0.0F), SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(std::nextafter(0.5F, 0.0F), TARGET));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(0.5F, SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(0.5F, TARGET));
    REQUIRE(farTerrainLodTerrainFog(0.42F, SOURCE) == Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodTerrainFog(0.5F, SOURCE) == Catch::Approx(1.0F));
    REQUIRE(farTerrainLodTerrainFog(0.58F, TARGET) == Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodConnectedGeometryVisible(SOURCE));
    REQUIRE_FALSE(farTerrainLodConnectedGeometryVisible(TARGET));

    constexpr unsigned int EMERGENCY_SOURCE = SOURCE | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    constexpr unsigned int EMERGENCY_TARGET = TARGET | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    const FarTerrainTransitionSample beforeEmergencySwap =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS - 0.001F);
    const FarTerrainTransitionSample afterEmergencySwap =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS + 0.001F);
    REQUIRE(farTerrainLodTerrainVisible(beforeEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(beforeEmergencySwap.progress, EMERGENCY_TARGET));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(afterEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(afterEmergencySwap.progress, EMERGENCY_TARGET));
    const float emergencySwapProgress =
        farTerrainLodTransitionProgressAtSeconds(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS);
    REQUIRE(farTerrainLodTerrainSwapProgress(EMERGENCY_SOURCE) ==
            Catch::Approx(emergencySwapProgress));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress, EMERGENCY_SOURCE) ==
            Catch::Approx(1.0F));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress - 0.030F, EMERGENCY_SOURCE) ==
            Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress + 0.030F, EMERGENCY_TARGET) ==
            Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(1.0F, SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(1.0F, TARGET));
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Water keeps one owner through far LOD and exact handoffs",
          "[render][far-terrain][water][transition][ownership][flicker][regression]") {
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    STATIC_REQUIRE(sizeof(Vertex) == 16);

    const auto requireSingleOwner = [=](bool exactOwned, float progress) {
        const bool sourceFarWater = !exactOwned && farTerrainLodConnectedGeometryVisible(SOURCE);
        const bool targetFarWater = !exactOwned && farTerrainLodConnectedGeometryVisible(TARGET);
        const unsigned int ownerCount = static_cast<unsigned int>(exactOwned) +
                                        static_cast<unsigned int>(sourceFarWater) +
                                        static_cast<unsigned int>(targetFarWater);
        CAPTURE(exactOwned, progress, sourceFarWater, targetFarWater);
        REQUIRE(ownerCount == 1U);
        if (!exactOwned) {
            REQUIRE(sourceFarWater);
            REQUIRE_FALSE(targetFarWater);
        }
    };

    for (int sample = 0; sample <= 64; ++sample) {
        const float progress = static_cast<float>(sample) / 64.0F;
        requireSingleOwner(false, progress);
    }

    bool exactOwned = false;
    for (const auto [builtRevision, currentRevision] :
         {std::pair{4U, 5U}, std::pair{5U, 5U}, std::pair{5U, 6U}, std::pair{6U, 6U}}) {
        exactOwned = farTerrainExactSectionOwnsSurface(exactOwned, builtRevision, currentRevision);
        for (const float progress : {0.0F, 0.25F, 0.5F, 0.75F, 1.0F}) {
            CAPTURE(builtRevision, currentRevision);
            requireSingleOwner(exactOwned, progress);
        }
    }
    REQUIRE(exactOwned);

    // Once the replacement completes, the scheduler submits only the new
    // regular draw. The transition helper must admit that sole owner.
    REQUIRE(farTerrainLodConnectedGeometryVisible(FAR_TERRAIN_DRAW_FLAG));
}

TEST_CASE("Far terrain skirts require both far owners",
          "[render][far-terrain][shader][transition][skirt][regression]") {
    REQUIRE(farTerrainSkirtOwnersVisible(false, false));
    REQUIRE_FALSE(farTerrainSkirtOwnersVisible(true, false));
    REQUIRE_FALSE(farTerrainSkirtOwnersVisible(false, true));
    REQUIRE_FALSE(farTerrainSkirtOwnersVisible(true, true));

    const simd_float2 negativeEdge = simd_make_float2(0.0F, 0.0F);
    const simd_float2 negativeXReceiver =
        farTerrainSkirtReceivingOwnershipSamplePosition(negativeEdge, FAR_TERRAIN_FACE_MINUS_X);
    const simd_float2 negativeZReceiver =
        farTerrainSkirtReceivingOwnershipSamplePosition(negativeEdge, FAR_TERRAIN_FACE_MINUS_Z);
    REQUIRE(negativeXReceiver.x < 0.0F);
    REQUIRE(negativeXReceiver.y == 0.0F);
    REQUIRE(negativeZReceiver.x == 0.0F);
    REQUIRE(negativeZReceiver.y < 0.0F);

    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    REQUIRE(farTerrainLodSkirtVisible(0.25F, SOURCE));
    REQUIRE_FALSE(farTerrainLodSkirtVisible(0.25F, TARGET));
    REQUIRE_FALSE(farTerrainLodSkirtVisible(0.75F, SOURCE));
    REQUIRE(farTerrainLodSkirtVisible(0.75F, TARGET));

    std::array<std::optional<FarTerrainStep>, 4> neighbors = {
        FarTerrainStep::FOUR, FarTerrainStep::TWO, std::nullopt, FarTerrainStep::EIGHT};
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::TWO, neighbors) ==
            ((1U << static_cast<uint8_t>(FaceNormal::PLUS_X)) |
             (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z))));
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::FOUR, neighbors) ==
            (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z)));
    REQUIRE(farTerrainSkirtEdgeMask(FarTerrainStep::SIXTEEN, neighbors) == 0U);

    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 80.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
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

TEST_CASE("Pinned seed forty-two handoff never leaves an orphan tile skirt",
          "[render][far-terrain][skirt][ownership][exact][regression]") {
    constexpr FarTerrainKey KEY{2, -6, FarTerrainStep::TWO};
    constexpr float CAMERA_X = 576.537F;
    constexpr float CAMERA_Z = -1528.19F;
    constexpr float EDGE_WORLD_X = 768.0F;
    constexpr float EDGE_WORLD_Z = -1518.0F;
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    const auto mesh = FarTerrainMesher::build(KEY, source);

    bool foundPinnedSkirt = false;
    for (uint32_t offset = 0; offset + 5 < mesh->opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh->vertices[mesh->indices[offset]];
        if ((first.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) == 0U ||
            unpackFace(first.faceAttr) != FaceNormal::PLUS_X) {
            continue;
        }
        float minimumZ = std::numeric_limits<float>::max();
        float maximumZ = std::numeric_limits<float>::lowest();
        float minimumY = std::numeric_limits<float>::max();
        float maximumY = std::numeric_limits<float>::lowest();
        for (const uint32_t corner : {offset, offset + 1, offset + 2, offset + 5}) {
            const Vertex& vertex = mesh->vertices[mesh->indices[corner]];
            REQUIRE(static_cast<float>(vertex.px) == FAR_TERRAIN_TILE_EDGE);
            minimumZ = std::min(minimumZ, static_cast<float>(vertex.pz));
            maximumZ = std::max(maximumZ, static_cast<float>(vertex.pz));
            minimumY = std::min(minimumY, static_cast<float>(vertex.py));
            maximumY = std::max(maximumY, static_cast<float>(vertex.py));
        }
        const float localSceneZ = EDGE_WORLD_Z - static_cast<float>(mesh->originZ);
        if (localSceneZ < minimumZ || localSceneZ > maximumZ)
            continue;
        foundPinnedSkirt = true;
        REQUIRE(maximumY - minimumY >= FAR_TERRAIN_SKIRT_DEPTH);
        break;
    }
    REQUIRE(foundPinnedSkirt);
    REQUIRE(std::abs(EDGE_WORLD_X - CAMERA_X) < 192.0F);
    REQUIRE(std::abs(EDGE_WORLD_Z - CAMERA_Z) < 16.0F);

    const simd_float2 edge =
        simd_make_float2(FAR_TERRAIN_TILE_EDGE, EDGE_WORLD_Z - static_cast<float>(mesh->originZ));
    const simd_float2 emitting =
        farTerrainExactOwnershipSamplePosition(edge, FAR_TERRAIN_FACE_PLUS_X, true);
    const simd_float2 receiving =
        farTerrainExactOwnershipSamplePosition(edge, FAR_TERRAIN_FACE_PLUS_X, false);
    REQUIRE(static_cast<int>(std::floor(emitting.x / CHUNK_EDGE)) == 15);
    REQUIRE(static_cast<int>(std::floor(receiving.x / CHUNK_EDGE)) == 16);

    double maximumExactSurface = std::numeric_limits<double>::lowest();
    for (int64_t x = 576; x <= 800; x += 8) {
        const int64_t z = static_cast<int64_t>(
            std::llround(static_cast<double>(CAMERA_Z) + static_cast<double>(x - 576) * 0.05275));
        maximumExactSurface =
            std::max(maximumExactSurface, generator->sampleExactSurface(x, z).terrainHeight);
    }
    REQUIRE(maximumExactSurface < 110.0);
    REQUIRE_FALSE(farTerrainSkirtOwnersVisible(true, false));
}

TEST_CASE("Far terrain view selection is circular ordered and negative-coordinate safe",
          "[render][far-terrain][selection]") {
    constexpr double cameraX = -320.5;
    constexpr double cameraZ = -511.25;
    constexpr int exactRadius = 32;
    constexpr int visibleRadius = 512;
    const double exactSquared = std::pow(exactRadius * CHUNK_EDGE, 2.0);
    const double visibleSquared = std::pow(visibleRadius * CHUNK_EDGE, 2.0);

    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(cameraX, cameraZ, visibleRadius, selected);
    REQUIRE_FALSE(selected.empty());

    std::array<bool, 4> reachedStep{};
    bool sawNegativeTile = false;
    bool sawExactBoundaryOverlap = false;
    bool sawTileWhollyInsideExactDisk = false;
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
        sawTileWhollyInsideExactDisk =
            sawTileWhollyInsideExactDisk || farthestSquared <= exactSquared;
    }

    REQUIRE(sawNegativeTile);
    REQUIRE(sawExactBoundaryOverlap);
    REQUIRE(sawTileWhollyInsideExactDisk);
    REQUIRE(
        std::all_of(reachedStep.begin(), reachedStep.end(), [](bool reached) { return reached; }));
}

TEST_CASE("Far terrain coverage stops before the nearest absent base",
          "[render][far-terrain][coverage][residency]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(128.0, 128.0, 64, selected);
    REQUIRE(selected.size() > 8);

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    for (const FarTerrainViewTile& tile : selected) {
        resident.insert({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    const auto isResident = [&](const FarTerrainKey& key) { return resident.contains(key); };
    const FarTerrainCoverageFrontier complete = farTerrainCoverageFrontier(selected, isResident);
    REQUIRE(complete.complete);
    REQUIRE(complete.missingBaseTiles == 0);
    REQUIRE(complete.distanceBlocks == 0.0F);

    const FarTerrainViewTile& missing = selected[selected.size() / 3];
    resident.erase({missing.key.tileX, missing.key.tileZ, FAR_TERRAIN_BASE_STEP});
    const FarTerrainCoverageFrontier incomplete = farTerrainCoverageFrontier(selected, isResident);
    REQUIRE_FALSE(incomplete.complete);
    REQUIRE(incomplete.missingBaseTiles == 1);
    REQUIRE(incomplete.distanceBlocks == Catch::Approx(std::sqrt(missing.distanceSquared)));
    REQUIRE(incomplete.distanceSquaredBlocks == missing.distanceSquared);
    REQUIRE(farTerrainCoverageFog(incomplete.distanceBlocks, incomplete.distanceBlocks) == 1.0F);
    REQUIRE_FALSE(farTerrainCoverageVisible(incomplete.distanceBlocks, incomplete.distanceBlocks));
    REQUIRE(farTerrainCoverageVisible(incomplete.distanceBlocks - 1.0F, incomplete.distanceBlocks));

    for (const FarTerrainViewTile& tile : selected) {
        CAPTURE(tile.key.tileX, tile.key.tileZ, tile.distanceSquared,
                incomplete.distanceSquaredBlocks);
        REQUIRE(farTerrainCoverageDrawEligible(tile.distanceSquared, incomplete) ==
                (tile.distanceSquared < missing.distanceSquared));
    }
}

TEST_CASE("A nearby cold coverage frontier preserves a clear camera neighborhood",
          "[render][far-terrain][coverage][fog][cold-start][regression]") {
    constexpr float NEAR_FRONTIER = 64.0F;
    STATIC_REQUIRE(FAR_TERRAIN_COVERAGE_MIN_FADE_BLOCKS == 16.0F);
    STATIC_REQUIRE(FAR_TERRAIN_COVERAGE_FADE_FRACTION == 0.125F);
    REQUIRE(farTerrainCoverageFadeBlocks(NEAR_FRONTIER) == 16.0F);
    REQUIRE(farTerrainCoverageFog(0.0F, NEAR_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(48.0F, NEAR_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(56.0F, NEAR_FRONTIER) == Catch::Approx(0.5F));
    REQUIRE(farTerrainCoverageFog(NEAR_FRONTIER, NEAR_FRONTIER) == 1.0F);

    // Once the connected prefix is at least eight tiles deep, retain the full
    // 256-block horizon taper used by settled long-distance coverage.
    constexpr float DISTANT_FRONTIER = 2048.0F;
    REQUIRE(farTerrainCoverageFadeBlocks(DISTANT_FRONTIER) == FAR_TERRAIN_COVERAGE_FADE_BLOCKS);
    REQUIRE(farTerrainCoverageFog(1792.0F, DISTANT_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(1920.0F, DISTANT_FRONTIER) == Catch::Approx(0.5F));
}

TEST_CASE("Connected parents refine every distance tier before full horizon coverage",
          "[render][far-terrain][coverage][lod][priority][cold-start][camera-jump][regression]") {
    FarTerrainViewTile near;
    near.key = {0, 0, FarTerrainStep::TWO};
    near.distanceSquared = 560.0 * 560.0;
    near.distanceChunks = 35.0;

    FarTerrainCoverageFrontier incomplete;
    incomplete.complete = false;
    incomplete.missingBaseTiles = 12;
    incomplete.distanceBlocks = 900.0F;
    incomplete.distanceSquaredBlocks = 900.0 * 900.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, false));

    // The camera target may build alongside its missing parent. It remains
    // undisplayable until the parent is resident, so this reduces cold
    // latency without exposing an isolated refinement.
    near.distanceSquared = 0.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, false, true));
    REQUIRE_FALSE(
        farTerrainInitialDisplayedStep(near.key.step, farTerrainStepMask(FarTerrainStep::TWO)));

    // A camera jump can contract the exact handoff to zero. Every connected
    // parent remains eligible, independent of that exact-residency radius.
    near.distanceSquared = 500.0 * 500.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, true));
    near.distanceSquared = 513.0 * 513.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, true));

    // No target may appear at or beyond the nearest missing base, even when
    // the tile is inside the urgent distance band and its own parent exists.
    near.distanceSquared = incomplete.distanceSquaredBlocks;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.distanceSquared = std::nextafter(incomplete.distanceSquaredBlocks, 0.0);
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));

    near.key.step = FarTerrainStep::FOUR;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::EIGHT;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::SIXTEEN;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::THIRTY_TWO;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::TWO;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(
        near, std::numeric_limits<float>::infinity(), incomplete, true));
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, -1.0F, incomplete, true));

    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT == 4);
    STATIC_REQUIRE(FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE == 4);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME == 4);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(1, true) == 1);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(2, true) == 2);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(4, true) == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(8, true) == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(4, false) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(1, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(2, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(4, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(5, true) == 1);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(8, true) == 4);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(4, false) == 4);
}

TEST_CASE("Full horizon residency orders every coarse parent before refinements",
          "[render][far-terrain][coverage][residency][priority]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(-257.25, 513.75, 512, selected);
    REQUIRE(selected.size() > 3'000);

    std::vector<FarTerrainKey> order;
    buildFarTerrainResidencyOrder(selected, order);
    REQUIRE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE(order.size() >= selected.size());
    for (size_t index = 0; index < selected.size(); ++index) {
        INFO("base index " << index);
        REQUIRE(order[index].tileX == selected[index].key.tileX);
        REQUIRE(order[index].tileZ == selected[index].key.tileZ);
        REQUIRE(farTerrainIsBaseStep(order[index].step));
        if (index > 0) {
            REQUIRE(selected[index - 1].distanceSquared <= selected[index].distanceSquared);
        }
    }
    for (size_t index = selected.size(); index < order.size(); ++index) {
        INFO("refinement index " << index);
        REQUIRE_FALSE(farTerrainIsBaseStep(order[index].step));
    }

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> unique(order.begin(), order.end());
    REQUIRE(unique.size() == order.size());
    REQUIRE(farTerrainResidencyMembershipMatches(selected, unique));
    for (size_t index = 0; index < selected.size(); ++index) {
        CAPTURE(index, selected[index].key.tileX, selected[index].key.tileZ,
                farTerrainStepSize(selected[index].key.step));
        REQUIRE(order[selected.size() + index] == selected[index].key);
    }
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::EIGHT, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::FOUR, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);

    std::reverse(selected.begin(), selected.end());
    REQUIRE_FALSE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE(farTerrainResidencyMembershipMatches(selected, unique));

    selected.front().key.step = selected.front().key.step == FarTerrainStep::TWO
                                    ? FarTerrainStep::FOUR
                                    : FarTerrainStep::TWO;
    REQUIRE_FALSE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE_FALSE(farTerrainResidencyMembershipMatches(selected, unique));
}

TEST_CASE("Cold near fallback distributes step eight before the selected step two target",
          "[render][far-terrain][coverage][lod][priority][cold-start][regression]") {
    constexpr ColumnPos CAMERA{0, 0};
    constexpr ColumnPos NEAR_A{1, 0};
    constexpr ColumnPos NEAR_B{0, 1};
    constexpr ColumnPos DISTANT{8, 0};
    constexpr ColumnPos TRANSITIONING{2, 0};
    std::array<FarTerrainRefinementCacheRequest, 5> requests{{
        {CAMERA, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {NEAR_A, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {NEAR_B, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {DISTANT, FarTerrainStep::THIRTY_TWO, FarTerrainStep::SIXTEEN},
        {TRANSITIONING, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, 0, true},
    }};
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    const std::vector<FarTerrainKey> expected = {
        {CAMERA.x, CAMERA.z, FarTerrainStep::EIGHT},
        {NEAR_A.x, NEAR_A.z, FarTerrainStep::EIGHT},
        {NEAR_B.x, NEAR_B.z, FarTerrainStep::EIGHT},
        {CAMERA.x, CAMERA.z, FarTerrainStep::TWO},
        {CAMERA.x, CAMERA.z, FarTerrainStep::FOUR},
        {NEAR_A.x, NEAR_A.z, FarTerrainStep::FOUR},
        {NEAR_B.x, NEAR_B.z, FarTerrainStep::FOUR},
        {NEAR_A.x, NEAR_A.z, FarTerrainStep::TWO},
        {NEAR_B.x, NEAR_B.z, FarTerrainStep::TWO},
        {DISTANT.x, DISTANT.z, FarTerrainStep::SIXTEEN},
    };
    REQUIRE(order == expected);
    REQUIRE(std::ranges::none_of(order, [](FarTerrainKey key) {
        return key.tileX == TRANSITIONING.x && key.tileZ == TRANSITIONING.z;
    }));

    auto oneSlot = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                oneSlot, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS - 1) == 1);
    REQUIRE_FALSE(oneSlot[0].deferIntermediate);
    for (size_t index = 1; index < 4; ++index)
        REQUIRE(oneSlot[index].deferIntermediate);

    auto noSlots = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                noSlots, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS) == 0);
    for (size_t index = 0; index < 4; ++index)
        REQUIRE(noSlots[index].deferIntermediate);
}

TEST_CASE("Far terrain parent lane outranks every refinement priority",
          "[render][far-terrain][coverage][scheduler][priority]") {
    const FarTerrainKey parent{12, -7, FAR_TERRAIN_BASE_STEP};
    const FarTerrainKey nearTarget{0, 0, FarTerrainStep::TWO};
    const FarTerrainKey fartherTarget{3, 4, FarTerrainStep::EIGHT};

    REQUIRE(farTerrainSubmissionBefore(parent, 10'000, nearTarget, 0));
    REQUIRE_FALSE(farTerrainSubmissionBefore(nearTarget, 0, parent, 10'000));
    REQUIRE(farTerrainSubmissionBefore(nearTarget, 4, fartherTarget, 8));

    FarTerrainCoverageFrontier frontier;
    frontier.complete = false;
    frontier.missingBaseTiles = 1;
    REQUIRE_FALSE(farTerrainRefinementLaneOpen(frontier, true));
    frontier.complete = true;
    frontier.missingBaseTiles = 0;
    REQUIRE_FALSE(farTerrainRefinementLaneOpen(frontier, false));
    REQUIRE(farTerrainRefinementLaneOpen(frontier, true));
}

TEST_CASE("Urgent nearby refinement shares all eight utility workers with parents",
          "[render][far-terrain][scheduler][priority][cold-start][camera-jump][performance]"
          "[regression]") {
    constexpr std::array<FarTerrainKey, 10> BASES{{
        {100, 0, FarTerrainStep::THIRTY_TWO},
        {200, 0, FarTerrainStep::THIRTY_TWO},
        {300, 0, FarTerrainStep::THIRTY_TWO},
        {400, 0, FarTerrainStep::THIRTY_TWO},
        {500, 0, FarTerrainStep::THIRTY_TWO},
        {600, 0, FarTerrainStep::THIRTY_TWO},
        {700, 0, FarTerrainStep::THIRTY_TWO},
        {800, 0, FarTerrainStep::THIRTY_TWO},
        {900, 0, FarTerrainStep::THIRTY_TWO},
        {1'000, 0, FarTerrainStep::THIRTY_TWO},
    }};
    constexpr std::array<FarTerrainKey, 5> URGENT{{
        {1'100, 0, FarTerrainStep::TWO},
        {1'200, 0, FarTerrainStep::FOUR},
        {1'300, 0, FarTerrainStep::EIGHT},
        {1'400, 0, FarTerrainStep::SIXTEEN},
        {1'500, 0, FarTerrainStep::TWO},
    }};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    std::unordered_set<int64_t> started;
    bool releaseInitialBases = false;
    bool releaseUrgent = false;

    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            const bool known =
                std::ranges::any_of(BASES, [&](FarTerrainKey key) { return key.tileX == tileX; }) ||
                std::ranges::any_of(URGENT, [&](FarTerrainKey key) { return key.tileX == tileX; });
            if (known) {
                std::unique_lock lock(gateMutex);
                if (started.insert(tileX).second) {
                    gateCv.notify_all();
                    if (std::find_if(BASES.begin(), BASES.begin() + 8, [&](FarTerrainKey key) {
                            return key.tileX == tileX;
                        }) != BASES.begin() + 8) {
                        gateCv.wait(lock, [&] { return releaseInitialBases; });
                    } else if (std::find_if(URGENT.begin(), URGENT.begin() + 4,
                                            [&](FarTerrainKey key) {
                                                return key.tileX == tileX;
                                            }) != URGENT.begin() + 4) {
                        gateCv.wait(lock, [&] { return releaseUrgent; });
                    }
                }
            }
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 16;
    limits.maxCompleted = 16;
    limits.maxCacheEntries = 16;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct GateRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& releaseInitial;
        bool& releaseUrgent;
        ~GateRelease() {
            {
                std::lock_guard lock(mutex);
                releaseInitial = true;
                releaseUrgent = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseInitialBases, releaseUrgent};
    scheduler.setWorkerBudget(FarTerrainScheduler::WORKER_COUNT);
    for (const FarTerrainKey base : BASES)
        REQUIRE(scheduler.enqueue(base));

    bool initialBasesStarted = false;
    {
        std::unique_lock lock(gateMutex);
        initialBasesStarted = gateCv.wait_for(lock, std::chrono::seconds(30), [&] {
            return std::all_of(BASES.begin(), BASES.begin() + 8,
                               [&](FarTerrainKey key) { return started.contains(key.tileX); });
        });
    }
    if (!initialBasesStarted) {
        {
            std::lock_guard lock(gateMutex);
            releaseInitialBases = true;
            releaseUrgent = true;
        }
        gateCv.notify_all();
        scheduler.shutdown();
    }
    REQUIRE(initialBasesStarted);

    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[0], 0));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[1], 1));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[2], 2));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[3], 3));
    REQUIRE_FALSE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE_FALSE(scheduler.enqueueUrgentRefinement(URGENT[4], 4));
    REQUIRE_FALSE(scheduler.enqueueUrgentRefinement(BASES[0], 0));
    {
        const FarTerrainSchedulerStats queued = scheduler.stats();
        REQUIRE(queued.urgentRefinementInFlight == FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT);
        REQUIRE(queued.queuedUrgentRefinement == FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT);
        REQUIRE(queued.queuedBase >= 2);
        REQUIRE(queued.activeBaseWorkers == 8);
        REQUIRE(queued.reservedBaseWorkers == 4);
        REQUIRE(queued.activeUrgentRefinement == 0);
        REQUIRE(queued.workerBudget == FarTerrainScheduler::WORKER_COUNT);
    }

    {
        std::lock_guard lock(gateMutex);
        releaseInitialBases = true;
    }
    gateCv.notify_all();

    bool nearbyAndParentAdvancedTogether = false;
    {
        std::unique_lock lock(gateMutex);
        nearbyAndParentAdvancedTogether = gateCv.wait_for(lock, std::chrono::seconds(30), [&] {
            const bool urgentStarted =
                std::all_of(URGENT.begin(), URGENT.begin() + 4,
                            [&](FarTerrainKey key) { return started.contains(key.tileX); });
            const bool nextBaseStarted =
                started.contains(BASES[8].tileX) || started.contains(BASES[9].tileX);
            return urgentStarted && nextBaseStarted;
        });
        releaseUrgent = true;
    }
    gateCv.notify_all();
    REQUIRE(nearbyAndParentAdvancedTogether);

    for (int attempt = 0; attempt < 500 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const FarTerrainSchedulerStats finished = scheduler.stats();
    scheduler.shutdown();
    REQUIRE(finished.inFlight == 0);
    REQUIRE(finished.urgentRefinementInFlight == 0);
    REQUIRE(finished.activeBaseWorkers == 0);
    REQUIRE(finished.reservedBaseWorkers == 0);
    REQUIRE(finished.activeUrgentRefinement == 0);
    REQUIRE(std::all_of(URGENT.begin(), URGENT.begin() + 4,
                        [&](FarTerrainKey key) { return started.contains(key.tileX); }));
    REQUIRE(started.contains(BASES[8].tileX));
    REQUIRE(started.contains(BASES[9].tileX));
}

TEST_CASE("Cold selected step two publishes a near step eight fallback first",
          "[render][far-terrain][scheduler][lod][cold-start][priority][regression]") {
    constexpr ColumnPos CAMERA{0, 0};
    std::array<FarTerrainRefinementCacheRequest, 3> requests{{
        {CAMERA, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {ColumnPos{1, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {ColumnPos{0, 1}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
    }};
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    REQUIRE(order.size() >= FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT);

    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool stepTwoStarted = false;
    bool releaseStepTwo = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (footprint == worldgen::SurfaceFootprint::BLOCK_2) {
            std::unique_lock lock(gateMutex);
            if (!stepTwoStarted) {
                stepTwoStarted = true;
                gateCv.notify_all();
            }
            gateCv.wait(lock, [&] { return releaseStepTwo; });
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct GateRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& release;
        ~GateRelease() {
            {
                std::lock_guard lock(mutex);
                release = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseStepTwo};
    for (const FarTerrainKey key :
         std::span(order).first(FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT)) {
        REQUIRE(scheduler.enqueueUrgentRefinement(key));
    }

    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(30), [&] { return stepTwoStarted; }));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().completedRefinement < 3; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(scheduler.stats().completedRefinement >= 3);
    FarTerrainRefinementCacheRequest cameraRequest{
        CAMERA, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, 0, false, true};
    std::vector<std::shared_ptr<const FarTerrainMesh>> cached;
    scheduler.findFinestCachedBatch(std::span(&cameraRequest, 1), 1, cached);
    REQUIRE(cached.empty());
    cameraRequest.residentSteps = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    cameraRequest.deferIntermediate = false;
    scheduler.findFinestCachedBatch(std::span(&cameraRequest, 1), 1, cached);
    REQUIRE(cached.size() == 1);
    REQUIRE(cached.front()->key.step == FarTerrainStep::EIGHT);

    {
        std::lock_guard lock(gateMutex);
        releaseStepTwo = true;
    }
    gateCv.notify_all();
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    cameraRequest.residentSteps = 0;
    cameraRequest.deferIntermediate = true;
    scheduler.findFinestCachedBatch(std::span(&cameraRequest, 1), 1, cached);
    REQUIRE(cached.size() == 1);
    REQUIRE(cached.front()->key.step == FarTerrainStep::TWO);
    scheduler.shutdown();
}

TEST_CASE("Camera jumps reset obsolete urgent refinement quota",
          "[render][far-terrain][scheduler][priority][camera-jump][cancellation][regression]") {
    constexpr FarTerrainKey BLOCKING_BASE{900, 0, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey OLD_FIRST{901, 0, FarTerrainStep::TWO};
    constexpr FarTerrainKey OLD_SECOND{902, 0, FarTerrainStep::FOUR};
    constexpr FarTerrainKey CURRENT{903, 0, FarTerrainStep::TWO};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool baseStarted = false;
    bool releaseBase = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKING_BASE.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!baseStarted) {
                baseStarted = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBase; });
            }
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct BaseRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& release;
        ~BaseRelease() {
            {
                std::lock_guard lock(mutex);
                release = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseBase};
    scheduler.setWorkerBudget(1);
    REQUIRE(scheduler.enqueue(BLOCKING_BASE));
    {
        std::unique_lock lock(gateMutex);
        const bool started =
            gateCv.wait_for(lock, std::chrono::seconds(30), [&] { return baseStarted; });
        if (!started)
            releaseBase = true;
        REQUIRE(started);
    }
    REQUIRE(scheduler.enqueueUrgentRefinement(OLD_FIRST));
    REQUIRE(scheduler.enqueueUrgentRefinement(OLD_SECOND));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 2);

    const uint64_t currentEpoch = scheduler.advanceEpoch();
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(CURRENT));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBase = true;
    }
    gateCv.notify_all();

    for (int attempt = 0; attempt < 500 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    const FarTerrainSchedulerStats finished = scheduler.stats();
    scheduler.shutdown();
    REQUIRE(finished.inFlight == 0);
    REQUIRE(finished.urgentRefinementInFlight == 0);
    REQUIRE(finished.canceled >= 3);
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().key == CURRENT);
    REQUIRE(completed.front().epoch == currentEpoch);
}

TEST_CASE("Full horizon wanted keys fit the CPU cache across tile offsets",
          "[render][far-terrain][coverage][residency][cache][capacity][regression]") {
    std::vector<double> offsets{0.0, std::nextafter(0.0, 1.0), 1.0};
    for (int offset = 8; offset < FAR_TERRAIN_TILE_EDGE; offset += 8)
        offsets.push_back(static_cast<double>(offset));
    offsets.push_back(static_cast<double>(FAR_TERRAIN_TILE_EDGE - 1));
    offsets.push_back(std::nextafter(static_cast<double>(FAR_TERRAIN_TILE_EDGE), 0.0));

    const size_t cacheCapacity = FarTerrainSchedulerLimits{}.maxCacheEntries;
    size_t maximumWanted = 0;
    double maximumCameraX = 0.0;
    double maximumCameraZ = 0.0;
    std::vector<FarTerrainViewTile> selected;
    std::vector<FarTerrainKey> wanted;
    for (const double cameraX : offsets) {
        for (const double cameraZ : offsets) {
            selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
            buildFarTerrainResidencyOrder(selected, wanted);
            CAPTURE(cameraX, cameraZ, selected.size(), wanted.size(), cacheCapacity);
            REQUIRE(farTerrainResidencyOrderMatches(selected, wanted));
            if (wanted.size() > maximumWanted) {
                maximumWanted = wanted.size();
                maximumCameraX = cameraX;
                maximumCameraZ = cameraZ;
            }
        }
    }
    CAPTURE(maximumCameraX, maximumCameraZ, maximumWanted, cacheCapacity);
    constexpr size_t MINIMUM_ENTRY_MARGIN = 32;
    REQUIRE(maximumWanted + MINIMUM_ENTRY_MARGIN <= cacheCapacity);
    REQUIRE(maximumWanted > 9'000);
}

TEST_CASE("Cold parent uploads fit the startup envelope without consuming refinement budget",
          "[render][far-terrain][coverage][upload][budget]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(0.0, 0.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    const size_t referenceColdBaseCount = selected.size();
    constexpr size_t SIXTY_FPS_FRAMES_IN_TWO_SECONDS = 120;
    const size_t parentFrames =
        (referenceColdBaseCount + FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - 1) /
        FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME;
    REQUIRE(referenceColdBaseCount > 3'000);
    REQUIRE(parentFrames < SIXTY_FPS_FRAMES_IN_TWO_SECONDS);
    REQUIRE(FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME == 12);
    REQUIRE(FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME == 32 * 1024 * 1024);
}

TEST_CASE("Exact handoff handles large snapshots empty meshes and stale revisions",
          "[render][far-terrain][coverage][exact][revision]") {
    std::vector<ChunkPos> required;
    for (int32_t y = -8; y <= -4; ++y) {
        for (int64_t z = -32; z <= 32; ++z) {
            for (int64_t x = -32; x <= 32; ++x) {
                required.push_back({x, y, z});
            }
        }
    }
    REQUIRE(required.size() > 16'384);

    std::unordered_set<ChunkPos> ready(required.begin(), required.end());
    const auto isReady = [&](ChunkPos position) { return ready.contains(position); };
    const FarTerrainExactHandoff complete =
        farTerrainExactHandoff(0.0, 0.0, 32, required, {}, isReady);
    REQUIRE(complete.requiredSections == required.size());
    REQUIRE(complete.readySections == required.size());
    REQUIRE(complete.unresolvedColumns == 0);
    REQUIRE(complete.distanceBlocks == 32 * CHUNK_EDGE);

    // Empty completed meshes own no GPU allocation, but their matching
    // revision still closes the exact coverage requirement.
    REQUIRE(farTerrainExactSectionReady(7, 7));
    REQUIRE_FALSE(farTerrainExactSectionReady(6, 7));

    constexpr ChunkPos STALE{4, -8, 0};
    ready.erase(STALE);
    const FarTerrainExactHandoff stale =
        farTerrainExactHandoff(0.0, 0.0, 32, required, {}, isReady);
    REQUIRE(stale.readySections + 1 == stale.requiredSections);
    REQUIRE(stale.distanceBlocks == 4 * CHUNK_EDGE);

    const std::array unresolved{ColumnPos{-3, 0}};
    const FarTerrainExactHandoff unresolvedResult =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved, isReady);
    REQUIRE(unresolvedResult.unresolvedColumns == 1);
    REQUIRE(unresolvedResult.distanceBlocks == 2 * CHUNK_EDGE);

    size_t readinessProbes = 0;
    FarTerrainExactCoverageCache cache;
    cache.rebuild(73, 32, required, {}, [&](ChunkPos position) {
        ++readinessProbes;
        return position != STALE;
    });
    REQUIRE(cache.matches(73, 32));
    REQUIRE_FALSE(cache.matches(74, 32));
    REQUIRE(readinessProbes == required.size());
    REQUIRE(cache.sample(0.0, 0.0).distanceBlocks == 4 * CHUNK_EDGE);
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(cache.sample(80.0, 0.0).distanceBlocks == 0.0F);
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(readinessProbes == required.size());

    REQUIRE(cache.setSectionReady(STALE, true));
    REQUIRE_FALSE(cache.setSectionReady(STALE, true));
    REQUIRE(cache.sample(0.0, 0.0).readySections == required.size());
    REQUIRE(cache.lastSampleColumnVisits() == 0);
    REQUIRE(cache.sample(0.0, 0.0).distanceBlocks == 32 * CHUNK_EDGE);
    REQUIRE(cache.setSectionReady(STALE, false));
    REQUIRE(cache.sample(0.0, 0.0).readySections + 1 == required.size());
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(readinessProbes == required.size());
}

TEST_CASE("Far refinement stays bounded until all exact streaming lanes drain",
          "[render][far-terrain][coverage][exact][priority][regression]") {
    REQUIRE_FALSE(farTerrainExactStreamingBusy(0, 0, 0, 24, 24, 0));

    REQUIRE(farTerrainExactStreamingBusy(1, 0, 0, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 1, 0, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 1, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 0, 24, 23, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 0, 24, 24, 1));

    // A duplicate publication can make an observational ready count exceed
    // its requirement, but cannot manufacture pending exact work.
    REQUIRE_FALSE(farTerrainExactStreamingBusy(0, 0, 0, 24, 25, 0));
}

TEST_CASE("Published exact sections retain ownership while replacement meshes build",
          "[render][far-terrain][coverage][exact][ownership][latch][regression]") {
    REQUIRE(farTerrainExactSectionOwnsSurface(false, 12, 12));
    REQUIRE_FALSE(farTerrainExactSectionOwnsSurface(false, 12, 13));
    REQUIRE(farTerrainExactSectionOwnsSurface(true, 12, 13));
}

TEST_CASE("Exact mesh registry waits when every capacity slot owns terrain",
          "[render][coverage][exact][ownership][capacity][regression]") {
    REQUIRE(chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES - 1, MAX_MESH_RESIDENT_CUBES, false,
                                      false));
    REQUIRE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, true, false));
    REQUIRE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, false, true));
    REQUIRE_FALSE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, false, false));
    REQUIRE_FALSE(chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES + 1, MAX_MESH_RESIDENT_CUBES,
                                            false, true));
}

TEST_CASE("One unresolved exact section preserves refinement in every ready column",
          "[render][far-terrain][coverage][exact][ownership][column][regression]") {
    std::vector<ChunkPos> required;
    required.reserve(16 * 16);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x)
            required.push_back({x, 4, z});
    }
    constexpr ChunkPos UNRESOLVED{7, 4, 9};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(128.0, 128.0, 32, required, {},
                               [UNRESOLVED](ChunkPos section) { return section != UNRESOLVED; });

    REQUIRE_FALSE(handoff.tileFullyReady({0, 0}));
    size_t exactOwnedColumns = 0;
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            const bool ready = handoff.columnFullyReady({x, z});
            CAPTURE(x, z);
            REQUIRE(ready == (ChunkPos{x, 4, z} != UNRESOLVED));
            exactOwnedColumns += ready ? 1 : 0;
        }
    }
    REQUIRE(exactOwnedColumns == 255);
}

TEST_CASE("Active far LOD transitions complete before selecting another target",
          "[render][far-terrain][lod][transition][monotonic][regression]") {
    const FarTerrainLodAdvance started =
        advanceFarTerrainLod(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO);
    REQUIRE(started.displayed == FarTerrainStep::THIRTY_TWO);
    REQUIRE(started.transitionTarget == FarTerrainStep::TWO);
    REQUIRE_FALSE(started.completedTransition);

    const FarTerrainLodAdvance redirected =
        advanceFarTerrainLod(started.displayed, FarTerrainStep::FOUR, started.transitionTarget,
                             FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.75F);
    REQUIRE(redirected.displayed == FarTerrainStep::THIRTY_TWO);
    REQUIRE(redirected.transitionTarget == FarTerrainStep::TWO);
    REQUIRE_FALSE(redirected.completedTransition);

    const FarTerrainLodAdvance completed =
        advanceFarTerrainLod(redirected.displayed, FarTerrainStep::FOUR,
                             redirected.transitionTarget, FAR_TERRAIN_LOD_TRANSITION_SECONDS);
    REQUIRE(completed.displayed == FarTerrainStep::TWO);
    REQUIRE_FALSE(completed.transitionTarget.has_value());
    REQUIRE(completed.completedTransition);

    const FarTerrainLodAdvance next =
        advanceFarTerrainLod(completed.displayed, FarTerrainStep::FOUR);
    REQUIRE(next.displayed == FarTerrainStep::TWO);
    REQUIRE(next.transitionTarget == FarTerrainStep::FOUR);
}

TEST_CASE("Nearby far fallback chooses its finest ready tier without regressing",
          "[render][far-terrain][lod][residency][exact][priority][regression]") {
    FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::THIRTY_TWO);

    ready |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  ready, true));
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::SIXTEEN);
    ready |= farTerrainStepMask(FarTerrainStep::FOUR);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::FOUR);
    ready |= farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::FOUR, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::TWO);

    // A step-16 replacement can finish while finer CPU work completes. The
    // active pair remains stable, then the next replacement skips directly
    // to the finest ready tier without serially dwelling at steps 8 and 4.
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE_FALSE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  ready, true));
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::TWO);

    // A transient cache observation cannot make an already displayed nearby
    // refinement fall back to an emergency parent.
    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
            farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::TWO);

    // Distance-selected coarsening remains intentional and requires the
    // requested replacement to be resident.
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::EIGHT, ready) ==
            FarTerrainStep::TWO);
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::EIGHT, ready) ==
            FarTerrainStep::EIGHT);
}

TEST_CASE("Resident nearby refinement initializes without displaying its coarse parent",
          "[render][far-terrain][lod][residency][camera-jump][flicker][regression]") {
    FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE_FALSE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready));

    ready |= farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready) == FarTerrainStep::TWO);

    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
            farTerrainStepMask(FarTerrainStep::SIXTEEN) | farTerrainStepMask(FarTerrainStep::FOUR);
    REQUIRE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready) == FarTerrainStep::FOUR);

    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready) ==
            FarTerrainStep::THIRTY_TWO);

    REQUIRE_FALSE(
        farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready, FarTerrainStep::EIGHT));
    ready |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(
        farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready, FarTerrainStep::EIGHT));
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready, FarTerrainStep::EIGHT) ==
            FarTerrainStep::EIGHT);
    REQUIRE_FALSE(
        farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::EIGHT));
    REQUIRE_FALSE(farTerrainDisplayedStepAllowed(FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT));
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::EIGHT, FarTerrainStep::EIGHT));
    REQUIRE_FALSE(farTerrainDisplayedStepAllowed(FarTerrainStep::EIGHT, FarTerrainStep::TWO));
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::THIRTY_TWO));

    ready |= farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE(farTerrainInitialDisplayedStep(FarTerrainStep::TWO, ready, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);
}

TEST_CASE("Every unresolved exact-loading tile receives a fine temporary LOD",
          "[render][far-terrain][coverage][lod][priority][exact][regression]") {
    constexpr size_t PROTECTED_TILE_COUNT = 24;
    constexpr size_t BLOCK_SCALE_TILE_COUNT = 4;
    std::array<FarTerrainRefinementCacheRequest, PROTECTED_TILE_COUNT> requests{};
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index] = {{static_cast<int64_t>(index), 0},
                           FarTerrainStep::THIRTY_TWO,
                           FarTerrainStep::TWO,
                           0,
                           false,
                           false,
                           true,
                           index < BLOCK_SCALE_TILE_COUNT};
    }
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);

    REQUIRE(order.size() >= PROTECTED_TILE_COUNT);
    for (size_t index = 0; index < PROTECTED_TILE_COUNT; ++index) {
        CAPTURE(index, static_cast<int>(order[index].step));
        REQUIRE(order[index].tileX == static_cast<int64_t>(index));
        REQUIRE(order[index].step ==
                (index < BLOCK_SCALE_TILE_COUNT ? FarTerrainStep::TWO : FarTerrainStep::EIGHT));
    }

    auto transitionLimited = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                transitionLimited, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS) == 0);
    REQUIRE(std::ranges::none_of(transitionLimited,
                                 [](const auto& request) { return request.deferIntermediate; }));

    std::vector<FarTerrainViewTile> selected{
        {{0, 0, FarTerrainStep::TWO}, {}, 0.0, 32.0},
        {{1, 0, FarTerrainStep::TWO}, {}, 256.0 * 256.0, 32.0},
        {{2, 0, FarTerrainStep::TWO}, {}, 512.0 * 512.0, 32.0},
    };
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    for (const FarTerrainViewTile& tile : selected) {
        resident.insert({tile.key.tileX, tile.key.tileZ, FarTerrainStep::THIRTY_TWO});
    }
    const auto drawable = [&](FarTerrainKey base) {
        if (!resident.contains(base))
            return false;
        if (base.tileX != 0)
            return true;
        return resident.contains({base.tileX, base.tileZ, FarTerrainStep::TWO});
    };
    FarTerrainCoverageFrontier frontier = farTerrainCoverageFrontier(selected, drawable);
    REQUIRE(frontier.missingBaseTiles == 1);
    REQUIRE_FALSE(farTerrainCoverageDrawEligible(selected[1].distanceSquared, frontier));
    resident.insert({0, 0, FarTerrainStep::TWO});
    frontier = farTerrainCoverageFrontier(selected, drawable);
    REQUIRE(frontier.complete);
}

TEST_CASE("Cold nearby parents coalesce refinements for a bounded interval",
          "[render][far-terrain][lod][residency][priority][cold-start][camera-jump][regression]") {
    REQUIRE(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS <= 0.12F);
    REQUIRE(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS < FAR_TERRAIN_LOD_TRANSITION_SECONDS);
    REQUIRE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                            FarTerrainStep::SIXTEEN, 0.0F));
    REQUIRE(farTerrainDeferNearIntermediate(
        FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, FarTerrainStep::FOUR,
        std::nextafter(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS, 0.0F)));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  FarTerrainStep::SIXTEEN,
                                                  FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS));

    // The final requested tier bypasses the grace, and a tile that has
    // already refined never regresses to a coarser placeholder after travel.
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  FarTerrainStep::TWO, 0.0F));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO,
                                                  FarTerrainStep::FOUR, 0.0F));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::FOUR,
                                                  FarTerrainStep::SIXTEEN, 0.0F));
}

TEST_CASE("Far trees exchange monotonically through exact and LOD transitions",
          "[render][far-terrain][canopy][lod][transition][ownership][flicker][regression]") {
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    for (const float threshold : {0.1F, 0.33F, 0.67F, 0.9F}) {
        bool sourcePreviouslyVisible = true;
        bool targetPreviouslyVisible = false;
        size_t sourceChanges = 0;
        size_t targetChanges = 0;
        for (int tick = 0; tick <= 100; ++tick) {
            const float elapsed =
                FAR_TERRAIN_LOD_TRANSITION_SECONDS * static_cast<float>(tick) / 100.0F;
            const float progress = sampleFarTerrainTransition(elapsed).progress;
            const bool sourceVisible = farTerrainLodCanopyVisible(progress, threshold, SOURCE);
            const bool targetVisible = farTerrainLodCanopyVisible(progress, threshold, TARGET);
            CAPTURE(threshold, tick, progress, sourceVisible, targetVisible);
            REQUIRE((sourceVisible || targetVisible));
            REQUIRE_FALSE((!sourcePreviouslyVisible && sourceVisible));
            REQUIRE_FALSE((targetPreviouslyVisible && !targetVisible));
            sourceChanges += sourceVisible != sourcePreviouslyVisible ? 1 : 0;
            targetChanges += targetVisible != targetPreviouslyVisible ? 1 : 0;
            sourcePreviouslyVisible = sourceVisible;
            targetPreviouslyVisible = targetVisible;
        }
        REQUIRE(sourceChanges == 1);
        REQUIRE(targetChanges == 1);
        REQUIRE_FALSE(farTerrainLodCanopyVisible(1.0F, threshold, SOURCE));
        REQUIRE(farTerrainLodCanopyVisible(1.0F, threshold, TARGET));
        REQUIRE(farTerrainLodCanopyVisible(0.5F, threshold, SOURCE));
        REQUIRE(farTerrainLodCanopyVisible(0.5F, threshold, TARGET));
    }

    // The nearby step-32 emergency parent swaps its terrain earlier, but its
    // canopy still uses the same overlap contract for the full transition.
    constexpr unsigned int EMERGENCY_SOURCE = SOURCE | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    constexpr unsigned int EMERGENCY_TARGET = TARGET | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    for (const float threshold : {0.05F, 0.25F, 0.5F, 0.75F, 0.95F}) {
        bool sourceVisibleLast = true;
        bool targetVisibleLast = false;
        for (int tick = 0; tick <= 100; ++tick) {
            const float progress = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS *
                                                              static_cast<float>(tick) / 100.0F)
                                       .progress;
            const bool sourceVisible =
                farTerrainLodCanopyVisible(progress, threshold, EMERGENCY_SOURCE);
            const bool targetVisible =
                farTerrainLodCanopyVisible(progress, threshold, EMERGENCY_TARGET);
            CAPTURE(threshold, tick, progress, sourceVisible, targetVisible);
            REQUIRE((sourceVisible || targetVisible));
            REQUIRE_FALSE((!sourceVisibleLast && sourceVisible));
            REQUIRE_FALSE((targetVisibleLast && !targetVisible));
            sourceVisibleLast = sourceVisible;
            targetVisibleLast = targetVisible;
        }
    }

    bool exactOwned = false;
    bool previouslyExactOwned = false;
    for (const auto [built, current] :
         {std::pair{4U, 5U}, std::pair{5U, 5U}, std::pair{5U, 6U}, std::pair{6U, 7U}}) {
        exactOwned = farTerrainExactSectionOwnsSurface(exactOwned, built, current);
        const bool farOwned = !exactOwned;
        CAPTURE(built, current, exactOwned, farOwned);
        REQUIRE(exactOwned != farOwned);
        REQUIRE(static_cast<unsigned int>(exactOwned) + static_cast<unsigned int>(farOwned) == 1U);
        REQUIRE_FALSE((previouslyExactOwned && !exactOwned));
        if (built == current)
            REQUIRE(exactOwned);
        previouslyExactOwned = exactOwned;
    }
    REQUIRE(exactOwned);
}

TEST_CASE("Far ownership requires every exact surface section in a column",
          "[render][far-terrain][coverage][exact][ownership][revision]") {
    constexpr std::array required{
        ChunkPos{0, 3, 0},   ChunkPos{0, 4, 0},   ChunkPos{1, 3, 0},     ChunkPos{1, 4, 0},
        ChunkPos{15, 3, 15}, ChunkPos{-1, 3, -1}, ChunkPos{-16, 3, -16}, ChunkPos{-17, 3, -17},
    };
    std::unordered_set<ChunkPos> ready(required.begin(), required.end());
    ready.erase(ChunkPos{1, 4, 0});
    constexpr std::array unresolved{ColumnPos{15, 15}};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved,
                               [&](ChunkPos position) { return ready.contains(position); });

    REQUIRE(handoff.columnFullyReady({0, 0}));
    REQUIRE_FALSE(handoff.columnFullyReady({1, 0}));
    REQUIRE_FALSE(handoff.columnFullyReady({15, 15}));
    REQUIRE(handoff.columnFullyReady({-1, -1}));
    REQUIRE(handoff.columnFullyReady({-16, -16}));
    REQUIRE(handoff.columnFullyReady({-17, -17}));

    const FarTerrainExactHandoff::ColumnMask positive = handoff.readyColumnMask({0, 0});
    REQUIRE((positive[0] & 1U) != 0U);
    REQUIRE((positive[0] & (1U << 1U)) == 0U);
    REQUIRE((positive[7] & (1U << 31U)) == 0U);

    // Floor division maps negative boundaries to the same far tile and mask
    // bit used by generation, streaming, and the fragment lookup.
    const FarTerrainExactHandoff::ColumnMask negativeOne = handoff.readyColumnMask({-1, -1});
    REQUIRE((negativeOne[7] & (1U << 31U)) != 0U);
    REQUIRE((negativeOne[0] & 1U) != 0U);
    const FarTerrainExactHandoff::ColumnMask negativeTwo = handoff.readyColumnMask({-2, -2});
    REQUIRE((negativeTwo[7] & (1U << 31U)) != 0U);
}

TEST_CASE("Submerged exact columns retain their parent until floor and water are ready",
          "[render][far-terrain][coverage][exact][ownership][water][floor][regression]") {
    // Keep the fixture pinned to a genuinely deep generated-water column so
    // the test exercises independent floor and water readiness.
    constexpr int64_t WORLD_X = -8'348;
    constexpr int64_t WORLD_Z = 2'281;
    const ColumnPos column{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunk(WORLD_Z)};
    ChunkGenerator generator(42);
    const auto plan = generator.getColumnPlan(column);
    const int localX = Chunk::worldToLocal(WORLD_X);
    const int localZ = Chunk::worldToLocal(WORLD_Z);
    const int32_t floorSection = Chunk::worldToChunkY(plan->surfaceY(localX, localZ));
    const worldgen::SurfaceSample surface = generator.sampleSurface(WORLD_X, WORLD_Z);
    REQUIRE((surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake));
    const int32_t waterSection =
        Chunk::worldToChunkY(static_cast<int>(std::ceil(surface.waterSurface)) - 1);
    REQUIRE(waterSection > floorSection);
    REQUIRE(plan->exposesSection(floorSection));
    REQUIRE(plan->exposesSection(waterSection));

    std::vector<ChunkPos> required;
    required.reserve(plan->exposedSections().size());
    for (const int32_t section : plan->exposedSections()) {
        required.push_back({column.x, section, column.z});
    }
    const auto handoffWithMissing = [&](int32_t missingSection) {
        return farTerrainExactHandoff(
            static_cast<double>(WORLD_X), static_cast<double>(WORLD_Z), 32, required, {},
            [&](ChunkPos position) { return position.y != missingSection; });
    };
    REQUIRE_FALSE(handoffWithMissing(floorSection).columnFullyReady(column));
    REQUIRE_FALSE(handoffWithMissing(waterSection).columnFullyReady(column));
    REQUIRE(farTerrainExactHandoff(static_cast<double>(WORLD_X), static_cast<double>(WORLD_Z), 32,
                                   required, {}, [](ChunkPos) { return true; })
                .columnFullyReady(column));
}

TEST_CASE("Exact and far ownership select one surface per ready column",
          "[render][far-terrain][coverage][exact][ownership]") {
    std::vector<ChunkPos> required;
    required.reserve(16 * 16 * 2);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            required.push_back({x, 3, z});
            required.push_back({x, 4, z});
        }
    }
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(128.0, 128.0, 32, required, {}, [](ChunkPos) { return true; });
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            const bool exactOwner = handoff.columnFullyReady({x, z});
            const bool farOwner = !exactOwner;
            CAPTURE(x, z);
            REQUIRE(exactOwner);
            REQUIRE(exactOwner != farOwner);
        }
    }
}

TEST_CASE("The ready camera column masks every far LOD underfoot",
          "[render][far-terrain][coverage][exact][ownership][camera][lod][regression]") {
    constexpr double CAMERA_X = -198.692;
    constexpr double CAMERA_Z = 63.7348;
    constexpr ColumnPos CAMERA_COLUMN{-13, 3};
    constexpr ColumnPos CAMERA_TILE{-1, 0};
    constexpr std::array REQUIRED{
        ChunkPos{CAMERA_COLUMN.x, 4, CAMERA_COLUMN.z},
        ChunkPos{CAMERA_COLUMN.x, 5, CAMERA_COLUMN.z},
        ChunkPos{CAMERA_COLUMN.x + 1, 4, CAMERA_COLUMN.z},
    };
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(CAMERA_X, CAMERA_Z, 32, REQUIRED, {}, [](ChunkPos) { return true; });
    REQUIRE(handoff.columnFullyReady(CAMERA_COLUMN));

    const FarTerrainExactHandoff::ColumnMask mask = handoff.readyColumnMask(CAMERA_TILE);
    constexpr uint32_t LOCAL_X = 3;
    constexpr uint32_t LOCAL_Z = 3;
    constexpr uint32_t BIT = LOCAL_Z * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE + LOCAL_X;
    REQUIRE((mask[BIT / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD] &
             (1U << (BIT % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD))) != 0U);

    // Match the shader's half-open lookup at the camera's horizontal sample.
    // Every far tier shares this ownership mask, so step 32 cannot overwrite
    // a published exact surface in the chunk the player occupies.
    const float localX = static_cast<float>(CAMERA_X - CAMERA_TILE.x * FAR_TERRAIN_TILE_EDGE);
    const float localZ = static_cast<float>(CAMERA_Z - CAMERA_TILE.z * FAR_TERRAIN_TILE_EDGE);
    const uint32_t sampledColumnX = static_cast<uint32_t>(std::floor(localX / CHUNK_EDGE));
    const uint32_t sampledColumnZ = static_cast<uint32_t>(std::floor(localZ / CHUNK_EDGE));
    REQUIRE(sampledColumnX == LOCAL_X);
    REQUIRE(sampledColumnZ == LOCAL_Z);
    const bool farTierVisibleAtCamera = !handoff.columnFullyReady(CAMERA_COLUMN);
    REQUIRE_FALSE(farTierVisibleAtCamera);
}

TEST_CASE("Far terrain risers query exact ownership from their emitting column",
          "[render][far-terrain][coverage][exact][ownership][riser][shader-contract][regression]") {
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::PLUS_X) == FAR_TERRAIN_FACE_PLUS_X);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::MINUS_X) == FAR_TERRAIN_FACE_MINUS_X);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::PLUS_Z) == FAR_TERRAIN_FACE_PLUS_Z);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::MINUS_Z) == FAR_TERRAIN_FACE_MINUS_Z);

    struct OwnershipCell {
        int tileX;
        int tileZ;
        int columnX;
        int columnZ;

        bool operator==(const OwnershipCell&) const = default;
    };
    const auto lookup = [](simd_float2 position) {
        const int tileX = static_cast<int>(
            std::floor(position.x / static_cast<float>(FAR_TERRAIN_TILE_EDGE_BLOCKS)));
        const int tileZ = static_cast<int>(
            std::floor(position.y / static_cast<float>(FAR_TERRAIN_TILE_EDGE_BLOCKS)));
        const float tileLocalX =
            position.x - static_cast<float>(tileX) * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        const float tileLocalZ =
            position.y - static_cast<float>(tileZ) * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        return OwnershipCell{
            tileX,
            tileZ,
            std::clamp(
                static_cast<int>(std::floor(tileLocalX / FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS)), 0,
                FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1),
            std::clamp(
                static_cast<int>(std::floor(tileLocalZ / FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS)), 0,
                FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1),
        };
    };
    struct Fixture {
        FaceNormal face;
        simd_float2 position;
        OwnershipCell emittingCell;
        OwnershipCell halfOpenCell;
    };
    const std::array fixtures = {
        Fixture{FaceNormal::PLUS_X, simd_make_float2(16.0F, 8.0F), {0, 0, 0, 0}, {0, 0, 1, 0}},
        Fixture{FaceNormal::MINUS_X, simd_make_float2(16.0F, 8.0F), {0, 0, 1, 0}, {0, 0, 1, 0}},
        Fixture{FaceNormal::PLUS_X, simd_make_float2(256.0F, 8.0F), {0, 0, 15, 0}, {1, 0, 0, 0}},
        Fixture{FaceNormal::MINUS_X, simd_make_float2(256.0F, 8.0F), {1, 0, 0, 0}, {1, 0, 0, 0}},
        Fixture{FaceNormal::PLUS_Z, simd_make_float2(8.0F, 16.0F), {0, 0, 0, 0}, {0, 0, 0, 1}},
        Fixture{FaceNormal::MINUS_Z, simd_make_float2(8.0F, 16.0F), {0, 0, 0, 1}, {0, 0, 0, 1}},
        Fixture{FaceNormal::PLUS_Z, simd_make_float2(8.0F, 256.0F), {0, 0, 0, 15}, {0, 1, 0, 0}},
        Fixture{FaceNormal::MINUS_Z, simd_make_float2(8.0F, 256.0F), {0, 1, 0, 0}, {0, 1, 0, 0}},
    };

    for (const Fixture& fixture : fixtures) {
        const unsigned int face = static_cast<unsigned int>(fixture.face);
        CAPTURE(face, fixture.position.x, fixture.position.y);
        REQUIRE(farTerrainOpaqueRiserUsesEmittingColumn(face, false, false));
        const simd_float2 emittingSample =
            farTerrainExactOwnershipSamplePosition(fixture.position, face, true);
        REQUIRE(lookup(emittingSample) == fixture.emittingCell);

        // Tops and water pass false directly. Canopies and skirts are excluded
        // by the shared classifier, so all four retain half-open destination
        // ownership at chunk and tile boundaries.
        const simd_float2 halfOpenSample =
            farTerrainExactOwnershipSamplePosition(fixture.position, face, false);
        REQUIRE(halfOpenSample.x == fixture.position.x);
        REQUIRE(halfOpenSample.y == fixture.position.y);
        REQUIRE(lookup(halfOpenSample) == fixture.halfOpenCell);
        REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(face, true, false));
        REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(face, false, true));
    }

    constexpr unsigned int TOP_FACE = static_cast<unsigned int>(FaceNormal::PLUS_Y);
    REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(TOP_FACE, false, false));
    const simd_float2 topPosition = simd_make_float2(16.0F, 256.0F);
    const simd_float2 topSample = farTerrainExactOwnershipSamplePosition(
        topPosition, TOP_FACE, farTerrainOpaqueRiserUsesEmittingColumn(TOP_FACE, false, false));
    REQUIRE(topSample.x == topPosition.x);
    REQUIRE(topSample.y == topPosition.y);
}

TEST_CASE("Seed 764891 coarse protrusions are hidden by exact column ownership",
          "[render][far-terrain][coverage][exact][ownership][regression]") {
    constexpr FarTerrainKey KEY{91, -437, FarTerrainStep::SIXTEEN};
    constexpr int64_t TILE_ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t TILE_ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    std::array<float, 17 * 17> heights{};
    for (int sampleZ = 0; sampleZ <= 16; ++sampleZ) {
        for (int sampleX = 0; sampleX <= 16; ++sampleX) {
            const FarSurfaceSample sample =
                source.sample(TILE_ORIGIN_X + sampleX * 16, TILE_ORIGIN_Z + sampleZ * 16,
                              worldgen::SurfaceFootprint::BLOCK_16);
            heights[static_cast<size_t>(sampleZ * 17 + sampleX)] =
                static_cast<float>(static_cast<float16_t>(sample.footprintMinimumTerrainHeight));
        }
    }

    std::vector<ChunkPos> required;
    required.reserve(16 * 16);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            required.push_back({KEY.tileX * 16 + x, 14, KEY.tileZ * 16 + z});
        }
    }
    const FarTerrainExactHandoff handoff = farTerrainExactHandoff(
        23'029.0, -111'726.0, 32, required, {}, [](ChunkPos) { return true; });

    size_t rawProtrusions = 0;
    size_t visibleProtrusions = 0;
    for (int cellZ = 0; cellZ < 16; ++cellZ) {
        for (int cellX = 0; cellX < 16; ++cellX) {
            const float northwest = heights[static_cast<size_t>(cellZ * 17 + cellX)];
            const float northeast = heights[static_cast<size_t>(cellZ * 17 + cellX + 1)];
            const float southwest = heights[static_cast<size_t>((cellZ + 1) * 17 + cellX)];
            const float southeast = heights[static_cast<size_t>((cellZ + 1) * 17 + cellX + 1)];
            const ColumnPos chunkColumn{KEY.tileX * 16 + cellX, KEY.tileZ * 16 + cellZ};
            for (int blockZ = 0; blockZ < 16; ++blockZ) {
                for (int blockX = 0; blockX < 16; ++blockX) {
                    const float u = (static_cast<float>(blockX) + 0.5F) / 16.0F;
                    const float v = (static_cast<float>(blockZ) + 0.5F) / 16.0F;
                    const float coarse =
                        u <= v
                            ? northwest + v * (southwest - northwest) + u * (southeast - southwest)
                            : northwest + u * (northeast - northwest) + v * (southeast - northeast);
                    const int64_t worldX = TILE_ORIGIN_X + cellX * 16 + blockX;
                    const int64_t worldZ = TILE_ORIGIN_Z + cellZ * 16 + blockZ;
                    const double exact =
                        generator->sampleExactSurface(worldX, worldZ).terrainHeight;
                    if (coarse <= exact + 1.0e-5)
                        continue;
                    ++rawProtrusions;
                    if (!handoff.columnFullyReady(chunkColumn))
                        ++visibleProtrusions;
                }
            }
        }
    }
    INFO("raw coarse protrusions " << rawProtrusions);
    REQUIRE(visibleProtrusions == 0);
}

TEST_CASE("Far terrain exact handoff uses unresolved column AABBs",
          "[render][far-terrain][coverage][exact]") {
    REQUIRE(farTerrainColumnDistanceSquared(8.0, 8.0, {0, 0}) == 0.0);
    REQUIRE(farTerrainColumnDistanceSquared(-1.0, -1.0, {0, 0}) == Catch::Approx(2.0));
    REQUIRE(farTerrainColumnDistanceSquared(40.0, 8.0, {1, 0}) == Catch::Approx(64.0));
    REQUIRE(farTerrainColumnDistanceSquared(-40.0, -8.0, {-2, -1}) == Catch::Approx(64.0));
}

TEST_CASE("Ready exact tiles retain ownership when another tile is stale",
          "[render][far-terrain][coverage][exact][flicker][regression]") {
    constexpr std::array required = {
        ChunkPos{0, 4, 0},  ChunkPos{1, 4, 0},  ChunkPos{-1, 4, -1},
        ChunkPos{16, 4, 0}, ChunkPos{17, 4, 0}, ChunkPos{32, 4, 0},
    };
    const std::unordered_set<ChunkPos> ready = {
        required[0], required[1], required[2], required[3], required[5],
    };
    constexpr std::array unresolved = {ColumnPos{48, 0}};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved,
                               [&](ChunkPos position) { return ready.contains(position); });

    REQUIRE(handoff.tileFullyReady({0, 0}));
    REQUIRE(handoff.tileFullyReady({-1, -1}));
    REQUIRE_FALSE(handoff.tileFullyReady({1, 0}));
    REQUIRE(handoff.tileFullyReady({2, 0}));
    REQUIRE_FALSE(handoff.tileFullyReady({3, 0}));
    REQUIRE_FALSE(handoff.tileFullyReady({4, 0}));

    constexpr float NOMINAL = 32.0F * CHUNK_EDGE;
    REQUIRE(handoff.distanceBlocksForTile({0, 0}, NOMINAL) == NOMINAL);
    REQUIRE(handoff.distanceBlocksForTile({2, 0}, NOMINAL) == NOMINAL);
    REQUIRE(handoff.distanceBlocksForTile({1, 0}, NOMINAL) == handoff.distanceBlocks);
    REQUIRE(handoff.distanceBlocksForTile({3, 0}, NOMINAL) == 48.0F * CHUNK_EDGE);
    REQUIRE(handoff.distanceBlocksForTile({4, 0}, NOMINAL) == handoff.distanceBlocks);

    REQUIRE_FALSE(handoff.tileFullyOwned({0, 0}));
    REQUIRE_FALSE(handoff.tileFullyOwned({2, 0}));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {0, 0}, NOMINAL, handoff));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, NOMINAL, handoff));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {2, 0}, NOMINAL, handoff));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {4, 0}, NOMINAL, handoff));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, 0.0F, handoff));
}

TEST_CASE("Only complete exact tile ownership releases fine boundary fallback",
          "[render][far-terrain][coverage][exact][lod][flicker][regression]") {
    std::vector<ChunkPos> completeTile;
    completeTile.reserve(FAR_TERRAIN_EXACT_COLUMNS_PER_TILE * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
    for (int64_t z = 0; z < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++z) {
        for (int64_t x = 0; x < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++x) {
            completeTile.push_back({x, 4, z});
        }
    }
    const FarTerrainExactHandoff complete =
        farTerrainExactHandoff(0.0, 0.0, 32, completeTile, {}, [](ChunkPos) { return true; });
    REQUIRE(complete.tileFullyReady({0, 0}));
    REQUIRE(complete.tileFullyOwned({0, 0}));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {0, 0}, 32.0F * CHUNK_EDGE, complete));

    constexpr std::array partialBoundary{ChunkPos{31, 4, 0}};
    const FarTerrainExactHandoff partial =
        farTerrainExactHandoff(0.0, 0.0, 32, partialBoundary, {}, [](ChunkPos) { return true; });
    REQUIRE(partial.tileFullyReady({1, 0}));
    REQUIRE_FALSE(partial.tileFullyOwned({1, 0}));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, 32.0F * CHUNK_EDGE, partial));
}

TEST_CASE("Far terrain scheduler defaults retain the full base and refinement set",
          "[render][far-terrain][scheduler][coverage]") {
    STATIC_REQUIRE(FarTerrainScheduler::WORKER_COUNT == 8);
    STATIC_REQUIRE(FarTerrainScheduler::LATENCY_WORKER_COUNT == 4);
    const FarTerrainSchedulerLimits limits;
    REQUIRE(limits.maxPending == 64);
    REQUIRE(limits.maxCompleted == 32);
    REQUIRE(limits.maxCacheEntries == 9280);
    REQUIRE(limits.maxCacheBytes == 3ull * 1024 * 1024 * 1024);
}

TEST_CASE("Far terrain scheduler exposes the finest useful cached refinement",
          "[render][far-terrain][scheduler][cache][lod][priority][regression]") {
    FarTerrainScheduler scheduler(farTerrainTestSource());
    constexpr ColumnPos COORDINATE{0, 0};
    for (FarTerrainStep step :
         {FarTerrainStep::SIXTEEN, FarTerrainStep::FOUR, FarTerrainStep::TWO}) {
        REQUIRE(scheduler.enqueue({COORDINATE.x, COORDINATE.z, step}));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(scheduler.stats().inFlight == 0);

    constexpr FarTerrainStepMask BASE_READY = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE_FALSE(scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                             FarTerrainStep::TWO, BASE_READY, true));
    const auto finest = scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_READY);
    REQUIRE(finest);
    REQUIRE(finest->key.step == FarTerrainStep::TWO);

    const auto distanceTier = scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                                         FarTerrainStep::FOUR, BASE_READY);
    REQUIRE(distanceTier);
    REQUIRE(distanceTier->key.step == FarTerrainStep::FOUR);

    REQUIRE_FALSE(scheduler.findFinestCached(COORDINATE, FarTerrainStep::TWO, FarTerrainStep::TWO,
                                             BASE_READY));
}

TEST_CASE("Production coverage minima stay below exact emitted surfaces",
          "[render][far-terrain][coverage][bounds][worldgen]") {
    struct Fixture {
        std::shared_ptr<ChunkGenerator> generator;
        int64_t centerX;
        int64_t centerZ;
        const char* name;
    };
    auto ordinary = std::make_shared<ChunkGenerator>(42);
    auto volcanic = std::make_shared<ChunkGenerator>(764891);
    const std::array fixtures{
        Fixture{ordinary, -513, -257, "negative dry terrain"},
        Fixture{ordinary, -8'352, 2'160, "negative lake shoreline"},
        Fixture{ordinary, -12'289, 2'653, "negative river"},
        Fixture{volcanic, 23'029, -111'486, "caldera lake"},
    };

    bool sawSurfaceWater = false;
    bool sawVolcanicTerrain = false;
    for (const Fixture& fixture : fixtures) {
        const FarTerrainSource source =
            FarTerrainMesher::generatorGeometrySource(fixture.generator);
        for (int64_t dz = -4; dz <= 4; dz += 2) {
            for (int64_t dx = -4; dx <= 4; dx += 2) {
                const int64_t x = fixture.centerX + dx;
                const int64_t z = fixture.centerZ + dz;
                const worldgen::SurfaceSample exact = fixture.generator->sampleExactSurface(x, z);
                const worldgen::SurfaceSample filtered =
                    fixture.generator->sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_16);
                const worldgen::SurfaceSample canonicalWater =
                    fixture.generator->sampleFarGeometrySurface(
                        x, z, worldgen::SurfaceFootprint::BLOCK_1);
                const FarSurfaceSample coverage =
                    source.sample(x, z, worldgen::SurfaceFootprint::BLOCK_16);
                CAPTURE(fixture.name, x, z, exact.terrainHeight, filtered.terrainHeight,
                        coverage.footprintMinimumTerrainHeight);
                double expectedMinimum = filtered.terrainHeight -
                                         FAR_TERRAIN_STEP16_RELIEF_ENVELOPE -
                                         ChunkGenerator::emittedSurfaceDetailAmplitude(
                                             filtered, FAR_TERRAIN_STEP16_RELIEF_SLOPE_ENVELOPE) -
                                         FAR_TERRAIN_EMITTED_SURFACE_ENVELOPE;
                const bool standingWater = canonicalWater.hydrology.ocean ||
                                           canonicalWater.hydrology.river ||
                                           canonicalWater.hydrology.lake;
                if (standingWater) {
                    expectedMinimum =
                        std::min(expectedMinimum, canonicalWater.hydrology.surfaceElevation);
                    expectedMinimum = std::min(
                        expectedMinimum, std::ceil(canonicalWater.hydrology.waterSurface) - 1.0);
                }
                if (canonicalWater.hydrology.waterfall &&
                    canonicalWater.hydrology.waterfallTop >=
                        canonicalWater.hydrology.waterfallBottom + 0.5) {
                    expectedMinimum = std::min(
                        expectedMinimum, std::ceil(canonicalWater.hydrology.waterfallBottom) - 1.0);
                }
                expectedMinimum = std::min(expectedMinimum, exact.terrainHeight);
                REQUIRE(coverage.footprintMinimumTerrainHeight == Catch::Approx(expectedMinimum));
                REQUIRE(coverage.footprintMinimumTerrainHeight <= exact.terrainHeight + 1.0e-6);
                sawSurfaceWater = sawSurfaceWater || exact.hydrology.ocean ||
                                  exact.hydrology.river || exact.hydrology.lake;
                sawVolcanicTerrain = sawVolcanicTerrain || exact.geology.volcanicActivity > 0.5;
            }
        }
    }
    REQUIRE(sawSurfaceWater);
    REQUIRE(sawVolcanicTerrain);
}

TEST_CASE("Production far terrain uses generator-owned material palettes and ranks",
          "[render][far-terrain][material][worldgen][determinism]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.materialRank);

    constexpr int64_t WORLD_X = 23'029;
    constexpr int64_t WORLD_Z = -111'486;
    const worldgen::SurfaceSample surface =
        generator->sampleFarSurface(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8);
    const FarSurfaceSample sampled =
        source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8);
    REQUIRE(sameMaterialPalette(sampled.materialPalette,
                                generator->farSurfaceMaterialPaletteAt(WORLD_X, WORLD_Z, surface)));
    REQUIRE(source.materialRank(WORLD_X, WORLD_Z) ==
            Catch::Approx(generator->farSurfaceMaterialRankAt(WORLD_X, WORLD_Z)));

    std::array<FarSurfaceSample, 9> grid{};
    source.sampleGrid(WORLD_X - 8, WORLD_Z - 8, 8, 3, worldgen::SurfaceFootprint::BLOCK_8, grid);
    for (int z = 0; z < 3; ++z) {
        for (int x = 0; x < 3; ++x) {
            const int64_t sampleX = WORLD_X - 8 + x * 8;
            const int64_t sampleZ = WORLD_Z - 8 + z * 8;
            const worldgen::SurfaceSample expectedSurface =
                generator->sampleFarSurface(sampleX, sampleZ, worldgen::SurfaceFootprint::BLOCK_8);
            CAPTURE(sampleX, sampleZ);
            REQUIRE(sameMaterialPalette(
                grid[static_cast<size_t>(z * 3 + x)].materialPalette,
                generator->farSurfaceMaterialPaletteAt(sampleX, sampleZ, expectedSurface)));
        }
    }
}

TEST_CASE("The first far LOD reduces exact terrain to flat voxel terraces",
          "[render][far-terrain][seam][exact][voxel]") {
    ChunkGenerator generator(42);
    constexpr int64_t WORLD_X = -81'792;
    constexpr int64_t WORLD_Z = 126'976;
    const worldgen::SurfaceSample planned = generator.sampleSurface(WORLD_X, WORLD_Z);
    const worldgen::SurfaceSample coarse =
        generator.sampleFarSurface(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(WORLD_X, WORLD_Z);
    // This fixture must resolve to a different emitted density voxel so the
    // test can distinguish the exact handoff callback from its macro parent.
    // The bounded density detail contract now caps that displacement well
    // below the former twenty-block fixture delta.
    REQUIRE(std::abs(coarse.terrainHeight - exact.terrainHeight) > 2.0);
    // The public two-coordinate wrapper is block-resolution authority and
    // therefore agrees with exact cube emission. Only the explicit far
    // sampler returns the filtered macro parent used beyond exact residency.
    REQUIRE(planned.terrainHeight == Catch::Approx(exact.terrainHeight).margin(1.0e-4));
    REQUIRE(exact.terrainHeight == generator.surfaceYAt(WORLD_X, WORLD_Z) + 1.0);

    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) {
            return generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_16);
        });
    REQUIRE(source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_1)
                .geometry.terrainHeight == exact.terrainHeight);
    REQUIRE(source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8)
                .geometry.terrainHeight == coarse.terrainHeight);
    constexpr FarTerrainKey KEY{world_coord::floorDiv(WORLD_X, int64_t{FAR_TERRAIN_TILE_EDGE}),
                                world_coord::floorDiv(WORLD_Z, int64_t{FAR_TERRAIN_TILE_EDGE}),
                                FarTerrainStep::ONE};
    const auto mesh = FarTerrainMesher::build(KEY, source);
    REQUIRE(farTerrainTopsAreVoxelFlat(*mesh));
    const int64_t localCellX = WORLD_X - KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    const int64_t localCellZ = WORLD_Z - KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    const float expectedHeight =
        expectedVoxelCellHeight(source, WORLD_X, WORLD_Z, FarTerrainStep::ONE);
    const std::optional<float> emittedHeight = farTerrainHeightAt(
        *mesh, static_cast<float>(localCellX) + 0.5F, static_cast<float>(localCellZ) + 0.5F);
    REQUIRE(emittedHeight);
    REQUIRE(*emittedHeight == expectedHeight);
}

TEST_CASE("Far LOD reduces horizontal voxel resolution without sloped faces",
          "[render][far-terrain][lod][voxel][regression]") {
    const FarTerrainSource source = farTerrainTestSource();
    for (const FarTerrainStep step :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const int width = farTerrainStepSize(step);
        const auto mesh = FarTerrainMesher::build({-1, 2, step}, source);
        CAPTURE(width, mesh->terrainQuadCount, mesh->mergedTerrainCellCount);
        REQUIRE(farTerrainUsesVoxelFaces(*mesh, width));
        REQUIRE(mesh->mergedTerrainCellCount ==
                static_cast<uint32_t>((FAR_TERRAIN_TILE_EDGE / width) *
                                      (FAR_TERRAIN_TILE_EDGE / width)));
    }
}

TEST_CASE("Fallback voxel tops remain independent of conservative footprint minima",
          "[render][far-terrain][lod][voxel][transition][bounds][regression]") {
    FarTerrainSource source;
    source.sample = [](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        const double height = 92.0 + std::sin(static_cast<double>(x) * 0.071) * 7.0 +
                              std::cos(static_cast<double>(z) * 0.053) * 5.0;
        const double support = static_cast<double>(worldgen::surfaceFootprintWidth(footprint));
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = height;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = height - support * 0.75,
            .footprintMaximumTerrainHeight = height + support * 0.75,
            .materialPalette = testMaterialPalette(BlockType::STONE),
        };
    };

    constexpr std::array TIERS = {FarTerrainStep::ONE,     FarTerrainStep::TWO,
                                  FarTerrainStep::FOUR,    FarTerrainStep::EIGHT,
                                  FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO};
    std::array<std::shared_ptr<const FarTerrainMesh>, TIERS.size()> meshes;
    for (size_t index = 0; index < TIERS.size(); ++index) {
        meshes[index] = FarTerrainMesher::build({0, 0, TIERS[index]}, source);
    }
    bool sawSeparatedCoverageMinimum = false;
    for (size_t tierIndex = 1; tierIndex < TIERS.size(); ++tierIndex) {
        const FarTerrainStep tier = TIERS[tierIndex];
        const int step = farTerrainStepSize(tier);
        for (int cellZ = 0; cellZ < FAR_TERRAIN_TILE_EDGE; cellZ += step) {
            for (int cellX = 0; cellX < FAR_TERRAIN_TILE_EDGE; cellX += step) {
                const float x = static_cast<float>(cellX) + 0.5F;
                const float z = static_cast<float>(cellZ) + 0.5F;
                const auto top = farTerrainHeightAt(*meshes[tierIndex], x, z);
                const float expected = expectedVoxelCellHeight(source, cellX, cellZ, tier);
                REQUIRE(top);
                CAPTURE(step, x, z, *top, expected);
                REQUIRE(*top == expected);
                const FarSurfaceSample sample =
                    source.sample(cellX, cellZ, farTerrainSurfaceFootprint(tier));
                sawSeparatedCoverageMinimum = sawSeparatedCoverageMinimum ||
                                              *top - sample.footprintMinimumTerrainHeight > 10.0;
            }
        }
    }
    REQUIRE(sawSeparatedCoverageMinimum);
}

TEST_CASE("Production terrain keeps filtered voxel tops through atomic LOD swaps",
          "[render][far-terrain][lod][voxel][transition][worldgen][regression]") {
    struct Fixture {
        uint64_t seed;
        int64_t tileX;
        int64_t tileZ;
        const char* name;
    };
    constexpr std::array fixtures{
        Fixture{42, -33, 8, "lake shoreline"},
        Fixture{42, 0, -6, "ocean river exact handoff"},
        Fixture{764891, 89, -436, "volcanic caldera"},
    };
    constexpr std::array TIERS = {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
                                  FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO};
    constexpr int RASTER_SPACING = 2;
    for (const Fixture& fixture : fixtures) {
        auto generator = std::make_shared<ChunkGenerator>(fixture.seed);
        FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        source.canopies = {};
        std::array<std::vector<float>, TIERS.size()> heights;
        std::array<double, TIERS.size()> means{};
        for (size_t tier = 0; tier < TIERS.size(); ++tier) {
            const auto mesh =
                FarTerrainMesher::build({fixture.tileX, fixture.tileZ, TIERS[tier]}, source);
            heights[tier] = farTerrainHeightRaster(*mesh, RASTER_SPACING);
            REQUIRE(std::ranges::none_of(heights[tier],
                                         [](float value) { return !std::isfinite(value); }));
            means[tier] = std::accumulate(heights[tier].begin(), heights[tier].end(), 0.0) /
                          static_cast<double>(heights[tier].size());
        }
        for (size_t tier = 1; tier < TIERS.size(); ++tier) {
            CAPTURE(fixture.name, farTerrainStepSize(TIERS[tier]), means[0], means[tier]);
            REQUIRE(std::abs(means[tier] - means[0]) <= 2.0);
        }
    }
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    for (const float progress : {0.0F, 0.25F, 0.499F, 0.5F, 0.75F, 1.0F}) {
        CAPTURE(progress);
        REQUIRE(farTerrainLodTerrainVisible(progress, SOURCE) !=
                farTerrainLodTerrainVisible(progress, TARGET));
    }
}

TEST_CASE("Seed forty-two coverage parents retain their filtered lowland surface",
          "[render][far-terrain][coverage][bounds][lod][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int PARENT_STEP = 32;
    constexpr int CHILD_STEP = 2;
    constexpr int CHILD_SCALE = PARENT_STEP / CHILD_STEP;
    struct Fixture {
        int64_t x;
        int64_t z;
    };
    constexpr std::array FIXTURES = {Fixture{64, -1'632}, Fixture{192, -1'472}};
    for (const Fixture fixture : FIXTURES) {
        std::array<FarTerrainCellBounds, 1> parents{};
        source.cellBoundsGrid(fixture.x, fixture.z, PARENT_STEP, 1, 1,
                              worldgen::SurfaceFootprint::BLOCK_32, parents);
        std::array<worldgen::SurfaceSample, CHILD_SCALE * CHILD_SCALE> children{};
        generator->sampleExactSurfaceGrid(fixture.x, fixture.z, CHILD_STEP, CHILD_SCALE, children);
        const FarTerrainCellBounds& parent = parents.front();
        double maximumAbsoluteError = 0.0;
        for (const worldgen::SurfaceSample& child : children) {
            REQUIRE(child.terrainHeight > SEA_LEVEL + 1.0);
            maximumAbsoluteError = std::max(maximumAbsoluteError,
                                            std::abs(child.terrainHeight - parent.terrainHeight));
        }
        CAPTURE(fixture.x, fixture.z, parent.terrainHeight, parent.minimumTerrainHeight,
                maximumAbsoluteError);
        REQUIRE(parent.minimumTerrainHeight <= SEA_LEVEL);
        REQUIRE(parent.terrainHeight > SEA_LEVEL);
        REQUIRE(maximumAbsoluteError <= 6.0);

        const int64_t tileX = world_coord::floorDiv(fixture.x, int64_t{FAR_TERRAIN_TILE_EDGE});
        const int64_t tileZ = world_coord::floorDiv(fixture.z, int64_t{FAR_TERRAIN_TILE_EDGE});
        const auto mesh =
            FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
        const float localX = static_cast<float>(fixture.x - mesh->originX + PARENT_STEP / 2);
        const float localZ = static_cast<float>(fixture.z - mesh->originZ + PARENT_STEP / 2);
        const std::optional<float> meshTop = farTerrainHeightAt(*mesh, localX, localZ);
        REQUIRE(meshTop);
        REQUIRE(*meshTop == static_cast<float>(std::ceil(parent.terrainHeight)));
    }
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
    const FarTerrainGeometrySample geometry = testFarGeometry(source, LAKE_X, LAKE_Z);
    REQUIRE(geometry.lake);
    REQUIRE(geometry.waterSurface == std::ceil(exact.waterSurface));
    REQUIRE(geometry.waterSurface ==
            std::ceil(exact.waterSurface) - 1.0 + fluidSurfaceHeight(FluidState::source()));
}

TEST_CASE("Seed 42 far ocean coverage has no eight-block grid gaps",
          "[render][far-terrain][water][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    constexpr int64_t CENTER_X = -557;
    constexpr int64_t CENTER_Z = 379;
    constexpr int RADIUS = 32;
    constexpr int SAMPLE_EDGE = RADIUS * 2 + 3;
    constexpr int64_t SAMPLE_ORIGIN_X = CENTER_X - RADIUS - 1;
    constexpr int64_t SAMPLE_ORIGIN_Z = CENTER_Z - RADIUS - 1;
    std::array<FarTerrainGeometrySample, SAMPLE_EDGE * SAMPLE_EDGE> exact{};
    for (int z = 0; z < SAMPLE_EDGE; ++z) {
        for (int x = 0; x < SAMPLE_EDGE; ++x) {
            exact[static_cast<size_t>(z * SAMPLE_EDGE + x)] =
                source
                    .sample(SAMPLE_ORIGIN_X + x, SAMPLE_ORIGIN_Z + z,
                            worldgen::SurfaceFootprint::BLOCK_1)
                    .geometry;
        }
    }
    const auto wetAt = [&](int x, int z) {
        const FarTerrainGeometrySample& sample =
            exact[static_cast<size_t>((z + RADIUS + 1) * SAMPLE_EDGE + x + RADIUS + 1)];
        return (sample.ocean || sample.river || sample.lake) &&
               sample.waterSurface > sample.terrainHeight + 0.01;
    };

    constexpr FarTerrainKey TILE{-3, 1, FarTerrainStep::TWO};
    // Step 32 intentionally uses one exact authority representative per
    // aligned 8x8 coverage cell and has dedicated ownership tests below.
    for (const FarTerrainStep step :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({TILE.tileX, TILE.tileZ, step}, source);
        size_t expectedWet = 0;
        size_t missing = 0;
        std::array<size_t, 64> missingByEightBlockPhase{};
        for (int dz = -RADIUS; dz <= RADIUS; ++dz) {
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                bool broadWater = true;
                for (int neighborZ = -1; neighborZ <= 1; ++neighborZ) {
                    for (int neighborX = -1; neighborX <= 1; ++neighborX)
                        broadWater = broadWater && wetAt(dx + neighborX, dz + neighborZ);
                }
                if (!broadWater)
                    continue;
                ++expectedWet;
                const int64_t worldX = CENTER_X + dx;
                const int64_t worldZ = CENTER_Z + dz;
                const float localX = static_cast<float>(worldX - mesh->originX) + 0.5F;
                const float localZ = static_cast<float>(worldZ - mesh->originZ) + 0.5F;
                if (farWaterTopCovers(*mesh, localX, localZ))
                    continue;
                ++missing;
                const size_t phaseX =
                    static_cast<size_t>(world_coord::floorMod(worldX, int64_t{8}));
                const size_t phaseZ =
                    static_cast<size_t>(world_coord::floorMod(worldZ, int64_t{8}));
                ++missingByEightBlockPhase[phaseZ * 8 + phaseX];
            }
        }
        CAPTURE(farTerrainStepSize(step), expectedWet, missing, missingByEightBlockPhase);
        REQUIRE(expectedWet > 1'000);
        REQUIRE(missing == 0);
    }
}

TEST_CASE("Seed 42 step 32 water respects exact ownership through a cold handoff",
          "[render][far-terrain][water][coverage][ownership][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    const auto geometryPointCount = std::make_shared<size_t>(0);
    const auto authorityPointCount = std::make_shared<size_t>(0);
    const auto authorityGridCount = std::make_shared<size_t>(0);
    const auto geometryPoints = source.geometryPoints;
    source.geometryPoints = [geometryPointCount,
                             geometryPoints](std::span<const ColumnPos> positions,
                                             worldgen::SurfaceFootprint footprint,
                                             std::span<FarTerrainGeometrySample> output) {
        if (footprint == worldgen::SurfaceFootprint::BLOCK_1)
            *geometryPointCount += positions.size();
        geometryPoints(positions, footprint, output);
    };
    const auto authorityPoints = source.waterAuthorityPoints;
    source.waterAuthorityPoints = [authorityPointCount,
                                   authorityPoints](std::span<const ColumnPos> positions,
                                                    worldgen::SurfaceFootprint footprint,
                                                    std::span<FarTerrainGeometrySample> output) {
        *authorityPointCount += positions.size();
        authorityPoints(positions, footprint, output);
    };
    const auto authorityGrid = source.waterAuthorityGrid;
    source.waterAuthorityGrid = [authorityGridCount,
                                 authorityGrid](int64_t originX, int64_t originZ, int spacingX,
                                                int spacingZ, int sampleWidth, int sampleHeight,
                                                worldgen::SurfaceFootprint footprint,
                                                std::span<FarTerrainGeometrySample> output) {
        *authorityGridCount += output.size();
        authorityGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight, footprint,
                      output);
    };

    constexpr FarTerrainKey LEFT{-1, -6, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey RIGHT{0, -6, FarTerrainStep::THIRTY_TWO};
    const auto left = FarTerrainMesher::build(LEFT, source);
    const auto right = FarTerrainMesher::build(RIGHT, source);

    constexpr int SAMPLE_STEP = 8;
    constexpr int SAMPLE_EDGE = FAR_TERRAIN_TILE_EDGE / SAMPLE_STEP + 2;
    constexpr int64_t SAMPLE_ORIGIN_X = -SAMPLE_STEP;
    constexpr int64_t SAMPLE_ORIGIN_Z = RIGHT.tileZ * FAR_TERRAIN_TILE_EDGE - SAMPLE_STEP;
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> exact{};
    generator->sampleExactSurfaceGrid(SAMPLE_ORIGIN_X, SAMPLE_ORIGIN_Z, SAMPLE_STEP, SAMPLE_EDGE,
                                      exact);
    const auto exactWet = [&](int x, int z) {
        const worldgen::SurfaceSample& sample = exact[static_cast<size_t>(z * SAMPLE_EDGE + x)];
        return (sample.hydrology.ocean || sample.hydrology.river || sample.hydrology.lake) &&
               sample.hydrology.waterSurface > sample.terrainHeight + 0.01;
    };
    size_t broadDry = 0;
    size_t broadWet = 0;
    size_t falseWater = 0;
    size_t missingWater = 0;
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        for (int x = 1; x + 1 < SAMPLE_EDGE; ++x) {
            bool neighborhoodDry = true;
            bool neighborhoodWet = true;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    neighborhoodDry = neighborhoodDry && !exactWet(x + dx, z + dz);
                    neighborhoodWet = neighborhoodWet && exactWet(x + dx, z + dz);
                }
            }
            const float localX = static_cast<float>((x - 1) * SAMPLE_STEP) + 0.5F;
            const float localZ = static_cast<float>((z - 1) * SAMPLE_STEP) + 0.5F;
            const bool meshWet = farWaterTopCovers(*right, localX, localZ);
            broadDry += neighborhoodDry;
            broadWet += neighborhoodWet;
            falseWater += neighborhoodDry && meshWet;
            missingWater += neighborhoodWet && !meshWet;
        }
    }
    CAPTURE(broadDry, broadWet, falseWater, missingWater, *geometryPointCount, *authorityPointCount,
            *authorityGridCount, right->waterQuadCount, right->waterContourTriangleCount);
    REQUIRE(broadDry > 100);
    REQUIRE(broadWet > 100);
    REQUIRE(falseWater == 0);
    REQUIRE(missingWater == 0);
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        for (int x = 1; x + 1 < SAMPLE_EDGE; ++x) {
            const float localX = static_cast<float>((x - 1) * SAMPLE_STEP) + 0.5F;
            const float localZ = static_cast<float>((z - 1) * SAMPLE_STEP) + 0.5F;
            CAPTURE(x, z, localX, localZ);
            REQUIRE(farWaterTopCovers(*right, localX, localZ) == exactWet(x, z));
        }
    }
    REQUIRE(left->waterContourTriangleCount == 0);
    REQUIRE(right->waterContourTriangleCount == 0);

    size_t sharedWet = 0;
    size_t sharedDry = 0;
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        bool neighborhoodDry = true;
        bool neighborhoodWet = true;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int x = 0; x <= 2; ++x) {
                neighborhoodDry = neighborhoodDry && !exactWet(x, z + dz);
                neighborhoodWet = neighborhoodWet && exactWet(x, z + dz);
            }
        }
        if (!neighborhoodDry && !neighborhoodWet)
            continue;
        const int localZ = (z - 1) * SAMPLE_STEP;
        const float sampleZ = static_cast<float>(localZ) + 0.5F;
        const bool leftWet =
            farWaterTopCovers(*left, static_cast<float>(FAR_TERRAIN_TILE_EDGE), sampleZ);
        const bool rightWet = farWaterTopCovers(*right, 0.0F, sampleZ);
        CAPTURE(localZ, neighborhoodDry, neighborhoodWet, leftWet, rightWet);
        REQUIRE((leftWet || rightWet) == neighborhoodWet);
        sharedWet += neighborhoodWet;
        sharedDry += neighborhoodDry;
    }
    REQUIRE(sharedWet > 4);
    REQUIRE(sharedDry > 0);
    REQUIRE(*authorityGridCount == 2 * 33 * 33);
    REQUIRE(*authorityPointCount == 0);
    REQUIRE(*geometryPointCount + *authorityPointCount + *authorityGridCount < 4'096);
}

TEST_CASE("Seed 42 step 32 water cells retain phase zero authority across twenty five tiles",
          "[render][far-terrain][water][coverage][ownership][seam][voxel][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int64_t FIRST_TILE_X = -1;
    constexpr int64_t FIRST_TILE_Z = -8;
    constexpr int TILE_EDGE = 5;
    constexpr int WATER_STEP = 8;
    constexpr int WATER_CELLS = FAR_TERRAIN_TILE_EDGE / WATER_STEP;
    constexpr int AUTHORITY_EDGE = TILE_EDGE * WATER_CELLS + 1;
    constexpr int64_t AUTHORITY_ORIGIN_X = FIRST_TILE_X * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t AUTHORITY_ORIGIN_Z = FIRST_TILE_Z * FAR_TERRAIN_TILE_EDGE;
    std::vector<worldgen::HydrologySample> authority(
        static_cast<size_t>(AUTHORITY_EDGE * AUTHORITY_EDGE));
    generator->sampleGeneratedWaterAuthorityGrid(AUTHORITY_ORIGIN_X, AUTHORITY_ORIGIN_Z, WATER_STEP,
                                                 AUTHORITY_EDGE, authority);
    const auto authorityAt = [&](int tileX, int tileZ, int cellX, int cellZ) {
        const int sampleX = (tileX - FIRST_TILE_X) * WATER_CELLS + cellX;
        const int sampleZ = (tileZ - FIRST_TILE_Z) * WATER_CELLS + cellZ;
        return authority[static_cast<size_t>(sampleZ * AUTHORITY_EDGE + sampleX)];
    };
    size_t wetCells = 0;
    size_t dryCells = 0;
    for (int tileZ = FIRST_TILE_Z; tileZ < FIRST_TILE_Z + TILE_EDGE; ++tileZ) {
        for (int tileX = FIRST_TILE_X; tileX < FIRST_TILE_X + TILE_EDGE; ++tileX) {
            const auto mesh =
                FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
            CAPTURE(tileX, tileZ, mesh->waterQuadCount, mesh->waterContourTriangleCount);
            REQUIRE(mesh->waterContourTriangleCount == 0);
            for (int cellZ = 0; cellZ < WATER_CELLS; ++cellZ) {
                for (int cellX = 0; cellX < WATER_CELLS; ++cellX) {
                    const worldgen::HydrologySample hydrology =
                        authorityAt(tileX, tileZ, cellX, cellZ);
                    const bool expectedWet =
                        (hydrology.ocean || hydrology.river || hydrology.lake) &&
                        hydrology.waterSurface > hydrology.surfaceElevation + 0.01;
                    const float localX = static_cast<float>(cellX * WATER_STEP) + 0.5F;
                    const float localZ = static_cast<float>(cellZ * WATER_STEP) + 0.5F;
                    CAPTURE(cellX, cellZ, localX, localZ, expectedWet, hydrology.waterBodyId,
                            hydrology.waterSurface, hydrology.surfaceElevation);
                    REQUIRE(farWaterTopCovers(*mesh, localX, localZ) == expectedWet);
                    wetCells += expectedWet;
                    dryCells += !expectedWet;
                }
            }
            for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size();
                 offset += 3) {
                const Vertex& first = mesh->vertices[mesh->indices[offset]];
                if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
                    unpackFluidFalling(first.faceAttr)) {
                    continue;
                }
                const Vertex& second = mesh->vertices[mesh->indices[offset + 1]];
                const Vertex& third = mesh->vertices[mesh->indices[offset + 2]];
                CAPTURE(tileX, tileZ, static_cast<float>(first.px), static_cast<float>(first.py),
                        static_cast<float>(first.pz));
                REQUIRE(static_cast<float>(first.py) == static_cast<float>(second.py));
                REQUIRE(static_cast<float>(first.py) == static_cast<float>(third.py));
                REQUIRE(std::fmod(static_cast<float>(first.py) * 8.0F, 1.0F) ==
                        Catch::Approx(0.0F).margin(1.0e-4F));
                const float signedArea =
                    (static_cast<float>(second.px) - static_cast<float>(first.px)) *
                        (static_cast<float>(third.pz) - static_cast<float>(first.pz)) -
                    (static_cast<float>(second.pz) - static_cast<float>(first.pz)) *
                        (static_cast<float>(third.px) - static_cast<float>(first.px));
                REQUIRE(std::abs(signedArea) == Catch::Approx(64.0F).margin(1.0e-4F));
            }
        }
    }
    REQUIRE(wetCells > 1'000);
    REQUIRE(dryCells > 1'000);

    constexpr std::array REPORTED_COLUMNS = {
        ColumnPos{-8, -1'896},  ColumnPos{8, -1'896},  ColumnPos{368, -1'880},
        ColumnPos{360, -1'864}, ColumnPos{-8, -1'264}, ColumnPos{248, -1'608},
    };
    for (const ColumnPos position : REPORTED_COLUMNS) {
        const worldgen::HydrologySample direct =
            generator->sampleGeneratedWaterAuthority(position.x, position.z);
        const worldgen::SurfaceSample exact = generator->sampleExactSurface(position.x, position.z);
        CAPTURE(position.x, position.z, direct.waterBodyId, exact.hydrology.waterBodyId);
        REQUIRE(direct.waterBodyId == exact.hydrology.waterBodyId);
        REQUIRE(direct.ocean == exact.hydrology.ocean);
        REQUIRE(direct.river == exact.hydrology.river);
        REQUIRE(direct.lake == exact.hydrology.lake);
    }
}

TEST_CASE("Step 32 shared water risers have one positive-side owner",
          "[render][far-terrain][water][coverage][seam][riser][negative][regression]") {
    const auto boundaryRisers = [](const FarTerrainMesh& mesh, FaceNormal face, float localX) {
        size_t result = 0;
        for (size_t offset = mesh.opaqueIndexCount; offset + 5 < mesh.indices.size(); offset += 6) {
            const Vertex& first = mesh.vertices[mesh.indices[offset]];
            if (unpackFace(first.faceAttr) != face || static_cast<float>(first.px) != localX) {
                continue;
            }
            ++result;
        }
        return result;
    };
    const auto flowingSource = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = x < 0 ? 64.875 : 64.625;
            sample.river = true;
            sample.generatedFluidLevel = x < 0 ? 1 : 3;
            sample.transitionOwnerKind = worldgen::WaterTransitionKind::RASTER_CHANNEL;
            sample.transitionOwnerId = 0x5154'4147'4552'4953ULL;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto left = FarTerrainMesher::build({-1, -1, FarTerrainStep::THIRTY_TWO}, flowingSource);
    const auto right = FarTerrainMesher::build({0, -1, FarTerrainStep::THIRTY_TWO}, flowingSource);
    REQUIRE(left->waterContourTriangleCount == 0);
    REQUIRE(right->waterContourTriangleCount == 0);
    REQUIRE(boundaryRisers(*left, FaceNormal::PLUS_X, static_cast<float>(FAR_TERRAIN_TILE_EDGE)) ==
            0);
    REQUIRE(boundaryRisers(*right, FaceNormal::PLUS_X, 0.0F) == 32);
    REQUIRE(boundaryRisers(*right, FaceNormal::MINUS_X, 0.0F) == 0);

    const auto shorelineSource = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = 64.0;
            sample.ocean = x < 0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
    const auto wet = FarTerrainMesher::build({-1, -1, FarTerrainStep::THIRTY_TWO}, shorelineSource);
    const auto dry = FarTerrainMesher::build({0, -1, FarTerrainStep::THIRTY_TWO}, shorelineSource);
    REQUIRE(boundaryRisers(*wet, FaceNormal::PLUS_X, static_cast<float>(FAR_TERRAIN_TILE_EDGE)) ==
            0);
    REQUIRE(boundaryRisers(*dry, FaceNormal::PLUS_X, 0.0F) == 0);
    REQUIRE(boundaryRisers(*dry, FaceNormal::MINUS_X, 0.0F) == 0);
}

TEST_CASE("Canonical lake contours agree through every far LOD",
          "[render][far-terrain][water][lake][seam][lod]") {
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });
    constexpr FarTerrainKey BASE_KEY{-32, 8, FarTerrainStep::TWO};
    constexpr int64_t WORLD_Z = 2'288;
    constexpr int64_t WET_X = -8'192;
    constexpr int64_t SCAN_END_X = -8'160;
    constexpr float LOCAL_Z = 240.0F;
    constexpr float MINIMUM_LOCAL_X = 0.0F;
    constexpr float MAXIMUM_LOCAL_X = 32.0F;
    constexpr int64_t TILE_ORIGIN_X = BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE;

    bool foundWetReference = false;
    int64_t firstDryReferenceX = std::numeric_limits<int64_t>::max();
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const FarTerrainGeometrySample sample =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_2);
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
        static_cast<float>(testFarGeometry(source, firstDryReferenceX - 1, WORLD_Z,
                                           worldgen::SurfaceFootprint::BLOCK_2)
                               .waterSurface);

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const int spacing = farTerrainStepSize(step);
        REQUIRE(world_coord::floorMod(WET_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
        REQUIRE(world_coord::floorMod(SCAN_END_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
    }
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const worldgen::SurfaceSample exactAuthority = generator.sampleExactSurface(x, WORLD_Z);
        const worldgen::SurfaceSample coarseAuthority =
            generator.sampleFarSurface(x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
        REQUIRE(exactAuthority.hydrology.lake == coarseAuthority.hydrology.lake);
        REQUIRE(exactAuthority.hydrology.waterBodyId == coarseAuthority.hydrology.waterBodyId);
        REQUIRE(exactAuthority.waterSurface ==
                Catch::Approx(coarseAuthority.waterSurface).margin(1.0e-5));
        const FarTerrainGeometrySample exact =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_2);
        const FarTerrainGeometrySample coarse =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
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
    constexpr int64_t FALL_X = -8'240;
    constexpr int64_t FALL_Z = 3'088;
    constexpr FarTerrainKey BASE_KEY{-33, 12, FarTerrainStep::TWO};
    constexpr float LOCAL_FALL_X = 208.0F;
    constexpr float LOCAL_FALL_Z = 16.0F;
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });

    const FarTerrainGeometrySample near =
        testFarGeometry(source, FALL_X, FALL_Z, worldgen::SurfaceFootprint::BLOCK_2);
    const FarTerrainGeometrySample coarse =
        testFarGeometry(source, FALL_X, FALL_Z, worldgen::SurfaceFootprint::BLOCK_16);
    for (const FarTerrainGeometrySample* sample : {&near, &coarse}) {
        REQUIRE(sample->ocean);
        REQUIRE_FALSE(sample->river);
        REQUIRE(sample->waterfall);
        REQUIRE(sample->waterfallAnchor);
        REQUIRE(sample->waterSurface == Catch::Approx(std::ceil(sample->waterfallBottom) - 1.0 +
                                                      fluidSurfaceHeight(FluidState::source())));
        REQUIRE(sample->waterfallBottom == Catch::Approx(SEA_LEVEL).margin(1.0e-4));
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
    const FarTerrainGeometrySample outside =
        testFarGeometry(source, outsideX, outsideZ, worldgen::SurfaceFootprint::BLOCK_16);
    REQUIRE_FALSE(outside.waterfall);
    REQUIRE(outside.ocean);
    REQUIRE_FALSE(outside.lake);
    REQUIRE_FALSE(outside.river);

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
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        std::vector<FarCanopy> canopies;
        size_t retained = 2;
        switch (step) {
            case FarTerrainStep::ONE:
            case FarTerrainStep::TWO:
                retained = 6;
                break;
            case FarTerrainStep::FOUR:
                retained = 5;
                break;
            case FarTerrainStep::EIGHT:
                retained = 4;
                break;
            case FarTerrainStep::SIXTEEN:
                retained = 3;
                break;
            case FarTerrainStep::THIRTY_TWO:
                retained = 2;
                break;
        }
        for (uint64_t index = 0; index < retained; ++index) {
            FarCanopy canopy;
            canopy.x = 20 + static_cast<int64_t>(index) * 40;
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
    const auto thirtyTwo = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(two->canopyAnchorCount == 6);
    REQUIRE(four->canopyAnchorCount == 5);
    REQUIRE(eight->canopyAnchorCount == 4);
    REQUIRE(sixteen->canopyAnchorCount == 3);
    REQUIRE(thirtyTwo->canopyAnchorCount == 2);

    for (const auto& mesh : {two, four, eight, sixteen, thirtyTwo}) {
        REQUIRE(mesh->canopyImpostorQuadCount == mesh->canopyAnchorCount * 19);
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

TEST_CASE("Step-two exact-anchor canopies reuse their resident voxel ground",
          "[render][far-terrain][canopy][lod][grounding][performance][regression]") {
    size_t blockTwoSamples = 0;
    FarTerrainSource source;
    source.sample = [&](int64_t x, int64_t, worldgen::SurfaceFootprint footprint) {
        if (footprint == worldgen::SurfaceFootprint::BLOCK_2)
            ++blockTwoSamples;
        const double terrainHeight = 64.0 + static_cast<double>(x) * 0.5;
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = terrainHeight;
        geometry.waterSurface = SEA_LEVEL;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = terrainHeight,
            .footprintMaximumTerrainHeight = terrainHeight,
            .materialPalette = testMaterialPalette(BlockType::GRASS),
        };
    };
    FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const size_t baselineBlockTwoSamples = blockTwoSamples;
    blockTwoSamples = 0;

    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{FarCanopy{
            .x = 5,
            .z = 5,
            .baseY = 65,
            .topY = 73,
            .canopyMinimumY = 69,
            .canopyMaximumY = 73,
            .canopyRadius = 2,
            .logBlock = BlockType::LOG,
            .leafBlock = BlockType::LEAVES,
            .anchorId = 1,
            .aggregate = false,
        }};
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    REQUIRE(blockTwoSamples == baselineBlockTwoSamples);
    REQUIRE(mesh->canopyAnchorCount == 1);

    float minimumCanopyY = std::numeric_limits<float>::max();
    for (const Vertex& vertex : mesh->vertices) {
        if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U)
            continue;
        minimumCanopyY = std::min(minimumCanopyY, static_cast<float>(vertex.py));
    }
    // The anchor lies in the [4, 6) voxel cell, whose filtered center-equivalent
    // top is 67. Conservative bounds do not lower its visible ground.
    REQUIRE(minimumCanopyY == 67.0F);
}

TEST_CASE("Far canopies keep one half-open owner across signed tile boundaries",
          "[render][far-terrain][canopy][ownership][seam][lod][flicker][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    const auto canopy = [](int64_t x, BlockType logBlock, BlockType leafBlock, uint64_t anchorId) {
        return FarCanopy{
            .x = x,
            .z = 96,
            .baseY = 65,
            .topY = 74,
            .canopyMinimumY = 69,
            .canopyMaximumY = 74,
            .canopyRadius = 3,
            .logBlock = logBlock,
            .leafBlock = leafBlock,
            .anchorId = anchorId,
            .aggregate = true,
        };
    };
    const std::array canopies = {
        canopy(-257, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES, 1),
        canopy(-256, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES, 2),
        canopy(-1, BlockType::PALM_LOG, BlockType::PALM_LEAVES, 3),
        canopy(0, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES, 4),
        canopy(255, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES, 5),
        canopy(256, BlockType::PALM_LOG, BlockType::PALM_LEAVES, 6),
    };
    source.canopies = [canopies](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector<FarCanopy>(canopies.begin(), canopies.end());
    };

    const auto canopyVertices = [](const FarTerrainMesh& mesh) {
        std::vector<Vertex> result;
        for (const Vertex& vertex : mesh.vertices) {
            if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U)
                result.push_back(vertex);
        }
        return result;
    };
    const auto sameVertices = [](std::span<const Vertex> first, std::span<const Vertex> second) {
        return first.size() == second.size() &&
               std::equal(first.begin(), first.end(), second.begin(),
                          [](const Vertex& lhs, const Vertex& rhs) {
                              return lhs.faceAttr == rhs.faceAttr &&
                                     static_cast<float>(lhs.px) == static_cast<float>(rhs.px) &&
                                     static_cast<float>(lhs.py) == static_cast<float>(rhs.py) &&
                                     static_cast<float>(lhs.pz) == static_cast<float>(rhs.pz) &&
                                     static_cast<float>(lhs.u) == static_cast<float>(rhs.u) &&
                                     static_cast<float>(lhs.v) == static_cast<float>(rhs.v);
                          });
    };

    constexpr std::array TILE_X = {-2LL, -1LL, 0LL, 1LL};
    constexpr std::array EXPECTED_OWNERS = {1U, 2U, 2U, 1U};
    size_t fineOwners = 0;
    size_t coarseOwners = 0;
    for (size_t index = 0; index < TILE_X.size(); ++index) {
        const auto fine = FarTerrainMesher::build({TILE_X[index], 0, FarTerrainStep::TWO}, source);
        const auto coarse =
            FarTerrainMesher::build({TILE_X[index], 0, FarTerrainStep::THIRTY_TWO}, source);
        CAPTURE(TILE_X[index]);
        REQUIRE(fine->canopyAnchorCount == EXPECTED_OWNERS[index]);
        REQUIRE(coarse->canopyAnchorCount == EXPECTED_OWNERS[index]);
        fineOwners += fine->canopyAnchorCount;
        coarseOwners += coarse->canopyAnchorCount;

        // Each owner emits the complete species silhouette even when its crown
        // crosses the tile face. The adjacent loaded tile does not emit a
        // replacement, and changing far tiers does not resize or relocate it.
        const std::vector<Vertex> fineCanopyVertices = canopyVertices(*fine);
        const std::vector<Vertex> coarseCanopyVertices = canopyVertices(*coarse);
        REQUIRE_FALSE(fineCanopyVertices.empty());
        REQUIRE(sameVertices(fineCanopyVertices, coarseCanopyVertices));
        for (const Vertex& vertex : fineCanopyVertices) {
            const unsigned int face = static_cast<unsigned int>(unpackFace(vertex.faceAttr));
            REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(face, true, false));
            const simd_float2 position =
                simd_make_float2(static_cast<float>(vertex.px), static_cast<float>(vertex.pz));
            const simd_float2 ownershipSample = farTerrainExactOwnershipSamplePosition(
                position, face, farTerrainOpaqueRiserUsesEmittingColumn(face, true, false));
            REQUIRE(ownershipSample.x == position.x);
            REQUIRE(ownershipSample.y == position.y);
        }
        const auto [minimumX, maximumX] =
            std::minmax_element(fineCanopyVertices.begin(), fineCanopyVertices.end(),
                                [](const Vertex& lhs, const Vertex& rhs) {
                                    return static_cast<float>(lhs.px) < static_cast<float>(rhs.px);
                                });
        REQUIRE((static_cast<float>(minimumX->px) < 0.0F ||
                 static_cast<float>(maximumX->px) > FAR_TERRAIN_TILE_EDGE_BLOCKS));
    }
    REQUIRE(fineOwners == canopies.size());
    REQUIRE(coarseOwners == canopies.size());
}

TEST_CASE("Far canopy leaf materials produce distinct voxel silhouettes",
          "[render][far-terrain][canopy][species][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        const auto canopy = [](int64_t x, BlockType logBlock, BlockType leafBlock,
                               feature_generation::TreeSpecies species, uint64_t anchorId) {
            return FarCanopy{
                .x = x,
                .z = 96,
                .baseY = 64,
                .topY = 73,
                .canopyMinimumY = 68,
                .canopyMaximumY = 73,
                .canopyRadius = 3,
                .logBlock = logBlock,
                .leafBlock = leafBlock,
                .anchorId = anchorId,
                .species = species,
            };
        };
        return std::vector{
            canopy(32, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES,
                   feature_generation::TreeSpecies::SPRUCE, 1),
            canopy(96, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES,
                   feature_generation::TreeSpecies::ACACIA, 5),
            canopy(160, BlockType::PALM_LOG, BlockType::PALM_LEAVES,
                   feature_generation::TreeSpecies::PALM, 9),
            // Species is authoritative even when a material differs. This
            // deliberately mismatched fixture must keep an oak silhouette
            // while using jungle leaves as its texture layer.
            canopy(224, BlockType::LOG, BlockType::JUNGLE_LEAVES,
                   feature_generation::TreeSpecies::OAK, 13),
        };
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    struct TopSpan {
        float width = 0.0F;
        float depth = 0.0F;
    };
    const auto topSpans = [&](BlockType leafBlock) {
        struct Extents {
            float minimumX = std::numeric_limits<float>::max();
            float maximumX = std::numeric_limits<float>::lowest();
            float minimumZ = std::numeric_limits<float>::max();
            float maximumZ = std::numeric_limits<float>::lowest();
        };
        std::map<float, Extents> byHeight;
        for (const Vertex& vertex : mesh->vertices) {
            if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U ||
                unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(leafBlock) ||
                unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y) {
                continue;
            }
            Extents& extents = byHeight[static_cast<float>(vertex.py)];
            extents.minimumX = std::min(extents.minimumX, static_cast<float>(vertex.px));
            extents.maximumX = std::max(extents.maximumX, static_cast<float>(vertex.px));
            extents.minimumZ = std::min(extents.minimumZ, static_cast<float>(vertex.pz));
            extents.maximumZ = std::max(extents.maximumZ, static_cast<float>(vertex.pz));
        }
        std::vector<TopSpan> result;
        for (const auto& [height, extents] : byHeight) {
            (void)height;
            result.push_back({.width = extents.maximumX - extents.minimumX,
                              .depth = extents.maximumZ - extents.minimumZ});
        }
        return result;
    };

    const std::vector<TopSpan> spruce = topSpans(BlockType::SPRUCE_LEAVES);
    REQUIRE(spruce.size() == 4);
    REQUIRE(spruce[0].width > spruce[1].width);
    REQUIRE(spruce[1].width > spruce[2].width);
    REQUIRE(spruce[2].width > spruce[3].width);
    REQUIRE(std::ranges::all_of(spruce, [](TopSpan span) { return span.width == span.depth; }));

    const std::vector<TopSpan> acacia = topSpans(BlockType::ACACIA_LEAVES);
    REQUIRE(acacia.size() == 2);
    REQUIRE(acacia[0].width > acacia[1].width);
    REQUIRE(acacia[0].width == acacia[0].depth);
    REQUIRE(acacia[1].width == acacia[1].depth);

    const std::vector<TopSpan> palm = topSpans(BlockType::PALM_LEAVES);
    REQUIRE(palm.size() == 3);
    REQUIRE(palm[0].width > palm[0].depth);
    REQUIRE(palm[1].width < palm[1].depth);
    REQUIRE(palm[2].width == palm[2].depth);
    REQUIRE(palm[2].width < palm[0].width);

    const std::vector<TopSpan> authoritativeOak = topSpans(BlockType::JUNGLE_LEAVES);
    REQUIRE(authoritativeOak.size() == 3);
    REQUIRE(authoritativeOak[0].width < authoritativeOak[1].width);
    REQUIRE(authoritativeOak[1].width > authoritativeOak[2].width);
    REQUIRE(authoritativeOak[0].width == authoritativeOak[2].width);
    REQUIRE(std::ranges::all_of(authoritativeOak,
                                [](TopSpan span) { return span.width == span.depth; }));
    REQUIRE(mesh->canopyImpostorQuadCount == 76);
    REQUIRE(mesh->deterministicHash ==
            FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source)->deterministicHash);
}

TEST_CASE("Step-two fallen logs retain their exact horizontal morphology",
          "[render][far-terrain][canopy][species][fallen-log][lod][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{FarCanopy{
            .x = 32,
            .z = 48,
            .baseY = 65,
            .topY = 65,
            .canopyMinimumY = 65,
            .canopyMaximumY = 65,
            .logBlock = BlockType::WILLOW_LOG,
            .anchorId = 17,
            .species = feature_generation::TreeSpecies::FALLEN_LOG,
            .formX = 1,
            .formExtent = 7,
        }};
    };

    const auto fine = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const auto coarse = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(fine->canopyAnchorCount == 1);
    REQUIRE(fine->canopyImpostorQuadCount == 5);
    REQUIRE(fine->canopyImpostorQuadCount == coarse->canopyImpostorQuadCount);

    float minimumX = std::numeric_limits<float>::max();
    float maximumX = std::numeric_limits<float>::lowest();
    float minimumY = std::numeric_limits<float>::max();
    float maximumY = std::numeric_limits<float>::lowest();
    float minimumZ = std::numeric_limits<float>::max();
    float maximumZ = std::numeric_limits<float>::lowest();
    for (const Vertex& vertex : fine->vertices) {
        if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U)
            continue;
        minimumX = std::min(minimumX, static_cast<float>(vertex.px));
        maximumX = std::max(maximumX, static_cast<float>(vertex.px));
        minimumY = std::min(minimumY, static_cast<float>(vertex.py));
        maximumY = std::max(maximumY, static_cast<float>(vertex.py));
        minimumZ = std::min(minimumZ, static_cast<float>(vertex.pz));
        maximumZ = std::max(maximumZ, static_cast<float>(vertex.pz));
    }
    REQUIRE(maximumX - minimumX == 7.0F);
    REQUIRE(maximumY - minimumY == 1.0F);
    REQUIRE(maximumZ - minimumZ == 1.0F);
    REQUIRE(fine->deterministicHash ==
            FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source)->deterministicHash);
}

TEST_CASE("Production step-two meshing uses shared exact tree roots without scalar basin work",
          "[render][far-terrain][canopy][worldgen][performance][handoff][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const uint64_t scalarCallsBefore = generator->basinCacheMetrics().scalarSampleCalls;
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const auto mesh = FarTerrainMesher::build({-1, 0, FarTerrainStep::TWO}, source);
    const std::vector<FarCanopy> canopies = generator->collectFarCanopiesForLod(-256, 0, 0, 256, 2);
    const size_t owned = std::ranges::count_if(canopies, [](const FarCanopy& canopy) {
        return canopy.x >= -256 && canopy.x < 0 && canopy.z >= 0 && canopy.z < 256;
    });
    REQUIRE(owned > 0);
    REQUIRE(mesh->canopyAnchorCount == owned);
    REQUIRE(generator->cachedColumnPlanCount() == 0);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    generator->clearMacroCaches();
    const auto rebuilt = FarTerrainMesher::build({-1, 0, FarTerrainStep::TWO}, source);
    REQUIRE(rebuilt->deterministicHash == mesh->deterministicHash);
    REQUIRE(rebuilt->canopyAnchorCount == mesh->canopyAnchorCount);
}

TEST_CASE("Step-thirty-two coverage batches water and aggregate forest authority",
          "[render][far-terrain][coverage][water][canopy][batch][performance][determinism]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.geometryPoints);
    // The reported seed-42 camera tile contains a mixed coastal forest. Use
    // it to prove the coverage preview keeps both land canopies and canonical
    // water without invoking scalar basin sampling.
    constexpr FarTerrainKey KEY{-1, 0, FarTerrainStep::THIRTY_TWO};
    size_t expectedCanopyCount = 0;
    const auto collectCanopies = source.canopies;
    source.canopies = [&](int64_t minimumX, int64_t minimumZ, int64_t maximumX, int64_t maximumZ,
                          FarTerrainStep step) {
        std::vector<FarCanopy> canopies =
            collectCanopies(minimumX, minimumZ, maximumX, maximumZ, step);
        expectedCanopyCount = std::ranges::count_if(canopies, [&](const FarCanopy& canopy) {
            return canopy.x >= minimumX && canopy.x < maximumX && canopy.z >= minimumZ &&
                   canopy.z < maximumZ;
        });
        return canopies;
    };
    const uint64_t scalarCallsBefore = generator->basinCacheMetrics().scalarSampleCalls;
    const auto first = FarTerrainMesher::build(KEY, source);
    REQUIRE(first->canopyAnchorCount > 0);
    REQUIRE(first->canopyAnchorCount == expectedCanopyCount);
    REQUIRE(first->waterQuadCount + first->waterContourTriangleCount > 0);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    generator->clearMacroCaches();
    const auto rebuilt = FarTerrainMesher::build(KEY, source);
    REQUIRE(rebuilt->deterministicHash == first->deterministicHash);
    REQUIRE(rebuilt->vertices.size() == first->vertices.size());
    REQUIRE(std::equal(first->vertices.begin(), first->vertices.end(), rebuilt->vertices.begin(),
                       [](const Vertex& left, const Vertex& right) {
                           return left.faceAttr == right.faceAttr && left.px == right.px &&
                                  left.py == right.py && left.pz == right.pz && left.u == right.u &&
                                  left.v == right.v;
                       }));
    REQUIRE(rebuilt->indices == first->indices);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
}

TEST_CASE("Step-eight and step-sixteen canonical water probes stay in bulk batches",
          "[render][far-terrain][water][batch][performance][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            const int64_t shoreline =
                128 + static_cast<int64_t>(std::lround(std::sin(z * 0.075) * 13.0));
            sample.lake = x < shoreline;
            sample.waterBodyId = sample.lake ? 0x4255'4C4B'5741'5445ULL : worldgen::NO_WATER_BODY;
            sample.terrainHeight = sample.lake ? 60.0 : 70.0;
            sample.waterSurface = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto sample = source.sample;
    std::vector<size_t> pointBatchSizes;
    source.geometryPoints = [&](std::span<const ColumnPos> positions,
                                worldgen::SurfaceFootprint footprint,
                                std::span<FarTerrainGeometrySample> output) {
        REQUIRE(output.size() == positions.size());
        pointBatchSizes.push_back(positions.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            output[index] = sample(positions[index].x, positions[index].z, footprint).geometry;
        }
    };

    for (const FarTerrainStep step : {FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({0, 0, step}, source);
        REQUIRE(mesh->waterQuadCount + mesh->waterContourTriangleCount > 0);
    }
    REQUIRE_FALSE(pointBatchSizes.empty());
    REQUIRE(std::ranges::none_of(pointBatchSizes, [](size_t size) { return size <= 1; }));
}

TEST_CASE("Far canopy layers stay inside exact bounds without LOD inflation",
          "[render][far-terrain][canopy][lod][bounds][regression]") {
    FarTerrainSource source;
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint footprint) {
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = 60.0 + worldgen::surfaceFootprintWidth(footprint);
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = geometry.terrainHeight,
            .footprintMaximumTerrainHeight = geometry.terrainHeight,
            .materialPalette = testMaterialPalette(BlockType::GRASS),
        };
    };
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        return std::vector{FarCanopy{
            .x = 64,
            .z = 64,
            .baseY = 64,
            .topY = 72,
            .canopyMinimumY = 67,
            .canopyMaximumY = 72,
            .canopyRadius = 3,
            .logBlock = BlockType::LOG,
            .leafBlock = BlockType::LEAVES,
            .anchorId = 1,
            .aggregate = farTerrainStepSize(step) >= 8,
        }};
    };

    const auto leafVertices = [](const FarTerrainMesh& mesh) {
        std::vector<Vertex> result;
        std::ranges::copy_if(mesh.vertices, std::back_inserter(result), [](const Vertex& vertex) {
            return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U &&
                   unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::LEAVES);
        });
        return result;
    };
    const auto requireSameVertices = [](const std::vector<Vertex>& first,
                                        const std::vector<Vertex>& second) {
        REQUIRE(first.size() == second.size());
        const auto minimumY = [](const std::vector<Vertex>& vertices) {
            return std::ranges::min(
                vertices, {}, [](const Vertex& vertex) { return static_cast<float>(vertex.py); });
        };
        const float firstMinimumY = static_cast<float>(minimumY(first).py);
        const float secondMinimumY = static_cast<float>(minimumY(second).py);
        for (size_t index = 0; index < first.size(); ++index) {
            CAPTURE(index);
            REQUIRE(first[index].faceAttr == second[index].faceAttr);
            REQUIRE(static_cast<float>(first[index].px) == static_cast<float>(second[index].px));
            REQUIRE(static_cast<float>(first[index].py) - firstMinimumY ==
                    static_cast<float>(second[index].py) - secondMinimumY);
            REQUIRE(static_cast<float>(first[index].pz) == static_cast<float>(second[index].pz));
            REQUIRE(static_cast<float>(first[index].u) == static_cast<float>(second[index].u));
            REQUIRE(static_cast<float>(first[index].v) == static_cast<float>(second[index].v));
        }
    };

    std::vector<std::vector<Vertex>> verticesByLod;
    for (FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
                                FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const auto mesh = FarTerrainMesher::build({0, 0, step}, source);
        const std::vector<Vertex> vertices = leafVertices(*mesh);
        REQUIRE(vertices.size() == 15 * 4);
        float minimumX = std::numeric_limits<float>::max();
        float maximumX = std::numeric_limits<float>::lowest();
        float minimumY = std::numeric_limits<float>::max();
        float maximumY = std::numeric_limits<float>::lowest();
        float minimumZ = std::numeric_limits<float>::max();
        float maximumZ = std::numeric_limits<float>::lowest();
        for (const Vertex& vertex : vertices) {
            minimumX = std::min(minimumX, static_cast<float>(vertex.px));
            maximumX = std::max(maximumX, static_cast<float>(vertex.px));
            minimumY = std::min(minimumY, static_cast<float>(vertex.py));
            maximumY = std::max(maximumY, static_cast<float>(vertex.py));
            minimumZ = std::min(minimumZ, static_cast<float>(vertex.pz));
            maximumZ = std::max(maximumZ, static_cast<float>(vertex.pz));
        }
        REQUIRE(maximumX - minimumX == 7.0F);
        REQUIRE(maximumY - minimumY == 6.0F);
        REQUIRE(maximumZ - minimumZ == 7.0F);

        for (size_t offset = 0; offset < vertices.size(); offset += 4) {
            float quadMinimumY = std::numeric_limits<float>::max();
            float quadMaximumY = std::numeric_limits<float>::lowest();
            float quadMinimumHorizontal = std::numeric_limits<float>::max();
            float quadMaximumHorizontal = std::numeric_limits<float>::lowest();
            const FaceNormal face = unpackFace(vertices[offset].faceAttr);
            for (size_t corner = 0; corner < 4; ++corner) {
                const Vertex& vertex = vertices[offset + corner];
                quadMinimumY = std::min(quadMinimumY, static_cast<float>(vertex.py));
                quadMaximumY = std::max(quadMaximumY, static_cast<float>(vertex.py));
                const float horizontal = face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X
                                             ? static_cast<float>(vertex.pz)
                                             : static_cast<float>(vertex.px);
                quadMinimumHorizontal = std::min(quadMinimumHorizontal, horizontal);
                quadMaximumHorizontal = std::max(quadMaximumHorizontal, horizontal);
            }
            const bool giantSide = quadMaximumY - quadMinimumY == 6.0F &&
                                   quadMaximumHorizontal - quadMinimumHorizontal == 7.0F;
            REQUIRE_FALSE(giantSide);
        }
        verticesByLod.push_back(vertices);
    }
    requireSameVertices(verticesByLod[0], verticesByLod[1]);
    requireSameVertices(verticesByLod[2], verticesByLod[3]);
    requireSameVertices(verticesByLod[3], verticesByLod[4]);
}

TEST_CASE("Coarse far forests retain hierarchical compact anchors",
          "[render][far-terrain][canopy][lod][worldgen][determinism][regression]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    ChunkGenerator generator(42);

    const std::vector<FarCanopy> nearAnchors =
        generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2);
    REQUIRE_FALSE(nearAnchors.empty());
    REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2) ==
            nearAnchors);
    for (const FarCanopy& canopy : nearAnchors) {
        REQUIRE_FALSE(canopy.aggregate);
        REQUIRE(canopy.anchorId != 0);
        REQUIRE((canopy.logBlock != BlockType::AIR || canopy.leafBlock != BlockType::AIR));
    }

    // Step two preserves exact accepted roots for a stable near handoff.
    // Aggregate cover begins at step four, then each coarser tier retains a
    // strict subset of the same fixed forest-cell candidates.
    constexpr std::array LOD_STEPS = {4, 8, 16, 32};
    constexpr std::array CROWN_LIMITS = {5U, 4U, 3U, 2U};
    using Cell = std::pair<int64_t, int64_t>;
    std::array<std::vector<FarCanopy>, LOD_STEPS.size()> tiers;
    std::array<std::map<Cell, size_t>, LOD_STEPS.size()> counts;

    for (size_t tierIndex = 0; tierIndex < LOD_STEPS.size(); ++tierIndex) {
        const int step = LOD_STEPS[tierIndex];
        tiers[tierIndex] =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, step);
        REQUIRE_FALSE(tiers[tierIndex].empty());
        REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z,
                                                   step) == tiers[tierIndex]);
        for (const FarCanopy& canopy : tiers[tierIndex]) {
            REQUIRE(canopy.aggregate);
            REQUIRE(canopy.canopyRadius <= 3);
            REQUIRE(static_cast<int>(canopy.canopyRadius) * 2 + 1 <= 7);
            ++counts[tierIndex][{world_coord::floorDiv(canopy.x, int64_t{64}),
                                 world_coord::floorDiv(canopy.z, int64_t{64})}];
        }
        if (tierIndex == 0)
            continue;
        REQUIRE(tiers[tierIndex].size() <= tiers[tierIndex - 1].size());
        std::unordered_map<uint64_t, FarCanopy> nearer;
        for (const FarCanopy& canopy : tiers[tierIndex - 1]) {
            REQUIRE(nearer.emplace(canopy.anchorId, canopy).second);
        }
        for (const FarCanopy& canopy : tiers[tierIndex]) {
            const auto matching = nearer.find(canopy.anchorId);
            REQUIRE(matching != nearer.end());
            REQUIRE(matching->second == canopy);
        }
    }

    bool sawSeveralCrowns = false;
    for (const auto& [cell, finestCount] : counts.front()) {
        REQUIRE(finestCount <= CROWN_LIMITS.front());
        for (size_t tierIndex = 1; tierIndex < counts.size(); ++tierIndex) {
            REQUIRE(counts[tierIndex][cell] ==
                    std::min<size_t>(finestCount, CROWN_LIMITS[tierIndex]));
        }
        sawSeveralCrowns = sawSeveralCrowns || counts.back()[cell] >= 2;
    }
    REQUIRE(sawSeveralCrowns);
    REQUIRE(generator.cachedColumnPlanCount() == 0);
}

TEST_CASE("Far terrain greedily merges flat terrain and water", "[render][far-terrain]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
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
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = x < FAR_TERRAIN_TILE_EDGE / 2;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });

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
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < 100 ? 74.0 : 84.0;
            sample.waterSurface = x < 100 ? 80.0 : SEA_LEVEL;
            sample.lake = x < 100;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });

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

TEST_CASE("Far water never triangulates between distinct standing bodies",
          "[render][far-terrain][water][authority][seam][lod][regression]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.lake = true;
            if (x + z < 400) {
                sample.waterBodyId = 0xA11CE;
                sample.waterSurface = 307.875;
            } else {
                sample.waterBodyId = 0xB0B;
                sample.waterSurface = 106.875;
            }
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });

    const auto coarse = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto fine = FarTerrainMesher::build({1, 0, FarTerrainStep::FOUR}, source);
    const auto requireNoWaterBridge = [](const FarTerrainMesh& mesh) {
        bool sawUpperBody = false;
        bool sawLowerBody = false;
        for (size_t offset = mesh.opaqueIndexCount; offset + 2 < mesh.indices.size(); offset += 3) {
            std::array<float, 3> heights{};
            for (size_t corner = 0; corner < heights.size(); ++corner) {
                heights[corner] =
                    static_cast<float>(mesh.vertices[mesh.indices[offset + corner]].py);
                sawUpperBody = sawUpperBody || heights[corner] > 300.0F;
                sawLowerBody = sawLowerBody || heights[corner] < 110.0F;
            }
            const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
            CAPTURE(mesh.key.tileX, mesh.key.tileZ, static_cast<int>(mesh.key.step), *minimum,
                    *maximum);
            REQUIRE(*maximum - *minimum <= 0.25F);
        }
        REQUIRE(sawUpperBody);
        REQUIRE(sawLowerBody);
    };
    requireNoWaterBridge(*coarse);
    requireNoWaterBridge(*fine);

    const auto seamVertices = [](const FarTerrainMesh& mesh, float localX) {
        std::set<std::pair<float, float>> result;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); ++offset) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset]];
            if (static_cast<float>(vertex.px) == localX) {
                result.emplace(static_cast<float>(vertex.pz), static_cast<float>(vertex.py));
            }
        }
        return result;
    };
    const auto coarseSeam = seamVertices(*coarse, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto fineSeam = seamVertices(*fine, 0.0F);
    REQUIRE_FALSE(coarseSeam.empty());
    REQUIRE_FALSE(fineSeam.empty());
    for (int z = 0; z <= FAR_TERRAIN_TILE_EDGE; z += 2) {
        const float expectedHeight =
            static_cast<float>(static_cast<float16_t>(256 + z < 400 ? 307.875F : 106.875F));
        CAPTURE(z, expectedHeight);
        REQUIRE(coarseSeam.contains({static_cast<float>(z), expectedHeight}));
        REQUIRE(fineSeam.contains({static_cast<float>(z), expectedHeight}));
    }
}

TEST_CASE("Seed 764891 caldera water has no coarse interpolation wall",
          "[render][far-terrain][water][authority][caldera][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const FarTerrainGeometrySample caldera =
        testFarGeometry(source, 23'029, -111'486, worldgen::SurfaceFootprint::BLOCK_16);
    REQUIRE(caldera.lake);
    REQUIRE(caldera.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(caldera.waterSurface > 250.0);

    const auto mesh = FarTerrainMesher::build({89, -436, FarTerrainStep::SIXTEEN}, source);
    bool sawCalderaSurface = false;
    for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size(); offset += 3) {
        std::array<float, 3> heights{};
        bool topSurface = true;
        for (size_t corner = 0; corner < heights.size(); ++corner) {
            const Vertex& vertex = mesh->vertices[mesh->indices[offset + corner]];
            topSurface = topSurface && unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                         !unpackFluidFalling(vertex.faceAttr);
            heights[corner] = static_cast<float>(vertex.py);
        }
        if (!topSurface)
            continue;
        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        sawCalderaSurface = sawCalderaSurface || *maximum > 250.0F;
        CAPTURE(*minimum, *maximum);
        REQUIRE(*maximum - *minimum <= 8.0F);
    }
    REQUIRE(sawCalderaSurface);
}

TEST_CASE("Step 32 keeps caldera water and volcanic island land on canonical cells",
          "[render][far-terrain][water][coverage][caldera][volcanic][ownership][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    struct Fixture {
        ColumnPos position;
        bool wet;
        const char* name;
    };
    constexpr std::array FIXTURES = {
        Fixture{{23'024, -111'488}, true, "caldera lake"},
        Fixture{{17'576, -9'632}, false, "volcanic island"},
    };
    for (const Fixture& fixture : FIXTURES) {
        const int64_t tileX =
            world_coord::floorDiv(fixture.position.x, int64_t{FAR_TERRAIN_TILE_EDGE});
        const int64_t tileZ =
            world_coord::floorDiv(fixture.position.z, int64_t{FAR_TERRAIN_TILE_EDGE});
        const auto mesh =
            FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
        const worldgen::SurfaceSample exact =
            generator->sampleExactSurface(fixture.position.x, fixture.position.z);
        const bool exactWet =
            (exact.hydrology.ocean || exact.hydrology.river || exact.hydrology.lake) &&
            exact.waterSurface > exact.terrainHeight + 0.01;
        const float localX = static_cast<float>(fixture.position.x - mesh->originX) + 0.5F;
        const float localZ = static_cast<float>(fixture.position.z - mesh->originZ) + 0.5F;
        CAPTURE(fixture.name, tileX, tileZ, exactWet, exact.hydrology.waterBodyId, localX, localZ);
        REQUIRE(exactWet == fixture.wet);
        REQUIRE(farWaterTopCovers(*mesh, localX, localZ) == exactWet);
        REQUIRE(mesh->waterContourTriangleCount == 0);
        if (fixture.wet)
            REQUIRE(exact.hydrology.waterBodyId != worldgen::NO_WATER_BODY);
    }
}

TEST_CASE("Far terrain shoreline contours stitch across tile faces",
          "[render][far-terrain][water][seam]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = z < x - FAR_TERRAIN_TILE_EDGE / 2;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
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
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.river = z >= 5 && z <= 7;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });
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

TEST_CASE("Orthogonal water boundary refinement owns each corner once",
          "[render][far-terrain][water][seam][ownership][regression]") {
    constexpr uint8_t WEST_EDGE = 1U << 0U;
    constexpr uint8_t EAST_EDGE = 1U << 1U;
    constexpr uint8_t NORTH_EDGE = 1U << 2U;
    constexpr uint8_t SOUTH_EDGE = 1U << 3U;
    for (uint8_t edgeMask = 0; edgeMask < 16; ++edgeMask) {
        CAPTURE(static_cast<unsigned>(edgeMask));
        const FarTerrainSource source = testFarTerrainSource(
            [edgeMask](int64_t x, int64_t z) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = 40.0;
                sample.waterSurface = 64.0;
                const auto active = [edgeMask](uint8_t edge) { return (edgeMask & edge) != 0; };
                bool wet = false;
                wet = wet || (active(WEST_EDGE) && x <= 16 && z >= 48 && z <= 64);
                wet = wet ||
                      (active(EAST_EDGE) && x >= FAR_TERRAIN_TILE_EDGE - 16 && z >= 48 && z <= 64);
                wet = wet || (active(NORTH_EDGE) && z <= 16 && x >= 48 && x <= 64);
                wet = wet ||
                      (active(SOUTH_EDGE) && z >= FAR_TERRAIN_TILE_EDGE - 16 && x >= 48 && x <= 64);
                wet = wet || (active(WEST_EDGE) && active(NORTH_EDGE) && x <= 32 && z <= 32);
                wet = wet || (active(EAST_EDGE) && active(NORTH_EDGE) &&
                              x >= FAR_TERRAIN_TILE_EDGE - 32 && z <= 32);
                wet = wet || (active(WEST_EDGE) && active(SOUTH_EDGE) && x <= 32 &&
                              z >= FAR_TERRAIN_TILE_EDGE - 32);
                wet = wet || (active(EAST_EDGE) && active(SOUTH_EDGE) &&
                              x >= FAR_TERRAIN_TILE_EDGE - 32 && z >= FAR_TERRAIN_TILE_EDGE - 32);
                sample.ocean = wet;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
        const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);

        std::array<double, 4> cornerAreas{};
        for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size();
             offset += 3) {
            std::array<const Vertex*, 3> vertices{};
            for (size_t corner = 0; corner < vertices.size(); ++corner) {
                vertices[corner] = &mesh->vertices[mesh->indices[offset + corner]];
            }
            const double x0 = static_cast<float>(vertices[0]->px);
            const double z0 = static_cast<float>(vertices[0]->pz);
            const double x1 = static_cast<float>(vertices[1]->px);
            const double z1 = static_cast<float>(vertices[1]->pz);
            const double x2 = static_cast<float>(vertices[2]->px);
            const double z2 = static_cast<float>(vertices[2]->pz);
            const double area = std::abs((x0 * (z1 - z2) + x1 * (z2 - z0) + x2 * (z0 - z1)) * 0.5);
            constexpr float TILE_EDGE = static_cast<float>(FAR_TERRAIN_TILE_EDGE);
            constexpr std::array<std::array<float, 4>, 4> CORNERS = {{
                {{0.0F, 16.0F, 0.0F, 16.0F}},
                {{TILE_EDGE - 16.0F, TILE_EDGE, 0.0F, 16.0F}},
                {{0.0F, 16.0F, TILE_EDGE - 16.0F, TILE_EDGE}},
                {{TILE_EDGE - 16.0F, TILE_EDGE, TILE_EDGE - 16.0F, TILE_EDGE}},
            }};
            for (size_t corner = 0; corner < CORNERS.size(); ++corner) {
                const auto [minimumX, maximumX, minimumZ, maximumZ] = CORNERS[corner];
                const bool inside = std::ranges::all_of(vertices, [&](const Vertex* vertex) {
                    const float x = static_cast<float>(vertex->px);
                    const float z = static_cast<float>(vertex->pz);
                    return x >= minimumX && x <= maximumX && z >= minimumZ && z <= maximumZ;
                });
                if (inside)
                    cornerAreas[corner] += area;
            }
        }
        constexpr std::array<std::pair<uint8_t, uint8_t>, 4> INCIDENT_EDGES = {{
            {WEST_EDGE, NORTH_EDGE},
            {EAST_EDGE, NORTH_EDGE},
            {WEST_EDGE, SOUTH_EDGE},
            {EAST_EDGE, SOUTH_EDGE},
        }};
        for (size_t corner = 0; corner < cornerAreas.size(); ++corner) {
            const auto [first, second] = INCIDENT_EDGES[corner];
            const double expected =
                (edgeMask & first) != 0 && (edgeMask & second) != 0 ? 16.0 * 16.0 : 0.0;
            REQUIRE(cornerAreas[corner] == Catch::Approx(expected));
        }
    }
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
    REQUIRE_FALSE(rightEdge.empty());
    REQUIRE(farTerrainTopsAreVoxelFlat(*leftFirst));
    REQUIRE(farTerrainTopsAreVoxelFlat(*rightFirst));

    bool sawHeightDifference = false;
    constexpr int STEP = 4;
    const int64_t sharedWorldX = rightKey.tileX * FAR_TERRAIN_TILE_EDGE;
    for (int cellZ = 0; cellZ < FAR_TERRAIN_TILE_EDGE / STEP; ++cellZ) {
        const int64_t worldZ = leftKey.tileZ * FAR_TERRAIN_TILE_EDGE + cellZ * STEP;
        const float leftHeight =
            expectedVoxelCellHeight(source, sharedWorldX - STEP, worldZ, FarTerrainStep::FOUR);
        const float rightHeight =
            expectedVoxelCellHeight(source, sharedWorldX, worldZ, FarTerrainStep::FOUR);
        if (leftHeight == rightHeight)
            continue;
        sawHeightDifference = true;
        if (leftHeight > rightHeight) {
            REQUIRE(hasTerrainBoundaryRiser(
                *leftFirst, FaceNormal::PLUS_X, static_cast<float>(FAR_TERRAIN_TILE_EDGE),
                static_cast<float>(cellZ * STEP), static_cast<float>((cellZ + 1) * STEP),
                rightHeight, leftHeight));
        } else {
            REQUIRE(hasTerrainBoundaryRiser(
                *rightFirst, FaceNormal::MINUS_X, 0.0F, static_cast<float>(cellZ * STEP),
                static_cast<float>((cellZ + 1) * STEP), leftHeight, rightHeight));
        }
    }
    REQUIRE(sawHeightDifference);
}

TEST_CASE("Far terrain LOD edges share aligned samples and carry downward skirts",
          "[render][far-terrain][seam]") {
    const FarTerrainSource source = farTerrainTestSource();
    const auto fine = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::FOUR}, source);
    const auto coarse =
        FarTerrainMesher::build(FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}, source);
    const std::map<int, float> fineEdge = farTerrainEdge(*fine, true);
    const std::map<int, float> coarseEdge = farTerrainEdge(*coarse, false);
    REQUIRE_FALSE(fineEdge.empty());
    REQUIRE_FALSE(coarseEdge.empty());
    for (const auto& [z, height] : fineEdge) {
        CAPTURE(z);
        REQUIRE(height == std::round(height));
    }
    for (const auto& [z, height] : coarseEdge) {
        CAPTURE(z);
        REQUIRE(height == std::round(height));
    }
    REQUIRE(farTerrainTopsAreVoxelFlat(*fine));
    REQUIRE(farTerrainTopsAreVoxelFlat(*coarse));
    REQUIRE(fine->skirtQuadCount == 256);
    REQUIRE(coarse->skirtQuadCount == 64);
    REQUIRE(fine->bounds.minY >= static_cast<float>(WORLD_MIN_Y));
    REQUIRE(fine->bounds.minY <= fine->surfaceBounds.maxY - FAR_TERRAIN_SKIRT_DEPTH + 0.01F);
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
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(threadMutex);
            workerThreads.insert(std::this_thread::get_id());
            if (workerThreads.size() == FarTerrainScheduler::WORKER_COUNT) {
                workersReleased = true;
                threadCv.notify_all();
            } else if (!workersReleased && !threadCv.wait_for(lock, std::chrono::seconds(30),
                                                              [&] { return workersReleased; })) {
                workerGateTimedOut = true;
                workersReleased = true;
                threadCv.notify_all();
            }
        }
        return sample(x, z, footprint);
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
    for (int attempt = 0; attempt < 400 && (scheduler.stats().inFlight != 0 ||
                                            scheduler.stats().maintenancePending != 0);
         ++attempt) {
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
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (!entered) {
                entered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return released; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainScheduler scheduler(source);
    REQUIRE(scheduler.enqueue({0, 0, FarTerrainStep::SIXTEEN}));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(30), [&] { return entered; }));
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
    for (int attempt = 0; attempt < 400; ++attempt) {
        const FarTerrainSchedulerStats current = scheduler.stats();
        if (current.maintenancePending == 0 && current.cacheEntries == wanted.size())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const FarTerrainSchedulerStats retained = scheduler.stats();
    REQUIRE(retained.maintenancePending == 0);
    REQUIRE(retained.cacheEntries == wanted.size());
    REQUIRE(retained.completed == wanted.size());
    REQUIRE(scheduler.findCached(keys[1]));
    REQUIRE(scheduler.findCached(keys[3]));
    REQUIRE_FALSE(scheduler.findCached(keys[0]));
    REQUIRE_FALSE(scheduler.findCached(keys[2]));
    REQUIRE_FALSE(scheduler.enqueue(keys[0]));
}

TEST_CASE("Far terrain scheduler reuses stable wanted state",
          "[render][far-terrain][scheduler][residency][performance]") {
    FarTerrainScheduler scheduler(farTerrainTestSource());
    const std::vector<FarTerrainKey> order{
        {0, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::SIXTEEN},
        {0, 0, FarTerrainStep::FOUR},
    };
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());

    REQUIRE(scheduler.retainWanted(wanted, order));
    const FarTerrainSchedulerStats first = scheduler.stats();
    REQUIRE(first.wantedUpdates == 1);
    REQUIRE(first.wantedNoops == 0);

    REQUIRE_FALSE(scheduler.retainWanted(wanted, order));
    const FarTerrainSchedulerStats second = scheduler.stats();
    REQUIRE(second.wantedUpdates == first.wantedUpdates);
    REQUIRE(second.wantedNoops == first.wantedNoops + 1);

    std::vector<FarTerrainKey> reprioritized = order;
    std::swap(reprioritized[0], reprioritized[1]);
    REQUIRE_FALSE(scheduler.retainWanted(wanted, reprioritized));
    const FarTerrainSchedulerStats reprioritizedStats = scheduler.stats();
    REQUIRE(reprioritizedStats.wantedUpdates == first.wantedUpdates);
    REQUIRE(reprioritizedStats.wantedNoops == second.wantedNoops + 1);
}

TEST_CASE("Far terrain cache residency retires obsolete meshes in bounded worker passes",
          "[render][far-terrain][scheduler][cache][performance][concurrency][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    limits.maxMaintenanceEntries = 1;
    limits.maxMaintenanceBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    std::array<FarTerrainKey, 6> keys{};
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = {static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN};
        REQUIRE(scheduler.enqueue(keys[index]));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().cacheEntries == keys.size());

    const std::vector<FarTerrainKey> order{keys[1], keys[4]};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    REQUIRE(scheduler.retainWanted(wanted, order));
    for (int attempt = 0; attempt < 1000; ++attempt) {
        const FarTerrainSchedulerStats current = scheduler.stats();
        if (current.maintenancePending == 0 && current.cacheEntries == wanted.size())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const FarTerrainSchedulerStats maintained = scheduler.stats();
    REQUIRE(maintained.maintenancePending == 0);
    REQUIRE(maintained.cacheEntries == wanted.size());
    REQUIRE(maintained.maintenanceEvicted == keys.size() - wanted.size());
    REQUIRE(maintained.maintenanceScanned >= keys.size());
    REQUIRE(maintained.maintenancePasses >= keys.size());
    REQUIRE(maintained.maximumMaintenanceScanned <= limits.maxMaintenanceEntries);
    REQUIRE(maintained.maximumMaintenanceBytes <= limits.maxMaintenanceBytes);
    REQUIRE(scheduler.findCached(keys[1]));
    REQUIRE(scheduler.findCached(keys[4]));
}

TEST_CASE("Far terrain cache batches preserve nearest useful refinement selection",
          "[render][far-terrain][scheduler][cache][batch][performance]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    constexpr std::array keys{
        FarTerrainKey{0, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{0, 0, FarTerrainStep::EIGHT},
        FarTerrainKey{1, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{1, 0, FarTerrainStep::FOUR},
    };
    for (const FarTerrainKey key : keys)
        REQUIRE(scheduler.enqueue(key));
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == keys.size());
    REQUIRE(scheduler.stats().cacheBaseEntries == 2);

    const std::array baseRequests{
        FarTerrainKey{-1, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{0, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{1, 0, FarTerrainStep::THIRTY_TWO},
    };
    std::vector<std::shared_ptr<const FarTerrainMesh>> batch;
    scheduler.findCachedBatch(baseRequests, 1, batch);
    REQUIRE(batch.size() == 1);
    REQUIRE(batch.front()->key == baseRequests[1]);

    constexpr FarTerrainStepMask BASE_RESIDENT = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    const std::array refinementRequests{
        FarTerrainRefinementCacheRequest{
            {0, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, false},
        FarTerrainRefinementCacheRequest{
            {1, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, false},
        FarTerrainRefinementCacheRequest{
            {2, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, true},
    };
    auto deferredRequests = refinementRequests;
    deferredRequests[0].deferIntermediate = true;
    deferredRequests[1].deferIntermediate = true;
    scheduler.findFinestCachedBatch(deferredRequests, 2, batch);
    REQUIRE(batch.empty());

    scheduler.findFinestCachedBatch(refinementRequests, 2, batch);
    REQUIRE(batch.size() == 2);
    REQUIRE((batch[0]->key == FarTerrainKey{0, 0, FarTerrainStep::EIGHT}));
    REQUIRE((batch[1]->key == FarTerrainKey{1, 0, FarTerrainStep::FOUR}));
    REQUIRE(batch[0] == scheduler.findFinestCached({0, 0}, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_RESIDENT));
    REQUIRE(batch[1] == scheduler.findFinestCached({1, 0}, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_RESIDENT));
}

TEST_CASE("Far terrain submission scans pass cache hits and stop at capacity",
          "[render][far-terrain][scheduler][capacity][cache][performance]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockBuilds = false;
    bool releaseBuilds = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (blockBuilds)
                gateCv.wait(lock, [&] { return releaseBuilds; });
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 32;
    limits.maxCacheEntries = 32;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);

    std::vector<FarTerrainKey> scanOrder;
    for (int64_t x = 0; x < 4; ++x) {
        scanOrder.push_back({x, 0, FarTerrainStep::SIXTEEN});
        REQUIRE(scheduler.enqueue(scanOrder.back(), static_cast<uint32_t>(x)));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == 4);

    for (int64_t x = 4; x < 13; ++x)
        scanOrder.push_back({x, 0, FarTerrainStep::SIXTEEN});
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(scanOrder.begin(),
                                                                      scanOrder.end());
    REQUIRE(scheduler.retainWanted(wanted, scanOrder));
    {
        std::lock_guard lock(gateMutex);
        blockBuilds = true;
    }

    size_t scanned = 0;
    size_t submitted = 0;
    for (size_t index = 0; index < scanOrder.size(); ++index) {
        if (!scheduler.hasSubmissionCapacity())
            break;
        ++scanned;
        if (scheduler.enqueue(scanOrder[index], static_cast<uint32_t>(index)))
            ++submitted;
    }
    REQUIRE(scanned == 12);
    REQUIRE(submitted == limits.maxPending);
    REQUIRE_FALSE(scheduler.hasSubmissionCapacity());
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);

    {
        std::lock_guard lock(gateMutex);
        releaseBuilds = true;
    }
    gateCv.notify_all();
    scheduler.shutdown();
}

TEST_CASE("Far terrain scheduler cancels obsolete view work",
          "[render][far-terrain][scheduler][cancellation][priority]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    size_t enteredWorkers = 0;
    bool released = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            ++enteredWorkers;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return released; });
        }
        return sample(x, z, footprint);
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
        allWorkersEntered = gateCv.wait_for(lock, std::chrono::seconds(30), [&] {
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
    // All AIR, no solid blocks

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
    // Uniform derived light (all zero) so smooth per-vertex lighting cannot
    // split the shaded underside; this isolates the greedy-merge behavior.
    snapshot.derivedSkyLightValid = true;
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
    // Uniform derived light so smooth per-vertex lighting cannot split the
    // shaded underside; this isolates the greedy merge the test targets.
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    // 2x2 square of STONE at local y=8
    snapshot.blocks[MeshSnapshot::index(0, 8, 0)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(1, 8, 0)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(0, 8, 1)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(1, 8, 1)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);

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
    // 2x2 grass floor with one flower on top: the floor's +Y face must still
    // merge into a single quad (flora neither occludes nor casts shade).
    // Uniform derived light keeps the shaded underside from splitting so the
    // test isolates the flora-versus-merge interaction.
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    for (int z = 0; z < 2; ++z)
        for (int x = 0; x < 2; ++x)
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::GRASS;
    snapshot.blocks[MeshSnapshot::index(0, 9, 0)] = BlockType::FLOWER_RED;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);

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

    // Implicit generated water is a full source block in every meshing path.
    // Animation belongs to the fragment normal, not vertex displacement, so
    // exact and far ownership can exchange this planar top without exposing
    // a triangle diagonal.
    size_t topIndexCount = 0;
    for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
        const Vertex& vertex = output.vertices[output.indices[offset]];
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        REQUIRE(static_cast<float>(vertex.py) == 6.0F);
        ++topIndexCount;
    }
    REQUIRE(topIndexCount == 6);
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

    STATIC_REQUIRE(fluidSurfaceHeight(FluidState::source()) == 1.0F);
    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::source().packed();
    MeshOutput source = LODMesher::buildMesh(snapshot, scratch);
    size_t sourceTopIndexCount = 0;
    for (size_t offset = source.opaqueIndexCount; offset + 2 < source.indices.size(); offset += 3) {
        const Vertex& first = source.vertices[source.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        std::array<Vec3, 3> triangle{};
        for (size_t corner = 0; corner < triangle.size(); ++corner) {
            const Vertex& vertex = source.vertices[source.indices[offset + corner]];
            REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
            REQUIRE(static_cast<float>(vertex.py) == 9.0F);
            triangle[corner] = {static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                                static_cast<float>(vertex.pz)};
            ++sourceTopIndexCount;
        }
        const Vec3 normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
        REQUIRE(std::abs(normal.x) <= 1.0e-6F);
        REQUIRE(std::abs(normal.z) <= 1.0e-6F);
        REQUIRE(std::abs(normal.y) > 0.0F);
    }
    REQUIRE(sourceTopIndexCount == 6);

    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::falling(3).packed();
    MeshOutput falling = LODMesher::buildMesh(snapshot, scratch);
    bool foundFallingFace = false;
    for (const Vertex& vertex : falling.vertices) {
        if (unpackFluidFalling(vertex.faceAttr))
            foundFallingFace = true;
    }
    REQUIRE(foundFallingFace);
}

TEST_CASE("Snapshot water keeps exterior reflection authority separate from skylight",
          "[render][mesher][water][lighting]") {
    constexpr int waterX = 8;
    constexpr int waterY = 8;
    constexpr int waterZ = 8;
    constexpr int32_t receiverWorldY = 4 * CHUNK_EDGE + waterY + 1;

    auto topHasExteriorSky = [](const MeshOutput& output) {
        bool foundTop = false;
        bool exteriorSky = false;
        for (const Vertex& vertex : output.vertices) {
            if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
                static_cast<float>(vertex.py) != static_cast<float>(waterY + 1)) {
                continue;
            }
            foundTop = true;
            exteriorSky = exteriorSky || unpackFluidExteriorSky(vertex.faceAttr);
        }
        REQUIRE(foundTop);
        return exteriorSky;
    };

    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.derivedSkyLightValid = true;
    snapshot.blocks[MeshSnapshot::index(waterX, waterY, waterZ)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(waterX, waterY, waterZ)] =
        FluidState::source().packed();
    snapshot.skyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] =
        MeshSnapshot::SKY_CUTOFF_INCOMPLETE;
    snapshot.visualSkyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] = receiverWorldY;

    MeshScratch scratch;
    REQUIRE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));

    // An edited roof must still suppress reflection while the packed skylight
    // remains conservatively dark.
    snapshot.visualSkyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] = receiverWorldY + 1;
    REQUIRE_FALSE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));

    // Actual propagated skylight reopens a cave mouth without relying on the
    // incomplete-column visual cutoff.
    snapshot.packedLight[MeshSnapshot::index(waterX, waterY + 1, waterZ)] = packDerivedLight(1, 0);
    REQUIRE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));
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
    constexpr int64_t RIVER_X = -12'801;
    constexpr int64_t RIVER_Z = 2'759;
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
    const int RIVER_LOCAL_Z = Chunk::worldToLocal(RIVER_Z);
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
// Neighbor-aware (snapshot) meshing, chunk border correctness
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
    // element past the vector's end, slow heap corruption that surfaced as
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

TEST_CASE("Segmented far arena grows lazily and routes allocations to their slab",
          "[render][megabuffer][far-terrain][residency]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    constexpr uint64_t SLAB_BYTES = 4 * 1024;
    SegmentedMegaBuffer arena(device, SLAB_BYTES * 2, SLAB_BYTES * 2, SLAB_BYTES, SLAB_BYTES);
    REQUIRE(arena.segmentCount() == 0);

    constexpr uint32_t VERTEX_COUNT = 200;
    constexpr uint32_t INDEX_COUNT = 400;
    std::vector<Vertex> vertices(VERTEX_COUNT);
    std::vector<uint32_t> indices(INDEX_COUNT);
    auto first = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 1);
    arena.uploadVertices(vertices.data(), vertices.size() * sizeof(Vertex), first);
    arena.uploadIndices(indices.data(), indices.size() * sizeof(uint32_t), first);

    auto second = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 2);
    REQUIRE(second.vertexBuffer != first.vertexBuffer);
    REQUIRE(second.indexBuffer != first.indexBuffer);
    const uint64_t usedBeforeDeferred = arena.vertexUsed() + arena.indexUsed();
    arena.deferFree(first, 7);
    arena.drainDeferredFrees(6);
    REQUIRE(arena.vertexUsed() + arena.indexUsed() == usedBeforeDeferred);
    arena.drainDeferredFrees(7);
    REQUIRE(arena.vertexUsed() + arena.indexUsed() < usedBeforeDeferred);

    auto reused = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 2);
    REQUIRE(reused.vertexBuffer != second.vertexBuffer);
    arena.free(reused);
    arena.free(second);
    REQUIRE(arena.vertexUsed() == 0);
    REQUIRE(arena.indexUsed() == 0);
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

TEST_CASE("Exact mesh scheduling reserves capacity and ordering for the camera band",
          "[render][scheduler][priority][cold-start][regression]") {
    STATIC_REQUIRE(EXACT_MESH_CAMERA_RESERVED_SLOTS == 32);
    STATIC_REQUIRE(EXACT_MESH_MAX_INFLIGHT == 64);

    REQUIRE(meshLaneCanReserve(31, 0, MeshPriorityLane::BROAD_SURFACE));
    REQUIRE_FALSE(meshLaneCanReserve(32, 0, MeshPriorityLane::BROAD_SURFACE));
    REQUIRE(meshLaneCanReserve(32, 0, MeshPriorityLane::CAMERA_BAND));
    REQUIRE(meshLaneCanReserve(32, 0, MeshPriorityLane::CAMERA_COLUMN));
    REQUIRE(meshLaneCanReserve(63, 0, MeshPriorityLane::CAMERA_BAND));
    REQUIRE_FALSE(meshLaneCanReserve(63, 1, MeshPriorityLane::CAMERA_BAND));

    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4'096, 8,
                               MeshPriorityLane::BROAD_SURFACE, 0, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_COLUMN, 4'096, 8,
                               MeshPriorityLane::CAMERA_BAND, 0, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4, 9, MeshPriorityLane::CAMERA_BAND,
                               64, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4, 9, MeshPriorityLane::CAMERA_BAND,
                               4, 10));
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
    REQUIRE(snapshot.skyLightAt(CHUNK_EDGE, 8, 8) == 0);
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
    REQUIRE(snapshot.blockLightAt(CHUNK_EDGE, 7, 8) == 9);
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
    // normally lit terrain material. A solid roof isolates the three air
    // blocks below it, which still represent a cave opening and must remain
    // sealed and dark.
    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = BlockType::GRASS;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            snapshot.blocks[MeshSnapshot::index(x, 3, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t sealedVertices = 0;
    size_t litSurfaceVertices = 0;
    float highestCapY = 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        highestCapY = std::max(highestCapY, static_cast<float>(vertex.py));
        if (unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::STONE)) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++sealedVertices;
        } else if (unpackTextureLayer(vertex.faceAttr) == TEXTURE_LAYER_GRASS_SIDE) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++litSurfaceVertices;
        }
    }
    REQUIRE(sealedVertices == 3 * CHUNK_EDGE * 4);
    REQUIRE(litSurfaceVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(highestCapY == 8.0F);
    REQUIRE(sealedVertices + litSurfaceVertices < CHUNK_EDGE * CHUNK_EDGE * 4);
}

TEST_CASE("Missing caps recognize outdoor air beneath a generated overhang",
          "[world][mesher][border][streaming][surface][overhang]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = BlockType::GRASS;
    }
    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            snapshot.blocks[MeshSnapshot::index(x, 3, z)] = BlockType::STONE;
        }
    }

    // This column's generated top describes an overhang at world Y=71. Its
    // undercut air joins the neighboring outdoor column inside the snapshot,
    // so the three cells beneath the roof and four above it all need outdoor
    // lighting even though the undercut lies below its column cutoff.
    constexpr int overhangZ = 8;
    snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, overhangZ)] = 72;
    snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, overhangZ)] = 76;
    snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, overhangZ)] =
        BlockType::LIMESTONE;
    snapshot.blocks[MeshSnapshot::index(CHUNK_EDGE - 1, 7, overhangZ)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t litOverhangVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE) ||
            unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::LIMESTONE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++litOverhangVertices;
    }
    REQUIRE(litOverhangVertices == 7 * 4);
}

TEST_CASE("Edited roofs keep enclosed missing-neighbor caps dark",
          "[world][mesher][border][streaming][surface][roof][underground]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 73;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t enclosedVertices = 0;
    size_t litVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        if (unpackSkyLight(vertex.faceAttr) == 0) {
            REQUIRE(unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::STONE));
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++enclosedVertices;
        } else {
            ++litVertices;
        }
    }
    REQUIRE(enclosedVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(litVertices == 0);
}

TEST_CASE("Top-of-world roofs remain distinct from incomplete sky paths",
          "[world][mesher][border][streaming][surface][roof][limit]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, WORLD_MAX_CHUNK_Y, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 500;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = WORLD_MAX_Y + 1;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = WORLD_MAX_Y + 1;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = WORLD_MAX_Y + 1;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
            snapshot.blocks[MeshSnapshot::index(x, CHUNK_EDGE - 1, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t enclosedVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++enclosedVertices;
    }
    REQUIRE(enclosedVertices == 11 * CHUNK_EDGE * 4);
}

TEST_CASE("Lowered exact sky cutoffs light opened missing-neighbor caps",
          "[world][mesher][border][streaming][surface][edit]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 96;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 80;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 80;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] =
            BlockType::LIMESTONE;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t litVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++litVertices;
    }
    REQUIRE(litVertices == 12 * CHUNK_EDGE * 4);
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

    // Settle the queued generation-time reconciles first. An edit floods its
    // cube synchronously, and against stale initial light that flood would
    // legitimately change border light and mark neighbors.
    for (int pass = 0; pass < 8; ++pass)
        world.reconcileLight(256);
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

    // buildMesh is pure, it does not modify the chunk
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

TEST_CASE("UI icon vertices share one layout between C++ and Metal", "[render][ui]") {
    REQUIRE(sizeof(UIIconVertex) == 48);
    REQUIRE(offsetof(UIIconVertex, position) == 0);
    REQUIRE(offsetof(UIIconVertex, uv) == 8);
    REQUIRE(offsetof(UIIconVertex, tint) == 16);
    REQUIRE(offsetof(UIIconVertex, layer) == 32);
}

TEST_CASE("Item icon layers map the non-block range past the block layers", "[render][textures]") {
    REQUIRE(TEXTURE_LAYER_ITEM_FIRST == TEXTURE_LAYER_COUNT);
    REQUIRE(itemIconLayer(ItemType::STICK) == TEXTURE_LAYER_ITEM_FIRST);
    REQUIRE(itemIconLayer(static_cast<ItemType>(static_cast<uint16_t>(ItemType::COUNT) - 1)) ==
            TEXTURE_LAYER_ITEM_FIRST + NON_BLOCK_ITEM_COUNT - 1);
    REQUIRE(TEXTURE_LAYER_TOTAL == TEXTURE_LAYER_ITEM_FIRST + ITEM_ICON_COUNT);
    REQUIRE(TEXTURE_LAYER_TOTAL <= 255);
}

TEST_CASE("Block textures: workshop blocks use per-face layers", "[render][textures]") {
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::PLUS_Y) ==
            TEXTURE_LAYER_CRAFTING_TABLE_TOP);
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::MINUS_Y) ==
            static_cast<uint8_t>(BlockType::PLANKS));
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::PLUS_X) ==
            static_cast<uint8_t>(BlockType::CRAFTING_TABLE));
    for (BlockType furnace : {BlockType::FURNACE, BlockType::FURNACE_LIT}) {
        REQUIRE(textureLayerFor(furnace, FaceNormal::PLUS_Y) == TEXTURE_LAYER_FURNACE_TOP);
        REQUIRE(textureLayerFor(furnace, FaceNormal::MINUS_Y) == TEXTURE_LAYER_FURNACE_TOP);
        REQUIRE(textureLayerFor(furnace, FaceNormal::MINUS_X) == static_cast<uint8_t>(furnace));
    }
    REQUIRE(textureLayerFor(BlockType::TORCH, FaceNormal::PLUS_X) ==
            static_cast<uint8_t>(BlockType::TORCH));
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
    const uint32_t attr =
        packFluidFaceAttr(FaceNormal::PLUS_Z, skyLight, 5, true, blockLight, true);
    const uint32_t sealed =
        packFluidFaceAttr(FaceNormal::PLUS_Z, skyLight, 5, true, blockLight, false);

    REQUIRE(unpackFace(attr) == FaceNormal::PLUS_Z);
    REQUIRE(unpackTextureLayer(attr) == static_cast<uint8_t>(BlockType::WATER));
    REQUIRE(unpackSkyLight(attr) == skyLight);
    REQUIRE(unpackCornerAO(attr) == 3);
    REQUIRE(unpackBlockLight(attr) == blockLight);
    REQUIRE_FALSE(unpackEmissive(attr));
    REQUIRE(unpackSway(attr) == 0);
    REQUIRE(unpackFluidDirection(attr) == 5);
    REQUIRE(unpackFluidFalling(attr));
    REQUIRE(unpackFluidExteriorSky(attr));
    REQUIRE_FALSE(unpackFluidExteriorSky(sealed));
    REQUIRE((attr & ~FLUID_EXTERIOR_SKY_ATTRIBUTE_MASK) ==
            (sealed & ~FLUID_EXTERIOR_SKY_ATTRIBUTE_MASK));
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
    // Smooth lighting averages each corner over the outward-plane 3x3 patch, so
    // fill the patch uniformly to keep every corner at the tested nibble.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 12;

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

TEST_CASE("Snapshot mesher decodes independent packed light channels",
          "[render][mesher][light][skylight]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // Uniform patch so each smoothed corner reads the same skylight and block
    // light nibble the two channels are asserted against.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 0xB5;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundBoundary = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 11);
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 5);
            foundBoundary = true;
        }
    }
    REQUIRE(foundBoundary);
}

TEST_CASE("Snapshot mesher smooths block light per vertex across a face",
          "[render][mesher][light][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // Block light rising along +Z in the air the +X face looks into: 0 at z=7,
    // 8 at z=8, 15 at z=9. Smooth lighting must give the +Z-side corners more
    // block light than the -Z-side corners instead of one flat value.
    for (int dy = -1; dy <= 1; ++dy) {
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 7)] = 0x00;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8)] = 0x08;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 9)] = 0x0F;
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    uint8_t low = 255;
    uint8_t high = 0;
    int faceVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            const uint8_t value = unpackBlockLight(vertex.faceAttr);
            low = std::min(low, value);
            high = std::max(high, value);
            ++faceVertices;
        }
    }
    REQUIRE(faceVertices == 4);
    REQUIRE(low < high); // a gradient, not one flat per-face value
    REQUIRE(low <= 5);   // the -Z corners see the dark end
    REQUIRE(high >= 10); // the +Z corners see the bright end
}

TEST_CASE("Snapshot mesher merges a uniformly lit face into one quad",
          "[render][mesher][light][smooth][greedy]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    // A 3x3 stone slab with uniform light in the air above it must still merge
    // its top face to one quad despite the widened per-corner key.
    for (int z = 6; z <= 8; ++z)
        for (int x = 6; x <= 8; ++x)
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::STONE;
    for (int z = 5; z <= 9; ++z)
        for (int x = 5; x <= 9; ++x)
            snapshot.packedLight[MeshSnapshot::index(x, 9, z)] = 0xF7; // sky 15, block 7

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    int topVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y) {
            ++topVertices;
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 7);
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        }
    }
    REQUIRE(topVertices == 4);
}

TEST_CASE("Snapshot mesher excludes opaque neighbors from smoothed corners",
          "[render][mesher][light][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // A solid neighbor sits beside the lit +X face. Its cell stores no
    // propagated light, so a corner touching it must average only the lit cells
    // and stay at 8 rather than being pulled toward zero (which would read 6).
    snapshot.blocks[MeshSnapshot::index(16, 8, 9)] = BlockType::STONE;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 0x08;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool found = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 8);
            found = true;
        }
    }
    REQUIRE(found);
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
    // Fill the 3x3 column patch the top-face corners read so smooth lighting
    // keeps the shaded top uniformly dark instead of blending in lit neighbors.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx)
            snapshot.skyCutoffY[MeshSnapshot::skyIndex(8 + dx, 8 + dz)] = 96;

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
    REQUIRE(separated.skyCutoffAt(8, 8) == MeshSnapshot::SKY_CUTOFF_INCOMPLETE);

    for (int32_t section = targetSection + 2; section <= surfaceSection; ++section) {
        REQUIRE(world.getChunk({0, section, 0}));
    }
    MeshSnapshot connected;
    REQUIRE(world.snapshotForMeshing(target, connected));
    REQUIRE(connected.skyCutoffAt(8, 8) <= WORLD_MAX_Y + 1);
}

TEST_CASE("Generated opaque features extend the exact density sky cutoff",
          "[render][mesher][light][skylight][feature]") {
    World world(42, 4);
    constexpr int64_t MINIMUM_X = -27'392;
    constexpr int64_t MINIMUM_Z = -17'152;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 512;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 512;
    const std::vector<FarCanopy> canopies =
        world.generator().collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z);
    const auto selected = std::ranges::find_if(canopies, [](const FarCanopy& canopy) {
        return canopy.logBlock != BlockType::AIR &&
               canopy.species != feature_generation::TreeSpecies::FALLEN_LOG;
    });
    REQUIRE(selected != canopies.end());
    const int64_t worldX = selected->x;
    const int64_t worldZ = selected->z;
    const ChunkPos target{Chunk::worldToChunk(worldX), Chunk::worldToChunkY(selected->baseY - 1),
                          Chunk::worldToChunk(worldZ)};
    const int32_t maximumTreeSection = Chunk::worldToChunkY(selected->topY);
    for (int32_t chunkY = target.y - 1; chunkY <= std::max(target.y + 1, maximumTreeSection);
         ++chunkY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                REQUIRE(world.getChunk({target.x + offsetX, chunkY, target.z + offsetZ}));
            }
        }
    }

    const auto plan = world.generator().getColumnPlan({target.x, target.z});
    const int plannedSurface =
        plan->surfaceY(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ));
    const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(worldX, worldZ);
    REQUIRE(loadedTop);
    REQUIRE(*loadedTop > plannedSurface);
    REQUIRE(world.getBlockIfLoaded(worldX, *loadedTop, worldZ) == selected->logBlock);

    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing(target, snapshot));
    REQUIRE(snapshot.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ))] == *loadedTop + 1);
    REQUIRE(snapshot.visualSkyCutoffAt(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ)) ==
            *loadedTop + 1);
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
    REQUIRE(covered.visualSkyCutoffAt(Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z)) ==
            roofY + 1);

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
    REQUIRE(opened.visualSkyCutoffAt(Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z)) ==
            *restoredTop + 1);
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
    REQUIRE(first.texture().arrayLength == TEXTURE_LAYER_TOTAL);

    const uint64_t firstHash = blockTextureHash(first);
    REQUIRE(blockTextureHash(second) == firstHash);
    REQUIRE(firstHash == 0x5bb2a0c9c1913dd6ULL);
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

// Hable "Uncharted 2" filmic tonemap replicated from post.metal (the
// composite owns tonemapping; this replaced Uchimura, whose shoulder
// plateaued the HDR sun to flat white within a few stops). Pins the curve's
// contract: black stays black, mids survive the fixed 2x gain near identity,
// highlights keep compressing across many stops, and it never decreases.
static float hableFilmicToneMap(float x) {
    auto curve = [](float v) {
        const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
    };
    const float W = 11.2f;             // linear white point
    return curve(x * 2.0f) / curve(W); // 2x gain matches displayColor
}

TEST_CASE("Post: filmic tone mapping curve", "[hdr][post]") {
    // Black in, black out
    REQUIRE(hableFilmicToneMap(0.0f) == Catch::Approx(0.0f).margin(0.001f));

    // The raw curve sits about a stop under identity at the mids; the
    // exposure key (0.85 in encodeExposure) compensates in the live path
    float atMid = hableFilmicToneMap(0.5f);
    REQUIRE(atMid > 0.22f);
    REQUIRE(atMid < 0.55f);

    // HDR highlights compress below display max and keep separating up to
    // the white point (5.6 scene units after the 2x gain); the auto-exposure
    // stop-down keeps the HDR-8 sun disc below it in the live path
    REQUIRE(hableFilmicToneMap(4.0f) < 1.0f);
    REQUIRE(hableFilmicToneMap(4.0f) - hableFilmicToneMap(2.0f) > 0.05f);
    REQUIRE(hableFilmicToneMap(8.0f) >= 1.0f); // past white: display max by design

    // Monotonically increasing across the range
    REQUIRE(hableFilmicToneMap(0.2f) < hableFilmicToneMap(0.5f));
    REQUIRE(hableFilmicToneMap(0.5f) < hableFilmicToneMap(1.0f));
    REQUIRE(hableFilmicToneMap(1.0f) < hableFilmicToneMap(2.0f));
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

TEST_CASE("Bloom: extract threshold, bright pixels pass, dark pixels blocked", "[phase8][bloom]") {
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

// ---- Shared shader struct layout pins ----
// shader_types.hpp is compiled by BOTH clang++ and the Metal compiler; simd
// types have the same layout in each. These pins catch accidental drift
// (reordered fields, ad-hoc padding) that previously corrupted fog, camera
// position, sky colors, and particle data.

TEST_CASE("Shader types: Uniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(FoliageWindUniforms) == 16);
    REQUIRE(offsetof(FoliageWindUniforms, direction) == 0);
    REQUIRE(offsetof(FoliageWindUniforms, speedBlocksPerSecond) == 8);
    REQUIRE(offsetof(FoliageWindUniforms, strength) == 12);
    REQUIRE(sizeof(Uniforms) == 320);
    REQUIRE(offsetof(Uniforms, sunDirection) == 192);
    REQUIRE(offsetof(Uniforms, fogColor) == 240);
    REQUIRE(offsetof(Uniforms, fogDensity) == 256);
    REQUIRE(offsetof(Uniforms, cameraPosition) == 272);
    REQUIRE(offsetof(Uniforms, foliageWind) == 288);
    REQUIRE(offsetof(Uniforms, time) == 304);
    REQUIRE(offsetof(Uniforms, wetness) == 308);
    REQUIRE(alignof(Uniforms) == 16);
    REQUIRE(sizeof(EntityModel) == 80);
    REQUIRE(offsetof(EntityModel, lighting) == 64);
    REQUIRE(sizeof(ChunkOrigin) == 48);
    REQUIRE(offsetof(ChunkOrigin, farMetadata) == 32);
    REQUIRE(sizeof(FarTerrainOwnershipUniforms) == 288);
    REQUIRE(offsetof(FarTerrainOwnershipUniforms, readyColumnMasks) == 0);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_WORD_COUNT * FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD ==
                   FAR_TERRAIN_EXACT_COLUMNS_PER_TILE * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE *
                       FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR ==
                   FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE * FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE ==
                   FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT);
    STATIC_REQUIRE(sizeof(FarTerrainOwnershipUniforms) ==
                   FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT * FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE *
                       sizeof(simd_uint4));
    STATIC_REQUIRE(FarTerrainExactHandoff::COLUMN_MASK_WORD_COUNT ==
                   FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
}

TEST_CASE("Shader types: ShadowUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ShadowPassUniforms) == 112);
    REQUIRE(offsetof(ShadowPassUniforms, projectionOrigin) == 64);
    REQUIRE(offsetof(ShadowPassUniforms, foliageWind) == 80);
    REQUIRE(offsetof(ShadowPassUniforms, time) == 96);
    REQUIRE(sizeof(ShadowCascadeUniforms) == 112);
    REQUIRE(offsetof(ShadowCascadeUniforms, projectionOrigin) == 64);
    REQUIRE(offsetof(ShadowCascadeUniforms, depthRange) == 80);
    REQUIRE(offsetof(ShadowCascadeUniforms, samplingParams) == 96);
    REQUIRE(sizeof(ShadowUniforms) == 592);
    REQUIRE(offsetof(ShadowUniforms, cameraPositionAndStrength) == 560);
    REQUIRE(offsetof(ShadowUniforms, cameraForwardAndPadding) == 576);
    REQUIRE(SHADOW_DETAILED_CASCADE_COUNT == 4);
    REQUIRE(SHADOW_CASCADE_COUNT == 5);
    REQUIRE(SHADOW_HORIZON_CASCADE_INDEX == 4);
}

TEST_CASE("Foliage wind preserves canonical weather direction and physical speed",
          "[render][weather][sway][shader-types]") {
    WeatherSample localWeather{};
    localWeather.windBlocksPerSecond = {3.0F, 4.0F};
    const FoliageWindUniforms wind = makeFoliageWindUniforms(
        localWeather.windBlocksPerSecond.x, localWeather.windBlocksPerSecond.y, true);

    REQUIRE(wind.direction.x == Catch::Approx(0.6F));
    REQUIRE(wind.direction.y == Catch::Approx(0.8F));
    REQUIRE(wind.speedBlocksPerSecond == Catch::Approx(5.0F));
    REQUIRE(wind.strength == Catch::Approx(1.0F));

    Uniforms scene{};
    ShadowPassUniforms shadow{};
    scene.foliageWind = wind;
    shadow.foliageWind = wind;
    REQUIRE(scene.foliageWind.direction.x == shadow.foliageWind.direction.x);
    REQUIRE(scene.foliageWind.direction.y == shadow.foliageWind.direction.y);
    REQUIRE(scene.foliageWind.speedBlocksPerSecond == shadow.foliageWind.speedBlocksPerSecond);
    REQUIRE(scene.foliageWind.strength == shadow.foliageWind.strength);

    const FoliageWindUniforms disabled = makeFoliageWindUniforms(
        localWeather.windBlocksPerSecond.x, localWeather.windBlocksPerSecond.y, false);
    REQUIRE(disabled.direction.x == Catch::Approx(wind.direction.x));
    REQUIRE(disabled.direction.y == Catch::Approx(wind.direction.y));
    REQUIRE(disabled.speedBlocksPerSecond == Catch::Approx(wind.speedBlocksPerSecond));
    REQUIRE(disabled.strength == Catch::Approx(0.0F));

    const FoliageWindUniforms bounded = makeFoliageWindUniforms(24.0F, 0.0F, true);
    REQUIRE(bounded.direction.x == Catch::Approx(1.0F));
    REQUIRE(bounded.direction.y == Catch::Approx(0.0F));
    REQUIRE(bounded.speedBlocksPerSecond == Catch::Approx(FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND));
}

TEST_CASE("Deferred shadow cascades hold foliage casters static", "[render][shadow][sway]") {
    FoliageWindUniforms wind{};
    wind.direction = simd_make_float2(0.6F, 0.8F);
    wind.speedBlocksPerSecond = 5.0F;
    wind.strength = 1.0F;

    for (uint32_t cascade = 0U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const FoliageWindUniforms casterWind = shadowFoliageWindForCascade(wind, cascade);
        REQUIRE(casterWind.direction.x == Catch::Approx(wind.direction.x));
        REQUIRE(casterWind.direction.y == Catch::Approx(wind.direction.y));
        REQUIRE(casterWind.speedBlocksPerSecond == Catch::Approx(wind.speedBlocksPerSecond));
        REQUIRE(casterWind.strength == Catch::Approx(cascade < 2U ? wind.strength : 0.0F));
        REQUIRE(shadowCascadeUsesAnimatedFoliage(cascade) == (cascade < 2U));
    }

    wind.strength = 0.0F;
    for (uint32_t cascade = 0U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        REQUIRE(shadowFoliageWindForCascade(wind, cascade).strength == Catch::Approx(0.0F));
    }
}

TEST_CASE("Shadow cascades: quality table pins splits targets and slices", "[render][shadow]") {
    const std::array<float, SHADOW_CASCADE_COUNT> highFar = {48.0f, 160.0f, 512.0f, 1536.0f,
                                                             8192.0f};
    const std::array<float, SHADOW_CASCADE_COUNT> mediumFar = {40.0f, 128.0f, 384.0f, 768.0f,
                                                               8192.0f};
    const std::array<uint32_t, SHADOW_CASCADE_COUNT> highResolution = {4096u, 4096u, 2048u, 2048u,
                                                                       2048u};
    const std::array<uint32_t, SHADOW_CASCADE_COUNT> mediumResolution = {2048u, 2048u, 1024u, 1024u,
                                                                         1024u};

    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration high = shadowCascadeConfiguration(2u, cascade);
        const ShadowCascadeConfiguration medium = shadowCascadeConfiguration(1u, cascade);
        REQUIRE(high.farDepth == Catch::Approx(highFar[cascade]));
        REQUIRE(medium.farDepth == Catch::Approx(mediumFar[cascade]));
        REQUIRE(high.resolution == highResolution[cascade]);
        REQUIRE(medium.resolution == mediumResolution[cascade]);
        if (cascade < 2u) {
            REQUIRE(high.textureGroup == ShadowTextureGroup::NEAR);
            REQUIRE(high.textureSlice == cascade);
        } else if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
            REQUIRE(high.textureGroup == ShadowTextureGroup::FAR);
            REQUIRE(high.textureSlice == cascade - 2u);
        } else {
            REQUIRE(high.textureGroup == ShadowTextureGroup::HORIZON);
            REQUIRE(high.textureSlice == 0u);
        }
    }
}

TEST_CASE("Shadow cascades: overlap selection uses camera-forward view depth", "[render][shadow]") {
    ShadowUniforms shadow{};
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration configuration = shadowCascadeConfiguration(2u, cascade);
        shadow.cascades[cascade].depthRange =
            simd_make_float4(configuration.nearDepth, configuration.farDepth,
                             shadowCascadeBlendStart(configuration), 1.0f);
    }

    const simd_float3 camera = simd_make_float3(10.0f, 2.0f, 3.0f);
    const simd_float3 forward = simd_make_float3(0.0f, 0.0f, -1.0f);
    REQUIRE(shadowViewDepth(simd_make_float3(10.0f, 2.0f, -7.0f), camera, forward) ==
            Catch::Approx(10.0f));
    REQUIRE(shadowViewDepth(simd_make_float3(10.0f, 2.0f, 13.0f), camera, forward) ==
            Catch::Approx(-10.0f));

    const float firstBlendStart = shadowCascadeBlendStart(shadowCascadeConfiguration(2u, 0u));
    ShadowCascadeSelection beforeBlend = shadowCascadeSelection(firstBlendStart - 0.01f, shadow);
    REQUIRE(beforeBlend.primary == 0u);
    REQUIRE(beforeBlend.secondary == 0u);
    REQUIRE(beforeBlend.secondaryWeight == Catch::Approx(0.0f));

    ShadowCascadeSelection insideBlend = shadowCascadeSelection(45.0f, shadow);
    REQUIRE(insideBlend.primary == 0u);
    REQUIRE(insideBlend.secondary == 1u);
    REQUIRE(insideBlend.secondaryWeight > 0.0f);
    REQUIRE(insideBlend.secondaryWeight < 1.0f);

    ShadowCascadeSelection horizonBlend = shadowCascadeSelection(1500.0f, shadow);
    REQUIRE(horizonBlend.primary == 3u);
    REQUIRE(horizonBlend.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizonBlend.secondaryWeight > 0.0f);
    REQUIRE(horizonBlend.secondaryWeight < 1.0f);

    ShadowCascadeSelection horizon = shadowCascadeSelection(2000.0f, shadow);
    REQUIRE(horizon.primary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizon.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizon.covered == 1u);

    const ShadowCascadeConfiguration horizonConfiguration =
        shadowCascadeConfiguration(2u, SHADOW_HORIZON_CASCADE_INDEX);
    const float terminalStart = shadowCascadeBlendStart(horizonConfiguration);
    ShadowCascadeSelection terminalFade =
        shadowCascadeSelection(0.5F * (terminalStart + horizonConfiguration.farDepth), shadow);
    REQUIRE(terminalFade.primary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(terminalFade.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(terminalFade.exteriorWeight > 0.0F);
    REQUIRE(terminalFade.exteriorWeight < 1.0F);
    REQUIRE(shadowCascadeSelection(horizonConfiguration.farDepth, shadow).exteriorWeight ==
            Catch::Approx(1.0F));
    REQUIRE(shadowCascadeSelection(9000.0f, shadow).covered == 0u);
    REQUIRE(shadowCascadeSelection(-1.0f, shadow).covered == 0u);
}

TEST_CASE("Shadow cascades: overlap and texel snap metadata are stable", "[render][shadow]") {
    for (uint32_t quality : {1u, 2u}) {
        for (uint32_t cascade = 0; cascade < SHADOW_HORIZON_CASCADE_INDEX; ++cascade) {
            const ShadowCascadeConfiguration configuration =
                shadowCascadeConfiguration(quality, cascade);
            REQUIRE(configuration.farDepth - shadowCascadeBlendStart(configuration) ==
                    Catch::Approx((configuration.farDepth - configuration.nearDepth) *
                                  SHADOW_CASCADE_BLEND_FRACTION));
        }
    }

    const ShadowCascadeConfiguration first = shadowCascadeConfiguration(2u, 0u);
    REQUIRE(shadowCascadeBlendStart(first) == Catch::Approx(42.0625f));

    const float tanHalfFov = std::tan(70.0F * static_cast<float>(M_PI) / 360.0F);
    const float analytic = shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 1.5F);
    REQUIRE(analytic == shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 1.5F));
    REQUIRE(std::fmod(analytic * 16.0F, 1.0F) == Catch::Approx(0.0F).margin(1.0e-6F));
    // Reconstructing absolute frustum corners at the acceptance coordinate
    // alternated between 66.3125 and 66.375 while moving. The analytic native
    // route radius has one exact authority independent of camera position.
    REQUIRE(shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 3456.0F / 2234.0F) ==
            Catch::Approx(66.3125F));

    const double texelWorldSize = 2.0 * analytic / first.resolution;
    const double lightCoordinate = -102'753.123456;
    const double snapped = shadowSnappedLightCoordinate(lightCoordinate, texelWorldSize);
    REQUIRE(snapped / texelWorldSize ==
            Catch::Approx(std::round(lightCoordinate / texelWorldSize)));
    REQUIRE(std::abs(snapped - lightCoordinate) <= texelWorldSize * 0.5);
}

TEST_CASE("Shadow cascades: refresh cadence bounds stale depth", "[render][shadow]") {
    REQUIRE(shadowCascadeMaximumRefreshInterval(0U) == 1U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(1U) == 1U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(2U) == 2U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(3U) == 4U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(4U) == 8U);
}

TEST_CASE("Shadow cascades: deferred depth coverage refresh is snap-safe", "[render][shadow]") {
    const Mat4 rendered = Mat4::identity();
    const Vec3 origin{22'784.0F, 0.0F, -111'872.0F};
    Mat4 candidate = rendered;
    REQUIRE_FALSE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));

    candidate(0, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(1, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(2, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(3, 3) += 0.125F;
    REQUIRE_FALSE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered,
                                           origin + Vec3{256.0F, 0.0F, 0.0F}));

    constexpr float radius = 10.0F;
    constexpr float casterMargin = 2.0F;
    constexpr float normalBias = 0.5F;
    constexpr uint32_t resolution = 1024U;
    const float guard =
        shadowCascadeReceiverDepthGuard(radius, casterMargin, resolution, normalBias);
    const float depthRange = shadowCascadeDepthRange(radius, casterMargin, guard);
    const float depthTexel =
        shadowCascadeDepthTexelWorldSize(radius, casterMargin, guard, resolution);
    REQUIRE(guard >= normalBias + depthTexel);

    const double depthCenter = static_cast<double>(depthTexel) * 1'024.0;
    REQUIRE(shadowSnappedLightCoordinate(depthCenter + 0.49 * depthTexel, depthTexel) ==
            Catch::Approx(shadowSnappedLightCoordinate(depthCenter, depthTexel)));
    REQUIRE(shadowSnappedLightCoordinate(depthCenter + 0.51 * depthTexel, depthTexel) !=
            Catch::Approx(shadowSnappedLightCoordinate(depthCenter, depthTexel)));

    const Vec3 receiverCenter{};
    const Vec3 light{0.0F, 0.0F, 1.0F};
    const Mat4 sampled =
        Mat4::orthographic(-radius, radius, -radius, radius, 0.0F, depthRange) *
        Mat4::lookAt(light * (radius + casterMargin + guard), receiverCenter, Vec3::up());
    REQUIRE(shadowCascadeReceiverDepthCovered(sampled, Vec3::zero(), receiverCenter, radius));
    REQUIRE_FALSE(shadowCascadeReceiverDepthCovered(
        sampled, Vec3::zero(), receiverCenter - light * (guard + 0.25F), radius));
}

TEST_CASE("Shadow cascades: projection anchors preserve large-coordinate precision",
          "[render][shadow][large-coordinate]") {
    const Vec3 camera{23'029.0F, 225.0F, -111'726.0F};
    const Vec3 origin = shadowProjectionOrigin(camera);
    REQUIRE(origin.x == Catch::Approx(22'784.0F));
    REQUIRE(origin.y == Catch::Approx(0.0F));
    REQUIRE(origin.z == Catch::Approx(-111'872.0F));

    const Vec3 local = camera - origin;
    REQUIRE(local.x == Catch::Approx(245.0F));
    REQUIRE(local.y == Catch::Approx(225.0F));
    REQUIRE(local.z == Catch::Approx(146.0F));
    REQUIRE(std::abs(local.x) < 256.0F);
    REQUIRE(std::abs(local.y) < 256.0F);
    REQUIRE(std::abs(local.z) < 256.0F);

    const Vec3 lightAxis = Vec3{0.37F, 0.81F, -0.45F}.normalize();
    const double expected = static_cast<double>(camera.x) * lightAxis.x +
                            static_cast<double>(camera.y) * lightAxis.y +
                            static_cast<double>(camera.z) * lightAxis.z;
    REQUIRE(shadowPreciseDot(camera, lightAxis) == Catch::Approx(expected));
}

TEST_CASE("Shadow cascades: nearby entities do not invalidate coarse high-sun slices",
          "[render][shadow][entity]") {
    const Vec3 camera{0.0F, 64.0F, 0.0F};
    const Vec3 forward{0.0F, 0.0F, 1.0F};
    const Vec3 highSun{0.0F, 1.0F, 0.0F};
    const AABB before{{-0.5F, 64.0F, 9.5F}, {0.5F, 66.0F, 10.5F}};
    const AABB after{{-0.5F, 64.0F, 10.5F}, {0.5F, 66.0F, 11.5F}};

    const ShadowCascadeConfiguration near = shadowCascadeConfiguration(2U, 0U);
    REQUIRE(shadowEntityCasterReachesDepthSlice(before, camera, forward, highSun, near.nearDepth,
                                                near.farDepth));
    REQUIRE(shadowEntityCasterReachesDepthSlice(after, camera, forward, highSun, near.nearDepth,
                                                near.farDepth));

    for (uint32_t cascade = 2U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration coarse = shadowCascadeConfiguration(2U, cascade);
        REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(before, camera, forward, highSun,
                                                          coarse.nearDepth, coarse.farDepth));
        REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(after, camera, forward, highSun,
                                                          coarse.nearDepth, coarse.farDepth));
    }
}

TEST_CASE("Shadow cascades: low-sun entity extrusion retains reachable coarse shadows",
          "[render][shadow][entity]") {
    const Vec3 camera{0.0F, 64.0F, 0.0F};
    const Vec3 forward{0.0F, 0.0F, 1.0F};
    const AABB entity{{-0.5F, 64.0F, 9.5F}, {0.5F, 66.0F, 10.5F}};

    // Light comes from behind the camera, so its shadow travels forward. At a
    // low elevation the ray can reach the horizon slice before the world
    // floor bounds it; at a steep elevation it cannot.
    const Vec3 lowSun = Vec3{0.0F, 0.10F, -0.995F}.normalize();
    const Vec3 highSun = Vec3{0.0F, 0.80F, -0.60F}.normalize();
    const ShadowCascadeConfiguration horizon =
        shadowCascadeConfiguration(2U, SHADOW_HORIZON_CASCADE_INDEX);
    const float horizonNear = shadowCascadeBlendStart(shadowCascadeConfiguration(2U, 3U));
    REQUIRE(shadowEntityCasterReachesDepthSlice(entity, camera, forward, lowSun, horizonNear,
                                                horizon.farDepth));
    REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(entity, camera, forward, highSun, horizonNear,
                                                      horizon.farDepth));

    // A caster beyond a receiver slice remains eligible when the light points
    // toward the camera and its shadow travels back into that slice.
    const AABB distant{{-0.5F, 300.0F, 1'599.5F}, {0.5F, 302.0F, 1'600.5F}};
    const Vec3 frontLight = Vec3{0.0F, 0.20F, 0.98F}.normalize();
    const ShadowCascadeConfiguration fourth = shadowCascadeConfiguration(2U, 3U);
    REQUIRE(shadowEntityCasterReachesDepthSlice(distant, camera, forward, frontLight,
                                                fourth.nearDepth, fourth.farDepth));
}

TEST_CASE("Shadow visibility follows the active celestial source strength",
          "[render][shadow][celestial]") {
    REQUIRE(shadowVisibilityWithStrength(0.0F, 0.0F) == Catch::Approx(1.0F));
    REQUIRE(shadowVisibilityWithStrength(0.0F, 0.14F) == Catch::Approx(0.86F));
    REQUIRE(shadowVisibilityWithStrength(0.0F, 1.0F) == Catch::Approx(0.0F));
    REQUIRE(shadowVisibilityWithStrength(0.35F, 0.5F) == Catch::Approx(0.675F));
    REQUIRE(shadowVisibilityWithStrength(-1.0F, 2.0F) == Catch::Approx(0.0F));
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
    REQUIRE(sizeof(SkyUniforms) == 176);
    REQUIRE(offsetof(SkyUniforms, sunDirection) == 48);
    REQUIRE(offsetof(SkyUniforms, moonDirection) == 64);
    REQUIRE(offsetof(SkyUniforms, moonColor) == 96);
    REQUIRE(offsetof(SkyUniforms, zenithColor) == 112);
    REQUIRE(offsetof(SkyUniforms, visibilityAndPhase) == 144);
    REQUIRE(offsetof(SkyUniforms, tanHalfFov) == 160);
}

TEST_CASE("Shader types: WaterUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(WaterUniforms) == 288);
    REQUIRE(offsetof(WaterUniforms, cameraRelativeViewProjection) == 64);
    REQUIRE(offsetof(WaterUniforms, zenithColor) == 128);
    REQUIRE(offsetof(WaterUniforms, resolution) == 224);
    REQUIRE(offsetof(WaterUniforms, fogDensity) == 232);
    REQUIRE(offsetof(WaterUniforms, time) == 236);
    REQUIRE(offsetof(WaterUniforms, cameraUnderwater) == 240);
    REQUIRE(offsetof(WaterUniforms, ssrStrength) == 244);
    REQUIRE(offsetof(WaterUniforms, skyExposure) == 248);
    REQUIRE(offsetof(WaterUniforms, waterSurfaceY) == 252);
    REQUIRE(offsetof(WaterUniforms, solarDirection) == 256);
    REQUIRE(offsetof(WaterUniforms, physicalSkyBlend) == 272);
    REQUIRE(offsetof(WaterUniforms, directSpecularFactor) == 276);
}

TEST_CASE("Celestial state forms physical full quarter and new Moon phases",
          "[render][celestial]") {
    const uint64_t fullTick = CELESTIAL_FULL_MOON_REFERENCE_TICK;
    const CelestialState full = computeCelestialState(fullTick);
    const CelestialState quarter =
        computeCelestialState(fullTick + CELESTIAL_SYNODIC_PERIOD_TICKS / 4U);
    const CelestialState fresh =
        computeCelestialState(fullTick + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U);

    REQUIRE(full.sunDirection.dot(full.moonDirection) == Catch::Approx(-1.0F).margin(1.0e-5F));
    REQUIRE(full.illuminatedFraction == Catch::Approx(1.0F).margin(1.0e-5F));
    REQUIRE(full.phaseEnergy == Catch::Approx(1.0F).margin(1.0e-5F));
    REQUIRE(quarter.illuminatedFraction == Catch::Approx(0.5F).margin(2.0e-5F));
    REQUIRE(quarter.phaseEnergy == Catch::Approx(1.0F / static_cast<float>(M_PI)).margin(2.0e-5F));
    REQUIRE(fresh.sunDirection.dot(fresh.moonDirection) == Catch::Approx(1.0F).margin(2.0e-5F));
    REQUIRE(fresh.illuminatedFraction <= 2.0e-5F);
    REQUIRE(fresh.phaseEnergy <= 2.0e-5F);
    REQUIRE(static_cast<double>(CELESTIAL_SYNODIC_PERIOD_TICKS) /
                static_cast<double>(CELESTIAL_TICKS_PER_DAY) ==
            Catch::Approx(29.530583).margin(1.0e-6));
}

TEST_CASE("Celestial state suppresses competing Moon light through twilight",
          "[render][celestial]") {
    const uint64_t fullMidnight = CELESTIAL_FULL_MOON_REFERENCE_TICK;
    const uint64_t fullDayStart = fullMidnight - 18'000U;
    const CelestialState justAfterSunset = computeCelestialState(fullDayStart + 12'020U);
    const CelestialState civilTwilight = computeCelestialState(fullDayStart + 12'400U);
    const CelestialState midnight = computeCelestialState(fullMidnight);

    REQUIRE(justAfterSunset.sunVisibility <= 1.0e-5F);
    REQUIRE(justAfterSunset.moonDirectVisibility <= 1.0e-5F);
    REQUIRE(justAfterSunset.directSource == CelestialLightSource::NONE);
    REQUIRE(civilTwilight.directLightRadiance.length() < 0.04F);
    REQUIRE(midnight.directSource == CelestialLightSource::MOON);
    // Full-moon direct light sits at the playable-night level: far below any
    // daylight value, bright enough that moonlit terrain reads in motion.
    REQUIRE(midnight.directLightRadiance.length() < 0.05F);
    // The disc peaks just past the bloom threshold for a slight glow while
    // staying an order of magnitude below the sun disc's 18x on-screen
    // radiance, so the Moon reads at a glance without becoming a second sun.
    REQUIRE(midnight.lunarDiscRadiance.length() < 3.60F);
    REQUIRE(midnight.shadowStrength < 0.08F);
    REQUIRE(midnight.directSpecularFactor == Catch::Approx(midnight.phaseEnergy));
}

TEST_CASE("Visible horizon Sun does not apply daytime irradiance to terrain",
          "[render][celestial][twilight]") {
    const CelestialState sunset = computeCelestialState(12'000U);
    REQUIRE(sunset.sunVisibility == Catch::Approx(0.5F).margin(5.0e-5F));
    REQUIRE(sunset.sunDirectVisibility <= 1.0e-6F);
    REQUIRE(sunset.directSource == CelestialLightSource::NONE);
    REQUIRE(sunset.directLightRadiance.length() <= 1.0e-6F);
    REQUIRE(sunset.shadowStrength <= 1.0e-6F);

    const auto closestMorningTick = [](float elevationDegrees) {
        const float target = std::sin(elevationDegrees * static_cast<float>(M_PI) / 180.0F);
        uint64_t closest = 0U;
        float error = std::numeric_limits<float>::max();
        for (uint64_t tick = 0U; tick <= 6'000U; ++tick) {
            const float candidate = computeCelestialState(tick).sunDirection.y;
            const float candidateError = std::abs(candidate - target);
            if (candidateError < error) {
                error = candidateError;
                closest = tick;
            }
        }
        return closest;
    };
    const CelestialState oneDegree = computeCelestialState(closestMorningTick(1.0F));
    const CelestialState fiveDegrees = computeCelestialState(closestMorningTick(5.0F));
    const CelestialState tenDegrees = computeCelestialState(closestMorningTick(10.0F));
    REQUIRE(oneDegree.sunVisibility > 0.99F);
    REQUIRE(oneDegree.sunDirectVisibility < 0.04F);
    REQUIRE(fiveDegrees.sunDirectVisibility > oneDegree.sunDirectVisibility);
    REQUIRE(fiveDegrees.sunDirectVisibility < 0.60F);
    REQUIRE(tenDegrees.sunDirectVisibility > 0.99F);
}

TEST_CASE("Night ambient is phase-aware and cannot resemble daylight",
          "[render][celestial][night]") {
    const CelestialState fullMoon = computeCelestialState(CELESTIAL_FULL_MOON_REFERENCE_TICK);
    const uint64_t exactNewMoon =
        CELESTIAL_FULL_MOON_REFERENCE_TICK + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U;
    const uint64_t newMoonMidnight = exactNewMoon + (18'000U + CELESTIAL_TICKS_PER_DAY -
                                                     exactNewMoon % CELESTIAL_TICKS_PER_DAY) %
                                                        CELESTIAL_TICKS_PER_DAY;
    const CelestialState newMoon = computeCelestialState(newMoonMidnight);
    const CelestialState noon = computeCelestialState(6'000U);

    REQUIRE(newMoon.phaseEnergy < 0.002F);
    REQUIRE(fullMoon.ambientRadiance.length() > newMoon.ambientRadiance.length());
    // Playable-night contract: a full moon sits about a tenth of daylight so
    // moonlit terrain stays legible, while a new moon keeps only the stellar
    // floor and daylight remains an order of magnitude above any night.
    REQUIRE(fullMoon.ambientRadiance.length() < 0.055F);
    REQUIRE(newMoon.ambientRadiance.length() < 0.025F);
    REQUIRE(noon.ambientRadiance.length() > fullMoon.ambientRadiance.length() * 10.0F);
}

TEST_CASE("Moon fades in only after civil twilight at sunset and sunrise",
          "[render][celestial][twilight]") {
    const auto closestTick = [](float elevationDegrees, bool beforeSunrise) {
        const float target = std::sin(elevationDegrees * static_cast<float>(M_PI) / 180.0F);
        const uint64_t begin = beforeSunrise ? 18'000U : 12'000U;
        const uint64_t end = beforeSunrise ? 24'000U : 18'000U;
        uint64_t closest = begin;
        float error = std::numeric_limits<float>::max();
        for (uint64_t tick = begin; tick <= end; ++tick) {
            const float candidate = computeCelestialState(tick).sunDirection.y;
            const float candidateError = std::abs(candidate - target);
            if (candidateError < error) {
                error = candidateError;
                closest = tick;
            }
        }
        return closest;
    };

    for (const bool beforeSunrise : {false, true}) {
        const CelestialState minusFive = computeCelestialState(closestTick(-5.0F, beforeSunrise));
        const CelestialState minusSix = computeCelestialState(closestTick(-6.0F, beforeSunrise));
        const CelestialState minusSeven = computeCelestialState(closestTick(-7.0F, beforeSunrise));
        const CelestialState minusTwelve =
            computeCelestialState(closestTick(-12.0F, beforeSunrise));

        REQUIRE(minusFive.moonDirectVisibility <= 1.0e-6F);
        REQUIRE(minusFive.directSource == CelestialLightSource::NONE);
        REQUIRE(minusSix.moonDirectVisibility < 0.001F);
        REQUIRE(minusSeven.moonDirectVisibility > minusSix.moonDirectVisibility);
        REQUIRE(minusSeven.moonDirectVisibility < 0.20F);
        REQUIRE(minusTwelve.moonDirectVisibility > 0.98F);
        REQUIRE(minusSeven.directLightRadiance.length() < 0.004F);
    }
}

TEST_CASE("Celestial source is exclusive continuous and phase-scaled", "[render][celestial]") {
    const uint64_t cycle = CELESTIAL_SYNODIC_PERIOD_TICKS;
    for (uint64_t tick = 0; tick < cycle; tick += 137U) {
        const CelestialState state = computeCelestialState(tick);
        REQUIRE(state.sunDirection.length() == Catch::Approx(1.0F).margin(1.0e-5F));
        REQUIRE(state.moonDirection.length() == Catch::Approx(1.0F).margin(1.0e-5F));
        REQUIRE(std::isfinite(state.directLightRadiance.length()));
        if (state.directSource == CelestialLightSource::SUN) {
            REQUIRE(state.sunDirectVisibility > 0.0F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(1.0F));
        } else if (state.directSource == CelestialLightSource::MOON) {
            REQUIRE(state.sunVisibility <= 0.0001F);
            REQUIRE(state.moonDirectVisibility > 0.0F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(state.phaseEnergy));
        } else {
            REQUIRE(state.directLightRadiance.length() <= 1.0e-6F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(0.0F));
        }
    }

    const CelestialState before = computeCelestialState(cycle - 1U);
    const CelestialState after = computeCelestialState(cycle);
    REQUIRE((before.sunDirection - after.sunDirection).length() < 0.001F);
    REQUIRE((before.moonDirection - after.moonDirection).length() < 0.001F);
    REQUIRE(std::abs(before.phaseEnergy - after.phaseEnergy) < 0.001F);
    const float angularDiameterDegrees =
        2.0F * LUNAR_ANGULAR_RADIUS_RADIANS * 180.0F / static_cast<float>(M_PI);
    REQUIRE(angularDiameterDegrees == Catch::Approx(0.518F).margin(0.01F));
}

TEST_CASE("New Moon cannot drive lighting shadows or a water glint", "[render][celestial]") {
    const uint64_t newMoonMidnight =
        CELESTIAL_FULL_MOON_REFERENCE_TICK + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U;
    const CelestialState state = computeCelestialState(newMoonMidnight);
    REQUIRE(state.phaseEnergy <= 1.0e-5F);
    REQUIRE(state.moonDirectVisibility <= 1.0e-5F);
    REQUIRE(state.directLightRadiance.length() <= 1.0e-6F);
    REQUIRE(state.shadowStrength <= 1.0e-6F);
    REQUIRE(state.directSpecularFactor <= 1.0e-6F);
}

TEST_CASE("Air precipitation does not leak into the underwater medium",
          "[render][weather][water]") {
    REQUIRE(weatherParticlesVisible(false));
    REQUIRE_FALSE(weatherParticlesVisible(true));
}

TEST_CASE("Water procedural bands fade before their phase aliases",
          "[render][water][shader-types][antialiasing]") {
    REQUIRE(waterBandVisibility(0.0F) == Catch::Approx(1.0F));
    REQUIRE(waterBandVisibility(0.45F) == Catch::Approx(1.0F));
    REQUIRE(waterBandVisibility(1.125F) == Catch::Approx(0.5F));
    REQUIRE(waterBandVisibility(1.8F) <= 1.0e-6F);
    REQUIRE(waterBandVisibility(4.0F) <= 1.0e-6F);

    float previous = waterBandVisibility(0.0F);
    for (int sample = 1; sample <= 64; ++sample) {
        const float current = waterBandVisibility(static_cast<float>(sample) / 16.0F);
        REQUIRE(current <= previous);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        previous = current;
    }
}

TEST_CASE("Underwater caustics reject wall, ceiling, and silhouette receivers",
          "[render][water][caustics]") {
    // UV-space derivative winding can produce a downward raw floor normal.
    // Facing it toward a camera above the floor restores its physical +Y side
    // before the strict receiver gate runs.
    const simd_float3 floorNormal = orientUnderwaterReceiverNormalTowardCamera(
        simd_make_float3(0.0F, -1.0F, 0.0F), simd_make_float3(0.0F, -8.0F, 0.0F));
    REQUIRE(floorNormal.y > 0.99F);
    REQUIRE(underwaterCausticSurfaceConfidence(floorNormal.y, 0.0F, 8.0F) == Catch::Approx(1.0F));

    // A ceiling viewed from below must orient downward and remain ineligible.
    const simd_float3 ceilingNormal = orientUnderwaterReceiverNormalTowardCamera(
        simd_make_float3(0.0F, 1.0F, 0.0F), simd_make_float3(0.0F, 8.0F, 0.0F));
    REQUIRE(ceilingNormal.y < -0.99F);
    REQUIRE(underwaterCausticSurfaceConfidence(ceilingNormal.y, 0.0F, 8.0F) <= 1.0e-6F);

    // Walls, opposite-oriented ceilings, and oblique normals must not turn
    // into false floors. In particular, this pins the absence of abs(normalY).
    REQUIRE(underwaterCausticSurfaceConfidence(0.0F, 0.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(-1.0F, 0.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(0.55F, 0.0F, 8.0F) <= 1.0e-6F);

    // A depth discontinuity invalidates even an otherwise up-facing estimate.
    REQUIRE(underwaterCausticSurfaceConfidence(1.0F, 4.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(std::numeric_limits<float>::quiet_NaN(), 0.0F,
                                               8.0F) <= 1.0e-6F);
}

TEST_CASE("Water SSR filters and retires only unstable grazing hits",
          "[render][water][ssr][antialiasing]") {
    // Near and non-grazing reflections preserve the full-resolution source
    // and the original narrow IGN stride range.
    REQUIRE(waterSsrReflectionMipLevel(0.0F, 0.0F) == Catch::Approx(0.0F));
    REQUIRE(waterSsrStabilityConfidence(0.0F, 1'000.0F) == Catch::Approx(1.0F));
    REQUIRE(waterSsrJitterAmplitude(0.0F) == Catch::Approx(0.24F));

    // Distant glancing rays are explicitly blurred then retire into the
    // analytic sky fallback before depth discontinuities form reflection
    // bands. Nearby and non-grazing geometry remains available to SSR.
    const float nearMip = waterSsrReflectionMipLevel(0.80F, 12.0F);
    const float farMip = waterSsrReflectionMipLevel(0.98F, 128.0F);
    REQUIRE(nearMip > 0.0F);
    REQUIRE(farMip > nearMip);
    REQUIRE(farMip <= 4.0F);
    REQUIRE(waterSsrStabilityConfidence(0.98F, 128.0F) <= 1.0e-5F);
    REQUIRE(waterSsrStabilityConfidence(0.55F, 96.0F) < 0.10F);
    REQUIRE(waterSsrStabilityConfidence(0.35F, 12.0F) > 0.90F);
    REQUIRE(waterSsrJitterAmplitude(0.98F) < waterSsrJitterAmplitude(0.50F));

    float previousMip = 0.0F;
    float previousConfidence = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float grazing = static_cast<float>(sample) / 64.0F;
        const float mip = waterSsrReflectionMipLevel(grazing, 96.0F);
        const float confidence = waterSsrStabilityConfidence(grazing, 96.0F);
        REQUIRE(mip >= previousMip);
        REQUIRE(confidence <= previousConfidence);
        REQUIRE(mip >= 0.0F);
        REQUIRE(mip <= 4.0F);
        REQUIRE(confidence >= 0.0F);
        REQUIRE(confidence <= 1.0F);
        previousMip = mip;
        previousConfidence = confidence;
    }
}

TEST_CASE("Water wave detail retires at a glancing view", "[render][water][antialiasing]") {
    REQUIRE(waterGrazingWaveDetail(1.0F) == Catch::Approx(1.0F));
    REQUIRE(waterGrazingWaveDetail(0.60F) > 0.90F);
    REQUIRE(waterGrazingWaveDetail(0.20F) < 0.05F);
    REQUIRE(waterGrazingWaveDetail(0.0F) <= 1.0e-6F);
    for (int sample = 0; sample <= 64; ++sample) {
        const float current = waterGrazingWaveDetail(static_cast<float>(sample) / 64.0F);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
    }
}

TEST_CASE("Water reflection normal filtering follows reflected-ray variation",
          "[render][water][reflection][antialiasing]") {
    // Ordinary camera projection changes only a few milliradians per pixel,
    // so resolved nearby waves retain their normal detail.
    REQUIRE(waterReflectionNormalVisibility(0.0F) == Catch::Approx(1.0F));
    REQUIRE(waterReflectionNormalVisibility(0.012F) == Catch::Approx(1.0F));
    REQUIRE(waterReflectionNormalVisibility(0.006F) > 0.99F);

    // When an analytic normal sends neighboring pixels to unrelated
    // reflection samples, retire that normal before it forms horizontal bands.
    REQUIRE(waterReflectionNormalVisibility(0.065F) <= 1.0e-6F);
    REQUIRE(waterReflectionNormalVisibility(0.25F) <= 1.0e-6F);
    REQUIRE(waterReflectionNormalVisibility(std::numeric_limits<float>::quiet_NaN()) <= 1.0e-6F);

    float previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float footprint = static_cast<float>(sample) / 256.0F;
        const float current = waterReflectionNormalVisibility(footprint);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }
}

TEST_CASE("Water refraction rejects unstable grazing receivers",
          "[render][water][refraction][antialiasing]") {
    // The close interface preserves a detailed underwater view at normal
    // incidence and on a short grazing path.
    REQUIRE(waterRefractionVisibility(0.95F, 2.0F, 32.0F, 8.0F, 0.01F, 0.25F, true) ==
            Catch::Approx(1.0F));
    REQUIRE(waterRefractionVisibility(0.24F, 2.0F, 2.0F, 8.0F, 0.01F, 0.25F, true) ==
            Catch::Approx(1.0F));

    // The far tail is a single screen-space source sample per water fragment.
    // It must fade before a long grazing path turns terrain or LOD edges into
    // a moving grid of refracted slabs.
    REQUIRE(waterRefractionVisibility(0.15F, 64.0F, 16.0F, 24.0F, 0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 16.0F, 24.0F, 0.01F, 12.0F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 64.0F, 0.01F, 0.25F, true) <= 1.0e-6F);
    // A distant top-down interface has the same one-sample opaque receiver
    // problem as a grazing interface. It must use reflection rather than
    // expose a coarse terrain tile through a small Fresnel transmission tail.
    REQUIRE(waterRefractionVisibility(0.95F, 96.0F, 32.0F, 512.0F, 0.25F, 32.0F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F, 0.2F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F, 0.01F, 0.25F, false) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(std::numeric_limits<float>::quiet_NaN(), 2.0F, 2.0F, 24.0F,
                                      0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 24.0F,
                                      0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F,
                                      std::numeric_limits<float>::quiet_NaN(), 0.25F,
                                      true) <= 1.0e-6F);

    // A shallow lake bed is long along a grazing refracted ray. Its vertical
    // depth, rather than its slant distance, must retire an under-sampled
    // coarse terrain receiver before individual cells become visible panes.
    REQUIRE(waterRefractionVisibility(0.15F, 20.0F, 2.0F, 8.0F, 0.2F, 0.25F, true) <= 1.0e-6F);

    // A receiver may remain smooth within one large terrain cell. Past the
    // nearby grazing region, retire its transmission fully rather than leave
    // a partially visible rectangular floor sample.
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 24.0F, 32.0F, 0.01F, 0.25F, true) <= 1.0e-6F);

    // The remaining Fresnel tail must not retain the raw depth of an unstable
    // receiver, otherwise a dark voxel or LOD rectangle leaks through the
    // reflection-only fallback.
    REQUIRE(waterStabilizedOpticalDepth(24.0F, 1.0F) == Catch::Approx(24.0F));
    REQUIRE(waterStabilizedOpticalDepth(64.0F, 0.0F) == Catch::Approx(4.0F));
    REQUIRE(waterStabilizedOpticalDepth(128.0F, 1.0F) == Catch::Approx(64.0F));
    REQUIRE(waterStabilizedOpticalDepth(std::numeric_limits<float>::quiet_NaN(), 0.5F) ==
            Catch::Approx(4.0F));

    float previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float footprint = static_cast<float>(sample) * 0.25F;
        const float current =
            waterRefractionVisibility(0.16F, 20.0F, 16.0F, 24.0F, 0.01F, footprint, true);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }

    // A flat far-terrain cell has a small receiver derivative in its interior.
    // The interface-distance guard must still retire its shallow grazing
    // transmission before the cell's different opaque color reads as a pane.
    previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float distance = static_cast<float>(sample) * 8.0F;
        const float current =
            waterRefractionVisibility(0.16F, 2.0F, 2.0F, distance, 0.01F, 0.25F, true);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }
}

TEST_CASE("Water exterior reflection gate ignores skylight nibble seams",
          "[render][water][lighting][seam]") {
    // Propagated skylight carries ambient accessibility. Open-water reflection
    // is either exterior or sealed, so harmless level differences between
    // exact and far geometry must not become a fractional reflection grid.
    REQUIRE(waterExteriorSkyVisibility(0.0F) <= 1.0e-6F);
    REQUIRE(waterExteriorSkyVisibility(1.0F / 15.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(14.0F / 15.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(1.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(std::numeric_limits<float>::quiet_NaN()) <= 1.0e-6F);

    float previous = 0.0F;
    for (int level = 0; level <= 15; ++level) {
        const float current = waterExteriorSkyVisibility(static_cast<float>(level) / 15.0F);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current >= previous);
        previous = current;
    }
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

TEST_CASE("Shader types: cloud layouts match MSL", "[render][shader-types]") {
    REQUIRE(sizeof(WeatherMapUniforms) == 32);
    REQUIRE(offsetof(WeatherMapUniforms, gridSize) == 16);
    REQUIRE(offsetof(WeatherMapUniforms, motionWrapBlocks) == 24);
    REQUIRE(sizeof(CloudRenderUniforms) == 352);
    REQUIRE(offsetof(CloudRenderUniforms, cameraForward) == 144);
    REQUIRE(offsetof(CloudRenderUniforms, weatherMap) == 288);
    REQUIRE(offsetof(CloudRenderUniforms, previousWeatherMap) == 320);
    REQUIRE(sizeof(CloudShadowUniforms) == 80);
    REQUIRE(offsetof(CloudShadowUniforms, weatherMap) == 48);
}

TEST_CASE("Shader types: atmospheric overhaul layouts match MSL", "[render][shader-types]") {
    REQUIRE(sizeof(AtmosphereUniforms) == 160);
    REQUIRE(offsetof(AtmosphereUniforms, cameraPositionKm) == 0);
    REQUIRE(offsetof(AtmosphereUniforms, sunDirection) == 16);
    REQUIRE(offsetof(AtmosphereUniforms, rayleighScatteringAndScaleHeight) == 64);
    REQUIRE(offsetof(AtmosphereUniforms, weatherOptics) == 128);
    REQUIRE(offsetof(AtmosphereUniforms, renderParams) == 144);

    REQUIRE(sizeof(IndirectLightingUniforms) == 336);
    REQUIRE(offsetof(IndirectLightingUniforms, projection) == 0);
    REQUIRE(offsetof(IndirectLightingUniforms, invViewProjection) == 128);
    REQUIRE(offsetof(IndirectLightingUniforms, previousViewProjection) == 192);
    REQUIRE(offsetof(IndirectLightingUniforms, resolutionAndQuality) == 256);
    REQUIRE(offsetof(IndirectLightingUniforms, traceParams) == 272);
    REQUIRE(offsetof(IndirectLightingUniforms, temporalParams) == 288);
    REQUIRE(offsetof(IndirectLightingUniforms, filterParams) == 304);
    REQUIRE(offsetof(IndirectLightingUniforms, ambientAndFrame) == 320);

    REQUIRE(sizeof(FroxelUniforms) == 368);
    REQUIRE(offsetof(FroxelUniforms, invViewProjection) == 0);
    REQUIRE(offsetof(FroxelUniforms, cameraPosition) == 192);
    REQUIRE(offsetof(FroxelUniforms, volumeDimensions) == 256);
    REQUIRE(offsetof(FroxelUniforms, depthParams) == 272);
    REQUIRE(offsetof(FroxelUniforms, renderParams) == 320);
    REQUIRE(offsetof(FroxelUniforms, weatherMap) == 336);

    REQUIRE(sizeof(LightningUniforms) == 128);
    REQUIRE(offsetof(LightningUniforms, viewProjection) == 0);
    REQUIRE(offsetof(LightningUniforms, cameraPosition) == 64);
    REQUIRE(offsetof(LightningUniforms, strikePosition) == 80);
    REQUIRE(offsetof(LightningUniforms, colorAndIntensity) == 96);
    REQUIRE(offsetof(LightningUniforms, eventAndShape) == 112);
}

TEST_CASE("Froxel media only composites onto finite receivers",
          "[render][volumetric][shader-contract]") {
    // The sky shader already integrates atmosphere to infinity. A clear sky
    // depth must preserve it rather than treating it as a far opaque surface.
    REQUIRE_FALSE(froxelHasFiniteReceiver(1.0F, 0.0F));
    REQUIRE_FALSE(froxelHasFiniteReceiver(0.999999F, 0.0F));

    REQUIRE(froxelHasFiniteReceiver(0.999F, 0.0F));
    REQUIRE(froxelHasFiniteReceiver(1.0F, 128.0F));
    REQUIRE_FALSE(froxelHasFiniteReceiver(1.0F, 65504.0F));
}

TEST_CASE("Froxel history and upscale use stable linear depth", "[render][volumetric][history]") {
    // Device depth has too little useful precision along a grazing cave floor.
    // The linear-depth threshold grows only enough to retain the same receiver.
    REQUIRE(froxelTemporalLinearDepthTolerance(2.0F) == Catch::Approx(0.05F));
    REQUIRE(froxelTemporalLinearDepthTolerance(12.0F) == Catch::Approx(0.096F));
    REQUIRE(froxelTemporalLinearDepthTolerance(96.0F) == Catch::Approx(0.768F));
    REQUIRE(froxelTemporalLinearDepthTolerance(0.0F) == 0.0F);
    REQUIRE(froxelTemporalLinearDepthTolerance(std::numeric_limits<float>::quiet_NaN()) == 0.0F);

    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.0F) == Catch::Approx(1.0F));
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.1F) > 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.1F) < 1.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 13.0F) == 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(96.0F, 97.0F) > 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(std::numeric_limits<float>::quiet_NaN(), 12.0F) ==
            0.0F);

    // The injection sequence is deterministic for replayable captures, while
    // each dimension advances between frames to break the fixed cell grid.
    for (unsigned int dimension = 0; dimension < 3U; ++dimension) {
        const float first = froxelLowDiscrepancySample(0U, dimension);
        const float second = froxelLowDiscrepancySample(1U, dimension);
        REQUIRE(first >= 0.0F);
        REQUIRE(first < 1.0F);
        REQUIRE(second >= 0.0F);
        REQUIRE(second < 1.0F);
        REQUIRE(first != second);
        REQUIRE(froxelLowDiscrepancySample(17U, dimension) ==
                Catch::Approx(froxelLowDiscrepancySample(17U, dimension)));
    }

    // The engine's perspective matrix writes negative view Z to clip W, so
    // reprojection can compare one linear-depth authority without storing a
    // second device-depth channel beside every froxel history sample.
    const Mat4 view =
        Mat4::lookAt(Vec3{11.0F, 72.0F, -9.0F}, Vec3{20.0F, 68.0F, 14.0F}, Vec3{0.0F, 1.0F, 0.0F});
    const Mat4 projection =
        Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F, 16.0F / 9.0F, 0.1F, 1000.0F);
    const Vec4 worldPoint{18.0F, 66.0F, 20.0F, 1.0F};
    const Vec4 viewPoint = view.transformVec4(worldPoint);
    const Vec4 clipPoint = (projection * view).transformVec4(worldPoint);
    REQUIRE(std::abs(clipPoint.w) == Catch::Approx(std::abs(viewPoint.z)).margin(1.0e-5F));
}

TEST_CASE("Cloud bilateral upscale preserves transparent silhouette coverage",
          "[render][cloud][shader-contract]") {
    const simd_float2 clear = cloudCompositeTapWeights(0.25F, 0.0F, 100.0F);
    REQUIRE(clear.x == 0.0F);
    REQUIRE(clear.y == Catch::Approx(0.25F));

    const simd_float2 occluded = cloudCompositeTapWeights(0.25F, 120.0F, 100.0F);
    REQUIRE(occluded.x == 0.0F);
    REQUIRE(occluded.y == Catch::Approx(0.25F));

    const simd_float2 visible = cloudCompositeTapWeights(0.25F, 80.0F, 100.0F);
    REQUIRE(visible.x > 0.0F);
    REQUIRE(visible.x < 0.25F);
    REQUIRE(visible.y == Catch::Approx(visible.x));

    // One cloudy quarter-resolution tap beside three terrain-occluded taps
    // must retain only its bilinear coverage instead of being normalized back
    // to a fully opaque cloud pixel.
    float colorWeight = visible.x;
    float normalizationWeight = visible.y;
    for (int tap = 0; tap < 3; ++tap) {
        const simd_float2 hidden = cloudCompositeTapWeights(0.25F, 120.0F, 100.0F);
        colorWeight += hidden.x;
        normalizationWeight += hidden.y;
    }
    REQUIRE(colorWeight / normalizationWeight < 0.25F);
}

TEST_CASE("Shader types: GPUParticle layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(GPUParticle) == 48);
    REQUIRE(offsetof(GPUParticle, velocity) == 16);
    REQUIRE(offsetof(GPUParticle, lifetime) == 32);
    REQUIRE(offsetof(GPUParticle, type) == 36);
}

TEST_CASE("Shader types: ParticleUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ParticleUniforms) == 160);
    REQUIRE(offsetof(ParticleUniforms, cameraPosition) == 128);
    REQUIRE(offsetof(ParticleUniforms, atmosphericExtinction) == 144);
}

TEST_CASE("Shader types: BloomUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(BloomUniforms) == 32);
    REQUIRE(offsetof(BloomUniforms, texelSize) == 8);
    REQUIRE(offsetof(BloomUniforms, threshold) == 16);
    REQUIRE(offsetof(BloomUniforms, blurRadius) == 24);
}

TEST_CASE("Shader types: PostUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(PostUniforms) == 48);
    REQUIRE(offsetof(PostUniforms, resolution) == 0);
    REQUIRE(offsetof(PostUniforms, exposure) == 8);
    REQUIRE(offsetof(PostUniforms, bloomIntensity) == 12);
    REQUIRE(offsetof(PostUniforms, vibrance) == 16);
    REQUIRE(offsetof(PostUniforms, sharpening) == 20);
    REQUIRE(offsetof(PostUniforms, frameIndex) == 24);
    REQUIRE(offsetof(PostUniforms, flareStrength) == 28);
    REQUIRE(offsetof(PostUniforms, sunScreenUV) == 32);
    REQUIRE(offsetof(PostUniforms, flareCloudOpacityTexture) == 40);
    REQUIRE(sizeof(FlareState) == 4);
}

TEST_CASE("Screen-space lighting keeps a projection-invariant view-space trace radius",
          "[render][indirect][projection]") {
    auto projectionFor = [](float fovDegrees) {
        const Mat4 matrix = Mat4::perspective(fovDegrees * static_cast<float>(M_PI) / 180.0F,
                                              16.0F / 9.0F, 0.1F, 1000.0F);
        simd_float4x4 projection;
        std::memcpy(&projection, matrix.data.data(), sizeof(projection));
        return projection;
    };
    auto reconstructFromLinearDepth = [](simd_float2 uv, float linearDepth,
                                         simd_float4x4 projection) {
        const simd_float4x4 inverse = simd_inverse(projection);
        const simd_float4 farClip =
            simd_make_float4(uv.x * 2.0F - 1.0F, 1.0F - uv.y * 2.0F, 1.0F, 1.0F);
        const simd_float4 farView = simd_mul(inverse, farClip);
        const simd_float3 ray = farView.xyz / farView.w;
        return ray * (linearDepth / std::abs(ray.z));
    };

    constexpr float RADIUS = 8.0F;
    const simd_float3 direction = simd_normalize(simd_make_float3(1.0F, 0.25F, 0.0F));
    const simd_float2 resolution = simd_make_float2(3456.0F, 2234.0F);
    std::array<float, 2> nearPixels{};
    std::array<float, 2> farPixels{};
    const std::array<float, 2> fieldsOfView = {50.0F, 90.0F};

    for (size_t field = 0; field < fieldsOfView.size(); ++field) {
        const simd_float4x4 projection = projectionFor(fieldsOfView[field]);
        for (float depth : {32.0F, 96.0F}) {
            const simd_float3 origin = simd_make_float3(0.0F, 0.0F, -depth);
            const simd_float3 endpoint = screenSpaceTraceViewSample(origin, direction, RADIUS);
            const simd_float2 endpointUv = screenSpaceProjectViewPosition(endpoint, projection);
            const simd_float3 reconstructed =
                reconstructFromLinearDepth(endpointUv, std::abs(endpoint.z), projection);

            REQUIRE(simd_length(endpoint - origin) == Catch::Approx(RADIUS).margin(1.0e-5F));
            REQUIRE(simd_length(reconstructed - endpoint) < 1.0e-3F);
            const simd_float2 originUv = screenSpaceProjectViewPosition(origin, projection);
            const float projectedPixels = simd_length((endpointUv - originUv) * resolution);
            if (depth < 64.0F) {
                nearPixels[field] = projectedPixels;
            } else {
                farPixels[field] = projectedPixels;
            }
        }
    }

    // Projection changes only screen footprint. The shared helper above still
    // reconstructs the same eight-block ray at either distance or FOV.
    REQUIRE(nearPixels[0] > nearPixels[1]);
    REQUIRE(farPixels[0] > farPixels[1]);
    REQUIRE(nearPixels[0] > farPixels[0]);
    REQUIRE(nearPixels[1] > farPixels[1]);
    REQUIRE(INDIRECT_MEDIUM_RAY_COUNT == 2u);
    REQUIRE(INDIRECT_HIGH_RAY_COUNT == 4u);
    REQUIRE(INDIRECT_MEDIUM_HIZ_ITERATION_CAP == 16u);
    REQUIRE(INDIRECT_HIGH_HIZ_ITERATION_CAP == 24u);
    REQUIRE(INDIRECT_MEDIUM_ATROUS_ITERATIONS == 2u);
    REQUIRE(INDIRECT_HIGH_ATROUS_ITERATIONS == 3u);
}

TEST_CASE("Screen-space lighting bilateral upsample rejects voxel depth discontinuities",
          "[render][indirect][upsample]") {
    // Equal-depth history remains a full contribution, while nearby values
    // soften smoothly across one physical receiver.
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.1F) > 0.0F);
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.1F) < 1.0F);

    // A close voxel face is outside the hard bilateral interval, so a
    // lower-resolution history sample cannot darken or brighten its neighbor.
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 13.0F) <= 1.0e-6F);
    REQUIRE(screenSpaceBilateralDepthWeight(128.0F, 140.0F) <= 1.0e-6F);
    REQUIRE(screenSpaceBilateralDepthWeight(std::numeric_limits<float>::quiet_NaN(), 12.0F) <=
            1.0e-6F);
}

TEST_CASE("Screen-space lighting fallback keeps a compatible voxel-face owner",
          "[render][indirect][upsample]") {
    const simd_float3 floorNormal = simd_make_float3(0.0F, 1.0F, 0.0F);
    const simd_float3 wallNormal = simd_make_float3(1.0F, 0.0F, 0.0F);

    // The regular footprint can contain only a perpendicular wall and reject
    // it completely. A nearby coplanar candidate is safe for the bounded
    // no-owner fallback, while a different depth remains rejected.
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, wallNormal) == 0.0F);
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, floorNormal) > 0.9F);
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 13.0F, floorNormal, floorNormal) <=
            1.0e-6F);
}

TEST_CASE("Screen-space history uses linear depth for grazing cave floors",
          "[render][indirect][history]") {
    // The bounded linear tolerance admits a continuous receiver at distance
    // while remaining much smaller than a different voxel face.
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(2.0F) == Catch::Approx(0.04F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(12.0F) == Catch::Approx(0.12F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(96.0F) == Catch::Approx(0.96F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(0.0F) == 0.0F);
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(std::numeric_limits<float>::quiet_NaN()) ==
            0.0F);
}

TEST_CASE("Hi-Z traversal steps cells and classifies exact hits", "[render][indirect][hiz]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();

    // Axis crossing with the epsilon nudge landing in the next cell.
    {
        const simd_float2 position = simd_make_float2(4.3F, 7.9F);
        const simd_float2 direction = simd_make_float2(1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(exit == Catch::Approx(0.75F).margin(0.06F));
        REQUIRE(std::floor(position.x + direction.x * exit) == 5.0F);
    }
    // Diagonal ray exits through the nearer boundary.
    {
        const simd_float2 position = simd_make_float2(0.5F, 0.5F);
        const simd_float2 direction = simd_normalize(simd_make_float2(2.0F, 1.0F));
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(std::floor(position.x + direction.x * exit) == 1.0F);
        REQUIRE(std::floor(position.y + direction.y * exit) == 0.0F);
    }
    // Negative direction crosses the low boundary.
    {
        const simd_float2 position = simd_make_float2(4.3F, 7.9F);
        const simd_float2 direction = simd_make_float2(-1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(std::floor(position.x + direction.x * exit) == 3.0F);
    }
    // Coarser mip levels step whole cells at once.
    {
        const simd_float2 position = simd_make_float2(5.0F, 6.0F);
        const simd_float2 direction = simd_make_float2(1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 4.0F);
        REQUIRE(exit == Catch::Approx(3.0F).margin(0.06F));
    }
    REQUIRE(screenSpaceHiZCellExit(simd_make_float2(1.0F, 1.0F), simd_make_float2(1.0F, 0.0F),
                                   0.0F) == Catch::Approx(0.05F));

    // Reciprocal depth interpolation matches a real perspective projection:
    // the point on the 3D segment at the helper's midpoint depth projects to
    // the screen-space midpoint of the segment's endpoints.
    {
        const Mat4 matrix = Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F,
                                              16.0F / 9.0F, 0.1F, 1000.0F);
        simd_float4x4 projection;
        std::memcpy(&projection, matrix.data.data(), sizeof(projection));
        const simd_float3 start = simd_make_float3(1.0F, 0.5F, -4.0F);
        const simd_float3 end = simd_make_float3(3.0F, -1.0F, -20.0F);
        const simd_float2 startUv = screenSpaceProjectViewPosition(start, projection);
        const simd_float2 endUv = screenSpaceProjectViewPosition(end, projection);
        const float midDepth = screenSpaceHiZRayDepth(0.5F, 4.0F, 20.0F);
        REQUIRE(midDepth == Catch::Approx(1.0F / ((0.25F + 0.05F) * 0.5F)));
        const float along = (midDepth - 4.0F) / 16.0F;
        const simd_float3 midPoint = start + (end - start) * along;
        const simd_float2 midUv = screenSpaceProjectViewPosition(midPoint, projection);
        REQUIRE(midUv.x == Catch::Approx((startUv.x + endUv.x) * 0.5F).margin(1.0e-3F));
        REQUIRE(midUv.y == Catch::Approx((startUv.y + endUv.y) * 0.5F).margin(1.0e-3F));
    }
    REQUIRE(screenSpaceHiZRayDepth(0.0F, 4.0F, 20.0F) == Catch::Approx(4.0F));
    REQUIRE(screenSpaceHiZRayDepth(1.0F, 4.0F, 20.0F) == Catch::Approx(20.0F));
    REQUIRE(screenSpaceHiZRayDepth(0.5F, NAN_VALUE, 20.0F) ==
            Catch::Approx(INDIRECT_SKY_LINEAR_DEPTH));

    // Empty-cell classification, including a ray moving toward the camera.
    REQUIRE(screenSpaceHiZAdvances(5.0F, 6.0F, 6.5F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(5.0F, 7.0F, 6.5F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(8.0F, 5.0F, 7.0F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(NAN_VALUE, 6.0F, 6.5F));

    // Exact mip-zero receiver test.
    REQUIRE(screenSpaceHiZSurfaceHit(10.05F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(9.9F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(10.3F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(65000.0F, 65504.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(NAN_VALUE, 10.0F, 0.15F));
}

TEST_CASE("Cosine hemisphere rays stay above the receiver surface", "[render][indirect][rays]") {
    const simd_float3 normal = simd_normalize(simd_make_float3(0.3F, 0.9F, -0.2F));
    double cosineSum = 0.0;
    int sampleCount = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            const simd_float2 xi = simd_make_float2((static_cast<float>(i) + 0.5F) / 16.0F,
                                                    (static_cast<float>(j) + 0.5F) / 16.0F);
            const simd_float3 direction = screenSpaceCosineHemisphereDirection(xi, normal);
            REQUIRE(simd_length(direction) == Catch::Approx(1.0F).margin(1.0e-4F));
            REQUIRE(simd_dot(direction, normal) > 0.0F);
            cosineSum += simd_dot(direction, normal);
            ++sampleCount;
        }
    }
    // The cosine-weighted density has an exact mean cosine of two thirds.
    REQUIRE(cosineSum / sampleCount == Catch::Approx(2.0 / 3.0).margin(0.02));

    const simd_float3 fallback = screenSpaceCosineHemisphereDirection(
        simd_make_float2(0.3F, 0.7F), simd_make_float3(0.0F, 0.0F, 0.0F));
    REQUIRE(fallback.z == Catch::Approx(1.0F));

    // R2 sequence samples stay in the unit square and move a meaningful
    // distance every frame so a pixel's rays never clump.
    const simd_float2 noise = simd_make_float2(0.42F, 0.17F);
    simd_float2 previous = screenSpaceRaySequenceSample(0, noise);
    for (uint32_t index = 1; index < 8; ++index) {
        const simd_float2 sample = screenSpaceRaySequenceSample(index, noise);
        REQUIRE(sample.x >= 0.0F);
        REQUIRE(sample.x < 1.0F);
        REQUIRE(sample.y >= 0.0F);
        REQUIRE(sample.y < 1.0F);
        REQUIRE(std::abs(sample.x - previous.x) + std::abs(sample.y - previous.y) > 0.05F);
        previous = sample;
    }
}

TEST_CASE("Temporal blend ramps with age and clamps fireflies", "[render][indirect][temporal]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(screenSpaceTemporalBlendWeight(0.0F, 0.90F) == 0.0F);
    REQUIRE(screenSpaceTemporalBlendWeight(1.0F, 0.90F) == Catch::Approx(0.5F));
    REQUIRE(screenSpaceTemporalBlendWeight(3.0F, 0.90F) == Catch::Approx(0.75F));
    REQUIRE(screenSpaceTemporalBlendWeight(9.0F, 0.90F) == Catch::Approx(0.90F));
    REQUIRE(screenSpaceTemporalBlendWeight(INDIRECT_HISTORY_MAX_AGE, 0.90F) ==
            Catch::Approx(0.90F));
    REQUIRE(screenSpaceTemporalBlendWeight(NAN_VALUE, 0.90F) == 0.0F);

    REQUIRE(screenSpaceLuminanceVariance(0.5F, 0.25F) == 0.0F);
    REQUIRE(screenSpaceLuminanceVariance(0.5F, 0.50F) == Catch::Approx(0.25F));
    REQUIRE(screenSpaceLuminanceVariance(NAN_VALUE, 1.0F) == 0.0F);

    REQUIRE(screenSpaceFireflyClampScale(2.0F, 4.0F) == 1.0F);
    REQUIRE(screenSpaceFireflyClampScale(16.0F, 4.0F) == Catch::Approx(0.25F));
    REQUIRE(screenSpaceFireflyClampScale(NAN_VALUE, 4.0F) == 0.0F);
}

TEST_CASE("Variance clamp collapses stale history over a converged neighborhood",
          "[render][indirect][history]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    // A converged region has near-zero deviation, so a stale bright ghost is
    // clamped to the floor within one frame.
    REQUIRE(screenSpaceVarianceClampHalfRange(0.0F, 2.0F, 0.001F) == Catch::Approx(0.001F));
    // A genuinely sparse bright source keeps a wide clamp because its
    // accumulated variance stays high.
    REQUIRE(screenSpaceVarianceClampHalfRange(0.5F, 2.0F, 0.001F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceVarianceClampHalfRange(NAN_VALUE, 2.0F, 0.001F) == Catch::Approx(0.001F));

    // Young pixels take the wider of the spatial and temporal estimates so
    // disocclusion opens the spatial filter instead of trusting two samples.
    REQUIRE(screenSpaceVarianceForAge(0.01F, 0.2F, 1.0F, 4.0F) == Catch::Approx(0.2F));
    REQUIRE(screenSpaceVarianceForAge(0.01F, 0.2F, 8.0F, 4.0F) == Catch::Approx(0.01F));
}

TEST_CASE("A-trous edge weight stops at voxel edges and follows variance",
          "[render][indirect][denoise]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.0F, 1.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceAtrousEdgeWeight(2.0F, 0.5F, 1.0F, 0.0F, 1.0F) == 0.0F);
    REQUIRE(screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 0.0F, 0.0F, 1.0F) == 0.0F);
    const float tight = screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.5F, 0.25F);
    const float loose = screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.5F, 1.0F);
    REQUIRE(loose > tight);
    REQUIRE(screenSpaceAtrousEdgeWeight(NAN_VALUE, 0.5F, 1.0F, 0.0F, 1.0F) == 0.0F);

    REQUIRE(screenSpaceOcclusionFalloff(0.0F, 8.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceOcclusionFalloff(8.0F, 8.0F) == 0.0F);
    REQUIRE(screenSpaceOcclusionFalloff(2.0F, 8.0F) > screenSpaceOcclusionFalloff(6.0F, 8.0F));

    REQUIRE(screenSpaceBounceSourceWeight(-0.5F, 4.0F, 24.0F) == 0.0F);
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 4.0F, 24.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 24.0F, 24.0F) == Catch::Approx(0.0F));
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 19.0F, 24.0F) >
            screenSpaceBounceSourceWeight(1.0F, 23.0F, 24.0F));
    REQUIRE(screenSpaceBounceSourceWeight(NAN_VALUE, 4.0F, 24.0F) == 0.0F);
}

TEST_CASE("Shader types: ExposureState + ExposureParams layout match MSL",
          "[render][shader-types]") {
    REQUIRE(sizeof(ExposureState) == 8);
    REQUIRE(offsetof(ExposureState, smoothedLogLum) == 0);
    REQUIRE(offsetof(ExposureState, exposure) == 4);

    REQUIRE(sizeof(ExposureParams) == 48);
    REQUIRE(offsetof(ExposureParams, keyValue) == 0);
    REQUIRE(offsetof(ExposureParams, adaptationDownRate) == 4);
    REQUIRE(offsetof(ExposureParams, minLogLum) == 8);
    REQUIRE(offsetof(ExposureParams, maxLogLum) == 12);
    REQUIRE(offsetof(ExposureParams, sampleGrid) == 16);
    REQUIRE(offsetof(ExposureParams, minExposure) == 24);
    REQUIRE(offsetof(ExposureParams, maxExposure) == 28);
    REQUIRE(offsetof(ExposureParams, adaptationUpRate) == 32);
    REQUIRE(offsetof(ExposureParams, highlightGain) == 36);
    REQUIRE(offsetof(ExposureParams, highlightKnee) == 40);
    REQUIRE(offsetof(ExposureParams, highlightRange) == 44);
}

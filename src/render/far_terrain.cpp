#include "render/far_terrain.hpp"

#include "common/thread_priority.hpp"
#include "render/block_textures.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace {

struct FarCell {
    std::array<float, 4> terrain{};
    std::array<float, 4> waterSurface{};
    BlockType material = BlockType::GRASS;
    bool flat = false;
    bool water = false;
    bool flatWater = false;
    bool centerWet = false;
    uint8_t waterMask = 0;
    float waterHeight = SEA_LEVEL;
    float centerWaterHeight = SEA_LEVEL;
};

struct Corner {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float u = 0.0F;
    float v = 0.0F;
};

struct WaterPoint {
    float x = 0.0F;
    float z = 0.0F;
    float height = 0.0F;
    bool wet = false;
};

using WaterEdgeRefiner =
    std::function<WaterPoint(const WaterPoint& first, const WaterPoint& second)>;

bool validStep(FarTerrainStep step) {
    return step == FarTerrainStep::TWO || step == FarTerrainStep::FOUR ||
           step == FarTerrainStep::EIGHT || step == FarTerrainStep::SIXTEEN;
}

int64_t tileOrigin(int64_t tileCoordinate) {
    const __int128 product =
        static_cast<__int128>(tileCoordinate) * static_cast<__int128>(FAR_TERRAIN_TILE_EDGE);
    if (product < std::numeric_limits<int64_t>::min() ||
        product > std::numeric_limits<int64_t>::max() - FAR_TERRAIN_TILE_EDGE) {
        throw std::out_of_range("far terrain tile origin exceeds int64 range");
    }
    return static_cast<int64_t>(product);
}

uint64_t mix64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xBF58'476D'1CE4'E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

uint16_t halfBits(const float16_t& value) {
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t hashMesh(const FarTerrainMesh& mesh) {
    uint64_t hash = mix64(static_cast<uint64_t>(mesh.key.tileX));
    hash = mix64(hash ^ static_cast<uint64_t>(mesh.key.tileZ));
    hash = mix64(hash ^ static_cast<uint8_t>(mesh.key.step));
    for (const Vertex& vertex : mesh.vertices) {
        hash = mix64(hash ^ vertex.faceAttr);
        hash = mix64(hash ^ (static_cast<uint64_t>(halfBits(vertex.px)) << 0U) ^
                     (static_cast<uint64_t>(halfBits(vertex.py)) << 16U) ^
                     (static_cast<uint64_t>(halfBits(vertex.pz)) << 32U));
        hash = mix64(hash ^ static_cast<uint64_t>(halfBits(vertex.u)) ^
                     (static_cast<uint64_t>(halfBits(vertex.v)) << 16U));
    }
    for (uint32_t index : mesh.indices)
        hash = mix64(hash ^ index);
    hash = mix64(hash ^ std::bit_cast<uint32_t>(mesh.complexity));
    return hash;
}

BlockType geologyMaterial(const worldgen::GeologySample& geology) {
    return worldgen::surface_material::outcrop(geology);
}

BlockType terrainMaterial(const worldgen::SurfaceSample& sample) {
    const Biome biome = sample.biome.primary;
    return worldgen::surface_material::surface(sample, biome, {},
                                               worldgen::surface_material::frozen(sample, biome),
                                               worldgen::surface_material::submerged(sample));
}

FarTerrainGeometrySample geometryFromSurface(const worldgen::SurfaceSample& sample) {
    FarTerrainGeometrySample result;
    result.terrainHeight = sample.terrainHeight;
    // Generation fills source blocks through ceil(spill)-1. The top source
    // uses the shared seven-eighths fluid height, while covered source blocks
    // below it remain full height. Retain the analytical spill level in world
    // samples and give far geometry the same visible top as exact water.
    result.waterSurface =
        (sample.hydrology.ocean || sample.hydrology.river || sample.hydrology.lake)
            ? std::ceil(sample.waterSurface) - 0.125
            : sample.waterSurface;
    result.discharge = sample.hydrology.discharge;
    result.sediment = sample.hydrology.sediment;
    result.waterfallTop = sample.hydrology.waterfallTop;
    result.waterfallBottom = sample.hydrology.waterfallBottom;
    result.waterfallWidth = sample.hydrology.waterfallWidth;
    result.flowX = sample.hydrology.flowDirection.x;
    result.flowZ = sample.hydrology.flowDirection.z;
    result.ocean = sample.hydrology.ocean;
    result.river = sample.hydrology.river;
    result.lake = sample.hydrology.lake;
    result.waterfall = sample.hydrology.waterfall;
    result.waterfallAnchor = sample.hydrology.waterfallAnchor;
    result.delta = sample.hydrology.delta;
    return result;
}

bool hasWater(const FarTerrainGeometrySample& sample) {
    return sample.ocean || sample.river || sample.lake;
}

BlockType macroGeometryMaterial(const worldgen::GeologySample& geology,
                                const FarTerrainGeometrySample& geometry) {
    if (geometry.delta) return BlockType::SILT;
    if (geology.volcanicActivity > 0.52) return BlockType::VOLCANIC_ASH;
    if (geometry.ocean) {
        return geometry.terrainHeight < SEA_LEVEL - 18.0 ? BlockType::GRAVEL : BlockType::SAND;
    }
    if (geometry.river || geometry.lake) {
        return geometry.sediment > geometry.discharge * 0.02 ? BlockType::MUD : BlockType::CLAY;
    }
    if (geometry.terrainHeight > 150.0 || geology.uplift > 0.55 || geology.faultStrength > 0.60) {
        return geologyMaterial(geology);
    }
    return BlockType::GRASS;
}

float vertexHeight(double height) {
    return static_cast<float>(static_cast<float16_t>(std::clamp(
        height, static_cast<double>(WORLD_MIN_Y), static_cast<double>(WORLD_MAX_Y + 1))));
}

void updateYBounds(FarTerrainMesh& mesh, float y) {
    mesh.bounds.minY = std::min(mesh.bounds.minY, y);
    mesh.bounds.maxY = std::max(mesh.bounds.maxY, y);
}

void updateSurfaceYBounds(FarTerrainMesh& mesh, float y) {
    mesh.surfaceBounds.minY = std::min(mesh.surfaceBounds.minY, y);
    mesh.surfaceBounds.maxY = std::max(mesh.surfaceBounds.maxY, y);
}

void pushQuad(FarTerrainMesh& mesh, uint32_t attribute, const std::array<Corner, 4>& corners,
              bool water) {
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (const Corner& corner : corners) {
        const Vertex vertex{attribute,
                            static_cast<float16_t>(corner.x),
                            static_cast<float16_t>(corner.y),
                            static_cast<float16_t>(corner.z),
                            static_cast<float16_t>(corner.u),
                            static_cast<float16_t>(corner.v)};
        mesh.vertices.push_back(vertex);
        updateYBounds(mesh, static_cast<float>(vertex.py));
    }
    mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    if (water) {
        ++mesh.waterQuadCount;
    } else {
        ++mesh.terrainQuadCount;
    }
}

void pushTerrainTop(FarTerrainMesh& mesh, BlockType material, float x0, float z0, float x1,
                    float z1, float northwest, float northeast, float southeast, float southwest) {
    const uint32_t attribute =
        packFaceAttr(FaceNormal::PLUS_Y, textureLayerFor(material, FaceNormal::PLUS_Y), 15);
    const float width = x1 - x0;
    const float depth = z1 - z0;
    pushQuad(mesh, attribute,
             {{{x0, northwest, z0, 0.0F, 0.0F},
               {x0, southwest, z1, 0.0F, depth},
               {x1, southeast, z1, width, depth},
               {x1, northeast, z0, width, 0.0F}}},
             false);
    updateSurfaceYBounds(mesh, northwest);
    updateSurfaceYBounds(mesh, northeast);
    updateSurfaceYBounds(mesh, southeast);
    updateSurfaceYBounds(mesh, southwest);
}

void pushWaterTop(FarTerrainMesh& mesh, float x0, float z0, float x1, float z1,
                  const std::array<float, 4>& heights) {
    const uint32_t attribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, false);
    const float width = x1 - x0;
    const float depth = z1 - z0;
    pushQuad(mesh, attribute,
             {{{x0, heights[0], z0, 0.0F, 0.0F},
               {x0, heights[3], z1, 0.0F, depth},
               {x1, heights[2], z1, width, depth},
               {x1, heights[1], z0, width, 0.0F}}},
             true);
    for (float height : heights)
        updateSurfaceYBounds(mesh, height);
}

void pushWaterTop(FarTerrainMesh& mesh, float x0, float z0, float x1, float z1, float height) {
    pushWaterTop(mesh, x0, z0, x1, z1, {height, height, height, height});
}

FaceNormal horizontalFace(float normalX, float normalZ) {
    if (std::abs(normalX) >= std::abs(normalZ)) {
        return normalX >= 0.0F ? FaceNormal::PLUS_X : FaceNormal::MINUS_X;
    }
    return normalZ >= 0.0F ? FaceNormal::PLUS_Z : FaceNormal::MINUS_Z;
}

void pushWaterfallPrism(FarTerrainMesh& mesh, float centerX, float centerZ,
                        const FarTerrainGeometrySample& sample) {
    const float bottom = vertexHeight(std::ceil(sample.waterfallBottom) - 1.0);
    const float top = vertexHeight(std::ceil(sample.waterfallTop));
    if (top <= bottom + 0.5F) return;

    float flowX = static_cast<float>(sample.flowX);
    float flowZ = static_cast<float>(sample.flowZ);
    const float flowLength = std::hypot(flowX, flowZ);
    if (flowLength <= 1.0e-5F) {
        flowX = 1.0F;
        flowZ = 0.0F;
    } else {
        flowX /= flowLength;
        flowZ /= flowLength;
    }
    const float crossX = -flowZ;
    const float crossZ = flowX;
    const float halfWidth =
        std::clamp(static_cast<float>(sample.waterfallWidth) * 0.5F, 1.5F, 8.0F);
    const float halfDepth = std::clamp(halfWidth * 0.18F, 0.75F, 1.5F);
    const std::array<std::array<float, 2>, 4> footprint = {{
        {{centerX - crossX * halfWidth - flowX * halfDepth,
          centerZ - crossZ * halfWidth - flowZ * halfDepth}},
        {{centerX + crossX * halfWidth - flowX * halfDepth,
          centerZ + crossZ * halfWidth - flowZ * halfDepth}},
        {{centerX + crossX * halfWidth + flowX * halfDepth,
          centerZ + crossZ * halfWidth + flowZ * halfDepth}},
        {{centerX - crossX * halfWidth + flowX * halfDepth,
          centerZ - crossZ * halfWidth + flowZ * halfDepth}},
    }};
    const float height = top - bottom;
    auto pushSide = [&](size_t firstIndex, size_t secondIndex, float normalX, float normalZ) {
        const auto& first = footprint[firstIndex];
        const auto& second = footprint[secondIndex];
        const float width = std::hypot(second[0] - first[0], second[1] - first[1]);
        const uint32_t attribute = packFluidFaceAttr(horizontalFace(normalX, normalZ), 15, 0, true);
        pushQuad(mesh, attribute,
                 {{{first[0], bottom, first[1], 0.0F, height},
                   {second[0], bottom, second[1], width, height},
                   {second[0], top, second[1], width, 0.0F},
                   {first[0], top, first[1], 0.0F, 0.0F}}},
                 true);
        ++mesh.waterfallQuadCount;
    };
    pushSide(0, 1, -flowX, -flowZ);
    pushSide(1, 2, crossX, crossZ);
    pushSide(2, 3, flowX, flowZ);
    pushSide(3, 0, -crossX, -crossZ);

    const uint32_t topAttribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, true);
    pushQuad(mesh, topAttribute,
             {{{footprint[0][0], top, footprint[0][1], 0.0F, 0.0F},
               {footprint[3][0], top, footprint[3][1], 0.0F, halfDepth * 2.0F},
               {footprint[2][0], top, footprint[2][1], halfWidth * 2.0F, halfDepth * 2.0F},
               {footprint[1][0], top, footprint[1][1], halfWidth * 2.0F, 0.0F}}},
             true);
    ++mesh.waterfallQuadCount;
    updateSurfaceYBounds(mesh, bottom);
    updateSurfaceYBounds(mesh, top);

    for (const auto& point : footprint) {
        const int64_t worldX = mesh.originX + static_cast<int64_t>(std::floor(point[0]));
        const int64_t worldZ = mesh.originZ + static_cast<int64_t>(std::floor(point[1]));
        const int64_t worldMaxX = mesh.originX + static_cast<int64_t>(std::ceil(point[0]));
        const int64_t worldMaxZ = mesh.originZ + static_cast<int64_t>(std::ceil(point[1]));
        mesh.bounds.minX = std::min(mesh.bounds.minX, worldX);
        mesh.bounds.maxX = std::max(mesh.bounds.maxX, worldMaxX);
        mesh.bounds.minZ = std::min(mesh.bounds.minZ, worldZ);
        mesh.bounds.maxZ = std::max(mesh.bounds.maxZ, worldMaxZ);
        mesh.surfaceBounds.minX = std::min(mesh.surfaceBounds.minX, worldX);
        mesh.surfaceBounds.maxX = std::max(mesh.surfaceBounds.maxX, worldMaxX);
        mesh.surfaceBounds.minZ = std::min(mesh.surfaceBounds.minZ, worldZ);
        mesh.surfaceBounds.maxZ = std::max(mesh.surfaceBounds.maxZ, worldMaxZ);
    }
}

// Clip one center-split cell triangle against the binary wet mask. The caller
// refines wet-to-dry edges against the coordinate-pure water predicate, which
// keeps a coarse LOD from extending a lake surface halfway across a dry cell.
void pushWaterContourTriangle(FarTerrainMesh& mesh, const std::array<WaterPoint, 3>& triangle,
                              const WaterEdgeRefiner& refineEdge) {
    std::array<WaterPoint, 4> polygon{};
    size_t polygonSize = 0;
    for (size_t index = 0; index < triangle.size(); ++index) {
        const WaterPoint& current = triangle[index];
        const WaterPoint& next = triangle[(index + 1) % triangle.size()];
        if (current.wet) polygon[polygonSize++] = current;
        if (current.wet != next.wet) {
            polygon[polygonSize++] = refineEdge(current, next);
        }
    }
    if (polygonSize < 3) return;

    const uint32_t attribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, false);
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (size_t index = 0; index < polygonSize; ++index) {
        const WaterPoint& point = polygon[index];
        const Vertex vertex{attribute,
                            static_cast<float16_t>(point.x),
                            static_cast<float16_t>(point.height),
                            static_cast<float16_t>(point.z),
                            static_cast<float16_t>(point.x),
                            static_cast<float16_t>(point.z)};
        mesh.vertices.push_back(vertex);
        updateYBounds(mesh, static_cast<float>(vertex.py));
        updateSurfaceYBounds(mesh, static_cast<float>(vertex.py));
    }
    for (size_t index = 1; index + 1 < polygonSize; ++index) {
        mesh.indices.insert(mesh.indices.end(), {base, base + static_cast<uint32_t>(index),
                                                 base + static_cast<uint32_t>(index + 1)});
        ++mesh.waterContourTriangleCount;
    }
}

void pushSkirt(FarTerrainMesh& mesh, FaceNormal face, BlockType material, float x0, float z0,
               float x1, float z1, float top0, float top1) {
    const float bottom0 = std::max(static_cast<float>(WORLD_MIN_Y), top0 - FAR_TERRAIN_SKIRT_DEPTH);
    const float bottom1 = std::max(static_cast<float>(WORLD_MIN_Y), top1 - FAR_TERRAIN_SKIRT_DEPTH);
    const float width = std::hypot(x1 - x0, z1 - z0);
    const uint32_t attribute =
        packFaceAttr(face, textureLayerFor(material, face), 15) | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK;
    const std::array<Corner, 4> corners = {{{x0, bottom0, z0, 0.0F, top0 - bottom0},
                                            {x1, bottom1, z1, width, top1 - bottom1},
                                            {x1, top1, z1, width, 0.0F},
                                            {x0, top0, z0, 0.0F, 0.0F}}};
    pushQuad(mesh, attribute, corners, false);
    --mesh.terrainQuadCount;
    ++mesh.skirtQuadCount;
}

void pushCanopyQuad(FarTerrainMesh& mesh, FaceNormal face, BlockType material,
                    const std::array<Corner, 4>& corners) {
    const uint32_t attribute =
        packFaceAttr(face, textureLayerFor(material, face), 15) | FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK;
    pushQuad(mesh, attribute, corners, false);
    --mesh.terrainQuadCount;
    ++mesh.canopyImpostorQuadCount;
    for (const Corner& corner : corners)
        updateSurfaceYBounds(mesh, corner.y);
}

void pushCanopyBox(FarTerrainMesh& mesh, BlockType material, float centerX, float centerZ,
                   float radius, float bottom, float top, bool includeTop) {
    const float x0 = centerX - radius;
    const float x1 = centerX + radius;
    const float z0 = centerZ - radius;
    const float z1 = centerZ + radius;
    const float width = radius * 2.0F;
    const float height = std::max(0.0F, top - bottom);
    if (height <= 0.0F || width <= 0.0F) return;
    pushCanopyQuad(mesh, FaceNormal::MINUS_X, material,
                   {{{x0, bottom, z0, 0.0F, height},
                     {x0, bottom, z1, width, height},
                     {x0, top, z1, width, 0.0F},
                     {x0, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::PLUS_X, material,
                   {{{x1, bottom, z1, 0.0F, height},
                     {x1, bottom, z0, width, height},
                     {x1, top, z0, width, 0.0F},
                     {x1, top, z1, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::MINUS_Z, material,
                   {{{x1, bottom, z0, 0.0F, height},
                     {x0, bottom, z0, width, height},
                     {x0, top, z0, width, 0.0F},
                     {x1, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::PLUS_Z, material,
                   {{{x0, bottom, z1, 0.0F, height},
                     {x1, bottom, z1, width, height},
                     {x1, top, z1, width, 0.0F},
                     {x0, top, z1, 0.0F, 0.0F}}});
    if (includeTop) {
        pushCanopyQuad(mesh, FaceNormal::PLUS_Y, material,
                       {{{x0, top, z0, 0.0F, 0.0F},
                         {x0, top, z1, 0.0F, width},
                         {x1, top, z1, width, width},
                         {x1, top, z0, width, 0.0F}}});
    }
}

bool retainsCanopy(FarTerrainStep step, uint64_t anchorId) {
    switch (step) {
        case FarTerrainStep::TWO:
        case FarTerrainStep::FOUR:
            return true;
        case FarTerrainStep::EIGHT:
            return (anchorId & 3U) != 0U;
        case FarTerrainStep::SIXTEEN:
            return (anchorId & 3U) == 1U || (anchorId & 3U) == 3U;
    }
    return false;
}

bool sameFlatTerrain(const FarCell& first, const FarCell& second) {
    return first.flat && second.flat && first.material == second.material &&
           first.terrain[0] == second.terrain[0];
}

bool sameWater(const FarCell& first, const FarCell& second) {
    return first.waterMask == 0x0FU && second.waterMask == 0x0FU && first.flatWater &&
           second.flatWater && first.waterHeight == second.waterHeight;
}

void validateLimits(FarTerrainSchedulerLimits& limits) {
    limits.maxPending = std::max<size_t>(1, limits.maxPending);
    limits.maxCompleted = std::max<size_t>(1, limits.maxCompleted);
    limits.maxCacheEntries = std::max<size_t>(1, limits.maxCacheEntries);
    limits.maxCacheBytes = std::max<size_t>(1, limits.maxCacheBytes);
}

struct AzimuthCoverage {
    double start = 0.0;
    double end = 0.0;
    bool cameraInside = false;
};

double normalizedAngle(double angle) {
    const double fullCircle = 2.0 * std::numbers::pi;
    angle = std::fmod(angle, fullCircle);
    return angle < 0.0 ? angle + fullCircle : angle;
}

AzimuthCoverage azimuthCoverage(const FarTerrainBounds& bounds, TerrainHorizonViewpoint viewpoint) {
    AzimuthCoverage result;
    result.cameraInside = viewpoint.x >= static_cast<double>(bounds.minX) &&
                          viewpoint.x <= static_cast<double>(bounds.maxX) &&
                          viewpoint.z >= static_cast<double>(bounds.minZ) &&
                          viewpoint.z <= static_cast<double>(bounds.maxZ);
    if (result.cameraInside) return result;
    std::array<double, 4> angles = {
        normalizedAngle(std::atan2(static_cast<double>(bounds.minZ) - viewpoint.z,
                                   static_cast<double>(bounds.minX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.minZ) - viewpoint.z,
                                   static_cast<double>(bounds.maxX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.maxZ) - viewpoint.z,
                                   static_cast<double>(bounds.maxX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.maxZ) - viewpoint.z,
                                   static_cast<double>(bounds.minX) - viewpoint.x)),
    };
    std::sort(angles.begin(), angles.end());
    const double fullCircle = 2.0 * std::numbers::pi;
    double largestGap = -1.0;
    size_t gapIndex = 0;
    for (size_t index = 0; index < angles.size(); ++index) {
        const double next = index + 1 < angles.size() ? angles[index + 1] : angles[0] + fullCircle;
        const double gap = next - angles[index];
        if (gap > largestGap) {
            largestGap = gap;
            gapIndex = index;
        }
    }
    result.start = gapIndex + 1 < angles.size() ? angles[gapIndex + 1] : angles[0];
    result.end = result.start + fullCircle - largestGap;
    return result;
}

template <typename Visitor>
size_t visitFullyCoveredBins(const AzimuthCoverage& coverage, Visitor&& visitor) {
    if (coverage.cameraInside || coverage.end <= coverage.start) return 0;
    constexpr double FULL_CIRCLE = 2.0 * std::numbers::pi;
    constexpr double BIN_WIDTH = FULL_CIRCLE / TerrainHorizonCuller::AZIMUTH_BIN_COUNT;
    constexpr double EPSILON = 1.0e-12;
    size_t count = 0;
    // Contract both ends so numerical tolerance can only omit an occluder
    // bin. Expanding here could classify a partially covered bin as fully
    // covered and create a false-positive horizon rejection.
    const int64_t first = static_cast<int64_t>(std::ceil((coverage.start + EPSILON) / BIN_WIDTH));
    const int64_t pastLast = static_cast<int64_t>(std::floor((coverage.end - EPSILON) / BIN_WIDTH));
    for (int64_t unwrapped = first; unwrapped < pastLast; ++unwrapped) {
        const size_t bin = static_cast<size_t>(world_coord::floorMod(
            unwrapped, static_cast<int32_t>(TerrainHorizonCuller::AZIMUTH_BIN_COUNT)));
        ++count;
        visitor(bin);
    }
    return count;
}

template <typename Visitor>
size_t visitIntersectedBins(const AzimuthCoverage& coverage, Visitor&& visitor) {
    if (coverage.cameraInside || coverage.end <= coverage.start) return 0;
    constexpr double FULL_CIRCLE = 2.0 * std::numbers::pi;
    constexpr double BIN_WIDTH = FULL_CIRCLE / TerrainHorizonCuller::AZIMUTH_BIN_COUNT;
    constexpr double EPSILON = 1.0e-12;
    const int64_t first = static_cast<int64_t>(std::floor(coverage.start / BIN_WIDTH));
    const int64_t pastLast =
        static_cast<int64_t>(std::floor((coverage.end - EPSILON) / BIN_WIDTH)) + 1;
    size_t count = 0;
    for (int64_t unwrapped = first; unwrapped < pastLast; ++unwrapped) {
        const size_t bin = static_cast<size_t>(world_coord::floorMod(
            unwrapped, static_cast<int32_t>(TerrainHorizonCuller::AZIMUTH_BIN_COUNT)));
        ++count;
        visitor(bin);
    }
    return count;
}

double farthestHorizontalDistance(const FarTerrainBounds& bounds,
                                  TerrainHorizonViewpoint viewpoint) {
    double farthestSquared = 0.0;
    for (int xSide = 0; xSide < 2; ++xSide) {
        for (int zSide = 0; zSide < 2; ++zSide) {
            const double x = static_cast<double>(xSide == 0 ? bounds.minX : bounds.maxX);
            const double z = static_cast<double>(zSide == 0 ? bounds.minZ : bounds.maxZ);
            farthestSquared = std::max(farthestSquared, (x - viewpoint.x) * (x - viewpoint.x) +
                                                            (z - viewpoint.z) * (z - viewpoint.z));
        }
    }
    return std::sqrt(farthestSquared);
}

} // namespace

uint32_t farTerrainSkirtEdgeMask(
    FarTerrainStep step,
    const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps) {
    uint32_t mask = 0;
    for (size_t edge = 0; edge < displayedNeighborSteps.size(); ++edge) {
        const std::optional<FarTerrainStep> neighbor = displayedNeighborSteps[edge];
        if (neighbor && farTerrainStepSize(step) < farTerrainStepSize(*neighbor)) {
            mask |= 1U << edge;
        }
    }
    return mask;
}

std::optional<FarTerrainStep> farTerrainStepForChunkDistance(double chunkDistance) {
    return farTerrainStepForMetrics(chunkDistance, 0.0F);
}

std::optional<FarTerrainStep> farTerrainStepForMetrics(double chunkDistance, float complexity,
                                                       std::optional<FarTerrainStep> previousStep) {
    if (!std::isfinite(chunkDistance) || chunkDistance < FAR_TERRAIN_NEAR_CHUNK_RADIUS ||
        chunkDistance >= FAR_TERRAIN_MAX_CHUNK_RADIUS) {
        return std::nullopt;
    }
    const double boundedComplexity = std::clamp(static_cast<double>(complexity), 0.0, 1.0);
    const double effectiveDistance = chunkDistance / (1.0 + boundedComplexity * 0.35);
    if (!previousStep) {
        if (effectiveDistance < 48.0) return FarTerrainStep::TWO;
        if (effectiveDistance < 72.0) return FarTerrainStep::FOUR;
        if (effectiveDistance < 136.0) return FarTerrainStep::EIGHT;
        return FarTerrainStep::SIXTEEN;
    }

    // A detailed tile must cross the upper edge before coarsening; a coarse
    // tile must cross the lower edge before refining. These bands are wider at
    // the distant transition because a 256-block tile crosses it more slowly.
    switch (*previousStep) {
        case FarTerrainStep::TWO:
            if (effectiveDistance < 53.0) return FarTerrainStep::TWO;
            if (effectiveDistance < 78.0) return FarTerrainStep::FOUR;
            if (effectiveDistance < 146.0) return FarTerrainStep::EIGHT;
            return FarTerrainStep::SIXTEEN;
        case FarTerrainStep::FOUR:
            if (effectiveDistance < 43.0) return FarTerrainStep::TWO;
            if (effectiveDistance < 78.0) return FarTerrainStep::FOUR;
            if (effectiveDistance < 146.0) return FarTerrainStep::EIGHT;
            return FarTerrainStep::SIXTEEN;
        case FarTerrainStep::EIGHT:
            if (effectiveDistance < 43.0) return FarTerrainStep::TWO;
            if (effectiveDistance < 66.0) return FarTerrainStep::FOUR;
            if (effectiveDistance < 146.0) return FarTerrainStep::EIGHT;
            return FarTerrainStep::SIXTEEN;
        case FarTerrainStep::SIXTEEN:
            if (effectiveDistance < 43.0) return FarTerrainStep::TWO;
            if (effectiveDistance < 66.0) return FarTerrainStep::FOUR;
            if (effectiveDistance < 126.0) return FarTerrainStep::EIGHT;
            return FarTerrainStep::SIXTEEN;
    }
    return FarTerrainStep::SIXTEEN;
}

FarTerrainTransitionSample sampleFarTerrainTransition(float elapsedSeconds) {
    if (!std::isfinite(elapsedSeconds) || elapsedSeconds <= 0.0F) return {};
    const float phase = elapsedSeconds / FAR_TERRAIN_LOD_TRANSITION_SECONDS;
    if (phase >= 1.0F) return {.drawTarget = true, .complete = true, .fogBlend = 0.0F};
    const auto smooth = [](float value) { return value * value * (3.0F - 2.0F * value); };
    if (phase < 0.5F) {
        return {.drawTarget = false, .complete = false, .fogBlend = smooth(phase * 2.0F)};
    }
    return {.drawTarget = true, .complete = false, .fogBlend = smooth((1.0F - phase) * 2.0F)};
}

void selectFarTerrainView(double cameraX, double cameraZ, int exactChunkRadius,
                          int visibleChunkRadius, std::vector<FarTerrainViewTile>& output) {
    output.clear();
    if (!std::isfinite(cameraX) || !std::isfinite(cameraZ)) return;
    exactChunkRadius = std::clamp(exactChunkRadius, 0, FAR_TERRAIN_MAX_CHUNK_RADIUS);
    visibleChunkRadius =
        std::clamp(visibleChunkRadius, exactChunkRadius, FAR_TERRAIN_MAX_CHUNK_RADIUS);
    if (visibleChunkRadius <= exactChunkRadius) return;

    const double exactBlocks = static_cast<double>(exactChunkRadius * CHUNK_EDGE);
    const double visibleBlocks = static_cast<double>(visibleChunkRadius * CHUNK_EDGE);
    const double exactSquared = exactBlocks * exactBlocks;
    const double visibleSquared = visibleBlocks * visibleBlocks;
    const int64_t cameraBlockX = static_cast<int64_t>(std::floor(cameraX));
    const int64_t cameraBlockZ = static_cast<int64_t>(std::floor(cameraZ));
    const int64_t centerTileX =
        world_coord::floorDiv(cameraBlockX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    const int64_t centerTileZ =
        world_coord::floorDiv(cameraBlockZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    const int tileRadius = static_cast<int>(std::ceil(visibleBlocks / FAR_TERRAIN_TILE_EDGE)) + 1;
    const size_t squareEdge = static_cast<size_t>(tileRadius * 2 + 1);
    output.reserve(squareEdge * squareEdge);

    auto horizontalDistanceSquared = [&](const FarTerrainBounds& bounds) {
        const double dx = cameraX < static_cast<double>(bounds.minX)
                              ? static_cast<double>(bounds.minX) - cameraX
                          : cameraX > static_cast<double>(bounds.maxX)
                              ? cameraX - static_cast<double>(bounds.maxX)
                              : 0.0;
        const double dz = cameraZ < static_cast<double>(bounds.minZ)
                              ? static_cast<double>(bounds.minZ) - cameraZ
                          : cameraZ > static_cast<double>(bounds.maxZ)
                              ? cameraZ - static_cast<double>(bounds.maxZ)
                              : 0.0;
        return dx * dx + dz * dz;
    };
    auto farthestDistanceSquared = [&](const FarTerrainBounds& bounds) {
        double result = 0.0;
        for (int xSide = 0; xSide < 2; ++xSide) {
            for (int zSide = 0; zSide < 2; ++zSide) {
                const double x = static_cast<double>(xSide == 0 ? bounds.minX : bounds.maxX);
                const double z = static_cast<double>(zSide == 0 ? bounds.minZ : bounds.maxZ);
                result =
                    std::max(result, (x - cameraX) * (x - cameraX) + (z - cameraZ) * (z - cameraZ));
            }
        }
        return result;
    };

    for (int dz = -tileRadius; dz <= tileRadius; ++dz) {
        for (int dx = -tileRadius; dx <= tileRadius; ++dx) {
            FarTerrainKey key{centerTileX + dx, centerTileZ + dz, FarTerrainStep::TWO};
            FarTerrainBounds bounds;
            bounds.minX = tileOrigin(key.tileX);
            bounds.maxX = bounds.minX + FAR_TERRAIN_TILE_EDGE;
            bounds.minZ = tileOrigin(key.tileZ);
            bounds.maxZ = bounds.minZ + FAR_TERRAIN_TILE_EDGE;
            bounds.minY = static_cast<float>(WORLD_MIN_Y);
            bounds.maxY = static_cast<float>(WORLD_MAX_Y + 1);
            const double nearestSquared = horizontalDistanceSquared(bounds);
            if (nearestSquared >= visibleSquared ||
                farthestDistanceSquared(bounds) <= exactSquared) {
                continue;
            }
            const double lodDistanceChunks =
                std::max(static_cast<double>(FAR_TERRAIN_NEAR_CHUNK_RADIUS),
                         std::sqrt(nearestSquared) / CHUNK_EDGE);
            const auto step = farTerrainStepForChunkDistance(lodDistanceChunks);
            if (!step) continue;
            key.step = *step;
            output.push_back({key, bounds, nearestSquared, lodDistanceChunks});
        }
    }
    std::sort(output.begin(), output.end(), [](const auto& first, const auto& second) {
        if (first.distanceSquared != second.distanceSquared) {
            return first.distanceSquared < second.distanceSquared;
        }
        if (first.key.tileX != second.key.tileX) return first.key.tileX < second.key.tileX;
        return first.key.tileZ < second.key.tileZ;
    });
}

size_t FarTerrainKeyHash::operator()(const FarTerrainKey& key) const noexcept {
    uint64_t hash = mix64(static_cast<uint64_t>(key.tileX));
    hash = mix64(hash ^ static_cast<uint64_t>(key.tileZ));
    hash = mix64(hash ^ static_cast<uint8_t>(key.step));
    return static_cast<size_t>(hash);
}

size_t FarTerrainMesh::byteSize() const {
    return sizeof(*this) + vertices.capacity() * sizeof(Vertex) +
           indices.capacity() * sizeof(uint32_t);
}

std::shared_ptr<const FarTerrainMesh> FarTerrainMesher::build(FarTerrainKey key,
                                                              const FarTerrainSource& source) {
    if (!source.geometry || !source.material) {
        throw std::invalid_argument("far terrain source is incomplete");
    }
    if (!validStep(key.step)) throw std::invalid_argument("unsupported far terrain LOD step");
    const bool useExactNearSource =
        (key.step == FarTerrainStep::TWO || key.step == FarTerrainStep::FOUR);
    const FarTerrainSource::GeometryFunction& geometryFunction =
        useExactNearSource && source.nearGeometry ? source.nearGeometry : source.geometry;
    const FarTerrainSource::MaterialFunction& materialFunction =
        useExactNearSource && source.nearMaterial ? source.nearMaterial : source.material;
    const int step = farTerrainStepSize(key.step);
    const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
    const int sampleEdge = cellEdge + 1;
    auto mesh = std::make_shared<FarTerrainMesh>();
    mesh->key = key;
    mesh->originX = tileOrigin(key.tileX);
    mesh->originZ = tileOrigin(key.tileZ);
    mesh->bounds.minX = mesh->originX;
    mesh->bounds.maxX = mesh->originX + FAR_TERRAIN_TILE_EDGE;
    mesh->bounds.minZ = mesh->originZ;
    mesh->bounds.maxZ = mesh->originZ + FAR_TERRAIN_TILE_EDGE;
    mesh->bounds.minY = std::numeric_limits<float>::max();
    mesh->bounds.maxY = std::numeric_limits<float>::lowest();
    mesh->surfaceBounds = mesh->bounds;
    constexpr int PATCHES_PER_EDGE = FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
    for (int patchZ = 0; patchZ < PATCHES_PER_EDGE; ++patchZ) {
        for (int patchX = 0; patchX < PATCHES_PER_EDGE; ++patchX) {
            FarTerrainBounds& patch =
                mesh->occluderPatches[static_cast<size_t>(patchZ * PATCHES_PER_EDGE + patchX)];
            patch.minX = mesh->originX + patchX * FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.maxX = patch.minX + FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.minZ = mesh->originZ + patchZ * FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.maxZ = patch.minZ + FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.minY = std::numeric_limits<float>::max();
            patch.maxY = std::numeric_limits<float>::lowest();
        }
    }

    std::vector<FarTerrainGeometrySample> samples(static_cast<size_t>(sampleEdge * sampleEdge));
    auto sampleAt = [&](int x, int z) -> FarTerrainGeometrySample& {
        return samples[static_cast<size_t>(z * sampleEdge + x)];
    };
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const int64_t worldX = mesh->originX + static_cast<int64_t>(x * step);
            const int64_t worldZ = mesh->originZ + static_cast<int64_t>(z * step);
            sampleAt(x, z) = geometryFunction(worldX, worldZ);
        }
    }

    double maximumSampleSlope = 0.0;
    bool hasWaterBoundary = false;
    bool hasChannelFeature = false;
    bool hasLakeFeature = false;
    auto sampleIsWet = [](const FarTerrainGeometrySample& sample) {
        return hasWater(sample) && sample.waterSurface > sample.terrainHeight + 0.01;
    };
    auto makeWaterEdgeRefiner = [&](FarTerrainSource::GeometryFunction sampler) {
        return WaterEdgeRefiner{
            [&, sampler = std::move(sampler)](const WaterPoint& first, const WaterPoint& second) {
                WaterPoint wet = first.wet ? first : second;
                const WaterPoint dry = first.wet ? second : first;
                const int subdivisions =
                    std::max(1, static_cast<int>(std::ceil(
                                    std::max(std::abs(dry.x - wet.x), std::abs(dry.z - wet.z)))));
                WaterPoint lastWet = wet;
                for (int subdivision = 1; subdivision <= subdivisions; ++subdivision) {
                    const float amount = static_cast<float>(subdivision) / subdivisions;
                    const float localX = wet.x + (dry.x - wet.x) * amount;
                    const float localZ = wet.z + (dry.z - wet.z) * amount;
                    const int64_t worldX =
                        mesh->originX + static_cast<int64_t>(std::llround(localX));
                    const int64_t worldZ =
                        mesh->originZ + static_cast<int64_t>(std::llround(localZ));
                    const FarTerrainGeometrySample sample = sampler(worldX, worldZ);
                    WaterPoint current{
                        .x = localX,
                        .z = localZ,
                        .height = vertexHeight(sample.waterSurface),
                        .wet = sampleIsWet(sample),
                    };
                    if (!current.wet) {
                        return WaterPoint{
                            .x = (lastWet.x + current.x) * 0.5F,
                            .z = (lastWet.z + current.z) * 0.5F,
                            .height = lastWet.height,
                            .wet = true,
                        };
                    }
                    lastWet = current;
                }
                return WaterPoint{
                    .x = (wet.x + dry.x) * 0.5F,
                    .z = (wet.z + dry.z) * 0.5F,
                    .height = wet.height,
                    .wet = true,
                };
            },
        };
    };
    const WaterEdgeRefiner refineWaterEdge = makeWaterEdgeRefiner(geometryFunction);

    // Every LOD samples tile-face water on the same two-block lattice. A
    // narrow river can be smaller than a coarse interior cell, but it cannot
    // disappear on one side of a fine/coarse tile seam.
    constexpr int SHARED_WATER_EDGE_STEP = 2;
    enum WaterEdge : size_t { WEST = 0, EAST = 1, NORTH = 2, SOUTH = 3 };
    std::array<bool, 4> refineWaterBoundary{};
    std::unordered_map<ColumnPos, FarTerrainGeometrySample> sharedWaterSamples;
    auto sharedWaterSample = [&](int localX, int localZ) -> const FarTerrainGeometrySample& {
        const ColumnPos key{mesh->originX + localX, mesh->originZ + localZ};
        auto found = sharedWaterSamples.find(key);
        if (found != sharedWaterSamples.end()) return found->second;
        return sharedWaterSamples.emplace(key, source.geometry(key.x, key.z)).first->second;
    };
    auto probeWaterBoundary = [&](WaterEdge edge) {
        bool sawWet = false;
        bool sawDry = false;
        for (int coordinate = 0; coordinate <= FAR_TERRAIN_TILE_EDGE;
             coordinate += SHARED_WATER_EDGE_STEP) {
            const int localX = edge == WEST ? 0 : edge == EAST ? FAR_TERRAIN_TILE_EDGE : coordinate;
            const int localZ = edge == NORTH   ? 0
                               : edge == SOUTH ? FAR_TERRAIN_TILE_EDGE
                                               : coordinate;
            const FarTerrainGeometrySample& sample = sharedWaterSample(localX, localZ);
            const bool wet = sampleIsWet(sample);
            sawWet = sawWet || wet;
            sawDry = sawDry || !wet;
            hasChannelFeature = hasChannelFeature || sample.river || sample.delta;
            hasLakeFeature = hasLakeFeature || sample.lake;
        }
        refineWaterBoundary[edge] = sawWet && sawDry;
        hasWaterBoundary = hasWaterBoundary || refineWaterBoundary[edge];
    };
    probeWaterBoundary(WEST);
    probeWaterBoundary(EAST);
    probeWaterBoundary(NORTH);
    probeWaterBoundary(SOUTH);
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const FarTerrainGeometrySample& sample = sampleAt(x, z);
            hasChannelFeature = hasChannelFeature || sample.river || sample.delta;
            hasLakeFeature = hasLakeFeature || sample.lake;
            if (x + 1 < sampleEdge) {
                const FarTerrainGeometrySample& east = sampleAt(x + 1, z);
                maximumSampleSlope = std::max(
                    maximumSampleSlope, std::abs(east.terrainHeight - sample.terrainHeight) / step);
                hasWaterBoundary = hasWaterBoundary || sampleIsWet(sample) != sampleIsWet(east);
            }
            if (z + 1 < sampleEdge) {
                const FarTerrainGeometrySample& south = sampleAt(x, z + 1);
                maximumSampleSlope =
                    std::max(maximumSampleSlope,
                             std::abs(south.terrainHeight - sample.terrainHeight) / step);
                hasWaterBoundary = hasWaterBoundary || sampleIsWet(sample) != sampleIsWet(south);
            }
        }
    }
    const float terrainComplexity =
        static_cast<float>(std::clamp(maximumSampleSlope / 1.5, 0.0, 1.0));
    const float hydrologyComplexity = hasWaterBoundary    ? 1.0F
                                      : hasChannelFeature ? 0.85F
                                      : hasLakeFeature    ? 0.45F
                                                          : 0.0F;
    mesh->complexity = std::max(terrainComplexity, hydrologyComplexity);

    const int64_t materialSpacing = key.step == FarTerrainStep::TWO
                                        ? FAR_TERRAIN_FINE_MATERIAL_SAMPLE_EDGE
                                        : FAR_TERRAIN_COARSE_MATERIAL_SAMPLE_EDGE;
    std::unordered_map<ColumnPos, BlockType> materialCache;
    auto materialAt = [&](int64_t worldX, int64_t worldZ) {
        const ColumnPos materialCell{
            world_coord::floorDiv(worldX, materialSpacing),
            world_coord::floorDiv(worldZ, materialSpacing),
        };
        auto found = materialCache.find(materialCell);
        if (found != materialCache.end()) return found->second;
        const int64_t alignedX = materialCell.x * materialSpacing;
        const int64_t alignedZ = materialCell.z * materialSpacing;
        const FarTerrainGeometrySample geometry = geometryFunction(alignedX, alignedZ);
        const BlockType material = materialFunction(alignedX, alignedZ, geometry);
        materialCache.emplace(materialCell, material);
        return material;
    };

    std::vector<FarCell> cells(static_cast<size_t>(cellEdge * cellEdge));
    auto cellAt = [&](int x, int z) -> FarCell& {
        return cells[static_cast<size_t>(z * cellEdge + x)];
    };
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const std::array<FarTerrainGeometrySample*, 4> corners = {
                &sampleAt(x, z), &sampleAt(x + 1, z), &sampleAt(x + 1, z + 1), &sampleAt(x, z + 1)};
            for (size_t corner = 0; corner < corners.size(); ++corner) {
                cell.terrain[corner] = vertexHeight(corners[corner]->terrainHeight);
            }
            const int patchX = x * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            const int patchZ = z * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            FarTerrainBounds& patch =
                mesh->occluderPatches[static_cast<size_t>(patchZ * PATCHES_PER_EDGE + patchX)];
            for (float height : cell.terrain) {
                patch.minY = std::min(patch.minY, height);
                patch.maxY = std::max(patch.maxY, height);
            }
            cell.flat = cell.terrain[0] == cell.terrain[1] && cell.terrain[0] == cell.terrain[2] &&
                        cell.terrain[0] == cell.terrain[3];
            const int64_t worldX = mesh->originX + static_cast<int64_t>(x * step);
            const int64_t worldZ = mesh->originZ + static_cast<int64_t>(z * step);
            cell.material = materialAt(worldX, worldZ);
            for (size_t corner = 0; corner < corners.size(); ++corner) {
                cell.waterSurface[corner] = vertexHeight(corners[corner]->waterSurface);
                if (sampleIsWet(*corners[corner])) {
                    cell.waterMask |= static_cast<uint8_t>(1U << corner);
                }
            }
            cell.water = cell.waterMask != 0;
            cell.flatWater = cell.waterMask == 0x0FU &&
                             cell.waterSurface[0] == cell.waterSurface[1] &&
                             cell.waterSurface[0] == cell.waterSurface[2] &&
                             cell.waterSurface[0] == cell.waterSurface[3];
            cell.waterHeight =
                *std::max_element(cell.waterSurface.begin(), cell.waterSurface.end());
            if (cell.waterMask != 0 && cell.waterMask != 0x0FU) {
                const int64_t centerX = worldX + step / 2;
                const int64_t centerZ = worldZ + step / 2;
                const FarTerrainGeometrySample center = geometryFunction(centerX, centerZ);
                cell.centerWet = sampleIsWet(center);
                cell.centerWaterHeight = vertexHeight(center.waterSurface);
            }
        }
    }

    std::vector<uint8_t> merged(static_cast<size_t>(cellEdge * cellEdge), 0);
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const size_t cellIndex = static_cast<size_t>(z * cellEdge + x);
            if (merged[cellIndex] != 0) continue;
            if (!cell.flat) {
                const float x0 = static_cast<float>(x * step);
                const float z0 = static_cast<float>(z * step);
                pushTerrainTop(*mesh, cell.material, x0, z0, x0 + step, z0 + step, cell.terrain[0],
                               cell.terrain[1], cell.terrain[2], cell.terrain[3]);
                merged[cellIndex] = 1;
                continue;
            }
            int width = 1;
            while (x + width < cellEdge &&
                   merged[static_cast<size_t>(z * cellEdge + x + width)] == 0 &&
                   sameFlatTerrain(cell, cellAt(x + width, z))) {
                ++width;
            }
            int depth = 1;
            while (z + depth < cellEdge) {
                bool compatible = true;
                for (int offset = 0; offset < width; ++offset) {
                    const size_t candidate =
                        static_cast<size_t>((z + depth) * cellEdge + x + offset);
                    if (merged[candidate] != 0 ||
                        !sameFlatTerrain(cell, cellAt(x + offset, z + depth))) {
                        compatible = false;
                        break;
                    }
                }
                if (!compatible) break;
                ++depth;
            }
            for (int dz = 0; dz < depth; ++dz) {
                for (int dx = 0; dx < width; ++dx) {
                    merged[static_cast<size_t>((z + dz) * cellEdge + x + dx)] = 1;
                }
            }
            mesh->mergedTerrainCellCount += static_cast<uint32_t>(width * depth);
            const float x0 = static_cast<float>(x * step);
            const float z0 = static_cast<float>(z * step);
            const float x1 = static_cast<float>((x + width) * step);
            const float z1 = static_cast<float>((z + depth) * step);
            pushTerrainTop(*mesh, cell.material, x0, z0, x1, z1, cell.terrain[0], cell.terrain[0],
                           cell.terrain[0], cell.terrain[0]);
        }
    }

    auto addEdgeSkirt = [&](int x, int z, int nextX, int nextZ, FaceNormal face) {
        const FarCell& materialCell = cellAt(std::min(x, cellEdge - 1), std::min(z, cellEdge - 1));
        const float x0 = static_cast<float>(x * step);
        const float z0 = static_cast<float>(z * step);
        const float x1 = static_cast<float>(nextX * step);
        const float z1 = static_cast<float>(nextZ * step);
        const float top0 = vertexHeight(sampleAt(x, z).terrainHeight);
        const float top1 = vertexHeight(sampleAt(nextX, nextZ).terrainHeight);
        pushSkirt(*mesh, face, materialCell.material, x0, z0, x1, z1, top0, top1);
    };
    for (int coordinate = 0; coordinate < cellEdge; ++coordinate) {
        addEdgeSkirt(0, coordinate, 0, coordinate + 1, FaceNormal::MINUS_X);
        addEdgeSkirt(cellEdge, coordinate + 1, cellEdge, coordinate, FaceNormal::PLUS_X);
        addEdgeSkirt(coordinate + 1, 0, coordinate, 0, FaceNormal::MINUS_Z);
        addEdgeSkirt(coordinate, cellEdge, coordinate + 1, cellEdge, FaceNormal::PLUS_Z);
    }

    if (source.canopies) {
        const std::vector<FarCanopy> canopies =
            source.canopies(mesh->originX, mesh->originZ, mesh->originX + FAR_TERRAIN_TILE_EDGE,
                            mesh->originZ + FAR_TERRAIN_TILE_EDGE, key.step);
        for (const FarCanopy& canopy : canopies) {
            // The anchor's half-open tile owns the complete impostor. Its
            // conservative box may cross the tile face, matching exact tree
            // ownership without duplicate coplanar foliage.
            if (canopy.x < mesh->originX || canopy.x >= mesh->originX + FAR_TERRAIN_TILE_EDGE ||
                canopy.z < mesh->originZ || canopy.z >= mesh->originZ + FAR_TERRAIN_TILE_EDGE ||
                (!canopy.aggregate && !retainsCanopy(key.step, canopy.anchorId))) {
                continue;
            }
            const FarTerrainGeometrySample anchorGeometry = geometryFunction(canopy.x, canopy.z);
            const float ground = vertexHeight(anchorGeometry.terrainHeight);
            const float localAnchorX = static_cast<float>(canopy.x - mesh->originX);
            const float localAnchorZ = static_cast<float>(canopy.z - mesh->originZ);
            const float canopyCenterX = localAnchorX + canopy.canopyOffsetX;
            const float canopyCenterZ = localAnchorZ + canopy.canopyOffsetZ;
            const float canopyBottom =
                ground + static_cast<float>(canopy.canopyMinimumY - canopy.baseY);
            const float canopyTop =
                ground + static_cast<float>(canopy.canopyMaximumY + 1 - canopy.baseY);
            const float trunkTop = ground + static_cast<float>(canopy.topY + 1 - canopy.baseY);
            const float lodExpansion = key.step == FarTerrainStep::EIGHT     ? 0.5F
                                       : key.step == FarTerrainStep::SIXTEEN ? 1.0F
                                                                             : 0.0F;
            const float canopyRadius =
                std::max(1.0F, static_cast<float>(canopy.canopyRadius) + 0.5F + lodExpansion);
            if (canopy.logBlock != BlockType::AIR && trunkTop > ground) {
                pushCanopyBox(*mesh, canopy.logBlock, localAnchorX + 0.5F, localAnchorZ + 0.5F,
                              0.45F, ground, trunkTop, false);
            }
            pushCanopyBox(*mesh, canopy.leafBlock, canopyCenterX + 0.5F, canopyCenterZ + 0.5F,
                          canopyRadius, canopyBottom, canopyTop, true);
            ++mesh->canopyAnchorCount;
        }
    }
    mesh->opaqueIndexCount = static_cast<uint32_t>(mesh->indices.size());

    // Lake outlets retain the downstream body's standing surface and add a
    // separately owned falling prism. The half-open tile containing the
    // outlet anchor owns the complete narrow prism, even when it crosses a
    // tile face, so no adjacent tile duplicates a coplanar waterfall wall.
    for (int z = 0; z < sampleEdge - 1; ++z) {
        for (int x = 0; x < sampleEdge - 1; ++x) {
            const FarTerrainGeometrySample& sample = sampleAt(x, z);
            if (!sample.waterfall || !sample.waterfallAnchor ||
                sample.waterfallTop < sample.waterfallBottom + 0.5) {
                continue;
            }
            pushWaterfallPrism(*mesh, static_cast<float>(x * step), static_cast<float>(z * step),
                               sample);
            mesh->complexity = 1.0F;
        }
    }

    std::fill(merged.begin(), merged.end(), 0);
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            const bool replacedBySharedBoundary =
                (x == 0 && refineWaterBoundary[WEST]) ||
                (x == cellEdge - 1 && refineWaterBoundary[EAST]) ||
                (z == 0 && refineWaterBoundary[NORTH]) ||
                (z == cellEdge - 1 && refineWaterBoundary[SOUTH]);
            if (replacedBySharedBoundary) {
                merged[static_cast<size_t>(z * cellEdge + x)] = 1;
            }
        }
    }
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const size_t cellIndex = static_cast<size_t>(z * cellEdge + x);
            if (!cell.water || merged[cellIndex] != 0) continue;
            const float x0 = static_cast<float>(x * step);
            const float z0 = static_cast<float>(z * step);
            const float x1 = static_cast<float>((x + 1) * step);
            const float z1 = static_cast<float>((z + 1) * step);
            if (cell.waterMask != 0x0FU) {
                const std::array<WaterPoint, 4> corners = {{
                    {x0, z0, cell.waterSurface[0], (cell.waterMask & (1U << 0U)) != 0},
                    {x1, z0, cell.waterSurface[1], (cell.waterMask & (1U << 1U)) != 0},
                    {x1, z1, cell.waterSurface[2], (cell.waterMask & (1U << 2U)) != 0},
                    {x0, z1, cell.waterSurface[3], (cell.waterMask & (1U << 3U)) != 0},
                }};
                const WaterPoint center{(x0 + x1) * 0.5F, (z0 + z1) * 0.5F, cell.centerWaterHeight,
                                        cell.centerWet};
                pushWaterContourTriangle(*mesh, {corners[0], corners[3], center}, refineWaterEdge);
                pushWaterContourTriangle(*mesh, {corners[3], corners[2], center}, refineWaterEdge);
                pushWaterContourTriangle(*mesh, {corners[2], corners[1], center}, refineWaterEdge);
                pushWaterContourTriangle(*mesh, {corners[1], corners[0], center}, refineWaterEdge);
                merged[cellIndex] = 1;
                continue;
            }
            if (!cell.flatWater) {
                pushWaterTop(*mesh, x0, z0, x1, z1, cell.waterSurface);
                merged[cellIndex] = 1;
                continue;
            }
            int width = 1;
            while (x + width < cellEdge &&
                   merged[static_cast<size_t>(z * cellEdge + x + width)] == 0 &&
                   sameWater(cell, cellAt(x + width, z))) {
                ++width;
            }
            int depth = 1;
            while (z + depth < cellEdge) {
                bool compatible = true;
                for (int offset = 0; offset < width; ++offset) {
                    const size_t candidate =
                        static_cast<size_t>((z + depth) * cellEdge + x + offset);
                    if (merged[candidate] != 0 || !sameWater(cell, cellAt(x + offset, z + depth))) {
                        compatible = false;
                        break;
                    }
                }
                if (!compatible) break;
                ++depth;
            }
            for (int dz = 0; dz < depth; ++dz) {
                for (int dx = 0; dx < width; ++dx) {
                    merged[static_cast<size_t>((z + dz) * cellEdge + x + dx)] = 1;
                }
            }
            pushWaterTop(*mesh, x0, z0, static_cast<float>((x + width) * step),
                         static_cast<float>((z + depth) * step), cell.waterHeight);
        }
    }

    const WaterEdgeRefiner refineSharedWaterEdge = makeWaterEdgeRefiner(source.geometry);
    auto emitSharedWaterCell = [&](int x0, int z0, int x1, int z1) {
        const std::array<std::array<int, 2>, 4> coordinates = {{
            {{x0, z0}},
            {{x1, z0}},
            {{x1, z1}},
            {{x0, z1}},
        }};
        std::array<WaterPoint, 4> corners{};
        bool allWet = true;
        bool anyWet = false;
        bool flatWater = true;
        float firstHeight = 0.0F;
        for (size_t index = 0; index < coordinates.size(); ++index) {
            const auto [localX, localZ] = coordinates[index];
            const FarTerrainGeometrySample& sample = sharedWaterSample(localX, localZ);
            corners[index] = {
                .x = static_cast<float>(localX),
                .z = static_cast<float>(localZ),
                .height = vertexHeight(sample.waterSurface),
                .wet = sampleIsWet(sample),
            };
            if (index == 0) firstHeight = corners[index].height;
            allWet = allWet && corners[index].wet;
            anyWet = anyWet || corners[index].wet;
            flatWater = flatWater && corners[index].height == firstHeight;
        }
        const int centerX = (x0 + x1) / 2;
        const int centerZ = (z0 + z1) / 2;
        const FarTerrainGeometrySample& centerSample = sharedWaterSample(centerX, centerZ);
        const WaterPoint center{
            .x = static_cast<float>(centerX),
            .z = static_cast<float>(centerZ),
            .height = vertexHeight(centerSample.waterSurface),
            .wet = sampleIsWet(centerSample),
        };
        anyWet = anyWet || center.wet;
        if (!anyWet) return;
        if (allWet && flatWater) {
            pushWaterTop(*mesh, static_cast<float>(x0), static_cast<float>(z0),
                         static_cast<float>(x1), static_cast<float>(z1), firstHeight);
            return;
        }
        pushWaterContourTriangle(*mesh, {corners[0], corners[3], center}, refineSharedWaterEdge);
        pushWaterContourTriangle(*mesh, {corners[3], corners[2], center}, refineSharedWaterEdge);
        pushWaterContourTriangle(*mesh, {corners[2], corners[1], center}, refineSharedWaterEdge);
        pushWaterContourTriangle(*mesh, {corners[1], corners[0], center}, refineSharedWaterEdge);
    };
    auto emitSharedBoundary = [&](WaterEdge edge) {
        if (!refineWaterBoundary[edge]) return;
        for (int coordinate = 0; coordinate < FAR_TERRAIN_TILE_EDGE;
             coordinate += SHARED_WATER_EDGE_STEP) {
            if (edge == WEST) {
                emitSharedWaterCell(0, coordinate, step, coordinate + SHARED_WATER_EDGE_STEP);
            } else if (edge == EAST) {
                emitSharedWaterCell(FAR_TERRAIN_TILE_EDGE - step, coordinate, FAR_TERRAIN_TILE_EDGE,
                                    coordinate + SHARED_WATER_EDGE_STEP);
            } else if (edge == NORTH) {
                emitSharedWaterCell(coordinate, 0, coordinate + SHARED_WATER_EDGE_STEP, step);
            } else {
                emitSharedWaterCell(coordinate, FAR_TERRAIN_TILE_EDGE - step,
                                    coordinate + SHARED_WATER_EDGE_STEP, FAR_TERRAIN_TILE_EDGE);
            }
        }
    };
    emitSharedBoundary(WEST);
    emitSharedBoundary(EAST);
    emitSharedBoundary(NORTH);
    emitSharedBoundary(SOUTH);

    if (mesh->vertices.empty()) {
        mesh->bounds.minY = 0.0F;
        mesh->bounds.maxY = 0.0F;
        mesh->surfaceBounds.minY = 0.0F;
        mesh->surfaceBounds.maxY = 0.0F;
    }
    for (FarTerrainBounds& patch : mesh->occluderPatches) {
        if (patch.minY > patch.maxY) {
            patch.minY = mesh->surfaceBounds.minY;
            patch.maxY = mesh->surfaceBounds.maxY;
        }
    }
    mesh->deterministicHash = hashMesh(*mesh);
    return mesh;
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainMesher::buildFromSurface(FarTerrainKey key, const SurfaceSampleFunction& sampleSurface) {
    return build(key, surfaceGeometrySource(sampleSurface));
}

FarTerrainSource FarTerrainMesher::surfaceGeometrySource(SurfaceSampleFunction sampleSurface) {
    if (!sampleSurface) throw std::invalid_argument("far terrain surface sampler is empty");
    FarTerrainSource source;
    source.geometry = [sampleSurface](int64_t x, int64_t z) {
        return geometryFromSurface(sampleSurface(x, z));
    };
    source.material = [sampleSurface](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        return terrainMaterial(sampleSurface(x, z));
    };
    return source;
}

FarTerrainSource
FarTerrainMesher::tieredSurfaceGeometrySource(SurfaceSampleFunction exactNearSurface,
                                              SurfaceSampleFunction coarseSurface) {
    FarTerrainSource source = surfaceGeometrySource(std::move(coarseSurface));
    if (!exactNearSurface) {
        throw std::invalid_argument("far terrain exact near surface sampler is empty");
    }
    source.nearGeometry = [exactNearSurface](int64_t x, int64_t z) {
        return geometryFromSurface(exactNearSurface(x, z));
    };
    source.nearMaterial = [exactNearSurface](int64_t x, int64_t z,
                                             const FarTerrainGeometrySample&) {
        return terrainMaterial(exactNearSurface(x, z));
    };
    return source;
}

FarTerrainSource
FarTerrainMesher::macroGeometrySource(std::shared_ptr<worldgen::MacroGenerationSampler> sampler) {
    if (!sampler) throw std::invalid_argument("far terrain macro sampler is empty");
    FarTerrainSource source;
    source.geometry = [sampler](int64_t x, int64_t z) {
        const worldgen::HydrologySample hydrology =
            sampler->sampleHydrology(static_cast<double>(x), static_cast<double>(z));
        FarTerrainGeometrySample result;
        result.terrainHeight = hydrology.surfaceElevation;
        result.waterSurface = (hydrology.ocean || hydrology.river || hydrology.lake)
                                  ? std::ceil(hydrology.waterSurface) - 0.125
                                  : hydrology.waterSurface;
        result.discharge = hydrology.discharge;
        result.sediment = hydrology.sediment;
        result.waterfallTop = hydrology.waterfallTop;
        result.waterfallBottom = hydrology.waterfallBottom;
        result.waterfallWidth = hydrology.waterfallWidth;
        result.flowX = hydrology.flowDirection.x;
        result.flowZ = hydrology.flowDirection.z;
        result.ocean = hydrology.ocean;
        result.river = hydrology.river;
        result.lake = hydrology.lake;
        result.waterfall = hydrology.waterfall;
        result.waterfallAnchor = hydrology.waterfallAnchor;
        result.delta = hydrology.delta;
        return result;
    };
    source.material = [sampler](int64_t x, int64_t z, const FarTerrainGeometrySample& geometry) {
        return macroGeometryMaterial(
            sampler->sampleGeology(static_cast<double>(x), static_cast<double>(z)), geometry);
    };
    return source;
}

TerrainHorizonCuller::TerrainHorizonCuller(TerrainHorizonViewpoint viewpoint) {
    reset(viewpoint);
}

void TerrainHorizonCuller::reset(TerrainHorizonViewpoint viewpoint) {
    viewpoint_ = viewpoint;
    horizonCounts_.fill(0);
}

double TerrainHorizonCuller::horizontalDistanceSquared(const FarTerrainBounds& bounds,
                                                       TerrainHorizonViewpoint viewpoint) {
    const double dx = viewpoint.x < static_cast<double>(bounds.minX)
                          ? static_cast<double>(bounds.minX) - viewpoint.x
                      : viewpoint.x > static_cast<double>(bounds.maxX)
                          ? viewpoint.x - static_cast<double>(bounds.maxX)
                          : 0.0;
    const double dz = viewpoint.z < static_cast<double>(bounds.minZ)
                          ? static_cast<double>(bounds.minZ) - viewpoint.z
                      : viewpoint.z > static_cast<double>(bounds.maxZ)
                          ? viewpoint.z - static_cast<double>(bounds.maxZ)
                          : 0.0;
    return dx * dx + dz * dz;
}

bool TerrainHorizonCuller::isOccluded(const FarTerrainBounds& surfaceBounds) const {
    const AzimuthCoverage coverage = azimuthCoverage(surfaceBounds, viewpoint_);
    const double nearestDistance = std::sqrt(horizontalDistanceSquared(surfaceBounds, viewpoint_));
    if (nearestDistance <= 1.0e-9) return false;
    const double farthestDistance = farthestHorizontalDistance(surfaceBounds, viewpoint_);
    const double verticalDelta = static_cast<double>(surfaceBounds.maxY) - viewpoint_.y;
    const double distance = verticalDelta >= 0.0 ? nearestDistance : farthestDistance;
    const double maximumElevation = std::atan2(verticalDelta, std::max(distance, 1.0e-9));
    bool occluded = true;
    const size_t count = visitIntersectedBins(coverage, [&](size_t bin) {
        bool binOccluded = false;
        for (uint8_t index = 0; index < horizonCounts_[bin]; ++index) {
            const HorizonEntry& horizon = horizons_[bin][index];
            if (horizon.farthestDistance < nearestDistance - 1.0e-9 &&
                maximumElevation < horizon.minimumElevation - 1.0e-12) {
                binOccluded = true;
                break;
            }
        }
        if (!binOccluded) {
            occluded = false;
        }
    });
    return count != 0 && occluded;
}

void TerrainHorizonCuller::addOccluder(const FarTerrainBounds& surfaceBounds) {
    const AzimuthCoverage coverage = azimuthCoverage(surfaceBounds, viewpoint_);
    const double nearestDistance = std::sqrt(horizontalDistanceSquared(surfaceBounds, viewpoint_));
    if (nearestDistance <= 1.0e-9) return;
    const double farthestDistance = farthestHorizontalDistance(surfaceBounds, viewpoint_);
    const double verticalDelta = static_cast<double>(surfaceBounds.minY) - viewpoint_.y;
    const double distance = verticalDelta >= 0.0 ? farthestDistance : nearestDistance;
    const double minimumElevation = std::atan2(verticalDelta, std::max(distance, 1.0e-9));
    visitFullyCoveredBins(coverage, [&](size_t bin) {
        auto& entries = horizons_[bin];
        uint8_t& count = horizonCounts_[bin];

        // Keep a bounded Pareto frontier. A nearer, higher horizon dominates
        // a farther, lower one for every possible candidate. Dropping an entry
        // when the fixed frontier is full can only reduce culling, never hide
        // visible terrain.
        for (uint8_t index = 0; index < count; ++index) {
            if (entries[index].farthestDistance <= farthestDistance &&
                entries[index].minimumElevation >= minimumElevation) {
                return;
            }
        }
        for (uint8_t index = 0; index < count;) {
            if (farthestDistance <= entries[index].farthestDistance &&
                minimumElevation >= entries[index].minimumElevation) {
                entries[index] = entries[--count];
            } else {
                ++index;
            }
        }
        if (count < MAX_HORIZONS_PER_BIN) {
            entries[count++] = {farthestDistance, minimumElevation};
        }
    });
}

bool TerrainHorizonCuller::testAndAdd(const FarTerrainBounds& surfaceBounds) {
    if (isOccluded(surfaceBounds)) return true;
    addOccluder(surfaceBounds);
    return false;
}

FarTerrainScheduler::FarTerrainScheduler(FarTerrainSource source, FarTerrainSchedulerLimits limits)
    : source_(std::move(source))
    , limits_(limits) {
    if (!source_.geometry || !source_.material) {
        throw std::invalid_argument("far terrain scheduler source is incomplete");
    }
    validateLimits(limits_);
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

FarTerrainScheduler::FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits)
    : limits_(limits) {
    validateLimits(limits_);
    auto generator = std::make_shared<ChunkGenerator>(static_cast<uint32_t>(worldSeed));
    source_ = FarTerrainMesher::tieredSurfaceGeometrySource(
        [generator](int64_t x, int64_t z) { return generator->sampleExactSurface(x, z); },
        [generator](int64_t x, int64_t z) { return generator->sampleFarSurface(x, z); });
    source_.geometry = [generator](int64_t x, int64_t z) {
        return geometryFromSurface(generator->sampleFarGeometrySurface(x, z));
    };
    source_.nearGeometry = [generator](int64_t x, int64_t z) {
        return geometryFromSurface(generator->sampleExactGeometrySurface(x, z));
    };
    source_.material = [generator](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        return generator->farSurfaceMaterialAt(x, z);
    };
    source_.nearMaterial = [generator](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
        return generator->surfaceMaterialAt(x, z);
    };
    source_.canopies = [generator](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                   int64_t maximumZ, FarTerrainStep step) {
        return generator->collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ,
                                                   farTerrainStepSize(step));
    };
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

FarTerrainScheduler::~FarTerrainScheduler() {
    shutdown();
}

bool FarTerrainScheduler::enqueue(FarTerrainKey key) {
    if (!validStep(key.step) || !running_.load(std::memory_order_acquire)) return false;
    if (findCached(key)) return false;
    const uint64_t current = epoch_.load(std::memory_order_acquire);
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            inFlight_.load(std::memory_order_relaxed) >= limits_.maxPending) {
            return false;
        }
        if (wantedFilterEnabled_ && !wantedKeys_.contains(key)) return false;
        auto active = activeKeys_.find(key);
        if (active != activeKeys_.end() && active->second == current) return false;
        activeKeys_[key] = current;
        jobs_.push_back({key, current});
        inFlight_.fetch_add(1, std::memory_order_relaxed);
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }
    jobCv_.notify_one();
    return true;
}

void FarTerrainScheduler::retainWanted(
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted) {
    size_t removed = 0;
    {
        std::lock_guard lock(jobMutex_);
        wantedFilterEnabled_ = true;
        wantedKeys_ = wanted;
        std::erase_if(jobs_, [&](const Job& job) {
            if (wanted.contains(job.key)) return false;
            const auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
            ++removed;
            return true;
        });
    }
    if (removed > 0) {
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(completedMutex_);
        std::erase_if(completed_,
                      [&](const FarTerrainResult& result) { return !wanted.contains(result.key); });
    }
    {
        std::lock_guard lock(cacheMutex_);
        std::erase_if(cache_, [&](const auto& entry) {
            if (wanted.contains(entry.first)) return false;
            cacheBytes_ -= entry.second.bytes;
            return true;
        });
    }
}

uint64_t FarTerrainScheduler::advanceEpoch() {
    const uint64_t next = epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    size_t removed = 0;
    {
        std::lock_guard lock(jobMutex_);
        removed = jobs_.size();
        for (const Job& job : jobs_) {
            auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
        }
        jobs_.clear();
    }
    if (removed > 0) {
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(completedMutex_);
        completed_.clear();
    }
    return next;
}

void FarTerrainScheduler::drainCompleted(std::vector<FarTerrainResult>& output) {
    std::lock_guard lock(completedMutex_);
    while (!completed_.empty()) {
        output.push_back(std::move(completed_.front()));
        completed_.pop_front();
    }
}

std::shared_ptr<const FarTerrainMesh> FarTerrainScheduler::findCached(FarTerrainKey key) const {
    std::lock_guard lock(cacheMutex_);
    auto found = cache_.find(key);
    if (found == cache_.end()) return {};
    found->second.lastAccess = ++accessClock_;
    cacheHits_.fetch_add(1, std::memory_order_relaxed);
    return found->second.mesh;
}

void FarTerrainScheduler::clearCache() {
    std::lock_guard lock(cacheMutex_);
    cache_.clear();
    cacheBytes_ = 0;
}

FarTerrainSchedulerStats FarTerrainScheduler::stats() const {
    FarTerrainSchedulerStats result;
    result.inFlight = inFlight_.load(std::memory_order_relaxed);
    result.epoch = epoch_.load(std::memory_order_relaxed);
    result.submitted = submitted_.load(std::memory_order_relaxed);
    result.built = built_.load(std::memory_order_relaxed);
    result.canceled = canceled_.load(std::memory_order_relaxed);
    result.failed = failed_.load(std::memory_order_relaxed);
    result.cacheHits = cacheHits_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(jobMutex_);
        result.queued = jobs_.size();
    }
    {
        std::lock_guard lock(completedMutex_);
        result.completed = completed_.size();
    }
    {
        std::lock_guard lock(cacheMutex_);
        result.cacheEntries = cache_.size();
        result.cacheBytes = cacheBytes_;
    }
    return result;
}

void FarTerrainScheduler::shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    size_t removed = 0;
    {
        std::lock_guard lock(jobMutex_);
        removed = jobs_.size();
        jobs_.clear();
        activeKeys_.clear();
    }
    if (removed > 0) {
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    jobCv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
}

void FarTerrainScheduler::finishJob(const Job& job) {
    {
        std::lock_guard lock(jobMutex_);
        auto active = activeKeys_.find(job.key);
        if (active != activeKeys_.end() && active->second == job.epoch) {
            activeKeys_.erase(active);
        }
    }
    inFlight_.fetch_sub(1, std::memory_order_relaxed);
}

void FarTerrainScheduler::storeCompleted(FarTerrainResult result) {
    std::lock_guard lock(completedMutex_);
    while (completed_.size() >= limits_.maxCompleted)
        completed_.pop_front();
    completed_.push_back(std::move(result));
}

void FarTerrainScheduler::storeCache(std::shared_ptr<const FarTerrainMesh> mesh) {
    const size_t bytes = mesh->byteSize();
    const FarTerrainKey key = mesh->key;
    if (bytes > limits_.maxCacheBytes) return;
    std::lock_guard lock(cacheMutex_);
    auto existing = cache_.find(key);
    if (existing != cache_.end()) {
        cacheBytes_ -= existing->second.bytes;
        cache_.erase(existing);
    }
    while (!cache_.empty() && (cache_.size() >= limits_.maxCacheEntries ||
                               cacheBytes_ + bytes > limits_.maxCacheBytes)) {
        auto oldest = std::min_element(
            cache_.begin(), cache_.end(), [](const auto& first, const auto& second) {
                return first.second.lastAccess < second.second.lastAccess;
            });
        cacheBytes_ -= oldest->second.bytes;
        cache_.erase(oldest);
    }
    cacheBytes_ += bytes;
    cache_.emplace(key, CacheEntry{std::move(mesh), bytes, ++accessClock_});
}

void FarTerrainScheduler::workerLoop() {
    setCurrentThreadPriority(ThreadPriority::UTILITY);
    while (true) {
        Job job;
        {
            std::unique_lock lock(jobMutex_);
            jobCv_.wait(lock, [this] {
                return !jobs_.empty() || !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_relaxed) && jobs_.empty()) return;
            job = jobs_.front();
            jobs_.pop_front();
        }
        if (job.epoch != epoch_.load(std::memory_order_acquire)) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }

        FarTerrainResult result;
        result.key = job.key;
        result.epoch = job.epoch;
        try {
            result.mesh = FarTerrainMesher::build(job.key, source_);
        } catch (...) {
            result.failed = true;
        }

        if (job.epoch != epoch_.load(std::memory_order_acquire) ||
            !running_.load(std::memory_order_acquire)) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }
        {
            std::lock_guard lock(jobMutex_);
            if (wantedFilterEnabled_ && !wantedKeys_.contains(job.key)) {
                canceled_.fetch_add(1, std::memory_order_relaxed);
                auto active = activeKeys_.find(job.key);
                if (active != activeKeys_.end() && active->second == job.epoch) {
                    activeKeys_.erase(active);
                }
                inFlight_.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
        }
        if (result.failed || !result.mesh) {
            failed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            storeCache(result.mesh);
            built_.fetch_add(1, std::memory_order_relaxed);
        }
        storeCompleted(std::move(result));
        finishJob(job);
    }
}

#pragma once

#include <common/math.hpp>
#include <world/block_properties.hpp>
#include <world/chunk.hpp>

#include <optional>
#include <utility>

// Forward declaration
class World;

// Rich ray result shared by block interaction and authored-shape highlights.
// Bounds remain local to the selected voxel so the renderer can reuse them
// without losing precision at large world coordinates.
struct VoxelRayHit {
    Vec3 blockPosition;
    Vec3 normal;
    BlockType block = BlockType::AIR;
    BlockSelectionBounds localBounds;
    float distance = 0.0F;
};

// ---------------------------------------------------------------------------
// VoxelTraversal — DDA (Digital Differential Analyzer) ray marching
//
// Efficiently traces a ray through a voxel grid, finding the first solid
// block intersected. Returns block position and optionally the face normal
// of the hit surface.
// ---------------------------------------------------------------------------
class VoxelTraversal {
public:
    // DDA ray marching through voxel grid.
    // Returns the first solid block hit, or std::nullopt if none within range.
    // maxDistance limits traversal range (default 6.0 blocks).
    static std::optional<Vec3> traceRay(const Vec3& origin, const Vec3& direction, World& world,
                                        float maxDistance = 6.0f);

    // Returns the hit block position AND the face normal of the hit surface.
    // Face normal indicates which face of the block was struck.
    static std::optional<std::pair<Vec3, Vec3>> traceRayWithNormal(const Vec3& origin,
                                                                   const Vec3& direction,
                                                                   World& world,
                                                                   float maxDistance = 6.0f);

    // Returns the authored shape, block type, and exact entry distance in
    // addition to the compatibility position and face-normal fields above.
    static std::optional<VoxelRayHit> traceRayDetailed(const Vec3& origin, const Vec3& direction,
                                                       World& world, float maxDistance = 6.0f);

private:
    struct ShapeIntersection {
        float distance = 0.0F;
        Vec3 normal;
    };

    static std::optional<ShapeIntersection>
    intersectSelectionBounds(const Vec3& origin, const Vec3& direction, const Vec3& blockPosition,
                             const BlockSelectionBounds& bounds, float maxDistance);
};

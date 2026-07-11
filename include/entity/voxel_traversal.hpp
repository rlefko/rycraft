#pragma once

#include <common/math.hpp>
#include <world/chunk.hpp>

#include <optional>
#include <utility>

// Forward declaration
class World;

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
    static std::optional<Vec3> traceRay(const Vec3& origin, const Vec3& direction,
                                         World& world, float maxDistance = 6.0f);

    // Returns the hit block position AND the face normal of the hit surface.
    // Face normal indicates which face of the block was struck.
    static std::optional<std::pair<Vec3, Vec3>> traceRayWithNormal(const Vec3& origin,
                                                                    const Vec3& direction,
                                                                    World& world,
                                                                    float maxDistance = 6.0f);

private:
    // Check if a block at integer coordinates is solid (for ray tracing).
    // Uses the same definition as PhysicsEngine::isSolid.
    static bool isBlockSolid(World& world, int x, int y, int z);
};

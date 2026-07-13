#include "entity/voxel_traversal.hpp"
#include "world/world.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// isBlockSolid — ray targeting, delegating to the shared block property
// table so raycasts agree with collision (glass is targetable).
// ---------------------------------------------------------------------------
bool VoxelTraversal::isBlockSolid(World& world, int x, int y, int z) {
    return isSolid(world.getBlock(x, y, z));
}

// ---------------------------------------------------------------------------
// traceRay — DDA ray marching, returns first solid block position
// ---------------------------------------------------------------------------
std::optional<Vec3> VoxelTraversal::traceRay(const Vec3& origin, const Vec3& direction,
                                              World& world, float maxDistance) {
    // Current voxel (floor of origin)
    int x = static_cast<int>(std::floor(origin.x));
    int y = static_cast<int>(std::floor(origin.y));
    int z = static_cast<int>(std::floor(origin.z));

    // Step direction (+1 or -1 per axis)
    int stepX = (direction.x >= 0.f) ? 1 : -1;
    int stepY = (direction.y >= 0.f) ? 1 : -1;
    int stepZ = (direction.z >= 0.f) ? 1 : -1;

    // Distance to next voxel boundary along each axis
    float tMaxX = (direction.x != 0.f) ?
        ((stepX > 0) ? (x + 1 - origin.x) : (x - origin.x)) / direction.x : INFINITY;
    float tMaxY = (direction.y != 0.f) ?
        ((stepY > 0) ? (y + 1 - origin.y) : (y - origin.y)) / direction.y : INFINITY;
    float tMaxZ = (direction.z != 0.f) ?
        ((stepZ > 0) ? (z + 1 - origin.z) : (z - origin.z)) / direction.z : INFINITY;

    // Distance between voxel boundaries along each axis
    float tDeltaX = (direction.x != 0.f) ? stepX / direction.x : INFINITY;
    float tDeltaY = (direction.y != 0.f) ? stepY / direction.y : INFINITY;
    float tDeltaZ = (direction.z != 0.f) ? stepZ / direction.z : INFINITY;

    float t = 0.f;

    // DDA loop — advance along axis with smallest tMax
    while (t < maxDistance) {
        // Check current voxel
        if (isBlockSolid(world, x, y, z)) {
            return Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
        }

        // Advance along axis with smallest tMax
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX;
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                t = tMaxZ;
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY;
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                t = tMaxZ;
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// traceRayWithNormal — DDA ray marching with face normal computation
// ---------------------------------------------------------------------------
std::optional<std::pair<Vec3, Vec3>> VoxelTraversal::traceRayWithNormal(
    const Vec3& origin, const Vec3& direction, World& world, float maxDistance) {

    // Current voxel (floor of origin)
    int x = static_cast<int>(std::floor(origin.x));
    int y = static_cast<int>(std::floor(origin.y));
    int z = static_cast<int>(std::floor(origin.z));

    // Previous voxel (for face normal computation)
    int prevX = x;
    int prevY = y;
    int prevZ = z;

    // Step direction (+1 or -1 per axis)
    int stepX = (direction.x >= 0.f) ? 1 : -1;
    int stepY = (direction.y >= 0.f) ? 1 : -1;
    int stepZ = (direction.z >= 0.f) ? 1 : -1;

    // Distance to next voxel boundary along each axis
    float tMaxX = (direction.x != 0.f) ?
        ((stepX > 0) ? (x + 1 - origin.x) : (x - origin.x)) / direction.x : INFINITY;
    float tMaxY = (direction.y != 0.f) ?
        ((stepY > 0) ? (y + 1 - origin.y) : (y - origin.y)) / direction.y : INFINITY;
    float tMaxZ = (direction.z != 0.f) ?
        ((stepZ > 0) ? (z + 1 - origin.z) : (z - origin.z)) / direction.z : INFINITY;

    // Distance between voxel boundaries along each axis
    float tDeltaX = (direction.x != 0.f) ? stepX / direction.x : INFINITY;
    float tDeltaY = (direction.y != 0.f) ? stepY / direction.y : INFINITY;
    float tDeltaZ = (direction.z != 0.f) ? stepZ / direction.z : INFINITY;

    float t = 0.f;

    // DDA loop
    while (t < maxDistance) {
        // Check current voxel
        if (isBlockSolid(world, x, y, z)) {
            // Compute face normal from previous to current position
            Vec3 normal;
            if (x != prevX) {
                normal.x = static_cast<float>(-stepX);
            } else if (y != prevY) {
                normal.y = static_cast<float>(-stepY);
            } else if (z != prevZ) {
                normal.z = static_cast<float>(-stepZ);
            }

            return std::make_pair(
                Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)},
                normal
            );
        }

        // Store previous position before advancing
        prevX = x;
        prevY = y;
        prevZ = z;

        // Advance along axis with smallest tMax
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX;
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                t = tMaxZ;
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY;
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                t = tMaxZ;
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    return std::nullopt;
}

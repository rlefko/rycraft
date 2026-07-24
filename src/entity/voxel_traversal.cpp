#include "entity/voxel_traversal.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

std::optional<VoxelTraversal::ShapeIntersection>
VoxelTraversal::intersectSelectionBounds(const Vec3& origin, const Vec3& direction,
                                         const Vec3& blockPosition,
                                         const BlockSelectionBounds& bounds, float maxDistance) {
    if (maxDistance < 0.0F) return std::nullopt;

    const Vec3 boxMin = blockPosition + bounds.min;
    const Vec3 boxMax = blockPosition + bounds.max;
    float nearDistance = -std::numeric_limits<float>::infinity();
    float farDistance = std::numeric_limits<float>::infinity();
    Vec3 nearNormal{};

    const auto intersectAxis = [&](float rayOrigin, float rayDirection, float minimum,
                                   float maximum, const Vec3& negativeNormal,
                                   const Vec3& positiveNormal) {
        constexpr float PARALLEL_EPSILON = 1.0e-8F;
        if (std::abs(rayDirection) <= PARALLEL_EPSILON) {
            return rayOrigin >= minimum && rayOrigin <= maximum;
        }

        float axisNear = 0.0F;
        float axisFar = 0.0F;
        Vec3 axisNearNormal;
        if (rayDirection > 0.0F) {
            axisNear = (minimum - rayOrigin) / rayDirection;
            axisFar = (maximum - rayOrigin) / rayDirection;
            axisNearNormal = negativeNormal;
        } else {
            axisNear = (maximum - rayOrigin) / rayDirection;
            axisFar = (minimum - rayOrigin) / rayDirection;
            axisNearNormal = positiveNormal;
        }

        if (axisNear > nearDistance) {
            nearDistance = axisNear;
            nearNormal = axisNearNormal;
        }
        farDistance = std::min(farDistance, axisFar);
        return nearDistance <= farDistance;
    };

    if (!intersectAxis(origin.x, direction.x, boxMin.x, boxMax.x, {-1.0F, 0.0F, 0.0F},
                       {1.0F, 0.0F, 0.0F}) ||
        !intersectAxis(origin.y, direction.y, boxMin.y, boxMax.y, {0.0F, -1.0F, 0.0F},
                       {0.0F, 1.0F, 0.0F}) ||
        !intersectAxis(origin.z, direction.z, boxMin.z, boxMax.z, {0.0F, 0.0F, -1.0F},
                       {0.0F, 0.0F, 1.0F})) {
        return std::nullopt;
    }

    const float hitDistance = std::max(nearDistance, 0.0F);
    if (farDistance < 0.0F || hitDistance > maxDistance) return std::nullopt;
    if (nearDistance < 0.0F) nearNormal = {};
    return ShapeIntersection{.distance = hitDistance, .normal = nearNormal};
}

// DDA visits only loaded voxels, then intersects targetable blocks against
// their authored selection bounds. A miss through the empty part of a bed or
// torch voxel continues to the next cell instead of intercepting the ray.
std::optional<VoxelRayHit> VoxelTraversal::traceRayDetailed(const Vec3& origin,
                                                            const Vec3& direction, World& world,
                                                            float maxDistance) {
    int x = static_cast<int>(std::floor(origin.x));
    int y = static_cast<int>(std::floor(origin.y));
    int z = static_cast<int>(std::floor(origin.z));

    const int stepX = direction.x >= 0.0F ? 1 : -1;
    const int stepY = direction.y >= 0.0F ? 1 : -1;
    const int stepZ = direction.z >= 0.0F ? 1 : -1;

    float tMaxX =
        direction.x != 0.0F
            ? ((stepX > 0 ? static_cast<float>(x + 1) : static_cast<float>(x)) - origin.x) /
                  direction.x
            : std::numeric_limits<float>::infinity();
    float tMaxY =
        direction.y != 0.0F
            ? ((stepY > 0 ? static_cast<float>(y + 1) : static_cast<float>(y)) - origin.y) /
                  direction.y
            : std::numeric_limits<float>::infinity();
    float tMaxZ =
        direction.z != 0.0F
            ? ((stepZ > 0 ? static_cast<float>(z + 1) : static_cast<float>(z)) - origin.z) /
                  direction.z
            : std::numeric_limits<float>::infinity();

    const float tDeltaX = direction.x != 0.0F ? static_cast<float>(stepX) / direction.x
                                              : std::numeric_limits<float>::infinity();
    const float tDeltaY = direction.y != 0.0F ? static_cast<float>(stepY) / direction.y
                                              : std::numeric_limits<float>::infinity();
    const float tDeltaZ = direction.z != 0.0F ? static_cast<float>(stepZ) / direction.z
                                              : std::numeric_limits<float>::infinity();

    float t = 0.0F;
    while (t < maxDistance) {
        const std::optional<BlockType> block = world.findBlockIfLoaded(x, y, z);
        if (!block) return std::nullopt;
        if (isTargetable(*block)) {
            const BlockSelectionBounds bounds = blockSelectionBounds(*block);
            const Vec3 blockPosition{static_cast<float>(x), static_cast<float>(y),
                                     static_cast<float>(z)};
            if (const auto intersection = intersectSelectionBounds(origin, direction, blockPosition,
                                                                   bounds, maxDistance)) {
                return VoxelRayHit{.blockPosition = blockPosition,
                                   .normal = intersection->normal,
                                   .block = *block,
                                   .localBounds = bounds,
                                   .distance = intersection->distance};
            }
        }

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
        } else if (tMaxY < tMaxZ) {
            t = tMaxY;
            y += stepY;
            tMaxY += tDeltaY;
        } else {
            t = tMaxZ;
            z += stepZ;
            tMaxZ += tDeltaZ;
        }
    }
    return std::nullopt;
}

std::optional<Vec3> VoxelTraversal::traceRay(const Vec3& origin, const Vec3& direction,
                                             World& world, float maxDistance) {
    const auto hit = traceRayDetailed(origin, direction, world, maxDistance);
    return hit ? std::optional<Vec3>{hit->blockPosition} : std::nullopt;
}

std::optional<std::pair<Vec3, Vec3>> VoxelTraversal::traceRayWithNormal(const Vec3& origin,
                                                                        const Vec3& direction,
                                                                        World& world,
                                                                        float maxDistance) {
    const auto hit = traceRayDetailed(origin, direction, world, maxDistance);
    return hit ? std::optional<std::pair<Vec3, Vec3>>{{hit->blockPosition, hit->normal}}
               : std::nullopt;
}

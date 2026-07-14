#include "entity/physics.hpp"
#include "world/world.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// isSolid — collision check, delegating to the shared block property table
// (block_properties.hpp). Notably glass IS solid: it used to be excluded
// here while the mesher rendered it as a full block, so entities fell
// through what looked like floor.
// ---------------------------------------------------------------------------
bool PhysicsEngine::isSolid(World& world, int x, int y, int z) {
    return ::isSolid(world.getBlock(x, y, z));
}

// ---------------------------------------------------------------------------
// isInWater — Check if any liquid block intersects the entity AABB (lava
// swims like water; damage is out of scope)
// ---------------------------------------------------------------------------
bool PhysicsEngine::isInWater(World& world, const AABB& entityAABB) {
    int minX = static_cast<int>(std::floor(entityAABB.min.x));
    int minY = static_cast<int>(std::floor(entityAABB.min.y));
    int minZ = static_cast<int>(std::floor(entityAABB.min.z));
    int maxX = static_cast<int>(std::ceil(entityAABB.max.x));
    int maxY = static_cast<int>(std::ceil(entityAABB.max.y));
    int maxZ = static_cast<int>(std::ceil(entityAABB.max.z));

    // Exclusive upper bounds: cell k spans [k, k+1), so the last cell the box
    // reaches is ceil(max)-1. The old inclusive <= scanned a full block past
    // the AABB on +X/+Y/+Z, reporting "in water" while merely standing beside
    // a pond — which would flick sprint into swim mode along every shoreline.
    for (int x = minX; x < maxX; ++x) {
        for (int y = minY; y < maxY; ++y) {
            for (int z = minZ; z < maxZ; ++z) {
                if (isLiquid(world.getBlock(x, y, z))) {
                    return true;
                }
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// collectObstacles — Gather all solid block AABBs intersecting the query box
// ---------------------------------------------------------------------------
std::vector<AABB> PhysicsEngine::collectObstacles(const AABB& expandedAABB, World& world) {
    std::vector<AABB> obstacles;

    int minX = static_cast<int>(std::floor(expandedAABB.min.x));
    int minY = static_cast<int>(std::floor(expandedAABB.min.y));
    int minZ = static_cast<int>(std::floor(expandedAABB.min.z));
    int maxX = static_cast<int>(std::ceil(expandedAABB.max.x));
    int maxY = static_cast<int>(std::ceil(expandedAABB.max.y));
    int maxZ = static_cast<int>(std::ceil(expandedAABB.max.z));

    // Expand by ±1 margin for edge cases
    minX -= 1;
    minY -= 1;
    minZ -= 1;
    maxX += 1;
    maxY += 1;
    maxZ += 1;

    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                if (PhysicsEngine::isSolid(world, x, y, z)) {
                    obstacles.emplace_back(
                        Vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)},
                        Vec3{static_cast<float>(x) + 1.f, static_cast<float>(y) + 1.f,
                             static_cast<float>(z) + 1.f});
                }
            }
        }
    }

    return obstacles;
}

// ---------------------------------------------------------------------------
// resolveAxis — Resolve collision along a single axis.
//
// Clamps the axis movement to the nearest obstacle face AHEAD of the entity
// that the entity actually overlaps on the two other axes. The previous
// version skipped that overlap test and treated every nearby solid block —
// including the ground underfoot — as a wall, so horizontal movement zeroed
// out the moment the player landed ("the stuck player" bug).
// ---------------------------------------------------------------------------
static Vec3 resolveAxis(const Vec3& position, const Vec3& entitySize, const Vec3& remainingMovement,
                        int axis, World& world, bool& wasPushed) {
    Vec3 newRemaining = remainingMovement;

    auto getAxis = [](const Vec3& v, int a) -> float {
        return (a == 0) ? v.x : (a == 1) ? v.y : v.z;
    };
    auto setAxis = [](Vec3& v, int a, float val) {
        if (a == 0)
            v.x = val;
        else if (a == 1)
            v.y = val;
        else
            v.z = val;
    };

    float moveAmount = getAxis(newRemaining, axis);

    // Early exit: no movement on this axis
    if (std::abs(moveAmount) < 1e-6f) {
        return newRemaining;
    }

    // Entity AABB at current position, and the volume the move sweeps through
    AABB entity{position, position + entitySize};
    AABB swept = entity;
    if (moveAmount > 0.f) {
        setAxis(swept.max, axis, getAxis(swept.max, axis) + moveAmount);
    } else {
        setAxis(swept.min, axis, getAxis(swept.min, axis) + moveAmount);
    }

    std::vector<AABB> obstacles = PhysicsEngine::collectObstacles(swept, world);

    const float sign = (moveAmount > 0.f) ? 1.f : -1.f;
    float allowed = std::abs(moveAmount);

    // Keep a hair of separation so surfaces never mathematically touch and
    // re-register as collisions on the next axis or tick
    constexpr float SKIN = 1e-4f;

    for (const auto& obstacle : obstacles) {
        // Only blocks the entity overlaps on the OTHER two axes can block
        // this axis — the floor underfoot must not stop a horizontal step.
        bool overlapsOthers = true;
        for (int other = 0; other < 3 && overlapsOthers; ++other) {
            if (other == axis) continue;
            if (getAxis(entity.max, other) <= getAxis(obstacle.min, other) + 1e-6f ||
                getAxis(entity.min, other) >= getAxis(obstacle.max, other) - 1e-6f) {
                overlapsOthers = false;
            }
        }
        if (!overlapsOthers) continue;

        // Gap from the entity's leading face to the obstacle's near face
        float gap = (sign > 0.f) ? getAxis(obstacle.min, axis) - getAxis(entity.max, axis)
                                 : getAxis(entity.min, axis) - getAxis(obstacle.max, axis);
        if (gap < -1e-6f) {
            // Obstacle is behind the leading face (or interpenetrating);
            // it cannot block this direction of travel
            continue;
        }

        float limit = std::max(gap - SKIN, 0.f);
        if (limit < allowed) {
            allowed = limit;
            wasPushed = true;
        }
    }

    setAxis(newRemaining, axis, sign * allowed);
    return newRemaining;
}

// ---------------------------------------------------------------------------
// sweepCollision — Main per-axis sweep collision entry point
// ---------------------------------------------------------------------------
Vec3 PhysicsEngine::sweepCollision(const AABB& entityAABB, const Vec3& movement, World& world) {
    Vec3 position = entityAABB.min;
    Vec3 entitySize = entityAABB.max - entityAABB.min;
    Vec3 remaining = movement;
    bool wasPushed = false;

    // Axis order: Y (1) first, then X/Z by magnitude (larger first)
    int firstAxis = 1; // Y
    int secondAxis = (std::abs(remaining.x) >= std::abs(remaining.z)) ? 0 : 2;
    int thirdAxis = (secondAxis == 0) ? 2 : 0;

    // Resolve Y axis
    remaining = resolveAxis(position, entitySize, remaining, firstAxis, world, wasPushed);
    position.y += remaining.y;

    // Resolve second horizontal axis
    remaining = resolveAxis(position, entitySize, remaining, secondAxis, world, wasPushed);
    if (secondAxis == 0)
        position.x += remaining.x;
    else
        position.z += remaining.z;

    // Resolve third horizontal axis
    remaining = resolveAxis(position, entitySize, remaining, thirdAxis, world, wasPushed);

    return remaining;
}

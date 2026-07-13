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
// isInWater — Check if any water block intersects the entity AABB
// ---------------------------------------------------------------------------
bool PhysicsEngine::isInWater(World& world, const AABB& entityAABB) {
    int minX = static_cast<int>(std::floor(entityAABB.min.x));
    int minY = static_cast<int>(std::floor(entityAABB.min.y));
    int minZ = static_cast<int>(std::floor(entityAABB.min.z));
    int maxX = static_cast<int>(std::ceil(entityAABB.max.x));
    int maxY = static_cast<int>(std::ceil(entityAABB.max.y));
    int maxZ = static_cast<int>(std::ceil(entityAABB.max.z));

    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            for (int z = minZ; z <= maxZ; ++z) {
                if (world.getBlock(x, y, z) == BlockType::WATER) {
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
// resolveAxis — Resolve collision along a single axis
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

    // Build entity AABB at current position
    AABB currentAABB{position, position + entitySize};

    // Build swept AABB: covers the range from current position to position+movement
    // on this axis (sweep volume, not translation)
    AABB sweptAABB = currentAABB;
    if (moveAmount > 0.f) {
        // Extend max in movement direction
        if (axis == 0)
            sweptAABB.max.x += moveAmount;
        else if (axis == 1)
            sweptAABB.max.y += moveAmount;
        else
            sweptAABB.max.z += moveAmount;
    } else if (moveAmount < 0.f) {
        // Extend min in movement direction
        if (axis == 0)
            sweptAABB.min.x += moveAmount;
        else if (axis == 1)
            sweptAABB.min.y += moveAmount;
        else
            sweptAABB.min.z += moveAmount;
    }

    // Collect obstacles intersecting the swept AABB
    std::vector<AABB> obstacles = PhysicsEngine::collectObstacles(sweptAABB, world);

    float sign = (moveAmount > 0.f) ? 1.f : -1.f;

    for (const auto& obstacle : obstacles) {
        float overlap = 0.f;

        if (sign > 0.f) {
            float entityMax = getAxis(sweptAABB.max, axis);
            float obstacleMin = getAxis(obstacle.min, axis);
            overlap = entityMax - obstacleMin;
        } else {
            float entityMin = getAxis(sweptAABB.min, axis);
            float obstacleMax = getAxis(obstacle.max, axis);
            overlap = obstacleMax - entityMin;
        }

        if (overlap > 0.f) {
            wasPushed = true;

            // Reduce remaining movement
            float reducedMove = moveAmount - overlap * sign;
            if (reducedMove * sign <= 1e-6f) {
                // Movement fully blocked
                setAxis(newRemaining, axis, 0.f);
                return newRemaining;
            }

            // Update remaining movement for this axis
            setAxis(newRemaining, axis, reducedMove);
            moveAmount = reducedMove;

            // Rebuild swept AABB with reduced movement for next obstacle
            sweptAABB = currentAABB;
            if (moveAmount > 0.f) {
                if (axis == 0)
                    sweptAABB.max.x += moveAmount;
                else if (axis == 1)
                    sweptAABB.max.y += moveAmount;
                else
                    sweptAABB.max.z += moveAmount;
            } else if (moveAmount < 0.f) {
                if (axis == 0)
                    sweptAABB.min.x += moveAmount;
                else if (axis == 1)
                    sweptAABB.min.y += moveAmount;
                else
                    sweptAABB.min.z += moveAmount;
            }
        }
    }

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

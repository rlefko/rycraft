#pragma once

#include <common/math.hpp>
#include <world/chunk.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// Forward declaration
class World;

// ---------------------------------------------------------------------------
// PhysicsEngine — Per-axis AABB sweep collision and world queries
//
// Implements the vanilla Minecraft collision algorithm: resolve movement
// one axis at a time (Y first, then X/Z by magnitude), expanding the
// entity AABB along each axis and pushing out of any solid block overlap.
// ---------------------------------------------------------------------------
class PhysicsEngine {
public:
    // Per-axis AABB sweep collision (vanilla Minecraft algorithm).
    // Y-axis is resolved first, then X/Z by movement magnitude (larger first).
    // Returns the resolved movement vector (may be reduced by collisions).
    Vec3 sweepCollision(const AABB& entityAABB, const Vec3& movement, World& world);

    // Collect all solid blocks that intersect an AABB.
    // Full cubes return a 1×1×1 AABB. Authored partial blocks retain their
    // exact collision height.
    static std::vector<AABB> collectObstacles(const AABB& expandedAABB, World& world);

    // Check if a block at world coordinates is solid.
    // Returns the shared gameplay solidity classification. Shape-aware callers
    // use collectObstacles for the exact collision volume.
    static bool isSolid(World& world, int64_t x, int32_t y, int64_t z);

    // Check if entity AABB overlaps any water block.
    static bool isInWater(World& world, const AABB& entityAABB);
};

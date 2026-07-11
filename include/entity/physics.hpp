#pragma once

#include <common/math.hpp>
#include <world/chunk.hpp>

#include <vector>
#include <optional>

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
    // Each returned AABB represents a 1×1×1 block at integer coordinates.
    static std::vector<AABB> collectObstacles(const AABB& expandedAABB, World& world);

    // Check if a block at world coordinates is solid.
    // Returns true for all BlockTypes except AIR, WATER, and GLASS.
    static bool isSolid(World& world, int x, int y, int z);

    // Check if entity AABB overlaps any water block.
    static bool isInWater(World& world, const AABB& entityAABB);
};

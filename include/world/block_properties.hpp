#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Block types and their properties — the single source of truth.
//
// Every subsystem answers "is this block solid/opaque/transparent?" from
// here. The engine previously carried four diverging isSolid definitions
// (meshing, physics, raycasting, spawning), which made glass render as a
// solid block that entities silently fell through.
// ---------------------------------------------------------------------------

enum class BlockType : uint8_t {
    AIR = 0,
    STONE = 1,
    GRASS = 2,
    DIRT = 3,
    SAND = 4,
    GRAVEL = 5,
    WATER = 6,
    BEDROCK = 7,
    LOG = 8,
    LEAVES = 9,
    SNOW = 10,
    COAL_ORE = 11,
    IRON_ORE = 12,
    GOLD_ORE = 13,
    DIAMOND_ORE = 14,
    PLANKS = 15,
    GLASS = 16,
    COUNT = 17
};

// Collision: blocks entities cannot pass through. Water is swimmable (see
// PhysicsEngine::isInWater); glass IS solid.
constexpr bool isSolid(BlockType type) {
    return type != BlockType::AIR && type != BlockType::WATER;
}

// Meshing/occlusion: blocks that produce visible faces and hide the faces of
// their neighbors. Matches the mesher's historical behavior (leaves and
// glass occlude; only air and water do not).
constexpr bool isOpaque(BlockType type) {
    return type != BlockType::AIR && type != BlockType::WATER;
}

// Light/visibility: blocks you can (partially) see through.
constexpr bool isTransparent(BlockType type) {
    return type == BlockType::AIR || type == BlockType::WATER ||
           type == BlockType::LEAVES || type == BlockType::GLASS;
}

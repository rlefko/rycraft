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
    // Values are persisted as raw bytes in saves: only append, never renumber.
    COBBLESTONE = 17,
    MOSSY_COBBLESTONE = 18,
    SANDSTONE = 19,
    BIRCH_LOG = 20,
    BIRCH_LEAVES = 21,
    SPRUCE_LOG = 22,
    SPRUCE_LEAVES = 23,
    CACTUS = 24,
    DEAD_BUSH = 25,
    TALL_GRASS = 26,
    FLOWER_YELLOW = 27,
    FLOWER_RED = 28,
    MUSHROOM_BROWN = 29,
    MUSHROOM_RED = 30,
    REED = 31,
    LAVA = 32,
    ICE = 33,
    COUNT = 34
};

// Cross-quad rendered, walk-through decoration blocks (plants). They occupy
// a cell but never collide, occlude, or cast column shadows.
constexpr bool isFlora(BlockType type) {
    switch (type) {
        case BlockType::DEAD_BUSH:
        case BlockType::TALL_GRASS:
        case BlockType::FLOWER_YELLOW:
        case BlockType::FLOWER_RED:
        case BlockType::MUSHROOM_BROWN:
        case BlockType::MUSHROOM_RED:
        case BlockType::REED:
            return true;
        default:
            return false;
    }
}

// Liquids are swimmable (see PhysicsEngine::isInLiquid) and render in the
// translucent pass, not the opaque chunk pass.
constexpr bool isLiquid(BlockType type) {
    return type == BlockType::WATER || type == BlockType::LAVA;
}

// Collision: blocks entities cannot pass through. Liquids are swimmable,
// flora is walk-through; glass and ice ARE solid.
constexpr bool isSolid(BlockType type) {
    return type != BlockType::AIR && !isLiquid(type) && !isFlora(type);
}

// Meshing/occlusion: blocks that fully hide their neighbors' faces. Leaf
// variants and glass are alpha-cutout textures (their transparent texels
// are discarded in the fragment shader), so the faces behind them must
// render; the same goes for non-cube flora and for liquids.
constexpr bool isOpaque(BlockType type) {
    switch (type) {
        case BlockType::AIR:
        case BlockType::WATER:
        case BlockType::LAVA:
        case BlockType::LEAVES:
        case BlockType::BIRCH_LEAVES:
        case BlockType::SPRUCE_LEAVES:
        case BlockType::GLASS:
            return false;
        default:
            return !isFlora(type);
    }
}

// Light/visibility: blocks you can (at least partially) see through.
constexpr bool isTransparent(BlockType type) {
    return !isOpaque(type);
}

// Interaction: blocks the crosshair raycast stops on. Everything you can
// stand on plus flora (breakable in place); liquids are click-through.
constexpr bool isTargetable(BlockType type) {
    return isSolid(type) || isFlora(type);
}

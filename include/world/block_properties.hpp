#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

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
    MUD = 34,
    CLAY = 35,
    SILT = 36,
    BASALT = 37,
    VOLCANIC_ASH = 38,
    LIMESTONE = 39,
    OBSIDIAN = 40,
    ACACIA_LOG = 41,
    ACACIA_LEAVES = 42,
    JUNGLE_LOG = 43,
    JUNGLE_LEAVES = 44,
    MANGROVE_LOG = 45,
    MANGROVE_LEAVES = 46,
    PALM_LOG = 47,
    PALM_LEAVES = 48,
    WILLOW_LOG = 49,
    WILLOW_LEAVES = 50,
    FERN = 51,
    SHRUB = 52,
    CATTAIL = 53,
    LILY_PAD = 54,
    FLOWER_BLUE = 55,
    SUCCULENT = 56,
    ANDESITE = 57,
    COUNT = 58
};

enum class BlockRenderShape : uint8_t {
    NONE,
    CUBE,
    CROSS,
    FLAT,
    LIQUID,
};

enum class BlockSound : uint8_t {
    UNDEFINED,
    NONE,
    STONE,
    SOIL,
    SAND,
    WOOD,
    PLANT,
    GLASS,
    WATER,
    LAVA,
    SNOW,
};

enum class BlockMaterial : uint8_t {
    UNDEFINED,
    AIR,
    ROCK,
    SOIL,
    GRANULAR,
    WOOD,
    LEAVES,
    PLANT,
    GLASS,
    WATER,
    LAVA,
    ICE,
};

struct BlockDefinition {
    BlockRenderShape renderShape = BlockRenderShape::CUBE;
    bool solid = true;
    bool opaque = true;
    bool targetable = true;
    bool liquid = false;
    bool leaf = false;
    BlockSound sound = BlockSound::UNDEFINED;
    BlockMaterial material = BlockMaterial::UNDEFINED;
    uint8_t lightEmission = 0;
    bool emissive = false;
    uint8_t sway = 0;
};

inline constexpr size_t BLOCK_TYPE_COUNT = static_cast<size_t>(BlockType::COUNT);

constexpr std::array<BlockDefinition, BLOCK_TYPE_COUNT> makeBlockDefinitions() {
    std::array<BlockDefinition, BLOCK_TYPE_COUNT> definitions{};
    definitions[static_cast<size_t>(BlockType::AIR)] = {
        BlockRenderShape::NONE, false, false, false, false, false, BlockSound::NONE,
        BlockMaterial::AIR};
    definitions[static_cast<size_t>(BlockType::WATER)] = {
        BlockRenderShape::LIQUID, false, false, false, true, false, BlockSound::WATER,
        BlockMaterial::WATER};
    definitions[static_cast<size_t>(BlockType::LAVA)] = {
        BlockRenderShape::CUBE, false, true, false, true, false, BlockSound::LAVA,
        BlockMaterial::LAVA};
    definitions[static_cast<size_t>(BlockType::LAVA)].lightEmission = 15;
    definitions[static_cast<size_t>(BlockType::LAVA)].emissive = true;
    definitions[static_cast<size_t>(BlockType::GLASS)] = {
        BlockRenderShape::CUBE, true, false, true, false, false, BlockSound::GLASS,
        BlockMaterial::GLASS};

    constexpr BlockType rocks[] = {
        BlockType::STONE,       BlockType::BEDROCK,           BlockType::COAL_ORE,
        BlockType::IRON_ORE,    BlockType::GOLD_ORE,          BlockType::DIAMOND_ORE,
        BlockType::COBBLESTONE, BlockType::MOSSY_COBBLESTONE, BlockType::SANDSTONE,
        BlockType::BASALT,      BlockType::LIMESTONE,         BlockType::OBSIDIAN,
        BlockType::ANDESITE,
    };
    for (BlockType block : rocks) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::STONE;
        definition.material = BlockMaterial::ROCK;
    }

    constexpr BlockType soils[] = {
        BlockType::GRASS, BlockType::DIRT, BlockType::MUD,
        BlockType::CLAY,  BlockType::SILT, BlockType::VOLCANIC_ASH,
    };
    for (BlockType block : soils) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::SOIL;
        definition.material = BlockMaterial::SOIL;
    }

    constexpr BlockType granular[] = {BlockType::SAND, BlockType::GRAVEL};
    for (BlockType block : granular) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::SAND;
        definition.material = BlockMaterial::GRANULAR;
    }

    constexpr BlockType wood[] = {
        BlockType::LOG,          BlockType::PLANKS,     BlockType::BIRCH_LOG,
        BlockType::SPRUCE_LOG,   BlockType::ACACIA_LOG, BlockType::JUNGLE_LOG,
        BlockType::MANGROVE_LOG, BlockType::PALM_LOG,   BlockType::WILLOW_LOG,
    };
    for (BlockType block : wood) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::WOOD;
        definition.material = BlockMaterial::WOOD;
    }

    constexpr BlockType leaves[] = {
        BlockType::LEAVES,        BlockType::BIRCH_LEAVES,  BlockType::SPRUCE_LEAVES,
        BlockType::ACACIA_LEAVES, BlockType::JUNGLE_LEAVES, BlockType::MANGROVE_LEAVES,
        BlockType::PALM_LEAVES,   BlockType::WILLOW_LEAVES,
    };
    for (BlockType block : leaves) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.opaque = false;
        definition.leaf = true;
        definition.sound = BlockSound::PLANT;
        definition.material = BlockMaterial::LEAVES;
        definition.sway = 2;
    }

    constexpr BlockType crossFlora[] = {
        BlockType::DEAD_BUSH,  BlockType::TALL_GRASS,     BlockType::FLOWER_YELLOW,
        BlockType::FLOWER_RED, BlockType::MUSHROOM_BROWN, BlockType::MUSHROOM_RED,
        BlockType::REED,       BlockType::FERN,           BlockType::SHRUB,
        BlockType::CATTAIL,    BlockType::FLOWER_BLUE,    BlockType::SUCCULENT,
    };
    for (BlockType block : crossFlora) {
        definitions[static_cast<size_t>(block)] = {
            BlockRenderShape::CROSS, false, false, true, false, false, BlockSound::PLANT,
            BlockMaterial::PLANT};
    }
    definitions[static_cast<size_t>(BlockType::LILY_PAD)] = {
        BlockRenderShape::FLAT, false, false, true, false, false, BlockSound::PLANT,
        BlockMaterial::PLANT};
    definitions[static_cast<size_t>(BlockType::CACTUS)].sound = BlockSound::PLANT;
    definitions[static_cast<size_t>(BlockType::CACTUS)].material = BlockMaterial::PLANT;
    definitions[static_cast<size_t>(BlockType::SNOW)].sound = BlockSound::SNOW;
    definitions[static_cast<size_t>(BlockType::SNOW)].material = BlockMaterial::ICE;
    definitions[static_cast<size_t>(BlockType::ICE)].sound = BlockSound::GLASS;
    definitions[static_cast<size_t>(BlockType::ICE)].material = BlockMaterial::ICE;

    constexpr BlockType rootBendingFlora[] = {
        BlockType::DEAD_BUSH,  BlockType::TALL_GRASS,  BlockType::FLOWER_YELLOW,
        BlockType::FLOWER_RED, BlockType::FERN,        BlockType::SHRUB,
        BlockType::CATTAIL,    BlockType::FLOWER_BLUE,
    };
    for (BlockType block : rootBendingFlora) {
        definitions[static_cast<size_t>(block)].sway = 1;
    }
    definitions[static_cast<size_t>(BlockType::REED)].sway = 2;
    return definitions;
}

inline constexpr auto BLOCK_DEFINITIONS = makeBlockDefinitions();
static_assert(BLOCK_DEFINITIONS.size() == BLOCK_TYPE_COUNT);
static_assert([] {
    for (const BlockDefinition& definition : BLOCK_DEFINITIONS) {
        if (definition.sound == BlockSound::UNDEFINED ||
            definition.material == BlockMaterial::UNDEFINED || definition.lightEmission > 15 ||
            definition.sway > 2 || definition.emissive != (definition.lightEmission > 0)) {
            return false;
        }
    }
    return true;
}());

constexpr const BlockDefinition& blockDefinition(BlockType type) {
    return BLOCK_DEFINITIONS[static_cast<size_t>(type)];
}

constexpr bool isFlora(BlockType type) {
    const auto shape = blockDefinition(type).renderShape;
    return shape == BlockRenderShape::CROSS || shape == BlockRenderShape::FLAT;
}

constexpr bool isLiquid(BlockType type) {
    return blockDefinition(type).liquid;
}

constexpr bool isLeafBlock(BlockType type) {
    return blockDefinition(type).leaf;
}

constexpr bool isSolid(BlockType type) {
    return blockDefinition(type).solid;
}

constexpr bool isOpaque(BlockType type) {
    return blockDefinition(type).opaque;
}

constexpr bool rendersAsCube(BlockType type) {
    return blockDefinition(type).renderShape == BlockRenderShape::CUBE;
}

constexpr bool isTransparent(BlockType type) {
    return !isOpaque(type);
}

constexpr bool isTargetable(BlockType type) {
    return blockDefinition(type).targetable;
}

// Block light: the level (0-15) a block emits into the world. Lava is the one
// source today; this is the single home the LightEngine and the mesher share.
// Light spreads through isTransparent cells losing one level per block.
constexpr uint8_t blockLightEmission(BlockType type) {
    return blockDefinition(type).lightEmission;
}

// Wind sway class the vertex shaders animate: 0 static, 1 flora (bends from
// the root, tip swings most), 2 leaves (whole canopy drifts). Mushrooms stay
// static — cave flora sits out of the wind.
constexpr uint8_t swayClass(BlockType type) {
    return blockDefinition(type).sway;
}

// Rendering: a self-lit block whose faces glow at a fixed HDR level regardless
// of sun, shadow, or skylight (and spill orange block light onto their
// surroundings). Derived from the emission so the two can never disagree.
constexpr bool isEmissive(BlockType type) {
    return blockDefinition(type).emissive;
}

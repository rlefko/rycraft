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
    CRAFTING_TABLE = 58,
    FURNACE = 59,
    FURNACE_LIT = 60,
    TORCH = 61,
    CHEST = 62,
    WOOL = 63,
    BED = 64,
    COUNT = 65
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

// Tool class a block prefers and the tier ladder for harvest gating. Survival
// mining speed and drop eligibility read these; Creative ignores them.
enum class ToolClass : uint8_t { NONE, PICKAXE, AXE, SHOVEL, SWORD };
enum class ToolTier : uint8_t { NONE = 0, WOOD = 1, STONE = 2, IRON = 3 };

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
    // Survival mining: seconds a bare hand needs; 0 breaks instantly and a
    // negative value never breaks (bedrock). Tools of `tool` class mine
    // faster, and drops require at least `minimumTier` of that class.
    float hardness = 0.0f;
    ToolClass tool = ToolClass::NONE;
    ToolTier minimumTier = ToolTier::NONE;
    bool interactable = false; // right-click opens a screen instead of placing
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
    definitions[static_cast<size_t>(BlockType::GLASS)].hardness = 0.3f;
    definitions[static_cast<size_t>(BlockType::CRAFTING_TABLE)] = {
        BlockRenderShape::CUBE, true, true, true, false, false, BlockSound::WOOD,
        BlockMaterial::WOOD};
    definitions[static_cast<size_t>(BlockType::CRAFTING_TABLE)].hardness = 2.5f;
    definitions[static_cast<size_t>(BlockType::CRAFTING_TABLE)].tool = ToolClass::AXE;
    definitions[static_cast<size_t>(BlockType::CRAFTING_TABLE)].interactable = true;
    for (BlockType furnace : {BlockType::FURNACE, BlockType::FURNACE_LIT}) {
        auto& definition = definitions[static_cast<size_t>(furnace)];
        definition = {BlockRenderShape::CUBE, true, true, true, false, false, BlockSound::STONE,
                      BlockMaterial::ROCK};
        definition.hardness = 3.5f;
        definition.tool = ToolClass::PICKAXE;
        definition.minimumTier = ToolTier::WOOD;
        definition.interactable = true;
    }
    definitions[static_cast<size_t>(BlockType::FURNACE_LIT)].lightEmission = 13;
    definitions[static_cast<size_t>(BlockType::FURNACE_LIT)].emissive = true;
    definitions[static_cast<size_t>(BlockType::TORCH)] = {
        BlockRenderShape::CROSS, false, false, true, false, false, BlockSound::WOOD,
        BlockMaterial::WOOD};
    definitions[static_cast<size_t>(BlockType::TORCH)].lightEmission = 14;
    definitions[static_cast<size_t>(BlockType::TORCH)].emissive = true;
    definitions[static_cast<size_t>(BlockType::CHEST)] = {
        BlockRenderShape::CUBE, true, true, true, false, false, BlockSound::WOOD,
        BlockMaterial::WOOD};
    definitions[static_cast<size_t>(BlockType::CHEST)].hardness = 2.5f;
    definitions[static_cast<size_t>(BlockType::CHEST)].tool = ToolClass::AXE;
    definitions[static_cast<size_t>(BlockType::CHEST)].interactable = true;
    definitions[static_cast<size_t>(BlockType::WOOL)] = {
        BlockRenderShape::CUBE, true, true, true, false, false, BlockSound::PLANT,
        BlockMaterial::LEAVES};
    definitions[static_cast<size_t>(BlockType::WOOL)].hardness = 0.8f;
    // A single-block bed (the cube format has no facing metadata for a two-part
    // bed): right-clicking it sleeps through the night and sets the spawn.
    definitions[static_cast<size_t>(BlockType::BED)] = {
        BlockRenderShape::CUBE, true, true, true, false, false, BlockSound::WOOD,
        BlockMaterial::WOOD};
    definitions[static_cast<size_t>(BlockType::BED)].hardness = 0.2f;

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
        definition.hardness = 1.5f;
        definition.tool = ToolClass::PICKAXE;
        definition.minimumTier = ToolTier::WOOD;
    }
    definitions[static_cast<size_t>(BlockType::BEDROCK)].hardness = -1.0f;
    definitions[static_cast<size_t>(BlockType::BEDROCK)].minimumTier = ToolTier::NONE;
    definitions[static_cast<size_t>(BlockType::COBBLESTONE)].hardness = 2.0f;
    definitions[static_cast<size_t>(BlockType::MOSSY_COBBLESTONE)].hardness = 2.0f;
    definitions[static_cast<size_t>(BlockType::SANDSTONE)].hardness = 0.8f;
    definitions[static_cast<size_t>(BlockType::BASALT)].hardness = 1.25f;
    definitions[static_cast<size_t>(BlockType::OBSIDIAN)].hardness = 50.0f;
    definitions[static_cast<size_t>(BlockType::OBSIDIAN)].minimumTier = ToolTier::IRON;
    constexpr BlockType ores[] = {
        BlockType::COAL_ORE,
        BlockType::IRON_ORE,
        BlockType::GOLD_ORE,
        BlockType::DIAMOND_ORE,
    };
    for (BlockType block : ores) {
        definitions[static_cast<size_t>(block)].hardness = 3.0f;
    }
    definitions[static_cast<size_t>(BlockType::IRON_ORE)].minimumTier = ToolTier::STONE;
    definitions[static_cast<size_t>(BlockType::GOLD_ORE)].minimumTier = ToolTier::IRON;
    definitions[static_cast<size_t>(BlockType::DIAMOND_ORE)].minimumTier = ToolTier::IRON;

    constexpr BlockType soils[] = {
        BlockType::GRASS, BlockType::DIRT, BlockType::MUD,
        BlockType::CLAY,  BlockType::SILT, BlockType::VOLCANIC_ASH,
    };
    for (BlockType block : soils) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::SOIL;
        definition.material = BlockMaterial::SOIL;
        definition.hardness = 0.5f;
        definition.tool = ToolClass::SHOVEL;
    }
    definitions[static_cast<size_t>(BlockType::GRASS)].hardness = 0.6f;
    definitions[static_cast<size_t>(BlockType::CLAY)].hardness = 0.6f;

    constexpr BlockType granular[] = {BlockType::SAND, BlockType::GRAVEL};
    for (BlockType block : granular) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::SAND;
        definition.material = BlockMaterial::GRANULAR;
        definition.hardness = 0.5f;
        definition.tool = ToolClass::SHOVEL;
    }
    definitions[static_cast<size_t>(BlockType::GRAVEL)].hardness = 0.6f;

    constexpr BlockType wood[] = {
        BlockType::LOG,          BlockType::PLANKS,     BlockType::BIRCH_LOG,
        BlockType::SPRUCE_LOG,   BlockType::ACACIA_LOG, BlockType::JUNGLE_LOG,
        BlockType::MANGROVE_LOG, BlockType::PALM_LOG,   BlockType::WILLOW_LOG,
    };
    for (BlockType block : wood) {
        auto& definition = definitions[static_cast<size_t>(block)];
        definition.sound = BlockSound::WOOD;
        definition.material = BlockMaterial::WOOD;
        definition.hardness = 2.0f;
        definition.tool = ToolClass::AXE;
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
        definition.hardness = 0.2f;
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
    definitions[static_cast<size_t>(BlockType::CACTUS)].hardness = 0.4f;
    definitions[static_cast<size_t>(BlockType::SNOW)].sound = BlockSound::SNOW;
    definitions[static_cast<size_t>(BlockType::SNOW)].material = BlockMaterial::ICE;
    definitions[static_cast<size_t>(BlockType::SNOW)].hardness = 0.2f;
    definitions[static_cast<size_t>(BlockType::SNOW)].tool = ToolClass::SHOVEL;
    definitions[static_cast<size_t>(BlockType::ICE)].sound = BlockSound::GLASS;
    definitions[static_cast<size_t>(BlockType::ICE)].material = BlockMaterial::ICE;
    definitions[static_cast<size_t>(BlockType::ICE)].hardness = 0.5f;
    definitions[static_cast<size_t>(BlockType::ICE)].tool = ToolClass::PICKAXE;

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
// Survival data invariants: bedrock is the only unbreakable block, every
// solid targetable block takes finite nonzero time, a tier requirement always
// names the tool class it gates, and only solid blocks open screens.
static_assert([] {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockDefinition& definition = BLOCK_DEFINITIONS[index];
        const bool isBedrock = static_cast<BlockType>(index) == BlockType::BEDROCK;
        if ((definition.hardness < 0.0f) != isBedrock) return false;
        if (definition.solid && definition.targetable && !isBedrock &&
            definition.hardness <= 0.0f) {
            return false;
        }
        if (definition.minimumTier != ToolTier::NONE && definition.tool == ToolClass::NONE) {
            return false;
        }
        if (definition.interactable && !definition.solid) return false;
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

// Survival mining: seconds of bare-hand work; 0 is instant, negative never
// breaks. The full break-time formula lives in world/item.hpp beside the
// tool speed table so the two cannot drift apart.
constexpr float blockHardness(BlockType type) {
    return blockDefinition(type).hardness;
}

// Right-clicking this block opens its screen (crafting table, furnace)
// instead of placing the held block.
constexpr bool isInteractable(BlockType type) {
    return blockDefinition(type).interactable;
}

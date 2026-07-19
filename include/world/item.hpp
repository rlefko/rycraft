#pragma once

#include "world/block_properties.hpp"

#include <array>
#include <cstdint>

// ---------------------------------------------------------------------------
// Items - the single source of truth for everything an inventory slot can
// hold: identifiers, stack rules, display names, swatch colors, tool and food
// data, per-block drops, and the mining-time formula.
//
// Ids 0..255 mirror BlockType exactly (BlockType is uint8_t, so the block
// range can never outgrow it); non-block items append from ITEM_ID_BASE in
// one contiguous range. Both ranges are append-only: ids persist as raw
// numbers in world metadata.
// ---------------------------------------------------------------------------

inline constexpr uint16_t ITEM_ID_BASE = 256;

enum class ItemType : uint16_t {
    NONE = 0, // == BlockType::AIR, the empty slot
    STICK = 256,
    COAL = 257,
    CHARCOAL = 258,
    IRON_INGOT = 259,
    GOLD_INGOT = 260,
    DIAMOND = 261,
    RAW_BEEF = 262,
    COOKED_BEEF = 263,
    RAW_PORKCHOP = 264,
    COOKED_PORKCHOP = 265,
    RAW_MUTTON = 266,
    COOKED_MUTTON = 267,
    RAW_CHICKEN = 268,
    COOKED_CHICKEN = 269,
    RAW_FISH = 270,
    COOKED_FISH = 271,
    WOODEN_PICKAXE = 272,
    WOODEN_AXE = 273,
    WOODEN_SHOVEL = 274,
    WOODEN_SWORD = 275,
    STONE_PICKAXE = 276,
    STONE_AXE = 277,
    STONE_SHOVEL = 278,
    STONE_SWORD = 279,
    IRON_PICKAXE = 280,
    IRON_AXE = 281,
    IRON_SHOVEL = 282,
    IRON_SWORD = 283,
    BUCKET = 284,
    WATER_BUCKET = 285,
    LAVA_BUCKET = 286,
    COUNT = 287
};

inline constexpr size_t NON_BLOCK_ITEM_COUNT = static_cast<size_t>(ItemType::COUNT) - ITEM_ID_BASE;

constexpr ItemType itemFromBlock(BlockType block) {
    return static_cast<ItemType>(static_cast<uint16_t>(block));
}

constexpr bool isBlockItem(ItemType type) {
    return static_cast<uint16_t>(type) < ITEM_ID_BASE;
}

// A defined id: a real block (0..BLOCK_TYPE_COUNT-1) or a non-block item
// (ITEM_ID_BASE..COUNT-1). The gap between the two ranges is reserved for
// future block ids, so a saved value there is not yet valid.
constexpr bool isValidItemId(uint16_t id) {
    return id < BLOCK_TYPE_COUNT ||
           (id >= ITEM_ID_BASE && id < static_cast<uint16_t>(ItemType::COUNT));
}

// AIR for non-block items and for block ids no block type defines.
constexpr BlockType blockFromItem(ItemType type) {
    const auto id = static_cast<uint16_t>(type);
    return id < BLOCK_TYPE_COUNT ? static_cast<BlockType>(id) : BlockType::AIR;
}

struct ItemStack {
    ItemType type = ItemType::NONE;
    uint8_t count = 0;
    uint16_t durability = 0; // remaining uses; nonzero only for tools

    constexpr bool empty() const { return type == ItemType::NONE || count == 0; }
    constexpr void clear() { *this = ItemStack{}; }
    constexpr bool operator==(const ItemStack&) const = default;
};

enum class ItemCategory : uint8_t { BLOCK, MATERIAL, FOOD, TOOL };

struct ItemDefinition {
    const char* name = "";
    ItemCategory category = ItemCategory::MATERIAL;
    uint8_t maxStack = 64;                 // tools stack to 1
    ToolClass toolClass = ToolClass::NONE; // tool/tier enums live beside the block table
    ToolTier toolTier = ToolTier::NONE;
    uint16_t maxDurability = 0; // uses before the tool breaks
    float miningSpeed = 1.0f;   // divisor in blockBreakTicks when the class matches
    uint8_t attackDamage = 1;   // half-hearts dealt to entities
    uint8_t foodValue = 0;      // hunger points restored on eating
};

// Display names for the block range. Charset is limited to [A-Za-z0-9 -] so
// the bitmap font's coverage test can enumerate every drawable string.
inline constexpr std::array<const char*, BLOCK_TYPE_COUNT> BLOCK_NAMES = {
    "Air",
    "Stone",
    "Grass",
    "Dirt",
    "Sand",
    "Gravel",
    "Water",
    "Bedrock",
    "Oak Log",
    "Oak Leaves",
    "Snow",
    "Coal Ore",
    "Iron Ore",
    "Gold Ore",
    "Diamond Ore",
    "Planks",
    "Glass",
    "Cobblestone",
    "Mossy Cobblestone",
    "Sandstone",
    "Birch Log",
    "Birch Leaves",
    "Spruce Log",
    "Spruce Leaves",
    "Cactus",
    "Dead Bush",
    "Tall Grass",
    "Yellow Flower",
    "Red Flower",
    "Brown Mushroom",
    "Red Mushroom",
    "Reed",
    "Lava",
    "Ice",
    "Mud",
    "Clay",
    "Silt",
    "Basalt",
    "Volcanic Ash",
    "Limestone",
    "Obsidian",
    "Acacia Log",
    "Acacia Leaves",
    "Jungle Log",
    "Jungle Leaves",
    "Mangrove Log",
    "Mangrove Leaves",
    "Palm Log",
    "Palm Leaves",
    "Willow Log",
    "Willow Leaves",
    "Fern",
    "Shrub",
    "Cattail",
    "Lily Pad",
    "Blue Flower",
    "Succulent",
    "Andesite",
    "Crafting Table",
    "Furnace",
    "Furnace",
    "Torch",
};
static_assert([] {
    for (const char* name : BLOCK_NAMES) {
        if (name == nullptr || name[0] == '\0') return false;
    }
    return true;
}());

namespace detail {

constexpr std::array<ItemDefinition, NON_BLOCK_ITEM_COUNT> makeItemDefinitions() {
    std::array<ItemDefinition, NON_BLOCK_ITEM_COUNT> items{};
    auto at = [&items](ItemType type) -> ItemDefinition& {
        return items[static_cast<uint16_t>(type) - ITEM_ID_BASE];
    };

    at(ItemType::STICK) = {"Stick", ItemCategory::MATERIAL};
    at(ItemType::COAL) = {"Coal", ItemCategory::MATERIAL};
    at(ItemType::CHARCOAL) = {"Charcoal", ItemCategory::MATERIAL};
    at(ItemType::IRON_INGOT) = {"Iron Ingot", ItemCategory::MATERIAL};
    at(ItemType::GOLD_INGOT) = {"Gold Ingot", ItemCategory::MATERIAL};
    at(ItemType::DIAMOND) = {"Diamond", ItemCategory::MATERIAL};

    // An empty bucket stacks to 16; a filled one is unstackable, exactly like
    // Minecraft, so a filled bucket can never merge and lose a fluid.
    at(ItemType::BUCKET) = {"Bucket", ItemCategory::MATERIAL, 16};
    at(ItemType::WATER_BUCKET) = {"Water Bucket", ItemCategory::MATERIAL, 1};
    at(ItemType::LAVA_BUCKET) = {"Lava Bucket", ItemCategory::MATERIAL, 1};

    struct Food {
        ItemType type;
        const char* name;
        uint8_t value;
    };
    constexpr Food foods[] = {
        {ItemType::RAW_BEEF, "Raw Beef", 3},
        {ItemType::COOKED_BEEF, "Cooked Beef", 8},
        {ItemType::RAW_PORKCHOP, "Raw Porkchop", 3},
        {ItemType::COOKED_PORKCHOP, "Cooked Porkchop", 8},
        {ItemType::RAW_MUTTON, "Raw Mutton", 2},
        {ItemType::COOKED_MUTTON, "Cooked Mutton", 6},
        {ItemType::RAW_CHICKEN, "Raw Chicken", 2},
        {ItemType::COOKED_CHICKEN, "Cooked Chicken", 6},
        {ItemType::RAW_FISH, "Raw Fish", 2},
        {ItemType::COOKED_FISH, "Cooked Fish", 5},
    };
    for (const Food& food : foods) {
        at(food.type) = {food.name, ItemCategory::FOOD};
        at(food.type).foodValue = food.value;
    }

    struct Tier {
        ItemType firstTool; // PICKAXE, then AXE, SHOVEL, SWORD follow
        ToolTier tier;
        uint16_t durability;
        float speed;
        uint8_t baseDamage; // pickaxe and shovel; axe +1, sword +2
    };
    constexpr Tier tiers[] = {
        {ItemType::WOODEN_PICKAXE, ToolTier::WOOD, 59, 2.0f, 2},
        {ItemType::STONE_PICKAXE, ToolTier::STONE, 131, 4.0f, 3},
        {ItemType::IRON_PICKAXE, ToolTier::IRON, 250, 6.0f, 4},
    };
    struct Kind {
        int offset;
        ToolClass toolClass;
        uint8_t damageBonus;
    };
    constexpr Kind kinds[] = {
        {0, ToolClass::PICKAXE, 0},
        {1, ToolClass::AXE, 1},
        {2, ToolClass::SHOVEL, 0},
        {3, ToolClass::SWORD, 2},
    };
    // The table stores string literals, so names are spelled out per tier and
    // kind rather than concatenated; the order matches kinds[].
    constexpr const char* toolNames[3][4] = {
        {"Wooden Pickaxe", "Wooden Axe", "Wooden Shovel", "Wooden Sword"},
        {"Stone Pickaxe", "Stone Axe", "Stone Shovel", "Stone Sword"},
        {"Iron Pickaxe", "Iron Axe", "Iron Shovel", "Iron Sword"},
    };
    for (size_t tierIndex = 0; tierIndex < 3; ++tierIndex) {
        const Tier& tier = tiers[tierIndex];
        for (const Kind& kind : kinds) {
            auto type = static_cast<ItemType>(static_cast<uint16_t>(tier.firstTool) + kind.offset);
            ItemDefinition& item = at(type);
            item = {toolNames[tierIndex][kind.offset], ItemCategory::TOOL, 1, kind.toolClass,
                    tier.tier};
            item.maxDurability = tier.durability;
            item.miningSpeed = tier.speed;
            item.attackDamage = static_cast<uint8_t>(tier.baseDamage + kind.damageBonus);
        }
    }
    return items;
}

inline constexpr auto ITEM_DEFINITIONS = makeItemDefinitions();
static_assert([] {
    for (const ItemDefinition& item : ITEM_DEFINITIONS) {
        if (item.name == nullptr || item.name[0] == '\0') return false;
        if (item.maxStack == 0) return false;
        if ((item.category == ItemCategory::TOOL) != (item.maxDurability > 0)) return false;
        if (item.category == ItemCategory::TOOL && item.maxStack != 1) return false;
        if ((item.category == ItemCategory::FOOD) != (item.foodValue > 0)) return false;
    }
    return true;
}());

} // namespace detail

// Block items synthesize their definition from the block table; the dense
// array above holds only the non-block range.
constexpr ItemDefinition itemDefinition(ItemType type) {
    if (isBlockItem(type)) {
        ItemDefinition definition{};
        const auto id = static_cast<uint16_t>(type);
        definition.name = id < BLOCK_TYPE_COUNT ? BLOCK_NAMES[id] : "";
        definition.category = ItemCategory::BLOCK;
        return definition;
    }
    return detail::ITEM_DEFINITIONS[static_cast<uint16_t>(type) - ITEM_ID_BASE];
}

constexpr const char* itemName(ItemType type) {
    return itemDefinition(type).name;
}

constexpr int maxStackSize(ItemType type) {
    return itemDefinition(type).maxStack;
}

constexpr bool isTool(ItemType type) {
    return itemDefinition(type).category == ItemCategory::TOOL;
}

constexpr bool isFood(ItemType type) {
    return itemDefinition(type).category == ItemCategory::FOOD;
}

// A tool fresh from the crafting grid carries its full durability.
constexpr ItemStack makeItemStack(ItemType type, uint8_t count = 1) {
    return ItemStack{type, count, itemDefinition(type).maxDurability};
}

// ---------------------------------------------------------------------------
// Swatch colors - 0xRRGGBB per item, used for dropped-item cubes and any
// untextured slot fallback. The block range keeps the palette the hotbar
// previously spread across a switch in ui_hud.mm.
// ---------------------------------------------------------------------------

namespace detail {

constexpr std::array<uint32_t, BLOCK_TYPE_COUNT> makeBlockSwatches() {
    std::array<uint32_t, BLOCK_TYPE_COUNT> colors{};
    auto at = [&colors](BlockType block) -> uint32_t& {
        return colors[static_cast<size_t>(block)];
    };
    for (uint32_t& color : colors) {
        color = 0x808080;
    }
    at(BlockType::AIR) = 0x000000;
    at(BlockType::STONE) = 0x808080;
    at(BlockType::GRASS) = 0x339933;
    at(BlockType::DIRT) = 0x8C5933;
    at(BlockType::SAND) = 0xD9C78C;
    at(BlockType::GRAVEL) = 0x8C8C8C;
    at(BlockType::WATER) = 0x4073D9;
    at(BlockType::BEDROCK) = 0x333333;
    at(BlockType::LOG) = 0x664026;
    at(BlockType::LEAVES) = 0x2E5926;
    at(BlockType::SNOW) = 0xF0F5FA;
    at(BlockType::COAL_ORE) = 0x262626;
    at(BlockType::IRON_ORE) = 0x998073;
    at(BlockType::GOLD_ORE) = 0xB09A45;
    at(BlockType::DIAMOND_ORE) = 0x7FB8B8;
    at(BlockType::PLANKS) = 0xA67340;
    at(BlockType::GLASS) = 0xD9E6F2;
    at(BlockType::COBBLESTONE) = 0x737378;
    at(BlockType::MOSSY_COBBLESTONE) = 0x59734D;
    at(BlockType::SANDSTONE) = 0xCCB880;
    at(BlockType::BIRCH_LOG) = 0xD9D4BF;
    at(BlockType::BIRCH_LEAVES) = 0x598C40;
    at(BlockType::SPRUCE_LOG) = 0x4D331A;
    at(BlockType::SPRUCE_LEAVES) = 0x1F592E;
    at(BlockType::CACTUS) = 0x337326;
    at(BlockType::DEAD_BUSH) = 0x8C6638;
    at(BlockType::TALL_GRASS) = 0x59A640;
    at(BlockType::FLOWER_YELLOW) = 0xE6D933;
    at(BlockType::FLOWER_RED) = 0xD93333;
    at(BlockType::MUSHROOM_BROWN) = 0x8C664D;
    at(BlockType::MUSHROOM_RED) = 0xCC2626;
    at(BlockType::REED) = 0x80BF59;
    at(BlockType::LAVA) = 0xE6661A;
    at(BlockType::ICE) = 0xB3D9F2;
    at(BlockType::MUD) = 0x4D3B2B;
    at(BlockType::CLAY) = 0x8F9499;
    at(BlockType::SILT) = 0x806E4F;
    at(BlockType::BASALT) = 0x333638;
    at(BlockType::VOLCANIC_ASH) = 0x454240;
    at(BlockType::LIMESTONE) = 0xB8B39E;
    at(BlockType::OBSIDIAN) = 0x1F142B;
    at(BlockType::ACACIA_LOG) = 0x6E4A32;
    at(BlockType::ACACIA_LEAVES) = 0x4A7331;
    at(BlockType::JUNGLE_LOG) = 0x59422B;
    at(BlockType::JUNGLE_LEAVES) = 0x2E6626;
    at(BlockType::MANGROVE_LOG) = 0x59332B;
    at(BlockType::MANGROVE_LEAVES) = 0x2E5C33;
    at(BlockType::PALM_LOG) = 0x73593B;
    at(BlockType::PALM_LEAVES) = 0x40732E;
    at(BlockType::WILLOW_LOG) = 0x4D3D26;
    at(BlockType::WILLOW_LEAVES) = 0x4D6E3A;
    at(BlockType::FERN) = 0x3A6629;
    at(BlockType::SHRUB) = 0x476B2F;
    at(BlockType::CATTAIL) = 0x6B7331;
    at(BlockType::LILY_PAD) = 0x2E6629;
    at(BlockType::FLOWER_BLUE) = 0x4059CC;
    at(BlockType::SUCCULENT) = 0x5C8547;
    at(BlockType::ANDESITE) = 0x6B6E6B;
    at(BlockType::CRAFTING_TABLE) = 0x99734D;
    at(BlockType::FURNACE) = 0x6B6B70;
    at(BlockType::FURNACE_LIT) = 0x8C6247;
    at(BlockType::TORCH) = 0xFFD966;
    return colors;
}

inline constexpr auto BLOCK_SWATCHES = makeBlockSwatches();

constexpr std::array<uint32_t, NON_BLOCK_ITEM_COUNT> makeItemSwatches() {
    std::array<uint32_t, NON_BLOCK_ITEM_COUNT> colors{};
    auto at = [&colors](ItemType type) -> uint32_t& {
        return colors[static_cast<uint16_t>(type) - ITEM_ID_BASE];
    };
    at(ItemType::STICK) = 0x735933;
    at(ItemType::COAL) = 0x262626;
    at(ItemType::CHARCOAL) = 0x33291F;
    at(ItemType::IRON_INGOT) = 0xD9D0C7;
    at(ItemType::GOLD_INGOT) = 0xF2D93B;
    at(ItemType::DIAMOND) = 0x66E6E6;
    at(ItemType::RAW_BEEF) = 0xB33B33;
    at(ItemType::COOKED_BEEF) = 0x8C4A26;
    at(ItemType::RAW_PORKCHOP) = 0xE68C8C;
    at(ItemType::COOKED_PORKCHOP) = 0xBF8033;
    at(ItemType::RAW_MUTTON) = 0xCC5247;
    at(ItemType::COOKED_MUTTON) = 0x995233;
    at(ItemType::RAW_CHICKEN) = 0xE6B8A6;
    at(ItemType::COOKED_CHICKEN) = 0xC7853B;
    at(ItemType::RAW_FISH) = 0x9FB8C7;
    at(ItemType::COOKED_FISH) = 0xB8926B;
    for (auto tool : {ItemType::WOODEN_PICKAXE, ItemType::WOODEN_AXE, ItemType::WOODEN_SHOVEL,
                      ItemType::WOODEN_SWORD}) {
        at(tool) = 0x996B33;
    }
    for (auto tool : {ItemType::STONE_PICKAXE, ItemType::STONE_AXE, ItemType::STONE_SHOVEL,
                      ItemType::STONE_SWORD}) {
        at(tool) = 0x8C8C8C;
    }
    for (auto tool : {ItemType::IRON_PICKAXE, ItemType::IRON_AXE, ItemType::IRON_SHOVEL,
                      ItemType::IRON_SWORD}) {
        at(tool) = 0xD9D9D9;
    }
    at(ItemType::BUCKET) = 0xB0B4BA;
    at(ItemType::WATER_BUCKET) = 0x4073D9;
    at(ItemType::LAVA_BUCKET) = 0xE6661A;
    return colors;
}

inline constexpr auto ITEM_SWATCHES = makeItemSwatches();

} // namespace detail

constexpr uint32_t itemSwatchColor(ItemType type) {
    const auto id = static_cast<uint16_t>(type);
    if (isBlockItem(type)) {
        return id < BLOCK_TYPE_COUNT ? detail::BLOCK_SWATCHES[id] : 0x808080;
    }
    return detail::ITEM_SWATCHES[id - ITEM_ID_BASE];
}

// ---------------------------------------------------------------------------
// Block drops - what survival mining yields once the tier gate passes.
// ---------------------------------------------------------------------------

struct BlockDrop {
    ItemType item = ItemType::NONE;
    uint8_t count = 0; // 0 drops nothing
};

namespace detail {

constexpr std::array<BlockDrop, BLOCK_TYPE_COUNT> makeBlockDrops() {
    std::array<BlockDrop, BLOCK_TYPE_COUNT> drops{};
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        drops[index] = {itemFromBlock(static_cast<BlockType>(index)), 1};
    }
    auto at = [&drops](BlockType block) -> BlockDrop& { return drops[static_cast<size_t>(block)]; };
    // Blocks that vanish instead of dropping: unbreakable, liquid, fragile,
    // and decorative flora without an item form.
    for (BlockType block :
         {BlockType::AIR, BlockType::WATER, BlockType::LAVA, BlockType::BEDROCK, BlockType::GLASS,
          BlockType::ICE, BlockType::TALL_GRASS, BlockType::DEAD_BUSH, BlockType::LEAVES,
          BlockType::BIRCH_LEAVES, BlockType::SPRUCE_LEAVES, BlockType::ACACIA_LEAVES,
          BlockType::JUNGLE_LEAVES, BlockType::MANGROVE_LEAVES, BlockType::PALM_LEAVES,
          BlockType::WILLOW_LEAVES}) {
        at(block) = {};
    }
    at(BlockType::STONE) = {itemFromBlock(BlockType::COBBLESTONE), 1};
    at(BlockType::GRASS) = {itemFromBlock(BlockType::DIRT), 1};
    at(BlockType::COAL_ORE) = {ItemType::COAL, 1};
    at(BlockType::DIAMOND_ORE) = {ItemType::DIAMOND, 1};
    at(BlockType::FURNACE_LIT) = {itemFromBlock(BlockType::FURNACE), 1};
    return drops;
}

inline constexpr auto BLOCK_DROPS = makeBlockDrops();
static_assert([] {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockDrop& drop = BLOCK_DROPS[index];
        if (drop.count > 0 && drop.item == ItemType::NONE) return false;
        const auto& definition = BLOCK_DEFINITIONS[index];
        if ((definition.liquid || definition.hardness < 0.0f) && drop.count != 0) return false;
    }
    return true;
}());

} // namespace detail

constexpr BlockDrop blockDrop(BlockType type) {
    return detail::BLOCK_DROPS[static_cast<size_t>(type)];
}

// ---------------------------------------------------------------------------
// Mining times - the one formula survival breaking uses. Seconds of work are
// hardness * 1.5 when the held tool can harvest the block and hardness * 5
// when it cannot; a matching tool class divides by its mining speed.
// ---------------------------------------------------------------------------

inline constexpr int UNBREAKABLE_BREAK_TICKS = 0x7FFFFFFF;
inline constexpr int TICKS_PER_SECOND = 20;

constexpr bool toolCanHarvest(BlockType block, ItemType held) {
    const BlockDefinition& definition = blockDefinition(block);
    if (definition.minimumTier == ToolTier::NONE) return true;
    const ItemDefinition item = itemDefinition(held);
    return item.toolClass == definition.tool && item.toolTier >= definition.minimumTier;
}

constexpr int blockBreakTicks(BlockType block, ItemType held) {
    const BlockDefinition& definition = blockDefinition(block);
    if (definition.hardness < 0.0f) return UNBREAKABLE_BREAK_TICKS;
    if (definition.hardness == 0.0f) return 0;
    const ItemDefinition item = itemDefinition(held);
    float seconds = definition.hardness * (toolCanHarvest(block, held) ? 1.5f : 5.0f);
    if (definition.tool != ToolClass::NONE && item.toolClass == definition.tool) {
        seconds /= item.miningSpeed;
    }
    const float ticks = seconds * TICKS_PER_SECOND;
    const int whole = static_cast<int>(ticks);
    return ticks > static_cast<float>(whole) ? whole + 1 : whole;
}

// ---------------------------------------------------------------------------
// Creative palette - every obtainable item in slot order: placeable blocks
// first (skipping the unobtainable lit furnace), then materials, foods, and
// tools in id order.
// ---------------------------------------------------------------------------

namespace detail {

inline constexpr size_t CREATIVE_PALETTE_SIZE = (BLOCK_TYPE_COUNT - 2) + NON_BLOCK_ITEM_COUNT;

constexpr std::array<ItemType, CREATIVE_PALETTE_SIZE> makeCreativePalette() {
    std::array<ItemType, CREATIVE_PALETTE_SIZE> palette{};
    size_t next = 0;
    for (size_t index = 1; index < BLOCK_TYPE_COUNT; ++index) {
        const auto block = static_cast<BlockType>(index);
        if (block == BlockType::FURNACE_LIT) continue;
        palette[next++] = itemFromBlock(block);
    }
    for (size_t index = 0; index < NON_BLOCK_ITEM_COUNT; ++index) {
        palette[next++] = static_cast<ItemType>(ITEM_ID_BASE + index);
    }
    return palette;
}

} // namespace detail

inline constexpr auto CREATIVE_PALETTE = detail::makeCreativePalette();

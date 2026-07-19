#include <catch2/catch_test_macros.hpp>

#include "world/item.hpp"

TEST_CASE("Item ids mirror blocks below the non-block range", "[item]") {
    REQUIRE(static_cast<uint16_t>(ItemType::NONE) == 0);
    REQUIRE(itemFromBlock(BlockType::AIR) == ItemType::NONE);
    REQUIRE(itemFromBlock(BlockType::STONE) == static_cast<ItemType>(1));
    REQUIRE(blockFromItem(itemFromBlock(BlockType::ANDESITE)) == BlockType::ANDESITE);
    REQUIRE(blockFromItem(ItemType::STICK) == BlockType::AIR);
    REQUIRE(blockFromItem(static_cast<ItemType>(200)) == BlockType::AIR);
    REQUIRE(isBlockItem(itemFromBlock(BlockType::TORCH)));
    REQUIRE_FALSE(isBlockItem(ItemType::STICK));
    REQUIRE(static_cast<uint16_t>(ItemType::STICK) == ITEM_ID_BASE);
    // The non-block range stays contiguous so icon layers can be an offset.
    REQUIRE(static_cast<size_t>(ItemType::COUNT) == ITEM_ID_BASE + NON_BLOCK_ITEM_COUNT);
}

TEST_CASE("Item definitions cover names stacks tools and foods", "[item]") {
    for (size_t index = 0; index < NON_BLOCK_ITEM_COUNT; ++index) {
        const auto type = static_cast<ItemType>(ITEM_ID_BASE + index);
        const ItemDefinition definition = itemDefinition(type);
        REQUIRE(definition.name[0] != '\0');
        REQUIRE(definition.maxStack >= 1);
        if (definition.category == ItemCategory::TOOL) {
            REQUIRE(definition.maxStack == 1);
            REQUIRE(definition.maxDurability > 0);
            REQUIRE(definition.toolClass != ToolClass::NONE);
            REQUIRE(definition.toolTier != ToolTier::NONE);
            REQUIRE(definition.miningSpeed > 1.0f);
        } else {
            REQUIRE(definition.maxDurability == 0);
        }
        if (definition.category == ItemCategory::FOOD) {
            REQUIRE(definition.foodValue > 0);
        }
        // Display names stay inside the bitmap-font charset.
        for (const char* c = definition.name; *c != '\0'; ++c) {
            const bool covered = (*c >= 'A' && *c <= 'Z') || (*c >= 'a' && *c <= 'z') ||
                                 (*c >= '0' && *c <= '9') || *c == ' ' || *c == '-';
            REQUIRE(covered);
        }
    }

    REQUIRE(maxStackSize(itemFromBlock(BlockType::STONE)) == 64);
    REQUIRE(maxStackSize(ItemType::IRON_PICKAXE) == 1);
    // Empty buckets stack; filled buckets never do, so a fluid cannot merge away.
    REQUIRE(maxStackSize(ItemType::BUCKET) == 16);
    REQUIRE(maxStackSize(ItemType::WATER_BUCKET) == 1);
    REQUIRE(maxStackSize(ItemType::LAVA_BUCKET) == 1);
    // Shears and beds are unstackable, exactly like Minecraft.
    REQUIRE(maxStackSize(ItemType::SHEARS) == 1);
    REQUIRE(maxStackSize(itemFromBlock(BlockType::BED)) == 1);
    REQUIRE(maxStackSize(itemFromBlock(BlockType::WOOL)) == 64);
    REQUIRE(itemDefinition(ItemType::IRON_SWORD).attackDamage == 6);
    REQUIRE(itemDefinition(ItemType::WOODEN_SWORD).attackDamage == 4);
    REQUIRE(itemDefinition(ItemType::COOKED_BEEF).foodValue == 8);
    REQUIRE(isFood(ItemType::RAW_FISH));
    REQUIRE_FALSE(isFood(ItemType::COAL));
    REQUIRE(isTool(ItemType::STONE_SHOVEL));
    REQUIRE(std::string(itemName(itemFromBlock(BlockType::CRAFTING_TABLE))) == "Crafting Table");

    const ItemStack fresh = makeItemStack(ItemType::WOODEN_PICKAXE);
    REQUIRE(fresh.durability == 59);
    REQUIRE(makeItemStack(ItemType::STICK, 4).durability == 0);
}

TEST_CASE("Every item has a nonzero swatch color", "[item]") {
    for (size_t index = 1; index < BLOCK_TYPE_COUNT; ++index) {
        REQUIRE(itemSwatchColor(itemFromBlock(static_cast<BlockType>(index))) != 0);
    }
    for (size_t index = 0; index < NON_BLOCK_ITEM_COUNT; ++index) {
        REQUIRE(itemSwatchColor(static_cast<ItemType>(ITEM_ID_BASE + index)) != 0);
    }
    REQUIRE(itemSwatchColor(itemFromBlock(BlockType::STONE)) == 0x808080);
}

TEST_CASE("Block drops follow survival rules", "[item][survival]") {
    REQUIRE(blockDrop(BlockType::STONE).item == itemFromBlock(BlockType::COBBLESTONE));
    REQUIRE(blockDrop(BlockType::GRASS).item == itemFromBlock(BlockType::DIRT));
    REQUIRE(blockDrop(BlockType::COAL_ORE).item == ItemType::COAL);
    REQUIRE(blockDrop(BlockType::DIAMOND_ORE).item == ItemType::DIAMOND);
    // Iron and gold ore drop themselves and smelt into ingots later.
    REQUIRE(blockDrop(BlockType::IRON_ORE).item == itemFromBlock(BlockType::IRON_ORE));
    REQUIRE(blockDrop(BlockType::GOLD_ORE).item == itemFromBlock(BlockType::GOLD_ORE));
    REQUIRE(blockDrop(BlockType::FURNACE_LIT).item == itemFromBlock(BlockType::FURNACE));
    REQUIRE(blockDrop(BlockType::DIRT).item == itemFromBlock(BlockType::DIRT));

    for (BlockType none : {BlockType::AIR, BlockType::WATER, BlockType::LAVA, BlockType::BEDROCK,
                           BlockType::GLASS, BlockType::ICE, BlockType::LEAVES,
                           BlockType::WILLOW_LEAVES, BlockType::TALL_GRASS, BlockType::DEAD_BUSH}) {
        REQUIRE(blockDrop(none).count == 0);
    }
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockDrop drop = blockDrop(static_cast<BlockType>(index));
        if (drop.count > 0) {
            REQUIRE(drop.item != ItemType::NONE);
        }
    }
}

TEST_CASE("Break ticks scale with hardness tool class and tier", "[item][survival]") {
    // Stone: 1.5 hardness. Bare hand cannot harvest: 1.5 * 5 = 7.5 s.
    REQUIRE(blockBreakTicks(BlockType::STONE, ItemType::NONE) == 150);
    // Wooden pickaxe harvests: 1.5 * 1.5 / 2 = 1.125 s.
    REQUIRE(blockBreakTicks(BlockType::STONE, ItemType::WOODEN_PICKAXE) == 23);
    // Iron pickaxe: 1.5 * 1.5 / 6 = 0.375 s.
    REQUIRE(blockBreakTicks(BlockType::STONE, ItemType::IRON_PICKAXE) == 8);
    // The wrong tool class neither speeds up nor harvests stone.
    REQUIRE(blockBreakTicks(BlockType::STONE, ItemType::IRON_AXE) == 150);
    // Iron ore requires stone tier: a wooden pickaxe mines slowly and fruitlessly.
    REQUIRE_FALSE(toolCanHarvest(BlockType::IRON_ORE, ItemType::WOODEN_PICKAXE));
    REQUIRE(toolCanHarvest(BlockType::IRON_ORE, ItemType::STONE_PICKAXE));
    REQUIRE(blockBreakTicks(BlockType::IRON_ORE, ItemType::WOODEN_PICKAXE) >
            blockBreakTicks(BlockType::IRON_ORE, ItemType::STONE_PICKAXE));
    // Dirt needs no tool: every held item harvests it.
    REQUIRE(toolCanHarvest(BlockType::DIRT, ItemType::NONE));
    // Dirt: 0.5 * 1.5 = 0.75 s by hand, halved twice by an iron shovel.
    REQUIRE(blockBreakTicks(BlockType::DIRT, ItemType::NONE) == 15);
    REQUIRE(blockBreakTicks(BlockType::DIRT, ItemType::IRON_SHOVEL) == 3);
    // Flora is instant, bedrock is never.
    REQUIRE(blockBreakTicks(BlockType::TALL_GRASS, ItemType::NONE) == 0);
    REQUIRE(blockBreakTicks(BlockType::TORCH, ItemType::NONE) == 0);
    REQUIRE(blockBreakTicks(BlockType::BEDROCK, ItemType::IRON_PICKAXE) == UNBREAKABLE_BREAK_TICKS);
}

TEST_CASE("Creative palette lists every obtainable item once", "[item]") {
    REQUIRE(CREATIVE_PALETTE.size() == (BLOCK_TYPE_COUNT - 2) + NON_BLOCK_ITEM_COUNT);
    for (size_t left = 0; left < CREATIVE_PALETTE.size(); ++left) {
        REQUIRE(CREATIVE_PALETTE[left] != ItemType::NONE);
        REQUIRE(CREATIVE_PALETTE[left] != itemFromBlock(BlockType::FURNACE_LIT));
        for (size_t right = left + 1; right < CREATIVE_PALETTE.size(); ++right) {
            REQUIRE(CREATIVE_PALETTE[left] != CREATIVE_PALETTE[right]);
        }
    }
}

TEST_CASE("Animal drops are valid food items", "[item][entity]") {
    // Included via entity config coverage in test_entity.mm; the item side
    // asserts the raw foods smelt targets exist in the id space.
    for (ItemType raw : {ItemType::RAW_BEEF, ItemType::RAW_PORKCHOP, ItemType::RAW_MUTTON,
                         ItemType::RAW_CHICKEN, ItemType::RAW_FISH}) {
        REQUIRE(isFood(raw));
    }
}

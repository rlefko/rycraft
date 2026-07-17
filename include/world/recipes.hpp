#pragma once

#include "world/item.hpp"

#include <optional>
#include <span>

// ---------------------------------------------------------------------------
// Recipes — the single source of truth for crafting, smelting, and fuel.
//
// Crafting grids are row-major ItemStack spans with gridWidth 2 (inventory)
// or 3 (crafting table); NONE marks an empty cell. Matching runs on grid
// changes, never per tick. Shaped patterns match at any offset and in
// horizontal mirror; interchangeable species fold through craftingKey first.
// ---------------------------------------------------------------------------

inline constexpr int FURNACE_COOK_TICKS = 200; // 10 s per item at 20 Hz

// All nine log species craft alike, and charcoal substitutes for coal.
constexpr ItemType craftingKey(ItemType type) {
    if (isBlockItem(type)) {
        switch (blockFromItem(type)) {
            case BlockType::BIRCH_LOG:
            case BlockType::SPRUCE_LOG:
            case BlockType::ACACIA_LOG:
            case BlockType::JUNGLE_LOG:
            case BlockType::MANGROVE_LOG:
            case BlockType::PALM_LOG:
            case BlockType::WILLOW_LOG:
                return itemFromBlock(BlockType::LOG);
            default:
                return type;
        }
    }
    return type == ItemType::CHARCOAL ? ItemType::COAL : type;
}

// The crafted result for the grid contents, or nullopt when nothing matches.
// Tool results carry full durability.
std::optional<ItemStack> matchCraftingRecipe(std::span<const ItemStack> grid, int gridWidth);

// One craft consumes one item from every occupied cell.
void consumeOneCraft(std::span<ItemStack> grid);

std::optional<ItemType> smeltingResult(ItemType input);
bool isSmeltable(ItemType input);

// Ticks of burn time one unit of this fuel provides; 0 means not a fuel.
int fuelBurnTicks(ItemType fuel);
bool isFurnaceFuel(ItemType fuel);

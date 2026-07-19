#include "world/recipes.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace {

// Shaped patterns are row-major width x height in craftingKey space; a 0x0
// shape marks a shapeless recipe whose first ingredientCount pattern entries
// are the required multiset.
struct CraftingRecipe {
    uint8_t width = 0;
    uint8_t height = 0;
    std::array<ItemType, 9> pattern{};
    uint8_t ingredientCount = 0;
    ItemStack result{};
};

constexpr ItemType LOG = itemFromBlock(BlockType::LOG);
constexpr ItemType PLANKS = itemFromBlock(BlockType::PLANKS);
constexpr ItemType COBBLESTONE = itemFromBlock(BlockType::COBBLESTONE);
constexpr ItemType NONE = ItemType::NONE;

constexpr CraftingRecipe shapeless(ItemType ingredient, ItemStack result) {
    CraftingRecipe recipe{};
    recipe.pattern[0] = ingredient;
    recipe.ingredientCount = 1;
    recipe.result = result;
    return recipe;
}

constexpr CraftingRecipe shaped(uint8_t width, uint8_t height, std::array<ItemType, 9> pattern,
                                ItemStack result) {
    return CraftingRecipe{width, height, pattern, 0, result};
}

// Tool rows share one shape per class with the material substituted per tier.
struct ToolShape {
    uint8_t width;
    uint8_t height;
    // 'M' material, 'S' stick, '.' empty, row-major top to bottom
    std::array<char, 9> cells;
};

constexpr ToolShape PICKAXE_SHAPE{3, 3, {'M', 'M', 'M', '.', 'S', '.', '.', 'S', '.'}};
constexpr ToolShape AXE_SHAPE{2, 3, {'M', 'M', 'M', 'S', '.', 'S'}};
constexpr ToolShape SHOVEL_SHAPE{1, 3, {'M', 'S', 'S'}};
constexpr ToolShape SWORD_SHAPE{1, 3, {'M', 'M', 'S'}};

constexpr CraftingRecipe toolRecipe(const ToolShape& shape, ItemType material, ItemType tool) {
    CraftingRecipe recipe{};
    recipe.width = shape.width;
    recipe.height = shape.height;
    for (size_t index = 0; index < static_cast<size_t>(shape.width) * shape.height; ++index) {
        recipe.pattern[index] = shape.cells[index] == 'M'   ? material
                                : shape.cells[index] == 'S' ? ItemType::STICK
                                                            : NONE;
    }
    recipe.result = makeItemStack(tool);
    return recipe;
}

constexpr std::array<ItemType, 3> TIER_MATERIALS = {PLANKS, COBBLESTONE, ItemType::IRON_INGOT};
constexpr std::array<ItemType, 3> TIER_PICKAXES = {ItemType::WOODEN_PICKAXE,
                                                   ItemType::STONE_PICKAXE, ItemType::IRON_PICKAXE};

constexpr auto makeRecipes() {
    // 6 fixed recipes plus 4 tool classes across 3 tiers.
    std::array<CraftingRecipe, 18> recipes{};
    size_t next = 0;
    recipes[next++] = shapeless(LOG, {PLANKS, 4});
    recipes[next++] = shaped(1, 2, {PLANKS, PLANKS}, {ItemType::STICK, 4});
    recipes[next++] = shaped(2, 2, {PLANKS, PLANKS, PLANKS, PLANKS},
                             {itemFromBlock(BlockType::CRAFTING_TABLE), 1});
    recipes[next++] = shaped(3, 3,
                             {COBBLESTONE, COBBLESTONE, COBBLESTONE, COBBLESTONE, NONE, COBBLESTONE,
                              COBBLESTONE, COBBLESTONE, COBBLESTONE},
                             {itemFromBlock(BlockType::FURNACE), 1});
    recipes[next++] =
        shaped(1, 2, {ItemType::COAL, ItemType::STICK}, {itemFromBlock(BlockType::TORCH), 4});
    // Three iron ingots in a V make an empty bucket.
    constexpr ItemType IRON = ItemType::IRON_INGOT;
    recipes[next++] = shaped(3, 2, {IRON, NONE, IRON, NONE, IRON, NONE}, {ItemType::BUCKET, 1});
    for (size_t tier = 0; tier < 3; ++tier) {
        const ItemType material = TIER_MATERIALS[tier];
        const auto pickaxe = static_cast<uint16_t>(TIER_PICKAXES[tier]);
        recipes[next++] = toolRecipe(PICKAXE_SHAPE, material, static_cast<ItemType>(pickaxe));
        recipes[next++] = toolRecipe(AXE_SHAPE, material, static_cast<ItemType>(pickaxe + 1));
        recipes[next++] = toolRecipe(SHOVEL_SHAPE, material, static_cast<ItemType>(pickaxe + 2));
        recipes[next++] = toolRecipe(SWORD_SHAPE, material, static_cast<ItemType>(pickaxe + 3));
    }
    return recipes;
}

constexpr auto RECIPES = makeRecipes();
static_assert([] {
    for (const CraftingRecipe& recipe : RECIPES) {
        if (recipe.result.empty()) return false;
        if (recipe.width == 0 && recipe.ingredientCount == 0) return false;
    }
    return true;
}());

struct GridBounds {
    int minX = 3, minY = 3, maxX = -1, maxY = -1;
    constexpr bool empty() const { return maxX < 0; }
    constexpr int width() const { return maxX - minX + 1; }
    constexpr int height() const { return maxY - minY + 1; }
};

GridBounds boundsOf(std::span<const ItemStack> grid, int gridWidth) {
    GridBounds bounds;
    const int rows = static_cast<int>(grid.size()) / gridWidth;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
            if (grid[static_cast<size_t>(y) * gridWidth + x].empty()) continue;
            bounds.minX = std::min(bounds.minX, x);
            bounds.minY = std::min(bounds.minY, y);
            bounds.maxX = std::max(bounds.maxX, x);
            bounds.maxY = std::max(bounds.maxY, y);
        }
    }
    return bounds;
}

bool matchesShaped(const CraftingRecipe& recipe, std::span<const ItemStack> grid, int gridWidth,
                   const GridBounds& bounds, bool mirrored) {
    if (bounds.width() != recipe.width || bounds.height() != recipe.height) return false;
    for (int y = 0; y < recipe.height; ++y) {
        for (int x = 0; x < recipe.width; ++x) {
            const int patternX = mirrored ? recipe.width - 1 - x : x;
            const ItemType expected =
                recipe.pattern[static_cast<size_t>(y) * recipe.width + patternX];
            const ItemStack& cell =
                grid[static_cast<size_t>(bounds.minY + y) * gridWidth + bounds.minX + x];
            const ItemType actual = cell.empty() ? NONE : craftingKey(cell.type);
            if (actual != expected) return false;
        }
    }
    return true;
}

bool matchesShapeless(const CraftingRecipe& recipe, std::span<const ItemStack> grid) {
    std::vector<ItemType> present;
    for (const ItemStack& cell : grid) {
        if (!cell.empty()) present.push_back(craftingKey(cell.type));
    }
    if (present.size() != recipe.ingredientCount) return false;
    std::vector<ItemType> required(recipe.pattern.begin(),
                                   recipe.pattern.begin() + recipe.ingredientCount);
    std::sort(present.begin(), present.end());
    std::sort(required.begin(), required.end());
    return present == required;
}

} // namespace

std::optional<ItemStack> matchCraftingRecipe(std::span<const ItemStack> grid, int gridWidth) {
    const GridBounds bounds = boundsOf(grid, gridWidth);
    if (bounds.empty()) return std::nullopt;
    for (const CraftingRecipe& recipe : RECIPES) {
        if (recipe.width == 0) {
            if (matchesShapeless(recipe, grid)) return recipe.result;
            continue;
        }
        if (matchesShaped(recipe, grid, gridWidth, bounds, false) ||
            matchesShaped(recipe, grid, gridWidth, bounds, true)) {
            return recipe.result;
        }
    }
    return std::nullopt;
}

void consumeOneCraft(std::span<ItemStack> grid) {
    for (ItemStack& cell : grid) {
        if (cell.empty()) continue;
        if (--cell.count == 0) cell.clear();
    }
}

std::optional<ItemType> smeltingResult(ItemType input) {
    const ItemType key = craftingKey(input);
    if (key == itemFromBlock(BlockType::IRON_ORE)) return ItemType::IRON_INGOT;
    if (key == itemFromBlock(BlockType::GOLD_ORE)) return ItemType::GOLD_INGOT;
    if (key == itemFromBlock(BlockType::SAND)) return itemFromBlock(BlockType::GLASS);
    if (key == itemFromBlock(BlockType::COBBLESTONE)) return itemFromBlock(BlockType::STONE);
    if (key == itemFromBlock(BlockType::LOG)) return ItemType::CHARCOAL;
    switch (key) {
        case ItemType::RAW_BEEF:
            return ItemType::COOKED_BEEF;
        case ItemType::RAW_PORKCHOP:
            return ItemType::COOKED_PORKCHOP;
        case ItemType::RAW_MUTTON:
            return ItemType::COOKED_MUTTON;
        case ItemType::RAW_CHICKEN:
            return ItemType::COOKED_CHICKEN;
        case ItemType::RAW_FISH:
            return ItemType::COOKED_FISH;
        default:
            return std::nullopt;
    }
}

bool isSmeltable(ItemType input) {
    return smeltingResult(input).has_value();
}

int fuelBurnTicks(ItemType fuel) {
    const ItemType key = craftingKey(fuel);
    if (key == ItemType::COAL) return 1600; // charcoal folds in
    if (key == PLANKS || key == LOG || key == itemFromBlock(BlockType::CRAFTING_TABLE)) {
        return 300;
    }
    if (key == ItemType::WOODEN_PICKAXE || key == ItemType::WOODEN_AXE ||
        key == ItemType::WOODEN_SHOVEL || key == ItemType::WOODEN_SWORD) {
        return 200;
    }
    if (key == ItemType::STICK) return 100;
    return 0;
}

bool isFurnaceFuel(ItemType fuel) {
    return fuelBurnTicks(fuel) > 0;
}

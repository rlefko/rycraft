#include <catch2/catch_test_macros.hpp>

#include "world/furnace.hpp"
#include "world/recipes.hpp"

#include <array>

namespace {

constexpr ItemType LOG = itemFromBlock(BlockType::LOG);
constexpr ItemType PLANKS = itemFromBlock(BlockType::PLANKS);
constexpr ItemType COBBLESTONE = itemFromBlock(BlockType::COBBLESTONE);

std::array<ItemStack, 4> grid2(ItemType a, ItemType b, ItemType c, ItemType d) {
    auto cell = [](ItemType type) {
        return type == ItemType::NONE ? ItemStack{} : ItemStack{type, 1, 0};
    };
    return {cell(a), cell(b), cell(c), cell(d)};
}

std::array<ItemStack, 9> grid3(std::array<ItemType, 9> cells) {
    std::array<ItemStack, 9> grid{};
    for (size_t index = 0; index < cells.size(); ++index) {
        if (cells[index] != ItemType::NONE)
            grid[index] = ItemStack{cells[index], 1, 0};
    }
    return grid;
}

constexpr ItemType N = ItemType::NONE;
constexpr ItemType S = ItemType::STICK;

} // namespace

TEST_CASE("Crafting key folds log species and charcoal", "[recipes]") {
    REQUIRE(craftingKey(itemFromBlock(BlockType::BIRCH_LOG)) == LOG);
    REQUIRE(craftingKey(itemFromBlock(BlockType::WILLOW_LOG)) == LOG);
    REQUIRE(craftingKey(LOG) == LOG);
    REQUIRE(craftingKey(ItemType::CHARCOAL) == ItemType::COAL);
    REQUIRE(craftingKey(PLANKS) == PLANKS);
    REQUIRE(craftingKey(ItemType::IRON_INGOT) == ItemType::IRON_INGOT);
}

TEST_CASE("Any log crafts planks anywhere in either grid", "[recipes]") {
    for (ItemType log :
         {LOG, itemFromBlock(BlockType::SPRUCE_LOG), itemFromBlock(BlockType::MANGROVE_LOG)}) {
        for (int slot = 0; slot < 4; ++slot) {
            std::array<ItemStack, 4> grid{};
            grid[slot] = ItemStack{log, 1, 0};
            const auto result = matchCraftingRecipe(grid, 2);
            REQUIRE(result.has_value());
            REQUIRE(result->type == PLANKS);
            REQUIRE(result->count == 4);
        }
    }
    auto big = grid3({N, N, N, N, LOG, N, N, N, N});
    REQUIRE(matchCraftingRecipe(big, 3).has_value());
}

TEST_CASE("Sticks torches tables and furnaces follow their shapes", "[recipes]") {
    const auto sticks = matchCraftingRecipe(grid2(PLANKS, N, PLANKS, N), 2);
    REQUIRE(sticks.has_value());
    REQUIRE(sticks->type == ItemType::STICK);
    REQUIRE(sticks->count == 4);
    // Side-by-side planks are not the stick shape.
    REQUIRE_FALSE(matchCraftingRecipe(grid2(PLANKS, PLANKS, N, N), 2).has_value());

    const auto table = matchCraftingRecipe(grid2(PLANKS, PLANKS, PLANKS, PLANKS), 2);
    REQUIRE(table.has_value());
    REQUIRE(table->type == itemFromBlock(BlockType::CRAFTING_TABLE));

    for (ItemType coal : {ItemType::COAL, ItemType::CHARCOAL}) {
        const auto torches = matchCraftingRecipe(grid2(coal, N, S, N), 2);
        REQUIRE(torches.has_value());
        REQUIRE(torches->type == itemFromBlock(BlockType::TORCH));
        REQUIRE(torches->count == 4);
    }

    constexpr ItemType C = COBBLESTONE;
    const auto furnace = matchCraftingRecipe(grid3({C, C, C, C, N, C, C, C, C}), 3);
    REQUIRE(furnace.has_value());
    REQUIRE(furnace->type == itemFromBlock(BlockType::FURNACE));
    // The furnace ring cannot fit the 2x2 inventory grid.
    REQUIRE_FALSE(matchCraftingRecipe(grid2(C, C, C, C), 2).has_value());
}

TEST_CASE("Three iron ingots in a V craft a bucket", "[recipes]") {
    constexpr ItemType I = ItemType::IRON_INGOT;
    const auto bucket = matchCraftingRecipe(grid3({I, N, I, N, I, N, N, N, N}), 3);
    REQUIRE(bucket.has_value());
    REQUIRE(bucket->type == ItemType::BUCKET);
    REQUIRE(bucket->count == 1);
    // A solid iron block is not the bucket shape.
    REQUIRE_FALSE(matchCraftingRecipe(grid3({I, I, I, I, I, I, I, I, I}), 3).has_value());
}

TEST_CASE("Wool over planks crafts a bed and iron crafts shears", "[recipes]") {
    constexpr ItemType I = ItemType::IRON_INGOT;
    constexpr ItemType WOOL = itemFromBlock(BlockType::WOOL);

    const auto shears = matchCraftingRecipe(grid2(N, I, I, N), 2);
    REQUIRE(shears.has_value());
    REQUIRE(shears->type == ItemType::SHEARS);
    REQUIRE(shears->count == 1);

    const auto bed =
        matchCraftingRecipe(grid3({WOOL, WOOL, WOOL, PLANKS, PLANKS, PLANKS, N, N, N}), 3);
    REQUIRE(bed.has_value());
    REQUIRE(bed->type == itemFromBlock(BlockType::BED));
    // Planks over wool (inverted) is not a bed.
    REQUIRE_FALSE(matchCraftingRecipe(grid3({PLANKS, PLANKS, PLANKS, WOOL, WOOL, WOOL, N, N, N}), 3)
                      .has_value());
}

TEST_CASE("Five planks in a U craft a boat", "[recipes]") {
    const auto boat =
        matchCraftingRecipe(grid3({PLANKS, N, PLANKS, PLANKS, PLANKS, PLANKS, N, N, N}), 3);
    REQUIRE(boat.has_value());
    REQUIRE(boat->type == ItemType::BOAT);
    REQUIRE(boat->count == 1);
}

TEST_CASE("Eight planks ringing an empty center craft a chest", "[recipes]") {
    const auto chest = matchCraftingRecipe(
        grid3({PLANKS, PLANKS, PLANKS, PLANKS, N, PLANKS, PLANKS, PLANKS, PLANKS}), 3);
    REQUIRE(chest.has_value());
    REQUIRE(chest->type == itemFromBlock(BlockType::CHEST));
    REQUIRE(chest->count == 1);
    // A filled center is the furnace shape family, never a chest.
    REQUIRE_FALSE(
        matchCraftingRecipe(
            grid3({PLANKS, PLANKS, PLANKS, PLANKS, PLANKS, PLANKS, PLANKS, PLANKS, PLANKS}), 3)
            .has_value());
}

TEST_CASE("Tool recipes cover tiers offsets and mirrors", "[recipes]") {
    constexpr ItemType I = ItemType::IRON_INGOT;
    const auto pickaxe = matchCraftingRecipe(grid3({I, I, I, N, S, N, N, S, N}), 3);
    REQUIRE(pickaxe.has_value());
    REQUIRE(pickaxe->type == ItemType::IRON_PICKAXE);
    REQUIRE(pickaxe->count == 1);
    REQUIRE(pickaxe->durability == itemDefinition(ItemType::IRON_PICKAXE).maxDurability);

    // Wooden shovel in the left column, then shifted to the right column.
    const auto shovel = matchCraftingRecipe(grid3({PLANKS, N, N, S, N, N, S, N, N}), 3);
    REQUIRE(shovel.has_value());
    REQUIRE(shovel->type == ItemType::WOODEN_SHOVEL);
    const auto shifted = matchCraftingRecipe(grid3({N, N, PLANKS, N, N, S, N, N, S}), 3);
    REQUIRE(shifted.has_value());
    REQUIRE(shifted->type == ItemType::WOODEN_SHOVEL);

    // Axe and its horizontal mirror both match.
    constexpr ItemType C = COBBLESTONE;
    const auto axe = matchCraftingRecipe(grid3({C, C, N, C, S, N, N, S, N}), 3);
    REQUIRE(axe.has_value());
    REQUIRE(axe->type == ItemType::STONE_AXE);
    const auto mirrored = matchCraftingRecipe(grid3({C, C, N, S, C, N, S, N, N}), 3);
    REQUIRE(mirrored.has_value());
    REQUIRE(mirrored->type == ItemType::STONE_AXE);

    const auto sword = matchCraftingRecipe(grid3({N, C, N, N, C, N, N, S, N}), 3);
    REQUIRE(sword.has_value());
    REQUIRE(sword->type == ItemType::STONE_SWORD);

    // Swords do not fit the 2x2 grid (height 3).
    REQUIRE_FALSE(matchCraftingRecipe(grid2(C, N, C, N), 2).has_value());
}

TEST_CASE("Consuming a craft decrements every occupied cell", "[recipes]") {
    auto grid = grid2(PLANKS, N, PLANKS, N);
    grid[0].count = 3;
    consumeOneCraft(grid);
    REQUIRE(grid[0].count == 2);
    REQUIRE(grid[2].empty());
    REQUIRE(grid[1].empty());
}

TEST_CASE("Smelting covers ores sand cobble logs and raw foods", "[recipes]") {
    REQUIRE(smeltingResult(itemFromBlock(BlockType::IRON_ORE)) == ItemType::IRON_INGOT);
    REQUIRE(smeltingResult(itemFromBlock(BlockType::GOLD_ORE)) == ItemType::GOLD_INGOT);
    REQUIRE(smeltingResult(itemFromBlock(BlockType::SAND)) == itemFromBlock(BlockType::GLASS));
    REQUIRE(smeltingResult(COBBLESTONE) == itemFromBlock(BlockType::STONE));
    REQUIRE(smeltingResult(itemFromBlock(BlockType::PALM_LOG)) == ItemType::CHARCOAL);
    REQUIRE(smeltingResult(ItemType::RAW_BEEF) == ItemType::COOKED_BEEF);
    REQUIRE(smeltingResult(ItemType::RAW_PORKCHOP) == ItemType::COOKED_PORKCHOP);
    REQUIRE(smeltingResult(ItemType::RAW_MUTTON) == ItemType::COOKED_MUTTON);
    REQUIRE(smeltingResult(ItemType::RAW_CHICKEN) == ItemType::COOKED_CHICKEN);
    REQUIRE(smeltingResult(ItemType::RAW_FISH) == ItemType::COOKED_FISH);
    REQUIRE_FALSE(isSmeltable(ItemType::STICK));
    REQUIRE_FALSE(isSmeltable(itemFromBlock(BlockType::STONE)));

    REQUIRE(fuelBurnTicks(ItemType::COAL) == 1600);
    REQUIRE(fuelBurnTicks(ItemType::CHARCOAL) == 1600);
    REQUIRE(fuelBurnTicks(PLANKS) == 300);
    REQUIRE(fuelBurnTicks(itemFromBlock(BlockType::BIRCH_LOG)) == 300);
    REQUIRE(fuelBurnTicks(ItemType::STICK) == 100);
    REQUIRE(fuelBurnTicks(ItemType::WOODEN_AXE) == 200);
    REQUIRE_FALSE(isFurnaceFuel(ItemType::IRON_INGOT));
    REQUIRE_FALSE(isFurnaceFuel(COBBLESTONE));
}

TEST_CASE("Furnace ignites cooks and cools deterministically", "[recipes][furnace]") {
    FurnaceState furnace;
    furnace.input = {ItemType::RAW_BEEF, 2, 0};
    furnace.fuel = {ItemType::STICK, 2, 0};

    // Ignition consumes exactly one fuel item and reports the lit change.
    REQUIRE(furnaceTick(furnace));
    REQUIRE(furnace.lit());
    REQUIRE(furnace.fuel.count == 1);
    REQUIRE(furnace.burnTicksTotal == 100);

    // A stick burns out before the 200-tick cook finishes; the second stick
    // reignites and the item completes on cumulative progress.
    int safety = 0;
    while (furnace.output.empty() && ++safety < 1000) {
        furnaceTick(furnace);
    }
    REQUIRE(safety < 1000);
    REQUIRE(furnace.output.type == ItemType::COOKED_BEEF);
    REQUIRE(furnace.output.count == 1);
    REQUIRE(furnace.input.count == 1);
    REQUIRE(furnace.fuel.empty());

    // Out of fuel: progress decays instead of holding.
    while (furnace.lit()) {
        furnaceTick(furnace);
    }
    furnace.cookTicks = 10;
    furnaceTick(furnace);
    REQUIRE(furnace.cookTicks == 8);

    // No smelting into a mismatched output.
    FurnaceState blocked;
    blocked.input = {ItemType::RAW_FISH, 1, 0};
    blocked.fuel = {ItemType::COAL, 1, 0};
    blocked.output = {ItemType::IRON_INGOT, 1, 0};
    REQUIRE_FALSE(furnaceTick(blocked));
    REQUIRE_FALSE(blocked.lit());
    REQUIRE(blocked.fuel.count == 1);

    // No ignition without smeltable input.
    FurnaceState idle;
    idle.fuel = {ItemType::COAL, 1, 0};
    REQUIRE_FALSE(furnaceTick(idle));
    REQUIRE_FALSE(idle.lit());
}

TEST_CASE("Furnace gauges expose cook and fuel fractions", "[recipes][furnace]") {
    FurnaceState furnace;
    REQUIRE(furnace.cookFraction() == 0.0f);
    REQUIRE(furnace.fuelFraction() == 0.0f);
    furnace.burnTicksTotal = 200;
    furnace.burnTicksRemaining = 50;
    furnace.cookTicks = 100;
    REQUIRE(furnace.cookFraction() == 0.5f);
    REQUIRE(furnace.fuelFraction() == 0.25f);
}

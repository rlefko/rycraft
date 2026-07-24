#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_helpers.hpp"
#include "world/save_manager.hpp"
#include "world/world_list.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

// TempDir clears its path at construction, so direct file writes must first
// recreate the directory a SaveManager constructor would have made.
void writeTextFile(const std::string& path, const std::string& content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    REQUIRE(out.is_open());
    out << content;
    REQUIRE(out.good());
}

} // namespace

TEST_CASE("World metadata v2 round-trips every field", "[saves][metadata]") {
    TempDir directory("metadata_v2_roundtrip");
    SaveManager saves(directory.path());

    SaveManager::WorldMetadata written;
    written.seed = 777;
    written.spawnPos = Vec3{8.f, 96.f, -12.f};
    written.bedSpawnSet = true;
    written.playerPos = Vec3{100.f, 64.f, 250.f};
    written.worldTime = 13500;
    written.name = "Frontier Valley 2";
    written.gameMode = GameMode::SURVIVAL;
    written.generation = GenerationSettings{false, true, false, true};
    written.createdMs = 1784290000000ULL;
    written.player.yaw = 92.5f;
    written.player.hunger = 11;
    written.player.inventory.fill(ItemStack{});
    written.player.inventory[3] = ItemStack{ItemType::COOKED_MUTTON, 7, 0};
    written.player.inventory[20] = ItemStack{ItemType::STONE_AXE, 1, 44};
    written.player.carriedStacks[0] = ItemStack{ItemType::DIAMOND, 17, 0};
    written.player.carriedStacks[9] = ItemStack{ItemType::IRON_PICKAXE, 1, 203};
    REQUIRE(saves.saveMetadata(written));

    const auto loaded = saves.loadMetadata();
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->seed == 777);
    REQUIRE(loaded->spawnPos == written.spawnPos);
    REQUIRE(loaded->bedSpawnSet);
    REQUIRE(loaded->playerPos == written.playerPos);
    REQUIRE(loaded->worldTime == 13500);
    REQUIRE(loaded->name == "Frontier Valley 2");
    REQUIRE(loaded->gameMode == GameMode::SURVIVAL);
    REQUIRE(loaded->generation == written.generation);
    REQUIRE(loaded->createdMs == written.createdMs);
    REQUIRE(loaded->lastPlayedMs > 0);
    REQUIRE(loaded->player.yaw == Catch::Approx(92.5f));
    REQUIRE(loaded->player.hunger == 11);
    REQUIRE(loaded->player.inventory[3] == ItemStack{ItemType::COOKED_MUTTON, 7, 0});
    REQUIRE(loaded->player.inventory[20] == ItemStack{ItemType::STONE_AXE, 1, 44});
    REQUIRE(loaded->player.inventory[0].empty());
    REQUIRE(loaded->player.carriedStacks[0] == ItemStack{ItemType::DIAMOND, 17, 0});
    REQUIRE(loaded->player.carriedStacks[9] == ItemStack{ItemType::IRON_PICKAXE, 1, 203});
}

TEST_CASE("Legacy metadata shapes load with creative defaults", "[saves][metadata]") {
    TempDir directory("metadata_legacy");

    // The oldest live shape: only seed, spawn, and time.
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 42,\n"
                  "  \"spawnPos\": { \"x\": 1.5, \"y\": 90, \"z\": -3 },\n"
                  "  \"worldTime\": 600\n"
                  "}\n");
    const auto oldest = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(oldest.has_value());
    REQUIRE(oldest->seed == 42);
    REQUIRE(oldest->gameMode == GameMode::CREATIVE);
    REQUIRE(oldest->generation == GenerationSettings{});
    REQUIRE(oldest->name.empty());
    REQUIRE_FALSE(oldest->bedSpawnSet);
    // Missing playerPos falls back to the spawn.
    REQUIRE(oldest->playerPos == oldest->spawnPos);
    // Missing player section keeps the classic starter hotbar.
    REQUIRE(oldest->player.inventory[0] == ItemStack{itemFromBlock(BlockType::STONE), 1, 0});
    REQUIRE(oldest->player.hunger == 20);
    for (const ItemStack& stack : oldest->player.carriedStacks)
        REQUIRE(stack.empty());

    // The nine-number hotbar shape converts to count-one stacks.
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 7,\n"
                  "  \"spawnPos\": { \"x\": 0, \"y\": 100, \"z\": 0 },\n"
                  "  \"worldTime\": 0,\n"
                  "  \"player\": {\n"
                  "    \"yaw\": 10, \"pitch\": 0, \"health\": 18, \"selectedSlot\": 2,\n"
                  "    \"inventory\": [1, 3, 2, 8, 15, 4, 19, 16, 28]\n"
                  "  }\n"
                  "}\n");
    const auto legacy = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(legacy.has_value());
    REQUIRE(legacy->player.inventory[0] == ItemStack{itemFromBlock(BlockType::STONE), 1, 0});
    REQUIRE(legacy->player.inventory[8] == ItemStack{itemFromBlock(BlockType::FLOWER_RED), 1, 0});
    for (size_t slot = 9; slot < SaveManager::PLAYER_INVENTORY_SLOTS; ++slot) {
        REQUIRE(legacy->player.inventory[slot].empty());
    }
    REQUIRE(legacy->player.health == 18);
}

TEST_CASE("Carried stacks validate ids and counts with inventory rules", "[saves][metadata]") {
    TempDir directory("metadata_carried_stack_validation");
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 1,\n"
                  "  \"spawnPos\": { \"x\": 0, \"y\": 64, \"z\": 0 },\n"
                  "  \"worldTime\": 0,\n"
                  "  \"player\": {\n"
                  "    \"carriedStacks\": [[65536, 5, 0], [257, 200, 0], [256, 3, 7]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [280, 1, 245]]\n"
                  "  }\n"
                  "}\n");
    const auto loaded = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(loaded.has_value());
    CHECK(loaded->player.carriedStacks[0].empty());
    CHECK(loaded->player.carriedStacks[1] == ItemStack{ItemType::COAL, 64, 0});
    CHECK(loaded->player.carriedStacks[2] == ItemStack{ItemType::STICK, 3, 7});
    CHECK(loaded->player.carriedStacks[9] == ItemStack{ItemType::IRON_PICKAXE, 1, 245});

    SaveManager saves(directory.path());
    SaveManager::WorldMetadata metadata;
    metadata.seed = 2;
    metadata.spawnPos = Vec3{0.5F, 64.0F, 0.5F};
    metadata.playerPos = metadata.spawnPos;
    metadata.player.carriedStacks[0] = ItemStack{ItemType::COAL, 65, 0};
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.player.carriedStacks[0] = ItemStack{static_cast<ItemType>(100), 1, 0};
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.player.carriedStacks[0] = ItemStack{ItemType::NONE, 0, 1};
    CHECK_FALSE(saves.saveMetadata(metadata));
}

TEST_CASE("Older generator v4 metadata defaults new player provenance fields",
          "[saves][metadata][generator-v4][migration]") {
    TempDir directory("metadata_v4_player_provenance_defaults");
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 3,\n"
                  "  \"generationFingerprint\": "
                  "\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\n"
                  "  \"spawnFinalized\": true,\n"
                  "  \"spawnSafetyRevision\": 4,\n"
                  "  \"safeSpawnPos\": { \"x\": 1.5, \"y\": 70, \"z\": -1.5 },\n"
                  "  \"spawnPos\": { \"x\": 1.5, \"y\": 70, \"z\": -1.5 },\n"
                  "  \"playerPos\": { \"x\": 4.5, \"y\": 71, \"z\": -6.5 },\n"
                  "  \"worldTime\": 7,\n"
                  "  \"generatorVersion\": 4\n"
                  "}\n");
    const auto loaded = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(loaded.has_value());
    CHECK_FALSE(loaded->bedSpawnSet);
    for (const ItemStack& stack : loaded->player.carriedStacks)
        CHECK(stack.empty());
}

TEST_CASE("Metadata writes reject positions unsafe for world conversion", "[saves][metadata]") {
    TempDir directory("metadata_position_write_bounds");
    SaveManager saves(directory.path());

    SaveManager::WorldMetadata metadata;
    metadata.seed = 91;
    metadata.spawnPos = Vec3{0.5F, static_cast<float>(WORLD_MIN_Y), -0.5F};
    metadata.playerPos = Vec3{-12.5F, static_cast<float>(WORLD_MAX_Y), 18.5F};
    REQUIRE(saves.saveMetadata(metadata));

    metadata.playerPos.y = static_cast<float>(WORLD_MAX_Y) + 1.0F;
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.playerPos.y = static_cast<float>(WORLD_MIN_Y) - 1.0F;
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.playerPos = Vec3{1.0e30F, 64.0F, 0.0F};
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.playerPos = Vec3{0.0F, 64.0F, -1.0e30F};
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.playerPos = Vec3{0.0F, std::numeric_limits<float>::quiet_NaN(), 0.0F};
    CHECK_FALSE(saves.saveMetadata(metadata));
    metadata.playerPos = Vec3{0.0F, 64.0F, 0.0F};
    metadata.safeSpawnPos = Vec3{0.0F, static_cast<float>(WORLD_MAX_Y) + 1.0F, 0.0F};
    CHECK_FALSE(saves.saveMetadata(metadata));
}

TEST_CASE("Generator v4 repairs unsafe positions before startup can cast them",
          "[saves][metadata][generator-v4][corruption]") {
    TempDir directory("metadata_v4_position_repair");
    constexpr std::string_view fingerprint{
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    std::ostringstream json;
    json << "{\n"
         << "  \"seed\": 123,\n"
         << "  \"generationFingerprint\": \"" << fingerprint << "\",\n"
         << "  \"spawnFinalized\": true,\n"
         << "  \"spawnSafetyRevision\": " << SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION
         << ",\n"
         << "  \"safeSpawnPos\": { \"x\": 24.5, \"y\": 81, \"z\": -8.5 },\n"
         << "  \"spawnPos\": { \"x\": 2, \"y\": " << WORLD_MAX_Y + 1 << ", \"z\": 3 },\n"
         << "  \"bedSpawnSet\": true,\n"
         << "  \"playerPos\": { \"x\": 1e30, \"y\": 64, \"z\": 0 },\n"
         << "  \"worldTime\": 4,\n"
         << "  \"generatorVersion\": " << SaveManager::GENERATOR_V4_VERSION << "\n"
         << "}\n";
    writeTextFile(directory.path() + "/metadata.json", json.str());

    const auto repaired = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(repaired.has_value());
    REQUIRE(repaired->safeSpawnPos.has_value());
    CHECK(repaired->spawnPos == *repaired->safeSpawnPos);
    CHECK(repaired->playerPos == *repaired->safeSpawnPos);
    CHECK_FALSE(repaired->bedSpawnSet);
    CHECK(repaired->spawnFinalized);
}

TEST_CASE("Generator v4 invalidates an unsafe safe spawn and repairs remaining fields",
          "[saves][metadata][generator-v4][corruption]") {
    TempDir directory("metadata_v4_safe_spawn_repair");
    constexpr std::string_view fingerprint{
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"};
    std::ostringstream json;
    json << "{\n"
         << "  \"seed\": 456,\n"
         << "  \"generationFingerprint\": \"" << fingerprint << "\",\n"
         << "  \"spawnFinalized\": true,\n"
         << "  \"spawnSafetyRevision\": " << SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION
         << ",\n"
         << "  \"safeSpawnPos\": { \"x\": 0, \"y\": " << WORLD_MIN_Y - 1 << ", \"z\": 0 },\n"
         << "  \"spawnPos\": { \"x\": 1e30, \"y\": 64, \"z\": 0 },\n"
         << "  \"bedSpawnSet\": true,\n"
         << "  \"playerPos\": { \"x\": -10.5, \"y\": 72, \"z\": 31.5 },\n"
         << "  \"worldTime\": 5,\n"
         << "  \"generatorVersion\": " << SaveManager::GENERATOR_V4_VERSION << "\n"
         << "}\n";
    writeTextFile(directory.path() + "/metadata.json", json.str());

    const auto repaired = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(repaired.has_value());
    CHECK_FALSE(repaired->safeSpawnPos.has_value());
    CHECK_FALSE(repaired->spawnFinalized);
    CHECK(repaired->spawnSafetyRevision == 0);
    CHECK(repaired->spawnPos == repaired->playerPos);
    CHECK_FALSE(repaired->bedSpawnSet);
}

TEST_CASE("Generator v4 refuses metadata with no safe position to repair from",
          "[saves][metadata][generator-v4][corruption]") {
    TempDir directory("metadata_v4_unrepairable_positions");
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 789,\n"
                  "  \"generationFingerprint\": "
                  "\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\n"
                  "  \"spawnFinalized\": false,\n"
                  "  \"spawnSafetyRevision\": 0,\n"
                  "  \"safeSpawnPos\": null,\n"
                  "  \"spawnPos\": { \"x\": 0, \"y\": 99999, \"z\": 0 },\n"
                  "  \"playerPos\": { \"x\": 1e30, \"y\": 64, \"z\": 0 },\n"
                  "  \"worldTime\": 6,\n"
                  "  \"generatorVersion\": 4\n"
                  "}\n");
    CHECK_FALSE(SaveManager::readMetadataFile(directory.path() + "/metadata.json").has_value());
}

TEST_CASE("Generator v4 locks structures but permits runtime generation toggles",
          "[saves][metadata][generator-v4]") {
    TempDir directory("metadata_v4_structures_identity");
    SaveManager saves(directory.path(), SaveManager::Profile::GeneratorV4);
    SaveManager::WorldMetadata metadata;
    metadata.seed = 0x1234'5678'9ABC'DEF0ULL;
    metadata.generationFingerprint =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    metadata.spawnFinalized = false;
    metadata.spawnSafetyRevision = 0;
    metadata.spawnPos = Vec3{0.5F, 80.0F, -0.5F};
    metadata.bedSpawnSet = true;
    metadata.playerPos = metadata.spawnPos;
    metadata.generatorVersion = SaveManager::GENERATOR_V4_VERSION;
    metadata.generation = GenerationSettings{true, true, true, true};
    metadata.player.carriedStacks[0] = ItemStack{ItemType::DIAMOND, 4, 0};
    metadata.player.carriedStacks[8] = ItemStack{ItemType::STONE_AXE, 1, 12};
    REQUIRE(saves.saveMetadata(metadata));

    metadata.generation.fauna = false;
    metadata.generation.weather = false;
    metadata.generation.dayCycle = false;
    REQUIRE(saves.saveMetadata(metadata));
    metadata.generation.structures = false;
    CHECK_FALSE(saves.saveMetadata(metadata));

    const auto loaded = saves.loadMetadata();
    REQUIRE(loaded.has_value());
    CHECK(loaded->generation.structures);
    CHECK(loaded->bedSpawnSet);
    CHECK_FALSE(loaded->generation.fauna);
    CHECK_FALSE(loaded->generation.weather);
    CHECK_FALSE(loaded->generation.dayCycle);
    CHECK(loaded->player.carriedStacks[0] == ItemStack{ItemType::DIAMOND, 4, 0});
    CHECK(loaded->player.carriedStacks[8] == ItemStack{ItemType::STONE_AXE, 1, 12});
}

TEST_CASE("Inventory slots tolerate out-of-range items and oversized counts", "[saves][metadata]") {
    TempDir directory("metadata_forward_tolerance");
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 1,\n"
                  "  \"spawnPos\": { \"x\": 0, \"y\": 0, \"z\": 0 },\n"
                  "  \"worldTime\": 0,\n"
                  "  \"player\": {\n"
                  "    \"hunger\": 99,\n"
                  "    \"inventorySlots\": [[9999, 5, 0], [257, 200, 0], [256, 3, 7]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0]]\n"
                  "  }\n"
                  "}\n");
    const auto loaded = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(loaded.has_value());
    // Unknown future item ids read as empty rather than failing the file.
    REQUIRE(loaded->player.inventory[0].empty());
    // Oversized counts clamp to the stack limit.
    REQUIRE(loaded->player.inventory[1] == ItemStack{ItemType::COAL, 64, 0});
    REQUIRE(loaded->player.inventory[2] == ItemStack{ItemType::STICK, 3, 7});
    REQUIRE(loaded->player.hunger == 20); // clamped
}

TEST_CASE("Reserved-gap item ids read as empty, not phantom items", "[saves][metadata]") {
    TempDir directory("metadata_reserved_gap");
    // Id 100 sits in the reserved gap between block ids (< 62) and non-block
    // items (>= 256); it must load as empty, honoring the parser's contract.
    writeTextFile(directory.path() + "/metadata.json",
                  "{\n"
                  "  \"seed\": 1,\n"
                  "  \"spawnPos\": { \"x\": 0, \"y\": 0, \"z\": 0 },\n"
                  "  \"worldTime\": 0,\n"
                  "  \"player\": {\n"
                  "    \"inventorySlots\": [[100, 5, 0], [57, 1, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]"
                  ", [0, 0, 0], [0, 0, 0], [0, 0, 0]]\n"
                  "  }\n"
                  "}\n");
    const auto loaded = SaveManager::readMetadataFile(directory.path() + "/metadata.json");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->player.inventory[0].empty()); // reserved-gap id 100 dropped
    // The last valid block id (andesite = 57 < 62) still loads.
    REQUIRE(loaded->player.inventory[1] == ItemStack{itemFromBlock(BlockType::ANDESITE), 1, 0});
}

TEST_CASE("Block entities sidecar round-trips furnaces and chests", "[saves][block-entities]") {
    TempDir directory("block_entities_roundtrip");
    SaveManager saves(directory.path());

    // Missing file reads as empty.
    REQUIRE(saves.loadBlockEntities().furnaces.empty());
    REQUIRE(saves.loadBlockEntities().chests.empty());

    FurnaceMap furnaces;
    FurnaceState lit;
    lit.input = ItemStack{ItemType::RAW_BEEF, 4, 0};
    lit.fuel = ItemStack{ItemType::COAL, 2, 0};
    lit.output = ItemStack{ItemType::COOKED_BEEF, 1, 0};
    lit.burnTicksRemaining = 900;
    lit.burnTicksTotal = 1600;
    lit.cookTicks = 77;
    furnaces[BlockPos{10, 64, -3}] = lit;
    furnaces[BlockPos{-2000000000LL, -40, 7}] = FurnaceState{};

    ChestMap chests;
    ChestState chest;
    chest.slots[0] = ItemStack{itemFromBlock(BlockType::COBBLESTONE), 64, 0};
    chest.slots[26] = ItemStack{ItemType::IRON_PICKAXE, 1, 131};
    chests[BlockPos{4, 70, 4}] = chest;
    chests[BlockPos{4, 70, 5}] = ChestState{}; // an empty chest still persists
    REQUIRE(saves.saveBlockEntities(furnaces, chests));

    const SaveManager::BlockEntities loaded = saves.loadBlockEntities();
    REQUIRE(loaded.furnaces.size() == 2);
    const auto& reloaded = loaded.furnaces.at(BlockPos{10, 64, -3});
    REQUIRE(reloaded.input == lit.input);
    REQUIRE(reloaded.fuel == lit.fuel);
    REQUIRE(reloaded.output == lit.output);
    REQUIRE(reloaded.burnTicksRemaining == 900);
    REQUIRE(reloaded.burnTicksTotal == 1600);
    REQUIRE(reloaded.cookTicks == 77);
    REQUIRE(loaded.furnaces.at(BlockPos{-2000000000LL, -40, 7}).input.empty());

    REQUIRE(loaded.chests.size() == 2);
    const auto& reloadedChest = loaded.chests.at(BlockPos{4, 70, 4});
    REQUIRE(reloadedChest.slots[0] == ItemStack{itemFromBlock(BlockType::COBBLESTONE), 64, 0});
    REQUIRE(reloadedChest.slots[26] == ItemStack{ItemType::IRON_PICKAXE, 1, 131});
    REQUIRE(loaded.chests.at(BlockPos{4, 70, 5}).empty());
}

TEST_CASE("Block entities sidecar skips unknown records and malformed lines",
          "[saves][block-entities]") {
    TempDir directory("block_entities_tolerance");
    writeTextFile(directory.path() + "/block_entities.dat",
                  "RYBE 1\n"
                  "beehive 1 2 3 some future payload\n"
                  "furnace 5 60 5 0 0 0 262 2 0 257 1 0 0 0 0\n"
                  "furnace broken line\n");
    SaveManager saves(directory.path());
    const FurnaceMap loaded = saves.loadBlockEntities().furnaces;
    REQUIRE(loaded.size() == 1);
    REQUIRE(loaded.at(BlockPos{5, 60, 5}).input == ItemStack{ItemType::RAW_BEEF, 2, 0});

    // An unrecognized header refuses the whole file rather than guessing.
    writeTextFile(directory.path() + "/block_entities.dat", "RYBX 9\nfurnace 1 1 1\n");
    REQUIRE(saves.loadBlockEntities().furnaces.empty());
}

TEST_CASE("Block entities write is atomic under injected failures", "[saves][block-entities]") {
    TempDir directory("block_entities_atomic");
    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    SaveManager saves(directory.path(), hooks);

    FurnaceMap first;
    first[BlockPos{1, 2, 3}].input = ItemStack{ItemType::RAW_FISH, 1, 0};
    REQUIRE(saves.saveBlockEntities(first));

    hooks->writeFailuresRemaining.store(100);
    FurnaceMap second;
    second[BlockPos{9, 9, 9}].input = ItemStack{ItemType::RAW_CHICKEN, 5, 0};
    REQUIRE_FALSE(saves.saveBlockEntities(second));

    // The prior durable contents survive the failed replacement.
    const FurnaceMap loaded = saves.loadBlockEntities().furnaces;
    REQUIRE(loaded.size() == 1);
    REQUIRE(loaded.count(BlockPos{1, 2, 3}) == 1);
}

TEST_CASE("World list enumerates saves and the legacy directory", "[saves][worlds]") {
    TempDir directory("world_list");
    const std::string root = directory.path();
    REQUIRE(listWorlds(root).empty());

    // The legacy world adopts in place and lists alongside created worlds.
    std::filesystem::create_directories(std::filesystem::path(root) / LEGACY_WORLD_DIRECTORY /
                                        SaveManager::CURRENT_REGIONS_DIRECTORY);
    const auto first = createWorld("Alpha Base", 1234, GameMode::SURVIVAL, {}, root);
    REQUIRE(first.has_value());
    const auto second = createWorld("Beta Cove", 99, GameMode::CREATIVE, {}, root);
    REQUIRE(second.has_value());

    const auto worlds = listWorlds(root);
    REQUIRE(worlds.size() == 3);
    // Stamped worlds sort ahead of the legacy world, which has no stamp.
    REQUIRE(worlds[2].metadata.name == LEGACY_WORLD_DIRECTORY);
    const auto* alpha = worlds[0].metadata.name == "Alpha Base" ? &worlds[0] : &worlds[1];
    REQUIRE(alpha->metadata.name == "Alpha Base");
    REQUIRE(alpha->metadata.seed == 1234);
    REQUIRE(alpha->metadata.gameMode == GameMode::SURVIVAL);
    // Survival worlds start with nothing; the world list read must show it.
    for (const ItemStack& stack : alpha->metadata.player.inventory) {
        REQUIRE(stack.empty());
    }

    // A stray file or empty directory under saves/ is not a world.
    std::filesystem::create_directories(std::filesystem::path(root) / SAVES_ROOT / "not_a_world");
    REQUIRE(listWorlds(root).size() == 3);
}

TEST_CASE("World directories sanitize and deduplicate", "[saves][worlds]") {
    TempDir directory("world_sanitize");
    const std::string root = directory.path();

    REQUIRE(sanitizeWorldDirectory("My World!", root) == "My_World");
    REQUIRE(sanitizeWorldDirectory("dots.and spaces", root) == "dots_and_spaces");
    REQUIRE(sanitizeWorldDirectory("", root) == "world");
    REQUIRE(sanitizeWorldDirectory("~~~", root) == "world");

    REQUIRE(createWorld("Twin", 1, GameMode::CREATIVE, {}, root).has_value());
    REQUIRE(sanitizeWorldDirectory("Twin", root) == "Twin_2");
    REQUIRE(createWorld("Twin", 2, GameMode::CREATIVE, {}, root).has_value());
    REQUIRE(sanitizeWorldDirectory("Twin", root) == "Twin_3");
}

TEST_CASE("Delete world refuses anything outside the world roots", "[saves][worlds]") {
    TempDir directory("world_delete");
    const std::string root = directory.path();

    const auto created = createWorld("Doomed", 5, GameMode::SURVIVAL, {}, root);
    REQUIRE(created.has_value());
    REQUIRE(std::filesystem::exists(*created));

    // A sibling non-world directory refuses deletion.
    const auto stray = std::filesystem::path(root) / SAVES_ROOT / "stray";
    std::filesystem::create_directories(stray);
    REQUIRE_FALSE(deleteWorld(stray.string(), root));

    // Paths outside the roots refuse even when they exist.
    REQUIRE_FALSE(deleteWorld(root, root));
    REQUIRE_FALSE(deleteWorld((std::filesystem::path(root) / SAVES_ROOT).string(), root));

    REQUIRE(deleteWorld(*created, root));
    REQUIRE_FALSE(std::filesystem::exists(*created));
    // Deleting twice reports failure instead of pretending success.
    REQUIRE_FALSE(deleteWorld(*created, root));
}

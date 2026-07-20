#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <lz4.h>
#include <world/chunk_generator.hpp>
#include <world/fluid.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <array>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

FluidCell residentCell(BlockType block = BlockType::AIR, FluidState state = FluidState::source()) {
    return {.loaded = true, .block = block, .state = state};
}

std::filesystem::path regionPath(const std::string& worldPath, ColumnPos column) {
    return std::filesystem::path(worldPath) / SaveManager::CURRENT_REGIONS_DIRECTORY /
           ("r." + std::to_string(world_coord::floorDiv(column.x, int64_t{32})) + "." +
            std::to_string(world_coord::floorDiv(column.z, int64_t{32})));
}

std::filesystem::path manifestPath(const std::string& worldPath, ColumnPos column) {
    return regionPath(worldPath, column) /
           ("m." + std::to_string(column.x) + "." + std::to_string(column.z) + ".manifest");
}

std::filesystem::path cubePath(const std::string& worldPath, ChunkPos position) {
    return regionPath(worldPath, {position.x, position.z}) /
           ("c." + std::to_string(position.x) + "." + std::to_string(position.y) + "." +
            std::to_string(position.z) + ".dat");
}

void refreshPayloadChecksum(std::vector<uint8_t>& data) {
    REQUIRE(data.size() >= HEADER_SIZE);
    ChunkSaveHeader header{};
    std::memcpy(&header, data.data(), sizeof(header));
    header.payloadChecksum = ChunkSerializer::payloadChecksum(
        std::span<const uint8_t>(data).subspan(static_cast<size_t>(HEADER_SIZE)));
    std::memcpy(data.data(), &header, sizeof(header));
}

std::vector<uint8_t> readCompressedCube(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    REQUIRE(input.is_open());
    const std::streamsize fileSize = input.tellg();
    REQUIRE(fileSize > static_cast<std::streamsize>(sizeof(uint32_t)));
    std::vector<uint8_t> compressed(static_cast<size_t>(fileSize));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(compressed.data()), fileSize);
    REQUIRE(input.good());

    uint32_t originalSize = 0;
    std::memcpy(&originalSize, compressed.data(), sizeof(originalSize));
    REQUIRE(originalSize > HEADER_SIZE);
    std::vector<uint8_t> decompressed(originalSize);
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed.data() + sizeof(originalSize)),
        reinterpret_cast<char*>(decompressed.data()),
        static_cast<int>(compressed.size() - sizeof(originalSize)), static_cast<int>(originalSize));
    REQUIRE(written == static_cast<int>(originalSize));
    return decompressed;
}

void writeCompressedCube(const std::filesystem::path& path, std::span<const uint8_t> payload) {
    REQUIRE(payload.size() <= static_cast<size_t>(INT_MAX));
    const int maximum = LZ4_compressBound(static_cast<int>(payload.size()));
    std::vector<uint8_t> compressed(sizeof(uint32_t) + static_cast<size_t>(maximum));
    const uint32_t originalSize = static_cast<uint32_t>(payload.size());
    std::memcpy(compressed.data(), &originalSize, sizeof(originalSize));
    const int written =
        LZ4_compress_default(reinterpret_cast<const char*>(payload.data()),
                             reinterpret_cast<char*>(compressed.data() + sizeof(originalSize)),
                             static_cast<int>(payload.size()), maximum);
    REQUIRE(written > 0);
    compressed.resize(sizeof(originalSize) + static_cast<size_t>(written));

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.is_open());
    output.write(reinterpret_cast<const char*>(compressed.data()),
                 static_cast<std::streamsize>(compressed.size()));
    REQUIRE(output.good());
}

void writeManifest(const std::string& worldPath, ColumnPos column, const std::string& contents) {
    std::filesystem::create_directories(regionPath(worldPath, column));
    std::ofstream output(manifestPath(worldPath, column), std::ios::trunc);
    REQUIRE(output.is_open());
    output << contents;
    REQUIRE(output.good());
}

} // namespace

TEST_CASE("Unavailable lower water cells defer without horizontal mutation",
          "[fluid][fluid-persistence][rules]") {
    FluidNeighborhood cells{
        .center = residentCell(BlockType::WATER),
        .down = {},
        .up = residentCell(),
        .west = residentCell(),
        .east = residentCell(),
        .north = residentCell(),
        .south = residentCell(),
    };

    const FluidRuleResult result = evaluateWaterRules(cells);
    REQUIRE(result.mutationCount == 0);
    REQUIRE(result.deferredCount == 1);
    REQUIRE(result.deferred[0] == FluidDirection::DOWN);
}

TEST_CASE("Unavailable flowing-water supports defer without provisional mutation",
          "[fluid][fluid-persistence][rules]") {
    FluidNeighborhood cells{
        .center = residentCell(BlockType::WATER, FluidState::flowing(3)),
        .down = residentCell(BlockType::STONE),
        .up = residentCell(),
        .west = residentCell(),
        .east = residentCell(),
        .north = residentCell(),
        .south = residentCell(),
    };

    SECTION("upper support") {
        cells.up = {};
        const FluidRuleResult result = evaluateWaterRules(cells);
        REQUIRE(result.mutationCount == 0);
        REQUIRE(result.deferredCount == 1);
        REQUIRE(result.deferred[0] == FluidDirection::UP);
    }

    SECTION("horizontal support") {
        cells.west = {};
        const FluidRuleResult result = evaluateWaterRules(cells);
        REQUIRE(result.mutationCount == 0);
        REQUIRE(result.deferredCount == 1);
        REQUIRE(result.deferred[0] == FluidDirection::WEST);
    }
}

TEST_CASE("RYCH rejects fluid state bytes that disagree with their water blocks",
          "[serialization][fluid-persistence]") {
    Chunk cube(ChunkPos{2, 4, -3});
    cube.fill(BlockType::STONE);
    cube.setBlock(1, 2, 3, BlockType::WATER);
    cube.setFluidState(1, 2, 3, FluidState::flowing(4));
    const std::vector<uint8_t> valid = ChunkSerializer::serialize(cube);
    REQUIRE(ChunkSerializer::deserialize(valid).has_value());

    const size_t fluidOffset = HEADER_SIZE + CHUNK_VOLUME;
    SECTION("explicit state on stone") {
        auto corrupt = valid;
        corrupt[fluidOffset + Chunk::index(0, 0, 0)] = FluidState::flowing(2).packed();
        refreshPayloadChecksum(corrupt);
        REQUIRE(ChunkSerializer::validatePayload(corrupt) == ChunkPayloadValidation::VALID);
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("falling level zero") {
        auto corrupt = valid;
        corrupt[fluidOffset + Chunk::index(1, 2, 3)] = FluidState::FALLING_MASK;
        REQUIRE_FALSE(FluidState::isValidPacked(FluidState::FALLING_MASK));
        refreshPayloadChecksum(corrupt);
        REQUIRE(ChunkSerializer::validatePayload(corrupt) == ChunkPayloadValidation::VALID);
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("explicit source instead of the implicit sentinel") {
        auto corrupt = valid;
        corrupt[fluidOffset + Chunk::index(1, 2, 3)] = FluidState::source().packed();
        refreshPayloadChecksum(corrupt);
        REQUIRE(ChunkSerializer::validatePayload(corrupt) == ChunkPayloadValidation::VALID);
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
}

TEST_CASE("RYCH payload checksum uses deterministic IEEE CRC-32",
          "[serialization][fluid-persistence][checksum]") {
    constexpr std::array<uint8_t, 9> reference{'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    REQUIRE(ChunkSerializer::payloadChecksum(reference) == 0xCBF43926U);
}

TEST_CASE("RYCH checksum rejects valid LZ4 block and fluid mutations and reports once",
          "[serialization][save][fluid-persistence][checksum]") {
    TempDir directory("cube_checksum");
    constexpr ChunkPos position{1, 4, 1};
    constexpr size_t waterIndex = Chunk::index(1, 2, 3);
    constexpr size_t oreIndex = Chunk::index(4, 5, 6);

    {
        SaveManager saves(directory.path());
        Chunk edited(position);
        edited.fill(BlockType::STONE);
        edited.setBlock(1, 2, 3, BlockType::WATER);
        edited.setFluidState(1, 2, 3, FluidState::flowing(4));
        edited.setBlock(4, 5, 6, BlockType::GOLD_ORE);
        edited.generated = true;
        saves.saveChunk(edited);
        REQUIRE(saves.flush());
    }

    const std::filesystem::path path = cubePath(directory.path(), position);
    std::vector<uint8_t> decompressed = readCompressedCube(path);
    REQUIRE(ChunkSerializer::validatePayload(decompressed) == ChunkPayloadValidation::VALID);

    SECTION("valid block value") {
        decompressed[HEADER_SIZE + oreIndex] = static_cast<uint8_t>(BlockType::DIAMOND_ORE);
        REQUIRE(decompressed[HEADER_SIZE + oreIndex] < static_cast<uint8_t>(BlockType::COUNT));
    }
    SECTION("valid fluid value") {
        const size_t fluidOffset = HEADER_SIZE + CHUNK_VOLUME;
        decompressed[fluidOffset + waterIndex] = FluidState::flowing(5).packed();
        REQUIRE(FluidState::isValidPacked(decompressed[fluidOffset + waterIndex]));
    }

    REQUIRE(ChunkSerializer::validatePayload(decompressed) ==
            ChunkPayloadValidation::CHECKSUM_MISMATCH);
    REQUIRE_FALSE(ChunkSerializer::deserialize(decompressed).has_value());
    writeCompressedCube(path, decompressed);
    REQUIRE(readCompressedCube(path) == decompressed);

    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    SaveManager corrupted(directory.path(), hooks);
    REQUIRE_FALSE(corrupted.loadChunk(position).has_value());
    REQUIRE_FALSE(corrupted.loadChunk(position).has_value());
    REQUIRE(hooks->loadFailuresReported.load() == 1);

    World world(42, 4);
    world.setSaveManager(&corrupted);
    const std::shared_ptr<Chunk> rebuilt = world.getChunk(position);
    ChunkGenerator generator(42);
    Chunk expected(position);
    generator.generateCube(expected);
    REQUIRE(rebuilt->copyBlocks() == expected.copyBlocks());
    REQUIRE(rebuilt->explicitFluidStates() == expected.explicitFluidStates());
    REQUIRE(hooks->loadFailuresReported.load() == 1);
}

TEST_CASE("World rejects direct cube access outside the supported vertical range",
          "[world][bounds][fluid-persistence]") {
    World world(42, 4);
    REQUIRE(world.getChunk({0, WORLD_MIN_CHUNK_Y - 1, 0}) == nullptr);
    REQUIRE(world.getChunk({0, WORLD_MAX_CHUNK_Y + 1, 0}) == nullptr);
    REQUIRE(world.getLoadedChunkCount() == 0);
}

TEST_CASE("Cube writes retry and retain the newest load shield after terminal failure",
          "[save][fluid-persistence][failure]") {
    TempDir directory("save_retry_and_shield");
    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    constexpr ChunkPos position{3, 4, 5};

    {
        SaveManager saves(directory.path(), hooks);
        Chunk original(position);
        original.setBlock(1, 2, 3, BlockType::GOLD_ORE);
        original.generated = true;
        saves.saveChunk(original);
        REQUIRE(saves.flush());

        Chunk retried(position);
        retried.setBlock(1, 2, 3, BlockType::IRON_ORE);
        retried.generated = true;
        hooks->writeFailuresRemaining.store(1);
        saves.saveChunk(retried);
        REQUIRE(saves.flush());
        REQUIRE(saves.loadChunk(position)->getBlock(1, 2, 3) == BlockType::IRON_ORE);

        Chunk unsaved(position);
        unsaved.setBlock(1, 2, 3, BlockType::DIAMOND_ORE);
        unsaved.generated = true;
        hooks->writeFailuresRemaining.store(100);
        saves.saveChunk(unsaved);
        REQUIRE_FALSE(saves.flush());
        const auto shielded = saves.loadChunk(position);
        REQUIRE(shielded.has_value());
        REQUIRE(shielded->getBlock(1, 2, 3) == BlockType::DIAMOND_ORE);
    }

    SaveManager reopened(directory.path());
    const auto durable = reopened.loadChunk(position);
    REQUIRE(durable.has_value());
    REQUIRE(durable->getBlock(1, 2, 3) == BlockType::IRON_ORE);
}

TEST_CASE("Atomic metadata and manifest replacement preserve prior durable state",
          "[save][fluid-persistence][failure]") {
    TempDir directory("atomic_save_replacement");
    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    const FluidBoundaryFrontier original{{15, 64, 3}, {16, 64, 3}};
    const FluidBoundaryFrontier replacement{{14, 65, 3}, {15, 65, 3}};

    {
        SaveManager saves(directory.path(), hooks);
        SaveManager::WorldMetadata first;
        first.seed = 11;
        first.spawnPos = Vec3{1.0f, 2.0f, 3.0f};
        first.worldTime = 4;
        REQUIRE(saves.saveMetadata(first));
        REQUIRE(saves.saveDeferredFluidFrontiers({original}));

        hooks->writeFailuresRemaining.store(100);
        SaveManager::WorldMetadata second;
        second.seed = 22;
        second.spawnPos = Vec3{4.0f, 5.0f, 6.0f};
        second.worldTime = 7;
        REQUIRE_FALSE(saves.saveMetadata(second));
        REQUIRE_FALSE(saves.saveDeferredFluidFrontiers({replacement}));
        REQUIRE(saves.loadMetadata()->seed == 11);
        REQUIRE(saves.loadDeferredFluidFrontiers() == std::vector<FluidBoundaryFrontier>{original});
    }

    SaveManager reopened(directory.path());
    REQUIRE(reopened.loadMetadata()->seed == 11);
    REQUIRE(reopened.loadDeferredFluidFrontiers() == std::vector<FluidBoundaryFrontier>{original});
}

TEST_CASE("Generator version three preserves metadata and isolates legacy cube data",
          "[save][migration][generator-version]") {
    TempDir directory("generator_v3_migration");
    constexpr ChunkPos position{7, 5, -9};
    {
        SaveManager current(directory.path());
        SaveManager::WorldMetadata metadata;
        metadata.seed = 9182;
        metadata.spawnPos = Vec3{12.0F, 91.0F, -33.0F};
        metadata.worldTime = 4455;
        metadata.player.yaw = 37.0F;
        metadata.player.inventory[2] = ItemStack{itemFromBlock(BlockType::OBSIDIAN), 1, 0};
        REQUIRE(current.saveMetadata(metadata));
        Chunk edited(position);
        edited.setBlock(3, 4, 5, BlockType::DIAMOND_ORE);
        current.saveChunk(edited);
        REQUIRE(current.flush());
    }

    const std::filesystem::path currentRegions =
        std::filesystem::path(directory.path()) / SaveManager::CURRENT_REGIONS_DIRECTORY;
    const std::filesystem::path legacyRegions = std::filesystem::path(directory.path()) / "regions";
    std::filesystem::rename(currentRegions, legacyRegions);
    const std::filesystem::path metadataPath =
        std::filesystem::path(directory.path()) / "metadata.json";
    {
        std::ifstream input(metadataPath);
        REQUIRE(input.is_open());
        std::string metadata((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
        const std::string currentVersion =
            "\"generatorVersion\": " + std::to_string(SaveManager::CURRENT_GENERATOR_VERSION);
        const size_t version = metadata.find(currentVersion);
        REQUIRE(version != std::string::npos);
        metadata.replace(version, currentVersion.size(), "\"generatorVersion\": 2");
        std::ofstream output(metadataPath, std::ios::trunc);
        REQUIRE(output.is_open());
        output << metadata;
        REQUIRE(output.good());
    }

    {
        SaveManager upgraded(directory.path());
        const auto metadata = upgraded.loadMetadata();
        REQUIRE(metadata.has_value());
        REQUIRE(metadata->generatorVersion == 2);
        REQUIRE(metadata->seed == 9182);
        REQUIRE(metadata->spawnPos == Vec3{12.0F, 91.0F, -33.0F});
        REQUIRE(metadata->worldTime == 4455);
        REQUIRE(metadata->player.yaw == 37.0F);
        REQUIRE(metadata->player.inventory[2] ==
                ItemStack{itemFromBlock(BlockType::OBSIDIAN), 1, 0});
        REQUIRE_FALSE(upgraded.loadChunk(position).has_value());
        REQUIRE(upgraded.savedSections({position.x, position.z}).empty());
        REQUIRE(upgraded.saveMetadata(*metadata));
    }

    REQUIRE(std::filesystem::exists(legacyRegions));
    REQUIRE(std::filesystem::exists(std::filesystem::path(directory.path()) /
                                    SaveManager::CURRENT_REGIONS_DIRECTORY));
    SaveManager reopened(directory.path());
    REQUIRE(reopened.loadMetadata()->generatorVersion == SaveManager::CURRENT_GENERATOR_VERSION);
}

TEST_CASE("Generated rapid and waterfall states survive cubic persistence",
          "[worldgen][save][fluid-persistence][waterfall][regression]") {
    TempDir directory("generated_flowing_water");
    ChunkGenerator generator(42);
    constexpr std::array<ChunkPos, 3> POSITIONS = {
        ChunkPos{-516, 5, 193},
        ChunkPos{-515, 5, 193},
        ChunkPos{-515, 4, 193},
    };

    std::array<std::vector<uint8_t>, POSITIONS.size()> expectedFluidStates;
    std::array<std::vector<BlockType>, POSITIONS.size()> expectedBlocks;
    std::array<bool, 8> flowingLevels{};
    size_t fallingCells = 0;
    {
        SaveManager saves(directory.path());
        for (size_t index = 0; index < POSITIONS.size(); ++index) {
            Chunk cube(POSITIONS[index]);
            generator.generateCube(cube);
            expectedBlocks[index] = cube.copyBlocks();
            expectedFluidStates[index] = cube.explicitFluidStates();
            for (int localY = 0; localY < CHUNK_EDGE; ++localY) {
                for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
                    for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                        if (cube.getBlock(localX, localY, localZ) != BlockType::WATER)
                            continue;
                        const FluidState state = cube.getFluidState(localX, localY, localZ);
                        if (state.isFalling()) {
                            ++fallingCells;
                        } else if (!state.isSource()) {
                            flowingLevels[state.level()] = true;
                        }
                    }
                }
            }
            saves.saveChunk(cube);
        }
        REQUIRE(saves.flush());
    }

    REQUIRE(fallingCells > 0);
    for (uint8_t level = 1; level <= 7; ++level) {
        CAPTURE(level);
        REQUIRE(flowingLevels[level]);
    }

    SaveManager reopened(directory.path());
    for (size_t index = 0; index < POSITIONS.size(); ++index) {
        CAPTURE(POSITIONS[index].x, POSITIONS[index].y, POSITIONS[index].z);
        const std::optional<Chunk> restored = reopened.loadChunk(POSITIONS[index]);
        REQUIRE(restored.has_value());
        REQUIRE(restored->copyBlocks() == expectedBlocks[index]);
        REQUIRE(restored->explicitFluidStates() == expectedFluidStates[index]);
    }
}

TEST_CASE("Valid orphaned v4 cube filenames rebuild the edited section manifest",
          "[save][fluid-persistence][recovery]") {
    TempDir directory("orphaned_cube_recovery");
    constexpr ChunkPos position{2, -3, 4};
    {
        SaveManager saves(directory.path());
        Chunk cube(position);
        cube.setBlock(4, 5, 6, BlockType::DIAMOND_ORE);
        cube.generated = true;
        saves.saveChunk(cube);
        REQUIRE(saves.flush());
    }

    REQUIRE(std::filesystem::remove(manifestPath(directory.path(), {position.x, position.z})));
    SaveManager reopened(directory.path());
    REQUIRE(reopened.savedSections({position.x, position.z}) == std::vector<int32_t>{position.y});
    REQUIRE(std::filesystem::exists(manifestPath(directory.path(), {position.x, position.z})));
    REQUIRE(reopened.loadChunk(position)->getBlock(4, 5, 6) == BlockType::DIAMOND_ORE);
}

TEST_CASE("Column manifests reject incompatible versions bounds and frontiers",
          "[save][fluid-persistence][validation]") {
    TempDir directory("manifest_validation");
    constexpr ColumnPos column{0, 0};

    SECTION("version") {
        writeManifest(directory.path(), column, "RYCM 2\ncolumn 0 0\nsection 4\n");
    }
    SECTION("section Y") {
        writeManifest(directory.path(), column,
                      "RYCM 1\ncolumn 0 0\nsection " + std::to_string(WORLD_MAX_CHUNK_Y + 1) +
                          "\n");
    }
    SECTION("frontier Y") {
        writeManifest(directory.path(), column,
                      "RYCM 1\ncolumn 0 0\nfrontier 0 " + std::to_string(WORLD_MAX_Y + 1) +
                          " 0 1 " + std::to_string(WORLD_MAX_Y + 1) + " 0\n");
    }
    SECTION("frontier adjacency") {
        writeManifest(directory.path(), column, "RYCM 1\ncolumn 0 0\nfrontier 0 64 0 2 64 0\n");
    }
    SECTION("frontier within one cube") {
        writeManifest(directory.path(), column, "RYCM 1\ncolumn 0 0\nfrontier 2 64 2 3 64 2\n");
    }

    SaveManager saves(directory.path());
    REQUIRE(saves.savedSections(column).empty());
    REQUIRE(saves.loadDeferredFluidFrontiers().empty());
}

TEST_CASE("Activated fluid frontiers resume across a real world restart",
          "[world][save][fluid-persistence][restart]") {
    TempDir directory("fluid_restart");
    constexpr ChunkPos leftPosition{0, 4, 0};
    constexpr ChunkPos rightPosition{1, 4, 0};
    constexpr BlockPos source{15, 65, 8};
    constexpr BlockPos resumed{16, 65, 8};

    {
        SaveManager saves(directory.path());
        Chunk left(leftPosition);
        left.fill(BlockType::AIR);
        left.setBlock(15, 0, 8, BlockType::STONE);
        left.generated = true;
        Chunk right(rightPosition);
        right.fill(BlockType::AIR);
        right.setBlock(0, 0, 8, BlockType::STONE);
        right.generated = true;
        saves.saveChunk(left);
        saves.saveChunk(right);
        REQUIRE(saves.flush());

        World world(42, 4);
        world.setSaveManager(&saves);
        REQUIRE(world.getChunk(leftPosition) != nullptr);
        world.setBlock(source.x, source.y, source.z, BlockType::WATER);
        for (uint32_t tick = 0; tick < WATER_UPDATE_DELAY_TICKS; ++tick)
            world.tickFluids(1.0 / FLUID_TICKS_PER_SECOND);
        REQUIRE(world.getPendingFluidCount() > 0);
        REQUIRE(world.saveModifiedChunks());
        REQUIRE(saves.flush());
        REQUIRE_FALSE(saves.loadDeferredFluidFrontiers().empty());
    }

    SaveManager reopened(directory.path());
    REQUIRE_FALSE(reopened.loadDeferredFluidFrontiers().empty());
    World restored(42, 4);
    restored.setSaveManager(&reopened);
    REQUIRE(restored.getChunk(leftPosition) != nullptr);
    REQUIRE(restored.getChunk(rightPosition) != nullptr);
    for (uint32_t tick = 0; tick <= WATER_UPDATE_DELAY_TICKS; ++tick)
        restored.tickFluids(1.0 / FLUID_TICKS_PER_SECOND);
    REQUIRE(restored.getBlockIfLoaded(resumed.x, resumed.y, resumed.z) == BlockType::WATER);
    REQUIRE(restored.readFluidCell(resumed).state == FluidState::flowing(1));
}

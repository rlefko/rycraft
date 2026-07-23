#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/counter_rng.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/block_properties.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/features.hpp>
#include <world/fluid.hpp>
#include <world/furnace.hpp>
#include <world/learned_terrain.hpp>
#include <world/light_engine.hpp>
#include <world/noise.hpp>
#include <world/ores.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/structures.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <numbers>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ===========================================================================
// Coordinate and math tests
// ===========================================================================

TEST_CASE("Cubic world constants describe the supported vertical range", "[chunk][coords]") {
    STATIC_REQUIRE(CHUNK_EDGE == 16);
    STATIC_REQUIRE(CHUNK_VOLUME == 4096);
    STATIC_REQUIRE(WORLD_MIN_Y == -128);
    STATIC_REQUIRE(WORLD_MAX_Y == 1407);
    STATIC_REQUIRE(WORLD_MIN_CHUNK_Y == -8);
    STATIC_REQUIRE(WORLD_MAX_CHUNK_Y == 87);
    STATIC_REQUIRE(WORLD_VERTICAL_CHUNKS == 96);
    STATIC_REQUIRE(VerticalSectionMask::WORD_COUNT == 2);
    STATIC_REQUIRE(SEA_LEVEL == 64);
}

TEST_CASE("World instances retain distinct renderer identities", "[world][render]") {
    World first(42, 4);
    World second(42, 4);

    REQUIRE(first.getSeed() == second.getSeed());
    REQUIRE(first.instanceId() != second.instanceId());
}

TEST_CASE("Vertical section masks span both words without undefined shifts",
          "[chunk][coords][regression]") {
    VerticalSectionMask sections;
    REQUIRE(sections.empty());
    REQUIRE(sections.highestSection() == WORLD_MIN_CHUNK_Y - 1);
    REQUIRE_FALSE(sections.contains(WORLD_MIN_CHUNK_Y - 1));
    REQUIRE_FALSE(sections.contains(WORLD_MAX_CHUNK_Y + 1));
    REQUIRE_FALSE(sections.containsRange(WORLD_MIN_CHUNK_Y, WORLD_MAX_CHUNK_Y));

    constexpr int32_t LAST_FIRST_WORD_SECTION = WORLD_MIN_CHUNK_Y + 63;
    constexpr int32_t FIRST_SECOND_WORD_SECTION = WORLD_MIN_CHUNK_Y + 64;
    for (int32_t section = LAST_FIRST_WORD_SECTION - 1; section <= FIRST_SECOND_WORD_SECTION + 1;
         ++section) {
        sections.set(section);
    }

    REQUIRE(sections.contains(LAST_FIRST_WORD_SECTION));
    REQUIRE(sections.contains(FIRST_SECOND_WORD_SECTION));
    REQUIRE(sections.highestSection() == FIRST_SECOND_WORD_SECTION + 1);
    REQUIRE(sections.containsRange(LAST_FIRST_WORD_SECTION - 1, FIRST_SECOND_WORD_SECTION + 1));
    REQUIRE_FALSE(sections.containsRange(FIRST_SECOND_WORD_SECTION, FIRST_SECOND_WORD_SECTION + 2));

    std::vector<int32_t> ascending;
    REQUIRE(sections.visitSetSections(LAST_FIRST_WORD_SECTION - 4, FIRST_SECOND_WORD_SECTION + 4,
                                      [&](int32_t section) {
                                          ascending.push_back(section);
                                          return true;
                                      }) == 4);
    REQUIRE(ascending == std::vector<int32_t>{LAST_FIRST_WORD_SECTION - 1, LAST_FIRST_WORD_SECTION,
                                              FIRST_SECOND_WORD_SECTION,
                                              FIRST_SECOND_WORD_SECTION + 1});
    std::vector<int32_t> descending;
    REQUIRE(sections.visitSetSectionsDescending(
                LAST_FIRST_WORD_SECTION - 4, FIRST_SECOND_WORD_SECTION + 4, [&](int32_t section) {
                    descending.push_back(section);
                    return descending.size() < 3;
                }) == 3);
    REQUIRE(descending == std::vector<int32_t>{FIRST_SECOND_WORD_SECTION + 1,
                                               FIRST_SECOND_WORD_SECTION, LAST_FIRST_WORD_SECTION});

    sections.reset(FIRST_SECOND_WORD_SECTION);
    REQUIRE_FALSE(sections.contains(FIRST_SECOND_WORD_SECTION));
    REQUIRE_FALSE(sections.containsRange(LAST_FIRST_WORD_SECTION, FIRST_SECOND_WORD_SECTION + 1));
    sections.reset(WORLD_MIN_CHUNK_Y - 1);
    sections.reset(WORLD_MAX_CHUNK_Y + 1);
    REQUIRE_FALSE(sections.empty());

    for (int32_t section = WORLD_MIN_CHUNK_Y; section <= WORLD_MAX_CHUNK_Y; ++section) {
        sections.set(section);
    }
    REQUIRE(sections.highestSection() == WORLD_MAX_CHUNK_Y);
    REQUIRE(sections.containsRange(WORLD_MIN_CHUNK_Y, WORLD_MAX_CHUNK_Y));
    REQUIRE_FALSE(sections.containsRange(WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_Y));
    for (int32_t section = WORLD_MIN_CHUNK_Y; section <= WORLD_MAX_CHUNK_Y; ++section) {
        sections.reset(section);
    }
    REQUIRE(sections.empty());
}

TEST_CASE("World exposes failures latched by shared far generation authority",
          "[world][generator-v4][failure][regression]") {
    worldgen::learned::GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x31U);
    identity.runtimeHash.fill(0x72U);
    auto backend = std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, std::filesystem::path{}, std::move(backend));
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, std::move(authority), worldgen::learned::AuthorityQuality::FINAL);
    World world(identity.seed, MIN_RENDER_DISTANCE_CHUNKS, MAX_LOADED_CUBES, context);

    context->latchFailure({.code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                           .message = "Far terrain authority failed",
                           .retriable = true});
    REQUIRE(world.generationFailure() == "Far terrain authority failed");
    REQUIRE(world.retryGeneration());
    REQUIRE_FALSE(world.generationFailure());
}

TEST_CASE("Floor division and local coordinates are canonical for negatives", "[chunk][coords]") {
    const std::array<int64_t, 10> world = {-33, -32, -17, -16, -1, 0, 15, 16, 31, 32};
    const std::array<int64_t, 10> chunks = {-3, -2, -2, -1, -1, 0, 0, 1, 1, 2};
    const std::array<int32_t, 10> locals = {15, 0, 15, 0, 15, 0, 15, 0, 15, 0};

    for (size_t i = 0; i < world.size(); ++i) {
        REQUIRE(Chunk::worldToChunk(world[i]) == chunks[i]);
        REQUIRE(Chunk::worldToLocal(world[i]) == locals[i]);
    }

    REQUIRE(Chunk::worldToChunkY(WORLD_MIN_Y) == WORLD_MIN_CHUNK_Y);
    REQUIRE(Chunk::worldToChunkY(-1) == -1);
    REQUIRE(Chunk::worldToChunkY(0) == 0);
    REQUIRE(Chunk::worldToChunkY(WORLD_MAX_Y) == WORLD_MAX_CHUNK_Y);
    REQUIRE(Chunk::worldToLocalY(-1) == 15);
    REQUIRE(Chunk::worldToLocalY(WORLD_MAX_Y) == 15);
}

TEST_CASE("Floating coordinate floors retain neighbor headroom at int64 limits",
          "[chunk][coords][regression]") {
    constexpr int64_t minimum = std::numeric_limits<int64_t>::min() + 1;
    constexpr int64_t maximum = std::numeric_limits<int64_t>::max() - 1;
    REQUIRE(world_coord::floorToNeighborSafeInt64(-1.01) == -2);
    REQUIRE(world_coord::floorToNeighborSafeInt64(1.99) == 1);
    REQUIRE(world_coord::floorToNeighborSafeInt64(std::numeric_limits<double>::quiet_NaN()) == 0);
    REQUIRE(world_coord::floorToNeighborSafeInt64(-std::numeric_limits<double>::infinity()) ==
            minimum);
    REQUIRE(world_coord::floorToNeighborSafeInt64(std::numeric_limits<double>::infinity()) ==
            maximum);
    REQUIRE(world_coord::floorToNeighborSafeInt64(-0x1p63) == minimum);
    REQUIRE(world_coord::floorToNeighborSafeInt64(0x1p63) == maximum);
    REQUIRE(world_coord::floorToNeighborSafeInt64(std::nextafter(0x1p63, 0.0)) <= maximum);
    REQUIRE(world_coord::floorToNeighborSafeInt64(std::nextafter(-0x1p63, 0.0)) >= minimum);
}

TEST_CASE("Three dimensional positions compare and hash distinctly", "[chunk][coords]") {
    std::unordered_map<ChunkPos, int> cubes;
    cubes[{5, -3, 7}] = 1;
    cubes[{5, -2, 7}] = 2;
    cubes[{-5, -3, 7}] = 3;
    cubes[{5, -3, -7}] = 4;

    REQUIRE(cubes.size() == 4);
    REQUIRE(cubes.at({5, -3, 7}) == 1);
    REQUIRE(cubes.at({5, -2, 7}) == 2);
    REQUIRE(ChunkPos{5, -3, 7} != ChunkPos{5, 7, -3});

    std::unordered_set<ColumnPos> columns{{5, 7}, {5, -7}, {-5, 7}};
    REQUIRE(columns.size() == 3);
    REQUIRE(columns.contains({5, 7}));

    std::unordered_set<BlockPos> blocks{{-1, -1, -1}, {-1, 0, -1}, {0, -1, -1}};
    REQUIRE(blocks.size() == 3);
}

TEST_CASE("CounterRng addresses full-width coordinates and candidate indices",
          "[counter-rng][random]") {
    constexpr CounterRng random(0x123456789ABCDEF0ULL);
    constexpr uint64_t stream = 0x4F52455F54455354ULL;
    constexpr int64_t aliasDistance = int64_t{1} << 32;

    const CounterRng::Block origin = random.block(stream, -17, -3, 29, 7);
    REQUIRE(origin == random.block(stream, -17, -3, 29, 7));
    REQUIRE(origin != random.block(stream, -17 + aliasDistance, -3, 29, 7));
    REQUIRE(origin != random.block(stream, -17, -3, 29 + aliasDistance, 7));
    REQUIRE(origin != random.block(stream + 1, -17, -3, 29, 7));
    REQUIRE(origin != random.block(stream, -17, -3, 29, 8));

    const uint32_t later = random.u32(stream, -17, -3, 29, 99);
    REQUIRE(random.u32(stream, -17, -3, 29, 7) == origin[0]);
    REQUIRE(random.u32(stream, -17, -3, 29, 99) == later);
}

TEST_CASE("Vec3 arithmetic and AABB intersection remain stable", "[math]") {
    Vec3 a{1.f, 2.f, 3.f};
    Vec3 b{-2.f, 4.f, 1.f};
    REQUIRE(a + b == Vec3{-1.f, 6.f, 4.f});
    REQUIRE(a.dot(b) == Catch::Approx(9.f));
    REQUIRE(a.cross(b) == Vec3{-10.f, -7.f, 8.f});
    REQUIRE(a.normalize().length() == Catch::Approx(1.f));

    AABB first{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
    AABB overlap{{0.5f, 0.5f, 0.5f}, {2.f, 2.f, 2.f}};
    AABB separate{{2.f, 2.f, 2.f}, {3.f, 3.f, 3.f}};
    REQUIRE(first.intersects(overlap));
    REQUIRE_FALSE(first.intersects(separate));
}

// ===========================================================================
// Block, noise, and climate tests
// ===========================================================================

TEST_CASE("Persisted block identifiers retain their original values", "[block]") {
    REQUIRE(static_cast<int>(BlockType::AIR) == 0);
    REQUIRE(static_cast<int>(BlockType::STONE) == 1);
    REQUIRE(static_cast<int>(BlockType::WATER) == 6);
    REQUIRE(static_cast<int>(BlockType::BEDROCK) == 7);
    REQUIRE(static_cast<int>(BlockType::GLASS) == 16);
    REQUIRE(static_cast<int>(BlockType::COBBLESTONE) == 17);
    REQUIRE(static_cast<int>(BlockType::ICE) == 33);
    REQUIRE(static_cast<int>(BlockType::MUD) == 34);
    REQUIRE(static_cast<int>(BlockType::OBSIDIAN) == 40);
    REQUIRE(static_cast<int>(BlockType::ACACIA_LOG) == 41);
    REQUIRE(static_cast<int>(BlockType::FERN) == 51);
    REQUIRE(static_cast<int>(BlockType::ANDESITE) == 57);
    REQUIRE(static_cast<int>(BlockType::CRAFTING_TABLE) == 58);
    REQUIRE(static_cast<int>(BlockType::FURNACE) == 59);
    REQUIRE(static_cast<int>(BlockType::FURNACE_LIT) == 60);
    REQUIRE(static_cast<int>(BlockType::TORCH) == 61);
    REQUIRE(static_cast<int>(BlockType::CHEST) == 62);
    REQUIRE(static_cast<int>(BlockType::WOOL) == 63);
    REQUIRE(static_cast<int>(BlockType::BED) == 64);
    REQUIRE(static_cast<int>(BlockType::COUNT) == 65);
}

TEST_CASE("Block survival data gates mining by hardness tool and tier", "[block][survival]") {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockType type = static_cast<BlockType>(index);
        const BlockDefinition& definition = blockDefinition(type);
        REQUIRE(blockHardness(type) == definition.hardness);
        if (type == BlockType::BEDROCK) {
            REQUIRE(definition.hardness < 0.0f);
        } else {
            REQUIRE(definition.hardness >= 0.0f);
        }
        if (definition.solid && definition.targetable && type != BlockType::BEDROCK) {
            REQUIRE(definition.hardness > 0.0f);
        }
        if (definition.minimumTier != ToolTier::NONE) {
            REQUIRE(definition.tool != ToolClass::NONE);
        }
        if (definition.interactable) {
            REQUIRE(definition.solid);
        }
    }

    REQUIRE(blockDefinition(BlockType::STONE).tool == ToolClass::PICKAXE);
    REQUIRE(blockDefinition(BlockType::STONE).minimumTier == ToolTier::WOOD);
    REQUIRE(blockDefinition(BlockType::IRON_ORE).minimumTier == ToolTier::STONE);
    REQUIRE(blockDefinition(BlockType::GOLD_ORE).minimumTier == ToolTier::IRON);
    REQUIRE(blockDefinition(BlockType::DIAMOND_ORE).minimumTier == ToolTier::IRON);
    REQUIRE(blockDefinition(BlockType::OBSIDIAN).hardness == 50.0f);
    REQUIRE(blockDefinition(BlockType::DIRT).tool == ToolClass::SHOVEL);
    REQUIRE(blockDefinition(BlockType::LOG).tool == ToolClass::AXE);
    REQUIRE(blockDefinition(BlockType::TALL_GRASS).hardness == 0.0f);
    REQUIRE(isInteractable(BlockType::CRAFTING_TABLE));
    REQUIRE(isInteractable(BlockType::FURNACE));
    REQUIRE(isInteractable(BlockType::FURNACE_LIT));
    REQUIRE_FALSE(isInteractable(BlockType::STONE));
}

TEST_CASE("New workshop blocks render light and break like their materials", "[block]") {
    REQUIRE(rendersAsCube(BlockType::CRAFTING_TABLE));
    REQUIRE(rendersAsCube(BlockType::FURNACE));
    REQUIRE(rendersAsCube(BlockType::FURNACE_LIT));
    REQUIRE(blockDefinition(BlockType::CRAFTING_TABLE).material == BlockMaterial::WOOD);
    REQUIRE(blockDefinition(BlockType::FURNACE).material == BlockMaterial::ROCK);
    REQUIRE(blockLightEmission(BlockType::FURNACE) == 0);
    REQUIRE(blockLightEmission(BlockType::FURNACE_LIT) == 13);
    REQUIRE_FALSE(isFlora(BlockType::TORCH));
    REQUIRE(isFloorTorch(BlockType::TORCH));
    REQUIRE(rendersAsCross(BlockType::TORCH));
    REQUIRE_FALSE(isSolid(BlockType::TORCH));
    REQUIRE(blockLightEmission(BlockType::TORCH) == 14);
    REQUIRE(isEmissive(BlockType::TORCH));
    REQUIRE(rendersAsLowBox(BlockType::BED));
    REQUIRE_FALSE(isOpaque(BlockType::BED));
    REQUIRE(blockCollisionHeight(BlockType::BED) == BED_COLLISION_HEIGHT);
    REQUIRE_FALSE(hasFullBlockCollision(BlockType::BED));
}

TEST_CASE("Block properties distinguish flora liquids and cubes", "[block]") {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const auto& definition = blockDefinition(static_cast<BlockType>(index));
        REQUIRE(definition.sound != BlockSound::UNDEFINED);
        REQUIRE(definition.material != BlockMaterial::UNDEFINED);
    }

    for (BlockType block : {BlockType::DEAD_BUSH, BlockType::TALL_GRASS, BlockType::FLOWER_YELLOW,
                            BlockType::FLOWER_RED, BlockType::MUSHROOM_BROWN,
                            BlockType::MUSHROOM_RED, BlockType::REED}) {
        REQUIRE(isFlora(block));
        REQUIRE_FALSE(isSolid(block));
        REQUIRE_FALSE(isOpaque(block));
        REQUIRE(isTargetable(block));
    }

    for (BlockType block : {BlockType::WATER, BlockType::LAVA}) {
        REQUIRE(isLiquid(block));
        REQUIRE_FALSE(isSolid(block));
        REQUIRE_FALSE(isTargetable(block));
    }
    REQUIRE_FALSE(isOpaque(BlockType::WATER));
    REQUIRE(isOpaque(BlockType::LAVA));
    REQUIRE(rendersAsCube(BlockType::LAVA));
    REQUIRE_FALSE(rendersAsCube(BlockType::WATER));

    for (BlockType block : {BlockType::FERN, BlockType::SHRUB, BlockType::CATTAIL,
                            BlockType::FLOWER_BLUE, BlockType::SUCCULENT, BlockType::LILY_PAD}) {
        REQUIRE(isFlora(block));
        REQUIRE_FALSE(isSolid(block));
    }
    REQUIRE(blockDefinition(BlockType::LILY_PAD).renderShape == BlockRenderShape::FLAT);
    REQUIRE(blockDefinition(BlockType::TORCH).renderShape == BlockRenderShape::TORCH_CROSS);
    REQUIRE(isLeafBlock(BlockType::MANGROVE_LEAVES));
    REQUIRE(isSolid(BlockType::BASALT));
    REQUIRE(isOpaque(BlockType::BASALT));
}

TEST_CASE("Simplex noise is deterministic and seed dependent", "[noise]") {
    SimplexNoise first(42);
    SimplexNoise same(42);
    SimplexNoise different(43);

    bool foundSeedDifference = false;
    for (int x = -20; x <= 20; ++x) {
        const double nx = static_cast<double>(x) * 0.37;
        const double a = first.noise2D(nx, 7.25);
        const double b = same.noise2D(nx, 7.25);
        const double c = different.noise2D(nx, 7.25);
        REQUIRE(a == b);
        REQUIRE(a >= -1.0);
        REQUIRE(a <= 1.0);
        foundSeedDifference = foundSeedDifference || a != c;
    }
    REQUIRE(foundSeedDifference);
}

TEST_CASE("Simplex three dimensional and octave samples stay bounded", "[noise]") {
    SimplexNoise noise(123);
    for (int z = -3; z <= 3; ++z) {
        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                const double value = noise.noise3D(x * 0.3, y * 0.3, z * 0.3);
                REQUIRE(value >= -1.0);
                REQUIRE(value <= 1.0);
                REQUIRE(value == noise.noise3D(x * 0.3, y * 0.3, z * 0.3));
            }
        }
    }

    for (int i = 0; i < 30; ++i) {
        const double octave = noise.octave2D(i * 0.17, i * -0.11, 6, 0.5, 2.0);
        const double ridged = noise.ridged2D(i * 0.17, i * -0.11, 4, 0.5, 2.0);
        REQUIRE(octave >= -1.0);
        REQUIRE(octave <= 1.0);
        REQUIRE(ridged >= 0.0);
        REQUIRE(ridged <= 1.0);
    }
}

TEST_CASE("Climate columns are deterministic continuous samples", "[climate]") {
    ClimateSampler first(123);
    ClimateSampler same(123);
    for (int i = -50; i <= 50; i += 7) {
        const ColumnShape a = first.shapeColumn(i * 31.0, i * 17.0);
        const ColumnShape b = same.shapeColumn(i * 31.0, i * 17.0);
        REQUIRE(a.height == b.height);
        REQUIRE(a.climate.temperature == b.climate.temperature);
        REQUIRE(a.climate.humidity == b.climate.humidity);
        REQUIRE(std::isfinite(a.height));
        REQUIRE(a.height >= WORLD_MIN_Y + 2);
        REQUIRE(a.height <= WORLD_MAX_Y);
        REQUIRE(a.detailAmp >= 0.0);
        REQUIRE(a.ravineEdge >= 0.0);
        REQUIRE(a.ravineEdge <= 1.0);
    }
}

TEST_CASE("Biome selection responds to climate and terrain", "[climate][biome]") {
    ColumnShape shape;
    shape.height = 40.0;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::DEEP_OCEAN);
    shape.height = 58.0;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::OCEAN);

    shape.height = 80.0;
    shape.climate.temperature = -0.6;
    shape.climate.humidity = 0.3;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::TAIGA);
    shape.climate.humidity = -0.3;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::ICE_SPIKES);

    shape.climate.temperature = 0.7;
    shape.climate.humidity = -0.5;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::DESERT);

    shape.climate.temperature = 0.3;
    shape.climate.humidity = 0.4;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::FOREST);
}

TEST_CASE("Terrain height remains finite across climate transitions", "[climate][worldgen]") {
    ChunkGenerator generator(4242);
    GenScratch scratch;
    scratch.reset(&generator);
    constexpr std::array<int, 5> coordinates{{-256, -97, 0, 113, 256}};
    for (int x : coordinates) {
        const double height = generator.baseHeightAt(x, 100, scratch);
        REQUIRE(std::isfinite(height));
        REQUIRE(height >= WORLD_MIN_Y + 2);
        REQUIRE(height <= WORLD_MAX_Y);
    }
}

// ===========================================================================
// Sparse cubic chunks and deterministic generation
// ===========================================================================

TEST_CASE("A new cube uses uniform air storage", "[chunk][storage]") {
    Chunk cube(ChunkPos{5, -3, 7});
    REQUIRE(cube.pos() == ChunkPos{5, -3, 7});
    REQUIRE(cube.isUniform());
    REQUIRE(cube.uniformBlock() == BlockType::AIR);
    REQUIRE(cube.denseBlocks().empty());
    REQUIRE(cube.copyBlocks().size() == static_cast<size_t>(CHUNK_VOLUME));
    REQUIRE(cube.getBlock(0, 0, 0) == BlockType::AIR);
    REQUIRE(cube.getBlock(15, 15, 15) == BlockType::AIR);
}

TEST_CASE("Editing a uniform cube materializes and can compact it", "[chunk][storage]") {
    Chunk cube(ChunkPos{1, 4, -2});
    cube.fill(BlockType::STONE);
    REQUIRE(cube.isUniform());
    REQUIRE(cube.uniformBlock() == BlockType::STONE);

    cube.setBlock(8, 9, 10, BlockType::DIRT);
    REQUIRE_FALSE(cube.isUniform());
    REQUIRE(cube.denseBlocks().size() == static_cast<size_t>(CHUNK_VOLUME));
    REQUIRE(cube.getBlock(8, 9, 10) == BlockType::DIRT);
    REQUIRE(cube.getBlock(0, 0, 0) == BlockType::STONE);

    cube.setBlock(8, 9, 10, BlockType::STONE);
    cube.compactStorage();
    REQUIRE(cube.isUniform());
    REQUIRE(cube.uniformBlock() == BlockType::STONE);
}

TEST_CASE("Cube access rejects local coordinates outside all six faces", "[chunk]") {
    Chunk cube(ChunkPos{0, 0, 0});
    cube.setBlock(8, 8, 8, BlockType::STONE);
    REQUIRE(cube.getBlock(8, 8, 8) == BlockType::STONE);
    REQUIRE(cube.getBlock(-1, 8, 8) == BlockType::AIR);
    REQUIRE(cube.getBlock(16, 8, 8) == BlockType::AIR);
    REQUIRE(cube.getBlock(8, -1, 8) == BlockType::AIR);
    REQUIRE(cube.getBlock(8, 16, 8) == BlockType::AIR);
    REQUIRE(cube.getBlock(8, 8, -1) == BlockType::AIR);
    REQUIRE(cube.getBlock(8, 8, 16) == BlockType::AIR);
}

TEST_CASE("Cube world access maps negative coordinates into local cells", "[chunk][coords]") {
    Chunk cube(ChunkPos{-2, -1, 3});
    cube.setBlockWorld(-17, -1, 63, BlockType::GRASS);
    REQUIRE(cube.getBlock(15, 15, 15) == BlockType::GRASS);
    REQUIRE(cube.getBlockWorld(-17, -1, 63) == BlockType::GRASS);
}

TEST_CASE("Cube position and bounds include the vertical section", "[chunk]") {
    Chunk cube(ChunkPos{1, -3, -1});
    const Vec3 position = cube.getWorldPosition();
    REQUIRE(position.x == Catch::Approx(16.f));
    REQUIRE(position.y == Catch::Approx(-48.f));
    REQUIRE(position.z == Catch::Approx(-16.f));

    const AABB bounds = cube.getAABB();
    REQUIRE(bounds.min == Vec3{16.f, -48.f, -16.f});
    REQUIRE(bounds.max == Vec3{32.f, -32.f, 0.f});
}

TEST_CASE("Fluid states remain implicit until a flowing cell is written", "[chunk][fluid]") {
    Chunk cube(ChunkPos{0, 4, 0});
    cube.setBlock(2, 3, 4, BlockType::WATER);
    REQUIRE(cube.getFluidState(2, 3, 4).isSource());
    REQUIRE_FALSE(cube.hasExplicitFluidStates());

    cube.setFluidState(2, 3, 4, FluidState::flowing(5));
    REQUIRE(cube.hasExplicitFluidStates());
    REQUIRE(cube.explicitFluidStates().size() == static_cast<size_t>(CHUNK_VOLUME));
    REQUIRE(cube.getFluidState(2, 3, 4) == FluidState::flowing(5));
    REQUIRE(cube.getFluidState(1, 3, 4).isSource());
}

namespace {

size_t oreCount(const Chunk& cube) {
    const std::vector<BlockType> blocks = cube.copyBlocks();
    return static_cast<size_t>(std::count_if(blocks.begin(), blocks.end(), [](BlockType block) {
        return block == BlockType::COAL_ORE || block == BlockType::IRON_ORE ||
               block == BlockType::GOLD_ORE || block == BlockType::DIAMOND_ORE;
    }));
}

std::array<int64_t, 6> structureRandomSignature(const StructurePlacement& placement,
                                                int64_t regionX, int64_t regionZ) {
    const int64_t anchorChunkX = Chunk::worldToChunk(placement.anchorX);
    const int64_t anchorChunkZ = Chunk::worldToChunk(placement.anchorZ);
    return {
        anchorChunkX - regionX * STRUCTURE_REGION_CHUNKS,
        anchorChunkZ - regionZ * STRUCTURE_REGION_CHUNKS,
        Chunk::worldToLocal(placement.anchorX),
        Chunk::worldToLocal(placement.anchorZ),
        static_cast<int64_t>(placement.kind),
        placement.rotation,
    };
}

bool samePlacement(const StructurePlacement& left, const StructurePlacement& right) {
    return left.valid == right.valid && left.kind == right.kind &&
           left.rotation == right.rotation && left.anchorX == right.anchorX &&
           left.anchorZ == right.anchorZ && left.floorY == right.floorY &&
           left.halfX == right.halfX && left.halfZ == right.halfZ;
}

} // namespace

TEST_CASE("Ore anchors use full-width coordinates and are order independent",
          "[ore-rng][worldgen]") {
    constexpr int64_t aliasDistance = int64_t{1} << 32;
    constexpr ChunkPos nearPosition{-13, -3, 21};
    constexpr ChunkPos farPosition{nearPosition.x + aliasDistance, nearPosition.y,
                                   nearPosition.z + aliasDistance};

    OrePlacer forwardPlacer(808);
    Chunk nearForward(nearPosition);
    Chunk farForward(farPosition);
    nearForward.fill(BlockType::STONE);
    farForward.fill(BlockType::STONE);
    forwardPlacer.place(nearForward);
    forwardPlacer.place(farForward);

    OrePlacer reversePlacer(808);
    Chunk nearReverse(nearPosition);
    Chunk farReverse(farPosition);
    nearReverse.fill(BlockType::STONE);
    farReverse.fill(BlockType::STONE);
    reversePlacer.place(farReverse);
    reversePlacer.place(nearReverse);

    REQUIRE(nearForward.copyBlocks() == nearReverse.copyBlocks());
    REQUIRE(farForward.copyBlocks() == farReverse.copyBlocks());
    REQUIRE(oreCount(nearForward) > 0);
    REQUIRE(oreCount(farForward) > 0);
    REQUIRE(nearForward.copyBlocks() != farForward.copyBlocks());

    OrePlacer differentSeed(809);
    Chunk changed(nearPosition);
    changed.fill(BlockType::STONE);
    differentSeed.place(changed);
    REQUIRE(changed.copyBlocks() != nearForward.copyBlocks());
}

TEST_CASE("Structure candidates use full-width region coordinates and stable order",
          "[structure-rng][worldgen]") {
    constexpr int64_t aliasDistance = int64_t{1} << 32;
    constexpr ColumnPos nearRegion{-7, 11};
    constexpr ColumnPos farRegion{nearRegion.x + aliasDistance, nearRegion.z + aliasDistance};
    ChunkGenerator generator(112233);
    StructurePlacer structures(112233);

    GenScratch forwardScratch;
    forwardScratch.reset(&generator);
    const StructurePlacement nearForward =
        structures.regionPlacement(nearRegion.x, nearRegion.z, generator, forwardScratch);
    const StructurePlacement farForward =
        structures.regionPlacement(farRegion.x, farRegion.z, generator, forwardScratch);

    GenScratch reverseScratch;
    reverseScratch.reset(&generator);
    const StructurePlacement farReverse =
        structures.regionPlacement(farRegion.x, farRegion.z, generator, reverseScratch);
    const StructurePlacement nearReverse =
        structures.regionPlacement(nearRegion.x, nearRegion.z, generator, reverseScratch);

    REQUIRE(samePlacement(nearForward, nearReverse));
    REQUIRE(samePlacement(farForward, farReverse));
    REQUIRE(structureRandomSignature(nearForward, nearRegion.x, nearRegion.z) !=
            structureRandomSignature(farForward, farRegion.x, farRegion.z));

    for (const auto& [placement, region] :
         {std::pair{nearForward, nearRegion}, std::pair{farForward, farRegion}}) {
        const auto signature = structureRandomSignature(placement, region.x, region.z);
        REQUIRE(signature[0] >= 1);
        REQUIRE(signature[0] <= 6);
        REQUIRE(signature[1] >= 1);
        REQUIRE(signature[1] <= 6);
        REQUIRE(signature[2] >= 2);
        REQUIRE(signature[2] <= 13);
        REQUIRE(signature[3] >= 2);
        REQUIRE(signature[3] <= 13);
        REQUIRE(signature[5] >= 0);
        REQUIRE(signature[5] <= 3);
    }
}

TEST_CASE("Structure placement validates only candidates that can reach the target chunk",
          "[structure][worldgen][work-limit]") {
    constexpr uint32_t SEED = 112233;
    ChunkGenerator generator(SEED);
    StructurePlacer structures(SEED);
    GenScratch scratch;
    scratch.reset(&generator);
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.fill(BlockType::AIR);

    structures.place(chunk, generator, scratch);

    // One 8 by 8 region owns the target chunk. Neighboring regions are
    // examined geometrically, but their distant candidates must not trigger
    // terrain plans or enter the validated-placement cache.
    REQUIRE(scratch.structurePlacements.size() <= 1);
}

TEST_CASE("Bounded structure validation preserves every reachable structure output",
          "[structure][worldgen][determinism][regression]") {
    constexpr uint32_t SEED = 112233;
    ChunkGenerator generator(SEED);
    StructurePlacer structures(SEED);
    GenScratch searchScratch;
    searchScratch.reset(&generator);
    std::optional<StructurePlacement> accepted;
    for (int64_t regionZ = -2; regionZ <= 2 && !accepted; ++regionZ) {
        for (int64_t regionX = -2; regionX <= 2; ++regionX) {
            const StructurePlacement placement =
                structures.regionPlacement(regionX, regionZ, generator, searchScratch);
            if (placement.valid) {
                accepted = placement;
                break;
            }
        }
    }
    REQUIRE(accepted);

    const int64_t minimumChunkX = Chunk::worldToChunk(accepted->anchorX - accepted->halfX);
    const int64_t maximumChunkX = Chunk::worldToChunk(accepted->anchorX + accepted->halfX);
    const int64_t minimumChunkZ = Chunk::worldToChunk(accepted->anchorZ - accepted->halfZ);
    const int64_t maximumChunkZ = Chunk::worldToChunk(accepted->anchorZ + accepted->halfZ);
    const int32_t minimumChunkY = Chunk::worldToChunkY(accepted->floorY - 8);
    const int32_t maximumChunkY = Chunk::worldToChunkY(accepted->floorY + 6);
    size_t comparedChunks = 0;
    size_t nonemptyChunks = 0;
    for (int64_t chunkZ = minimumChunkZ; chunkZ <= maximumChunkZ; ++chunkZ) {
        for (int64_t chunkX = minimumChunkX; chunkX <= maximumChunkX; ++chunkX) {
            for (int32_t chunkY = minimumChunkY; chunkY <= maximumChunkY; ++chunkY) {
                const ChunkPos position{chunkX, chunkY, chunkZ};
                Chunk eager(position);
                eager.fill(BlockType::AIR);
                GenScratch eagerScratch;
                eagerScratch.reset(&generator);
                const int64_t minimumRegionX = world_coord::floorDiv(
                    chunkX - 1, static_cast<int64_t>(STRUCTURE_REGION_CHUNKS));
                const int64_t maximumRegionX = world_coord::floorDiv(
                    chunkX + 1, static_cast<int64_t>(STRUCTURE_REGION_CHUNKS));
                const int64_t minimumRegionZ = world_coord::floorDiv(
                    chunkZ - 1, static_cast<int64_t>(STRUCTURE_REGION_CHUNKS));
                const int64_t maximumRegionZ = world_coord::floorDiv(
                    chunkZ + 1, static_cast<int64_t>(STRUCTURE_REGION_CHUNKS));
                for (int64_t regionZ = minimumRegionZ; regionZ <= maximumRegionZ; ++regionZ) {
                    for (int64_t regionX = minimumRegionX; regionX <= maximumRegionX; ++regionX) {
                        static_cast<void>(
                            structures.regionPlacement(regionX, regionZ, generator, eagerScratch));
                    }
                }
                structures.place(eager, generator, eagerScratch);

                Chunk bounded(position);
                bounded.fill(BlockType::AIR);
                GenScratch boundedScratch;
                boundedScratch.reset(&generator);
                structures.place(bounded, generator, boundedScratch);

                REQUIRE(bounded.copyBlocks() == eager.copyBlocks());
                ++comparedChunks;
                if (std::ranges::any_of(bounded.copyBlocks(),
                                        [](BlockType block) { return block != BlockType::AIR; })) {
                    ++nonemptyChunks;
                }
            }
        }
    }
    REQUIRE(comparedChunks > 0);
    REQUIRE(nonemptyChunks > 0);
}

TEST_CASE("Generator handles cubes beyond supported vertical bounds", "[worldgen][bounds]") {
    ChunkGenerator generator(9876);
    Chunk below(ChunkPos{0, WORLD_MIN_CHUNK_Y - 1, 0});
    Chunk above(ChunkPos{0, WORLD_MAX_CHUNK_Y + 1, 0});
    generator.generateCube(below);
    generator.generateCube(above);

    REQUIRE(below.isUniform());
    REQUIRE(below.uniformBlock() == BlockType::BEDROCK);
    REQUIRE(above.isUniform());
    REQUIRE(above.uniformBlock() == BlockType::AIR);
}

TEST_CASE("Generator seals the floor and leaves headroom at the ceiling", "[worldgen][bounds]") {
    ChunkGenerator generator(9876);
    Chunk bottom(ChunkPos{0, WORLD_MIN_CHUNK_Y, 0});
    Chunk top(ChunkPos{0, WORLD_MAX_CHUNK_Y, 0});
    generator.generateCube(bottom);
    generator.generateCube(top);

    for (int z = 0; z < CHUNK_EDGE; ++z) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            REQUIRE(bottom.getBlock(x, 0, z) == BlockType::BEDROCK);
            REQUIRE(bottom.getBlock(x, 1, z) == BlockType::BEDROCK);
        }
    }
    REQUIRE(top.isUniform());
    REQUIRE(top.uniformBlock() == BlockType::AIR);
}

TEST_CASE("Hotspot volcanoes emit conduits and settled crater lakes", "[worldgen][volcano]") {
    constexpr int64_t x = 23'029;
    constexpr int64_t z = -111'486;
    ChunkGenerator generator(764891);
    const int64_t chunkX = Chunk::worldToChunk(x);
    const int64_t chunkZ = Chunk::worldToChunk(z);
    const int localX = Chunk::worldToLocal(x);
    const int localZ = Chunk::worldToLocal(z);
    const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);
    const int surfaceY = generator.surfaceYAt(x, z);
    const int conduitY = surfaceY - 24;
    const int waterTopY = static_cast<int>(std::ceil(surface.waterSurface)) - 1;

    REQUIRE(surface.hydrology.lake);
    REQUIRE(surface.hydrology.endorheic);
    REQUIRE(surface.hydrology.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(surface.waterSurface > surface.terrainHeight);
    REQUIRE(waterTopY > surfaceY);
    const worldgen::WaterBodyId craterWaterBodyId = surface.hydrology.waterBodyId;

    Chunk conduit(ChunkPos{chunkX, Chunk::worldToChunkY(conduitY), chunkZ});
    Chunk craterFloor(ChunkPos{chunkX, Chunk::worldToChunkY(surfaceY), chunkZ});
    Chunk craterLake(ChunkPos{chunkX, Chunk::worldToChunkY(waterTopY), chunkZ});
    generator.generateCube(conduit);
    generator.generateCube(craterFloor);
    generator.generateCube(craterLake);

    REQUIRE(conduit.getBlock(localX, Chunk::worldToLocalY(conduitY), localZ) == BlockType::LAVA);
    REQUIRE(craterFloor.getBlock(localX, Chunk::worldToLocalY(surfaceY), localZ) ==
            BlockType::BASALT);
    REQUIRE(craterFloor.getBlock(localX, Chunk::worldToLocalY(surfaceY + 1), localZ) ==
            BlockType::WATER);
    REQUIRE(craterLake.getBlock(localX, Chunk::worldToLocalY(waterTopY), localZ) ==
            BlockType::WATER);
    REQUIRE(craterLake.getBlock(localX, Chunk::worldToLocalY(waterTopY + 1), localZ) ==
            BlockType::AIR);
    REQUIRE_FALSE(craterFloor.hasExplicitFluidStates());
    REQUIRE_FALSE(craterLake.hasExplicitFluidStates());

    const std::vector<VolcanoPrimitive> volcanoes = generator.hotspotVolcanoesForCell(1, -7);
    const auto primitive = std::ranges::find_if(volcanoes, [](const VolcanoPrimitive& volcano) {
        return volcano.craterLake && std::abs(volcano.centerX - 23'029.177516) < 0.01 &&
               std::abs(volcano.centerZ + 111'485.810195) < 0.01;
    });
    REQUIRE(primitive != volcanoes.end());
    REQUIRE(primitive->craterLakeRadius > 2.0);
    REQUIRE(primitive->craterLakeRadius < primitive->craterRadius - 1.0);
    REQUIRE(primitive->craterLakeSurface <= primitive->craterRimElevation - 1.0);
    REQUIRE(primitive->craterRimWidth >= 12.0);

    struct RingTransition {
        int64_t wetX = 0;
        int64_t wetZ = 0;
        int64_t dryX = 0;
        int64_t dryZ = 0;
        int radius = 0;
        double wetFarTerrain = 0.0;
        double dryFarTerrain = 0.0;
        double wetExactTerrain = 0.0;
        double dryExactTerrain = 0.0;
        double waterSurface = 0.0;
        bool dryBank = false;

        bool operator==(const RingTransition&) const = default;
    };

    constexpr int RING_DIRECTIONS = 96;
    const int64_t centerX = static_cast<int64_t>(std::llround(primitive->centerX - 0.5));
    const int64_t centerZ = static_cast<int64_t>(std::llround(primitive->centerZ - 0.5));
    const int scanRadius = static_cast<int>(std::ceil(primitive->craterRadius * 1.25));
    auto scanDirection = [&](int direction) {
        const double angle =
            static_cast<double>(direction) / RING_DIRECTIONS * 2.0 * std::numbers::pi;
        int64_t wetX = centerX;
        int64_t wetZ = centerZ;
        bool sawWet = false;
        bool foundTransition = false;
        RingTransition transition;
        double priorHeight = generator.sampleFarGeometrySurface(centerX, centerZ).terrainHeight;
        int64_t priorX = centerX;
        int64_t priorZ = centerZ;
        double maximumStep = 0.0;
        int64_t maximumStepFromX = centerX;
        int64_t maximumStepFromZ = centerZ;
        int64_t maximumStepToX = centerX;
        int64_t maximumStepToZ = centerZ;
        for (int radius = 1; radius <= scanRadius; ++radius) {
            const int64_t sampleX =
                static_cast<int64_t>(std::llround(centerX + std::cos(angle) * radius));
            const int64_t sampleZ =
                static_cast<int64_t>(std::llround(centerZ + std::sin(angle) * radius));
            const worldgen::SurfaceSample far =
                generator.sampleFarGeometrySurface(sampleX, sampleZ);
            if (sampleX != priorX || sampleZ != priorZ) {
                const double step = std::abs(far.terrainHeight - priorHeight);
                if (step > maximumStep) {
                    maximumStep = step;
                    maximumStepFromX = priorX;
                    maximumStepFromZ = priorZ;
                    maximumStepToX = sampleX;
                    maximumStepToZ = sampleZ;
                }
                priorHeight = far.terrainHeight;
                priorX = sampleX;
                priorZ = sampleZ;
            }
            const bool craterWater =
                far.hydrology.lake &&
                std::abs(far.waterSurface - primitive->craterLakeSurface) < 1.0e-9;
            if (craterWater) {
                REQUIRE(far.hydrology.waterBodyId == craterWaterBodyId);
                REQUIRE_FALSE(foundTransition);
                sawWet = true;
                wetX = sampleX;
                wetZ = sampleZ;
                continue;
            }
            if (!sawWet || foundTransition)
                continue;

            foundTransition = true;
            const worldgen::SurfaceSample wetFar = generator.sampleFarGeometrySurface(wetX, wetZ);
            const worldgen::SurfaceSample wetExact =
                generator.sampleExactGeometrySurface(wetX, wetZ);
            const worldgen::SurfaceSample dryExact =
                generator.sampleExactGeometrySurface(sampleX, sampleZ);
            REQUIRE(wetFar.hydrology.lake);
            REQUIRE(wetFar.hydrology.waterBodyId == craterWaterBodyId);
            REQUIRE(wetFar.hydrology.lakeShoreDistance > 0.0);
            REQUIRE(wetFar.hydrology.lakeDepth <= 2.0);
            REQUIRE(wetExact.hydrology.lake);
            REQUIRE(wetExact.hydrology.waterBodyId == craterWaterBodyId);
            REQUIRE_FALSE(dryExact.hydrology.lake);
            REQUIRE(dryExact.hydrology.lakeBank);
            REQUIRE(dryExact.hydrology.lakeShoreDistance <= 0.0);
            REQUIRE(dryExact.hydrology.shoreWaterSurface == primitive->craterLakeSurface);
            REQUIRE(dryExact.terrainHeight >= std::ceil(primitive->craterLakeSurface));
            REQUIRE(std::abs(wetExact.terrainHeight - wetFar.terrainHeight) <= 1.0);
            REQUIRE(std::abs(dryExact.terrainHeight - far.terrainHeight) <= 1.0);
            transition = {
                .wetX = wetX,
                .wetZ = wetZ,
                .dryX = sampleX,
                .dryZ = sampleZ,
                .radius = radius,
                .wetFarTerrain = wetFar.terrainHeight,
                .dryFarTerrain = far.terrainHeight,
                .wetExactTerrain = wetExact.terrainHeight,
                .dryExactTerrain = dryExact.terrainHeight,
                .waterSurface = wetFar.waterSurface,
                .dryBank = dryExact.hydrology.lakeBank,
            };
        }
        REQUIRE(sawWet);
        REQUIRE(foundTransition);
        CAPTURE(direction, maximumStepFromX, maximumStepFromZ, maximumStepToX, maximumStepToZ);
        REQUIRE(maximumStep <= 2.0);
        return transition;
    };

    std::vector<RingTransition> forwardRing(static_cast<size_t>(RING_DIRECTIONS));
    int minimumShoreRadius = scanRadius;
    int maximumShoreRadius = 0;
    std::unordered_set<int> shoreRadii;
    for (int direction = 0; direction < RING_DIRECTIONS; ++direction) {
        RingTransition transition = scanDirection(direction);
        minimumShoreRadius = std::min(minimumShoreRadius, transition.radius);
        maximumShoreRadius = std::max(maximumShoreRadius, transition.radius);
        shoreRadii.insert(transition.radius);
        forwardRing[static_cast<size_t>(direction)] = transition;
    }
    REQUIRE(maximumShoreRadius - minimumShoreRadius >= 6);
    REQUIRE(shoreRadii.size() >= 6);

    std::unordered_map<ChunkPos, std::unique_ptr<Chunk>> emittedCubes;
    auto emittedCube = [&](int64_t worldX, int worldY, int64_t worldZ) -> Chunk& {
        const ChunkPos position{Chunk::worldToChunk(worldX), Chunk::worldToChunkY(worldY),
                                Chunk::worldToChunk(worldZ)};
        auto found = emittedCubes.find(position);
        if (found == emittedCubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = emittedCubes.emplace(position, std::move(cube)).first;
        }
        return *found->second;
    };
    auto requireSourceColumn = [&](int64_t worldX, int64_t worldZ) {
        const int floorY = generator.surfaceYAt(worldX, worldZ);
        REQUIRE(floorY < waterTopY);
        for (int worldY = floorY + 1; worldY <= waterTopY; ++worldY) {
            CAPTURE(worldX, worldY, worldZ, floorY, waterTopY);
            Chunk& cube = emittedCube(worldX, worldY, worldZ);
            const int localX = Chunk::worldToLocal(worldX);
            const int localY = Chunk::worldToLocalY(worldY);
            const int localZ = Chunk::worldToLocal(worldZ);
            const BlockType block = cube.getBlock(localX, localY, localZ);
            if (worldY == waterTopY && block == BlockType::ICE) {
                // Climate can cross the freezing threshold within a broad
                // caldera. An ice cap is a valid top cell as long as the full
                // liquid volume beneath it remains source water.
                continue;
            }
            REQUIRE(block == BlockType::WATER);
            REQUIRE(cube.getFluidState(localX, localY, localZ).isSource());
        }
    };

    requireSourceColumn(centerX, centerZ);
    const int64_t crossCubeWaterX = centerX + CHUNK_EDGE;
    REQUIRE(Chunk::worldToChunk(crossCubeWaterX) != Chunk::worldToChunk(centerX));
    const worldgen::SurfaceSample crossCubeWater =
        generator.sampleExactGeometrySurface(crossCubeWaterX, centerZ);
    REQUIRE(crossCubeWater.hydrology.lake);
    REQUIRE(crossCubeWater.hydrology.waterBodyId == craterWaterBodyId);
    requireSourceColumn(crossCubeWaterX, centerZ);
    for (const RingTransition& transition : forwardRing) {
        Chunk& dryCube = emittedCube(transition.dryX, waterTopY, transition.dryZ);
        REQUIRE(isSolid(dryCube.getBlock(Chunk::worldToLocal(transition.dryX),
                                         Chunk::worldToLocalY(waterTopY),
                                         Chunk::worldToLocal(transition.dryZ))));
    }
    for (const auto& [position, cube] : emittedCubes) {
        CAPTURE(position.x, position.y, position.z);
        REQUIRE_FALSE(cube->hasExplicitFluidStates());
    }

    generator.clearMacroCaches();
    const worldgen::SurfaceSample rebuiltCenter = generator.sampleSurface(x, z);
    REQUIRE(rebuiltCenter.hydrology.waterBodyId == craterWaterBodyId);
    REQUIRE(rebuiltCenter.waterSurface == surface.waterSurface);
    std::vector<RingTransition> rebuiltRing(static_cast<size_t>(RING_DIRECTIONS));
    for (int direction = RING_DIRECTIONS - 1; direction >= 0; --direction) {
        rebuiltRing[static_cast<size_t>(direction)] = scanDirection(direction);
    }
    REQUIRE(rebuiltRing == forwardRing);
}

TEST_CASE("Aquifers stay inside deterministic sealed pockets", "[worldgen][aquifer]") {
    constexpr int64_t x = -1443;
    constexpr int y = -84;
    constexpr int64_t z = -1500;
    ChunkGenerator generator(764891);
    Chunk aquifer(
        ChunkPos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)});
    generator.generateCube(aquifer);

    const int localX = Chunk::worldToLocal(x);
    const int localY = Chunk::worldToLocalY(y);
    const int localZ = Chunk::worldToLocal(z);
    REQUIRE(aquifer.getBlock(localX, localY, localZ) == BlockType::WATER);
    REQUIRE(isSolid(aquifer.getBlock(localX, localY, localZ + 7)));
    REQUIRE(aquifer.getBlock(localX, localY, localZ + 9) != BlockType::WATER);
    REQUIRE_FALSE(aquifer.hasExplicitFluidStates());
}

TEST_CASE("Cubic generation is seed deterministic", "[worldgen][determinism]") {
    const std::array<ChunkPos, 1> positions{{
        {0, 4, 0},
    }};
    ChunkGenerator first(777);
    ChunkGenerator same(777);
    ChunkGenerator different(778);
    bool foundSeedDifference = false;

    for (ChunkPos position : positions) {
        Chunk a(position);
        Chunk b(position);
        Chunk c(position);
        first.generateCube(a);
        same.generateCube(b);
        different.generateCube(c);
        REQUIRE(a.copyBlocks() == b.copyBlocks());
        REQUIRE(a.generated);
        REQUIRE(b.generated);
        foundSeedDifference = foundSeedDifference || a.copyBlocks() != c.copyBlocks();
    }
    REQUIRE(foundSeedDifference);
}

TEST_CASE("Cubic generation is independent of request order", "[worldgen][determinism]") {
    const std::array<ChunkPos, 4> positions{{
        {0, 3, 0},
        {0, 4, 0},
        {1, 4, 0},
        {0, 4, 1},
    }};
    ChunkGenerator forwardGenerator(5150);
    ChunkGenerator reverseGenerator(5150);
    std::unordered_map<ChunkPos, std::vector<BlockType>> forward;
    std::unordered_map<ChunkPos, std::vector<BlockType>> reverse;

    for (ChunkPos position : positions) {
        Chunk cube(position);
        forwardGenerator.generateCube(cube);
        forward.emplace(position, cube.copyBlocks());
    }
    for (auto iterator = positions.rbegin(); iterator != positions.rend(); ++iterator) {
        Chunk cube(*iterator);
        reverseGenerator.generateCube(cube);
        reverse.emplace(*iterator, cube.copyBlocks());
    }

    REQUIRE(forward == reverse);
}

TEST_CASE("Surface queries agree across scratch instances", "[worldgen][determinism]") {
    ChunkGenerator generator(9999);
    GenScratch first;
    GenScratch second;
    first.reset(&generator);
    second.reset(&generator);
    constexpr std::array<ColumnPos, 2> samples{{
        {-80, -80},
        {80, 17},
    }};
    for (ColumnPos sample : samples) {
        REQUIRE(generator.surfaceYAt(sample.x, sample.z, first) ==
                generator.surfaceYAt(sample.x, sample.z, second));
        REQUIRE(generator.biomeAt(sample.x, sample.z, first) ==
                generator.biomeAt(sample.x, sample.z, second));
    }
}

TEST_CASE("Thread-local generation scratch does not survive generator address reuse",
          "[worldgen][determinism][scratch]") {
    alignas(ChunkGenerator) std::array<std::byte, sizeof(ChunkGenerator)> storage{};
    auto* first = std::construct_at(reinterpret_cast<ChunkGenerator*>(storage.data()), 111);
    const worldgen::SurfaceSample firstSample = first->sampleSurface(1234, -5678);
    std::destroy_at(first);

    auto* reused = std::construct_at(reinterpret_cast<ChunkGenerator*>(storage.data()), 222);
    const worldgen::SurfaceSample reusedSample = reused->sampleSurface(1234, -5678);
    ChunkGenerator control(222);
    const worldgen::SurfaceSample controlSample = control.sampleSurface(1234, -5678);
    std::destroy_at(reused);

    REQUIRE(firstSample.terrainHeight != Catch::Approx(controlSample.terrainHeight));
    REQUIRE(reusedSample.terrainHeight == Catch::Approx(controlSample.terrainHeight));
    REQUIRE(reusedSample.biome.primary == controlSample.biome.primary);
    REQUIRE(reusedSample.hydrology.surfaceElevation ==
            Catch::Approx(controlSample.hydrology.surfaceElevation));
}

TEST_CASE("Column plans cache immutable cubic surface data", "[worldgen][column-plan]") {
    ChunkGenerator generator(2468);
    constexpr ColumnPos column{-2, 3};
    auto first = generator.getColumnPlan(column);
    auto cached = generator.getColumnPlan(column);
    REQUIRE(first == cached);
    REQUIRE(first->chunkColumn() == column);
    REQUIRE_FALSE(first->exposedSections().empty());
    REQUIRE_FALSE(first->floraOwnershipSections().empty());
    REQUIRE(std::is_sorted(first->exposedSections().begin(), first->exposedSections().end()));
    REQUIRE(std::is_sorted(first->floraOwnershipSections().begin(),
                           first->floraOwnershipSections().end()));
    REQUIRE(std::ranges::includes(first->exposedSections(), first->floraOwnershipSections()));
    for (int32_t section : first->exposedSections()) {
        REQUIRE(section >= WORLD_MIN_CHUNK_Y);
        REQUIRE(section <= WORLD_MAX_CHUNK_Y);
        REQUIRE(first->exposesSection(section));
    }

    constexpr int localX = 8;
    constexpr int localZ = 12;
    const int64_t worldX = column.x * CHUNK_EDGE + localX;
    const int64_t worldZ = column.z * CHUNK_EDGE + localZ;
    const worldgen::SurfaceSample planned = first->sample(localX, localZ);
    const worldgen::SurfaceSample direct = generator.sampleSurface(worldX, worldZ);
    // ColumnPlan::sample retains the continuous hydrologic substrate while
    // the exact surface grid records the emitted voxel top. The public
    // block-footprint sampler must expose that exact mesh plane.
    REQUIRE(first->surfaceY(localX, localZ) + 1 == Catch::Approx(direct.terrainHeight));
    REQUIRE(std::isfinite(planned.terrainHeight));
    REQUIRE(planned.waterSurface == Catch::Approx(planned.hydrology.waterSurface));
    REQUIRE(direct.waterSurface == Catch::Approx(direct.hydrology.waterSurface));
    REQUIRE(planned.biome.primary == direct.biome.primary);
    REQUIRE(std::isfinite(planned.climate.temperatureC));
    REQUIRE(std::isfinite(planned.climate.annualPrecipitationMm));
    REQUIRE(planned.soil.moisture >= 0.0);
    REQUIRE(planned.soil.moisture <= 1.0);
}

TEST_CASE("Column plans expose flora ownership separately from generation support",
          "[worldgen][column-plan][flora][streaming]") {
    ChunkGenerator generator(2468);
    const auto plan = generator.getColumnPlan({0, 0});
    REQUIRE_FALSE(plan->floraOwnershipSections().empty());
    REQUIRE(std::ranges::includes(plan->exposedSections(), plan->floraOwnershipSections()));
    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const int surfaceY = plan->surfaceY(localX, localZ);
            for (int offset = 1; offset <= feature_generation::GROUND_FLORA_MAXIMUM_VERTICAL_OFFSET;
                 ++offset) {
                REQUIRE(std::ranges::binary_search(plan->floraOwnershipSections(),
                                                   Chunk::worldToChunkY(surfaceY + offset)));
            }
        }
    }

    const auto flatSurface = [](int64_t, int64_t) {
        worldgen::SurfaceSample surface;
        surface.terrainHeight = 40.0;
        surface.hydrology.surfaceElevation = 40.0;
        surface.waterSurface = 40.0;
        return surface;
    };
    const auto waterfall = [](int64_t, int64_t) {
        worldgen::HydrologySample hydrology;
        hydrology.surfaceElevation = 40.0;
        hydrology.waterSurface = 600.0;
        hydrology.waterfall = true;
        hydrology.waterfallTop = 600.0;
        hydrology.waterfallBottom = 40.0;
        hydrology.waterfallWidth = 2.0;
        hydrology.flowDirection = {1.0, 0.0};
        return hydrology;
    };
    ColumnPlan verticalWater(
        {0, 0}, flatSurface, [](int64_t, int64_t) { return 40.0; },
        [](const ColumnPlan&) {
            ColumnPlanSurfaceGrid surfaces{};
            surfaces.fill(40);
            return surfaces;
        },
        waterfall);
    const int32_t waterfallTopSection = Chunk::worldToChunkY(599);
    REQUIRE(std::ranges::binary_search(verticalWater.exposedSections(), waterfallTopSection));
    REQUIRE(
        std::ranges::binary_search(verticalWater.surfaceOwnershipSections(), waterfallTopSection));
    REQUIRE_FALSE(
        std::ranges::binary_search(verticalWater.floraOwnershipSections(), waterfallTopSection));
}

TEST_CASE("Column plan exposure covers every generated block above its terrain cutoff",
          "[worldgen][column-plan][skylight][occupancy][regression]") {
    ChunkGenerator generator(2468);
    constexpr ColumnPos column{-2, 3};
    const auto plan = generator.getColumnPlan(column);
    REQUIRE(plan);

    int minimumTerrainY = WORLD_MAX_Y;
    for (int z = 0; z < CHUNK_EDGE; ++z) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            minimumTerrainY = std::min(minimumTerrainY, plan->surfaceY(x, z));
        }
    }
    const int32_t firstSection = Chunk::worldToChunkY(minimumTerrainY);
    const int32_t lastSection =
        std::min(WORLD_MAX_CHUNK_Y, Chunk::worldToChunkY(plan->maximumSurfaceY()) + 1);
    size_t provenEmptySections = 0;
    for (int32_t section = firstSection; section <= lastSection; ++section) {
        if (plan->exposesSection(section))
            continue;
        Chunk cube(ChunkPos{column.x, section, column.z});
        generator.generate(cube);
        ++provenEmptySections;
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            for (int x = 0; x < CHUNK_EDGE; ++x) {
                const int cutoffY = plan->surfaceY(x, z) + 1;
                for (int y = 0; y < CHUNK_EDGE; ++y) {
                    const int worldY = section * CHUNK_EDGE + y;
                    if (worldY < cutoffY)
                        continue;
                    CAPTURE(section, x, y, z, cutoffY, cube.getBlock(x, y, z));
                    REQUIRE(cube.getBlock(x, y, z) == BlockType::AIR);
                }
            }
        }
    }
    REQUIRE(provenEmptySections > 0);
}

TEST_CASE("Column plans retain canonical parent-owned wetland authority",
          "[worldgen][column-plan][wetland][exact][determinism]") {
    constexpr int WETLAND_X = 7;
    constexpr int WETLAND_Z = 9;
    constexpr worldgen::WaterBodyId PARENT_BODY = 0x5745'544C'414E'4404ULL;
    const auto drySurface = [](int64_t, int64_t) {
        worldgen::SurfaceSample sample;
        sample.terrainHeight = 63.875;
        sample.hydrology.surfaceElevation = 63.875;
        sample.hydrology.waterSurface = 0.0;
        sample.waterSurface = 0.0;
        return sample;
    };
    const auto hydrology = [](int64_t x, int64_t z) {
        worldgen::HydrologySample sample;
        sample.surfaceElevation = 63.875;
        if (x != WETLAND_X || z != WETLAND_Z)
            return sample;
        sample.waterBodyId = PARENT_BODY;
        sample.waterSurface = 64.0;
        sample.wetland = true;
        sample.groundwaterHead = 64.25;
        sample.hydroperiod = 0.80;
        return sample;
    };
    ColumnPlan plan(
        {0, 0}, drySurface, [](int64_t, int64_t) { return 63.875; },
        [](const ColumnPlan&) {
            ColumnPlanSurfaceGrid surfaces{};
            surfaces.fill(63);
            return surfaces;
        },
        hydrology);

    const worldgen::SurfaceSample wetland = plan.sample(WETLAND_X, WETLAND_Z);
    REQUIRE(wetland.hydrology.wetland);
    REQUIRE_FALSE(wetland.hydrology.ocean);
    REQUIRE_FALSE(wetland.hydrology.lake);
    REQUIRE_FALSE(wetland.hydrology.river);
    REQUIRE(wetland.hydrology.waterBodyId == PARENT_BODY);
    REQUIRE(wetland.hydrology.waterSurface == Catch::Approx(64.0));
    REQUIRE(wetland.hydrology.surfaceElevation == Catch::Approx(63.875));
    REQUIRE(wetland.hydrology.groundwaterHead == Catch::Approx(64.25));
    REQUIRE(wetland.hydrology.hydroperiod == Catch::Approx(0.80).margin(1.0 / 255.0));
    REQUIRE(wetland.waterSurface == Catch::Approx(64.0));

    const worldgen::SurfaceSample dry = plan.sample(WETLAND_X - 1, WETLAND_Z);
    REQUIRE_FALSE(dry.hydrology.wetland);
    REQUIRE(dry.hydrology.waterBodyId == worldgen::NO_WATER_BODY);
    REQUIRE(dry.hydrology.waterSurface == Catch::Approx(0.0));
}

TEST_CASE("Column plans retain canonical estuary and distributary identity",
          "[worldgen][column-plan][hydrology][estuary][delta][brackish]") {
    constexpr int MOUTH_X = 11;
    constexpr int MOUTH_Z = 5;
    constexpr worldgen::WaterBodyId RIVER_BODY = 0x4553'5455'4152'5901ULL;
    const auto drySurface = [](int64_t, int64_t) {
        worldgen::SurfaceSample sample;
        sample.terrainHeight = 63.875;
        sample.hydrology.surfaceElevation = 63.875;
        return sample;
    };
    const auto hydrology = [](int64_t x, int64_t z) {
        worldgen::HydrologySample sample;
        sample.surfaceElevation = 63.875;
        if (x != MOUTH_X || z != MOUTH_Z)
            return sample;
        sample.waterBodyId = RIVER_BODY;
        sample.waterSurface = 64.0;
        sample.flowDirection = {0.8, 0.6};
        sample.discharge = 640.0;
        sample.channelDistance = 0.0;
        sample.channelWidth = 6.0;
        sample.channelDepth = 0.125;
        sample.channelGradient = 0.002;
        sample.groundwaterHead = 64.0;
        sample.streamOrder = 4;
        sample.distributaryCount = 2;
        sample.river = true;
        sample.delta = true;
        sample.estuary = true;
        sample.brackish = true;
        return sample;
    };
    ColumnPlan plan(
        {0, 0}, drySurface, [](int64_t, int64_t) { return 63.875; },
        [](const ColumnPlan&) {
            ColumnPlanSurfaceGrid surfaces{};
            surfaces.fill(63);
            return surfaces;
        },
        hydrology);

    const worldgen::SurfaceSample mouth = plan.sample(MOUTH_X, MOUTH_Z);
    REQUIRE(mouth.hydrology.river);
    REQUIRE(mouth.hydrology.delta);
    REQUIRE(mouth.hydrology.estuary);
    REQUIRE(mouth.hydrology.brackish);
    REQUIRE_FALSE(mouth.hydrology.ocean);
    REQUIRE_FALSE(mouth.hydrology.lake);
    REQUIRE(mouth.hydrology.waterBodyId == RIVER_BODY);
    REQUIRE(mouth.hydrology.waterSurface == Catch::Approx(64.0));
    REQUIRE(mouth.hydrology.surfaceElevation == Catch::Approx(63.875));

    const worldgen::SurfaceSample dry = plan.sample(MOUTH_X - 1, MOUTH_Z);
    REQUIRE_FALSE(dry.hydrology.river);
    REQUIRE_FALSE(dry.hydrology.delta);
    REQUIRE_FALSE(dry.hydrology.estuary);
    REQUIRE_FALSE(dry.hydrology.brackish);
}

TEST_CASE("Column plans expose tree reach from every neighboring face",
          "[worldgen][column-plan][flora]") {
    constexpr ColumnPos column{-3, 2};
    constexpr int64_t baseX = column.x * CHUNK_EDGE;
    constexpr int64_t baseZ = column.z * CHUNK_EDGE;
    constexpr std::array<ColumnPos, 4> directions{{
        {-1, 0},
        {1, 0},
        {0, -1},
        {0, 1},
    }};

    for (const ColumnPos direction : directions) {
        size_t surfaceSamples = 0;
        size_t heightSamples = 0;
        auto heightAt = [&](int64_t x, int64_t z) {
            REQUIRE(x >= baseX - COLUMN_PLAN_LATTICE_SPACING);
            REQUIRE(x <= baseX + CHUNK_EDGE + COLUMN_PLAN_LATTICE_SPACING);
            REQUIRE(z >= baseZ - COLUMN_PLAN_LATTICE_SPACING);
            REQUIRE(z <= baseZ + CHUNK_EDGE + COLUMN_PLAN_LATTICE_SPACING);

            const bool neighboringCliff =
                (direction.x < 0 && x < baseX) || (direction.x > 0 && x > baseX + CHUNK_EDGE) ||
                (direction.z < 0 && z < baseZ) || (direction.z > 0 && z > baseZ + CHUNK_EDGE);
            return neighboringCliff ? 180.0 : 40.0;
        };
        ColumnPlan plan(
            column,
            [&](int64_t x, int64_t z) {
                ++surfaceSamples;
                const double height = heightAt(x, z);
                const bool neighboringCliff = height > 100.0;
                worldgen::SurfaceSample result;
                result.terrainHeight = height;
                result.hydrology.surfaceElevation = result.terrainHeight;
                result.hydrology.ocean = !neighboringCliff;
                result.hydrology.waterSurface = neighboringCliff ? result.terrainHeight : SEA_LEVEL;
                result.waterSurface = result.hydrology.waterSurface;
                result.biome.primary = neighboringCliff ? Biome::FOREST : Biome::OCEAN;
                result.biome.secondary = result.biome.primary;
                return result;
            },
            [&](int64_t x, int64_t z) {
                ++heightSamples;
                return heightAt(x, z);
            },
            [](const ColumnPlan&) {
                ColumnPlanSurfaceGrid surfaces{};
                surfaces.fill(40);
                return surfaces;
            });

        constexpr int treeApron =
            (feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH + COLUMN_PLAN_LATTICE_SPACING - 1) /
            COLUMN_PLAN_LATTICE_SPACING;
        constexpr size_t sampledEdge = COLUMN_PLAN_LATTICE_EDGE + treeApron * 2;
        constexpr size_t retainedSamples = COLUMN_PLAN_LATTICE_EDGE * COLUMN_PLAN_LATTICE_EDGE;
        REQUIRE(surfaceSamples == sampledEdge * sampledEdge);
        REQUIRE(heightSamples == sampledEdge * sampledEdge - retainedSamples);
        const int minimumTreeY = 40 - feature_generation::TREE_MAXIMUM_SURFACE_DEVIATION + 1 +
                                 feature_generation::TREE_MINIMUM_VERTICAL_OFFSET;
        const int maximumTreeY = 180 + feature_generation::TREE_MAXIMUM_SURFACE_DEVIATION + 1 +
                                 feature_generation::TREE_MAXIMUM_VERTICAL_OFFSET;
        REQUIRE(plan.minimumSurfaceY() == minimumTreeY);
        REQUIRE(plan.maximumSurfaceY() == maximumTreeY);
        for (int y = minimumTreeY; y <= maximumTreeY; y += CHUNK_EDGE) {
            REQUIRE(plan.exposesSection(Chunk::worldToChunkY(y)));
        }
        REQUIRE(plan.exposesSection(Chunk::worldToChunkY(maximumTreeY)));
    }
}

TEST_CASE("Column plans expose the exact density surface below macro relief",
          "[worldgen][column-plan][density]") {
    ChunkGenerator generator(42);
    constexpr int64_t worldX = 18;
    constexpr int64_t worldZ = 86;
    const ColumnPos column{Chunk::worldToChunk(worldX), Chunk::worldToChunk(worldZ)};
    const auto plan = generator.getColumnPlan(column);
    const int localX = Chunk::worldToLocal(worldX);
    const int localZ = Chunk::worldToLocal(worldZ);
    const int exactSurface = generator.surfaceYAt(worldX, worldZ);

    REQUIRE(exactSurface == 79);
    REQUIRE(plan->surfaceY(localX, localZ) == exactSurface);
    REQUIRE(exactSurface <
            static_cast<int>(std::floor(plan->sample(localX, localZ).terrainHeight)));
    REQUIRE(plan->exposesSection(Chunk::worldToChunkY(exactSurface)));
}

TEST_CASE("Bounded basins expose stable river canyon waterfall and delta features",
          "[worldgen][hydrology]") {
    worldgen::MacroGenerationSampler sampler(42);
    const worldgen::SurfaceSample lakeLip = sampler.sampleSurface(-8235.0, 2976.0);
    REQUIRE(lakeLip.hydrology.lake);
    REQUIRE_FALSE(lakeLip.hydrology.endorheic);
    REQUIRE(lakeLip.waterSurface > lakeLip.terrainHeight);

    for (const double z : {2759.0, 2760.0}) {
        const worldgen::SurfaceSample riverLeft = sampler.sampleSurface(-12801.0, z);
        const worldgen::SurfaceSample riverRight = sampler.sampleSurface(-12800.0, z);
        for (const worldgen::SurfaceSample* river : {&riverLeft, &riverRight}) {
            REQUIRE(river->hydrology.river);
            REQUIRE_FALSE(river->hydrology.lake);
            REQUIRE_FALSE(river->hydrology.waterfall);
            REQUIRE(worldgen::hasEcotope(river->ecotopes, worldgen::Ecotope::RIVERBANK));
            REQUIRE(river->hydrology.streamOrder >= 2);
            REQUIRE(river->hydrology.discharge > 0.0);
            REQUIRE(river->hydrology.erosionDepth > 4.5);
        }
        REQUIRE(std::abs(riverLeft.waterSurface - riverRight.waterSurface) < 0.05);
    }

    const worldgen::SurfaceSample canyon = sampler.sampleSurface(-23904.0, 0.0);
    REQUIRE(worldgen::hasEcotope(canyon.ecotopes, worldgen::Ecotope::CANYON));
    REQUIRE(canyon.hydrology.streamOrder >= 2);
    REQUIRE(canyon.hydrology.discharge > 0.0);
    REQUIRE(canyon.hydrology.erosionDepth > 4.5);
    REQUIRE(canyon.hydrology.channelGradient > 0.012);

    const worldgen::SurfaceSample waterfall = sampler.sampleSurface(-8240.0, 3088.0);
    REQUIRE(waterfall.hydrology.waterfall);
    REQUIRE(waterfall.hydrology.waterfallAnchor);
    REQUIRE(waterfall.hydrology.waterfallTop >= waterfall.hydrology.waterfallBottom + 2.5);

    const worldgen::SurfaceSample delta = sampler.sampleSurface(-23904.0, 0.0);
    REQUIRE(delta.hydrology.delta);
    REQUIRE(worldgen::hasEcotope(delta.ecotopes, worldgen::Ecotope::DELTA));
    REQUIRE(delta.hydrology.distributaryCount >= 2);
    REQUIRE(delta.hydrology.distributaryCount <= 4);
    REQUIRE(delta.hydrology.sediment > 0.0);

    ChunkGenerator generator(42);
    const worldgen::SurfaceSample emittedDelta = generator.sampleFarSurface(-23904, 0);
    REQUIRE_FALSE(emittedDelta.hydrology.ocean);
    REQUIRE(emittedDelta.hydrology.river);
    REQUIRE(emittedDelta.hydrology.delta);
    REQUIRE(emittedDelta.waterSurface == SEA_LEVEL);
    REQUIRE(emittedDelta.waterSurface > emittedDelta.terrainHeight);
    REQUIRE(emittedDelta.hydrology.distributaryCount == delta.hydrology.distributaryCount);
    REQUIRE(worldgen::hasEcotope(emittedDelta.ecotopes, worldgen::Ecotope::DELTA));
}

TEST_CASE("Eroded macro relief exposes cliff ecotopes", "[worldgen][terrain]") {
    worldgen::MacroGenerationSampler sampler(42);
    const worldgen::SurfaceSample cliff = sampler.sampleSurface(-23904.0, 0.0);
    REQUIRE(cliff.slope > 0.75);
    REQUIRE(worldgen::hasEcotope(cliff.ecotopes, worldgen::Ecotope::CLIFF));
}

TEST_CASE("Basin cache is single flight bounded and eviction deterministic",
          "[worldgen][hydrology][concurrency]") {
    worldgen::MacroGenerationSampler sampler(42);
    sampler.clearBasinCache();
    std::array<std::future<worldgen::HydrologySample>, 8> requests;
    for (auto& request : requests) {
        request = std::async(std::launch::async,
                             [&sampler] { return sampler.sampleHydrology(-8235.0, 2976.0); });
    }
    std::array<worldgen::HydrologySample, 8> samples;
    for (size_t index = 0; index < requests.size(); ++index)
        samples[index] = requests[index].get();
    for (const auto& sample : samples) {
        REQUIRE(sample.surfaceElevation == samples.front().surfaceElevation);
        REQUIRE(sample.waterSurface == samples.front().waterSurface);
        REQUIRE(sample.discharge == samples.front().discharge);
        REQUIRE(sample.streamOrder == samples.front().streamOrder);
    }

    const worldgen::BasinCacheMetrics warm = sampler.basinCacheMetrics();
    // This contour page straddles one catchment face, so its canonical
    // authority has exactly two immutable basin dependencies. Concurrent
    // callers must still construct each dependency only once.
    REQUIRE(warm.builds == 2);
    REQUIRE(warm.failures == 0);
    REQUIRE(warm.entries == 2);
    REQUIRE(warm.bytes > 0);
    REQUIRE(warm.bytes <= worldgen::BASIN_CACHE_BYTE_BUDGET);
    REQUIRE(warm.shorelineBuilds == 1);
    REQUIRE(warm.shorelineMisses == 1);
    REQUIRE(warm.shorelineHits >= requests.size() - 1);
    REQUIRE(warm.shorelineFailures == 0);
    REQUIRE(warm.shorelineEntries == 1);
    REQUIRE(warm.shorelineBytes > 0);
    REQUIRE(warm.shorelineBytes <= worldgen::SHORELINE_CACHE_BYTE_BUDGET);

    sampler.clearBasinCache();
    const worldgen::HydrologySample rebuilt = sampler.sampleHydrology(-8235.0, 2976.0);
    REQUIRE(rebuilt.surfaceElevation == samples.front().surfaceElevation);
    REQUIRE(rebuilt.waterSurface == samples.front().waterSurface);
    REQUIRE(rebuilt.discharge == samples.front().discharge);
    const worldgen::BasinCacheMetrics afterEviction = sampler.basinCacheMetrics();
    REQUIRE(afterEviction.failures == 0);
    REQUIRE(afterEviction.builds == 4);
    REQUIRE(afterEviction.entries == 2);
    REQUIRE(afterEviction.shorelineBuilds == 2);
    REQUIRE(afterEviction.shorelineFailures == 0);
    REQUIRE(afterEviction.shorelineEntries == 1);
    REQUIRE(afterEviction.shorelineBytes > 0);
}

// ===========================================================================
// RYCH v4 serialization and cubic persistence
// ===========================================================================

TEST_CASE("RYCH v4 dense cube round-trips coordinates blocks and fluid", "[serialization]") {
    Chunk original(ChunkPos{5, -3, -7});
    original.setBlock(8, 9, 10, BlockType::STONE);
    original.setBlock(0, 0, 0, BlockType::WATER);
    original.setFluidState(0, 0, 0, FluidState::falling(4));
    original.generated = true;

    const std::vector<uint8_t> bytes = ChunkSerializer::serialize(original);
    REQUIRE(bytes.size() == HEADER_SIZE + CHUNK_VOLUME * 2);

    ChunkSaveHeader header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    REQUIRE(header.magic == CHUNK_MAGIC);
    REQUIRE(header.version == 4);
    REQUIRE(header.chunkX == 5);
    REQUIRE(header.chunkY == -3);
    REQUIRE(header.chunkZ == -7);
    REQUIRE((header.flags & CHUNK_FLAG_UNIFORM) == 0);
    REQUIRE((header.flags & CHUNK_FLAG_FLUID_STATES) != 0);

    auto restored = ChunkSerializer::deserialize(bytes);
    REQUIRE(restored.has_value());
    REQUIRE(restored->pos() == original.pos());
    REQUIRE(restored->copyBlocks() == original.copyBlocks());
    REQUIRE(restored->getFluidState(0, 0, 0) == FluidState::falling(4));
    REQUIRE(restored->generated);
}

TEST_CASE("RYCH v4 preserves edits and fluid at both vertical limits",
          "[serialization][bounds][regression]") {
    Chunk bottom(ChunkPos{-2, WORLD_MIN_CHUNK_Y, 3});
    bottom.setBlock(4, 0, 9, BlockType::BEDROCK);
    bottom.generated = true;
    auto restoredBottom = ChunkSerializer::deserialize(ChunkSerializer::serialize(bottom));
    REQUIRE(restoredBottom.has_value());
    REQUIRE(restoredBottom->pos() == bottom.pos());
    REQUIRE(restoredBottom->getBlock(4, 0, 9) == BlockType::BEDROCK);
    REQUIRE(restoredBottom->getWorldPosition().y == Catch::Approx(static_cast<float>(WORLD_MIN_Y)));

    Chunk top(ChunkPos{5, WORLD_MAX_CHUNK_Y, -7});
    top.setBlock(12, CHUNK_EDGE - 1, 1, BlockType::WATER);
    top.setFluidState(12, CHUNK_EDGE - 1, 1, FluidState::falling(3));
    top.generated = true;
    auto restoredTop = ChunkSerializer::deserialize(ChunkSerializer::serialize(top));
    REQUIRE(restoredTop.has_value());
    REQUIRE(restoredTop->pos() == top.pos());
    REQUIRE(restoredTop->getBlock(12, CHUNK_EDGE - 1, 1) == BlockType::WATER);
    REQUIRE(restoredTop->getFluidState(12, CHUNK_EDGE - 1, 1) == FluidState::falling(3));
    REQUIRE(restoredTop->getWorldPosition().y + CHUNK_EDGE - 1 ==
            Catch::Approx(static_cast<float>(WORLD_MAX_Y)));
}

TEST_CASE("RYCH v4 preserves compact uniform cubes", "[serialization][storage]") {
    Chunk original(ChunkPos{-12, 31, 44});
    original.fill(BlockType::STONE);
    const std::vector<uint8_t> bytes = ChunkSerializer::serialize(original);
    REQUIRE(bytes.size() == HEADER_SIZE + 1);

    auto restored = ChunkSerializer::deserialize(bytes);
    REQUIRE(restored.has_value());
    REQUIRE(restored->pos() == ChunkPos{-12, 31, 44});
    REQUIRE(restored->isUniform());
    REQUIRE(restored->uniformBlock() == BlockType::STONE);
}

TEST_CASE("RYCH rejects legacy corrupt and out of range cube headers", "[serialization]") {
    Chunk cube(ChunkPos{1, 2, 3});
    std::vector<uint8_t> bytes = ChunkSerializer::serialize(cube);

    SECTION("legacy version") {
        auto corrupt = bytes;
        auto* header = reinterpret_cast<ChunkSaveHeader*>(corrupt.data());
        header->version = 3;
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("wrong magic") {
        auto corrupt = bytes;
        auto* header = reinterpret_cast<ChunkSaveHeader*>(corrupt.data());
        header->magic = 0;
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("invalid vertical section") {
        auto corrupt = bytes;
        auto* header = reinterpret_cast<ChunkSaveHeader*>(corrupt.data());
        header->chunkY = WORLD_MAX_CHUNK_Y + 1;
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("truncated payload") {
        auto corrupt = bytes;
        corrupt.pop_back();
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("invalid block byte") {
        auto corrupt = bytes;
        corrupt[HEADER_SIZE] = 0xFF;
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
    SECTION("unknown header flags") {
        auto corrupt = bytes;
        ChunkSaveHeader header{};
        std::memcpy(&header, corrupt.data(), sizeof(header));
        header.flags |= 1U << 31U;
        std::memcpy(corrupt.data(), &header, sizeof(header));
        REQUIRE_FALSE(ChunkSerializer::deserialize(corrupt).has_value());
    }
}

TEST_CASE("SaveManager keys cubes by full three dimensional position", "[save]") {
    TempDir directory("cubic_save");
    SaveManager saves(directory.path());

    Chunk lower(ChunkPos{7, 4, -5});
    lower.setBlock(1, 2, 3, BlockType::GOLD_ORE);
    lower.generated = true;
    Chunk upper(ChunkPos{7, 6, -5});
    upper.setBlock(1, 2, 3, BlockType::DIAMOND_ORE);
    upper.generated = true;
    saves.saveChunk(lower);
    saves.saveChunk(upper);
    saves.flush();

    auto loadedLower = saves.loadChunk({7, 4, -5});
    auto loadedUpper = saves.loadChunk({7, 6, -5});
    REQUIRE(loadedLower.has_value());
    REQUIRE(loadedUpper.has_value());
    REQUIRE(loadedLower->getBlock(1, 2, 3) == BlockType::GOLD_ORE);
    REQUIRE(loadedUpper->getBlock(1, 2, 3) == BlockType::DIAMOND_ORE);
    REQUIRE_FALSE(saves.loadChunk({7, 5, -5}).has_value());
    REQUIRE(saves.savedSections({7, -5}) == std::vector<int32_t>{4, 6});
}

TEST_CASE("SaveManager exposes queued cubic edits before disk write", "[save]") {
    TempDir directory("cubic_save_shield");
    SaveManager saves(directory.path());
    auto cube = std::make_shared<Chunk>(ChunkPos{3, 9, 4});
    cube->setBlock(2, 10, 2, BlockType::DIAMOND_ORE);
    cube->generated = true;
    saves.saveChunkAsync(cube);

    auto loaded = saves.loadChunk({3, 9, 4});
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->getBlock(2, 10, 2) == BlockType::DIAMOND_ORE);
    saves.flush();
}

TEST_CASE("SaveManager coalesces queued snapshots by cubic position", "[save][performance]") {
    TempDir directory("cubic_save_coalescing");
    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    hooks->pauseWrites.store(true, std::memory_order_release);
    SaveManager saves(directory.path(), hooks);
    constexpr ChunkPos POSITION{11, 7, -13};

    Chunk initial(POSITION);
    initial.setBlock(2, 3, 4, BlockType::STONE);
    saves.saveChunk(initial);
    for (int revision = 0; revision < 100; ++revision) {
        Chunk replacement(POSITION);
        replacement.setBlock(2, 3, 4,
                             revision == 99 ? BlockType::OBSIDIAN : BlockType::DIAMOND_ORE);
        saves.saveChunk(replacement);
    }

    REQUIRE(saves.pendingSaveCount() <= 2);
    REQUIRE(saves.coalescedSaveCount() >= 99);
    const auto pending = saves.loadChunk(POSITION);
    REQUIRE(pending.has_value());
    REQUIRE(pending->getBlock(2, 3, 4) == BlockType::OBSIDIAN);

    hooks->pauseWrites.store(false, std::memory_order_release);
    hooks->pauseWrites.notify_all();
    REQUIRE(saves.flush());
    const auto durable = saves.loadChunk(POSITION);
    REQUIRE(durable.has_value());
    REQUIRE(durable->getBlock(2, 3, 4) == BlockType::OBSIDIAN);
}

TEST_CASE("SaveManager teardown releases a paused writer and drains accepted saves",
          "[save][thread][shutdown][regression]") {
    TempDir directory("save_teardown_drain");
    const auto hooks = std::make_shared<SaveManager::TestHooks>();
    hooks->pauseWrites.store(true, std::memory_order_release);
    constexpr ChunkPos POSITION{-17, 11, 23};
    {
        auto saves = std::make_unique<SaveManager>(directory.path(), hooks);
        Chunk edited(POSITION);
        edited.setBlock(4, 5, 6, BlockType::DIAMOND_ORE);
        edited.generated = true;
        saves->saveChunk(edited);
        REQUIRE(saves->pendingSaveCount() == 1);
        REQUIRE_NOTHROW(saves.reset());
    }

    SaveManager reopened(directory.path());
    const std::optional<Chunk> loaded = reopened.loadChunk(POSITION);
    REQUIRE(loaded);
    REQUIRE(loaded->getBlock(4, 5, 6) == BlockType::DIAMOND_ORE);
}

TEST_CASE("SaveManager snapshots edited sections under one manifest lock", "[save][performance]") {
    TempDir directory("manifest_bulk_lookup");
    SaveManager saves(directory.path());
    for (ChunkPos position : {ChunkPos{2, 4, -3}, ChunkPos{2, 8, -3}, ChunkPos{-5, 1, 7}}) {
        Chunk cube(position);
        cube.setBlock(1, 1, 1, BlockType::GOLD_ORE);
        saves.saveChunk(cube);
    }
    REQUIRE(saves.flush());

    const std::vector<ColumnPos> columns = {{2, -3}, {-5, 7}, {99, 99}, {2, -3}};
    const auto sections = saves.savedSectionsForColumns(columns);
    REQUIRE(sections.size() == 2);
    REQUIRE(sections.at({2, -3}) == std::vector<int32_t>{4, 8});
    REQUIRE(sections.at({-5, 7}) == std::vector<int32_t>{1});
}

TEST_CASE("SaveManager metadata records the cubic format version", "[save]") {
    TempDir directory("cubic_metadata");
    SaveManager saves(directory.path());
    SaveManager::WorldMetadata written;
    written.seed = 12345;
    written.spawnPos = Vec3{100.f, 80.f, -50.f};
    written.worldTime = 9876543210ULL;
    saves.saveMetadata(written);

    auto metadata = saves.loadMetadata();
    REQUIRE(metadata.has_value());
    REQUIRE(metadata->seed == 12345);
    REQUIRE(metadata->playerPos == Vec3{100.f, 80.f, -50.f});
    REQUIRE(metadata->worldTime == 9876543210ULL);
    REQUIRE(metadata->chunkFormatVersion == CHUNK_VERSION);
    REQUIRE(metadata->generatorVersion == SaveManager::CURRENT_GENERATOR_VERSION);
}

TEST_CASE("SaveManager preserves non-chunk player metadata", "[save][metadata]") {
    TempDir directory("rycraft_player_metadata");
    SaveManager saves(directory.path());
    SaveManager::WorldMetadata written;
    written.seed = 9191;
    written.spawnPos = Vec3{12.0f, 88.0f, -4.0f};
    written.worldTime = 123456;
    written.player.yaw = 127.5f;
    written.player.pitch = -31.25f;
    written.player.health = 13;
    written.player.selectedSlot = 7;
    written.player.inventory[0] = ItemStack{itemFromBlock(BlockType::BASALT), 12, 0};
    written.player.inventory[35] = ItemStack{ItemType::IRON_PICKAXE, 1, 187};

    saves.saveMetadata(written);
    const auto loaded = saves.loadMetadata();

    REQUIRE(loaded.has_value());
    REQUIRE(loaded->seed == 9191);
    REQUIRE(loaded->playerPos == Vec3{12.0f, 88.0f, -4.0f});
    REQUIRE(loaded->worldTime == 123456);
    REQUIRE(loaded->player.yaw == Catch::Approx(127.5f));
    REQUIRE(loaded->player.pitch == Catch::Approx(-31.25f));
    REQUIRE(loaded->player.health == 13);
    REQUIRE(loaded->player.selectedSlot == 7);
    REQUIRE(loaded->player.inventory[0] == ItemStack{itemFromBlock(BlockType::BASALT), 12, 0});
    REQUIRE(loaded->player.inventory[35] == ItemStack{ItemType::IRON_PICKAXE, 1, 187});
}

TEST_CASE("SaveManager persists activated fluid frontiers across restart", "[save][fluid]") {
    TempDir directory("fluid_frontiers");
    const std::vector<FluidBoundaryFrontier> frontiers{
        {{-1, -64, -17}, {-1, -64, -16}},
        {{15, 64, 3}, {16, 64, 3}},
    };
    {
        SaveManager saves(directory.path());
        saves.saveDeferredFluidFrontiers(frontiers);
        REQUIRE(saves.loadDeferredFluidFrontiers() == frontiers);
    }
    {
        SaveManager reopened(directory.path());
        REQUIRE(reopened.loadDeferredFluidFrontiers() == frontiers);
    }
}

// ===========================================================================
// Cubic world access and streaming contracts
// ===========================================================================

TEST_CASE("World returns boundary blocks without loading invalid cubes", "[world][bounds]") {
    World world(42);
    REQUIRE(world.getLoadedChunkCount() == 0);
    REQUIRE(world.getBlock(0, WORLD_MIN_Y - 1, 0) == BlockType::BEDROCK);
    REQUIRE(world.getBlock(0, WORLD_MAX_Y + 1, 0) == BlockType::AIR);
    REQUIRE(world.getLoadedChunkCount() == 0);
}

TEST_CASE("World block edits report only committed resident changes",
          "[world][edit][transaction][regression]") {
    World world(42);
    const uint64_t initialLightingRevision = world.lightingRevision();

    REQUIRE_FALSE(world.trySetBlock(0, SEA_LEVEL, 0, BlockType::STONE));
    REQUIRE_FALSE(world.trySetBlock(0, WORLD_MIN_Y - 1, 0, BlockType::STONE));
    REQUIRE_FALSE(world.trySetBlock(0, WORLD_MAX_Y + 1, 0, BlockType::STONE));
    REQUIRE(world.getLoadedChunkCount() == 0);
    REQUIRE(world.getPendingFluidCount() == 0);
    REQUIRE(world.lightingRevision() == initialLightingRevision);

    auto bottom = world.getChunk({0, WORLD_MIN_CHUNK_Y, 0});
    auto top = world.getChunk({0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(bottom);
    REQUIRE(top);
    bottom->fill(BlockType::AIR);
    top->fill(BlockType::AIR);
    bottom->modifiedSinceSave = false;
    top->modifiedSinceSave = false;
    bottom->needsMeshUpdate = false;
    top->needsMeshUpdate = false;

    const uint64_t bottomVersionBefore = bottom->version.load(std::memory_order_relaxed);
    REQUIRE(world.trySetBlock(2, WORLD_MIN_Y, 3, BlockType::STONE));
    REQUIRE(world.findBlockIfLoaded(2, WORLD_MIN_Y, 3) == BlockType::STONE);
    REQUIRE(bottom->modifiedSinceSave);
    REQUIRE(bottom->needsMeshUpdate);
    REQUIRE(bottom->version.load(std::memory_order_relaxed) > bottomVersionBefore);
    REQUIRE(world.lightingRevision() > initialLightingRevision);

    const uint64_t noOpLightingRevision = world.lightingRevision();
    const uint64_t noOpVersion = bottom->version.load(std::memory_order_relaxed);
    const size_t noOpPendingFluids = world.getPendingFluidCount();
    REQUIRE_FALSE(world.trySetBlock(2, WORLD_MIN_Y, 3, BlockType::STONE));
    REQUIRE(world.lightingRevision() == noOpLightingRevision);
    REQUIRE(bottom->version.load(std::memory_order_relaxed) == noOpVersion);
    REQUIRE(world.getPendingFluidCount() == noOpPendingFluids);

    REQUIRE(world.trySetBlock(4, WORLD_MAX_Y, 5, BlockType::TORCH));
    REQUIRE(world.findBlockIfLoaded(4, WORLD_MAX_Y, 5) == BlockType::TORCH);
    REQUIRE(top->getBlockLight(4, CHUNK_EDGE - 1, 5) == 14);
}

TEST_CASE("World skips proven-empty gaps in sparse sky authority",
          "[world][snapshot][skylight][publication][regression]") {
    World world(42, 4, MAX_LOADED_CUBES, GenerationSettings{.structures = false});
    std::array<std::shared_ptr<const ColumnPlan>, 9> plans;
    int32_t firstRequiredSection = WORLD_MAX_CHUNK_Y;
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            auto plan = world.generator().getColumnPlan({offsetX, offsetZ});
            REQUIRE(plan);
            REQUIRE_FALSE(plan->exposedSections().empty());
            firstRequiredSection = std::min(firstRequiredSection, plan->exposedSections().front());
            plans[static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1)] = std::move(plan);
        }
    }
    const int32_t targetSection = firstRequiredSection - 4;
    REQUIRE(targetSection >= WORLD_MIN_CHUNK_Y);
    const ChunkPos target{0, targetSection, 0};
    REQUIRE(world.getChunk(target));

    const auto& centerPlan = plans[4];
    const int32_t missingRequiredSection = centerPlan->exposedSections().back();
    REQUIRE(missingRequiredSection > targetSection);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto& plan = plans[static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1)];
            for (const int32_t section : plan->exposedSections()) {
                if (offsetX == 0 && offsetZ == 0 && section == missingRequiredSection)
                    continue;
                REQUIRE(world.getChunk({offsetX, section, offsetZ}));
            }
        }
    }

    MeshSnapshot missingRequired;
    REQUIRE_FALSE(world.snapshotForMeshing(target, missingRequired));

    std::optional<int32_t> provenEmptyGap;
    for (int32_t section = targetSection + 1; section < missingRequiredSection; ++section) {
        if (!centerPlan->exposesSection(section)) {
            provenEmptyGap = section;
            break;
        }
    }
    REQUIRE(provenEmptyGap);
    REQUIRE_FALSE(world.isChunkLoaded({0, *provenEmptyGap, 0}));

    REQUIRE(world.getChunk({0, missingRequiredSection, 0}));
    MeshSnapshot connected;
    REQUIRE(world.snapshotForMeshing(target, connected));
    REQUIRE(connected.derivedSkyLightValid);
    REQUIRE_FALSE(world.isChunkLoaded({0, *provenEmptyGap, 0}));

    const auto firstVisibleLight = connected.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(target, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);

    // Removing the planned surface invalidates the sparse proof below it. A
    // nonexposed density section can still own the next opaque block, so the
    // edited column must fail closed until its bounded contiguous hull loads.
    constexpr int LOCAL_X = 8;
    constexpr int LOCAL_Z = 8;
    const int32_t localSurfaceY = centerPlan->surfaceY(LOCAL_X, LOCAL_Z);
    const int32_t localSurfaceSection = Chunk::worldToChunkY(localSurfaceY);
    std::optional<int32_t> unloadedOpaqueSection;
    for (int32_t section = localSurfaceSection - 1; section > targetSection; --section) {
        if (centerPlan->exposesSection(section) || world.isChunkLoaded({0, section, 0}))
            continue;
        Chunk probe(ChunkPos{0, section, 0});
        world.generator().generateCube(probe);
        for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
            if (isOpaque(probe.getBlock(LOCAL_X, localY, LOCAL_Z))) {
                unloadedOpaqueSection = section;
                break;
            }
        }
        if (unloadedOpaqueSection)
            break;
    }
    REQUIRE(unloadedOpaqueSection);
    REQUIRE_FALSE(world.isChunkLoaded({0, *unloadedOpaqueSection, 0}));
    const auto plannedSurfaceBlock = world.findBlockIfLoaded(LOCAL_X, localSurfaceY, LOCAL_Z);
    REQUIRE(plannedSurfaceBlock);
    REQUIRE(isOpaque(*plannedSurfaceBlock));

    int removedOpaqueBlocks = 0;
    for (int32_t section = *unloadedOpaqueSection + 1; section <= WORLD_MAX_CHUNK_Y; ++section) {
        if (!world.isChunkLoaded({0, section, 0}))
            continue;
        for (int localY = 0; localY < CHUNK_EDGE; ++localY) {
            const int32_t worldY = section * CHUNK_EDGE + localY;
            const auto block = world.findBlockIfLoaded(LOCAL_X, worldY, LOCAL_Z);
            if (!block || !isOpaque(*block))
                continue;
            REQUIRE(world.trySetBlock(LOCAL_X, worldY, LOCAL_Z, BlockType::AIR));
            ++removedOpaqueBlocks;
        }
    }
    REQUIRE(removedOpaqueBlocks > 0);
    MeshSnapshot incompleteLoweredCutoff;
    REQUIRE_FALSE(world.snapshotForMeshing(target, incompleteLoweredCutoff));
    REQUIRE_FALSE(world.isChunkLoaded({0, *unloadedOpaqueSection, 0}));

    const int32_t maximumSurfaceSection =
        Chunk::worldToChunkY(std::clamp(centerPlan->maximumSurfaceY(), WORLD_MIN_Y, WORLD_MAX_Y));
    for (int32_t section = targetSection; section <= maximumSurfaceSection; ++section) {
        REQUIRE(world.getChunk({0, section, 0}));
    }

    std::optional<int32_t> expectedCutoff;
    for (int32_t worldY = maximumSurfaceSection * CHUNK_EDGE + CHUNK_EDGE - 1;
         worldY >= targetSection * CHUNK_EDGE; --worldY) {
        const auto block = world.findBlockIfLoaded(LOCAL_X, worldY, LOCAL_Z);
        if (!block || !isOpaque(*block))
            continue;
        expectedCutoff = worldY + 1;
        break;
    }
    REQUIRE(expectedCutoff);
    MeshSnapshot loweredCutoff;
    REQUIRE(world.snapshotForMeshing(target, loweredCutoff));
    REQUIRE(loweredCutoff.skyCutoffAt(LOCAL_X, LOCAL_Z) == *expectedCutoff);
}

TEST_CASE("World withholds a first mesh until every sparse saved sky section is loaded",
          "[world][snapshot][skylight][save][publication][regression]") {
    TempDir directory("sparse_saved_skylight");
    SaveManager saves(directory.path());
    constexpr int32_t TARGET_SECTION = 4;
    constexpr int32_t ROOF_SECTION = 20;
    constexpr ChunkPos TARGET{0, TARGET_SECTION, 0};
    constexpr ChunkPos ROOF{0, ROOF_SECTION, 0};

    Chunk target(TARGET);
    target.fill(BlockType::AIR);
    target.generated = true;
    saves.saveChunk(target);
    Chunk roof(ROOF);
    roof.fill(BlockType::AIR);
    roof.setBlock(8, 8, 8, BlockType::STONE);
    roof.generated = true;
    saves.saveChunk(roof);
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setSaveManager(&saves);
    REQUIRE(world.getChunk(TARGET));
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ColumnPos column{offsetX, offsetZ};
            const auto plan = world.generator().getColumnPlan(column);
            REQUIRE(plan);
            // This same-section halo also completes the empty save-manifest
            // lookup for neighboring columns.
            REQUIRE(world.getChunk({column.x, TARGET_SECTION, column.z}));
            for (const int32_t section : plan->exposedSections()) {
                if (section < TARGET_SECTION)
                    continue;
                REQUIRE(world.getChunk({column.x, section, column.z}));
            }
        }
    }

    MeshSnapshot missingRoof;
    REQUIRE_FALSE(world.snapshotForMeshing(TARGET, missingRoof));
    REQUIRE_FALSE(world.isChunkLoaded({0, 12, 0}));

    REQUIRE(world.getChunk(ROOF));
    MeshSnapshot firstVisible;
    REQUIRE(world.snapshotForMeshing(TARGET, firstVisible));
    REQUIRE(firstVisible.derivedSkyLightValid);
    REQUIRE(firstVisible.skyCutoffAt(8, 8) == ROOF_SECTION * CHUNK_EDGE + 9);
    REQUIRE_FALSE(world.isChunkLoaded({0, 12, 0}));

    const auto firstVisibleLight = firstVisible.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(TARGET, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);
}

TEST_CASE("Resident flora waits for saved sky authority before its first mesh",
          "[world][snapshot][skylight][save][publication][flora][regression]") {
    TempDir directory("resident_saved_skylight");
    SaveManager saves(directory.path());
    constexpr ChunkPos TARGET{0, WORLD_MAX_CHUNK_Y, 0};
    constexpr int32_t TARGET_Y = WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8;
    constexpr int64_t MOVED_CHUNK_X = 3;

    Chunk saved(TARGET);
    // Keep the saved plant in a sky-open stone shaft. When this column's
    // manifest is unknown, authoritative skylight from the still-active east
    // neighbor cannot leak sideways into the test cell and make the stale
    // value depend on light-queue order.
    saved.fill(BlockType::STONE);
    for (int localY = 8; localY < CHUNK_EDGE; ++localY)
        saved.setBlock(8, localY, 8, BlockType::AIR);
    saved.setBlock(8, 7, 8, BlockType::GRASS);
    saved.setBlock(8, 8, 8, BlockType::SHRUB);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    world.setSaveManager(&saves);
    std::shared_ptr<Chunk> target;
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({offsetX, offsetZ}));
            const std::shared_ptr<Chunk> resident = world.getChunk({offsetX, TARGET.y, offsetZ});
            REQUIRE(resident);
            if (offsetX == 0 && offsetZ == 0)
                target = resident;
        }
    }
    REQUIRE(target);
    REQUIRE(target->getBlock(8, 8, 8) == BlockType::SHRUB);
    REQUIRE(target->getSkyLight(8, 8, 8) == MAX_DERIVED_LIGHT_LEVEL);
    REQUIRE(world.isChunkLoaded(TARGET));
    world.generateAroundPlayer(0, TARGET_Y, 0);
    REQUIRE(world.isChunkLoaded(TARGET));

    // Moving the active manifest away while retaining the loaded cube is a
    // production streaming state. Ordinary reconciliation removes its direct
    // sky seed because the saved ceiling is unknown, but unloading has not run.
    world.generateAroundPlayer(MOVED_CHUNK_X * CHUNK_EDGE, TARGET_Y, 0);
    REQUIRE(target->generated);
    REQUIRE(world.isChunkLoaded(TARGET));
    for (int pass = 0; pass < 8; ++pass)
        world.reconcileLight(64);
    REQUIRE(world.isChunkLoaded(TARGET));
    const uint8_t staleSky = target->getSkyLight(8, 8, 8);
    REQUIRE(staleSky < MAX_DERIVED_LIGHT_LEVEL);

    MeshSnapshot unknownManifest;
    REQUIRE_FALSE(world.snapshotForMeshing(TARGET, unknownManifest));

    // The return rebuild publishes the complete manifest in one bulk snapshot.
    // The already resident cube must remain unavailable until its packed light
    // catches up with that newly authoritative snapshot.
    world.generateAroundPlayer(0, TARGET_Y, 0);
    MeshSnapshot firstVisible;
    bool ready = world.snapshotForMeshing(TARGET, firstVisible);
    if (ready) {
        // A lightly loaded worker set may finish the bounded publication
        // transaction inside the active-set rebuild. It may publish only the
        // final sky value, never the stale dim mesh from the prior manifest.
        REQUIRE(firstVisible.skyLightAt(8, 8, 8) == MAX_DERIVED_LIGHT_LEVEL);
    }
    for (int pass = 0; pass < 32 && !ready; ++pass) {
        world.reconcileLight(32);
        ready = world.snapshotForMeshing(TARGET, firstVisible);
    }
    REQUIRE(ready);
    REQUIRE(firstVisible.derivedSkyLightValid);
    REQUIRE(firstVisible.skyLightAt(8, 8, 8) == MAX_DERIVED_LIGHT_LEVEL);

    MeshScratch scratch;
    const MeshOutput firstMesh = LODMesher::buildMesh(firstVisible, scratch);
    int shrubVertices = 0;
    for (const Vertex& vertex : firstMesh.vertices) {
        if (unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::SHRUB)) {
            continue;
        }
        ++shrubVertices;
        REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::CROSS);
        REQUIRE(unpackSkyLight(vertex.faceAttr) == MAX_DERIVED_LIGHT_LEVEL);
    }
    REQUIRE(shrubVertices > 0);

    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(TARGET, settled));
    REQUIRE(settled.packedLight == firstVisible.packedLight);
    const MeshOutput settledMesh = LODMesher::buildMesh(settled, scratch);
    REQUIRE(settledMesh.vertices.size() == firstMesh.vertices.size());
    REQUIRE(std::memcmp(settledMesh.vertices.data(), firstMesh.vertices.data(),
                        firstMesh.vertices.size() * sizeof(Vertex)) == 0);
    REQUIRE(settledMesh.indices == firstMesh.indices);
}

TEST_CASE("World caches cubes by X Y and Z", "[world]") {
    World world(42);
    auto lower = world.getChunk({5, 3, -3});
    auto lowerAgain = world.getChunk({5, 3, -3});
    auto upper = world.getChunk({5, 4, -3});
    REQUIRE(lower == lowerAgain);
    REQUIRE(lower != upper);
    REQUIRE(lower->pos() == ChunkPos{5, 3, -3});
    REQUIRE(upper->pos() == ChunkPos{5, 4, -3});
    REQUIRE(world.getLoadedChunkCount() == 2);
}

TEST_CASE("World edits address the correct vertical cube", "[world]") {
    World world(42);
    constexpr int32_t worldY = 200;
    const ChunkPos position{0, Chunk::worldToChunkY(worldY), 0};
    auto cube = world.getChunk(position);
    cube->needsMeshUpdate = false;
    world.setBlock(5, worldY, 6, BlockType::PLANKS);

    REQUIRE(world.getBlockIfLoaded(5, worldY, 6) == BlockType::PLANKS);
    REQUIRE(cube->getBlock(5, Chunk::worldToLocalY(worldY), 6) == BlockType::PLANKS);
    REQUIRE(cube->needsMeshUpdate);
    REQUIRE(cube->modifiedSinceSave);
}

TEST_CASE("Loaded world snapshots are reused until the cube set changes", "[world][snapshot]") {
    World world(42);
    world.getChunk({0, 4, 0});
    world.publishLoadedSnapshot();
    auto first = world.getLoadedSnapshot();
    auto same = world.getLoadedSnapshot();
    REQUIRE(first == same);
    REQUIRE(first->size() == 1);

    world.getChunk({0, 5, 0});
    REQUIRE(world.getLoadedSnapshot() == first);
    world.publishLoadedSnapshot();
    auto changed = world.getLoadedSnapshot();
    REQUIRE(changed != first);
    REQUIRE(changed->size() == 2);
}

TEST_CASE("Chunk distance sorting includes vertical distance", "[world][priority]") {
    std::vector<ChunkPos> cubes{{0, 4, 0}, {0, 10, 0}, {3, 4, 4}, {-1, 4, 0}};
    sortChunksByDistance(cubes, 0, 4, 0);
    REQUIRE(cubes.front() == ChunkPos{0, 10, 0});
    REQUIRE(cubes.back() == ChunkPos{0, 4, 0});
}

TEST_CASE("Exact streaming priority preserves camera lanes across cold work and camera jumps",
          "[world][streaming][priority][cold-start][camera-jump][regression]") {
    STATIC_REQUIRE(EXACT_GENERATION_WORKER_COUNT == 6);
    STATIC_REQUIRE(EXACT_LATENCY_WORKER_COUNT == 4);
    STATIC_REQUIRE(EXACT_GENERATION_SUBMISSION_LIMIT == EXACT_GENERATION_WORKER_COUNT + 1);
    STATIC_REQUIRE(EXACT_GENERATION_SUBMISSION_LIMIT <= MAX_INFLIGHT_GEN);
    STATIC_REQUIRE(EXACT_MESH_WORKER_COUNT == 4);

    constexpr uint64_t EPOCH = 37;
    const int64_t camera = exactStreamingTaskPriority(EPOCH, 7, 128);
    const int64_t explorationEdge = exactStreamingTaskPriority(EPOCH, 6, 0);
    const int64_t broadSurface = exactStreamingTaskPriority(EPOCH, 4, 0);
    REQUIRE(camera > explorationEdge);
    REQUIRE(explorationEdge > broadSurface);

    // Distance resolves work only inside one lane. A new active-set epoch
    // jumps every still-queued task from the old camera position.
    REQUIRE(exactStreamingTaskPriority(EPOCH, 6, 4) > exactStreamingTaskPriority(EPOCH, 6, 64));
    REQUIRE(exactStreamingTaskPriority(EPOCH + 1, 3, 4'096) >
            exactStreamingTaskPriority(EPOCH, 7, 0));

    // Broad and medium work never occupy every exact worker. A newly queued
    // camera epoch can therefore begin without waiting for a complete stale
    // generation wave to finish.
    STATIC_REQUIRE(exactStreamingPlanSubmissionLimit(EXACT_STREAMING_SURFACE_PRIORITY_LANE) == 2);
    STATIC_REQUIRE(exactStreamingPlanSubmissionLimit(EXACT_STREAMING_EDITED_PRIORITY_LANE) == 2);
    STATIC_REQUIRE(exactStreamingPlanSubmissionLimit(EXACT_STREAMING_EXPLORATION_PRIORITY_LANE) ==
                   3);
    STATIC_REQUIRE(exactStreamingPlanSubmissionLimit(EXACT_STREAMING_CAMERA_PRIORITY_LANE) ==
                   MAX_COLD_COLUMN_PLANS);
    STATIC_REQUIRE(exactStreamingCubeSubmissionLimit(EXACT_STREAMING_SURFACE_PRIORITY_LANE) == 3);
    STATIC_REQUIRE(exactStreamingCubeSubmissionLimit(EXACT_STREAMING_EDITED_PRIORITY_LANE) == 3);
    STATIC_REQUIRE(exactStreamingCubeSubmissionLimit(EXACT_STREAMING_EXPLORATION_PRIORITY_LANE) ==
                   4);
    STATIC_REQUIRE(exactStreamingCubeSubmissionLimit(EXACT_STREAMING_CAMERA_PRIORITY_LANE) ==
                   EXACT_GENERATION_SUBMISSION_LIMIT);
    STATIC_REQUIRE(exactStreamingPlanSubmissionLimit(EXACT_STREAMING_EDITED_PRIORITY_LANE) +
                       exactStreamingCubeSubmissionLimit(EXACT_STREAMING_EDITED_PRIORITY_LANE) <
                   EXACT_GENERATION_WORKER_COUNT);

    constexpr ChunkPos LIGHT_CENTER{200, 8, -300};
    STATIC_REQUIRE(exactPublicationLightPriority(LIGHT_CENTER, LIGHT_CENTER) >
                   exactPublicationLightPriority(
                       {LIGHT_CENTER.x + 20, LIGHT_CENTER.y, LIGHT_CENTER.z}, LIGHT_CENTER));
    STATIC_REQUIRE(exactPublicationLightPriority(
                       {LIGHT_CENTER.x + 1, LIGHT_CENTER.y, LIGHT_CENTER.z}, LIGHT_CENTER) >
                   exactPublicationLightPriority(
                       {LIGHT_CENTER.x + 1, LIGHT_CENTER.y + 20, LIGHT_CENTER.z}, LIGHT_CENTER));
}

TEST_CASE("Airborne exact streaming prioritizes the surface below negative movement",
          "[world][streaming][priority][collision][flight][regression]") {
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(0, 0) == EXACT_STREAMING_CAMERA_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingPrimarySurfacePriorityLane(0, 0) ==
                   EXACT_STREAMING_CAMERA_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(-EXPLORATION_RADIUS_CHUNKS, 0) ==
                   EXACT_STREAMING_EXPLORATION_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(0, -EXPLORATION_RADIUS_CHUNKS) ==
                   EXACT_STREAMING_EXPLORATION_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(-EXPLORATION_RADIUS_CHUNKS, -1) ==
                   EXACT_STREAMING_EDITED_PRIORITY_LANE);
    STATIC_REQUIRE(
        exactStreamingSurfacePriorityLane(EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS,
                                          0) == EXACT_STREAMING_EDITED_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(
                       EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS + 1, 0) ==
                   EXACT_STREAMING_SURFACE_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingPrimarySurfacePriorityLane(
                       EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS + 1, 0) ==
                   EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingFloraPriorityLane(0, 0) == EXACT_STREAMING_CAMERA_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingFloraPriorityLane(EXPLORATION_RADIUS_CHUNKS, 0) ==
                   EXACT_STREAMING_EXPLORATION_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingFloraPriorityLane(EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS,
                                                   0) == EXACT_STREAMING_FLORA_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingFloraPriorityLane(EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS + 1,
                                                   0) == EXACT_STREAMING_SURFACE_PRIORITY_LANE);

    constexpr uint64_t EPOCH = 91;
    constexpr uint64_t MAXIMUM_VERTICAL_DISTANCE_SQUARED =
        static_cast<uint64_t>(WORLD_VERTICAL_CHUNKS) * WORLD_VERTICAL_CHUNKS;
    const int64_t surfaceBelowCamera = exactStreamingTaskPriority(
        EPOCH, exactStreamingSurfacePriorityLane(0, 0), MAXIMUM_VERTICAL_DISTANCE_SQUARED);
    const int64_t optionalAirborneNeighbor =
        exactStreamingTaskPriority(EPOCH, EXACT_STREAMING_EXPLORATION_PRIORITY_LANE, 0);
    const int64_t explorationSurface = exactStreamingTaskPriority(
        EPOCH, exactStreamingSurfacePriorityLane(-EXPLORATION_RADIUS_CHUNKS, 0),
        MAXIMUM_VERTICAL_DISTANCE_SQUARED);
    const int64_t broadPrimary =
        exactStreamingTaskPriority(EPOCH, EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE, 0);

    REQUIRE(surfaceBelowCamera > optionalAirborneNeighbor);
    REQUIRE(explorationSurface > broadPrimary);
    REQUIRE(exactStreamingTaskPriority(
                EPOCH,
                exactStreamingFloraPriorityLane(EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS, 0),
                EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS *
                    EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS) >
            exactStreamingTaskPriority(
                EPOCH, EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE,
                (EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS + 1) *
                    (EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS + 1)));
    REQUIRE(
        exactStreamingTaskPriority(EPOCH,
                                   exactStreamingSurfacePriorityLane(
                                       EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS, 0),
                                   MAXIMUM_VERTICAL_DISTANCE_SQUARED) >
        exactStreamingTaskPriority(
            EPOCH, exactStreamingFloraPriorityLane(EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS, 0),
            0));
}

TEST_CASE("Complete exact disk surfaces retain protected multi-section priority",
          "[world][streaming][priority][surface][exact-disk][multi-section][regression]") {
    constexpr uint64_t EPOCH = 117;
    constexpr int EDGE_DX = EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS;
    constexpr int OUTSIDE_DX = EXACT_STREAMING_REQUIRED_SURFACE_PRIORITY_RADIUS_CHUNKS + 1;
    STATIC_REQUIRE(EDGE_DX == MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
    STATIC_REQUIRE(exactStreamingSurfacePriorityLane(EDGE_DX, 0) ==
                   EXACT_STREAMING_EDITED_PRIORITY_LANE);
    STATIC_REQUIRE(exactStreamingPrimarySurfacePriorityLane(EDGE_DX, 0) ==
                   exactStreamingSurfacePriorityLane(EDGE_DX, 0));
    STATIC_REQUIRE(exactStreamingPrimarySurfacePriorityLane(OUTSIDE_DX, 0) ==
                   EXACT_STREAMING_PRIMARY_SURFACE_PRIORITY_LANE);

    const int64_t outsidePrimary = exactStreamingTaskPriority(
        EPOCH, exactStreamingPrimarySurfacePriorityLane(OUTSIDE_DX, 0), 0);
    constexpr std::array<int32_t, 4> EDGE_SURFACE_SECTIONS{-8, 0, 24, 87};
    for (const int32_t section : EDGE_SURFACE_SECTIONS) {
        const int64_t verticalDistance = static_cast<int64_t>(section) - 40;
        const uint64_t distanceSquared =
            static_cast<uint64_t>(EDGE_DX * EDGE_DX) +
            static_cast<uint64_t>(verticalDistance * verticalDistance) * 2U;
        CAPTURE(section, distanceSquared);
        const int64_t required = exactStreamingTaskPriority(
            EPOCH, exactStreamingSurfacePriorityLane(EDGE_DX, 0), distanceSquared);
        REQUIRE(required > outsidePrimary);
        REQUIRE(required ==
                exactStreamingTaskPriority(
                    EPOCH, exactStreamingPrimarySurfacePriorityLane(EDGE_DX, 0), distanceSquared));
    }

    constexpr ChunkPos CENTER{0, 40, 0};
    constexpr ChunkPos NEAR_TALL{7, WORLD_MAX_CHUNK_Y, 0};
    constexpr ChunkPos FAR_FLAT{EDGE_DX, CENTER.y, 0};
    STATIC_REQUIRE(exactStreamingCubePriorityDistance(NEAR_TALL, CENTER) <
                   exactStreamingCubePriorityDistance(FAR_FLAT, CENTER));
    STATIC_REQUIRE(
        exactStreamingTaskPriority(EPOCH,
                                   exactStreamingSurfacePriorityLane(NEAR_TALL.x, NEAR_TALL.z),
                                   exactStreamingCubePriorityDistance(NEAR_TALL, CENTER)) >
        exactStreamingTaskPriority(EPOCH, exactStreamingSurfacePriorityLane(FAR_FLAT.x, FAR_FLAT.z),
                                   exactStreamingCubePriorityDistance(FAR_FLAT, CENTER)));
}

TEST_CASE("Stale column plans requeue when a camera jump requires them again",
          "[world][streaming][priority][cold-start][camera-jump][regression]") {
    using Action = ColumnPlanCompletionAction;
    REQUIRE(columnPlanCompletionAction(true, false, false, false) == Action::PUBLISH);
    REQUIRE(columnPlanCompletionAction(false, false, true, false) == Action::REQUEUE);
    REQUIRE(columnPlanCompletionAction(false, false, true, true) == Action::DROP);
    REQUIRE(columnPlanCompletionAction(false, false, false, false) == Action::DROP);
    REQUIRE(columnPlanCompletionAction(false, true, true, false) == Action::DROP);
}

TEST_CASE("Deferred column plan retries wait for prior future reaping",
          "[world][streaming][generator-v4][cold-start][regression]") {
    using Action = ColumnPlanRetryPublicationAction;

    // The worker may have recorded a retry and released its function body,
    // but a concurrent preparation pump must retain the active reservation
    // until the corresponding future itself reports ready.
    REQUIRE(columnPlanRetryPublicationAction(false, true, true, false, false, true) ==
            Action::HOLD);
    REQUIRE(columnPlanRetryPublicationAction(true, true, true, false, false, true) ==
            Action::REQUEUE);
    REQUIRE(columnPlanRetryPublicationAction(true, true, false, false, false, true) ==
            Action::DROP);
    REQUIRE(columnPlanRetryPublicationAction(true, true, true, true, false, true) == Action::DROP);
    REQUIRE(columnPlanRetryPublicationAction(true, true, true, false, true, true) == Action::DROP);
    REQUIRE(columnPlanRetryPublicationAction(true, true, true, false, false, false) ==
            Action::DROP);
}

TEST_CASE("World loads a saved cubic edit before generation", "[world][save]") {
    TempDir directory("world_cubic_load");
    SaveManager saves(directory.path());
    Chunk saved(ChunkPos{2, 12, -1});
    saved.setBlock(4, 8, 9, BlockType::DIAMOND_ORE);
    saved.generated = true;
    saves.saveChunk(saved);
    saves.flush();

    World world(42);
    world.setSaveManager(&saves);
    auto loaded = world.getChunk({2, 12, -1});
    REQUIRE(loaded->getBlock(4, 8, 9) == BlockType::DIAMOND_ORE);
}

TEST_CASE("World streaming remains within the cubic loaded cap", "[world][async]") {
    World world(42, 1);
    REQUIRE_FALSE(world.exactSpawnBandReady(0, SEA_LEVEL, 0, 0));
    world.updatePlayerPosition(0, SEA_LEVEL, 0);
    for (int attempt = 0; attempt < 2000 && !world.exactSpawnBandReady(0, SEA_LEVEL, 0, 0);
         ++attempt) {
        world.updatePlayerPosition(0, SEA_LEVEL, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(world.getLoadedChunkCount() <= MAX_LOADED_CUBES);
    REQUIRE(world.getLoadedChunkCount() > 0);
    REQUIRE(world.getStreamingWorkStats().loadedCubeHighWater <= MAX_LOADED_CUBES);
    REQUIRE(world.exactSpawnBandReady(0, SEA_LEVEL, 0, 0));
    const std::optional<Vec3> safeSpawn = world.safeSpawnFromReadyPlans(0, 0, 0);
    REQUIRE(safeSpawn);
    const int64_t spawnX = static_cast<int64_t>(std::floor(safeSpawn->x));
    const int32_t spawnY = static_cast<int32_t>(std::floor(safeSpawn->y));
    const int64_t spawnZ = static_cast<int64_t>(std::floor(safeSpawn->z));
    REQUIRE_FALSE(isSolid(world.getBlockIfLoaded(spawnX, spawnY, spawnZ)));
    REQUIRE_FALSE(isSolid(world.getBlockIfLoaded(spawnX, spawnY + 1, spawnZ)));
    REQUIRE(safeSpawn->y > WORLD_MIN_Y);
    REQUIRE(safeSpawn->y < WORLD_MAX_Y);
}

TEST_CASE("Playable spawn waits for its complete closed-collision halo",
          "[world][spawn][streaming][collision][regression]") {
    World world(42, 1);
    constexpr int64_t spawnX = -1;
    constexpr int32_t spawnY = 200;
    constexpr int64_t spawnZ = -1;
    const ChunkPos center{Chunk::worldToChunk(spawnX), Chunk::worldToChunkY(spawnY),
                          Chunk::worldToChunk(spawnZ)};
    const ChunkPos finalNeighbor{center.x + PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS,
                                 center.y + PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES,
                                 center.z + PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS};

    REQUIRE_FALSE(world.playableSpawnCollisionReady(spawnX, spawnY, spawnZ));
    for (int offsetZ = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
         offsetZ <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetZ) {
        for (int offsetX = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
             offsetX <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetX) {
            for (int offsetY = -PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES;
                 offsetY <= PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES; ++offsetY) {
                const ChunkPos position{center.x + offsetX, center.y + offsetY, center.z + offsetZ};
                if (position == finalNeighbor)
                    continue;
                REQUIRE(world.getChunk(position));
            }
        }
    }

    REQUIRE_FALSE(world.isChunkLoaded(finalNeighbor));
    REQUIRE_FALSE(world.playableSpawnCollisionReady(spawnX, spawnY, spawnZ));
    REQUIRE(world.getChunk(finalNeighbor));
    REQUIRE(world.playableSpawnCollisionReady(spawnX, spawnY, spawnZ));
}

TEST_CASE("Cold radius-zero streaming generates the playable collision halo",
          "[world][spawn][streaming][collision][cold-start][regression]") {
    STATIC_REQUIRE(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS == 0);
    STATIC_REQUIRE(exactStreamingActiveSetRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) ==
                   1);

    World world(42, 1);
    REQUIRE_FALSE(world.generationContext());
    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);

    constexpr int64_t spawnX = -1;
    constexpr int32_t spawnY = 200;
    constexpr int64_t spawnZ = -1;
    const ChunkPos center{Chunk::worldToChunk(spawnX), Chunk::worldToChunkY(spawnY),
                          Chunk::worldToChunk(spawnZ)};

    world.updatePlayerPosition(spawnX, spawnY, spawnZ);
    for (int attempt = 0;
         attempt < 2'000 && !world.playableSpawnCollisionReady(spawnX, spawnY, spawnZ); ++attempt) {
        world.updatePlayerPosition(spawnX, spawnY, spawnZ);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    REQUIRE(world.playableSpawnCollisionReady(spawnX, spawnY, spawnZ));
    size_t residentHaloCubes = 0;
    for (int offsetZ = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
         offsetZ <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetZ) {
        for (int offsetX = -PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS;
             offsetX <= PLAYABLE_SPAWN_COLLISION_HORIZONTAL_HALO_CHUNKS; ++offsetX) {
            for (int offsetY = -PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES;
                 offsetY <= PLAYABLE_SPAWN_COLLISION_VERTICAL_HALO_CUBES; ++offsetY) {
                const ChunkPos position{center.x + offsetX, center.y + offsetY, center.z + offsetZ};
                CAPTURE(position.x, position.y, position.z);
                REQUIRE(world.isChunkLoaded(position));
                ++residentHaloCubes;
            }
        }
    }
    REQUIRE(residentHaloCubes == 27);
}

TEST_CASE("Safe spawn rejects lava in both breathing cells", "[world][spawn][lava][regression]") {
    World world(42, 1);
    world.updatePlayerPosition(0, SEA_LEVEL, 0);
    for (int attempt = 0;
         attempt < 2000 &&
         !world.exactSpawnBandReady(0, SEA_LEVEL, 0, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
         ++attempt) {
        world.updatePlayerPosition(0, SEA_LEVEL, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(world.exactSpawnBandReady(0, SEA_LEVEL, 0, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS));
    REQUIRE(world.safeSpawnFromReadyPlans(0, 0, 0));

    const std::shared_ptr<const ColumnPlan> plan = world.generator().findColumnPlan({0, 0});
    REQUIRE(plan);
    struct SpawnCellPair {
        std::shared_ptr<Chunk> chunk;
        int64_t x = 0;
        int32_t feetY = 0;
        int64_t z = 0;
        BlockType feet = BlockType::AIR;
        BlockType head = BlockType::AIR;
    };
    std::vector<SpawnCellPair> cells;
    cells.reserve(CHUNK_EDGE * CHUNK_EDGE);
    for (int64_t z = 0; z < CHUNK_EDGE; ++z) {
        for (int64_t x = 0; x < CHUNK_EDGE; ++x) {
            const worldgen::SurfaceSample sample =
                plan->sample(static_cast<int>(x), static_cast<int>(z));
            const int32_t feetY = static_cast<int32_t>(std::ceil(sample.terrainHeight));
            REQUIRE(feetY >= WORLD_MIN_Y + 1);
            REQUIRE(feetY + 1 <= WORLD_MAX_Y);
            const ChunkPos cubePosition{x / CHUNK_EDGE, Chunk::worldToChunkY(feetY),
                                        z / CHUNK_EDGE};
            REQUIRE(world.isChunkLoaded(cubePosition));
            std::shared_ptr<Chunk> chunk = world.getChunk(cubePosition);
            REQUIRE(chunk);
            const std::optional<BlockType> feet = world.findBlockIfLoaded(x, feetY, z);
            const std::optional<BlockType> head = world.findBlockIfLoaded(x, feetY + 1, z);
            REQUIRE(feet);
            REQUIRE(head);
            cells.push_back({chunk, x, feetY, z, *feet, *head});
        }
    }

    for (const SpawnCellPair& cell : cells) {
        cell.chunk->setBlockWorld(cell.x, cell.feetY, cell.z, BlockType::LAVA);
    }
    REQUIRE_FALSE(world.safeSpawnFromReadyPlans(0, 0, 0));

    for (const SpawnCellPair& cell : cells) {
        cell.chunk->setBlockWorld(cell.x, cell.feetY, cell.z, cell.feet);
        cell.chunk->setBlockWorld(cell.x, cell.feetY + 1, cell.z, BlockType::LAVA);
    }
    REQUIRE_FALSE(world.safeSpawnFromReadyPlans(0, 0, 0));

    for (const SpawnCellPair& cell : cells) {
        cell.chunk->setBlockWorld(cell.x, cell.feetY + 1, cell.z, cell.head);
    }
    REQUIRE(world.safeSpawnFromReadyPlans(0, 0, 0));
}

TEST_CASE("World rejects synchronous cube admission at its configured hard cap",
          "[world][streaming][capacity][performance][regression]") {
    constexpr size_t TEST_CAP = 2;
    World world(42, MIN_RENDER_DISTANCE_CHUNKS, TEST_CAP);
    REQUIRE(world.getChunk({0, WORLD_MAX_CHUNK_Y, 0}));
    REQUIRE(world.getChunk({0, WORLD_MAX_CHUNK_Y - 1, 0}));
    REQUIRE_FALSE(world.getChunk({0, WORLD_MAX_CHUNK_Y - 2, 0}));
    REQUIRE(world.getLoadedChunkCount() == TEST_CAP);

    const StreamingWorkStats capped = world.getStreamingWorkStats();
    REQUIRE(capped.loadedCubeAdmissionsRejected == 1);
    REQUIRE(capped.loadedCubeHighWater == TEST_CAP);
    REQUIRE(capped.loadedCubeHighWater <= MAX_LOADED_CUBES);

    world.unloadDistantChunks();
    REQUIRE(world.getLoadedChunkCount() == 0);
    REQUIRE(world.getChunk({0, WORLD_MAX_CHUNK_Y - 2, 0}));
    REQUIRE(world.getLoadedChunkCount() == 1);
    REQUIRE(world.getStreamingWorkStats().loadedCubeHighWater == TEST_CAP);
}

TEST_CASE("Exact priority metadata survives a cap too small for complete mesh halos",
          "[world][streaming][priority][capacity][regression]") {
    constexpr size_t TINY_CAP = 2;
    World world(42, MIN_RENDER_DISTANCE_CHUNKS, TINY_CAP);
    REQUIRE_NOTHROW(world.generateAroundPlayer(0, SEA_LEVEL, 0));
    REQUIRE(world.getLoadedChunkCount() <= TINY_CAP);
    REQUIRE(world.getStreamingWorkStats().loadedCubeHighWater <= TINY_CAP);
}

TEST_CASE("World teardown drains queued plans and chunks across repeated worlds",
          "[world][streaming][thread][shutdown][regression]") {
    constexpr int WORLD_COUNT = 12;
    constexpr int64_t WORLD_SPACING_CHUNKS = 16;
    for (int index = 0; index < WORLD_COUNT; ++index) {
        INFO("world iteration=" << index);
        auto world = std::make_unique<World>(42 + static_cast<uint64_t>(index),
                                             MIN_RENDER_DISTANCE_CHUNKS, 64);
        const int64_t centerChunkX = static_cast<int64_t>(index) * WORLD_SPACING_CHUNKS;
        const int64_t centerBlockX = centerChunkX * CHUNK_EDGE;

        // A cached owning plan lets non-surface cubes enter the generation
        // backlog immediately. The other worlds begin with plan work only, so
        // repeated teardown covers both scheduler phases without test hooks.
        if (index == 1) {
            REQUIRE(world->generator().getColumnPlan({centerChunkX, 0}));
        }
        world->generateAroundPlayer(centerBlockX, SEA_LEVEL, 0);
        REQUIRE(world->getPendingChunkCount() > 0);
        REQUIRE_NOTHROW(world.reset());
    }
}

TEST_CASE("Concurrent cube admission cannot race past the loaded hard cap",
          "[world][streaming][capacity][performance][concurrency][regression]") {
    constexpr size_t TEST_CAP = 2;
    constexpr size_t REQUEST_COUNT = 4;
    World world(42, MIN_RENDER_DISTANCE_CHUNKS, TEST_CAP);
    std::array<std::shared_ptr<Chunk>, REQUEST_COUNT> results;
    std::atomic<bool> start{false};
    std::array<std::thread, REQUEST_COUNT> workers;
    for (size_t index = 0; index < workers.size(); ++index) {
        workers[index] = std::thread([&, index] {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            results[index] =
                world.getChunk({0, WORLD_MAX_CHUNK_Y - static_cast<int32_t>(index), 0});
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread& worker : workers)
        worker.join();

    REQUIRE(std::ranges::count_if(results, [](const auto& result) { return result != nullptr; }) ==
            TEST_CAP);
    REQUIRE(world.getLoadedChunkCount() == TEST_CAP);
    const StreamingWorkStats stats = world.getStreamingWorkStats();
    REQUIRE(stats.loadedCubeAdmissionsRejected == REQUEST_COUNT - TEST_CAP);
    REQUIRE(stats.loadedCubeHighWater == TEST_CAP);
}

TEST_CASE("Column plan completions wake only registered cube dependencies",
          "[world][streaming][performance]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    world.updatePlayerPosition(0, SEA_LEVEL, 0);

    StreamingWorkStats work;
    for (int attempt = 0; attempt < 3000; ++attempt) {
        work = world.getStreamingWorkStats();
        if (work.completedColumnPlans >= COLUMN_PLAN_REBUILD_BATCH)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    INFO("apron centers=" << work.planApronCenters
                          << " actual expansions=" << work.planApronExpansionAttempts
                          << " per-cube equivalent=" << work.planApronCubeExpansionEquivalent);
    INFO("completed plans=" << work.completedColumnPlans
                            << " dependent checks=" << work.planDependentChecks
                            << " full-scan equivalent=" << work.fullRetainedScanEquivalent
                            << " rebuild notifications=" << work.activeSetRebuildNotifications);
    REQUIRE(work.activeSetRebuilds == 1);
    REQUIRE(work.planApronCenters > 0);
    REQUIRE(work.planApronExpansionAttempts == work.planApronCenters * 25);
    REQUIRE(work.planApronExpansionAttempts < work.planApronCubeExpansionEquivalent);
    REQUIRE(work.completedColumnPlans >= COLUMN_PLAN_REBUILD_BATCH);
    REQUIRE(work.planDependentChecks < work.fullRetainedScanEquivalent);
    REQUIRE(work.activeSetRebuildNotifications <=
            work.completedColumnPlans / COLUMN_PLAN_REBUILD_BATCH + 1);

    const uint64_t rebuildsBeforeCooldown = work.activeSetRebuilds;
    for (size_t tick = 0; tick < COLUMN_PLAN_REBUILD_COOLDOWN_TICKS; ++tick) {
        world.updatePlayerPosition(0, SEA_LEVEL, 0);
        REQUIRE(world.getStreamingWorkStats().activeSetRebuilds == rebuildsBeforeCooldown);
    }
    world.updatePlayerPosition(0, SEA_LEVEL, 0);
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (world.getStreamingWorkStats().activeSetRebuilds > rebuildsBeforeCooldown)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const StreamingWorkStats rebuilt = world.getStreamingWorkStats();
    REQUIRE(rebuilt.activeSetRebuilds == rebuildsBeforeCooldown + 1);
    REQUIRE(rebuilt.activeSetRequests >= 2);
    REQUIRE(rebuilt.activeSetBuildMs > 0.0F);
}

TEST_CASE("Gameplay coalesces active-set movement away from the fixed tick",
          "[world][streaming][performance][concurrency]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    constexpr int REQUEST_COUNT = 12;
    constexpr int64_t REQUEST_STRIDE = CHUNK_EDGE * 3;
    for (int request = 0; request < REQUEST_COUNT; ++request) {
        world.updatePlayerPosition(request * REQUEST_STRIDE, SEA_LEVEL, 0);
    }

    const int64_t finalChunkX = (REQUEST_COUNT - 1) * 3;
    const ChunkPos finalCameraCube{finalChunkX, Chunk::worldToChunkY(SEA_LEVEL), 0};
    // Mesh-candidate publication precedes the final timing sample by a few
    // instructions. Wait for both observables so this concurrency test does
    // not race the diagnostics write on a fast worker.
    for (int attempt = 0; attempt < 2000; ++attempt) {
        const StreamingWorkStats current = world.getStreamingWorkStats();
        if (world.shouldMeshChunk(finalCameraCube) && current.activeSetBuildMs > 0.0F) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const StreamingWorkStats work = world.getStreamingWorkStats();
    REQUIRE(world.shouldMeshChunk(finalCameraCube));
    REQUIRE(work.activeSetRequests == REQUEST_COUNT);
    REQUIRE(work.activeSetRequestsCoalesced > 0);
    REQUIRE(work.activeSetRebuilds < work.activeSetRequests);
    REQUIRE(work.activeSetBuildMs > 0.0F);
}

TEST_CASE("Cubic streaming unload hysteresis does not add mesh candidates",
          "[world][streaming][hysteresis]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    constexpr int32_t INITIAL_BLOCK_Y = 400;
    constexpr int32_t initialY = INITIAL_BLOCK_Y / CHUNK_EDGE;
    constexpr int retainedHorizontalRadius =
        exactStreamingActiveSetRadiusChunks(MIN_RENDER_DISTANCE_CHUNKS) +
        EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
    constexpr ChunkPos horizontalEdge{-retainedHorizontalRadius, initialY, 0};
    constexpr ChunkPos verticalEdge{0, initialY + EXPLORATION_VERTICAL_RADIUS_CUBES + 1, 0};

    world.generateAroundPlayer(0, INITIAL_BLOCK_Y, 0);
    REQUIRE(world.getChunk(horizontalEdge));
    REQUIRE(world.getChunk(verticalEdge));

    world.generateAroundPlayer(CHUNK_EDGE * HORIZONTAL_UNLOAD_HYSTERESIS_CHUNKS,
                               INITIAL_BLOCK_Y - CHUNK_EDGE, 0);
    world.unloadDistantChunks();
    REQUIRE(world.isChunkLoaded(horizontalEdge));
    REQUIRE(world.isChunkLoaded(verticalEdge));
    REQUIRE_FALSE(world.shouldMeshChunk(horizontalEdge));
    REQUIRE_FALSE(world.shouldMeshChunk(verticalEdge));
    REQUIRE(world.getStreamingWorkStats().hysteresisRetainedCubes >= 2);

    world.generateAroundPlayer(CHUNK_EDGE * 4, INITIAL_BLOCK_Y - CHUNK_EDGE * 4, 0);
    world.unloadDistantChunks();
    REQUIRE_FALSE(world.isChunkLoaded(horizontalEdge));
    REQUIRE_FALSE(world.isChunkLoaded(verticalEdge));
}

TEST_CASE("The underground exploration band is a hard mesh and retention priority",
          "[world][streaming][priority]") {
    World world(42, EXPLORATION_RADIUS_CHUNKS, MAX_LOADED_CUBES,
                GenerationSettings{.structures = false});
    constexpr int32_t centerY = SEA_LEVEL / CHUNK_EDGE;
    constexpr ChunkPos retainedEdge{EXPLORATION_RADIUS_CHUNKS, centerY, 0};
    REQUIRE(world.getChunk(retainedEdge));

    world.generateAroundPlayer(0, SEA_LEVEL, 0);
    world.unloadDistantChunks();

    REQUIRE(world.isChunkLoaded(retainedEdge));
    for (int dz = -EXPLORATION_RADIUS_CHUNKS; dz <= EXPLORATION_RADIUS_CHUNKS; ++dz) {
        for (int dx = -EXPLORATION_RADIUS_CHUNKS; dx <= EXPLORATION_RADIUS_CHUNKS; ++dx) {
            if (dx * dx + dz * dz > EXPLORATION_RADIUS_CHUNKS * EXPLORATION_RADIUS_CHUNKS) {
                continue;
            }
            for (int dy = -EXPLORATION_VERTICAL_RADIUS_CUBES;
                 dy <= EXPLORATION_VERTICAL_RADIUS_CUBES; ++dy) {
                REQUIRE(world.shouldMeshChunk({dx, centerY + dy, dz}));
            }
        }
    }
}

TEST_CASE("Active streaming completes deep sky authority before the first underground mesh",
          "[world][streaming][light][skylight][publication][regression]") {
    // Clamp this scheduling fixture's visible horizon at the exploration
    // boundary. The production release request below is still the full exact
    // distance, while unrelated outer-disk surface work cannot dominate the
    // bounded regression.
    World world(42, EXPLORATION_RADIUS_CHUNKS);
    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    const auto centerPlan = world.generator().getColumnPlan({0, 0});
    REQUIRE(centerPlan);

    const int32_t surfaceSection = Chunk::worldToChunkY(centerPlan->minimumSurfaceY());
    const int32_t cameraSection = surfaceSection - EXPLORATION_VERTICAL_RADIUS_CUBES - 2;
    REQUIRE(cameraSection >= WORLD_MIN_CHUNK_Y);
    const int32_t cameraY = cameraSection * CHUNK_EDGE + CHUNK_EDGE / 2;
    constexpr int64_t cameraX = CHUNK_EDGE / 2;
    constexpr int64_t cameraZ = CHUNK_EDGE / 2;
    const ChunkPos target{0, cameraSection, 0};

    const auto streamingStart = std::chrono::steady_clock::now();
    world.updatePlayerPosition(cameraX, cameraY, cameraZ);
    MeshSnapshot firstVisible;
    bool ready = false;
    for (int attempt = 0; attempt < 20'000; ++attempt) {
        world.updatePlayerPosition(cameraX, cameraY, cameraZ);
        if (world.shouldMeshChunk(target) && world.snapshotForMeshing(target, firstVisible)) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    INFO("loaded=" << world.getLoadedChunkCount() << " pending=" << world.getPendingChunkCount()
                   << " publication="
                   << world.getStreamingWorkStats().publicationLightDeferredQueue);
    const double centerReadySeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - streamingStart).count();
    const StreamingWorkStats centerStats = world.getStreamingWorkStats();
    CAPTURE(centerReadySeconds, centerStats.loadedCubeHighWater,
            centerStats.publicationLightDeferredQueue, world.getLoadedChunkCount(),
            world.getPendingChunkCount());
    REQUIRE(ready);
    REQUIRE(firstVisible.derivedSkyLightValid);
    REQUIRE(world.getLoadedChunkCount() < 2'048);
    REQUIRE(world.getStreamingWorkStats().loadedCubeHighWater <= MAX_LOADED_CUBES);

    const auto firstVisibleLight = firstVisible.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(target, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);

    // The active-set radius and priority-lane assertions keep the x=7 meshing
    // halo ahead of broad surface work. Exercise publication there with
    // bounded direct loads instead of timing it behind every nearer
    // asynchronous exploration job.
    const ChunkPos explorationEdge{EXPLORATION_RADIUS_CHUNKS, cameraSection, 0};
    const ChunkPos explorationHalo{EXPLORATION_RADIUS_CHUNKS + 1, cameraSection, 0};
    STATIC_REQUIRE(exactStreamingActiveSetRadiusChunks(EXPLORATION_RADIUS_CHUNKS) ==
                   EXPLORATION_RADIUS_CHUNKS + 1);
    STATIC_REQUIRE(exactStreamingTaskPriority(0, 6, 49) > exactStreamingTaskPriority(0, 4, 0));
    REQUIRE(withinExactStreamingRadius(EXPLORATION_RADIUS_CHUNKS, 0, EXPLORATION_RADIUS_CHUNKS));
    const auto edgeStart = std::chrono::steady_clock::now();
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ColumnPos column{explorationEdge.x + offsetX, explorationEdge.z + offsetZ};
            const auto plan = world.generator().getColumnPlan(column);
            REQUIRE(plan);
            REQUIRE(world.getChunk({column.x, cameraSection, column.z}));
            for (const int32_t section : plan->exposedSections()) {
                if (section >= cameraSection) {
                    REQUIRE(world.getChunk({column.x, section, column.z}));
                }
            }
        }
    }

    MeshSnapshot edgeFirstVisible;
    REQUIRE(world.snapshotForMeshing(explorationEdge, edgeFirstVisible));
    const double targetedEdgeSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - edgeStart).count();
    const StreamingWorkStats edgeStats = world.getStreamingWorkStats();
    CAPTURE(centerReadySeconds, targetedEdgeSeconds, edgeStats.completedColumnPlans,
            edgeStats.activeSetRebuilds, edgeStats.activeSetRequests,
            edgeStats.activeSetRequestsCoalesced, edgeStats.activeSetBuildsCanceled,
            edgeStats.loadedCubeAdmissionsRejected, world.getLoadedChunkCount(),
            world.getPendingChunkCount());
    REQUIRE(world.getExactViewDistance() == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE_FALSE(world.generationFailure());
    REQUIRE(edgeFirstVisible.derivedSkyLightValid);
    REQUIRE(world.isChunkLoaded(explorationHalo));
    REQUIRE(edgeStats.loadedCubeAdmissionsRejected == 0);
    REQUIRE(edgeStats.loadedCubeHighWater <= MAX_LOADED_CUBES);

    const auto edgeFirstVisibleLight = edgeFirstVisible.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot edgeSettled;
    REQUIRE(world.snapshotForMeshing(explorationEdge, edgeSettled));
    REQUIRE(edgeSettled.packedLight == edgeFirstVisibleLight);
}

TEST_CASE("Visible distance does not expand exact cubic simulation beyond 32 chunks",
          "[world][streaming][lod]") {
    World world(42, MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getViewDistance() == 512);
    REQUIRE(world.getExactViewDistance() == 32);

    world.setViewDistance(12);
    REQUIRE(world.getViewDistance() == 12);
    REQUIRE(world.getExactViewDistance() == 12);

    world.setViewDistance(1000);
    REQUIRE(world.getViewDistance() == MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getExactViewDistance() == MAX_EXACT_CUBIC_DISTANCE_CHUNKS);

    world.setViewDistance(1);
    REQUIRE(world.getViewDistance() == MIN_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getExactViewDistance() == MIN_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Exact collision publication rejects stale coverage epochs",
          "[world][streaming][collision][exact-ownership][epoch][regression]") {
    World world(42, 4);
    constexpr int64_t WORLD_X = 8;
    constexpr int64_t WORLD_Z = 8;
    constexpr ColumnPos COLUMN{0, 0};
    const auto plan = world.generator().getColumnPlan(COLUMN);
    REQUIRE(plan);
    const int32_t surfaceY = plan->surfaceY(8, 8);
    const ChunkPos surfaceSection{COLUMN.x, Chunk::worldToChunkY(surfaceY), COLUMN.z};
    REQUIRE(world.getChunk(surfaceSection));
    world.setBlock(WORLD_X, surfaceY, WORLD_Z, BlockType::AIR);
    REQUIRE(world.findBlockIfLoaded(WORLD_X, surfaceY, WORLD_Z) == BlockType::AIR);

    const auto coverage = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(coverage);
    const uint64_t staleEpoch = coverage->epoch == std::numeric_limits<uint64_t>::max()
                                    ? coverage->epoch - 1
                                    : coverage->epoch + 1;
    const std::array sections{surfaceSection};
    REQUIRE_FALSE(world.publishExactCollisionOwnership(staleEpoch, sections));
    REQUIRE(world.getExactCollisionOwnershipSnapshot()->sections.empty());

    // Loaded exact air cannot open the canonical surface until the same
    // coverage epoch publishes the exact section.
    REQUIRE(world.getCollisionBlockIfLoaded(WORLD_X, surfaceY, WORLD_Z) == BlockType::STONE);
    REQUIRE(world.getCollisionBlockIfLoaded(WORLD_X, surfaceY + 1, WORLD_Z) == BlockType::AIR);
}

TEST_CASE("Published exact collision sections use loaded blocks and close missing cubes",
          "[world][streaming][collision][exact-ownership][publication][regression]") {
    World world(42, 4);
    constexpr int64_t WORLD_X = 8;
    constexpr int64_t WORLD_Z = 8;
    constexpr ColumnPos COLUMN{0, 0};
    const auto plan = world.generator().getColumnPlan(COLUMN);
    REQUIRE(plan);
    const int32_t surfaceY = plan->surfaceY(8, 8);
    const ChunkPos surfaceSection{COLUMN.x, Chunk::worldToChunkY(surfaceY), COLUMN.z};
    REQUIRE(world.getChunk(surfaceSection));
    world.setBlock(WORLD_X, surfaceY, WORLD_Z, BlockType::AIR);
    REQUIRE(world.getCollisionBlockIfLoaded(WORLD_X, surfaceY, WORLD_Z) == BlockType::STONE);

    const int32_t missingSectionY = surfaceSection.y + 2;
    REQUIRE(missingSectionY >= WORLD_MIN_CHUNK_Y);
    REQUIRE(missingSectionY <= WORLD_MAX_CHUNK_Y);
    const int32_t missingWorldY = missingSectionY * CHUNK_EDGE + CHUNK_EDGE / 2;
    REQUIRE(missingWorldY > surfaceY);
    REQUIRE_FALSE(world.isChunkLoaded({COLUMN.x, missingSectionY, COLUMN.z}));

    const auto coverage = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(coverage);
    const std::array sections{surfaceSection, ChunkPos{COLUMN.x, missingSectionY, COLUMN.z}};
    REQUIRE(world.publishExactCollisionOwnership(coverage->epoch, sections));
    const auto publication = world.getExactCollisionOwnershipSnapshot();
    REQUIRE(publication->coverageEpoch == coverage->epoch);
    REQUIRE(publication->owns(surfaceSection));
    REQUIRE(publication->owns({COLUMN.x, missingSectionY, COLUMN.z}));
    REQUIRE(world.getCollisionBlockIfLoaded(WORLD_X, surfaceY, WORLD_Z) == BlockType::AIR);
    REQUIRE(world.getCollisionBlockIfLoaded(WORLD_X, missingWorldY, WORLD_Z) == BlockType::BEDROCK);
}

TEST_CASE("Cold v4 entry bounds exact streaming without shrinking the far horizon",
          "[world][streaming][lod][generator-v4][startup]") {
    World world(42, MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getViewDistance() == MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getExactViewDistance() == MAX_EXACT_CUBIC_DISTANCE_CHUNKS);

    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(world.getViewDistance() == MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getExactViewDistance() == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);

    world.setExactStreamingDistance(0);
    REQUIRE(world.getExactViewDistance() == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);

    world.setExactStreamingDistance(MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(world.getViewDistance() == MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(world.getExactViewDistance() == MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
}

TEST_CASE("Cold spawn readiness matches the circular entry footprint",
          "[world][streaming][generator-v4][startup][spawn][regression]") {
    REQUIRE(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS == 0);
    REQUIRE(boundedColdStartExactRadiusChunks(-1) == 0);
    REQUIRE(boundedColdStartExactRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) ==
            COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(boundedColdStartExactRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS + 1) ==
            COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(withinExactStreamingRadius(0, 0, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS));
    REQUIRE_FALSE(withinExactStreamingRadius(1, 0, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS));
    REQUIRE(exactStreamingActiveSetRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) ==
            COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS + 1);
    REQUIRE(exactStreamingMeshRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) == 1);
    REQUIRE(withinExactStreamingRadius(
        1, 0, exactStreamingMeshRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS)));
    REQUIRE_FALSE(withinExactStreamingRadius(
        1, 1, exactStreamingMeshRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS)));
    REQUIRE(exactStreamingPlanCoverageRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) ==
            COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS + 1 +
                EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS +
                EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS);
    REQUIRE(exactStreamingActiveSetRadiusChunks(MAX_EXACT_CUBIC_DISTANCE_CHUNKS) ==
            MAX_EXACT_CUBIC_DISTANCE_CHUNKS + 1);
    REQUIRE(exactStreamingMeshRadiusChunks(MAX_EXACT_CUBIC_DISTANCE_CHUNKS) ==
            MAX_EXACT_CUBIC_DISTANCE_CHUNKS + 1);
    REQUIRE(exactStreamingActiveSetRadiusChunks(MAX_EXACT_CUBIC_DISTANCE_CHUNKS + 1) ==
            MAX_EXACT_CUBIC_DISTANCE_CHUNKS + 1);
    REQUIRE(exactStreamingActiveSetRadiusChunks(-1) == 1);
}

TEST_CASE("Cold v4 exact cap retains the mandatory one-chunk halo before entry",
          "[world][streaming][lod][generator-v4][startup][regression]") {
    // A tiny residency cap keeps this a scheduling test. The active-set
    // snapshot is published before any asynchronous column-plan work can
    // expand the mutable world, so it directly proves the v4 entry cap is
    // applied to a 512-chunk render horizon.
    World world(42, MAX_RENDER_DISTANCE_CHUNKS, 1);
    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    world.updatePlayerPosition(0, SEA_LEVEL, 0);

    std::shared_ptr<const ExactSurfaceCoverageSnapshot> coverage;
    for (int attempt = 0; attempt < 2'000; ++attempt) {
        coverage = world.getExactSurfaceCoverageSnapshot();
        if (coverage && coverage->nominalRadiusChunks == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(coverage);
    REQUIRE(coverage->nominalRadiusChunks == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(world.getViewDistance() == MAX_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Capped exact mesh selection keeps resident surfaces stable across small movement",
          "[world][streaming][mesh-cap][residency][regression]") {
    constexpr size_t EXTRA_REQUIREMENTS = 2'048;
    constexpr uint8_t SURFACE_PRIORITY = 3;
    constexpr uint8_t EDITED_PRIORITY = 5;
    std::unordered_map<ChunkPos, uint8_t> requirements;
    requirements.reserve(MAX_MESH_RESIDENT_CUBES + EXTRA_REQUIREMENTS);
    for (size_t index = 0; index < MAX_MESH_RESIDENT_CUBES + EXTRA_REQUIREMENTS; ++index) {
        const int64_t x = static_cast<int64_t>(index % 192) - 96;
        const int64_t z = static_cast<int64_t>(index / 192) - 48;
        requirements.emplace(ChunkPos{x, 4, z}, SURFACE_PRIORITY);
    }
    REQUIRE(requirements.size() > MAX_MESH_RESIDENT_CUBES);

    const std::unordered_set<ChunkPos> empty;
    const auto initial =
        selectStableMeshCandidates(requirements, empty, {0, 4, 0}, MAX_MESH_RESIDENT_CUBES);
    REQUIRE(initial.size() == MAX_MESH_RESIDENT_CUBES);

    // A rebuild within the same camera cube retains the complete prior set.
    // A one-cube diagonal movement may replace only the candidates whose
    // distance advantage exceeds the bounded residency credit.
    const auto sameCube =
        selectStableMeshCandidates(requirements, initial, {0, 4, 0}, MAX_MESH_RESIDENT_CUBES);
    const auto adjacentCube =
        selectStableMeshCandidates(requirements, sameCube, {1, 4, -1}, MAX_MESH_RESIDENT_CUBES);
    REQUIRE(sameCube == initial);
    REQUIRE(std::count_if(initial.begin(), initial.end(), [&](ChunkPos position) {
                return adjacentCube.contains(position);
            }) >= static_cast<std::ptrdiff_t>(MAX_MESH_RESIDENT_CUBES - 32));

    std::vector<ChunkPos> promoted;
    for (const auto& [position, priority] : requirements) {
        if (initial.contains(position))
            continue;
        requirements[position] = EDITED_PRIORITY;
        promoted.push_back(position);
        if (promoted.size() == 128)
            break;
    }
    REQUIRE(promoted.size() == 128);
    const auto withEdits =
        selectStableMeshCandidates(requirements, adjacentCube, {1, 4, -1}, MAX_MESH_RESIDENT_CUBES);
    for (ChunkPos position : promoted)
        REQUIRE(withEdits.contains(position));
    const auto newlyPromoted =
        std::count_if(promoted.begin(), promoted.end(),
                      [&](ChunkPos position) { return !adjacentCube.contains(position); });
    REQUIRE(std::count_if(adjacentCube.begin(), adjacentCube.end(), [&](ChunkPos position) {
                return withEdits.contains(position);
            }) == static_cast<std::ptrdiff_t>(MAX_MESH_RESIDENT_CUBES) - newlyPromoted);
}

TEST_CASE("Capped exact mesh selection replaces stale residency after a camera jump",
          "[world][streaming][mesh-cap][residency][regression]") {
    constexpr int64_t CLUSTER_EDGE = 128;
    constexpr int64_t NEW_CLUSTER_X = 4'096;
    constexpr uint8_t SURFACE_PRIORITY = 3;
    static_assert(CLUSTER_EDGE * CLUSTER_EDGE == MAX_MESH_RESIDENT_CUBES);

    std::unordered_map<ChunkPos, uint8_t> requirements;
    std::unordered_set<ChunkPos> oldCluster;
    std::unordered_set<ChunkPos> newCluster;
    requirements.reserve(MAX_MESH_RESIDENT_CUBES * 2);
    oldCluster.reserve(MAX_MESH_RESIDENT_CUBES);
    newCluster.reserve(MAX_MESH_RESIDENT_CUBES);
    for (int64_t z = 0; z < CLUSTER_EDGE; ++z) {
        for (int64_t x = 0; x < CLUSTER_EDGE; ++x) {
            const ChunkPos oldPosition{x, 4, z};
            const ChunkPos newPosition{NEW_CLUSTER_X + x, 4, z};
            requirements.emplace(oldPosition, SURFACE_PRIORITY);
            requirements.emplace(newPosition, SURFACE_PRIORITY);
            oldCluster.insert(oldPosition);
            newCluster.insert(newPosition);
        }
    }

    const std::unordered_set<ChunkPos> empty;
    const auto beforeJump = selectStableMeshCandidates(
        requirements, empty, {CLUSTER_EDGE / 2, 4, CLUSTER_EDGE / 2}, MAX_MESH_RESIDENT_CUBES);
    REQUIRE(beforeJump == oldCluster);

    const auto afterJump = selectStableMeshCandidates(
        requirements, beforeJump, {NEW_CLUSTER_X + CLUSTER_EDGE / 2, 4, CLUSTER_EDGE / 2},
        MAX_MESH_RESIDENT_CUBES);
    REQUIRE(afterJump == newCluster);
}

TEST_CASE("Exact mesh selection preserves hard gameplay priority at capacity",
          "[world][streaming][mesh-cap][priority]") {
    const std::unordered_map<ChunkPos, uint8_t> requirements{
        {{0, 4, 0}, 3}, // additional surface or cliff
        {{1, 4, 0}, 4}, // primary surface
        {{2, 4, 0}, 5}, // edited section
        {{3, 4, 0}, 6}, // exploration and collision
    };
    const std::unordered_set<ChunkPos> previous{{0, 4, 0}};
    const auto selected = selectStableMeshCandidates(requirements, previous, {0, 4, 0}, 3);
    REQUIRE_FALSE(selected.contains({0, 4, 0}));
    REQUIRE(selected.contains({1, 4, 0}));
    REQUIRE(selected.contains({2, 4, 0}));
    REQUIRE(selected.contains({3, 4, 0}));
}

TEST_CASE("Exact mesh capacity keeps complete nearer vertical surfaces before flatter distance",
          "[world][streaming][mesh-cap][priority][surface][flight][regression]") {
    constexpr uint8_t REQUIRED_SURFACE_PRIORITY = EXACT_STREAMING_EDITED_PRIORITY_LANE;
    constexpr ChunkPos CAMERA{0, 40, 0};
    constexpr ChunkPos NEAR_LOW{1, WORLD_MIN_CHUNK_Y, 0};
    constexpr ChunkPos NEAR_HIGH{1, WORLD_MAX_CHUNK_Y, 0};
    constexpr ChunkPos FAR_FLAT{2, CAMERA.y, 0};
    const std::unordered_map<ChunkPos, uint8_t> requirements{
        {NEAR_LOW, REQUIRED_SURFACE_PRIORITY},
        {NEAR_HIGH, REQUIRED_SURFACE_PRIORITY},
        {FAR_FLAT, REQUIRED_SURFACE_PRIORITY},
    };

    const auto selected = selectStableMeshCandidates(requirements, {}, CAMERA, /*capacity=*/2);

    REQUIRE(selected.size() == 2);
    REQUIRE(selected.contains(NEAR_LOW));
    REQUIRE(selected.contains(NEAR_HIGH));
    REQUIRE_FALSE(selected.contains(FAR_FLAT));
}

// ===========================================================================
// Java-style water rules and scheduler
// ===========================================================================

namespace {

FluidCell loadedCell(BlockType block = BlockType::AIR, FluidState state = FluidState::source()) {
    return {.loaded = true, .block = block, .state = state};
}

const FluidMutation* findMutation(const FluidRuleResult& result, FluidDirection direction,
                                  FluidMutationType type) {
    for (uint8_t index = 0; index < result.mutationCount; ++index) {
        const FluidMutation& mutation = result.mutations[index];
        if (mutation.direction == direction && mutation.type == type)
            return &mutation;
    }
    return nullptr;
}

class TestFluidWorld final : public FluidWorldAccess {
public:
    explicit TestFluidWorld(FluidBounds bounds) : bounds_(bounds) {}

    FluidCell readFluidCell(FluidPos position) const override {
        if (!bounds_.contains(position))
            return {};
        auto iterator = cells_.find(position);
        if (iterator != cells_.end())
            return iterator->second;
        return loadedCell();
    }

    void writeWater(FluidPos position, FluidState state) override {
        cells_[position] = loadedCell(BlockType::WATER, state);
        ++writes;
    }

    void removeWater(FluidPos position) override {
        cells_[position] = loadedCell();
        ++removals;
    }

    void setBlock(FluidPos position, BlockType block, FluidState state = FluidState::source()) {
        cells_[position] = loadedCell(block, state);
    }

    FluidBounds bounds_;
    std::unordered_map<FluidPos, FluidCell> cells_;
    size_t writes = 0;
    size_t removals = 0;
};

} // namespace

TEST_CASE("FluidState packs source levels and falling flow", "[fluid]") {
    STATIC_REQUIRE(sizeof(FluidState) == 1);
    REQUIRE(FluidState::source().packed() == 0);
    REQUIRE(FluidState::source().isSource());
    REQUIRE(FluidState::flowing(0).level() == 1);
    REQUIRE(FluidState::flowing(9).level() == 7);
    REQUIRE(FluidState::falling(4).level() == 4);
    REQUIRE(FluidState::falling(4).isFalling());
    STATIC_REQUIRE(fluidSurfaceHeight(FluidState::source()) == 1.0F);
    STATIC_REQUIRE(fluidSurfaceHeight(FluidState::flowing(7)) == 0.125F);
    STATIC_REQUIRE(fluidSurfaceHeight(FluidState::falling(4)) == 1.0F);
    REQUIRE(FluidState::isValidPacked(0x0F));
    REQUIRE_FALSE(FluidState::isValidPacked(0x10));
}

TEST_CASE("Water falls before spreading horizontally", "[fluid][rules]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER),
        .down = loadedCell(BlockType::AIR),
        .up = loadedCell(),
        .west = loadedCell(),
        .east = loadedCell(),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    const FluidMutation* downward =
        findMutation(result, FluidDirection::DOWN, FluidMutationType::SET_WATER);
    REQUIRE(downward != nullptr);
    REQUIRE(downward->state.isFalling());
    REQUIRE(findMutation(result, FluidDirection::WEST, FluidMutationType::SET_WATER) == nullptr);
}

TEST_CASE("Supported falling water does not spread sideways", "[fluid][rules][waterfall]") {
    for (const FluidState support : {FluidState::falling(7), FluidState::source()}) {
        CAPTURE(support.packed());
        const FluidNeighborhood cells{
            .center = loadedCell(BlockType::WATER, FluidState::falling(7)),
            .down = loadedCell(BlockType::WATER, support),
            .up = loadedCell(BlockType::WATER, FluidState::falling(7)),
            .west = loadedCell(),
            .east = loadedCell(),
            .north = loadedCell(),
            .south = loadedCell(),
        };
        const FluidRuleResult result = evaluateWaterRules(cells);
        REQUIRE(result.deferredCount == 0);
        REQUIRE(result.mutationCount == 0);
    }
}

TEST_CASE("Supported source water spreads at level one", "[fluid][rules]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER),
        .down = loadedCell(BlockType::STONE),
        .up = loadedCell(),
        .west = loadedCell(),
        .east = loadedCell(),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    for (FluidDirection direction : {FluidDirection::WEST, FluidDirection::EAST,
                                     FluidDirection::NORTH, FluidDirection::SOUTH}) {
        const FluidMutation* spread = findMutation(result, direction, FluidMutationType::SET_WATER);
        REQUIRE(spread != nullptr);
        REQUIRE(spread->state == FluidState::flowing(1));
    }
}

TEST_CASE("Source water cannot replace a floor torch",
          "[fluid][rules][torch][replacement][regression]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER),
        .down = loadedCell(BlockType::STONE),
        .up = loadedCell(),
        .west = loadedCell(),
        .east = loadedCell(BlockType::TORCH),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    REQUIRE(findMutation(result, FluidDirection::EAST, FluidMutationType::SET_WATER) == nullptr);
    for (FluidDirection direction :
         {FluidDirection::WEST, FluidDirection::NORTH, FluidDirection::SOUTH}) {
        REQUIRE(findMutation(result, direction, FluidMutationType::SET_WATER) != nullptr);
    }
}

TEST_CASE("Two adjacent sources form a source over support", "[fluid][rules]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER, FluidState::flowing(4)),
        .down = loadedCell(BlockType::STONE),
        .up = loadedCell(),
        .west = loadedCell(BlockType::WATER),
        .east = loadedCell(BlockType::WATER),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    const FluidMutation* center =
        findMutation(result, FluidDirection::CENTER, FluidMutationType::SET_WATER);
    REQUIRE(center != nullptr);
    REQUIRE(center->state.isSource());
}

TEST_CASE("Unsupported flowing water is removed", "[fluid][rules]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER, FluidState::flowing(4)),
        .down = loadedCell(BlockType::STONE),
        .up = loadedCell(),
        .west = loadedCell(),
        .east = loadedCell(),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    REQUIRE(findMutation(result, FluidDirection::CENTER, FluidMutationType::REMOVE_WATER) !=
            nullptr);
}

TEST_CASE("Water rules defer unavailable face neighbors", "[fluid][rules]") {
    FluidNeighborhood cells{
        .center = loadedCell(BlockType::WATER),
        .down = {},
        .up = {},
        .west = {},
        .east = loadedCell(),
        .north = loadedCell(),
        .south = loadedCell(),
    };
    const FluidRuleResult result = evaluateWaterRules(cells);
    REQUIRE(result.deferredCount == 1);
    REQUIRE(std::find(result.deferred.begin(), result.deferred.begin() + result.deferredCount,
                      FluidDirection::DOWN) != result.deferred.begin() + result.deferredCount);
    REQUIRE(result.mutationCount == 0);
}

TEST_CASE("Fluid scheduler stays idle until a gameplay edit activates water",
          "[fluid][scheduler]") {
    TestFluidWorld world({-4, -2, -4, 4, 2, 4});
    for (int64_t z = -4; z <= 4; ++z) {
        for (int64_t x = -4; x <= 4; ++x) {
            world.setBlock({x, -1, z}, BlockType::STONE);
        }
    }
    world.setBlock({0, 0, 0}, BlockType::WATER);
    FluidScheduler scheduler;

    REQUIRE(scheduler.pendingCount() == 0);
    REQUIRE(scheduler.tick(world) == 0);
    REQUIRE(world.writes == 0);

    REQUIRE(scheduler.activateBlockChange({0, 0, 0}) == 7);
    for (uint32_t tick = 1; tick < WATER_UPDATE_DELAY_TICKS; ++tick) {
        REQUIRE(scheduler.tick(world) == 0);
    }
    REQUIRE(scheduler.tick(world) > 0);
    REQUIRE(world.writes > 0);
}

TEST_CASE("World generation and loading do not enqueue fluid ticks", "[fluid][worldgen]") {
    World world(42);
    REQUIRE(world.getPendingFluidCount() == 0);
    world.getChunk({0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(world.getPendingFluidCount() == 0);

    world.setBlock(8, WORLD_MAX_Y - 1, 8, BlockType::AIR);
    REQUIRE(world.getPendingFluidCount() == 0);

    world.setBlock(8, WORLD_MAX_Y - 1, 8, BlockType::WATER);
    REQUIRE(world.getPendingFluidCount() > 0);
}

TEST_CASE("Fluid scheduler enforces its per-tick work budget", "[fluid][scheduler]") {
    TestFluidWorld world({-4, -2, -4, 4, 2, 4});
    world.setBlock({0, 0, 0}, BlockType::WATER);
    FluidScheduler scheduler({.updatesPerTick = 2});
    scheduler.activateBlockChange({0, 0, 0});
    for (uint32_t tick = 1; tick < WATER_UPDATE_DELAY_TICKS; ++tick) {
        scheduler.tick(world);
    }
    REQUIRE(scheduler.tick(world) == 2);
    REQUIRE(scheduler.pendingCount() > 0);
}

TEST_CASE("Fluid scheduler persists and resumes only activated boundaries", "[fluid][scheduler]") {
    TestFluidWorld world({0, 0, 0, 0, 0, 0});
    world.setBlock({0, 0, 0}, BlockType::WATER);
    FluidScheduler scheduler;
    REQUIRE(scheduler.resumeDeferredIn({-1, -1, -1, 1, 1, 1}) == 0);
    scheduler.activateBlockChange({0, 0, 0});
    for (uint32_t tick = 0; tick < WATER_UPDATE_DELAY_TICKS; ++tick) {
        scheduler.tick(world);
    }
    const std::vector<FluidBoundaryFrontier> persisted = scheduler.deferredFrontiers();
    REQUIRE_FALSE(persisted.empty());

    FluidScheduler restored;
    for (const FluidBoundaryFrontier& frontier : persisted) {
        REQUIRE(restored.restoreDeferredFrontier(frontier));
    }
    REQUIRE(restored.deferredCount() == persisted.size());
    REQUIRE(restored.resumeDeferredIn({-1, -1, -1, 1, 1, 1}) > 0);
    REQUIRE(restored.pendingCount() > 0);
}

TEST_CASE("Fluid frontier resume is deterministic and bounded by unavailable cube",
          "[fluid][scheduler][frontier]") {
    const FluidBoundaryFrontier first{{-1, 1, 2}, {0, 1, 2}};
    const FluidBoundaryFrontier second{{-1, 4, 2}, {0, 4, 2}};
    const FluidBoundaryFrontier third{{-1, 7, 2}, {0, 7, 2}};
    const FluidBoundaryFrontier unrelated{{15, 1, 2}, {16, 1, 2}};

    FluidScheduler scheduler;
    REQUIRE(scheduler.restoreDeferredFrontier(third));
    REQUIRE(scheduler.restoreDeferredFrontier(unrelated));
    REQUIRE(scheduler.restoreDeferredFrontier(first));
    REQUIRE(scheduler.restoreDeferredFrontier(second));
    REQUIRE(scheduler.restoreDeferredFrontier(second));
    REQUIRE(scheduler.deferredCount() == 4);

    const FluidBounds loadedCube{0, 0, 0, CHUNK_EDGE - 1, CHUNK_EDGE - 1, CHUNK_EDGE - 1};
    const FluidBounds unrelatedCube{CHUNK_EDGE,    0, 0, CHUNK_EDGE * 2 - 1, CHUNK_EDGE - 1,
                                    CHUNK_EDGE - 1};
    REQUIRE(scheduler.deferredCountIn(loadedCube) == 3);
    REQUIRE(scheduler.deferredCountIn(unrelatedCube) == 1);
    REQUIRE(scheduler.resumeDeferredIn(loadedCube, 0) == 0);
    REQUIRE(scheduler.resumeDeferredIn(loadedCube, 2) == 2);
    REQUIRE(scheduler.deferredCountIn(loadedCube) == 1);
    REQUIRE(scheduler.pendingCount() == 4);
    REQUIRE(scheduler.deferredFrontiers() == std::vector<FluidBoundaryFrontier>{third, unrelated});

    REQUIRE(scheduler.resumeDeferredIn(loadedCube, 1) == 1);
    REQUIRE(scheduler.deferredCountIn(loadedCube) == 0);
    REQUIRE(scheduler.deferredFrontiers() == std::vector<FluidBoundaryFrontier>{unrelated});
}

TEST_CASE("Fluid frontier resume budget bounds failed scheduling attempts",
          "[fluid][scheduler][frontier]") {
    FluidScheduler scheduler({.pendingUpdates = 7});
    REQUIRE(scheduler.activateBlockChange({100, 100, 100}) == 7);
    for (int32_t y = 1; y <= 3; ++y) {
        REQUIRE(scheduler.restoreDeferredFrontier({{-1, y, 2}, {0, y, 2}}));
    }

    REQUIRE(scheduler.resumeDeferredIn({0, 0, 0, 15, 15, 15}, 2) == 0);
    REQUIRE(scheduler.droppedUpdateCount() == 4);
    REQUIRE(scheduler.deferredCount() == 3);
}

TEST_CASE("World resumes one loaded cube through bounded fluid batches",
          "[fluid][world][performance][frontier]") {
    TempDir directory("world_fluid_resume_budget");
    SaveManager saves(directory.path());
    std::vector<FluidBoundaryFrontier> frontiers;
    for (int32_t y = 64; y <= 65; ++y) {
        for (int64_t z = 0; z < 10; ++z) {
            frontiers.push_back({{-1, y, z}, {0, y, z}});
        }
    }
    REQUIRE(frontiers.size() > MAX_FLUID_FRONTIER_RESUMES_PER_CUBE);
    REQUIRE(saves.saveDeferredFluidFrontiers(frontiers));

    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    world.setSaveManager(&saves);
    REQUIRE(world.getChunk({0, 4, 0}));

    world.tickFluids(0.0);
    REQUIRE(world.saveModifiedChunks());
    REQUIRE(saves.loadDeferredFluidFrontiers().size() ==
            frontiers.size() - MAX_FLUID_FRONTIER_RESUMES_PER_CUBE);

    world.tickFluids(0.0);
    REQUIRE(world.saveModifiedChunks());
    REQUIRE(saves.loadDeferredFluidFrontiers().empty());
    REQUIRE(world.getDroppedFluidUpdateCount() == 0);
    REQUIRE(world.getDroppedFluidFrontierCount() == 0);
}

TEST_CASE("Fluid frontier clear and restore preserve ordering caps and index state",
          "[fluid][scheduler][frontier]") {
    const FluidBoundaryFrontier earlier{{-17, -1, 0}, {-16, -1, 0}};
    const FluidBoundaryFrontier later{{15, 1, 0}, {16, 1, 0}};
    const FluidBoundaryFrontier excess{{31, 1, 0}, {32, 1, 0}};
    FluidScheduler scheduler({.deferredFrontiers = 2});

    REQUIRE(scheduler.restoreDeferredFrontier(later));
    REQUIRE(scheduler.restoreDeferredFrontier(earlier));
    REQUIRE(scheduler.restoreDeferredFrontier(earlier));
    REQUIRE_FALSE(scheduler.restoreDeferredFrontier(excess));
    REQUIRE(scheduler.droppedFrontierCount() == 1);
    REQUIRE(scheduler.deferredFrontiers() == std::vector<FluidBoundaryFrontier>{earlier, later});

    scheduler.clear();
    REQUIRE(scheduler.deferredCount() == 0);
    REQUIRE(scheduler.pendingCount() == 0);
    REQUIRE(scheduler.droppedFrontierCount() == 0);
    REQUIRE(scheduler.deferredFrontiers().empty());

    REQUIRE(scheduler.restoreDeferredFrontier(later));
    REQUIRE(scheduler.restoreDeferredFrontier(earlier));
    REQUIRE(scheduler.deferredFrontiers() == std::vector<FluidBoundaryFrontier>{earlier, later});
    REQUIRE(scheduler.resumeDeferredIn({-16, -16, -16, -1, 15, 15}, 1) == 1);
    REQUIRE(scheduler.deferredFrontiers() == std::vector<FluidBoundaryFrontier>{later});
}

// ===========================================================================
// LightEngine: block-light propagation from lava
// ===========================================================================

TEST_CASE("Block properties expose gameplay light emitters", "[world][light]") {
    REQUIRE(blockLightEmission(BlockType::LAVA) == 15);
    REQUIRE(blockLightEmission(BlockType::TORCH) == 14);
    REQUIRE(blockLightEmission(BlockType::FURNACE_LIT) == 13);
    REQUIRE(blockLightEmission(BlockType::FURNACE) == 0);
    REQUIRE(blockLightEmission(BlockType::STONE) == 0);
    REQUIRE(blockLightEmission(BlockType::AIR) == 0);
    REQUIRE(isEmissive(BlockType::LAVA));
    REQUIRE_FALSE(isEmissive(BlockType::STONE));
    REQUIRE_FALSE(isEmissive(BlockType::GLASS));
}

TEST_CASE("Chunk packs skylight and block light into independent nibbles", "[world][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setSkyLight(3, 5, 7, 12);
    chunk.setBlockLight(3, 5, 7, 6);

    REQUIRE(chunk.getPackedLight(3, 5, 7) == 0xC6);
    REQUIRE(chunk.getSkyLight(3, 5, 7) == 12);
    REQUIRE(chunk.getBlockLight(3, 5, 7) == 6);

    chunk.setSkyLight(3, 5, 7, 4);
    REQUIRE(chunk.getPackedLight(3, 5, 7) == 0x46);
    chunk.setBlockLight(3, 5, 7, 0);
    REQUIRE(chunk.getPackedLight(3, 5, 7) == 0x40);
    REQUIRE(chunk.getSkyLight(3, 5, 7) == 4);
    REQUIRE_FALSE(chunk.hasBlockLight());
    REQUIRE(chunk.hasDerivedLight());
}

TEST_CASE("LightEngine: isolated opaque cover admits lateral skylight", "[world][light][sky]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 12, 8, BlockType::LOG);
    LightEngine::SkyLightSeedColumns seeds;
    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            seeds.set(x, z, chunk.chunkY * CHUNK_EDGE);
    seeds.set(8, 8, chunk.chunkY * CHUNK_EDGE + 13);

    REQUIRE(LightEngine::floodChunk(chunk, {}, seeds));
    REQUIRE(chunk.getSkyLight(8, 13, 8) == 15);
    REQUIRE(chunk.getSkyLight(8, 12, 8) == 0);
    REQUIRE(chunk.getSkyLight(8, 11, 8) == 14);
}

TEST_CASE("LightEngine: broad roofs and sealed caves remain dark", "[world][light][sky]") {
    Chunk roof(ChunkPos{0, 4, 0});
    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            roof.setBlock(x, 8, z, BlockType::STONE);
    LightEngine::SkyLightSeedColumns roofSeeds;
    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            roofSeeds.set(x, z, roof.chunkY * CHUNK_EDGE + 9);
    REQUIRE(LightEngine::floodChunk(roof, {}, roofSeeds));
    REQUIRE(roof.getSkyLight(8, 9, 8) == 15);
    REQUIRE(roof.getSkyLight(8, 7, 8) == 0);

    Chunk cave(ChunkPos{0, 4, 0});
    cave.fill(BlockType::STONE);
    cave.setBlock(8, 8, 8, BlockType::AIR);
    REQUIRE_FALSE(LightEngine::floodChunk(cave, {}));
    REQUIRE(cave.getSkyLight(8, 8, 8) == 0);
}

TEST_CASE("LightEngine: cave mouths receive attenuated skylight", "[world][light][sky]") {
    Chunk cave(ChunkPos{0, 4, 0});
    cave.fill(BlockType::STONE);
    for (int x = 8; x < CHUNK_EDGE; ++x)
        cave.setBlock(x, 8, 8, BlockType::AIR);

    Chunk exterior(ChunkPos{1, 4, 0});
    exterior.setSkyLight(0, 8, 8, 15);
    LightEngine::FaceNeighbors neighbors{};
    neighbors[1] = &exterior;
    REQUIRE(LightEngine::floodChunk(cave, neighbors));
    REQUIRE(cave.getSkyLight(15, 8, 8) == 14);
    REQUIRE(cave.getSkyLight(12, 8, 8) == 11);
    REQUIRE(cave.getSkyLight(8, 8, 8) == 7);
}

TEST_CASE("LightEngine: skylight spills through all six cubic faces",
          "[world][light][sky][cubic]") {
    struct FaceCase {
        size_t neighborIndex;
        std::array<int, 3> neighborCell;
        std::array<int, 3> borderCell;
        std::array<int, 3> inwardCell;
    };
    constexpr std::array<FaceCase, 6> cases{{
        {0, {15, 8, 8}, {0, 8, 8}, {1, 8, 8}},
        {1, {0, 8, 8}, {15, 8, 8}, {14, 8, 8}},
        {2, {8, 8, 15}, {8, 8, 0}, {8, 8, 1}},
        {3, {8, 8, 0}, {8, 8, 15}, {8, 8, 14}},
        {4, {8, 15, 8}, {8, 0, 8}, {8, 1, 8}},
        {5, {8, 0, 8}, {8, 15, 8}, {8, 14, 8}},
    }};

    for (const FaceCase& test : cases) {
        Chunk self(ChunkPos{0, 0, 0});
        Chunk neighbor(ChunkPos{0, 0, 0});
        neighbor.setSkyLight(test.neighborCell[0], test.neighborCell[1], test.neighborCell[2], 10);
        LightEngine::FaceNeighbors neighbors{};
        neighbors[test.neighborIndex] = &neighbor;

        const LightEngine::FloodResult result = LightEngine::floodChunk(self, neighbors);
        REQUIRE(result.changedState);
        REQUIRE((result.changedFaceMask & (1U << test.neighborIndex)) != 0);
        REQUIRE(self.getSkyLight(test.borderCell[0], test.borderCell[1], test.borderCell[2]) == 9);
        REQUIRE(self.getSkyLight(test.inwardCell[0], test.inwardCell[1], test.inwardCell[2]) == 8);
    }
}

TEST_CASE("LightEngine: skylight converges independently of load order", "[world][light][sky]") {
    const auto settle = [](bool sourceFirst) {
        Chunk source(ChunkPos{0, 4, 0});
        Chunk target(ChunkPos{1, 4, 0});
        LightEngine::SkyLightSeedColumns sourceSeeds;
        for (int z = 0; z < CHUNK_EDGE; ++z)
            for (int x = 0; x < CHUNK_EDGE; ++x)
                sourceSeeds.set(x, z, source.chunkY * CHUNK_EDGE);

        LightEngine::FaceNeighbors targetNeighbors{};
        targetNeighbors[0] = &source;
        if (sourceFirst) {
            LightEngine::floodChunk(source, {}, sourceSeeds);
            LightEngine::floodChunk(target, targetNeighbors);
        } else {
            LightEngine::floodChunk(target, targetNeighbors);
            LightEngine::floodChunk(source, {}, sourceSeeds);
            LightEngine::floodChunk(target, targetNeighbors);
        }
        return target.packedLightData();
    };

    REQUIRE(settle(true) == settle(false));
}

TEST_CASE("LightEngine: sky authority updates remove and restore direct light",
          "[world][light][sky][edit]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    LightEngine::SkyLightSeedColumns openSeeds;
    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            openSeeds.set(x, z, chunk.chunkY * CHUNK_EDGE);
    REQUIRE(LightEngine::floodChunk(chunk, {}, openSeeds));
    REQUIRE(chunk.getSkyLight(8, 7, 8) == 15);

    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            chunk.setBlock(x, 8, z, BlockType::STONE);
    LightEngine::SkyLightSeedColumns roofSeeds;
    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            roofSeeds.set(x, z, chunk.chunkY * CHUNK_EDGE + 9);
    REQUIRE(LightEngine::floodChunk(chunk, {}, roofSeeds));
    REQUIRE(chunk.getSkyLight(8, 7, 8) == 0);

    for (int z = 0; z < CHUNK_EDGE; ++z)
        for (int x = 0; x < CHUNK_EDGE; ++x)
            chunk.setBlock(x, 8, z, BlockType::AIR);
    REQUIRE(LightEngine::floodChunk(chunk, {}, openSeeds));
    REQUIRE(chunk.getSkyLight(8, 7, 8) == 15);
}

TEST_CASE("LightEngine: incomplete sky authority stays conservatively dark",
          "[world][light][sky][streaming]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    REQUIRE_FALSE(LightEngine::floodChunk(chunk, {}));
    REQUIRE_FALSE(chunk.hasDerivedLight());

    LightEngine::SkyLightSeedColumns complete;
    complete.set(8, 8, chunk.chunkY * CHUNK_EDGE);
    REQUIRE(LightEngine::floodChunk(chunk, {}, complete));
    REQUIRE(chunk.getSkyLight(8, 8, 8) == 15);
}

TEST_CASE("Skylight authority requires sparse generated and saved occupancy sections",
          "[world][light][sky][streaming][save]") {
    VerticalSectionMask generated;
    generated.set(4);
    generated.set(7);
    VerticalSectionMask saved;
    saved.set(12);
    generated.merge(saved);

    VerticalSectionMask loaded;
    loaded.set(4);
    loaded.set(12);
    REQUIRE_FALSE(loaded.containsAllSetSections(generated, 4));
    loaded.set(7);
    REQUIRE(loaded.containsAllSetSections(generated, 4));
    REQUIRE(loaded.containsAllSetSections(generated, 8));
    REQUIRE_FALSE(loaded.contains(6));
}

TEST_CASE("World publishes generated cubes with initial derived skylight", "[world][light][sky]") {
    World world(42, 4);
    auto chunk = world.getChunk({0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(chunk);
    REQUIRE(chunk->getBlock(8, 8, 8) == BlockType::AIR);
    REQUIRE(chunk->getSkyLight(8, 8, 8) == 15);
}

TEST_CASE("Generated tree cutoffs settle before their first mesh",
          "[world][light][sky][streaming][publication][flora][regression]") {
    // This seed has a birch rooted at (-27286, 75, -17100). The tree raises
    // the real opaque cutoff above ColumnPlan's density surface, which used to
    // become visible once with stale skylight and relight on a later tick.
    constexpr BlockPos ROOT{-27'286, 75, -17'100};
    World world(42, 4, 512);
    const ChunkPos target{Chunk::worldToChunk(ROOT.x), Chunk::worldToChunkY(ROOT.y),
                          Chunk::worldToChunk(ROOT.z)};

    std::array<std::shared_ptr<const ColumnPlan>, 9> plans{};
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const size_t index = static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1);
            plans[index] =
                world.generator().getColumnPlan({target.x + offsetX, target.z + offsetZ});
            REQUIRE(plans[index]);
        }
    }
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const size_t index = static_cast<size_t>((offsetZ + 1) * 3 + offsetX + 1);
            for (const int32_t section : plans[index]->exposedSections()) {
                if (section < target.y)
                    continue;
                REQUIRE(world.getChunk({target.x + offsetX, section, target.z + offsetZ}));
            }
        }
    }

    const std::shared_ptr<Chunk> rootCube = world.getChunk(target);
    REQUIRE(rootCube);
    REQUIRE(rootCube->getBlockWorld(ROOT.x, ROOT.y, ROOT.z) == BlockType::BIRCH_LOG);

    MeshSnapshot firstVisible;
    REQUIRE(world.snapshotForMeshing(target, firstVisible));
    const int localX = Chunk::worldToLocal(ROOT.x);
    const int localZ = Chunk::worldToLocal(ROOT.z);
    REQUIRE(firstVisible.skyCutoffY[MeshSnapshot::skyIndex(localX, localZ)] >
            firstVisible.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(localX, localZ)]);

    const std::vector<uint8_t> firstLight = rootCube->packedLightData();
    world.markChunkMeshed(target);
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    REQUIRE(rootCube->packedLightData() == firstLight);
    REQUIRE_FALSE(rootCube->needsMeshUpdate);
}

TEST_CASE("A newly published emitter settles resident boundary light before the first mesh",
          "[world][light][streaming][publication][torch][regression]") {
    TempDir directory("publication_boundary_light");
    SaveManager saves(directory.path());
    constexpr ChunkPos WEST{0, WORLD_MAX_CHUNK_Y, 0};
    constexpr ChunkPos EAST{1, WORLD_MAX_CHUNK_Y, 0};

    Chunk west(WEST);
    west.fill(BlockType::AIR);
    west.generated = true;
    Chunk east(EAST);
    east.fill(BlockType::AIR);
    east.setBlock(0, 8, 8, BlockType::TORCH);
    east.generated = true;
    saves.saveChunk(west);
    saves.saveChunk(east);
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setSaveManager(&saves);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({WEST.x + offsetX, WEST.z + offsetZ}));
        }
    }

    const auto resident = world.getChunk(WEST);
    REQUIRE(resident);
    REQUIRE(resident->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 0);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ColumnPos column{WEST.x + offsetX, WEST.z + offsetZ};
            const auto plan = world.generator().getColumnPlan(column);
            REQUIRE(plan);
            int32_t topSection =
                std::max(WEST.y, Chunk::worldToChunkY(std::clamp(plan->maximumSurfaceY(),
                                                                 WORLD_MIN_Y, WORLD_MAX_Y)));
            for (const int32_t exposed : plan->exposedSections()) {
                topSection = std::max(topSection, exposed);
            }
            for (int32_t section = WEST.y; section <= topSection; ++section) {
                const ChunkPos authority{column.x, section, column.z};
                if (authority != EAST)
                    REQUIRE(world.getChunk(authority));
            }
        }
    }
    world.markChunkMeshed(WEST);
    const uint32_t versionBeforeArrival = resident->version.load(std::memory_order_relaxed);

    const auto arriving = world.getChunk(EAST);
    REQUIRE(arriving);
    REQUIRE(arriving->getBlockLight(0, 8, 8) == 14);
    REQUIRE(resident->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 13);
    REQUIRE(resident->needsMeshUpdate);
    REQUIRE(resident->version.load(std::memory_order_relaxed) > versionBeforeArrival);

    MeshSnapshot firstVisible;
    REQUIRE(world.snapshotForMeshing(WEST, firstVisible));
    REQUIRE(firstVisible.blockLightAt(CHUNK_EDGE - 1, 8, 8) == 13);
    REQUIRE(firstVisible.blockLightAt(CHUNK_EDGE, 8, 8) == 14);

    const std::vector<uint8_t> settled = resident->packedLightData();
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    REQUIRE(resident->packedLightData() == settled);

    World reverse(42, 4);
    reverse.setSaveManager(&saves);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 2; ++offsetX) {
            REQUIRE(reverse.generator().getColumnPlan({offsetX, offsetZ}));
        }
    }
    const auto reverseEast = reverse.getChunk(EAST);
    const auto reverseWest = reverse.getChunk(WEST);
    REQUIRE(reverseEast);
    REQUIRE(reverseWest);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ColumnPos column{WEST.x + offsetX, WEST.z + offsetZ};
            const auto plan = reverse.generator().getColumnPlan(column);
            REQUIRE(plan);
            int32_t topSection =
                std::max(WEST.y, Chunk::worldToChunkY(std::clamp(plan->maximumSurfaceY(),
                                                                 WORLD_MIN_Y, WORLD_MAX_Y)));
            for (const int32_t exposed : plan->exposedSections()) {
                topSection = std::max(topSection, exposed);
            }
            for (int32_t section = WEST.y; section <= topSection; ++section) {
                REQUIRE(reverse.getChunk({column.x, section, column.z}));
            }
        }
    }
    REQUIRE(reverseEast->getBlockLight(0, 8, 8) == 14);
    REQUIRE(reverseWest->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 13);
    REQUIRE(reverseEast->packedLightData() == arriving->packedLightData());
    REQUIRE(reverseWest->packedLightData() == resident->packedLightData());
}

TEST_CASE("A section crossing the vertical mask words publishes completed skylight synchronously",
          "[world][light][sky][streaming][publication][vertical][v4][regression]") {
    TempDir directory("publication_vertical_skylight");
    SaveManager saves(directory.path());
    constexpr int32_t LOWER_SECTION = WORLD_MIN_CHUNK_Y + 63;
    constexpr int32_t BRIDGE_SECTION = LOWER_SECTION + 1;
    constexpr int32_t UPPER_SECTION = BRIDGE_SECTION + 1;
    STATIC_REQUIRE(LOWER_SECTION == 55);
    STATIC_REQUIRE(BRIDGE_SECTION == 56);

    for (int32_t section : {LOWER_SECTION, BRIDGE_SECTION, UPPER_SECTION}) {
        Chunk saved(ChunkPos{0, section, 0});
        saved.fill(BlockType::AIR);
        saved.generated = true;
        saves.saveChunk(saved);
    }
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setSaveManager(&saves);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({offsetX, offsetZ}));
            if (offsetX != 0 || offsetZ != 0) {
                REQUIRE(world.getChunk({offsetX, LOWER_SECTION, offsetZ}));
            }
        }
    }

    const auto lower = world.getChunk({0, LOWER_SECTION, 0});
    REQUIRE(lower);
    REQUIRE(world.getChunk({0, UPPER_SECTION, 0}));
    const uint8_t lightBeforeBridge = lower->getSkyLight(8, 8, 8);
    REQUIRE(lightBeforeBridge < MAX_DERIVED_LIGHT_LEVEL);
    world.markChunkMeshed({0, LOWER_SECTION, 0});
    const uint32_t versionBeforeBridge = lower->version.load(std::memory_order_relaxed);

    REQUIRE(world.getChunk({0, BRIDGE_SECTION, 0}));
    REQUIRE(lower->getSkyLight(8, 8, 8) == 15);
    REQUIRE(lower->needsMeshUpdate);
    REQUIRE(lower->version.load(std::memory_order_relaxed) > versionBeforeBridge);

    MeshSnapshot firstVisible;
    REQUIRE(world.snapshotForMeshing({0, LOWER_SECTION, 0}, firstVisible));
    REQUIRE(firstVisible.derivedSkyLightValid);
    REQUIRE(firstVisible.skyLightAt(8, 8, 8) == 15);
}

TEST_CASE("Publication light reaches an available diagonal before its first mesh",
          "[world][light][streaming][publication][torch][corner][regression]") {
    TempDir directory("publication_corner_light");
    SaveManager saves(directory.path());
    constexpr ChunkPos SOURCE{1, WORLD_MAX_CHUNK_Y, 0};
    constexpr ChunkPos FACE_X{0, WORLD_MAX_CHUNK_Y, 0};
    constexpr ChunkPos FACE_Z{1, WORLD_MAX_CHUNK_Y, -1};
    constexpr ChunkPos DIAGONAL{0, WORLD_MAX_CHUNK_Y, -1};

    for (ChunkPos position : {SOURCE, FACE_X, FACE_Z, DIAGONAL}) {
        Chunk saved(position);
        saved.fill(BlockType::AIR);
        if (position == SOURCE)
            saved.setBlock(0, 8, 0, BlockType::TORCH);
        saved.generated = true;
        saves.saveChunk(saved);
    }
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setSaveManager(&saves);
    for (int64_t chunkZ = -2; chunkZ <= 1; ++chunkZ) {
        for (int64_t chunkX = -1; chunkX <= 2; ++chunkX) {
            REQUIRE(world.generator().getColumnPlan({chunkX, chunkZ}));
        }
    }
    const auto faceX = world.getChunk(FACE_X);
    const auto faceZ = world.getChunk(FACE_Z);
    const auto diagonal = world.getChunk(DIAGONAL);
    REQUIRE(faceX);
    REQUIRE(faceZ);
    REQUIRE(diagonal);
    REQUIRE(diagonal->getBlockLight(CHUNK_EDGE - 1, 8, CHUNK_EDGE - 1) == 0);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const ChunkPos halo{DIAGONAL.x + offsetX, DIAGONAL.y, DIAGONAL.z + offsetZ};
            if (halo != SOURCE)
                REQUIRE(world.getChunk(halo));
        }
    }

    const auto source = world.getChunk(SOURCE);
    REQUIRE(source);
    REQUIRE(source->getBlockLight(0, 8, 0) == 14);
    REQUIRE(faceX->getBlockLight(CHUNK_EDGE - 1, 8, 0) == 13);
    REQUIRE(faceZ->getBlockLight(0, 8, CHUNK_EDGE - 1) == 13);
    REQUIRE(diagonal->getBlockLight(CHUNK_EDGE - 1, 8, CHUNK_EDGE - 1) == 12);

    MeshSnapshot firstVisible;
    REQUIRE(world.snapshotForMeshing(DIAGONAL, firstVisible));
    REQUIRE(firstVisible.blockLightAt(CHUNK_EDGE - 1, 8, CHUNK_EDGE - 1) == 12);
}

TEST_CASE("Publication lighting defers a tall saved stack without publishing stale meshes",
          "[world][light][streaming][publication][vertical][performance][regression]") {
    TempDir directory("publication_tall_stack");
    SaveManager saves(directory.path());
    constexpr int32_t LOWER_SECTION = 20;
    constexpr int32_t BRIDGE_SECTION = 52;
    constexpr int32_t UPPER_SECTION = BRIDGE_SECTION + 1;
    for (int32_t section = LOWER_SECTION; section <= UPPER_SECTION; ++section) {
        Chunk saved(ChunkPos{0, section, 0});
        saved.fill(BlockType::AIR);
        saved.generated = true;
        saves.saveChunk(saved);
    }
    REQUIRE(saves.flush());

    World world(42, 4);
    world.setSaveManager(&saves);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({offsetX, offsetZ}));
            if (offsetX != 0 || offsetZ != 0) {
                REQUIRE(world.getChunk({offsetX, BRIDGE_SECTION, offsetZ}));
            }
        }
    }
    for (int32_t section = LOWER_SECTION; section < BRIDGE_SECTION; ++section) {
        REQUIRE(world.getChunk({0, section, 0}));
    }
    REQUIRE(world.getChunk({0, UPPER_SECTION, 0}));
    const StreamingWorkStats beforeBridge = world.getStreamingWorkStats();
    REQUIRE(world.getChunk({0, BRIDGE_SECTION, 0}));

    MeshSnapshot snapshot;
    REQUIRE_FALSE(world.snapshotForMeshing({0, BRIDGE_SECTION, 0}, snapshot));
    const StreamingWorkStats deferred = world.getStreamingWorkStats();
    REQUIRE(deferred.publicationLightDeferredCubes > 0);
    REQUIRE(deferred.publicationLightMaxDeferredQueue > 0);
    REQUIRE(deferred.publicationLightMaxSyncFloods <= 32);
    REQUIRE(deferred.publicationLightSectionVisits - beforeBridge.publicationLightSectionVisits <=
            static_cast<uint64_t>(WORLD_VERTICAL_CHUNKS));

    bool ready = false;
    for (int pass = 0; pass < 64 && !ready; ++pass) {
        world.reconcileLight(16);
        ready = world.snapshotForMeshing({0, BRIDGE_SECTION, 0}, snapshot);
    }
    REQUIRE(ready);
    REQUIRE(snapshot.derivedSkyLightValid);
    REQUIRE(snapshot.skyLightAt(8, 8, 8) == 15);
    REQUIRE(world.getStreamingWorkStats().publicationLightMaxSyncFloods <= 32);
    for (int pass = 0;
         pass < 64 && world.getStreamingWorkStats().publicationLightDeferredQueue != 0; ++pass) {
        world.reconcileLight(16);
    }
    REQUIRE(world.getStreamingWorkStats().publicationLightDeferredQueue == 0);
}

TEST_CASE("World reconciles skylight after opaque roof edits", "[world][light][sky][edit]") {
    World world(42, 4);
    auto chunk = world.getChunk({0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(chunk);
    constexpr int32_t roofY = WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8;

    for (int64_t z = 0; z < CHUNK_EDGE; ++z)
        for (int64_t x = 0; x < CHUNK_EDGE; ++x)
            world.setBlock(x, roofY, z, BlockType::STONE);
    world.reconcileLight(64);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 0);
    REQUIRE(chunk->getSkyLight(8, 9, 8) == 15);

    for (int64_t z = 0; z < CHUNK_EDGE; ++z)
        for (int64_t x = 0; x < CHUNK_EDGE; ++x)
            world.setBlock(x, roofY, z, BlockType::AIR);
    world.reconcileLight(64);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 15);
}

TEST_CASE("World skylight edits preserve negative-coordinate column mapping",
          "[world][light][sky][edit][negative]") {
    World world(42, 4);
    auto chunk = world.getChunk({-1, WORLD_MAX_CHUNK_Y, -1});
    REQUIRE(chunk);
    constexpr int64_t worldX = -1;
    constexpr int64_t worldZ = -1;
    constexpr int32_t coverY = WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8;

    world.setBlock(worldX, coverY, worldZ, BlockType::LOG);
    world.reconcileLight(64);
    REQUIRE(chunk->getSkyLight(Chunk::worldToLocal(worldX), 7, Chunk::worldToLocal(worldZ)) == 14);

    world.setBlock(worldX, coverY, worldZ, BlockType::AIR);
    world.reconcileLight(64);
    REQUIRE(chunk->getSkyLight(Chunk::worldToLocal(worldX), 7, Chunk::worldToLocal(worldZ)) == 15);
}

TEST_CASE("World relights player edits synchronously before remeshing", "[world][light][edit]") {
    World world(42, 4);
    auto chunk = world.getChunk({0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(chunk);
    // The snapshot below needs the full 3x3 column-plan neighborhood, which
    // only exists once the neighboring cubes have generated.
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.getChunk({offsetX, WORLD_MAX_CHUNK_Y, offsetZ}));
        }
    }
    constexpr int32_t placeY = WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8;
    const uint32_t versionBefore = chunk->version.load(std::memory_order_relaxed);

    // The render thread rebuilds an edited near-camera cube on the very next
    // frame, so the packed light must be correct the moment setBlock returns,
    // with no reconcileLight pass in between.
    world.setBlock(8, placeY, 8, BlockType::STONE);
    REQUIRE(chunk->getSkyLight(8, 8, 8) == 0);
    REQUIRE(chunk->getSkyLight(8, 9, 8) == 15);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 14);
    // One bump for the block write and one for the flood's light diff.
    REQUIRE(chunk->version.load(std::memory_order_relaxed) >= versionBefore + 2);

    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing({0, WORLD_MAX_CHUNK_Y, 0}, snapshot));
    REQUIRE(snapshot.derivedSkyLightValid);
    REQUIRE(snapshot.skyLightAt(8, 9, 8) == 15);
    REQUIRE(snapshot.skyLightAt(8, 7, 8) == 14);

    // The synchronous flood must already be the fixed point: later queued
    // reconcile passes may not move any value it produced.
    const uint8_t below = chunk->getSkyLight(8, 7, 8);
    const uint8_t beside = chunk->getSkyLight(9, 8, 8);
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == below);
    REQUIRE(chunk->getSkyLight(9, 8, 8) == beside);
}

TEST_CASE("World spreads a placed emitter's light without a reconcile pass",
          "[world][light][edit]") {
    World world(42);
    auto chunk = world.getChunk(ChunkPos{0, 4, 0});
    REQUIRE(chunk);
    chunk->fill(BlockType::AIR);

    world.setBlock(8, 4 * CHUNK_EDGE + 8, 8, BlockType::LAVA);
    REQUIRE(chunk->getBlockLight(8, 8, 8) == 15);
    REQUIRE(chunk->getBlockLight(8, 9, 8) == 14);
    REQUIRE(chunk->getBlockLight(9, 8, 8) == 14);
}

TEST_CASE("World relights the face neighbor of a border edit synchronously",
          "[world][light][edit]") {
    World world(42);
    auto lower = world.getChunk(ChunkPos{0, 4, 0});
    auto upper = world.getChunk(ChunkPos{0, 5, 0});
    REQUIRE(lower);
    REQUIRE(upper);
    lower->fill(BlockType::AIR);
    upper->fill(BlockType::AIR);

    // ly == 15: the emitter's light crosses into the +Y face neighbor, which
    // must flood in the same setBlock call rather than on a later tick.
    world.setBlock(8, 5 * CHUNK_EDGE - 1, 8, BlockType::LAVA);
    REQUIRE(lower->getBlockLight(8, CHUNK_EDGE - 1, 8) == 15);
    REQUIRE(upper->getBlockLight(8, 0, 8) == 14);
}

TEST_CASE("World relights a non-border edit's face neighbor synchronously",
          "[world][light][edit]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    auto west = world.getChunk(ChunkPos{-1, 4, 0});
    REQUIRE(home);
    REQUIRE(west);
    home->fill(BlockType::AIR);
    west->fill(BlockType::AIR);

    const uint32_t westVersionBefore = west->version.load(std::memory_order_relaxed);
    // Local x == 1 is not on the chunk border, yet the emitter still reaches the
    // -X neighbor. The old immediate flood only covered a cell sitting exactly
    // on a face, so this neighbor used to relight and remesh a tick late.
    world.setBlock(1, 4 * CHUNK_EDGE + 8, 8, BlockType::LAVA);
    REQUIRE(home->getBlockLight(1, 8, 8) == 15);
    REQUIRE(home->getBlockLight(0, 8, 8) == 14);
    REQUIRE(west->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 13);
    REQUIRE(west->version.load(std::memory_order_relaxed) > westVersionBefore);
}

TEST_CASE("World relights the diagonal neighbor of a corner edit synchronously",
          "[world][light][edit]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    auto east = world.getChunk(ChunkPos{1, 4, 0});
    auto south = world.getChunk(ChunkPos{0, 4, 1});
    auto diagonal = world.getChunk(ChunkPos{1, 4, 1});
    REQUIRE(home);
    REQUIRE(east);
    REQUIRE(south);
    REQUIRE(diagonal);
    home->fill(BlockType::AIR);
    east->fill(BlockType::AIR);
    south->fill(BlockType::AIR);
    diagonal->fill(BlockType::AIR);

    const uint32_t diagonalVersionBefore = diagonal->version.load(std::memory_order_relaxed);
    // A +X+Z corner emitter reaches the edge-diagonal cube, which shares no face
    // with the edited cube. The flood chain hops home -> face -> diagonal and
    // must light and dirty it in the same call.
    world.setBlock(CHUNK_EDGE - 1, 4 * CHUNK_EDGE + 8, CHUNK_EDGE - 1, BlockType::LAVA);
    REQUIRE(diagonal->getBlockLight(0, 8, 0) == 13);
    REQUIRE(diagonal->version.load(std::memory_order_relaxed) > diagonalVersionBefore);
}

TEST_CASE("World removes a broken emitter's neighbor light synchronously", "[world][light][edit]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    auto west = world.getChunk(ChunkPos{-1, 4, 0});
    REQUIRE(home);
    REQUIRE(west);
    home->fill(BlockType::AIR);
    west->fill(BlockType::AIR);

    world.setBlock(1, 4 * CHUNK_EDGE + 8, 8, BlockType::LAVA);
    REQUIRE(west->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 13);
    // Breaking the emitter must darken the neighbor in the same call, not a tick
    // later; floodChunk recomputes from scratch, so removal converges too.
    world.setBlock(1, 4 * CHUNK_EDGE + 8, 8, BlockType::AIR);
    REQUIRE(west->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 0);

    // The synchronous cross-chunk result is already the fixed point.
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    REQUIRE(west->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 0);
}

TEST_CASE("Packed skylight crosses both vertical mask words at the world limits",
          "[world][light][sky][vertical][v4]") {
    VerticalSectionMask loaded;
    for (int32_t section = WORLD_MIN_CHUNK_Y; section <= WORLD_MAX_CHUNK_Y; ++section) {
        loaded.set(section);
    }
    REQUIRE(loaded.containsRange(WORLD_MIN_CHUNK_Y, WORLD_MAX_CHUNK_Y));
    VerticalSectionMask required;
    required.set(WORLD_MIN_CHUNK_Y);
    required.set(WORLD_MAX_CHUNK_Y);
    REQUIRE(loaded.containsAllSetSections(required, WORLD_MIN_CHUNK_Y));

    Chunk bottom(ChunkPos{0, WORLD_MIN_CHUNK_Y, 0});
    LightEngine::SkyLightSeedColumns bottomSeeds;
    bottomSeeds.set(8, 8, WORLD_MIN_Y);
    REQUIRE(LightEngine::floodChunk(bottom, {}, bottomSeeds));
    REQUIRE(bottom.getSkyLight(8, 0, 8) == 15);

    Chunk top(ChunkPos{0, WORLD_MAX_CHUNK_Y, 0});
    LightEngine::SkyLightSeedColumns topSeeds;
    topSeeds.set(8, 8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE);
    REQUIRE(LightEngine::floodChunk(top, {}, topSeeds));
    REQUIRE(top.getSkyLight(8, CHUNK_EDGE - 1, 8) == 15);
}

TEST_CASE("World relights a torch across a cube face before setBlock returns",
          "[world][light][edit][torch]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    auto west = world.getChunk(ChunkPos{-1, 4, 0});
    REQUIRE(home);
    REQUIRE(west);
    home->fill(BlockType::AIR);
    west->fill(BlockType::AIR);

    const uint64_t revisionBefore = world.lightingRevision();
    world.setBlock(1, 4 * CHUNK_EDGE + 8, 8, BlockType::TORCH);
    REQUIRE(home->getBlockLight(1, 8, 8) == 14);
    REQUIRE(home->getBlockLight(0, 8, 8) == 13);
    REQUIRE(west->getBlockLight(CHUNK_EDGE - 1, 8, 8) == 12);
    REQUIRE(world.lightingRevision() > revisionBefore);
}

TEST_CASE("World batches dynamic object light probes under resident authority",
          "[world][light][entity][snapshot]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    REQUIRE(home);
    home->fill(BlockType::AIR);
    constexpr int32_t WORLD_Y = 4 * CHUNK_EDGE + 8;
    world.setBlock(1, WORLD_Y, 8, BlockType::TORCH);

    const std::array<BlockPos, 4> positions{
        BlockPos{1, WORLD_Y, 8},
        BlockPos{2, WORLD_Y, 8},
        BlockPos{1'000'000, WORLD_Y, 1'000'000},
        BlockPos{1, WORLD_MAX_Y + 1, 8},
    };
    std::array<uint8_t, 5> samples{0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    world.samplePackedLightsIfLoaded(positions, samples);

    REQUIRE(samples[0] == home->getPackedLight(1, 8, 8));
    REQUIRE(samples[1] == home->getPackedLight(2, 8, 8));
    REQUIRE((samples[0] & 0x0F) == 14);
    REQUIRE((samples[1] & 0x0F) == 13);
    REQUIRE(samples[2] == 0);
    REQUIRE(samples[3] == 0);
    REQUIRE(samples[4] == 0);
}

TEST_CASE("World relights furnace state transitions synchronously",
          "[world][light][edit][furnace]") {
    World world(42);
    auto chunk = world.getChunk(ChunkPos{0, 4, 0});
    REQUIRE(chunk);
    chunk->fill(BlockType::AIR);
    constexpr int32_t WORLD_Y = 4 * CHUNK_EDGE + 8;

    world.setBlock(8, WORLD_Y, 8, BlockType::FURNACE_LIT);
    REQUIRE(chunk->getBlockLight(8, 8, 8) == 13);
    REQUIRE(chunk->getBlockLight(9, 8, 8) == 12);

    world.setBlock(8, WORLD_Y, 8, BlockType::FURNACE);
    REQUIRE(chunk->getBlockLight(8, 8, 8) == 0);
    REQUIRE(chunk->getBlockLight(9, 8, 8) == 0);
}

TEST_CASE("World normalizes an orphan saved furnace emitter before lighting",
          "[world][light][save][torch][furnace]") {
    TempDir directory("gameplay_light_reload");
    SaveManager saves(directory.path());
    Chunk saved(ChunkPos{0, 4, 0});
    saved.fill(BlockType::AIR);
    saved.setBlock(1, 8, 8, BlockType::TORCH);
    saved.setBlock(8, 8, 8, BlockType::FURNACE_LIT);
    saved.setBlock(15, 8, 8, BlockType::LAVA);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    World world(42);
    world.setSaveManager(&saves);
    const uint64_t revisionBeforeLoad = world.lightingRevision();
    auto loaded = world.getChunk(ChunkPos{0, 4, 0});
    REQUIRE(loaded);
    REQUIRE(loaded->getBlockLight(1, 8, 8) == 14);
    REQUIRE(loaded->getBlock(8, 8, 8) == BlockType::FURNACE);
    REQUIRE(loaded->getBlockLight(8, 8, 8) < 13);
    REQUIRE(loaded->getBlockLight(15, 8, 8) == 15);
    REQUIRE(world.lightingRevision() == revisionBeforeLoad);
}

TEST_CASE("Saved active furnaces publish their emissive block and derived light atomically",
          "[world][light][save][furnace][publication][render][regression]") {
    TempDir directory("active_furnace_first_publication");
    SaveManager saves(directory.path());
    constexpr ChunkPos CHUNK{0, WORLD_MAX_CHUNK_Y, 0};
    constexpr BlockPos ACTIVE{8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8};
    constexpr BlockPos ORPHAN{12, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8};

    Chunk saved(CHUNK);
    saved.fill(BlockType::AIR);
    saved.setBlock(8, 8, 8, BlockType::FURNACE_LIT);
    saved.setBlock(12, 8, 8, BlockType::FURNACE_LIT);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    FurnaceState burning;
    burning.burnTicksRemaining = 400;
    burning.burnTicksTotal = 800;
    FurnaceMap furnaces{{ACTIVE, burning}};
    auto visuals = std::make_shared<FurnaceVisualAuthority>();
    visuals->replace(furnaces);

    World world(42);
    world.setSavedChunkProjection({
        .apply = [visuals](Chunk& chunk) { return visuals->projectSavedChunk(chunk); },
        .currentRevision = [visuals] { return visuals->revision(); },
    });
    world.setSaveManager(&saves);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto plan = world.generator().getColumnPlan({offsetX, offsetZ});
            REQUIRE(plan);
            int32_t topSection =
                std::max(CHUNK.y, Chunk::worldToChunkY(std::clamp(plan->maximumSurfaceY(),
                                                                  WORLD_MIN_Y, WORLD_MAX_Y)));
            for (const int32_t exposed : plan->exposedSections()) {
                topSection = std::max(topSection, exposed);
            }
            for (int32_t section = CHUNK.y; section <= topSection; ++section) {
                if (offsetX == 0 && offsetZ == 0 && section == CHUNK.y)
                    continue;
                REQUIRE(world.getChunk({offsetX, section, offsetZ}));
            }
        }
    }
    const uint64_t lightingRevision = world.lightingRevision();
    const std::shared_ptr<Chunk> loaded = world.getChunk(CHUNK);
    REQUIRE(loaded);

    // The live sidecar-backed furnace is active in the first resident cube
    // and its initial publication flood already contains the source. The
    // orphan is sanitized in the same transform and never emits.
    REQUIRE(loaded->getBlockWorld(ACTIVE.x, ACTIVE.y, ACTIVE.z) == BlockType::FURNACE_LIT);
    REQUIRE(loaded->getBlockLight(8, 8, 8) == 13);
    REQUIRE(loaded->getBlockLight(9, 8, 8) == 12);
    REQUIRE(loaded->getBlockWorld(ORPHAN.x, ORPHAN.y, ORPHAN.z) == BlockType::FURNACE);
    REQUIRE(world.lightingRevision() == lightingRevision);

    MeshSnapshot firstSnapshot;
    REQUIRE(world.snapshotForMeshing(CHUNK, firstSnapshot));
    MeshScratch scratch;
    const MeshOutput firstMesh = LODMesher::buildMesh(firstSnapshot, scratch);
    bool foundActiveMouth = false;
    for (const Vertex& vertex : firstMesh.vertices) {
        if (unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::FURNACE_LIT)) {
            continue;
        }
        foundActiveMouth = true;
        REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::MINUS_Z);
        REQUIRE(unpackEmissive(vertex.faceAttr));
        REQUIRE(unpackBlockLight(vertex.faceAttr) > 0);
    }
    REQUIRE(foundActiveMouth);
}

TEST_CASE("Saved chunk publication retries a stale visual projection before insertion",
          "[world][save][furnace][publication][concurrency][regression]") {
    TempDir directory("stale_furnace_projection");
    SaveManager saves(directory.path());
    constexpr ChunkPos CHUNK{-2, 4, 3};
    constexpr BlockPos POSITION{-2 * CHUNK_EDGE + 5, 4 * CHUNK_EDGE + 6, 3 * CHUNK_EDGE + 7};

    Chunk saved(CHUNK);
    saved.fill(BlockType::AIR);
    saved.setBlock(5, 6, 7, BlockType::FURNACE);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    FurnaceMap furnaces{{POSITION, FurnaceState{}}};
    auto visuals = std::make_shared<FurnaceVisualAuthority>();
    visuals->replace(furnaces);
    std::atomic<int> applications{0};

    World world(42);
    world.setSavedChunkProjection({
        .apply =
            [visuals, &applications, POSITION](Chunk& chunk) {
                const uint64_t applied = visuals->projectSavedChunk(chunk);
                if (applications.fetch_add(1, std::memory_order_relaxed) == 0) {
                    visuals->set(POSITION, BlockType::FURNACE_LIT);
                }
                return applied;
            },
        .currentRevision = [visuals] { return visuals->revision(); },
    });
    world.setSaveManager(&saves);
    const std::shared_ptr<Chunk> loaded = world.getChunk(CHUNK);
    REQUIRE(loaded);
    REQUIRE(applications.load(std::memory_order_relaxed) == 2);
    REQUIRE(loaded->getBlock(5, 6, 7) == BlockType::FURNACE_LIT);
    REQUIRE(loaded->getBlockLight(5, 6, 7) == 13);
}

TEST_CASE("Beds and chests synchronously update derived sky paths",
          "[world][light][sky][edit][gameplay]") {
    World world(42);
    auto chunk = world.getChunk(ChunkPos{0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(chunk);
    constexpr int32_t BLOCK_Y = WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8;

    world.setBlock(8, BLOCK_Y, 8, BlockType::BED);
    REQUIRE(chunk->getSkyLight(8, 8, 8) == 15);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 15);
    world.setBlock(8, BLOCK_Y, 8, BlockType::AIR);

    world.setBlock(8, BLOCK_Y, 8, BlockType::CHEST);
    REQUIRE(chunk->getSkyLight(8, 8, 8) == 0);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 14);
    world.setBlock(8, BLOCK_Y, 8, BlockType::AIR);
    REQUIRE(chunk->getSkyLight(8, 7, 8) == 15);
}

TEST_CASE("Floor torches are supported decorations and resist fluid replacement",
          "[world][torch][fluid][support][regression]") {
    REQUIRE(isFloorTorch(BlockType::TORCH));
    REQUIRE_FALSE(isFlora(BlockType::TORCH));
    REQUIRE_FALSE(isWaterReplaceable(BlockType::TORCH));
    REQUIRE(isWaterReplaceable(BlockType::TALL_GRASS));
    REQUIRE(hasFullBlockCollision(BlockType::STONE));
    REQUIRE(hasFullBlockCollision(BlockType::GLASS));
    REQUIRE_FALSE(hasFullBlockCollision(BlockType::BED));
}

TEST_CASE("Torch support dependency crosses a vertical cube boundary",
          "[world][torch][support][boundary][regression]") {
    World world(42);
    constexpr int32_t SUPPORT_Y = 4 * CHUNK_EDGE + CHUNK_EDGE - 1;
    auto supportChunk = world.getChunk(ChunkPos{0, 4, 0});
    auto torchChunk = world.getChunk(ChunkPos{0, 5, 0});
    REQUIRE(supportChunk);
    REQUIRE(torchChunk);
    supportChunk->fill(BlockType::AIR);
    torchChunk->fill(BlockType::AIR);

    REQUIRE(world.trySetBlock(8, SUPPORT_Y, 8, BlockType::STONE));
    REQUIRE(world.trySetBlock(8, SUPPORT_Y + 1, 8, BlockType::TORCH));
    const std::optional<BlockType> decoration = world.findBlockIfLoaded(8, SUPPORT_Y + 1, 8);
    REQUIRE(decoration == BlockType::TORCH);
    REQUIRE(losesSupportWhenBlockBelowBreaks(*decoration));
    REQUIRE(hasFullBlockCollision(*world.findBlockIfLoaded(8, SUPPORT_Y, 8)));

    // Engine::breakBlock follows this shared predicate before dropping the
    // decoration, and World resolves both resident cubes without truncating Y.
    REQUIRE(world.trySetBlock(8, SUPPORT_Y, 8, BlockType::AIR));
    REQUIRE(world.trySetBlock(8, SUPPORT_Y + 1, 8, BlockType::AIR));
    REQUIRE(world.findBlockIfLoaded(8, SUPPORT_Y + 1, 8) == BlockType::AIR);
}

TEST_CASE("Static source water stays transparent without activating runtime fluid work",
          "[world][light][water][fluid][regression]") {
    REQUIRE_FALSE(blockEditResetsIndirectLighting(BlockType::AIR, BlockType::WATER));
    REQUIRE_FALSE(blockEditResetsIndirectLighting(BlockType::WATER, BlockType::AIR));
    REQUIRE_FALSE(blockEditResetsIndirectLighting(BlockType::WATER, BlockType::WATER));
    REQUIRE(blockEditResetsIndirectLighting(BlockType::AIR, BlockType::STONE));
    REQUIRE(blockEditResetsIndirectLighting(BlockType::STONE, BlockType::AIR));
    REQUIRE(blockEditResetsIndirectLighting(BlockType::AIR, BlockType::TORCH));
    REQUIRE(blockEditResetsIndirectLighting(BlockType::WATER, BlockType::LAVA));

    World world(42);
    auto loaded = world.getChunk(ChunkPos{0, WORLD_MAX_CHUNK_Y, 0});
    REQUIRE(loaded);
    // Model canonical water as an already-published implicit source. Replacing
    // transparent air directly preserves its derived skylight, and the public
    // no-op placement must not reinterpret it as mutable bucket water.
    loaded->setBlock(8, 8, 8, BlockType::WATER);
    loaded->setFluidState(8, 8, 8, FluidState::source());
    REQUIRE(loaded->getSkyLight(8, 8, 8) == 15);
    REQUIRE(world.getPendingFluidCount() == 0);

    const uint64_t revisionBefore = world.lightingRevision();
    REQUIRE_FALSE(world.trySetBlock(8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8, BlockType::WATER));
    REQUIRE(world.getPendingFluidCount() == 0);
    REQUIRE(world.lightingRevision() == revisionBefore);

    loaded->setFluidState(8, 8, 8, FluidState::flowing(4));
    const uint64_t flowingRevision = world.lightingRevision();
    REQUIRE(world.trySetBlock(8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8, BlockType::WATER));
    REQUIRE(world.readFluidCell({8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8}).state ==
            FluidState::source());
    REQUIRE(world.getPendingFluidCount() > 0);
    REQUIRE(world.lightingRevision() == flowingRevision);

    const uint64_t sourceNoOpRevision = world.lightingRevision();
    const size_t sourceNoOpPending = world.getPendingFluidCount();
    REQUIRE_FALSE(world.trySetBlock(8, WORLD_MAX_CHUNK_Y * CHUNK_EDGE + 8, 8, BlockType::WATER));
    REQUIRE(world.lightingRevision() == sourceNoOpRevision);
    REQUIRE(world.getPendingFluidCount() == sourceNoOpPending);
}

TEST_CASE("Snapshot mesher decodes independent packed light channels",
          "[world][light][mesher][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 0xB5;
        }
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundBoundary = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_X ||
            static_cast<float>(vertex.px) != 16.0F) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 11);
        REQUIRE(unpackBlockLight(vertex.faceAttr) == 5);
        foundBoundary = true;
    }
    REQUIRE(foundBoundary);
}

TEST_CASE("Snapshot mesher preserves a smooth per-vertex block-light gradient",
          "[world][light][mesher][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    for (int dy = -1; dy <= 1; ++dy) {
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 7)] = 0x00;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8)] = 0x08;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 9)] = 0x0F;
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    uint8_t low = 255;
    uint8_t high = 0;
    int faceVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_X ||
            static_cast<float>(vertex.px) != 16.0F) {
            continue;
        }
        const uint8_t value = unpackBlockLight(vertex.faceAttr);
        low = std::min(low, value);
        high = std::max(high, value);
        ++faceVertices;
    }
    REQUIRE(faceVertices == 4);
    REQUIRE(low <= 5);
    REQUIRE(high >= 10);
    REQUIRE(low < high);
}

TEST_CASE("Snapshot mesher retains greedy merging under uniform packed light",
          "[world][light][mesher][smooth][greedy]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    for (int z = 6; z <= 8; ++z) {
        for (int x = 6; x <= 8; ++x) {
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::STONE;
        }
    }
    for (int z = 5; z <= 9; ++z) {
        for (int x = 5; x <= 9; ++x) {
            snapshot.packedLight[MeshSnapshot::index(x, 9, z)] = 0xF7;
        }
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    int topVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        ++topVertices;
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        REQUIRE(unpackBlockLight(vertex.faceAttr) == 7);
    }
    REQUIRE(topVertices == 4);
}

TEST_CASE("World removes a corner emitter's diagonal light synchronously",
          "[world][light][edit][diagonal]") {
    World world(42);
    auto home = world.getChunk(ChunkPos{0, 4, 0});
    auto east = world.getChunk(ChunkPos{1, 4, 0});
    auto south = world.getChunk(ChunkPos{0, 4, 1});
    auto diagonal = world.getChunk(ChunkPos{1, 4, 1});
    REQUIRE(home);
    REQUIRE(east);
    REQUIRE(south);
    REQUIRE(diagonal);
    home->fill(BlockType::AIR);
    east->fill(BlockType::AIR);
    south->fill(BlockType::AIR);
    diagonal->fill(BlockType::AIR);

    world.setBlock(CHUNK_EDGE - 1, 4 * CHUNK_EDGE + 8, CHUNK_EDGE - 1, BlockType::LAVA);
    REQUIRE(diagonal->getBlockLight(0, 8, 0) == 13);
    world.setBlock(CHUNK_EDGE - 1, 4 * CHUNK_EDGE + 8, CHUNK_EDGE - 1, BlockType::AIR);
    REQUIRE(diagonal->getBlockLight(0, 8, 0) == 0);
}

TEST_CASE("LightEngine: lava light falls off one level per block", "[world][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA); // one source in open air

    REQUIRE(LightEngine::computeSelfLight(chunk));

    REQUIRE(chunk.getBlockLight(8, 8, 8) == 15); // the source itself
    REQUIRE(chunk.getBlockLight(9, 8, 8) == 14); // one block away
    REQUIRE(chunk.getBlockLight(10, 8, 8) == 13);
    REQUIRE(chunk.getBlockLight(8, 11, 8) == 12); // three up
    REQUIRE(chunk.getBlockLight(15, 8, 8) == 8);  // seven away
    // A cell 15+ blocks away (across two axes) is dark again.
    REQUIRE(chunk.getBlockLight(0, 8, 0) == 0);
}

TEST_CASE("LightEngine: a placed torch emits and propagates light", "[world][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::TORCH); // a lone torch in open air

    REQUIRE(LightEngine::computeSelfLight(chunk));

    REQUIRE(chunk.getBlockLight(8, 8, 8) == 14); // the torch's own emission
    REQUIRE(chunk.getBlockLight(9, 8, 8) == 13); // falls off one level per block
    REQUIRE(chunk.getBlockLight(8, 12, 8) == 10);
    REQUIRE(chunk.getBlockLight(8, 8, 8 - 6) == 8);
}

TEST_CASE("LightEngine: opaque blocks do not receive light", "[world][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA);
    chunk.setBlock(9, 8, 8, BlockType::STONE); // opaque neighbor

    LightEngine::computeSelfLight(chunk);

    // The stone cell stays dark (light never enters an opaque cell), but light
    // still routes around it through the open air above.
    REQUIRE(chunk.getBlockLight(9, 8, 8) == 0);
    REQUIRE(chunk.getBlockLight(8, 9, 8) == 14); // air above the source
}

TEST_CASE("LightEngine: a lava-free chunk allocates no light", "[world][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    REQUIRE_FALSE(LightEngine::computeSelfLight(chunk)); // nothing changed
    REQUIRE_FALSE(chunk.hasBlockLight());                // stays unallocated
    REQUIRE(chunk.getBlockLight(8, 8, 8) == 0);
}

TEST_CASE("LightEngine: flood is a pure function of chunk contents", "[world][light]") {
    // The same blocks produce bit-identical light, so order cannot affect the fixed point.
    Chunk a(ChunkPos{0, 4, 0}), b(ChunkPos{0, 4, 0});
    for (Chunk* c : {&a, &b}) {
        c->setBlock(4, 8, 4, BlockType::LAVA);
        c->setBlock(11, 8, 11, BlockType::LAVA);
        c->setBlock(7, 8, 7, BlockType::STONE);
    }
    LightEngine::computeSelfLight(a);
    LightEngine::computeSelfLight(b);
    REQUIRE(a.packedLightData() == b.packedLightData());
}

TEST_CASE("LightEngine: light spills across a chunk border", "[world][light]") {
    // A neighbor's border light seeds this chunk's edge (minus one), then floods
    // inward. Cross-cube reconciliation relies on exactly this behavior.
    Chunk neighbor(ChunkPos{-1, 4, 0});
    neighbor.setBlockLight(CHUNK_WIDTH - 1, 8, 8, 10); // its +X wall glows

    Chunk self(ChunkPos{0, 4, 0}); // all air, no own source
    LightEngine::FaceNeighbors faces{&neighbor, nullptr, nullptr, nullptr, nullptr, nullptr};
    REQUIRE(LightEngine::floodChunk(self, faces));

    REQUIRE(self.getBlockLight(0, 8, 8) == 9); // border pulls neighbor - 1
    REQUIRE(self.getBlockLight(1, 8, 8) == 8); // then floods inward
}

TEST_CASE("LightEngine: light spills across a vertical cube border", "[world][light]") {
    Chunk below(ChunkPos{0, 3, 0});
    below.setBlockLight(8, CHUNK_EDGE - 1, 8, 10);

    Chunk self(ChunkPos{0, 4, 0});
    LightEngine::FaceNeighbors faces{nullptr, nullptr, nullptr, nullptr, &below, nullptr};
    REQUIRE(LightEngine::floodChunk(self, faces));
    REQUIRE(self.getBlockLight(8, 0, 8) == 9);
    REQUIRE(self.getBlockLight(8, 1, 8) == 8);
}

TEST_CASE("World reconciles light across vertical cube borders", "[world][light]") {
    World world(42);
    auto lower = world.getChunk(ChunkPos{0, 4, 0});
    auto upper = world.getChunk(ChunkPos{0, 5, 0});
    lower->fill(BlockType::AIR);
    upper->fill(BlockType::AIR);

    world.setBlock(8, 79, 8, BlockType::LAVA);
    for (int pass = 0; pass < 8; ++pass)
        world.reconcileLight(64);

    REQUIRE(lower->getBlockLight(8, CHUNK_EDGE - 1, 8) == 15);
    REQUIRE(upper->getBlockLight(8, 0, 8) == 14);
}

TEST_CASE("World reports the highest opaque block in loaded cubic columns", "[world][weather]") {
    World world(42);
    REQUIRE_FALSE(world.surfaceHeightIfLoaded(-1, -1).has_value());

    auto lower = world.getChunk(ChunkPos{-1, 4, -1});
    auto upper = world.getChunk(ChunkPos{-1, 6, -1});
    lower->fill(BlockType::AIR);
    upper->fill(BlockType::AIR);
    lower->setBlock(15, 11, 15, BlockType::STONE);
    upper->setBlock(15, 4, 15, BlockType::STONE);

    REQUIRE(world.surfaceHeightIfLoaded(-1, -1) == 100);
}

TEST_CASE("LightEngine: block light is derived, never serialized", "[world][light]") {
    Chunk original(ChunkPos{2, 4, -1});
    original.setBlock(8, 8, 8, BlockType::LAVA);
    LightEngine::computeSelfLight(original);
    original.setSkyLight(2, 3, 4, 13);
    REQUIRE(original.hasBlockLight());
    REQUIRE(original.getSkyLight(2, 3, 4) == 13);

    // The save size accounts for cubic block and fluid state, not light.
    size_t before = ChunkSerializer::serializedSize(original);
    auto data = ChunkSerializer::serialize(original);
    REQUIRE(data.size() == before);

    auto restored = ChunkSerializer::deserialize(data);
    REQUIRE(restored.has_value());
    REQUIRE(restored->getBlock(8, 8, 8) == BlockType::LAVA);
    REQUIRE_FALSE(restored->hasBlockLight()); // not carried through the save
    REQUIRE_FALSE(restored->hasDerivedLight());
    // ...but recomputable from the blocks alone.
    LightEngine::computeSelfLight(*restored);
    REQUIRE(restored->getBlockLight(9, 8, 8) == 14);
}

TEST_CASE("Default generation settings produce byte-identical cubes", "[worldgen][settings]") {
    ChunkGenerator legacy(42);
    ChunkGenerator configured(42, GenerationSettings{});
    // Surface, underground, and structure-region fixtures around spawn.
    constexpr std::array<ChunkPos, 5> POSITIONS = {
        ChunkPos{0, 6, 0},  ChunkPos{3, 5, -2}, ChunkPos{-4, 4, 7},
        ChunkPos{12, 6, 9}, ChunkPos{1, -2, 1},
    };
    for (ChunkPos position : POSITIONS) {
        Chunk fromLegacy(position);
        legacy.generate(fromLegacy);
        Chunk fromConfigured(position);
        configured.generate(fromConfigured);
        REQUIRE(ChunkSerializer::serialize(fromLegacy) ==
                ChunkSerializer::serialize(fromConfigured));
    }
}

TEST_CASE("Disabled structures reserve no footprints", "[worldgen][settings][structures]") {
    StructurePlacer enabled(42);
    StructurePlacer disabled(42, false);

    // The enabled placer reserves at least one candidate footprint in a wide
    // scan; the disabled placer must reserve none anywhere.
    bool foundReserved = false;
    for (int64_t chunkX = -48; chunkX <= 48 && !foundReserved; ++chunkX) {
        for (int64_t chunkZ = -48; chunkZ <= 48 && !foundReserved; ++chunkZ) {
            const int64_t x = chunkX * CHUNK_WIDTH + 8;
            const int64_t z = chunkZ * CHUNK_DEPTH + 8;
            if (enabled.insideStructure(x, z, chunkX, chunkZ, 2)) {
                foundReserved = true;
                REQUIRE_FALSE(disabled.insideStructure(x, z, chunkX, chunkZ, 2));
            }
        }
    }
    REQUIRE(foundReserved);

    // Disabled placement is invalid for every region.
    ChunkGenerator generator(42, GenerationSettings{.structures = false});
    GenScratch scratch;
    REQUIRE_FALSE(disabled.regionPlacement(0, 0, generator, scratch).valid);
    REQUIRE_FALSE(disabled.regionPlacement(-3, 7, generator, scratch).valid);
}

TEST_CASE("World carries its generation settings", "[world][settings]") {
    const GenerationSettings settings{
        .structures = false, .fauna = false, .weather = true, .dayCycle = false};
    World world(42, MIN_RENDER_DISTANCE_CHUNKS, 64, settings);
    REQUIRE(world.getGenerationSettings() == settings);
    World defaults(42, MIN_RENDER_DISTANCE_CHUNKS, 64);
    REQUIRE(defaults.getGenerationSettings() == GenerationSettings{});
}

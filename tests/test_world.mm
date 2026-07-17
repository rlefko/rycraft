#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/counter_rng.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/inventory.hpp>
#include <engine/input_bindings.hpp>
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
#include <world/fluid.hpp>
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
    STATIC_REQUIRE(WORLD_MAX_Y == 511);
    STATIC_REQUIRE(WORLD_MIN_CHUNK_Y == -8);
    STATIC_REQUIRE(WORLD_MAX_CHUNK_Y == 31);
    STATIC_REQUIRE(WORLD_VERTICAL_CHUNKS == 40);
    STATIC_REQUIRE(SEA_LEVEL == 64);
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
    REQUIRE(static_cast<int>(BlockType::COUNT) == 62);
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
    REQUIRE(isFlora(BlockType::TORCH));
    REQUIRE_FALSE(isSolid(BlockType::TORCH));
    REQUIRE(blockLightEmission(BlockType::TORCH) == 14);
    REQUIRE(isEmissive(BlockType::TORCH));
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
    REQUIRE(std::is_sorted(first->exposedSections().begin(), first->exposedSections().end()));
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
    saves.saveMetadata(12345, Vec3{100.f, 80.f, -50.f}, 9876543210ULL);

    auto metadata = saves.loadMetadata();
    REQUIRE(metadata.has_value());
    REQUIRE(metadata->seed == 12345);
    REQUIRE(metadata->spawnPos == Vec3{100.f, 80.f, -50.f});
    REQUIRE(metadata->worldTime == 9876543210ULL);
    REQUIRE(metadata->chunkFormatVersion == CHUNK_VERSION);
    REQUIRE(metadata->generatorVersion == SaveManager::CURRENT_GENERATOR_VERSION);
}

TEST_CASE("SaveManager preserves non-chunk player metadata", "[save][metadata]") {
    TempDir directory("rycraft_player_metadata");
    SaveManager saves(directory.path());
    SaveManager::PlayerMetadata player;
    player.yaw = 127.5f;
    player.pitch = -31.25f;
    player.health = 13;
    player.selectedSlot = 7;
    player.inventory[0] = BlockType::BASALT;
    player.inventory[7] = BlockType::LILY_PAD;

    saves.saveMetadata(9191, Vec3{12.0f, 88.0f, -4.0f}, 123456, player);
    const auto loaded = saves.loadMetadata();

    REQUIRE(loaded.has_value());
    REQUIRE(loaded->seed == 9191);
    REQUIRE(loaded->spawnPos == Vec3{12.0f, 88.0f, -4.0f});
    REQUIRE(loaded->worldTime == 123456);
    REQUIRE(loaded->player.yaw == Catch::Approx(127.5f));
    REQUIRE(loaded->player.pitch == Catch::Approx(-31.25f));
    REQUIRE(loaded->player.health == 13);
    REQUIRE(loaded->player.selectedSlot == 7);
    REQUIRE(loaded->player.inventory[0] == BlockType::BASALT);
    REQUIRE(loaded->player.inventory[7] == BlockType::LILY_PAD);
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
    world.updatePlayerPosition(0, SEA_LEVEL, 0);
    for (int attempt = 0; attempt < 1000 && world.getPendingChunkCount() > 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(world.getLoadedChunkCount() <= MAX_LOADED_CUBES);
    REQUIRE(world.getLoadedChunkCount() > 0);
    REQUIRE(world.getStreamingWorkStats().loadedCubeHighWater <= MAX_LOADED_CUBES);
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
    for (int attempt = 0; attempt < 2000; ++attempt) {
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
    constexpr ChunkPos horizontalEdge{-EXPLORATION_RADIUS_CHUNKS - 1, initialY, 0};
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
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
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

    // A rebuild within the same camera cube and a one-cube movement retain
    // the complete prior intersection instead of selecting a new equal-rank
    // subset from unordered iteration or tiny distance changes.
    const auto sameCube =
        selectStableMeshCandidates(requirements, initial, {0, 4, 0}, MAX_MESH_RESIDENT_CUBES);
    const auto adjacentCube =
        selectStableMeshCandidates(requirements, sameCube, {1, 4, -1}, MAX_MESH_RESIDENT_CUBES);
    REQUIRE(sameCube == initial);
    REQUIRE(adjacentCube == initial);

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
    REQUIRE(std::count_if(initial.begin(), initial.end(), [&](ChunkPos position) {
                return withEdits.contains(position);
            }) == static_cast<std::ptrdiff_t>(MAX_MESH_RESIDENT_CUBES - promoted.size()));
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

TEST_CASE("Block properties: lava emits light, nothing else does", "[world][light]") {
    REQUIRE(blockLightEmission(BlockType::LAVA) == 15);
    REQUIRE(blockLightEmission(BlockType::STONE) == 0);
    REQUIRE(blockLightEmission(BlockType::AIR) == 0);
    REQUIRE(isEmissive(BlockType::LAVA));
    REQUIRE_FALSE(isEmissive(BlockType::STONE));
    REQUIRE_FALSE(isEmissive(BlockType::GLASS));
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
    REQUIRE(a.blockLightData() == b.blockLightData());
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
    REQUIRE(original.hasBlockLight());

    // The save size accounts for cubic block and fluid state, not light.
    size_t before = ChunkSerializer::serializedSize(original);
    auto data = ChunkSerializer::serialize(original);
    REQUIRE(data.size() == before);

    auto restored = ChunkSerializer::deserialize(data);
    REQUIRE(restored.has_value());
    REQUIRE(restored->getBlock(8, 8, 8) == BlockType::LAVA);
    REQUIRE_FALSE(restored->hasBlockLight()); // not carried through the save
    // ...but recomputable from the blocks alone.
    LightEngine::computeSelfLight(*restored);
    REQUIRE(restored->getBlockLight(9, 8, 8) == 14);
}

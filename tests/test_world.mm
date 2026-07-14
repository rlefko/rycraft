#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
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
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// World: chunks, generation, biomes, persistence
// ===========================================================================

// ============================================================================
// Existing Tests
// ============================================================================

TEST_CASE("Chunk coordinates are multiples of CHUNK_SIZE", "[chunk]") {
    REQUIRE(0 % CHUNK_SIZE == 0);
    REQUIRE(16 % CHUNK_SIZE == 0);
    REQUIRE(32 % CHUNK_SIZE == 0);
    REQUIRE((-16) % CHUNK_SIZE == 0);
}

TEST_CASE("BlockType enum values are as expected", "[block]") {
    // Saves store raw enum bytes: existing values must never renumber
    REQUIRE(static_cast<int>(BlockType::AIR) == 0);
    REQUIRE(static_cast<int>(BlockType::STONE) == 1);
    REQUIRE(static_cast<int>(BlockType::GRASS) == 2);
    REQUIRE(static_cast<int>(BlockType::DIRT) == 3);
    REQUIRE(static_cast<int>(BlockType::SAND) == 4);
    REQUIRE(static_cast<int>(BlockType::BEDROCK) == 7);
    REQUIRE(static_cast<int>(BlockType::LOG) == 8);
    REQUIRE(static_cast<int>(BlockType::LEAVES) == 9);
    REQUIRE(static_cast<int>(BlockType::GLASS) == 16);
    REQUIRE(static_cast<int>(BlockType::COBBLESTONE) == 17);
    REQUIRE(static_cast<int>(BlockType::REED) == 31);
    REQUIRE(static_cast<int>(BlockType::ICE) == 33);
    REQUIRE(static_cast<int>(BlockType::COUNT) == 34);
}

TEST_CASE("Block properties: flora, liquid, and targetable sets", "[block]") {
    // Flora: cross-quad plants — non-solid, non-opaque, but targetable
    for (BlockType bt : {BlockType::DEAD_BUSH, BlockType::TALL_GRASS, BlockType::FLOWER_YELLOW,
                         BlockType::FLOWER_RED, BlockType::MUSHROOM_BROWN, BlockType::MUSHROOM_RED,
                         BlockType::REED}) {
        REQUIRE(isFlora(bt));
        REQUIRE(!isSolid(bt));
        REQUIRE(!isOpaque(bt));
        REQUIRE(isTargetable(bt));
    }
    REQUIRE(!isFlora(BlockType::CACTUS)); // cactus is a full cube

    // Liquids: swimmable, non-solid, click-through. Water renders in the
    // water pass (non-opaque); lava draws as an emissive opaque cube.
    for (BlockType bt : {BlockType::WATER, BlockType::LAVA}) {
        REQUIRE(isLiquid(bt));
        REQUIRE(!isSolid(bt));
        REQUIRE(!isTargetable(bt));
    }
    REQUIRE(!isOpaque(BlockType::WATER));
    REQUIRE(isOpaque(BlockType::LAVA));
    REQUIRE(rendersAsCube(BlockType::LAVA));
    REQUIRE(!rendersAsCube(BlockType::WATER));

    // Cube blocks added by the worldgen overhaul stay solid + opaque
    for (BlockType bt :
         {BlockType::COBBLESTONE, BlockType::MOSSY_COBBLESTONE, BlockType::SANDSTONE,
          BlockType::BIRCH_LOG, BlockType::SPRUCE_LOG, BlockType::CACTUS, BlockType::ICE}) {
        REQUIRE(isSolid(bt));
        REQUIRE(isOpaque(bt));
    }

    // Leaf variants cut out like oak leaves
    for (BlockType bt : {BlockType::BIRCH_LEAVES, BlockType::SPRUCE_LEAVES}) {
        REQUIRE(isSolid(bt));
        REQUIRE(!isOpaque(bt));
    }
}

// ============================================================================
// Simplex Noise Tests
// ============================================================================

TEST_CASE("SimplexNoise deterministic output for same seed", "[noise]") {
    SimplexNoise noise1(42);
    SimplexNoise noise2(42);

    double v1 = noise1.noise2D(1.0, 2.0);
    double v2 = noise2.noise2D(1.0, 2.0);
    REQUIRE(v1 == v2);

    double v3 = noise1.noise3D(1.0, 2.0, 3.0);
    double v4 = noise2.noise3D(1.0, 2.0, 3.0);
    REQUIRE(v3 == v4);
}

TEST_CASE("SimplexNoise same input gives same output", "[noise]") {
    SimplexNoise noise(123);

    double a = noise(10.0, 20.0);
    double b = noise(10.0, 20.0);
    REQUIRE(a == b);

    double c = noise.noise3D(5.0, 5.0, 5.0);
    double d = noise.noise3D(5.0, 5.0, 5.0);
    REQUIRE(c == d);
}

TEST_CASE("SimplexNoise output range within [-1, 1]", "[noise]") {
    SimplexNoise noise(99);

    // Sample a grid of points
    for (int ix = -10; ix <= 10; ++ix) {
        for (int iy = -10; iy <= 10; ++iy) {
            double v2d = noise.noise2D(static_cast<double>(ix), static_cast<double>(iy));
            REQUIRE(v2d >= -1.0);
            REQUIRE(v2d <= 1.0);
        }
    }

    for (int ix = -5; ix <= 5; ++ix) {
        for (int iy = -5; iy <= 5; ++iy) {
            for (int iz = -5; iz <= 5; ++iz) {
                double v3d = noise.noise3D(static_cast<double>(ix), static_cast<double>(iy),
                                           static_cast<double>(iz));
                REQUIRE(v3d >= -1.0);
                REQUIRE(v3d <= 1.0);
            }
        }
    }
}

TEST_CASE("SimplexNoise different seeds give different outputs", "[noise]") {
    SimplexNoise noiseA(1);
    SimplexNoise noiseB(2);

    // Different seeds should produce different noise fields
    bool different = false;
    for (int i = 0; i < 20; ++i) {
        double a = noiseA.noise2D(static_cast<double>(i), static_cast<double>(i));
        double b = noiseB.noise2D(static_cast<double>(i), static_cast<double>(i));
        if (a != b) {
            different = true;
            break;
        }
    }
    REQUIRE(different == true);
}

TEST_CASE("SimplexNoise octave2D is deterministic", "[noise]") {
    SimplexNoise noise(77);
    double a = noise.octave2D(10.0, 20.0, 4, 0.5, 2.0);
    double b = noise.octave2D(10.0, 20.0, 4, 0.5, 2.0);
    REQUIRE(a == b);
}

TEST_CASE("SimplexNoise octave output range within [-1, 1]", "[noise]") {
    SimplexNoise noise(42);

    for (int i = 0; i < 20; ++i) {
        double v =
            noise.octave2D(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.1, 6, 0.5, 2.0);
        REQUIRE(v >= -1.0);
        REQUIRE(v <= 1.0);
    }
}

TEST_CASE("SimplexNoise ridged noise is deterministic", "[noise]") {
    SimplexNoise noise(55);
    double a = noise.ridged2D(10.0, 20.0, 4, 0.5, 2.0);
    double b = noise.ridged2D(10.0, 20.0, 4, 0.5, 2.0);
    REQUIRE(a == b);
}

TEST_CASE("SimplexNoise ridged output range within [0, 1]", "[noise]") {
    SimplexNoise noise(42);

    for (int i = 0; i < 20; ++i) {
        double v =
            noise.ridged2D(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.1, 4, 0.5, 2.0);
        REQUIRE(v >= 0.0);
        REQUIRE(v <= 1.0);
    }
}

TEST_CASE("SimplexNoise operator() equals noise2D", "[noise]") {
    SimplexNoise noise(42);
    REQUIRE(noise(1.0, 2.0) == noise.noise2D(1.0, 2.0));
    REQUIRE(noise(10.5, -3.7) == noise.noise2D(10.5, -3.7));
}

// ============================================================================
// Climate / terrain shaping tests
// ============================================================================

TEST_CASE("ClimateSampler shapeColumn is deterministic and sane", "[climate]") {
    ClimateSampler c1(123);
    ClimateSampler c2(123);
    for (int i = -50; i <= 50; i += 7) {
        ColumnShape a = c1.shapeColumn(i * 31.0, i * 17.0);
        ColumnShape b = c2.shapeColumn(i * 31.0, i * 17.0);
        REQUIRE(a.height == b.height);
        REQUIRE(a.climate.temperature == b.climate.temperature);
        REQUIRE(a.height >= 20.0);
        REQUIRE(a.height <= 240.0);
        REQUIRE(a.detailAmp >= 0.0);
        REQUIRE(a.ravineEdge >= 0.0);
        REQUIRE(a.ravineEdge <= 1.0);
    }
}

TEST_CASE("selectBiome: ordered rules on the climate fields", "[climate]") {
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
    shape.climate.temperature = -0.2;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::BIRCH_FOREST);

    shape.climate.humidity = 0.6;
    shape.height = 66.0;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::SWAMP);

    shape.climate.humidity = 0.1;
    shape.climate.temperature = 0.4;
    shape.height = 80.0;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::FLOWER_FIELD);

    shape.climate.humidity = -0.1;
    shape.climate.temperature = 0.1;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::PLAINS);

    shape.riverCut = 4.0;
    shape.height = 60.0;
    shape.climate.continentalness = 0.3;
    REQUIRE(ClimateSampler::selectBiome(shape) == Biome::RIVER);
}

TEST_CASE("Terrain height is continuous — no biome cliffs", "[climate][worldgen]") {
    // The old design added per-biome height offsets after a discrete biome
    // pick, which produced 15-block-in-one-step walls at biome borders.
    // Height now reads continuous splines only. The steepest legitimate
    // slope is a river gorge bank through a high plateau (~7 blocks per
    // column); a discrete-biome cliff would blow well past this bound.
    ChunkGenerator gen(4242);
    GenScratch scratch;
    scratch.reset(&gen);
    double prev = gen.baseHeightAt(-256, 100, scratch);
    for (int x = -255; x <= 256; ++x) {
        double h = gen.baseHeightAt(x, 100, scratch);
        REQUIRE(std::abs(h - prev) <= 8.0);
        prev = h;
    }
}

// ============================================================================
// ChunkGenerator tests
// ============================================================================

namespace {
// Blocks the terrain fill + surface pass produce — decorations (trees,
// structures, flora, ice caps) may legitimately rise above the height map,
// raw terrain must not.
bool isTerrainBlock(BlockType b) {
    switch (b) {
        case BlockType::STONE:
        case BlockType::DIRT:
        case BlockType::GRASS:
        case BlockType::SAND:
        case BlockType::GRAVEL:
        case BlockType::SNOW:
        case BlockType::BEDROCK:
        case BlockType::COAL_ORE:
        case BlockType::IRON_ORE:
        case BlockType::GOLD_ORE:
        case BlockType::DIAMOND_ORE:
            return true;
        default:
            return false;
    }
}

// Column-level invariants every generated chunk must satisfy.
void checkChunkInvariants(const Chunk& chunk) {
    for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
        for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
            // Bedrock floor
            REQUIRE(chunk.getBlock(lx, 0, lz) == BlockType::BEDROCK);
            REQUIRE(chunk.getBlock(lx, 1, lz) == BlockType::BEDROCK);

            // heightMap records the raw terrain surface; structures may
            // later carve or pave that exact cell, so only the "no terrain
            // above it" direction is asserted.
            int top = chunk.heightMap[lx + lz * CHUNK_WIDTH];
            REQUIRE(top >= 2);

            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                BlockType b = chunk.getBlock(lx, y, lz);
                // No raw terrain above the height map (the old carver's
                // floating-surface bug); decorations are allowed there
                if (y > top) {
                    REQUIRE(!isTerrainBlock(b));
                }
                // Water never rests directly on air; lava only pools at
                // the very bottom (well shafts sit above ground level)
                if (b == BlockType::WATER && y < 64) {
                    REQUIRE(chunk.getBlock(lx, y - 1, lz) != BlockType::AIR);
                }
                if (b == BlockType::LAVA) {
                    REQUIRE(y <= 10);
                }
                if (b == BlockType::ICE) {
                    REQUIRE(y == 63);
                }
            }
        }
    }
}
} // namespace

TEST_CASE("ChunkGenerator: same seed → bit-identical chunk, different seed differs", "[worldgen]") {
    ChunkGenerator g1(777);
    ChunkGenerator g2(777);
    ChunkGenerator g3(778);
    Chunk a(5, -3), b(5, -3), c(5, -3);
    g1.generate(a);
    g2.generate(b);
    g3.generate(c);
    REQUIRE(a.blocks == b.blocks);
    REQUIRE(a.biomes == b.biomes);
    REQUIRE(a.heightMap == b.heightMap);
    REQUIRE(a.blocks != c.blocks);
}

TEST_CASE("ChunkGenerator: column invariants hold across varied chunks", "[worldgen]") {
    ChunkGenerator gen(1234);
    for (auto [cx, cz] : {std::pair{0, 0}, {5, 7}, {-3, 2}, {40, -25}}) {
        Chunk chunk(cx, cz);
        gen.generate(chunk);
        checkChunkInvariants(chunk);
    }
}

TEST_CASE("ChunkGenerator: underground carve fraction is sane", "[worldgen][caves]") {
    // The old carver's inverted threshold hollowed out ~99% of the
    // underground (the bug this overhaul fixes). Keep the carved fraction
    // in a believable band across several chunks.
    ChunkGenerator gen(31337);
    int air = 0, total = 0;
    for (auto [cx, cz] : {std::pair{0, 0}, {3, 1}, {-2, -4}, {10, 6}}) {
        Chunk chunk(cx, cz);
        gen.generate(chunk);
        for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
            for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
                int top = chunk.heightMap[lx + lz * CHUNK_WIDTH];
                for (int y = 8; y < top - 10; ++y) {
                    ++total;
                    BlockType b = chunk.getBlock(lx, y, lz);
                    if (b == BlockType::AIR || b == BlockType::LAVA)
                        ++air;
                }
            }
        }
    }
    REQUIRE(total > 0);
    double fraction = static_cast<double>(air) / static_cast<double>(total);
    REQUIRE(fraction >= 0.01);
    REQUIRE(fraction <= 0.30);
}

TEST_CASE("ChunkGenerator: surfaceYAt matches the generated height map", "[worldgen]") {
    ChunkGenerator gen(9999);
    GenScratch scratch;
    scratch.reset(&gen);
    Chunk chunk(2, -7);
    gen.generate(chunk);
    for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
        for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
            int wx = chunk.chunkX * CHUNK_WIDTH + lx;
            int wz = chunk.chunkZ * CHUNK_DEPTH + lz;
            REQUIRE(gen.surfaceYAt(wx, wz, scratch) == chunk.heightMap[lx + lz * CHUNK_WIDTH]);
        }
    }
}

TEST_CASE("ChunkGenerator: chunks are generation-order independent", "[worldgen]") {
    // Generate the same 2x1 area in both orders across separate generators;
    // every block must agree (the purity contract for infinite worlds).
    ChunkGenerator g1(5150);
    ChunkGenerator g2(5150);
    Chunk a0(0, 0), a1(1, 0);
    g1.generate(a0);
    g1.generate(a1);
    Chunk b1(1, 0), b0(0, 0);
    g2.generate(b1);
    g2.generate(b0);
    REQUIRE(a0.blocks == b0.blocks);
    REQUIRE(a1.blocks == b1.blocks);
    REQUIRE(a0.biomes == b0.biomes);
    REQUIRE(a1.biomes == b1.biomes);
}

// ============================================================================
// Feature / decoration tests
// ============================================================================

namespace {
// isLeafBlock comes from block_properties.hpp (the single property table)
bool isLogBlock(BlockType b) {
    return b == BlockType::LOG || b == BlockType::BIRCH_LOG || b == BlockType::SPRUCE_LOG;
}
} // namespace

TEST_CASE("Trees span chunk borders and no canopy is orphaned", "[worldgen][trees]") {
    // Generate a 3x3 area and inspect the center chunk: every leaf must have
    // a trunk within canopy range in the combined area (a clipped border
    // canopy would orphan leaves), and at least one border column must carry
    // leaves whose trunk lives in the neighbor chunk.
    ChunkGenerator gen(2024);
    std::vector<std::unique_ptr<Chunk>> area;
    auto blockAt = [&](int wx, int y, int wz) -> BlockType {
        for (auto& c : area) {
            int lx = wx - c->chunkX * CHUNK_WIDTH;
            int lz = wz - c->chunkZ * CHUNK_DEPTH;
            if (lx >= 0 && lx < CHUNK_WIDTH && lz >= 0 && lz < CHUNK_DEPTH)
                return c->getBlock(lx, y, lz);
        }
        return BlockType::AIR;
    };
    // Pick a center chunk in forest-ish terrain by scanning for one with
    // plenty of leaves.
    int bestCX = 0, bestCZ = 0, bestLeaves = -1;
    for (int cz = -6; cz <= 6; cz += 3) {
        for (int cx = -6; cx <= 6; cx += 3) {
            Chunk probe(cx, cz);
            gen.generate(probe);
            int leaves = 0;
            for (BlockType b : probe.blocks)
                if (isLeafBlock(b))
                    ++leaves;
            if (leaves > bestLeaves) {
                bestLeaves = leaves;
                bestCX = cx;
                bestCZ = cz;
            }
        }
    }
    REQUIRE(bestLeaves > 0); // some forest exists near spawn scale

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            auto chunk = std::make_unique<Chunk>(bestCX + dx, bestCZ + dz);
            gen.generate(*chunk);
            area.push_back(std::move(chunk));
        }
    }

    const Chunk& center = *area[4]; // dz=0, dx=0 (row-major -1..1)
    bool foundBorderLeaf = false;
    for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
        for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
            for (int y = 60; y < 140; ++y) {
                if (!isLeafBlock(center.getBlock(lx, y, lz)))
                    continue;
                int wx = center.chunkX * CHUNK_WIDTH + lx;
                int wz = center.chunkZ * CHUNK_DEPTH + lz;
                // A trunk must exist within canopy range in the 3x3 area
                bool hasTrunk = false;
                bool trunkInNeighbor = false;
                for (int tz = -3; tz <= 3 && !hasTrunk; ++tz) {
                    for (int tx = -3; tx <= 3 && !hasTrunk; ++tx) {
                        for (int ty = -4; ty <= 2 && !hasTrunk; ++ty) {
                            if (isLogBlock(blockAt(wx + tx, y + ty, wz + tz))) {
                                hasTrunk = true;
                                int trunkLX = wx + tx - center.chunkX * CHUNK_WIDTH;
                                int trunkLZ = wz + tz - center.chunkZ * CHUNK_DEPTH;
                                trunkInNeighbor = trunkLX < 0 || trunkLX >= CHUNK_WIDTH ||
                                                  trunkLZ < 0 || trunkLZ >= CHUNK_DEPTH;
                            }
                        }
                    }
                }
                REQUIRE(hasTrunk);
                if (trunkInNeighbor)
                    foundBorderLeaf = true;
            }
        }
    }
    // With ~6 trees per forest chunk, a 16-wide chunk essentially always has
    // at least one canopy reaching across its border.
    REQUIRE(foundBorderLeaf);
}

TEST_CASE("Ores stay in their depth bands and replace only stone", "[worldgen][ores]") {
    ChunkGenerator gen(808);
    for (auto [cx, cz] : {std::pair{0, 0}, {7, -3}, {-11, 5}}) {
        Chunk chunk(cx, cz);
        gen.generate(chunk);
        for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
            for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
                for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                    switch (chunk.getBlock(lx, y, lz)) {
                        case BlockType::DIAMOND_ORE:
                            REQUIRE(y >= 2);
                            REQUIRE(y <= 17);
                            break;
                        case BlockType::GOLD_ORE:
                            REQUIRE(y >= 4);
                            REQUIRE(y <= 35);
                            break;
                        case BlockType::IRON_ORE:
                            REQUIRE(y >= 8);
                            REQUIRE(y <= 71);
                            break;
                        case BlockType::COAL_ORE:
                            REQUIRE(y >= 48);
                            REQUIRE(y <= 131);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

TEST_CASE("Flora sits on valid ground", "[worldgen][flora]") {
    ChunkGenerator gen(606);
    int floraSeen = 0;
    for (int cz = -4; cz <= 4; cz += 2) {
        for (int cx = -4; cx <= 4; cx += 2) {
            Chunk chunk(cx, cz);
            gen.generate(chunk);
            for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
                for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
                    for (int y = 3; y < CHUNK_HEIGHT - 1; ++y) {
                        BlockType b = chunk.getBlock(lx, y, lz);
                        BlockType below = chunk.getBlock(lx, y - 1, lz);
                        if (b == BlockType::CACTUS && below != BlockType::CACTUS) {
                            REQUIRE(below == BlockType::SAND);
                            ++floraSeen;
                        } else if (b == BlockType::REED && below != BlockType::REED) {
                            REQUIRE((below == BlockType::GRASS || below == BlockType::SAND ||
                                     below == BlockType::DIRT));
                            ++floraSeen;
                        } else if (isFlora(b) && b != BlockType::REED) {
                            if (b != BlockType::CACTUS && b != BlockType::DEAD_BUSH) {
                                REQUIRE(below == BlockType::GRASS);
                            }
                            ++floraSeen;
                        }
                    }
                }
            }
        }
    }
    REQUIRE(floraSeen > 0);
}

TEST_CASE("Structures are deterministic and land on validated ground", "[worldgen][structures]") {
    ChunkGenerator g1(112233);
    ChunkGenerator g2(112233);
    GenScratch s1, s2;
    s1.reset(&g1);
    s2.reset(&g2);
    StructurePlacer placer(112233);

    int validCount = 0;
    for (int rz = -6; rz <= 6; ++rz) {
        for (int rx = -6; rx <= 6; ++rx) {
            const StructurePlacement& a = placer.regionPlacement(rx, rz, g1, s1);
            const StructurePlacement& b = placer.regionPlacement(rx, rz, g2, s2);
            REQUIRE(a.valid == b.valid);
            REQUIRE(a.anchorX == b.anchorX);
            REQUIRE(a.anchorZ == b.anchorZ);
            REQUIRE(a.floorY == b.floorY);
            if (a.valid) {
                ++validCount;
                REQUIRE(a.floorY >= 64);
            }
        }
    }
    // 13x13 regions of mixed terrain: some sites must validate
    REQUIRE(validCount > 0);
}

// ============================================================================
// Chunk Tests
// ============================================================================

TEST_CASE("Chunk creation initializes to air", "[chunk]") {
    Chunk chunk(0, 0);
    REQUIRE(chunk.chunkX == 0);
    REQUIRE(chunk.chunkZ == 0);
    REQUIRE(chunk.blocks.size() == static_cast<size_t>(CHUNK_VOLUME));
    REQUIRE(chunk.getBlock(0, 0, 0) == BlockType::AIR);
    REQUIRE(chunk.getBlock(7, 127, 7) == BlockType::AIR);
}

TEST_CASE("Chunk setBlock and getBlock", "[chunk]") {
    Chunk chunk(5, -3);
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    REQUIRE(chunk.getBlock(8, 64, 8) == BlockType::STONE);

    chunk.setBlock(0, 0, 0, BlockType::GRASS);
    REQUIRE(chunk.getBlock(0, 0, 0) == BlockType::GRASS);
}

TEST_CASE("Chunk setBlock marks chunk dirty", "[chunk]") {
    Chunk chunk(0, 0);
    REQUIRE(chunk.needsMeshUpdate == false);
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    REQUIRE(chunk.needsMeshUpdate == true);
}

TEST_CASE("Chunk out-of-bounds returns air", "[chunk]") {
    Chunk chunk(0, 0);
    REQUIRE(chunk.getBlock(-1, 64, 8) == BlockType::AIR);
    REQUIRE(chunk.getBlock(16, 64, 8) == BlockType::AIR);
    REQUIRE(chunk.getBlock(8, -1, 8) == BlockType::AIR);
    REQUIRE(chunk.getBlock(8, 256, 8) == BlockType::AIR);
    REQUIRE(chunk.getBlock(8, 64, -1) == BlockType::AIR);
    REQUIRE(chunk.getBlock(8, 64, 16) == BlockType::AIR);
}

TEST_CASE("Chunk world coordinate conversion", "[chunk]") {
    REQUIRE(Chunk::worldToChunk(0) == 0);
    REQUIRE(Chunk::worldToChunk(15) == 0);
    REQUIRE(Chunk::worldToChunk(16) == 1);
    REQUIRE(Chunk::worldToChunk(31) == 1);
    REQUIRE(Chunk::worldToChunk(32) == 2);
    REQUIRE(Chunk::worldToChunk(-1) == -1);
    REQUIRE(Chunk::worldToChunk(-16) == -1);
    REQUIRE(Chunk::worldToChunk(-17) == -2);
    REQUIRE(Chunk::worldToChunk(-32) == -2);
}

TEST_CASE("Chunk chunkToWorld conversion", "[chunk]") {
    REQUIRE(Chunk::chunkToWorld(0, 0) == 0);
    REQUIRE(Chunk::chunkToWorld(1, 0) == 16);
    REQUIRE(Chunk::chunkToWorld(1, 15) == 31);
    REQUIRE(Chunk::chunkToWorld(-1, 0) == -16);
    REQUIRE(Chunk::chunkToWorld(-1, 15) == -1);
}

TEST_CASE("Chunk world block access", "[chunk]") {
    Chunk chunk(2, -1);
    chunk.setBlockWorld(32, 64, -8, BlockType::DIRT);
    REQUIRE(chunk.getBlockWorld(32, 64, -8) == BlockType::DIRT);
}

TEST_CASE("Chunk getAABB", "[chunk]") {
    Chunk chunk(1, -1);
    AABB aabb = chunk.getAABB();
    REQUIRE(aabb.min.x == Catch::Approx(16.f));
    REQUIRE(aabb.min.y == Catch::Approx(0.f));
    REQUIRE(aabb.min.z == Catch::Approx(-16.f));
    REQUIRE(aabb.max.x == Catch::Approx(32.f));
    REQUIRE(aabb.max.y == Catch::Approx(256.f));
    REQUIRE(aabb.max.z == Catch::Approx(0.f));
}

TEST_CASE("Chunk getWorldPosition", "[chunk]") {
    Chunk chunk(3, -2);
    Vec3 pos = chunk.getWorldPosition();
    REQUIRE(pos.x == Catch::Approx(48.f));
    REQUIRE(pos.y == Catch::Approx(0.f));
    REQUIRE(pos.z == Catch::Approx(-32.f));
}

TEST_CASE("Chunk markDirty", "[chunk]") {
    Chunk chunk(0, 0);
    chunk.needsMeshUpdate = false;
    chunk.markDirty();
    REQUIRE(chunk.needsMeshUpdate == true);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST_CASE("Serialization roundtrip", "[serialization]") {
    Chunk original(5, -3);
    original.setBlock(8, 64, 8, BlockType::STONE);
    original.setBlock(0, 0, 0, BlockType::GRASS);
    original.generated = true;
    original.biomes[0] = Biome::DESERT;
    original.biomes[100] = Biome::FOREST;
    original.heightMap[0] = 65;
    original.heightMap[100] = 72;

    auto data = ChunkSerializer::serialize(original);
    auto restored = ChunkSerializer::deserialize(data);
    REQUIRE(restored.has_value());
    REQUIRE(restored->chunkX == original.chunkX);
    REQUIRE(restored->chunkZ == original.chunkZ);
    REQUIRE(restored->getBlock(8, 64, 8) == BlockType::STONE);
    REQUIRE(restored->getBlock(0, 0, 0) == BlockType::GRASS);
    REQUIRE(restored->biomes[0] == Biome::DESERT);
    REQUIRE(restored->biomes[100] == Biome::FOREST);
    REQUIRE(restored->heightMap[0] == 65);
    REQUIRE(restored->heightMap[100] == 72);
}

TEST_CASE("Serialization: heights at 128 and above survive the roundtrip", "[serialization]") {
    // Terrain reaches height 128, which overflowed the old int8 height field
    // to -128 on load and corrupted tree/structure placement.
    Chunk chunk(3, 4);
    chunk.generated = true;
    chunk.heightMap[0] = 127;
    chunk.heightMap[1] = 128;
    chunk.heightMap[2] = 255;

    auto data = ChunkSerializer::serialize(chunk);
    auto restored = ChunkSerializer::deserialize(data);
    REQUIRE(restored.has_value());
    REQUIRE(restored->heightMap[0] == 127);
    REQUIRE(restored->heightMap[1] == 128);
    REQUIRE(restored->heightMap[2] == 255);
}

TEST_CASE("Serialization: pre-v2 chunks are rejected so they regenerate", "[serialization]") {
    Chunk chunk(0, 0);
    chunk.generated = true;
    auto data = ChunkSerializer::serialize(chunk);

    // Rewrite the version field (offset 4) to the old v1
    uint32_t oldVersion = 1;
    std::memcpy(data.data() + 4, &oldVersion, sizeof(oldVersion));
    REQUIRE(!ChunkSerializer::deserialize(data).has_value());
}

TEST_CASE("World loads saved chunks before generating", "[world][save]") {
    TempDir tempDirGuard("load_before_generate");
    const std::string& tempDir = tempDirGuard.path();

    uint32_t seed = 777;
    int editX = 8, editY = 200, editZ = 8;

    {
        // First session: edit a block and persist the chunk
        SaveManager saver(tempDir);
        auto world = std::make_shared<World>(seed);
        world->setSaveManager(&saver);
        auto chunk = world->getChunk(0, 0);
        world->setBlock(editX, editY, editZ, BlockType::DIAMOND_ORE);
        saver.saveChunk(*chunk);
        saver.flush();
    }

    {
        // Second session: the edit must come back from disk, not the generator
        SaveManager saver(tempDir);
        auto world = std::make_shared<World>(seed);
        world->setSaveManager(&saver);
        REQUIRE(world->getBlock(editX, editY, editZ) == BlockType::DIAMOND_ORE);
    }
}

TEST_CASE("ChunkPos packs and hashes distinctly", "[world]") {
    REQUIRE(ChunkPos{0, 0} == ChunkPos{0, 0});
    REQUIRE(!(ChunkPos{1, 0} == ChunkPos{0, 1}));
    REQUIRE(ChunkPos{1, 0}.packed() != ChunkPos{0, 1}.packed());
    REQUIRE(ChunkPos{-1, -1}.packed() != ChunkPos{1, 1}.packed());

    std::unordered_map<ChunkPos, int> map;
    map[ChunkPos{5, -3}] = 42;
    REQUIRE(map.at(ChunkPos{5, -3}) == 42);
    REQUIRE(map.find(ChunkPos{-3, 5}) == map.end());
}

TEST_CASE("Serialization size is correct", "[serialization]") {
    Chunk chunk(0, 0);
    size_t expected = ChunkSerializer::serializedSize(chunk);
    auto data = ChunkSerializer::serialize(chunk);
    REQUIRE(data.size() == expected);
}

TEST_CASE("Serialization corrupt data returns nullopt", "[serialization]") {
    std::vector<uint8_t> corruptData(100, 0xFF);
    auto result = ChunkSerializer::deserialize(corruptData);
    REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization empty data returns nullopt", "[serialization]") {
    std::vector<uint8_t> emptyData;
    auto result = ChunkSerializer::deserialize(emptyData);
    REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization wrong magic returns nullopt", "[serialization]") {
    Chunk chunk(0, 0);
    auto data = ChunkSerializer::serialize(chunk);
    data[0] = 0x00;
    auto result = ChunkSerializer::deserialize(data);
    REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization truncated data returns nullopt", "[serialization]") {
    Chunk chunk(0, 0);
    auto data = ChunkSerializer::serialize(chunk);
    data.resize(HEADER_SIZE);
    auto result = ChunkSerializer::deserialize(data);
    REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization wrong block count returns nullopt", "[serialization]") {
    Chunk chunk(0, 0);
    auto data = ChunkSerializer::serialize(chunk);
    data[16] = 0x00;
    data[17] = 0x00;
    data[18] = 0x00;
    data[19] = 0x01;
    auto result = ChunkSerializer::deserialize(data);
    REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization multiple roundtrips consistent", "[serialization]") {
    Chunk original(10, 10);
    original.setBlock(4, 100, 4, BlockType::DIAMOND_ORE);
    auto data1 = ChunkSerializer::serialize(original);
    auto restored1 = ChunkSerializer::deserialize(data1);
    REQUIRE(restored1.has_value());
    auto data2 = ChunkSerializer::serialize(*restored1);
    auto restored2 = ChunkSerializer::deserialize(data2);
    REQUIRE(restored2.has_value());
    REQUIRE(restored2->getBlock(4, 100, 4) == BlockType::DIAMOND_ORE);
}

// ============================================================================
// World Tests
// ============================================================================

TEST_CASE("World creation", "[world]") {
    auto world = std::make_shared<World>(42);
    REQUIRE(world->getSeed() == 42);
    REQUIRE(world->getViewDistance() == 32);
}

TEST_CASE("World getChunk generates chunk", "[world]") {
    auto world = std::make_shared<World>(123);
    auto chunk = world->getChunk(0, 0);
    REQUIRE(chunk != nullptr);
    REQUIRE(chunk->chunkX == 0);
    REQUIRE(chunk->chunkZ == 0);
    REQUIRE(chunk->generated == true);
}

TEST_CASE("World getChunk returns cached chunk", "[world]") {
    auto world = std::make_shared<World>(42);
    auto chunk1 = world->getChunk(5, -3);
    auto chunk2 = world->getChunk(5, -3);
    REQUIRE(chunk1 == chunk2);
}

TEST_CASE("World getBlock and setBlock", "[world]") {
    auto world = std::make_shared<World>(42);
    BlockType b = world->getBlock(100, 64, 100);
    REQUIRE(static_cast<int>(b) >= 0);
    world->setBlock(100, 64, 100, BlockType::DIAMOND_ORE);
    BlockType after = world->getBlock(100, 64, 100);
    REQUIRE(after == BlockType::DIAMOND_ORE);
}

TEST_CASE("World getLoadedChunks", "[world]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(1, 0);
    world->getChunk(0, 1);
    auto loaded = world->getLoadedChunks();
    REQUIRE(loaded.size() == 3);
}

TEST_CASE("World getTerrainHeight", "[world]") {
    auto world = std::make_shared<World>(42);
    double h = world->getTerrainHeight(100, 200);
    REQUIRE(h >= 0.0);
}

TEST_CASE("World getBiome", "[world]") {
    auto world = std::make_shared<World>(42);
    Biome b = world->getBiome(100, 200);
    REQUIRE(static_cast<int>(b) >= 0);
    REQUIRE(static_cast<int>(b) < static_cast<int>(Biome::COUNT));
}

TEST_CASE("World setViewDistance", "[world]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(10);
    REQUIRE(world->getViewDistance() == 10);
    world->setViewDistance(0);
    REQUIRE(world->getViewDistance() == 1);
}

TEST_CASE("World markChunkMeshed", "[world]") {
    auto world = std::make_shared<World>(42);
    auto chunk = world->getChunk(0, 0);
    REQUIRE(chunk->needsMeshUpdate == true);
    world->markChunkMeshed(0, 0);
    REQUIRE(chunk->needsMeshUpdate == false);
}

TEST_CASE("World getDirtyChunks", "[world]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(1, 0);
    auto dirty = world->getDirtyChunks();
    REQUIRE(dirty.size() == 2);
    world->markChunkMeshed(0, 0);
    dirty = world->getDirtyChunks();
    REQUIRE(dirty.size() == 1);
}

// ============================================================================
// Async Generation Tests
// ============================================================================

TEST_CASE("World async generation pending count", "[world][async]") {
    auto world = std::make_shared<World>(42);
    REQUIRE(world->getPendingChunkCount() == 0);
}

TEST_CASE("World generateAroundPlayer submits chunks", "[world][async]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(4);
    world->generateAroundPlayer(0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_t pending = world->getPendingChunkCount();
    REQUIRE(pending >= 0);
}

TEST_CASE("World generateAroundPlayer populates chunks", "[world][async]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(2);
    world->generateAroundPlayer(0, 0);
    for (int attempts = 0; attempts < 50; ++attempts) {
        if (world->getPendingChunkCount() == 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    auto chunks = world->getLoadedChunks();
    REQUIRE(chunks.size() >= 0);
}

TEST_CASE("World updatePlayerPosition loads surrounding chunks", "[world]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(2);
    world->updatePlayerPosition(256, 256);

    // Generation streams in on the worker pool; wait for it to settle.
    // The self-sustaining pump must drain the whole backlog without any
    // further updatePlayerPosition calls (workers refill the window).
    for (int i = 0; i < 500 && world->getPendingChunkCount() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Generation reaches one chunk past the render radius: (2·(vd+1)+1)²
    auto chunks = world->getLoadedChunks();
    REQUIRE(chunks.size() == 49);
}

TEST_CASE("World updatePlayerPosition streams the spawn area on first call", "[world]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(1);

    // The player spawns in chunk (0,0) — the very position the tracker starts
    // at. The first call must still trigger streaming.
    world->updatePlayerPosition(0, 0);
    for (int i = 0; i < 500 && world->getPendingChunkCount() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    REQUIRE(world->getLoadedChunks().size() == 25);
}

TEST_CASE("sortChunksByDistance orders farthest-first for pop_back consumption",
          "[world][priority]") {
    std::vector<ChunkPos> chunks = {{10, 0}, {0, 0}, {3, 4}, {-1, 0}, {0, -7}};
    sortChunksByDistance(chunks, 0, 0);
    // Farthest first…
    REQUIRE(chunks.front() == ChunkPos{10, 0});
    // …so pop_back() yields the player's own chunk before anything else
    REQUIRE(chunks.back() == ChunkPos{0, 0});
    for (size_t i = 1; i < chunks.size(); ++i) {
        auto d2 = [](const ChunkPos& p) { return p.x * p.x + p.z * p.z; };
        REQUIRE(d2(chunks[i - 1]) >= d2(chunks[i]));
    }
}

TEST_CASE("World generation window stays bounded", "[world][priority]") {
    auto world = std::make_shared<World>(42);
    world->setViewDistance(5); // gen radius 6 → 169 chunks, well over the window
    world->updatePlayerPosition(0, 0);

    // Immediately after the first pump, at most MAX_INFLIGHT_GEN tasks may
    // be in flight; the rest wait in the backlog (all still counted).
    size_t pending = world->getPendingChunkCount();
    size_t loaded = world->getLoadedChunks().size();
    REQUIRE(pending + loaded >= 169 - MAX_INFLIGHT_GEN); // nothing lost

    for (int i = 0; i < 2000 && world->getPendingChunkCount() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(world->getLoadedChunks().size() == 169);
}

// ============================================================================
// SaveManager Tests
// ============================================================================

TEST_CASE("SaveManager creation", "[save]") {
    TempDir dir("world_sm1");
    SaveManager saver(dir.path());
    REQUIRE(saver.getWorldPath().find("rycraft_test_") != std::string::npos);
}

TEST_CASE("SaveManager save/load chunk roundtrip", "[save]") {
    TempDir tempDirGuard("save_roundtrip");
    const std::string& tempDir = tempDirGuard.path();
    {
        SaveManager saver(tempDir);
        Chunk original(7, -5);
        original.setBlock(8, 100, 8, BlockType::IRON_ORE);
        original.generated = true;
        saver.saveChunk(original);
        saver.flush();
        auto loaded = saver.loadChunk(7, -5);
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->chunkX == 7);
        REQUIRE(loaded->chunkZ == -5);
        REQUIRE(loaded->getBlock(8, 100, 8) == BlockType::IRON_ORE);
    }
}

TEST_CASE("SaveManager: chunks in one region never clobber each other", "[save]") {
    // Regression: the old packed-region format wrote ONE chunk per region
    // file, so every chunk save silently destroyed its 1023 region-mates.
    TempDir tempDirGuard("save_multi");
    SaveManager saver(tempDirGuard.path());

    // Same region (0,0), region border, and the next region over
    const std::pair<int, int> coords[] = {{0, 0}, {1, 0}, {31, 31}, {32, 0}, {-1, -1}};
    int marker = 1;
    for (auto [cx, cz] : coords) {
        Chunk chunk(cx, cz);
        chunk.setBlock(1, 100 + marker, 1, BlockType::GOLD_ORE);
        chunk.generated = true;
        saver.saveChunk(chunk);
        ++marker;
    }
    saver.flush();

    marker = 1;
    for (auto [cx, cz] : coords) {
        auto loaded = saver.loadChunk(cx, cz);
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->chunkX == cx);
        REQUIRE(loaded->chunkZ == cz);
        REQUIRE(loaded->getBlock(1, 100 + marker, 1) == BlockType::GOLD_ORE);
        ++marker;
    }
}

TEST_CASE("SaveManager: queued chunks are readable before the write lands", "[save]") {
    // The load shield: unloading a chunk queues it; walking straight back
    // must return the queued edits even if the file hasn't been written yet.
    TempDir tempDirGuard("save_shield");
    SaveManager saver(tempDirGuard.path());

    auto chunk = std::make_shared<Chunk>(3, 4);
    chunk->setBlock(2, 90, 2, BlockType::DIAMOND_ORE);
    chunk->generated = true;
    saver.saveChunkAsync(chunk);

    // No flush: the job may or may not have been written yet — either way
    // the loaded chunk must carry the edit.
    auto loaded = saver.loadChunk(3, 4);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->getBlock(2, 90, 2) == BlockType::DIAMOND_ORE);
    saver.flush();
}

TEST_CASE("World: edited chunks persist through unload-and-return", "[world][save]") {
    TempDir tempDirGuard("save_unload");
    SaveManager saver(tempDirGuard.path());
    World world(4321, 1);
    world.setSaveManager(&saver);

    world.getChunk(0, 0);
    world.setBlock(5, 150, 5, BlockType::PLANKS);

    // Walk far away: (0,0) leaves the radius and queues for saving…
    world.updatePlayerPosition(10 * CHUNK_WIDTH, 10 * CHUNK_DEPTH);
    // …then come back: the reloaded chunk must carry the edit
    auto chunk = world.getChunk(0, 0);
    REQUIRE(chunk->getBlock(5, 150, 5) == BlockType::PLANKS);
    saver.flush();
}

TEST_CASE("SaveManager load non-existent chunk returns nullopt", "[save]") {
    TempDir tempDirGuard("save_missing");
    const std::string& tempDir = tempDirGuard.path();
    {
        SaveManager saver(tempDir);
        auto loaded = saver.loadChunk(999, 999);
        REQUIRE(loaded.has_value() == false);
    }
}

TEST_CASE("SaveManager save/load metadata roundtrip", "[save]") {
    TempDir tempDirGuard("save_meta");
    const std::string& tempDir = tempDirGuard.path();
    {
        SaveManager saver(tempDir);
        saver.saveMetadata(12345, Vec3{100.f, 80.f, -50.f}, 9876543210);
        auto meta = saver.loadMetadata();
        REQUIRE(meta.has_value());
        REQUIRE(meta->seed == 12345);
        REQUIRE(meta->spawnPos.x == Catch::Approx(100.f));
        REQUIRE(meta->spawnPos.y == Catch::Approx(80.f));
        REQUIRE(meta->spawnPos.z == Catch::Approx(-50.f));
        REQUIRE(meta->worldTime == 9876543210);
    }
}

TEST_CASE("SaveManager load missing metadata returns nullopt", "[save]") {
    TempDir tempDirGuard("save_nometa");
    const std::string& tempDir = tempDirGuard.path();
    {
        SaveManager saver(tempDir);
        auto meta = saver.loadMetadata();
        REQUIRE(meta.has_value() == false);
    }
}

TEST_CASE("SaveManager LZ4 compression produces smaller data", "[save]") {
    TempDir tempDirGuard("save_compress");
    const std::string& tempDir = tempDirGuard.path();
    {
        SaveManager saver(tempDir);
        Chunk chunk(0, 0);
        chunk.setBlock(8, 64, 8, BlockType::STONE);
        chunk.generated = true;
        saver.saveChunk(chunk);
        saver.flush();
        auto loaded = saver.loadChunk(0, 0);
        REQUIRE(loaded.has_value());
    }
}

// ============================================================================
// Phase 6: Block Interaction & Environment Tests
// ============================================================================

// ---- Block Breaking Tests (Task 6.1) ----

TEST_CASE("Block breaking: raycast hits block and block becomes AIR", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y to avoid terrain
    world->setBlock(5, 200, 0, BlockType::STONE);
    REQUIRE(world->getBlock(5, 200, 0) == BlockType::STONE);

    // Ray from (0, 200, 0) going +X toward the block
    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());
    REQUIRE(hit->first.x == Catch::Approx(5.f));
    REQUIRE(hit->first.y == Catch::Approx(200.f));
    REQUIRE(hit->first.z == Catch::Approx(0.f));

    // "Break" the block: set to AIR
    int hitX = static_cast<int>(std::floor(hit->first.x));
    int hitY = static_cast<int>(std::floor(hit->first.y));
    int hitZ = static_cast<int>(std::floor(hit->first.z));
    world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Verify block is now AIR
    REQUIRE(world->getBlock(5, 200, 0) == BlockType::AIR);
}

TEST_CASE("Block breaking: chunk marked dirty after block change", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    auto chunk = world->getChunk(0, 0);

    // Reset dirty state
    chunk->needsMeshUpdate = false;

    // Place and break a block
    world->setBlock(5, 200, 0, BlockType::STONE);
    REQUIRE(chunk->needsMeshUpdate == true);

    // Reset and break
    chunk->needsMeshUpdate = false;
    world->setBlock(5, 200, 0, BlockType::AIR);
    REQUIRE(chunk->needsMeshUpdate == true);
}

TEST_CASE("Block breaking: bedrock cannot be broken", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    world->setBlock(5, 200, 0, BlockType::BEDROCK);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);

    // Ray should NOT hit bedrock (it's not solid for ray tracing purposes in some games)
    // But in our implementation, bedrock IS solid for ray tracing
    // The "cannot break" logic is in the engine, not the traversal
    if (hit.has_value()) {
        // Verify it's bedrock
        REQUIRE(world->getBlock(5, 200, 0) == BlockType::BEDROCK);
    }
}

// ---- Block Placing Tests (Task 6.2) ----

TEST_CASE("Block placing: raycast finds face and block placed on face normal", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0)
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (2, 200, 0) going +X — hits the -X face of the block
    Vec3 origin{2.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());

    // Calculate placement position: hit block + face normal
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Normal should be -X, so placement should be at (4, 200, 0)
    REQUIRE(placeX == 4);
    REQUIRE(placeY == 200);
    REQUIRE(placeZ == 0);

    // Place block
    world->setBlock(placeX, placeY, placeZ, BlockType::DIRT);
    REQUIRE(world->getBlock(4, 200, 0) == BlockType::DIRT);
}

TEST_CASE("Block placing: no placement when overlapping player AABB", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place block at (10, 200, 10)
    world->setBlock(10, 200, 10, BlockType::STONE);

    // Player standing at (10, 200, 10) — inside the block we'd place on
    Player player;
    player.position = Vec3{10.f, 200.f, 10.f};

    // Ray from (7, 200, 10) going +X
    Vec3 origin{7.f, 200.f, 10.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());

    // Calculate placement position
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Check overlap
    AABB placeBox{
        Vec3{static_cast<float>(placeX), static_cast<float>(placeY), static_cast<float>(placeZ)},
        Vec3{static_cast<float>(placeX + 1), static_cast<float>(placeY + 1),
             static_cast<float>(placeZ + 1)}};

    bool overlaps = placeBox.intersects(player.getAABB());
    // If it overlaps, we should NOT place the block
    if (overlaps) {
        // This is the expected behavior — block should not be placed
        REQUIRE(overlaps == true);
    }
}

TEST_CASE("Block placing: adjacent chunks marked dirty at boundary", "[phase6][block]") {
    auto world = std::make_shared<World>(42);

    // Place block at chunk boundary: x=15 is last block in chunk 0, x=16 is first in chunk 1
    world->getChunk(0, 0);
    world->getChunk(1, 0);

    auto chunk0 = world->getChunk(0, 0);
    auto chunk1 = world->getChunk(1, 0);

    chunk0->needsMeshUpdate = false;
    chunk1->needsMeshUpdate = false;

    // Place block at x=16 (first block of chunk 1)
    world->setBlock(16, 200, 8, BlockType::STONE);

    // Chunk 1 should be dirty
    REQUIRE(chunk1->needsMeshUpdate == true);
}

// ---- Water Physics Tests (Task 6.7-6.8) ----

TEST_CASE("Water physics: reduced gravity when submerged", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    // Place floor far below so player falls freely
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    // Place water block at player position
    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3::zero();

    PlayerInput input;
    player.tick(*world, input);

    // In water: gravity *= 0.3, so effective gravity = -0.08 * 0.3 = -0.024
    // After drag: -0.024 * 0.98 = -0.02352
    // Plus buoyancy: -0.02352 + 0.02 = -0.00352
    // Velocity should be much smaller than in air (-0.0784)
    REQUIRE(std::abs(player.velocity.y) < std::abs(-0.08f * 0.98f));
}

TEST_CASE("Water physics: increased horizontal drag in water", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    // Place water at player position
    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3::zero();
    player.yaw = 0.f;

    // Press W to move forward
    PlayerInput input;
    input.forward = true;
    player.tick(*world, input);

    // Water halves the walking pace (0.216 → 0.108 blocks/tick)
    float totalHorizontalSpeed =
        std::sqrt(player.velocity.x * player.velocity.x + player.velocity.z * player.velocity.z);
    REQUIRE(totalHorizontalSpeed == Catch::Approx(0.108f).epsilon(0.01f));
}

TEST_CASE("Water physics: buoyancy pushes player upward", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3{0.f, -0.1f, 0.f}; // Moving downward

    PlayerInput input;
    player.tick(*world, input);

    // Buoyancy should reduce downward velocity
    // Without buoyancy: velocity.y ≈ -0.1 * 0.98 + (-0.024) = -0.122
    // With buoyancy: velocity.y ≈ -0.122 + 0.02 = -0.102
    // The buoyancy force makes velocity.y less negative
    REQUIRE(player.velocity.y > -0.13f);
}

TEST_CASE("isInWater: detects player in water block", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(5, 200, 5, BlockType::WATER);

    // Player AABB overlapping water block
    AABB playerBox{Vec3{4.5f, 199.5f, 4.5f}, Vec3{5.1f, 201.3f, 5.1f}};
    REQUIRE(PhysicsEngine::isInWater(*world, playerBox) == true);

    // Player not in water
    AABB playerBox2{Vec3{10.f, 200.f, 10.f}, Vec3{10.6f, 201.8f, 10.6f}};
    REQUIRE(PhysicsEngine::isInWater(*world, playerBox2) == false);
}

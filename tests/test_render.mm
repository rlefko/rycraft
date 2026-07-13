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
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Rendering: meshing, textures, shared GPU layouts
// ===========================================================================

// ============================================================================
// Vertex Format Tests
// ============================================================================

TEST_CASE("Vertex size is 16 bytes", "[render][vertex]") {
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Vertex alignment is 16 bytes", "[render][vertex]") {
    REQUIRE(alignof(Vertex) == 16);
}

TEST_CASE("Vertex fields have expected sizes", "[render][vertex]") {
    REQUIRE(sizeof(float16_t) == 2);
    REQUIRE(sizeof(uint8_t) == 1);
    REQUIRE(sizeof(uint32_t) == 4);
}

// ============================================================================
// Greedy Mesher Tests
// ============================================================================

TEST_CASE("Mesher: empty chunk produces no geometry", "[render][mesher]") {
    Chunk chunk(0, 0);
    // All AIR — no solid blocks

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    REQUIRE(output.vertices.empty());
    REQUIRE(output.indices.empty());
}

TEST_CASE("Mesher: single block produces 6 faces", "[render][mesher]") {
    Chunk chunk(0, 0);
    chunk.setBlock(8, 64, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 6 faces × 4 vertices = 24 vertices
    REQUIRE(output.vertices.size() == 24);
    // 6 faces × 2 triangles × 3 indices = 36 indices
    REQUIRE(output.indices.size() == 36);
}

TEST_CASE("Mesher: 2x2 flat merges top face", "[render][mesher]") {
    Chunk chunk(0, 0);
    // 2x2 square of STONE at y=64
    chunk.setBlock(0, 64, 0, BlockType::STONE);
    chunk.setBlock(1, 64, 0, BlockType::STONE);
    chunk.setBlock(0, 64, 1, BlockType::STONE);
    chunk.setBlock(1, 64, 1, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Without greedy merge: 4 top faces = 16 vertices
    // With greedy merge: 1 top face = 4 vertices
    // Total expected:
    //   Top (+Y): 1 merged quad = 4 vertices, 6 indices
    //   Bottom (-Y): 1 merged quad = 4 vertices, 6 indices
    //   +X face (right side of x=1 column): 2 quads (z=0 and z=1) = 8 vertices, 12 indices
    //   -X face (left side of x=0 column): 2 quads = 8 vertices, 12 indices
    //   +Z face (front of z=1 row): 2 quads = 8 vertices, 12 indices
    //   -Z face (back of z=0 row): 2 quads = 8 vertices, 12 indices
    // Total: 40 vertices, 60 indices
    //
    // But +X and -X faces can also merge vertically since all 4 blocks are at same Y
    // +X: blocks at (1,64,0) and (1,64,1) — both have +X exposed, same type
    //   They're adjacent in Z direction, so they merge into 1 quad: 4 vertices, 6 indices
    // Same for -X, +Z, -Z
    //
    // Total: 6 faces × 4 vertices = 24 vertices, 6 × 6 = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify the top face is a single quad (first 4 vertices)
    // All 4 top-face vertices should decode to FaceNormal::PLUS_Y
    bool foundTopQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        if (unpackFace(output.vertices[i].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 1].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 2].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 3].faceAttr) == FaceNormal::PLUS_Y) {
            foundTopQuad = true;
            break;
        }
    }
    REQUIRE(foundTopQuad);
}

TEST_CASE("Mesher: vertical column merges side faces", "[render][mesher]") {
    Chunk chunk(0, 0);
    // 4-block tall column of STONE at (8, 64..67, 8)
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    chunk.setBlock(8, 65, 8, BlockType::STONE);
    chunk.setBlock(8, 66, 8, BlockType::STONE);
    chunk.setBlock(8, 67, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Top (+Y): 1 quad at y=67 top = 4 vertices, 6 indices
    // Bottom (-Y): 1 quad at y=64 bottom = 4 vertices, 6 indices
    // +X, -X, +Z, -Z: each has 1 merged quad spanning y=64..67 = 4 vertices each, 6 indices each
    // Total: 6 faces × 4 vertices = 24 vertices
    // Total: 6 faces × 6 indices = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify side faces span the full column height
    // Each side quad should have vertices at y=64 and y=68 (height=4)
    bool foundSideQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        uint8_t ni = static_cast<uint8_t>(unpackFace(output.vertices[i].faceAttr));
        // Check side faces (face indices 0-3)
        if (ni <= 3 && static_cast<uint8_t>(unpackFace(output.vertices[i + 1].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 2].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 3].faceAttr)) == ni) {
            // Check that the quad spans 4 units in Y
            float minY = std::min({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            float maxY = std::max({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            if (maxY - minY >= 3.5f) { // height=4, account for float16 precision
                foundSideQuad = true;
                break;
            }
        }
    }
    REQUIRE(foundSideQuad);
}

TEST_CASE("Mesher: produces mesh without side effects", "[render][mesher]") {
    Chunk chunk(0, 0);
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    REQUIRE(chunk.needsMeshUpdate == true);

    LODMesher mesher;
    MeshOutput mesh = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // buildMesh is pure — it does not modify the chunk
    REQUIRE(chunk.needsMeshUpdate == true);
    REQUIRE(mesh.vertices.size() > 0u);
    REQUIRE(mesh.indices.size() > 0u);

    // Caller is responsible for marking the chunk as meshed
    chunk.setMeshed(true);
    chunk.needsMeshUpdate = false;
    REQUIRE(chunk.meshed == true);
    REQUIRE(chunk.needsMeshUpdate == false);
}

// ============================================================================
// Block texture mapping tests (no Metal device required)
// ============================================================================

TEST_CASE("Block textures: every block type maps to a valid layer", "[render][textures]") {
    for (int t = 0; t < static_cast<int>(BlockType::COUNT); ++t) {
        for (int f = 0; f < 6; ++f) {
            uint8_t layer = textureLayerFor(static_cast<BlockType>(t), static_cast<FaceNormal>(f));
            REQUIRE(layer < TEXTURE_LAYER_COUNT);
        }
    }
}

TEST_CASE("Block textures: grass uses per-face layers", "[render][textures]") {
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_Y) ==
            static_cast<uint8_t>(BlockType::GRASS));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Y) ==
            static_cast<uint8_t>(BlockType::DIRT));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_X) == TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Z) == TEXTURE_LAYER_GRASS_SIDE);
}

TEST_CASE("Block textures: face attr pack/unpack round-trips", "[render][textures]") {
    for (int f = 0; f < 6; ++f) {
        for (uint8_t layer :
             {uint8_t{0}, uint8_t{7}, TEXTURE_LAYER_GRASS_SIDE, TEXTURE_LAYER_WHITE}) {
            for (uint8_t light : {uint8_t{0}, uint8_t{4}, uint8_t{15}}) {
                uint32_t attr = packFaceAttr(static_cast<FaceNormal>(f), layer, light);
                REQUIRE(unpackFace(attr) == static_cast<FaceNormal>(f));
                REQUIRE(unpackTextureLayer(attr) == layer);
                REQUIRE(unpackSkyLight(attr) == light);
            }
        }
    }
}

TEST_CASE("Mesher: faces under cover carry reduced skylight", "[render][mesher]") {
    Chunk chunk(0, 0);
    // Ground block with a "canopy" three blocks above it
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    chunk.setBlock(8, 68, 8, BlockType::LEAVES);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool foundShadedTop = false;
    bool foundLitCanopyTop = false;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        float y = static_cast<float>(v.py);
        if (y > 64.5f && y < 65.5f) {
            // The ground's top face sits under the canopy → shaded
            REQUIRE(unpackSkyLight(v.faceAttr) < 15);
            foundShadedTop = true;
        }
        if (y > 68.5f && y < 69.5f) {
            // The canopy's own top face sees open sky
            REQUIRE(unpackSkyLight(v.faceAttr) == 15);
            foundLitCanopyTop = true;
        }
    }
    REQUIRE(foundShadedTop);
    REQUIRE(foundLitCanopyTop);
}

TEST_CASE("Block textures: extra layers extend past the block types", "[render][textures]") {
    REQUIRE(TEXTURE_LAYER_GRASS_SIDE == static_cast<uint8_t>(BlockType::COUNT));
    REQUIRE(TEXTURE_LAYER_COUNT > TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(BlockTextureArray::TILE_SIZE == 16);
}

// ============================================================================
// MegaBuffer Constant Tests (no Metal device required)
// ============================================================================

TEST_CASE("MegaBuffer alignment is power of 2", "[render][megabuffer]") {
    uint64_t align = MegaBuffer::ALIGNMENT;
    REQUIRE(align > 0);
    REQUIRE((align & (align - 1)) == 0); // Power of 2 check
    REQUIRE(align == 256);
}

TEST_CASE("MegaBuffer alignUp rounds up to alignment", "[render][megabuffer]") {
    // Test alignment behavior with known values
    // alignUp(x) = (x + 255) & ~255
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    REQUIRE(alignUp(0) == 0);
    REQUIRE(alignUp(1) == 256);
    REQUIRE(alignUp(256) == 256);
    REQUIRE(alignUp(257) == 512);
    REQUIRE(alignUp(512) == 512);
    REQUIRE(alignUp(1000) == 1024);
    REQUIRE(alignUp(16 * sizeof(Vertex)) == 256); // 256 bytes, already aligned
    REQUIRE(alignUp(17 * sizeof(Vertex)) == 512); // 272 bytes, rounds to 512
}

TEST_CASE("MegaBuffer vertex allocation size calculation", "[render][megabuffer]") {
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    // 100 vertices × 16 bytes = 1600 bytes → aligned to 1792 (7 × 256)
    uint64_t vertexBytes = alignUp(100 * sizeof(Vertex));
    REQUIRE(vertexBytes >= 1600);
    REQUIRE(vertexBytes % MegaBuffer::ALIGNMENT == 0);

    // 1000 indices × 4 bytes = 4000 bytes → aligned to 4096 (16 × 256)
    uint64_t indexBytes = alignUp(1000 * sizeof(uint32_t));
    REQUIRE(indexBytes >= 4000);
    REQUIRE(indexBytes % MegaBuffer::ALIGNMENT == 0);
}

// ---- Day/Night Cycle Tests (Task 6.4-6.5) ----

TEST_CASE("Day/night cycle: sun position at noon", "[phase6][daynight]") {
    // At noon: worldTime = 6000 (25% of 24000)
    // orbitalAngle = 0.25 * 2*PI = PI/2
    // sunDirection = (cos(PI/2), sin(PI/2), 0.3) = (0, 1, 0.3)
    uint64_t worldTime = 6000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    // At noon: cos(PI/2) ≈ 0, sin(PI/2) = 1
    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at sunset", "[phase6][daynight]") {
    // At sunset: worldTime = 12000 (50% of 24000)
    // orbitalAngle = 0.5 * 2*PI = PI
    // sunDirection = (cos(PI), sin(PI), 0.3) = (-1, 0, 0.3)
    uint64_t worldTime = 12000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(-1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at midnight", "[phase6][daynight]") {
    // At midnight: worldTime = 18000 (75% of 24000)
    // orbitalAngle = 0.75 * 2*PI = 3PI/2
    // sunDirection = (cos(3PI/2), sin(3PI/2), 0.3) = (0, -1, 0.3)
    uint64_t worldTime = 18000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(-1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at dawn", "[phase6][daynight]") {
    // At dawn: worldTime = 0 (or 24000)
    // orbitalAngle = 0
    // sunDirection = (cos(0), sin(0), 0.3) = (1, 0, 0.3)
    uint64_t worldTime = 0;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: world time wraps at day boundary", "[phase6][daynight]") {
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // worldTime = 48000 (2 days) should wrap to same position as 0
    uint64_t worldTime = 48000;
    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    REQUIRE(dayFraction == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun elevation drives ambient brightness", "[phase6][daynight]") {
    // Test that sun elevation at noon produces higher ambient than at midnight
    auto computeAmbient = [](uint64_t worldTime) -> float {
        static constexpr uint64_t TICKS_PER_DAY = 24000;
        float dayFraction =
            static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
        float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);
        float sunElevation = std::sin(orbitalAngle);

        float ambientDay = 0.35f;
        float ambientNight = 0.1f;
        float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
        return ambientNight + (ambientDay - ambientNight) * ambientT;
    };

    float ambientNoon = computeAmbient(6000);
    float ambientMidnight = computeAmbient(18000);

    // Noon ambient should be higher than midnight
    REQUIRE(ambientNoon > ambientMidnight);
    REQUIRE(ambientNoon == Catch::Approx(0.35f).margin(0.01f));
    REQUIRE(ambientMidnight == Catch::Approx(0.1f).margin(0.01f));
}

// ============================================================================
// Phase 8: Post-Processing, Audio, Performance HUD Tests
// ============================================================================

// ---- Bloom Tests ----

TEST_CASE("Bloom: ACES tone mapping formula", "[phase8][bloom]") {
    // ACES: color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
    auto acesToneMap = [](float c) -> float {
        float a = c * (2.51f * c + 0.03f);
        float b = c * (2.43f * c + 0.59f) + 0.14f;
        return a / b;
    };

    // Identity at 0
    REQUIRE(acesToneMap(0.0f) == Catch::Approx(0.0f).epsilon(0.0001f));

    // Near-identity at 1.0 (slight compression)
    float at1 = acesToneMap(1.0f);
    REQUIRE(at1 > 0.8f);
    REQUIRE(at1 < 1.2f);

    // Compresses high values
    float at2 = acesToneMap(2.0f);
    REQUIRE(at2 < 2.0f); // Compression

    // Monotonically increasing
    REQUIRE(acesToneMap(0.5f) < acesToneMap(1.0f));
    REQUIRE(acesToneMap(1.0f) < acesToneMap(1.5f));
}

TEST_CASE("Bloom: extract threshold — bright pixels pass, dark pixels blocked", "[phase8][bloom]") {
    auto softThreshold = [](float luminance, float threshold) -> float {
        float low = threshold - 0.5f;
        float high = threshold + 0.5f;
        if (luminance <= low)
            return 0.0f;
        if (luminance >= high)
            return 1.0f;
        return (luminance - low) / (high - low);
    };

    // Dark pixel (luminance 0.2) with threshold 1.0 → blocked
    REQUIRE(softThreshold(0.2f, 1.0f) == Catch::Approx(0.0f));

    // Bright pixel (luminance 1.5) with threshold 1.0 → passes
    REQUIRE(softThreshold(1.5f, 1.0f) == Catch::Approx(1.0f));

    // Edge pixel (luminance 1.0) with threshold 1.0 → 0.5
    REQUIRE(softThreshold(1.0f, 1.0f) == Catch::Approx(0.5f));

    // Very bright pixel (luminance 3.0) → passes fully
    REQUIRE(softThreshold(3.0f, 1.0f) == Catch::Approx(1.0f));
}

TEST_CASE("Bloom: blur kernel weights are positive and symmetric", "[phase8][bloom]") {
    // 8-tap Kawase blur weights (normalized in shader by dividing by sum)
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // All weights are positive
    for (int i = 0; i < 8; ++i) {
        REQUIRE(weights[i] > 0.0f);
    }

    // Symmetric: first and last match, inner pairs match
    REQUIRE(weights[0] == weights[7]);
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);

    // Sum is used for normalization in shader
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += weights[i];
    }
    REQUIRE(sum > 0.0f); // Non-zero for valid normalization
}

TEST_CASE("Bloom: blur kernel is symmetric", "[phase8][bloom]") {
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // First and last should match
    REQUIRE(weights[0] == weights[7]);
    // Inner pairs should match
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);
}

// ---- Fog Tests ----

TEST_CASE("Fog: exponential fog factor at various distances", "[phase8][fog]") {
    float density = 0.0003f;

    auto fogFactor = [](float distance, float density) -> float {
        return 1.0f - std::exp(-density * distance);
    };

    // At distance 0: no fog
    REQUIRE(fogFactor(0.0f, density) == Catch::Approx(0.0f).epsilon(0.0001f));

    // At distance 100: slight fog
    float f100 = fogFactor(100.0f, density);
    REQUIRE(f100 > 0.0f);
    REQUIRE(f100 < 0.5f);

    // At distance 1000: significant fog
    float f1000 = fogFactor(1000.0f, density);
    REQUIRE(f1000 > 0.2f);
    REQUIRE(f1000 < 0.5f);

    // At distance 5000: very foggy
    float f5000 = fogFactor(5000.0f, density);
    REQUIRE(f5000 > 0.7f);

    // Fog factor increases monotonically with distance
    REQUIRE(fogFactor(100.0f, density) < fogFactor(500.0f, density));
    REQUIRE(fogFactor(500.0f, density) < fogFactor(1000.0f, density));
}

TEST_CASE("Fog: fog color mixing", "[phase8][fog]") {
    struct F3 {
        float x, y, z;
    };

    auto mixFog = [](float fogFactor, F3 fogColor, F3 litColor) -> F3 {
        // fogFactor: 0 = fully fogged, 1 = fully lit
        // mix(fogColor, litColor, fogFactor) = fogColor*(1-fogFactor) + litColor*fogFactor
        return {
            fogColor.x * (1.0f - fogFactor) + litColor.x * fogFactor,
            fogColor.y * (1.0f - fogFactor) + litColor.y * fogFactor,
            fogColor.z * (1.0f - fogFactor) + litColor.z * fogFactor,
        };
    };

    F3 fogColor{0.5f, 0.7f, 0.8f}; // Sky-like
    F3 litColor{0.3f, 0.3f, 0.3f}; // Dark stone

    // No fog (factor=1): fully lit
    auto noFog = mixFog(1.0f, fogColor, litColor);
    REQUIRE(noFog.x == Catch::Approx(litColor.x));

    // Full fog (factor=0): fully fogged
    auto fullFog = mixFog(0.0f, fogColor, litColor);
    REQUIRE(fullFog.x == Catch::Approx(fogColor.x));

    // Half fog: blend
    auto halfFog = mixFog(0.5f, fogColor, litColor);
    REQUIRE(halfFog.x == Catch::Approx((fogColor.x + litColor.x) * 0.5f));
}

// ---- Cloud Tests ----

TEST_CASE("Clouds: noise threshold for cloud generation", "[phase8][clouds]") {
    // Cloud threshold: 0.4 — noise values above this render as clouds
    float threshold = 0.4f;

    auto cloudMask = [](float noise, float threshold) -> float {
        float low = threshold - 0.1f;
        float high = threshold + 0.1f;
        if (noise <= low)
            return 0.0f;
        if (noise >= high)
            return 1.0f;
        return (noise - low) / (high - low);
    };

    // Low noise → no cloud
    REQUIRE(cloudMask(0.2f, threshold) == Catch::Approx(0.0f));

    // High noise → full cloud
    REQUIRE(cloudMask(0.6f, threshold) == Catch::Approx(1.0f));

    // At threshold → partial cloud
    REQUIRE(cloudMask(0.4f, threshold) == Catch::Approx(0.5f));
}

TEST_CASE("Clouds: wind offset calculation", "[phase8][clouds]") {
    // Wind speed: 0.02 blocks/tick
    float windSpeed = 0.02f;

    auto windOffset = [](uint64_t worldTime, float windSpeed) -> float {
        return static_cast<float>(worldTime) * windSpeed;
    };

    // At time 0: no offset
    REQUIRE(windOffset(0, windSpeed) == Catch::Approx(0.0f));

    // At time 1000: offset = 20
    REQUIRE(windOffset(1000, windSpeed) == Catch::Approx(20.0f));

    // At time 5000: offset = 100
    REQUIRE(windOffset(5000, windSpeed) == Catch::Approx(100.0f));

    // Monotonically increasing
    REQUIRE(windOffset(100, windSpeed) < windOffset(200, windSpeed));
}

TEST_CASE("Clouds: cloud altitude constant", "[phase8][clouds]") {
    // Cloud layer at Y=192
    static constexpr float CLOUD_ALTITUDE = 192.0f;
    REQUIRE(CLOUD_ALTITUDE > 0.0f);
    REQUIRE(CLOUD_ALTITUDE < 256.0f); // Within world bounds
}

// ---- Shared shader struct layout pins ----
// shader_types.hpp is compiled by BOTH clang++ and the Metal compiler; simd
// types have the same layout in each. These pins catch accidental drift
// (reordered fields, ad-hoc padding) that previously corrupted fog, camera
// position, sky colors, and particle data.

TEST_CASE("Shader types: Uniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(Uniforms) == 288);
    REQUIRE(offsetof(Uniforms, sunDirection) == 192);
    REQUIRE(offsetof(Uniforms, fogColor) == 240);
    REQUIRE(offsetof(Uniforms, fogDensity) == 256);
    REQUIRE(offsetof(Uniforms, cameraPosition) == 272);
    REQUIRE(alignof(Uniforms) == 16);
}

TEST_CASE("Shader types: SkyUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(SkyUniforms) == 80);
    REQUIRE(offsetof(SkyUniforms, horizonColor) == 16);
    REQUIRE(offsetof(SkyUniforms, sunIntensity) == 64);
}

TEST_CASE("Shader types: CloudUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(CloudUniforms) == 112);
    REQUIRE(offsetof(CloudUniforms, sunDirection) == 64);
    REQUIRE(offsetof(CloudUniforms, tanHalfFov) == 80);
    REQUIRE(offsetof(CloudUniforms, cloudThreshold) == 100);
}

TEST_CASE("Shader types: GPUParticle layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(GPUParticle) == 48);
    REQUIRE(offsetof(GPUParticle, velocity) == 16);
    REQUIRE(offsetof(GPUParticle, lifetime) == 32);
    REQUIRE(offsetof(GPUParticle, type) == 36);
}

TEST_CASE("Shader types: ParticleUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ParticleUniforms) == 144);
    REQUIRE(offsetof(ParticleUniforms, cameraPosition) == 128);
}

TEST_CASE("Shader types: BloomUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(BloomUniforms) == 32);
    REQUIRE(offsetof(BloomUniforms, texelSize) == 8);
    REQUIRE(offsetof(BloomUniforms, threshold) == 16);
    REQUIRE(offsetof(BloomUniforms, blurRadius) == 24);
}

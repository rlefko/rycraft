#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

bool isStrataMaterial(BlockType block) {
    switch (block) {
        case BlockType::STONE:
        case BlockType::CLAY:
        case BlockType::SILT:
        case BlockType::SANDSTONE:
        case BlockType::BASALT:
        case BlockType::LIMESTONE:
        case BlockType::OBSIDIAN:
        case BlockType::ANDESITE:
            return true;
        default:
            return false;
    }
}

class GeneratedCubes {
public:
    explicit GeneratedCubes(uint32_t seed) : generator_(seed) {}

    BlockType blockAt(int64_t x, int y, int64_t z) {
        const ChunkPos pos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)};
        auto [entry, inserted] = cubes_.try_emplace(pos, pos);
        if (inserted)
            generator_.generate(entry->second);
        return entry->second.getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                      Chunk::worldToLocal(z));
    }

    ChunkGenerator& generator() { return generator_; }

private:
    ChunkGenerator generator_;
    std::unordered_map<ChunkPos, Chunk> cubes_;
};

worldgen::SurfaceSample paletteFixture() {
    worldgen::SurfaceSample sample;
    sample.geology.crust = worldgen::CrustType::CONTINENTAL;
    sample.geology.rock = worldgen::RockType::GRANITE;
    sample.geology.lithology = {
        .primary = worldgen::RockType::GRANITE,
        .secondary = worldgen::RockType::LIMESTONE,
        .transition = 0.35,
        .contactDistance = 80.0,
    };
    sample.geology.distanceToBoundary = 1024.0;
    sample.hydrology.surfaceElevation = 96.0;
    sample.hydrology.channelDistance = 4096.0;
    sample.climate.temperatureC = 14.0;
    sample.climate.annualPrecipitationMm = 780.0;
    sample.climate.potentialEvapotranspirationMm = 740.0;
    sample.climate.aridity = 0.95;
    sample.climate.relativeHumidity = 0.45;
    sample.soil.moisture = 0.48;
    sample.soil.fertility = 0.46;
    sample.soil.drainage = 0.52;
    sample.soil.waterTable = 70.0;
    sample.biome = {.primary = Biome::PLAINS, .secondary = Biome::DESERT, .transition = 0.2};
    sample.terrainHeight = 96.0;
    return sample;
}

int paletteWeight(const worldgen::surface_material::SurfaceMaterialPalette& palette,
                  BlockType material) {
    for (size_t index = 0; index < palette.count; ++index) {
        if (palette.entries[index].material == material)
            return palette.entries[index].weight;
    }
    return 0;
}

int transitionHeight(GeneratedCubes& cubes, int64_t x, int64_t z) {
    for (int y = 48; y >= -32; --y) {
        const BlockType upper = cubes.blockAt(x, y, z);
        const BlockType lower = cubes.blockAt(x, y - 1, z);
        if (isStrataMaterial(upper) && isStrataMaterial(lower) && upper != lower)
            return y;
    }
    return WORLD_MIN_Y - 1;
}

size_t longestConstantRun(const std::vector<int>& values, int edge, int dx, int dz) {
    size_t longest = 0;
    for (int z = 0; z < edge; ++z) {
        for (int x = 0; x < edge; ++x) {
            const int previousX = x - dx;
            const int previousZ = z - dz;
            if (previousX >= 0 && previousX < edge && previousZ >= 0 && previousZ < edge) {
                continue;
            }
            size_t run = 0;
            int previous = WORLD_MIN_Y - 2;
            for (int sampleX = x, sampleZ = z;
                 sampleX >= 0 && sampleX < edge && sampleZ >= 0 && sampleZ < edge;
                 sampleX += dx, sampleZ += dz) {
                const int value = values[static_cast<size_t>(sampleZ * edge + sampleX)];
                run = value > WORLD_MIN_Y && value == previous ? run + 1
                                                               : (value > WORLD_MIN_Y ? 1 : 0);
                longest = std::max(longest, run);
                previous = value;
            }
        }
    }
    return longest;
}

std::vector<BlockType> generateRoute(ChunkGenerator& generator,
                                     const std::vector<ChunkPos>& route) {
    std::vector<BlockType> result;
    result.reserve(route.size() * CHUNK_VOLUME);
    for (const ChunkPos pos : route) {
        Chunk cube(pos);
        generator.generate(cube);
        const std::vector<BlockType> blocks = cube.copyBlocks();
        result.insert(result.end(), blocks.begin(), blocks.end());
    }
    return result;
}

} // namespace

TEST_CASE("Surface palettes blend four biomes and exposed lithology with positive weights",
          "[worldgen][geology][material][palette][organic][organic-geology]") {
    using namespace worldgen::surface_material;
    worldgen::SurfaceSample sample = paletteFixture();
    sample.suitability.scores[static_cast<size_t>(Biome::PLAINS)] = 1.0F;
    sample.suitability.scores[static_cast<size_t>(Biome::DESERT)] = 0.9F;
    sample.suitability.scores[static_cast<size_t>(Biome::BADLANDS)] = 0.8F;
    sample.suitability.scores[static_cast<size_t>(Biome::SWAMP)] = 0.7F;

    const SurfaceMaterialPalette biomePalette = materialPalette(sample, {}, false, false);
    REQUIRE(biomePalette.count == 4);
    REQUIRE(paletteWeight(biomePalette, BlockType::GRASS) > 0);
    REQUIRE(paletteWeight(biomePalette, BlockType::SAND) > 0);
    REQUIRE(paletteWeight(biomePalette, BlockType::SANDSTONE) > 0);
    REQUIRE(paletteWeight(biomePalette, BlockType::MUD) > 0);
    int biomeWeight = 0;
    for (size_t index = 0; index < biomePalette.count; ++index) {
        REQUIRE(biomePalette.entries[index].weight > 0);
        biomeWeight += biomePalette.entries[index].weight;
    }
    REQUIRE(biomeWeight == 255);

    sample.slope = 0.96;
    const SurfaceMaterialPalette rockPalette = materialPalette(sample, {}, false, false);
    REQUIRE(exposesLithology(sample, Biome::PLAINS, {}, false, false));
    REQUIRE(paletteWeight(rockPalette, BlockType::STONE) > 0);
    REQUIRE(paletteWeight(rockPalette, BlockType::LIMESTONE) > 0);
    REQUIRE(paletteWeight(rockPalette, BlockType::LIMESTONE) ==
            Catch::Approx(255.0 * sample.geology.lithology.transition).margin(3.0));
}

TEST_CASE("Physical surface constraints override lithology and biome palettes",
          "[worldgen][geology][material][palette][exclusion][organic-geology]") {
    using namespace worldgen::surface_material;
    worldgen::SurfaceSample sample = paletteFixture();
    sample.suitability.scores[static_cast<size_t>(Biome::PLAINS)] = 1.0F;
    sample.suitability.scores[static_cast<size_t>(Biome::DESERT)] = 0.8F;
    sample.slope = 0.96;

    SECTION("source water retains sediment") {
        sample.hydrology.river = true;
        sample.soil.moisture = 0.8;
        const SurfaceMaterialPalette palette = materialPalette(sample, {}, false, true);
        REQUIRE(palette.count == 1);
        REQUIRE(palette.entries[0].material == BlockType::MUD);
        REQUIRE(palette.entries[0].weight == 255);
    }

    SECTION("deltas retain silt") {
        sample.hydrology.delta = true;
        const SurfaceMaterialPalette palette = materialPalette(sample, {}, false, false);
        REQUIRE(palette.count == 1);
        REQUIRE(palette.entries[0].material == BlockType::SILT);
        REQUIRE(palette.entries[0].weight == 255);
    }

    SECTION("snow remains climate-owned") {
        const SurfaceMaterialPalette palette = materialPalette(sample, {}, true, false);
        REQUIRE(palette.count == 1);
        REQUIRE(palette.entries[0].material == BlockType::SNOW);
        REQUIRE(palette.entries[0].weight == 255);
    }

    SECTION("active submerged conduits retain basalt") {
        sample.hydrology.lake = true;
        const VolcanicSignals conduit{.conduitExposure = true};
        const SurfaceMaterialPalette palette = materialPalette(sample, conduit, false, true);
        REQUIRE(palette.count == 1);
        REQUIRE(palette.entries[0].material == BlockType::BASALT);
        REQUIRE(palette.entries[0].weight == 255);
    }
}

TEST_CASE("Curved strata avoid long cardinal and diagonal block runs",
          "[worldgen][geology][strata][continuity][artifact][organic][organic-geology]") {
    constexpr int EDGE = 48;
    GeneratedCubes cubes(42);
    std::vector<int> contacts(static_cast<size_t>(EDGE * EDGE));
    int minimumHeight = WORLD_MAX_Y;
    int maximumHeight = WORLD_MIN_Y;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const int height = transitionHeight(cubes, x, 1200 + z);
            INFO("missing stratal contact at " << x << ',' << 1200 + z);
            REQUIRE(height > WORLD_MIN_Y);
            contacts[static_cast<size_t>(z * EDGE + x)] = height;
            minimumHeight = std::min(minimumHeight, height);
            maximumHeight = std::max(maximumHeight, height);
        }
    }
    REQUIRE(maximumHeight - minimumHeight >= 4);
    for (const auto [dx, dz] :
         std::array<std::pair<int, int>, 4>{{{1, 0}, {0, 1}, {1, 1}, {1, -1}}}) {
        const size_t longest = longestConstantRun(contacts, EDGE, dx, dz);
        INFO("direction " << dx << ',' << dz << " longest run " << longest);
        REQUIRE(longest <= 24);
    }

    std::array<bool, 16> observedRunLengths{};
    for (int64_t x = -64; x <= 64; x += 8) {
        BlockType previous = BlockType::AIR;
        size_t run = 0;
        for (int y = -64; y <= 48; ++y) {
            const BlockType block = cubes.blockAt(x, y, 1234);
            if (block == previous && isStrataMaterial(block)) {
                ++run;
                continue;
            }
            if (run > 0)
                observedRunLengths[std::min(run, observedRunLengths.size() - 1)] = true;
            previous = block;
            run = isStrataMaterial(block) ? 1 : 0;
        }
        if (run > 0)
            observedRunLengths[std::min(run, observedRunLengths.size() - 1)] = true;
    }
    REQUIRE(std::count(observedRunLengths.begin(), observedRunLengths.end(), true) >= 8);
}

TEST_CASE("Strata remain continuous across storage and former phase boundaries",
          "[worldgen][geology][strata][continuity][seam][wrap][organic-geology]") {
    struct SeamFixture {
        int64_t rightX;
        int64_t z;
        size_t maximumMismatches;
    };
    constexpr std::array<SeamFixture, 5> FIXTURES = {{
        {16, 1234, 4},
        {64, 1234, 4},
        {-8192, 4096, 4},
        {295689, 0, 4},
        {20993900, 0, 4},
    }};
    GeneratedCubes cubes(42);
    for (const SeamFixture fixture : FIXTURES) {
        const worldgen::GeologySample leftGeology =
            cubes.generator().sampleFarSurface(fixture.rightX - 1, fixture.z).geology;
        const worldgen::GeologySample rightGeology =
            cubes.generator().sampleFarSurface(fixture.rightX, fixture.z).geology;
        REQUIRE(leftGeology.plateId == rightGeology.plateId);
        const bool taggedFault = (leftGeology.boundary == worldgen::PlateBoundary::TRANSFORM &&
                                  leftGeology.faultStrength > 0.25) ||
                                 (rightGeology.boundary == worldgen::PlateBoundary::TRANSFORM &&
                                  rightGeology.faultStrength > 0.25);
        REQUIRE_FALSE(taggedFault);
        const int maximumY = std::min(cubes.generator().surfaceYAt(fixture.rightX - 1, fixture.z),
                                      cubes.generator().surfaceYAt(fixture.rightX, fixture.z)) -
                             20;
        size_t comparable = 0;
        size_t mismatches = 0;
        for (int y = maximumY - 63; y <= maximumY; ++y) {
            const BlockType left = cubes.blockAt(fixture.rightX - 1, y, fixture.z);
            const BlockType right = cubes.blockAt(fixture.rightX, y, fixture.z);
            if (!isStrataMaterial(left) || !isStrataMaterial(right))
                continue;
            ++comparable;
            mismatches += left != right;
        }
        INFO("seam " << fixture.rightX << ',' << fixture.z << " comparable " << comparable
                     << " mismatches " << mismatches);
        REQUIRE(comparable >= 24);
        REQUIRE(mismatches <= fixture.maximumMismatches);
    }
}

TEST_CASE("Lithology lenses are bounded within their host rock",
          "[worldgen][geology][strata][lens][reachability][organic-geology]") {
    constexpr int64_t X = 3776;
    constexpr int Y = 5;
    constexpr int64_t Z = 1600;
    GeneratedCubes cubes(42);
    const worldgen::GeologySample geology = cubes.generator().sampleFarSurface(X, Z).geology;
    REQUIRE(geology.lithology.primary == worldgen::RockType::GRANITE);
    REQUIRE(geology.lithology.transition < 0.01);
    REQUIRE(geology.volcanicActivity < 0.1);
    REQUIRE(geology.uplift < 0.3);
    REQUIRE(cubes.blockAt(X, Y, Z) == BlockType::LIMESTONE);
    REQUIRE(cubes.blockAt(X, Y - 1, Z) == BlockType::STONE);
    REQUIRE(cubes.blockAt(X, Y - 8, Z) == BlockType::STONE);
}

TEST_CASE("Strata are identical after reverse generation and macro cache eviction",
          "[worldgen][geology][strata][determinism][cache][organic-geology]") {
    const std::vector<ChunkPos> route = {
        {0, 2, 77},
        {1, 2, 77},
        {3, 2, 77},
        {4, 2, 77},
        {-513, 4, 256},
        {-512, 4, 256},
        {Chunk::worldToChunk(295689), 2, 0},
        {Chunk::worldToChunk(20993900), 2, 0},
    };
    ChunkGenerator generator(42);
    const std::vector<BlockType> forward = generateRoute(generator, route);
    generator.clearMacroCaches();
    std::vector<ChunkPos> reverseRoute(route.rbegin(), route.rend());
    const std::vector<BlockType> reverse = generateRoute(generator, reverseRoute);

    // Reverse whole cubes, not individual cube storage.
    std::vector<BlockType> reordered;
    reordered.reserve(reverse.size());
    for (size_t cube = route.size(); cube-- > 0;) {
        const auto first = reverse.begin() + static_cast<std::ptrdiff_t>(cube * CHUNK_VOLUME);
        reordered.insert(reordered.end(), first, first + CHUNK_VOLUME);
    }
    REQUIRE(reordered == forward);
}

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common/counter_rng.hpp"
#include "world/artifact_analysis.hpp"
#include "world/chunk_generator.hpp"
#include "world/density_field.hpp"
#include "world/features.hpp"
#include "world/macro_generation.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace artifact = worldgen::artifact_analysis;

BlockType generatedBlockAt(ChunkGenerator& generator, int64_t x, int y, int64_t z) {
    Chunk cube(ChunkPos{Chunk::worldToChunk(x), Chunk::worldToChunkY(y), Chunk::worldToChunk(z)});
    generator.generate(cube);
    return cube.getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y), Chunk::worldToLocal(z));
}

bool isOceanBiome(Biome biome) {
    return biome == Biome::OCEAN || biome == Biome::DEEP_OCEAN || biome == Biome::FROZEN_OCEAN;
}

worldgen::SurfaceSample materialFixture() {
    worldgen::SurfaceSample sample;
    sample.geology.crust = worldgen::CrustType::CONTINENTAL;
    sample.geology.rock = worldgen::RockType::GRANITE;
    sample.geology.distanceToBoundary = 1024.0;
    sample.hydrology.surfaceElevation = 90.0;
    sample.hydrology.channelDistance = 4096.0;
    sample.climate.temperatureC = 14.0;
    sample.climate.annualPrecipitationMm = 900.0;
    sample.climate.potentialEvapotranspirationMm = 720.0;
    sample.climate.aridity = 0.8;
    sample.climate.relativeHumidity = 0.48;
    sample.soil.moisture = 0.45;
    sample.soil.fertility = 0.48;
    sample.soil.drainage = 0.5;
    sample.soil.waterTable = 72.0;
    sample.biome.primary = Biome::PLAINS;
    sample.biome.secondary = Biome::PLAINS;
    sample.terrainHeight = 90.0;
    return sample;
}

worldgen::SurfaceSample treeHabitatFixture(Biome biome) {
    worldgen::SurfaceSample sample = materialFixture();
    sample.biome.primary = biome;
    sample.biome.secondary = biome;
    sample.biome.transition = 0.0;
    sample.suitability.scores.fill(0.0F);
    sample.suitability.scores[static_cast<size_t>(biome)] = 1.0F;
    sample.terrainHeight = 90.0;
    sample.hydrology.surfaceElevation = 90.0;
    sample.waterSurface = 0.0;
    sample.hydrology.waterSurface = 0.0;
    sample.slope = 0.12;
    sample.geology.rock = worldgen::RockType::GRANITE;
    sample.geology.lithology.primary = worldgen::RockType::GRANITE;
    sample.geology.lithology.secondary = worldgen::RockType::GRANITE;
    sample.geology.lithology.contactDistance = 512.0;
    sample.geology.volcanicActivity = 0.0;
    sample.geology.faultStrength = 0.0;
    sample.geology.uplift = 0.0;
    sample.climate.relativeHumidity = 0.72;
    sample.soil.drainage = 0.55;
    sample.soil.waterTable = 84.0;
    sample.ecotopes = worldgen::Ecotope::NONE;
    return sample;
}

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

std::vector<BlockType> generatedColumn(ChunkGenerator& generator, int64_t x, int minimumY,
                                       int maximumY, int64_t z) {
    std::vector<BlockType> result;
    result.reserve(static_cast<size_t>(maximumY - minimumY + 1));
    for (int32_t sectionY = Chunk::worldToChunkY(minimumY);
         sectionY <= Chunk::worldToChunkY(maximumY); ++sectionY) {
        Chunk cube(ChunkPos{Chunk::worldToChunk(x), sectionY, Chunk::worldToChunk(z)});
        generator.generate(cube);
        const int firstY = std::max(minimumY, sectionY * CHUNK_EDGE);
        const int lastY = std::min(maximumY, sectionY * CHUNK_EDGE + CHUNK_EDGE - 1);
        for (int y = firstY; y <= lastY; ++y) {
            result.push_back(cube.getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                           Chunk::worldToLocal(z)));
        }
    }
    return result;
}

void requireContinuousFieldsEqual(const worldgen::SurfaceSample& first,
                                  const worldgen::SurfaceSample& second) {
    REQUIRE(first.climate.wind.x == second.climate.wind.x);
    REQUIRE(first.climate.wind.z == second.climate.wind.z);
    REQUIRE(first.climate.temperatureC == second.climate.temperatureC);
    REQUIRE(first.climate.annualPrecipitationMm == second.climate.annualPrecipitationMm);
    REQUIRE(first.climate.potentialEvapotranspirationMm ==
            second.climate.potentialEvapotranspirationMm);
    REQUIRE(first.climate.aridity == second.climate.aridity);
    REQUIRE(first.climate.relativeHumidity == second.climate.relativeHumidity);
    REQUIRE(first.soil.moisture == second.soil.moisture);
    REQUIRE(first.soil.fertility == second.soil.fertility);
    REQUIRE(first.soil.drainage == second.soil.drainage);
    REQUIRE(first.soil.waterTable == second.soil.waterTable);
    REQUIRE(first.suitability.scores == second.suitability.scores);
    REQUIRE(first.biome.primary == second.biome.primary);
    REQUIRE(first.biome.secondary == second.biome.secondary);
    REQUIRE(first.biome.transition == second.biome.transition);
}

BlockType dominantSurfaceMaterial(ChunkGenerator& generator, int64_t x, int64_t z) {
    return generator.farSurfaceMaterialAt(x, z);
}

int surfacePaletteWeight(const worldgen::surface_material::SurfaceMaterialPalette& palette,
                         BlockType material) {
    for (size_t index = 0; index < palette.count; ++index) {
        if (palette.entries[index].material == material)
            return palette.entries[index].weight;
    }
    return 0;
}

bool isTaggedLithologyDiscontinuity(const worldgen::GeologySample& first,
                                    const worldgen::GeologySample& second) {
    return first.faultStrength > 0.45 || second.faultStrength > 0.45 ||
           first.boundary == worldgen::PlateBoundary::TRANSFORM ||
           second.boundary == worldgen::PlateBoundary::TRANSFORM;
}

bool hasHardSurfaceMaterialConstraint(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.ocean || sample.hydrology.lake || sample.hydrology.river ||
           sample.hydrology.waterfall || sample.hydrology.delta ||
           sample.geology.volcanicActivity > 0.52 ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::CLIFF) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SCREE) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SNOWFIELD) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::GLACIER);
}

} // namespace

TEST_CASE("Geology and terrain signals select every natural material coherently",
          "[worldgen][geology][material]") {
    using namespace worldgen::surface_material;
    worldgen::SurfaceSample sample = materialFixture();

    REQUIRE(outcrop(sample.geology) == BlockType::STONE);
    sample.geology.boundary = worldgen::PlateBoundary::CONVERGENT;
    sample.geology.uplift = 0.7;
    sample.geology.volcanicActivity = 0.3;
    REQUIRE(outcrop(sample.geology) == BlockType::ANDESITE);
    sample.geology = materialFixture().geology;
    sample.geology.rock = worldgen::RockType::BASALT;
    REQUIRE(outcrop(sample.geology) == BlockType::BASALT);
    sample.geology.rock = worldgen::RockType::LIMESTONE;
    REQUIRE(outcrop(sample.geology) == BlockType::LIMESTONE);
    sample.geology.rock = worldgen::RockType::SANDSTONE;
    REQUIRE(outcrop(sample.geology) == BlockType::SANDSTONE);

    SECTION("rooted vegetation requires soil instead of bare outcrop") {
        for (const BlockType substrate : {BlockType::GRASS, BlockType::DIRT, BlockType::SAND,
                                          BlockType::MUD, BlockType::CLAY, BlockType::SILT}) {
            REQUIRE(supportsSurfaceFlora(substrate));
            REQUIRE(supportsTreeRooting(substrate));
        }
        REQUIRE_FALSE(supportsSurfaceFlora(BlockType::SNOW));
        REQUIRE(supportsTreeRooting(BlockType::SNOW));
        for (const BlockType substrate :
             {BlockType::STONE, BlockType::GRAVEL, BlockType::SANDSTONE, BlockType::BASALT,
              BlockType::VOLCANIC_ASH, BlockType::LIMESTONE, BlockType::OBSIDIAN,
              BlockType::ANDESITE}) {
            REQUIRE_FALSE(supportsSurfaceFlora(substrate));
            REQUIRE_FALSE(supportsTreeRooting(substrate));
        }
    }

    SECTION("weathered cliffs and scree expose coherent host material") {
        sample = materialFixture();
        sample.geology.rock = worldgen::RockType::LIMESTONE;
        sample.slope = 1.15;
        REQUIRE(surface(sample, Biome::PLAINS, {}, false, false) == BlockType::LIMESTONE);

        sample.geology.rock = worldgen::RockType::GRANITE;
        sample.terrainHeight = 180.0;
        sample.slope = 0.92;
        REQUIRE(surface(sample, Biome::PLAINS, {}, false, false) == BlockType::GRAVEL);
    }

    SECTION("depositional sediment covers inactive submerged lava") {
        sample = materialFixture();
        sample.geology.rock = worldgen::RockType::VOLCANIC;
        sample.geology.volcanicActivity = 0.55;
        sample.hydrology.ocean = true;
        sample.hydrology.delta = true;
        sample.terrainHeight = 52.0;
        sample.waterSurface = SEA_LEVEL;
        sample.hydrology.waterSurface = SEA_LEVEL;
        const VolcanicSignals weatheredField{.basaltField = 0.45};
        REQUIRE(surface(sample, Biome::VOLCANIC_BARREN, weatheredField, false, true) ==
                BlockType::SILT);

        const VolcanicSignals activeCrater{
            .basaltField = 0.45, .craterFactor = 0.8, .conduitExposure = true};
        REQUIRE(surface(sample, Biome::VOLCANIC_BARREN, activeCrater, false, true) ==
                BlockType::BASALT);
    }

    SECTION("water depth moisture and alluvium select related sediments") {
        sample = materialFixture();
        sample.hydrology.lake = true;
        sample.terrainHeight = 60.0;
        sample.waterSurface = 64.0;
        sample.hydrology.waterSurface = 64.0;
        sample.soil.moisture = 0.52;
        REQUIRE(surface(sample, Biome::PLAINS, {}, false, true) == BlockType::CLAY);
        sample.soil.moisture = 0.78;
        REQUIRE(surface(sample, Biome::PLAINS, {}, false, true) == BlockType::MUD);

        sample.hydrology = {};
        sample.terrainHeight = 90.0;
        REQUIRE(surface(sample, Biome::PLAINS, {}, false, false, true) == BlockType::MUD);
    }

    SECTION("ash obsidian and snow require their geological or climate context") {
        sample = materialFixture();
        sample.geology.rock = worldgen::RockType::VOLCANIC;
        sample.geology.volcanicActivity = 0.82;
        REQUIRE(surface(sample, Biome::VOLCANIC_BARREN, {}, false, false) ==
                BlockType::VOLCANIC_ASH);

        sample.slope = 0.82;
        const VolcanicSignals conduit{.craterFactor = 0.84, .conduitExposure = true};
        REQUIRE(surface(sample, Biome::VOLCANIC_BARREN, conduit, false, false) ==
                BlockType::OBSIDIAN);

        sample = materialFixture();
        sample.climate.temperatureC = -8.0;
        sample.climate.annualPrecipitationMm = 700.0;
        REQUIRE(frozen(sample, Biome::MONTANE_GRASSLAND));
        REQUIRE(surface(sample, Biome::MONTANE_GRASSLAND, {}, true, false) == BlockType::SNOW);
    }
}

TEST_CASE("Only adapted trees root in shallow supported water",
          "[worldgen][ecology][tree][water][habitat]") {
    using feature_generation::TreeSpecies;
    worldgen::SurfaceSample mangrove = treeHabitatFixture(Biome::MANGROVE);
    mangrove.climate.temperatureC = 27.0;
    mangrove.climate.annualPrecipitationMm = 2400.0;
    mangrove.climate.potentialEvapotranspirationMm = 900.0;
    mangrove.soil.moisture = 0.88;
    mangrove.soil.fertility = 0.62;
    mangrove.terrainHeight = 80.0;
    mangrove.hydrology.surfaceElevation = 80.0;
    mangrove.hydrology.lake = true;
    mangrove.hydrology.lakeDepth = 2.0;
    mangrove.hydrology.waterSurface = 82.0;
    mangrove.waterSurface = 82.0;
    mangrove.ecotopes = worldgen::Ecotope::LAKESHORE;

    const auto mangroveRoots =
        feature_generation::evaluateTreeHabitat(TreeSpecies::MANGROVE, mangrove, 79);
    REQUIRE(mangroveRoots.allowed);
    REQUIRE(mangroveRoots.submerged);
    REQUIRE(mangroveRoots.waterDepthBlocks == 2);
    for (const TreeSpecies ordinary :
         {TreeSpecies::OAK, TreeSpecies::BIRCH, TreeSpecies::SPRUCE, TreeSpecies::ACACIA,
          TreeSpecies::JUNGLE, TreeSpecies::PALM, TreeSpecies::FALLEN_LOG}) {
        CAPTURE(static_cast<int>(ordinary));
        const auto habitat = feature_generation::evaluateTreeHabitat(ordinary, mangrove, 79);
        REQUIRE(habitat.submerged);
        REQUIRE_FALSE(habitat.allowed);
    }

    worldgen::SurfaceSample willow = treeHabitatFixture(Biome::SWAMP);
    willow.climate.temperatureC = 15.0;
    willow.climate.annualPrecipitationMm = 1900.0;
    willow.climate.potentialEvapotranspirationMm = 650.0;
    willow.soil.moisture = 0.82;
    willow.soil.fertility = 0.58;
    willow.hydrology.river = true;
    willow.hydrology.channelWidth = 8.0;
    willow.hydrology.channelDistance = 0.0;
    willow.hydrology.waterSurface = 91.0;
    willow.waterSurface = 91.0;
    willow.ecotopes = worldgen::Ecotope::RIVERBANK;
    const auto willowRoots =
        feature_generation::evaluateTreeHabitat(TreeSpecies::WILLOW, willow, 89);
    REQUIRE(willowRoots.allowed);
    REQUIRE(willowRoots.waterDepthBlocks == 1);

    willow.hydrology.ocean = true;
    REQUIRE_FALSE(feature_generation::evaluateTreeHabitat(TreeSpecies::WILLOW, willow, 89).allowed);

    mangrove.hydrology.waterSurface = 86.0;
    mangrove.waterSurface = 86.0;
    const auto deepRoots =
        feature_generation::evaluateTreeHabitat(TreeSpecies::MANGROVE, mangrove, 79);
    REQUIRE(deepRoots.submerged);
    REQUIRE(deepRoots.waterDepthBlocks == 6);
    REQUIRE_FALSE(deepRoots.allowed);
}

TEST_CASE("Tree cover and species vary continuously with habitat",
          "[worldgen][ecology][tree][climate][biome][density]") {
    using feature_generation::TreeSpecies;
    worldgen::SurfaceSample rainforest = treeHabitatFixture(Biome::TROPICAL_RAINFOREST);
    rainforest.climate.temperatureC = 28.0;
    rainforest.climate.annualPrecipitationMm = 3000.0;
    rainforest.climate.potentialEvapotranspirationMm = 1000.0;
    rainforest.climate.relativeHumidity = 0.90;
    rainforest.soil.moisture = 0.90;
    rainforest.soil.fertility = 0.82;
    const double denseCover = feature_generation::treeCoverDensity(rainforest);
    REQUIRE(denseCover > 0.78);
    const auto jungle =
        feature_generation::evaluateTreeHabitat(TreeSpecies::JUNGLE, rainforest, 89);
    REQUIRE(jungle.allowed);
    REQUIRE(
        jungle.suitability >
        feature_generation::evaluateTreeHabitat(TreeSpecies::ACACIA, rainforest, 89).suitability);

    worldgen::SurfaceSample taiga = treeHabitatFixture(Biome::TAIGA);
    taiga.climate.temperatureC = -2.0;
    taiga.climate.annualPrecipitationMm = 900.0;
    taiga.climate.potentialEvapotranspirationMm = 380.0;
    taiga.soil.moisture = 0.62;
    taiga.soil.fertility = 0.40;
    REQUIRE(feature_generation::evaluateTreeHabitat(TreeSpecies::SPRUCE, taiga, 89).allowed);
    REQUIRE_FALSE(feature_generation::evaluateTreeHabitat(TreeSpecies::JUNGLE, taiga, 89).allowed);

    worldgen::SurfaceSample savanna = treeHabitatFixture(Biome::SAVANNA);
    savanna.climate.temperatureC = 29.0;
    savanna.climate.annualPrecipitationMm = 680.0;
    savanna.climate.potentialEvapotranspirationMm = 1250.0;
    savanna.soil.moisture = 0.30;
    savanna.soil.fertility = 0.38;
    REQUIRE(feature_generation::evaluateTreeHabitat(TreeSpecies::ACACIA, savanna, 89).allowed);
    REQUIRE_FALSE(
        feature_generation::evaluateTreeHabitat(TreeSpecies::SPRUCE, savanna, 89).allowed);

    worldgen::SurfaceSample desert = treeHabitatFixture(Biome::DESERT);
    desert.climate.temperatureC = 31.0;
    desert.climate.annualPrecipitationMm = 90.0;
    desert.climate.potentialEvapotranspirationMm = 1800.0;
    desert.soil.moisture = 0.04;
    desert.soil.fertility = 0.06;
    REQUIRE(feature_generation::treeCoverDensity(desert) < 0.02);

    worldgen::SurfaceSample stressed = rainforest;
    stressed.geology.rock = worldgen::RockType::VOLCANIC;
    stressed.geology.volcanicActivity = 0.80;
    stressed.ecotopes = worldgen::Ecotope::GEOTHERMAL;
    REQUIRE(feature_generation::treeCoverDensity(stressed) < denseCover * 0.24);
    REQUIRE_FALSE(
        feature_generation::evaluateTreeHabitat(TreeSpecies::JUNGLE, stressed, 89).allowed);
}

TEST_CASE("Material dither forms deterministic connected transition patches",
          "[worldgen][geology][material][determinism]") {
    worldgen::SurfaceSample sample = materialFixture();
    sample.biome.primary = Biome::PLAINS;
    sample.biome.secondary = Biome::VOLCANIC_BARREN;
    sample.biome.transition = 0.36;
    CounterRng random(42);

    std::array<Biome, 64 * 64> first{};
    size_t matchingNeighbors = 0;
    size_t neighborCount = 0;
    for (int z = 0; z < 64; ++z) {
        for (int x = 0; x < 64; ++x) {
            const size_t index = static_cast<size_t>(z * 64 + x);
            first[index] =
                worldgen::surface_material::materialBiome(sample, random, x - 64, z - 64);
            if (x > 0) {
                matchingNeighbors += first[index] == first[index - 1];
                ++neighborCount;
            }
            if (z > 0) {
                matchingNeighbors += first[index] == first[index - 64];
                ++neighborCount;
            }
        }
    }

    REQUIRE(static_cast<double>(matchingNeighbors) / neighborCount > 0.68);
    for (int z = 63; z >= 0; --z) {
        for (int x = 63; x >= 0; --x) {
            const size_t index = static_cast<size_t>(z * 64 + x);
            REQUIRE(worldgen::surface_material::materialBiome(sample, random, x - 64, z - 64) ==
                    first[index]);
        }
    }

    size_t secondaryCount = 0;
    constexpr int COARSE_EDGE = 32;
    for (int cellZ = -COARSE_EDGE / 2; cellZ < COARSE_EDGE / 2; ++cellZ) {
        for (int cellX = -COARSE_EDGE / 2; cellX < COARSE_EDGE / 2; ++cellX) {
            secondaryCount +=
                worldgen::surface_material::materialBiome(
                    sample, random, cellX * 64 + 17, cellZ * 64 + 29) == Biome::VOLCANIC_BARREN;
        }
    }
    REQUIRE(static_cast<double>(secondaryCount) / (COARSE_EDGE * COARSE_EDGE) ==
            Catch::Approx(sample.biome.transition).margin(0.05));
}

TEST_CASE("Multiscale biome dithering is coordinate-pure and does not expose aligned cells",
          "[worldgen][climate][biome][dither][determinism]") {
    constexpr uint64_t STREAM = 0x434C494D41544531ULL;
    CounterRng random(42);
    constexpr std::array<ColumnPos, 8> samples = {
        ColumnPos{0, 0},   ColumnPos{3, 3},     ColumnPos{15, 15},   ColumnPos{63, 63},
        ColumnPos{-1, -1}, ColumnPos{-16, -16}, ColumnPos{-63, -63}, ColumnPos{9'223, -17'009},
    };
    for (const ColumnPos sample : samples) {
        const double first =
            worldgen::multiscaleDitherThreshold(random, STREAM, sample.x, sample.z);
        const double repeated =
            worldgen::multiscaleDitherThreshold(random, STREAM, sample.x, sample.z);
        REQUIRE(first == repeated);
        REQUIRE(first > 0.0);
        REQUIRE(first < 1.0);
    }

    std::array<bool, 4> positiveRanks{};
    std::array<bool, 4> negativeRanks{};
    for (int64_t offset = 1; offset < 64; offset += 7) {
        positiveRanks[static_cast<size_t>(
            worldgen::multiscaleDitherThreshold(random, STREAM, offset, 64 - offset) * 4.0)] = true;
        negativeRanks[static_cast<size_t>(
            worldgen::multiscaleDitherThreshold(random, STREAM, -offset, offset - 64) * 4.0)] =
            true;
    }
    REQUIRE(std::count(positiveRanks.begin(), positiveRanks.end(), true) >= 2);
    REQUIRE(std::count(negativeRanks.begin(), negativeRanks.end(), true) >= 2);

    size_t belowTransition = 0;
    size_t sampleCount = 0;
    for (int64_t z = -512; z < 512; z += 4) {
        for (int64_t x = -512; x < 512; x += 4) {
            belowTransition += worldgen::multiscaleDitherThreshold(random, STREAM, x, z) < 0.37;
            ++sampleCount;
        }
    }
    REQUIRE(static_cast<double>(belowTransition) / static_cast<double>(sampleCount) ==
            Catch::Approx(0.37).margin(0.025));
}

TEST_CASE("Surface footprints filter detail without changing macro ownership",
          "[worldgen][surface-footprint][lod][determinism]") {
    STATIC_REQUIRE(worldgen::surfaceFootprintWidth(worldgen::SurfaceFootprint::BLOCK_1) == 1);
    STATIC_REQUIRE(worldgen::surfaceFootprintWidth(worldgen::SurfaceFootprint::BLOCK_16) == 16);
    ChunkGenerator generator(42);
    constexpr int64_t X = 1'184;
    constexpr int64_t Z = -2'736;
    const worldgen::SurfaceSample block =
        generator.sampleFarSurface(X, Z, worldgen::SurfaceFootprint::BLOCK_1);
    for (const worldgen::SurfaceFootprint footprint : {
             worldgen::SurfaceFootprint::BLOCK_2,
             worldgen::SurfaceFootprint::BLOCK_4,
             worldgen::SurfaceFootprint::BLOCK_8,
             worldgen::SurfaceFootprint::BLOCK_16,
         }) {
        const worldgen::SurfaceSample filtered = generator.sampleFarSurface(X, Z, footprint);
        REQUIRE(filtered.geology.plateId == block.geology.plateId);
        REQUIRE(filtered.hydrology.waterBodyId == block.hydrology.waterBodyId);
        REQUIRE(filtered.hydrology.ocean == block.hydrology.ocean);
        REQUIRE(filtered.hydrology.lake == block.hydrology.lake);
        REQUIRE(filtered.hydrology.river == block.hydrology.river);
        REQUIRE(filtered.waterSurface == block.waterSurface);
        REQUIRE(generator.sampleFarSurface(X, Z, footprint).terrainHeight ==
                filtered.terrainHeight);
        REQUIRE(std::abs(filtered.terrainHeight - block.terrainHeight) < 6.0);
    }
}

TEST_CASE("Column plans retain block-resolution lithology authority",
          "[worldgen][geology][lithology][column-plan][seam]") {
    ChunkGenerator generator(42);
    constexpr int64_t BASE_X = -32'272;
    constexpr int64_t BASE_Z = -32'768;
    for (int offset = 1; offset < CHUNK_EDGE; ++offset) {
        const int64_t x = BASE_X + offset;
        const int64_t z = BASE_Z + (offset * 5) % CHUNK_EDGE;
        const worldgen::SurfaceSample exact = generator.sampleSurface(x, z);
        const worldgen::SurfaceSample direct = generator.sampleFarSurface(x, z);
        REQUIRE(exact.geology.plateId == direct.geology.plateId);
        REQUIRE(exact.geology.crust == direct.geology.crust);
        REQUIRE(exact.geology.boundary == direct.geology.boundary);
        REQUIRE(exact.geology.rock == direct.geology.rock);
        REQUIRE(exact.geology.lithology.primary == direct.geology.lithology.primary);
        REQUIRE(exact.geology.lithology.secondary == direct.geology.lithology.secondary);
        REQUIRE(exact.geology.lithology.transition ==
                Catch::Approx(direct.geology.lithology.transition).margin(1.0 / 65535.0));
    }
}

TEST_CASE("Column plans retain block-resolution water authority and source support",
          "[worldgen][hydrology][column-plan][water][source][support][regression]") {
    struct Fixture {
        uint64_t seed;
        int64_t x;
        int64_t z;
        bool ocean;
        bool river;
        bool lake;
    };
    constexpr std::array fixtures = {
        Fixture{42, -557, 379, true, false, false},
        Fixture{764891, 22'000, -111'421, false, true, false},
        Fixture{42, -8'348, 2'281, false, false, true},
    };

    for (const Fixture fixture : fixtures) {
        CAPTURE(fixture.seed, fixture.x, fixture.z);
        ChunkGenerator generator(fixture.seed);
        const worldgen::SurfaceSample direct =
            generator.sampleFarSurface(fixture.x, fixture.z, worldgen::SurfaceFootprint::BLOCK_1);
        const worldgen::SurfaceSample planned = generator.sampleSurface(fixture.x, fixture.z);
        const worldgen::SurfaceSample exact = generator.sampleExactSurface(fixture.x, fixture.z);

        REQUIRE(direct.hydrology.ocean == fixture.ocean);
        REQUIRE(direct.hydrology.river == fixture.river);
        REQUIRE(direct.hydrology.lake == fixture.lake);
        REQUIRE(planned.hydrology.ocean == direct.hydrology.ocean);
        REQUIRE(planned.hydrology.river == direct.hydrology.river);
        REQUIRE(planned.hydrology.lake == direct.hydrology.lake);
        REQUIRE(planned.hydrology.delta == direct.hydrology.delta);
        REQUIRE(planned.hydrology.estuary == direct.hydrology.estuary);
        REQUIRE(planned.hydrology.brackish == direct.hydrology.brackish);
        REQUIRE(planned.hydrology.waterfall == direct.hydrology.waterfall);
        REQUIRE(planned.hydrology.waterBodyId == direct.hydrology.waterBodyId);
        REQUIRE(planned.waterSurface == Catch::Approx(direct.waterSurface).margin(1.0e-4));
        REQUIRE(planned.hydrology.surfaceElevation ==
                Catch::Approx(direct.hydrology.surfaceElevation).margin(1.0e-4));
        REQUIRE(exact.hydrology.ocean == direct.hydrology.ocean);
        REQUIRE(exact.hydrology.river == direct.hydrology.river);
        REQUIRE(exact.hydrology.lake == direct.hydrology.lake);
        REQUIRE(exact.hydrology.estuary == direct.hydrology.estuary);
        REQUIRE(exact.hydrology.brackish == direct.hydrology.brackish);
        REQUIRE(exact.waterSurface == Catch::Approx(direct.waterSurface).margin(1.0e-6));

        const int surfaceY = generator.surfaceYAt(fixture.x, fixture.z);
        const int plannedFloorY =
            static_cast<int>(std::ceil(direct.hydrology.surfaceElevation)) - 1;
        const int waterTopY = static_cast<int>(std::ceil(direct.waterSurface)) - 1;
        REQUIRE(surfaceY >= plannedFloorY);
        REQUIRE(surfaceY < waterTopY);
        size_t sourceCount = 0;
        for (int32_t sectionY = Chunk::worldToChunkY(surfaceY);
             sectionY <= Chunk::worldToChunkY(waterTopY); ++sectionY) {
            Chunk cube(
                ChunkPos{Chunk::worldToChunk(fixture.x), sectionY, Chunk::worldToChunk(fixture.z)});
            generator.generateCube(cube);
            const int firstY = std::max(surfaceY, sectionY * CHUNK_EDGE);
            const int lastY = std::min(waterTopY, sectionY * CHUNK_EDGE + CHUNK_EDGE - 1);
            for (int y = firstY; y <= lastY; ++y) {
                const BlockType block =
                    cube.getBlock(Chunk::worldToLocal(fixture.x), Chunk::worldToLocalY(y),
                                  Chunk::worldToLocal(fixture.z));
                if (y == surfaceY) {
                    REQUIRE(isSolid(block));
                    continue;
                }
                REQUIRE(block == BlockType::WATER);
                REQUIRE(cube.getFluidState(Chunk::worldToLocal(fixture.x), Chunk::worldToLocalY(y),
                                           Chunk::worldToLocal(fixture.z))
                            .isSource());
                ++sourceCount;
            }
        }
        REQUIRE(sourceCount == static_cast<size_t>(waterTopY - surfaceY));
    }
}

TEST_CASE("Exact cube emission performs no scalar basin sampling after its plan is built",
          "[worldgen][hydrology][column-plan][performance][regression]") {
    constexpr int64_t FALL_X = -8'240;
    constexpr int64_t FALL_Z = 3'088;
    const ColumnPos column{Chunk::worldToChunk(FALL_X), Chunk::worldToChunk(FALL_Z)};
    ChunkGenerator generator(42);
    const std::shared_ptr<const ColumnPlan> plan = generator.getColumnPlan(column);
    // Density interpolation at the positive cube faces consumes the east,
    // south, and southeast plan authorities. Warm those legitimate plan-build
    // dependencies before measuring ordinary cube emission.
    static_cast<void>(generator.getColumnPlan({column.x + 1, column.z}));
    static_cast<void>(generator.getColumnPlan({column.x, column.z + 1}));
    static_cast<void>(generator.getColumnPlan({column.x + 1, column.z + 1}));
    REQUIRE(
        plan->sample(Chunk::worldToLocal(FALL_X), Chunk::worldToLocal(FALL_Z)).hydrology.waterfall);

    const uint64_t scalarSamplesBefore = generator.basinCacheMetrics().scalarSampleCalls;
    REQUIRE(generator.sampleSurface(FALL_X, FALL_Z).hydrology.waterfall);
    static_cast<void>(generator.surfaceMaterialAt(FALL_X, FALL_Z));

    const int32_t deepSection = Chunk::worldToChunkY(WORLD_MIN_Y);
    REQUIRE_FALSE(plan->exposesSection(deepSection));
    Chunk cube(ChunkPos{column.x, deepSection, column.z});
    generator.generateCube(cube);

    const uint64_t scalarSamplesAfter = generator.basinCacheMetrics().scalarSampleCalls;
    REQUIRE(scalarSamplesAfter == scalarSamplesBefore);
}

TEST_CASE("Surface material palettes retain four bounded weighted entries",
          "[worldgen][material][palette]") {
    worldgen::SurfaceSample sample = materialFixture();
    sample.biome = {
        .primary = Biome::PLAINS, .secondary = Biome::VOLCANIC_BARREN, .transition = 0.36};
    const worldgen::surface_material::SurfaceMaterialPalette palette =
        worldgen::surface_material::materialPalette(sample, {}, false, false);
    REQUIRE(palette.count >= 1);
    REQUIRE(palette.count <= palette.entries.size());
    int totalWeight = 0;
    for (size_t index = 0; index < palette.count; ++index) {
        REQUIRE(palette.entries[index].material != BlockType::AIR);
        totalWeight += palette.entries[index].weight;
    }
    REQUIRE(totalWeight == 255);
}

TEST_CASE("Exact top blocks use the shared material palette and rank",
          "[worldgen][material][palette][lod][exact][determinism]") {
    ChunkGenerator generator(42);
    constexpr std::array<ColumnPos, 6> coordinates = {
        ColumnPos{0, 0},       ColumnPos{-32, -32},         ColumnPos{3'264, 480},
        ColumnPos{-23'904, 0}, ColumnPos{-104'448, 42'176}, ColumnPos{-14'208, 27'200},
    };

    for (const ColumnPos coordinate : coordinates) {
        const auto palette = generator.surfaceMaterialPaletteAt(coordinate.x, coordinate.z);
        const double rank = generator.farSurfaceMaterialRankAt(coordinate.x, coordinate.z);
        const BlockType expected = worldgen::surface_material::selectMaterial(palette, rank);
        CAPTURE(coordinate.x, coordinate.z, palette.count, rank);
        REQUIRE(generator.surfaceMaterialAt(coordinate.x, coordinate.z) == expected);
        REQUIRE(generatedBlockAt(generator, coordinate.x,
                                 generator.surfaceYAt(coordinate.x, coordinate.z),
                                 coordinate.z) == expected);
    }
}

TEST_CASE("Climate soil and suitability stitch exactly across cubic column faces",
          "[worldgen][climate][biome][column-plan][seam]") {
    ChunkGenerator generator(42);
    constexpr std::array<ColumnPos, 3> columns = {
        ColumnPos{0, 0},
        ColumnPos{-1, 2},
        ColumnPos{127, -65},
    };
    constexpr std::array<int, 4> offsets = {1, 7, 9, 15};
    for (const ColumnPos column : columns) {
        const auto center = generator.getColumnPlan(column);
        const auto east = generator.getColumnPlan({column.x + 1, column.z});
        const auto south = generator.getColumnPlan({column.x, column.z + 1});
        for (const int offset : offsets) {
            requireContinuousFieldsEqual(center->sample(CHUNK_EDGE, offset),
                                         east->sample(0, offset));
            requireContinuousFieldsEqual(center->sample(offset, CHUNK_EDGE),
                                         south->sample(offset, 0));
        }
    }
}

TEST_CASE("Macro climate and biome fields remain continuous across synthesis cell boundaries",
          "[worldgen][climate][biome][catchment][seam]") {
    worldgen::MacroGenerationSampler sampler(42);
    constexpr std::array<int64_t, 2> boundaries = {2'048, -8'192};
    for (const int64_t boundary : boundaries) {
        for (const bool alongX : {true, false}) {
            auto sample = [&](double offset) {
                return alongX ? sampler.sampleSurface(boundary + offset, 1008.0)
                              : sampler.sampleSurface(1008.0, boundary + offset);
            };
            const worldgen::SurfaceSample before = sample(-1.0);
            const worldgen::SurfaceSample on = sample(0.0);
            const worldgen::SurfaceSample after = sample(1.0);
            for (const auto& [first, second] : {std::pair{&before, &on}, std::pair{&on, &after}}) {
                REQUIRE(std::abs(first->climate.temperatureC - second->climate.temperatureC) < 1.0);
                REQUIRE(std::abs(first->climate.annualPrecipitationMm -
                                 second->climate.annualPrecipitationMm) < 120.0);
                REQUIRE(std::abs(first->soil.moisture - second->soil.moisture) < 0.12);
                REQUIRE(std::abs(first->soil.fertility - second->soil.fertility) < 0.12);
                for (size_t index = 0; index < first->suitability.scores.size(); ++index) {
                    REQUIRE(std::abs(first->suitability.scores[index] -
                                     second->suitability.scores[index]) < 0.18);
                }
            }
        }
    }

    const worldgen::SurfaceSample expected = sampler.sampleSurface(2048.0, 1008.0);
    sampler.clearBasinCache();
    const worldgen::SurfaceSample rebuilt = sampler.sampleSurface(2048.0, 1008.0);
    requireContinuousFieldsEqual(expected, rebuilt);
}

TEST_CASE("Former storage lines do not bias macro terrain derivatives",
          "[worldgen][geology][continuity][artifact]") {
    worldgen::MacroGenerationSampler sampler(42);
    for (const int64_t spacing : artifact::FORMER_GRID_SPACINGS) {
        const int64_t lineX = spacing;
        const int64_t centerZ = spacing / 2;
        for (const artifact::Field field : artifact::FIELDS) {
            const double boundaryEnergy =
                artifact::derivativeEnergy(sampler, field, lineX, centerZ);
            const double shiftedEnergy =
                artifact::nearbyDerivativeEnergy(sampler, field, lineX, centerZ);
            const double ratio = artifact::energyRatio(boundaryEnergy, shiftedEnergy);
            INFO("former spacing " << spacing << " field " << static_cast<int>(field)
                                   << " boundary energy " << boundaryEnergy << " shifted energy "
                                   << shiftedEnergy);
            REQUIRE(ratio >= artifact::DERIVATIVE_RATIO_MINIMUM);
            REQUIRE(ratio <= artifact::DERIVATIVE_RATIO_MAXIMUM);
        }
    }
}

TEST_CASE("Former storage lines have no structured orientation excess across world seeds",
          "[worldgen][geology][climate][continuity][artifact]") {
    constexpr std::array<uint64_t, 6> seeds = {42, 764'891, 1, 7, 12'345, 0xDEAD'BEEF};
    for (const uint64_t seed : seeds) {
        worldgen::MacroGenerationSampler sampler(seed);
        artifact::OrientationHistogram seedHistogram;
        for (const int64_t spacing : artifact::FORMER_GRID_SPACINGS) {
            artifact::OrientationHistogram spacingHistogram;
            for (const artifact::Field field : artifact::FIELDS) {
                artifact::add(spacingHistogram,
                              artifact::orientationHistogram(sampler, field, spacing, spacing / 2));
            }
            const double spacingRatio = artifact::structuredOrientationRatio(spacingHistogram);
            INFO("seed " << seed << " former spacing " << spacing
                         << " structured orientation ratio " << spacingRatio);
            REQUIRE(spacingRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);
            artifact::add(seedHistogram, spacingHistogram);
        }
        const double seedRatio = artifact::structuredOrientationRatio(seedHistogram);
        INFO("seed " << seed << " aggregate structured orientation ratio " << seedRatio);
        REQUIRE(seedRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);

        const artifact::OrientationBins globalHistogram =
            artifact::globalOrientationHistogram(sampler);
        const double globalRatio = artifact::structuredOrientationRatio(globalHistogram);
        INFO("seed " << seed << " global structured orientation ratio " << globalRatio);
        REQUIRE(globalRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);
    }
}

TEST_CASE("Former-line orientation analysis detects an aligned synthetic seam",
          "[worldgen][continuity][artifact]") {
    constexpr int64_t line = 64;
    auto alignedField = [](double x, double z) {
        const double smoothBackground = x * 0.025 + z * 0.017;
        return smoothBackground + (x >= static_cast<double>(line) ? 100.0 : 0.0);
    };
    const artifact::OrientationHistogram histogram =
        artifact::orientationHistogram(alignedField, line, 32);
    REQUIRE(artifact::structuredOrientationRatio(histogram) >
            artifact::STRUCTURED_ORIENTATION_LIMIT);
}

TEST_CASE("Orientation bias scales sparse priors with sample exposure",
          "[worldgen][continuity][artifact][unit]") {
    const artifact::OrientationBins former = {15, 11, 0, 0, 0, 0, 0, 7};
    const artifact::OrientationBins nearby = {120, 88, 0, 0, 0, 0, 0, 56};

    const std::array<double, 8> bias = artifact::orientationBias(former, nearby);
    for (const double value : bias)
        CHECK(value == Catch::Approx(1.0).margin(1.0e-12));
    CHECK(artifact::structuredOrientationRatio(bias) == Catch::Approx(1.0).margin(1.0e-12));
}

TEST_CASE("Orientation bias retains structured seam detection with scaled priors",
          "[worldgen][continuity][artifact][unit][negative-control]") {
    const artifact::OrientationBins former = {64, 0, 0, 0, 0, 0, 0, 0};
    const artifact::OrientationBins nearby = {8, 8, 8, 8, 8, 8, 8, 8};

    CHECK(artifact::structuredOrientationRatio(artifact::orientationBias(former, nearby)) >
          artifact::STRUCTURED_ORIENTATION_LIMIT);
}

TEST_CASE("Orientation bias preserves the equal-exposure prior",
          "[worldgen][continuity][artifact][unit]") {
    const artifact::OrientationBins former = {8, 4, 0, 0, 0, 0, 4, 0};
    const artifact::OrientationBins nearby = {4, 8, 0, 0, 0, 0, 0, 4};
    const std::array<double, 8> bias = artifact::orientationBias(former, nearby);

    for (size_t index = 0; index < bias.size(); ++index) {
        const double originalBias =
            (static_cast<double>(former[index]) + 1.0) / (static_cast<double>(nearby[index]) + 1.0);
        CHECK(bias[index] == Catch::Approx(originalBias).margin(1.0e-12));
    }
}

TEST_CASE("Former storage lines do not own untagged categorical boundaries",
          "[worldgen][hydrology][geology][material][continuity][artifact]") {
    ChunkGenerator generator(42);
    for (const int64_t spacing : artifact::FORMER_GRID_SPACINGS) {
        const int64_t lineX = spacing;
        const int64_t centerZ = spacing / 2;
        int lakeRun = 0;
        int materialRun = 0;
        int lithologyRun = 0;
        int longestLakeRun = 0;
        int longestMaterialRun = 0;
        int longestLithologyRun = 0;
        for (int64_t offset = -artifact::CATEGORICAL_HALF_WINDOW;
             offset <= artifact::CATEGORICAL_HALF_WINDOW; ++offset) {
            const int64_t z = centerZ + offset;
            const worldgen::SurfaceSample left = generator.sampleFarSurface(lineX - 1, z);
            const worldgen::SurfaceSample right = generator.sampleFarSurface(lineX + 1, z);
            const bool outlet = left.hydrology.river || right.hydrology.river ||
                                left.hydrology.waterfall || right.hydrology.waterfall ||
                                left.hydrology.delta || right.hydrology.delta;
            const bool lakeBoundary = !outlet && (left.hydrology.lake || right.hydrology.lake) &&
                                      (left.hydrology.lake != right.hydrology.lake ||
                                       left.hydrology.waterBodyId != right.hydrology.waterBodyId);
            lakeRun = lakeBoundary ? lakeRun + 1 : 0;
            longestLakeRun = std::max(longestLakeRun, lakeRun);

            const bool constrainedMaterial =
                hasHardSurfaceMaterialConstraint(left) || hasHardSurfaceMaterialConstraint(right);
            const bool materialBoundary =
                !constrainedMaterial && dominantSurfaceMaterial(generator, lineX - 1, z) !=
                                            dominantSurfaceMaterial(generator, lineX + 1, z);
            materialRun = materialBoundary ? materialRun + 1 : 0;
            longestMaterialRun = std::max(longestMaterialRun, materialRun);

            const bool lithologyBoundary =
                !isTaggedLithologyDiscontinuity(left.geology, right.geology) &&
                left.geology.lithology.primary != right.geology.lithology.primary;
            lithologyRun = lithologyBoundary ? lithologyRun + 1 : 0;
            longestLithologyRun = std::max(longestLithologyRun, lithologyRun);
        }
        INFO("former spacing " << spacing << " lake " << longestLakeRun << " material "
                               << longestMaterialRun << " lithology " << longestLithologyRun);
        REQUIRE(longestLakeRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(longestMaterialRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(longestLithologyRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
    }
}

TEST_CASE("Rock and water habitat effects fade through their physical boundaries",
          "[worldgen][climate][soil][biome][continuity]") {
    worldgen::MacroGenerationSampler sampler(42);
    worldgen::HydrologySample hydrology;
    hydrology.surfaceElevation = 80.0;
    hydrology.waterSurface = SEA_LEVEL;
    worldgen::ClimateFields climate;
    climate.temperatureC = 18.0;
    climate.annualPrecipitationMm = 650.0;
    climate.potentialEvapotranspirationMm = 700.0;
    climate.aridity = climate.potentialEvapotranspirationMm / climate.annualPrecipitationMm;

    worldgen::GeologySample sandstone;
    sandstone.rock = worldgen::RockType::SANDSTONE;
    sandstone.distanceToBoundary = 0.0;
    worldgen::GeologySample basalt = sandstone;
    basalt.rock = worldgen::RockType::BASALT;

    const worldgen::SoilSample sandstoneBoundary =
        sampler.sampleSoil(512.0, -768.0, sandstone, hydrology, climate);
    const worldgen::SoilSample basaltBoundary =
        sampler.sampleSoil(512.0, -768.0, basalt, hydrology, climate);
    REQUIRE(sandstoneBoundary.drainage == basaltBoundary.drainage);
    REQUIRE(sandstoneBoundary.moisture == basaltBoundary.moisture);
    REQUIRE(sandstoneBoundary.fertility == basaltBoundary.fertility);

    sandstone.distanceToBoundary = 768.0;
    basalt.distanceToBoundary = 768.0;
    const worldgen::SoilSample sandstoneInterior =
        sampler.sampleSoil(512.0, -768.0, sandstone, hydrology, climate);
    const worldgen::SoilSample basaltInterior =
        sampler.sampleSoil(512.0, -768.0, basalt, hydrology, climate);
    REQUIRE(sandstoneInterior.drainage != basaltInterior.drainage);
    REQUIRE(sandstoneInterior.fertility != basaltInterior.fertility);

    worldgen::HydrologySample channel;
    channel.channelWidth = 12.0;
    channel.channelDistance = 0.0;
    const double channelCenter = worldgen::climateWaterInfluence(channel);
    channel.channelDistance = 12.0;
    const double channelBank = worldgen::climateWaterInfluence(channel);
    channel.channelDistance = 30.0;
    const double channelExterior = worldgen::climateWaterInfluence(channel);
    REQUIRE(channelCenter > channelBank);
    REQUIRE(channelBank > channelExterior);
    REQUIRE(channelExterior == 0.0);

    worldgen::HydrologySample shallowOcean;
    shallowOcean.ocean = true;
    shallowOcean.surfaceElevation = SEA_LEVEL;
    REQUIRE(worldgen::climateWaterInfluence(shallowOcean) == 0.0);
    shallowOcean.surfaceElevation = SEA_LEVEL - 6.0;
    REQUIRE(worldgen::climateWaterInfluence(shallowOcean) > 0.0);
}

TEST_CASE("Biome blends retain every biome and cross score ties continuously",
          "[worldgen][climate][biome][reachability][transition]") {
    for (size_t index = 0; index < static_cast<size_t>(Biome::COUNT); ++index) {
        worldgen::BiomeSuitability suitability;
        suitability.scores[index] = 1.0;
        const worldgen::BiomeBlend blend =
            worldgen::MacroGenerationSampler::selectBiome(suitability);
        REQUIRE(blend.primary == static_cast<Biome>(index));
        REQUIRE(worldgen::biomeBlendWeight(blend, static_cast<Biome>(index)) == 1.0);
    }

    worldgen::BiomeSuitability before;
    before.scores[static_cast<size_t>(Biome::FOREST)] = 0.51;
    before.scores[static_cast<size_t>(Biome::PLAINS)] = 0.49;
    worldgen::BiomeSuitability tie;
    tie.scores[static_cast<size_t>(Biome::FOREST)] = 0.50;
    tie.scores[static_cast<size_t>(Biome::PLAINS)] = 0.50;
    worldgen::BiomeSuitability after;
    after.scores[static_cast<size_t>(Biome::FOREST)] = 0.49;
    after.scores[static_cast<size_t>(Biome::PLAINS)] = 0.51;

    const worldgen::BiomeBlend beforeBlend = worldgen::MacroGenerationSampler::selectBiome(before);
    const worldgen::BiomeBlend tieBlend = worldgen::MacroGenerationSampler::selectBiome(tie);
    const worldgen::BiomeBlend afterBlend = worldgen::MacroGenerationSampler::selectBiome(after);
    REQUIRE(worldgen::biomeBlendWeight(beforeBlend, Biome::PLAINS) == Catch::Approx(0.49));
    REQUIRE(worldgen::biomeBlendWeight(tieBlend, Biome::PLAINS) == Catch::Approx(0.50));
    REQUIRE(worldgen::biomeBlendWeight(afterBlend, Biome::PLAINS) == Catch::Approx(0.51));
}

TEST_CASE("Continuous ecotope influence tapers flora and material overlays",
          "[worldgen][ecotope][continuity]") {
    worldgen::SurfaceSample surface;
    surface.hydrology.channelWidth = 12.0;
    surface.hydrology.channelDistance = 2.0;
    const double nearRiver =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK);
    surface.hydrology.channelDistance = 16.0;
    const double farRiver =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK);
    REQUIRE(nearRiver > farRiver);
    REQUIRE(farRiver > 0.0);

    surface.slope = 0.45;
    const double gentle =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::CLIFF);
    surface.slope = 0.80;
    const double transitional =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::CLIFF);
    surface.slope = 1.15;
    const double cliff =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::CLIFF);
    REQUIRE(gentle < transitional);
    REQUIRE(transitional < cliff);
}

TEST_CASE("Coordinate-generated incision reaches a steep canyon ecotope",
          "[worldgen][hydrology][ecotope][canyon][reachability]") {
    constexpr int64_t x = -23'904;
    constexpr int64_t z = 0;
    ChunkGenerator generator(42);
    const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);

    REQUIRE(surface.hydrology.streamOrder >= 4);
    REQUIRE(surface.hydrology.erosionDepth >= 8.0);
    REQUIRE(surface.hydrology.channelGradient >= 0.012);
    REQUIRE(surface.slope >= 0.75);
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::CANYON));
}

TEST_CASE("Elevation ecotopes overlap continuously from valleys through exposed peaks",
          "[worldgen][ecotope][elevation][continuity][reachability]") {
    struct ElevationFixture {
        worldgen::Ecotope ecotope;
        double height;
        double temperature;
        double precipitation;
        double slope;
        double uplift;
    };
    constexpr std::array<ElevationFixture, 8> fixtures = {
        ElevationFixture{worldgen::Ecotope::VALLEY, 68.0, 16.0, 800.0, 0.08, 0.0},
        ElevationFixture{worldgen::Ecotope::FOOTHILL, 108.0, 12.0, 800.0, 0.35, 0.1},
        ElevationFixture{worldgen::Ecotope::MONTANE, 158.0, 7.0, 900.0, 0.40, 0.2},
        ElevationFixture{worldgen::Ecotope::SUBALPINE, 214.0, 1.0, 1000.0, 0.48, 0.3},
        ElevationFixture{worldgen::Ecotope::ALPINE_ZONE, 278.0, -5.0, 1000.0, 0.60, 0.5},
        ElevationFixture{worldgen::Ecotope::SNOWFIELD, 300.0, -8.0, 1200.0, 0.55, 0.5},
        ElevationFixture{worldgen::Ecotope::GLACIER, 340.0, -17.0, 1600.0, 0.48, 0.5},
        ElevationFixture{worldgen::Ecotope::EXPOSED_PEAK, 360.0, -8.0, 700.0, 1.20, 0.9},
    };

    for (const ElevationFixture& fixture : fixtures) {
        worldgen::SurfaceSample surface;
        surface.terrainHeight = fixture.height;
        surface.hydrology.surfaceElevation = fixture.height;
        surface.climate.temperatureC = fixture.temperature;
        surface.climate.annualPrecipitationMm = fixture.precipitation;
        surface.slope = fixture.slope;
        surface.geology.uplift = fixture.uplift;
        surface.soil.waterTable = fixture.height - 64.0;
        const double influence =
            worldgen::MacroGenerationSampler::ecotopeInfluence(surface, fixture.ecotope);
        INFO("elevation ecotope " << static_cast<uint32_t>(fixture.ecotope));
        REQUIRE(influence >= 0.28);
        REQUIRE(worldgen::hasEcotope(worldgen::MacroGenerationSampler::classifyEcotopes(surface),
                                     fixture.ecotope));
    }

    struct OverlapFixture {
        worldgen::Ecotope lower;
        worldgen::Ecotope upper;
        double height;
        double temperature;
        double precipitation;
        double slope;
        double uplift;
    };
    constexpr std::array<OverlapFixture, 7> overlaps = {
        OverlapFixture{worldgen::Ecotope::VALLEY, worldgen::Ecotope::FOOTHILL, 88.0, 12.0, 800.0,
                       0.25, 0.1},
        OverlapFixture{worldgen::Ecotope::FOOTHILL, worldgen::Ecotope::MONTANE, 132.0, 9.0, 850.0,
                       0.35, 0.2},
        OverlapFixture{worldgen::Ecotope::MONTANE, worldgen::Ecotope::SUBALPINE, 184.0, 4.0, 900.0,
                       0.40, 0.3},
        OverlapFixture{worldgen::Ecotope::SUBALPINE, worldgen::Ecotope::ALPINE_ZONE, 246.0, -2.0,
                       1050.0, 0.50, 0.5},
        OverlapFixture{worldgen::Ecotope::ALPINE_ZONE, worldgen::Ecotope::SNOWFIELD, 294.0, -7.0,
                       1200.0, 0.60, 0.6},
        OverlapFixture{worldgen::Ecotope::SNOWFIELD, worldgen::Ecotope::GLACIER, 326.0, -14.0,
                       1550.0, 0.65, 0.7},
        OverlapFixture{worldgen::Ecotope::GLACIER, worldgen::Ecotope::EXPOSED_PEAK, 350.0, -17.0,
                       1600.0, 0.80, 0.85},
    };
    for (const OverlapFixture& overlap : overlaps) {
        worldgen::SurfaceSample surface;
        surface.terrainHeight = overlap.height;
        surface.hydrology.surfaceElevation = overlap.height;
        surface.climate.temperatureC = overlap.temperature;
        surface.climate.annualPrecipitationMm = overlap.precipitation;
        surface.slope = overlap.slope;
        surface.geology.uplift = overlap.uplift;
        surface.soil.waterTable = overlap.height - 64.0;
        INFO("elevation transition " << static_cast<uint32_t>(overlap.lower) << " to "
                                     << static_cast<uint32_t>(overlap.upper));
        REQUIRE(worldgen::MacroGenerationSampler::ecotopeInfluence(surface, overlap.lower) > 0.28);
        REQUIRE(worldgen::MacroGenerationSampler::ecotopeInfluence(surface, overlap.upper) > 0.28);
    }

    worldgen::SurfaceSample transition;
    transition.terrainHeight = overlaps[2].height;
    transition.hydrology.surfaceElevation = transition.terrainHeight;
    transition.climate.temperatureC = overlaps[2].temperature;
    transition.climate.annualPrecipitationMm = overlaps[2].precipitation;
    transition.slope = overlaps[2].slope;
    transition.geology.uplift = overlaps[2].uplift;
    transition.soil.waterTable = transition.terrainHeight - 64.0;

    worldgen::SurfaceSample adjacent = transition;
    adjacent.terrainHeight += 1.0;
    adjacent.hydrology.surfaceElevation += 1.0;
    REQUIRE(std::abs(worldgen::MacroGenerationSampler::ecotopeInfluence(
                         transition, worldgen::Ecotope::MONTANE) -
                     worldgen::MacroGenerationSampler::ecotopeInfluence(
                         adjacent, worldgen::Ecotope::MONTANE)) < 0.03);
}

TEST_CASE("Karst and transform fault density fields are bounded and deterministic",
          "[worldgen][geology][caves]") {
    DensityField density(42);
    ColumnShape column;
    column.height = 160.0;
    column.detailAmp = 0.0;
    column.entrance = -1.0;
    column.ravineFloor = column.height;

    worldgen::GeologySample granite;
    granite.rock = worldgen::RockType::GRANITE;
    worldgen::GeologySample limestone = granite;
    limestone.rock = worldgen::RockType::LIMESTONE;
    worldgen::GeologySample transform = granite;
    transform.boundary = worldgen::PlateBoundary::TRANSFORM;
    transform.faultStrength = 1.0;

    bool foundKarstCarve = false;
    bool foundFaultCarve = false;
    for (int z = -192; z <= 192 && (!foundKarstCarve || !foundFaultCarve); z += 8) {
        for (int x = -192; x <= 192 && (!foundKarstCarve || !foundFaultCarve); x += 8) {
            const DensityColumnContext graniteContext = density.columnContext(x, z, granite);
            const DensityColumnContext limestoneContext = density.columnContext(x, z, limestone);
            const DensityColumnContext transformContext = density.columnContext(x, z, transform);
            for (int y = 40; y <= 128; y += 8) {
                const double ordinary = density.density(x, y, z, column, graniteContext);
                const double karst = density.density(x, y, z, column, limestoneContext);
                const double fault = density.density(x, y, z, column, transformContext);
                for (double value : {ordinary, karst, fault}) {
                    REQUIRE(std::isfinite(value));
                    REQUIRE(value >= -DENSITY_CAP);
                    REQUIRE(value <= DENSITY_CAP);
                }
                foundKarstCarve = foundKarstCarve || (ordinary > 1.0 && karst < -0.25);
                foundFaultCarve = foundFaultCarve || (ordinary > 1.0 && fault < -0.25);
            }
        }
    }

    REQUIRE(foundKarstCarve);
    REQUIRE(foundFaultCarve);
    const DensityColumnContext repeatedContext = density.columnContext(64.0, -96.0, limestone);
    REQUIRE(density.density(64.0, 80.0, -96.0, column, repeatedContext) ==
            density.density(64.0, 80.0, -96.0, column, repeatedContext));
}

TEST_CASE("Public surface samples expose deterministic cave ecotopes",
          "[worldgen][geology][ecotope]") {
    ChunkGenerator generator(42);
    constexpr std::array<std::pair<int64_t, int64_t>, 4> limestoneCaves = {
        std::pair<int64_t, int64_t>{7168, -32768},
        {9728, -32768},
        {14848, -32768},
        {7168, -32256},
    };

    std::pair<int64_t, int64_t> selected{};
    bool found = false;
    for (const auto& coordinate : limestoneCaves) {
        const worldgen::SurfaceSample far =
            generator.sampleFarSurface(coordinate.first, coordinate.second);
        if (far.geology.rock != worldgen::RockType::LIMESTONE ||
            !worldgen::hasEcotope(far.ecotopes, worldgen::Ecotope::CAVE)) {
            continue;
        }
        selected = coordinate;
        found = true;
        break;
    }
    REQUIRE(found);

    const worldgen::SurfaceSample exact = generator.sampleSurface(selected.first, selected.second);
    const worldgen::SurfaceSample repeated =
        generator.sampleSurface(selected.first, selected.second);
    REQUIRE(worldgen::hasEcotope(exact.ecotopes, worldgen::Ecotope::CAVE));
    REQUIRE(exact.terrainHeight == repeated.terrainHeight);
    REQUIRE(exact.ecotopes == repeated.ecotopes);
}

TEST_CASE("Vertical strata follow the sampled limestone geology", "[worldgen][geology][strata]") {
    ChunkGenerator generator(42);
    constexpr int64_t x = -26355;
    constexpr int64_t z = 29672;
    const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);
    REQUIRE(surface.geology.rock == worldgen::RockType::LIMESTONE);

    const int32_t firstSection =
        Chunk::worldToChunkY(static_cast<int32_t>(std::floor(surface.terrainHeight)) - 72);
    int limestone = 0;
    int sandstone = 0;
    int clay = 0;
    for (int offset = 0; offset < 3; ++offset) {
        Chunk cube(ChunkPos{Chunk::worldToChunk(x), firstSection + offset, Chunk::worldToChunk(z)});
        generator.generate(cube);
        const int localX = Chunk::worldToLocal(x);
        const int localZ = Chunk::worldToLocal(z);
        for (int localY = 0; localY < CHUNK_EDGE; ++localY) {
            const BlockType block = cube.getBlock(localX, localY, localZ);
            limestone += block == BlockType::LIMESTONE;
            sandstone += block == BlockType::SANDSTONE;
            clay += block == BlockType::CLAY;
        }
    }

    REQUIRE(limestone >= 12);
    REQUIRE(sandstone >= 1);
    REQUIRE(clay >= 1);
}

TEST_CASE("Alpine scrub generation is order independent", "[worldgen][ecology][flora]") {
    constexpr int64_t x = -80'225;
    constexpr int64_t z = 124'607;
    ChunkGenerator generator(42);
    const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::ALPINE_ZONE));

    const ChunkPos lowerPos{Chunk::worldToChunk(x),
                            Chunk::worldToChunkY(generator.surfaceYAt(x, z)),
                            Chunk::worldToChunk(z)};
    const ChunkPos upperPos{lowerPos.x, lowerPos.y + 1, lowerPos.z};
    Chunk lowerForward(lowerPos);
    Chunk upperForward(upperPos);
    generator.generate(lowerForward);
    generator.generate(upperForward);

    Chunk upperReverse(upperPos);
    Chunk lowerReverse(lowerPos);
    generator.generate(upperReverse);
    generator.generate(lowerReverse);

    REQUIRE(lowerForward.copyBlocks() == lowerReverse.copyBlocks());
    REQUIRE(upperForward.copyBlocks() == upperReverse.copyBlocks());
    int scrubCount = 0;
    for (BlockType block : lowerForward.copyBlocks())
        scrubCount += block == BlockType::SHRUB;
    for (BlockType block : upperForward.copyBlocks())
        scrubCount += block == BlockType::SHRUB;
    REQUIRE(scrubCount >= 8);
}

TEST_CASE("Coordinate-generated snow peaks combine cold climate and exposed alpine ground",
          "[worldgen][climate][ecotope][snow][reachability]") {
    constexpr int64_t x = -81'896;
    constexpr int64_t z = 126'960;
    ChunkGenerator generator(42);
    const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);

    REQUIRE(surface.terrainHeight >= 340.0);
    REQUIRE(surface.climate.temperatureC < 0.0);
    REQUIRE(surface.climate.annualPrecipitationMm > 1'000.0);
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::ALPINE_ZONE));
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::SNOWFIELD));
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::EXPOSED_PEAK));
    REQUIRE(generator.surfaceMaterialAt(x, z) == BlockType::SNOW);
    REQUIRE(generatedBlockAt(generator, x, generator.surfaceYAt(x, z), z) == BlockType::SNOW);
}

TEST_CASE("Hotspot fields and emitted cones share the source plate velocity",
          "[worldgen][geology][hotspot]") {
    constexpr uint32_t seed = 42;
    worldgen::MacroGenerationSampler macro(seed);
    ChunkGenerator generator(seed);

    worldgen::HotspotChainPrimitive selected;
    std::vector<VolcanoPrimitive> volcanoes;
    for (int64_t cellZ = -8; cellZ <= 8 && volcanoes.empty(); ++cellZ) {
        for (int64_t cellX = -8; cellX <= 8 && volcanoes.empty(); ++cellX) {
            const worldgen::HotspotChainPrimitive candidate = macro.hotspotChain(cellX, cellZ);
            if (!candidate.active)
                continue;
            selected = candidate;
            volcanoes = generator.hotspotVolcanoesForCell(cellX, cellZ);
        }
    }
    REQUIRE(selected.active);
    REQUIRE_FALSE(volcanoes.empty());

    const double velocityLength =
        std::hypot(selected.sourcePlateVelocity.x, selected.sourcePlateVelocity.z);
    REQUIRE(velocityLength > 0.0);
    REQUIRE(selected.direction.x ==
            Catch::Approx(-selected.sourcePlateVelocity.x / velocityLength).margin(1.0e-12));
    REQUIRE(selected.direction.z ==
            Catch::Approx(-selected.sourcePlateVelocity.z / velocityLength).margin(1.0e-12));

    const worldgen::Vector2d transverse{-selected.direction.z, selected.direction.x};
    for (const VolcanoPrimitive& volcano : volcanoes) {
        const double offsetX = volcano.centerX - selected.sourceX;
        const double offsetZ = volcano.centerZ - selected.sourceZ;
        const double along = offsetX * selected.direction.x + offsetZ * selected.direction.z;
        const double across = offsetX * transverse.x + offsetZ * transverse.z;
        REQUIRE(along >= -1.0e-9);
        REQUIRE(along <= selected.length + 1.0e-9);
        REQUIRE(std::abs(across) <= 150.0 + 1.0e-9);
    }

    const worldgen::GeologySample sourceGeology =
        macro.sampleGeology(selected.sourceX, selected.sourceZ);
    REQUIRE(sourceGeology.hotspotInfluence == Catch::Approx(1.0).margin(1.0e-12));

    const int64_t sampleX = static_cast<int64_t>(std::llround(volcanoes.front().centerX));
    const int64_t sampleZ = static_cast<int64_t>(std::llround(volcanoes.front().centerZ));
    const worldgen::SurfaceSample beforeEviction = generator.sampleFarSurface(sampleX, sampleZ);
    generator.clearMacroCaches();
    const worldgen::SurfaceSample afterEviction = generator.sampleFarSurface(sampleX, sampleZ);
    REQUIRE(beforeEviction.terrainHeight == afterEviction.terrainHeight);
    REQUIRE(beforeEviction.geology.hotspotInfluence == afterEviction.geology.hotspotInfluence);
    REQUIRE(beforeEviction.biome.primary == afterEviction.biome.primary);
}

TEST_CASE("Oceanic cones become internally consistent generated islands",
          "[worldgen][geology][volcano][island]") {
    constexpr uint32_t seed = 42;
    worldgen::MacroGenerationSampler macro(seed);
    ChunkGenerator generator(seed);
    constexpr int64_t ISLAND_CELL_X = -28;
    constexpr int64_t ISLAND_CELL_Z = -20;
    constexpr int64_t volcanoX = -449'951;
    constexpr int64_t volcanoZ = -313'477;
    constexpr int64_t islandX = -449'951;
    constexpr int64_t islandZ = -313'477;
    const std::vector<VolcanoPrimitive> volcanoes =
        generator.hotspotVolcanoesForCell(ISLAND_CELL_X, ISLAND_CELL_Z);
    const auto islandVolcano =
        std::ranges::find_if(volcanoes, [=](const VolcanoPrimitive& volcano) {
            return static_cast<int64_t>(std::llround(volcano.centerX)) == volcanoX &&
                   static_cast<int64_t>(std::llround(volcano.centerZ)) == volcanoZ;
        });
    REQUIRE(islandVolcano != volcanoes.end());
    REQUIRE(std::hypot(static_cast<double>(islandX) + 0.5 - islandVolcano->centerX,
                       static_cast<double>(islandZ) + 0.5 - islandVolcano->centerZ) <
            islandVolcano->radius);
    REQUIRE(macro.sampleHydrology(islandX, islandZ).ocean);
    REQUIRE(macro.sampleGeology(islandX, islandZ).crust == worldgen::CrustType::OCEANIC);

    const worldgen::SurfaceSample surface = generator.sampleSurface(islandX, islandZ);
    REQUIRE(surface.terrainHeight >= SEA_LEVEL);
    REQUIRE_FALSE(surface.hydrology.ocean);
    REQUIRE_FALSE(surface.hydrology.lake);
    REQUIRE_FALSE(surface.hydrology.river);
    REQUIRE_FALSE(isOceanBiome(surface.biome.primary));
    REQUIRE_FALSE(isOceanBiome(surface.biome.secondary));
    REQUIRE(surface.climate.aridity == Catch::Approx(surface.climate.potentialEvapotranspirationMm /
                                                     surface.climate.annualPrecipitationMm));
    REQUIRE(surface.soil.waterTable < surface.terrainHeight);

    const int surfaceY = generator.surfaceYAt(islandX, islandZ);
    REQUIRE(surfaceY >= SEA_LEVEL);
    REQUIRE(generatedBlockAt(generator, islandX, surfaceY, islandZ) != BlockType::WATER);
    REQUIRE(generatedBlockAt(generator, islandX, surfaceY + 1, islandZ) != BlockType::WATER);

    generator.clearMacroCaches();
    const worldgen::SurfaceSample rebuilt = generator.sampleSurface(islandX, islandZ);
    REQUIRE(rebuilt.terrainHeight == surface.terrainHeight);
    REQUIRE(rebuilt.hydrology.ocean == surface.hydrology.ocean);
    REQUIRE(rebuilt.climate.temperatureC == surface.climate.temperatureC);
    REQUIRE(rebuilt.soil.moisture == surface.soil.moisture);
    REQUIRE(rebuilt.biome.primary == surface.biome.primary);
}

TEST_CASE("Crater lake samples and generated water share final dependent state",
          "[worldgen][geology][volcano][lake]") {
    constexpr int64_t x = 23'029;
    constexpr int64_t z = -111'486;
    ChunkGenerator generator(764891);
    const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);

    REQUIRE(surface.hydrology.lake);
    REQUIRE(surface.hydrology.endorheic);
    REQUIRE_FALSE(surface.hydrology.ocean);
    REQUIRE_FALSE(surface.hydrology.river);
    REQUIRE(surface.waterSurface == surface.hydrology.waterSurface);
    REQUIRE(surface.hydrology.lakeDepth ==
            Catch::Approx(surface.waterSurface - surface.terrainHeight));
    REQUIRE(surface.waterSurface > surface.terrainHeight);
    REQUIRE_FALSE(isOceanBiome(surface.biome.primary));
    REQUIRE(surface.climate.aridity == Catch::Approx(surface.climate.potentialEvapotranspirationMm /
                                                     surface.climate.annualPrecipitationMm));
    REQUIRE(surface.biome.primary ==
            worldgen::MacroGenerationSampler::selectBiome(surface.suitability).primary);

    const int surfaceY = generator.surfaceYAt(x, z);
    const int waterTopY = static_cast<int>(std::ceil(surface.waterSurface)) - 1;
    REQUIRE(waterTopY > surfaceY);
    REQUIRE(generatedBlockAt(generator, x, surfaceY, z) == BlockType::BASALT);
    REQUIRE(generatedBlockAt(generator, x, surfaceY + 1, z) == BlockType::WATER);
    REQUIRE(generatedBlockAt(generator, x, waterTopY, z) == BlockType::WATER);
    REQUIRE(generatedBlockAt(generator, x, waterTopY + 1, z) == BlockType::AIR);

    const int32_t waterSection = Chunk::worldToChunkY(waterTopY);
    REQUIRE(generator.getColumnPlan({Chunk::worldToChunk(x), Chunk::worldToChunk(z)})
                ->exposesSection(waterSection));
}

TEST_CASE("Exact cubes and far terrain share aligned material regions",
          "[worldgen][geology][material][lod][seam]") {
    ChunkGenerator generator(42);
    constexpr std::array<ColumnPos, 7> coordinates = {
        ColumnPos{0, 0},         ColumnPos{-32, -32},       ColumnPos{3264, 480},
        ColumnPos{-23904, 0},    ColumnPos{-104448, 42176}, ColumnPos{-14208, 27200},
        ColumnPos{7040, -32768},
    };

    std::array<BlockType, coordinates.size()> forward{};
    for (size_t index = 0; index < coordinates.size(); ++index) {
        const ColumnPos coordinate = coordinates[index];
        const BlockType material = generator.surfaceMaterialAt(coordinate.x, coordinate.z);
        const int surfaceY = generator.surfaceYAt(coordinate.x, coordinate.z);
        INFO("coordinate " << coordinate.x << ',' << coordinate.z << " surface " << surfaceY
                           << " material " << static_cast<int>(material));
        REQUIRE(generatedBlockAt(generator, coordinate.x, surfaceY, coordinate.z) == material);
        forward[index] = material;
    }

    generator.clearMacroCaches();
    for (size_t reverse = coordinates.size(); reverse-- > 0;) {
        REQUIRE(generator.surfaceMaterialAt(coordinates[reverse].x, coordinates[reverse].z) ==
                forward[reverse]);
    }
}

TEST_CASE("Fine far materials preserve exact wet and dry cube margins",
          "[worldgen][geology][material][water][lod][seam][regression]") {
    struct MarginFixture {
        int64_t x;
        int64_t z;
        int spacing;
        bool submerged;
    };
    constexpr std::array<MarginFixture, 5> MARGINS = {{
        {-8'064, 2'496, 32, true},
        {-9'120, 2'944, 32, false},
        {-13'760, 832, 64, true},
        {-13'696, 832, 64, false},
        {-13'760, 896, 64, false},
    }};
    ChunkGenerator generator(42);

    std::array<BlockType, MARGINS.size()> exactMaterials{};
    std::array<BlockType, MARGINS.size()> coarseMaterials{};
    for (size_t fixtureIndex = 0; fixtureIndex < MARGINS.size(); ++fixtureIndex) {
        const MarginFixture& fixture = MARGINS[fixtureIndex];
        INFO("coordinate " << fixture.x << ',' << fixture.z << " spacing " << fixture.spacing);
        REQUIRE(world_coord::floorMod(fixture.x, static_cast<int64_t>(fixture.spacing)) == 0);
        REQUIRE(world_coord::floorMod(fixture.z, static_cast<int64_t>(fixture.spacing)) == 0);
        const worldgen::SurfaceSample exact = generator.sampleExactSurface(fixture.x, fixture.z);
        const int surfaceY = generator.surfaceYAt(fixture.x, fixture.z);
        const int waterTopY = static_cast<int>(std::ceil(exact.waterSurface)) - 1;
        const bool hasWater =
            exact.hydrology.ocean || exact.hydrology.river || exact.hydrology.lake;
        REQUIRE((hasWater && surfaceY < waterTopY) == fixture.submerged);
        const BlockType exactMaterial = generator.surfaceMaterialAt(fixture.x, fixture.z);
        const BlockType coarseMaterial = generator.farSurfaceMaterialAt(fixture.x, fixture.z);
        exactMaterials[fixtureIndex] = exactMaterial;
        coarseMaterials[fixtureIndex] = coarseMaterial;
        REQUIRE(exactMaterial != BlockType::AIR);
        REQUIRE(coarseMaterial != BlockType::AIR);
        REQUIRE(generatedBlockAt(generator, fixture.x, surfaceY, fixture.z) == exactMaterial);
        if (fixture.submerged) {
            const auto isSedimentOrOutcrop = [](BlockType material) {
                return material == BlockType::MUD || material == BlockType::CLAY ||
                       material == BlockType::SILT || material == BlockType::SAND ||
                       material == BlockType::GRAVEL || material == BlockType::BASALT ||
                       material == BlockType::ANDESITE;
            };
            REQUIRE(isSedimentOrOutcrop(exactMaterial));
            const worldgen::SurfaceSample coarseSurface =
                generator.sampleFarSurface(fixture.x, fixture.z);
            if (worldgen::surface_material::submerged(coarseSurface)) {
                REQUIRE(isSedimentOrOutcrop(coarseMaterial));
            }
            REQUIRE(generatedBlockAt(generator, fixture.x, surfaceY + 1, fixture.z) ==
                    BlockType::WATER);
        }
    }

    generator.clearMacroCaches();
    for (size_t reverse = MARGINS.size(); reverse-- > 0;) {
        REQUIRE(generator.surfaceMaterialAt(MARGINS[reverse].x, MARGINS[reverse].z) ==
                exactMaterials[reverse]);
        REQUIRE(generator.farSurfaceMaterialAt(MARGINS[reverse].x, MARGINS[reverse].z) ==
                coarseMaterials[reverse]);
    }
}

TEST_CASE("Volcanic deposits cross chunk seams without material confetti",
          "[worldgen][geology][material][coherence][seam]") {
    ChunkGenerator generator(42);
    constexpr int64_t z = -23'060;
    constexpr int64_t firstX = -10'080;
    std::array<BlockType, 17> forward{};
    int previousBasaltWeight = -1;
    int materialTransitions = 0;
    for (size_t index = 0; index < forward.size(); ++index) {
        const int64_t x = firstX + static_cast<int64_t>(index);
        const auto palette = generator.surfaceMaterialPaletteAt(x, z);
        const int basaltWeight = surfacePaletteWeight(palette, BlockType::BASALT);
        const int ashWeight = surfacePaletteWeight(palette, BlockType::VOLCANIC_ASH);
        CAPTURE(x, basaltWeight, ashWeight);
        REQUIRE(basaltWeight > 0);
        REQUIRE(ashWeight > 0);
        REQUIRE(basaltWeight + ashWeight == 255);
        if (previousBasaltWeight >= 0) {
            REQUIRE(std::abs(basaltWeight - previousBasaltWeight) <= 8);
        }
        previousBasaltWeight = basaltWeight;

        forward[index] = generator.surfaceMaterialAt(x, z);
        REQUIRE((forward[index] == BlockType::BASALT || forward[index] == BlockType::VOLCANIC_ASH));
        if (index > 0 && forward[index] != forward[index - 1])
            ++materialTransitions;
    }
    REQUIRE(materialTransitions <= 4);
    generator.clearMacroCaches();
    for (size_t reverse = forward.size(); reverse-- > 0;) {
        REQUIRE(generator.surfaceMaterialAt(firstX + static_cast<int64_t>(reverse), z) ==
                forward[reverse]);
    }
}

TEST_CASE("Implicit strata stay coherent across column boundaries",
          "[worldgen][geology][strata][seam]") {
    ChunkGenerator generator(42);
    constexpr int64_t rightX = -32'256;
    constexpr int64_t z = -32'768;
    const worldgen::SurfaceSample leftSurface = generator.sampleSurface(rightX - 1, z);
    const worldgen::SurfaceSample rightSurface = generator.sampleSurface(rightX, z);
    REQUIRE(leftSurface.geology.plateId == rightSurface.geology.plateId);
    REQUIRE(leftSurface.geology.rock == rightSurface.geology.rock);

    const int maximumY =
        std::min(generator.surfaceYAt(rightX - 1, z), generator.surfaceYAt(rightX, z)) - 20;
    const int minimumY = maximumY - 63;
    const std::vector<BlockType> left =
        generatedColumn(generator, rightX - 1, minimumY, maximumY, z);
    const std::vector<BlockType> right = generatedColumn(generator, rightX, minimumY, maximumY, z);
    REQUIRE(left.size() == right.size());

    size_t comparable = 0;
    size_t mismatches = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        if (!isStrataMaterial(left[index]) || !isStrataMaterial(right[index]))
            continue;
        ++comparable;
        mismatches += left[index] != right[index];
    }
    INFO("comparable strata " << comparable << " mismatches " << mismatches);
    REQUIRE(comparable >= 40);
    REQUIRE(mismatches <= 10);
}

TEST_CASE("Continental volcanic arcs emit reachable andesite intrusions",
          "[worldgen][geology][material][strata][andesite][reachability]") {
    constexpr int64_t x = -26'304;
    constexpr int64_t z = 29'696;
    ChunkGenerator generator(42);
    const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);

    REQUIRE(surface.geology.crust == worldgen::CrustType::CONTINENTAL);
    REQUIRE(surface.geology.boundary == worldgen::PlateBoundary::CONVERGENT);
    REQUIRE(surface.geology.rock == worldgen::RockType::GRANITE);
    REQUIRE(surface.geology.uplift > 0.9);
    const int surfaceY = generator.surfaceYAt(x, z);
    REQUIRE(surfaceY > WORLD_MIN_Y + 96);
    int firstAndesiteY = WORLD_MIN_Y;
    for (int y = surfaceY - 96; y <= surfaceY - 8; ++y) {
        if (generatedBlockAt(generator, x, y, z) != BlockType::ANDESITE)
            continue;
        firstAndesiteY = y;
        break;
    }
    REQUIRE(firstAndesiteY > WORLD_MIN_Y);
    int andesiteCount = 0;
    for (int y = firstAndesiteY; y < firstAndesiteY + 10; ++y)
        andesiteCount += generatedBlockAt(generator, x, y, z) == BlockType::ANDESITE;
    REQUIRE(andesiteCount >= 3);
}

TEST_CASE("Obsidian conduits are reachable but remain rarer than basalt",
          "[worldgen][geology][material][rarity]") {
    ChunkGenerator generator(764891);
    Chunk conduit(ChunkPos{Chunk::worldToChunk(23'029), Chunk::worldToChunkY(192),
                           Chunk::worldToChunk(-111'486)});
    generator.generate(conduit);

    size_t obsidian = 0;
    size_t basalt = 0;
    for (const BlockType block : conduit.copyBlocks()) {
        obsidian += block == BlockType::OBSIDIAN;
        basalt += block == BlockType::BASALT;
    }
    REQUIRE(obsidian > 0);
    REQUIRE(obsidian < basalt);
    REQUIRE(obsidian < CHUNK_VOLUME / 4);
}

TEST_CASE("Volcanic ground suppresses ordinary surface flora",
          "[worldgen][geology][material][flora]") {
    ChunkGenerator generator(42);
    constexpr int64_t x = -10'066;
    constexpr int64_t z = -23'060;
    const auto palette = generator.surfaceMaterialPaletteAt(x, z);
    REQUIRE(surfacePaletteWeight(palette, BlockType::VOLCANIC_ASH) > 0);
    REQUIRE(surfacePaletteWeight(palette, BlockType::BASALT) > 0);
    const BlockType volcanicGround = generator.surfaceMaterialAt(x, z);
    REQUIRE((volcanicGround == BlockType::VOLCANIC_ASH || volcanicGround == BlockType::BASALT));
    const int surfaceY = generator.surfaceYAt(x, z);
    REQUIRE(generatedBlockAt(generator, x, surfaceY, z) == volcanicGround);
    const BlockType above = generatedBlockAt(generator, x, surfaceY + 1, z);
    REQUIRE(above != BlockType::TALL_GRASS);
    REQUIRE(above != BlockType::FLOWER_YELLOW);
    REQUIRE(above != BlockType::FLOWER_RED);
    REQUIRE(above != BlockType::FLOWER_BLUE);
    REQUIRE(above != BlockType::FERN);
    REQUIRE(above != BlockType::SHRUB);

    constexpr int64_t basaltX = -10'066;
    constexpr int64_t basaltZ = -23'060;
    REQUIRE(generator.surfaceMaterialAt(basaltX, basaltZ) == BlockType::BASALT);
    const worldgen::SurfaceSample basalt = generator.sampleSurface(basaltX, basaltZ);
    REQUIRE(worldgen::hasEcotope(basalt.ecotopes, worldgen::Ecotope::GEOTHERMAL));
    const int basaltSurfaceY = generator.surfaceYAt(basaltX, basaltZ);
    const BlockType basaltAbove = generatedBlockAt(generator, basaltX, basaltSurfaceY + 1, basaltZ);
    for (const BlockType log :
         {BlockType::LOG, BlockType::BIRCH_LOG, BlockType::SPRUCE_LOG, BlockType::ACACIA_LOG,
          BlockType::JUNGLE_LOG, BlockType::MANGROVE_LOG, BlockType::PALM_LOG,
          BlockType::WILLOW_LOG}) {
        REQUIRE(basaltAbove != log);
    }
}

TEST_CASE("High elevation flora does not root directly in bare sandstone scree",
          "[worldgen][geology][material][flora][substrate][regression]") {
    constexpr int64_t x = -80'320;
    constexpr int64_t z = 124'992;
    ChunkGenerator generator(42);
    const worldgen::SurfaceSample surface = generator.sampleExactSurface(x, z);

    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::SCREE));
    REQUIRE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::EXPOSED_PEAK));
    REQUIRE_FALSE(worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL));
    const auto palette = generator.surfaceMaterialPaletteAt(x, z);
    REQUIRE(surfacePaletteWeight(palette, BlockType::SANDSTONE) > 0);
    REQUIRE(generator.surfaceMaterialAt(x, z) == BlockType::SANDSTONE);
    const int surfaceY = generator.surfaceYAt(x, z);
    REQUIRE(generatedBlockAt(generator, x, surfaceY, z) == BlockType::SANDSTONE);
    REQUIRE(generatedBlockAt(generator, x, surfaceY + 1, z) == BlockType::AIR);
}

#include "world/chunk_generator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <queue>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

template <typename Value> void fingerprintValue(uint64_t& hash, Value value) {
    uint64_t bits = 0;
    if constexpr (std::is_same_v<Value, double>) {
        bits = std::bit_cast<uint64_t>(value);
    } else if constexpr (std::is_same_v<Value, float>) {
        bits = std::bit_cast<uint32_t>(value);
    } else if constexpr (std::is_enum_v<Value>) {
        bits = static_cast<uint64_t>(value);
    } else {
        bits = static_cast<uint64_t>(value);
    }
    hash ^= bits;
    hash *= 1099511628211ULL;
}

uint64_t surfaceFingerprint(const worldgen::SurfaceSample& surface) {
    uint64_t hash = 1469598103934665603ULL;
    const auto value = [&](auto field) { fingerprintValue(hash, field); };
    value(surface.geology.plateId);
    value(surface.geology.crust);
    value(surface.geology.boundary);
    value(surface.geology.rock);
    value(surface.geology.lithology.primary);
    value(surface.geology.lithology.secondary);
    value(surface.geology.lithology.transition);
    value(surface.geology.lithology.contactDistance);
    value(surface.geology.plateVelocity.x);
    value(surface.geology.plateVelocity.z);
    value(surface.geology.continentalFraction);
    value(surface.geology.crustAge);
    value(surface.geology.crustThickness);
    value(surface.geology.crustDensity);
    value(surface.geology.distanceToBoundary);
    value(surface.geology.uplift);
    value(surface.geology.rift);
    value(surface.geology.faultStrength);
    value(surface.geology.hotspotInfluence);
    value(surface.geology.volcanicActivity);
    value(surface.hydrology.waterBodyId);
    value(surface.hydrology.generatedFluidLevel);
    value(surface.hydrology.flowDirection.x);
    value(surface.hydrology.flowDirection.z);
    value(surface.hydrology.surfaceElevation);
    value(surface.hydrology.waterSurface);
    value(surface.hydrology.discharge);
    value(surface.hydrology.sediment);
    value(surface.hydrology.channelDistance);
    value(surface.hydrology.channelWidth);
    value(surface.hydrology.channelDepth);
    value(surface.hydrology.channelGradient);
    value(surface.hydrology.erosionDepth);
    value(surface.hydrology.lakeDepth);
    value(surface.hydrology.lakeShoreDistance);
    value(surface.hydrology.shoreWaterSurface);
    value(surface.hydrology.lakeBankTarget);
    value(surface.hydrology.lakeBankInfluence);
    value(surface.hydrology.lakeAreaSquareKilometers);
    value(surface.hydrology.lakeVolumeCubicMeters);
    value(surface.hydrology.lakeRunoffMmSquareKilometers);
    value(surface.hydrology.lakeLossMm);
    value(surface.hydrology.lakeOverflowMmSquareKilometers);
    value(surface.hydrology.lakeSpillSurface);
    value(surface.hydrology.waterfallTop);
    value(surface.hydrology.waterfallBottom);
    value(surface.hydrology.waterfallWidth);
    value(surface.hydrology.streamOrder);
    value(surface.hydrology.distributaryCount);
    value(surface.hydrology.ocean);
    value(surface.hydrology.river);
    value(surface.hydrology.lake);
    value(surface.hydrology.lakeBank);
    value(surface.hydrology.endorheic);
    value(surface.hydrology.waterfall);
    value(surface.hydrology.waterfallAnchor);
    value(surface.hydrology.delta);
    value(surface.climate.wind.x);
    value(surface.climate.wind.z);
    value(surface.climate.temperatureC);
    value(surface.climate.annualPrecipitationMm);
    value(surface.climate.potentialEvapotranspirationMm);
    value(surface.climate.aridity);
    value(surface.climate.relativeHumidity);
    value(surface.soil.moisture);
    value(surface.soil.fertility);
    value(surface.soil.drainage);
    value(surface.soil.waterTable);
    for (const float score : surface.suitability.scores)
        value(score);
    value(surface.biome.primary);
    value(surface.biome.secondary);
    value(surface.biome.transition);
    value(surface.ecotopes);
    value(surface.terrainHeight);
    value(surface.waterSurface);
    value(surface.slope);
    return hash;
}

void requireHabitatCandidateInputs(const worldgen::SurfaceSample& actual,
                                   const worldgen::SurfaceSample& expected) {
    REQUIRE(actual.geology.plateId == expected.geology.plateId);
    REQUIRE(actual.geology.crust == expected.geology.crust);
    REQUIRE(actual.geology.boundary == expected.geology.boundary);
    REQUIRE(actual.geology.rock == expected.geology.rock);
    REQUIRE(actual.geology.lithology.primary == expected.geology.lithology.primary);
    REQUIRE(actual.geology.lithology.secondary == expected.geology.lithology.secondary);
    REQUIRE(actual.geology.lithology.transition == expected.geology.lithology.transition);
    REQUIRE(actual.geology.lithology.contactDistance == expected.geology.lithology.contactDistance);
    REQUIRE(actual.geology.crustAge == expected.geology.crustAge);
    REQUIRE(actual.geology.distanceToBoundary == expected.geology.distanceToBoundary);
    REQUIRE(actual.geology.uplift == expected.geology.uplift);
    REQUIRE(actual.geology.faultStrength == expected.geology.faultStrength);
    REQUIRE(actual.geology.volcanicActivity == expected.geology.volcanicActivity);

    REQUIRE(actual.hydrology.waterSurface == expected.hydrology.waterSurface);
    REQUIRE(actual.hydrology.surfaceElevation == expected.hydrology.surfaceElevation);
    REQUIRE(actual.hydrology.channelDistance == expected.hydrology.channelDistance);
    REQUIRE(actual.hydrology.channelWidth == expected.hydrology.channelWidth);
    REQUIRE(actual.hydrology.channelGradient == expected.hydrology.channelGradient);
    REQUIRE(actual.hydrology.erosionDepth == expected.hydrology.erosionDepth);
    REQUIRE(actual.hydrology.lakeDepth == expected.hydrology.lakeDepth);
    REQUIRE(actual.hydrology.streamOrder == expected.hydrology.streamOrder);
    REQUIRE(actual.hydrology.ocean == expected.hydrology.ocean);
    REQUIRE(actual.hydrology.river == expected.hydrology.river);
    REQUIRE(actual.hydrology.lake == expected.hydrology.lake);
    REQUIRE(actual.hydrology.waterfall == expected.hydrology.waterfall);
    REQUIRE(actual.hydrology.delta == expected.hydrology.delta);

    REQUIRE(actual.climate.temperatureC == expected.climate.temperatureC);
    REQUIRE(actual.climate.annualPrecipitationMm == expected.climate.annualPrecipitationMm);
    REQUIRE(actual.climate.aridity == expected.climate.aridity);
    REQUIRE(actual.climate.relativeHumidity == expected.climate.relativeHumidity);
    REQUIRE(actual.soil.moisture == expected.soil.moisture);
    REQUIRE(actual.soil.fertility == expected.soil.fertility);
    REQUIRE(actual.soil.waterTable == expected.soil.waterTable);
    REQUIRE(actual.suitability.scores == expected.suitability.scores);
    REQUIRE(actual.biome.primary == expected.biome.primary);
    REQUIRE(actual.biome.secondary == expected.biome.secondary);
    REQUIRE(actual.biome.transition == expected.biome.transition);
    REQUIRE(actual.ecotopes == expected.ecotopes);
    REQUIRE(actual.terrainHeight == expected.terrainHeight);
    REQUIRE(actual.waterSurface == expected.waterSurface);
    REQUIRE(actual.slope == expected.slope);

    const int actualGroundY = static_cast<int>(std::ceil(actual.terrainHeight)) - 1;
    const int expectedGroundY = static_cast<int>(std::ceil(expected.terrainHeight)) - 1;
    REQUIRE(actualGroundY == expectedGroundY);
    REQUIRE(feature_generation::treeCoverDensity(actual) ==
            feature_generation::treeCoverDensity(expected));
    for (size_t index = 0; index < static_cast<size_t>(feature_generation::TreeSpecies::COUNT);
         ++index) {
        const auto species = static_cast<feature_generation::TreeSpecies>(index);
        const feature_generation::TreeHabitatEvaluation actualHabitat =
            feature_generation::evaluateTreeHabitat(species, actual, actualGroundY);
        const feature_generation::TreeHabitatEvaluation expectedHabitat =
            feature_generation::evaluateTreeHabitat(species, expected, expectedGroundY);
        CAPTURE(index);
        REQUIRE(actualHabitat.suitability == expectedHabitat.suitability);
        REQUIRE(actualHabitat.waterDepthBlocks == expectedHabitat.waterDepthBlocks);
        REQUIRE(actualHabitat.spacing == expectedHabitat.spacing);
        REQUIRE(actualHabitat.submerged == expectedHabitat.submerged);
        REQUIRE(actualHabitat.allowed == expectedHabitat.allowed);
    }
}

std::vector<ColumnPos> treeCandidateBatchPositions() {
    constexpr uint64_t STREAM = 0x5452454543414E44ULL;
    constexpr int EDGE = feature_generation::TREE_CELL_EDGE;
    constexpr int COMPETITOR_RADIUS = 3;
    constexpr int64_t MINIMUM_X = -10'048;
    constexpr int64_t MINIMUM_Z = 4'992;
    constexpr int64_t MAXIMUM_X = -9'536;
    constexpr int64_t MAXIMUM_Z = 5'504;
    const int64_t minimumCellX =
        world_coord::floorDiv(MINIMUM_X - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(EDGE)) -
        COMPETITOR_RADIUS;
    const int64_t maximumCellX =
        world_coord::floorDiv(MAXIMUM_X - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(EDGE)) +
        COMPETITOR_RADIUS;
    const int64_t minimumCellZ =
        world_coord::floorDiv(MINIMUM_Z - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(EDGE)) -
        COMPETITOR_RADIUS;
    const int64_t maximumCellZ =
        world_coord::floorDiv(MAXIMUM_Z - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(EDGE)) +
        COMPETITOR_RADIUS;

    CounterRng random(42);
    std::vector<ColumnPos> result;
    result.reserve(
        static_cast<size_t>((maximumCellX - minimumCellX + 1) * (maximumCellZ - minimumCellZ + 1)) +
        2);
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            result.push_back({
                cellX * EDGE + random.uniformInt(STREAM, cellX, 0, cellZ, 0, 0, EDGE - 1),
                cellZ * EDGE + random.uniformInt(STREAM, cellX, 0, cellZ, 1, 0, EDGE - 1),
            });
        }
    }
    result.push_back(ColumnPos{-8'936, 4'519});
    result.push_back(ColumnPos{720, -1'665});
    return result;
}

void requireFaceConnectedTreeMaterial(ChunkGenerator& generator, const FarCanopy& canopy,
                                      bool includeLeaves) {
    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> cubes;
    const auto blockAt = [&](const BlockPos& position) {
        const ChunkPos cubePosition{Chunk::worldToChunk(position.x),
                                    Chunk::worldToChunkY(position.y),
                                    Chunk::worldToChunk(position.z)};
        auto [found, inserted] = cubes.try_emplace(cubePosition);
        if (inserted) {
            found->second = std::make_shared<Chunk>(cubePosition);
            generator.generateCube(*found->second);
        }
        return found->second->getBlock(Chunk::worldToLocal(position.x),
                                       Chunk::worldToLocalY(position.y),
                                       Chunk::worldToLocal(position.z));
    };

    std::unordered_set<BlockPos> material;
    size_t logCount = 0;
    size_t leafCount = 0;
    for (int y = canopy.baseY + feature_generation::TREE_MINIMUM_VERTICAL_OFFSET; y <= canopy.topY;
         ++y) {
        for (int64_t z = canopy.z - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH;
             z <= canopy.z + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH; ++z) {
            for (int64_t x = canopy.x - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH;
                 x <= canopy.x + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH; ++x) {
                const BlockPos position{x, y, z};
                const BlockType block = blockAt(position);
                if (block == canopy.logBlock) {
                    material.insert(position);
                    ++logCount;
                } else if (includeLeaves && block == canopy.leafBlock) {
                    material.insert(position);
                    ++leafCount;
                }
            }
        }
    }
    REQUIRE(logCount > 0);
    if (includeLeaves)
        REQUIRE(leafCount > 0);
    const BlockPos root{canopy.x, canopy.baseY, canopy.z};
    REQUIRE(material.contains(root));

    constexpr std::array<BlockPos, 6> NEIGHBORS = {
        BlockPos{1, 0, 0},  BlockPos{-1, 0, 0}, BlockPos{0, 1, 0},
        BlockPos{0, -1, 0}, BlockPos{0, 0, 1},  BlockPos{0, 0, -1},
    };
    std::unordered_set<BlockPos> connected;
    std::queue<BlockPos> pending;
    connected.insert(root);
    pending.push(root);
    while (!pending.empty()) {
        const BlockPos current = pending.front();
        pending.pop();
        for (const BlockPos direction : NEIGHBORS) {
            const BlockPos neighbor{current.x + direction.x, current.y + direction.y,
                                    current.z + direction.z};
            if (material.contains(neighbor) && connected.insert(neighbor).second)
                pending.push(neighbor);
        }
    }
    REQUIRE(connected.size() == material.size());
}

} // namespace

TEST_CASE("Sparse block surface batches match scalar tree habitat authority",
          "[worldgen][ecology][tree][batch][determinism][cache][regression]") {
    const std::vector<ColumnPos> positions = treeCandidateBatchPositions();
    REQUIRE(positions.size() == 5'186);
    REQUIRE(std::ranges::find(positions, ColumnPos{-9'738, 5'065}) != positions.end());
    REQUIRE(std::ranges::find(positions, ColumnPos{-8'936, 4'519}) != positions.end());

    ChunkGenerator scalarGenerator(42);
    std::vector<uint64_t> expected;
    expected.reserve(positions.size());
    for (const ColumnPos position : positions) {
        expected.push_back(surfaceFingerprint(scalarGenerator.sampleFarSurface(
            position.x, position.z, worldgen::SurfaceFootprint::BLOCK_1)));
    }

    ChunkGenerator batchGenerator(42);
    const uint64_t scalarSamplesBefore = batchGenerator.basinCacheMetrics().scalarSampleCalls;
    std::vector<worldgen::SurfaceSample> forward(positions.size());
    batchGenerator.sampleFarSurfacePoints(positions, worldgen::SurfaceFootprint::BLOCK_1, forward);
    REQUIRE(batchGenerator.basinCacheMetrics().scalarSampleCalls == scalarSamplesBefore);
    for (size_t index = 0; index < positions.size(); ++index) {
        CAPTURE(index, positions[index].x, positions[index].z);
        REQUIRE(surfaceFingerprint(forward[index]) == expected[index]);
    }

    std::vector<ColumnPos> reversedPositions(positions.rbegin(), positions.rend());
    batchGenerator.clearMacroCaches();
    std::vector<worldgen::SurfaceSample> reversed(reversedPositions.size());
    batchGenerator.sampleFarSurfacePoints(reversedPositions, worldgen::SurfaceFootprint::BLOCK_1,
                                          reversed);
    REQUIRE(batchGenerator.basinCacheMetrics().scalarSampleCalls == scalarSamplesBefore);
    for (size_t index = 0; index < reversed.size(); ++index) {
        CAPTURE(index, reversedPositions[index].x, reversedPositions[index].z);
        REQUIRE(surfaceFingerprint(reversed[index]) == expected[expected.size() - index - 1]);
    }
}

TEST_CASE("Batched far habitat inputs match scalar exact canopy authority",
          "[worldgen][ecology][tree][habitat][batch][determinism][cache][regression]") {
    enum class FixtureKind : uint8_t {
        DRY,
        RIPARIAN,
        SHALLOW_WATER,
        VOLCANIC,
        CLIMATE_TRANSITION,
        NEGATIVE_COORDINATE,
        FORMER_GRID_BOUNDARY,
    };
    struct Fixture {
        ColumnPos position;
        FixtureKind kind;
        const char* label;
    };
    constexpr std::array SEED_42_FIXTURES = {
        Fixture{{720, -1'665}, FixtureKind::DRY, "dry forest candidate"},
        Fixture{{-8'240, 3'088}, FixtureKind::RIPARIAN, "riparian channel"},
        Fixture{{-9'781, 5'125}, FixtureKind::SHALLOW_WATER, "shallow-water root"},
        Fixture{{-26'355, 29'672}, FixtureKind::CLIMATE_TRANSITION, "climate transition"},
        Fixture{{-557, 379}, FixtureKind::NEGATIVE_COORDINATE, "negative-coordinate scene"},
        Fixture{{-8'192, 2'048}, FixtureKind::FORMER_GRID_BOUNDARY, "plate and catchment boundary"},
        Fixture{
            {-2'048, -64}, FixtureKind::FORMER_GRID_BOUNDARY, "catchment and material boundary"},
        Fixture{{-64, -32}, FixtureKind::FORMER_GRID_BOUNDARY, "material and biome boundary"},
        Fixture{{-16, 8}, FixtureKind::FORMER_GRID_BOUNDARY, "surface and control boundary"},
        Fixture{{0, 0}, FixtureKind::FORMER_GRID_BOUNDARY, "world-origin boundary"},
    };
    constexpr std::array SEED_764891_FIXTURES = {
        Fixture{{23'029, -111'486}, FixtureKind::VOLCANIC, "volcanic caldera"},
    };

    const auto exercise = [](uint32_t seed, std::span<const Fixture> fixtures) {
        std::vector<ColumnPos> positions;
        positions.reserve(fixtures.size());
        for (const Fixture& fixture : fixtures)
            positions.push_back(fixture.position);

        ChunkGenerator scalar(seed);
        std::vector<worldgen::SurfaceSample> expected;
        expected.reserve(positions.size());
        for (const ColumnPos position : positions) {
            expected.push_back(scalar.sampleFarSurface(position.x, position.z,
                                                       worldgen::SurfaceFootprint::BLOCK_1));
        }

        ChunkGenerator batched(seed);
        const uint64_t scalarCallsBefore = batched.basinCacheMetrics().scalarSampleCalls;
        std::vector<worldgen::SurfaceSample> actual(positions.size());
        batched.sampleFarHabitatPoints(positions, actual);
        REQUIRE(batched.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
        for (size_t index = 0; index < fixtures.size(); ++index) {
            const Fixture& fixture = fixtures[index];
            CAPTURE(seed, fixture.label, fixture.position.x, fixture.position.z);
            requireHabitatCandidateInputs(actual[index], expected[index]);
            REQUIRE(batched.farSurfaceMaterialAt(fixture.position.x, fixture.position.z,
                                                 actual[index]) ==
                    scalar.farSurfaceMaterialAt(fixture.position.x, fixture.position.z,
                                                expected[index]));

            switch (fixture.kind) {
                case FixtureKind::DRY:
                    REQUIRE_FALSE(expected[index].hydrology.ocean);
                    REQUIRE_FALSE(expected[index].hydrology.river);
                    REQUIRE_FALSE(expected[index].hydrology.lake);
                    break;
                case FixtureKind::RIPARIAN:
                    REQUIRE((worldgen::hasEcotope(expected[index].ecotopes,
                                                  worldgen::Ecotope::RIVERBANK) ||
                             worldgen::hasEcotope(expected[index].ecotopes,
                                                  worldgen::Ecotope::FLOODPLAIN) ||
                             worldgen::hasEcotope(expected[index].ecotopes,
                                                  worldgen::Ecotope::LAKESHORE)));
                    break;
                case FixtureKind::SHALLOW_WATER: {
                    REQUIRE((expected[index].hydrology.ocean || expected[index].hydrology.river ||
                             expected[index].hydrology.lake));
                    const int groundY =
                        static_cast<int>(std::ceil(expected[index].terrainHeight)) - 1;
                    const int waterY =
                        static_cast<int>(std::ceil(expected[index].waterSurface)) - 1;
                    REQUIRE(waterY > groundY);
                    REQUIRE(waterY - groundY <= 3);
                    break;
                }
                case FixtureKind::VOLCANIC:
                    REQUIRE(expected[index].geology.volcanicActivity > 0.75);
                    break;
                case FixtureKind::CLIMATE_TRANSITION:
                    REQUIRE(expected[index].biome.primary != expected[index].biome.secondary);
                    REQUIRE(expected[index].biome.transition > 0.35);
                    break;
                case FixtureKind::NEGATIVE_COORDINATE:
                    REQUIRE((fixture.position.x < 0 || fixture.position.z < 0));
                    break;
                case FixtureKind::FORMER_GRID_BOUNDARY:
                    REQUIRE((fixture.position.x % 8 == 0 || fixture.position.z % 8 == 0));
                    break;
            }
        }

        std::vector<ColumnPos> reversedPositions(positions.rbegin(), positions.rend());
        batched.clearMacroCaches();
        std::vector<worldgen::SurfaceSample> reversed(reversedPositions.size());
        batched.sampleFarHabitatPoints(reversedPositions, reversed);
        REQUIRE(batched.basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
        for (size_t index = 0; index < reversed.size(); ++index) {
            CAPTURE(seed, reversedPositions[index].x, reversedPositions[index].z);
            requireHabitatCandidateInputs(reversed[index], expected[expected.size() - index - 1]);
        }
    };

    exercise(42, SEED_42_FIXTURES);
    exercise(764891, SEED_764891_FIXTURES);
}

TEST_CASE("Bent acacia trunks remain face connected in the seed forty two scene",
          "[worldgen][ecology][tree][geometry][connectivity][regression]") {
    constexpr int64_t ROOT_X = 595;
    constexpr int64_t ROOT_Z = -1'753;
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> canopies =
        generator.collectFarCanopies(ROOT_X - 8, ROOT_Z - 8, ROOT_X + 9, ROOT_Z + 9);
    const auto selected =
        std::find_if(canopies.begin(), canopies.end(), [](const FarCanopy& canopy) {
            return canopy.x == ROOT_X && canopy.z == ROOT_Z &&
                   canopy.species == feature_generation::TreeSpecies::ACACIA;
        });
    REQUIRE(selected != canopies.end());
    requireFaceConnectedTreeMaterial(generator, *selected, false);
}

TEST_CASE("Palm trunks and drooping fronds remain face connected",
          "[worldgen][ecology][tree][palm][geometry][connectivity][regression]") {
    constexpr int64_t ROOT_X = 518;
    constexpr int64_t ROOT_Z = -1'557;
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> canopies =
        generator.collectFarCanopies(ROOT_X - 8, ROOT_Z - 8, ROOT_X + 9, ROOT_Z + 9);
    const auto selected =
        std::find_if(canopies.begin(), canopies.end(), [](const FarCanopy& canopy) {
            return canopy.x == ROOT_X && canopy.z == ROOT_Z &&
                   canopy.species == feature_generation::TreeSpecies::PALM;
        });
    REQUIRE(selected != canopies.end());
    requireFaceConnectedTreeMaterial(generator, *selected, true);
}

TEST_CASE("Willow crowns and hanging leaves remain face connected",
          "[worldgen][ecology][tree][willow][geometry][connectivity][regression]") {
    constexpr int64_t ROOT_X = 778;
    constexpr int64_t ROOT_Z = -1'722;
    ChunkGenerator generator(42);
    const std::vector<FarCanopy> canopies =
        generator.collectFarCanopies(ROOT_X - 8, ROOT_Z - 8, ROOT_X + 9, ROOT_Z + 9);
    const auto selected =
        std::find_if(canopies.begin(), canopies.end(), [](const FarCanopy& canopy) {
            return canopy.x == ROOT_X && canopy.z == ROOT_Z &&
                   canopy.species == feature_generation::TreeSpecies::WILLOW;
        });
    REQUIRE(selected != canopies.end());
    requireFaceConnectedTreeMaterial(generator, *selected, true);
}

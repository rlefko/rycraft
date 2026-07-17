#include <catch2/catch_test_macros.hpp>
#include <render/block_textures.hpp>
#include <render/far_terrain.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/macro_generation.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t REPORTED_SEED = 42;
constexpr double MAXIMUM_UNTAGGED_VISIBLE_STEP = 0.125001;
constexpr std::array<std::pair<ColumnPos, ColumnPos>, 2> REPORTED_WET_FACES = {{
    {{52, -1'509}, {53, -1'509}},
    {{287, -1'583}, {288, -1'583}},
}};

constexpr std::array<ColumnPos, 5> PARITY_PROBES = {{
    {52, -1'509},
    {53, -1'509},
    {287, -1'583},
    {288, -1'583},
    {100, -1'518},
}};

constexpr std::array<worldgen::SurfaceFootprint, 6> FOOTPRINTS = {{
    worldgen::SurfaceFootprint::BLOCK_1,
    worldgen::SurfaceFootprint::BLOCK_2,
    worldgen::SurfaceFootprint::BLOCK_4,
    worldgen::SurfaceFootprint::BLOCK_8,
    worldgen::SurfaceFootprint::BLOCK_16,
    worldgen::SurfaceFootprint::BLOCK_32,
}};

constexpr std::array<FarTerrainStep, 5> FAR_STEPS = {{
    FarTerrainStep::TWO,
    FarTerrainStep::FOUR,
    FarTerrainStep::EIGHT,
    FarTerrainStep::SIXTEEN,
    FarTerrainStep::THIRTY_TWO,
}};

bool explicitFallOwns(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.waterfall &&
           sample.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL &&
           sample.hydrology.transitionOwnerId != 0;
}

bool sameWaterTopology(const worldgen::SurfaceSample& first,
                       const worldgen::SurfaceSample& second) {
    return first.hydrology.ocean == second.hydrology.ocean &&
           first.hydrology.river == second.hydrology.river &&
           first.hydrology.lake == second.hydrology.lake &&
           first.hydrology.delta == second.hydrology.delta &&
           first.hydrology.waterfall == second.hydrology.waterfall &&
           first.hydrology.waterBodyId == second.hydrology.waterBodyId;
}

struct FarWaterObservation {
    float interpolatedHeight = 0.0F;
    float minimumHeight = 0.0F;
    float maximumHeight = 0.0F;
};

std::vector<FarWaterObservation> farWaterAt(const FarTerrainMesh& mesh, float x, float z) {
    const auto signedArea = [](float ax, float az, float bx, float bz, float px, float pz) {
        return (px - bx) * (az - bz) - (ax - bx) * (pz - bz);
    };
    std::vector<FarWaterObservation> observations;
    for (size_t offset = mesh.opaqueIndexCount; offset + 2 < mesh.indices.size(); offset += 3) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            unpackTextureLayer(first.faceAttr) != static_cast<uint8_t>(BlockType::WATER)) {
            continue;
        }
        const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
        const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
        const float firstSign =
            signedArea(static_cast<float>(first.px), static_cast<float>(first.pz),
                       static_cast<float>(second.px), static_cast<float>(second.pz), x, z);
        const float secondSign =
            signedArea(static_cast<float>(second.px), static_cast<float>(second.pz),
                       static_cast<float>(third.px), static_cast<float>(third.pz), x, z);
        const float thirdSign =
            signedArea(static_cast<float>(third.px), static_cast<float>(third.pz),
                       static_cast<float>(first.px), static_cast<float>(first.pz), x, z);
        constexpr float EPSILON = 0.001F;
        const bool hasNegative =
            firstSign < -EPSILON || secondSign < -EPSILON || thirdSign < -EPSILON;
        const bool hasPositive = firstSign > EPSILON || secondSign > EPSILON || thirdSign > EPSILON;
        if (hasNegative && hasPositive)
            continue;

        const float denominator =
            (static_cast<float>(second.pz) - static_cast<float>(third.pz)) *
                (static_cast<float>(first.px) - static_cast<float>(third.px)) +
            (static_cast<float>(third.px) - static_cast<float>(second.px)) *
                (static_cast<float>(first.pz) - static_cast<float>(third.pz));
        if (std::abs(denominator) <= 1.0e-6F)
            continue;
        const float firstWeight = ((static_cast<float>(second.pz) - static_cast<float>(third.pz)) *
                                       (x - static_cast<float>(third.px)) +
                                   (static_cast<float>(third.px) - static_cast<float>(second.px)) *
                                       (z - static_cast<float>(third.pz))) /
                                  denominator;
        const float secondWeight = ((static_cast<float>(third.pz) - static_cast<float>(first.pz)) *
                                        (x - static_cast<float>(third.px)) +
                                    (static_cast<float>(first.px) - static_cast<float>(third.px)) *
                                        (z - static_cast<float>(third.pz))) /
                                   denominator;
        const float thirdWeight = 1.0F - firstWeight - secondWeight;
        const std::array heights = {static_cast<float>(first.py), static_cast<float>(second.py),
                                    static_cast<float>(third.py)};
        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        observations.push_back({
            .interpolatedHeight =
                firstWeight * heights[0] + secondWeight * heights[1] + thirdWeight * heights[2],
            .minimumHeight = *minimum,
            .maximumHeight = *maximum,
        });
    }
    return observations;
}

bool canonicalStateExistsNear(const ChunkGenerator& generator, ColumnPos position, int radius,
                              bool wet, double visibleSurface, bool requireSurface) {
    for (int offsetZ = -radius; offsetZ <= radius; ++offsetZ) {
        for (int offsetX = -radius; offsetX <= radius; ++offsetX) {
            const worldgen::SurfaceSample sample = generator.sampleFarSurface(
                position.x + offsetX, position.z + offsetZ, worldgen::SurfaceFootprint::BLOCK_1);
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(sample);
            if (fluid.wet != wet)
                continue;
            if (!requireSurface ||
                std::abs(fluid.visibleSurface - visibleSurface) <= MAXIMUM_UNTAGGED_VISIBLE_STEP) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

TEST_CASE("Reported seed 42 camera corridor keeps generated water physically continuous",
          "[worldgen][hydrology][water][reported-water-continuity][regression]") {
    ChunkGenerator generator(REPORTED_SEED);

    for (const auto& [firstPosition, secondPosition] : REPORTED_WET_FACES) {
        const worldgen::SurfaceSample first =
            generator.sampleExactSurface(firstPosition.x, firstPosition.z);
        const worldgen::SurfaceSample second =
            generator.sampleExactSurface(secondPosition.x, secondPosition.z);
        const worldgen::GeneratedFluidColumn firstFluid = worldgen::generatedFluidColumn(first);
        const worldgen::GeneratedFluidColumn secondFluid = worldgen::generatedFluidColumn(second);
        const double visibleStep = std::abs(firstFluid.visibleSurface - secondFluid.visibleSurface);
        const bool taggedFall = explicitFallOwns(first) || explicitFallOwns(second);
        CAPTURE(firstPosition.x, firstPosition.z, secondPosition.x, secondPosition.z,
                first.terrainHeight, second.terrainHeight, firstFluid.visibleSurface,
                secondFluid.visibleSurface, first.hydrology.generatedFluidLevel,
                second.hydrology.generatedFluidLevel, first.hydrology.transitionOwnerKind,
                second.hydrology.transitionOwnerKind, first.hydrology.transitionOwnerId,
                second.hydrology.transitionOwnerId, visibleStep, taggedFall);
        CHECK(firstFluid.wet);
        CHECK(secondFluid.wet);
        CHECK((taggedFall || visibleStep <= MAXIMUM_UNTAGGED_VISIBLE_STEP));
    }

    constexpr int64_t MINIMUM_X = 32;
    constexpr int64_t MAXIMUM_X = 384;
    constexpr int64_t MINIMUM_Z = -1'584;
    constexpr int64_t MAXIMUM_Z = -1'440;
    constexpr int WIDTH = static_cast<int>(MAXIMUM_X - MINIMUM_X + 1);
    constexpr int DEPTH = static_cast<int>(MAXIMUM_Z - MINIMUM_Z + 1);
    std::vector<worldgen::SurfaceSample> samples(static_cast<size_t>(WIDTH * DEPTH));
    generator.sampleFarGeometryGrid(MINIMUM_X, MINIMUM_Z, 1, 1, WIDTH, DEPTH,
                                    worldgen::SurfaceFootprint::BLOCK_1, samples);
    const auto sampleAt = [&](int x, int z) -> const worldgen::SurfaceSample& {
        return samples[static_cast<size_t>(z) * WIDTH + x];
    };

    size_t wetColumns = 0;
    size_t unsupportedWetColumns = 0;
    size_t untaggedWetJumps = 0;
    double maximumUntaggedStep = 0.0;
    std::map<std::pair<worldgen::WaterTransitionKind, worldgen::WaterTransitionKind>, size_t>
        illegalOwnerKindPairs;
    std::map<std::pair<int64_t, int64_t>, size_t> illegalStagePairs;
    std::map<bool, size_t> illegalFlowAlignment;
    std::vector<std::string> illegalFaceExamples;
    size_t scalarBatchMismatches = 0;
    std::vector<std::string> scalarBatchMismatchExamples;
    int64_t worstX = 0;
    int64_t worstZ = 0;
    int worstDx = 0;
    int worstDz = 0;
    for (int z = 0; z < DEPTH; ++z) {
        for (int x = 0; x < WIDTH; ++x) {
            const worldgen::SurfaceSample& sample = sampleAt(x, z);
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(sample);
            if (fluid.wet) {
                ++wetColumns;
                unsupportedWetColumns +=
                    fluid.visibleSurface <= sample.terrainHeight + 0.01 ? 1U : 0U;
            }
            for (const auto [dx, dz] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + dx >= WIDTH || z + dz >= DEPTH)
                    continue;
                const worldgen::SurfaceSample& neighbor = sampleAt(x + dx, z + dz);
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                if (!fluid.wet || !neighborFluid.wet || explicitFallOwns(sample) ||
                    explicitFallOwns(neighbor)) {
                    continue;
                }
                const double step = std::abs(fluid.visibleSurface - neighborFluid.visibleSurface);
                if (step <= MAXIMUM_UNTAGGED_VISIBLE_STEP)
                    continue;
                ++untaggedWetJumps;
                const auto ownerKinds = std::minmax(sample.hydrology.transitionOwnerKind,
                                                    neighbor.hydrology.transitionOwnerKind);
                ++illegalOwnerKindPairs[{ownerKinds.first, ownerKinds.second}];
                const int64_t firstStage =
                    static_cast<int64_t>(std::llround(fluid.visibleSurface * 8.0));
                const int64_t secondStage =
                    static_cast<int64_t>(std::llround(neighborFluid.visibleSurface * 8.0));
                const auto quantizedStages = std::minmax(firstStage, secondStage);
                ++illegalStagePairs[{quantizedStages.first, quantizedStages.second}];
                const auto flowAligns = [&](const worldgen::SurfaceSample& flowSample) {
                    const double alongFace = dx != 0
                                                 ? std::abs(flowSample.hydrology.flowDirection.x)
                                                 : std::abs(flowSample.hydrology.flowDirection.z);
                    const double acrossFace = dx != 0
                                                  ? std::abs(flowSample.hydrology.flowDirection.z)
                                                  : std::abs(flowSample.hydrology.flowDirection.x);
                    return alongFace >= acrossFace && alongFace > 1.0e-6;
                };
                const bool alignsEitherFlow = flowAligns(sample) || flowAligns(neighbor);
                ++illegalFlowAlignment[alignsEitherFlow];
                const auto compareScalar = [&](const worldgen::SurfaceSample& batched, int sampleX,
                                               int sampleZ) {
                    const worldgen::SurfaceSample scalar =
                        generator.sampleFarSurface(MINIMUM_X + sampleX, MINIMUM_Z + sampleZ,
                                                   worldgen::SurfaceFootprint::BLOCK_1);
                    const worldgen::GeneratedFluidColumn scalarFluid =
                        worldgen::generatedFluidColumn(scalar);
                    const worldgen::GeneratedFluidColumn batchedFluid =
                        worldgen::generatedFluidColumn(batched);
                    if (std::abs(scalarFluid.visibleSurface - batchedFluid.visibleSurface) <=
                            1.0e-9 &&
                        scalar.hydrology.transitionOwnerKind ==
                            batched.hydrology.transitionOwnerKind &&
                        scalar.hydrology.transitionOwnerId == batched.hydrology.transitionOwnerId) {
                        return;
                    }
                    ++scalarBatchMismatches;
                    if (scalarBatchMismatchExamples.size() < 8) {
                        scalarBatchMismatchExamples.push_back(
                            "(" + std::to_string(MINIMUM_X + sampleX) + "," +
                            std::to_string(MINIMUM_Z + sampleZ) +
                            ") scalar=" + std::to_string(scalarFluid.visibleSurface) +
                            " batch=" + std::to_string(batchedFluid.visibleSurface) + " owners=" +
                            std::to_string(
                                static_cast<unsigned>(scalar.hydrology.transitionOwnerKind)) +
                            "/" +
                            std::to_string(
                                static_cast<unsigned>(batched.hydrology.transitionOwnerKind)));
                    }
                };
                compareScalar(sample, x, z);
                compareScalar(neighbor, x + dx, z + dz);
                if (illegalFaceExamples.size() < 8) {
                    illegalFaceExamples.push_back(
                        "(" + std::to_string(MINIMUM_X + x) + "," + std::to_string(MINIMUM_Z + z) +
                        ")->(" + std::to_string(MINIMUM_X + x + dx) + "," +
                        std::to_string(MINIMUM_Z + z + dz) + ") owners=" +
                        std::to_string(
                            static_cast<unsigned>(sample.hydrology.transitionOwnerKind)) +
                        "/" +
                        std::to_string(
                            static_cast<unsigned>(neighbor.hydrology.transitionOwnerKind)) +
                        " stages=" + std::to_string(quantizedStages.first) + "/" +
                        std::to_string(quantizedStages.second) +
                        " aligned=" + (alignsEitherFlow ? "true" : "false"));
                }
                if (step > maximumUntaggedStep) {
                    maximumUntaggedStep = step;
                    worstX = MINIMUM_X + x;
                    worstZ = MINIMUM_Z + z;
                    worstDx = dx;
                    worstDz = dz;
                }
            }
        }
    }
    std::vector<std::string> illegalOwnerKindSummary;
    illegalOwnerKindSummary.reserve(illegalOwnerKindPairs.size());
    for (const auto& [pair, count] : illegalOwnerKindPairs) {
        illegalOwnerKindSummary.push_back(std::to_string(static_cast<unsigned>(pair.first)) + "/" +
                                          std::to_string(static_cast<unsigned>(pair.second)) + "=" +
                                          std::to_string(count));
    }
    std::vector<std::string> illegalStageSummary;
    illegalStageSummary.reserve(illegalStagePairs.size());
    for (const auto& [pair, count] : illegalStagePairs) {
        illegalStageSummary.push_back(std::to_string(pair.first) + "/" +
                                      std::to_string(pair.second) + "=" + std::to_string(count));
    }
    std::vector<std::string> illegalAlignmentSummary;
    illegalAlignmentSummary.reserve(illegalFlowAlignment.size());
    for (const auto& [aligned, count] : illegalFlowAlignment) {
        illegalAlignmentSummary.push_back(std::string(aligned ? "aligned=" : "lateral=") +
                                          std::to_string(count));
    }
    CAPTURE(wetColumns, unsupportedWetColumns, untaggedWetJumps, maximumUntaggedStep, worstX,
            worstZ, worstDx, worstDz, illegalOwnerKindSummary, illegalStageSummary,
            illegalAlignmentSummary, illegalFaceExamples, scalarBatchMismatches,
            scalarBatchMismatchExamples);
    CHECK(wetColumns >= 10'000);
    CHECK(unsupportedWetColumns == 0);
    CHECK(untaggedWetJumps == 0);

    World world(REPORTED_SEED, 4);
    CHECK(world.getPendingFluidCount() == 0);
    for (const ColumnPos position : PARITY_PROBES) {
        const auto plan = generator.getColumnPlan(
            {Chunk::worldToChunk(position.x), Chunk::worldToChunk(position.z)});
        const int localX = Chunk::worldToLocal(position.x);
        const int localZ = Chunk::worldToLocal(position.z);
        const int surfaceY = plan->surfaceY(localX, localZ);
        worldgen::SurfaceSample exact = plan->sample(localX, localZ);
        exact.terrainHeight = static_cast<double>(surfaceY + 1);
        const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(exact);
        const int maximumY = fluid.wet ? fluid.topY : surfaceY;
        for (int32_t section = Chunk::worldToChunkY(surfaceY);
             section <= Chunk::worldToChunkY(maximumY); ++section) {
            REQUIRE(world.getChunk(
                {Chunk::worldToChunk(position.x), section, Chunk::worldToChunk(position.z)}));
        }
        const FluidCell floor = world.readFluidCell({position.x, surfaceY, position.z});
        CAPTURE(position.x, position.z, surfaceY, maximumY, fluid.wet, fluid.topState.packed());
        CHECK(floor.loaded);
        CHECK(isSolid(floor.block));
        for (int y = surfaceY + 1; y <= maximumY; ++y) {
            const FluidCell cell = world.readFluidCell({position.x, y, position.z});
            CHECK(cell.loaded);
            CHECK(cell.block == BlockType::WATER);
            if (y == maximumY)
                CHECK(cell.state == fluid.topState);
        }
    }
    CHECK(world.getPendingFluidCount() == 0);
}

TEST_CASE("Final exact terrain preserves routed water and supported lateral banks",
          "[worldgen][hydrology][water][exact][volcanic][bank][regression]") {
    constexpr int64_t CENTER_X = 141;
    constexpr int64_t CENTER_Z = -1'555;
    constexpr int RADIUS = 48;
    constexpr int EDGE = RADIUS * 2 + 1;
    constexpr int64_t ORIGIN_X = CENTER_X - RADIUS;
    constexpr int64_t ORIGIN_Z = CENTER_Z - RADIUS;

    ChunkGenerator generator(REPORTED_SEED);
    std::vector<worldgen::SurfaceSample> exact(static_cast<size_t>(EDGE * EDGE));
    std::vector<worldgen::HydrologySample> canonical(static_cast<size_t>(EDGE * EDGE));
    generator.sampleExactSurfaceGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, exact);
    generator.sampleGeneratedWaterAuthorityGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, canonical);

    const auto index = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    size_t supportableCanonicalColumns = 0;
    size_t deletedCanonicalColumns = 0;
    size_t adjacentWetFaces = 0;
    size_t untaggedWetSteps = 0;
    size_t lateralBanks = 0;
    size_t unsupportedLateralBanks = 0;
    double maximumUntaggedWetStep = 0.0;
    double maximumLateralBankDeficit = 0.0;
    ColumnPos firstDeleted{};
    ColumnPos firstWetStep{};
    ColumnPos firstWetStepNeighbor{};
    double firstWetStage = 0.0;
    double firstWetNeighborStage = 0.0;
    worldgen::WaterTransitionKind firstWetKind = worldgen::WaterTransitionKind::NONE;
    worldgen::WaterTransitionKind firstWetNeighborKind = worldgen::WaterTransitionKind::NONE;
    bool firstWetOcean = false;
    bool firstWetRiver = false;
    bool firstWetBank = false;
    double firstWetTerrain = 0.0;
    bool firstWetNeighborOcean = false;
    bool firstWetNeighborRiver = false;
    bool firstWetNeighborBank = false;
    double firstWetNeighborTerrain = 0.0;
    ColumnPos firstUnsupportedBank{};
    ColumnPos firstUnsupportedBankNeighbor{};
    double firstUnsupportedWetSurface = 0.0;
    double firstUnsupportedDryTerrain = 0.0;
    bool firstUnsupportedDryBank = false;

    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const size_t currentIndex = index(x, z);
            const worldgen::SurfaceSample& sample = exact[currentIndex];
            const worldgen::HydrologySample& authority = canonical[currentIndex];
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(sample);
            const bool canonicalWet = authority.ocean || authority.river || authority.lake;
            if (canonicalWet && std::isfinite(authority.waterSurface) &&
                authority.waterSurface > sample.terrainHeight + 0.01) {
                ++supportableCanonicalColumns;
                if (!fluid.wet) {
                    if (deletedCanonicalColumns == 0)
                        firstDeleted = {ORIGIN_X + x, ORIGIN_Z + z};
                    ++deletedCanonicalColumns;
                }
            }

            for (const auto [dx, dz] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + dx >= EDGE || z + dz >= EDGE)
                    continue;
                const worldgen::SurfaceSample& neighbor = exact[index(x + dx, z + dz)];
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                if (fluid.wet && neighborFluid.wet) {
                    ++adjacentWetFaces;
                    if (explicitFallOwns(sample) || explicitFallOwns(neighbor))
                        continue;
                    const double step =
                        std::abs(fluid.visibleSurface - neighborFluid.visibleSurface);
                    maximumUntaggedWetStep = std::max(maximumUntaggedWetStep, step);
                    if (step > MAXIMUM_UNTAGGED_VISIBLE_STEP) {
                        if (untaggedWetSteps == 0) {
                            firstWetStep = {ORIGIN_X + x, ORIGIN_Z + z};
                            firstWetStepNeighbor = {ORIGIN_X + x + dx, ORIGIN_Z + z + dz};
                            firstWetStage = fluid.visibleSurface;
                            firstWetNeighborStage = neighborFluid.visibleSurface;
                            firstWetKind = sample.hydrology.transitionOwnerKind;
                            firstWetNeighborKind = neighbor.hydrology.transitionOwnerKind;
                            firstWetOcean = sample.hydrology.ocean;
                            firstWetRiver = sample.hydrology.river;
                            firstWetBank = sample.hydrology.channelBank;
                            firstWetTerrain = sample.terrainHeight;
                            firstWetNeighborOcean = neighbor.hydrology.ocean;
                            firstWetNeighborRiver = neighbor.hydrology.river;
                            firstWetNeighborBank = neighbor.hydrology.channelBank;
                            firstWetNeighborTerrain = neighbor.terrainHeight;
                        }
                        ++untaggedWetSteps;
                    }
                    continue;
                }
                if (fluid.wet == neighborFluid.wet)
                    continue;

                const worldgen::SurfaceSample& wet = fluid.wet ? sample : neighbor;
                const worldgen::SurfaceSample& dry = fluid.wet ? neighbor : sample;
                const worldgen::GeneratedFluidColumn& wetFluid = fluid.wet ? fluid : neighborFluid;
                const double normalFlow = dx != 0 ? std::abs(wet.hydrology.flowDirection.x)
                                                  : std::abs(wet.hydrology.flowDirection.z);
                const double tangentialFlow = dx != 0 ? std::abs(wet.hydrology.flowDirection.z)
                                                      : std::abs(wet.hydrology.flowDirection.x);
                const bool standingWater = std::hypot(wet.hydrology.flowDirection.x,
                                                      wet.hydrology.flowDirection.z) <= 1.0e-6;
                if (!standingWater && normalFlow + 1.0e-6 >= tangentialFlow)
                    continue;

                ++lateralBanks;
                const double deficit = wetFluid.visibleSurface - dry.terrainHeight;
                maximumLateralBankDeficit = std::max(maximumLateralBankDeficit, deficit);
                // A one-block source-water face against solid terrain is a
                // supported natural bank. Only a deeper exposed face is a
                // floating wall without lateral or floor support.
                if (deficit > 1.000001) {
                    if (unsupportedLateralBanks == 0) {
                        firstUnsupportedBank = {ORIGIN_X + x, ORIGIN_Z + z};
                        firstUnsupportedBankNeighbor = {ORIGIN_X + x + dx, ORIGIN_Z + z + dz};
                        firstUnsupportedWetSurface = wetFluid.visibleSurface;
                        firstUnsupportedDryTerrain = dry.terrainHeight;
                        firstUnsupportedDryBank = dry.hydrology.channelBank;
                    }
                    ++unsupportedLateralBanks;
                }
            }
        }
    }

    CAPTURE(supportableCanonicalColumns, deletedCanonicalColumns, adjacentWetFaces,
            untaggedWetSteps, maximumUntaggedWetStep, lateralBanks, unsupportedLateralBanks,
            maximumLateralBankDeficit, firstDeleted.x, firstDeleted.z, firstWetStep.x,
            firstWetStep.z, firstWetStepNeighbor.x, firstWetStepNeighbor.z, firstWetStage,
            firstWetNeighborStage, firstWetKind, firstWetNeighborKind, firstUnsupportedBank.x,
            firstUnsupportedBank.z, firstWetOcean, firstWetRiver, firstWetBank, firstWetTerrain,
            firstWetNeighborOcean, firstWetNeighborRiver, firstWetNeighborBank,
            firstWetNeighborTerrain, firstUnsupportedBankNeighbor.x, firstUnsupportedBankNeighbor.z,
            firstUnsupportedWetSurface, firstUnsupportedDryTerrain, firstUnsupportedDryBank);
    CHECK(supportableCanonicalColumns > 0);
    CHECK(deletedCanonicalColumns == 0);
    CHECK(adjacentWetFaces > 0);
    CHECK(untaggedWetSteps == 0);
    CHECK(lateralBanks > 0);
    CHECK(unsupportedLateralBanks == 0);
}

TEST_CASE("Reported seed 42 camera water agrees across exact and far ownership",
          "[worldgen][render][far-terrain][water][reported-water-continuity][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(REPORTED_SEED);
    for (const ColumnPos position : PARITY_PROBES) {
        const worldgen::SurfaceSample exact = generator->sampleExactSurface(position.x, position.z);
        const worldgen::GeneratedFluidColumn exactFluid = worldgen::generatedFluidColumn(exact);
        for (const worldgen::SurfaceFootprint footprint : FOOTPRINTS) {
            const worldgen::SurfaceSample far =
                generator->sampleFarSurface(position.x, position.z, footprint);
            const worldgen::GeneratedFluidColumn farFluid = worldgen::generatedFluidColumn(far);
            CAPTURE(position.x, position.z, worldgen::surfaceFootprintWidth(footprint),
                    exact.terrainHeight, far.terrainHeight, exactFluid.wet, farFluid.wet,
                    exactFluid.visibleSurface, farFluid.visibleSurface,
                    exact.hydrology.generatedFluidLevel, far.hydrology.generatedFluidLevel,
                    exact.hydrology.transitionOwnerKind, far.hydrology.transitionOwnerKind,
                    exact.hydrology.transitionOwnerId, far.hydrology.transitionOwnerId,
                    exact.hydrology.waterBodyId, far.hydrology.waterBodyId);
            CHECK(sameWaterTopology(exact, far));
            CHECK(exact.hydrology.generatedFluidLevel == far.hydrology.generatedFluidLevel);
            CHECK(exact.hydrology.transitionOwnerKind == far.hydrology.transitionOwnerKind);
            CHECK(exact.hydrology.transitionOwnerId == far.hydrology.transitionOwnerId);
            CHECK(std::abs(exact.hydrology.waterSurface - far.hydrology.waterSurface) <= 1.0e-6);
            CHECK(exactFluid.wet == farFluid.wet);
            if (exactFluid.wet && farFluid.wet) {
                CHECK(std::abs(exactFluid.visibleSurface - farFluid.visibleSurface) <=
                      MAXIMUM_UNTAGGED_VISIBLE_STEP);
            }
        }
    }

    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    std::optional<FarTerrainStep> activeBuildStep;
    size_t sentinelBatchCalls = 0;
    size_t sentinelBatchSamples = 0;
    size_t maximumSentinelBatchSamples = 0;
    size_t exactPointBatchCalls = 0;
    size_t exactPointBatchSamples = 0;
    size_t maximumExactPointBatchSamples = 0;
    double sentinelBatchMilliseconds = 0.0;
    double exactPointBatchMilliseconds = 0.0;
    std::map<FarTerrainStep, double> sentinelMillisecondsByStep;
    std::map<FarTerrainStep, double> exactPointMillisecondsByStep;
    std::map<FarTerrainStep, size_t> exactPointSamplesByStep;
    const FarTerrainSource::GeometryGridSampleFunction geometryGrid = source.geometryGrid;
    source.geometryGrid = [&](int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                              int sampleWidth, int sampleHeight,
                              worldgen::SurfaceFootprint footprint,
                              std::span<FarTerrainGeometrySample> output) {
        if (activeBuildStep && farTerrainStepSize(*activeBuildStep) <= 4 && spacingX == 2 &&
            spacingZ == 2 && footprint == worldgen::SurfaceFootprint::BLOCK_1) {
            ++sentinelBatchCalls;
            sentinelBatchSamples += output.size();
            maximumSentinelBatchSamples = std::max(maximumSentinelBatchSamples, output.size());
        }
        const auto started = std::chrono::steady_clock::now();
        geometryGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight, footprint,
                     output);
        if (activeBuildStep && farTerrainStepSize(*activeBuildStep) <= 4 && spacingX == 2 &&
            spacingZ == 2 && footprint == worldgen::SurfaceFootprint::BLOCK_1) {
            const double milliseconds = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - started)
                                            .count();
            sentinelBatchMilliseconds += milliseconds;
            sentinelMillisecondsByStep[*activeBuildStep] += milliseconds;
        }
    };
    const FarTerrainSource::GeometryPointSampleFunction geometryPoints = source.geometryPoints;
    source.geometryPoints = [&](std::span<const ColumnPos> positions,
                                worldgen::SurfaceFootprint footprint,
                                std::span<FarTerrainGeometrySample> output) {
        if (activeBuildStep && farTerrainStepSize(*activeBuildStep) <= 4 &&
            footprint == worldgen::SurfaceFootprint::BLOCK_1) {
            ++exactPointBatchCalls;
            exactPointBatchSamples += output.size();
            exactPointSamplesByStep[*activeBuildStep] += output.size();
            maximumExactPointBatchSamples = std::max(maximumExactPointBatchSamples, output.size());
        }
        const auto started = std::chrono::steady_clock::now();
        geometryPoints(positions, footprint, output);
        if (activeBuildStep && farTerrainStepSize(*activeBuildStep) <= 4 &&
            footprint == worldgen::SurfaceFootprint::BLOCK_1) {
            const double milliseconds = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - started)
                                            .count();
            exactPointBatchMilliseconds += milliseconds;
            exactPointMillisecondsByStep[*activeBuildStep] += milliseconds;
        }
    };
    source.canopies = {};
    using MeshKey = std::tuple<int64_t, int64_t, FarTerrainStep>;
    std::map<MeshKey, std::shared_ptr<const FarTerrainMesh>> meshes;
    std::map<MeshKey, double> meshBuildMilliseconds;
    for (const ColumnPos position : PARITY_PROBES) {
        const worldgen::SurfaceSample exact = generator->sampleExactSurface(position.x, position.z);
        const worldgen::GeneratedFluidColumn exactFluid = worldgen::generatedFluidColumn(exact);
        const int64_t tileX = world_coord::floorDiv(position.x, int64_t{FAR_TERRAIN_TILE_EDGE});
        const int64_t tileZ = world_coord::floorDiv(position.z, int64_t{FAR_TERRAIN_TILE_EDGE});
        for (const FarTerrainStep step : FAR_STEPS) {
            const MeshKey key{tileX, tileZ, step};
            auto [entry, inserted] = meshes.try_emplace(key);
            if (inserted) {
                const auto started = std::chrono::steady_clock::now();
                activeBuildStep = step;
                entry->second = FarTerrainMesher::build({tileX, tileZ, step}, source);
                activeBuildStep.reset();
                meshBuildMilliseconds[key] = std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count();
            }
            const std::shared_ptr<const FarTerrainMesh>& mesh = entry->second;
            const std::vector<FarWaterObservation> observations =
                farWaterAt(*mesh, static_cast<float>(position.x - mesh->originX) + 0.5F,
                           static_cast<float>(position.z - mesh->originZ) + 0.5F);
            const int stepSize = farTerrainStepSize(step);
            const bool meshWet = !observations.empty();
            double maximumHeightError = 0.0;
            double maximumTriangleRange = 0.0;
            bool everyHeightHasLocalAuthority = true;
            for (const FarWaterObservation& observation : observations) {
                maximumHeightError =
                    std::max(maximumHeightError,
                             std::abs(static_cast<double>(observation.interpolatedHeight) -
                                      exactFluid.visibleSurface));
                maximumTriangleRange =
                    std::max(maximumTriangleRange, static_cast<double>(observation.maximumHeight -
                                                                       observation.minimumHeight));
                if (stepSize >= 8 &&
                    !canonicalStateExistsNear(*generator, position, stepSize, true,
                                              observation.interpolatedHeight, true)) {
                    everyHeightHasLocalAuthority = false;
                }
            }
            CAPTURE(position.x, position.z, stepSize, exactFluid.wet, exactFluid.visibleSurface,
                    meshWet, observations.size(), maximumHeightError, maximumTriangleRange,
                    everyHeightHasLocalAuthority, mesh->vertices.size(), mesh->indices.size(),
                    mesh->waterQuadCount, mesh->waterContourTriangleCount, mesh->byteSize(),
                    meshBuildMilliseconds.at(key));
            CHECK(maximumTriangleRange <= MAXIMUM_UNTAGGED_VISIBLE_STEP);
            if (stepSize <= 4) {
                CHECK(meshWet == exactFluid.wet);
                CHECK(mesh->waterContourTriangleCount == 0);
            } else {
                if (meshWet != exactFluid.wet) {
                    CHECK(canonicalStateExistsNear(*generator, position, stepSize, meshWet, 0.0,
                                                   false));
                }
                CHECK(everyHeightHasLocalAuthority);
            }
            if (stepSize <= 4 && exactFluid.wet && meshWet) {
                CHECK(maximumHeightError <= MAXIMUM_UNTAGGED_VISIBLE_STEP);
            }
        }
    }
    const size_t exactNearMeshCount =
        std::count_if(meshes.begin(), meshes.end(), [](const auto& entry) {
            const FarTerrainStep step = std::get<2>(entry.first);
            return step == FarTerrainStep::TWO || step == FarTerrainStep::FOUR;
        });
    const size_t stepTwoMeshCount =
        std::count_if(meshes.begin(), meshes.end(), [](const auto& entry) {
            return std::get<2>(entry.first) == FarTerrainStep::TWO;
        });
    const size_t stepFourMeshCount =
        std::count_if(meshes.begin(), meshes.end(), [](const auto& entry) {
            return std::get<2>(entry.first) == FarTerrainStep::FOUR;
        });
    const size_t expectedSentinelSamples = stepTwoMeshCount * static_cast<size_t>(128 * 128) +
                                           stepFourMeshCount * static_cast<size_t>(129 * 129);
    constexpr size_t MAXIMUM_SENTINEL_SAMPLES = 129 * 129;
    constexpr size_t MAXIMUM_EXACT_POINT_SAMPLES = FAR_TERRAIN_TILE_EDGE * FAR_TERRAIN_TILE_EDGE;
    constexpr size_t MAXIMUM_POINT_CALLBACK_SAMPLES = 4'096;
    const size_t maximumSentinelBytes =
        maximumSentinelBatchSamples * sizeof(FarTerrainGeometrySample);
    const size_t maximumPointCallbackBytes =
        maximumExactPointBatchSamples *
        (sizeof(ColumnPos) + sizeof(FarTerrainGeometrySample) + sizeof(worldgen::SurfaceSample));
    CAPTURE(exactNearMeshCount, sentinelBatchCalls, sentinelBatchSamples,
            maximumSentinelBatchSamples, exactPointBatchCalls, exactPointBatchSamples,
            maximumExactPointBatchSamples, expectedSentinelSamples, maximumSentinelBytes,
            maximumPointCallbackBytes);
    CAPTURE(sentinelBatchMilliseconds, exactPointBatchMilliseconds);
    const double stepTwoSentinelMilliseconds = sentinelMillisecondsByStep[FarTerrainStep::TWO];
    const double stepFourSentinelMilliseconds = sentinelMillisecondsByStep[FarTerrainStep::FOUR];
    const double stepTwoExactPointMilliseconds = exactPointMillisecondsByStep[FarTerrainStep::TWO];
    const double stepFourExactPointMilliseconds =
        exactPointMillisecondsByStep[FarTerrainStep::FOUR];
    const size_t stepTwoExactPointSamples = exactPointSamplesByStep[FarTerrainStep::TWO];
    const size_t stepFourExactPointSamples = exactPointSamplesByStep[FarTerrainStep::FOUR];
    CAPTURE(stepTwoSentinelMilliseconds, stepFourSentinelMilliseconds,
            stepTwoExactPointMilliseconds, stepFourExactPointMilliseconds, stepTwoExactPointSamples,
            stepFourExactPointSamples);
    CHECK(sentinelBatchCalls == exactNearMeshCount);
    CHECK(sentinelBatchSamples == expectedSentinelSamples);
    CHECK(maximumSentinelBatchSamples <= MAXIMUM_SENTINEL_SAMPLES);
    CHECK(maximumExactPointBatchSamples <= MAXIMUM_POINT_CALLBACK_SAMPLES);
    CHECK(exactPointBatchSamples <= exactNearMeshCount * MAXIMUM_EXACT_POINT_SAMPLES);
    CHECK(exactPointBatchCalls <= exactNearMeshCount * 16);
}

TEST_CASE("Near far-water authority retains generated caldera lakes",
          "[worldgen][render][far-terrain][water][caldera][near-water-authority][regression]") {
    constexpr uint32_t SEED = 764891;
    constexpr ColumnPos POSITION{23'029, -111'486};
    auto generator = std::make_shared<ChunkGenerator>(SEED);
    const worldgen::SurfaceSample exact = generator->sampleExactSurface(POSITION.x, POSITION.z);
    const worldgen::GeneratedFluidColumn exactFluid = worldgen::generatedFluidColumn(exact);
    REQUIRE(exact.hydrology.lake);
    REQUIRE(exactFluid.wet);

    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    const int64_t tileX = world_coord::floorDiv(POSITION.x, int64_t{FAR_TERRAIN_TILE_EDGE});
    const int64_t tileZ = world_coord::floorDiv(POSITION.z, int64_t{FAR_TERRAIN_TILE_EDGE});
    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR}) {
        const std::shared_ptr<const FarTerrainMesh> mesh =
            FarTerrainMesher::build({tileX, tileZ, step}, source);
        const std::vector<FarWaterObservation> observations =
            farWaterAt(*mesh, static_cast<float>(POSITION.x - mesh->originX) + 0.5F,
                       static_cast<float>(POSITION.z - mesh->originZ) + 0.5F);
        double maximumHeightError = 0.0;
        double maximumTriangleRange = 0.0;
        for (const FarWaterObservation& observation : observations) {
            maximumHeightError = std::max(
                maximumHeightError, std::abs(static_cast<double>(observation.interpolatedHeight) -
                                             exactFluid.visibleSurface));
            maximumTriangleRange =
                std::max(maximumTriangleRange, static_cast<double>(observation.maximumHeight -
                                                                   observation.minimumHeight));
        }
        CAPTURE(farTerrainStepSize(step), exactFluid.visibleSurface, observations.size(),
                maximumHeightError, maximumTriangleRange, mesh->vertices.size(),
                mesh->indices.size(), mesh->byteSize());
        CHECK_FALSE(observations.empty());
        CHECK(maximumHeightError <= MAXIMUM_UNTAGGED_VISIBLE_STEP);
        CHECK(maximumTriangleRange <= MAXIMUM_UNTAGGED_VISIBLE_STEP);
        CHECK(mesh->waterContourTriangleCount == 0);
    }
}

TEST_CASE("Near far-water sentinels retain a minimum-width route at a cell edge",
          "[render][far-terrain][water][route][near-water-authority][regression]") {
    const FarTerrainSource source =
        FarTerrainMesher::surfaceGeometrySource([](int64_t x, int64_t, worldgen::SurfaceFootprint) {
            worldgen::SurfaceSample sample;
            sample.terrainHeight = 50.0;
            sample.hydrology.surfaceElevation = 50.0;
            sample.hydrology.channelWidth = 4.0;
            sample.hydrology.channelDistance = std::abs(static_cast<double>(x) - 1.0);
            sample.hydrology.river = sample.hydrology.channelDistance <= 2.0;
            if (sample.hydrology.river) {
                sample.hydrology.streamOrder = 1;
                sample.hydrology.waterSurface = 70.0;
                sample.hydrology.waterBodyId = 0xA11C'E001ULL;
                sample.waterSurface = 70.0;
            }
            return sample;
        });
    const std::shared_ptr<const FarTerrainMesh> mesh =
        FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const std::vector<FarWaterObservation> edgeRoute = farWaterAt(*mesh, 1.5F, 0.5F);
    const std::vector<FarWaterObservation> dry = farWaterAt(*mesh, 5.5F, 0.5F);
    REQUIRE_FALSE(edgeRoute.empty());
    CHECK(dry.empty());
    for (const FarWaterObservation& observation : edgeRoute) {
        CHECK(observation.interpolatedHeight == 70.0F);
        CHECK(observation.maximumHeight == observation.minimumHeight);
    }
}

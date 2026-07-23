#include <catch2/catch_all.hpp>

#include "test_helpers.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/learned_terrain.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace worldgen::learned;

constexpr uint64_t V4_WATER_ANTIPATTERN_SEED = 0x5644'5741'5445'5255ULL;
constexpr int NATIVE_SAMPLE_SPACING = worldgen::NATIVE_HYDROLOGY_RASTER_SPACING;
constexpr int SAMPLE_EDGE = 257;
constexpr int64_t SAMPLE_ORIGIN_X = 384;
constexpr int64_t SAMPLE_ORIGIN_Z = 512;

GenerationIdentity v4WaterIdentity() {
    GenerationIdentity identity;
    identity.seed = V4_WATER_ANTIPATTERN_SEED;
    identity.modelPackHash =
        *parseSha256("543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0");
    identity.runtimeHash =
        *parseSha256("e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6");
    return identity;
}

double learnedLakeHeight(int64_t worldX, int64_t worldZ) {
    // This is deliberately a smooth, asymmetric closed basin. It catches a
    // return to categorical sixteen-block bank dilation because the canonical
    // shore must curve through subcell coordinates before cube emission.
    const double localX = static_cast<double>(worldX) - 1'024.0;
    const double localZ = static_cast<double>(worldZ) - 1'024.0;
    const double warpedX = localX + 13.0 * std::sin(localZ / 71.0);
    const double warpedZ = (localZ + 9.0 * std::sin(localX / 61.0)) / 0.83;
    const double radius = std::hypot(warpedX, warpedZ);
    return 126.0 + std::min(22.0, radius * 0.25);
}

double quantizedLearnedLakeHeight(int64_t worldX, int64_t worldZ) {
    const double elevationMeters =
        (learnedLakeHeight(worldX, worldZ) - LEARNED_SEA_LEVEL) * WORLD_METERS_PER_BLOCK;
    const int16_t quantized = static_cast<int16_t>(
        std::clamp<long long>(std::llround(elevationMeters), std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()));
    return learnedElevationMetersToWorldHeight(static_cast<double>(quantized));
}

class CurvedValleyTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        ++calls_;
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeRow = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE + row;
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                const int64_t worldX = nativeColumn * MODEL_BLOCK_SCALE;
                const int64_t worldZ = nativeRow * MODEL_BLOCK_SCALE;
                const double elevationMeters =
                    (learnedLakeHeight(worldX, worldZ) - LEARNED_SEA_LEVEL) *
                    WORLD_METERS_PER_BLOCK;
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = static_cast<int16_t>(std::clamp<long long>(
                        std::llround(elevationMeters), std::numeric_limits<int16_t>::min(),
                        std::numeric_limits<int16_t>::max())),
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 0,
                    .precipitationCoefficientBasisPoints = 2'300,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    [[nodiscard]] uint64_t calls() const noexcept { return calls_.load(); }

private:
    std::atomic<uint64_t> calls_{0};
};

template <typename Operation>
auto awaitV4Authority(Operation&& operation) -> std::invoke_result_t<Operation> {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
    while (std::chrono::steady_clock::now() < deadline) {
        try {
            return operation();
        } catch (const GenerationFailureException& failure) {
            if (failure.status() != AuthorityStatus::DEFERRED)
                throw;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    throw std::runtime_error("v4 learned authority did not become ready before the test deadline");
}

class V4CurvedValleyFixture {
public:
    V4CurvedValleyFixture()
        : store_("v4_water_antipatterns"), identity_(v4WaterIdentity()),
          backend_(std::make_shared<CurvedValleyTerrainBackend>()),
          authority_(std::make_shared<CachedTerrainAuthority>(identity_, store_.path(), backend_)),
          context_(std::make_shared<WorldGenerationContext>(
              identity_, authority_, AuthorityQuality::FINAL,
              std::filesystem::path(store_.path()) / "hydrology-authority-v1")),
          generator_(identity_.seed, context_) {}

    ChunkGenerator& generator() { return generator_; }

    [[nodiscard]] uint64_t inferenceCalls() const noexcept { return backend_->calls(); }

private:
    TempDir store_;
    GenerationIdentity identity_;
    std::shared_ptr<CurvedValleyTerrainBackend> backend_;
    std::shared_ptr<CachedTerrainAuthority> authority_;
    std::shared_ptr<WorldGenerationContext> context_;
    ChunkGenerator generator_;
};

bool wet(const worldgen::SurfaceSample& surface) {
    return worldgen::generatedFluidColumn(surface).wet;
}

bool explicitFall(const worldgen::SurfaceSample& surface) {
    return surface.hydrology.waterfall &&
           surface.hydrology.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL &&
           surface.hydrology.transitionOwnerId != 0;
}

struct NativeRouteCell {
    int x = 0;
    int z = 0;
};

std::vector<NativeRouteCell>
largestConnectedWetRoute(const std::vector<worldgen::SurfaceSample>& samples) {
    const auto indexOf = [](int x, int z) { return static_cast<size_t>(z * SAMPLE_EDGE + x); };
    std::vector<uint8_t> visited(samples.size());
    std::vector<NativeRouteCell> largest;
    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};
    for (int startZ = 0; startZ < SAMPLE_EDGE; ++startZ) {
        for (int startX = 0; startX < SAMPLE_EDGE; ++startX) {
            const size_t start = indexOf(startX, startZ);
            if (visited[start] || !wet(samples[start]))
                continue;
            std::vector<NativeRouteCell> component;
            component.push_back({startX, startZ});
            visited[start] = 1;
            for (size_t cursor = 0; cursor < component.size(); ++cursor) {
                const NativeRouteCell cell = component[cursor];
                for (const auto [offsetX, offsetZ] : CARDINAL) {
                    const int nextX = cell.x + offsetX;
                    const int nextZ = cell.z + offsetZ;
                    if (nextX < 0 || nextZ < 0 || nextX >= SAMPLE_EDGE || nextZ >= SAMPLE_EDGE)
                        continue;
                    const size_t next = indexOf(nextX, nextZ);
                    if (visited[next] || !wet(samples[next]))
                        continue;
                    visited[next] = 1;
                    component.push_back({nextX, nextZ});
                }
            }
            if (component.size() > largest.size())
                largest = std::move(component);
        }
    }
    return largest;
}

} // namespace

TEST_CASE("V4 learned curved lakes retain supported continuous water without legacy banks",
          "[worldgen][hydrology][water][v4][learned][reported-water-continuity][regression]") {
    V4CurvedValleyFixture fixture;
    ChunkGenerator& generator = fixture.generator();
    REQUIRE(generator.usesLearnedAuthority());

    std::vector<worldgen::SurfaceSample> nativeSamples(
        static_cast<size_t>(SAMPLE_EDGE * SAMPLE_EDGE));
    std::vector<worldgen::HydrologySample> nativeAuthority(nativeSamples.size());
    awaitV4Authority([&] {
        generator.sampleNativeHydrologyGeometryGrid(SAMPLE_ORIGIN_X, SAMPLE_ORIGIN_Z,
                                                    NATIVE_SAMPLE_SPACING, NATIVE_SAMPLE_SPACING,
                                                    SAMPLE_EDGE, SAMPLE_EDGE, nativeSamples);
        return true;
    });
    awaitV4Authority([&] {
        generator.sampleNativeHydrologyAuthorityGrid(SAMPLE_ORIGIN_X, SAMPLE_ORIGIN_Z,
                                                     NATIVE_SAMPLE_SPACING, NATIVE_SAMPLE_SPACING,
                                                     SAMPLE_EDGE, SAMPLE_EDGE, nativeAuthority);
        return true;
    });

    REQUIRE(fixture.inferenceCalls() > 0);

    const std::vector<NativeRouteCell> route = largestConnectedWetRoute(nativeSamples);
    CAPTURE(route.size(), fixture.inferenceCalls());
    REQUIRE(route.size() >= 48);

    const auto sampleAt =
        [&nativeSamples](const NativeRouteCell& cell) -> worldgen::SurfaceSample const& {
        return nativeSamples[static_cast<size_t>(cell.z * SAMPLE_EDGE + cell.x)];
    };

    size_t unsupportedWetColumns = 0;
    size_t legacyBankFlags = 0;
    size_t dryNeighborColumns = 0;
    size_t lakeColumns = 0;
    size_t unownedWetTransitions = 0;
    size_t dryTerrainRises = 0;
    for (const NativeRouteCell& cell : route) {
        const worldgen::SurfaceSample& surface = sampleAt(cell);
        const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(surface);
        if (!fluid.wet)
            continue;
        unsupportedWetColumns += surface.terrainHeight >= fluid.visibleSurface - 0.01 ? 1U : 0U;
        legacyBankFlags += surface.hydrology.channelBank || surface.hydrology.lakeBank ? 1U : 0U;
        lakeColumns += surface.hydrology.lake ? 1U : 0U;

        for (const auto [offsetX, offsetZ] : std::array<std::pair<int, int>, 4>{
                 std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}}) {
            const int neighborX = cell.x + offsetX;
            const int neighborZ = cell.z + offsetZ;
            if (neighborX < 0 || neighborZ < 0 || neighborX >= SAMPLE_EDGE ||
                neighborZ >= SAMPLE_EDGE)
                continue;
            const int64_t worldX = SAMPLE_ORIGIN_X + neighborX * NATIVE_SAMPLE_SPACING;
            const int64_t worldZ = SAMPLE_ORIGIN_Z + neighborZ * NATIVE_SAMPLE_SPACING;
            const worldgen::SurfaceSample& neighbor =
                nativeSamples[static_cast<size_t>(neighborZ * SAMPLE_EDGE + neighborX)];
            if (wet(neighbor)) {
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                const double visibleStep =
                    std::abs(fluid.visibleSurface - neighborFluid.visibleSurface);
                unownedWetTransitions +=
                    !(explicitFall(surface) || explicitFall(neighbor) || visibleStep <= 0.125001)
                        ? 1U
                        : 0U;
                continue;
            }
            ++dryNeighborColumns;
            const size_t neighborIndex = static_cast<size_t>(neighborZ * SAMPLE_EDGE + neighborX);
            const worldgen::HydrologySample& authority = nativeAuthority[neighborIndex];
            const double learnedHeight = quantizedLearnedLakeHeight(worldX, worldZ);
            // The v4 route may cut a bed but must never raise a dry neighbor
            // to contain it. The synthetic v3 bank path did exactly that.
            dryTerrainRises += authority.surfaceElevation > learnedHeight + 1.0e-6 ? 1U : 0U;
            legacyBankFlags += authority.channelBank || authority.lakeBank ? 1U : 0U;
        }
    }
    CAPTURE(unsupportedWetColumns, legacyBankFlags, dryNeighborColumns, lakeColumns,
            unownedWetTransitions, dryTerrainRises);
    REQUIRE(unsupportedWetColumns == 0);
    REQUIRE(legacyBankFlags == 0);
    REQUIRE(unownedWetTransitions == 0);
    REQUIRE(dryTerrainRises == 0);
    REQUIRE(dryNeighborColumns > 0);
    REQUIRE(lakeColumns == route.size());

    const auto shoreline = std::ranges::min_element(
        route, [&](const NativeRouteCell& first, const NativeRouteCell& second) {
            return std::abs(sampleAt(first).hydrology.lakeShoreDistance) <
                   std::abs(sampleAt(second).hydrology.lakeShoreDistance);
        });
    REQUIRE(shoreline != route.end());
    const int64_t representativeX = SAMPLE_ORIGIN_X + shoreline->x * NATIVE_SAMPLE_SPACING;
    const int64_t representativeZ = SAMPLE_ORIGIN_Z + shoreline->z * NATIVE_SAMPLE_SPACING;

    // Native four-block input may not force a four-block staircase. Search
    // exact block coordinates around the curved shore for an oblique signed
    // distance gradient and a wet subcell sample.
    constexpr int EXACT_EDGE = 33;
    std::vector<worldgen::SurfaceSample> exactSamples(static_cast<size_t>(EXACT_EDGE * EXACT_EDGE));
    awaitV4Authority([&] {
        generator.sampleNativeHydrologyGeometryGrid(representativeX - EXACT_EDGE / 2,
                                                    representativeZ - EXACT_EDGE / 2, 1, 1,
                                                    EXACT_EDGE, EXACT_EDGE, exactSamples);
        return true;
    });
    bool foundCurvedSubcellShore = false;
    for (int z = 1; z + 1 < EXACT_EDGE; ++z) {
        for (int x = 1; x + 1 < EXACT_EDGE; ++x) {
            const int64_t worldX = representativeX - EXACT_EDGE / 2 + x;
            const int64_t worldZ = representativeZ - EXACT_EDGE / 2 + z;
            const worldgen::SurfaceSample& sample =
                exactSamples[static_cast<size_t>(z * EXACT_EDGE + x)];
            const double gradientX = (exactSamples[static_cast<size_t>(z * EXACT_EDGE + x + 1)]
                                          .hydrology.lakeShoreDistance -
                                      exactSamples[static_cast<size_t>(z * EXACT_EDGE + x - 1)]
                                          .hydrology.lakeShoreDistance) *
                                     0.5;
            const double gradientZ = (exactSamples[static_cast<size_t>((z + 1) * EXACT_EDGE + x)]
                                          .hydrology.lakeShoreDistance -
                                      exactSamples[static_cast<size_t>((z - 1) * EXACT_EDGE + x)]
                                          .hydrology.lakeShoreDistance) *
                                     0.5;
            if (wet(sample) && std::abs(sample.hydrology.lakeShoreDistance) < 2.5 &&
                (worldX % NATIVE_SAMPLE_SPACING != 0 || worldZ % NATIVE_SAMPLE_SPACING != 0) &&
                std::abs(gradientX) > 0.12 && std::abs(gradientZ) > 0.12) {
                foundCurvedSubcellShore = true;
            }
        }
    }
    REQUIRE(foundCurvedSubcellShore);

    // Exact cube emission must preserve a real solid bed below the canonical
    // fluid column. This rules out dry-route deletion and artificial support
    // columns independently of the far representation.
    std::array<ColumnPos, 3> exactPositions{};
    for (size_t index = 0; index < exactPositions.size(); ++index) {
        const NativeRouteCell& cell = route[(route.size() * (index + 1)) / 4];
        exactPositions[index] = {
            SAMPLE_ORIGIN_X + cell.x * NATIVE_SAMPLE_SPACING,
            SAMPLE_ORIGIN_Z + cell.z * NATIVE_SAMPLE_SPACING,
        };
    }
    std::array<worldgen::SurfaceSample, exactPositions.size()> emitted{};
    awaitV4Authority([&] {
        generator.sampleGeneratedWaterGeometryPoints(exactPositions, emitted);
        return true;
    });
    const worldgen::WaterBodyId lakeId = emitted.front().hydrology.waterBodyId;
    REQUIRE(lakeId != worldgen::NO_WATER_BODY);
    for (size_t index = 0; index < emitted.size(); ++index) {
        const worldgen::SurfaceSample& surface = emitted[index];
        const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(surface);
        CAPTURE(exactPositions[index].x, exactPositions[index].z, surface.terrainHeight,
                fluid.visibleSurface, fluid.topY, fluid.wet);
        REQUIRE(fluid.wet);
        REQUIRE(fluid.standing);
        REQUIRE(surface.terrainHeight < fluid.visibleSurface - 0.01);
        REQUIRE(surface.hydrology.waterBodyId == lakeId);
        const ColumnPos chunkColumn{Chunk::worldToChunk(exactPositions[index].x),
                                    Chunk::worldToChunk(exactPositions[index].z)};
        const std::shared_ptr<const ColumnPlan> plan =
            awaitV4Authority([&] { return generator.getColumnPlan(chunkColumn); });
        const int surfaceY = plan->surfaceY(Chunk::worldToLocal(exactPositions[index].x),
                                            Chunk::worldToLocal(exactPositions[index].z));
        Chunk bed(ChunkPos{chunkColumn.x, Chunk::worldToChunkY(surfaceY), chunkColumn.z});
        awaitV4Authority([&] {
            generator.generate(bed);
            return true;
        });
        REQUIRE(isSolid(bed.getBlock(Chunk::worldToLocal(exactPositions[index].x),
                                     Chunk::worldToLocalY(surfaceY),
                                     Chunk::worldToLocal(exactPositions[index].z))));
        Chunk water(ChunkPos{chunkColumn.x, Chunk::worldToChunkY(fluid.topY), chunkColumn.z});
        awaitV4Authority([&] {
            generator.generate(water);
            return true;
        });
        REQUIRE(water.getBlock(Chunk::worldToLocal(exactPositions[index].x),
                               Chunk::worldToLocalY(fluid.topY),
                               Chunk::worldToLocal(exactPositions[index].z)) == BlockType::WATER);
    }
}

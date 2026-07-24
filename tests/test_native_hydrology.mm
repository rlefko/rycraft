#include <catch2/catch_all.hpp>

#include "render/far_terrain.hpp"
#include "test_helpers.hpp"
#include "world/learned_terrain.hpp"
#include "world/macro_generation.hpp"
#include "world/native_hydrology.hpp"
#include "world/terrain_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace worldgen;

TEST_CASE("V4 explicit falls publish receiving stage across standing-water overlap",
          "[worldgen][hydrology][native-4][v4][lake][waterfall][stage][regression]") {
    constexpr double LAKE_STAGE = 80.0;
    constexpr double RECEIVING_STAGE = 64.0;
    REQUIRE(native_hydrology_detail::explicitFallPublishedWaterSurface(
                true, RECEIVING_STAGE, LAKE_STAGE) == RECEIVING_STAGE);
    REQUIRE(native_hydrology_detail::explicitFallPublishedWaterSurface(false, RECEIVING_STAGE,
                                                                       LAKE_STAGE) == LAKE_STAGE);
}

TEST_CASE("V4 bowed wide ribbons use monotone receiver-axis stage progress",
          "[worldgen][hydrology][native-4][v4][curve][stage][backwater][regression]") {
    constexpr double SOURCE_X = 4.0;
    constexpr double SOURCE_Z = 2.5;
    constexpr double RECEIVER_X = 0.0;
    constexpr double RECEIVER_Z = 2.5;
    constexpr double CONTROL_X = 2.0;
    constexpr double CONTROL_Z = 3.62;
    constexpr std::array<std::pair<double, double>, 2> ADJACENT_COLUMNS{{
        {1.5, 0.0},
        {2.5, 0.0},
    }};
    const auto curvePoint = [](double amount) {
        const double inverse = 1.0 - amount;
        return std::pair{
            inverse * inverse * SOURCE_X + 2.0 * inverse * amount * CONTROL_X +
                amount * amount * RECEIVER_X,
            inverse * inverse * SOURCE_Z + 2.0 * inverse * amount * CONTROL_Z +
                amount * amount * RECEIVER_Z,
        };
    };
    const auto nearestCurveAmount = [&](double worldX, double worldZ) {
        double selectedAmount = 0.0;
        double selectedDistance = std::numeric_limits<double>::infinity();
        for (int sample = 0; sample <= 10'000; ++sample) {
            const double amount = static_cast<double>(sample) / 10'000.0;
            const auto point = curvePoint(amount);
            const double distance = std::hypot(worldX - point.first, worldZ - point.second);
            if (distance < selectedDistance) {
                selectedAmount = amount;
                selectedDistance = distance;
            }
        }
        return selectedAmount;
    };

    std::array<double, 2> curveAmounts{};
    std::array<double, 2> hydraulicAmounts{};
    for (size_t index = 0; index < ADJACENT_COLUMNS.size(); ++index) {
        const auto [worldX, worldZ] = ADJACENT_COLUMNS[index];
        curveAmounts[index] = nearestCurveAmount(worldX, worldZ);
        hydraulicAmounts[index] = native_hydrology_detail::receiverAxisProgress(
            worldX, worldZ, SOURCE_X, SOURCE_Z, RECEIVER_X, RECEIVER_Z);
    }
    // These adjacent lateral samples sit on opposite sides of the bowed
    // curve's medial axis. Nearest-centerline arclength snaps almost an entire
    // receiver edge even though their physical separation is one block.
    REQUIRE(std::abs(curveAmounts[0] - curveAmounts[1]) > 0.75);
    REQUIRE(hydraulicAmounts[0] == Catch::Approx(0.625));
    REQUIRE(hydraulicAmounts[1] == Catch::Approx(0.375));

    const auto visibleStage = [](double stage) { return std::round(stage * 8.0) / 8.0; };
    const auto outletBackwaterStage = [](double amount) {
        constexpr double SOURCE_STAGE = 108.75;
        constexpr double RECEIVER_STAGE = 108.603553772;
        constexpr double STANDING_STAGE = 109.0;
        const double routedStage = std::lerp(SOURCE_STAGE, RECEIVER_STAGE, amount);
        const double smooth = amount * amount * (3.0 - 2.0 * amount);
        const double standingTarget = STANDING_STAGE - 0.5 * smooth;
        return std::lerp(routedStage, standingTarget, 1.0 - smooth);
    };
    const double firstStage = outletBackwaterStage(hydraulicAmounts[0]);
    const double secondStage = outletBackwaterStage(hydraulicAmounts[1]);
    REQUIRE(std::abs(visibleStage(firstStage) - visibleStage(secondStage)) <= 0.125001);
    REQUIRE(std::abs(visibleStage(outletBackwaterStage(curveAmounts[0])) -
                     visibleStage(outletBackwaterStage(curveAmounts[1]))) > 0.125001);
}

TEST_CASE("V4 reported tile 30,-5 keeps overlapping ordinary reaches exact",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty())
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_ROOT to a disposable copy of the reported preview "
             "hydrology store");
    constexpr uint64_t SEED = 11'940'042'767'486'971'292ULL;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::PREVIEW);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("reported persisted page unexpectedly rebuilt");
    };
    for (int64_t z = -1'284; z <= -1'020; z += 4) {
        for (int64_t x = 7'676; x <= 7'940; x += 4) {
            CAPTURE(x, z);
            REQUIRE(router.sample(x, z, forbidden).valid);
        }
    }
    const BasinSample hairpin = router.sample(7'922, -1'065, forbidden);
    const BasinSample adjacentRibbon = router.sample(7'922, -1'064, forbidden);
    REQUIRE(hairpin.river);
    REQUIRE(adjacentRibbon.river);
    REQUIRE_FALSE(hairpin.waterfall);
    REQUIRE_FALSE(adjacentRibbon.waterfall);
    REQUIRE(hairpin.waterBodyId == adjacentRibbon.waterBodyId);
    REQUIRE(std::abs(hairpin.waterSurface - adjacentRibbon.waterSurface) <= 0.125001);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads == 1);
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

TEST_CASE("V4 reported same-edge projection stays monotone across a bowed outlet ribbon",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_FINAL_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty()) {
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_FINAL_ROOT to a disposable copy of the reported "
             "final hydrology store");
    }
    constexpr uint64_t SEED = 7'551'868'678'105'131'611ULL;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("reported persisted page unexpectedly rebuilt");
    };
    for (int64_t z = 7'264; z < 7'296; ++z) {
        for (int64_t x = 3'264; x < 3'296; ++x) {
            CAPTURE(x, z);
            REQUIRE(router.sample(x, z, forbidden).valid);
        }
    }
    const BasinSample first = router.sample(3'291, 7'275, forbidden);
    const BasinSample second = router.sample(3'292, 7'275, forbidden);
    REQUIRE(first.river);
    REQUIRE(second.river);
    REQUIRE_FALSE(first.waterfall);
    REQUIRE_FALSE(second.waterfall);
    REQUIRE(first.waterBodyId == second.waterBodyId);
    REQUIRE(std::abs(first.waterSurface - second.waterSurface) <= 0.125001);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads == 4);
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

TEST_CASE("V4 reported consecutive-edge contact stays graded after receiver-axis projection",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_CURRENT_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty()) {
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_CURRENT_ROOT to a disposable copy of the reported "
             "preview hydrology store");
    }
    constexpr uint64_t SEED = 3'526'921'319'008'933'461ULL;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::PREVIEW);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("reported persisted page unexpectedly rebuilt");
    };
    // The reported contact lies in the certified halo of a neighboring
    // 32-column stage tile, not in the wet core of its own tile. Exercise the
    // complete surrounding 3-by-3 tile footprint so that exact sampling must
    // build every stage authority that can admit this pair.
    for (int64_t z = -1'600; z < -1'504; ++z) {
        for (int64_t x = -128; x < -32; ++x) {
            CAPTURE(x, z);
            REQUIRE(router.sample(x, z, forbidden).valid);
        }
    }
    const BasinSample upstream = router.sample(-85, -1'545, forbidden);
    const BasinSample downstream = router.sample(-85, -1'544, forbidden);
    REQUIRE(upstream.valid);
    REQUIRE(downstream.valid);
    if (upstream.river && downstream.river && !upstream.waterfall && !downstream.waterfall &&
        upstream.waterBodyId == downstream.waterBodyId) {
        REQUIRE(std::abs(upstream.waterSurface - downstream.waterSurface) <= 0.125001);
    }
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

TEST_CASE("V4 seed-42 production channel contacts retain ordinary stage authority",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_SEED42_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty()) {
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_SEED42_ROOT to an isolated copy of the seed-42 "
             "final hydrology store");
    }
    constexpr uint64_t SEED = 42;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("seed-42 persisted page unexpectedly rebuilt");
    };
    // The first and third pairs are wide round-cap contacts whose projections
    // are several routed blocks from their shared junction. The middle pair
    // stays on one edge but crosses the neighboring-ribbon blend boundary.
    constexpr std::array contacts{
        std::pair{ColumnPos{2'933, -4'323}, ColumnPos{2'934, -4'323}},
        std::pair{ColumnPos{2'977, -5'162}, ColumnPos{2'978, -5'162}},
        std::pair{ColumnPos{3'544, -4'335}, ColumnPos{3'545, -4'335}},
    };
    for (const auto& [firstPosition, secondPosition] : contacts) {
        CAPTURE(firstPosition.x, firstPosition.z, secondPosition.x, secondPosition.z);
        const BasinSample first = router.sample(firstPosition.x, firstPosition.z, forbidden);
        const BasinSample second = router.sample(secondPosition.x, secondPosition.z, forbidden);
        REQUIRE(first.valid);
        REQUIRE(second.valid);
        REQUIRE(first.river);
        REQUIRE(second.river);
        REQUIRE_FALSE(first.waterfall);
        REQUIRE_FALSE(second.waterfall);
        REQUIRE(first.waterBodyId == second.waterBodyId);
        REQUIRE(std::abs(first.waterSurface - second.waterSurface) <= 0.125001);
    }
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads > 0);
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

TEST_CASE("V4 seed-42 collapsed fall cap accepts a reconciled ordinary stage",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][waterfall][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_SEED42_ENTRY_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty()) {
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_SEED42_ENTRY_ROOT to the isolated seed-42 "
             "entry-capture final hydrology store");
    }
    constexpr uint64_t SEED = 42;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("seed-42 entry persisted page unexpectedly rebuilt");
    };

    // The second column selects a collapsed fall endpoint whose route stage
    // remains at the fall threshold (1408), while spatial ordinary authority
    // has already reconciled both published stages to 1404.
    const BasinSample first = router.sample(4'108, -4'976, forbidden);
    const BasinSample second = router.sample(4'109, -4'976, forbidden);
    REQUIRE(first.valid);
    REQUIRE(second.valid);
    REQUIRE(first.river);
    REQUIRE(second.river);
    REQUIRE_FALSE(first.waterfall);
    REQUIRE_FALSE(second.waterfall);
    REQUIRE(first.waterBodyId == second.waterBodyId);
    REQUIRE(first.waterSurface == Catch::Approx(175.5).margin(1.0e-6));
    REQUIRE(second.waterSurface == Catch::Approx(175.5).margin(1.0e-6));

    bool sawNearbyExplicitFall = false;
    for (int64_t z = -4'984; z <= -4'968; ++z) {
        for (int64_t x = 4'100; x <= 4'116; ++x) {
            const BasinSample sample = router.sample(x, z, forbidden);
            if (!sample.waterfall ||
                sample.transitionOwnerKind != WaterTransitionKind::EXPLICIT_FALL) {
                continue;
            }
            sawNearbyExplicitFall = true;
            REQUIRE(sample.waterfallTop >= sample.waterfallBottom + 0.5);
        }
    }
    REQUIRE(sawNearbyExplicitFall);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads > 0);
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

TEST_CASE("V4 user-seed consecutive junction remains monotonically graded",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][regression]"
          "[.reported-stage-tile]") {
    const char* rootEnvironment = std::getenv("RYCRAFT_REPORTED_HYDROLOGY_USER_SEED_ROOT");
    if (rootEnvironment == nullptr || std::string_view(rootEnvironment).empty()) {
        SKIP("Set RYCRAFT_REPORTED_HYDROLOGY_USER_SEED_ROOT to an isolated copy of the "
             "reported final hydrology store");
    }
    constexpr uint64_t SEED = 9'522'641'655'513'703'167ULL;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(rootEnvironment),
        worldgen::runtime::productionGenerationIdentity(SEED),
        worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    const NativeHydrologyInputFunction forbidden = [](std::span<const NativeHydrologyPosition>,
                                                      std::span<NativeHydrologyInput>) {
        throw std::runtime_error("user-seed persisted page unexpectedly rebuilt");
    };
    // These columns select the consecutive edges (20,-5080)->(16,-5076) and
    // (16,-5076)->(20,-5072), but meet far from their shared endpoint through
    // the broad round caps. Their visible same-reach union must still grade.
    const BasinSample first = router.sample(22, -5'075, forbidden);
    const BasinSample second = router.sample(22, -5'074, forbidden);
    REQUIRE(first.valid);
    REQUIRE(second.valid);
    REQUIRE(first.river);
    REQUIRE(second.river);
    REQUIRE_FALSE(first.waterfall);
    REQUIRE_FALSE(second.waterfall);
    REQUIRE(first.waterBodyId == second.waterBodyId);
    REQUIRE(std::abs(first.waterSurface - second.waterSurface) <= 0.125001);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.persistedLoads > 0);
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
}

NativeHydrologyInput climateInput(double elevationMeters) {
    return {
        .elevationMeters = elevationMeters,
        .climate =
            {
                .meanTemperatureC = 16.0,
                .temperatureVariabilityC = 9.0,
                .annualPrecipitationMm = 2'400.0,
                .precipitationCoefficientOfVariation = 0.22,
                .lapseRateCPerMeter = -0.0065,
                .potentialEvapotranspirationMm = 420.0,
            },
    };
}

NativeHydrologyInputFunction planarNativeInput() {
    return [](std::span<const NativeHydrologyPosition> positions,
              std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const double elevation =
                130.0 - positions[index].x * 0.004 - positions[index].z * 0.002;
            output[index] = climateInput(elevation);
        }
    };
}

NativeHydrologyInputFunction oceanVentedDryIslandInput(int64_t centerX = 1'024,
                                                       int64_t centerZ = 1'024) {
    return [centerX, centerZ](std::span<const NativeHydrologyPosition> positions,
                              std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const double radius =
                std::hypot(positions[index].x - centerX, positions[index].z - centerZ);
            output[index] = climateInput(radius <= 320.0 ? 24.0 - radius * 0.02 : -1.0);
            output[index].climate.annualPrecipitationMm = 0.0;
            output[index].climate.potentialEvapotranspirationMm = 1'800.0;
        }
    };
}

NativeHydrologyInputFunction steppedSlopeNativeInput() {
    return [](std::span<const NativeHydrologyPosition> positions,
              std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const NativeHydrologyPosition position = positions[index];
            // The center sample is exactly on a positive-height conversion
            // boundary. Its east and south neighbors each occupy the next
            // lower world height, making the persisted four-block terrain
            // gradient observably nonzero.
            const double elevation = 97.5 - (static_cast<double>(position.x) - 1'024.0) * 0.1 -
                                     (static_cast<double>(position.z) - 1'024.0) * 0.05;
            output[index] = climateInput(elevation);
        }
    };
}

NativeHydrologyInputFunction isolatedOceanWetlandInput() {
    return [](std::span<const NativeHydrologyPosition> positions,
              std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const NativeHydrologyPosition position = positions[index];
            // A two-by-two native fringe remains resolvable after the shared
            // align_corners=false block-center reconstruction. Its seaward
            // cells drain directly into the ocean while local runoff remains
            // below the channel threshold.
            const bool fringe = (position.x == 1'024 || position.x == 1'028) &&
                                (position.z == 1'024 || position.z == 1'028);
            output[index] = climateInput(fringe ? 0.2 : -1.0);
            output[index].climate.annualPrecipitationMm = 1'280.0;
            output[index].climate.potentialEvapotranspirationMm = 0.0;
            output[index].climate.precipitationCoefficientOfVariation = 0.20;
        }
    };
}

NativeHydrologyInputFunction connectedCoastalWetlandInput() {
    return [](std::span<const NativeHydrologyPosition> positions,
              std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const NativeHydrologyPosition position = positions[index];
            constexpr int64_t COAST_X = 2'084;
            const bool corridor =
                position.z == 1'024 && position.x >= 1'980 && position.x < COAST_X;
            if (position.x >= COAST_X) {
                output[index] = climateInput(-1.0);
            } else if (!corridor) {
                output[index] = climateInput(50.0);
            } else {
                output[index] = climateInput(0.25 + (COAST_X - position.x) * 0.002);
            }
            output[index].climate.annualPrecipitationMm = corridor ? 1'280.0 : 0.0;
            output[index].climate.potentialEvapotranspirationMm = 0.0;
            output[index].climate.precipitationCoefficientOfVariation = 0.20;
        }
    };
}

NativeHydrologyInputFunction lowGradientEstuaryInput(int64_t coastX = 2'060,
                                                     int64_t valleyCenterZ = 1'024) {
    return [coastX, valleyCenterZ](std::span<const NativeHydrologyPosition> positions,
                                   std::span<NativeHydrologyInput> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            const NativeHydrologyPosition position = positions[index];
            if (position.x >= coastX) {
                output[index] = climateInput(-0.05);
            } else {
                const double valleyRise = std::abs(position.z - valleyCenterZ) * 0.02;
                output[index] = climateInput(0.25 + (coastX - position.x) * 0.002 + valleyRise);
            }
            output[index].climate.annualPrecipitationMm = 2'800.0;
            output[index].climate.potentialEvapotranspirationMm = 350.0;
            output[index].climate.precipitationCoefficientOfVariation = 0.18;
        }
    };
}

worldgen::learned::GenerationIdentity nativeHydrologyIdentity(uint64_t seed) {
    using worldgen::learned::parseSha256;
    worldgen::learned::GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash =
        *parseSha256("543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0");
    identity.runtimeHash =
        *parseSha256("e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6");
    return identity;
}

class DryIslandTerrainAuthority final : public worldgen::learned::TerrainAuthority {
public:
    explicit DryIslandTerrainAuthority(worldgen::learned::GenerationIdentity identity)
        : identity_(std::move(identity)) {}

    [[nodiscard]] const worldgen::learned::GenerationIdentity&
    generationIdentity() const noexcept override {
        return identity_;
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey,
                worldgen::learned::AuthorityRequestPriority) override {
        ++pageQueries_;
        return worldgen::learned::AuthorityResult<
            std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::failed({
            .code = worldgen::learned::GenerationFailureCode::INVALID_REQUEST,
            .message = "Dry-island authority does not persist pages",
            .retriable = false,
        });
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    queryNative(worldgen::learned::NativeRect region, worldgen::learned::AuthorityQuality,
                worldgen::learned::AuthorityRequestPriority) override {
        ++nativeGridQueries_;
        return worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>::ready(
            makeGrid(region));
    }

    worldgen::learned::AuthorityResult<std::vector<worldgen::learned::PhysicalTerrainSample>>
    queryNativePoints(std::span<const worldgen::learned::NativePoint> points,
                      worldgen::learned::AuthorityQuality,
                      worldgen::learned::AuthorityRequestPriority) override {
        ++pointQueries_;
        std::vector<worldgen::learned::PhysicalTerrainSample> output;
        output.reserve(points.size());
        for (const worldgen::learned::NativePoint point : points)
            output.push_back(sample(point.row, point.column));
        return worldgen::learned::AuthorityResult<
            std::vector<worldgen::learned::PhysicalTerrainSample>>::ready(std::move(output));
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(worldgen::learned::NativeRect region,
                                  worldgen::learned::AuthorityRequestPriority) override {
        ++transientGridQueries_;
        return worldgen::learned::
            AuthorityResult<std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>::ready(
                std::make_shared<const worldgen::learned::PhysicalTerrainGrid>(makeGrid(region)));
    }

    [[nodiscard]] worldgen::learned::TerrainAuthorityCacheMetrics cacheMetrics() const override {
        return {};
    }

    [[nodiscard]] uint64_t pointQueries() const noexcept { return pointQueries_.load(); }
    [[nodiscard]] uint64_t transientGridQueries() const noexcept {
        return transientGridQueries_.load();
    }
    [[nodiscard]] uint64_t pageQueries() const noexcept { return pageQueries_.load(); }

private:
    static worldgen::learned::PhysicalTerrainSample sample(int64_t row, int64_t column) {
        constexpr int64_t CENTER_X = 1'024;
        constexpr int64_t CENTER_Z = 1'024;
        const double worldX = static_cast<double>(column * worldgen::learned::MODEL_BLOCK_SCALE);
        const double worldZ = static_cast<double>(row * worldgen::learned::MODEL_BLOCK_SCALE);
        const double radius = std::hypot(worldX - CENTER_X, worldZ - CENTER_Z);
        return {
            .elevationMeters = radius <= 320.0 ? 24.0 - radius * 0.02 : -1.0,
            .meanTemperatureC = 16.0,
            .temperatureVariabilityC = 9.0,
            .annualPrecipitationMm = 0.0,
            .precipitationCoefficientOfVariation = 0.22,
            .lapseRateCPerMeter = -0.0065,
        };
    }

    static worldgen::learned::PhysicalTerrainGrid makeGrid(worldgen::learned::NativeRect region) {
        worldgen::learned::PhysicalTerrainGrid grid{.region = region};
        if (!region.valid())
            return grid;
        grid.samples.reserve(static_cast<size_t>(region.height() * region.width()));
        for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
            for (int64_t column = region.columnBegin; column < region.columnEnd; ++column)
                grid.samples.push_back(sample(row, column));
        }
        return grid;
    }

    worldgen::learned::GenerationIdentity identity_;
    std::atomic<uint64_t> pageQueries_{0};
    std::atomic<uint64_t> nativeGridQueries_{0};
    std::atomic<uint64_t> pointQueries_{0};
    std::atomic<uint64_t> transientGridQueries_{0};
};

uint64_t nativeWaterHash(const BasinSample& sample) {
    uint64_t hash = 1'469'598'103'934'665'603ULL;
    const auto combine = [&](uint64_t value) {
        hash ^= value;
        hash *= 1'099'511'628'211ULL;
    };
    combine(sample.waterBodyId);
    combine(sample.generatedFluidLevel);
    combine(static_cast<uint64_t>(sample.transitionOwnerKind));
    combine(sample.transitionOwnerId);
    combine(std::bit_cast<uint64_t>(sample.surfaceElevation));
    combine(std::bit_cast<uint64_t>(sample.terrainSlope));
    combine(std::bit_cast<uint64_t>(sample.waterSurface));
    combine(std::bit_cast<uint64_t>(sample.discharge));
    combine(std::bit_cast<uint64_t>(sample.lakeShoreDistance));
    combine(sample.streamOrder);
    combine(sample.distributaryCount);
    combine(
        static_cast<uint64_t>(sample.ocean) << 0U | static_cast<uint64_t>(sample.river) << 1U |
        static_cast<uint64_t>(sample.lake) << 2U | static_cast<uint64_t>(sample.waterfall) << 3U |
        static_cast<uint64_t>(sample.delta) << 4U | static_cast<uint64_t>(sample.wetland) << 5U |
        static_cast<uint64_t>(sample.estuary) << 6U | static_cast<uint64_t>(sample.brackish) << 7U);
    return hash;
}

TEST_CASE("V4 hydrology samples native spacing across signed page boundaries",
          "[worldgen][hydrology][native-4][v4][page-edge]") {
    struct InputBatch {
        int64_t minimumX = std::numeric_limits<int64_t>::max();
        int64_t maximumX = std::numeric_limits<int64_t>::min();
        int64_t minimumZ = std::numeric_limits<int64_t>::max();
        int64_t maximumZ = std::numeric_limits<int64_t>::min();
        size_t count = 0;
        bool aligned = true;
    };
    std::vector<InputBatch> batches;
    const NativeHydrologyInputFunction input =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            InputBatch batch;
            batch.count = positions.size();
            for (size_t index = 0; index < positions.size(); ++index) {
                batch.minimumX = std::min(batch.minimumX, positions[index].x);
                batch.maximumX = std::max(batch.maximumX, positions[index].x);
                batch.minimumZ = std::min(batch.minimumZ, positions[index].z);
                batch.maximumZ = std::max(batch.maximumZ, positions[index].z);
                batch.aligned =
                    batch.aligned && positions[index].x % 4 == 0 && positions[index].z % 4 == 0;
                output[index] =
                    climateInput(120.0 - positions[index].x * 0.001 - positions[index].z * 0.0005);
            }
            batches.push_back(batch);
        };

    NativeHydrologyRouter router(0x1234'5678'9ABC'DEF0ULL);
    REQUIRE(router.sample(-1.0, -1.0, input).valid);
    REQUIRE(router.sample(0.0, 0.0, input).valid);
    constexpr size_t RASTER_EDGE = 2'048 / 4 + 1 + 4;
    constexpr size_t RASTER_CELLS = RASTER_EDGE * RASTER_EDGE;
    std::vector<InputBatch> nativeBatches;
    for (const InputBatch& batch : batches) {
        REQUIRE(batch.count == RASTER_CELLS);
        REQUIRE(batch.aligned);
        nativeBatches.push_back(batch);
    }
    REQUIRE(nativeBatches.size() >= 4);
    REQUIRE(nativeBatches.size() <= 8);
    REQUIRE(batches.size() == nativeBatches.size());
    const auto observed = [&](int64_t minimumX, int64_t maximumX, int64_t minimumZ,
                              int64_t maximumZ) {
        return std::ranges::any_of(nativeBatches, [&](const InputBatch& batch) {
            return batch.minimumX == minimumX && batch.maximumX == maximumX &&
                   batch.minimumZ == minimumZ && batch.maximumZ == maximumZ;
        });
    };
    REQUIRE(observed(-2'056, 8, -2'056, 8));
    REQUIRE(observed(-8, 2'056, -2'056, 8));
    REQUIRE(observed(-2'056, 8, -8, 2'056));
    REQUIRE(observed(-8, 2'056, -8, 2'056));

    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.builds == nativeBatches.size());
    REQUIRE(metrics.entries <= NATIVE_HYDROLOGY_MAX_HANDOFF_PAGES);
    REQUIRE(metrics.bytes <= NATIVE_HYDROLOGY_CACHE_BYTE_BUDGET);
    REQUIRE(metrics.peakBuildBytes <= NATIVE_HYDROLOGY_MAX_BUILD_BYTES);
}

TEST_CASE("V4 native wetlands inherit one connected parent body and stage",
          "[worldgen][hydrology][native-4][v4][wetland][exact][determinism]") {
    constexpr uint64_t SEED = 0x5745'544C'414E'4404ULL;
    constexpr std::array<BasinSamplePosition, 9> positions{{
        {1'018.0, 1'018.0},
        {1'026.0, 1'018.0},
        {1'034.0, 1'018.0},
        {1'018.0, 1'026.0},
        {1'026.0, 1'026.0},
        {1'034.0, 1'026.0},
        {1'018.0, 1'034.0},
        {1'026.0, 1'034.0},
        {1'034.0, 1'034.0},
    }};

    NativeHydrologyRouter router(SEED);
    std::array<BasinSample, positions.size()> exact{};
    router.samplePoints(positions, isolatedOceanWetlandInput(), exact);
    const BasinSample& wetland = exact[4];
    CAPTURE(wetland.ocean, wetland.lake, wetland.river, wetland.waterfall, wetland.surfaceElevation,
            wetland.waterSurface, wetland.groundwaterHead, wetland.discharge,
            wetland.lakeShoreDistance);
    REQUIRE(wetland.valid);
    REQUIRE(wetland.wetland);
    REQUIRE_FALSE(wetland.ocean);
    REQUIRE_FALSE(wetland.lake);
    REQUIRE_FALSE(wetland.river);
    REQUIRE(wetland.waterBodyId != NO_WATER_BODY);
    REQUIRE(wetland.waterSurface == Catch::Approx(64.0));
    REQUIRE(wetland.surfaceElevation < wetland.waterSurface);
    REQUIRE(wetland.groundwaterHead >= wetland.waterSurface);
    REQUIRE(wetland.hydroperiod >= 0.55);
    for (size_t index = 0; index < exact.size(); ++index) {
        if (index == 4)
            continue;
        REQUIRE(exact[index].ocean);
        REQUIRE(exact[index].waterBodyId == wetland.waterBodyId);
        REQUIRE(exact[index].waterSurface == wetland.waterSurface);
    }

    std::array<BasinSample, positions.size()> grid{};
    router.sampleGrid(1'018, 1'018, 8, 8, 3, 3, isolatedOceanWetlandInput(), grid);
    for (size_t index = 0; index < grid.size(); ++index)
        REQUIRE(nativeWaterHash(grid[index]) == nativeWaterHash(exact[index]));

    router.clear();
    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    std::array<BasinSample, positions.size()> reversed{};
    router.samplePoints(reversedPositions, isolatedOceanWetlandInput(), reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < exact.size(); ++index)
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(exact[index]));
}

TEST_CASE("V4 native wetlands inherit connected groundwater authority across page seams",
          "[worldgen][hydrology][native-4][v4][wetland][groundwater][page-edge][persistence]") {
    constexpr uint64_t SEED = 0x5745'544C'414E'4405ULL;
    constexpr std::array<BasinSamplePosition, 5> positions{{
        {2'002.0, 1'026.0},
        {2'026.0, 1'026.0},
        {2'046.0, 1'026.0},
        {2'054.0, 1'026.0},
        {2'086.0, 1'026.0},
    }};
    const NativeHydrologyInputFunction input = connectedCoastalWetlandInput();
    NativeHydrologyRouter router(SEED);
    std::array<BasinSample, positions.size()> forward{};
    router.samplePoints(positions, input, forward);
    REQUIRE(router.cacheMetrics().connectedWetlandEntries > 0);
    REQUIRE(forward[3].river);
    REQUIRE(forward.back().ocean);
    for (size_t index = 0; index < 3; ++index) {
        const BasinSample& wetland = forward[index];
        CAPTURE(index, wetland.ocean, wetland.lake, wetland.river, wetland.wetland,
                wetland.waterSurface, wetland.surfaceElevation, wetland.discharge,
                wetland.groundwaterHead);
        REQUIRE(wetland.valid);
        REQUIRE(wetland.wetland);
        REQUIRE_FALSE(wetland.ocean);
        REQUIRE_FALSE(wetland.lake);
        REQUIRE_FALSE(wetland.river);
        REQUIRE(wetland.waterBodyId != NO_WATER_BODY);
        REQUIRE(wetland.waterBodyId == forward[3].waterBodyId);
        REQUIRE(wetland.waterSurface == forward[3].waterSurface);
        if (index > 0)
            REQUIRE(wetland.waterSurface >= forward[index - 1].waterSurface);
        REQUIRE(wetland.surfaceElevation < wetland.waterSurface);
        REQUIRE(wetland.surfaceElevation <= 64.0);
        REQUIRE(wetland.groundwaterHead >= wetland.waterSurface);
        REQUIRE(wetland.erosionDepth >= 0.0);
    }

    std::array<BasinSample, 7> grid{};
    router.sampleGrid(2'038, 1'026, 4, 4, static_cast<int>(grid.size()), 1, input, grid);
    for (size_t index = 0; index < grid.size(); ++index) {
        const BasinSample point = router.sample(2'038.0 + index * 4.0, 1'026.0, input);
        REQUIRE(nativeWaterHash(grid[index]) == nativeWaterHash(point));
        REQUIRE((grid[index].wetland || grid[index].river || grid[index].ocean));
    }
    std::array<NativeHydrologyTopologyCell, 2> topology{};
    router.sampleTopologyGrid(2'016, 1'024, 2, 1, input, topology);
    REQUIRE(std::ranges::all_of(topology, [](const NativeHydrologyTopologyCell& cell) {
        return cell.waterTopologyPossible;
    }));

    router.clear();
    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    std::array<BasinSample, positions.size()> reversed{};
    router.samplePoints(reversedPositions, input, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index)
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(forward[index]));

    TempDir directory("native_hydrology_connected_wetland_restart");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    std::array<BasinSample, positions.size()> persisted{};
    {
        NativeHydrologyRouter writer(SEED, store);
        writer.samplePoints(positions, input, persisted);
        REQUIRE(writer.cacheMetrics().persistedWrites == 2);
    }
    bool queriedInput = false;
    NativeHydrologyRouter reader(SEED, store);
    std::array<BasinSample, positions.size()> restored{};
    reader.samplePoints(
        reversedPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted wetland page unexpectedly rebuilt");
        },
        restored);
    std::ranges::reverse(restored);
    REQUIRE_FALSE(queriedInput);
    REQUIRE(reader.cacheMetrics().persistedLoads == 2);
    for (size_t index = 0; index < restored.size(); ++index)
        REQUIRE(nativeWaterHash(restored[index]) == nativeWaterHash(persisted[index]));
}

TEST_CASE("V4 native hydrology output is independent of build concurrency",
          "[worldgen][hydrology][native-4][v4][concurrency]") {
    constexpr uint64_t SEED = 0x5745'544C'414E'4405ULL;
    constexpr std::array<BasinSamplePosition, 5> positions{{
        {2'002.0, 1'026.0},
        {2'026.0, 1'026.0},
        {2'046.0, 1'026.0},
        {2'054.0, 1'026.0},
        {2'086.0, 1'026.0},
    }};
    const NativeHydrologyInputFunction input = connectedCoastalWetlandInput();

    std::array<BasinSample, positions.size()> baseline{};
    {
        NativeHydrologyRouter router(SEED);
        router.samplePoints(positions, input, baseline);
    }

    // Many workers share one router so concurrent page builds pass through the
    // camera-aware admission gate. Per-page single flight and the reservation
    // gate must still yield output identical to the serial baseline.
    NativeHydrologyRouter shared(SEED);
    std::atomic<bool> mismatch{false};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&]() {
            for (int repeat = 0; repeat < 4; ++repeat) {
                std::array<BasinSample, positions.size()> observed{};
                shared.samplePoints(positions, input, observed);
                for (size_t index = 0; index < observed.size(); ++index) {
                    if (nativeWaterHash(observed[index]) != nativeWaterHash(baseline[index]))
                        mismatch.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread& worker : workers)
        worker.join();
    CHECK_FALSE(mismatch.load());
}

TEST_CASE("V4 low-gradient mouths retain sea backwater and deterministic distributaries",
          "[worldgen][hydrology][native-4][v4][estuary][delta][brackish][page-edge][negative]") {
    constexpr uint64_t SEED = 0x4553'5455'4152'5901ULL;
    const NativeHydrologyInputFunction input = lowGradientEstuaryInput();
    NativeHydrologyRouter router(SEED);
    const BasinSample outsideBackwater = router.sample(1'702.0, 1'026.0, input);
    const BasinSample estuary = router.sample(1'810.0, 1'026.0, input);
    CAPTURE(outsideBackwater.river, outsideBackwater.estuary, outsideBackwater.channelGradient,
            outsideBackwater.discharge, outsideBackwater.waterSurface, estuary.river,
            estuary.estuary, estuary.channelGradient, estuary.discharge, estuary.waterSurface);
    REQUIRE(outsideBackwater.valid);
    REQUIRE(outsideBackwater.river);
    REQUIRE_FALSE(outsideBackwater.estuary);
    REQUIRE_FALSE(outsideBackwater.brackish);
    REQUIRE(estuary.valid);
    REQUIRE(estuary.river);
    REQUIRE(estuary.estuary);
    REQUIRE(estuary.brackish);
    REQUIRE(estuary.waterSurface >= 64.0);
    REQUIRE(outsideBackwater.waterSurface >= estuary.waterSurface);
    REQUIRE(estuary.surfaceElevation <= estuary.waterSurface - 0.125);
    REQUIRE(estuary.surfaceElevation <= 64.0);
    REQUIRE(estuary.groundwaterHead >= estuary.waterSurface);
    REQUIRE(estuary.erosionDepth >= 0.0);
    REQUIRE(router.cacheMetrics().seaBackwaterEntries > 0);

    std::vector<BasinSamplePosition> mouthPositions;
    for (int64_t z = 1'000; z <= 1'048; ++z) {
        for (int64_t x = 2'048; x <= 2'080; ++x)
            mouthPositions.push_back({.x = static_cast<double>(x), .z = static_cast<double>(z)});
    }
    std::vector<BasinSample> mouth(mouthPositions.size());
    router.samplePoints(mouthPositions, input, mouth);
    const auto delta = std::ranges::find_if(mouth, [](const BasinSample& sample) {
        return sample.delta && sample.distributaryCount >= 2;
    });
    REQUIRE(delta != mouth.end());
    CAPTURE(delta->ocean, delta->river, delta->waterSurface, delta->surfaceElevation,
            delta->discharge, delta->channelGradient, delta->streamOrder, delta->distributaryCount,
            delta->waterBodyId);
    REQUIRE(delta->estuary);
    REQUIRE(delta->brackish);
    REQUIRE(delta->waterBodyId != NO_WATER_BODY);
    REQUIRE(delta->streamOrder > 0);
    REQUIRE(delta->waterSurface >= delta->surfaceElevation);
    REQUIRE(delta->surfaceElevation <= 64.0);
    REQUIRE(delta->distributaryCount == 2);
    const bool distinctBranchDirection = std::ranges::any_of(mouth, [&](const BasinSample& sample) {
        return sample.delta && sample.waterBodyId == delta->waterBodyId &&
               (std::abs(sample.flowX - delta->flowX) > 0.05 ||
                std::abs(sample.flowZ - delta->flowZ) > 0.05);
    });
    REQUIRE(distinctBranchDirection);
    std::array<NativeHydrologyTopologyCell, 2> topology{};
    router.sampleTopologyGrid(2'048, 992, 1, 2, input, topology);
    REQUIRE(std::ranges::any_of(topology, [](const NativeHydrologyTopologyCell& cell) {
        return cell.waterTopologyPossible;
    }));

    constexpr int GRID_EDGE = 9;
    std::array<BasinSample, GRID_EDGE> grid{};
    router.sampleGrid(1'790, 1'026, 4, 4, GRID_EDGE, 1, input, grid);
    for (size_t index = 0; index < grid.size(); ++index) {
        const BasinSample point = router.sample(1'790.0 + index * 4.0, 1'026.0, input);
        REQUIRE(nativeWaterHash(grid[index]) == nativeWaterHash(point));
    }

    std::vector<BasinSamplePosition> reversedPositions = mouthPositions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(SEED);
    std::vector<BasinSample> reversed(reversedPositions.size());
    reverseRouter.samplePoints(reversedPositions, input, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < mouth.size(); ++index)
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(mouth[index]));
    const BasinSample reverseEstuary = reverseRouter.sample(1'810.0, 1'026.0, input);
    REQUIRE(nativeWaterHash(reverseEstuary) == nativeWaterHash(estuary));

    const NativeHydrologyInputFunction signedInput = lowGradientEstuaryInput(12, -1'024);
    NativeHydrologyRouter signedRouter(SEED);
    const BasinSample signedOutside = signedRouter.sample(-350.0, -1'022.0, signedInput);
    const BasinSample signedEstuary = signedRouter.sample(-238.0, -1'022.0, signedInput);
    REQUIRE(signedOutside.river);
    REQUIRE_FALSE(signedOutside.estuary);
    REQUIRE(signedEstuary.river);
    REQUIRE(signedEstuary.estuary);
    REQUIRE(signedEstuary.brackish);
    std::vector<BasinSamplePosition> signedMouthPositions;
    for (int64_t z = -1'048; z <= -1'000; ++z) {
        for (int64_t x = 0; x <= 32; ++x)
            signedMouthPositions.push_back(
                {.x = static_cast<double>(x), .z = static_cast<double>(z)});
    }
    std::vector<BasinSample> signedMouth(signedMouthPositions.size());
    signedRouter.samplePoints(signedMouthPositions, signedInput, signedMouth);
    REQUIRE(std::ranges::any_of(signedMouth, [](const BasinSample& sample) {
        return sample.delta && sample.estuary && sample.brackish && sample.distributaryCount == 2;
    }));

    TempDir directory("native_hydrology_estuary_delta_restart");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    std::vector<BasinSample> persistedMouth(mouthPositions.size());
    BasinSample persistedEstuary;
    {
        NativeHydrologyRouter writer(SEED, store);
        writer.samplePoints(mouthPositions, input, persistedMouth);
        persistedEstuary = writer.sample(1'810.0, 1'026.0, input);
        REQUIRE(writer.cacheMetrics().persistedWrites == 2);
    }
    bool queriedInput = false;
    NativeHydrologyRouter reader(SEED, store);
    const BasinSample restoredEstuary = reader.sample(
        1'810.0, 1'026.0,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted estuary page unexpectedly rebuilt");
        });
    std::vector<BasinSample> restoredMouth(mouthPositions.size());
    reader.samplePoints(
        mouthPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted delta page unexpectedly rebuilt");
        },
        restoredMouth);
    REQUIRE_FALSE(queriedInput);
    REQUIRE(reader.cacheMetrics().persistedLoads == 2);
    REQUIRE(nativeWaterHash(restoredEstuary) == nativeWaterHash(persistedEstuary));
    for (size_t index = 0; index < persistedMouth.size(); ++index)
        REQUIRE(nativeWaterHash(restoredMouth[index]) == nativeWaterHash(persistedMouth[index]));
}

TEST_CASE("V4 cold native pages issue one owner-bounded learned input query",
          "[worldgen][hydrology][native-4][v4][performance][query-bound]") {
    size_t calls = 0;
    size_t samples = 0;
    int64_t minimumX = std::numeric_limits<int64_t>::max();
    int64_t maximumX = std::numeric_limits<int64_t>::min();
    int64_t minimumZ = std::numeric_limits<int64_t>::max();
    int64_t maximumZ = std::numeric_limits<int64_t>::min();
    const NativeHydrologyInputFunction observed =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            ++calls;
            samples += positions.size();
            for (size_t index = 0; index < positions.size(); ++index) {
                minimumX = std::min(minimumX, positions[index].x);
                maximumX = std::max(maximumX, positions[index].x);
                minimumZ = std::min(minimumZ, positions[index].z);
                maximumZ = std::max(maximumZ, positions[index].z);
                const double normalizedX = (positions[index].x - 1'024.0) / 2'800.0;
                const double normalizedZ = (positions[index].z - 1'024.0) / 600.0;
                output[index] = climateInput(
                    90.0 + std::min(10.0, std::hypot(normalizedX, normalizedZ) * 10.0));
            }
        };
    NativeHydrologyRouter router(0xB01D'ED00'0004ULL);
    const BasinSample routed = router.sample(1'024.0, 1'024.0, observed);
    REQUIRE(routed.valid);
    REQUIRE(routed.lake);
    constexpr size_t RASTER_EDGE = 2'048 / 4 + 1 + 4;
    REQUIRE(calls == 1);
    REQUIRE(samples == RASTER_EDGE * RASTER_EDGE);
    REQUIRE(minimumX == -8);
    REQUIRE(maximumX == 2'056);
    REQUIRE(minimumZ == -8);
    REQUIRE(maximumZ == 2'056);
}

TEST_CASE("V4 direct native owner preparation never opens a neighboring semantic closure",
          "[worldgen][hydrology][native-4][v4][spawn][direct-owner]") {
    size_t calls = 0;
    int64_t minimumX = std::numeric_limits<int64_t>::max();
    int64_t maximumX = std::numeric_limits<int64_t>::min();
    int64_t minimumZ = std::numeric_limits<int64_t>::max();
    int64_t maximumZ = std::numeric_limits<int64_t>::min();
    const NativeHydrologyInputFunction observed =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            ++calls;
            for (size_t index = 0; index < positions.size(); ++index) {
                minimumX = std::min(minimumX, positions[index].x);
                maximumX = std::max(maximumX, positions[index].x);
                minimumZ = std::min(minimumZ, positions[index].z);
                maximumZ = std::max(maximumZ, positions[index].z);
                const double edgeBowlRadius =
                    std::hypot(positions[index].x + 2'048.0, positions[index].z - 7'168.0);
                output[index] = climateInput(80.0 + std::min(20.0, edgeBowlRadius * 0.2));
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
            }
        };

    NativeHydrologyRouter router(0xD1EC'7004'0000'0001ULL);
    router.prepareOwner(-2, 3, observed);
    router.prepareOwner(-2, 3, observed);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(calls == 1);
    REQUIRE(metrics.builds == 1);
    REQUIRE(metrics.entries == 1);
    REQUIRE(metrics.reconciliationEntries == 0);
    REQUIRE(metrics.openDepressionEntries == 0);
    REQUIRE(metrics.connectedWetlandEntries == 0);
    REQUIRE(metrics.seaBackwaterEntries == 0);
    REQUIRE(minimumX == -4'104);
    REQUIRE(maximumX == -2'040);
    REQUIRE(minimumZ == 6'136);
    REQUIRE(maximumZ == 8'200);
}

TEST_CASE("V4 native dry locality certificate accepts only ocean-vented owner interiors",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][signed]") {
    const auto dryIsland = [](int64_t centerX, int64_t centerZ) {
        return [centerX, centerZ](std::span<const NativeHydrologyPosition> positions,
                                  std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - centerX, positions[index].z - centerZ);
                output[index] = climateInput(radius <= 320.0 ? 24.0 - radius * 0.02 : -1.0);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
            }
        };
    };

    constexpr std::array<BasinSamplePosition, 3> positivePositions{{
        {1'024.0, 1'024.0},
        {1'032.0, 1'024.0},
        {1'024.0, 1'032.0},
    }};
    NativeHydrologyRouter positiveRouter(0xD1EC'7004'0000'0002ULL);
    std::array<BasinSample, positivePositions.size()> positive{};
    std::array<uint8_t, positivePositions.size()> positiveCertified{};
    positiveRouter.certifyDryPoints(positivePositions, dryIsland(1'024, 1'024), positive,
                                    positiveCertified);
    REQUIRE(std::ranges::all_of(positiveCertified, [](uint8_t value) { return value == 1; }));
    for (const BasinSample& sample : positive) {
        REQUIRE(sample.valid);
        REQUIRE_FALSE(sample.ocean);
        REQUIRE_FALSE(sample.lake);
        REQUIRE_FALSE(sample.river);
        REQUIRE_FALSE(sample.wetland);
        REQUIRE_FALSE(sample.waterfall);
        REQUIRE(sample.waterBodyId == NO_WATER_BODY);
        REQUIRE(sample.transitionOwnerKind == WaterTransitionKind::NONE);
    }

    constexpr std::array<BasinSamplePosition, 1> signedPositions{{{-1'024.0, -1'024.0}}};
    NativeHydrologyRouter signedRouter(0xD1EC'7004'0000'0003ULL);
    std::array<BasinSample, 1> signedSample{};
    std::array<uint8_t, 1> signedCertified{};
    signedRouter.certifyDryPoints(signedPositions, dryIsland(-1'024, -1'024), signedSample,
                                  signedCertified);
    REQUIRE(signedCertified.front() == 1);
    REQUIRE(signedSample.front().valid);
    REQUIRE_FALSE(signedSample.front().ocean);
    REQUIRE(signedRouter.cacheMetrics().builds == 1);
}

TEST_CASE("V4 native dry locality certificate rejects uncertain water and handoff footprints",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][fail-closed]") {
    const NativeHydrologyInputFunction island =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - 1'024.0, positions[index].z - 1'024.0);
                output[index] = climateInput(radius <= 320.0 ? 24.0 - radius * 0.02 : -1.0);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
            }
        };
    constexpr std::array<BasinSamplePosition, 1> handoff{{{8.0, 1'024.0}}};
    NativeHydrologyRouter handoffRouter(0xD1EC'7004'0000'0004ULL);
    std::array<BasinSample, 1> output{};
    std::array<uint8_t, 1> certified{{1}};
    handoffRouter.certifyDryPoints(handoff, island, output, certified);
    REQUIRE(certified.front() == 0);
    REQUIRE_FALSE(output.front().valid);
    REQUIRE(handoffRouter.cacheMetrics().builds == 0);

    constexpr std::array<BasinSamplePosition, 2> mixedOwners{{
        {1'024.0, 1'024.0},
        {3'072.0, 1'024.0},
    }};
    NativeHydrologyRouter mixedRouter(0xD1EC'7004'0000'0009ULL);
    std::array<BasinSample, mixedOwners.size()> mixedOutput{};
    std::array<uint8_t, mixedOwners.size()> mixedCertified{{1, 1}};
    mixedRouter.certifyDryPoints(mixedOwners, island, mixedOutput, mixedCertified);
    REQUIRE(std::ranges::all_of(mixedCertified, [](uint8_t value) { return value == 0; }));
    REQUIRE(
        std::ranges::none_of(mixedOutput, [](const BasinSample& sample) { return sample.valid; }));
    REQUIRE(mixedRouter.cacheMetrics().builds == 0);

    const NativeHydrologyInputFunction closedLake =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - 1'024.0, positions[index].z - 1'024.0);
                const double elevation =
                    radius < 500.0 ? 8.0 + std::min(20.0, radius * 0.04) : -1.0;
                output[index] = climateInput(elevation);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
            }
        };
    constexpr std::array<BasinSamplePosition, 1> center{{{1'024.0, 1'024.0}}};
    NativeHydrologyRouter lakeRouter(0xD1EC'7004'0000'0005ULL);
    certified.front() = 1;
    lakeRouter.certifyDryPoints(center, closedLake, output, certified);
    REQUIRE(certified.front() == 0);
    REQUIRE_FALSE(output.front().valid);
    REQUIRE(lakeRouter.sample(center.front().x, center.front().z, closedLake).lake);

    NativeHydrologyRouter wetlandRouter(0xD1EC'7004'0000'0006ULL);
    constexpr std::array<BasinSamplePosition, 1> wetlandPosition{{{1'026.0, 1'026.0}}};
    certified.front() = 1;
    wetlandRouter.certifyDryPoints(wetlandPosition, isolatedOceanWetlandInput(), output, certified);
    REQUIRE(certified.front() == 0);
    REQUIRE_FALSE(output.front().valid);
    REQUIRE(wetlandRouter
                .sample(wetlandPosition.front().x, wetlandPosition.front().z,
                        isolatedOceanWetlandInput())
                .wetland);
}

TEST_CASE("V4 native dry locality certificate follows both D-infinity receivers from boundaries",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][d-infinity]") {
    const NativeHydrologyInputFunction diagonalDrainage =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                output[index] =
                    climateInput(40.0 - positions[index].x * 0.018 - positions[index].z * 0.009);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
                output[index].climate.precipitationCoefficientOfVariation = 1.5;
            }
        };
    constexpr std::array<BasinSamplePosition, 1> positions{{{1'024.0, 1'024.0}}};
    NativeHydrologyRouter router(0xD1EC'7004'0000'0007ULL);
    const BasinSample canonical =
        router.sample(positions.front().x, positions.front().z, diagonalDrainage);
    CAPTURE(canonical.ocean, canonical.lake, canonical.river, canonical.wetland,
            canonical.waterfall, canonical.flowX, canonical.flowZ, canonical.discharge);
    REQUIRE(canonical.valid);
    REQUIRE_FALSE(canonical.ocean);
    REQUIRE_FALSE(canonical.lake);
    REQUIRE_FALSE(canonical.river);
    REQUIRE_FALSE(canonical.wetland);
    REQUIRE_FALSE(canonical.waterfall);
    REQUIRE(canonical.flowX > 0.7);
    REQUIRE(canonical.flowZ > 0.2);

    std::array<BasinSample, 1> certifiedSample{};
    std::array<uint8_t, 1> certified{{1}};
    router.certifyDryPoints(positions, diagonalDrainage, certifiedSample, certified);
    REQUIRE(certified.front() == 0);
    REQUIRE_FALSE(certifiedSample.front().valid);
}

TEST_CASE("V4 native dry locality certificate includes the complete channel source halo",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][channel]") {
    const NativeHydrologyInputFunction boundaryValley =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double valleyRise = std::abs(positions[index].z - 1'024) * 0.01;
                output[index] = climateInput(28.0 - positions[index].x * 0.014 + valleyRise);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'800.0;
                output[index].climate.precipitationCoefficientOfVariation = 1.5;
            }
        };
    constexpr std::array<BasinSamplePosition, 1> positions{{{1'024.0, 1'032.0}}};
    NativeHydrologyRouter router(0xD1EC'7004'0000'0008ULL);
    const BasinSample canonical =
        router.sample(positions.front().x, positions.front().z, boundaryValley);
    CAPTURE(canonical.ocean, canonical.lake, canonical.river, canonical.wetland,
            canonical.waterfall, canonical.channelDistance, canonical.channelWidth,
            canonical.discharge);
    REQUIRE(canonical.valid);
    REQUIRE_FALSE(canonical.ocean);
    REQUIRE_FALSE(canonical.lake);
    REQUIRE_FALSE(canonical.river);
    REQUIRE_FALSE(canonical.wetland);
    REQUIRE_FALSE(canonical.waterfall);
    REQUIRE(canonical.channelDistance <=
            NATIVE_HYDROLOGY_RASTER_SPACING * 8 + NATIVE_HYDROLOGY_RASTER_SPACING);

    std::array<BasinSample, 1> certifiedSample{};
    std::array<uint8_t, 1> certified{{1}};
    router.certifyDryPoints(positions, boundaryValley, certifiedSample, certified);
    REQUIRE(certified.front() == 0);
    REQUIRE_FALSE(certifiedSample.front().valid);
}

TEST_CASE("V4 certified dry footprints serve exact point and grid hits without page input",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][footprint]") {
    constexpr std::array<BasinSamplePosition, 4> positions{{
        {1'024.0, 1'024.0},
        {1'032.0, 1'024.0},
        {1'024.0, 1'032.0},
        {1'032.0, 1'032.0},
    }};
    NativeHydrologyRouter router(0xD1EC'7004'0000'0010ULL, 1);
    const std::optional<NativeHydrologyDryFootprintCertificate> certificate =
        router.certifyDryFootprint(positions, oceanVentedDryIslandInput());
    REQUIRE(certificate);
    REQUIRE(certificate->size() == positions.size());
    REQUIRE_FALSE(router.certifiedDryFootprintContains(positions));
    REQUIRE(router.replaceCertifiedDryFootprint(*certificate));
    REQUIRE(router.certifiedDryFootprintContains(positions));

    const NativeHydrologyInputFunction forbiddenInput = [](std::span<const NativeHydrologyPosition>,
                                                           std::span<NativeHydrologyInput>) {
        throw std::runtime_error("certified dry hit requested page input");
    };
    bool hit = false;
    const BasinSample exact =
        router.sample(positions.front().x, positions.front().z, forbiddenInput, &hit);
    REQUIRE(hit);
    REQUIRE(nativeWaterHash(exact) == nativeWaterHash(certificate->samples().front()));

    std::array<BasinSample, positions.size()> pointOutput{};
    std::array<uint8_t, positions.size()> pointHits{};
    router.samplePoints(positions, forbiddenInput, pointOutput, pointHits);
    REQUIRE(std::ranges::all_of(pointHits, [](uint8_t value) { return value == 1; }));
    for (size_t index = 0; index < positions.size(); ++index) {
        REQUIRE(nativeWaterHash(pointOutput[index]) ==
                nativeWaterHash(certificate->samples()[index]));
    }

    std::array<BasinSample, positions.size()> gridOutput{};
    std::array<uint8_t, positions.size()> gridHits{};
    router.sampleGrid(1'024, 1'024, 8, 8, 2, 2, forbiddenInput, gridOutput, gridHits);
    REQUIRE(std::ranges::all_of(gridHits, [](uint8_t value) { return value == 1; }));
    for (size_t index = 0; index < positions.size(); ++index) {
        REQUIRE(nativeWaterHash(gridOutput[index]) ==
                nativeWaterHash(certificate->samples()[index]));
    }

    constexpr std::array<BasinSamplePosition, 2> mixed{{
        {1'024.0, 1'024.0},
        {1'040.0, 1'024.0},
    }};
    std::array<BasinSample, mixed.size()> mixedOutput{};
    std::array<uint8_t, mixed.size()> mixedHits{{9, 9}};
    router.samplePoints(mixed, oceanVentedDryIslandInput(), mixedOutput, mixedHits);
    REQUIRE((mixedHits == std::array<uint8_t, 2>{1, 0}));
    REQUIRE(
        std::ranges::all_of(mixedOutput, [](const BasinSample& sample) { return sample.valid; }));

    hit = true;
    const BasinSample fractional =
        router.sample(1'024.25, 1'024.0, oceanVentedDryIslandInput(), &hit);
    REQUIRE_FALSE(hit);
    REQUIRE(fractional.valid);

    router.clearCertifiedDryFootprint();
    REQUIRE_FALSE(router.certifiedDryFootprintContains(positions));
    hit = true;
    REQUIRE(router.sample(1'024.0, 1'024.0, oceanVentedDryIslandInput(), &hit).valid);
    REQUIRE_FALSE(hit);
}

TEST_CASE("V4 topology cells short circuit only from complete native dry certificates",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][topology]") {
    std::vector<BasinSamplePosition> complete;
    complete.reserve(64);
    for (int z = 0; z < 8; ++z) {
        for (int x = 0; x < 8; ++x) {
            complete.push_back({.x = 1'024.0 + x * NATIVE_HYDROLOGY_RASTER_SPACING,
                                .z = 1'024.0 + z * NATIVE_HYDROLOGY_RASTER_SPACING});
        }
    }

    NativeHydrologyRouter router(0xD1EC'7004'0000'0020ULL, 1);
    const auto certificate = router.certifyDryFootprint(complete, oceanVentedDryIslandInput());
    REQUIRE(certificate);
    REQUIRE(router.replaceCertifiedDryFootprint(*certificate));
    const NativeHydrologyInputFunction forbiddenInput = [](std::span<const NativeHydrologyPosition>,
                                                           std::span<NativeHydrologyInput>) {
        throw std::runtime_error("certified topology cell requested page input");
    };
    std::array<NativeHydrologyTopologyCell, 1> topology{};
    std::array<uint8_t, 1> hits{};
    router.sampleTopologyGrid(1'024, 1'024, 1, 1, forbiddenInput, topology, hits);
    CHECK(hits.front() == 1);
    CHECK_FALSE(topology.front().waterTopologyPossible);
    CHECK_FALSE(topology.front().waterfallPossible);

    complete.pop_back();
    NativeHydrologyRouter partialRouter(0xD1EC'7004'0000'0021ULL);
    const auto partialCertificate =
        partialRouter.certifyDryFootprint(complete, oceanVentedDryIslandInput());
    REQUIRE(partialCertificate);
    REQUIRE(partialRouter.replaceCertifiedDryFootprint(*partialCertificate));
    size_t inputCalls = 0;
    const NativeHydrologyInputFunction observedInput =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            ++inputCalls;
            oceanVentedDryIslandInput()(positions, output);
        };
    topology.front() = {};
    hits.front() = 9;
    const NativeHydrologyCacheMetrics metricsBefore = partialRouter.cacheMetrics();
    partialRouter.sampleTopologyGrid(1'024, 1'024, 1, 1, observedInput, topology, hits);
    const NativeHydrologyCacheMetrics metricsAfter = partialRouter.cacheMetrics();
    CHECK(hits.front() == 0);
    // Certification already prepared the native page, so the canonical
    // fallback may reuse it without invoking the input callback again.
    CHECK(inputCalls == 0);
    CHECK(metricsAfter.hits > metricsBefore.hits);
}

TEST_CASE("V4 certified dry footprint replacement is transactional across rejection and clear",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][transaction]") {
    NativeHydrologyRouter router(0xD1EC'7004'0000'0011ULL);
    constexpr std::array<BasinSamplePosition, 1> original{{{1'024.0, 1'024.0}}};
    const auto originalCertificate =
        router.certifyDryFootprint(original, oceanVentedDryIslandInput());
    REQUIRE(originalCertificate);
    REQUIRE(router.replaceCertifiedDryFootprint(*originalCertificate));
    REQUIRE(router.certifiedDryFootprintContains(original));

    constexpr std::array<BasinSamplePosition, 2> partial{{
        {1'024.0, 1'024.0},
        {1'700.0, 1'024.0},
    }};
    REQUIRE_FALSE(router.certifyDryFootprint(partial, oceanVentedDryIslandInput()));
    REQUIRE(router.certifiedDryFootprintContains(original));

    constexpr std::array<BasinSamplePosition, 2> duplicates{{
        {1'024.0, 1'024.0},
        {1'024.0, 1'024.0},
    }};
    REQUIRE_FALSE(router.certifyDryFootprint(duplicates, oceanVentedDryIslandInput()));
    constexpr std::array<BasinSamplePosition, 1> fractional{{{1'024.5, 1'024.0}}};
    REQUIRE_FALSE(router.certifyDryFootprint(fractional, oceanVentedDryIslandInput()));
    REQUIRE(router.certifiedDryFootprintContains(original));

    NativeHydrologyRouter foreignRouter(0xD1EC'7004'0000'0012ULL);
    const auto foreignCertificate =
        foreignRouter.certifyDryFootprint(original, oceanVentedDryIslandInput());
    REQUIRE(foreignCertificate);
    REQUIRE_FALSE(router.replaceCertifiedDryFootprint(*foreignCertificate));
    REQUIRE(router.certifiedDryFootprintContains(original));

    constexpr std::array<BasinSamplePosition, 1> replacement{{{1'040.0, 1'040.0}}};
    const auto replacementCertificate =
        router.certifyDryFootprint(replacement, oceanVentedDryIslandInput());
    REQUIRE(replacementCertificate);
    REQUIRE(router.replaceCertifiedDryFootprint(*replacementCertificate));
    REQUIRE_FALSE(router.certifiedDryFootprintContains(original));
    REQUIRE(router.certifiedDryFootprintContains(replacement));

    router.clear();
    REQUIRE_FALSE(router.certifiedDryFootprintContains(replacement));

    NativeHydrologyRouter signedRouter(0xD1EC'7004'0000'0013ULL);
    constexpr std::array<BasinSamplePosition, 1> signedPosition{{{-1'024.0, -1'024.0}}};
    const auto signedCertificate =
        signedRouter.certifyDryFootprint(signedPosition, oceanVentedDryIslandInput(-1'024, -1'024));
    REQUIRE(signedCertificate);
    REQUIRE(signedRouter.replaceCertifiedDryFootprint(*signedCertificate));
    bool signedHit = false;
    REQUIRE(signedRouter
                .sample(-1'024.0, -1'024.0, oceanVentedDryIslandInput(-1'024, -1'024), &signedHit)
                .valid);
    REQUIRE(signedHit);
}

TEST_CASE("V4 native dry proofs are owner-cached and bounded to 65536 samples",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][cache][bound]") {
    size_t inputCalls = 0;
    const NativeHydrologyInputFunction input =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            ++inputCalls;
            oceanVentedDryIslandInput()(positions, output);
        };
    NativeHydrologyRouter router(0xD1EC'7004'0000'0014ULL);
    constexpr std::array<BasinSamplePosition, 1> first{{{1'024.0, 1'024.0}}};
    constexpr std::array<BasinSamplePosition, 1> second{{{1'032.0, 1'032.0}}};
    std::array<BasinSample, 1> output{};
    std::array<uint8_t, 1> mask{};
    router.certifyDryPoints(first, input, output, mask);
    REQUIRE(mask.front() == 1);
    const size_t callsAfterFirst = inputCalls;
    REQUIRE(callsAfterFirst > 0);
    REQUIRE(router.cacheMetrics().builds == 1);
    router.certifyDryPoints(second, input, output, mask);
    REQUIRE(mask.front() == 1);
    REQUIRE(inputCalls == callsAfterFirst);
    REQUIRE(router.cacheMetrics().builds == 1);

    std::vector<BasinSamplePosition> excessive(NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES + 1,
                                               first.front());
    std::vector<BasinSample> excessiveOutput(excessive.size());
    std::vector<uint8_t> excessiveMask(excessive.size());
    REQUIRE_THROWS_AS(router.certifyDryPoints(excessive, input, excessiveOutput, excessiveMask),
                      std::invalid_argument);
    REQUIRE_FALSE(router.certifyDryFootprint(excessive, input));
    REQUIRE(inputCalls == callsAfterFirst);
}

TEST_CASE("V4 native dry-proof metrics exclude structurally rejected batches",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][metrics]"
          "[regression]") {
    size_t inputCalls = 0;
    const NativeHydrologyInputFunction input =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            ++inputCalls;
            oceanVentedDryIslandInput()(positions, output);
        };
    NativeHydrologyRouter router(0xD1EC'7004'0000'0017ULL);
    std::array<BasinSample, 2> output{};
    std::array<uint8_t, 2> mask{};

    constexpr std::array<BasinSamplePosition, 2> mixedOwners{{
        {1'024.0, 1'024.0},
        {3'072.0, 1'024.0},
    }};
    router.certifyDryPoints(mixedOwners, input, output, mask);
    CHECK(std::ranges::all_of(mask, [](uint8_t value) { return value == 0; }));
    CHECK(router.cacheMetrics().dryCertificateSamples == 0);
    CHECK(inputCalls == 0);

    constexpr std::array<BasinSamplePosition, 2> handoffEdge{{
        {8.0, 1'024.0},
        {16.0, 1'024.0},
    }};
    router.certifyDryPoints(handoffEdge, input, output, mask);
    CHECK(std::ranges::all_of(mask, [](uint8_t value) { return value == 0; }));
    CHECK(router.cacheMetrics().dryCertificateSamples == 0);
    CHECK(inputCalls == 0);

    constexpr std::array<BasinSamplePosition, 1> valid{{{1'024.0, 1'024.0}}};
    std::array<BasinSample, 1> validOutput{};
    std::array<uint8_t, 1> validMask{};
    router.certifyDryPoints(valid, input, validOutput, validMask);
    CHECK(router.cacheMetrics().dryCertificateSamples == valid.size());
    CHECK(inputCalls > 0);
}

TEST_CASE("V4 macro dry footprints preserve adaptation and record only fallback owners",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][macro][readiness]") {
    const worldgen::learned::GenerationIdentity identity =
        nativeHydrologyIdentity(0xD1EC'7004'0000'0015ULL);
    const auto authority = std::make_shared<DryIslandTerrainAuthority>(identity);
    const auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    MacroGenerationSampler sampler(identity.seed, context);

    constexpr std::array<ColumnPos, 4> footprint{{
        ColumnPos{1'024, 1'024},
        ColumnPos{1'032, 1'024},
        ColumnPos{1'024, 1'032},
        ColumnPos{1'032, 1'032},
    }};
    std::array<uint8_t, footprint.size()> mask{};
    sampler.certifyNativeHydrologyDryMask(footprint, mask);
    REQUIRE(std::ranges::all_of(mask, [](uint8_t value) { return value == 1; }));
    REQUIRE(authority->transientGridQueries() == 1);
    REQUIRE(authority->pointQueries() == 0);
    REQUIRE(authority->pageQueries() == 0);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);

    std::array<HydrologySample, footprint.size()> installed{};
    REQUIRE(sampler.replaceNativeHydrologyDryFootprint(footprint, installed));
    REQUIRE(sampler.nativeHydrologyDryFootprintContains(footprint));
    REQUIRE(authority->pointQueries() == 1);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);

    const uint64_t pointQueriesBeforeHits = authority->pointQueries();
    std::array<HydrologySample, footprint.size()> pointHits{};
    sampler.sampleHydrologyPoints(footprint, pointHits);
    REQUIRE(authority->pointQueries() == pointQueriesBeforeHits + 1);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);
    for (size_t index = 0; index < footprint.size(); ++index) {
        CHECK(pointHits[index].surfaceElevation ==
              Catch::Approx(installed[index].surfaceElevation));
        CHECK(pointHits[index].waterSurface == Catch::Approx(installed[index].waterSurface));
        CHECK(pointHits[index].waterBodyId == NO_WATER_BODY);
    }

    const HydrologySample scalarHit = sampler.sampleHydrology(1'024.0, 1'024.0);
    CHECK(scalarHit.surfaceElevation == Catch::Approx(installed.front().surfaceElevation));
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);

    std::array<HydrologySample, footprint.size()> gridHits{};
    sampler.sampleHydrologyGrid(1'024, 1'024, 8, 8, 2, 2, gridHits);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);
    for (size_t index = 0; index < footprint.size(); ++index)
        CHECK(gridHits[index].surfaceElevation == Catch::Approx(installed[index].surfaceElevation));

    constexpr std::array<ColumnPos, 2> rejected{{
        ColumnPos{1'024, 1'024},
        ColumnPos{1'700, 1'024},
    }};
    std::array<HydrologySample, rejected.size()> rejectedOutput{};
    const uint64_t pointQueriesBeforeRejection = authority->pointQueries();
    REQUIRE_FALSE(sampler.replaceNativeHydrologyDryFootprint(rejected, rejectedOutput));
    REQUIRE(authority->pointQueries() == pointQueriesBeforeRejection);
    REQUIRE(sampler.nativeHydrologyDryFootprintContains(footprint));

    constexpr std::array<ColumnPos, 2> mixed{{
        ColumnPos{1'024, 1'024},
        ColumnPos{3'072, 1'024},
    }};
    std::array<HydrologySample, mixed.size()> mixedOutput{};
    sampler.sampleHydrologyPoints(mixed, mixedOutput);
    REQUIRE_FALSE(context->nativeHydrologyOwnerPrepared(0, 0));
    REQUIRE(context->nativeHydrologyOwnerPrepared(1, 0));
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 1);

    sampler.clearNativeHydrologyDryFootprint();
    REQUIRE_FALSE(sampler.nativeHydrologyDryFootprintContains(footprint));
}

TEST_CASE("V4 macro dry certificates reject integer coordinates that alias as doubles",
          "[worldgen][hydrology][native-4][v4][spawn][dry-certificate][macro][precision]"
          "[regression]") {
    const worldgen::learned::GenerationIdentity identity =
        nativeHydrologyIdentity(0xD1EC'7004'0000'0016ULL);
    const auto authority = std::make_shared<DryIslandTerrainAuthority>(identity);
    const auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    MacroGenerationSampler sampler(identity.seed, context);

    constexpr std::array<ColumnPos, 1> installedPosition{{ColumnPos{1'024, 1'024}}};
    std::array<HydrologySample, installedPosition.size()> installed{};
    REQUIRE(sampler.replaceNativeHydrologyDryFootprint(installedPosition, installed));
    REQUIRE(sampler.nativeHydrologyDryFootprintContains(installedPosition));

    constexpr int64_t MAXIMUM_EXACT_DOUBLE_INTEGER = 9'007'199'254'740'992LL;
    constexpr std::array<ColumnPos, 2> aliasedPositions{{
        ColumnPos{MAXIMUM_EXACT_DOUBLE_INTEGER + 1, 1'024},
        ColumnPos{-MAXIMUM_EXACT_DOUBLE_INTEGER - 1, 1'024},
    }};
    const uint64_t transientQueriesBefore = authority->transientGridQueries();
    const uint64_t pointQueriesBefore = authority->pointQueries();

    std::array<uint8_t, aliasedPositions.size()> mask{};
    mask.fill(1);
    sampler.certifyNativeHydrologyDryMask(aliasedPositions, mask);
    CHECK(std::ranges::all_of(mask, [](uint8_t value) { return value == 0; }));

    std::array<HydrologySample, aliasedPositions.size()> pointOutput{};
    std::array<uint8_t, aliasedPositions.size()> pointCertified{};
    pointCertified.fill(1);
    sampler.certifyNativeHydrologyDryPoints(aliasedPositions, pointOutput, pointCertified);
    CHECK(std::ranges::all_of(pointCertified, [](uint8_t value) { return value == 0; }));

    std::array<HydrologySample, aliasedPositions.size()> replacement{};
    REQUIRE_FALSE(sampler.replaceNativeHydrologyDryFootprint(aliasedPositions, replacement));
    REQUIRE_FALSE(sampler.nativeHydrologyDryFootprintContains(aliasedPositions));
    REQUIRE(sampler.nativeHydrologyDryFootprintContains(installedPosition));
    CHECK(authority->transientGridQueries() == transientQueriesBefore);
    CHECK(authority->pointQueries() == pointQueriesBefore);
}

TEST_CASE("V4 native horizon pages use bounded parallel CPU builds",
          "[worldgen][hydrology][native-4][v4][performance][concurrency]") {
    constexpr size_t REQUEST_COUNT = 4;
    const size_t reported = std::thread::hardware_concurrency() == 0
                                ? 1U
                                : static_cast<size_t>(std::thread::hardware_concurrency());
    const size_t admissionLimit =
        std::max<size_t>(1, std::min({reported, NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS,
                                      NATIVE_HYDROLOGY_PARALLEL_BUILD_MEMORY_BUDGET /
                                          NATIVE_HYDROLOGY_MAX_BUILD_BYTES}));
    const size_t expectedActive = std::min(REQUEST_COUNT, admissionLimit);
    std::barrier requestStart(static_cast<std::ptrdiff_t>(REQUEST_COUNT + 1));
    std::mutex inputMutex;
    std::condition_variable inputReady;
    size_t enteredInput = 0;
    bool releaseInput = false;
    const NativeHydrologyInputFunction heldInput =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            {
                std::unique_lock lock(inputMutex);
                ++enteredInput;
                inputReady.notify_all();
                inputReady.wait(lock, [&] { return releaseInput; });
            }
            for (size_t index = 0; index < positions.size(); ++index) {
                output[index] =
                    climateInput(150.0 - positions[index].x * 0.002 - positions[index].z * 0.001);
            }
        };

    NativeHydrologyRouter router(0x16C0'BE50'0004ULL);
    constexpr std::array<std::array<int64_t, 2>, REQUEST_COUNT> ownerPages{{
        {0, 0},
        {2, 0},
        {0, 2},
        {2, 2},
    }};
    std::array<std::future<void>, REQUEST_COUNT> requests;
    for (size_t index = 0; index < ownerPages.size(); ++index) {
        requests[index] = std::async(
            std::launch::async, [&router, &heldInput, &requestStart, owner = ownerPages[index]] {
                requestStart.arrive_and_wait();
                router.prepareOwner(owner[0], owner[1], heldInput);
            });
    }
    requestStart.arrive_and_wait();

    bool reachedExpectedConcurrency = false;
    {
        std::unique_lock lock(inputMutex);
        reachedExpectedConcurrency = inputReady.wait_for(
            lock, std::chrono::seconds(10), [&] { return enteredInput >= expectedActive; });
        releaseInput = true;
    }
    inputReady.notify_all();
    for (std::future<void>& request : requests)
        request.get();

    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(reachedExpectedConcurrency);
    REQUIRE(metrics.activeBuilds == 0);
    REQUIRE(metrics.peakConcurrentBuilds >= expectedActive);
    REQUIRE(metrics.peakConcurrentBuilds <= admissionLimit);
}

TEST_CASE("V4 distant hydrology cannot occupy every build lane ahead of the exact band",
          "[worldgen][hydrology][native-4][v4][performance][concurrency][priority]"
          "[regression]") {
    const size_t reported = std::thread::hardware_concurrency() == 0
                                ? 1U
                                : static_cast<size_t>(std::thread::hardware_concurrency());
    const size_t expectedActive =
        std::max<size_t>(1, std::min({reported, NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS,
                                      NATIVE_HYDROLOGY_PARALLEL_BUILD_MEMORY_BUDGET /
                                          NATIVE_HYDROLOGY_MAX_BUILD_BYTES}));
    constexpr size_t BLOCKER_COUNT = NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS;
    constexpr int64_t EXACT_OWNER = 101;

    std::mutex inputMutex;
    std::condition_variable inputReady;
    std::vector<int64_t> enteredOwners;
    std::set<int64_t> releasedOwners;
    bool releaseAll = false;
    const NativeHydrologyInputFunction heldInput =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            if (positions.empty())
                throw std::invalid_argument("priority admission input is empty");
            const int64_t owner = positions[positions.size() / 2].x /
                                  static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE);
            {
                std::unique_lock lock(inputMutex);
                enteredOwners.push_back(owner);
                inputReady.notify_all();
                inputReady.wait(lock, [&] { return releaseAll || releasedOwners.contains(owner); });
            }
            for (size_t index = 0; index < positions.size(); ++index) {
                output[index] =
                    climateInput(150.0 - positions[index].x * 0.0002 - positions[index].z * 0.0001);
            }
        };

    NativeHydrologyRouter router(0x16C0'BE50'0005ULL);
    // The low reservation caps distant SPECULATIVE builds at half the lanes so
    // the exact band always retains reserved build capacity (issue #17).
    const size_t lowCap = std::max<size_t>(1, expectedActive / 2);
    std::vector<std::future<BasinSample>> blockers;
    blockers.reserve(BLOCKER_COUNT);
    for (size_t index = 0; index < BLOCKER_COUNT; ++index) {
        const double x =
            static_cast<double>((static_cast<int64_t>(index) + 1) * NATIVE_HYDROLOGY_PAGE_EDGE +
                                NATIVE_HYDROLOGY_PAGE_EDGE / 2);
        blockers.push_back(std::async(std::launch::async, [&router, &heldInput, x] {
            return router.sample(x, 1'024.0, heldInput, nullptr,
                                 learned::AuthorityRequestPriority::SPECULATIVE_PREFETCH);
        }));
    }
    struct ReleaseAllOnExit {
        std::mutex& mutex;
        std::condition_variable& ready;
        bool& releaseAll;
        ~ReleaseAllOnExit() {
            {
                std::lock_guard lock(mutex);
                releaseAll = true;
            }
            ready.notify_all();
        }
    } releaseAllOnExit{inputMutex, inputReady, releaseAll};

    // At most lowCap distant builds run concurrently; the reservation keeps the
    // remaining SPECULATIVE requests waiting so exact lanes stay free.
    {
        std::unique_lock lock(inputMutex);
        REQUIRE(inputReady.wait_for(lock, std::chrono::seconds(5),
                                    [&] { return enteredOwners.size() >= lowCap; }));
    }

    if (lowCap < expectedActive) {
        // A reserved lane exists: distant work never exceeds its cap, and an
        // exact request enters immediately without releasing any distant build.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard lock(inputMutex);
            CHECK(enteredOwners.size() == lowCap);
        }
        std::future<BasinSample> exact = std::async(std::launch::async, [&] {
            return router.sample(
                static_cast<double>(EXACT_OWNER * NATIVE_HYDROLOGY_PAGE_EDGE + 1'024), 1'024.0,
                heldInput, nullptr, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
        });
        {
            std::unique_lock lock(inputMutex);
            REQUIRE(inputReady.wait_for(lock, std::chrono::seconds(10), [&] {
                return std::ranges::find(enteredOwners, EXACT_OWNER) != enteredOwners.end();
            }));
        }
        {
            std::lock_guard lock(inputMutex);
            releaseAll = true;
        }
        inputReady.notify_all();
        REQUIRE(exact.get().valid);
    } else {
        std::lock_guard lock(inputMutex);
        releaseAll = true;
        inputReady.notify_all();
    }

    for (std::future<BasinSample>& blocker : blockers)
        REQUIRE(blocker.get().valid);
    CHECK(router.cacheMetrics().buildAdmissionWaits >= 1);
}

TEST_CASE("V4 distant hydrology pages cannot evict exact cached owners",
          "[worldgen][hydrology][native-4][v4][performance][grid-cache][priority]"
          "[regression]") {
    const NativeHydrologyInputFunction flatInput =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index)
                output[index] = climateInput(100.0);
        };
    constexpr uint64_t SEED = 0x16C0'BE50'0006ULL;
    size_t pageBytes = 0;
    {
        NativeHydrologyRouter probe(SEED, NATIVE_HYDROLOGY_MAX_PAGE_BYTES);
        probe.prepareOwner(1, 0, flatInput, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
        const NativeHydrologyCacheMetrics metrics = probe.cacheMetrics();
        REQUIRE(metrics.entries == 1);
        REQUIRE(metrics.bytes > 0);
        pageBytes = metrics.bytes;
    }

    REQUIRE(pageBytes <= std::numeric_limits<size_t>::max() / 2);
    NativeHydrologyRouter router(SEED, pageBytes * 2);
    router.prepareOwner(1, 0, flatInput, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
    router.prepareOwner(2, 0, flatInput,
                        learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(router.cacheMetrics().entries == 2);
    router.prepareOwner(3, 0, flatInput,
                        learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    const NativeHydrologyCacheMetrics displaced = router.cacheMetrics();
    REQUIRE(displaced.entries == 2);
    REQUIRE(displaced.builds == 3);

    router.prepareOwner(1, 0, flatInput, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
    const NativeHydrologyCacheMetrics exactHit = router.cacheMetrics();
    CHECK(exactHit.builds == 3);
    CHECK(exactHit.hits >= 1);

    router.prepareOwner(2, 0, flatInput,
                        learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    CHECK(router.cacheMetrics().builds == 4);
}

TEST_CASE("V4 native owners enumerate signed learned authority apron pages",
          "[worldgen][hydrology][native-4][v4][authority][signed]") {
    const auto origin = nativeHydrologyRequiredAuthorityPages(0, 0);
    REQUIRE(origin.size() == 16);
    REQUIRE(origin.front() == learned::TerrainPageCoordinate{-1, -1});
    REQUIRE(origin.back() == learned::TerrainPageCoordinate{2, 2});
    REQUIRE(std::ranges::is_sorted(origin));

    const auto signedPages = nativeHydrologyRequiredAuthorityPages(-1, 1);
    REQUIRE(signedPages.size() == 16);
    REQUIRE(signedPages.front() == learned::TerrainPageCoordinate{1, -3});
    REQUIRE(signedPages.back() == learned::TerrainPageCoordinate{4, 0});
    REQUIRE(std::ranges::is_sorted(signedPages));
}

TEST_CASE("V4 native owners expose their exact overflow-checked final terrain rectangle",
          "[worldgen][hydrology][native-4][v4][authority][transient][signed]") {
    const learned::NativeRect origin = nativeHydrologyFinalTerrainRegion(0, 0);
    REQUIRE(origin == learned::NativeRect{-2, -2, 515, 515});
    REQUIRE(origin.height() == 517);
    REQUIRE(origin.width() == 517);

    const learned::NativeRect signedOwner = nativeHydrologyFinalTerrainRegion(-1, 1);
    REQUIRE(signedOwner == learned::NativeRect{510, -514, 1'027, 3});
    REQUIRE(signedOwner.height() == 517);
    REQUIRE(signedOwner.width() == 517);

    REQUIRE_THROWS_AS(nativeHydrologyFinalTerrainRegion(std::numeric_limits<int64_t>::max(), 0),
                      std::out_of_range);
    REQUIRE_THROWS_AS(nativeHydrologyFinalTerrainRegion(0, std::numeric_limits<int64_t>::min()),
                      std::out_of_range);
}

TEST_CASE("V4 exact spawn bands bound final topology and refinement pages",
          "[worldgen][hydrology][native-4][v4][authority][spawn]") {
    // A three-chunk band centered exactly on both 2,048-block owner
    // boundaries is the maximum dependency case. The eight-block plan apron
    // expands it to [2024, 2088) on both axes.
    const NativeHydrologyAuthorityRequirements boundary =
        nativeHydrologyAuthorityRequirementsForWorldRect(2'024, 2'024, 2'088, 2'088);
    REQUIRE(boundary.finalTopologyPages.size() == 36);
    REQUIRE(boundary.finalRefinementPages.size() == 4);
    REQUIRE(boundary.totalPageCount() == 36);
    REQUIRE(boundary.totalPageCount() <= learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS);
    REQUIRE(std::ranges::is_sorted(boundary.finalTopologyPages));
    REQUIRE(std::ranges::is_sorted(boundary.finalRefinementPages));

    // The fresh default sits inside both FINAL authority closures. Its full
    // exact band needs one refinement page while page-wide topology needs 16.
    const NativeHydrologyAuthorityRequirements interior =
        nativeHydrologyAuthorityRequirementsForWorldRect(488, 488, 552, 552);
    REQUIRE(interior.finalTopologyPages.size() == 16);
    REQUIRE(interior.finalRefinementPages.size() == 1);
    REQUIRE(interior.totalPageCount() == 16);

    const NativeHydrologyAuthorityRequirements signedBoundary =
        nativeHydrologyAuthorityRequirementsForWorldRect(-24, -24, 40, 40);
    REQUIRE(signedBoundary.finalTopologyPages.size() == 36);
    REQUIRE(signedBoundary.finalRefinementPages.size() == 4);
    REQUIRE(signedBoundary.totalPageCount() == 36);
}

TEST_CASE("V4 native D-infinity routing follows learned elevation without raising terrain",
          "[worldgen][hydrology][native-4][v4][d-infinity][no-raising]") {
    NativeHydrologyRouter router(42);
    const BasinSample routed = router.sample(1'024.0, 1'024.0, planarNativeInput());
    const double learnedElevation = 130.0 - 1'024.0 * 0.004 - 1'024.0 * 0.002;

    REQUIRE(routed.valid);
    CAPTURE(routed.discharge, NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE);
    REQUIRE(routed.flowX > 0.2);
    REQUIRE(routed.flowZ > 0.1);
    REQUIRE(std::abs(std::hypot(routed.flowX, routed.flowZ) - 1.0) < 1.0e-6);
    REQUIRE(routed.river);
    REQUIRE(routed.discharge >= NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE);
    REQUIRE(routed.waterBodyId != NO_WATER_BODY);
    REQUIRE(routed.surfaceElevation <= learnedElevation + 1.0e-6);
    REQUIRE(routed.erosionDepth >= 0.0);
    REQUIRE_FALSE(routed.lakeBank);
    REQUIRE_FALSE(routed.channelBank);
}

TEST_CASE("V4 native D-infinity retains sub-block learned meter slopes",
          "[worldgen][hydrology][native-4][v4][d-infinity][meters][regression]") {
    // Each neighboring four-block native sample descends by only 0.04 or
    // 0.02 meters. All five values therefore emit at the same Rycraft block
    // height, but the native solver must still route continuously toward the
    // southeast instead of falling back to a cardinal raster direction.
    const NativeHydrologyInputFunction shallowInclinedPlane =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const NativeHydrologyPosition position = positions[index];
                output[index] = climateInput(100.25 - static_cast<double>(position.x) * 0.01 -
                                             static_cast<double>(position.z) * 0.005);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'200.0;
            }
        };

    constexpr double SAMPLE_X = 1'024.0;
    constexpr double SAMPLE_Z = 1'024.0;
    const auto elevationAt = [](double x, double z) { return 100.25 - x * 0.01 - z * 0.005; };
    const double centerElevation = elevationAt(SAMPLE_X, SAMPLE_Z);
    const double centerHeight = learned::learnedElevationMetersToWorldHeight(centerElevation);
    REQUIRE(learned::learnedElevationMetersToWorldHeight(
                elevationAt(SAMPLE_X + NATIVE_HYDROLOGY_RASTER_SPACING, SAMPLE_Z)) == centerHeight);
    REQUIRE(learned::learnedElevationMetersToWorldHeight(
                elevationAt(SAMPLE_X, SAMPLE_Z + NATIVE_HYDROLOGY_RASTER_SPACING)) == centerHeight);

    NativeHydrologyRouter router(0x5A0F'1E00'0004ULL);
    const BasinSample routed = router.sample(SAMPLE_X, SAMPLE_Z, shallowInclinedPlane);
    REQUIRE(routed.valid);
    REQUIRE_FALSE(routed.ocean);
    REQUIRE_FALSE(routed.lake);
    REQUIRE_FALSE(routed.river);
    REQUIRE(routed.flowX > 0.80);
    REQUIRE(routed.flowX < 0.90);
    REQUIRE(routed.flowZ > 0.45);
    REQUIRE(routed.flowZ < 0.58);
    REQUIRE(std::abs(std::hypot(routed.flowX, routed.flowZ) - 1.0) < 1.0e-6);
}

TEST_CASE("V4 native waterfalls retain canonical identities through point and lattice sampling",
          "[worldgen][hydrology][native-4][v4][waterfall][identity][regression]") {
    // A broad, downhill valley crosses one 50-meter escarpment. The router
    // must classify the drop before any exact or far consumer sees it, then
    // derive the same owner from the routed river body and source cell for
    // both sampling paths.
    const NativeHydrologyInputFunction escarpment =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const NativeHydrologyPosition position = positions[index];
                const double valley = std::abs(static_cast<double>(position.z) - 1'024.0) * 0.001;
                const double elevation =
                    position.x < 1'024
                        ? 120.0 + (1'024.0 - static_cast<double>(position.x)) * 0.004 + valley
                        : 70.0 - (static_cast<double>(position.x) - 1'024.0) * 0.001 + valley;
                output[index] = climateInput(elevation);
                output[index].climate.annualPrecipitationMm = 4'000.0;
                output[index].climate.potentialEvapotranspirationMm = 0.0;
            }
        };

    constexpr int64_t ORIGIN_X = 960;
    constexpr int64_t ORIGIN_Z = 960;
    constexpr int EDGE = 129;
    NativeHydrologyRouter router(0x4641'4C4C'1D00'0004ULL);
    std::array<BasinSample, EDGE * EDGE> lattice{};
    router.sampleGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, escarpment, lattice);

    std::array<NativeHydrologyTopologyCell, 5 * 5> topology{};
    router.sampleTopologyGrid(ORIGIN_X, ORIGIN_Z, 5, 5, escarpment, topology);
    const auto latticeAt = [&](int sampleX, int sampleZ) -> const BasinSample& {
        return lattice[static_cast<size_t>(sampleZ * EDGE + sampleX)];
    };
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };
    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            const BasinSample& sample = latticeAt(sampleX, sampleZ);
            if (sample.waterfall) {
                const size_t topologyX = static_cast<size_t>(sampleX / 32);
                const size_t topologyZ = static_cast<size_t>(sampleZ / 32);
                CAPTURE(sampleX, sampleZ, topologyX, topologyZ, sample.waterfallAnchor,
                        sample.transitionOwnerKind, sample.transitionOwnerId);
                REQUIRE(topology[topologyZ * 5 + topologyX].waterfallPossible);
                REQUIRE(sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL);
                REQUIRE(sample.transitionOwnerId != 0);
            }
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (sampleX + offsetX >= EDGE || sampleZ + offsetZ >= EDGE)
                    continue;
                const BasinSample& adjacent = latticeAt(sampleX + offsetX, sampleZ + offsetZ);
                if (!sample.river || !adjacent.river ||
                    sample.waterBodyId != adjacent.waterBodyId || sample.waterfall ||
                    adjacent.waterfall) {
                    continue;
                }
                CAPTURE(sampleX, sampleZ, offsetX, offsetZ, sample.waterSurface,
                        adjacent.waterSurface, sample.waterBodyId);
                REQUIRE(std::abs(visibleStage(sample) - visibleStage(adjacent)) <= 0.125001);
            }
        }
    }

    const auto found = std::ranges::find_if(lattice, [](const BasinSample& sample) {
        return sample.river && sample.waterfall && sample.waterfallAnchor;
    });
    REQUIRE(found != lattice.end());
    const size_t index = static_cast<size_t>(std::distance(lattice.begin(), found));
    const int64_t x = ORIGIN_X + static_cast<int64_t>(index % EDGE);
    const int64_t z = ORIGIN_Z + static_cast<int64_t>(index / EDGE);
    const BasinSample& grid = *found;
    const BasinSample point =
        router.sample(static_cast<double>(x), static_cast<double>(z), escarpment);

    CAPTURE(x, z, grid.waterBodyId, grid.transitionOwnerId, grid.waterfallTop,
            grid.waterfallBottom);
    REQUIRE(grid.waterBodyId != NO_WATER_BODY);
    REQUIRE(grid.generatedFluidLevel == 7);
    REQUIRE(grid.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL);
    REQUIRE(grid.transitionOwnerId != 0);
    REQUIRE(grid.waterfallAnchor);
    REQUIRE(grid.waterfallTop >= grid.waterfallBottom + 0.5);
    // The compact lip column carries the receiving reach as its base stage.
    // It must not synthesize a deeper endpoint from the physical gradient.
    REQUIRE(grid.waterSurface == Catch::Approx(grid.waterfallBottom).margin(1.0e-6));
    REQUIRE(nativeWaterHash(point) == nativeWaterHash(grid));
    REQUIRE(point.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL);
    REQUIRE(point.transitionOwnerId == grid.transitionOwnerId);
    REQUIRE(point.waterBodyId == grid.waterBodyId);

    router.clear();
    const BasinSample rebuilt =
        router.sample(static_cast<double>(x), static_cast<double>(z), escarpment);
    REQUIRE(nativeWaterHash(rebuilt) == nativeWaterHash(grid));
    REQUIRE(rebuilt.transitionOwnerId == grid.transitionOwnerId);
}

TEST_CASE("V4 native runoff and lake storage use the 7.5 meter physical scale",
          "[worldgen][hydrology][native-4][v4][units]") {
    REQUIRE(NATIVE_HYDROLOGY_CELL_EDGE_METERS == Catch::Approx(30.0));
    REQUIRE(NATIVE_HYDROLOGY_CELL_AREA_SQUARE_KILOMETERS == Catch::Approx(0.0009));
    REQUIRE(NATIVE_HYDROLOGY_PAGE_EDGE_KILOMETERS == Catch::Approx(15.36));
    REQUIRE(NATIVE_HYDROLOGY_PAGE_AREA_SQUARE_KILOMETERS == Catch::Approx(235.9296));

    const NativeHydrologyInputFunction singleCellBasin =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const bool center = positions[index] == NativeHydrologyPosition{1'024, 1'024};
                output[index] = climateInput(center ? 90.0 : 100.0);
            }
        };
    NativeHydrologyRouter router(0x7515'CA1EULL);
    const BasinSample lake = router.sample(1'025.0, 1'025.0, singleCellBasin);
    REQUIRE(lake.lake);
    REQUIRE(lake.lakeAreaSquareKilometers == Catch::Approx(0.0009));
    REQUIRE(lake.lakeVolumeCubicMeters == Catch::Approx(9'000.0));
}

TEST_CASE("V4 native lake spill identities survive cache clears and request reversal",
          "[worldgen][hydrology][native-4][v4][lake][determinism]") {
    const NativeHydrologyInputFunction bowl = [](std::span<const NativeHydrologyPosition> positions,
                                                 std::span<NativeHydrologyInput> output) {
        for (size_t index = 0; index < positions.size(); ++index) {
            const double radius =
                std::hypot(positions[index].x - 1'024.0, positions[index].z - 1'024.0);
            output[index] = climateInput(90.0 + std::min(20.0, radius * 0.02));
        }
    };
    constexpr std::array<BasinSamplePosition, 3> positions{{
        {1'024.0, 1'024.0},
        {1'028.0, 1'024.0},
        {1'024.0, 1'028.0},
    }};
    NativeHydrologyRouter router(77);
    std::array<BasinSample, positions.size()> forward{};
    router.samplePoints(positions, bowl, forward);
    REQUIRE(std::ranges::all_of(forward, [](const BasinSample& sample) { return sample.lake; }));
    REQUIRE(forward[0].waterBodyId != NO_WATER_BODY);
    REQUIRE(forward[1].waterBodyId == forward[0].waterBodyId);
    REQUIRE(forward[2].waterBodyId == forward[0].waterBodyId);
    REQUIRE(forward[0].surfaceElevation <= 90.0);
    REQUIRE(forward[0].waterSurface > forward[0].surfaceElevation);

    router.clear();
    std::array<BasinSamplePosition, positions.size()> reversed = positions;
    std::ranges::reverse(reversed);
    std::array<BasinSample, positions.size()> rebuilt{};
    router.samplePoints(reversed, bowl, rebuilt);
    REQUIRE(rebuilt[2].waterBodyId == forward[0].waterBodyId);
    REQUIRE(rebuilt[2].waterSurface == forward[0].waterSurface);
    REQUIRE(rebuilt[2].surfaceElevation == forward[0].surfaceElevation);
}

TEST_CASE("V4 native step-32 topology preserves disconnected local lake identities",
          "[worldgen][hydrology][native-4][v4][lake][topology][step-32][regression]") {
    // Two four-block pits share an owner page but not a spill route. A former
    // page-wide summary overlay used the low west-edge port to apply one lake
    // identity to every depression below the unrelated edge stage. Local
    // Priority-Flood ownership must keep these lakes distinct, and its compact
    // step-32 reduction must still request canonical coverage for each one.
    const NativeHydrologyInputFunction disconnectedDepressions =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const NativeHydrologyPosition position = positions[index];
                const bool firstPit = position == NativeHydrologyPosition{512, 512};
                const bool secondPit = position == NativeHydrologyPosition{1'536, 1'536};
                // Keep the low port away from page corners. It is deliberately
                // unrelated to both pits, so it must not merge their bodies.
                const bool westPort = position.x == 0 && position.z == 1'024;
                output[index] = climateInput((firstPit || secondPit || westPort) ? 90.0 : 100.0);
            }
        };
    constexpr std::array<BasinSamplePosition, 2> positions{{
        {513.0, 513.0},
        {1'537.0, 1'537.0},
    }};

    NativeHydrologyRouter forwardRouter(0xD15C'0A11'CEED'0001ULL);
    std::array<BasinSample, positions.size()> forward{};
    forwardRouter.samplePoints(positions, disconnectedDepressions, forward);
    REQUIRE(forward[0].lake);
    REQUIRE(forward[1].lake);
    REQUIRE(forward[0].waterSurface > forward[0].surfaceElevation);
    REQUIRE(forward[1].waterSurface > forward[1].surfaceElevation);
    REQUIRE(forward[0].waterBodyId != NO_WATER_BODY);
    REQUIRE(forward[1].waterBodyId != NO_WATER_BODY);
    REQUIRE(forward[0].waterBodyId != forward[1].waterBodyId);

    std::array<NativeHydrologyTopologyCell, 64 * 64> topology{};
    forwardRouter.sampleTopologyGrid(0, 0, 64, 64, disconnectedDepressions, topology);
    constexpr int FIRST_CELL = 512 / NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
    constexpr int SECOND_CELL = 1'536 / NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
    REQUIRE(topology[static_cast<size_t>(FIRST_CELL * 64 + FIRST_CELL)].waterTopologyPossible);
    REQUIRE(topology[static_cast<size_t>(SECOND_CELL * 64 + SECOND_CELL)].waterTopologyPossible);

    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(0xD15C'0A11'CEED'0001ULL);
    std::array<BasinSample, positions.size()> reversed{};
    reverseRouter.samplePoints(reversedPositions, disconnectedDepressions, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(reversed[index].waterBodyId == forward[index].waterBodyId);
        REQUIRE(reversed[index].waterSurface == forward[index].waterSurface);
        REQUIRE(reversed[index].surfaceElevation == forward[index].surfaceElevation);
    }
}

TEST_CASE("V4 native lakes in one coarse anchor retain separate component identities",
          "[worldgen][hydrology][native-4][v4][lake][identity][coarse-anchor][regression]") {
    // Both pits deliberately fall in the [1024, 1280) coarse identity anchor.
    // They have no shared spill route, so they must not share a water body or
    // aggregate their local lake statistics merely because their anchors do.
    const NativeHydrologyInputFunction sameAnchorDepressions =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const NativeHydrologyPosition position = positions[index];
                const bool firstPit = position == NativeHydrologyPosition{1'056, 1'056};
                const bool secondPit = position == NativeHydrologyPosition{1'184, 1'184};
                output[index] = climateInput((firstPit || secondPit) ? 90.0 : 100.0);
            }
        };
    constexpr std::array<BasinSamplePosition, 2> positions{{
        {1'057.0, 1'057.0},
        {1'185.0, 1'185.0},
    }};

    NativeHydrologyRouter router(0xD15C'0A11'CEED'0002ULL);
    std::array<BasinSample, positions.size()> forward{};
    router.samplePoints(positions, sameAnchorDepressions, forward);
    for (const BasinSample& sample : forward) {
        REQUIRE(sample.lake);
        REQUIRE(sample.waterBodyId != NO_WATER_BODY);
        REQUIRE(sample.lakeAreaSquareKilometers == Catch::Approx(0.0009));
    }
    REQUIRE(forward[0].waterBodyId != forward[1].waterBodyId);

    router.clear();
    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    std::array<BasinSample, positions.size()> reversed{};
    router.samplePoints(reversedPositions, sameAnchorDepressions, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(reversed[index].waterBodyId == forward[index].waterBodyId);
        REQUIRE(reversed[index].waterSurface == forward[index].waterSurface);
        REQUIRE(reversed[index].lakeAreaSquareKilometers ==
                forward[index].lakeAreaSquareKilometers);
    }
}

TEST_CASE("V4 native dry ridges preserve separate lakes across a page handoff",
          "[worldgen][hydrology][native-4][v4][lake][page-edge][handoff][regression]") {
    // The two pits sit inside the handoff apron on opposite sides of the
    // 2048-block owner edge. A high dry ridge between them is not an outlet
    // and must remain dry rather than acquiring either neighboring lake's ID
    // or stage during handoff reconciliation.
    const NativeHydrologyInputFunction dryRidge =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const NativeHydrologyPosition position = positions[index];
                const bool firstPit = position == NativeHydrologyPosition{2'040, 1'024};
                const bool secondPit = position == NativeHydrologyPosition{2'056, 1'024};
                const bool ridge = position.x == 2'048 && std::abs(position.z - 1'024) <= 8;
                output[index] = climateInput(ridge                     ? 140.0
                                             : (firstPit || secondPit) ? 90.0
                                                                       : 100.0);
                output[index].climate.annualPrecipitationMm = 0.0;
            }
        };
    constexpr std::array<BasinSamplePosition, 3> positions{{
        {2'041.0, 1'025.0},
        {2'049.0, 1'025.0},
        {2'057.0, 1'025.0},
    }};

    NativeHydrologyRouter router(0xD15C'0A11'CEED'0003ULL);
    std::array<BasinSample, positions.size()> forward{};
    router.samplePoints(positions, dryRidge, forward);
    REQUIRE(forward[0].lake);
    REQUIRE_FALSE(forward[1].lake);
    REQUIRE(forward[2].lake);
    REQUIRE(forward[0].waterBodyId != NO_WATER_BODY);
    REQUIRE(forward[2].waterBodyId != NO_WATER_BODY);
    REQUIRE(forward[0].waterBodyId != forward[2].waterBodyId);
    REQUIRE(forward[1].waterBodyId == NO_WATER_BODY);
    REQUIRE(forward[1].waterSurface == 0.0);

    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(0xD15C'0A11'CEED'0003ULL);
    std::array<BasinSample, positions.size()> reversed{};
    reverseRouter.samplePoints(reversedPositions, dryRidge, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(forward[index]));
    }
}

TEST_CASE("V4 tiled spill summaries reconcile long signed lake chains",
          "[worldgen][hydrology][native-4][v4][lake][hierarchy][spill-summary][long-chain]") {
    const auto buildChain = [](int64_t firstPageX, size_t pageCount, bool withOutlet) {
        std::vector<NativeHydrologySpillNodeSummary> nodes;
        std::vector<NativeHydrologySpillPortalSummary> portals;
        nodes.reserve(pageCount);
        portals.reserve(pageCount - 1);
        for (size_t index = 0; index < pageCount; ++index) {
            const int64_t pageX = firstPageX + static_cast<int64_t>(index);
            nodes.push_back({
                .pageX = pageX,
                .pageZ = -3,
                .localBodyId = 10'000 + index,
                .localAnchorX = pageX * NATIVE_HYDROLOGY_PAGE_EDGE + 64,
                .localAnchorZ = -3 * NATIVE_HYDROLOGY_PAGE_EDGE + 64,
                .localStage = index + 1 == pageCount ? 78.0 : 82.0,
                .coreAreaSquareKilometers = 1.0 + static_cast<double>(index),
                .coreVolumeCubicMeters = 10'000.0 + static_cast<double>(index) * 100.0,
                .coreRunoffMmSquareKilometers = 20.0 + static_cast<double>(index),
                .naturalOutlet = withOutlet && index + 1 == pageCount ? BasinOutlet::SHARED_PORTAL
                                                                      : BasinOutlet::NONE,
                .naturalOutletX = (pageX + 1) * NATIVE_HYDROLOGY_PAGE_EDGE - 4,
                .naturalOutletZ = -3 * NATIVE_HYDROLOGY_PAGE_EDGE + 512,
                .naturalOutletStage = withOutlet && index + 1 == pageCount ? 78.0 : 0.0,
            });
            if (index == 0)
                continue;
            portals.push_back({
                .firstPageX = pageX - 1,
                .firstPageZ = -3,
                .firstLocalBodyId = 10'000 + index - 1,
                .secondPageX = pageX,
                .secondPageZ = -3,
                .secondLocalBodyId = 10'000 + index,
                .minimumWetStage = 70.0,
                .compatibleStage = 78.0,
                .x = pageX * NATIVE_HYDROLOGY_PAGE_EDGE,
                .z = -3 * NATIVE_HYDROLOGY_PAGE_EDGE + 512,
            });
        }
        return std::pair{nodes, portals};
    };

    for (const size_t pageCount : {size_t{3}, size_t{8}}) {
        auto [nodes, portals] = buildChain(-11, pageCount, true);
        const double expectedArea =
            std::accumulate(nodes.begin(), nodes.end(), 0.0, [](double total, const auto& node) {
                return total + node.coreAreaSquareKilometers;
            });
        const double expectedVolume =
            std::accumulate(nodes.begin(), nodes.end(), 0.0, [](double total, const auto& node) {
                return total + node.coreVolumeCubicMeters;
            });
        const double expectedRunoff =
            std::accumulate(nodes.begin(), nodes.end(), 0.0, [](double total, const auto& node) {
                return total + node.coreRunoffMmSquareKilometers;
            });
        const auto expected = resolveNativeHydrologySpillSummaries(
            nodes.front().pageX, nodes.front().pageZ, nodes.front().localBodyId, nodes, portals);
        REQUIRE(expected);
        REQUIRE(expected->pageCount == pageCount);
        REQUIRE(expected->canonicalBodyId == nodes.front().localBodyId);
        REQUIRE(expected->stage == 78.0);
        REQUIRE(expected->areaSquareKilometers == expectedArea);
        REQUIRE(expected->volumeCubicMeters == expectedVolume);
        REQUIRE(expected->runoffMmSquareKilometers == expectedRunoff);
        REQUIRE(expected->outlet == BasinOutlet::SHARED_PORTAL);
        REQUIRE(expected->outletX == nodes.back().naturalOutletX);
        REQUIRE(expected->outletZ == nodes.back().naturalOutletZ);

        for (const NativeHydrologySpillNodeSummary& source : nodes) {
            const auto fromPage = resolveNativeHydrologySpillSummaries(
                source.pageX, source.pageZ, source.localBodyId, nodes, portals);
            REQUIRE(fromPage == expected);
        }

        std::ranges::reverse(nodes);
        std::ranges::reverse(portals);
        const auto reversed = resolveNativeHydrologySpillSummaries(
            nodes.back().pageX, nodes.back().pageZ, nodes.back().localBodyId, nodes, portals);
        REQUIRE(reversed == expected);

        std::array<std::future<std::optional<NativeHydrologySpillResolution>>, 4> concurrent;
        for (size_t index = 0; index < concurrent.size(); ++index) {
            concurrent[index] = std::async(std::launch::async, [&, index] {
                const auto& source = nodes[index % nodes.size()];
                return resolveNativeHydrologySpillSummaries(source.pageX, source.pageZ,
                                                            source.localBodyId, nodes, portals);
            });
        }
        for (auto& result : concurrent)
            REQUIRE(result.get() == expected);
    }
}

TEST_CASE("V4 tiled spill summaries preserve legal closure and wet portals",
          "[worldgen][hydrology][native-4][v4][lake][hierarchy][spill-summary][mass]") {
    std::array<NativeHydrologySpillNodeSummary, 3> nodes{{
        {.pageX = -3,
         .pageZ = -1,
         .localBodyId = 31,
         .localAnchorX = -6'080,
         .localAnchorZ = -1'900,
         .localStage = 80.0,
         .coreAreaSquareKilometers = 2.0,
         .coreVolumeCubicMeters = 2'000.0,
         .coreRunoffMmSquareKilometers = 20.0},
        {.pageX = -2,
         .pageZ = -1,
         .localBodyId = 32,
         .localAnchorX = -4'032,
         .localAnchorZ = -1'900,
         .localStage = 80.0,
         .coreAreaSquareKilometers = 3.0,
         .coreVolumeCubicMeters = 3'000.0,
         .coreRunoffMmSquareKilometers = 30.0},
        {.pageX = -1,
         .pageZ = -1,
         .localBodyId = 33,
         .localAnchorX = -1'984,
         .localAnchorZ = -1'900,
         .localStage = 70.0,
         .coreAreaSquareKilometers = 4.0,
         .coreVolumeCubicMeters = 4'000.0,
         .coreRunoffMmSquareKilometers = 40.0},
    }};
    const std::array<NativeHydrologySpillPortalSummary, 2> portals{{
        {.firstPageX = -3,
         .firstPageZ = -1,
         .firstLocalBodyId = 31,
         .secondPageX = -2,
         .secondPageZ = -1,
         .secondLocalBodyId = 32,
         .minimumWetStage = 60.0,
         .compatibleStage = 80.0,
         .x = -4'096,
         .z = -1'536},
        {.firstPageX = -2,
         .firstPageZ = -1,
         .firstLocalBodyId = 32,
         .secondPageX = -1,
         .secondPageZ = -1,
         .secondLocalBodyId = 33,
         .minimumWetStage = 75.0,
         .compatibleStage = 80.0,
         .x = -2'048,
         .z = -1'536},
    }};

    const auto retained = resolveNativeHydrologySpillSummaries(-3, -1, 31, nodes, portals);
    REQUIRE(retained);
    REQUIRE(retained->pageCount == 2);
    REQUIRE(retained->stage == 80.0);
    REQUIRE(retained->areaSquareKilometers == 5.0);
    REQUIRE(retained->volumeCubicMeters == 5'000.0);
    REQUIRE(retained->runoffMmSquareKilometers == 50.0);
    REQUIRE(retained->outlet == BasinOutlet::ENDORHEIC);

    const auto isolated = resolveNativeHydrologySpillSummaries(-1, -1, 33, nodes, portals);
    REQUIRE_FALSE(isolated);

    std::vector<NativeHydrologySpillNodeSummary> overBound(
        NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES + 1);
    for (size_t index = 0; index < overBound.size(); ++index) {
        overBound[index] = {
            .pageX = static_cast<int64_t>(index),
            .pageZ = 0,
            .localBodyId = 1'000 + index,
            .localAnchorX = static_cast<int64_t>(index) * NATIVE_HYDROLOGY_PAGE_EDGE,
            .localAnchorZ = 0,
            .localStage = 80.0,
        };
    }
    REQUIRE_THROWS_AS(
        resolveNativeHydrologySpillSummaries(0, 0, overBound.front().localBodyId, overBound, {}),
        std::invalid_argument);
}

TEST_CASE("V4 tiled lake hierarchy merges only proven opposing edge portals",
          "[worldgen][hydrology][native-4][v4][lake][page-edge][hierarchy][order][concurrency]") {
    const auto crossingBowl = [](int64_t centerX) {
        return [centerX](std::span<const NativeHydrologyPosition> positions,
                         std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double deltaX = static_cast<double>(positions[index].x - centerX);
                const double sideRise =
                    deltaX < 0.0 ? std::min(20.0, -deltaX) : std::min(30.0, deltaX * 2.0);
                const double crossRise =
                    std::min(24.0, std::abs(positions[index].z - 1'024.0) * 0.25);
                output[index] = climateInput(90.0 + sideRise + crossRise);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'200.0;
            }
        };
    };
    constexpr std::array<BasinSamplePosition, 3> positions{{
        {2'047.0, 1'024.0},
        {2'048.0, 1'024.0},
        {2'049.0, 1'024.0},
    }};
    constexpr uint64_t SEED = 0xF111'5A11'4E43'0001ULL;

    NativeHydrologyRouter forwardRouter(SEED);
    std::array<BasinSample, positions.size()> forward{};
    forwardRouter.samplePoints(positions, crossingBowl(2'048), forward);
    for (size_t index = 0; index < forward.size(); ++index) {
        const BasinSample& sample = forward[index];
        CAPTURE(index, sample.lake, sample.waterBodyId, sample.waterSurface,
                sample.surfaceElevation, sample.outletX, sample.outletZ);
        REQUIRE(sample.valid);
        REQUIRE(sample.lake);
        REQUIRE_FALSE(sample.ocean);
        REQUIRE_FALSE(sample.river);
        REQUIRE(sample.waterBodyId != NO_WATER_BODY);
        REQUIRE(sample.waterBodyId == forward.front().waterBodyId);
        REQUIRE(sample.waterSurface == forward.front().waterSurface);
        REQUIRE(sample.outlet == BasinOutlet::ENDORHEIC);
        REQUIRE(sample.endorheic);
        REQUIRE(sample.surfaceElevation <= sample.waterSurface - 0.125);
        REQUIRE(sample.erosionDepth >= 0.0);
        REQUIRE(sample.lakeVolumeCubicMeters >= 0.0);
        REQUIRE(sample.lakeRunoffMmSquareKilometers >= 0.0);
    }
    std::array<BasinSample, positions.size()> grid{};
    forwardRouter.sampleGrid(2'047, 1'024, 1, 1, static_cast<int>(grid.size()), 1,
                             crossingBowl(2'048), grid);
    for (size_t index = 0; index < grid.size(); ++index) {
        REQUIRE(nativeWaterHash(grid[index]) == nativeWaterHash(forward[index]));
        REQUIRE(grid[index].outletX == forward[index].outletX);
        REQUIRE(grid[index].outletZ == forward[index].outletZ);
    }
    const NativeHydrologyCacheMetrics reconciliationMetrics = forwardRouter.cacheMetrics();
    REQUIRE(reconciliationMetrics.openDepressionMisses == 1);
    REQUIRE(reconciliationMetrics.openDepressionHits >= positions.size() + grid.size() - 1);
    REQUIRE(reconciliationMetrics.openDepressionBuilds == 1);
    REQUIRE(reconciliationMetrics.openDepressionEntries == 1);
    REQUIRE(reconciliationMetrics.openDepressionBytes <=
            NATIVE_HYDROLOGY_MAX_HANDOFF_PAGES * NATIVE_HYDROLOGY_MAX_PAGE_BYTES);

    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(SEED);
    std::array<BasinSample, positions.size()> reversed{};
    reverseRouter.samplePoints(reversedPositions, crossingBowl(2'048), reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(forward[index]));
        REQUIRE(reversed[index].outletX == forward[index].outletX);
        REQUIRE(reversed[index].outletZ == forward[index].outletZ);
    }

    NativeHydrologyRouter concurrentRouter(SEED);
    std::array<std::future<BasinSample>, positions.size()> concurrent;
    for (size_t index = 0; index < positions.size(); ++index) {
        concurrent[index] = std::async(std::launch::async, [&, index] {
            return concurrentRouter.sample(positions[index].x, positions[index].z,
                                           crossingBowl(2'048));
        });
    }
    for (size_t index = 0; index < concurrent.size(); ++index) {
        const BasinSample sample = concurrent[index].get();
        REQUIRE(nativeWaterHash(sample) == nativeWaterHash(forward[index]));
        REQUIRE(sample.outletX == forward[index].outletX);
        REQUIRE(sample.outletZ == forward[index].outletZ);
    }

    // The same half-open portal contract must hold at a negative owner edge.
    constexpr std::array<BasinSamplePosition, 3> signedPositions{{
        {-1.0, 1'024.0},
        {0.0, 1'024.0},
        {1.0, 1'024.0},
    }};
    NativeHydrologyRouter signedRouter(SEED);
    std::array<BasinSample, signedPositions.size()> signedSamples{};
    signedRouter.samplePoints(signedPositions, crossingBowl(0), signedSamples);
    for (const BasinSample& sample : signedSamples) {
        REQUIRE(sample.lake);
        REQUIRE(sample.waterBodyId == signedSamples.front().waterBodyId);
        REQUIRE(sample.waterSurface == signedSamples.front().waterSurface);
        REQUIRE(sample.outlet == BasinOutlet::ENDORHEIC);
        REQUIRE(sample.endorheic);
        REQUIRE(sample.surfaceElevation <= sample.waterSurface - 0.125);
    }
}

TEST_CASE("V4 tiled lake hierarchy restarts from both immutable opposing pages",
          "[worldgen][hydrology][native-4][v4][lake][hierarchy][persistence][restart]") {
    constexpr uint64_t SEED = 0xF111'5A11'4E43'0002ULL;
    const NativeHydrologyInputFunction crossingBowl =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double deltaX = static_cast<double>(positions[index].x - 2'048);
                const double sideRise =
                    deltaX < 0.0 ? std::min(20.0, -deltaX) : std::min(30.0, deltaX * 2.0);
                const double crossRise =
                    std::min(24.0, std::abs(positions[index].z - 1'024.0) * 0.25);
                output[index] = climateInput(90.0 + sideRise + crossRise);
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'200.0;
            }
        };
    constexpr std::array<BasinSamplePosition, 3> positions{{
        {2'047.0, 1'024.0},
        {2'048.0, 1'024.0},
        {2'049.0, 1'024.0},
    }};
    TempDir directory("native_hydrology_lake_hierarchy_restart");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);

    std::array<BasinSample, positions.size()> expected{};
    {
        NativeHydrologyRouter writer(SEED, store);
        writer.samplePoints(positions, crossingBowl, expected);
        REQUIRE(writer.cacheMetrics().persistedWrites == 2);
    }
    REQUIRE(std::filesystem::exists(store->pagePath({0, 0})));
    REQUIRE(std::filesystem::exists(store->pagePath({1, 0})));

    bool queriedInput = false;
    NativeHydrologyRouter reader(SEED, store);
    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    std::array<BasinSample, positions.size()> restored{};
    reader.samplePoints(
        reversedPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted opposing lake page unexpectedly rebuilt");
        },
        restored);
    std::ranges::reverse(restored);
    REQUIRE_FALSE(queriedInput);
    REQUIRE(reader.cacheMetrics().persistedLoads == 2);
    for (size_t index = 0; index < restored.size(); ++index) {
        REQUIRE(nativeWaterHash(restored[index]) == nativeWaterHash(expected[index]));
        REQUIRE(restored[index].outlet == BasinOutlet::ENDORHEIC);
        REQUIRE(restored[index].endorheic);
        REQUIRE(restored[index].outletX == expected[index].outletX);
        REQUIRE(restored[index].outletZ == expected[index].outletZ);
        REQUIRE(restored[index].lakeVolumeCubicMeters == expected[index].lakeVolumeCubicMeters);
        REQUIRE(restored[index].lakeRunoffMmSquareKilometers ==
                expected[index].lakeRunoffMmSquareKilometers);
    }
}

TEST_CASE("V4 open depressions resolve from the locally dry receiving page first",
          "[worldgen][hydrology][native-4][v4][lake][open-depression][order][topology]") {
    constexpr uint64_t SEED = 0x0D47'51DE'F1A5'0001ULL;
    const NativeHydrologyInputFunction crossingBowl =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - 2'000.0, positions[index].z - 1'024.0);
                output[index] = climateInput(90.0 + std::min(20.0, radius * 0.2));
                output[index].climate.annualPrecipitationMm = 0.0;
                output[index].climate.potentialEvapotranspirationMm = 1'200.0;
            }
        };
    constexpr BasinSamplePosition DRY_SIDE{2'080.0, 1'024.0};
    constexpr BasinSamplePosition SOURCE_SIDE{2'000.0, 1'024.0};

    NativeHydrologyRouter dryFirstRouter(SEED);
    const BasinSample dryFirst = dryFirstRouter.sample(DRY_SIDE.x, DRY_SIDE.z, crossingBowl);
    const BasinSample sourceAfter =
        dryFirstRouter.sample(SOURCE_SIDE.x, SOURCE_SIDE.z, crossingBowl);
    REQUIRE(dryFirst.lake);
    REQUIRE(sourceAfter.lake);
    REQUIRE(dryFirst.waterBodyId == sourceAfter.waterBodyId);
    REQUIRE(dryFirst.waterSurface == sourceAfter.waterSurface);
    REQUIRE(dryFirst.outlet == BasinOutlet::ENDORHEIC);
    REQUIRE(dryFirst.surfaceElevation <= dryFirst.waterSurface - 0.125);
    REQUIRE(dryFirstRouter.cacheMetrics().openDepressionBuilds == 1);

    std::array<NativeHydrologyTopologyCell, 1> topology{};
    dryFirstRouter.sampleTopologyGrid(2'080, 1'024, 1, 1, crossingBowl, topology);
    REQUIRE(topology.front().waterTopologyPossible);

    NativeHydrologyRouter topologyFirstRouter(SEED);
    topology.front() = {};
    topologyFirstRouter.sampleTopologyGrid(2'080, 1'024, 1, 1, crossingBowl, topology);
    REQUIRE(topology.front().waterTopologyPossible);
    REQUIRE(topologyFirstRouter.cacheMetrics().openDepressionBuilds == 1);
    const BasinSample afterTopology =
        topologyFirstRouter.sample(DRY_SIDE.x, DRY_SIDE.z, crossingBowl);
    REQUIRE(nativeWaterHash(afterTopology) == nativeWaterHash(dryFirst));

    NativeHydrologyRouter sourceFirstRouter(SEED);
    const BasinSample sourceFirst =
        sourceFirstRouter.sample(SOURCE_SIDE.x, SOURCE_SIDE.z, crossingBowl);
    const BasinSample dryAfter = sourceFirstRouter.sample(DRY_SIDE.x, DRY_SIDE.z, crossingBowl);
    REQUIRE(nativeWaterHash(sourceFirst) == nativeWaterHash(sourceAfter));
    REQUIRE(nativeWaterHash(dryAfter) == nativeWaterHash(dryFirst));

    dryFirstRouter.clear();
    const BasinSample rebuilt = dryFirstRouter.sample(DRY_SIDE.x, DRY_SIDE.z, crossingBowl);
    REQUIRE(nativeWaterHash(rebuilt) == nativeWaterHash(dryFirst));

    TempDir directory("native_hydrology_open_depression_restart");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    {
        NativeHydrologyRouter writer(SEED, store);
        const BasinSample persisted = writer.sample(DRY_SIDE.x, DRY_SIDE.z, crossingBowl);
        REQUIRE(nativeWaterHash(persisted) == nativeWaterHash(dryFirst));
        REQUIRE(writer.cacheMetrics().persistedWrites == 2);
    }
    bool queriedInput = false;
    NativeHydrologyRouter reader(SEED, store);
    const BasinSample restored = reader.sample(
        DRY_SIDE.x, DRY_SIDE.z,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted open-depression page unexpectedly rebuilt");
        });
    REQUIRE_FALSE(queriedInput);
    REQUIRE(reader.cacheMetrics().persistedLoads == 2);
    REQUIRE(nativeWaterHash(restored) == nativeWaterHash(dryFirst));
}

TEST_CASE("V4 native grids reuse immutable owner pages within one mesh query",
          "[worldgen][hydrology][native-4][v4][performance][grid-cache]") {
    NativeHydrologyRouter router(0x1A2B'3C4DULL);
    constexpr int edge = 9;
    std::array<BasinSample, edge * edge> samples{};
    router.sampleGrid(512, 512, 4, 4, edge, edge, planarNativeInput(), samples);

    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.misses == 1);
    REQUIRE(metrics.hits == 0);
    REQUIRE(metrics.batchPageReuses == edge * edge - 1);
    REQUIRE(std::ranges::all_of(samples, [](const BasinSample& sample) { return sample.valid; }));

    std::array<BasinSamplePosition, edge * edge> positions{};
    for (int z = 0; z < edge; ++z) {
        for (int x = 0; x < edge; ++x) {
            positions[static_cast<size_t>(z * edge + x)] = {
                static_cast<double>(512 + x * NATIVE_HYDROLOGY_RASTER_SPACING),
                static_cast<double>(512 + z * NATIVE_HYDROLOGY_RASTER_SPACING),
            };
        }
    }
    std::array<BasinSample, edge * edge> points{};
    router.samplePoints(positions, planarNativeInput(), points);
    for (size_t index = 0; index < samples.size(); ++index) {
        const BasinSample& grid = samples[index];
        const BasinSample& point = points[index];
        REQUIRE(nativeWaterHash(grid) == nativeWaterHash(point));
        REQUIRE(std::bit_cast<uint64_t>(grid.terrainSlope) ==
                std::bit_cast<uint64_t>(point.terrainSlope));
        REQUIRE(std::bit_cast<uint64_t>(grid.flowX) == std::bit_cast<uint64_t>(point.flowX));
        REQUIRE(std::bit_cast<uint64_t>(grid.flowZ) == std::bit_cast<uint64_t>(point.flowZ));
        REQUIRE(std::bit_cast<uint64_t>(grid.baseflow) == std::bit_cast<uint64_t>(point.baseflow));
        REQUIRE(std::bit_cast<uint64_t>(grid.groundwaterHead) ==
                std::bit_cast<uint64_t>(point.groundwaterHead));
        REQUIRE(std::bit_cast<uint64_t>(grid.precipitationSeasonality) ==
                std::bit_cast<uint64_t>(point.precipitationSeasonality));
        REQUIRE(grid.transitionOwnerId == point.transitionOwnerId);
        REQUIRE(grid.transitionOwnerKind == point.transitionOwnerKind);
        REQUIRE(grid.generatedFluidLevel == point.generatedFluidLevel);
        REQUIRE(grid.outlet == point.outlet);
        REQUIRE(grid.wetland == point.wetland);
        REQUIRE(grid.perennial == point.perennial);
        REQUIRE(grid.ephemeral == point.ephemeral);
    }
}

TEST_CASE("V4 compact topology grids retain every signed cross-page output cell",
          "[worldgen][hydrology][native-4][v4][topology][negative][page-edge]") {
    NativeHydrologyRouter router(0x544F'504F'4C4F'4759ULL);
    std::vector<NativeHydrologyTopologyCell> topology(100);
    router.sampleTopologyGrid(-544, -2'080, 10, 10, planarNativeInput(), topology);
    REQUIRE(topology.size() == 100);
    REQUIRE(std::ranges::any_of(topology, [](const NativeHydrologyTopologyCell& cell) {
        return cell.waterTopologyPossible || cell.waterfallPossible;
    }));
}

TEST_CASE("V4 compact topology leaves uniform standing water on the fast path",
          "[worldgen][hydrology][native-4][v4][topology][uniform-water][fast-path]") {
    NativeHydrologyRouter router(0x554E'4946'4F52'4DULL);
    std::array<NativeHydrologyTopologyCell, 4> topology{};
    const NativeHydrologyInputFunction openWater =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            REQUIRE(positions.size() == output.size());
            for (NativeHydrologyInput& sample : output)
                sample = climateInput(-1.0);
        };
    router.sampleTopologyGrid(-64, -64, 2, 2, openWater, topology);
    REQUIRE(std::ranges::none_of(topology, [](const NativeHydrologyTopologyCell& cell) {
        return cell.waterTopologyPossible || cell.waterfallPossible;
    }));
}

TEST_CASE("V4 native handoffs reconcile stages and identities on signed page edges",
          "[worldgen][hydrology][native-4][v4][page-edge][handoff]") {
    constexpr std::array<BasinSamplePosition, 9> positions{{
        {2'047.0, 1'024.0},
        {2'048.0, 1'024.0},
        {2'049.0, 1'024.0},
        {1'024.0, 2'047.0},
        {1'024.0, 2'048.0},
        {1'024.0, 2'049.0},
        {-1.0, -1'024.0},
        {0.0, -1'024.0},
        {1.0, -1'024.0},
    }};
    NativeHydrologyRouter forwardRouter(0x8A7B'6C5D'4E3F'2011ULL);
    std::array<BasinSample, positions.size()> forward{};
    forwardRouter.samplePoints(positions, planarNativeInput(), forward);
    for (const BasinSample& sample : forward) {
        CAPTURE(sample.discharge, NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE);
        REQUIRE(sample.valid);
        REQUIRE(sample.river);
        REQUIRE(sample.waterBodyId != NO_WATER_BODY);
        REQUIRE(std::isfinite(sample.channelDistance));
    }
    for (size_t begin : {size_t{0}, size_t{3}, size_t{6}}) {
        REQUIRE(forward[begin].waterBodyId == forward[begin + 1].waterBodyId);
        REQUIRE(forward[begin + 1].waterBodyId == forward[begin + 2].waterBodyId);
        REQUIRE(std::abs(forward[begin].waterSurface - forward[begin + 1].waterSurface) < 0.1);
        REQUIRE(std::abs(forward[begin + 1].waterSurface - forward[begin + 2].waterSurface) < 0.1);
    }

    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(0x8A7B'6C5D'4E3F'2011ULL);
    std::array<BasinSample, positions.size()> reversed{};
    reverseRouter.samplePoints(reversedPositions, planarNativeInput(), reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(reversed[index].waterBodyId == forward[index].waterBodyId);
        REQUIRE(reversed[index].waterSurface == forward[index].waterSurface);
        REQUIRE(reversed[index].surfaceElevation == forward[index].surfaceElevation);
    }
}

TEST_CASE("V4 native routing does not synthesize a lake through dry page summaries",
          "[worldgen][hydrology][native-4][v4][hierarchy][lake][order][regression]") {
    const NativeHydrologyInputFunction elongatedBasin =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double normalizedX = (positions[index].x - 3'072.0) / 2'800.0;
                const double normalizedZ = (positions[index].z - 1'024.0) / 600.0;
                const double radius = std::hypot(normalizedX, normalizedZ);
                output[index] = climateInput(90.0 + std::min(10.0, radius * 10.0));
            }
        };
    constexpr std::array<BasinSamplePosition, 3> positions{{
        {512.0, 1'024.0},
        {3'072.0, 1'024.0},
        {5'632.0, 1'024.0},
    }};
    NativeHydrologyRouter forwardRouter(0x3A3A'B451'0001ULL);
    std::array<BasinSample, positions.size()> forward{};
    forwardRouter.samplePoints(positions, elongatedBasin, forward);
    // The tiled hierarchy joins only components proven wet at the same
    // opposing edge sample. It deliberately does not turn one page-edge
    // extremum into a basin stage through dry summaries.
    const bool synthesizedChainLake =
        std::ranges::all_of(forward, [](const BasinSample& sample) { return sample.lake; }) &&
        forward[0].waterBodyId == forward[1].waterBodyId &&
        forward[1].waterBodyId == forward[2].waterBodyId &&
        forward[0].waterSurface == forward[1].waterSurface &&
        forward[1].waterSurface == forward[2].waterSurface;
    REQUIRE_FALSE(synthesizedChainLake);

    std::array<BasinSamplePosition, positions.size()> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    NativeHydrologyRouter reverseRouter(0x3A3A'B451'0001ULL);
    std::array<BasinSample, positions.size()> reversed{};
    reverseRouter.samplePoints(reversedPositions, elongatedBasin, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(reversed[index].waterBodyId == forward[index].waterBodyId);
        REQUIRE(reversed[index].waterSurface == forward[index].waterSurface);
        REQUIRE(reversed[index].surfaceElevation == forward[index].surfaceElevation);
    }

    NativeHydrologyRouter scalarRouter(0x3A3A'B451'0001ULL);
    const BasinSample scalar = scalarRouter.sample(positions[1].x, positions[1].z, elongatedBasin);
    NativeHydrologyRouter singletonRouter(0x3A3A'B451'0001ULL);
    std::array<BasinSample, 1> singleton{};
    singletonRouter.samplePoints(std::span(positions).subspan(1, 1), elongatedBasin, singleton);
    constexpr std::array<BasinSamplePosition, 3> companionPositions{{
        positions[1],
        positions[0],
        positions[2],
    }};
    NativeHydrologyRouter companionRouter(0x3A3A'B451'0001ULL);
    std::array<BasinSample, companionPositions.size()> companions{};
    companionRouter.samplePoints(companionPositions, elongatedBasin, companions);
    constexpr std::array<BasinSamplePosition, 3> reverseCompanionPositions{{
        positions[2],
        positions[0],
        positions[1],
    }};
    NativeHydrologyRouter reverseCompanionRouter(0x3A3A'B451'0001ULL);
    std::array<BasinSample, reverseCompanionPositions.size()> reverseCompanions{};
    reverseCompanionRouter.samplePoints(reverseCompanionPositions, elongatedBasin,
                                        reverseCompanions);
    const uint64_t canonicalHash = nativeWaterHash(scalar);
    REQUIRE(nativeWaterHash(singleton[0]) == canonicalHash);
    REQUIRE(nativeWaterHash(companions[0]) == canonicalHash);
    REQUIRE(nativeWaterHash(reverseCompanions[2]) == canonicalHash);
}

TEST_CASE("V4 bounded river inflow survives beyond both shared page aprons",
          "[worldgen][hydrology][native-4][v4][hierarchy][river][discharge]") {
    const NativeHydrologyInputFunction valley =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double crossValley =
                    std::min(24.0, std::abs(positions[index].z - 1'024.0) * 0.018);
                output[index] = climateInput(170.0 - positions[index].x * 0.003 + crossValley);
            }
        };
    constexpr std::array<BasinSamplePosition, 6> positions{{
        {2'049.0, 1'024.0},
        {2'304.0, 1'024.0},
        {4'000.0, 1'024.0},
        {4'095.0, 1'024.0},
        {4'096.0, 1'024.0},
        {4'352.0, 1'024.0},
    }};
    NativeHydrologyRouter router(0x3A3A'B451'0002ULL);
    std::array<BasinSample, positions.size()> samples{};
    router.samplePoints(positions, valley, samples);
    const WaterBodyId reach = samples.front().waterBodyId;
    REQUIRE(reach != NO_WATER_BODY);
    for (size_t index = 0; index < samples.size(); ++index) {
        REQUIRE(samples[index].river);
        REQUIRE(samples[index].waterBodyId == reach);
        REQUIRE(samples[index].discharge >= NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE);
        if (index > 0)
            REQUIRE(samples[index].waterSurface <= samples[index - 1].waterSurface + 0.05);
    }
}

TEST_CASE("V4 native channel projection and lake distance avoid categorical raster geometry",
          "[worldgen][hydrology][native-4][v4][curve][shore-distance]") {
    NativeHydrologyRouter riverRouter(0x1357'2468ULL);
    bool foundSubcellCenter = false;
    for (int z = 996; z <= 1'052; ++z) {
        double bestDistance = std::numeric_limits<double>::infinity();
        int bestX = 0;
        for (int x = 996; x <= 1'052; ++x) {
            const BasinSample sample = riverRouter.sample(x, z, planarNativeInput());
            if (sample.channelDistance < bestDistance) {
                bestDistance = sample.channelDistance;
                bestX = x;
            }
        }
        if (bestDistance < 0.6 && bestX % NATIVE_HYDROLOGY_RASTER_SPACING != 0) {
            foundSubcellCenter = true;
            break;
        }
    }
    REQUIRE(foundSubcellCenter);

    const NativeHydrologyInputFunction circularBowl =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - 1'024.0, positions[index].z - 1'024.0);
                output[index] = climateInput(90.0 + std::min(10.0, radius * 0.05));
            }
        };
    NativeHydrologyRouter lakeRouter(0x2468'1357ULL);
    const auto shoreDistance = [&](int64_t x, int64_t z) {
        return lakeRouter.sample(static_cast<double>(x), static_cast<double>(z), circularBowl)
            .lakeShoreDistance;
    };
    constexpr int64_t DIAGONAL = 1'162;
    const double gradientX =
        (shoreDistance(DIAGONAL + 1, DIAGONAL) - shoreDistance(DIAGONAL - 1, DIAGONAL)) * 0.5;
    const double gradientZ =
        (shoreDistance(DIAGONAL, DIAGONAL + 1) - shoreDistance(DIAGONAL, DIAGONAL - 1)) * 0.5;
    REQUIRE(std::abs(gradientX) > 0.15);
    REQUIRE(std::abs(gradientZ) > 0.15);
    REQUIRE(std::abs(gradientX / gradientZ) > 0.5);
    REQUIRE(std::abs(gradientX / gradientZ) < 2.0);
}

TEST_CASE("V4 ordinary channel grades do not become proximity waterfalls",
          "[worldgen][hydrology][native-4][v4][curve][stage][waterfall][regression]") {
    constexpr int64_t ORIGIN_X = 992;
    constexpr int64_t ORIGIN_Z = 992;
    constexpr int EDGE = 65;
    const NativeHydrologyInputFunction steepOrdinaryPlane =
        [](std::span<const NativeHydrologyPosition> positions,
           std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                // One diagonal native receiver interval descends about 3.2
                // meters, safely below the 30-meter explicit-fall threshold.
                // The accumulated ordinary grade still spans half a block
                // across nearby longitudinal splines, so proximity-only
                // contact logic would incorrectly relabel this plane as a
                // sequence of unrelated falls. Body filtering alone leaves
                // visible steps where adjacent ribbons choose different IDs.
                output[index] =
                    climateInput(1'000.0 - positions[index].x * 0.4 - positions[index].z * 0.4);
            }
        };

    NativeHydrologyRouter router(0x5354'4545'5000'0004ULL);
    std::array<BasinSample, EDGE * EDGE> samples{};
    router.sampleGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, steepOrdinaryPlane, samples);
    const auto indexOf = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };

    size_t riverCount = 0;
    size_t waterfallCount = 0;
    size_t staleTransitionOwners = 0;
    size_t uphillStageSteps = 0;
    size_t unownedStageJumps = 0;
    double maximumStageJump = 0.0;
    double minimumStage = std::numeric_limits<double>::infinity();
    double maximumStage = -std::numeric_limits<double>::infinity();
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const BasinSample& sample = samples[indexOf(x, z)];
            riverCount += sample.river ? 1U : 0U;
            waterfallCount += sample.waterfall ? 1U : 0U;
            if (!sample.river) {
                // A dry fringe is physical only outside the protected routed
                // core. This catches wet deletion while permitting terrain
                // above the stage to clip the outer width of a broad ribbon.
                REQUIRE(sample.channelDistance > std::min(sample.channelWidth * 0.5,
                                                          NATIVE_HYDROLOGY_RASTER_SPACING * 0.75) -
                                                     1.0e-6);
                REQUIRE(sample.waterBodyId == NO_WATER_BODY);
                continue;
            }
            if (sample.waterfall)
                continue;
            minimumStage = std::min(minimumStage, visibleStage(sample));
            maximumStage = std::max(maximumStage, visibleStage(sample));
            staleTransitionOwners += sample.transitionOwnerKind != WaterTransitionKind::NONE ||
                                             sample.transitionOwnerId != 0
                                         ? 1U
                                         : 0U;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE)
                    continue;
                const BasinSample& adjacent = samples[indexOf(x + offsetX, z + offsetZ)];
                if (!adjacent.river || adjacent.waterfall)
                    continue;
                const double stageStep = visibleStage(sample) - visibleStage(adjacent);
                uphillStageSteps += stageStep < -1.0e-6 ? 1U : 0U;
                unownedStageJumps += stageStep > 0.125001 ? 1U : 0U;
                maximumStageJump = std::max(maximumStageJump, std::abs(stageStep));
            }
        }
    }
    CAPTURE(riverCount, waterfallCount, staleTransitionOwners, uphillStageSteps, unownedStageJumps,
            maximumStageJump, minimumStage, maximumStage);
    // Terrain may legitimately clip a ribbon fringe that rises above its
    // routed stage. It must never clip the native routed core itself. Every
    // native source in this uniformly draining fixture has its canonical
    // curve endpoint at the 1.5-block cell center, so probe those endpoints
    // directly rather than treating every off-center ribbon pixel as water.
    std::array<BasinSamplePosition, 16 * 16> routedCenters{};
    for (int z = 0; z < 16; ++z) {
        for (int x = 0; x < 16; ++x) {
            routedCenters[static_cast<size_t>(z * 16 + x)] = {
                .x = static_cast<double>(ORIGIN_X + x * NATIVE_HYDROLOGY_RASTER_SPACING) + 1.5,
                .z = static_cast<double>(ORIGIN_Z + z * NATIVE_HYDROLOGY_RASTER_SPACING) + 1.5,
            };
        }
    }
    std::array<BasinSample, routedCenters.size()> routedSamples{};
    router.samplePoints(routedCenters, steepOrdinaryPlane, routedSamples);
    size_t routedSourceCount = 0;
    for (size_t index = 0; index < routedSamples.size(); ++index) {
        const BasinSample& sample = routedSamples[index];
        // A native river source owns at least one spline beginning exactly at
        // this canonical endpoint. Non-source raster cells can lie in another
        // route's outer ribbon and are deliberately not source obligations.
        if (sample.channelDistance > 1.0e-6)
            continue;
        ++routedSourceCount;
        CAPTURE(index, routedCenters[index].x, routedCenters[index].z, sample.river,
                sample.waterfall, sample.surfaceElevation, sample.waterSurface,
                sample.channelDistance, sample.channelWidth, sample.transitionOwnerKind,
                sample.transitionOwnerId);
        REQUIRE(sample.river);
        REQUIRE_FALSE(sample.waterfall);
        REQUIRE(sample.surfaceElevation <= sample.waterSurface - 0.124999);
        REQUIRE(sample.waterBodyId != NO_WATER_BODY);
        REQUIRE(sample.transitionOwnerKind == WaterTransitionKind::NONE);
        REQUIRE(sample.transitionOwnerId == 0);
    }
    CAPTURE(routedSourceCount);
    REQUIRE(routedSourceCount > 16);
    REQUIRE(waterfallCount == 0);
    REQUIRE(staleTransitionOwners == 0);
    REQUIRE(uphillStageSteps == 0);
    REQUIRE(unownedStageJumps == 0);
    // Reconciliation must retain the ordinary downhill grade rather than
    // satisfying continuity by flattening a connected page to one stage.
    REQUIRE(maximumStage >= minimumStage + 1.0);
}

TEST_CASE("V4 standing shore contacts are continuous or own a routed fall",
          "[worldgen][hydrology][native-4][v4][lake][river][shoreline][waterfall]"
          "[persistence][regression]") {
    constexpr int64_t ORIGIN_X = 900;
    constexpr int64_t ORIGIN_Z = 1'016;
    constexpr int EDGE = 17;
    constexpr uint64_t SEED = 0x4859'4452'4F50'5242ULL;
    const NativeHydrologyInputFunction bowl = [](std::span<const NativeHydrologyPosition> positions,
                                                 std::span<NativeHydrologyInput> output) {
        for (size_t index = 0; index < positions.size(); ++index) {
            const double dx = positions[index].x - 1'024.0;
            const double dz = positions[index].z - 1'024.0;
            const double radius = std::hypot(dx, dz);
            const double elevation =
                radius <= 120.0 ? 80.0 + radius * 0.25 : 110.0 - std::min(35.0, radius - 120.0);
            output[index] = climateInput(elevation);
        }
    };
    const auto indexOf = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    const auto explicitFall = [](const BasinSample& sample) {
        return sample.waterfall &&
               sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
               sample.transitionOwnerId != 0;
    };
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };

    TempDir directory("native_hydrology_standing_shore_contact");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    std::array<BasinSample, EDGE * EDGE> samples{};
    router.sampleGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, bowl, samples);
    const auto at = [&](int x, int z) -> const BasinSample& { return samples[indexOf(x, z)]; };
    const BasinSample& routedOutlet = at(8, 8); // world (908, 1024)
    REQUIRE(routedOutlet.river);
    REQUIRE_FALSE(routedOutlet.lake);
    const WaterBodyId outletBody = routedOutlet.waterBodyId;
    REQUIRE(outletBody != NO_WATER_BODY);

    size_t lakeCount = 0;
    size_t riverCount = 0;
    size_t compatibleLakeRiverEdges = 0;
    double minimumOutletStage = std::numeric_limits<double>::infinity();
    double maximumOutletStage = -std::numeric_limits<double>::infinity();
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const BasinSample& sample = at(x, z);
            lakeCount += sample.lake ? 1U : 0U;
            riverCount += sample.river ? 1U : 0U;
            if (sample.river && sample.waterBodyId == outletBody) {
                minimumOutletStage = std::min(minimumOutletStage, visibleStage(sample));
                maximumOutletStage = std::max(maximumOutletStage, visibleStage(sample));
            }
            if (sample.river) {
                CAPTURE(x, z, sample.surfaceElevation, sample.waterSurface, sample.lake,
                        sample.river, sample.waterfall);
                REQUIRE(sample.surfaceElevation <= sample.waterSurface - 0.124999);
            }
            if (sample.waterfall) {
                REQUIRE(explicitFall(sample));
                REQUIRE(sample.waterfallTop >= sample.waterfallBottom + 0.5);
            }
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE)
                    continue;
                const BasinSample& adjacent = at(x + offsetX, z + offsetZ);
                if (sample.river && adjacent.river && sample.waterBodyId == outletBody &&
                    adjacent.waterBodyId == outletBody) {
                    const double stageStep =
                        std::abs(visibleStage(sample) - visibleStage(adjacent));
                    CAPTURE(x, z, offsetX, offsetZ, stageStep, sample.waterSurface,
                            adjacent.waterSurface, sample.transitionOwnerKind,
                            sample.transitionOwnerId, adjacent.transitionOwnerKind,
                            adjacent.transitionOwnerId);
                    REQUIRE(
                        (stageStep <= 0.125001 || explicitFall(sample) || explicitFall(adjacent)));
                }
                if (!((sample.lake && adjacent.river) || (sample.river && adjacent.lake))) {
                    continue;
                }
                const BasinSample& river = sample.river ? sample : adjacent;
                if (river.waterBodyId != outletBody)
                    continue;
                ++compatibleLakeRiverEdges;
                const double stageStep = std::abs(visibleStage(sample) - visibleStage(adjacent));
                CAPTURE(x, z, offsetX, offsetZ, stageStep, sample.lake, sample.river,
                        sample.waterfall, sample.waterSurface, sample.waterBodyId,
                        sample.transitionOwnerKind, sample.transitionOwnerId, adjacent.lake,
                        adjacent.river, adjacent.waterfall, adjacent.waterSurface,
                        adjacent.waterBodyId, adjacent.transitionOwnerKind,
                        adjacent.transitionOwnerId);
                REQUIRE((stageStep <= 0.125001 || explicitFall(sample) || explicitFall(adjacent)));
            }
        }
    }
    CAPTURE(lakeCount, riverCount, compatibleLakeRiverEdges, minimumOutletStage,
            maximumOutletStage);
    REQUIRE(lakeCount > 20);
    REQUIRE(riverCount > 20);
    REQUIRE(compatibleLakeRiverEdges > 10);
    REQUIRE(maximumOutletStage >= minimumOutletStage + 0.125);

    // The routed outlet approaches the standing stage without changing its
    // categorical body. A different reach touching the north shore is a
    // proximity-only neighbor and must not be raised into that lake.
    const BasinSample& outletEdge = at(11, 8); // world (911, 1024)
    const BasinSample& standing = at(12, 8);   // world (912, 1024)
    REQUIRE(outletEdge.river);
    REQUIRE(outletEdge.waterBodyId == outletBody);
    REQUIRE(standing.lake);
    REQUIRE_FALSE(standing.river);
    REQUIRE(outletEdge.waterBodyId != standing.waterBodyId);
    const BasinSample& unrelated = at(15, 0);         // world (915, 1016)
    const BasinSample& unrelatedStanding = at(16, 0); // world (916, 1016)
    REQUIRE(unrelated.river);
    REQUIRE(unrelated.waterBodyId != outletBody);
    REQUIRE(unrelatedStanding.lake);
    REQUIRE(visibleStage(unrelated) <= visibleStage(unrelatedStanding) - 0.125);

    std::array<NativeHydrologyTopologyCell, 2> topology{};
    router.sampleTopologyGrid(896, 992, 1, 2, bowl, topology);
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const BasinSample& sample = at(x, z);
            if (!sample.waterfall)
                continue;
            const size_t topologyZ = static_cast<size_t>((ORIGIN_Z + z - 992) / 32);
            REQUIRE(topology[topologyZ].waterfallPossible);
        }
    }

    std::vector<BasinSamplePosition> forwardPositions;
    forwardPositions.reserve(samples.size());
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            forwardPositions.push_back(
                {.x = static_cast<double>(ORIGIN_X + x), .z = static_cast<double>(ORIGIN_Z + z)});
        }
    }
    std::vector<BasinSamplePosition> reversedPositions = forwardPositions;
    std::ranges::reverse(reversedPositions);
    std::vector<BasinSample> points(samples.size());
    router.samplePoints(reversedPositions, bowl, points);
    std::ranges::reverse(points);
    for (size_t index = 0; index < samples.size(); ++index) {
        REQUIRE(nativeWaterHash(points[index]) == nativeWaterHash(samples[index]));
        REQUIRE(std::bit_cast<uint64_t>(points[index].waterfallTop) ==
                std::bit_cast<uint64_t>(samples[index].waterfallTop));
        REQUIRE(std::bit_cast<uint64_t>(points[index].waterfallBottom) ==
                std::bit_cast<uint64_t>(samples[index].waterfallBottom));
        REQUIRE(points[index].waterfallAnchor == samples[index].waterfallAnchor);
        const BasinSample scalar =
            router.sample(forwardPositions[index].x, forwardPositions[index].z, bowl);
        REQUIRE(nativeWaterHash(scalar) == nativeWaterHash(samples[index]));
    }

    router.clear();
    const NativeHydrologyCacheMetrics metricsBeforeReload = router.cacheMetrics();
    std::vector<BasinSample> restored(samples.size());
    bool queriedInput = false;
    router.samplePoints(
        reversedPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted standing-shore page unexpectedly rebuilt");
        },
        restored);
    std::ranges::reverse(restored);
    REQUIRE_FALSE(queriedInput);
    REQUIRE(router.cacheMetrics().persistedLoads > metricsBeforeReload.persistedLoads);
    for (size_t index = 0; index < samples.size(); ++index) {
        REQUIRE(nativeWaterHash(restored[index]) == nativeWaterHash(samples[index]));
        REQUIRE(std::bit_cast<uint64_t>(restored[index].waterfallTop) ==
                std::bit_cast<uint64_t>(samples[index].waterfallTop));
        REQUIRE(std::bit_cast<uint64_t>(restored[index].waterfallBottom) ==
                std::bit_cast<uint64_t>(samples[index].waterfallBottom));
        REQUIRE(restored[index].waterfallAnchor == samples[index].waterfallAnchor);
    }
}

TEST_CASE("V4 unrelated parallel channels do not excavate an unowned stage seam",
          "[worldgen][hydrology][native-4][v4][river][curve][stage][body][regression]") {
    constexpr uint64_t SEED = 0x5041'5241'4C4C'454CULL;
    const auto valleys = [](bool includeLowValley) {
        return [includeLowValley](std::span<const NativeHydrologyPosition> positions,
                                  std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double high = 160.0 + 1.25 * std::abs(positions[index].z - 1'056.0);
                const double low = 100.0 + 5.0 * std::abs(positions[index].z - 1'024.0);
                // Both sides meet continuously at a 180-meter drainage
                // divide. Every longitudinal native receiver drop is
                // ordinary; the distinct valleys never own a cross-flow fall.
                const double crossSection =
                    includeLowValley && positions[index].z <= 1'040 ? low : high;
                output[index] = climateInput(crossSection - 0.02 * (positions[index].x - 1'024.0));
            }
        };
    };
    std::array<BasinSamplePosition, 9> positions{};
    for (size_t index = 0; index < positions.size(); ++index) {
        positions[index] = {.x = 1'024.0, .z = 1'024.0 + static_cast<double>(index * 4)};
    }

    NativeHydrologyRouter dualRouter(SEED);
    std::array<BasinSample, positions.size()> dual{};
    dualRouter.samplePoints(positions, valleys(true), dual);
    const BasinSample& lowCenter = dual.front();
    const BasinSample& highCenter = dual.back();
    REQUIRE(lowCenter.river);
    REQUIRE(highCenter.river);
    REQUIRE(lowCenter.waterBodyId != highCenter.waterBodyId);
    REQUIRE(highCenter.waterSurface >= lowCenter.waterSurface + 4.0);

    NativeHydrologyRouter isolatedRouter(SEED);
    const BasinSample isolatedHigh =
        isolatedRouter.sample(positions.back().x, positions.back().z, valleys(false));
    CAPTURE(isolatedHigh.waterBodyId, highCenter.waterBodyId, isolatedHigh.waterSurface,
            highCenter.waterSurface, isolatedHigh.channelDistance, highCenter.channelDistance,
            isolatedHigh.channelWidth, highCenter.channelWidth, isolatedHigh.flowX,
            isolatedHigh.flowZ, highCenter.flowX, highCenter.flowZ);
    REQUIRE(isolatedHigh.river);
    REQUIRE(isolatedHigh.waterBodyId != NO_WATER_BODY);
    REQUIRE(std::abs(isolatedHigh.waterSurface - highCenter.waterSurface) <= 0.125001);

    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };
    size_t dryDivideSamples = 0;
    for (size_t index = 0; index < dual.size(); ++index) {
        const BasinSample& sample = dual[index];
        CAPTURE(index, positions[index].z, sample.river, sample.waterfall, sample.surfaceElevation,
                sample.waterSurface, sample.waterBodyId, sample.transitionOwnerKind,
                sample.transitionOwnerId);
        REQUIRE(sample.valid);
        REQUIRE_FALSE(sample.waterfall);
        REQUIRE(sample.transitionOwnerKind == WaterTransitionKind::NONE);
        REQUIRE(sample.transitionOwnerId == 0);
        if (sample.river) {
            REQUIRE(sample.surfaceElevation <= sample.waterSurface - 0.124999);
        } else {
            ++dryDivideSamples;
            REQUIRE_FALSE(sample.ocean);
            REQUIRE_FALSE(sample.lake);
            REQUIRE_FALSE(sample.wetland);
            REQUIRE(sample.waterBodyId == NO_WATER_BODY);
            REQUIRE(sample.surfaceElevation > lowCenter.waterSurface + 0.5);
        }
        if (index == 0 || !dual[index - 1].river || !sample.river)
            continue;
        REQUIRE(std::abs(visibleStage(sample) - visibleStage(dual[index - 1])) <= 0.125001);
    }
    // Clipping the false cross-divide ribbon is allowed, but the actual
    // centerlines above prove that doing so did not delete either river.
    CAPTURE(dryDivideSamples);
}

TEST_CASE("V4 persisted closed outlets own branch-specific native and far falls",
          "[worldgen][hydrology][native-4][v4][persistence][waterfall][far-terrain]"
          "[regression]") {
    constexpr uint64_t SEED = 0x0A71'1E7F'A110'0004ULL;
    constexpr size_t HEADER_BYTES = 52;
    constexpr size_t RASTER_EDGE = 2'048 / 4 + 1 + 4;
    constexpr size_t RASTER_CELLS = RASTER_EDGE * RASTER_EDGE;
    constexpr size_t LAKE_SUMMARY_BYTES = 32;
    constexpr size_t DEPRESSION_SUMMARY_BYTES = 76;
    constexpr uint8_t CELL_RIVER = 1U << 2U;
    constexpr size_t LAKE_STATS_COUNT_OFFSET = 40;
    constexpr size_t DEPRESSION_COUNT_OFFSET = 44;
    constexpr size_t WATER_SURFACE_OFFSET = HEADER_BYTES + 3 * 4 * RASTER_CELLS;
    constexpr size_t RECEIVER_FIRST_OFFSET = HEADER_BYTES + 13 * 4 * RASTER_CELLS;
    constexpr size_t RECEIVER_SECOND_OFFSET = RECEIVER_FIRST_OFFSET + 4 * RASTER_CELLS;
    constexpr size_t RECEIVER_SECOND_WEIGHT_OFFSET = RECEIVER_SECOND_OFFSET + 4 * RASTER_CELLS;
    constexpr size_t WATER_BODY_OFFSET = RECEIVER_SECOND_WEIGHT_OFFSET + 4 * RASTER_CELLS;
    constexpr size_t STREAM_ORDER_OFFSET = WATER_BODY_OFFSET + 8 * RASTER_CELLS;
    constexpr size_t FLAGS_OFFSET = STREAM_ORDER_OFFSET + RASTER_CELLS;
    constexpr size_t WATERFALL_BRANCH_MASK_OFFSET = FLAGS_OFFSET + RASTER_CELLS;
    constexpr size_t VARIABLE_SUMMARIES_OFFSET = WATERFALL_BRANCH_MASK_OFFSET + RASTER_CELLS;

    const auto readU32 = [](std::span<const uint8_t> bytes, size_t offset) {
        REQUIRE(offset <= bytes.size());
        REQUIRE(bytes.size() - offset >= 4);
        uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            value |= static_cast<uint32_t>(bytes[offset++]) << shift;
        return value;
    };
    const auto readU64 = [](std::span<const uint8_t> bytes, size_t offset) {
        REQUIRE(offset <= bytes.size());
        REQUIRE(bytes.size() - offset >= 8);
        uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
            value |= static_cast<uint64_t>(bytes[offset++]) << shift;
        return value;
    };
    const auto readFloat = [&](std::span<const uint8_t> bytes, size_t offset) {
        return std::bit_cast<float>(readU32(bytes, offset));
    };
    const auto overwriteU32 = [](std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
        REQUIRE(offset <= bytes.size());
        REQUIRE(bytes.size() - offset >= 4);
        for (unsigned shift = 0; shift < 32; shift += 8)
            bytes[offset++] = static_cast<uint8_t>(value >> shift);
    };
    const auto appendU32 = [](std::vector<uint8_t>& bytes, uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8)
            bytes.push_back(static_cast<uint8_t>(value >> shift));
    };
    const auto appendU64 = [](std::vector<uint8_t>& bytes, uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8)
            bytes.push_back(static_cast<uint8_t>(value >> shift));
    };
    const auto appendFloat = [&](std::vector<uint8_t>& bytes, float value) {
        appendU32(bytes, std::bit_cast<uint32_t>(value));
    };
    const auto appendDouble = [&](std::vector<uint8_t>& bytes, double value) {
        appendU64(bytes, std::bit_cast<uint64_t>(value));
    };

    TempDir directory("native_hydrology_persisted_outlet_fall");
    constexpr worldgen::hydrology::HydrologyPageCoordinate COORDINATE{0, 0};
    const auto validStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "valid-hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    {
        NativeHydrologyRouter writer(SEED, validStore);
        REQUIRE(writer.sample(1'024.0, 1'024.0, planarNativeInput()).valid);
        REQUIRE(writer.cacheMetrics().persistedWrites == 1);
    }
    auto validPayloadResult = validStore->load(COORDINATE);
    REQUIRE(validPayloadResult.isReady());
    std::vector<uint8_t> payload = *validPayloadResult.value();
    REQUIRE(payload.size() > VARIABLE_SUMMARIES_OFFSET);
    REQUIRE(std::ranges::equal(std::span(payload).first(4),
                               std::array<uint8_t, 4>{'N', 'H', '4', 'P'}));
    const uint32_t lakeStatsCount = readU32(payload, LAKE_STATS_COUNT_OFFSET);
    const uint32_t depressionCount = readU32(payload, DEPRESSION_COUNT_OFFSET);
    const size_t depressionOffset =
        VARIABLE_SUMMARIES_OFFSET + static_cast<size_t>(lakeStatsCount) * LAKE_SUMMARY_BYTES;
    const size_t edgeSummaryOffset =
        depressionOffset + static_cast<size_t>(depressionCount) * DEPRESSION_SUMMARY_BYTES;
    REQUIRE(edgeSummaryOffset <= payload.size());

    std::set<int32_t> existingOutletSources;
    for (uint32_t depression = 0; depression < depressionCount; ++depression) {
        const size_t offset = depressionOffset + depression * DEPRESSION_SUMMARY_BYTES;
        const int64_t outletX = static_cast<int64_t>(readU64(payload, offset + 52));
        const int64_t outletZ = static_cast<int64_t>(readU64(payload, offset + 60));
        if (outletX % NATIVE_HYDROLOGY_RASTER_SPACING != 0 ||
            outletZ % NATIVE_HYDROLOGY_RASTER_SPACING != 0) {
            continue;
        }
        const int rasterX = static_cast<int>(outletX / NATIVE_HYDROLOGY_RASTER_SPACING) + 2;
        const int rasterZ = static_cast<int>(outletZ / NATIVE_HYDROLOGY_RASTER_SPACING) + 2;
        if (rasterX >= 0 && rasterX < static_cast<int>(RASTER_EDGE) && rasterZ >= 0 &&
            rasterZ < static_cast<int>(RASTER_EDGE)) {
            existingOutletSources.insert(
                static_cast<int32_t>(rasterZ * static_cast<int>(RASTER_EDGE) + rasterX));
        }
    }

    int32_t source = -1;
    std::array<int32_t, 2> targets{-1, -1};
    float secondWeight = 0.0F;
    for (int rasterZ = 96; rasterZ < static_cast<int>(RASTER_EDGE) - 96 && source < 0; ++rasterZ) {
        for (int rasterX = 96; rasterX < static_cast<int>(RASTER_EDGE) - 96; ++rasterX) {
            const int32_t candidate = rasterZ * static_cast<int32_t>(RASTER_EDGE) + rasterX;
            const size_t cell = static_cast<size_t>(candidate);
            if ((payload[FLAGS_OFFSET + cell] & CELL_RIVER) == 0 ||
                payload[WATERFALL_BRANCH_MASK_OFFSET + cell] != 0 ||
                existingOutletSources.contains(candidate)) {
                continue;
            }
            const int32_t first =
                static_cast<int32_t>(readU32(payload, RECEIVER_FIRST_OFFSET + cell * 4));
            const int32_t second =
                static_cast<int32_t>(readU32(payload, RECEIVER_SECOND_OFFSET + cell * 4));
            const float weight = readFloat(payload, RECEIVER_SECOND_WEIGHT_OFFSET + cell * 4);
            if (first < 0 || second < 0 || weight < 0.2F || weight > 0.8F)
                continue;
            const size_t firstCell = static_cast<size_t>(first);
            const size_t secondCell = static_cast<size_t>(second);
            if ((payload[FLAGS_OFFSET + firstCell] & 0x0FU) == 0 ||
                (payload[FLAGS_OFFSET + secondCell] & 0x0FU) == 0 ||
                readU64(payload, WATER_BODY_OFFSET + cell * 8) == NO_WATER_BODY ||
                readU64(payload, WATER_BODY_OFFSET + firstCell * 8) == NO_WATER_BODY ||
                readU64(payload, WATER_BODY_OFFSET + secondCell * 8) == NO_WATER_BODY) {
                continue;
            }
            source = candidate;
            targets = {first, second};
            secondWeight = weight;
            break;
        }
    }
    CAPTURE(source, targets[0], targets[1], secondWeight, lakeStatsCount, depressionCount);
    REQUIRE(source >= 0);
    REQUIRE(targets[0] >= 0);
    REQUIRE(targets[1] >= 0);

    const int sourceRasterX = source % static_cast<int32_t>(RASTER_EDGE);
    const int sourceRasterZ = source / static_cast<int32_t>(RASTER_EDGE);
    const int64_t sourceWorldX =
        static_cast<int64_t>(sourceRasterX - 2) * NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t sourceWorldZ =
        static_cast<int64_t>(sourceRasterZ - 2) * NATIVE_HYDROLOGY_RASTER_SPACING;
    const double firstTargetStage =
        readFloat(payload, WATER_SURFACE_OFFSET + static_cast<size_t>(targets[0]) * 4);
    const double secondTargetStage =
        readFloat(payload, WATER_SURFACE_OFFSET + static_cast<size_t>(targets[1]) * 4);
    const float standingStage =
        static_cast<float>(std::max(firstTargetStage, secondTargetStage) + 6.0);
    constexpr WaterBodyId STANDING_BODY = 0xC105'ED0A'711E'7004ULL;

    std::vector<uint8_t> summary;
    summary.reserve(DEPRESSION_SUMMARY_BYTES);
    appendU64(summary, STANDING_BODY);
    appendU64(summary, static_cast<uint64_t>(sourceWorldX));
    appendU64(summary, static_cast<uint64_t>(sourceWorldZ));
    appendFloat(summary, standingStage);
    appendDouble(summary, 1.0);
    appendDouble(summary, 1.0);
    appendDouble(summary, 1.0);
    // Production summaries name the first routed cell of the outlet edge.
    appendU64(summary, static_cast<uint64_t>(sourceWorldX));
    appendU64(summary, static_cast<uint64_t>(sourceWorldZ));
    appendFloat(summary, standingStage);
    summary.push_back(0); // Closed local component with immutable standing stage.
    summary.push_back(static_cast<uint8_t>(BasinOutlet::SHARED_PORTAL));
    summary.insert(summary.end(), 2, 0);
    REQUIRE(summary.size() == DEPRESSION_SUMMARY_BYTES);
    payload.insert(payload.begin() + static_cast<std::ptrdiff_t>(edgeSummaryOffset),
                   summary.begin(), summary.end());
    overwriteU32(payload, DEPRESSION_COUNT_OFFSET, depressionCount + 1);

    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "outlet-hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    REQUIRE(store->write(COORDINATE, payload).isReady());
    auto storedPayload = store->load(COORDINATE);
    REQUIRE(storedPayload.isReady());
    REQUIRE(*storedPayload.value() == payload);

    const NativeHydrologyInputFunction forbiddenInput = [](std::span<const NativeHydrologyPosition>,
                                                           std::span<NativeHydrologyInput>) {
        throw std::runtime_error("persisted outlet fixture unexpectedly rebuilt");
    };
    NativeHydrologyRouter router(SEED, store);
    constexpr int SEARCH_MARGIN = 12;
    const int64_t firstTargetWorldX =
        static_cast<int64_t>(targets[0] % static_cast<int32_t>(RASTER_EDGE) - 2) *
        NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t firstTargetWorldZ =
        static_cast<int64_t>(targets[0] / static_cast<int32_t>(RASTER_EDGE) - 2) *
        NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t secondTargetWorldX =
        static_cast<int64_t>(targets[1] % static_cast<int32_t>(RASTER_EDGE) - 2) *
        NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t secondTargetWorldZ =
        static_cast<int64_t>(targets[1] / static_cast<int32_t>(RASTER_EDGE) - 2) *
        NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t searchOriginX =
        std::min({sourceWorldX, firstTargetWorldX, secondTargetWorldX}) - SEARCH_MARGIN;
    const int64_t searchOriginZ =
        std::min({sourceWorldZ, firstTargetWorldZ, secondTargetWorldZ}) - SEARCH_MARGIN;
    const int searchWidth =
        static_cast<int>(std::max({sourceWorldX, firstTargetWorldX, secondTargetWorldX}) +
                         SEARCH_MARGIN - searchOriginX + 1);
    const int searchHeight =
        static_cast<int>(std::max({sourceWorldZ, firstTargetWorldZ, secondTargetWorldZ}) +
                         SEARCH_MARGIN - searchOriginZ + 1);
    std::vector<BasinSample> grid(static_cast<size_t>(searchWidth * searchHeight));
    router.sampleGrid(searchOriginX, searchOriginZ, 1, 1, searchWidth, searchHeight, forbiddenInput,
                      grid);

    const auto explicitFall = [](const BasinSample& sample) {
        return sample.waterfall &&
               sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
               sample.transitionOwnerId != 0;
    };
    std::map<uint64_t, ColumnPos> anchors;
    std::map<uint64_t, size_t> anchorCounts;
    size_t fallSamples = 0;
    for (int z = 0; z < searchHeight; ++z) {
        for (int x = 0; x < searchWidth; ++x) {
            const BasinSample& sample = grid[static_cast<size_t>(z * searchWidth + x)];
            if (!sample.waterfall)
                continue;
            ++fallSamples;
            REQUIRE(explicitFall(sample));
            REQUIRE(sample.waterfallTop == Catch::Approx(standingStage));
            REQUIRE(sample.waterfallBottom <= standingStage - 0.5);
            REQUIRE(sample.channelDistance <= sample.channelWidth * 0.5 + 1.0e-6);
            if (!sample.waterfallAnchor)
                continue;
            const ColumnPos position{searchOriginX + x, searchOriginZ + z};
            REQUIRE(world_coord::floorMod(position.x, NATIVE_HYDROLOGY_RASTER_SPACING) == 0);
            REQUIRE(world_coord::floorMod(position.z, NATIVE_HYDROLOGY_RASTER_SPACING) == 0);
            anchors.emplace(sample.transitionOwnerId, position);
            ++anchorCounts[sample.transitionOwnerId];
        }
    }
    CAPTURE(fallSamples, anchors.size(), anchorCounts.size());
    REQUIRE(fallSamples > 0);
    REQUIRE(anchors.size() == 2);
    REQUIRE(anchorCounts.size() == anchors.size());
    std::set<std::pair<int64_t, int64_t>> uniqueAnchorPositions;
    for (const auto& [owner, position] : anchors) {
        CAPTURE(owner, position.x, position.z);
        REQUIRE(anchorCounts[owner] == 1);
        REQUIRE(uniqueAnchorPositions.insert({position.x, position.z}).second);
        const BasinSample scalar = router.sample(position.x, position.z, forbiddenInput);
        REQUIRE(explicitFall(scalar));
        REQUIRE(scalar.waterfallAnchor);
        REQUIRE(scalar.transitionOwnerId == owner);
        std::array<NativeHydrologyTopologyCell, 1> topology{};
        const int64_t topologyX =
            world_coord::floorDiv(position.x,
                                  static_cast<int64_t>(NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE)) *
            NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
        const int64_t topologyZ =
            world_coord::floorDiv(position.z,
                                  static_cast<int64_t>(NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE)) *
            NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
        router.sampleTopologyGrid(topologyX, topologyZ, 1, 1, forbiddenInput, topology);
        REQUIRE(topology.front().waterfallPossible);
    }

    std::vector<BasinSamplePosition> positions;
    positions.reserve(grid.size());
    for (int z = 0; z < searchHeight; ++z) {
        for (int x = 0; x < searchWidth; ++x) {
            positions.push_back({.x = static_cast<double>(searchOriginX + x),
                                 .z = static_cast<double>(searchOriginZ + z)});
        }
    }
    std::vector<BasinSamplePosition> reversedPositions = positions;
    std::ranges::reverse(reversedPositions);
    std::vector<BasinSample> reversed(grid.size());
    router.samplePoints(reversedPositions, forbiddenInput, reversed);
    std::ranges::reverse(reversed);
    for (size_t index = 0; index < grid.size(); ++index) {
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(grid[index]));
        REQUIRE(reversed[index].transitionOwnerId == grid[index].transitionOwnerId);
        REQUIRE(reversed[index].waterfallAnchor == grid[index].waterfallAnchor);
    }

    router.clear();
    NativeHydrologyRouter warmRouter(SEED, store);
    std::vector<BasinSample> restored(grid.size());
    warmRouter.samplePoints(positions, forbiddenInput, restored);
    REQUIRE(warmRouter.cacheMetrics().persistedLoads == 1);
    for (size_t index = 0; index < grid.size(); ++index) {
        REQUIRE(nativeWaterHash(restored[index]) == nativeWaterHash(grid[index]));
        REQUIRE(restored[index].transitionOwnerId == grid[index].transitionOwnerId);
        REQUIRE(restored[index].waterfallAnchor == grid[index].waterfallAnchor);
    }

    const auto farRouter = std::make_shared<NativeHydrologyRouter>(SEED, store);
    const auto farGeometry = [](const BasinSample& sample) {
        FarTerrainGeometrySample result;
        result.waterBodyId = sample.waterBodyId;
        result.transitionOwnerId = sample.transitionOwnerId;
        result.terrainHeight = sample.surfaceElevation;
        result.waterSurface = sample.waterSurface;
        result.discharge = sample.discharge;
        result.sediment = sample.sediment;
        result.waterfallTop = sample.waterfallTop;
        result.waterfallBottom = sample.waterfallBottom;
        result.waterfallWidth = sample.waterfallWidth;
        result.flowX = sample.flowX;
        result.flowZ = sample.flowZ;
        result.transitionOwnerKind = sample.transitionOwnerKind;
        result.generatedFluidLevel = sample.generatedFluidLevel;
        result.ocean = sample.ocean;
        result.river = sample.river;
        result.lake = sample.lake;
        result.wetland = sample.wetland;
        result.waterfall = sample.waterfall;
        result.waterfallAnchor = sample.waterfallAnchor;
        result.delta = sample.delta;
        return result;
    };
    FarTerrainSource farSource;
    farSource.sample = [farRouter, forbiddenInput, farGeometry](int64_t x, int64_t z,
                                                                SurfaceFootprint) {
        const FarTerrainGeometrySample geometry =
            farGeometry(farRouter->sample(x, z, forbiddenInput));
        worldgen::surface_material::SurfaceMaterialPalette palette;
        palette.count = 1;
        palette.entries[0] = {.material = BlockType::STONE, .weight = 255};
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = geometry.terrainHeight,
            .footprintMaximumTerrainHeight = geometry.terrainHeight,
            .materialPalette = palette,
        };
    };
    farSource.canonicalWaterGrid = [farRouter, forbiddenInput, farGeometry](
                                       int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                       int sampleWidth, int sampleHeight, SurfaceFootprint,
                                       std::span<FarTerrainGeometrySample> output) {
        std::vector<BasinSample> samples(output.size());
        farRouter->sampleGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
                              forbiddenInput, samples);
        std::ranges::transform(samples, output.begin(), farGeometry);
    };
    farSource.cellBoundsGrid = [anchors, farRouter,
                                forbiddenInput](int64_t originX, int64_t originZ, int step,
                                                int cellWidth, int cellHeight, SurfaceFootprint,
                                                std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t cellX = originX + static_cast<int64_t>(x) * step;
                const int64_t cellZ = originZ + static_cast<int64_t>(z) * step;
                const BasinSample sample = farRouter->sample(cellX, cellZ, forbiddenInput);
                const bool ownsAnchor = std::ranges::any_of(anchors, [&](const auto& entry) {
                    const ColumnPos position = entry.second;
                    return position.x >= cellX && position.x < cellX + step &&
                           position.z >= cellZ && position.z < cellZ + step;
                });
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = sample.surfaceElevation,
                    .minimumTerrainHeight = sample.surfaceElevation,
                    .maximumTerrainHeight = sample.surfaceElevation,
                    .waterfallPossible = ownsAnchor,
                };
            }
        }
    };

    std::set<std::pair<int64_t, int64_t>> anchorOwnerTiles;
    for (const auto& [owner, position] : anchors) {
        static_cast<void>(owner);
        anchorOwnerTiles.insert(
            {world_coord::floorDiv(position.x, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
             world_coord::floorDiv(position.z, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE))});
    }
    for (const FarTerrainStep step :
         {FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        size_t waterfallQuads = 0;
        for (const auto [tileX, tileZ] : anchorOwnerTiles) {
            const FarTerrainKey key{tileX, tileZ, step};
            const auto mesh = FarTerrainMesher::build(key, farSource);
            waterfallQuads += mesh->waterfallQuadCount;
            const auto repeated = FarTerrainMesher::build(key, farSource);
            REQUIRE(repeated->deterministicHash == mesh->deterministicHash);
            REQUIRE(repeated->waterfallQuadCount == mesh->waterfallQuadCount);
        }
        CAPTURE(static_cast<int>(step), waterfallQuads, anchorOwnerTiles.size());
        REQUIRE(waterfallQuads >= anchors.size() * 5);
    }
}

TEST_CASE("V4 native channel contacts are graded or own an explicit fall",
          "[worldgen][hydrology][native-4][v4][curve][stage][waterfall][regression]") {
    constexpr int64_t ORIGIN_X = 992;
    constexpr int64_t ORIGIN_Z = 992;
    constexpr int EDGE = 65;
    constexpr uint64_t SEED = 0x5354'4147'4500'0004ULL;
    TempDir directory("native_hydrology_channel_contacts");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    std::array<BasinSample, EDGE * EDGE> forward{};
    NativeHydrologyRouter router(SEED, store);
    router.sampleGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, planarNativeInput(), forward);

    const auto indexOf = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    const auto explicitFall = [](const BasinSample& sample) {
        return sample.waterfall &&
               sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
               sample.transitionOwnerId != 0;
    };
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };

    size_t riverCount = 0;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const BasinSample& sample = forward[indexOf(x, z)];
            if (!sample.river)
                continue;
            ++riverCount;
            if (sample.waterfall) {
                REQUIRE(explicitFall(sample));
                REQUIRE(sample.waterfallTop >= sample.waterfallBottom + 0.5);
                REQUIRE(sample.generatedFluidLevel == 7);
            }
            const double elevationMeters = 130.0 - static_cast<double>(ORIGIN_X + x) * 0.004 -
                                           static_cast<double>(ORIGIN_Z + z) * 0.002;
            const double learnedHeight =
                learned::learnedElevationMetersToWorldHeight(elevationMeters);
            REQUIRE(sample.surfaceElevation <= learnedHeight + 1.0e-6);
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE)
                    continue;
                const BasinSample& adjacent = forward[indexOf(x + offsetX, z + offsetZ)];
                if (!adjacent.river)
                    continue;
                const double stageStep = std::abs(visibleStage(sample) - visibleStage(adjacent));
                CAPTURE(x, z, offsetX, offsetZ, sample.waterSurface, adjacent.waterSurface,
                        stageStep, sample.waterBodyId, adjacent.waterBodyId, sample.waterfall,
                        adjacent.waterfall);
                REQUIRE((stageStep <= 0.125001 || explicitFall(sample) || explicitFall(adjacent)));
            }

            const int flowOffsetX = std::abs(sample.flowX) >= std::abs(sample.flowZ)
                                        ? (sample.flowX < 0.0 ? -1 : 1)
                                        : 0;
            const int flowOffsetZ = flowOffsetX == 0 ? (sample.flowZ < 0.0 ? -1 : 1) : 0;
            if (x + flowOffsetX >= 0 && x + flowOffsetX < EDGE && z + flowOffsetZ >= 0 &&
                z + flowOffsetZ < EDGE) {
                const BasinSample& downstream = forward[indexOf(x + flowOffsetX, z + flowOffsetZ)];
                if (downstream.river && downstream.waterBodyId == sample.waterBodyId) {
                    CAPTURE(x, z, flowOffsetX, flowOffsetZ, sample.flowX, sample.flowZ,
                            sample.waterSurface, downstream.waterSurface, sample.waterfall,
                            downstream.waterfall);
                    REQUIRE((visibleStage(downstream) <= visibleStage(sample) + 0.125001 ||
                             explicitFall(sample) || explicitFall(downstream)));
                }
            }
        }
    }
    REQUIRE(riverCount > 100);

    std::vector<BasinSamplePosition> reversedPositions;
    reversedPositions.reserve(forward.size());
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x)
            reversedPositions.push_back(
                {.x = static_cast<double>(ORIGIN_X + x), .z = static_cast<double>(ORIGIN_Z + z)});
    }
    std::ranges::reverse(reversedPositions);
    router.clear();
    const NativeHydrologyCacheMetrics metricsBeforeReload = router.cacheMetrics();
    std::vector<BasinSample> reversed(forward.size());
    bool queriedInput = false;
    router.samplePoints(
        reversedPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted channel contact page unexpectedly rebuilt");
        },
        reversed);
    std::ranges::reverse(reversed);
    REQUIRE_FALSE(queriedInput);
    REQUIRE(router.cacheMetrics().persistedLoads > metricsBeforeReload.persistedLoads);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE(nativeWaterHash(reversed[index]) == nativeWaterHash(forward[index]));
        REQUIRE(std::bit_cast<uint64_t>(reversed[index].waterfallTop) ==
                std::bit_cast<uint64_t>(forward[index].waterfallTop));
        REQUIRE(std::bit_cast<uint64_t>(reversed[index].waterfallBottom) ==
                std::bit_cast<uint64_t>(forward[index].waterfallBottom));
        REQUIRE(reversed[index].waterfallAnchor == forward[index].waterfallAnchor);
    }
}

TEST_CASE("V4 ordinary stage tiles are signed, cross-tile, bounded, and restart-stable",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][negative]"
          "[persistence][regression]") {
    constexpr uint64_t SEED = 0x5354'4147'4554'494CULL;
    constexpr int64_t ORIGIN_X = -1'040;
    constexpr int64_t ORIGIN_Z = -1'040;
    constexpr int EDGE = 33;
    TempDir directory("native_hydrology_signed_stage_tiles");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    NativeHydrologyRouter router(SEED, store);
    std::array<BasinSample, EDGE * EDGE> grid{};
    router.sampleGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, planarNativeInput(), grid);
    const auto at = [&](int x, int z) -> const BasinSample& {
        return grid[static_cast<size_t>(z * EDGE + x)];
    };
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };
    const auto explicitFall = [](const BasinSample& sample) {
        return sample.waterfall &&
               sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
               sample.transitionOwnerId != 0;
    };

    size_t riverCount = 0;
    size_t crossTileContacts = 0;
    std::optional<size_t> scalarIndex;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const BasinSample& sample = at(x, z);
            if (!sample.river)
                continue;
            ++riverCount;
            if (!scalarIndex)
                scalarIndex = static_cast<size_t>(z * EDGE + x);
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE)
                    continue;
                const BasinSample& adjacent = at(x + offsetX, z + offsetZ);
                if (!adjacent.river || explicitFall(sample) || explicitFall(adjacent))
                    continue;
                const int64_t worldX = ORIGIN_X + x;
                const int64_t worldZ = ORIGIN_Z + z;
                if (world_coord::floorDiv(worldX, int64_t{32}) !=
                        world_coord::floorDiv(worldX + offsetX, int64_t{32}) ||
                    world_coord::floorDiv(worldZ, int64_t{32}) !=
                        world_coord::floorDiv(worldZ + offsetZ, int64_t{32})) {
                    ++crossTileContacts;
                }
                CAPTURE(worldX, worldZ, offsetX, offsetZ, sample.waterSurface,
                        adjacent.waterSurface, sample.waterBodyId, adjacent.waterBodyId);
                REQUIRE(std::abs(visibleStage(sample) - visibleStage(adjacent)) <= 0.125001);
            }
        }
    }
    CAPTURE(riverCount, crossTileContacts);
    REQUIRE(riverCount > 100);
    REQUIRE(crossTileContacts > 0);
    REQUIRE(scalarIndex);
    const int scalarX = static_cast<int>(*scalarIndex % EDGE);
    const int scalarZ = static_cast<int>(*scalarIndex / EDGE);
    const BasinSample scalar =
        router.sample(ORIGIN_X + scalarX, ORIGIN_Z + scalarZ, planarNativeInput());
    REQUIRE(nativeWaterHash(scalar) == nativeWaterHash(grid[*scalarIndex]));

    const NativeHydrologyCacheMetrics tileMetrics = router.cacheMetrics();
    CAPTURE(tileMetrics.ordinaryStageTileEntries, tileMetrics.ordinaryStageTileBytes,
            tileMetrics.ordinaryStageTilePeakPageBytes, tileMetrics.ordinaryStageTileHits,
            tileMetrics.ordinaryStageTileMisses, tileMetrics.ordinaryStageTileBuilds,
            tileMetrics.ordinaryStageTileFailures, tileMetrics.ordinaryStageTileBuildNanoseconds,
            tileMetrics.ordinaryStageTileExpandedBuilds);
    REQUIRE(tileMetrics.ordinaryStageTileBuilds >= 2);
    REQUIRE(tileMetrics.ordinaryStageTileHits > 0);
    REQUIRE(tileMetrics.ordinaryStageTileFailures == 0);
    REQUIRE(tileMetrics.ordinaryStageTileBuildNanoseconds > 0);
    REQUIRE(tileMetrics.ordinaryStageTilePeakPageBytes <=
            NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET);
    REQUIRE(tileMetrics.ordinaryStageTileBytes <=
            tileMetrics.entries * NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET);

    // Load the persisted owner without requesting an integer-column stage
    // tile, then release several same-key readers together. Exactly one
    // builder owns the expensive halo solve and every waiter observes its
    // immutable result.
    NativeHydrologyRouter concurrentRouter(SEED, store);
    std::atomic<bool> concurrentInputQueried = false;
    const NativeHydrologyInputFunction persistedOnly = [&](std::span<const NativeHydrologyPosition>,
                                                           std::span<NativeHydrologyInput>) {
        concurrentInputQueried.store(true, std::memory_order_relaxed);
        throw std::runtime_error("persisted stage-tile owner unexpectedly rebuilt");
    };
    NativeHydrologyRouter coarseRouter(SEED, store);
    std::array<BasinSample, 16> coarse{};
    coarseRouter.sampleGrid(ORIGIN_X, ORIGIN_Z, 32, 32, 4, 4, persistedOnly, coarse);
    REQUIRE(std::ranges::all_of(coarse, &BasinSample::valid));
    REQUIRE_FALSE(concurrentInputQueried.load(std::memory_order_relaxed));
    const NativeHydrologyCacheMetrics coarseMetrics = coarseRouter.cacheMetrics();
    REQUIRE(coarseMetrics.ordinaryStageCoarseGridSamples == coarse.size());
    REQUIRE(coarseMetrics.ordinaryStageTileBuilds == 0);

    std::vector<BasinSamplePosition> coarsePointPositions;
    coarsePointPositions.reserve(coarse.size());
    for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
            coarsePointPositions.push_back({.x = static_cast<double>(ORIGIN_X + x * 32),
                                            .z = static_cast<double>(ORIGIN_Z + z * 32)});
        }
    }
    std::ranges::reverse(coarsePointPositions);
    NativeHydrologyRouter coarsePointRouter(SEED, store);
    std::vector<BasinSample> coarsePoints(coarsePointPositions.size());
    coarsePointRouter.sampleCoarsePoints(coarsePointPositions, persistedOnly, coarsePoints);
    std::ranges::reverse(coarsePoints);
    for (size_t index = 0; index < coarse.size(); ++index)
        REQUIRE(nativeWaterHash(coarsePoints[index]) == nativeWaterHash(coarse[index]));
    const NativeHydrologyCacheMetrics coarsePointMetrics = coarsePointRouter.cacheMetrics();
    REQUIRE(coarsePointMetrics.ordinaryStageCoarseGridSamples == coarsePoints.size());
    REQUIRE(coarsePointMetrics.ordinaryStageTileBuilds == 0);

    const BasinSample warmOwner = concurrentRouter.sample(ORIGIN_X + scalarX + 0.25,
                                                          ORIGIN_Z + scalarZ + 0.25, persistedOnly);
    REQUIRE(warmOwner.valid);
    REQUIRE_FALSE(concurrentInputQueried.load(std::memory_order_relaxed));
    REQUIRE(concurrentRouter.cacheMetrics().ordinaryStageTileBuilds == 0);
    constexpr size_t CONCURRENT_READERS = 8;
    std::promise<void> releaseReaders;
    const std::shared_future<void> readerGate = releaseReaders.get_future().share();
    std::atomic<size_t> readersReady = 0;
    std::array<std::future<BasinSample>, CONCURRENT_READERS> concurrentReads;
    for (auto& read : concurrentReads) {
        read = std::async(std::launch::async, [&] {
            readersReady.fetch_add(1, std::memory_order_release);
            readerGate.wait();
            return concurrentRouter.sample(ORIGIN_X + scalarX, ORIGIN_Z + scalarZ, persistedOnly);
        });
    }
    while (readersReady.load(std::memory_order_acquire) != CONCURRENT_READERS)
        std::this_thread::yield();
    releaseReaders.set_value();
    for (auto& read : concurrentReads)
        REQUIRE(nativeWaterHash(read.get()) == nativeWaterHash(grid[*scalarIndex]));
    REQUIRE_FALSE(concurrentInputQueried.load(std::memory_order_relaxed));
    const NativeHydrologyCacheMetrics concurrentMetrics = concurrentRouter.cacheMetrics();
    CAPTURE(concurrentMetrics.ordinaryStageTileBuilds, concurrentMetrics.ordinaryStageTileMisses,
            concurrentMetrics.ordinaryStageTileBuildWaits);
    REQUIRE(concurrentMetrics.ordinaryStageTileBuilds == 1);
    REQUIRE(concurrentMetrics.ordinaryStageTileBuildWaits > 0);
    REQUIRE(concurrentMetrics.ordinaryStageCoarseGridSamples == 0);

    std::vector<BasinSamplePosition> reversedPositions;
    reversedPositions.reserve(grid.size());
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x)
            reversedPositions.push_back(
                {.x = static_cast<double>(ORIGIN_X + x), .z = static_cast<double>(ORIGIN_Z + z)});
    }
    std::ranges::reverse(reversedPositions);
    router.clear();
    std::vector<BasinSample> restored(grid.size());
    bool queriedInput = false;
    router.samplePoints(
        reversedPositions,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedInput = true;
            throw std::runtime_error("persisted signed stage-tile page unexpectedly rebuilt");
        },
        restored);
    std::ranges::reverse(restored);
    REQUIRE_FALSE(queriedInput);
    for (size_t index = 0; index < grid.size(); ++index)
        REQUIRE(nativeWaterHash(restored[index]) == nativeWaterHash(grid[index]));
}

TEST_CASE("V4 ordinary stage halos resolve signed native-page owners",
          "[worldgen][hydrology][native-4][v4][curve][stage][tile][page-edge]"
          "[negative][regression]") {
    constexpr uint64_t SEED = 0x4841'4C4F'5041'4745ULL;
    constexpr int ROWS_PER_BOUNDARY = 24;
    constexpr std::array<int64_t, 2> BOUNDARIES{-2'048, 2'048};
    std::vector<BasinSamplePosition> positions;
    positions.reserve(BOUNDARIES.size() * ROWS_PER_BOUNDARY * 5);
    uint64_t state = 0x9E37'79B9'7F4A'7C15ULL;
    for (const int64_t boundary : BOUNDARIES) {
        for (int row = 0; row < ROWS_PER_BOUNDARY; ++row) {
            state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
            const int64_t z = 256 + static_cast<int64_t>((state >> 16U) % 1'536U);
            for (int offset = -2; offset <= 2; ++offset)
                positions.push_back(
                    {.x = static_cast<double>(boundary + offset), .z = static_cast<double>(z)});
        }
    }

    NativeHydrologyRouter router(SEED);
    std::vector<BasinSample> samples(positions.size());
    router.samplePoints(positions, planarNativeInput(), samples);
    const auto visibleStage = [](const BasinSample& sample) {
        return std::round(sample.waterSurface * 8.0) / 8.0;
    };
    const auto explicitFall = [](const BasinSample& sample) {
        return sample.waterfall &&
               sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
               sample.transitionOwnerId != 0;
    };
    size_t ordinaryContacts = 0;
    for (size_t row = 0; row < BOUNDARIES.size() * ROWS_PER_BOUNDARY; ++row) {
        const size_t begin = row * 5;
        for (size_t offset = 0; offset + 1 < 5; ++offset) {
            const BasinSample& sample = samples[begin + offset];
            const BasinSample& adjacent = samples[begin + offset + 1];
            if (!sample.river || !adjacent.river || explicitFall(sample) ||
                explicitFall(adjacent)) {
                continue;
            }
            ++ordinaryContacts;
            CAPTURE(row, offset, positions[begin + offset].x, positions[begin + offset].z,
                    sample.waterSurface, adjacent.waterSurface, sample.waterBodyId,
                    adjacent.waterBodyId);
            REQUIRE(std::abs(visibleStage(sample) - visibleStage(adjacent)) <= 0.125001);
        }
    }
    CAPTURE(ordinaryContacts);
    REQUIRE(ordinaryContacts > ROWS_PER_BOUNDARY);
    const NativeHydrologyCacheMetrics metrics = router.cacheMetrics();
    REQUIRE(metrics.ordinaryStageTileBuilds > 0);
    REQUIRE(metrics.ordinaryStageTileFailures == 0);
    REQUIRE(metrics.ordinaryStageTilePeakPageBytes <=
            NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET);
}

TEST_CASE("V4 native pages restart from their immutable RYHY payload",
          "[worldgen][hydrology][native-4][v4][persistence][restart]") {
    constexpr uint64_t SEED = 0xBADC'0FFE'E0DD'F00DULL;
    STATIC_REQUIRE(NATIVE_HYDROLOGY_PAYLOAD_SCHEMA_VERSION == 6);
    TempDir directory("native_hydrology_restart");
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    const NativeHydrologyInputFunction input = steppedSlopeNativeInput();
    BasinSample expected;
    {
        NativeHydrologyRouter writer(SEED, store);
        expected = writer.sample(1'026.0, 1'026.0, input);
        REQUIRE(expected.valid);
        REQUIRE(expected.terrainSlope > 0.3);
        REQUIRE(expected.terrainSlope < 0.4);
        std::array<BasinSample, 1> lattice{};
        writer.sampleGrid(1'026, 1'026, NATIVE_HYDROLOGY_RASTER_SPACING,
                          NATIVE_HYDROLOGY_RASTER_SPACING, 1, 1, input, lattice);
        REQUIRE(std::bit_cast<uint64_t>(lattice.front().terrainSlope) ==
                std::bit_cast<uint64_t>(expected.terrainSlope));
        const NativeHydrologyCacheMetrics metrics = writer.cacheMetrics();
        REQUIRE(metrics.persistedWrites == 1);
        REQUIRE(metrics.persistedLoads == 0);
    }
    auto payload = store->load({0, 0});
    REQUIRE(payload.isReady());
    REQUIRE(payload.value()->size() > 52);
    REQUIRE(payload.value()->size() < NATIVE_HYDROLOGY_MAX_PAGE_BYTES);
    REQUIRE(std::ranges::equal(std::span(*payload.value()).first(4),
                               std::array<uint8_t, 4>{'N', 'H', '4', 'P'}));

    bool queriedLearnedInput = false;
    const NativeHydrologyInputFunction forbiddenInput =
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            queriedLearnedInput = true;
            throw std::runtime_error("persisted native page unexpectedly queried learned input");
        };
    NativeHydrologyRouter reader(SEED, store);
    const auto warmStart = std::chrono::steady_clock::now();
    const BasinSample restored = reader.sample(1'026.0, 1'026.0, forbiddenInput);
    const uint64_t observedWarmNanoseconds =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - warmStart)
                                  .count());
    REQUIRE_FALSE(queriedLearnedInput);
    REQUIRE(restored.valid);
    REQUIRE(restored.waterBodyId == expected.waterBodyId);
    REQUIRE(restored.waterSurface == expected.waterSurface);
    REQUIRE(restored.surfaceElevation == expected.surfaceElevation);
    REQUIRE(std::bit_cast<uint64_t>(restored.terrainSlope) ==
            std::bit_cast<uint64_t>(expected.terrainSlope));
    REQUIRE(restored.discharge == expected.discharge);
    REQUIRE(restored.channelDistance == expected.channelDistance);
    const NativeHydrologyCacheMetrics metrics = reader.cacheMetrics();
    REQUIRE(metrics.persistedLoads == 1);
    REQUIRE(metrics.persistedWrites == 0);
    REQUIRE(metrics.lastPersistedPayloadBytes == payload.value()->size());
    REQUIRE(metrics.lastWarmLoadNanoseconds > 0);
    REQUIRE(metrics.lastWarmLoadNanoseconds <= observedWarmNanoseconds);
}

TEST_CASE("V4 preview and final hydrology payloads remain request-order isolated",
          "[worldgen][hydrology][native-4][v4][persistence][quality][order]") {
    constexpr uint64_t SEED = 0x0A11'7EED'0004ULL;
    const auto offsetInput = [](double offset) {
        return [offset](std::span<const NativeHydrologyPosition> positions,
                        std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                output[index] = climateInput(130.0 + offset - positions[index].x * 0.004 -
                                             positions[index].z * 0.002);
            }
        };
    };
    struct QualitySamples {
        BasinSample preview;
        BasinSample final;
    };
    const auto run = [&](const std::filesystem::path& root, bool reverse) {
        const auto previewStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
            root / "preview", nativeHydrologyIdentity(SEED),
            worldgen::learned::AuthorityQuality::PREVIEW);
        const auto finalStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
            root / "final", nativeHydrologyIdentity(SEED),
            worldgen::learned::AuthorityQuality::FINAL);
        NativeHydrologyRouter previewRouter(SEED, previewStore);
        NativeHydrologyRouter finalRouter(SEED, finalStore);
        QualitySamples samples;
        if (reverse) {
            samples.final = finalRouter.sample(1'024.0, 1'024.0, offsetInput(18.0));
            samples.preview = previewRouter.sample(1'024.0, 1'024.0, offsetInput(0.0));
        } else {
            samples.preview = previewRouter.sample(1'024.0, 1'024.0, offsetInput(0.0));
            samples.final = finalRouter.sample(1'024.0, 1'024.0, offsetInput(18.0));
        }
        REQUIRE(std::filesystem::exists(previewStore->pagePath({0, 0})));
        REQUIRE(std::filesystem::exists(finalStore->pagePath({0, 0})));
        return samples;
    };

    TempDir forwardDirectory("native_hydrology_quality_forward");
    TempDir reverseDirectory("native_hydrology_quality_reverse");
    const QualitySamples forward = run(forwardDirectory.path(), false);
    const QualitySamples reverse = run(reverseDirectory.path(), true);
    REQUIRE(forward.preview.surfaceElevation == reverse.preview.surfaceElevation);
    REQUIRE(forward.preview.waterSurface == reverse.preview.waterSurface);
    REQUIRE(forward.preview.waterBodyId == reverse.preview.waterBodyId);
    REQUIRE(forward.final.surfaceElevation == reverse.final.surfaceElevation);
    REQUIRE(forward.final.waterSurface == reverse.final.waterSurface);
    REQUIRE(forward.final.waterBodyId == reverse.final.waterBodyId);
    REQUIRE(forward.preview.waterBodyId == forward.final.waterBodyId);
    REQUIRE(forward.preview.surfaceElevation != forward.final.surfaceElevation);
}

TEST_CASE("V4 preview and final one-component lakes retain coarse anchor identities",
          "[worldgen][hydrology][native-4][v4][quality][identity][lake]") {
    constexpr uint64_t SEED = 0x1D3A'71A5'0004ULL;
    const auto bowl = [](double centerX, double depth) {
        return [centerX, depth](std::span<const NativeHydrologyPosition> positions,
                                std::span<NativeHydrologyInput> output) {
            for (size_t index = 0; index < positions.size(); ++index) {
                const double radius =
                    std::hypot(positions[index].x - centerX, positions[index].z - 1'024.0);
                output[index] = climateInput(100.0 - depth + std::min(depth, radius * 0.035));
            }
        };
    };
    const auto identities = std::make_shared<NativeHydrologyIdentityRegistry>(SEED);
    NativeHydrologyRouter preview(SEED, nullptr, identities);
    NativeHydrologyRouter final(SEED, nullptr, identities);
    constexpr std::array<BasinSamplePosition, 2> positions{{
        {1'108.0, 1'024.0},
        {1'112.0, 1'024.0},
    }};
    std::array<BasinSample, positions.size()> previewSamples{};
    std::array<BasinSample, positions.size()> finalSamples{};
    preview.samplePoints(positions, bowl(1'100.0, 9.0), previewSamples);
    std::array<BasinSamplePosition, positions.size()> reversed = positions;
    std::ranges::reverse(reversed);
    final.samplePoints(reversed, bowl(1'120.0, 13.0), finalSamples);
    std::ranges::reverse(finalSamples);
    for (size_t index = 0; index < positions.size(); ++index) {
        REQUIRE(previewSamples[index].lake);
        REQUIRE(finalSamples[index].lake);
        REQUIRE(previewSamples[index].waterBodyId != NO_WATER_BODY);
        REQUIRE(previewSamples[index].waterBodyId == finalSamples[index].waterBodyId);
    }
    // This proves only the ordinary one-component coarse-anchor behavior. The
    // bounded opposing-edge hierarchy covers a cross-page component, while a
    // quality-changing global topology still needs a higher-level spill graph.
    REQUIRE(previewSamples[0].waterBodyId == previewSamples[1].waterBodyId);
    REQUIRE(finalSamples[0].waterBodyId == finalSamples[1].waterBodyId);
}

TEST_CASE("V4 native rebuilds an outer-valid corrupt RYHY payload without a fallback",
          "[worldgen][hydrology][native-4][v4][persistence][corruption][repair]") {
    constexpr uint64_t SEED = 0xC077'B10B'0001ULL;
    TempDir directory("native_hydrology_inner_corruption");
    const auto validStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "valid-hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    constexpr worldgen::hydrology::HydrologyPageCoordinate coordinate{0, 0};
    {
        NativeHydrologyRouter writer(SEED, validStore);
        REQUIRE(writer.sample(1'024.0, 1'024.0, planarNativeInput()).valid);
        REQUIRE(writer.cacheMetrics().persistedWrites == 1);
    }
    auto validPayload = validStore->load(coordinate);
    REQUIRE(validPayload.isReady());
    std::vector<uint8_t> invalidPayload = *validPayload.value();
    constexpr size_t RASTER_EDGE = 2'048 / 4 + 1 + 4;
    constexpr size_t RASTER_CELLS = RASTER_EDGE * RASTER_EDGE;
    // Thirteen float fields, two int32 receiver fields, one float weight,
    // one uint64 body ID, stream order, and flags precede the schema-6 frozen
    // two-bit waterfall-branch mask.
    constexpr size_t WATERFALL_BRANCH_MASK_OFFSET = 52 + 74 * RASTER_CELLS;
    REQUIRE(invalidPayload.size() > WATERFALL_BRANCH_MASK_OFFSET);
    invalidPayload[WATERFALL_BRANCH_MASK_OFFSET] = 0x04U;
    const auto store = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        std::filesystem::path(directory.path()) / "corrupt-hydrology-authority-v1",
        nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    REQUIRE(store->write(coordinate, invalidPayload).isReady());

    bool queriedLearnedInput = false;
    const NativeHydrologyInputFunction observedInput =
        [&](std::span<const NativeHydrologyPosition> positions,
            std::span<NativeHydrologyInput> output) {
            queriedLearnedInput = true;
            planarNativeInput()(positions, output);
        };
    NativeHydrologyRouter repaired(SEED, store);
    const BasinSample rebuilt = repaired.sample(1'024.0, 1'024.0, observedInput);
    REQUIRE(queriedLearnedInput);
    REQUIRE(rebuilt.valid);
    const NativeHydrologyCacheMetrics repairMetrics = repaired.cacheMetrics();
    REQUIRE(repairMetrics.persistedLoads == 0);
    REQUIRE(repairMetrics.persistedWrites == 1);
    REQUIRE(repairMetrics.persistedRepairs == 1);
    auto repairedPayload = store->load(coordinate);
    REQUIRE(repairedPayload.isReady());
    REQUIRE(repairedPayload.value()->size() > 52);
    REQUIRE(std::ranges::equal(std::span(*repairedPayload.value()).first(4),
                               std::array<uint8_t, 4>{'N', 'H', '4', 'P'}));

    bool warmInputQueried = false;
    NativeHydrologyRouter warmReader(SEED, store);
    const BasinSample restored = warmReader.sample(
        1'024.0, 1'024.0,
        [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
            warmInputQueried = true;
            throw std::runtime_error("repaired native hydrology unexpectedly rebuilt");
        });
    REQUIRE_FALSE(warmInputQueried);
    REQUIRE(nativeWaterHash(restored) == nativeWaterHash(rebuilt));
    REQUIRE(warmReader.cacheMetrics().persistedLoads == 1);
}

TEST_CASE("V4 native persistence failures other than corruption stay fail-closed",
          "[worldgen][hydrology][native-4][v4][persistence][fingerprint]") {
    constexpr uint64_t SEED = 0xC077'B10B'0002ULL;
    TempDir directory("native_hydrology_incompatible_persistence");
    const std::filesystem::path root =
        std::filesystem::path(directory.path()) / "hydrology-authority-v1";
    const auto foreignStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        root, nativeHydrologyIdentity(SEED + 1), worldgen::learned::AuthorityQuality::FINAL);
    constexpr worldgen::hydrology::HydrologyPageCoordinate coordinate{0, 0};
    const std::array<uint8_t, 8> opaquePayload{'N', 'H', '4', 'P', 0, 0, 0, 0};
    REQUIRE(foreignStore->write(coordinate, opaquePayload).isReady());

    const auto currentStore = std::make_shared<worldgen::hydrology::HydrologyAuthorityStore>(
        root, nativeHydrologyIdentity(SEED), worldgen::learned::AuthorityQuality::FINAL);
    bool queriedLearnedInput = false;
    NativeHydrologyRouter router(SEED, currentStore);
    try {
        static_cast<void>(router.sample(
            1'024.0, 1'024.0,
            [&](std::span<const NativeHydrologyPosition>, std::span<NativeHydrologyInput>) {
                queriedLearnedInput = true;
                return;
            }));
        FAIL("Incompatible native hydrology persistence unexpectedly rebuilt");
    } catch (const worldgen::learned::GenerationFailureException& failure) {
        REQUIRE(failure.status() == worldgen::learned::AuthorityStatus::FAILED);
        REQUIRE(failure.failure().code ==
                worldgen::learned::GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    }
    REQUIRE_FALSE(queriedLearnedInput);
    REQUIRE(router.cacheMetrics().persistedRepairs == 0);
}

TEST_CASE("Native routing leaves the v3 16-block basin authority unchanged",
          "[worldgen][hydrology][native-4][v3][regression]") {
    STATIC_REQUIRE(BASIN_RASTER_SPACING == 16.0);
    STATIC_REQUIRE(NATIVE_HYDROLOGY_RASTER_SPACING == 4);
    const auto elevation = [](double x, double z) {
        return 118.0 - x * 0.002 - z * 0.001 + std::sin(x / 180.0) * 3.0 +
               std::cos(z / 220.0) * 2.0;
    };
    const auto rainfall = [](double, double, double) { return 1'100.0; };
    const auto resistance = [](double, double) { return 0.55; };
    BasinSolver legacy(42);
    const BasinSample expected = legacy.sample(773.0, 911.0, elevation, rainfall, resistance);

    NativeHydrologyRouter native(42);
    REQUIRE(native.sample(773.0, 911.0, planarNativeInput()).valid);
    legacy.clear();
    const BasinSample rebuilt = legacy.sample(773.0, 911.0, elevation, rainfall, resistance);

    REQUIRE(rebuilt.valid == expected.valid);
    REQUIRE(rebuilt.surfaceElevation == expected.surfaceElevation);
    REQUIRE(rebuilt.waterSurface == expected.waterSurface);
    REQUIRE(rebuilt.discharge == expected.discharge);
    REQUIRE(rebuilt.erosionDepth == expected.erosionDepth);
    REQUIRE(rebuilt.lakeDepth == expected.lakeDepth);
    REQUIRE(rebuilt.waterBodyId == expected.waterBodyId);
    REQUIRE(rebuilt.outlet == expected.outlet);
    REQUIRE(rebuilt.river == expected.river);
    REQUIRE(rebuilt.lake == expected.lake);
    REQUIRE(rebuilt.ocean == expected.ocean);
}

TEST_CASE("V4 channel projection preserves the unpruned canonical sample hash",
          "[worldgen][hydrology][native-4][v4][curve][projection][equivalence]") {
    NativeHydrologyRouter router(0x5EED'CAFE'0102'0304ULL);
    constexpr int EDGE = 49;
    std::array<BasinSample, EDGE * EDGE> samples{};
    router.sampleGrid(928, 928, 2, 2, EDGE, EDGE, planarNativeInput(), samples);
    uint64_t hash = 1'469'598'103'934'665'603ULL;
    const auto combine = [&](uint64_t value) {
        hash ^= value;
        hash *= 1'099'511'628'211ULL;
    };
    for (const BasinSample& sample : samples) {
        combine(sample.waterBodyId);
        combine(std::bit_cast<uint64_t>(sample.channelDistance));
        combine(std::bit_cast<uint64_t>(sample.channelWidth));
        combine(std::bit_cast<uint64_t>(sample.channelDepth));
        combine(std::bit_cast<uint64_t>(sample.waterSurface));
        combine(std::bit_cast<uint64_t>(sample.flowX));
        combine(std::bit_cast<uint64_t>(sample.flowZ));
        combine(static_cast<uint64_t>(sample.river));
        combine(static_cast<uint64_t>(sample.waterfall));
    }
    // Captured after receiver branches without a canonical wet target were
    // removed, ordinary spline projection became reach-scoped, and explicit
    // fall ownership was frozen per receiver branch. This is intentionally a
    // dense, sub-raster grid so it exercises river selection, stage projection,
    // and dry samples that still retain a finite canonical channel distance.
    REQUIRE(hash == 5'543'624'156'635'433'171ULL);
}

} // namespace

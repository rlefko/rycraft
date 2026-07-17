#include <catch2/catch_test_macros.hpp>

#include "world/column_plan.hpp"

namespace {

worldgen::SurfaceSample drySubstrateSample(bool lakeBank) {
    worldgen::SurfaceSample result;
    result.terrainHeight = 40.0;
    result.waterSurface = 72.0;
    result.hydrology.surfaceElevation = 40.0;
    result.hydrology.waterSurface = 72.0;
    result.hydrology.channelDistance = 8.0;
    result.hydrology.channelWidth = 8.0;
    result.hydrology.streamOrder = 2;
    result.hydrology.lakeBank = lakeBank;
    result.hydrology.lakeBankInfluence = lakeBank ? 1.0 : 0.0;
    result.hydrology.lakeBankTarget = lakeBank ? 72.0 : 0.0;
    result.hydrology.lakeShoreDistance = lakeBank ? -1.0 : -1.0e9;
    result.hydrology.shoreWaterSurface = lakeBank ? 72.0 : 0.0;
    return result;
}

ColumnPlan drySubstratePlan(bool lakeBank) {
    return ColumnPlan(
        {0, 0}, [=](int64_t, int64_t) { return drySubstrateSample(lakeBank); },
        [](int64_t, int64_t) { return 40.0; },
        [](const ColumnPlan&) {
            ColumnPlanSurfaceGrid result{};
            result.fill(39);
            return result;
        });
}

} // namespace

TEST_CASE("Column plans never raise dry substrate to sampled water stages",
          "[worldgen][column-plan][hydrology][bank][regression]") {
    SECTION("channel edge metadata") {
        const ColumnPlan plan = drySubstratePlan(false);
        const worldgen::SurfaceSample sample = plan.sample(4, 4);
        REQUIRE_FALSE(sample.hydrology.ocean);
        REQUIRE_FALSE(sample.hydrology.river);
        REQUIRE_FALSE(sample.hydrology.lake);
        REQUIRE(sample.terrainHeight == 40.0);
        REQUIRE(sample.hydrology.surfaceElevation == 40.0);
    }

    SECTION("shoreline bank metadata") {
        const ColumnPlan plan = drySubstratePlan(true);
        const worldgen::SurfaceSample sample = plan.sample(4, 4);
        REQUIRE_FALSE(sample.hydrology.ocean);
        REQUIRE_FALSE(sample.hydrology.river);
        REQUIRE_FALSE(sample.hydrology.lake);
        REQUIRE(sample.hydrology.lakeBank);
        REQUIRE(sample.hydrology.lakeBankInfluence == 1.0);
        REQUIRE(sample.terrainHeight == 40.0);
        REQUIRE(sample.hydrology.surfaceElevation == 40.0);
        REQUIRE(sample.hydrology.lakeBankTarget == 40.0);
    }
}

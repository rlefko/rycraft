#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "entity/spawner.hpp"
#include "world/world.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Wild fauna scores are independent of biome switches", "[fauna][habitat]") {
    HabitatSample forest;
    forest.biome = Biome::FOREST;
    forest.temperature = 0.58f;
    forest.moisture = 0.64f;
    forest.food = 0.73f;
    forest.cover = 0.48f;
    forest.fertility = 0.69f;
    forest.slope = 0.18f;
    forest.riverSize = 0.35f;
    forest.waterAccess = 0.62f;
    forest.waterDepth = 5;
    forest.nearWater = true;

    HabitatSample barren = forest;
    barren.biome = Biome::VOLCANIC_BARREN;

    for (size_t index = 0; index < ENTITY_TYPE_COUNT; ++index) {
        const EntityType type = static_cast<EntityType>(index);
        REQUIRE_THAT(Spawner::getHabitatScore(type, forest),
                     WithinAbs(Spawner::getHabitatScore(type, barren), 0.000001f));
    }
}

TEST_CASE("Appended fauna respond to their continuous habitat traits", "[fauna][habitat]") {
    HabitatSample forestEdge;
    forestEdge.temperature = 0.55f;
    forestEdge.moisture = 0.72f;
    forestEdge.food = 0.82f;
    forestEdge.fertility = 0.78f;
    forestEdge.cover = 0.55f;
    forestEdge.slope = 0.12f;
    forestEdge.waterAccess = 0.65f;
    forestEdge.riverSize = 0.35f;

    HabitatSample denseForest = forestEdge;
    denseForest.cover = 1.0f;
    REQUIRE(Spawner::getHabitatScore(EntityType::DEER, forestEdge) >
            Spawner::getHabitatScore(EntityType::DEER, denseForest));

    HabitatSample alpine = forestEdge;
    alpine.temperature = 0.30f;
    alpine.cover = 0.08f;
    alpine.slope = 0.92f;
    alpine.surfaceY = 142;
    REQUIRE(Spawner::getHabitatScore(EntityType::GOAT, alpine) >
            Spawner::getHabitatScore(EntityType::GOAT, forestEdge));

    HabitatSample openSteppe = forestEdge;
    openSteppe.moisture = 0.34f;
    openSteppe.cover = 0.08f;
    openSteppe.slope = 0.10f;
    REQUIRE(Spawner::getHabitatScore(EntityType::RABBIT, openSteppe) >
            Spawner::getHabitatScore(EntityType::RABBIT, denseForest));

    HabitatSample wetMargin = forestEdge;
    wetMargin.temperature = 0.78f;
    wetMargin.moisture = 0.95f;
    wetMargin.waterAccess = 1.0f;
    REQUIRE(Spawner::getHabitatScore(EntityType::FROG, wetMargin) > 0.60f);
    wetMargin.waterAccess = 0.0f;
    wetMargin.nearWater = false;
    REQUIRE(Spawner::getHabitatScore(EntityType::FROG, wetMargin) == 0.0f);
}

TEST_CASE("Fish require deep generated surface water", "[fauna][habitat][fish]") {
    HabitatSample water;
    water.temperature = 0.62f;
    water.moisture = 0.9f;
    water.fertility = 0.7f;
    water.waterAccess = 1.0f;
    water.riverSize = 0.6f;
    water.waterDepth = 5;
    water.generatedWaterBody = true;
    water.generatedRiver = true;
    REQUIRE(Spawner::getHabitatScore(EntityType::FISH, water) > 0.5f);

    water.generatedWaterBody = false;
    water.generatedRiver = false;
    REQUIRE(Spawner::getHabitatScore(EntityType::FISH, water) == 0.0f);

    water.generatedWaterBody = true;
    water.generatedLake = true;
    water.waterDepth = 1;
    REQUIRE(Spawner::getHabitatScore(EntityType::FISH, water) == 0.0f);
}

TEST_CASE("Habitat lookup never constructs a cold column plan", "[fauna][habitat][streaming]") {
    World world(42, 4);
    Spawner spawner(world);

    REQUIRE(world.cachedColumnPlanCount() == 0);
    REQUIRE_FALSE(spawner.findHabitatSample(12, -19).has_value());
    REQUIRE(world.cachedColumnPlanCount() == 0);

    const ColumnPos column{Chunk::worldToChunk(12), Chunk::worldToChunk(-19)};
    const auto warmed = world.generator().getColumnPlan(column);
    REQUIRE(warmed != nullptr);
    const size_t warmPlanCount = world.cachedColumnPlanCount();
    REQUIRE(warmPlanCount == 1);

    REQUIRE(spawner.findHabitatSample(12, -19).has_value());
    REQUIRE(world.cachedColumnPlanCount() == warmPlanCount);
}

TEST_CASE("Unloaded territories remain eligible for reevaluation",
          "[fauna][territory][streaming]") {
    World world(42, 4);
    Spawner spawner(world);
    const Vec2 anchor = Spawner::getTerritoryAnchor(world.getSeed(), 0, 0);
    const Vec3 player{anchor.x, static_cast<float>(SEA_LEVEL), anchor.y};

    spawner.updatePopulation(player);
    REQUIRE(spawner.populatedTerritoryCount() == 0);
    REQUIRE(spawner.getEntities().empty());

    world.sampleSurface(static_cast<int64_t>(anchor.x), static_cast<int64_t>(anchor.y));
    for (int tick = 0; tick < Spawner::POPULATION_UPDATE_TICKS; ++tick) {
        spawner.updatePopulation(player);
    }
    REQUIRE(spawner.populatedTerritoryCount() == 0);
    REQUIRE(spawner.getEntities().empty());
}

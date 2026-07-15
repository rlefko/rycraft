#include <catch2/catch_test_macros.hpp>
#include <render/mesh_scheduler.hpp>
#include <world/column_plan.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

TEST_CASE("Exact mesh scheduling shares one bounded result budget",
          "[render][scheduler][concurrency]") {
    World world(42, 4);
    MeshScheduler scheduler(world, 0);

    std::vector<MeshResult> consumer(1);
    scheduler.drainCompleted(consumer);
    REQUIRE(scheduler.stats().consumerPending == 1);

    for (size_t index = 0; index < MeshScheduler::MAX_INFLIGHT_MESH - 1; ++index) {
        REQUIRE(scheduler.enqueue({static_cast<int64_t>(index), 0, 0}));
    }
    REQUIRE_FALSE(scheduler.enqueue({1000, 0, 0}));

    MeshSchedulerStats full = scheduler.stats();
    REQUIRE(full.schedulerOwned + full.consumerPending == MeshScheduler::MAX_INFLIGHT_MESH);
    REQUIRE(full.highWater == MeshScheduler::MAX_INFLIGHT_MESH);

    consumer.clear();
    scheduler.acknowledgeConsumerPending(consumer.size());
    REQUIRE(scheduler.stats().consumerPending == 0);
    REQUIRE(scheduler.enqueue({1000, 0, 0}));
    REQUIRE_FALSE(scheduler.enqueue({1001, 0, 0}));

    scheduler.shutdown();
    REQUIRE(scheduler.stats().schedulerOwned == 0);
}

TEST_CASE("Exact mesh completion coalesces duplicate cube revisions",
          "[render][scheduler][concurrency]") {
    World world(42, 4);
    MeshScheduler scheduler(world, 1);
    constexpr ChunkPos missing{900, 4, -900};
    constexpr size_t REQUESTS = 16;

    for (size_t request = 0; request < REQUESTS; ++request)
        REQUIRE(scheduler.enqueue(missing));

    MeshSchedulerStats stats;
    for (int attempt = 0; attempt < 500; ++attempt) {
        stats = scheduler.stats();
        if (stats.schedulerOwned == 1 && stats.completed == 1)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(stats.schedulerOwned == 1);
    REQUIRE(stats.completed == 1);
    REQUIRE(stats.coalesced + stats.droppedStale == REQUESTS - 1);

    std::vector<MeshResult> results;
    scheduler.drainCompleted(results);
    REQUIRE(results.size() == 1);
    REQUIRE(results.front().pos == missing);
    REQUIRE_FALSE(results.front().snapshotOk);
    REQUIRE(scheduler.stats().schedulerOwned == 0);
}

TEST_CASE("Mesh sky cutoffs use immutable plans across negative chunk boundaries",
          "[world][render][snapshot][coordinates]") {
    World world(4242, 4);
    constexpr ChunkPos center{-1, 4, -1};
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }

    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing(center, snapshot));
    for (int z : {-1, 0, 15, 16}) {
        const int64_t worldZ = center.z * CHUNK_EDGE + z;
        for (int x : {-1, 0, 15, 16}) {
            const int64_t worldX = center.x * CHUNK_EDGE + x;
            const ColumnPos column{Chunk::worldToChunk(worldX), Chunk::worldToChunk(worldZ)};
            const auto plan = world.generator().findColumnPlan(column);
            REQUIRE(plan);
            const auto sample =
                plan->sample(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ));
            REQUIRE(std::isfinite(sample.terrainHeight));
            REQUIRE(snapshot.skyCutoffAt(x, z) ==
                    plan->surfaceY(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ)) + 1);
        }
    }
}

TEST_CASE("World publishes mesh candidates with the loaded cube snapshot",
          "[world][render][snapshot][concurrency]") {
    World world(42, 4);
    const auto initial = world.getMeshCandidateSnapshot();
    REQUIRE(initial);
    REQUIRE(initial->empty());

    world.generateAroundPlayer(0, SEA_LEVEL, 0);
    REQUIRE(world.getMeshCandidateSnapshot() == initial);
    world.publishLoadedSnapshot();

    const auto published = world.getMeshCandidateSnapshot();
    REQUIRE(published != initial);
    REQUIRE_FALSE(published->empty());
    for (ChunkPos pos : *published)
        REQUIRE(world.shouldMeshChunk(pos));
}

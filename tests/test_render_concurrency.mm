#include <catch2/catch_test_macros.hpp>
#include <render/mesh_scheduler.hpp>
#include <render/render_pipeline.hpp>
#include <world/column_plan.hpp>
#include <world/world.hpp>

#include <algorithm>
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
        REQUIRE(scheduler.enqueue({static_cast<int64_t>(index), 0, 0}, 0,
                                  MeshPriorityLane::CAMERA_COLUMN));
    }
    REQUIRE_FALSE(scheduler.enqueue({1000, 0, 0}, 0, MeshPriorityLane::CAMERA_COLUMN));

    MeshSchedulerStats full = scheduler.stats();
    REQUIRE(full.schedulerOwned + full.consumerPending == MeshScheduler::MAX_INFLIGHT_MESH);
    REQUIRE(full.highWater == MeshScheduler::MAX_INFLIGHT_MESH);

    consumer.clear();
    scheduler.acknowledgeConsumerPending(consumer.size());
    REQUIRE(scheduler.stats().consumerPending == 0);
    REQUIRE(scheduler.enqueue({1000, 0, 0}, 0, MeshPriorityLane::CAMERA_COLUMN));
    REQUIRE_FALSE(scheduler.enqueue({1001, 0, 0}, 0, MeshPriorityLane::CAMERA_COLUMN));

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
        REQUIRE(scheduler.enqueue(missing, static_cast<uint32_t>(request + 1)));

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
    REQUIRE(results.front().requestedVersion == REQUESTS);
    REQUIRE_FALSE(results.front().snapshotOk);
    REQUIRE(scheduler.stats().schedulerOwned == 0);
}

TEST_CASE("Exact mesh publication rejects results from older renderer drains",
          "[render][scheduler][concurrency][revision][regression]") {
    uint32_t residentVersion = 0;

    REQUIRE(chunkMeshAsyncResultCanReplace(2, 2, residentVersion));
    residentVersion = 2;

    // Revision one completed after revision two was already published by an
    // earlier drain. It cannot replace the newer resident mesh.
    REQUIRE_FALSE(chunkMeshAsyncResultCanReplace(1, 2, residentVersion));
    REQUIRE(residentVersion == 2);

    // Duplicate and future results also cannot publish against revision two.
    REQUIRE_FALSE(chunkMeshAsyncResultCanReplace(2, 2, residentVersion));
    REQUIRE_FALSE(chunkMeshAsyncResultCanReplace(3, 2, residentVersion));
}

TEST_CASE("Exact mesh completion preserves unrelated newer requests",
          "[render][scheduler][concurrency][revision][regression]") {
    REQUIRE(chunkMeshRequestAfterCompletion(2, 2) == 0);
    REQUIRE(chunkMeshRequestAfterCompletion(3, 1) == 3);
    REQUIRE(chunkMeshRequestAfterCompletion(3, 2) == 3);
    REQUIRE(chunkMeshRequestAfterCompletion(3, 3) == 0);
}

TEST_CASE("Exact mesh coalescing preserves the newest failed request",
          "[render][scheduler][concurrency][revision][regression]") {
    MeshResult successfulOld;
    successfulOld.requestedVersion = 1;
    successfulOld.builtVersion = 1;
    successfulOld.snapshotOk = true;

    MeshResult failedNew;
    failedNew.requestedVersion = 2;
    failedNew.snapshotOk = false;

    REQUIRE(meshResultSupersedes(successfulOld, failedNew));
    REQUIRE_FALSE(meshResultSupersedes(failedNew, successfulOld));

    MeshResult successfulNew;
    successfulNew.requestedVersion = 2;
    successfulNew.builtVersion = 2;
    successfulNew.snapshotOk = true;
    REQUIRE(meshResultSupersedes(failedNew, successfulNew));
}

TEST_CASE("Mesh snapshots preserve planned and exact cutoffs across negative chunk boundaries",
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
            const int32_t plannedCutoff =
                plan->surfaceY(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ)) + 1;
            REQUIRE(snapshot.generatedSurfaceCutoffAt(x, z) == plannedCutoff);

            const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(worldX, worldZ);
            REQUIRE(loadedTop);
            REQUIRE(snapshot.skyCutoffAt(x, z) == *loadedTop + 1);
            REQUIRE(snapshot.skyCutoffAt(x, z) >= plannedCutoff);
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

TEST_CASE("World publishes pre-cap exact surface coverage requirements",
          "[world][render][coverage][snapshot]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    const auto initial = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(initial);
    REQUIRE(initial->epoch == 0);
    REQUIRE(initial->requiredSections.empty());

    world.generateAroundPlayer(0, SEA_LEVEL, 0);
    const auto coverage = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(coverage != initial);
    REQUIRE(coverage->epoch > 0);
    REQUIRE(coverage->nominalRadiusChunks == MIN_RENDER_DISTANCE_CHUNKS);
    REQUIRE_FALSE(coverage->requiredSections.empty());
    REQUIRE(std::is_sorted(coverage->requiredSections.begin(), coverage->requiredSections.end(),
                           [](ChunkPos left, ChunkPos right) {
                               if (left.x != right.x)
                                   return left.x < right.x;
                               if (left.z != right.z)
                                   return left.z < right.z;
                               return left.y < right.y;
                           }));
    REQUIRE(
        std::adjacent_find(coverage->requiredSections.begin(), coverage->requiredSections.end()) ==
        coverage->requiredSections.end());
    for (ChunkPos required : coverage->requiredSections) {
        const int64_t distanceSquared = required.x * required.x + required.z * required.z;
        constexpr int EXPECTED_RADIUS =
            std::max(MIN_RENDER_DISTANCE_CHUNKS + 1, EXPLORATION_RADIUS_CHUNKS);
        REQUIRE(distanceSquared <= EXPECTED_RADIUS * EXPECTED_RADIUS);
    }
}

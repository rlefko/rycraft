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

TEST_CASE("Nearby exact meshes displace saturated distant and stale queued work",
          "[render][scheduler][concurrency][priority][camera-jump][regression]") {
    World world(42, 4);
    MeshScheduler scheduler(world, 0);

    for (size_t index = 0; index < EXACT_MESH_MAX_INFLIGHT / 2; ++index) {
        REQUIRE(scheduler.enqueue({static_cast<int64_t>(index), 0, 100}, 1,
                                  MeshPriorityLane::BROAD_SURFACE, index));
    }
    for (size_t index = 0; index < EXACT_MESH_MAX_INFLIGHT / 2; ++index) {
        REQUIRE(scheduler.enqueue({static_cast<int64_t>(index), 0, 10}, 1,
                                  MeshPriorityLane::CAMERA_BAND, index));
    }
    REQUIRE(scheduler.stats().schedulerOwned == EXACT_MESH_MAX_INFLIGHT);

    std::optional<MeshCanceledRequest> displaced;
    constexpr ChunkPos NEW_CAMERA{900, 5, -900};
    REQUIRE(scheduler.enqueue(NEW_CAMERA, 7, MeshPriorityLane::CAMERA_COLUMN, 0, &displaced));
    REQUIRE(displaced);
    REQUIRE(displaced->pos == ChunkPos{31, 0, 100});
    REQUIRE(displaced->requestedVersion == 1);
    REQUIRE(scheduler.stats().schedulerOwned == EXACT_MESH_MAX_INFLIGHT);
    REQUIRE(scheduler.stats().displaced == 1);

    const std::unordered_set<ChunkPos> currentCandidates{NEW_CAMERA};
    const std::vector<MeshCanceledRequest> canceled =
        scheduler.cancelQueuedOutside(currentCandidates);
    REQUIRE(canceled.size() == EXACT_MESH_MAX_INFLIGHT - 1);
    REQUIRE(scheduler.stats().schedulerOwned == 1);
    REQUIRE(scheduler.stats().canceledQueued == EXACT_MESH_MAX_INFLIGHT - 1);

    scheduler.shutdown();
}

TEST_CASE("Queued exact mesh work follows a moving camera without duplicate builds",
          "[render][scheduler][concurrency][priority][movement][regression]") {
    World world(42, 4);
    MeshScheduler scheduler(world, 0);
    constexpr ChunkPos MOVED_NEAR{24, 4, -10};
    constexpr ChunkPos STILL_DISTANT{-24, 4, 10};
    REQUIRE(scheduler.enqueue(MOVED_NEAR, 11, MeshPriorityLane::BROAD_SURFACE, 900));
    REQUIRE(scheduler.enqueue(STILL_DISTANT, 12, MeshPriorityLane::BROAD_SURFACE, 800));

    const size_t changed = scheduler.reprioritizeQueued([=](ChunkPos position) {
        if (position == MOVED_NEAR)
            return MeshRequestPriority{MeshPriorityLane::CAMERA_COLUMN, 0};
        return MeshRequestPriority{MeshPriorityLane::BROAD_SURFACE, 800};
    });
    REQUIRE(changed == 1);
    REQUIRE(scheduler.stats().schedulerOwned == 2);

    const std::optional<MeshCanceledRequest> canceled = scheduler.cancelQueued(MOVED_NEAR);
    REQUIRE(canceled);
    REQUIRE(canceled->requestedVersion == 11);
    REQUIRE(scheduler.stats().schedulerOwned == 1);
    REQUIRE(scheduler.stats().canceledQueued == 1);
    REQUIRE_FALSE(scheduler.cancelQueued(MOVED_NEAR));

    scheduler.shutdown();
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
    for (int offsetZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
         offsetZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetZ) {
        for (int offsetX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             offsetX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({center.x + offsetX, center.z + offsetZ}));
        }
    }
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto plan =
                world.generator().getColumnPlan({center.x + offsetX, center.z + offsetZ});
            REQUIRE(plan);
            for (const int32_t section : plan->exposedSections()) {
                if (section >= center.y) {
                    REQUIRE(world.getChunk({center.x + offsetX, section, center.z + offsetZ}));
                }
            }
        }
    }
    for (int pass = 0;
         pass < 64 && world.getStreamingWorkStats().publicationLightDeferredQueue != 0; ++pass) {
        world.reconcileLight(1'024);
    }
    REQUIRE(world.getStreamingWorkStats().publicationLightDeferredQueue == 0);
    REQUIRE(world.getStreamingWorkStats().publicationLightMaxSyncFloods <= 32);

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
    const auto firstVisibleLight = snapshot.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(center, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);
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

TEST_CASE("V4 preparation publishes exact candidates without a gameplay tick",
          "[world][render][snapshot][v4][startup][regression]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    const auto initialLoaded = world.getLoadedSnapshot();
    const auto initialCandidates = world.getMeshCandidateSnapshot();
    REQUIRE(initialLoaded);
    REQUIRE(initialLoaded->empty());
    REQUIRE(initialCandidates);
    REQUIRE(initialCandidates->empty());

    REQUIRE(world.generator().getColumnPlan({0, 0}));
    world.generateAroundPlayer(0, SEA_LEVEL, 0);
    const auto coverage = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(coverage);
    REQUIRE_FALSE(coverage->requiredSections.empty());

    size_t internallySelectedRequirements = 0;
    for (const ChunkPos section : coverage->requiredSections)
        internallySelectedRequirements += world.shouldMeshChunk(section) ? 1U : 0U;
    REQUIRE(internallySelectedRequirements > 0);
    REQUIRE(world.getMeshCandidateSnapshot() == initialCandidates);

    publishV4PreparationWorldSnapshot(world);

    const auto publishedCandidates = world.getMeshCandidateSnapshot();
    REQUIRE(publishedCandidates != initialCandidates);
    size_t publishedRequirements = 0;
    for (const ChunkPos section : coverage->requiredSections)
        publishedRequirements += publishedCandidates->contains(section) ? 1U : 0U;
    REQUIRE(publishedRequirements == internallySelectedRequirements);
}

TEST_CASE("World publishes pre-cap exact surface coverage requirements",
          "[world][render][coverage][snapshot]") {
    World world(42, MIN_RENDER_DISTANCE_CHUNKS);
    const auto initial = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(initial);
    REQUIRE(initial->epoch == 0);
    REQUIRE(initial->requiredSections.empty());
    REQUIRE(initial->floraRequiredSections.empty());

    // Seed one completed plan so the snapshot exercises the broad tree
    // interval instead of containing only unresolved-column fallbacks.
    REQUIRE(world.generator().getColumnPlan({0, 0}));
    world.generateAroundPlayer(0, SEA_LEVEL, 0);
    const auto coverage = world.getExactSurfaceCoverageSnapshot();
    REQUIRE(coverage != initial);
    REQUIRE(coverage->epoch > 0);
    REQUIRE(coverage->nominalRadiusChunks == MIN_RENDER_DISTANCE_CHUNKS);
    REQUIRE_FALSE(coverage->requiredSections.empty());
    REQUIRE_FALSE(coverage->floraRequiredSections.empty());
    const auto sectionOrder = [](ChunkPos left, ChunkPos right) {
        if (left.x != right.x)
            return left.x < right.x;
        if (left.z != right.z)
            return left.z < right.z;
        return left.y < right.y;
    };
    REQUIRE(std::is_sorted(coverage->requiredSections.begin(), coverage->requiredSections.end(),
                           sectionOrder));
    REQUIRE(std::is_sorted(coverage->floraRequiredSections.begin(),
                           coverage->floraRequiredSections.end(), sectionOrder));
    REQUIRE(
        std::adjacent_find(coverage->requiredSections.begin(), coverage->requiredSections.end()) ==
        coverage->requiredSections.end());
    REQUIRE(std::adjacent_find(coverage->floraRequiredSections.begin(),
                               coverage->floraRequiredSections.end()) ==
            coverage->floraRequiredSections.end());
    for (ChunkPos required : coverage->requiredSections) {
        const int64_t distanceSquared = required.x * required.x + required.z * required.z;
        constexpr int EXPECTED_RADIUS =
            std::max(MIN_RENDER_DISTANCE_CHUNKS + 1, EXPLORATION_RADIUS_CHUNKS);
        REQUIRE(distanceSquared <= EXPECTED_RADIUS * EXPECTED_RADIUS);
    }
    for (ChunkPos required : coverage->floraRequiredSections) {
        const int64_t distanceSquared = required.x * required.x + required.z * required.z;
        constexpr int EXPECTED_RADIUS =
            std::max(MIN_RENDER_DISTANCE_CHUNKS + 1, EXPLORATION_RADIUS_CHUNKS);
        REQUIRE(distanceSquared <= EXPECTED_RADIUS * EXPECTED_RADIUS);
    }
}

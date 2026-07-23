#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>
#import <mach/mach.h>

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <audio/thunder.hpp>
#include <common/error.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <engine/application_termination.hpp>
#include <engine/camera.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <engine/mining.hpp>
#include <engine/playtest_fixture.hpp>
#include <engine/slot_interaction.hpp>
#include <engine/survival.hpp>
#include <engine/v4_world_startup.hpp>
#include <entity/ai.hpp>
#include <entity/boat.hpp>
#include <entity/entity_picking.hpp>
#include <entity/item_entity.hpp>
#include <entity/player.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/celestial.hpp>
#include <render/far_terrain.hpp>
#include <render/graphics_settings.hpp>
#include <render/pixel_formats.hpp>
#include <render/render_pipeline.hpp>
#include <render/ui_menu.hpp>
#include <world/furnace.hpp>
#include <world/native_hydrology.hpp>
#include <world/recipes.hpp>
#include <world/save_manager.hpp>
#include <world/terrain_bootstrap.hpp>
#include <world/terrain_runtime.hpp>
#include <world/weather.hpp>
#include <world/world.hpp>
#include <world/world_list.hpp>

#include <random>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Engine, Singleton (Objective-C class with C++ internals)
// ---------------------------------------------------------------------------

// Raycast hit result type for block interaction
using BlockRayHit = std::optional<std::pair<Vec3, Vec3>>;

struct PerformanceCapture {
    uint64_t warmupFrames = 0;
    uint64_t requestedFrames = 0;
    std::vector<double> frameMilliseconds;
    std::vector<double> fixedTickMilliseconds;
    uint32_t maxLoadedCubes = 0;
    uint32_t maxMeshedCubes = 0;
    size_t maxGenerationQueue = 0;
    uint32_t maxMeshQueue = 0;
    uint32_t maxMeshQueueHighWater = 0;
    uint64_t maxMeshCoalesced = 0;
    uint64_t maxMeshDroppedStale = 0;
    uint32_t maxFarWanted = 0;
    uint32_t maxFarResident = 0;
    uint32_t maxFarDrawn = 0;
    uint32_t maxFarFrustumCulled = 0;
    uint32_t maxFarOcclusionCulled = 0;
    uint32_t maxFarPending = 0;
    uint32_t maxExactUploads = 0;
    uint32_t maxFarUploads = 0;
    float maxFarCacheMB = 0.0f;
    float maxFarArenaMB = 0.0f;
    float maxExactArenaMB = 0.0f;
    float maxGpuFrameMs = 0.0f;
    std::array<uint32_t, SHADOW_CASCADE_COUNT> maxShadowCasterCounts{};
    std::array<uint64_t, SHADOW_CASCADE_COUNT> shadowRefreshCounts{};
    uint64_t atmosphereSlowRefreshCount = 0;
    uint64_t atmosphereSkyRefreshCount = 0;
    uint64_t indirectHistoryInvalidFrames = 0;
    uint64_t cloudHistoryInvalidFrames = 0;
    uint64_t froxelHistoryInvalidFrames = 0;
    uint64_t maxIndirectPersistentBytes = 0;
    uint64_t maxCloudPersistentBytes = 0;
    uint64_t maxFroxelPersistentBytes = 0;
    uint64_t maxIntegratedAtmosphericPersistentBytes = 0;
    uint64_t weatherRequests = 0;
    uint64_t weatherCoalescedRequests = 0;
    uint64_t weatherBuildsStarted = 0;
    uint64_t weatherBuildsDeferred = 0;
    uint64_t weatherBuildsFailed = 0;
    uint64_t weatherSnapshotsPublished = 0;
    uint64_t weatherStaleBuildsDiscarded = 0;
    size_t maxWeatherPendingRequests = 0;
    uint64_t weatherWorkerBusyFrames = 0;
    size_t maxThunderPending = 0;
    uint64_t peakResidentBytes = 0;
    uint64_t peakMetalBytes = 0;
    double settleStartSeconds = -1.0;
    double settleSeconds = -1.0;
    bool reported = false;

    bool enabled() const { return requestedFrames > 0; }
};

void recordPerformanceFixedTick(PerformanceCapture& capture, uint64_t frameCount,
                                double milliseconds) {
    if (!capture.enabled() || capture.reported || frameCount < capture.warmupFrames ||
        capture.frameMilliseconds.size() >= capture.requestedFrames) {
        return;
    }
    capture.fixedTickMilliseconds.push_back(milliseconds);
}

uint64_t unsignedEnvironmentValue(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end == value || *end != '\0' ? fallback : static_cast<uint64_t>(parsed);
}

std::optional<Vec3> vectorEnvironmentValue(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value)
        return std::nullopt;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    char trailing = '\0';
    if (std::sscanf(value, "%f,%f,%f %c", &x, &y, &z, &trailing) != 3 || !std::isfinite(x) ||
        !std::isfinite(y) || !std::isfinite(z)) {
        return std::nullopt;
    }
    return Vec3{x, y, z};
}

std::optional<float> finiteEnvironmentValue(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value)
        return std::nullopt;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed))
        return std::nullopt;
    return parsed;
}

static_assert(V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS * CHUNK_EDGE ==
              (FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_RADIUS_TILES + 1) * FAR_TERRAIN_TILE_EDGE);
static_assert(FAR_TERRAIN_COVERAGE_FADE_BLOCKS == FAR_TERRAIN_TILE_EDGE);

float v4ConnectedParentRadiusChunks(const RenderPipeline::ChunkRenderStats& stats,
                                    int configuredHorizonChunks) noexcept {
    if (configuredHorizonChunks <= 0 || stats.farBaseWantedTileCount == 0)
        return 0.0F;
    if (stats.farBaseMissingTileCount == 0 &&
        stats.farBaseResidentTileCount >= stats.farBaseWantedTileCount) {
        return static_cast<float>(configuredHorizonChunks);
    }
    const float radiusChunks = stats.farCoverageFrontierBlocks / CHUNK_EDGE;
    return std::isfinite(radiusChunks) && radiusChunks > 0.0F ? radiusChunks : 0.0F;
}

void resolveLightningTerrainHeightIfLoaded(const World& world, LightningEvent& event) {
    const std::optional<float> loadedHeight = world.strikeSurfaceHeightIfLoaded(
        static_cast<int64_t>(std::floor(event.x)), static_cast<int64_t>(std::floor(event.z)));
    if (loadedHeight)
        event.y = *loadedHeight;
    event.cloudY = std::max(event.cloudY, event.y + 1.0F);
}

uint64_t processResidentBytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t result = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                           reinterpret_cast<task_info_t>(&info), &count);
    return result == KERN_SUCCESS ? static_cast<uint64_t>(info.resident_size) : 0;
}

bool updatePerformanceCapture(PerformanceCapture& capture, uint64_t frameCount,
                              double frameMilliseconds, uint32_t loadedCubes,
                              uint64_t autopilotStopFrame, World& world, RenderPipeline& renderer,
                              id<MTLDevice> device, const WeatherSystemStats& weatherStats,
                              size_t thunderPending) {
    if (!capture.enabled() || capture.reported)
        return false;

    const RenderPipeline::ChunkRenderStats stats = renderer.chunkRenderStats();
    const RenderPipeline::AtmosphericRenderStats atmospheric = renderer.atmosphericRenderStats();
    if (autopilotStopFrame != std::numeric_limits<uint64_t>::max() &&
        frameCount >= autopilotStopFrame) {
        if (capture.settleStartSeconds < 0.0)
            capture.settleStartSeconds = CACurrentMediaTime();
        const StreamingWorkStats streaming = world.getStreamingWorkStats();
        const bool settled = world.getPendingChunkCount() == 0 && stats.meshPendingCount == 0 &&
                             stats.farPendingTileCount == 0 &&
                             stats.farResidentTileCount >= stats.farWantedTileCount &&
                             streaming.publicationLightDeferredQueue == 0;
        if (settled && capture.settleSeconds < 0.0) {
            capture.settleSeconds = CACurrentMediaTime() - capture.settleStartSeconds;
        }
    }

    if (frameCount < capture.warmupFrames ||
        capture.frameMilliseconds.size() >= capture.requestedFrames) {
        return false;
    }

    capture.frameMilliseconds.push_back(frameMilliseconds);
    capture.maxLoadedCubes = std::max(capture.maxLoadedCubes, loadedCubes);
    capture.maxMeshedCubes = std::max(capture.maxMeshedCubes, stats.meshCubeCount);
    capture.maxGenerationQueue = std::max(capture.maxGenerationQueue, world.getPendingChunkCount());
    capture.maxMeshQueue = std::max(capture.maxMeshQueue, stats.meshPendingCount);
    capture.maxMeshQueueHighWater =
        std::max(capture.maxMeshQueueHighWater, stats.meshQueueHighWater);
    capture.maxMeshCoalesced = std::max(capture.maxMeshCoalesced, stats.meshCoalescedCount);
    capture.maxMeshDroppedStale =
        std::max(capture.maxMeshDroppedStale, stats.meshDroppedStaleCount);
    capture.maxFarWanted = std::max(capture.maxFarWanted, stats.farWantedTileCount);
    capture.maxFarResident = std::max(capture.maxFarResident, stats.farResidentTileCount);
    capture.maxFarDrawn = std::max(capture.maxFarDrawn, stats.farDrawnTileCount);
    capture.maxFarFrustumCulled =
        std::max(capture.maxFarFrustumCulled, stats.farFrustumCulledTileCount);
    capture.maxFarOcclusionCulled =
        std::max(capture.maxFarOcclusionCulled, stats.farOcclusionCulledTileCount);
    capture.maxFarPending = std::max(capture.maxFarPending, stats.farPendingTileCount);
    capture.maxExactUploads = std::max(capture.maxExactUploads, stats.meshBuildsLastFrame);
    capture.maxFarUploads = std::max(capture.maxFarUploads, stats.farUploadsLastFrame);
    capture.maxFarCacheMB = std::max(capture.maxFarCacheMB, stats.farCacheMB);
    capture.maxFarArenaMB = std::max(capture.maxFarArenaMB, stats.farMegaUsedMB);
    capture.maxExactArenaMB = std::max(capture.maxExactArenaMB, stats.megaUsedMB);
    capture.maxGpuFrameMs = std::max(capture.maxGpuFrameMs, renderer.gpuFrameMs());
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        capture.maxShadowCasterCounts[cascade] = std::max(capture.maxShadowCasterCounts[cascade],
                                                          atmospheric.shadowCasterCounts[cascade]);
        capture.shadowRefreshCounts[cascade] = std::max(capture.shadowRefreshCounts[cascade],
                                                        atmospheric.shadowRefreshCounts[cascade]);
    }
    capture.atmosphereSlowRefreshCount =
        std::max(capture.atmosphereSlowRefreshCount, atmospheric.atmosphereSlowRefreshCount);
    capture.atmosphereSkyRefreshCount =
        std::max(capture.atmosphereSkyRefreshCount, atmospheric.atmosphereSkyRefreshCount);
    capture.indirectHistoryInvalidFrames += atmospheric.indirectHistoryValid ? 0U : 1U;
    capture.cloudHistoryInvalidFrames += atmospheric.cloudHistoryValid ? 0U : 1U;
    capture.froxelHistoryInvalidFrames += atmospheric.froxelHistoryValid ? 0U : 1U;
    capture.maxIndirectPersistentBytes =
        std::max(capture.maxIndirectPersistentBytes, atmospheric.indirectPersistentBytes);
    capture.maxCloudPersistentBytes =
        std::max(capture.maxCloudPersistentBytes, atmospheric.cloudPersistentBytes);
    capture.maxFroxelPersistentBytes =
        std::max(capture.maxFroxelPersistentBytes, atmospheric.froxelPersistentBytes);
    capture.maxIntegratedAtmosphericPersistentBytes = std::max(
        capture.maxIntegratedAtmosphericPersistentBytes, atmospheric.integratedPersistentBytes);
    capture.weatherRequests = std::max(capture.weatherRequests, weatherStats.requests);
    capture.weatherCoalescedRequests =
        std::max(capture.weatherCoalescedRequests, weatherStats.coalescedRequests);
    capture.weatherBuildsStarted =
        std::max(capture.weatherBuildsStarted, weatherStats.buildsStarted);
    capture.weatherBuildsDeferred =
        std::max(capture.weatherBuildsDeferred, weatherStats.buildsDeferred);
    capture.weatherBuildsFailed = std::max(capture.weatherBuildsFailed, weatherStats.buildsFailed);
    capture.weatherSnapshotsPublished =
        std::max(capture.weatherSnapshotsPublished, weatherStats.snapshotsPublished);
    capture.weatherStaleBuildsDiscarded =
        std::max(capture.weatherStaleBuildsDiscarded, weatherStats.staleBuildsDiscarded);
    capture.maxWeatherPendingRequests =
        std::max(capture.maxWeatherPendingRequests, weatherStats.pendingRequests);
    capture.weatherWorkerBusyFrames += weatherStats.workerBusy ? 1U : 0U;
    capture.maxThunderPending = std::max(capture.maxThunderPending, thunderPending);
    if (frameCount % 30 == 0 || capture.frameMilliseconds.size() == 1) {
        capture.peakResidentBytes = std::max(capture.peakResidentBytes, processResidentBytes());
        capture.peakMetalBytes =
            std::max(capture.peakMetalBytes, static_cast<uint64_t>([device currentAllocatedSize]));
    }

    if (capture.frameMilliseconds.size() < capture.requestedFrames)
        return false;

    std::vector<double> sorted = capture.frameMilliseconds;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double fraction) {
        const size_t index =
            static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0);
        return sorted[std::min(index, sorted.size() - 1)];
    };
    std::vector<double> prefix(capture.frameMilliseconds.size() + 1, 0.0);
    for (size_t index = 0; index < capture.frameMilliseconds.size(); ++index) {
        prefix[index + 1] = prefix[index] + capture.frameMilliseconds[index];
    }
    double lowestOneSecondFps = std::numeric_limits<double>::infinity();
    for (size_t start = 0; start < capture.frameMilliseconds.size(); ++start) {
        const auto end = std::lower_bound(prefix.begin() + static_cast<ptrdiff_t>(start + 1),
                                          prefix.end(), prefix[start] + 1000.0);
        if (end == prefix.end())
            break;
        const size_t endIndex = static_cast<size_t>(std::distance(prefix.begin(), end));
        const double elapsed = prefix[endIndex] - prefix[start];
        const double fps = static_cast<double>(endIndex - start) * 1000.0 / elapsed;
        lowestOneSecondFps = std::min(lowestOneSecondFps, fps);
    }
    if (!std::isfinite(lowestOneSecondFps))
        lowestOneSecondFps = 1000.0 / percentile(0.95);
    const size_t overTwentyMilliseconds = static_cast<size_t>(
        std::count_if(capture.frameMilliseconds.begin(), capture.frameMilliseconds.end(),
                      [](double milliseconds) { return milliseconds > 20.0; }));
    const double maximumFrame =
        *std::max_element(capture.frameMilliseconds.begin(), capture.frameMilliseconds.end());
    constexpr double MEBIBYTE = 1024.0 * 1024.0;
    const double credibleUnifiedMB =
        static_cast<double>(std::max(capture.peakResidentBytes, capture.peakMetalBytes)) / MEBIBYTE;

    char timing[512];
    std::snprintf(timing, sizeof(timing),
                  "Performance summary: %zu frames p50 %.3f ms p95 %.3f ms max %.3f ms "
                  "lowest 1s %.2f FPS over 20ms %zu gpu EMA max %.3f ms",
                  capture.frameMilliseconds.size(), percentile(0.50), percentile(0.95),
                  maximumFrame, lowestOneSecondFps, overTwentyMilliseconds, capture.maxGpuFrameMs);
    RY_LOG_INFO(timing);

    if (!capture.fixedTickMilliseconds.empty()) {
        std::vector<double> sortedTicks = capture.fixedTickMilliseconds;
        std::sort(sortedTicks.begin(), sortedTicks.end());
        const auto tickPercentile = [&](double fraction) {
            const size_t index = static_cast<size_t>(
                std::ceil(fraction * static_cast<double>(sortedTicks.size())) - 1.0);
            return sortedTicks[std::min(index, sortedTicks.size() - 1)];
        };
        const double maximumTick = *std::max_element(sortedTicks.begin(), sortedTicks.end());
        char fixedTick[256];
        std::snprintf(fixedTick, sizeof(fixedTick),
                      "Performance fixed tick: %zu ticks p50 %.3f ms p95 %.3f ms max %.3f ms",
                      sortedTicks.size(), tickPercentile(0.50), tickPercentile(0.95), maximumTick);
        RY_LOG_INFO(fixedTick);
    } else {
        RY_LOG_INFO("Performance fixed tick: no gameplay ticks in capture window");
    }

    char residency[640];
    std::snprintf(residency, sizeof(residency),
                  "Performance residency: cubes loaded %u meshed %u queues gen %zu mesh %u high %u "
                  "coalesced %llu stale %llu uploads exact %u far wanted %u resident %u drawn %u "
                  "frustum %u occluded %u pending %u uploads %u cache %.1f MB arena %.1f MB exact "
                  "arena %.1f MB",
                  capture.maxLoadedCubes, capture.maxMeshedCubes, capture.maxGenerationQueue,
                  capture.maxMeshQueue, capture.maxMeshQueueHighWater,
                  static_cast<unsigned long long>(capture.maxMeshCoalesced),
                  static_cast<unsigned long long>(capture.maxMeshDroppedStale),
                  capture.maxExactUploads, capture.maxFarWanted, capture.maxFarResident,
                  capture.maxFarDrawn, capture.maxFarFrustumCulled, capture.maxFarOcclusionCulled,
                  capture.maxFarPending, capture.maxFarUploads, capture.maxFarCacheMB,
                  capture.maxFarArenaMB, capture.maxExactArenaMB);
    RY_LOG_INFO(residency);

    const StreamingWorkStats streaming = world.getStreamingWorkStats();
    const RenderPipeline::ChunkRenderStats farPlanner = renderer.chunkRenderStats();
    char planner[896];
    std::snprintf(planner, sizeof(planner),
                  "Performance streaming planner: rebuilds %llu requests %llu coalesced %llu "
                  "canceled %llu build EMA %.3f ms notifications %llu | far last/p95/max "
                  "%.3f/%.3f/%.3f ms phase p95 %.3f/%.3f/%.3f select/publish/resident arena "
                  "denied %llu critical %u/%u/%u displaced %llu optional evictions %llu %.1f MB "
                  "| publication light queue %zu deferred %llu max sync %zu",
                  static_cast<unsigned long long>(streaming.activeSetRebuilds),
                  static_cast<unsigned long long>(streaming.activeSetRequests),
                  static_cast<unsigned long long>(streaming.activeSetRequestsCoalesced),
                  static_cast<unsigned long long>(streaming.activeSetBuildsCanceled),
                  streaming.activeSetBuildMs,
                  static_cast<unsigned long long>(streaming.activeSetRebuildNotifications),
                  farPlanner.farPlannerMsLast, farPlanner.farPlannerMsP95,
                  farPlanner.farPlannerMsMax, farPlanner.farPlannerSelectionMsP95,
                  farPlanner.farPlannerPublicationMsP95, farPlanner.farPlannerResidencyMsP95,
                  static_cast<unsigned long long>(farPlanner.farArenaAdmissionDeniedCount),
                  farPlanner.farCriticalWantedTileCount, farPlanner.farCriticalResidentTileCount,
                  farPlanner.farCriticalMissingTileCount,
                  static_cast<unsigned long long>(farPlanner.farCriticalSchedulerDisplacementCount),
                  static_cast<unsigned long long>(farPlanner.farNearArenaReclaimCount),
                  static_cast<double>(farPlanner.farNearArenaReclaimedBytes) / (1024.0 * 1024.0),
                  streaming.publicationLightDeferredQueue,
                  static_cast<unsigned long long>(streaming.publicationLightDeferredCubes),
                  streaming.publicationLightMaxSyncFloods);
    RY_LOG_INFO(planner);

    char memory[384];
    std::snprintf(memory, sizeof(memory),
                  "Performance memory: process RSS %.1f MB Metal allocated %.1f MB credible "
                  "unified %.1f MB queue settle %.3f s",
                  static_cast<double>(capture.peakResidentBytes) / MEBIBYTE,
                  static_cast<double>(capture.peakMetalBytes) / MEBIBYTE, credibleUnifiedMB,
                  capture.settleSeconds);
    RY_LOG_INFO(memory);

    char atmosphericSummary[768];
    std::snprintf(
        atmosphericSummary, sizeof(atmosphericSummary),
        "Performance atmosphere: shadow selected %u/%u/%u/%u/%u refreshes %llu/%llu/%llu/%llu/"
        "%llu LUT slow %llu sky %llu history invalid frames indirect %llu cloud %llu froxel %llu",
        capture.maxShadowCasterCounts[0], capture.maxShadowCasterCounts[1],
        capture.maxShadowCasterCounts[2], capture.maxShadowCasterCounts[3],
        capture.maxShadowCasterCounts[4],
        static_cast<unsigned long long>(capture.shadowRefreshCounts[0]),
        static_cast<unsigned long long>(capture.shadowRefreshCounts[1]),
        static_cast<unsigned long long>(capture.shadowRefreshCounts[2]),
        static_cast<unsigned long long>(capture.shadowRefreshCounts[3]),
        static_cast<unsigned long long>(capture.shadowRefreshCounts[4]),
        static_cast<unsigned long long>(capture.atmosphereSlowRefreshCount),
        static_cast<unsigned long long>(capture.atmosphereSkyRefreshCount),
        static_cast<unsigned long long>(capture.indirectHistoryInvalidFrames),
        static_cast<unsigned long long>(capture.cloudHistoryInvalidFrames),
        static_cast<unsigned long long>(capture.froxelHistoryInvalidFrames));
    RY_LOG_INFO(atmosphericSummary);

    char atmosphericMemory[384];
    std::snprintf(atmosphericMemory, sizeof(atmosphericMemory),
                  "Performance atmospheric memory: indirect %.1f MB cloud %.1f MB froxel %.1f MB "
                  "integrated %.1f MB",
                  static_cast<double>(capture.maxIndirectPersistentBytes) / MEBIBYTE,
                  static_cast<double>(capture.maxCloudPersistentBytes) / MEBIBYTE,
                  static_cast<double>(capture.maxFroxelPersistentBytes) / MEBIBYTE,
                  static_cast<double>(capture.maxIntegratedAtmosphericPersistentBytes) / MEBIBYTE);
    RY_LOG_INFO(atmosphericMemory);

    char weather[512];
    std::snprintf(
        weather, sizeof(weather),
        "Performance weather worker: requests %llu coalesced %llu builds %llu deferred %llu "
        "failed %llu published %llu stale %llu pending max %zu busy samples %llu thunder pending "
        "max %zu",
        static_cast<unsigned long long>(capture.weatherRequests),
        static_cast<unsigned long long>(capture.weatherCoalescedRequests),
        static_cast<unsigned long long>(capture.weatherBuildsStarted),
        static_cast<unsigned long long>(capture.weatherBuildsDeferred),
        static_cast<unsigned long long>(capture.weatherBuildsFailed),
        static_cast<unsigned long long>(capture.weatherSnapshotsPublished),
        static_cast<unsigned long long>(capture.weatherStaleBuildsDiscarded),
        capture.maxWeatherPendingRequests,
        static_cast<unsigned long long>(capture.weatherWorkerBusyFrames),
        capture.maxThunderPending);
    RY_LOG_INFO(weather);
    capture.reported = true;
    return true;
}

// Internal C++ state
struct EngineState {
    // ---- Game loop state ----
    double lastTime = 0;
    double deltaTime = 0;
    uint64_t frameCount = 0;
    FrameCaptureClock frameCaptureClock;
    uint64_t autoPauseFrame = std::numeric_limits<uint64_t>::max();

    // ---- Fixed timestep accumulator ----
    static constexpr double TICK_RATE = 20.0;
    static constexpr double TICK_DT = 1.0 / TICK_RATE;
    double accumulator = 0;

    // ---- Projection ----
    Mat4 projectionMatrix = Mat4::identity();
    CGSize drawableSize = {0, 0};

    // ---- Field of view ----
    // Sprinting widens the FOV for a feeling of speed; the value eases
    // per frame (render parameter, not sim state) toward the mode's target.
    static constexpr float BASE_FOV = 70.0f;
    static constexpr float SPRINT_FOV = 77.0f; // +10%, the classic sprint cue
    static constexpr float FOV_EASE_SECONDS = 0.1f;
    float fovCurrent = BASE_FOV;

    // ---- Day/Night Cycle & Weather ----
    uint64_t worldTime = 0;
    std::shared_ptr<const WeatherSnapshot> weatherSnapshot;
    WeatherSample localWeather;
    std::vector<LightningEvent> lightningEvents;
    uint64_t lastWeatherEventTick = 0;
    ThunderScheduler thunder;
    std::optional<CaptureLightningOverride> captureLightningOverride;
    bool captureLightningInjected = false;
    float wetness = 0.0f; // 0 dry .. 1 soaked; integrates local precipitation
    static constexpr float SOAK_SECONDS = 15.0f;
    static constexpr float DRY_SECONDS = 45.0f;

    // ---- Block Interaction ----
    Inventory inventory;
    BlockHighlight highlightedBlock; // Authored bounds currently targeted by crosshair
    bool hasHighlightedBlock = false;

    // ---- Per-world configuration & stateful blocks ----
    // Round-tripped through metadata.json and the block-entities sidecar.
    std::string worldName;
    GameMode gameMode = GameMode::CREATIVE;
    GenerationSettings generation;
    uint64_t worldCreatedMs = 0;
    Vec3 worldSpawn{0.f, 100.f, 0.f}; // respawn anchor, distinct from playerPos
    bool bedSpawnSet = false;
    bool pendingBedSpawnValidation = false;
    SurvivalStats survival;
    EatingState eatingState;
    std::string deathMessage;
    int hurtSoundCooldown = 0;
    FurnaceMap furnaces;
    std::shared_ptr<FurnaceVisualAuthority> furnaceVisualAuthority =
        std::make_shared<FurnaceVisualAuthority>();
    std::optional<BlockPos> openFurnace; // the furnace the FURNACE screen edits
    ChestMap chests;
    std::optional<BlockPos> openChest; // the chest the CHEST screen edits
    ItemEntityManager itemEntities;
    BoatManager boats;
    int ridingBoat = -1; // index into boats, -1 when on foot
    int pickupSoundCooldown = 0;
    MiningState miningState;

    // ---- Container screens ----
    std::array<ItemStack, 9> craftGrid{}; // first 4 used on the inventory screen
    ItemStack craftResult;
    ItemStack cursorStack;
    int creativePage = 0;
    int hoveredSlot = -1;

    // ---- Slot drag and double-click (Minecraft-style) ----
    // A drag session paints every slot a held button passes over, then splits
    // (left) or spreads one-per-slot (right) the cursor across them on release.
    bool dragActive = false;
    SlotClickKind dragKind = SlotClickKind::LEFT;
    bool dragMoved = false; // painted a second distinct slot: a real drag
    std::vector<SlotRef> dragSlots;
    double lastSlotClickSeconds = -1.0; // for double-click gather detection
    SlotRef lastSlotClickRef;
    static constexpr double DOUBLE_CLICK_WINDOW = 0.3; // seconds

    // ---- Game flow & UI ----
    std::vector<WorldSummary> worldList; // cached on entering the world menus
    WorldSelectState worldSelect;
    WorldCreateState worldCreate;
    GameFlow flow;                   // Title → Playing ⇄ Paused ⇄ Settings
    bool spawnValidated = false;     // player unstuck from stale-save terrain
    SettingsValues settings;         // live values shown in the settings menu
    GraphicsSettings gfx;            // video screen values (persisted with settings)
    bool envOverridesActive = false; // RYCRAFT_* session: never save settings
    MenuLayout menuLayout;           // rebuilt each frame while a menu is open
    int hoveredButton = -1;
    bool showDebugHud = false;
    std::string startupFailure;
    std::optional<bool> startupFailureRetryable;
    bool startupFailureAllowsWorldSelect = false;
    bool startupFailureIsSave = false;
    std::optional<GameScreen> generationRecoveryReturnScreen;
    std::optional<GameScreen> saveFailureReturnScreen;
    bool diagnosticV3 = false;
    bool v4OpenRequested = false;
    bool v4EntryReady = false;
    bool v4ProfileNewlyCreated = false;
    bool v4SpawnFinalized = false;
    bool v4SpawnSafetyValidated = false;
    bool v4SpawnCandidateActive = false;
    // A continental candidate without a page-local ocean escape proof may
    // start radius-zero exact generation, but it cannot authorize horizon
    // work or metadata until World validates canonical water and collision.
    bool v4SpawnCandidateProvisional = false;
    bool v4ExactStreamingReleased = false;
    bool v4ProfileOpened = false;
    V4SpawnAuthorityPrequeueStatus v4SpawnAuthorityStatus =
        V4SpawnAuthorityPrequeueStatus::Deferred;
    uint32_t v4SpawnSafetyRevision = 0;
    uint32_t v4SpawnSearchOrdinal = 0;
    std::optional<Vec3> v4SafeSpawnPos;
    std::optional<Vec3> v4SpawnFallbackSearchOrigin;
    std::unique_ptr<V4SpawnWaterScreen> v4SpawnWaterScreen;
    std::optional<ColumnPos> v4HorizonAnchorTile;
    uint64_t v4HorizonWorldEpoch = 0;
    uint64_t v4HorizonViewEpoch = 0;
    uint32_t v4HorizonFreshFrames = 0;
    std::optional<GameScreen> requestedStartScreen;
    bool requestedStartScreenApplied = false;
    std::optional<std::filesystem::path> preferredV4ProfilePath;
    std::optional<V4WorldCreationRequest> pendingV4Creation;
    double v4EntryStartedAt = 0.0;

    // ---- Performance stats (exponential moving averages) ----
    float smoothedFrameMs = 16.7f;
    uint32_t cachedChunkCount = 0;
    uint32_t cachedPendingChunks = 0;
    PerformanceCapture performance;
    uint64_t autopilotStartFrame = 0;
    uint64_t autopilotStopFrame = std::numeric_limits<uint64_t>::max();

    // ---- Player & World ----
    Player player;
    // Reverse declaration order gives default teardown the same dependency
    // order as stopWorld: WeatherSystem, World, then SaveManager.
    std::unique_ptr<SaveManager> saveManager;
    std::shared_ptr<World> world;
    // Weather owns a non-owning ChunkGenerator reference and must stop first.
    // World generation owns a non-owning SaveManager pointer and stops next.
    std::unique_ptr<WeatherSystem> weatherSystem;
    bool weatherSessionInitialized = false;
    uint64_t generationSeed = 42;
    std::string generationFingerprint;
    Vec3 requestedSpawn = GENERATOR_V4_INITIAL_SPAWN;
    Vec3 v4SpawnSearchOrigin = GENERATOR_V4_INITIAL_SPAWN;
    std::unique_ptr<worldgen::bootstrap::TerrainModelTransport> terrainTransport;
    worldgen::bootstrap::Sha256TerrainAssetVerifier terrainVerifier;
    std::unique_ptr<worldgen::bootstrap::TerrainModelInstaller> terrainInstaller;
    std::unique_ptr<worldgen::bootstrap::TerrainRuntimePreparation> terrainRuntime;
    std::unique_ptr<worldgen::bootstrap::TerrainGenerationBootstrap> terrainBootstrap;
    std::jthread terrainBootstrapThread;
    std::atomic<bool> terrainBootstrapRunning{false};
    std::atomic<bool> terrainRepairRecovery{false};
    std::shared_ptr<worldgen::learned::WorldGenerationContext> repairedGenerationContext;
    Camera camera;

    // ---- Animals ----
    std::unique_ptr<Spawner> spawner;
    std::unordered_map<uint64_t, StateMachine> entityBrains;
    int animalCallCooldown = 0;

    // ---- Audio ----
    std::unique_ptr<AudioEngine> audio;
    std::vector<float> sfxBlockBreak;
    std::vector<float> sfxBlockPlace;
    std::vector<float> sfxFootstep;
    std::vector<float> sfxWind;
    std::vector<float> sfxRain;
    std::vector<float> sfxSnow;
    std::vector<float> sfxHurt;
    std::vector<float> sfxEat;
    std::vector<float> sfxDeath;
    std::vector<float> sfxClick;
    std::vector<float> sfxPickup;
    std::vector<float> sfxFurnacePop;
    std::array<std::vector<float>, ENTITY_TYPE_COUNT> sfxAnimal;
    int32_t windVoice = -1;
    int32_t rainVoice = -1;
    int32_t snowVoice = -1;
    float windGain = 0.0F;
    float rainGain = 0.0F;
    float snowGain = 0.0F;
    float footstepDistance = 0.f; // ground distance walked since last step
    Vec3 lastFootstepPos{0.f, 0.f, 0.f};

    // ---- Input manager (set after window creation) ----
    std::unique_ptr<InputManager> inputManager;

    ~EngineState() {
        // The bootstrap worker captures this state directly. Stop and join it
        // while every field it can access is still alive instead of relying
        // on reverse member destruction order during application teardown.
        if (terrainBootstrap)
            terrainBootstrap->cancel();
        if (terrainBootstrapThread.joinable())
            terrainBootstrapThread.join();
    }
};

void installSavedFurnaceProjection(EngineState& state) {
    if (!state.world || !state.furnaceVisualAuthority)
        return;
    const std::shared_ptr<FurnaceVisualAuthority> authority = state.furnaceVisualAuthority;
    state.world->setSavedChunkProjection({
        .apply = [authority](Chunk& chunk) { return authority->projectSavedChunk(chunk); },
        .currentRevision = [authority] { return authority->revision(); },
    });
}

void clearStartupFailure(EngineState& state) {
    state.startupFailure.clear();
    state.startupFailureRetryable.reset();
    state.startupFailureAllowsWorldSelect = false;
    state.startupFailureIsSave = false;
}

void latchStartupFailure(EngineState& state, std::string message,
                         std::optional<bool> retryable = std::nullopt,
                         bool allowWorldSelect = false, bool saveFailure = false) {
    state.startupFailure = std::move(message);
    state.startupFailureRetryable = retryable;
    state.startupFailureAllowsWorldSelect = allowWorldSelect;
    state.startupFailureIsSave = saveFailure;
}

void latchGenerationFailure(EngineState& state,
                            const std::optional<worldgen::learned::GenerationFailure>& failure,
                            std::string fallback, bool allowWorldSelect = false) {
    latchStartupFailure(
        state, failure && !failure->message.empty() ? failure->message : std::move(fallback),
        failure ? std::optional<bool>{failure->retriable} : std::nullopt, allowWorldSelect);
}

V4SpawnWaterScreenResult screenV4SpawnCandidateWater(
    EngineState& state,
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& generationContext,
    Vec3 candidate) {
    if (!state.v4SpawnWaterScreen)
        state.v4SpawnWaterScreen = std::make_unique<V4SpawnWaterScreen>();
    return state.v4SpawnWaterScreen->screen(generationContext, candidate);
}

bool advanceV4DrySpawnRecoverySearch(EngineState& state) {
    if (!state.v4SpawnFallbackSearchOrigin)
        return false;
    state.v4SpawnSearchOrigin = *state.v4SpawnFallbackSearchOrigin;
    state.v4SpawnFallbackSearchOrigin.reset();
    state.v4SpawnSearchOrdinal = 0;
    state.v4SpawnCandidateActive = false;
    state.v4SpawnCandidateProvisional = false;
    if (state.v4SpawnWaterScreen)
        state.v4SpawnWaterScreen->reset();
    return true;
}

bool retryV4DrySpawnFromFallbackAfterFailure(EngineState& state,
                                             const worldgen::learned::GenerationFailure* failure) {
    // INVALID_REQUEST is the selector's terminal "no dry coarse candidate"
    // result. Backend, persistence, and runtime errors must stay visible and
    // are never hidden behind a different recovery anchor.
    return failure && failure->code == worldgen::learned::GenerationFailureCode::INVALID_REQUEST &&
           advanceV4DrySpawnRecoverySearch(state);
}

bool retryV4DrySpawnFromFallbackAfterExhaustion(EngineState& state) {
    return advanceV4DrySpawnRecoverySearch(state);
}

SaveManager::WorldMetadata worldMetadataSnapshot(const EngineState& state, Vec3 playerPosition) {
    SaveManager::WorldMetadata metadata;
    metadata.seed = state.world ? state.world->getSeed() : state.generationSeed;
    metadata.generationFingerprint = state.generationFingerprint;
    metadata.spawnFinalized = state.v4SpawnFinalized;
    metadata.spawnSafetyRevision = state.v4SpawnSafetyRevision;
    metadata.safeSpawnPos = state.v4SafeSpawnPos;
    metadata.spawnPos = state.worldSpawn;
    metadata.bedSpawnSet = state.bedSpawnSet;
    metadata.playerPos = playerPosition;
    metadata.worldTime = state.worldTime;
    metadata.name = state.worldName;
    metadata.gameMode = state.gameMode;
    metadata.generation = state.generation;
    metadata.createdMs = state.worldCreatedMs;
    metadata.player.yaw = state.player.yaw;
    metadata.player.pitch = state.player.pitch;
    metadata.player.health = state.player.health;
    metadata.player.hunger = state.survival.food;
    metadata.player.selectedSlot = state.inventory.getSelectedIndex();
    for (int slot = 0; slot < Inventory::SLOTS; ++slot) {
        metadata.player.inventory[static_cast<size_t>(slot)] = state.inventory.getSlot(slot);
    }
    metadata.player.carriedStacks[0] = state.cursorStack;
    for (size_t cell = 0; cell < state.craftGrid.size(); ++cell) {
        metadata.player.carriedStacks[cell + 1] = state.craftGrid[cell];
    }
    return metadata;
}

void applyRequestedGameplayScreen(EngineState& state) {
    if (!state.world || !state.requestedStartScreen || state.requestedStartScreenApplied)
        return;

    state.flow.screen = *state.requestedStartScreen;
    state.requestedStartScreenApplied = true;
    if (state.inputManager) {
        if (state.flow.screen == GameScreen::PLAYING)
            state.inputManager->captureMouse();
        else
            state.inputManager->releaseMouse();
    }

    // Capture fixtures remain available after the asynchronous v4 entry
    // gate. They exercise the same inventory and block-entity paths as a
    // normally opened gameplay session.
    if (state.flow.screen == GameScreen::DEATH) {
        state.deathMessage = "YOU FELL";
    } else if (state.flow.screen == GameScreen::FURNACE) {
        const BlockPos pos{0, 0, 0};
        FurnaceState& furnace = state.furnaces[pos];
        furnace.input = ItemStack{ItemType::RAW_BEEF, 3, 0};
        furnace.fuel = ItemStack{ItemType::COAL, 5, 0};
        furnace.output = ItemStack{ItemType::COOKED_BEEF, 2, 0};
        furnace.burnTicksRemaining = 800;
        furnace.burnTicksTotal = 1600;
        furnace.cookTicks = 120;
        state.furnaceVisualAuthority->set(pos, furnaceBlockForState(furnace));
        state.openFurnace = pos;
    } else if (state.flow.screen == GameScreen::CHEST) {
        const BlockPos pos{0, 0, 0};
        ChestState& chest = state.chests[pos];
        chest.slots[0] = ItemStack{itemFromBlock(BlockType::COBBLESTONE), 64, 0};
        chest.slots[1] = ItemStack{ItemType::IRON_INGOT, 12, 0};
        chest.slots[9] = ItemStack{ItemType::COOKED_BEEF, 8, 0};
        chest.slots[13] = ItemStack{ItemType::DIAMOND, 3, 0};
        chest.slots[26] = ItemStack{ItemType::IRON_PICKAXE, 1, 131};
        state.openChest = pos;
    }
}

void initializeWeatherSession(EngineState& state, Vec3 anchor) {
    state.weatherSystem.reset();
    state.weatherSnapshot.reset();
    state.localWeather = {};
    state.lightningEvents.clear();
    state.captureLightningInjected = false;
    state.lastWeatherEventTick = state.worldTime;
    state.wetness = 0.0F;
    const WorldPhysicalScale physicalScale =
        state.world ? worldPhysicalScale(state.world->generator().usesLearnedAuthority())
                    : LEGACY_WORLD_PHYSICAL_SCALE;
    state.thunder.beginTimeline(state.worldTime, physicalScale);
    state.weatherSessionInitialized = true;
    if (!state.world || !state.generation.weather)
        return;

    state.weatherSystem = std::make_unique<WeatherSystem>(state.world->generator());
    state.weatherSystem->requestSnapshot(static_cast<int64_t>(std::floor(anchor.x)),
                                         static_cast<int64_t>(std::floor(anchor.z)),
                                         state.worldTime);
    state.localWeather = state.weatherSystem->sample(anchor.x, anchor.z, state.worldTime);

    // Fixed presets are capture and test authorities, so their material
    // state starts settled instead of visibly ramping after world creation.
    const WeatherPreset preset = state.weatherSystem->preset();
    if (preset == WeatherPreset::RAIN || preset == WeatherPreset::STORM) {
        state.wetness = 1.0F;
    }
}

@interface Engine ()
- (void)startTerrainBootstrapWithRetry:(BOOL)retry repair:(BOOL)repair;
- (BOOL)saveCurrentWorld;
- (BOOL)stopWorld;
- (void)releaseTerrainGenerationOwners;
- (void)releaseTerrainRuntime;
- (void)quiesceTerrainRuntimeForProfileDeletion;
- (BOOL)quiesceForApplicationTerminationRequiringSave:(BOOL)requireSave;
- (void)reconcileFurnaceBlocks:(EngineState*)state;
- (void)reconcileChestBlocks:(EngineState*)state;
- (void)updateDynamicObjectLighting:(EngineState*)state;
@end

@implementation Engine {
    NSApplication* _app;
    NSWindow* _window;
    MTKView* _view;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;

    // Render pipeline (created after device/queue + shader library)
    std::unique_ptr<RenderPipeline> _renderPipeline;

    // C++ game state
    std::unique_ptr<EngineState> _state;

    // Scroll wheel tracking
    float _scrollAccumulator;

    // Save-on-quit guard (terminate can be reached from several paths)
    bool _savedWorld;
    ApplicationTerminationQuiescence _terminationQuiescence;
}

// ---- C++ bridge helper, must be inside @implementation to access _state ----
static EngineState* _engineGetState(Engine* engine) {
    return engine->_state.get();
}

+ (instancetype)sharedEngine {
    static Engine* shared = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        shared = [[Engine alloc] init];
    });
    return shared;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _app = nil;
        _window = nil;
        _view = nil;
        _device = nil;
        _queue = nil;
        _scrollAccumulator = 0;
        _savedWorld = NO;
        _terminationQuiescence.resetForWorldSession();
        _state = std::make_unique<EngineState>();
        _state->performance.requestedFrames =
            std::min<uint64_t>(unsignedEnvironmentValue("RYCRAFT_PERF_FRAMES", 0), 36'000);
        _state->performance.warmupFrames =
            unsignedEnvironmentValue("RYCRAFT_PERF_WARMUP_FRAMES", 600);
        if (_state->performance.enabled()) {
            _state->performance.frameMilliseconds.reserve(
                static_cast<size_t>(_state->performance.requestedFrames));
            _state->performance.fixedTickMilliseconds.reserve(
                static_cast<size_t>(_state->performance.requestedFrames));
        }
        _state->autopilotStartFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPILOT_START_FRAME", 0);
        _state->autopilotStopFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPILOT_STOP_FRAME",
                                                              std::numeric_limits<uint64_t>::max());
        _state->autoPauseFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPAUSE_FRAME",
                                                          std::numeric_limits<uint64_t>::max());
        uint64_t seed = 42;
        Vec3 spawnPos = GENERATOR_V4_INITIAL_SPAWN;
        const char* seedEnv = std::getenv("RYCRAFT_WORLD_SEED");
        const char* spawnEnv = std::getenv("RYCRAFT_SPAWN");
        if (seedEnv) {
            seed = std::strtoull(seedEnv, nullptr, 0);
        }
        if (spawnEnv) {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            if (std::sscanf(spawnEnv, "%f,%f,%f", &x, &y, &z) == 3)
                spawnPos = {x, y, z};
        }

        // Gameplay launch hooks select one exact v4 profile. Resolve relative
        // values beneath Application Support and use its immutable seed for
        // runtime qualification unless the caller explicitly supplied a seed
        // override. Merely setting this hook never starts a title-screen
        // world session or creates a profile.
        const char* worldDirectoryEnv = std::getenv("RYCRAFT_WORLD_DIR");
        if (worldDirectoryEnv && *worldDirectoryEnv) {
            _state->preferredV4ProfilePath = resolveV4LaunchProfilePath(
                worldgen::bootstrap::defaultRycraftApplicationSupportPath(), worldDirectoryEnv);
            if (_state->preferredV4ProfilePath) {
                const std::optional<SaveManager::WorldMetadata> selectedMetadata =
                    SaveManager::readMetadataFile(
                        (*_state->preferredV4ProfilePath / "metadata.json").string());
                if (selectedMetadata &&
                    selectedMetadata->generatorVersion == SaveManager::GENERATOR_V4_VERSION) {
                    if (!seedEnv)
                        seed = selectedMetadata->seed;
                    if (!spawnEnv) {
                        spawnPos =
                            selectedMetadata->safeSpawnPos.value_or(selectedMetadata->playerPos);
                    }
                }
            }
        }

        // Playtest hook: pin absolute saved world time. Values beyond one day
        // select a repeatable lunar phase while the remainder selects time of
        // day (6000 = noon).
        if (const char* timeEnv = std::getenv("RYCRAFT_TIME")) {
            if (const auto parsed = parseUnsignedDecimal(timeEnv)) {
                _state->worldTime = *parsed;
            } else {
                RY_LOG_ERROR("Ignoring invalid RYCRAFT_TIME; expected unsigned decimal ticks");
            }
        }

        // Playtest hook: RYCRAFT_CAPTURE plus RYCRAFT_CAPTURE_LIGHTNING
        // injects one deterministic strike once the weather snapshot is
        // ready; parsed at launch so an invalid override fails loudly.
        const char* capturePath = std::getenv("RYCRAFT_CAPTURE");
        const char* captureLightning = std::getenv("RYCRAFT_CAPTURE_LIGHTNING");
        if (capturePath && *capturePath && captureLightning && *captureLightning) {
            _state->captureLightningOverride = parseCaptureLightningOverride(captureLightning);
            if (!_state->captureLightningOverride) {
                RY_LOG_ERROR("Ignoring invalid RYCRAFT_CAPTURE_LIGHTNING; expected "
                             "x,z,id,ageTicks");
            }
        }

        // Persisted settings load before any World exists (view distance
        // feeds its constructor); env overrides win over the file for
        // headless playtests. Playtest hook: RYCRAFT_VIEW_DISTANCE=<4..512>.
        // An env-overridden session never saves settings, a playtest run
        // must not rewrite the user's file with its overrides.
        LoadedSettings loaded = loadSettings(settingsPath());
        _state->settings = loaded.values;
        _state->gfx = loaded.gfx;
        _state->envOverridesActive = _state->gfx.applyEnvOverrides();
        if (const char* vdEnv = std::getenv("RYCRAFT_VIEW_DISTANCE")) {
            _state->settings.viewDistance =
                std::clamp(std::atoi(vdEnv), SettingsValues::MIN_VIEW_DISTANCE,
                           SettingsValues::MAX_VIEW_DISTANCE);
            _state->envOverridesActive = true;
        }
        if (const char* overlayEnv = std::getenv("RYCRAFT_WORLDGEN_OVERLAY")) {
            _state->showDebugHud = *overlayEnv != '\0';
        }
        if (const char* debugEnv = std::getenv("RYCRAFT_SHOW_DEBUG")) {
            _state->showDebugHud = *debugEnv != '\0' && std::strcmp(debugEnv, "0") != 0;
        }
        _state->generationSeed = seed;
        _state->requestedSpawn = spawnPos;
        _state->player.position = spawnPos;
        const char* diagnostic = std::getenv("RYCRAFT_DIAGNOSTIC_V3");
        _state->diagnosticV3 =
            diagnostic != nullptr && *diagnostic != '\0' && std::strcmp(diagnostic, "0") != 0;
        if (_state->diagnosticV3) {
            // The legacy generator is available only as an explicit no-save
            // diagnostic. It never opens or modifies rycraft_world.
            _state->world = std::make_shared<World>(seed, _state->settings.viewDistance);
            _state->spawner = std::make_unique<Spawner>(*_state->world);
            initializeWeatherSession(*_state, spawnPos);
        } else {
            _state->terrainTransport = worldgen::bootstrap::makeAppleTerrainModelTransport();
            _state->terrainInstaller = std::make_unique<worldgen::bootstrap::TerrainModelInstaller>(
                worldgen::bootstrap::defaultRycraftApplicationSupportPath(),
                *_state->terrainTransport, _state->terrainVerifier);
            _state->terrainRuntime =
                worldgen::runtime::makeProductionTerrainRuntime(_state->generationSeed);
            _state->terrainBootstrap =
                std::make_unique<worldgen::bootstrap::TerrainGenerationBootstrap>(
                    *_state->terrainInstaller, *_state->terrainRuntime);
        }
    }
    return self;
}

- (void)dealloc {
    // The normal AppKit path quiesces before exit. Retain a defensive owner
    // fallback for tests, initialization failures, and any future embedding
    // that releases Engine without first asking the application to terminate.
    if (_state && !_terminationQuiescence.quiesced() &&
        ![self quiesceForApplicationTerminationRequiringSave:YES]) {
        RY_LOG_ERROR("Engine destruction is forcing runtime quiescence after a save failure");
        static_cast<void>([self quiesceForApplicationTerminationRequiringSave:NO]);
    }
#if !__has_feature(objc_arc)
    [super dealloc];
#endif
}

// A legacy generator or an older v4 fingerprint can be used only as an
// explicit source for a new profile. Comparing the pure identity here needs
// no model load and prevents an obsolete profile from first failing after the
// player has already waited for runtime qualification.
static bool worldRequiresCurrentV4Successor(const WorldSummary& world) {
    if (world.requiresGeneratorV4Successor() ||
        world.metadata.chunkFormatVersion != CHUNK_VERSION) {
        return true;
    }
    const worldgen::learned::GenerationIdentity identity =
        worldgen::runtime::productionGenerationIdentity(world.metadata.seed);
    return !identity.valid() || world.metadata.generationFingerprint !=
                                    worldgen::learned::sha256Hex(identity.fingerprint());
}

// Build the live world session from a world directory. The caller drives the
// screen flow (onWorldStarted) only after this succeeds.
- (BOOL)startWorldAtPath:(const std::string&)worldDir {
    EngineState* state = _state.get();
    if (state->world) {
        RY_LOG_ERROR("A world session is already live");
        return NO;
    }

    state->saveManager = std::make_unique<SaveManager>(worldDir);

    // Per-session defaults, then the saved world when one exists.
    uint64_t seed = 42;
    Vec3 spawnPos{0.f, 100.f, 0.f};
    Vec3 playerPos = spawnPos;
    SaveManager::PlayerMetadata playerMeta{};
    state->worldTime = 0;
    state->worldName.clear();
    state->gameMode = GameMode::CREATIVE;
    state->generation = GenerationSettings{};
    state->worldCreatedMs = 0;
    if (auto meta = state->saveManager->loadMetadata()) {
        seed = meta->seed;
        spawnPos = meta->spawnPos;
        playerPos = meta->playerPos;
        state->worldTime = meta->worldTime;
        state->worldName = meta->name;
        state->gameMode = meta->gameMode;
        state->generation = meta->generation;
        state->worldCreatedMs = meta->createdMs;
        playerMeta = meta->player;
    }
    if (const char* seedEnv = std::getenv("RYCRAFT_WORLD_SEED")) {
        seed = std::strtoull(seedEnv, nullptr, 0);
    }
    if (const char* spawnEnv = std::getenv("RYCRAFT_SPAWN")) {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (std::sscanf(spawnEnv, "%f,%f,%f", &x, &y, &z) == 3) {
            spawnPos = {x, y, z};
            playerPos = spawnPos;
        }
    }
    // Playtest hook: pin absolute saved world time. Values beyond one day
    // select a repeatable lunar phase while the remainder selects time of
    // day (6000 = noon).
    if (const char* timeEnv = std::getenv("RYCRAFT_TIME")) {
        if (const auto parsed = parseUnsignedDecimal(timeEnv)) {
            state->worldTime = *parsed;
        } else {
            RY_LOG_ERROR("Ignoring invalid RYCRAFT_TIME; expected unsigned decimal ticks");
        }
    }
    // Playtest hook: force the mode without touching the saved metadata
    // (env-override capture sessions never save).
    if (const char* modeEnv = std::getenv("RYCRAFT_GAME_MODE")) {
        if (std::strcmp(modeEnv, "survival") == 0) {
            state->gameMode = GameMode::SURVIVAL;
        } else if (std::strcmp(modeEnv, "creative") == 0) {
            state->gameMode = GameMode::CREATIVE;
        }
    }

    state->worldSpawn = spawnPos;
    state->player = Player{};
    state->player.position = playerPos;
    state->player.yaw = playerMeta.yaw;
    state->player.pitch = playerMeta.pitch;
    state->player.health = playerMeta.health;
    state->camera.setLook(playerMeta.yaw, playerMeta.pitch);
    state->survival = SurvivalStats{};
    state->survival.food = playerMeta.hunger;
    state->inventory.clear();
    state->inventory.selectSlot(playerMeta.selectedSlot);
    for (size_t slot = 0; slot < SaveManager::PLAYER_INVENTORY_SLOTS; ++slot) {
        state->inventory.setSlot(static_cast<int>(slot), playerMeta.inventory[slot]);
    }
    SaveManager::BlockEntities blockEntities = state->saveManager->loadBlockEntities();
    state->furnaces = std::move(blockEntities.furnaces);
    state->furnaceVisualAuthority->replace(state->furnaces);
    state->chests = std::move(blockEntities.chests);
    state->openFurnace.reset();
    state->openChest.reset();
    state->itemEntities.clear();
    state->boats.clear();
    state->ridingBoat = -1;
    state->pickupSoundCooldown = 0;
    state->miningState.reset();
    state->eatingState.reset();
    state->deathMessage.clear();
    state->craftGrid.fill(ItemStack{});
    state->craftResult.clear();
    state->cursorStack.clear();
    state->creativePage = 0;
    state->hoveredSlot = -1;
    state->entityBrains.clear();
    state->spawnValidated = false;
    state->hasHighlightedBlock = false;
    state->wetness = 0.0f;
    _scrollAccumulator = 0;
    _savedWorld = NO;
    _terminationQuiescence.resetForWorldSession();

    state->world = std::make_shared<World>(seed, state->settings.viewDistance, MAX_LOADED_CUBES,
                                           state->generation);
    installSavedFurnaceProjection(*state);
    // Chunks load from disk before regenerating, so block edits persist
    state->world->setSaveManager(state->saveManager.get());
    if (state->generation.fauna) {
        state->spawner = std::make_unique<Spawner>(*state->world);
    }
    initializeWeatherSession(*state, playerPos);
    return YES;
}

// Persist the live session: sweep edited cubes, write metadata and block
// entities, and drain the save queue. Capture-run gating stays with the
// callers so quit and world-switch share one body.
- (BOOL)saveCurrentWorld {
    EngineState* state = _state.get();
    if (!state->saveManager || !state->world)
        return YES;
    [self reconcileFurnaceBlocks:state];
    [self reconcileChestBlocks:state];
    // Edited chunks persist on unload; this path sweeps the rest
    const bool frontiersSaved = state->world->saveModifiedChunks();
    SaveManager::WorldMetadata worldMetadata =
        worldMetadataSnapshot(*state, state->player.position);
    const bool metadataSaved = state->saveManager->saveMetadata(worldMetadata);
    const bool blockEntitiesSaved =
        state->saveManager->saveBlockEntities(state->furnaces, state->chests);
    const bool cubesSaved = state->saveManager->flush();
    if (frontiersSaved && metadataSaved && blockEntitiesSaved && cubesSaved) {
        RY_LOG_INFO("World state saved");
        if (state->startupFailureIsSave) {
            const std::optional<GameScreen> returnScreen = state->saveFailureReturnScreen;
            state->saveFailureReturnScreen.reset();
            clearStartupFailure(*state);
            if (returnScreen)
                [self applyFlowEffects:state->flow.onGenerationRecovered(*returnScreen)];
        }
        return YES;
    } else {
        RY_LOG_ERROR("World state save did not complete");
        if (!state->saveFailureReturnScreen && state->flow.worldScreens())
            state->saveFailureReturnScreen = state->flow.screen;
        if (state->flow.worldScreens()) {
            state->flow.screen = GameScreen::TITLE;
            if (state->inputManager)
                state->inputManager->releaseMouse();
            state->accumulator = 0.0;
        }
        latchStartupFailure(*state, "World state could not be saved", true, false, true);
        return NO;
    }
}

// Save and tear down the live session, returning to the title. Order
// matters: the renderer detaches first (its mesh scheduler captured a
// const World&), then the Spawner (holds World&), then the World (joins
// generation workers), then the SaveManager (joins the save thread).
- (BOOL)stopWorld {
    EngineState* state = _state.get();
    const bool ownsPendingV4Session = state->v4OpenRequested || state->v4ProfileOpened;
    if (!state->world && !state->saveManager && !ownsPendingV4Session)
        return YES;
    const bool wasGeneratorV4 =
        !state->diagnosticV3 &&
        (ownsPendingV4Session || (state->saveManager && state->saveManager->profile() ==
                                                            SaveManager::Profile::GeneratorV4));
    if (state->world && !_savedWorld && !std::getenv("RYCRAFT_CAPTURE")) {
        if (![self saveCurrentWorld])
            return NO;
    }
    if (_renderPipeline) {
        _renderPipeline->endWorldSession();
    }
    state->spawner.reset();
    state->entityBrains.clear();
    state->weatherSystem.reset();
    state->weatherSessionInitialized = false;
    state->world.reset();
    state->saveManager.reset();
    state->furnaces.clear();
    state->furnaceVisualAuthority->replace(state->furnaces);
    state->openFurnace.reset();
    state->chests.clear();
    state->openChest.reset();
    state->itemEntities.clear();
    state->boats.clear();
    state->ridingBoat = -1;
    state->miningState.reset();
    state->inventory.clear();
    state->craftGrid.fill(ItemStack{});
    state->craftResult.clear();
    state->cursorStack.clear();
    state->hasHighlightedBlock = false;
    state->worldName.clear();
    state->bedSpawnSet = false;
    state->pendingBedSpawnValidation = false;
    if (state->v4SpawnWaterScreen)
        state->v4SpawnWaterScreen->reset();
    if (wasGeneratorV4) {
        state->v4OpenRequested = false;
        state->v4ProfileOpened = false;
        state->v4ProfileNewlyCreated = false;
        state->v4EntryReady = false;
        state->v4SpawnCandidateActive = false;
        state->v4SpawnCandidateProvisional = false;
        state->v4ExactStreamingReleased = false;
        state->requestedStartScreen.reset();
        state->requestedStartScreenApplied = false;
        state->preferredV4ProfilePath.reset();
        state->pendingV4Creation.reset();
        state->terrainRepairRecovery.store(false, std::memory_order_release);
        state->repairedGenerationContext.reset();
    }
    state->generationRecoveryReturnScreen.reset();
    state->saveFailureReturnScreen.reset();
    clearStartupFailure(*state);
    [self applyFlowEffects:state->flow.onWorldStopped()];
    return YES;
}

// Deleting a profile is stronger than leaving it. Model and hydrology workers
// can outlive World requests through the qualified runtime context and may
// still publish immutable pages. Destroy every owner of that context before
// remove_all so no background publication can recreate the deleted profile.
- (void)releaseTerrainGenerationOwners {
    EngineState* state = _state.get();
    if (state->terrainBootstrap)
        state->terrainBootstrap->cancel();
    if (state->terrainBootstrapThread.joinable())
        state->terrainBootstrapThread.join();
    state->terrainBootstrapRunning.store(false, std::memory_order_release);
    state->v4SpawnWaterScreen.reset();
    state->repairedGenerationContext.reset();
    state->terrainRepairRecovery.store(false, std::memory_order_release);
    state->terrainBootstrap.reset();
}

- (void)releaseTerrainRuntime {
    EngineState* state = _state.get();
    state->terrainRuntime.reset();
}

- (void)quiesceTerrainRuntimeForProfileDeletion {
    [self releaseTerrainGenerationOwners];
    [self releaseTerrainRuntime];
}

- (BOOL)initialize {
    // 1. Create NSApplication
    _app = [NSApplication sharedApplication];
    if (!_app) {
        RY_LOG_ERROR("Failed to create NSApplication");
        return NO;
    }
    [_app setActivationPolicy:NSApplicationActivationPolicyRegular];
    [_app activateIgnoringOtherApps:true];
    [_app setDelegate:self];

    // 2. Create Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        RY_LOG_ERROR("No Metal-capable device found; Metal is not supported on this hardware");
        return NO;
    }

    // 3. Create command queue
    _queue = [_device newCommandQueue];
    if (!_queue) {
        RY_LOG_ERROR("Failed to create Metal command queue");
        return NO;
    }

    // 4. Create NSWindow
    const bool nativeWindow = [] {
        const char* value = std::getenv("RYCRAFT_NATIVE_WINDOW");
        return value && *value && std::strcmp(value, "0") != 0;
    }();
    NSRect screenFrame =
        nativeWindow ? [[NSScreen mainScreen] frame] : [[NSScreen mainScreen] visibleFrame];
    const CGFloat windowWidth = nativeWindow ? screenFrame.size.width : 1024.0;
    const CGFloat windowHeight = nativeWindow ? screenFrame.size.height : 768.0;
    NSRect windowRect =
        NSMakeRect(screenFrame.origin.x + (screenFrame.size.width - windowWidth) * 0.5,
                   screenFrame.origin.y + (screenFrame.size.height - windowHeight) * 0.5,
                   windowWidth, windowHeight);
    const NSWindowStyleMask styleMask =
        nativeWindow ? NSWindowStyleMaskBorderless
                     : NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    _window = [[NSWindow alloc] initWithContentRect:windowRect
                                          styleMask:styleMask
                                            backing:NSBackingStoreBuffered
                                              defer:false];
    if (!_window) {
        RY_LOG_ERROR("Failed to create NSWindow");
        return NO;
    }
    [_window setTitle:@"rycraft"];
    [_window setDelegate:self];
    [_window makeKeyAndOrderFront:nil];

    // 5. Create and configure MTKView
    _view = [[MTKView alloc] initWithFrame:NSMakeRect(0.0, 0.0, windowWidth, windowHeight)
                                    device:_device];
    if (!_view) {
        RY_LOG_ERROR("Failed to create MTKView");
        return NO;
    }

    // Drawable pixel format. The render pipeline builds its own MSAA render
    // passes, so the view carries no sample count or depth buffer of its own.
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _view.preferredFramesPerSecond = 120;

    // Disable automatic setNeedsDisplay, we drive rendering from the game loop
    _view.enableSetNeedsDisplay = false;
    _view.framebufferOnly = false;

    // Delegate
    _view.delegate = self;

    // Set as window content
    [_window setContentView:_view];

    // 7. Create InputManager (the game opens on the title screen with a
    // free cursor; clicking PLAY captures the mouse)
    _state->inputManager = std::make_unique<InputManager>(_window);

    // 6. Load shader library and create render pipeline
    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dirPath = [exePath stringByDeletingLastPathComponent];
    NSString* libPath = [dirPath stringByAppendingPathComponent:@"pipeline.metallib"];
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    NSError* libError = nil;
    id<MTLLibrary> library = [_device newLibraryWithURL:libURL error:&libError];
    if (!library || libError) {
        NSString* msg = [NSString
            stringWithFormat:@"Failed to load shader library: %@", libError.localizedDescription];
        RY_LOG_ERROR([msg UTF8String]);
        return NO;
    }

    // No world session exists yet; the first render's resetFarTerrain and
    // endWorldSession own per-world identity from here on.
    _renderPipeline = std::make_unique<RenderPipeline>(
        _device, library, static_cast<uint32_t>(_view.bounds.size.width),
        static_cast<uint32_t>(_view.bounds.size.height));

    // Apply the persisted settings to the live systems (world view distance
    // was already applied through the World constructor; RYCRAFT_BLOOM rides
    // GraphicsSettings::applyEnvOverrides now).
    _renderPipeline->setGraphicsSettings(_state->gfx);
    _renderPipeline->setFogDensity(fogDensityForLevel(_state->settings.fogLevel));
    _state->camera.setMouseSensitivity(mouseSensitivityForLevel(_state->settings.sensitivityLevel));

    // Keep gameplay screen requests pending while generator v4 verifies its
    // model, prepares a dry spawn, and fills the entry horizon. Menu-only
    // requests remain immediately available without constructing a world.
    const char* screenEnv = std::getenv("RYCRAFT_START_SCREEN");
    const std::string screenName = screenEnv ? screenEnv : "";
    _state->requestedStartScreen = gameScreenFromEnvironment(screenName);
    const bool automaticGameplay =
        screenName.empty() && (std::getenv("RYCRAFT_CAPTURE") || std::getenv("RYCRAFT_AUTOPILOT") ||
                               _state->performance.enabled());
    if (automaticGameplay)
        _state->requestedStartScreen = GameScreen::PLAYING;
    if (launchRequestsWorldSession(_state->requestedStartScreen, automaticGameplay)) {
        _state->v4OpenRequested = true;
    }

    if (screenName == "worlds" || screenName == "delete") {
        _state->v4OpenRequested = false;
        _state->worldList =
            _state->diagnosticV3
                ? listWorlds()
                : listWorldsForGeneratorV4(
                      worldgen::bootstrap::defaultRycraftApplicationSupportPath().string());
        if (screenName == "delete" && !_state->worldList.empty()) {
            _state->worldSelect.selected = 0;
            _state->flow.screen = GameScreen::WORLD_DELETE_CONFIRM;
        } else {
            _state->flow.screen = GameScreen::WORLD_SELECT;
        }
    } else if (screenName == "create") {
        _state->v4OpenRequested = false;
        _state->flow.screen = GameScreen::WORLD_CREATE;
    }
    if (_state->diagnosticV3)
        applyRequestedGameplayScreen(*_state);

    // 8. Audio: non-fatal on failure, the game is fully playable silent
    _state->audio = std::make_unique<AudioEngine>();
    if (_state->audio->initialize()) {
        _state->sfxBlockBreak = SoundEffect::generateBlockBreak();
        _state->sfxBlockPlace = SoundEffect::generateBlockPlace();
        _state->sfxFootstep = SoundEffect::generateFootstep();
        _state->sfxWind = SoundEffect::generateAmbientWind();
        _state->sfxRain = SoundEffect::generateRainAmbience();
        _state->sfxSnow = SoundEffect::generateSnowAmbience();
        _state->sfxHurt = SoundEffect::generateHurt();
        _state->sfxEat = SoundEffect::generateEat();
        _state->sfxDeath = SoundEffect::generateDeath();
        _state->sfxClick = SoundEffect::generateClick();
        _state->sfxPickup = SoundEffect::generatePickup();
        _state->sfxFurnacePop = SoundEffect::generateFurnacePop();
        for (size_t index = 0; index < ENTITY_TYPE_COUNT; ++index) {
            _state->sfxAnimal[index] =
                SoundEffect::generateAnimalCall(static_cast<EntityType>(index));
        }
        [self syncAudioVolume];
    } else {
        RY_LOG_ERROR("Audio engine failed to initialize, continuing without sound");
        _state->audio.reset();
    }

    // A previously installed pack is verified and loaded without presenting
    // another download action. Missing packs remain user-initiated so the
    // initial multi-gigabyte transfer is never surprising.
    if (_state->v4OpenRequested && _state->terrainInstaller &&
        _state->terrainInstaller->hasInstalledPackCandidate()) {
        [self startTerrainBootstrapWithRetry:NO repair:NO];
    }

    RY_LOG_INFO(std::string("Engine initialized - window: ") +
                std::to_string(static_cast<int>(_view.bounds.size.width)) + "x" +
                std::to_string(static_cast<int>(_view.bounds.size.height)) +
                ", drawable: " + std::to_string(static_cast<int>(_view.drawableSize.width)) + "x" +
                std::to_string(static_cast<int>(_view.drawableSize.height)) +
                ", device: " + std::string([[_device name] UTF8String]));

    return YES;
}

- (void)run {
    [_app run];
}

- (void)terminate {
    if (_app) {
        [_app terminate:nil];
    }
}

- (void)startTerrainBootstrapWithRetry:(BOOL)retry repair:(BOOL)repair {
    EngineState* state = _state.get();
    if (!state->terrainBootstrap || state->terrainBootstrapRunning.exchange(true))
        return;
    if (state->terrainBootstrapThread.joinable())
        state->terrainBootstrapThread.join();
    state->terrainBootstrapThread = std::jthread([state, retry, repair] {
        if (repair) {
            state->terrainBootstrap->repair();
        } else if (retry) {
            state->terrainBootstrap->retry();
        } else {
            state->terrainBootstrap->run();
        }
        state->terrainBootstrapRunning.store(false, std::memory_order_release);
    });
}

- (void)openReadyV4World {
    EngineState* state = _state.get();
    if (!state->v4OpenRequested || state->world || !state->terrainBootstrap ||
        !state->terrainBootstrap->ready() || !state->startupFailure.empty()) {
        return;
    }
    if (!state->v4ProfileOpened) {
        V4WorldOpenResult opened = openQualifiedV4World(
            *state->terrainBootstrap, state->generationSeed, state->requestedSpawn,
            state->worldTime, nullptr, state->preferredV4ProfilePath, state->pendingV4Creation);
        if (!opened.ready()) {
            latchStartupFailure(*state, opened.message, v4WorldOpenFailureRetryable(opened.status),
                                v4WorldOpenFailureAllowsWorldSelection(opened.status));
            return;
        }

        if (opened.usingSeparateProfile && !opened.message.empty()) {
            RY_LOG_INFO(opened.message.c_str());
        }
        RY_LOG_INFO(("Generator v4 persistence profile: " + opened.profilePath.string()).c_str());

        state->saveManager = std::move(opened.saveManager);
        state->preferredV4ProfilePath = opened.profilePath;
        state->pendingV4Creation.reset();
        state->v4ProfileNewlyCreated = opened.newlyCreated;
        state->generationFingerprint = opened.metadata.generationFingerprint;
        state->generationSeed = opened.metadata.seed;
        state->v4SpawnFinalized = opened.metadata.spawnFinalized;
        state->v4SpawnSafetyRevision = opened.metadata.spawnSafetyRevision;
        state->v4SafeSpawnPos = opened.metadata.safeSpawnPos;
        const bool requiresStrictDryValidation = v4SpawnRequiresStrictDryValidation(
            state->v4SpawnFinalized, state->v4SpawnSafetyRevision,
            state->v4SafeSpawnPos.has_value());
        const V4DrySpawnRecoverySearch recoverySearch = v4DrySpawnRecoverySearch(
            state->v4SafeSpawnPos, state->requestedSpawn, requiresStrictDryValidation);
        state->v4SpawnSearchOrigin = recoverySearch.primary;
        state->v4SpawnFallbackSearchOrigin = recoverySearch.fallback;
        state->v4SpawnSafetyValidated = !requiresStrictDryValidation;
        // A record written before the safe-spawn field was introduced must
        // remain provisional until final authority validates a replacement.
        // Clearing the stale field also keeps an interrupted migration from
        // later claiming the old location was checked under this revision.
        if (!state->v4SpawnSafetyValidated) {
            state->v4SpawnFinalized = false;
            state->v4SafeSpawnPos.reset();
        }
        state->v4SpawnCandidateActive = false;
        state->v4SpawnCandidateProvisional = false;
        state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Deferred;
        state->v4SpawnSearchOrdinal = 0;
        state->v4HorizonAnchorTile.reset();
        state->v4HorizonWorldEpoch = 0;
        state->v4HorizonViewEpoch = 0;
        state->v4HorizonFreshFrames = 0;
        state->worldTime = opened.metadata.worldTime;
        if (const char* timeEnv = std::getenv("RYCRAFT_TIME")) {
            if (const auto parsed = parseUnsignedDecimal(timeEnv)) {
                state->worldTime = *parsed;
            }
        }
        state->worldName = opened.metadata.name;
        state->gameMode = opened.metadata.gameMode;
        state->generation = opened.metadata.generation;
        state->worldCreatedMs = opened.metadata.createdMs;
        state->worldSpawn = opened.metadata.spawnPos;
        state->bedSpawnSet = opened.metadata.bedSpawnSet;
        state->player.position = opened.metadata.playerPos;
        state->player.yaw = opened.metadata.player.yaw;
        state->player.pitch = opened.metadata.player.pitch;
        state->player.health = opened.metadata.player.health;
        state->camera.setLook(opened.metadata.player.yaw, opened.metadata.player.pitch);
        state->survival = SurvivalStats{};
        state->survival.food = opened.metadata.player.hunger;
        state->inventory.clear();
        state->inventory.selectSlot(opened.metadata.player.selectedSlot);
        for (int slot = 0; slot < Inventory::SLOTS; ++slot) {
            state->inventory.setSlot(slot,
                                     opened.metadata.player.inventory[static_cast<size_t>(slot)]);
        }
        state->cursorStack = opened.metadata.player.carriedStacks[0];
        for (size_t cell = 0; cell < state->craftGrid.size(); ++cell) {
            state->craftGrid[cell] = opened.metadata.player.carriedStacks[cell + 1];
        }
        SaveManager::BlockEntities blockEntities = state->saveManager->loadBlockEntities();
        state->furnaces = std::move(blockEntities.furnaces);
        state->furnaceVisualAuthority->replace(state->furnaces);
        state->chests = std::move(blockEntities.chests);
        state->openFurnace.reset();
        state->openChest.reset();
        state->itemEntities.clear();
        state->boats.clear();
        state->ridingBoat = -1;
        state->pickupSoundCooldown = 0;
        state->miningState.reset();
        state->eatingState.reset();
        state->deathMessage.clear();
        state->craftResult.clear();
        state->creativePage = 0;
        state->hoveredSlot = -1;
        state->entityBrains.clear();
        state->spawnValidated = false;
        state->pendingBedSpawnValidation = false;
        state->hasHighlightedBlock = false;
        state->wetness = 0.0f;
        _scrollAccumulator = 0;
        _savedWorld = NO;
        _terminationQuiescence.resetForWorldSession();
        state->v4ProfileOpened = true;
        // Include nonpersistent coarse dry-land selection in the visible
        // preparation interval. It is deliberately completed before a World
        // or far horizon exists, so resetting this clock at world creation
        // would otherwise make a slow selector look like an inert READY UI.
        state->v4EntryStartedAt = CACurrentMediaTime();
    }
    const std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext =
        state->terrainBootstrap->qualifiedGenerationContext();
    if (!generationContext) {
        latchStartupFailure(*state, "The qualified generator v4 authority is unavailable", true);
        state->saveManager.reset();
        state->v4ProfileOpened = false;
        state->v4ProfileNewlyCreated = false;
        return;
    }

    // A fresh world, or a profile created before strict dry-spawn validation,
    // must not build its exact spawn band around an arbitrary ocean coordinate.
    // A coarse learned selector proposes an inland candidate without blocking
    // the render thread; final cubes validate it below before metadata is finalized.
    if (!state->v4SpawnSafetyValidated && !state->v4SpawnCandidateActive) {
        const auto selected = findV4DryLandSpawnCandidate(
            generationContext, static_cast<int64_t>(std::floor(state->v4SpawnSearchOrigin.x)),
            static_cast<int64_t>(std::floor(state->v4SpawnSearchOrigin.z)),
            state->v4SpawnSearchOrdinal);
        if (selected.status() == worldgen::learned::AuthorityStatus::FAILED) {
            if (retryV4DrySpawnFromFallbackAfterFailure(*state, selected.failure()))
                return;
            const std::optional<worldgen::learned::GenerationFailure> failure =
                selected.failure() ? std::optional{*selected.failure()} : std::nullopt;
            latchGenerationFailure(*state, failure, "A dry generator v4 spawn could not be located",
                                   failure && !failure->retriable);
            return;
        }
        if (!selected.isReady())
            return;
        if (!selected.value() || !*selected.value()) {
            ++state->v4SpawnSearchOrdinal;
            if (state->v4SpawnSearchOrdinal >= V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES) {
                if (retryV4DrySpawnFromFallbackAfterExhaustion(*state))
                    return;
                latchStartupFailure(*state,
                                    "No dry generator v4 spawn was found in the bounded "
                                    "learned-terrain search",
                                    false, true);
            }
            return;
        }
        const Vec3 candidate = **selected.value();
        const V4SpawnWaterScreenResult waterScreen =
            screenV4SpawnCandidateWater(*state, generationContext, candidate);
        if (waterScreen.failed()) {
            latchGenerationFailure(*state, waterScreen.failure,
                                   "The canonical safe-spawn water screen failed");
            return;
        }
        if (waterScreen.deferred())
            return;
        if (waterScreen.water()) {
            // Do not allocate the cold exact band for a proposal without a
            // locally safe FINAL canonical dry site. Advancing only here
            // keeps ordinal order stable across retries.
            ++state->v4SpawnSearchOrdinal;
            state->v4SpawnWaterScreen->reset();
            if (state->v4SpawnSearchOrdinal >= V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES) {
                if (retryV4DrySpawnFromFallbackAfterExhaustion(*state))
                    return;
                latchStartupFailure(*state,
                                    "No dry generator v4 spawn was found in the bounded "
                                    "learned-terrain search",
                                    false, true);
            }
            return;
        }
        const Vec3 resolvedCandidate = waterScreen.resolvedCandidate.value_or(candidate);
        state->v4SpawnWaterScreen->reset();
        state->player.position = resolvedCandidate;
        state->requestedSpawn = state->player.position;
        state->v4SpawnCandidateActive = true;
        state->v4SpawnCandidateProvisional = waterScreen.provisionalLearnedDry;
    }
    // A canonically proven candidate owns one FINAL hydrology page and a
    // local 5x5 dry certificate. A continental provisional candidate owns
    // FINAL learned terrain only and remains gated on radius-zero exact
    // validation. Construct the World now so that exact validation can run.
    state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Ready;
    state->world = std::make_shared<World>(state->generationSeed, state->settings.viewDistance,
                                           MAX_LOADED_CUBES, generationContext, state->generation);
    installSavedFurnaceProjection(*state);
    // Keep all requested far terrain visible, but establish collision and a
    // safe spawn from the bounded exploration band before starting the full
    // exact cubic disk. The latter is released once entry is complete.
    state->world->setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    state->v4ExactStreamingReleased = false;
    state->v4HorizonAnchorTile.reset();
    state->v4HorizonWorldEpoch = 0;
    state->v4HorizonViewEpoch = 0;
    state->v4HorizonFreshFrames = 0;
    state->world->setSaveManager(state->saveManager.get());
    if (state->generation.fauna)
        state->spawner = std::make_unique<Spawner>(*state->world);
    state->weatherSessionInitialized = false;
    if (state->v4EntryStartedAt <= 0.0)
        state->v4EntryStartedAt = CACurrentMediaTime();
    state->world->updatePlayerPosition(static_cast<int64_t>(std::floor(state->player.position.x)),
                                       static_cast<int32_t>(std::floor(state->player.position.y)),
                                       static_cast<int64_t>(std::floor(state->player.position.z)));
}

// ---- MTKViewDelegate: game loop with fixed timestep ----

- (void)drawInMTKView:(MTKView*)view {
    if (!_device || !_queue)
        return;

    EngineState* state = _state.get();
    [self openReadyV4World];

    if (state->world && state->terrainRepairRecovery.load(std::memory_order_acquire) &&
        !state->terrainBootstrapRunning.load(std::memory_order_acquire)) {
        const worldgen::bootstrap::TerrainBootstrapSnapshot repaired =
            state->terrainBootstrap->snapshot();
        if (repaired.state == worldgen::bootstrap::TerrainBootstrapState::Ready) {
            if (!state->repairedGenerationContext) {
                const std::optional<std::string> fingerprint =
                    state->terrainBootstrap->qualifiedGenerationFingerprint();
                const std::filesystem::path profilePath =
                    state->saveManager ? state->saveManager->getWorldPath() : std::string{};
                if (!fingerprint || *fingerprint != state->generationFingerprint ||
                    profilePath.empty() ||
                    !state->terrainBootstrap->bindWorldProfile(profilePath)) {
                    latchStartupFailure(
                        *state, "The repaired runtime does not match this generator v4 world",
                        false, true);
                    state->terrainRepairRecovery.store(false, std::memory_order_release);
                    state->repairedGenerationContext.reset();
                } else {
                    const auto context = state->terrainBootstrap->qualifiedGenerationContext();
                    if (!context || context->identity().seed != state->generationSeed ||
                        worldgen::learned::sha256Hex(context->fingerprint()) !=
                            state->generationFingerprint) {
                        latchStartupFailure(
                            *state, "The repaired runtime could not bind this generator v4 profile",
                            false, true);
                        state->terrainRepairRecovery.store(false, std::memory_order_release);
                    } else {
                        state->repairedGenerationContext = context;
                    }
                }
            }
            if (state->repairedGenerationContext &&
                state->terrainRepairRecovery.load(std::memory_order_acquire)) {
                const int64_t repairX = static_cast<int64_t>(std::floor(state->player.position.x));
                const int64_t repairZ = static_cast<int64_t>(std::floor(state->player.position.z));
                const V4SpawnAuthorityPrequeueResult authority =
                    prequeueV4SpawnAuthority(state->repairedGenerationContext, repairX, repairZ,
                                             COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
                if (authority.failed()) {
                    latchGenerationFailure(
                        *state, authority.failure,
                        "The repaired terrain authority could not prepare this world");
                    state->terrainRepairRecovery.store(false, std::memory_order_release);
                    state->repairedGenerationContext.reset();
                } else if (authority.ready()) {
                    // Renderer jobs and the Spawner retain World references.
                    // Detach them before replacing the failed context, while
                    // keeping SaveManager and gameplay state bound to the
                    // same selected profile.
                    if (_renderPipeline)
                        _renderPipeline->endWorldSession();
                    state->spawner.reset();
                    state->entityBrains.clear();
                    state->weatherSystem.reset();
                    state->weatherSnapshot.reset();
                    state->weatherSessionInitialized = false;
                    state->world.reset();
                    state->world = std::make_shared<World>(
                        state->generationSeed, state->settings.viewDistance, MAX_LOADED_CUBES,
                        state->repairedGenerationContext, state->generation);
                    installSavedFurnaceProjection(*state);
                    state->world->setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
                    state->world->setSaveManager(state->saveManager.get());
                    if (state->generation.fauna)
                        state->spawner = std::make_unique<Spawner>(*state->world);
                    state->world->updatePlayerPosition(
                        static_cast<int64_t>(std::floor(state->player.position.x)),
                        static_cast<int32_t>(std::floor(state->player.position.y)),
                        static_cast<int64_t>(std::floor(state->player.position.z)));
                    clearStartupFailure(*state);
                    state->v4EntryReady = false;
                    state->v4ExactStreamingReleased = false;
                    state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Ready;
                    state->v4HorizonAnchorTile.reset();
                    state->v4HorizonWorldEpoch = 0;
                    state->v4HorizonViewEpoch = 0;
                    state->v4HorizonFreshFrames = 0;
                    state->spawnValidated = false;
                    state->v4EntryStartedAt = CACurrentMediaTime();
                    state->terrainRepairRecovery.store(false, std::memory_order_release);
                    state->repairedGenerationContext.reset();
                }
            }
        } else if (repaired.state == worldgen::bootstrap::TerrainBootstrapState::Failed) {
            latchStartupFailure(*state, repaired.detail,
                                repaired.failure ? std::optional<bool>{repaired.failure->retryable}
                                                 : std::nullopt);
            state->terrainRepairRecovery.store(false, std::memory_order_release);
            state->repairedGenerationContext.reset();
        }
    }

    if (state->world && !state->diagnosticV3 && state->startupFailure.empty()) {
        if (const std::optional<std::string> failure = state->world->generationFailure()) {
            if (!state->generationRecoveryReturnScreen && state->flow.worldScreens())
                state->generationRecoveryReturnScreen = state->flow.screen;
            const auto context = state->world->generationContext();
            latchGenerationFailure(*state, context ? context->failure() : std::nullopt, *failure);
            if (state->flow.worldScreens()) {
                state->flow.screen = GameScreen::TITLE;
                if (state->inputManager)
                    state->inputManager->releaseMouse();
                state->accumulator = 0.0;
            }
        } else if (!state->v4EntryReady) {
            state->world->updatePlayerPosition(
                static_cast<int64_t>(std::floor(state->player.position.x)),
                static_cast<int32_t>(std::floor(state->player.position.y)),
                static_cast<int64_t>(std::floor(state->player.position.z)));
            const RenderPipeline::ChunkRenderStats streaming = _renderPipeline->chunkRenderStats();
            bool safeSpawnReady = state->world->exactSpawnBandReady(
                static_cast<int64_t>(std::floor(state->player.position.x)),
                static_cast<int32_t>(std::floor(state->player.position.y)),
                static_cast<int64_t>(std::floor(state->player.position.z)),
                V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS);
            bool spawnRelocatedThisFrame = false;
            if (safeSpawnReady && !state->v4SpawnSafetyValidated) {
                const std::optional<Vec3> safeSpawn = state->world->safeSpawnFromReadyPlans(
                    static_cast<int64_t>(std::floor(state->player.position.x)),
                    static_cast<int64_t>(std::floor(state->player.position.z)),
                    V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS);
                if (safeSpawn) {
                    SaveManager::WorldMetadata metadata = worldMetadataSnapshot(*state, *safeSpawn);
                    metadata.spawnFinalized = true;
                    metadata.spawnSafetyRevision = SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
                    metadata.safeSpawnPos = *safeSpawn;
                    if (state->v4ProfileNewlyCreated || !state->bedSpawnSet)
                        metadata.spawnPos = *safeSpawn;
                    if (!state->saveManager->saveMetadata(metadata)) {
                        latchStartupFailure(
                            *state, "The dry generator v4 spawn could not be persisted", true);
                    } else {
                        state->player.position = *safeSpawn;
                        state->player.velocity = Vec3{0.0F, 0.0F, 0.0F};
                        state->requestedSpawn = *safeSpawn;
                        state->v4SpawnFinalized = true;
                        state->v4SpawnSafetyValidated = true;
                        state->v4SpawnSafetyRevision =
                            SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
                        state->v4SafeSpawnPos = *safeSpawn;
                        if (state->v4ProfileNewlyCreated || !state->bedSpawnSet)
                            state->worldSpawn = *safeSpawn;
                        state->v4ProfileNewlyCreated = false;
                        state->v4SpawnFallbackSearchOrigin.reset();
                        state->v4SpawnCandidateActive = false;
                        state->v4SpawnCandidateProvisional = false;
                        state->spawnValidated = true;
                        state->v4HorizonAnchorTile.reset();
                        state->v4HorizonWorldEpoch = 0;
                        state->v4HorizonViewEpoch = 0;
                        state->v4HorizonFreshFrames = 0;
                        state->world->updatePlayerPosition(
                            static_cast<int64_t>(std::floor(safeSpawn->x)),
                            static_cast<int32_t>(std::floor(safeSpawn->y)),
                            static_cast<int64_t>(std::floor(safeSpawn->z)));
                        safeSpawnReady = false;
                        spawnRelocatedThisFrame = true;
                    }
                } else {
                    if (state->v4SpawnCandidateActive) {
                        _renderPipeline->cancelV4Preparation();
                        ++state->v4SpawnSearchOrdinal;
                        state->v4SpawnCandidateActive = false;
                        state->v4SpawnCandidateProvisional = false;
                    }
                    const std::shared_ptr<worldgen::learned::WorldGenerationContext>
                        generationContext = state->world->generationContext();
                    const auto selected = findV4DryLandSpawnCandidate(
                        generationContext,
                        static_cast<int64_t>(std::floor(state->v4SpawnSearchOrigin.x)),
                        static_cast<int64_t>(std::floor(state->v4SpawnSearchOrigin.z)),
                        state->v4SpawnSearchOrdinal);
                    if (selected.status() == worldgen::learned::AuthorityStatus::FAILED) {
                        if (retryV4DrySpawnFromFallbackAfterFailure(*state, selected.failure())) {
                            safeSpawnReady = false;
                        } else {
                            const std::optional<worldgen::learned::GenerationFailure> failure =
                                selected.failure() ? std::optional{*selected.failure()}
                                                   : std::nullopt;
                            latchGenerationFailure(*state, failure,
                                                   "A dry generator v4 spawn could not be located",
                                                   failure && !failure->retriable);
                        }
                    } else if (selected.isReady()) {
                        if (!selected.value() || !*selected.value()) {
                            ++state->v4SpawnSearchOrdinal;
                            if (state->v4SpawnSearchOrdinal >= V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES) {
                                if (retryV4DrySpawnFromFallbackAfterExhaustion(*state)) {
                                    safeSpawnReady = false;
                                } else {
                                    latchStartupFailure(
                                        *state,
                                        "No dry generator v4 spawn was found in the bounded "
                                        "learned-terrain search",
                                        false, true);
                                }
                            }
                        } else {
                            const Vec3 candidate = **selected.value();
                            const V4SpawnWaterScreenResult waterScreen =
                                screenV4SpawnCandidateWater(*state, generationContext, candidate);
                            if (waterScreen.failed()) {
                                latchGenerationFailure(
                                    *state, waterScreen.failure,
                                    "The canonical safe-spawn water screen failed");
                            } else if (waterScreen.deferred()) {
                                // Keep the previous bounded exact band
                                // resident while a worker checks this one
                                // candidate. The render thread does not wait
                                // for learned authority or hydrology here.
                            } else if (waterScreen.water()) {
                                ++state->v4SpawnSearchOrdinal;
                                state->v4SpawnWaterScreen->reset();
                                if (state->v4SpawnSearchOrdinal >=
                                    V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES) {
                                    if (retryV4DrySpawnFromFallbackAfterExhaustion(*state)) {
                                        safeSpawnReady = false;
                                    } else {
                                        latchStartupFailure(
                                            *state,
                                            "No dry generator v4 spawn was found in the bounded "
                                            "learned-terrain search",
                                            false, true);
                                    }
                                }
                            } else {
                                const Vec3 resolvedCandidate =
                                    waterScreen.resolvedCandidate.value_or(candidate);
                                state->v4SpawnWaterScreen->reset();
                                if (waterScreen.provisionalLearnedDry) {
                                    _renderPipeline->cancelV4Preparation();
                                }
                                // The FINAL owner is ready. A canonical result
                                // also installed its local 5x5 certificate; a
                                // continental provisional result remains
                                // gated on radius-zero exact validation.
                                state->v4SpawnAuthorityStatus =
                                    V4SpawnAuthorityPrequeueStatus::Ready;
                                state->player.position = resolvedCandidate;
                                state->player.velocity = Vec3{0.0F, 0.0F, 0.0F};
                                state->requestedSpawn = resolvedCandidate;
                                state->v4SpawnFinalized = false;
                                state->v4SpawnSafetyValidated = false;
                                state->v4SafeSpawnPos.reset();
                                state->v4SpawnCandidateActive = true;
                                state->v4SpawnCandidateProvisional =
                                    waterScreen.provisionalLearnedDry;
                                state->v4HorizonAnchorTile.reset();
                                state->v4HorizonWorldEpoch = 0;
                                state->v4HorizonViewEpoch = 0;
                                state->v4HorizonFreshFrames = 0;
                                state->world->setExactStreamingDistance(
                                    COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
                                state->world->updatePlayerPosition(
                                    static_cast<int64_t>(std::floor(resolvedCandidate.x)),
                                    static_cast<int32_t>(std::floor(resolvedCandidate.y)),
                                    static_cast<int64_t>(std::floor(resolvedCandidate.z)));
                                safeSpawnReady = false;
                                spawnRelocatedThisFrame = true;
                            }
                        }
                    }
                }
            }
            const int64_t horizonWorldX =
                static_cast<int64_t>(std::floor(state->player.position.x));
            const int32_t horizonWorldY =
                static_cast<int32_t>(std::floor(state->player.position.y));
            const int64_t horizonWorldZ =
                static_cast<int64_t>(std::floor(state->player.position.z));
            const ColumnPos requestedHorizonTile{
                world_coord::floorDiv(horizonWorldX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
                world_coord::floorDiv(horizonWorldZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
            };
            const int requiredHorizonChunks =
                farTerrainEntryHorizonViewDistance(state->world->getViewDistance());
            const bool horizonMatchesSpawn =
                streaming.farBaseWorldEpoch != 0 &&
                streaming.farBaseViewDistanceChunks == requiredHorizonChunks &&
                streaming.farBaseCenterTileX == requestedHorizonTile.x &&
                streaming.farBaseCenterTileZ == requestedHorizonTile.z;
            if (!horizonMatchesSpawn) {
                state->v4HorizonAnchorTile.reset();
                state->v4HorizonWorldEpoch = 0;
                state->v4HorizonViewEpoch = 0;
                state->v4HorizonFreshFrames = 0;
            } else if (!state->v4HorizonAnchorTile ||
                       *state->v4HorizonAnchorTile != requestedHorizonTile ||
                       state->v4HorizonWorldEpoch != streaming.farBaseWorldEpoch ||
                       state->v4HorizonViewEpoch != streaming.farBaseViewEpoch) {
                state->v4HorizonAnchorTile = requestedHorizonTile;
                state->v4HorizonWorldEpoch = streaming.farBaseWorldEpoch;
                state->v4HorizonViewEpoch = streaming.farBaseViewEpoch;
                state->v4HorizonFreshFrames = 1;
            } else if (state->v4HorizonFreshFrames < std::numeric_limits<uint32_t>::max()) {
                ++state->v4HorizonFreshFrames;
            }
            const bool horizonPermitted = state->v4SpawnSafetyValidated && safeSpawnReady;
            const int entryHorizonChunks = v4RequiredEntryParentRadiusChunks(requiredHorizonChunks);
            const float connectedParentRadiusChunks =
                v4ConnectedParentRadiusChunks(streaming, requiredHorizonChunks);
            const bool horizonReady = v4EntryHorizonReady(
                horizonPermitted, horizonMatchesSpawn, state->v4HorizonFreshFrames,
                requiredHorizonChunks, streaming.farBaseViewDistanceChunks, entryHorizonChunks,
                connectedParentRadiusChunks, streaming.farBaseWantedTileCount,
                streaming.farBaseResidentTileCount, streaming.farBaseMissingTileCount);
            // Canonical support and headroom certify the spawn column. Control
            // is released only after the already-retained local cube halo is
            // resident too, otherwise the intentional missing-cube collision
            // fallback becomes an invisible wall at a chunk edge. Horizon
            // preparation remains independent and overlaps this small warmup.
            const bool playableCollisionReady =
                state->v4SpawnFinalized && state->world->playableSpawnCollisionReady(
                                               horizonWorldX, horizonWorldY, horizonWorldZ);
            const ColumnPos protectedAnchor =
                farTerrainProtectedNearAnchor(horizonWorldX, horizonWorldZ);
            const bool connectedPreviewParentPrefixReady = v4EntryConnectedParentReady(
                requiredHorizonChunks, entryHorizonChunks, connectedParentRadiusChunks,
                streaming.farBaseWantedTileCount, streaming.farBaseResidentTileCount,
                streaming.farBaseMissingTileCount);
            const bool exactMeshesCurrent =
                streaming.exactSurfaceRequiredCount != 0 &&
                streaming.exactSurfaceReadyCount == streaming.exactSurfaceRequiredCount &&
                streaming.exactSurfaceUnresolvedColumnCount == 0;
            const V4NearEntryClosureInput nearEntry{
                .currentViewEpoch = streaming.farBaseViewEpoch,
                .closureViewEpoch = streaming.farProtectedNearViewEpoch,
                .currentWorldEpoch = streaming.farBaseWorldEpoch,
                .closureWorldEpoch = streaming.farProtectedNearWorldEpoch,
                .currentProtectedEpoch = streaming.farProtectedNearCurrentEpoch,
                .closureProtectedEpoch = streaming.farProtectedNearClosureEpoch,
                .currentAnchor = {protectedAnchor.x, protectedAnchor.z},
                .closureAnchor = {streaming.farProtectedNearAnchorTileX,
                                  streaming.farProtectedNearAnchorTileZ},
                .connectedPreviewParentPrefixReady = connectedPreviewParentPrefixReady,
                .finalTargetCountsByStep = streaming.farProtectedNearTargetCountsByStep,
                .matchingFinalParentsUploaded = streaming.farProtectedNearFinalParentCount,
                .matchingFinalParentsResident = streaming.farProtectedNearFinalParentCount,
                .matchingFinalChildrenUploaded = streaming.farProtectedNearFinalTargetCount,
                .matchingFinalChildrenResident = streaming.farProtectedNearFinalTargetCount,
                .exactCompatibleTargets = streaming.farProtectedNearExactCompatibleTargetCount,
                .lodTransitionMismatches = streaming.farProtectedNearLodMismatchCount,
                .authorityTransitionMismatches = streaming.farProtectedNearAuthorityMismatchCount,
                .collisionCubesReady = playableCollisionReady ? V4_ENTRY_COLLISION_CUBE_COUNT : 0,
                .exactMeshesRequired = streaming.exactSurfaceRequiredCount,
                .matchingExactMeshesReady = streaming.exactSurfaceReadyCount,
                .currentExactMeshRevision = streaming.exactSurfaceEpoch,
                .readyExactMeshRevision = exactMeshesCurrent ? streaming.exactSurfaceEpoch : 0,
            };
            const bool nearEntryReady =
                streaming.farProtectedNearReady && v4NearEntryClosureReady(nearEntry);
            const bool playableSpawnReady = state->v4SpawnFinalized &&
                                            state->v4SpawnSafetyValidated && safeSpawnReady &&
                                            !spawnRelocatedThisFrame && playableCollisionReady;
            state->v4EntryReady = playableSpawnReady && horizonReady && nearEntryReady;
            if (state->v4EntryReady && !state->v4ExactStreamingReleased) {
                if (!state->weatherSessionInitialized) {
                    initializeWeatherSession(*state, state->player.position);
                }
                state->world->setExactStreamingDistance(MAX_EXACT_CUBIC_DISTANCE_CHUNKS);
                state->v4ExactStreamingReleased = true;
            }
            if (state->v4EntryReady && state->generationRecoveryReturnScreen) {
                const GameScreen returnScreen = *state->generationRecoveryReturnScreen;
                state->generationRecoveryReturnScreen.reset();
                [self applyFlowEffects:state->flow.onGenerationRecovered(returnScreen)];
            } else if (state->v4EntryReady && !state->requestedStartScreenApplied &&
                       state->requestedStartScreen) {
                applyRequestedGameplayScreen(*state);
            }
        }
    }

    // 1. Calculate elapsed time
    double currentTime = CACurrentMediaTime();
    double frameTime =
        (state->lastTime > 0) ? (currentTime - state->lastTime) : EngineState::TICK_DT;
    state->lastTime = currentTime;

    // Clamp frame time to prevent spiral of death after long pause
    if (frameTime > 0.25) {
        frameTime = 0.25;
    }

    state->deltaTime = frameTime;

    // 2. Add to accumulator
    state->accumulator += frameTime;

    // 3. Screen-level input (ESC, menu clicks, F3), runs every frame so
    // menus stay responsive while the simulation is frozen
    [self handleGlobalInput];

    // Cooking ticks stay frozen in pause and non-container menus, but saved
    // furnace visuals still need to bind to their sidecars as cubes arrive.
    // Six checks per second are prompt without adding a per-frame world lock.
    if (state->world && state->flow.screen != GameScreen::PLAYING && !state->flow.inContainer() &&
        state->frameCount % 20 == 0) {
        [self reconcileFurnaceBlocks:state];
    }

    // Performance hook: freeze the exact same streamed scene at a fixed frame
    // so playing and paused windows can be compared without relying on a
    // synthetic key event or a separately generated world.
    if (state->frameCount == state->autoPauseFrame && state->flow.screen == GameScreen::PLAYING) {
        state->flow.screen = GameScreen::PAUSED;
        state->accumulator = 0.0;
    }

    // 4. Fixed timestep game tick, menus freeze the world
    if (state->world && state->flow.screen == GameScreen::PLAYING) {
        while (state->accumulator >= EngineState::TICK_DT) {
            const double tickStart = CACurrentMediaTime();
            [self gameTick:state];
            recordPerformanceFixedTick(state->performance, state->frameCount,
                                       (CACurrentMediaTime() - tickStart) * 1000.0);
            state->accumulator -= EngineState::TICK_DT;
        }

        // Render the latest simulated position directly, no inter-tick
        // interpolation. Interpolation trails the sim by up to a tick (50 ms)
        // and read as floaty/light; the 20 Hz camera step is preferred over
        // that added latency. Falls stay gradual because velocity no longer
        // saturates (see Player::tick), so this no longer looks instantaneous.
        state->camera.setPosition(state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f});
    } else if (state->world && state->flow.inContainer()) {
        // Container screens freeze the world but keep furnaces cooking, so the
        // cook arrow advances while the player watches. Bounded catch-up like
        // the fluid scheduler to survive a hitch.
        int steps = 0;
        while (state->accumulator >= EngineState::TICK_DT && steps < 8) {
            [self tickFurnaces:state];
            state->accumulator -= EngineState::TICK_DT;
            ++steps;
        }
        if (steps > 0) {
            [self updateDynamicObjectLighting:state];
        }
        if (state->accumulator >= EngineState::TICK_DT) {
            state->accumulator = 0;
        }
    } else {
        state->accumulator = 0;
    }

    // 5. Render
    [self render];

    // 5. Increment frame count
    state->frameCount++;

    // 6. Consume input state for next frame
    if (state->inputManager) {
        state->inputManager->state().update();
    }

    (void)view;
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)newSize {
    if (_renderPipeline && newSize.width > 0 && newSize.height > 0) {
        _renderPipeline->resize(static_cast<uint32_t>(newSize.width),
                                static_cast<uint32_t>(newSize.height));
    }
    (void)view;
}

// ---- Audio ----

// Master volume follows the settings slider while playing and mutes in
// menus (paused world = paused soundscape). The ambient wind bed starts on
// the first transition into gameplay.
- (void)syncAudioVolume {
    EngineState* state = _state.get();
    if (!state->audio)
        return;

    // Master volume follows the slider at all times so GUI clicks are audible
    // in menus. The "paused world = paused soundscape" feel is kept by
    // stopping the wind bed off the playing screen; frozen ticks produce no
    // world one-shots anyway.
    const bool playing = state->flow.screen == GameScreen::PLAYING;
    float volume = static_cast<float>(state->settings.volumeLevel) / 10.0f;
    state->audio->setMasterVolume(volume);

    if (playing && state->windVoice < 0 && !state->sfxWind.empty()) {
        state->windVoice = state->audio->playSound(state->sfxWind, SoundEffect::SAMPLE_RATE, 0.0F,
                                                   /*looping=*/true);
    } else if (!playing && state->windVoice >= 0) {
        state->audio->stopVoice(state->windVoice);
        state->windVoice = -1;
    }
    if (playing && state->rainVoice < 0 && !state->sfxRain.empty()) {
        state->rainVoice = state->audio->playSound(state->sfxRain, SoundEffect::SAMPLE_RATE, 0.0F,
                                                   /*looping=*/true);
    }
    if (playing && state->snowVoice < 0 && !state->sfxSnow.empty()) {
        state->snowVoice = state->audio->playSound(state->sfxSnow, SoundEffect::SAMPLE_RATE, 0.0F,
                                                   /*looping=*/true);
    }
}

- (void)syncWeatherAudio {
    EngineState* state = _state.get();
    if (!state->audio || state->flow.screen != GameScreen::PLAYING)
        return;

    const float wind =
        std::clamp(state->localWeather.windBlocksPerSecond.length() / 12.0F, 0.0F, 1.0F);
    const float precipitation = std::clamp(state->localWeather.precipitationIntensity, 0.0F, 1.0F);
    const float targetWind =
        0.12F + 0.30F * wind + 0.18F * std::clamp(state->localWeather.stormPotential, 0.0F, 1.0F);
    const float targetRain = state->localWeather.precipitationKind == PrecipitationKind::RAIN
                                 ? precipitation * 0.52F
                                 : 0.0F;
    const float targetSnow = state->localWeather.precipitationKind == PrecipitationKind::SNOW
                                 ? precipitation * 0.34F
                                 : 0.0F;
    const float blend = 1.0F - std::exp(-static_cast<float>(EngineState::TICK_DT) / 1.25F);
    state->windGain += (targetWind - state->windGain) * blend;
    state->rainGain += (targetRain - state->rainGain) * blend;
    state->snowGain += (targetSnow - state->snowGain) * blend;
    state->audio->setVoiceGain(state->windVoice, state->windGain);
    state->audio->setVoiceGain(state->rainVoice, state->rainGain);
    state->audio->setVoiceGain(state->snowVoice, state->snowGain);
}

- (void)playSfx:(const std::vector<float>&)buffer gain:(float)gain {
    EngineState* state = _state.get();
    if (!state->audio || state->flow.screen != GameScreen::PLAYING)
        return;
    state->audio->playSound(buffer, SoundEffect::SAMPLE_RATE, gain);
}

// Interface sounds bypass the playing-screen gate so buttons and slot clicks
// respond on every menu screen.
- (void)playUiSfx:(const std::vector<float>&)buffer gain:(float)gain {
    EngineState* state = _state.get();
    if (!state->audio)
        return;
    state->audio->playSound(buffer, SoundEffect::SAMPLE_RATE, gain);
}

// ---- Screen-level input: ESC, menu interaction, debug HUD toggle ----

- (void)applyFlowEffects:(GameFlowEffects)effects {
    EngineState* state = _state.get();
    if (effects.captureCursor && state->inputManager) {
        state->inputManager->captureMouse();
    }
    if (effects.releaseCursor && state->inputManager) {
        state->inputManager->releaseMouse();
    }
    if (effects.resetTiming) {
        state->accumulator = 0;
        if (state->inputManager) {
            // Drop buffered look deltas AND pending tick presses, the click
            // that pressed RESUME must not break a block on the next tick
            state->inputManager->state().clearMouseDelta();
            state->inputManager->state().clearTickPresses();
        }
    }
    if (effects.requestQuit) {
        [self requestQuit];
    }
    [self syncAudioVolume];
}

// Settings steppers mutate live engine state; the screen doesn't change.
- (void)applySettingAction:(MenuAction)action {
    EngineState* state = _state.get();
    SettingsValues& settings = state->settings;

    switch (action) {
        case MenuAction::VIEW_DISTANCE_DOWN:
        case MenuAction::VIEW_DISTANCE_UP: {
            constexpr const auto& distances = SettingsValues::VIEW_DISTANCES;
            const bool increase = action == MenuAction::VIEW_DISTANCE_UP;
            auto current =
                std::lower_bound(distances.begin(), distances.end(), settings.viewDistance);
            if (increase) {
                if (current == distances.end()) {
                    current = std::prev(distances.end());
                } else if (*current <= settings.viewDistance) {
                    const auto next = std::next(current);
                    if (next != distances.end())
                        current = next;
                }
            } else {
                if (current == distances.end() || *current >= settings.viewDistance) {
                    if (current != distances.begin())
                        --current;
                }
            }
            settings.viewDistance = *current;
            if (state->world) {
                state->world->setViewDistance(settings.viewDistance);
            }
            break;
        }
        case MenuAction::FOG_DOWN:
        case MenuAction::FOG_UP: {
            int step = (action == MenuAction::FOG_UP) ? 1 : -1;
            settings.fogLevel = std::clamp(settings.fogLevel + step, 0, 10);
            if (_renderPipeline) {
                _renderPipeline->setFogDensity(fogDensityForLevel(settings.fogLevel));
            }
            return;
        }
        case MenuAction::SENSITIVITY_DOWN:
        case MenuAction::SENSITIVITY_UP: {
            int step = (action == MenuAction::SENSITIVITY_UP) ? 1 : -1;
            settings.sensitivityLevel = std::clamp(settings.sensitivityLevel + step, 1, 10);
            state->camera.setMouseSensitivity(mouseSensitivityForLevel(settings.sensitivityLevel));
            return;
        }
        case MenuAction::VOLUME_DOWN:
        case MenuAction::VOLUME_UP: {
            int step = (action == MenuAction::VOLUME_UP) ? 1 : -1;
            settings.volumeLevel = std::clamp(settings.volumeLevel + step, 0, 10);
            [self syncAudioVolume];
            return;
        }
        case MenuAction::SHADOWS_DOWN:
        case MenuAction::SHADOWS_UP: {
            int step = (action == MenuAction::SHADOWS_UP) ? 1 : -1;
            state->gfx.shadowQuality = std::clamp(state->gfx.shadowQuality + step, 0,
                                                  GraphicsSettings::SHADOW_QUALITY_MAX);
            break;
        }
        case MenuAction::VL_TOGGLE:
            state->gfx.volumetricLight = !state->gfx.volumetricLight;
            break;
        case MenuAction::CLOUDS_DOWN:
        case MenuAction::CLOUDS_UP: {
            int step = (action == MenuAction::CLOUDS_UP) ? 1 : -1;
            state->gfx.cloudQuality =
                std::clamp(state->gfx.cloudQuality + step, 0, GraphicsSettings::QUALITY_MAX);
            break;
        }
        case MenuAction::INDIRECT_DOWN:
        case MenuAction::INDIRECT_UP: {
            int step = (action == MenuAction::INDIRECT_UP) ? 1 : -1;
            state->gfx.indirectLightingQuality = std::clamp(
                state->gfx.indirectLightingQuality + step, 0, GraphicsSettings::QUALITY_MAX);
            break;
        }
        case MenuAction::SSR_TOGGLE:
            state->gfx.waterReflections = !state->gfx.waterReflections;
            break;
        case MenuAction::WAVING_TOGGLE:
            state->gfx.wavingFoliage = !state->gfx.wavingFoliage;
            break;
        case MenuAction::LENS_FLARE_TOGGLE:
            state->gfx.lensFlare = !state->gfx.lensFlare;
            break;
        case MenuAction::BLOOM_DOWN:
        case MenuAction::BLOOM_UP: {
            int step = (action == MenuAction::BLOOM_UP) ? 1 : -1;
            state->gfx.bloomLevel =
                std::clamp(state->gfx.bloomLevel + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        case MenuAction::VIBRANCE_DOWN:
        case MenuAction::VIBRANCE_UP: {
            int step = (action == MenuAction::VIBRANCE_UP) ? 1 : -1;
            state->gfx.vibrance =
                std::clamp(state->gfx.vibrance + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        case MenuAction::SHARPEN_DOWN:
        case MenuAction::SHARPEN_UP: {
            int step = (action == MenuAction::SHARPEN_UP) ? 1 : -1;
            state->gfx.sharpening =
                std::clamp(state->gfx.sharpening + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        default:
            return; // not a settings action
    }

    // Only the video cases fall through: push the changed copy so the
    // renderer picks it up next frame.
    if (_renderPipeline) {
        _renderPipeline->setGraphicsSettings(state->gfx);
    }
}

// Advance every player-placed furnace one 20 Hz step and reconcile its world
// block with the persisted cooking state. The block check runs every tick so
// a transition that occurred while its cube was absent is applied as soon as
// that cube becomes resident again.
- (void)reconcileFurnaceBlocks:(EngineState*)state {
    if (!state->world)
        return;
    for (auto entry = state->furnaces.begin(); entry != state->furnaces.end();) {
        const BlockPos pos = entry->first;
        const std::optional<BlockType> current =
            state->world->findBlockIfLoaded(pos.x, pos.y, pos.z);
        if (!furnaceSidecarMatchesLoadedBlock(current)) {
            if (state->openFurnace && *state->openFurnace == pos)
                state->openFurnace.reset();
            state->furnaceVisualAuthority->erase(pos);
            entry = state->furnaces.erase(entry);
            continue;
        }
        if (current) {
            const BlockType desired = furnaceBlockForState(entry->second);
            if (*current != desired)
                state->world->setBlock(pos.x, pos.y, pos.z, desired);
        }
        ++entry;
    }
}

- (void)tickFurnaces:(EngineState*)state {
    if (!state->world)
        return;
    for (auto entry = state->furnaces.begin(); entry != state->furnaces.end();) {
        const BlockPos pos = entry->first;
        FurnaceState& furnace = entry->second;
        const std::optional<BlockType> current =
            state->world->findBlockIfLoaded(pos.x, pos.y, pos.z);
        if (!furnaceSidecarMatchesLoadedBlock(current)) {
            if (state->openFurnace && *state->openFurnace == pos)
                state->openFurnace.reset();
            state->furnaceVisualAuthority->erase(pos);
            entry = state->furnaces.erase(entry);
            continue;
        }
        const bool litChanged = furnaceTick(furnace);
        const BlockType desired = furnaceBlockForState(furnace);
        if (litChanged) {
            // Publish the immutable offscreen visual before the loaded-only
            // block write. A concurrently loading saved cube either observes
            // this revision or fails its pre-insertion revision check.
            state->furnaceVisualAuthority->set(pos, desired);
        }
        if (!current || *current != desired) {
            (void)state->world->trySetBlock(pos.x, pos.y, pos.z, desired);
        }
        if (current && litChanged && furnace.lit()) {
            [self playSfx:state->sfxFurnacePop gain:0.4f];
        }
        ++entry;
    }
}

// Retain an unloaded chest sidecar, but remove one once a resident cell proves
// that its owning block no longer exists. This prevents old contents from
// returning when a new chest is later placed at the same coordinates.
- (void)reconcileChestBlocks:(EngineState*)state {
    if (!state->world)
        return;
    for (auto entry = state->chests.begin(); entry != state->chests.end();) {
        const BlockPos pos = entry->first;
        const std::optional<BlockType> current =
            state->world->findBlockIfLoaded(pos.x, pos.y, pos.z);
        if (!chestSidecarMatchesLoadedBlock(current)) {
            if (state->openChest && *state->openChest == pos)
                state->openChest.reset();
            entry = state->chests.erase(entry);
            continue;
        }
        ++entry;
    }
}

// Strike an animal: subtract the held item's damage, apply knockback, and on
// death scatter its meat drops and remove it. Works in both modes so the
// hunger loop can close from a fresh creative test too.
- (void)attackEntity:(EngineState*)state entityId:(uint64_t)entityId {
    if (!state->spawner)
        return;
    for (auto& entity : state->spawner->getEntities()) {
        if (!entity || entity->id != entityId || !entity->alive)
            continue;

        const ItemStack held = state->inventory.getSelectedStack();
        const int damage = static_cast<int>(itemDefinition(held.type).attackDamage);
        entity->health -= damage;

        // Minimal knockback away from the player, plus a small hop.
        Vec3 away{entity->position.x - state->player.position.x, 0.f,
                  entity->position.z - state->player.position.z};
        const float length = away.length();
        if (length > 1e-4f) {
            away.x /= length;
            away.z /= length;
        }
        entity->velocity.x += away.x * 0.3f;
        entity->velocity.z += away.z * 0.3f;
        entity->velocity.y += 0.2f;

        if (modeDrainsHunger(state->gameMode)) {
            state->survival.exhaustion += SurvivalStats::EXHAUST_ATTACK;
        }
        if (isTool(held.type) && modeConsumesItems(state->gameMode)) {
            state->inventory.damageSelectedTool();
        }
        [self playSfx:state->sfxHurt gain:0.5f];

        if (entity->health <= 0) {
            const EntityConfig config = Entity::getConfig(entity->type);
            if (config.drop != ItemType::NONE && config.dropCount > 0) {
                [self spawnDrop:state
                          stack:ItemStack{config.drop, config.dropCount, 0}
                            atX:static_cast<int64_t>(std::floor(entity->position.x))
                              y:static_cast<int32_t>(std::floor(entity->position.y))
                              z:static_cast<int64_t>(std::floor(entity->position.z))];
            }
            entity->alive = false;
            state->spawner->removeEntity(entityId);
        }
        return;
    }
}

// Apply damage in survival, playing a rate-limited hurt sound and recording
// the cause for the death screen. Never drops health below zero.
- (void)damagePlayer:(EngineState*)state amount:(int)amount message:(const char*)message {
    if (amount <= 0 || !modeTakesDamage(state->gameMode))
        return;
    state->player.health = std::max(0, state->player.health - amount);
    state->deathMessage = message;
    if (state->hurtSoundCooldown == 0) {
        [self playSfx:state->sfxHurt gain:0.7f];
        state->hurtSoundCooldown = 10;
    }
}

// Death: scatter every player-owned stack, clear transient interaction state,
// and freeze the world on the death screen until the player respawns.
- (void)killPlayer:(EngineState*)state {
    const Vec3 at = state->player.position + Vec3{0.f, 0.5f, 0.f};
    const std::vector<ItemStack> drops =
        collectDeathDrops(state->inventory.slots(), state->cursorStack, state->craftGrid);
    for (size_t index = 0; index < drops.size(); ++index) {
        const uint64_t h = hash64(static_cast<uint64_t>(index) * 2654435761ull +
                                  static_cast<uint64_t>(state->frameCount));
        const float vx = (static_cast<float>(h & 0xFF) / 255.f - 0.5f) * 0.4f;
        const float vz = (static_cast<float>((h >> 8) & 0xFF) / 255.f - 0.5f) * 0.4f;
        state->itemEntities.spawn(drops[index], at, Vec3{vx, 0.25f, vz}, 20);
    }
    state->craftResult.clear();
    state->ridingBoat = -1;
    state->dragActive = false;
    state->dragMoved = false;
    state->dragSlots.clear();
    state->hoveredSlot = -1;
    state->miningState.reset();
    state->eatingState.reset();
    if (state->deathMessage.empty()) {
        state->deathMessage = "YOU DIED";
    }
    [self playSfx:state->sfxDeath gain:0.8f];
    [self applyFlowEffects:state->flow.onPlayerDied()];
}

// Respawn at the world's creation-time spawn anchor with reset stats. The
// unstick pass places the player on solid ground next tick.
- (void)respawnPlayer:(EngineState*)state {
    state->player.position = state->worldSpawn;
    state->player.velocity = Vec3{0.f, 0.f, 0.f};
    state->player.health = 20;
    state->player.flying = false;
    state->player.sprinting = false;
    state->player.swimming = false;
    state->player.onGround = false;
    state->player.blockedHorizontally = false;
    state->player.jumpCooldown = 0;
    state->player.fallResetTimer = 0;
    state->player.resetFallDistance();
    state->ridingBoat = -1;
    state->survival = SurvivalStats{};
    state->deathMessage.clear();
    state->spawnValidated = false;
    state->pendingBedSpawnValidation = state->bedSpawnSet;
    if (state->world) {
        state->world->updatePlayerPosition(
            static_cast<int64_t>(std::floor(state->player.position.x)),
            static_cast<int32_t>(std::floor(state->player.position.y)),
            static_cast<int64_t>(std::floor(state->player.position.z)));
    }
    [self applyFlowEffects:state->flow.onRespawn()];
}

// Dropped-item physics and the pickup pass. The manager owns physics/merge/
// despawn; the inventory mutation lives here so the entity module never
// depends on the engine's Inventory.
- (void)tickItemEntities:(EngineState*)state {
    if (!state->world)
        return;
    state->itemEntities.tick(*state->world, state->player.position);

    if (state->pickupSoundCooldown > 0)
        --state->pickupSoundCooldown;

    const AABB playerBox = state->player.getAABB();
    const AABB pickupBox{Vec3{playerBox.min.x - ItemEntityManager::PICKUP_XZ,
                              playerBox.min.y - ItemEntityManager::PICKUP_Y,
                              playerBox.min.z - ItemEntityManager::PICKUP_XZ},
                         Vec3{playerBox.max.x + ItemEntityManager::PICKUP_XZ,
                              playerBox.max.y + ItemEntityManager::PICKUP_Y,
                              playerBox.max.z + ItemEntityManager::PICKUP_XZ}};

    bool pickedUp = false;
    for (ItemEntity& item : state->itemEntities.items()) {
        if (item.pickupDelay > 0 || item.stack.empty())
            continue;
        if (!pickupBox.intersects(item.getAABB()))
            continue;
        const int absorbed = state->inventory.add(item.stack);
        if (absorbed > 0) {
            item.stack.count = static_cast<uint8_t>(item.stack.count - absorbed);
            if (item.stack.count == 0)
                item.stack.clear();
            pickedUp = true;
        }
    }
    if (pickedUp) {
        state->itemEntities.compact();
        if (state->pickupSoundCooldown == 0) {
            [self playSfx:state->sfxPickup gain:0.4f];
            state->pickupSoundCooldown = 2;
        }
    }
}

// Cache one packed sky/block-light probe per dynamic object on the fixed
// simulation tick. World batches the lookups beneath one map lock; the render
// thread only decodes these stable bytes, independent of display refresh rate.
- (void)updateDynamicObjectLighting:(EngineState*)state {
    if (!state->world)
        return;

    static thread_local std::vector<BlockPos> positions;
    static thread_local std::vector<uint8_t*> destinations;
    static thread_local std::vector<uint8_t> samples;
    positions.clear();
    destinations.clear();
    const size_t entityCount = state->spawner ? state->spawner->getEntities().size() : 0U;
    const size_t capacity =
        entityCount + state->itemEntities.items().size() + state->boats.boats().size();
    positions.reserve(capacity);
    destinations.reserve(capacity);

    const auto appendProbe = [&](const Vec3& position, float height, uint8_t& destination) {
        positions.emplace_back(static_cast<int64_t>(std::floor(position.x)),
                               static_cast<int32_t>(std::floor(position.y + height * 0.5F)),
                               static_cast<int64_t>(std::floor(position.z)));
        destinations.push_back(&destination);
    };

    if (state->spawner) {
        for (const auto& entity : state->spawner->getEntities()) {
            if (!entity || !entity->alive)
                continue;
            appendProbe(entity->position, Entity::getConfig(entity->type).height,
                        entity->renderPackedLight);
        }
    }
    for (ItemEntity& item : state->itemEntities.items()) {
        if (item.stack.empty())
            continue;
        appendProbe(item.position, ItemEntity::SIZE, item.renderPackedLight);
    }
    for (Boat& boat : state->boats.boats()) {
        appendProbe(boat.position, Boat::HEIGHT, boat.renderPackedLight);
    }

    if (positions.empty())
        return;
    samples.resize(positions.size());
    state->world->samplePackedLightsIfLoaded(positions, samples);
    for (size_t index = 0; index < samples.size(); ++index) {
        *destinations[index] = samples[index];
    }
}

// Mutable stack access for the currently open container screen.
- (SlotAccess)slotAccessForScreen {
    EngineState* state = _state.get();
    SlotAccess access;
    access.inventory = state->inventory.slots().data();
    const GameScreen screen = state->flow.screen;
    const bool creative = state->gameMode == GameMode::CREATIVE;
    if (screen == GameScreen::INVENTORY && creative) {
        access.palette = CREATIVE_PALETTE.data();
        access.paletteSize = static_cast<int>(CREATIVE_PALETTE.size());
    } else if (screen == GameScreen::INVENTORY) {
        access.craftGrid = state->craftGrid.data();
        access.craftGridSize = 4;
        access.craftGridWidth = 2;
        access.craftResult = &state->craftResult;
    } else if (screen == GameScreen::CRAFTING) {
        access.craftGrid = state->craftGrid.data();
        access.craftGridSize = 9;
        access.craftGridWidth = 3;
        access.craftResult = &state->craftResult;
    } else if (screen == GameScreen::FURNACE && state->openFurnace) {
        auto it = state->furnaces.find(*state->openFurnace);
        if (it != state->furnaces.end()) {
            access.furnaceInput = &it->second.input;
            access.furnaceFuel = &it->second.fuel;
            access.furnaceOutput = &it->second.output;
        }
    } else if (screen == GameScreen::CHEST && state->openChest) {
        auto it = state->chests.find(*state->openChest);
        if (it != state->chests.end()) {
            access.chest = it->second.slots.data();
            access.chestSize = ChestState::SLOT_COUNT;
        }
    }
    return access;
}

// Leaving a container returns the craft grid and held stack to inventory.
// Anything that cannot fit stays in the same persisted carried-slot contract
// used by an open-container quit. It is exposed on the next inventory action
// and never depends on transient item entities surviving a world switch.
- (void)closeContainerSession {
    EngineState* state = _state.get();
    SlotAccess access = [self slotAccessForScreen];
    std::vector<ItemStack> overflow = collectOnClose(access, state->cursorStack);
    if (!preserveCarriedOverflow(overflow, state->cursorStack, state->craftGrid)) {
        // collectOnClose can return at most the cursor plus the visible craft
        // inputs. Those cells have just been cleared, so exhausting all ten
        // carrier slots is an invariant violation and must remain visible.
        latchStartupFailure(*state, "Container overflow could not be preserved", false);
    }
    state->hoveredSlot = -1;
    state->openFurnace.reset();
    state->openChest.reset();
}

// Close the focused text field, committing its filtered contents.
- (void)commitTextEntry {
    EngineState* state = _state.get();
    InputState& input = state->inputManager->state();
    if (!input.textEntryActive)
        return;
    const bool digits = state->worldCreate.focusedField == 1;
    const std::string value = filterTextField(input.endTextEntry(), digits,
                                              digits ? WorldCreateState::MAX_SEED_LENGTH
                                                     : WorldCreateState::MAX_NAME_LENGTH);
    if (state->worldCreate.focusedField == 0) {
        state->worldCreate.name = value;
    } else if (state->worldCreate.focusedField == 1) {
        state->worldCreate.seedText = value;
    }
    state->worldCreate.focusedField = -1;
}

// One menu click: engine-side effects first (world lifecycle, list edits),
// then the pure flow transition.
- (void)handleMenuAction:(MenuAction)action payload:(int)payload {
    EngineState* state = _state.get();
    const int worldCount = static_cast<int>(state->worldList.size());
    const auto beginV4WorldRequest = [&](uint64_t seed, Vec3 requestedSpawn,
                                         uint64_t initialWorldTime,
                                         std::optional<std::filesystem::path> profilePath,
                                         std::optional<V4WorldCreationRequest> creation) {
        if (state->diagnosticV3 || !state->terrainInstaller)
            return false;
        if (state->world || state->saveManager || state->v4OpenRequested ||
            state->v4ProfileOpened) {
            if (![self stopWorld])
                return false;
        }
        if (state->terrainBootstrap)
            state->terrainBootstrap->cancel();
        if (state->terrainBootstrapThread.joinable())
            state->terrainBootstrapThread.join();
        state->terrainBootstrapRunning.store(false, std::memory_order_release);
        state->generationSeed = seed;
        state->requestedSpawn = requestedSpawn;
        state->player.position = requestedSpawn;
        state->worldTime = initialWorldTime;
        state->preferredV4ProfilePath = std::move(profilePath);
        state->pendingV4Creation = std::move(creation);
        clearStartupFailure(*state);
        state->v4OpenRequested = true;
        state->v4ProfileOpened = false;
        state->v4ProfileNewlyCreated = false;
        state->v4EntryReady = false;
        state->v4SpawnFinalized = false;
        state->v4SpawnSafetyValidated = false;
        state->v4SpawnCandidateActive = false;
        state->v4SpawnCandidateProvisional = false;
        state->v4ExactStreamingReleased = false;
        state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Deferred;
        state->requestedStartScreen = GameScreen::PLAYING;
        state->requestedStartScreenApplied = false;
        state->terrainRuntime = worldgen::runtime::makeProductionTerrainRuntime(seed);
        state->terrainBootstrap = std::make_unique<worldgen::bootstrap::TerrainGenerationBootstrap>(
            *state->terrainInstaller, *state->terrainRuntime);
        [self startTerrainBootstrapWithRetry:NO repair:NO];
        return true;
    };
    switch (action) {
        case MenuAction::DOWNLOAD_MODEL:
            [self startTerrainBootstrapWithRetry:NO repair:NO];
            return;
        case MenuAction::CANCEL_MODEL:
            if (state->terrainBootstrap)
                state->terrainBootstrap->cancel();
            return;
        case MenuAction::RETRY_MODEL:
            if (state->startupFailureIsSave) {
                [self saveCurrentWorld];
            } else if (state->world && !state->startupFailure.empty()) {
                if (state->world->retryGeneration()) {
                    clearStartupFailure(*state);
                    state->v4EntryReady = false;
                    state->v4EntryStartedAt = CACurrentMediaTime();
                }
            } else if (state->terrainBootstrap && state->terrainBootstrap->ready()) {
                if (const auto context = state->terrainBootstrap->qualifiedGenerationContext())
                    context->clearRetriableFailure();
                clearStartupFailure(*state);
                state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Deferred;
                if (state->v4SpawnWaterScreen)
                    state->v4SpawnWaterScreen->reset();
            } else {
                [self startTerrainBootstrapWithRetry:YES repair:NO];
            }
            return;
        case MenuAction::REPAIR_MODEL:
            if (state->startupFailureIsSave) {
                [self saveCurrentWorld];
                return;
            }
            state->repairedGenerationContext.reset();
            if (state->world) {
                // The repaired runtime cannot be installed into an existing
                // World in place. Persist edits now, keep its resident render
                // state visible during repair, then rebuild it from the newly
                // qualified context when repair completes.
                if (!std::getenv("RYCRAFT_CAPTURE") && ![self saveCurrentWorld])
                    return;
                state->terrainRepairRecovery.store(true, std::memory_order_release);
            } else {
                // A profile can already be selected while dry-spawn or page
                // preparation is still pre-world. Release its persistence
                // owner so openReadyV4World reloads metadata and rebinds the
                // repaired runtime to this exact path.
                if (state->saveManager) {
                    state->preferredV4ProfilePath = state->saveManager->getWorldPath();
                    state->saveManager.reset();
                }
                state->v4ProfileOpened = false;
                state->v4ProfileNewlyCreated = false;
                state->v4SpawnCandidateActive = false;
                state->v4SpawnCandidateProvisional = false;
                state->v4SpawnAuthorityStatus = V4SpawnAuthorityPrequeueStatus::Deferred;
                clearStartupFailure(*state);
            }
            if (state->v4SpawnWaterScreen)
                state->v4SpawnWaterScreen->reset();
            [self startTerrainBootstrapWithRetry:YES repair:YES];
            return;
        case MenuAction::OPEN_WORLD_SELECT:
            if (state->diagnosticV3 && state->world) {
                [self applyFlowEffects:state->flow.onWorldStarted()];
                return;
            }
            // Selecting worlds owns startup explicitly. Abandon an automatic
            // default-world request, including a profile that has opened but
            // is still preparing its dry spawn, before showing the list.
            if (!state->diagnosticV3 &&
                (state->v4OpenRequested || state->v4ProfileOpened || state->saveManager)) {
                if (![self stopWorld])
                    return;
            }
            state->worldList = listWorldsForGeneratorV4(
                worldgen::bootstrap::defaultRycraftApplicationSupportPath().string());
            state->worldSelect = WorldSelectState{};
            break;
        case MenuAction::OPEN_WORLD_CREATE:
            if (state->diagnosticV3)
                return;
            state->worldCreate = WorldCreateState{};
            break;
        case MenuAction::SELECT_WORLD:
            if (payload >= 0 && payload < worldCount) {
                state->worldSelect.selected = payload;
            }
            return;
        case MenuAction::WORLD_LIST_UP:
            state->worldSelect.scroll = std::max(0, state->worldSelect.scroll - 1);
            return;
        case MenuAction::WORLD_LIST_DOWN:
            state->worldSelect.scroll =
                std::min(std::max(0, worldCount - WorldSelectState::VISIBLE_ROWS),
                         state->worldSelect.scroll + 1);
            return;
        case MenuAction::REQUEST_V4_SUCCESSOR: {
            const int selected = state->worldSelect.selected;
            if (selected < 0 || selected >= worldCount)
                return;
            const WorldSummary& selectedWorld = state->worldList[static_cast<size_t>(selected)];
            if (state->diagnosticV3 || !worldRequiresCurrentV4Successor(selectedWorld))
                return;
            [self applyFlowEffects:state->flow.onMenuAction(action)];
            return;
        }
        case MenuAction::PLAY_SELECTED_WORLD: {
            const int selected = state->worldSelect.selected;
            if (selected < 0 || selected >= worldCount)
                return;
            const WorldSummary& selectedWorld = state->worldList[static_cast<size_t>(selected)];
            if (state->diagnosticV3)
                return;
            // A stale row can never bypass the explicit successor dialog,
            // even if the cached menu layout delivered an old play action.
            if (worldRequiresCurrentV4Successor(selectedWorld)) {
                [self applyFlowEffects:state->flow.onMenuAction(MenuAction::REQUEST_V4_SUCCESSOR)];
                return;
            }
            if (state->world && state->saveManager &&
                std::filesystem::path(state->saveManager->getWorldPath()) ==
                    std::filesystem::path(selectedWorld.directory)) {
                [self applyFlowEffects:state->flow.onWorldStarted()];
                return;
            }

            static_cast<void>(beginV4WorldRequest(
                selectedWorld.metadata.seed,
                selectedWorld.metadata.safeSpawnPos.value_or(selectedWorld.metadata.playerPos),
                selectedWorld.metadata.worldTime, selectedWorld.directory, std::nullopt));
            return;
        }
        case MenuAction::CONFIRM_V4_SUCCESSOR: {
            if (state->diagnosticV3 || state->flow.screen != GameScreen::WORLD_SUCCESSOR_CONFIRM) {
                return;
            }
            const int selected = state->worldSelect.selected;
            if (selected < 0 || selected >= worldCount)
                return;
            const WorldSummary& selectedWorld = state->worldList[static_cast<size_t>(selected)];
            if (!worldRequiresCurrentV4Successor(selectedWorld))
                return;
            V4WorldCreationRequest successor;
            successor.displayName = filterTextField(selectedWorld.metadata.name, false,
                                                    WorldCreateState::MAX_NAME_LENGTH - 3);
            if (successor.displayName.empty())
                successor.displayName = "World";
            successor.displayName += " v4";
            successor.gameMode = selectedWorld.metadata.gameMode;
            successor.generation = selectedWorld.metadata.generation;
            successor.player = selectedWorld.metadata.player;
            static_cast<void>(beginV4WorldRequest(
                selectedWorld.metadata.seed, GENERATOR_V4_INITIAL_SPAWN,
                selectedWorld.metadata.worldTime, std::nullopt, std::move(successor)));
            return;
        }
        case MenuAction::RANDOM_SEED:
            state->worldCreate.seedText =
                std::to_string((static_cast<uint64_t>(std::random_device{}()) << 32U) ^
                               static_cast<uint64_t>(std::random_device{}()));
            return;
        case MenuAction::TOGGLE_GEN_STRUCTURES:
            state->worldCreate.structures = !state->worldCreate.structures;
            return;
        case MenuAction::TOGGLE_GEN_FAUNA:
            state->worldCreate.fauna = !state->worldCreate.fauna;
            return;
        case MenuAction::TOGGLE_GEN_WEATHER:
            state->worldCreate.weather = !state->worldCreate.weather;
            return;
        case MenuAction::TOGGLE_GEN_DAY_CYCLE:
            state->worldCreate.dayCycle = !state->worldCreate.dayCycle;
            return;
        case MenuAction::TOGGLE_CREATE_MODE:
            state->worldCreate.creative = !state->worldCreate.creative;
            return;
        case MenuAction::CREATE_WORLD_CONFIRM: {
            if (state->diagnosticV3)
                return;
            if (state->inputManager && state->inputManager->state().textEntryActive)
                [self commitTextEntry];
            const std::string name =
                filterTextField(state->worldCreate.name, false, WorldCreateState::MAX_NAME_LENGTH);
            if (name.find_first_not_of(' ') == std::string::npos)
                return;
            uint64_t seed = 0;
            if (state->worldCreate.seedText.empty()) {
                seed = (static_cast<uint64_t>(std::random_device{}()) << 32U) ^
                       static_cast<uint64_t>(std::random_device{}());
            } else {
                const std::optional<uint64_t> parsed =
                    parseUnsignedDecimal(state->worldCreate.seedText);
                if (!parsed) {
                    RY_LOG_ERROR("World seed is outside the unsigned 64-bit range");
                    return;
                }
                seed = *parsed;
            }
            V4WorldCreationRequest creation;
            creation.displayName = name;
            creation.gameMode =
                state->worldCreate.creative ? GameMode::CREATIVE : GameMode::SURVIVAL;
            creation.generation = {
                .structures = state->worldCreate.structures,
                .fauna = state->worldCreate.fauna,
                .weather = state->worldCreate.weather,
                .dayCycle = state->worldCreate.dayCycle,
            };
            if (creation.gameMode == GameMode::SURVIVAL)
                creation.player.inventory.fill(ItemStack{});
            const uint64_t initialWorldTime = creation.generation.dayCycle ? 0 : 6'000;
            if (beginV4WorldRequest(seed, GENERATOR_V4_INITIAL_SPAWN, initialWorldTime,
                                    std::nullopt, std::move(creation))) {
                state->worldCreate = WorldCreateState{};
            }
            return;
        }
        case MenuAction::CONFIRM_DELETE: {
            if (state->diagnosticV3)
                return;
            const int selected = state->worldSelect.selected;
            if (selected >= 0 && selected < worldCount) {
                const std::string directory =
                    state->worldList[static_cast<size_t>(selected)].directory;
                if (state->saveManager &&
                    std::filesystem::path(state->saveManager->getWorldPath()) ==
                        std::filesystem::path(directory)) {
                    if (![self stopWorld])
                        return;
                }
                [self quiesceTerrainRuntimeForProfileDeletion];
                deleteWorld(directory,
                            worldgen::bootstrap::defaultRycraftApplicationSupportPath().string());
            }
            state->worldList = listWorldsForGeneratorV4(
                worldgen::bootstrap::defaultRycraftApplicationSupportPath().string());
            state->worldSelect = WorldSelectState{};
            state->flow.screen = GameScreen::WORLD_SELECT;
            return;
        }
        case MenuAction::CREATIVE_PAGE_PREV:
            state->creativePage = std::max(0, state->creativePage - 1);
            return;
        case MenuAction::CREATIVE_PAGE_NEXT: {
            const int pageCount =
                (static_cast<int>(CREATIVE_PALETTE.size()) + CREATIVE_PALETTE_PAGE_SIZE - 1) /
                CREATIVE_PALETTE_PAGE_SIZE;
            state->creativePage = std::min(pageCount - 1, state->creativePage + 1);
            return;
        }
        case MenuAction::TOGGLE_GAME_MODE:
            state->gameMode =
                state->gameMode == GameMode::CREATIVE ? GameMode::SURVIVAL : GameMode::CREATIVE;
            return;
        case MenuAction::RESPAWN:
            [self respawnPlayer:state];
            return;
        case MenuAction::SAVE_QUIT_TO_TITLE:
            [self stopWorld];
            return;
        case MenuAction::PLAY:
            if (!state->diagnosticV3 && !state->v4EntryReady)
                return;
            break;
        default:
            break;
    }
    [self applySettingAction:action];
    [self applyFlowEffects:state->flow.onMenuAction(action)];
    if ((action == MenuAction::CLOSE_SETTINGS || action == MenuAction::CLOSE_VIDEO_SETTINGS) &&
        !state->envOverridesActive) {
        saveSettings(settingsPath(), state->settings, state->gfx);
    }
}

- (void)handleGlobalInput {
    EngineState* state = _state.get();
    if (!state->inputManager)
        return;
    InputState& input = state->inputManager->state();

    if (state->world && !input.textEntryActive &&
        input.isJustPressed(InputBindings{}.inventory.key)) {
        if (state->flow.inContainer()) {
            [self closeContainerSession];
            [self applyFlowEffects:state->flow.onInventoryKey()];
        } else if (state->flow.screen == GameScreen::PLAYING &&
                   hasExtendedCarriedCrafting(state->craftGrid)) {
            // A restored 3x3 input cannot fit the normal 2x2 inventory view.
            // Open the complete grid once so all persisted cells are visible.
            [self applyFlowEffects:state->flow.onContainerOpened(GameScreen::CRAFTING)];
        } else {
            [self applyFlowEffects:state->flow.onInventoryKey()];
        }
    }
    if (input.isJustPressed(Key::Escape)) {
        if (input.textEntryActive) {
            // Escape closes the field, not the screen.
            [self commitTextEntry];
        } else {
            if (state->flow.inContainer()) {
                [self closeContainerSession];
            }
            // Leaving a settings screen persists the values (not per click,
            // no disk I/O while stepping)
            bool leavingSettings = state->flow.screen == GameScreen::SETTINGS ||
                                   state->flow.screen == GameScreen::VIDEO_SETTINGS;
            [self applyFlowEffects:state->flow.onEscape()];
            if (leavingSettings && !state->envOverridesActive) {
                saveSettings(settingsPath(), state->settings, state->gfx);
            }
        }
    }
    if (input.isJustPressed(Key::F3)) {
        state->showDebugHud = !state->showDebugHud;
    }

    if (state->flow.inMenu()) {
        // Rebuild the layout each frame (settings values can change) and
        // hit-test in window points, the same normalized space it's drawn in
        float boundsW = static_cast<float>(_view.bounds.size.width);
        float boundsH = static_cast<float>(_view.bounds.size.height);

        // Live text entry streams into the focused create-screen field.
        if (input.textEntryActive && state->worldCreate.focusedField >= 0) {
            const bool digits = state->worldCreate.focusedField == 1;
            const std::string filtered = filterTextField(
                input.textBuffer, digits,
                digits ? WorldCreateState::MAX_SEED_LENGTH : WorldCreateState::MAX_NAME_LENGTH);
            if (filtered != input.textBuffer) {
                input.textBuffer = filtered;
            }
            if (state->worldCreate.focusedField == 0) {
                state->worldCreate.name = filtered;
            } else {
                state->worldCreate.seedText = filtered;
            }
            if (input.textSubmitted) {
                [self commitTextEntry];
            }
        }

        MenuContext menuCtx;
        menuCtx.settings = state->settings;
        menuCtx.gfx = &state->gfx;
        menuCtx.mode = state->gameMode;
        menuCtx.allowWorldCreation = !state->diagnosticV3;
        if (state->diagnosticV3)
            menuCtx.worldCreationUnavailableReason = "DIAGNOSTIC WORLDS ARE NOT SAVED";
        menuCtx.caretVisible = (state->frameCount / 30) % 2 == 0;
        menuCtx.deathMessage = state->deathMessage;
        if (state->flow.screen == GameScreen::WORLD_SELECT ||
            state->flow.screen == GameScreen::WORLD_DELETE_CONFIRM ||
            state->flow.screen == GameScreen::WORLD_SUCCESSOR_CONFIRM) {
            menuCtx.worldRows.reserve(state->worldList.size());
            for (const WorldSummary& world : state->worldList) {
                menuCtx.worldRows.push_back(world.metadata.name + " - " +
                                            gameModeName(world.metadata.gameMode) + " - Seed " +
                                            std::to_string(world.metadata.seed));
            }
            menuCtx.worldSelect = state->worldSelect;
            const int selected = state->worldSelect.selected;
            if (selected >= 0 && selected < static_cast<int>(state->worldList.size())) {
                menuCtx.selectedWorldRequiresV4Successor = worldRequiresCurrentV4Successor(
                    state->worldList[static_cast<size_t>(selected)]);
                menuCtx.deleteWorldName =
                    state->worldList[static_cast<size_t>(selected)].metadata.name;
                menuCtx.successorWorldName =
                    state->worldList[static_cast<size_t>(selected)].metadata.name;
            }
        }
        if (state->flow.screen == GameScreen::WORLD_CREATE) {
            menuCtx.worldCreate = state->worldCreate;
        }
        if (state->flow.inContainer()) {
            ContainerView& view = menuCtx.container;
            view.inventory = state->inventory.slots();
            view.craftGrid = state->craftGrid;
            view.craftGridSize = state->flow.screen == GameScreen::CRAFTING ? 9 : 4;
            view.craftResult = state->craftResult;
            view.creative = state->flow.screen == GameScreen::INVENTORY &&
                            state->gameMode == GameMode::CREATIVE;
            view.creativePage = state->creativePage;
            if (state->flow.screen == GameScreen::FURNACE && state->openFurnace) {
                auto it = state->furnaces.find(*state->openFurnace);
                if (it != state->furnaces.end()) {
                    view.furnaceInput = it->second.input;
                    view.furnaceFuel = it->second.fuel;
                    view.furnaceOutput = it->second.output;
                    view.furnaceCook = it->second.cookFraction();
                    view.furnaceFuelLeft = it->second.fuelFraction();
                }
            }
            if (state->flow.screen == GameScreen::CHEST && state->openChest) {
                auto it = state->chests.find(*state->openChest);
                if (it != state->chests.end()) {
                    view.chestSlots = it->second.slots;
                }
            }
        }
        if (state->flow.inContainer() && state->flow.screen != GameScreen::FURNACE &&
            state->flow.screen != GameScreen::CHEST && state->gameMode != GameMode::CREATIVE) {
            const int gridSize = state->flow.screen == GameScreen::CRAFTING ? 9 : 4;
            const int gridWidth = state->flow.screen == GameScreen::CRAFTING ? 3 : 2;
            const auto result = matchCraftingRecipe(
                std::span<const ItemStack>(state->craftGrid.data(), static_cast<size_t>(gridSize)),
                gridWidth);
            state->craftResult = result.value_or(ItemStack{});
            menuCtx.container.craftResult = state->craftResult;
        }
        state->menuLayout = buildScreenLayout(state->flow.screen, boundsW, boundsH, menuCtx);

        // Installation, qualification, and entry preparation are modal. The
        // complete gameplay menu is still built above so normal world,
        // inventory, and container screens resume without losing state as
        // soon as the v4 gate opens.
        const bool preparingPreWorldV4 = !state->diagnosticV3 && state->v4ProfileOpened &&
                                         !state->world && !state->v4EntryReady &&
                                         state->startupFailure.empty() && state->terrainBootstrap &&
                                         state->terrainBootstrap->ready();
        if (preparingPreWorldV4 || (state->world && !state->diagnosticV3 && !state->v4EntryReady &&
                                    state->startupFailure.empty())) {
            const RenderPipeline::ChunkRenderStats streaming = _renderPipeline->chunkRenderStats();
            const int configuredHorizonRadiusChunks =
                state->world ? farTerrainEntryHorizonViewDistance(state->world->getViewDistance())
                             : 0;
            const int entryHorizonRadiusChunks =
                v4RequiredEntryParentRadiusChunks(configuredHorizonRadiusChunks);
            const V4WorldPreparationSnapshot preparation{
                // A canonically certified candidate has located dry land
                // while exact collision and headroom finish. A learned-only
                // continental proposal remains in the land-search phase
                // until radius-zero exact validation accepts it.
                .drySpawnValidated = v4CanonicalDrySpawnAccepted(state->v4SpawnCandidateActive,
                                                                 state->v4SpawnCandidateProvisional,
                                                                 state->v4SpawnSafetyValidated),
                .finalSpawnTerrainReady =
                    state->v4SpawnAuthorityStatus == V4SpawnAuthorityPrequeueStatus::Ready,
                .safeSpawnReady = state->world && state->v4SpawnFinalized &&
                                  state->world->exactSpawnBandReady(
                                      static_cast<int64_t>(std::floor(state->player.position.x)),
                                      static_cast<int32_t>(std::floor(state->player.position.y)),
                                      static_cast<int64_t>(std::floor(state->player.position.z)),
                                      V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS) &&
                                  state->world->playableSpawnCollisionReady(
                                      static_cast<int64_t>(std::floor(state->player.position.x)),
                                      static_cast<int32_t>(std::floor(state->player.position.y)),
                                      static_cast<int64_t>(std::floor(state->player.position.z))),
                .configuredHorizonRadiusChunks = configuredHorizonRadiusChunks,
                .entryHorizonRadiusChunks = entryHorizonRadiusChunks,
                .connectedParentRadiusChunks =
                    state->world
                        ? v4ConnectedParentRadiusChunks(streaming, configuredHorizonRadiusChunks)
                        : 0.0F,
                .farBaseReady = streaming.farBaseResidentTileCount,
                .farBaseRequired = streaming.farBaseWantedTileCount,
                // The renderer's count includes only FINAL-compatible
                // children with FINAL parents and legal shared boundaries.
                // Hide a retained prior closure during its one-frame epoch
                // revalidation instead of reporting stale detail as ready.
                .nearFinalReady = v4NearEntryFinalCompatibleProgress(
                    streaming.farProtectedNearCurrentEpoch, streaming.farProtectedNearClosureEpoch,
                    streaming.farProtectedNearResidentTileCount),
                .nearFinalRequired = V4_ENTRY_FINAL_TARGET_COUNT,
                .elapsedSeconds =
                    state->v4EntryStartedAt > 0.0
                        ? std::max(0.0, CACurrentMediaTime() - state->v4EntryStartedAt)
                        : 0.0,
            };
            state->menuLayout = buildV4WorldPreparationLayout(preparation, boundsW, boundsH);
        } else if (((!state->world && state->v4OpenRequested) || !state->startupFailure.empty()) &&
                   state->terrainBootstrap) {
            worldgen::bootstrap::TerrainBootstrapSnapshot startup =
                state->terrainBootstrap->snapshot();
            if (!state->startupFailure.empty() &&
                !state->terrainRepairRecovery.load(std::memory_order_acquire)) {
                bool retryable = false;
                if (state->startupFailureRetryable) {
                    retryable = *state->startupFailureRetryable;
                } else if (state->world && state->world->generationContext()) {
                    const std::optional<worldgen::learned::GenerationFailure> failure =
                        state->world->generationContext()->failure();
                    retryable = failure && failure->retriable;
                } else if (state->terrainBootstrap->ready()) {
                    // Authority failures latch before World construction. A
                    // retry must follow that failure's policy instead of
                    // treating every qualified runtime as retryable.
                    const auto context = state->terrainBootstrap->qualifiedGenerationContext();
                    const std::optional<worldgen::learned::GenerationFailure> failure =
                        context ? context->failure() : std::nullopt;
                    // Persistence and fresh-profile publication failures have
                    // no context failure and may retry after identity checks.
                    retryable = !failure || failure->retriable;
                }
                startup.state = worldgen::bootstrap::TerrainBootstrapState::Failed;
                startup.detail = state->startupFailure;
                startup.failure = worldgen::bootstrap::TerrainBootstrapFailure{
                    .code = worldgen::bootstrap::TerrainBootstrapFailureCode::Qualification,
                    .message = state->startupFailure,
                    .retryable = retryable,
                };
            }
            state->menuLayout = buildTerrainBootstrapLayout(startup, boundsW, boundsH,
                                                            state->startupFailureAllowsWorldSelect);
        }

        Vec2 mouse = input.mousePosition;
        const UIHit hit = uiHitTest(state->menuLayout, mouse.x / boundsW, mouse.y / boundsH);
        state->hoveredButton = hit.kind == UIHitKind::BUTTON ? hit.index : -1;
        state->hoveredSlot = hit.kind == UIHitKind::SLOT ? hit.index : -1;

        // Slot interaction. A held cursor plus a press over a slot begins a
        // drag that paints every slot the button crosses and distributes on
        // release (left even-splits, right spreads one apiece); a quick second
        // left click on the same slot gathers instead; a lone click picks,
        // places, or quick-moves through applySlotClick.
        SlotAccess access = [self slotAccessForScreen];
        const SlotRef hovered = hit.kind == UIHitKind::SLOT
                                    ? state->menuLayout.slots[static_cast<size_t>(hit.index)].ref
                                    : SlotRef{};
        const auto sameSlot = [](SlotRef a, SlotRef b) {
            return a.domain == b.domain && a.index == b.index;
        };

        if (hit.kind == UIHitKind::SLOT &&
            (input.isJustPressed(Key::MouseLeft) || input.isJustPressed(Key::MouseRight))) {
            const bool isLeft = input.isJustPressed(Key::MouseLeft);
            const double now = CACurrentMediaTime();
            const bool doubleClick =
                isLeft && !state->cursorStack.empty() &&
                now - state->lastSlotClickSeconds <= EngineState::DOUBLE_CLICK_WINDOW &&
                sameSlot(state->lastSlotClickRef, hovered);
            if (doubleClick) {
                if (applyDoubleClick(access, state->cursorStack).changed) {
                    [self playUiSfx:state->sfxClick gain:0.3f];
                }
                state->lastSlotClickSeconds = -1.0; // consume: a third click restarts
            } else if (!state->cursorStack.empty()) {
                state->dragActive = true;
                state->dragKind = isLeft ? SlotClickKind::LEFT : SlotClickKind::RIGHT;
                state->dragMoved = false;
                state->dragSlots.clear();
                state->dragSlots.push_back(hovered);
                if (isLeft) {
                    state->lastSlotClickSeconds = now;
                    state->lastSlotClickRef = hovered;
                }
            } else {
                const SlotClickKind kind = !isLeft ? SlotClickKind::RIGHT
                                           : input.isDown(Key::LeftShift)
                                               ? SlotClickKind::SHIFT_LEFT
                                               : SlotClickKind::LEFT;
                if (applySlotClick(access, state->cursorStack, hovered, kind).changed) {
                    [self playUiSfx:state->sfxClick gain:0.3f];
                }
                if (isLeft) {
                    state->lastSlotClickSeconds = now;
                    state->lastSlotClickRef = hovered;
                }
            }
        }

        // Paint newly crossed slots while the drag button stays down.
        if (state->dragActive && hit.kind == UIHitKind::SLOT) {
            bool present = false;
            for (const SlotRef ref : state->dragSlots) {
                if (sameSlot(ref, hovered)) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                state->dragSlots.push_back(hovered);
                state->dragMoved = true;
            }
        }

        // Release finalizes: a real multi-slot drag distributes, a lone
        // press-release falls back to a normal place click.
        const Key dragButton =
            state->dragKind == SlotClickKind::RIGHT ? Key::MouseRight : Key::MouseLeft;
        if (state->dragActive && input.isJustReleased(dragButton)) {
            if (state->dragMoved && state->dragSlots.size() >= 2) {
                if (applySlotDrag(access, state->cursorStack,
                                  std::span<const SlotRef>(state->dragSlots), state->dragKind)
                        .changed) {
                    [self playUiSfx:state->sfxClick gain:0.3f];
                }
            } else if (!state->dragSlots.empty() &&
                       applySlotClick(access, state->cursorStack, state->dragSlots.front(),
                                      state->dragKind)
                           .changed) {
                [self playUiSfx:state->sfxClick gain:0.3f];
            }
            state->dragActive = false;
            state->dragMoved = false;
            state->dragSlots.clear();
        }

        if (input.isJustPressed(Key::MouseLeft)) {
            if (hit.kind == UIHitKind::TEXT_FIELD) {
                [self commitTextEntry];
                state->worldCreate.focusedField = hit.index;
                input.beginTextEntry(hit.index == 0 ? state->worldCreate.name
                                                    : state->worldCreate.seedText);
                [self playUiSfx:state->sfxClick gain:0.3f];
            } else {
                [self commitTextEntry];
            }
            if (hit.kind == UIHitKind::BUTTON) {
                const MenuButton& button =
                    state->menuLayout.buttons[static_cast<size_t>(hit.index)];
                [self playUiSfx:state->sfxClick gain:0.3f];
                [self handleMenuAction:button.action payload:button.payload];
            }
        }
    } else {
        state->hoveredButton = -1;
        // No container open: abandon any half-finished drag so it never leaks
        // a phantom deposit into the next screen.
        state->dragActive = false;
        state->dragSlots.clear();

        // Hotbar: scroll wheel cycles slots (frame-level, gameplay only)
        _scrollAccumulator += input.scrollDelta;
        while (_scrollAccumulator >= 10.0f) {
            state->inventory.selectNext();
            _scrollAccumulator -= 10.0f;
        }
        while (_scrollAccumulator <= -10.0f) {
            state->inventory.selectPrev();
            _scrollAccumulator += 10.0f;
        }
    }
}

// ---- Clean quit: save the world, then terminate through AppKit ----

- (BOOL)quiesceForApplicationTerminationRequiringSave:(BOOL)requireSave {
    EngineState* state = _state.get();
    if (!requireSave)
        _savedWorld = true;

    ApplicationTerminationActions actions{
        .saveDurableState =
            [self, state] {
                // Capture runs are disposable and must not persist injected blocks
                // or a drifted player position. Every ordinary quit remains
                // cancelable until the world save succeeds.
                if (!std::getenv("RYCRAFT_CAPTURE") && ![self saveCurrentWorld])
                    return false;
                if (!std::getenv("RYCRAFT_CAPTURE") && !state->envOverridesActive)
                    saveSettings(settingsPath(), state->settings, state->gfx);
                self->_savedWorld = true;
                return true;
            },
        .cancelBootstrap =
            [state] {
                if (state->terrainBootstrap)
                    state->terrainBootstrap->cancel();
            },
        .stopRenderWorkers =
            [self] {
                if (self->_renderPipeline)
                    self->_renderPipeline->shutdownMeshWorkers();
            },
        .stopWorldAndGenerationWorkers =
            [self] {
                // Durable save completion makes stopWorld infallible here. It
                // joins render-owned schedulers again idempotently before World
                // joins exact generation and SaveManager joins persistence.
                static_cast<void>([self stopWorld]);
            },
        .releaseGenerationOwners = [self] { [self releaseTerrainGenerationOwners]; },
        .releaseRuntime = [self] { [self releaseTerrainRuntime]; },
    };
    return _terminationQuiescence.quiesce(actions, requireSave) ? YES : NO;
}

- (BOOL)saveWorldState {
    // AppKit terminates through exit rather than releasing the shared Engine
    // singleton. Explicit quiescence keeps ONNX static finalizers from racing
    // live runtime workers. The sequence is idempotent across every AppKit
    // callback and capture-driven quit.
    return [self quiesceForApplicationTerminationRequiringSave:YES];
}

- (void)requestQuit {
    if (_state->inputManager) {
        _state->inputManager->releaseMouse();
    }
    [NSApp terminate:self];
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    (void)sender;
    return [self saveWorldState] ? NSTerminateNow : NSTerminateCancel;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    (void)sender;
    // The red close button routes through applicationShouldTerminate: above,
    // so the save-on-quit path has no duplicate
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    (void)notification;
    // Belt and braces: the mouse association is global system state
    CGAssociateMouseAndMouseCursorPosition(true);
    static_cast<void>([self saveWorldState]);
}

- (void)windowDidResignKey:(NSNotification*)notification {
    (void)notification;
    // Automated movement must keep exercising streaming when the launcher's
    // terminal regains focus. Release the global pointer lock while leaving
    // the simulation on the playing screen; gameTick restores only the
    // synthetic movement keys on its next fixed tick.
    if (std::getenv("RYCRAFT_AUTOPILOT")) {
        if (_state->inputManager) {
            _state->inputManager->releaseMouse();
        }
        return;
    }
    // Cmd-Tab (or any focus loss) must never leave the pointer locked
    [self applyFlowEffects:_state->flow.onFocusLost()];
}

// ---- Game Tick (fixed timestep at 20Hz) ----

- (void)gameTick:(EngineState*)state {
    if (!state->inputManager || !state->world)
        return;

    InputState& input = state->inputManager->state();

    // Playtest hook: walk and sprint exercise ground physics. Fly holds a
    // level aerial route above obstacles so streaming benchmarks cross many
    // chunk boundaries without depending on one terrain fixture.
    static const char* autopilot = std::getenv("RYCRAFT_AUTOPILOT");
    if (autopilot) {
        const std::string_view mode{autopilot};
        const bool active = state->frameCount >= state->autopilotStartFrame &&
                            state->frameCount < state->autopilotStopFrame;
        const bool aerial = mode == "fly";
        if (aerial)
            state->player.flying = true;
        input.keysDown[Key::W] = active && (mode == "walk" || mode == "sprint" || aerial);
        input.keysDown[Key::LeftControl] = active && (mode == "sprint" || aerial);
    }

    // 1. Advance world time (1 tick per game tick). Playtest hooks:
    // RYCRAFT_TIME=<absolute world tick> pins time and lunar phase at launch;
    // RYCRAFT_TIME_FREEZE=1 stops it advancing, so captures at noon /
    // sunset / midnight don't depend on hand-editing the save.
    static const bool freezeTime = [] {
        const char* f = std::getenv("RYCRAFT_TIME_FREEZE");
        return f && *f && std::strcmp(f, "0") != 0;
    }();
    if (!freezeTime && state->generation.dayCycle &&
        state->worldTime < std::numeric_limits<uint64_t>::max()) {
        state->worldTime++;
    }

    // Furnaces are part of fixed-step simulation, not their container UI.
    // Keeping this here lets every placed furnace cook during normal play;
    // the container-only loop above covers the intentionally paused screen.
    [self tickFurnaces:state];
    if (state->frameCount % 20 == 0) {
        [self reconcileChestBlocks:state];
    }

    // 1b. Unstick a stale spawn: a resumed save can place the player inside
    // terrain when world generation has changed shape since the save was
    // written, collision then zeroes every move. Once the spawn chunk
    // exists, lift the player to the surface if they are embedded.
    if (!state->spawnValidated) {
        int64_t px = static_cast<int64_t>(std::floor(state->player.position.x));
        int64_t pz = static_cast<int64_t>(std::floor(state->player.position.z));
        int32_t feetY = static_cast<int32_t>(std::floor(state->player.position.y));
        auto chunk = state->world->getChunk(Chunk::worldToChunk(px), Chunk::worldToChunkY(feetY),
                                            Chunk::worldToChunk(pz));
        if (chunk && chunk->generated) {
            if (state->pendingBedSpawnValidation) {
                const BedSpawnValidation validation =
                    validateBedSpawnCells(state->world->findBlockIfLoaded(px, feetY - 1, pz),
                                          state->world->findBlockIfLoaded(px, feetY, pz),
                                          state->world->findBlockIfLoaded(px, feetY + 1, pz));
                if (validation == BedSpawnValidation::DEFERRED) {
                    state->world->updatePlayerPosition(px, feetY, pz);
                    state->world->publishLoadedSnapshot();
                    input.clearTickPresses();
                    return;
                }
                state->pendingBedSpawnValidation = false;
                if (validation == BedSpawnValidation::INVALID) {
                    state->bedSpawnSet = false;
                    state->worldSpawn = state->v4SafeSpawnPos.value_or(state->requestedSpawn);
                    state->player.position = state->worldSpawn;
                    state->player.velocity = Vec3{0.f, 0.f, 0.f};
                    const int64_t fallbackX =
                        static_cast<int64_t>(std::floor(state->player.position.x));
                    const int32_t fallbackY =
                        static_cast<int32_t>(std::floor(state->player.position.y));
                    const int64_t fallbackZ =
                        static_cast<int64_t>(std::floor(state->player.position.z));
                    state->world->updatePlayerPosition(fallbackX, fallbackY, fallbackZ);
                    state->world->publishLoadedSnapshot();
                    input.clearTickPresses();
                    RY_LOG_INFO("Bed spawn was missing or obstructed; using the safe world spawn");
                    return;
                }
            }
            bool embedded = isSolid(state->world->getBlock(px, feetY, pz)) ||
                            isSolid(state->world->getBlock(px, feetY + 1, pz));
            if (embedded) {
                int32_t surfaceY = static_cast<int32_t>(state->world->getTerrainHeight(px, pz));
                state->player.position.y = static_cast<float>(surfaceY + 2);
                state->player.velocity = Vec3{0.f, 0.f, 0.f};
                RY_LOG_INFO("Spawn was inside terrain and moved to the surface");
            }

            // Playtest hook: RYCRAFT_SPAWN_LAVA=1 carves a small surface lava
            // pool a few blocks from spawn so headless captures can show the
            // block-light glow (natural lava only forms in deep caves). Best
            // paired with a night RYCRAFT_TIME so the orange spill stands out.
            static const bool spawnLava = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_LAVA");
                return v && *v && std::strcmp(v, "0") != 0;
            }();
            if (spawnLava) {
                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int64_t wx = px + dx;
                        int64_t wz = pz + 6 + dz; // ahead (+Z) at spawn
                        int32_t searchTop = static_cast<int32_t>(
                            std::clamp(state->world->getTerrainHeight(wx, wz) + CHUNK_EDGE,
                                       static_cast<double>(WORLD_MIN_Y + 1),
                                       static_cast<double>(WORLD_MAX_Y - 1)));
                        for (int32_t y = searchTop; y > WORLD_MIN_Y; --y) {
                            if (isSolid(state->world->getBlock(wx, y, wz))) {
                                state->world->setBlock(wx, y, wz, BlockType::LAVA);
                                state->world->setBlock(wx, y + 1, wz, BlockType::AIR);
                                break;
                            }
                        }
                    }
                }
            }

            // Playtest hook: RYCRAFT_SPAWN_WATER=1 sinks a broad pool ahead of
            // spawn (between the camera and the tree line) so headless captures
            // can show the water reflections/refraction at a grazing angle.
            static const bool spawnWater = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_WATER");
                return v && *v && std::strcmp(v, "0") != 0;
            }();
            if (spawnWater) {
                auto isTrunk = [](BlockType b) {
                    return b == BlockType::LOG || b == BlockType::BIRCH_LOG ||
                           b == BlockType::SPRUCE_LOG || b == BlockType::ACACIA_LOG ||
                           b == BlockType::JUNGLE_LOG || b == BlockType::MANGROVE_LOG ||
                           b == BlockType::PALM_LOG || b == BlockType::WILLOW_LOG;
                };
                for (int dz = 3; dz <= 22; ++dz) {
                    for (int dx = -10; dx <= 10; ++dx) {
                        int64_t wx = px + dx;
                        int64_t wz = pz + dz;
                        // Scan down to the terrain surface, past any tree trunk,
                        // so the pool sits at ground level (no floating water).
                        int32_t searchTop = static_cast<int32_t>(
                            std::clamp(state->world->getTerrainHeight(wx, wz) + CHUNK_EDGE,
                                       static_cast<double>(WORLD_MIN_Y + 1),
                                       static_cast<double>(WORLD_MAX_Y - 1)));
                        for (int32_t y = searchTop; y > WORLD_MIN_Y; --y) {
                            BlockType b = state->world->getBlock(wx, y, wz);
                            if (isSolid(b) && !isTrunk(b)) {
                                // Carve a pool six blocks deep (over-water shots
                                // see the surface; walking in submerges the
                                // camera for underwater captures), air above.
                                for (int d = 0; d < 6; ++d) {
                                    state->world->setBlock(wx, y - d, wz, BlockType::WATER);
                                }
                                state->world->setBlock(wx, y + 1, wz, BlockType::AIR);
                                break;
                            }
                        }
                    }
                }
            }

            // The capture-only material fixture exposes bed geometry, fixed
            // -Z chest and furnace fronts, torch emission, and active-furnace
            // emission in one automated Metal frame. The capture requirement
            // keeps these transient edits out of normal saves even when
            // RYCRAFT_SPAWN_MATERIALS is set accidentally.
            static const bool spawnMaterials = materialPlaytestFixtureEnabled(
                std::getenv("RYCRAFT_SPAWN_MATERIALS"), std::getenv("RYCRAFT_CAPTURE"));
            if (spawnMaterials) {
                constexpr int PLATFORM_HALF_WIDTH = 6;
                constexpr int PLATFORM_FAR_Z = 7;
                constexpr int PLATFORM_HEADROOM = 6;
                const auto isTrunk = [](BlockType block) {
                    return block == BlockType::LOG || block == BlockType::BIRCH_LOG ||
                           block == BlockType::SPRUCE_LOG || block == BlockType::ACACIA_LOG ||
                           block == BlockType::JUNGLE_LOG || block == BlockType::MANGROVE_LOG ||
                           block == BlockType::PALM_LOG || block == BlockType::WILLOW_LOG;
                };
                const int32_t predictedSurface = static_cast<int32_t>(std::floor(std::clamp(
                    state->world->getTerrainHeight(px, pz), static_cast<double>(WORLD_MIN_Y + 1),
                    static_cast<double>(WORLD_MAX_Y - 8))));

                // The ordinary spawn check only guarantees that the cube
                // containing the requested feet position is resident. The
                // generated surface can occupy a different vertical section,
                // and the fixture crosses horizontal cube boundaries at many
                // useful capture coordinates. Explicitly admit the bounded
                // capture-only footprint before using loaded-only queries or
                // edits. Without this, the hook silently completes without
                // placing anything on a first visit to a world.
                const int64_t minChunkX = Chunk::worldToChunk(px - PLATFORM_HALF_WIDTH);
                const int64_t maxChunkX = Chunk::worldToChunk(px + PLATFORM_HALF_WIDTH);
                const int64_t minChunkZ = Chunk::worldToChunk(pz);
                const int64_t maxChunkZ = Chunk::worldToChunk(pz + PLATFORM_FAR_Z);
                const int32_t minChunkY =
                    Chunk::worldToChunkY(std::max(WORLD_MIN_Y, predictedSurface - 2 * CHUNK_EDGE));
                const int32_t maxChunkY = Chunk::worldToChunkY(
                    std::min(WORLD_MAX_Y, predictedSurface + PLATFORM_HEADROOM + 3));
                for (int32_t chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY) {
                    for (int64_t chunkZ = minChunkZ; chunkZ <= maxChunkZ; ++chunkZ) {
                        for (int64_t chunkX = minChunkX; chunkX <= maxChunkX; ++chunkX) {
                            state->world->getChunk(chunkX, chunkY, chunkZ);
                        }
                    }
                }

                std::optional<int32_t> fixtureSupportY;
                for (int32_t y = predictedSurface + 2;
                     y >= std::max(WORLD_MIN_Y + 1, predictedSurface - 2 * CHUNK_EDGE); --y) {
                    const std::optional<BlockType> support =
                        state->world->findBlockIfLoaded(px, y, pz);
                    if (!support || !hasFullBlockCollision(*support) || isLeafBlock(*support) ||
                        isTrunk(*support)) {
                        continue;
                    }
                    fixtureSupportY = y;
                    break;
                }

                if (fixtureSupportY) {
                    // A deterministic neutral pad makes every requested face
                    // visible even when the spawn is wooded, steep, or covered
                    // in flora. This path is already capture-only and no-save.
                    for (int dz = 0; dz <= PLATFORM_FAR_Z; ++dz) {
                        for (int dx = -PLATFORM_HALF_WIDTH; dx <= PLATFORM_HALF_WIDTH; ++dx) {
                            const int64_t wx = px + dx;
                            const int64_t wz = pz + dz;
                            state->world->setBlock(wx, *fixtureSupportY, wz, BlockType::STONE);
                            for (int dy = 1; dy <= PLATFORM_HEADROOM; ++dy) {
                                state->world->setBlock(wx, *fixtureSupportY + dy, wz,
                                                       BlockType::AIR);
                            }
                        }
                    }

                    const int32_t placementY = *fixtureSupportY + 1;
                    for (size_t index = 0; index < MATERIAL_PLAYTEST_BLOCKS.size(); ++index) {
                        const BlockType fixtureBlock = MATERIAL_PLAYTEST_BLOCKS[index];
                        const int64_t wx = px + (static_cast<int64_t>(index) - 2) * 2;
                        const int64_t wz = pz + 5;
                        state->world->setBlock(wx, placementY, wz, fixtureBlock);
                        const BlockPos position{wx, placementY, wz};
                        if (fixtureBlock == BlockType::CHEST) {
                            state->chests[position] = ChestState{};
                        } else if (fixtureBlock == BlockType::FURNACE) {
                            state->furnaces[position] = FurnaceState{};
                            state->furnaceVisualAuthority->set(position, BlockType::FURNACE);
                        } else if (fixtureBlock == BlockType::FURNACE_LIT) {
                            FurnaceState burning;
                            burning.input = ItemStack{ItemType::RAW_BEEF, 1, 0};
                            burning.fuel = ItemStack{ItemType::COAL, 1, 0};
                            burning.burnTicksRemaining = 800;
                            burning.burnTicksTotal = 1600;
                            burning.cookTicks = 40;
                            state->furnaces[position] = burning;
                            state->furnaceVisualAuthority->set(position, BlockType::FURNACE_LIT);
                        }
                    }
                }
            }

            // Playtest hook: RYCRAFT_SPAWN_ITEMS=N scatters N dropped items on
            // the loaded ground ahead of spawn so headless captures can show
            // the item entities falling and resting (never in unloaded space,
            // where closed missing cubes would freeze them mid-air).
            static const int spawnItems = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_ITEMS");
                return v ? std::clamp(std::atoi(v), 0, 64) : 0;
            }();
            for (int i = 0; i < spawnItems; ++i) {
                const auto type =
                    CREATIVE_PALETTE[static_cast<size_t>(i) % CREATIVE_PALETTE.size()];
                const int64_t wx = px + (i % 8) - 3;
                const int64_t wz = pz + 3 + i / 8;
                const int32_t surface = static_cast<int32_t>(std::clamp(
                    state->world->getTerrainHeight(wx, wz), static_cast<double>(WORLD_MIN_Y + 1),
                    static_cast<double>(WORLD_MAX_Y - 1)));
                const Vec3 pos{static_cast<float>(wx) + 0.5f, static_cast<float>(surface) + 3.0f,
                               static_cast<float>(wz) + 0.5f};
                state->itemEntities.spawn(makeItemStack(type, 1), pos, Vec3{0.f, 0.f, 0.f}, 0);
            }

            // Playtest hook: RYCRAFT_SPAWN_BOAT=1 drops a boat a few blocks
            // ahead of spawn so headless captures can show it rendering (on
            // the RYCRAFT_SPAWN_WATER pool it floats).
            static const bool spawnBoat = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_BOAT");
                return v && *v && std::strcmp(v, "0") != 0;
            }();
            if (spawnBoat) {
                const int64_t wx = px;
                const int64_t wz = pz + 4;
                const int32_t surface = static_cast<int32_t>(std::clamp(
                    state->world->getTerrainHeight(wx, wz), static_cast<double>(WORLD_MIN_Y + 1),
                    static_cast<double>(WORLD_MAX_Y - 1)));
                state->boats.spawn(Vec3{static_cast<float>(wx) + 0.5f,
                                        static_cast<float>(surface) + 2.0f,
                                        static_cast<float>(wz) + 0.5f},
                                   0.f);
            }

            // Playtest hook: RYCRAFT_YAW / RYCRAFT_PITCH (degrees) point the
            // camera for captures, e.g. face the afternoon sun for the lens
            // flare. Yaw 0 looks +Z; -90 looks -X.
            static const char* yawEnv = std::getenv("RYCRAFT_YAW");
            static const char* pitchEnv = std::getenv("RYCRAFT_PITCH");
            if (yawEnv || pitchEnv) {
                constexpr float DEG = static_cast<float>(M_PI) / 180.0f;
                state->camera.setLook(yawEnv ? std::atof(yawEnv) * DEG : 0.0f,
                                      pitchEnv ? std::atof(pitchEnv) * DEG : 0.0f);
            }

            state->spawnValidated = true;
        }
    }

    // 2. Hotbar input: keys 1-9 select slots
    for (int i = 0; i < Inventory::HOTBAR_SLOTS; ++i) {
        Key key = static_cast<Key>(static_cast<int>(Key::One) + i);
        if (input.isPressedForTick(key)) {
            state->inventory.selectSlot(i);
            break;
        }
    }

    // 3. Update camera from player state (at EYE height, not the feet), then
    // consume the look delta so a second tick in the same frame doesn't
    // re-apply it
    Vec3 eyePosition = state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f};
    state->camera.setPosition(eyePosition);
    InputBindings bindings;
    state->camera.update(state->deltaTime, input, bindings, eyePosition);
    input.clearMouseDelta();

    // 4. Sync player yaw/pitch from camera so WASD uses the correct
    // direction and swimming can follow the look direction
    state->player.yaw = state->camera.yaw();
    state->player.pitch = state->camera.pitch();

    // 4b. Boat riding: while seated, WASD paddles the boat and the rider is
    // slaved to the seat instead of running the normal player physics. Jump or
    // sneak dismounts to the side.
    bool riding =
        state->ridingBoat >= 0 && state->ridingBoat < static_cast<int>(state->boats.boats().size());
    Vec3 look = state->camera.forward();
    Vec3 horizForward{look.x, 0.f, look.z};
    const float horizLen = horizForward.length();
    if (horizLen > 1e-4f) {
        horizForward.x /= horizLen;
        horizForward.z /= horizLen;
    }
    const Vec3 horizRight{-horizForward.z, 0.f, horizForward.x};
    if (riding &&
        (input.isPressedForTick(bindings.jump.key) || input.isPressedForTick(bindings.sneak.key))) {
        Boat& boat = state->boats.boats()[static_cast<size_t>(state->ridingBoat)];
        state->player.position =
            boat.position + horizRight + Vec3{0.f, 0.4f, 0.f}; // step out to the side
        state->player.velocity = Vec3{0.f, 0.f, 0.f};
        state->ridingBoat = -1;
        riding = false;
    }
    if (riding) {
        Boat& boat = state->boats.boats()[static_cast<size_t>(state->ridingBoat)];
        constexpr float PADDLE = 0.02f;
        Vec3 accel{0.f, 0.f, 0.f};
        if (input.isDown(bindings.forward.key))
            accel += horizForward * PADDLE;
        if (input.isDown(bindings.backward.key))
            accel -= horizForward * PADDLE;
        if (input.isDown(bindings.right.key))
            accel += horizRight * PADDLE;
        if (input.isDown(bindings.left.key))
            accel -= horizRight * PADDLE;
        boat.yaw = std::atan2(horizForward.x, horizForward.z);
        state->boats.tick(*state->world, state->player.position, state->ridingBoat, accel);
        // Seat the rider on the boat and freeze player physics for the tick.
        state->player.position = boat.position + Vec3{0.f, 0.3f, 0.f};
        state->player.velocity = Vec3{0.f, 0.f, 0.f};
        state->player.onGround = true;
        state->player.resetFallDistance();
    }

    // 5. Player physics tick. All movement keys are decoded through the
    // bindings here (sprint once read a hardcoded key and fired on sneak);
    // Player itself never touches the input layer.
    bool jumpedThisTick = false;
    Vec3 playerPositionBeforeMove = state->player.position;
    if (!riding) {
        PlayerInput playerInput;
        playerInput.forward = input.isDown(bindings.forward.key);
        playerInput.backward = input.isDown(bindings.backward.key);
        playerInput.left = input.isDown(bindings.left.key);
        playerInput.right = input.isDown(bindings.right.key);
        playerInput.jumpHeld = input.isDown(bindings.jump.key);
        playerInput.jumpPressed = input.isPressedForTick(bindings.jump.key);
        playerInput.sprintHeld = input.isDown(bindings.sprint.key);
        playerInput.descendHeld = input.isDown(bindings.sneak.key);
        playerInput.doubleTapForward = input.isDoubleTappedForTick(bindings.forward.key);
        playerInput.doubleTapJump = input.isDoubleTappedForTick(bindings.jump.key);
        playerInput.allowFlight = modeAllowsFlight(state->gameMode);
        playerInput.takesFallDamage = modeTakesDamage(state->gameMode);
        // Low food disables sprinting in survival.
        if (modeDrainsHunger(state->gameMode) &&
            state->survival.food <= SurvivalStats::SPRINT_DISABLE_FOOD) {
            playerInput.sprintHeld = false;
            playerInput.doubleTapForward = false;
        }
        jumpedThisTick = playerInput.jumpPressed && state->player.onGround;
        const int healthBeforeMove = state->player.health;
        state->player.tick(*state->world, playerInput);
        // Player::tick applies fall damage directly to health; catch it here for
        // the hurt sound and the death-screen cause.
        if (state->player.health < healthBeforeMove && modeTakesDamage(state->gameMode)) {
            state->deathMessage = "YOU FELL";
            if (state->hurtSoundCooldown == 0) {
                [self playSfx:state->sfxHurt gain:0.7f];
                state->hurtSoundCooldown = 10;
            }
        }
        // On foot, drift every floating boat once this tick (the ridden boat
        // already stepped above with the rider's paddle).
        state->boats.tick(*state->world, state->player.position, -1, Vec3{});
    }
    if (state->hurtSoundCooldown > 0)
        --state->hurtSoundCooldown;

    // 5b. Footsteps: one thud roughly every two blocks walked on the ground
    if (state->audio) {
        Vec3 moved = state->player.position - state->lastFootstepPos;
        moved.y = 0.f;
        state->lastFootstepPos = state->player.position;
        if (state->player.onGround) {
            state->footstepDistance += moved.length();
            if (state->footstepDistance >= 2.2f) {
                state->footstepDistance = 0.f;
                [self playSfx:state->sfxFootstep gain:0.5f];
            }
        } else {
            state->footstepDistance = 0.f;
        }
    }

    // 5c. Survival stats: exhaustion, hunger, air, regen, starvation, and
    // drowning. Creative freezes them all.
    if (modeTakesDamage(state->gameMode)) {
        const Vec3 eye = state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f};
        const int64_t ex = static_cast<int64_t>(std::floor(eye.x));
        const int32_t ey = static_cast<int32_t>(std::floor(eye.y));
        const int64_t ez = static_cast<int64_t>(std::floor(eye.z));
        const bool eyesUnderwater =
            state->world->getBlockIfLoaded(ex, ey, ez) == BlockType::WATER &&
            eye.y < static_cast<float>(ey) + state->world->getFluidHeightIfLoaded(ex, ey, ez);

        SurvivalTickInputs inputs;
        inputs.sprinting = state->player.sprinting && state->player.onGround;
        inputs.swimming = state->player.swimming;
        inputs.jumped = jumpedThisTick;
        inputs.eyesUnderwater = eyesUnderwater;
        const int delta = tickSurvivalStats(state->survival, inputs, state->player.health);
        if (delta < 0) {
            [self damagePlayer:state
                        amount:-delta
                       message:eyesUnderwater ? "YOU DROWNED" : "YOU STARVED"];
        } else if (delta > 0) {
            state->player.health = std::min(20, state->player.health + delta);
        }
    }

    // 6. Update loaded chunks around player
    // updatePlayerPosition takes WORLD coordinates (it converts to chunk
    // coords itself), passing pre-converted chunk coords made streaming
    // track chunk/16, so the world unloaded under the player ~200 blocks out
    state->world->updatePlayerPosition(static_cast<int64_t>(std::floor(state->player.position.x)),
                                       static_cast<int32_t>(std::floor(state->player.position.y)),
                                       static_cast<int64_t>(std::floor(state->player.position.z)));
    state->world->tickFluids(EngineState::TICK_DT);

    // 6b. Animals: initial population once the spawn area streamed in, then
    // per-tick AI steering + physics for every living entity
    if (state->spawner) {
        state->spawner->updatePopulation(state->player.position);

        auto& entities = state->spawner->getEntities();
        auto& spatialHash = state->spawner->getSpatialHash();
        std::unordered_set<uint64_t> liveEntityIds;
        liveEntityIds.reserve(entities.size());
        for (const auto& entity : entities) {
            if (entity)
                liveEntityIds.insert(entity->id);
        }
        std::erase_if(state->entityBrains,
                      [&](const auto& entry) { return !liveEntityIds.contains(entry.first); });
        // Index-based over a size snapshot with a shared_ptr copy: breeding
        // inside StateMachine::update can push_back into this very vector,
        // which would invalidate range-for iterators mid-loop.
        const size_t entityCount = entities.size();
        for (size_t i = 0; i < entityCount; ++i) {
            std::shared_ptr<Entity> entity = entities[i];
            if (!entity || !entity->alive)
                continue;

            // Simulation distance: distant animals stand still (cheap and
            // invisible, they are beyond the fog anyway)
            Vec3 offset = entity->position - state->player.position;
            if (offset.length() > 96.f)
                continue;

            StateMachine& brain = state->entityBrains[entity->id];
            const bool playerMovingToward = StateMachine::playerMovedToward(
                playerPositionBeforeMove, state->player.position, entity->position);
            Vec3 steering =
                brain.update(*entity, *state->world, state->player.position, playerMovingToward,
                             /*playerHoldingFood=*/false, *state->spawner);
            entity->velocity.x += steering.x;
            entity->velocity.y += steering.y;
            entity->velocity.z += steering.z;
            entity->tick(*state->world);

            // Keep the spatial hash in sync for flocking neighbor queries
            spatialHash.remove(entity->id);
            spatialHash.insert(entity->id, entity->position);
        }

        // Occasional ambient animal call from something nearby
        if (state->animalCallCooldown > 0) {
            --state->animalCallCooldown;
        } else if (state->audio && !entities.empty()) {
            static SeededRng callRng(0xA111CA11u);
            const auto& candidate =
                entities[callRng.nextInt(0, static_cast<int>(entities.size()) - 1)];
            if (candidate && candidate->alive) {
                Vec3 toPlayer = candidate->position - state->player.position;
                if (toPlayer.length() < 24.f && callRng.nextFloat() < 0.3f) {
                    int type = static_cast<int>(candidate->type);
                    [self playSfx:state->sfxAnimal[type] gain:0.35f];
                }
            }
            state->animalCallCooldown = 120 + callRng.nextInt(0, 120); // 6-12s
        }
    }

    // 7. Regional weather. Snapshot construction is asynchronous and
    // latest-wins; the engine retains the prior immutable snapshot while a
    // time-slice or recenter replacement is being built.
    if (state->weatherSystem) {
        const int64_t playerX = static_cast<int64_t>(std::floor(state->player.position.x));
        const int64_t playerZ = static_cast<int64_t>(std::floor(state->player.position.z));
        state->weatherSystem->requestSnapshot(playerX, playerZ, state->worldTime);
        if (std::shared_ptr<const WeatherSnapshot> snapshot =
                state->weatherSystem->latestSnapshot()) {
            state->weatherSnapshot = std::move(snapshot);
        }
        state->localWeather =
            state->weatherSnapshot
                ? state->weatherSnapshot->sample(state->player.position.x, state->player.position.z,
                                                 state->worldTime)
                : state->weatherSystem->sample(state->player.position.x, state->player.position.z,
                                               state->worldTime);
        std::vector<LightningEvent> newLightningEvents =
            state->weatherSystem->lightningEvents(state->lastWeatherEventTick, state->worldTime);
        state->lastWeatherEventTick = state->worldTime;

        // The weather grid supplies a coarse strike height. Resolve the final
        // endpoint against exact deterministic terrain at the generated X/Z
        // before rendering the bolt or scheduling its distance-based thunder.
        for (LightningEvent& event : newLightningEvents) {
            resolveLightningTerrainHeightIfLoaded(*state->world, event);
        }

        const double audioNow = CACurrentMediaTime();
        for (const LightningEvent& event : newLightningEvents) {
            state->thunder.schedule(event, state->player.position.x, state->player.position.y,
                                    state->player.position.z, audioNow);
        }
        state->lightningEvents.insert(state->lightningEvents.end(), newLightningEvents.begin(),
                                      newLightningEvents.end());

        // A fixed capture can inject one visual strike without advancing or
        // replaying the canonical weather event timeline. Waiting for the
        // immutable snapshot gives the bolt the same regional cloud bounds as
        // a generated strike. This hook deliberately bypasses thunder so it
        // cannot alter the scheduler's load-boundary and backlog semantics.
        if (!state->captureLightningInjected && state->captureLightningOverride &&
            state->weatherSnapshot) {
            state->captureLightningInjected = true;
            const CaptureLightningOverride& capture = *state->captureLightningOverride;
            if (capture.ageTicks <= state->worldTime) {
                LightningEvent event;
                event.id = capture.id;
                event.tick = state->worldTime - capture.ageTicks;
                event.x = capture.x;
                event.z = capture.z;
                event.intensity = 1.0F;
                const WeatherSample strikeWeather =
                    state->weatherSnapshot->sample(event.x, event.z, event.tick);
                const uint32_t heightBits =
                    static_cast<uint32_t>(event.id) ^ static_cast<uint32_t>(event.id >> 32U);
                const float heightUnit = static_cast<float>(heightBits) /
                                         static_cast<float>(std::numeric_limits<uint32_t>::max());
                event.cloudY = std::lerp(strikeWeather.cloudBaseY, strikeWeather.cloudTopY,
                                         0.68F + heightUnit * 0.24F);
                resolveLightningTerrainHeightIfLoaded(*state->world, event);
                state->lightningEvents.insert(state->lightningEvents.begin(), event);
            } else {
                RY_LOG_ERROR("Ignoring RYCRAFT_CAPTURE_LIGHTNING whose age exceeds world time");
            }
        }
        std::erase_if(state->lightningEvents, [worldTime = state->worldTime](const auto& event) {
            return worldTime > event.tick && worldTime - event.tick > 11U;
        });
        for (const ScheduledThunder& thunder : state->thunder.popDue(audioNow)) {
            const std::vector<float> samples =
                SoundEffect::generateThunder(thunder.eventId, thunder.gain);
            [self playSfx:samples gain:1.0F];
        }

        const WeatherPreset preset = state->weatherSystem->preset();
        if (preset != WeatherPreset::NATURAL) {
            // Captures need a fully settled, reproducible material state.
            state->wetness =
                preset == WeatherPreset::RAIN || preset == WeatherPreset::STORM ? 1.0f : 0.0f;
        } else {
            // Rain soaks surfaces. Snow remains non-destructive and only adds
            // meltwater near freezing. Temperature, wind, and sunlight all
            // accelerate drying once precipitation eases.
            constexpr float TWO_PI = 6.28318530717958647692f;
            const float dayFraction =
                static_cast<float>(state->worldTime % CELESTIAL_TICKS_PER_DAY) /
                static_cast<float>(CELESTIAL_TICKS_PER_DAY);
            const float sunlight = std::max(0.0f, std::sin(dayFraction * TWO_PI));
            const float wind =
                std::clamp(state->localWeather.windBlocksPerSecond.length() / 12.0f, 0.0f, 1.0f);
            const float warmth =
                std::clamp((state->localWeather.temperatureC + 5.0f) / 30.0f, 0.0f, 1.0f);
            float wetting = state->localWeather.precipitationIntensity;
            if (state->localWeather.precipitationKind == PrecipitationKind::SNOW) {
                wetting *=
                    0.2f * std::clamp((state->localWeather.temperatureC + 2.0f) / 4.0f, 0.0f, 1.0f);
            } else if (state->localWeather.precipitationKind == PrecipitationKind::NONE) {
                wetting = 0.0f;
            }
            const float drying =
                (0.25f + 0.35f * warmth + 0.25f * wind + 0.55f * sunlight) * (1.0f - wetting);
            const float dt = static_cast<float>(EngineState::TICK_DT);
            state->wetness = std::clamp(state->wetness + dt * (wetting / EngineState::SOAK_SECONDS -
                                                               drying / EngineState::DRY_SECONDS),
                                        0.0f, 1.0f);
        }
    }
    if (_renderPipeline) {
        // Particles integrate per fixed tick too, the frame delta here made
        // rainfall speed depend on the frame rate (1/3 speed at 60 FPS).
        _renderPipeline->tickParticles(static_cast<float>(EngineState::TICK_DT), *state->world,
                                       state->player.position, state->localWeather);
        _renderPipeline->setWetness(state->wetness);
    }
    [self syncWeatherAudio];

    // 8. Single raycast for block interaction + highlight
    Vec3 cameraPos = state->camera.position();
    Vec3 forward = state->camera.forward();
    const auto detailedRayHit =
        VoxelTraversal::traceRayDetailed(cameraPos, forward, *state->world, 6.0F);
    const BlockRayHit rayHit =
        detailedRayHit ? BlockRayHit{{detailedRayHit->blockPosition, detailedRayHit->normal}}
                       : std::nullopt;

    // Update block highlight
    if (detailedRayHit) {
        state->highlightedBlock = {.blockPosition = detailedRayHit->blockPosition,
                                   .localBounds = detailedRayHit->localBounds};
        state->hasHighlightedBlock = true;
    } else {
        state->hasHighlightedBlock = false;
    }

    // Attack precedence: a left-click edge on an animal nearer than the block
    // target hits it instead of mining. Reach is shorter than block reach.
    bool attackedEntity = NO;

    // A left click on a boat pops it back into a boat item and suppresses
    // mining for the tick.
    if (input.isPressedForTick(Key::MouseLeft) && !state->boats.empty()) {
        const int boatIndex = state->boats.pick(cameraPos, forward, 3.5f);
        if (boatIndex >= 0) {
            const Vec3 pos = state->boats.boats()[static_cast<size_t>(boatIndex)].position;
            if (state->ridingBoat == boatIndex)
                state->ridingBoat = -1;
            else if (state->ridingBoat > boatIndex)
                --state->ridingBoat; // the removal shifts later indices down
            const ItemStack boatItem{ItemType::BOAT, 1, 0};
            if (modeConsumesItems(state->gameMode)) {
                [self spawnDrop:state
                          stack:boatItem
                            atX:static_cast<int64_t>(std::floor(pos.x))
                              y:static_cast<int32_t>(std::floor(pos.y))
                              z:static_cast<int64_t>(std::floor(pos.z))];
            } else {
                state->inventory.add(boatItem);
            }
            state->boats.remove(static_cast<size_t>(boatIndex));
            [self playSfx:state->sfxBlockBreak gain:0.5f];
            attackedEntity = YES;
        }
    }

    if (!attackedEntity && input.isPressedForTick(Key::MouseLeft) && state->spawner) {
        const auto entityHit = pickEntity(cameraPos, forward, 3.0f, state->spawner->getEntities());
        if (entityHit) {
            float blockDistance = 1e9f;
            if (detailedRayHit)
                blockDistance = detailedRayHit->distance;
            if (entityHit->distance <= blockDistance) {
                [self attackEntity:state entityId:entityHit->entityId];
                attackedEntity = YES;
            }
        }
    }

    // Block breaking. Creative is instant on the click edge; survival mines
    // held-left over time, its speed set by hardness and the held tool.
    if (attackedEntity) {
        // Attacking suppresses mining this tick but keeps any held progress.
    } else if (modeInstantBreak(state->gameMode)) {
        state->miningState.reset();
        if (input.isPressedForTick(Key::MouseLeft)) {
            [self breakBlock:state hit:rayHit withDrops:NO];
        }
    } else {
        bool hasTarget = NO;
        int64_t tx = 0;
        int32_t ty = 0;
        int64_t tz = 0;
        BlockType targetBlock = BlockType::AIR;
        if (rayHit.has_value()) {
            tx = static_cast<int64_t>(std::floor(rayHit->first.x));
            ty = static_cast<int32_t>(std::floor(rayHit->first.y));
            tz = static_cast<int64_t>(std::floor(rayHit->first.z));
            const std::optional<BlockType> block = state->world->findBlockIfLoaded(tx, ty, tz);
            if (block && *block != BlockType::AIR) {
                hasTarget = YES;
                targetBlock = *block;
            }
        }
        const ItemType held = state->inventory.getSelectedStack().type;
        if (tickMining(state->miningState, input.mouseLeftDown, hasTarget, tx, ty, tz, targetBlock,
                       held)) {
            [self breakBlock:state hit:rayHit withDrops:YES];
            state->survival.exhaustion += SurvivalStats::EXHAUST_MINE_BLOCK;
        }
    }

    // Right click precedence: open an interactable block, else eat held food,
    // else place. Opening and placing are click edges; eating is held.
    {
        InputBindings bindings;
        bool interactableTarget = NO;
        if (input.isPressedForTick(Key::MouseRight) && rayHit.has_value() &&
            !input.isDown(bindings.sneak.key)) {
            const int64_t bx = static_cast<int64_t>(std::floor(rayHit->first.x));
            const int32_t by = static_cast<int32_t>(std::floor(rayHit->first.y));
            const int64_t bz = static_cast<int64_t>(std::floor(rayHit->first.z));
            const std::optional<BlockType> block = state->world->findBlockIfLoaded(bx, by, bz);
            if (block && isInteractable(*block)) {
                interactableTarget = YES;
                const BlockPos pos{bx, by, bz};
                if (*block == BlockType::CRAFTING_TABLE) {
                    state->craftResult.clear();
                    [self applyFlowEffects:state->flow.onContainerOpened(GameScreen::CRAFTING)];
                } else if (*block == BlockType::CHEST) {
                    state->openChest = pos;
                    state->chests.try_emplace(pos);
                    [self applyFlowEffects:state->flow.onContainerOpened(GameScreen::CHEST)];
                } else {
                    state->openFurnace = pos;
                    const auto [furnace, inserted] = state->furnaces.try_emplace(pos);
                    if (inserted) {
                        state->furnaceVisualAuthority->set(pos,
                                                           furnaceBlockForState(furnace->second));
                    }
                    [self applyFlowEffects:state->flow.onContainerOpened(GameScreen::FURNACE)];
                }
            } else if (block && *block == BlockType::BED) {
                interactableTarget = YES;
                [self sleepInBed:state atX:bx y:by z:bz];
            }
        }

        // Mounting or placing a boat consumes the right-click before eating or
        // block placement, and works whether or not a block was aimed at.
        if (!interactableTarget && input.isPressedForTick(Key::MouseRight) &&
            [self tryBoatInteraction:state cameraPos:cameraPos forward:forward]) {
            interactableTarget = YES;
        }

        const ItemStack selected = state->inventory.getSelectedStack();
        const bool holdingFood = modeDrainsHunger(state->gameMode) && isFood(selected.type) &&
                                 state->survival.food < SurvivalStats::MAX_FOOD;
        if (!interactableTarget && holdingFood) {
            if (tickEating(state->eatingState, input.mouseRightDown,
                           state->inventory.getSelectedIndex(), true, state->survival.food)) {
                const ItemDefinition def = itemDefinition(selected.type);
                state->survival.food =
                    std::min(SurvivalStats::MAX_FOOD, state->survival.food + def.foodValue);
                state->survival.saturation = std::min(static_cast<float>(state->survival.food),
                                                      state->survival.saturation + def.foodValue);
                state->inventory.consumeSelected();
                [self playSfx:state->sfxEat gain:0.6f];
            }
        } else {
            state->eatingState.reset();
            if (!interactableTarget && input.isPressedForTick(Key::MouseRight)) {
                // Shears cut wool from a sheep; a held bucket fills or empties a
                // fluid; otherwise place the held block.
                if ([self tryShearSheep:state cameraPos:cameraPos forward:forward]) {
                    // handled
                } else if (![self tryBucketInteraction:state
                                             cameraPos:cameraPos
                                               forward:forward
                                                   hit:rayHit]) {
                    [self placeBlock:state hit:rayHit];
                }
            }
        }
    }

    // Q drops one item from the selected hotbar stack, thrown ahead of the
    // eye. Creative drops without decrementing.
    {
        InputBindings bindings;
        if (input.isPressedForTick(bindings.drop.key)) {
            const ItemStack selected = state->inventory.getSelectedStack();
            if (!selected.empty()) {
                const Vec3 eye = state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f};
                const Vec3 dir = state->camera.forward();
                const Vec3 velocity{dir.x * 0.3f, dir.y * 0.3f + 0.15f, dir.z * 0.3f};
                state->itemEntities.spawn(ItemStack{selected.type, 1, selected.durability},
                                          eye + dir * 0.4f, velocity, 40);
                if (modeConsumesItems(state->gameMode)) {
                    state->inventory.consumeSelected();
                }
            }
        }
    }

    // Dropped items: physics, merge, despawn, then a pickup pass into the
    // inventory. Beyond the manager's active radius items freeze.
    [self tickItemEntities:state];

    // Death: any survival damage that emptied the health bar ends the tick on
    // the death screen with the inventory scattered.
    if (modeTakesDamage(state->gameMode) && state->player.health <= 0) {
        [self killPlayer:state];
    }

    [self updateDynamicObjectLighting:state];

    // Publish the immutable loaded-cube registry once per simulation tick.
    // Rendering reads this pointer without copying or taking the world lock.
    state->world->publishLoadedSnapshot();

    // Tick-edge input consumed, a second tick in this frame must not re-fire
    input.clearTickPresses();
}

// ---- Block Breaking (Task 6.1) ----

- (void)breakBlock:(EngineState*)state hit:(BlockRayHit)hit withDrops:(BOOL)withDrops {
    if (!hit.has_value())
        return;

    int64_t hitX = static_cast<int64_t>(std::floor(hit->first.x));
    int32_t hitY = static_cast<int32_t>(std::floor(hit->first.y));
    int64_t hitZ = static_cast<int64_t>(std::floor(hit->first.z));

    // Cannot break bedrock
    const std::optional<BlockType> current = state->world->findBlockIfLoaded(hitX, hitY, hitZ);
    if (!current || *current == BlockType::BEDROCK)
        return;

    const auto releaseContainerStack = [&](const ItemStack& stack) {
        if (stack.empty())
            return;
        if (withDrops) {
            [self spawnDrop:state stack:stack atX:hitX y:hitY z:hitZ];
            return;
        }
        const int absorbed = state->inventory.add(stack);
        if (absorbed < stack.count) {
            ItemStack remainder = stack;
            remainder.count = static_cast<uint8_t>(stack.count - absorbed);
            [self spawnDrop:state stack:remainder atX:hitX y:hitY z:hitZ];
        }
    };

    // A broken furnace loses its state entry; its contents scatter (survival)
    // or return to the player (creative-instant, which never drops).
    if (*current == BlockType::FURNACE || *current == BlockType::FURNACE_LIT) {
        const BlockPos furnacePosition{hitX, hitY, hitZ};
        state->furnaceVisualAuthority->erase(furnacePosition);
        auto it = state->furnaces.find(furnacePosition);
        if (it != state->furnaces.end()) {
            for (const ItemStack& stack : {it->second.input, it->second.fuel, it->second.output}) {
                releaseContainerStack(stack);
            }
            state->furnaces.erase(it);
        }
    }

    // A broken chest empties its whole 27-slot store the same way.
    if (*current == BlockType::CHEST) {
        auto it = state->chests.find(BlockPos{hitX, hitY, hitZ});
        if (it != state->chests.end()) {
            for (const ItemStack& stack : it->second.slots) {
                releaseContainerStack(stack);
            }
            state->chests.erase(it);
        }
    }

    const BlockType broken = *current;
    const ItemStack heldStack = state->inventory.getSelectedStack();

    if (broken == BlockType::BED && state->bedSpawnSet &&
        bedSpawnAnchoredToBlock(state->worldSpawn, hitX, hitY, hitZ)) {
        state->bedSpawnSet = false;
        state->pendingBedSpawnValidation = false;
        state->worldSpawn = state->v4SafeSpawnPos.value_or(state->requestedSpawn);
    }

    // Set block to air
    state->world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Plants and floor torches standing on the broken block lose their
    // support and pop with it (same column, so the dirty/save below covers it).
    const std::optional<BlockType> decoration =
        state->world->findBlockIfLoaded(hitX, hitY + 1, hitZ);
    if (decoration && losesSupportWhenBlockBelowBreaks(*decoration)) {
        const BlockType decorationBlock = *decoration;
        state->world->setBlock(hitX, hitY + 1, hitZ, BlockType::AIR);
        if (withDrops) {
            const BlockDrop decorationDrop = blockDrop(decorationBlock);
            if (decorationDrop.count > 0) {
                [self spawnDrop:state
                          stack:ItemStack{decorationDrop.item, decorationDrop.count, 0}
                            atX:hitX
                              y:hitY + 1
                              z:hitZ];
            }
        }
    }

    // Survival loot: the block's drop, gated by the tool tier, plus one point
    // of tool wear.
    if (withDrops) {
        const BlockDrop drop = blockDrop(broken);
        if (drop.count > 0 && toolCanHarvest(broken, heldStack.type)) {
            [self spawnDrop:state stack:ItemStack{drop.item, drop.count, 0} atX:hitX y:hitY z:hitZ];
        }
        if (isTool(heldStack.type)) {
            state->inventory.damageSelectedTool();
        }
    }

    // World::setBlock marks the chunk (and boundary neighbors) dirty and
    // flags it for save-on-unload

    // Clear highlight since block is now air
    state->hasHighlightedBlock = false;

    [self playSfx:state->sfxBlockBreak gain:0.8f];
}

// Spawn a dropped item at the center of a broken block with a small random
// pop so drops do not stack into a single point.
- (void)spawnDrop:(EngineState*)state
            stack:(ItemStack)stack
              atX:(int64_t)x
                y:(int32_t)y
                z:(int64_t)z {
    const Vec3 center{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.25f,
                      static_cast<float>(z) + 0.5f};
    const uint64_t h =
        hash64(static_cast<uint64_t>(x) * 6364136223846793005ull +
               static_cast<uint64_t>(z) * 1442695040888963407ull + static_cast<uint64_t>(y));
    const float vx = (static_cast<float>(h & 0xFF) / 255.f - 0.5f) * 0.1f;
    const float vz = (static_cast<float>((h >> 8) & 0xFF) / 255.f - 0.5f) * 0.1f;
    state->itemEntities.spawn(stack, center, Vec3{vx, 0.15f, vz}, 10);
}

// ---- Block Placing (Task 6.2) ----

// Right-clicking a boat mounts it; right-clicking water with a boat item
// places one on the surface. Returns YES when it handled the click.
- (BOOL)tryBoatInteraction:(EngineState*)state cameraPos:(Vec3)cameraPos forward:(Vec3)forward {
    if (!state->world)
        return NO;

    // Mounting an aimed boat takes priority and works with any held item.
    const int boatIndex = state->boats.pick(cameraPos, forward, 4.0f);
    if (boatIndex >= 0) {
        state->ridingBoat = boatIndex;
        [self playSfx:state->sfxBlockPlace gain:0.3f];
        return YES;
    }

    if (state->inventory.getSelectedStack().type != ItemType::BOAT)
        return NO;

    // March the aim ray for the first water surface within reach; fluids are
    // not solid, so the standard block ray passes straight through them.
    constexpr float REACH = 5.0f;
    constexpr float STEP = 0.1f;
    for (float t = 0.f; t <= REACH; t += STEP) {
        const int64_t bx = static_cast<int64_t>(std::floor(cameraPos.x + forward.x * t));
        const int32_t by = static_cast<int32_t>(std::floor(cameraPos.y + forward.y * t));
        const int64_t bz = static_cast<int64_t>(std::floor(cameraPos.z + forward.z * t));
        const std::optional<BlockType> block = state->world->findBlockIfLoaded(bx, by, bz);
        if (!block)
            return NO;
        if (*block == BlockType::WATER) {
            const FluidCell water = state->world->readFluidCell({bx, by, bz});
            const std::optional<BlockType> above = state->world->findBlockIfLoaded(bx, by + 1, bz);
            const std::optional<float> surface =
                boatPlacementSurfaceY(bx, by, bz, water, above, cameraPos, forward, REACH);
            if (!surface)
                return NO;
            const BoatSpawnResult spawned = state->boats.spawn(
                Vec3{static_cast<float>(bx) + 0.5f, *surface, static_cast<float>(bz) + 0.5f},
                std::atan2(forward.x, forward.z));
            if (spawned.evictedOldest && state->ridingBoat >= 0) {
                --state->ridingBoat;
            }
            if (modeConsumesItems(state->gameMode))
                state->inventory.consumeSelected();
            [self playSfx:state->sfxBlockPlace gain:0.5f];
            return YES;
        }
        if (isSolid(*block))
            return NO; // an opaque block blocks the aim before any water
    }
    return NO;
}

// Shearing a sheep with shears drops wool without killing it, the Minecraft
// way to gather the wool a bed needs. Returns YES when it handled the click.
- (BOOL)tryShearSheep:(EngineState*)state cameraPos:(Vec3)cameraPos forward:(Vec3)forward {
    if (state->inventory.getSelectedStack().type != ItemType::SHEARS || !state->spawner)
        return NO;
    const auto hit = pickEntity(cameraPos, forward, 3.0f, state->spawner->getEntities());
    if (!hit)
        return NO;
    for (auto& entity : state->spawner->getEntities()) {
        if (!entity || entity->id != hit->entityId || !entity->alive ||
            entity->type != EntityType::SHEEP) {
            continue;
        }
        [self spawnDrop:state
                  stack:ItemStack{itemFromBlock(BlockType::WOOL), 2, 0}
                    atX:static_cast<int64_t>(std::floor(entity->position.x))
                      y:static_cast<int32_t>(std::floor(entity->position.y))
                      z:static_cast<int64_t>(std::floor(entity->position.z))];
        [self playSfx:state->sfxBlockBreak gain:0.4f];
        return YES;
    }
    return NO;
}

// Right-clicking a bed sets the spawn anchor and, at night, sleeps through to
// dawn, exactly like Minecraft.
- (void)sleepInBed:(EngineState*)state atX:(int64_t)bx y:(int32_t)by z:(int64_t)bz {
    const int32_t feetY = by + 1;
    const int32_t headY = by + 2;
    if (feetY < WORLD_MIN_Y || headY > WORLD_MAX_Y)
        return;
    const auto breathable = [state, bx, bz](int32_t y) {
        const std::optional<BlockType> block = state->world->findBlockIfLoaded(bx, y, bz);
        return block && !isSolid(*block) && *block != BlockType::WATER && *block != BlockType::LAVA;
    };
    if (!breathable(feetY) || !breathable(headY))
        return;
    state->worldSpawn = Vec3{static_cast<float>(bx) + 0.5f, static_cast<float>(feetY),
                             static_cast<float>(bz) + 0.5f};
    state->bedSpawnSet = true;
    // Sleepable night runs from dusk to just before dawn; sleeping jumps the
    // clock to the next dawn (the tick-0 rollover the sun orbit treats as day).
    const uint64_t tod = state->worldTime % CELESTIAL_TICKS_PER_DAY;
    constexpr uint64_t NIGHT_START = 12500;
    constexpr uint64_t NIGHT_END = 23500;
    if (tod >= NIGHT_START && tod < NIGHT_END) {
        state->worldTime =
            ((state->worldTime / CELESTIAL_TICKS_PER_DAY) + 1) * CELESTIAL_TICKS_PER_DAY;
    }
    [self playSfx:state->sfxBlockPlace gain:0.3f];
}

// Empty and filled buckets exchange a single fluid source with the world,
// exactly like Minecraft: an empty bucket scoops the water source or lava it
// is aimed at, and a filled bucket empties its fluid against the targeted
// block face. Returns YES when it handled the click.
- (BOOL)tryBucketInteraction:(EngineState*)state
                   cameraPos:(Vec3)cameraPos
                     forward:(Vec3)forward
                         hit:(BlockRayHit)hit {
    const ItemStack selected = state->inventory.getSelectedStack();
    const ItemType held = selected.type;
    if (held != ItemType::BUCKET && held != ItemType::WATER_BUCKET &&
        held != ItemType::LAVA_BUCKET) {
        return NO;
    }

    const bool creative = !modeConsumesItems(state->gameMode);
    auto swapSelected = [&](ItemType to) {
        if (creative)
            return;
        const ItemStack overflow = state->inventory.exchangeOneSelected(ItemStack{to, 1, 0});
        if (!overflow.empty()) {
            const Vec3 drop = state->player.position;
            [self spawnDrop:state
                      stack:overflow
                        atX:static_cast<int64_t>(std::floor(drop.x))
                          y:static_cast<int32_t>(std::floor(drop.y))
                          z:static_cast<int64_t>(std::floor(drop.z))];
        }
    };

    if (held == ItemType::BUCKET) {
        // March the aim ray for the first fluid source within reach; fluids are
        // not solid, so the standard block ray passes straight through them.
        constexpr float REACH = 5.0f;
        constexpr float STEP = 0.1f;
        for (float t = 0.f; t <= REACH; t += STEP) {
            const int64_t bx = static_cast<int64_t>(std::floor(cameraPos.x + forward.x * t));
            const int32_t by = static_cast<int32_t>(std::floor(cameraPos.y + forward.y * t));
            const int64_t bz = static_cast<int64_t>(std::floor(cameraPos.z + forward.z * t));
            const std::optional<BlockType> block = state->world->findBlockIfLoaded(bx, by, bz);
            if (!block)
                return NO;
            if (*block == BlockType::WATER) {
                const FluidCell cell = state->world->readFluidCell(FluidPos{bx, by, bz});
                if (!cell.loaded || !cell.state.isSource())
                    continue; // only a full source fills a bucket
                if (!state->world->trySetBlock(bx, by, bz, BlockType::AIR))
                    return NO;
                swapSelected(ItemType::WATER_BUCKET);
                [self playSfx:state->sfxBlockPlace gain:0.5f];
                return YES;
            }
            if (*block == BlockType::LAVA) {
                const FluidCell cell = state->world->readFluidCell(FluidPos{bx, by, bz});
                if (!cell.loaded || !cell.state.isSource())
                    continue;
                if (!state->world->trySetBlock(bx, by, bz, BlockType::AIR))
                    return NO;
                swapSelected(ItemType::LAVA_BUCKET);
                [self playSfx:state->sfxBlockPlace gain:0.5f];
                return YES;
            }
            if (isSolid(*block))
                return NO; // an opaque block blocks the aim before any fluid
        }
        return NO;
    }

    // A filled bucket pours its fluid against the targeted block face.
    if (!hit.has_value())
        return NO;
    const BlockType fluid = held == ItemType::WATER_BUCKET ? BlockType::WATER : BlockType::LAVA;
    const int64_t px =
        static_cast<int64_t>(std::floor(hit->first.x)) + static_cast<int64_t>(hit->second.x);
    const int32_t py =
        static_cast<int32_t>(std::floor(hit->first.y)) + static_cast<int32_t>(hit->second.y);
    const int64_t pz =
        static_cast<int64_t>(std::floor(hit->first.z)) + static_cast<int64_t>(hit->second.z);
    if (py < WORLD_MIN_Y || py > WORLD_MAX_Y)
        return NO;
    const ChunkPos placeChunk{Chunk::worldToChunk(px), Chunk::worldToChunkY(py),
                              Chunk::worldToChunk(pz)};
    if (!state->world->isChunkLoaded(placeChunk))
        return NO;
    const std::optional<BlockType> target = state->world->findBlockIfLoaded(px, py, pz);
    if (!target || (*target != BlockType::AIR && *target != BlockType::WATER))
        return NO;
    if (!state->world->trySetBlock(px, py, pz, fluid))
        return NO;
    swapSelected(ItemType::BUCKET);
    [self playSfx:state->sfxBlockPlace gain:0.5f];
    return YES;
}

- (void)placeBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value())
        return;

    // An empty slot or a non-block item has nothing to place.
    const BlockType selectedType = state->inventory.getSelectedBlockType();
    if (selectedType == BlockType::AIR)
        return;

    // Calculate placement position: hit block + face normal
    int64_t placeX =
        static_cast<int64_t>(std::floor(hit->first.x)) + static_cast<int64_t>(hit->second.x);
    int32_t placeY =
        static_cast<int32_t>(std::floor(hit->first.y)) + static_cast<int32_t>(hit->second.y);
    int64_t placeZ =
        static_cast<int64_t>(std::floor(hit->first.z)) + static_cast<int64_t>(hit->second.z);

    const ChunkPos placeChunk{Chunk::worldToChunk(placeX), Chunk::worldToChunkY(placeY),
                              Chunk::worldToChunk(placeZ)};
    if (!state->world->isChunkLoaded(placeChunk))
        return;

    const std::optional<BlockType> destination =
        state->world->findBlockIfLoaded(placeX, placeY, placeZ);
    if (!destination || (*destination != BlockType::AIR && !isFlora(*destination)))
        return;

    // Floor torches are authored decorations, not generic replaceable flora.
    // Their supporting cell must be resident and expose a full solid top.
    if (isFloorTorch(selectedType)) {
        if (placeY <= WORLD_MIN_Y)
            return;
        const std::optional<BlockType> support =
            state->world->findBlockIfLoaded(placeX, placeY - 1, placeZ);
        if (!support || !hasFullBlockCollision(*support))
            return;
    }

    // Validate the authored collision volume against the player. A bed stops
    // at 9/16 height; nonsolid crosses have no placement collision volume.
    const float collisionHeight = blockCollisionHeight(selectedType);
    if (collisionHeight > 0.0F) {
        const AABB placeBox{Vec3{static_cast<float>(placeX), static_cast<float>(placeY),
                                 static_cast<float>(placeZ)},
                            Vec3{static_cast<float>(placeX + 1),
                                 static_cast<float>(placeY) + collisionHeight,
                                 static_cast<float>(placeZ + 1)}};
        if (placeBox.intersects(state->player.getAABB()))
            return;
    }

    // Place block (World::setBlock marks the chunk and boundary neighbors
    // dirty and flags the chunk for save-on-unload).
    if (!state->world->trySetBlock(placeX, placeY, placeZ, selectedType))
        return;

    // A placed furnace or chest gets an empty state entry so it persists.
    if (selectedType == BlockType::FURNACE) {
        const BlockPos position{placeX, placeY, placeZ};
        state->furnaces.insert_or_assign(position, FurnaceState{});
        state->furnaceVisualAuthority->set(position, BlockType::FURNACE);
    } else if (selectedType == BlockType::CHEST) {
        state->chests.insert_or_assign(BlockPos{placeX, placeY, placeZ}, ChestState{});
    }

    // Survival consumes the placed block; creative keeps its infinite stack.
    if (modeConsumesItems(state->gameMode)) {
        state->inventory.consumeSelected();
    }

    [self playSfx:state->sfxBlockPlace gain:0.8f];
}

// ---- Block Highlight Update (Task 6.9) ----
// Merged into gameTick: via single raycast (Major #4 fix)

// ---- Render ----

- (void)render {
    EngineState* state = _state.get();

    // Capture cameras are independent from the validated player spawn. This
    // allows an isolated playtest to inspect water, handoffs, or tall terrain
    // without authorizing an unsafe player position or changing saved state.
    static const bool captureActive = [] {
        const char* value = std::getenv("RYCRAFT_CAPTURE");
        return value && *value;
    }();
    static const std::optional<Vec3> captureCameraPosition =
        captureActive ? vectorEnvironmentValue("RYCRAFT_CAPTURE_CAMERA") : std::nullopt;
    static const std::optional<float> captureYaw =
        captureActive ? finiteEnvironmentValue("RYCRAFT_YAW") : std::nullopt;
    static const std::optional<float> capturePitch =
        captureActive ? finiteEnvironmentValue("RYCRAFT_PITCH") : std::nullopt;
    if (captureActive && (captureYaw || capturePitch)) {
        constexpr float DEGREES_TO_RADIANS = static_cast<float>(M_PI) / 180.0F;
        state->camera.setLook(captureYaw.value_or(0.0F) * DEGREES_TO_RADIANS,
                              capturePitch.value_or(0.0F) * DEGREES_TO_RADIANS);
    }
    if (captureCameraPosition && state->world && (state->diagnosticV3 || state->v4EntryReady))
        state->camera.setPosition(*captureCameraPosition);

    // Ease the FOV toward the movement mode's target (dt-correct exponential,
    // so the zoom speed doesn't depend on frame rate) and hand it to the
    // camera, the cloud shader reads camera.FOV(), so routing the projection
    // through the same value keeps clouds registered with the world mid-zoom.
    float targetFov = state->player.sprinting ? EngineState::SPRINT_FOV : EngineState::BASE_FOV;
    state->fovCurrent +=
        (targetFov - state->fovCurrent) *
        (1.0f - std::exp(-static_cast<float>(state->deltaTime) / EngineState::FOV_EASE_SECONDS));
    state->camera.setFOV(state->fovCurrent);

    // Update projection matrix from current drawable size
    CGSize currentSize = _view.drawableSize;
    if (currentSize.width > 0 && currentSize.height > 0) {
        state->drawableSize = currentSize;
        float aspect =
            static_cast<float>(currentSize.width) / static_cast<float>(currentSize.height);
        const float farPlane =
            std::max(1000.0f, static_cast<float>(state->settings.viewDistance * CHUNK_EDGE +
                                                 FAR_TERRAIN_TILE_EDGE * 2));
        state->projectionMatrix = Mat4::perspective(
            state->camera.FOV() * (static_cast<float>(M_PI) / 180.0f), aspect, 0.1f, farPlane);
    }

    id<CAMetalDrawable> drawable = _view.currentDrawable;
    if (!drawable)
        return;

    if (!_renderPipeline)
        return;

    // A capture frame budget measures frames of the requested scene, not
    // bootstrap or entry preparation frames. The clock is advanced directly
    // before each eligible render so asynchronous v4 startup cannot skip the
    // equality point. Explicit menu start screens remain eligible too.
    const auto advanceFrameCapture = [self, state] {
        static const char* capturePath = std::getenv("RYCRAFT_CAPTURE");
        if (!capturePath || !*capturePath)
            return;
        static const uint64_t captureFrame = unsignedEnvironmentValue("RYCRAFT_CAPTURE_FRAME", 240);
        const FrameCaptureActions actions = state->frameCaptureClock.onRenderedFrame(captureFrame);
        if (actions.capture) {
            const Vec3 cameraPosition = state->camera.getPosition();
            const auto stats = self->_renderPipeline->chunkRenderStats();
            const int configuredHorizonChunks = stats.farBaseViewDistanceChunks;
            const int entryHorizonChunks =
                v4RequiredEntryParentRadiusChunks(configuredHorizonChunks);
            const float connectedParentRadiusChunks =
                v4ConnectedParentRadiusChunks(stats, configuredHorizonChunks);
            const StreamingWorkStats streaming =
                state->world ? state->world->getStreamingWorkStats() : StreamingWorkStats{};
            const char* fingerprint = state->generationFingerprint.empty()
                                          ? "none"
                                          : state->generationFingerprint.c_str();
            const int fingerprintShortLength = static_cast<int>(std::min<size_t>(
                12, state->generationFingerprint.empty() ? std::string_view("none").size()
                                                         : state->generationFingerprint.size()));
            const char* profile =
                state->saveManager ? state->saveManager->getWorldPath().c_str() : "none";
            const auto environmentFlagEnabled = [](const char* name) {
                const char* value = std::getenv(name);
                return value && *value && std::strcmp(value, "0") != 0;
            };
            char evidence[4096];
            std::snprintf(
                evidence, sizeof(evidence),
                "Capture evidence: output \"%s\" profile \"%s\" scene frame %llu engine frame "
                "%llu seed %llu fingerprint %s short %.*s | player %.3f,%.3f,%.3f camera "
                "%.3f,%.3f,%.3f yaw %.3f pitch %.3f fov %.2f drawable %.0fx%.0f msaa %lu view "
                "%d time %llu entry %.3f seconds | validation api %u shader %u | graphics shadow "
                "%d cloud %d indirect %d bloom %d volumetric %u reflections %u waving %u flare %u",
                capturePath, profile,
                static_cast<unsigned long long>(
                    state->frameCaptureClock.capturedAt.value_or(captureFrame)),
                static_cast<unsigned long long>(state->frameCount),
                static_cast<unsigned long long>(state->generationSeed), fingerprint,
                fingerprintShortLength, fingerprint, state->player.position.x,
                state->player.position.y, state->player.position.z, cameraPosition.x,
                cameraPosition.y, cameraPosition.z,
                static_cast<double>(state->camera.yaw()) * 180.0 / M_PI,
                static_cast<double>(state->camera.pitch()) * 180.0 / M_PI, state->camera.FOV(),
                state->drawableSize.width, state->drawableSize.height,
                static_cast<unsigned long>(PixelFormats::SCENE_SAMPLE_COUNT),
                state->settings.viewDistance, static_cast<unsigned long long>(state->worldTime),
                state->v4EntryStartedAt > 0.0
                    ? std::max(0.0, CACurrentMediaTime() - state->v4EntryStartedAt)
                    : 0.0,
                static_cast<unsigned>(environmentFlagEnabled("MTL_DEBUG_LAYER")),
                static_cast<unsigned>(environmentFlagEnabled("MTL_SHADER_VALIDATION")),
                state->gfx.shadowQuality, state->gfx.cloudQuality,
                state->gfx.indirectLightingQuality, state->gfx.bloomLevel,
                static_cast<unsigned>(state->gfx.volumetricLight),
                static_cast<unsigned>(state->gfx.waterReflections),
                static_cast<unsigned>(state->gfx.wavingFoliage),
                static_cast<unsigned>(state->gfx.lensFlare));
            RY_LOG_INFO(evidence);
            char streamingEvidence[2048];
            std::snprintf(
                streamingEvidence, sizeof(streamingEvidence),
                "Capture streaming: exact %u/%u ready/required unresolved %u handoff %.0f | "
                "protected-near %u/%u required/ready missing %u boundary-mismatch %u | "
                "base %u/%u/%u "
                "wanted/resident/drawn missing %u | refine %u/%u/%u wanted/resident/drawn | "
                "frontier %.0f blocks entry %.1f/%d chunks configured %d | "
                "final-handoff missing %u queues %u/%u base/refine pending %u | canopy %u "
                "in-flight %u active %u queued %u parked %u completed cache %u/%.1f MB failed "
                "%llu deferred %llu resumes %llu | light queue %zu deferred %llu",
                stats.exactSurfaceReadyCount, stats.exactSurfaceRequiredCount,
                stats.exactSurfaceUnresolvedColumnCount, stats.exactSurfaceHandoffBlocks,
                stats.farProtectedNearWantedTileCount, stats.farProtectedNearResidentTileCount,
                stats.farProtectedNearMissingTileCount, stats.farProtectedNearBoundaryMismatchCount,
                stats.farBaseWantedTileCount, stats.farBaseResidentTileCount,
                stats.farBaseDrawnTileCount, stats.farBaseMissingTileCount,
                stats.farRefinementWantedTileCount, stats.farRefinementResidentTileCount,
                stats.farRefinementDrawnTileCount, stats.farCoverageFrontierBlocks,
                connectedParentRadiusChunks, entryHorizonChunks, configuredHorizonChunks,
                stats.farExactHandoffMissingFinalParentCount, stats.farQueuedBaseTileCount,
                stats.farQueuedRefinementTileCount, stats.farPendingTileCount,
                stats.farCanopyInFlightCount, stats.farActiveCanopyWorkerCount,
                stats.farQueuedCanopyCount, stats.farParkedCanopyCount,
                stats.farCompletedCanopyCount, stats.farCanopyCacheEntryCount,
                stats.farCanopyCacheMB, static_cast<unsigned long long>(stats.farCanopyFailedCount),
                static_cast<unsigned long long>(stats.farCanopyDeferredCount),
                static_cast<unsigned long long>(stats.farCanopyAuthorityCompletionResumeCount),
                streaming.publicationLightDeferredQueue,
                static_cast<unsigned long long>(streaming.publicationLightDeferredCubes));
            RY_LOG_INFO(streamingEvidence);
            char tierEvidence[768];
            std::snprintf(tierEvidence, sizeof(tierEvidence),
                          "Capture tiers [step1,step2,step4,step8,step16,step32]: desired "
                          "[%u,%u,%u,%u,%u,%u] resident [%u,%u,%u,%u,%u,%u] displayed "
                          "[%u,%u,%u,%u,%u,%u] drawn [%u,%u,%u,%u,%u,%u]",
                          stats.farTierDesiredTileCounts[0], stats.farTierDesiredTileCounts[1],
                          stats.farTierDesiredTileCounts[2], stats.farTierDesiredTileCounts[3],
                          stats.farTierDesiredTileCounts[4], stats.farTierDesiredTileCounts[5],
                          stats.farTierResidentMeshCounts[0], stats.farTierResidentMeshCounts[1],
                          stats.farTierResidentMeshCounts[2], stats.farTierResidentMeshCounts[3],
                          stats.farTierResidentMeshCounts[4], stats.farTierResidentMeshCounts[5],
                          stats.farTierDisplayedTileCounts[0], stats.farTierDisplayedTileCounts[1],
                          stats.farTierDisplayedTileCounts[2], stats.farTierDisplayedTileCounts[3],
                          stats.farTierDisplayedTileCounts[4], stats.farTierDisplayedTileCounts[5],
                          stats.farTierDrawnTileCounts[0], stats.farTierDrawnTileCounts[1],
                          stats.farTierDrawnTileCounts[2], stats.farTierDrawnTileCounts[3],
                          stats.farTierDrawnTileCounts[4], stats.farTierDrawnTileCounts[5]);
            RY_LOG_INFO(tierEvidence);
            char authorityTierEvidence[1024];
            std::snprintf(
                authorityTierEvidence, sizeof(authorityTierEvidence),
                "Capture authority tiers [step1,step2,step4,step8,step16,step32]: resident "
                "preview [%u,%u,%u,%u,%u,%u] final [%u,%u,%u,%u,%u,%u] | displayed "
                "preview [%u,%u,%u,%u,%u,%u] final [%u,%u,%u,%u,%u,%u] | pending "
                "transitions %u",
                stats.farTierResidentPreviewCounts[0], stats.farTierResidentPreviewCounts[1],
                stats.farTierResidentPreviewCounts[2], stats.farTierResidentPreviewCounts[3],
                stats.farTierResidentPreviewCounts[4], stats.farTierResidentPreviewCounts[5],
                stats.farTierResidentFinalCounts[0], stats.farTierResidentFinalCounts[1],
                stats.farTierResidentFinalCounts[2], stats.farTierResidentFinalCounts[3],
                stats.farTierResidentFinalCounts[4], stats.farTierResidentFinalCounts[5],
                stats.farTierDisplayedPreviewCounts[0], stats.farTierDisplayedPreviewCounts[1],
                stats.farTierDisplayedPreviewCounts[2], stats.farTierDisplayedPreviewCounts[3],
                stats.farTierDisplayedPreviewCounts[4], stats.farTierDisplayedPreviewCounts[5],
                stats.farTierDisplayedFinalCounts[0], stats.farTierDisplayedFinalCounts[1],
                stats.farTierDisplayedFinalCounts[2], stats.farTierDisplayedFinalCounts[3],
                stats.farTierDisplayedFinalCounts[4], stats.farTierDisplayedFinalCounts[5],
                stats.farPendingAuthorityTransitionCount);
            RY_LOG_INFO(authorityTierEvidence);
            char perceptualLodEvidence[768];
            std::snprintf(
                perceptualLodEvidence, sizeof(perceptualLodEvidence),
                "Capture perceptual LOD: worst %.3f px tile (%lld,%lld) desired step%u FINAL "
                "displayed step%u %s resident masks preview 0x%08x final 0x%08x | violations %u "
                "visible FINAL requests %u",
                stats.farWorstVisibleProjectedErrorPixels,
                static_cast<long long>(stats.farWorstVisibleTileX),
                static_cast<long long>(stats.farWorstVisibleTileZ),
                static_cast<unsigned>(stats.farWorstVisibleDesiredStep),
                static_cast<unsigned>(stats.farWorstVisibleDisplayedStep),
                stats.farWorstVisibleDisplayedQuality ==
                        static_cast<uint8_t>(FarTerrainAuthorityQuality::FINAL)
                    ? "FINAL"
                    : "PREVIEW",
                stats.farWorstVisiblePreviewResidentMask, stats.farWorstVisibleFinalResidentMask,
                stats.farVisibleProjectedErrorViolationCount,
                stats.farVisiblePerceptualFinalRequestCount);
            RY_LOG_INFO(perceptualLodEvidence);
            if (const auto* runtime =
                    dynamic_cast<const worldgen::runtime::ProductionTerrainRuntime*>(
                        state->terrainRuntime.get())) {
                const worldgen::runtime::TerrainRuntimeMetrics runtimeMetrics = runtime->metrics();
                const std::string qualificationHash =
                    runtimeMetrics.qualificationDigest
                        ? worldgen::learned::sha256Hex(*runtimeMetrics.qualificationDigest)
                        : "none";
                worldgen::learned::WorldGenerationMetrics generationMetrics;
                const auto generationContext =
                    state->world ? state->world->generationContext() : nullptr;
                if (generationContext)
                    generationMetrics = generationContext->metrics();
                const char* contextQuality =
                    !generationContext ? "none"
                    : generationContext->quality() == worldgen::learned::AuthorityQuality::FINAL
                        ? "final"
                        : "preview";
                const std::string modelHash =
                    generationContext
                        ? worldgen::learned::sha256Hex(generationContext->identity().modelPackHash)
                        : "none";
                const std::string runtimeHash =
                    generationContext
                        ? worldgen::learned::sha256Hex(generationContext->identity().runtimeHash)
                        : "none";
                char authorityEvidence[1536];
                std::snprintf(
                    authorityEvidence, sizeof(authorityEvidence),
                    "Capture authority: context %s qualification %s model %s runtime %s "
                    "provider coreml %llu/%llu "
                    "cpu-fallback %llu/%llu inference active %u queued %u calls %llu failures "
                    "%llu max-concurrent %u cpu-threads %u | authority queries %llu ready %llu "
                    "deferred %llu failed %llu cache %zu/%.1f MB builds %zu/%zu active/queued "
                    "publication %zu/%zu active/queued transient disk/write/repair %llu/%llu/%llu",
                    contextQuality, qualificationHash.c_str(), modelHash.c_str(),
                    runtimeHash.c_str(),
                    static_cast<unsigned long long>(runtimeMetrics.coreMlPartitions),
                    static_cast<unsigned long long>(runtimeMetrics.coreMlNodes),
                    static_cast<unsigned long long>(runtimeMetrics.cpuFallbackPartitions),
                    static_cast<unsigned long long>(runtimeMetrics.cpuFallbackNodes),
                    runtimeMetrics.activeInferenceCalls, runtimeMetrics.queuedInferenceCalls,
                    static_cast<unsigned long long>(runtimeMetrics.inferenceCalls),
                    static_cast<unsigned long long>(runtimeMetrics.inferenceFailures),
                    runtimeMetrics.maximumConcurrentInferenceCalls,
                    runtimeMetrics.cpuFallbackIntraOpThreads,
                    static_cast<unsigned long long>(generationMetrics.queries),
                    static_cast<unsigned long long>(generationMetrics.readyQueries),
                    static_cast<unsigned long long>(generationMetrics.deferredQueries),
                    static_cast<unsigned long long>(generationMetrics.failedQueries),
                    generationMetrics.authorityCache.entries,
                    static_cast<double>(generationMetrics.authorityCache.bytes) / (1024.0 * 1024.0),
                    generationMetrics.authorityCache.activeBuilds,
                    generationMetrics.authorityCache.queuedBuilds,
                    generationMetrics.authorityCache.activePublications,
                    generationMetrics.authorityCache.queuedPublications,
                    static_cast<unsigned long long>(
                        generationMetrics.authorityCache.transientDiskLoads),
                    static_cast<unsigned long long>(
                        generationMetrics.authorityCache.transientPublicationWrites),
                    static_cast<unsigned long long>(
                        generationMetrics.authorityCache.transientRepairs));
                RY_LOG_INFO(authorityEvidence);
            }
            self->_renderPipeline->requestFrameCapture(capturePath);
        }
        if (actions.quit)
            [self requestQuit];
    };

    if (!state->world) {
        static double lastCaptureBootstrapEvidence = -1.0;
        const double bootstrapNow = CACurrentMediaTime();
        if (captureActive && state->v4OpenRequested &&
            (lastCaptureBootstrapEvidence < 0.0 ||
             bootstrapNow - lastCaptureBootstrapEvidence >= 1.0)) {
            lastCaptureBootstrapEvidence = bootstrapNow;
            worldgen::bootstrap::TerrainBootstrapSnapshot snapshot;
            if (state->terrainBootstrap)
                snapshot = state->terrainBootstrap->snapshot();
            worldgen::runtime::TerrainRuntimeMetrics runtimeMetrics;
            if (const auto* runtime =
                    dynamic_cast<const worldgen::runtime::ProductionTerrainRuntime*>(
                        state->terrainRuntime.get())) {
                runtimeMetrics = runtime->metrics();
            }
            worldgen::learned::WorldGenerationMetrics generationMetrics;
            if (state->terrainBootstrap) {
                if (const auto context = state->terrainBootstrap->qualifiedGenerationContext())
                    generationMetrics = context->metrics();
            }
            char bootstrapEvidence[1536];
            std::snprintf(
                bootstrapEvidence, sizeof(bootstrapEvidence),
                "Capture bootstrap: entry %.3f seconds state %u running %u installed-reuse %u "
                "bytes %llu/%llu asset \"%s\" | profile-open %u spawn ordinal %u candidate %u "
                "provisional %u safe %u authority %u | inference %u/%u active/queued calls %llu "
                "failures %llu | authority queries %llu ready %llu deferred %llu builds %zu/%zu "
                "active/queued",
                state->v4EntryStartedAt > 0.0
                    ? std::max(0.0, bootstrapNow - state->v4EntryStartedAt)
                    : 0.0,
                static_cast<unsigned>(snapshot.state),
                static_cast<unsigned>(
                    state->terrainBootstrapRunning.load(std::memory_order_acquire)),
                static_cast<unsigned>(snapshot.reusingInstalledPack),
                static_cast<unsigned long long>(snapshot.completedBytes),
                static_cast<unsigned long long>(snapshot.totalBytes), snapshot.currentAsset.c_str(),
                static_cast<unsigned>(state->v4ProfileOpened), state->v4SpawnSearchOrdinal,
                static_cast<unsigned>(state->v4SpawnCandidateActive),
                static_cast<unsigned>(state->v4SpawnCandidateProvisional),
                static_cast<unsigned>(state->v4SpawnSafetyValidated),
                static_cast<unsigned>(state->v4SpawnAuthorityStatus),
                runtimeMetrics.activeInferenceCalls, runtimeMetrics.queuedInferenceCalls,
                static_cast<unsigned long long>(runtimeMetrics.inferenceCalls),
                static_cast<unsigned long long>(runtimeMetrics.inferenceFailures),
                static_cast<unsigned long long>(generationMetrics.queries),
                static_cast<unsigned long long>(generationMetrics.readyQueries),
                static_cast<unsigned long long>(generationMetrics.deferredQueries),
                generationMetrics.authorityCache.activeBuilds,
                generationMetrics.authorityCache.queuedBuilds);
            RY_LOG_INFO(bootstrapEvidence);
        }
        const bool explicitMenuStart =
            state->requestedStartScreen && !screenHasWorldSession(*state->requestedStartScreen);
        if (!state->v4OpenRequested || explicitMenuStart)
            advanceFrameCapture();
        UIFrameState startupFrame;
        startupFrame.screen = GameScreen::TITLE;
        startupFrame.hoveredButton = state->hoveredButton;
        startupFrame.menu = state->menuLayout;
        _renderPipeline->renderMenuOnly(_queue, drawable, startupFrame);
        return;
    }

    if (!state->diagnosticV3 && !state->v4EntryReady) {
        // World preparation must not be paced by the full exact-world render.
        // A growing cold horizon otherwise slows the UI frames that drain and
        // publish it, producing a feedback loop that looks like a stalled load.
        // A terminal generation failure deliberately freezes new work until
        // the user retries or repairs the qualified runtime. Continuing to
        // pump here could keep refilling scheduler queues behind the repair
        // UI and make a latched failure look like active progress.
        const bool canonicalDryCandidateAccepted = v4CanonicalDrySpawnAccepted(
            state->v4SpawnCandidateActive, state->v4SpawnCandidateProvisional,
            state->v4SpawnSafetyValidated);
        const bool finalSpawnAuthorityReady =
            state->v4SpawnAuthorityStatus == V4SpawnAuthorityPrequeueStatus::Ready;
        const bool mayPrepareV4World =
            state->startupFailure.empty() &&
            v4MayPrepareHorizon(canonicalDryCandidateAccepted, finalSpawnAuthorityReady);
        if (mayPrepareV4World) {
            state->camera.setPosition(state->player.position +
                                      Vec3{0.0F, Player::EYE_HEIGHT, 0.0F});
        }
        static double lastCapturePreparationEvidence = -1.0;
        const double preparationNow = CACurrentMediaTime();
        if (captureActive && (lastCapturePreparationEvidence < 0.0 ||
                              preparationNow - lastCapturePreparationEvidence >= 1.0)) {
            lastCapturePreparationEvidence = preparationNow;
            const auto stats = _renderPipeline->chunkRenderStats();
            const int configuredHorizonChunks = stats.farBaseViewDistanceChunks;
            const int entryHorizonChunks =
                v4RequiredEntryParentRadiusChunks(configuredHorizonChunks);
            const float connectedParentRadiusChunks =
                v4ConnectedParentRadiusChunks(stats, configuredHorizonChunks);
            worldgen::runtime::TerrainRuntimeMetrics runtimeMetrics;
            if (const auto* runtime =
                    dynamic_cast<const worldgen::runtime::ProductionTerrainRuntime*>(
                        state->terrainRuntime.get())) {
                runtimeMetrics = runtime->metrics();
            }
            worldgen::learned::WorldGenerationMetrics generationMetrics;
            if (const auto context = state->world->generationContext())
                generationMetrics = context->metrics();
            const auto& qualificationPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::Qualification)];
            const auto& drySpawnPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::DrySpawnCoarseSearch)];
            const auto& finalSpawnPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::FinalSpawnCertification)];
            const auto& explorationExactPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::ExplorationExact)];
            const auto& horizonPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::HorizonPreview)];
            const auto& protectedPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::ProtectedFinal)];
            const auto& visibleFinalPhase = runtimeMetrics.phases[static_cast<size_t>(
                worldgen::runtime::TerrainRuntimeInferencePhase::VisibleFinalRefinement)];
            char preparationEvidence[4096];
            std::snprintf(
                preparationEvidence, sizeof(preparationEvidence),
                "Capture preparation: entry %.3f seconds spawn candidate %u provisional %u "
                "safe %u authority %u | exact %u/%u ready/required unresolved %u | "
                "protected-near %u/%u required/ready missing %u boundary-mismatch %u | "
                "base %u/%u "
                "wanted/resident missing %u | entry %.1f/%d chunks configured %d selection view "
                "%d epochs %llu/%llu | tier resident "
                "[%u,%u,%u,%u,%u,%u] displayed [%u,%u,%u,%u,%u,%u] | far queues %u/%u "
                "base/refine pending %u workers %u+%u/%u active+reserved/budget | inference %u/%u "
                "active/queued calls %llu failures %llu models c/b/d %llu/%llu/%llu | phases "
                "qualification %llu[%llu/%llu/%llu] dry-coarse %llu[%llu/%llu/%llu] "
                "spawn-final %llu[%llu/%llu/%llu] exploration-exact %llu[%llu/%llu/%llu] "
                "horizon-preview %llu[%llu/%llu/%llu] protected-final "
                "%llu[%llu/%llu/%llu] visible-final %llu[%llu/%llu/%llu] | authority "
                "%zu/%zu active/queued "
                "transient disk/write/repair %llu/%llu/%llu | "
                "step32 water grid %llu/%llu calls/samples dense %llu point %llu",
                state->v4EntryStartedAt > 0.0
                    ? std::max(0.0, preparationNow - state->v4EntryStartedAt)
                    : 0.0,
                static_cast<unsigned>(state->v4SpawnCandidateActive),
                static_cast<unsigned>(state->v4SpawnCandidateProvisional),
                static_cast<unsigned>(state->v4SpawnSafetyValidated),
                static_cast<unsigned>(state->v4SpawnAuthorityStatus), stats.exactSurfaceReadyCount,
                stats.exactSurfaceRequiredCount, stats.exactSurfaceUnresolvedColumnCount,
                stats.farProtectedNearWantedTileCount, stats.farProtectedNearResidentTileCount,
                stats.farProtectedNearMissingTileCount, stats.farProtectedNearBoundaryMismatchCount,
                stats.farBaseWantedTileCount, stats.farBaseResidentTileCount,
                stats.farBaseMissingTileCount, connectedParentRadiusChunks, entryHorizonChunks,
                configuredHorizonChunks, stats.farBaseViewDistanceChunks,
                static_cast<unsigned long long>(stats.farBaseWorldEpoch),
                static_cast<unsigned long long>(stats.farBaseViewEpoch),
                stats.farTierResidentMeshCounts[0], stats.farTierResidentMeshCounts[1],
                stats.farTierResidentMeshCounts[2], stats.farTierResidentMeshCounts[3],
                stats.farTierResidentMeshCounts[4], stats.farTierResidentMeshCounts[5],
                stats.farTierDisplayedTileCounts[0], stats.farTierDisplayedTileCounts[1],
                stats.farTierDisplayedTileCounts[2], stats.farTierDisplayedTileCounts[3],
                stats.farTierDisplayedTileCounts[4], stats.farTierDisplayedTileCounts[5],
                stats.farQueuedBaseTileCount, stats.farQueuedRefinementTileCount,
                stats.farPendingTileCount, stats.farActiveBaseWorkerCount,
                stats.farReservedBaseWorkerCount, stats.farWorkerBudget,
                runtimeMetrics.activeInferenceCalls, runtimeMetrics.queuedInferenceCalls,
                static_cast<unsigned long long>(runtimeMetrics.inferenceCalls),
                static_cast<unsigned long long>(runtimeMetrics.inferenceFailures),
                static_cast<unsigned long long>(runtimeMetrics.models[0].calls),
                static_cast<unsigned long long>(runtimeMetrics.models[1].calls),
                static_cast<unsigned long long>(runtimeMetrics.models[2].calls),
                static_cast<unsigned long long>(qualificationPhase.calls),
                static_cast<unsigned long long>(qualificationPhase.modelCalls[0]),
                static_cast<unsigned long long>(qualificationPhase.modelCalls[1]),
                static_cast<unsigned long long>(qualificationPhase.modelCalls[2]),
                static_cast<unsigned long long>(drySpawnPhase.calls),
                static_cast<unsigned long long>(drySpawnPhase.modelCalls[0]),
                static_cast<unsigned long long>(drySpawnPhase.modelCalls[1]),
                static_cast<unsigned long long>(drySpawnPhase.modelCalls[2]),
                static_cast<unsigned long long>(finalSpawnPhase.calls),
                static_cast<unsigned long long>(finalSpawnPhase.modelCalls[0]),
                static_cast<unsigned long long>(finalSpawnPhase.modelCalls[1]),
                static_cast<unsigned long long>(finalSpawnPhase.modelCalls[2]),
                static_cast<unsigned long long>(explorationExactPhase.calls),
                static_cast<unsigned long long>(explorationExactPhase.modelCalls[0]),
                static_cast<unsigned long long>(explorationExactPhase.modelCalls[1]),
                static_cast<unsigned long long>(explorationExactPhase.modelCalls[2]),
                static_cast<unsigned long long>(horizonPhase.calls),
                static_cast<unsigned long long>(horizonPhase.modelCalls[0]),
                static_cast<unsigned long long>(horizonPhase.modelCalls[1]),
                static_cast<unsigned long long>(horizonPhase.modelCalls[2]),
                static_cast<unsigned long long>(protectedPhase.calls),
                static_cast<unsigned long long>(protectedPhase.modelCalls[0]),
                static_cast<unsigned long long>(protectedPhase.modelCalls[1]),
                static_cast<unsigned long long>(protectedPhase.modelCalls[2]),
                static_cast<unsigned long long>(visibleFinalPhase.calls),
                static_cast<unsigned long long>(visibleFinalPhase.modelCalls[0]),
                static_cast<unsigned long long>(visibleFinalPhase.modelCalls[1]),
                static_cast<unsigned long long>(visibleFinalPhase.modelCalls[2]),
                generationMetrics.authorityCache.activeBuilds,
                generationMetrics.authorityCache.queuedBuilds,
                static_cast<unsigned long long>(
                    generationMetrics.authorityCache.transientDiskLoads),
                static_cast<unsigned long long>(
                    generationMetrics.authorityCache.transientPublicationWrites),
                static_cast<unsigned long long>(generationMetrics.authorityCache.transientRepairs),
                static_cast<unsigned long long>(stats.farStep32WaterGridCalls),
                static_cast<unsigned long long>(stats.farStep32WaterGridSamples),
                static_cast<unsigned long long>(stats.farStep32WaterDenseGridCalls),
                static_cast<unsigned long long>(stats.farStep32WaterPointSamples));
            RY_LOG_INFO(preparationEvidence);
            char preparationAuthorityEvidence[1024];
            std::snprintf(
                preparationAuthorityEvidence, sizeof(preparationAuthorityEvidence),
                "Capture preparation authority tiers [step1,step2,step4,step8,step16,step32]: "
                "resident preview [%u,%u,%u,%u,%u,%u] final [%u,%u,%u,%u,%u,%u] | "
                "displayed preview [%u,%u,%u,%u,%u,%u] final [%u,%u,%u,%u,%u,%u] | "
                "pending transitions %u",
                stats.farTierResidentPreviewCounts[0], stats.farTierResidentPreviewCounts[1],
                stats.farTierResidentPreviewCounts[2], stats.farTierResidentPreviewCounts[3],
                stats.farTierResidentPreviewCounts[4], stats.farTierResidentPreviewCounts[5],
                stats.farTierResidentFinalCounts[0], stats.farTierResidentFinalCounts[1],
                stats.farTierResidentFinalCounts[2], stats.farTierResidentFinalCounts[3],
                stats.farTierResidentFinalCounts[4], stats.farTierResidentFinalCounts[5],
                stats.farTierDisplayedPreviewCounts[0], stats.farTierDisplayedPreviewCounts[1],
                stats.farTierDisplayedPreviewCounts[2], stats.farTierDisplayedPreviewCounts[3],
                stats.farTierDisplayedPreviewCounts[4], stats.farTierDisplayedPreviewCounts[5],
                stats.farTierDisplayedFinalCounts[0], stats.farTierDisplayedFinalCounts[1],
                stats.farTierDisplayedFinalCounts[2], stats.farTierDisplayedFinalCounts[3],
                stats.farTierDisplayedFinalCounts[4], stats.farTierDisplayedFinalCounts[5],
                stats.farPendingAuthorityTransitionCount);
            RY_LOG_INFO(preparationAuthorityEvidence);
        }
        UIFrameState preparationFrame;
        preparationFrame.screen = GameScreen::TITLE;
        preparationFrame.hoveredButton = state->hoveredButton;
        preparationFrame.menu = state->menuLayout;
        if (mayPrepareV4World) {
            _renderPipeline->renderV4Preparation(_queue, drawable, preparationFrame, *state->world,
                                                 state->camera);
        } else {
            _renderPipeline->renderMenuOnly(_queue, drawable, preparationFrame);
        }
        return;
    }

    // Playtest captures request one full scene frame, then leave 60 rendered
    // frames for the asynchronous PNG write before terminating.
    advanceFrameCapture();

    // Menu-only frame while no world session is live (title, world menus).
    if (!state->world) {
        UIFrameState uiFrame;
        uiFrame.screen = state->flow.screen;
        uiFrame.hoveredButton = state->hoveredButton;
        uiFrame.menu = state->menuLayout;
        _renderPipeline->renderMenuOnly(_queue, drawable, uiFrame);
        return;
    }

    // Log render + streaming diagnostics every 60 frames (the same numbers
    // the F3 HUD shows, so headless playtests can measure against budgets;
    // the chunk count reuses the HUD's 30-frame sample instead of copying
    // the whole chunk vector)
    if (state->frameCount % 60 == 1) {
        auto chunkStats = _renderPipeline->chunkRenderStats();
        const int configuredHorizonChunks = chunkStats.farBaseViewDistanceChunks;
        const int entryHorizonChunks = v4RequiredEntryParentRadiusChunks(configuredHorizonChunks);
        const float connectedParentRadiusChunks =
            v4ConnectedParentRadiusChunks(chunkStats, configuredHorizonChunks);
        const StreamingWorkStats streaming = state->world->getStreamingWorkStats();
        char line[1152];
        snprintf(line, sizeof(line),
                 "Render: frame %llu player (%.1f, %.1f, %.1f) | %.2f ms/frame gpu %.2f ms "
                 "cubes %u loaded %u meshed gen %.2f ms mesh %.2f ms queues %zu/%u high %u "
                 "exact %.0f/%.0f MB far %u wanted %u resident %u drawn %u frustum %u occluded "
                 "%u pending entry %.1f/%d configured %d chunks %.0f MB cache %.0f MB arena "
                 "planner %.1f ms %llu/%llu/%llu farplan %.2f/%.2f/%.2f ms phase "
                 "%.2f/%.2f/%.2f deny %llu critical %u/%u/%u displace %llu evict %llu/%.1f MB "
                 "light %zu/%llu/%zu",
                 static_cast<unsigned long long>(state->frameCount), state->player.position.x,
                 state->player.position.y, state->player.position.z, state->smoothedFrameMs,
                 _renderPipeline->gpuFrameMs(), state->cachedChunkCount, chunkStats.meshCubeCount,
                 state->world->averageGenMs(), chunkStats.meshMsAvg,
                 state->world->getPendingChunkCount(), chunkStats.meshPendingCount,
                 chunkStats.meshQueueHighWater, chunkStats.megaUsedMB, chunkStats.megaCapMB,
                 chunkStats.farWantedTileCount, chunkStats.farResidentTileCount,
                 chunkStats.farDrawnTileCount, chunkStats.farFrustumCulledTileCount,
                 chunkStats.farOcclusionCulledTileCount, chunkStats.farPendingTileCount,
                 connectedParentRadiusChunks, entryHorizonChunks, configuredHorizonChunks,
                 chunkStats.farCacheMB, chunkStats.farMegaUsedMB, streaming.activeSetBuildMs,
                 static_cast<unsigned long long>(streaming.activeSetRequests),
                 static_cast<unsigned long long>(streaming.activeSetRequestsCoalesced),
                 static_cast<unsigned long long>(streaming.activeSetBuildsCanceled),
                 chunkStats.farPlannerMsLast, chunkStats.farPlannerMsP95,
                 chunkStats.farPlannerMsMax, chunkStats.farPlannerSelectionMsP95,
                 chunkStats.farPlannerPublicationMsP95, chunkStats.farPlannerResidencyMsP95,
                 static_cast<unsigned long long>(chunkStats.farArenaAdmissionDeniedCount),
                 chunkStats.farCriticalWantedTileCount, chunkStats.farCriticalResidentTileCount,
                 chunkStats.farCriticalMissingTileCount,
                 static_cast<unsigned long long>(chunkStats.farCriticalSchedulerDisplacementCount),
                 static_cast<unsigned long long>(chunkStats.farNearArenaReclaimCount),
                 static_cast<double>(chunkStats.farNearArenaReclaimedBytes) / (1024.0 * 1024.0),
                 streaming.publicationLightDeferredQueue,
                 static_cast<unsigned long long>(streaming.publicationLightDeferredCubes),
                 streaming.publicationLightMaxSyncFloods);
        RY_LOG_INFO(line);
        static const bool captureDiagnostics = [] {
            const char* capture = std::getenv("RYCRAFT_CAPTURE");
            return capture != nullptr && *capture != '\0';
        }();
        if (state->showDebugHud || state->performance.enabled() || captureDiagnostics) {
            char coverageLine[640];
            snprintf(coverageLine, sizeof(coverageLine),
                     "Coverage: exact %u/%u ready unresolved %u handoff %.0f blocks | base r%d "
                     "%u/%u/%u wanted/resident/drawn missing %u cached %u | refine %u/%u/%u "
                     "wanted/resident/drawn | final handoff missing %u | frontier %.0f blocks "
                     "queues %u/%u base/refine | "
                     "workers %u/%u/%u base/reserved/budget urgent %u",
                     chunkStats.exactSurfaceReadyCount, chunkStats.exactSurfaceRequiredCount,
                     chunkStats.exactSurfaceUnresolvedColumnCount,
                     chunkStats.exactSurfaceHandoffBlocks, chunkStats.farBaseViewDistanceChunks,
                     chunkStats.farBaseWantedTileCount, chunkStats.farBaseResidentTileCount,
                     chunkStats.farBaseDrawnTileCount, chunkStats.farBaseMissingTileCount,
                     chunkStats.farCachedBaseTileCount, chunkStats.farRefinementWantedTileCount,
                     chunkStats.farRefinementResidentTileCount,
                     chunkStats.farRefinementDrawnTileCount,
                     chunkStats.farExactHandoffMissingFinalParentCount,
                     chunkStats.farCoverageFrontierBlocks, chunkStats.farQueuedBaseTileCount,
                     chunkStats.farQueuedRefinementTileCount, chunkStats.farActiveBaseWorkerCount,
                     chunkStats.farReservedBaseWorkerCount, chunkStats.farWorkerBudget,
                     chunkStats.farActiveUrgentRefinementCount);
            RY_LOG_INFO(coverageLine);
            char canopyLine[384];
            snprintf(canopyLine, sizeof(canopyLine),
                     "Canopy: %u in-flight %u active %u queued %u parked %u completed | "
                     "cache %u/%.1f MB | failed %llu deferred %llu authority resumes %llu",
                     chunkStats.farCanopyInFlightCount, chunkStats.farActiveCanopyWorkerCount,
                     chunkStats.farQueuedCanopyCount, chunkStats.farParkedCanopyCount,
                     chunkStats.farCompletedCanopyCount, chunkStats.farCanopyCacheEntryCount,
                     chunkStats.farCanopyCacheMB,
                     static_cast<unsigned long long>(chunkStats.farCanopyFailedCount),
                     static_cast<unsigned long long>(chunkStats.farCanopyDeferredCount),
                     static_cast<unsigned long long>(
                         chunkStats.farCanopyAuthorityCompletionResumeCount));
            RY_LOG_INFO(canopyLine);
        }
        // Per-pass GPU breakdown (RYCRAFT_GPU_COUNTERS=1) mirrors to the log
        // so headless runs can attribute frame cost to individual passes.
        std::string passes = _renderPipeline->gpuPassBreakdown();
        if (!passes.empty()) {
            RY_LOG_INFO(("GPU passes (ms): " + passes).c_str());
        }
    }

    // Camera view matrix
    Mat4 viewMatrix = state->camera.viewMatrix();

    // Real performance stats for the F3 HUD (EMA-smoothed frame time; chunk
    // count sampled every 30 frames, getLoadedChunks copies under a lock)
    state->smoothedFrameMs =
        state->smoothedFrameMs * 0.95f + static_cast<float>(state->deltaTime) * 1000.0f * 0.05f;
    if (state->frameCount % 30 == 0) {
        state->cachedChunkCount = static_cast<uint32_t>(state->world->getLoadedChunkCount());
        state->cachedPendingChunks = static_cast<uint32_t>(state->world->getPendingChunkCount());
    }

    UIFrameState uiFrame;
    uiFrame.screen = state->flow.screen;
    uiFrame.hoveredButton = state->hoveredButton;
    uiFrame.showDebugHud = state->showDebugHud;
    uiFrame.hotbar.selected = state->inventory.getSelectedIndex();
    for (int slot = 0; slot < Inventory::HOTBAR_SLOTS; ++slot) {
        uiFrame.hotbar.slots[static_cast<size_t>(slot)] = state->inventory.getSlot(slot);
    }
    uiFrame.hoveredSlot = state->hoveredSlot;
    uiFrame.cursorStack = state->cursorStack;
    uiFrame.miningProgress = state->miningState.active ? state->miningState.progress : 0.f;
    uiFrame.mode = state->gameMode;
    uiFrame.health = state->player.health;
    uiFrame.food = state->survival.food;
    uiFrame.air = state->survival.air;
    uiFrame.maxAir = SurvivalStats::MAX_AIR;
    uiFrame.deathMessage = state->deathMessage;
    if (state->inputManager && _view) {
        const Vec2 mouse = state->inputManager->state().mousePosition;
        const float boundsW = static_cast<float>(_view.bounds.size.width);
        const float boundsH = static_cast<float>(_view.bounds.size.height);
        if (boundsW > 0.f && boundsH > 0.f) {
            uiFrame.mouseX = mouse.x / boundsW;
            uiFrame.mouseY = mouse.y / boundsH;
        }
    }
    if (state->hoveredSlot >= 0 && state->cursorStack.empty() &&
        state->hoveredSlot < static_cast<int>(state->menuLayout.slots.size())) {
        const ItemStack& hovered =
            state->menuLayout.slots[static_cast<size_t>(state->hoveredSlot)].stack;
        if (!hovered.empty()) {
            uiFrame.tooltipText = itemName(hovered.type);
        }
    }
    // Underwater view (veil, god rays, dense fog): the camera cell is water.
    // Non-generating read, a streaming lag must never stall the frame.
    {
        Vec3 camPos = state->camera.getPosition();
        const int64_t waterX = static_cast<int64_t>(std::floor(camPos.x));
        const int32_t waterY = static_cast<int32_t>(std::floor(camPos.y));
        const int64_t waterZ = static_cast<int64_t>(std::floor(camPos.z));
        uiFrame.cameraUnderwater =
            state->world->getBlockIfLoaded(waterX, waterY, waterZ) == BlockType::WATER &&
            camPos.y < static_cast<float>(waterY) +
                           state->world->getFluidHeightIfLoaded(waterX, waterY, waterZ);
    }
    uiFrame.stats.frameTimeMs = state->smoothedFrameMs;
    uiFrame.stats.gpuFrameMs = _renderPipeline->gpuFrameMs();
    uiFrame.stats.fps = state->smoothedFrameMs > 0.f ? 1000.0f / state->smoothedFrameMs : 0.f;
    uiFrame.stats.chunkCount = state->cachedChunkCount;
    uiFrame.stats.entityCount =
        state->spawner ? static_cast<uint32_t>(state->spawner->getEntities().size()) : 0;
    uiFrame.stats.pendingChunks = state->cachedPendingChunks;
    uiFrame.stats.genMsAvg = state->world->averageGenMs();
    auto chunkStats = _renderPipeline->chunkRenderStats();
    uiFrame.stats.meshMsAvg = chunkStats.meshMsAvg;
    uiFrame.stats.meshBuildsFrame = chunkStats.meshBuildsLastFrame;
    uiFrame.stats.megaUsedMB = chunkStats.megaUsedMB;
    uiFrame.stats.megaCapMB = chunkStats.megaCapMB;
    uiFrame.stats.meshedCubeCount = chunkStats.meshCubeCount;
    uiFrame.stats.exactSurfaceRequired = chunkStats.exactSurfaceRequiredCount;
    uiFrame.stats.exactSurfaceReady = chunkStats.exactSurfaceReadyCount;
    uiFrame.stats.exactSurfaceUnresolvedColumns = chunkStats.exactSurfaceUnresolvedColumnCount;
    uiFrame.stats.exactSurfaceHandoffBlocks = chunkStats.exactSurfaceHandoffBlocks;
    uiFrame.stats.farWantedTiles = chunkStats.farWantedTileCount;
    uiFrame.stats.farResidentTiles = chunkStats.farResidentTileCount;
    uiFrame.stats.farDrawnTiles = chunkStats.farDrawnTileCount;
    uiFrame.stats.farBaseWantedTiles = chunkStats.farBaseWantedTileCount;
    uiFrame.stats.farBaseResidentTiles = chunkStats.farBaseResidentTileCount;
    uiFrame.stats.farBaseDrawnTiles = chunkStats.farBaseDrawnTileCount;
    uiFrame.stats.farBaseMissingTiles = chunkStats.farBaseMissingTileCount;
    uiFrame.stats.farRefinementWantedTiles = chunkStats.farRefinementWantedTileCount;
    uiFrame.stats.farRefinementResidentTiles = chunkStats.farRefinementResidentTileCount;
    uiFrame.stats.farRefinementDrawnTiles = chunkStats.farRefinementDrawnTileCount;
    uiFrame.stats.farFrustumCulledTiles = chunkStats.farFrustumCulledTileCount;
    uiFrame.stats.farOcclusionCulledTiles = chunkStats.farOcclusionCulledTileCount;
    uiFrame.stats.farPendingTiles = chunkStats.farPendingTileCount;
    uiFrame.stats.farQueuedBaseTiles = chunkStats.farQueuedBaseTileCount;
    uiFrame.stats.farQueuedRefinementTiles = chunkStats.farQueuedRefinementTileCount;
    uiFrame.stats.farActiveBaseWorkers = chunkStats.farActiveBaseWorkerCount;
    uiFrame.stats.farReservedBaseWorkers = chunkStats.farReservedBaseWorkerCount;
    uiFrame.stats.farActiveUrgentRefinements = chunkStats.farActiveUrgentRefinementCount;
    uiFrame.stats.farWorkerBudget = chunkStats.farWorkerBudget;
    uiFrame.stats.farCachedBaseTiles = chunkStats.farCachedBaseTileCount;
    uiFrame.stats.farCanopyInFlight = chunkStats.farCanopyInFlightCount;
    uiFrame.stats.farActiveCanopyWorkers = chunkStats.farActiveCanopyWorkerCount;
    uiFrame.stats.farQueuedCanopies = chunkStats.farQueuedCanopyCount;
    uiFrame.stats.farParkedCanopies = chunkStats.farParkedCanopyCount;
    uiFrame.stats.farCompletedCanopies = chunkStats.farCompletedCanopyCount;
    uiFrame.stats.farCanopyCacheEntries = chunkStats.farCanopyCacheEntryCount;
    uiFrame.stats.farCanopyFailures = chunkStats.farCanopyFailedCount;
    uiFrame.stats.farCanopyDeferrals = chunkStats.farCanopyDeferredCount;
    uiFrame.stats.farCanopyAuthorityResumes = chunkStats.farCanopyAuthorityCompletionResumeCount;
    uiFrame.stats.farCoverageFrontierBlocks = chunkStats.farCoverageFrontierBlocks;
    uiFrame.stats.farCacheMB = chunkStats.farCacheMB;
    uiFrame.stats.farCanopyCacheMB = chunkStats.farCanopyCacheMB;
    uiFrame.stats.farMeshMB = chunkStats.farMegaUsedMB;
    uiFrame.stats.farPlannerMsLast = chunkStats.farPlannerMsLast;
    uiFrame.stats.farPlannerMsP95 = chunkStats.farPlannerMsP95;
    uiFrame.stats.farPlannerMsMax = chunkStats.farPlannerMsMax;
    uiFrame.stats.farArenaAdmissionDenials = chunkStats.farArenaAdmissionDeniedCount;
    const StreamingWorkStats streamingStats = state->world->getStreamingWorkStats();
    uiFrame.stats.publicationLightDeferredQueue = static_cast<uint32_t>(std::min<size_t>(
        streamingStats.publicationLightDeferredQueue, std::numeric_limits<uint32_t>::max()));
    uiFrame.stats.publicationLightDeferredCubes = streamingStats.publicationLightDeferredCubes;
    uiFrame.stats.publicationLightMaxSyncFloods = static_cast<uint32_t>(std::min<size_t>(
        streamingStats.publicationLightMaxSyncFloods, std::numeric_limits<uint32_t>::max()));
    const int64_t playerX = static_cast<int64_t>(std::floor(state->player.position.x));
    const int32_t playerY = static_cast<int32_t>(std::floor(state->player.position.y));
    const int64_t playerZ = static_cast<int64_t>(std::floor(state->player.position.z));
    uiFrame.stats.cubeX = Chunk::worldToChunk(playerX);
    uiFrame.stats.cubeY = Chunk::worldToChunkY(playerY);
    uiFrame.stats.cubeZ = Chunk::worldToChunk(playerZ);
    if (state->showDebugHud) {
        const size_t planEntries = state->world->cachedColumnPlanCount();
        const worldgen::BasinCacheMetrics basinCache =
            state->world->generator().basinCacheMetrics();
        const worldgen::MacroControlCacheMetrics macroControlCache =
            state->world->generator().macroControlCacheMetrics();
        const FarTerrainGenerationCacheStats farGenerationCache =
            _renderPipeline->farGenerationCacheStats();
        uiFrame.stats.macroCacheEntries =
            static_cast<uint32_t>(planEntries + basinCache.entries + basinCache.shorelineEntries +
                                  macroControlCache.entries + farGenerationCache.entries);
        uiFrame.stats.macroCacheMB =
            static_cast<float>(planEntries * sizeof(ColumnPlan) + basinCache.bytes +
                               basinCache.shorelineBytes + macroControlCache.bytes +
                               farGenerationCache.bytes) /
            (1024.0f * 1024.0f);
    }
    uiFrame.stats.pendingFluids = static_cast<uint32_t>(state->world->getPendingFluidCount());
    uiFrame.stats.droppedFluidUpdates = state->world->getDroppedFluidUpdateCount();
    uiFrame.stats.droppedFluidFrontiers = state->world->getDroppedFluidFrontierCount();
    const RenderPipeline::AtmosphericRenderStats atmospheric =
        _renderPipeline->atmosphericRenderStats();
    const WeatherSystemStats weatherSystemStats =
        state->weatherSystem ? state->weatherSystem->stats() : WeatherSystemStats{};
    uiFrame.stats.shadowRefreshMask = atmospheric.shadowRefreshMask;
    uiFrame.stats.shadowCasterCounts = atmospheric.shadowCasterCounts;
    uiFrame.stats.shadowRefreshCounts = atmospheric.shadowRefreshCounts;
    uiFrame.stats.indirectHistoryResetMask = atmospheric.indirectHistoryResetMask;
    uiFrame.stats.indirectHistoryValid = atmospheric.indirectHistoryValid;
    uiFrame.stats.cloudHistoryValid = atmospheric.cloudHistoryValid;
    uiFrame.stats.froxelHistoryValid = atmospheric.froxelHistoryValid;
    uiFrame.stats.atmosphereSlowRefreshCount = atmospheric.atmosphereSlowRefreshCount;
    uiFrame.stats.atmosphereSkyRefreshCount = atmospheric.atmosphereSkyRefreshCount;
    constexpr float BYTES_TO_MEBIBYTES = 1.0F / (1024.0F * 1024.0F);
    uiFrame.stats.indirectPersistentMB =
        static_cast<float>(atmospheric.indirectPersistentBytes) * BYTES_TO_MEBIBYTES;
    uiFrame.stats.cloudPersistentMB =
        static_cast<float>(atmospheric.cloudPersistentBytes) * BYTES_TO_MEBIBYTES;
    uiFrame.stats.froxelPersistentMB =
        static_cast<float>(atmospheric.froxelPersistentBytes) * BYTES_TO_MEBIBYTES;
    uiFrame.stats.integratedAtmosphericPersistentMB =
        static_cast<float>(atmospheric.integratedPersistentBytes) * BYTES_TO_MEBIBYTES;
    uiFrame.stats.weatherRequests = weatherSystemStats.requests;
    uiFrame.stats.weatherCoalescedRequests = weatherSystemStats.coalescedRequests;
    uiFrame.stats.weatherBuildsStarted = weatherSystemStats.buildsStarted;
    uiFrame.stats.weatherBuildsDeferred = weatherSystemStats.buildsDeferred;
    uiFrame.stats.weatherBuildsFailed = weatherSystemStats.buildsFailed;
    uiFrame.stats.weatherSnapshotsPublished = weatherSystemStats.snapshotsPublished;
    uiFrame.stats.weatherStaleBuildsDiscarded = weatherSystemStats.staleBuildsDiscarded;
    uiFrame.stats.weatherPendingRequests =
        static_cast<uint32_t>(weatherSystemStats.pendingRequests);
    uiFrame.stats.weatherWorkerBusy = weatherSystemStats.workerBusy;
    uiFrame.stats.thunderPending = static_cast<uint32_t>(state->thunder.pendingCount());
    uiFrame.stats.weatherPressureHpa = state->localWeather.pressureHpa;
    uiFrame.stats.weatherHumidity = state->localWeather.relativeHumidity;
    uiFrame.stats.weatherTemperatureC = state->localWeather.temperatureC;
    uiFrame.stats.weatherWindX = state->localWeather.windBlocksPerSecond.x;
    uiFrame.stats.weatherWindZ = state->localWeather.windBlocksPerSecond.y;
    uiFrame.stats.cloudCoverage = state->localWeather.cloudCoverage;
    uiFrame.stats.cloudType = static_cast<uint8_t>(state->localWeather.cloudType);
    uiFrame.stats.precipitationIntensity = state->localWeather.precipitationIntensity;
    uiFrame.stats.precipitationKind = static_cast<uint8_t>(state->localWeather.precipitationKind);
    uiFrame.stats.stormPotential = state->localWeather.stormPotential;
    uiFrame.stats.weatherFogExtinction = state->localWeather.fogExtinction;
    uiFrame.stats.aerosolDensity = state->localWeather.aerosolDensity;
    uiFrame.stats.stormId = atmospheric.lightningEventId;
    uiFrame.stats.lunarPhaseEnergy = atmospheric.lunarPhaseEnergy;
    uiFrame.stats.lunarPhaseCycle = atmospheric.lunarPhaseCycle;
    if (state->showDebugHud) {
        if (const auto surface = state->world->findSurfaceSample(playerX, playerZ)) {
            uiFrame.stats.plateId = surface->geology.plateId;
            uiFrame.stats.boundary = surface->geology.boundary;
            uiFrame.stats.temperatureC = static_cast<float>(surface->climate.temperatureC);
            uiFrame.stats.precipitationMm =
                static_cast<float>(surface->climate.annualPrecipitationMm);
            uiFrame.stats.primaryBiome = surface->biome.primary;
            uiFrame.stats.secondaryBiome = surface->biome.secondary;
            uiFrame.stats.biomeTransition = static_cast<float>(surface->biome.transition);
            uiFrame.stats.riverOrder = surface->hydrology.streamOrder;
        }
    }
    uiFrame.menu = state->menuLayout;

    _renderPipeline->render(_queue, drawable, viewMatrix, state->projectionMatrix, *state->world,
                            state->camera, state->worldTime, state->deltaTime,
                            state->hasHighlightedBlock
                                ? std::optional<BlockHighlight>(state->highlightedBlock)
                                : std::nullopt,
                            uiFrame, state->spawner ? &state->spawner->getEntities() : nullptr,
                            &state->itemEntities.items(), &state->boats.boats(),
                            state->weatherSnapshot, &state->lightningEvents);

    if (updatePerformanceCapture(state->performance, state->frameCount, state->deltaTime * 1000.0,
                                 state->cachedChunkCount, state->autopilotStopFrame, *state->world,
                                 *_renderPipeline, _device, weatherSystemStats,
                                 state->thunder.pendingCount())) {
        [self requestQuit];
    }
}

- (double)deltaTime {
    return _state->deltaTime;
}

- (uint64_t)frameCount {
    return _state->frameCount;
}

@end

// ---- C++ bridge functions (visible from header) ----

Mat4 engineProjectionMatrix(Engine* engine) {
    // Access internal state via the static helper defined in @implementation
    extern EngineState* _engineGetState(Engine*);
    EngineState* state = _engineGetState(engine);
    return state ? state->projectionMatrix : Mat4::identity();
}

CGSize engineDrawableSize(Engine* engine) {
    extern EngineState* _engineGetState(Engine*);
    EngineState* state = _engineGetState(engine);
    if (!state)
        return {0, 0};
    return state->drawableSize;
}

#pragma once

#include "common/math.hpp"
#include "world/macro_generation.hpp"
#include "world/physical_scale.hpp"
#include "world/weather_grid.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

class ChunkGenerator;

enum class PrecipitationKind : uint8_t {
    NONE,
    RAIN,
    SNOW,
};

enum class CloudType : uint8_t {
    CLEAR,
    CIRRUS,
    STRATUS,
    CUMULUS,
    CUMULONIMBUS,
};

enum class WeatherPreset : uint8_t {
    NATURAL,
    CLEAR,
    OVERCAST,
    RAIN,
    STORM,
    SNOW,
};

struct CloudLayerBounds {
    float baseY = 192.0f;
    float topY = 236.0f;
};

struct WeatherSample {
    // Vec2::x is world X and Vec2::y is world Z. Speeds are blocks per second.
    Vec2 windBlocksPerSecond;
    // Deterministic integral of the canonical wind field. Renderers use this
    // phase instead of integrating frame time, so saved time and fixed
    // captures reconstruct identical cloud positions. Double precision keeps
    // the integral continuous when saved worlds have very large tick counts.
    worldgen::Vector2d cloudOffsetBlocks;
    // High clouds use a separate physical flow map. Keeping both phases in
    // double precision lets cirrus and storm tops move independently without
    // deriving a new phase from a categorical cloud-type transition.
    worldgen::Vector2d highCloudOffsetBlocks;
    float pressureHpa = 1013.25f;
    float relativeHumidity = 0.35f;
    float temperatureC = 15.0f;
    float cloudCoverage = 0.0f;
    float precipitationIntensity = 0.0f;
    float stormPotential = 0.0f;
    float fogExtinction = 0.00015f;
    float aerosolDensity = 0.08f;
    float cloudBaseY = 192.0f;
    float cloudTopY = 236.0f;
    float terrainHeight = 64.0f;
    PrecipitationKind precipitationKind = PrecipitationKind::NONE;
    CloudType cloudType = CloudType::CLEAR;
};

constexpr uint64_t weatherTimeSliceStart(uint64_t worldTick, uint64_t sliceTicks) noexcept {
    if (sliceTicks == 0U) return worldTick;
    const uint64_t maximumStart = std::numeric_limits<uint64_t>::max() - sliceTicks;
    return std::min(worldTick - worldTick % sliceTicks, maximumStart);
}

struct LightningEvent {
    uint64_t id = 0;
    uint64_t tick = 0;
    double x = 0.0;
    float y = 0.0f;
    double z = 0.0;
    float cloudY = 0.0f;
    float intensity = 0.0f;
};

struct WeatherSystemStats {
    uint64_t requests = 0;
    uint64_t coalescedRequests = 0;
    uint64_t buildsStarted = 0;
    uint64_t buildsDeferred = 0;
    uint64_t buildsFailed = 0;
    uint64_t snapshotsPublished = 0;
    uint64_t staleBuildsDiscarded = 0;
    uint64_t lightningDiscoveryBuilds = 0;
    uint64_t lightningDiscoveryCacheHits = 0;
    size_t pendingRequests = 0;
    bool workerBusy = false;
};

std::optional<WeatherPreset> weatherPresetFromString(std::string_view value) noexcept;
Vec2 weatherAdvectionVelocity(const worldgen::ClimateFields& climate) noexcept;
CloudLayerBounds
cloudLayerBounds(CloudType type, float terrainHeight, float relativeHumidity,
                 WorldPhysicalScale physicalScale = LEGACY_WORLD_PHYSICAL_SCALE) noexcept;
float cloudProfileDensity(CloudType type, float normalizedHeight) noexcept;
WeatherSample
deriveWeatherSample(uint64_t worldSeed, double worldX, double worldZ, uint64_t worldTick,
                    const worldgen::SurfaceSample& staticClimate,
                    WeatherPreset preset = WeatherPreset::NATURAL,
                    WorldPhysicalScale physicalScale = LEGACY_WORLD_PHYSICAL_SCALE) noexcept;
std::optional<LightningEvent> lightningEventForCell(uint64_t worldSeed, int64_t stormCellX,
                                                    int64_t stormCellZ, uint64_t timeBucket,
                                                    const WeatherSample& weather) noexcept;
double thunderDelaySeconds(const LightningEvent& event, double listenerX, double listenerY,
                           double listenerZ,
                           WorldPhysicalScale physicalScale = LEGACY_WORLD_PHYSICAL_SCALE) noexcept;

// An immutable pair of weather grids. Readers may retain this object while a
// replacement is built and published by WeatherSystem's utility worker.
class WeatherSnapshot {
public:
    static constexpr int GRID_EDGE = WEATHER_GRID_EDGE;
    static constexpr int GRID_SPACING = static_cast<int>(WEATHER_GRID_CELL_SPACING_BLOCKS);
    static constexpr int TIME_SLICE_TICKS = 200;
    static constexpr size_t GRID_SAMPLE_COUNT =
        static_cast<size_t>(GRID_EDGE) * static_cast<size_t>(GRID_EDGE);

    WeatherSnapshot(uint64_t requestId, int64_t centerX, int64_t centerZ, uint64_t firstTick,
                    WeatherPreset preset, std::vector<WeatherSample> first,
                    std::vector<WeatherSample> second);

    WeatherSnapshot(const WeatherSnapshot&) = delete;
    WeatherSnapshot& operator=(const WeatherSnapshot&) = delete;

    uint64_t requestId() const noexcept { return requestId_; }
    int64_t centerX() const noexcept { return centerX_; }
    int64_t centerZ() const noexcept { return centerZ_; }
    int64_t originX() const noexcept { return originX_; }
    int64_t originZ() const noexcept { return originZ_; }
    uint64_t firstTick() const noexcept { return firstTick_; }
    uint64_t secondTick() const noexcept { return secondTick_; }
    WeatherPreset preset() const noexcept { return preset_; }

    bool covers(double worldX, double worldZ) const noexcept;
    const WeatherSample& gridSample(int sampleX, int sampleZ, int timeSlice) const;
    std::span<const WeatherSample> timeSlice(int timeSlice) const;
    WeatherSample sample(double worldX, double worldZ, uint64_t worldTick) const noexcept;

private:
    uint64_t requestId_ = 0;
    int64_t centerX_ = 0;
    int64_t centerZ_ = 0;
    int64_t originX_ = 0;
    int64_t originZ_ = 0;
    uint64_t firstTick_ = 0;
    uint64_t secondTick_ = TIME_SLICE_TICKS;
    WeatherPreset preset_ = WeatherPreset::NATURAL;
    std::vector<WeatherSample> first_;
    std::vector<WeatherSample> second_;
};

// Builds the horizon weather map on one joinable utility worker. Admission is
// latest-wins: at most one request waits while one request is being built.
class WeatherSystem {
public:
    static constexpr int RECENTER_DISTANCE = 1'024;
    static constexpr int LIGHTNING_CELL_EDGE = 1'024;
    static constexpr uint64_t LIGHTNING_BUCKET_TICKS = 40;
    static constexpr size_t MAX_LIGHTNING_EVENTS_PER_QUERY = 8;

    static constexpr uint64_t lightningTickForBucket(uint64_t timeBucket,
                                                     uint64_t tickOffset) noexcept {
        constexpr uint64_t MAXIMUM_TICK = std::numeric_limits<uint64_t>::max();
        if (timeBucket > MAXIMUM_TICK / LIGHTNING_BUCKET_TICKS) return MAXIMUM_TICK;
        const uint64_t bucketStart = timeBucket * LIGHTNING_BUCKET_TICKS;
        if (tickOffset > MAXIMUM_TICK - bucketStart) return MAXIMUM_TICK;
        return bucketStart + tickOffset;
    }

    using ClimateGridSampler =
        std::function<void(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                           std::span<worldgen::SurfaceSample> output)>;

    explicit WeatherSystem(const ChunkGenerator& generator);
    WeatherSystem(uint64_t worldSeed, ClimateGridSampler climateGridSampler,
                  WeatherPreset preset = WeatherPreset::NATURAL,
                  WorldPhysicalScale physicalScale = LEGACY_WORLD_PHYSICAL_SCALE);
    ~WeatherSystem();

    WeatherSystem(const WeatherSystem&) = delete;
    WeatherSystem& operator=(const WeatherSystem&) = delete;
    WeatherSystem(WeatherSystem&&) = delete;
    WeatherSystem& operator=(WeatherSystem&&) = delete;

    uint64_t requestSnapshot(int64_t cameraX, int64_t cameraZ, uint64_t worldTick);
    void setPreset(WeatherPreset preset);
    WeatherPreset preset() const;
    WorldPhysicalScale physicalScale() const noexcept { return physicalScale_; }

    std::shared_ptr<const WeatherSnapshot> latestSnapshot() const;
    WeatherSample sample(double worldX, double worldZ, uint64_t worldTick) const;
    std::vector<LightningEvent> lightningEvents(uint64_t previousTick, uint64_t currentTick) const;
    WeatherSystemStats stats() const;

    // Test and capture synchronization only. Ordinary gameplay retains the
    // previous snapshot and never waits for weather construction.
    bool waitForSnapshot(uint64_t requestId, std::chrono::milliseconds timeout) const;

private:
    struct Request {
        uint64_t id = 0;
        int64_t centerX = 0;
        int64_t centerZ = 0;
        uint64_t firstTick = 0;
        WeatherPreset preset = WeatherPreset::NATURAL;

        bool sameBuild(const Request& other) const noexcept;
    };

    struct LightningDiscoveryCacheEntry {
        uint64_t snapshotRequestId = 0;
        uint64_t timeBucket = 0;
        std::vector<LightningEvent> events;
    };

    uint64_t enqueueLocked(int64_t cameraX, int64_t cameraZ, uint64_t worldTick,
                           WeatherPreset preset);
    std::shared_ptr<const WeatherSnapshot> buildSnapshot(const Request& request) const;
    std::vector<LightningEvent> lightningEventsForBucket(const WeatherSnapshot& snapshot,
                                                         uint64_t timeBucket) const;
    void workerMain();

    uint64_t worldSeed_ = 0;
    ClimateGridSampler climateGridSampler_;
    WorldPhysicalScale physicalScale_ = LEGACY_WORLD_PHYSICAL_SCALE;

    mutable std::mutex mutex_;
    mutable std::condition_variable requestCondition_;
    mutable std::condition_variable snapshotCondition_;
    std::thread worker_;
    bool stopping_ = false;
    bool workerBusy_ = false;
    WeatherPreset preset_ = WeatherPreset::NATURAL;
    uint64_t nextRequestId_ = 1;
    uint64_t latestRequestId_ = 0;
    std::optional<Request> desiredRequest_;
    std::optional<Request> pendingRequest_;
    std::shared_ptr<const WeatherSnapshot> snapshot_;
    WeatherSystemStats stats_;

    static constexpr size_t MAX_LIGHTNING_DISCOVERY_CACHE_ENTRIES = 4;
    mutable std::mutex lightningDiscoveryMutex_;
    mutable std::vector<LightningDiscoveryCacheEntry> lightningDiscoveryCache_;
    mutable uint64_t lightningDiscoveryBuilds_ = 0;
    mutable uint64_t lightningDiscoveryCacheHits_ = 0;
};

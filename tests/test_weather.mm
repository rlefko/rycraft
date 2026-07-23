#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <world/weather.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

worldgen::SurfaceSample climateSample(double temperatureC = 12.0, double relativeHumidity = 0.62,
                                      double windX = 0.8, double windZ = 0.3) {
    worldgen::SurfaceSample sample;
    sample.terrainHeight = 72.0;
    sample.climate.wind = {windX, windZ};
    sample.climate.temperatureC = temperatureC;
    sample.climate.annualPrecipitationMm = 1'100.0;
    sample.climate.potentialEvapotranspirationMm = 760.0;
    sample.climate.aridity = 760.0 / 1'100.0;
    sample.climate.relativeHumidity = relativeHumidity;
    return sample;
}

void fillClimateGrid(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                     std::span<worldgen::SurfaceSample> output) {
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const int64_t worldX = originX + static_cast<int64_t>(x) * spacing;
            const int64_t worldZ = originZ + static_cast<int64_t>(z) * spacing;
            worldgen::SurfaceSample sample = climateSample();
            sample.terrainHeight = 64.0 + static_cast<double>((worldX + worldZ) % 17) * 0.1;
            sample.climate.temperatureC += static_cast<double>(worldX) / 100'000.0;
            sample.climate.relativeHumidity =
                std::clamp(0.58 + static_cast<double>(worldZ) / 1'000'000.0, 0.2, 0.95);
            output[static_cast<size_t>(z * sampleEdge + x)] = sample;
        }
    }
}

void requireAtmosphereEqual(const WeatherSample& first, const WeatherSample& second) {
    REQUIRE(first.windBlocksPerSecond.x == Catch::Approx(second.windBlocksPerSecond.x));
    REQUIRE(first.windBlocksPerSecond.y == Catch::Approx(second.windBlocksPerSecond.y));
    REQUIRE(first.pressureHpa == Catch::Approx(second.pressureHpa));
    REQUIRE(first.relativeHumidity == Catch::Approx(second.relativeHumidity));
    REQUIRE(first.temperatureC == Catch::Approx(second.temperatureC));
    REQUIRE(first.cloudCoverage == Catch::Approx(second.cloudCoverage));
    REQUIRE(first.precipitationIntensity == Catch::Approx(second.precipitationIntensity));
    REQUIRE(first.stormPotential == Catch::Approx(second.stormPotential));
    REQUIRE(first.fogExtinction == Catch::Approx(second.fogExtinction));
    REQUIRE(first.aerosolDensity == Catch::Approx(second.aerosolDensity));
    REQUIRE(first.cloudBaseY == Catch::Approx(second.cloudBaseY));
    REQUIRE(first.cloudTopY == Catch::Approx(second.cloudTopY));
    REQUIRE(first.precipitationKind == second.precipitationKind);
    REQUIRE(first.cloudType == second.cloudType);
}

} // namespace

TEST_CASE("Weather grid dimensions cover the horizon with two ten-second slices",
          "[weather][snapshot][bounds]") {
    STATIC_REQUIRE(WeatherSnapshot::GRID_EDGE == 81);
    STATIC_REQUIRE(WeatherSnapshot::GRID_SPACING == 256);
    STATIC_REQUIRE(WeatherSnapshot::TIME_SLICE_TICKS == 200);
    STATIC_REQUIRE(WeatherSnapshot::GRID_SAMPLE_COUNT == 6'561);
    STATIC_REQUIRE((WeatherSnapshot::GRID_EDGE / 2) * WeatherSnapshot::GRID_SPACING == 10'240);
    STATIC_REQUIRE(WeatherSystem::RECENTER_DISTANCE == 1'024);
}

TEST_CASE("Weather time slices remain representable at the final world tick",
          "[weather][snapshot][bounds]") {
    constexpr uint64_t slice = WeatherSnapshot::TIME_SLICE_TICKS;
    constexpr uint64_t maximumTick = std::numeric_limits<uint64_t>::max();

    REQUIRE(weatherTimeSliceStart(401, slice) == 400);
    REQUIRE(weatherTimeSliceStart(maximumTick, slice) == maximumTick - slice);
    REQUIRE(weatherTimeSliceStart(maximumTick - 1, slice) == maximumTick - slice);
    REQUIRE(weatherTimeSliceStart(maximumTick, 0) == maximumTick);

    const uint64_t finalStart = weatherTimeSliceStart(maximumTick, slice);
    REQUIRE(finalStart + slice == maximumTick);
}

TEST_CASE("Weather presets parse only the stable capture vocabulary", "[weather][override]") {
    REQUIRE(weatherPresetFromString("clear") == WeatherPreset::CLEAR);
    REQUIRE(weatherPresetFromString("overcast") == WeatherPreset::OVERCAST);
    REQUIRE(weatherPresetFromString("rain") == WeatherPreset::RAIN);
    REQUIRE(weatherPresetFromString("storm") == WeatherPreset::STORM);
    REQUIRE(weatherPresetFromString("snow") == WeatherPreset::SNOW);
    REQUIRE_FALSE(weatherPresetFromString("natural"));
    REQUIRE_FALSE(weatherPresetFromString("RAIN"));
    REQUIRE_FALSE(weatherPresetFromString("drizzle"));
}

TEST_CASE("Coordinate-pure weather is deterministic and continuous", "[weather][determinism]") {
    const worldgen::SurfaceSample climate = climateSample();
    const WeatherSample first = deriveWeatherSample(764891, -12'345.25, 67'890.5, 8'200, climate);
    const WeatherSample repeated =
        deriveWeatherSample(764891, -12'345.25, 67'890.5, 8'200, climate);
    requireAtmosphereEqual(first, repeated);
    REQUIRE(first.cloudOffsetBlocks.x == repeated.cloudOffsetBlocks.x);
    REQUIRE(first.cloudOffsetBlocks.z == repeated.cloudOffsetBlocks.z);
    REQUIRE(first.highCloudOffsetBlocks.x == repeated.highCloudOffsetBlocks.x);
    REQUIRE(first.highCloudOffsetBlocks.z == repeated.highCloudOffsetBlocks.z);

    const WeatherSample adjacent =
        deriveWeatherSample(764891, -12'344.25, 67'890.5, 8'200, climate);
    REQUIRE(std::abs(first.pressureHpa - adjacent.pressureHpa) < 0.1f);
    REQUIRE(std::abs(first.relativeHumidity - adjacent.relativeHumidity) < 0.01f);
    REQUIRE(std::abs(first.cloudCoverage - adjacent.cloudCoverage) < 0.02f);

    const WeatherSample otherSeed =
        deriveWeatherSample(764892, -12'345.25, 67'890.5, 8'200, climate);
    REQUIRE(otherSeed.pressureHpa != first.pressureHpa);
}

TEST_CASE("Weather fronts advect with the static climate wind", "[weather][advection]") {
    const worldgen::SurfaceSample climate = climateSample(14.0, 0.7, -0.4, 0.9);
    const Vec2 velocity = weatherAdvectionVelocity(climate.climate);
    const WeatherSample start = deriveWeatherSample(42, 4'000.0, -9'000.0, 0, climate);
    constexpr uint64_t TEN_SECONDS = 200;
    const WeatherSample advected = deriveWeatherSample(
        42, 4'000.0 + velocity.x * 10.0, -9'000.0 + velocity.y * 10.0, TEN_SECONDS, climate);
    REQUIRE(advected.pressureHpa == Catch::Approx(start.pressureHpa).margin(0.0002f));
}

TEST_CASE("Weather wind is finite and bounded in physical blocks per second",
          "[weather][wind][bounds]") {
    constexpr std::array<double, 4> COORDINATES = {-1'000'000.0, -512.0, 512.0, 1'000'000.0};
    for (double coordinate : COORDINATES) {
        const WeatherSample sample = deriveWeatherSample(
            764891, coordinate, -coordinate * 0.37, 9'876, climateSample(16.0, 0.72, 8.0, -3.0));
        const float speed = sample.windBlocksPerSecond.length();
        REQUIRE(std::isfinite(speed));
        REQUIRE(speed >= 1.0f);
        REQUIRE(speed <= 12.0f);
    }

    const WeatherSample storm =
        deriveWeatherSample(1, 0.0, 0.0, 0, climateSample(), WeatherPreset::STORM);
    REQUIRE(storm.windBlocksPerSecond.length() <= 12.0f);
}

TEST_CASE("Cloud motion retains sub-block continuity at large world ticks",
          "[weather][cloud][precision][continuity]") {
    constexpr uint64_t LARGE_TICK = (1ULL << 40U) + 137U;
    const worldgen::SurfaceSample climate = climateSample();
    const WeatherSample first =
        deriveWeatherSample(764891, 0.0, 0.0, LARGE_TICK, climate, WeatherPreset::STORM);
    const WeatherSample next =
        deriveWeatherSample(764891, 0.0, 0.0, LARGE_TICK + 1U, climate, WeatherPreset::STORM);

    REQUIRE(std::abs(first.cloudOffsetBlocks.x) > 100'000'000'000.0);
    REQUIRE(next.cloudOffsetBlocks.x - first.cloudOffsetBlocks.x ==
            Catch::Approx(0.5).margin(0.001));
    REQUIRE(next.cloudOffsetBlocks.z - first.cloudOffsetBlocks.z ==
            Catch::Approx(0.15).margin(0.001));
    REQUIRE(next.highCloudOffsetBlocks.x - first.highCloudOffsetBlocks.x ==
            Catch::Approx(0.675).margin(0.001));
    REQUIRE(next.highCloudOffsetBlocks.z - first.highCloudOffsetBlocks.z ==
            Catch::Approx(0.2025).margin(0.001));

    const double floatQuantizationError =
        std::abs(first.cloudOffsetBlocks.x -
                 static_cast<double>(static_cast<float>(first.cloudOffsetBlocks.x)));
    REQUIRE(floatQuantizationError > 1.0);
}

TEST_CASE("Natural cloud phase stays spatially continuous at large world ticks",
          "[weather][cloud][precision][continuity]") {
    constexpr uint64_t LARGE_TICK = (1ULL << 40U) + 137U;
    const worldgen::SurfaceSample climate = climateSample();
    const WeatherSample first =
        deriveWeatherSample(764891, 12'000.0, -8'000.0, LARGE_TICK, climate);
    const WeatherSample adjacent =
        deriveWeatherSample(764891, 12'001.0, -8'000.0, LARGE_TICK, climate);

    REQUIRE(std::abs(first.cloudOffsetBlocks.x - adjacent.cloudOffsetBlocks.x) < 0.25);
    REQUIRE(std::abs(first.cloudOffsetBlocks.z - adjacent.cloudOffsetBlocks.z) < 0.25);
    REQUIRE(std::abs(first.highCloudOffsetBlocks.x - adjacent.highCloudOffsetBlocks.x) < 0.25);
    REQUIRE(std::abs(first.highCloudOffsetBlocks.z - adjacent.highCloudOffsetBlocks.z) < 0.25);
}

TEST_CASE("Static climate biases weather temperature humidity and precipitation kind",
          "[weather][climate][precipitation]") {
    worldgen::SurfaceSample coldWet = climateSample(-16.0, 1.0);
    coldWet.climate.annualPrecipitationMm = 3'200.0;
    coldWet.climate.potentialEvapotranspirationMm = 180.0;
    coldWet.climate.aridity = 0.06;
    worldgen::SurfaceSample warmWet = coldWet;
    warmWet.climate.temperatureC = 24.0;
    const WeatherSample cold = deriveWeatherSample(91, 512.0, -768.0, 1'000, coldWet);
    const WeatherSample warm = deriveWeatherSample(91, 512.0, -768.0, 1'000, warmWet);
    REQUIRE(cold.temperatureC < warm.temperatureC - 30.0f);
    REQUIRE(cold.relativeHumidity == Catch::Approx(warm.relativeHumidity));
    REQUIRE(cold.precipitationIntensity > 0.0f);
    REQUIRE(warm.precipitationIntensity > 0.0f);
    REQUIRE(cold.precipitationKind == PrecipitationKind::SNOW);
    REQUIRE(warm.precipitationKind == PrecipitationKind::RAIN);

    const WeatherSample dry =
        deriveWeatherSample(91, 512.0, -768.0, 1'000, climateSample(24.0, 0.12));
    REQUIRE(dry.relativeHumidity < warm.relativeHumidity);
    REQUIRE(dry.cloudCoverage < warm.cloudCoverage);
}

TEST_CASE("Capture presets are stable across seeds coordinates and climates",
          "[weather][override][determinism]") {
    struct Expected {
        WeatherPreset preset;
        PrecipitationKind precipitation;
        CloudType cloud;
        float intensity;
    };
    constexpr std::array<Expected, 5> EXPECTED = {
        Expected{WeatherPreset::CLEAR, PrecipitationKind::NONE, CloudType::CLEAR, 0.0f},
        Expected{WeatherPreset::OVERCAST, PrecipitationKind::NONE, CloudType::STRATUS, 0.0f},
        Expected{WeatherPreset::RAIN, PrecipitationKind::RAIN, CloudType::STRATUS, 0.78f},
        Expected{WeatherPreset::STORM, PrecipitationKind::RAIN, CloudType::CUMULONIMBUS, 1.0f},
        Expected{WeatherPreset::SNOW, PrecipitationKind::SNOW, CloudType::STRATUS, 0.72f},
    };
    for (const Expected& expected : EXPECTED) {
        const WeatherSample first = deriveWeatherSample(
            1, -500'000.0, 300'000.0, 0, climateSample(-30.0, 0.05), expected.preset);
        const WeatherSample second = deriveWeatherSample(999, 800'000.0, -700'000.0, 77'777,
                                                         climateSample(40.0, 1.0), expected.preset);
        requireAtmosphereEqual(first, second);
        REQUIRE(first.precipitationKind == expected.precipitation);
        REQUIRE(first.cloudType == expected.cloud);
        REQUIRE(first.precipitationIntensity == Catch::Approx(expected.intensity));
    }
}

TEST_CASE("Cloud profile helpers keep physical layers finite and bounded", "[weather][cloud]") {
    constexpr std::array<CloudType, 5> TYPES = {
        CloudType::CLEAR,   CloudType::CIRRUS,       CloudType::STRATUS,
        CloudType::CUMULUS, CloudType::CUMULONIMBUS,
    };
    for (CloudType type : TYPES) {
        const CloudLayerBounds bounds = cloudLayerBounds(type, 120.0f, 0.8f);
        REQUIRE(std::isfinite(bounds.baseY));
        REQUIRE(std::isfinite(bounds.topY));
        REQUIRE(bounds.baseY >= 120.0f);
        REQUIRE(bounds.topY > bounds.baseY);
        REQUIRE(bounds.topY <= 500.0f);
        REQUIRE(cloudProfileDensity(type, -0.1f) == 0.0f);
        REQUIRE(cloudProfileDensity(type, 0.0f) == 0.0f);
        REQUIRE(cloudProfileDensity(type, 1.0f) == 0.0f);
        REQUIRE(cloudProfileDensity(type, 1.1f) == 0.0f);
        if (type == CloudType::CLEAR) {
            REQUIRE(cloudProfileDensity(type, 0.5f) == 0.0f);
        } else {
            REQUIRE(cloudProfileDensity(type, 0.5f) > 0.0f);
        }
    }
}

TEST_CASE("Generator v4 cloud layers remain above the tallest terrain",
          "[weather][cloud][v4][height]") {
    constexpr float SUMMIT_Y = 1'407.0F;
    for (const CloudType type : {CloudType::CLEAR, CloudType::CIRRUS, CloudType::STRATUS,
                                 CloudType::CUMULUS, CloudType::CUMULONIMBUS}) {
        const CloudLayerBounds bounds =
            cloudLayerBounds(type, SUMMIT_Y, 0.9F, GENERATOR_V4_PHYSICAL_SCALE);
        REQUIRE(std::isfinite(bounds.baseY));
        REQUIRE(std::isfinite(bounds.topY));
        REQUIRE(bounds.baseY > SUMMIT_Y);
        REQUIRE(bounds.topY > bounds.baseY);
    }

    worldgen::SurfaceSample summitClimate = climateSample(8.0, 0.9);
    summitClimate.terrainHeight = SUMMIT_Y;
    const WeatherSample storm = deriveWeatherSample(
        7, 0.0, 0.0, 0, summitClimate, WeatherPreset::STORM, GENERATOR_V4_PHYSICAL_SCALE);
    REQUIRE(storm.cloudBaseY > SUMMIT_Y);
    REQUIRE(storm.cloudTopY > storm.cloudBaseY);
}

TEST_CASE("Weather worker batches one immutable grid and interpolates it",
          "[weather][worker][snapshot][batch]") {
    struct Probe {
        std::mutex mutex;
        size_t calls = 0;
        int64_t originX = 0;
        int64_t originZ = 0;
        int spacing = 0;
        int edge = 0;
    } probe;
    WeatherSystem system(764891, [&](int64_t originX, int64_t originZ, int spacing, int edge,
                                     std::span<worldgen::SurfaceSample> output) {
        {
            std::lock_guard lock(probe.mutex);
            ++probe.calls;
            probe.originX = originX;
            probe.originZ = originZ;
            probe.spacing = spacing;
            probe.edge = edge;
        }
        fillClimateGrid(originX, originZ, spacing, edge, output);
    });

    const uint64_t request = system.requestSnapshot(-300, 700, 275);
    REQUIRE(system.waitForSnapshot(request, std::chrono::seconds(3)));
    const std::shared_ptr<const WeatherSnapshot> snapshot = system.latestSnapshot();
    REQUIRE(snapshot);
    REQUIRE(snapshot->requestId() == request);
    REQUIRE(snapshot->centerX() == -512);
    REQUIRE(snapshot->centerZ() == 512);
    REQUIRE(snapshot->firstTick() == 200);
    REQUIRE(snapshot->secondTick() == 400);
    REQUIRE(snapshot->timeSlice(0).size() == WeatherSnapshot::GRID_SAMPLE_COUNT);
    REQUIRE(snapshot->timeSlice(1).size() == WeatherSnapshot::GRID_SAMPLE_COUNT);
    REQUIRE(snapshot->covers(snapshot->centerX() - 8'192, snapshot->centerZ()));
    REQUIRE(snapshot->covers(snapshot->centerX() + 8'192, snapshot->centerZ()));

    {
        std::lock_guard lock(probe.mutex);
        REQUIRE(probe.calls == 1);
        REQUIRE(probe.originX == snapshot->centerX() - 10'240);
        REQUIRE(probe.originZ == snapshot->centerZ() - 10'240);
        REQUIRE(probe.spacing == WeatherSnapshot::GRID_SPACING);
        REQUIRE(probe.edge == WeatherSnapshot::GRID_EDGE);
    }

    const WeatherSample exact = snapshot->sample(snapshot->originX(), snapshot->originZ(), 200);
    REQUIRE(exact.pressureHpa == Catch::Approx(snapshot->gridSample(0, 0, 0).pressureHpa));
    const WeatherSample middleTime =
        snapshot->sample(snapshot->originX(), snapshot->originZ(), 300);
    REQUIRE(middleTime.pressureHpa ==
            Catch::Approx(std::lerp(snapshot->gridSample(0, 0, 0).pressureHpa,
                                    snapshot->gridSample(0, 0, 1).pressureHpa, 0.5f)));

    const WeatherSample midpoint =
        snapshot->sample(snapshot->originX() + 128, snapshot->originZ() + 128, 200);
    const float expectedPressure =
        (snapshot->gridSample(0, 0, 0).pressureHpa + snapshot->gridSample(1, 0, 0).pressureHpa +
         snapshot->gridSample(0, 1, 0).pressureHpa + snapshot->gridSample(1, 1, 0).pressureHpa) /
        4.0f;
    REQUIRE(midpoint.pressureHpa == Catch::Approx(expectedPressure));
}

TEST_CASE("Weather recentering retains the previous snapshot until replacement",
          "[weather][worker][recenter]") {
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        size_t calls = 0;
        bool secondEntered = false;
        bool releaseSecond = false;
    } gate;
    WeatherSystem system(42, [&](int64_t originX, int64_t originZ, int spacing, int edge,
                                 std::span<worldgen::SurfaceSample> output) {
        {
            std::unique_lock lock(gate.mutex);
            ++gate.calls;
            if (gate.calls == 2) {
                gate.secondEntered = true;
                gate.condition.notify_all();
                gate.condition.wait(lock, [&] { return gate.releaseSecond; });
            }
        }
        fillClimateGrid(originX, originZ, spacing, edge, output);
    });

    const uint64_t firstRequest = system.requestSnapshot(0, 0, 0);
    REQUIRE(system.waitForSnapshot(firstRequest, std::chrono::seconds(3)));
    REQUIRE(system.requestSnapshot(1'023, 0, 0) == firstRequest);
    {
        std::lock_guard lock(gate.mutex);
        REQUIRE(gate.calls == 1);
    }

    const uint64_t secondRequest = system.requestSnapshot(1'024, 0, 0);
    {
        std::unique_lock lock(gate.mutex);
        REQUIRE(gate.condition.wait_for(lock, std::chrono::seconds(3),
                                        [&] { return gate.secondEntered; }));
    }
    REQUIRE(system.latestSnapshot()->requestId() == firstRequest);
    {
        std::lock_guard lock(gate.mutex);
        gate.releaseSecond = true;
    }
    gate.condition.notify_all();
    REQUIRE(system.waitForSnapshot(secondRequest, std::chrono::seconds(3)));
    REQUIRE(system.latestSnapshot()->centerX() == 1'024);
}

TEST_CASE("Weather worker keeps one latest pending request", "[weather][worker][latest-wins]") {
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        size_t calls = 0;
        bool firstEntered = false;
        bool releaseFirst = false;
        std::vector<int64_t> origins;
    } gate;
    WeatherSystem system(9, [&](int64_t originX, int64_t originZ, int spacing, int edge,
                                std::span<worldgen::SurfaceSample> output) {
        {
            std::unique_lock lock(gate.mutex);
            ++gate.calls;
            gate.origins.push_back(originX);
            if (gate.calls == 1) {
                gate.firstEntered = true;
                gate.condition.notify_all();
                gate.condition.wait(lock, [&] { return gate.releaseFirst; });
            }
        }
        fillClimateGrid(originX, originZ, spacing, edge, output);
    });

    system.requestSnapshot(0, 0, 0);
    {
        std::unique_lock lock(gate.mutex);
        REQUIRE(gate.condition.wait_for(lock, std::chrono::seconds(3),
                                        [&] { return gate.firstEntered; }));
    }
    system.requestSnapshot(2'048, 0, 0);
    const uint64_t latestRequest = system.requestSnapshot(4'096, 0, 0);
    REQUIRE(system.stats().pendingRequests == 1);
    {
        std::lock_guard lock(gate.mutex);
        gate.releaseFirst = true;
    }
    gate.condition.notify_all();
    REQUIRE(system.waitForSnapshot(latestRequest, std::chrono::seconds(3)));

    const std::shared_ptr<const WeatherSnapshot> snapshot = system.latestSnapshot();
    REQUIRE(snapshot->requestId() == latestRequest);
    REQUIRE(snapshot->centerX() == 4'096);
    const WeatherSystemStats stats = system.stats();
    REQUIRE(stats.buildsStarted == 2);
    REQUIRE(stats.snapshotsPublished == 1);
    REQUIRE(stats.staleBuildsDiscarded == 1);
    REQUIRE(stats.pendingRequests == 0);
    {
        std::lock_guard lock(gate.mutex);
        REQUIRE(gate.calls == 2);
        REQUIRE(gate.origins.back() == 4'096 - 10'240);
    }
}

TEST_CASE("Weather worker survives a climate sampler failure and accepts a retry",
          "[weather][worker][failure]") {
    std::atomic<uint32_t> calls = 0;
    WeatherSystem system(91, [&](int64_t originX, int64_t originZ, int spacing, int edge,
                                 std::span<worldgen::SurfaceSample> output) {
        if (calls.fetch_add(1, std::memory_order_relaxed) == 0)
            throw std::runtime_error("synthetic climate failure");
        fillClimateGrid(originX, originZ, spacing, edge, output);
    });

    const uint64_t failedRequest = system.requestSnapshot(0, 0, 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (system.stats().buildsFailed == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    REQUIRE(system.stats().buildsFailed == 1);
    REQUIRE_FALSE(system.latestSnapshot());

    const uint64_t retryRequest = system.requestSnapshot(0, 0, 0);
    REQUIRE(retryRequest > failedRequest);
    REQUIRE(system.waitForSnapshot(retryRequest, std::chrono::seconds(3)));
    REQUIRE(system.stats().snapshotsPublished == 1);
    REQUIRE(calls.load(std::memory_order_relaxed) == 2);
}

TEST_CASE("Weather teardown joins an active climate build before releasing its sampler",
          "[weather][worker][thread][shutdown][regression]") {
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false;
        bool release = false;
    } gate;
    auto system = std::make_unique<WeatherSystem>(
        91, [&](int64_t originX, int64_t originZ, int spacing, int edge,
                std::span<worldgen::SurfaceSample> output) {
            {
                std::unique_lock lock(gate.mutex);
                gate.entered = true;
                gate.condition.notify_all();
                gate.condition.wait(lock, [&] { return gate.release; });
            }
            fillClimateGrid(originX, originZ, spacing, edge, output);
        });
    system->requestSnapshot(0, 0, 0);
    bool entered = false;
    {
        std::unique_lock lock(gate.mutex);
        entered =
            gate.condition.wait_for(lock, std::chrono::seconds(3), [&] { return gate.entered; });
        if (!entered)
            gate.release = true;
    }
    if (!entered)
        gate.condition.notify_all();
    REQUIRE(entered);

    std::future<void> teardown = std::async(std::launch::async, [&] { system.reset(); });
    REQUIRE(teardown.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    {
        std::lock_guard lock(gate.mutex);
        gate.release = true;
    }
    gate.condition.notify_all();
    REQUIRE(teardown.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    REQUIRE_NOTHROW(teardown.get());
}

TEST_CASE("Lightning events are deterministic non-destructive weather data",
          "[weather][lightning][determinism]") {
    WeatherSample storm =
        deriveWeatherSample(1, 0.0, 0.0, 0, climateSample(), WeatherPreset::STORM);
    std::optional<LightningEvent> selected;
    int64_t selectedX = 0;
    int64_t selectedZ = 0;
    constexpr uint64_t BUCKET = 123;
    for (int64_t z = -20; z <= 20 && !selected; ++z) {
        for (int64_t x = -20; x <= 20 && !selected; ++x) {
            selected = lightningEventForCell(764891, x, z, BUCKET, storm);
            selectedX = x;
            selectedZ = z;
        }
    }
    REQUIRE(selected);
    const std::optional<LightningEvent> repeated =
        lightningEventForCell(764891, selectedX, selectedZ, BUCKET, storm);
    REQUIRE(repeated);
    REQUIRE(repeated->id == selected->id);
    REQUIRE(repeated->tick == selected->tick);
    REQUIRE(repeated->x == Catch::Approx(selected->x));
    REQUIRE(repeated->z == Catch::Approx(selected->z));
    REQUIRE(selected->tick >= BUCKET * WeatherSystem::LIGHTNING_BUCKET_TICKS);
    REQUIRE(selected->tick < (BUCKET + 1) * WeatherSystem::LIGHTNING_BUCKET_TICKS);
    REQUIRE(selected->x >= static_cast<double>(selectedX * WeatherSystem::LIGHTNING_CELL_EDGE));
    REQUIRE(selected->x <
            static_cast<double>((selectedX + 1) * WeatherSystem::LIGHTNING_CELL_EDGE));
    REQUIRE(selected->z >= static_cast<double>(selectedZ * WeatherSystem::LIGHTNING_CELL_EDGE));
    REQUIRE(selected->z <
            static_cast<double>((selectedZ + 1) * WeatherSystem::LIGHTNING_CELL_EDGE));

    WeatherSample clear = storm;
    clear.stormPotential = 0.0f;
    clear.cloudType = CloudType::CLEAR;
    REQUIRE_FALSE(lightningEventForCell(764891, selectedX, selectedZ, BUCKET, clear));
}

TEST_CASE("Lightning ticks saturate in the final representable bucket",
          "[weather][lightning][overflow]") {
    constexpr uint64_t MAXIMUM_TICK = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t FINAL_BUCKET = MAXIMUM_TICK / WeatherSystem::LIGHTNING_BUCKET_TICKS;
    constexpr uint64_t FINAL_BUCKET_START = FINAL_BUCKET * WeatherSystem::LIGHTNING_BUCKET_TICKS;
    constexpr uint64_t FINAL_OFFSET = MAXIMUM_TICK - FINAL_BUCKET_START;

    STATIC_REQUIRE(WeatherSystem::lightningTickForBucket(FINAL_BUCKET, 0) == FINAL_BUCKET_START);
    STATIC_REQUIRE(WeatherSystem::lightningTickForBucket(FINAL_BUCKET, FINAL_OFFSET) ==
                   MAXIMUM_TICK);
    STATIC_REQUIRE(WeatherSystem::lightningTickForBucket(FINAL_BUCKET, FINAL_OFFSET + 1) ==
                   MAXIMUM_TICK);
    STATIC_REQUIRE(WeatherSystem::lightningTickForBucket(FINAL_BUCKET + 1, 0) == MAXIMUM_TICK);
}

TEST_CASE("Thunder delay uses physical sound speed", "[weather][lightning][thunder]") {
    LightningEvent event;
    event.x = 343.0;
    event.y = 0.0f;
    event.z = 0.0;
    REQUIRE(thunderDelaySeconds(event, 0.0, 0.0, 0.0) == Catch::Approx(1.0));
    REQUIRE(thunderDelaySeconds(event, 343.0, 0.0, 0.0) == Catch::Approx(0.0));

    event.x = 343.0 / GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock;
    event.y = static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY);
    REQUIRE(thunderDelaySeconds(event, 0.0, GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY, 0.0,
                                GENERATOR_V4_PHYSICAL_SCALE) == Catch::Approx(1.0));
}

TEST_CASE("Lightning queries cap work and do not replay an unloaded backlog",
          "[weather][lightning][worker][bounds]") {
    WeatherSystem system(764891, fillClimateGrid, WeatherPreset::STORM);
    const uint64_t request = system.requestSnapshot(0, 0, 0);
    REQUIRE(system.waitForSnapshot(request, std::chrono::seconds(3)));
    const std::vector<LightningEvent> first = system.lightningEvents(0, 40);
    const std::vector<LightningEvent> repeated = system.lightningEvents(0, 40);
    REQUIRE(first.size() <= WeatherSystem::MAX_LIGHTNING_EVENTS_PER_QUERY);
    REQUIRE(repeated.size() == first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        REQUIRE(repeated[i].id == first[i].id);
        REQUIRE(repeated[i].tick == first[i].tick);
    }

    const std::vector<LightningEvent> afterLongGap = system.lightningEvents(0, 400);
    REQUIRE(afterLongGap.size() <= WeatherSystem::MAX_LIGHTNING_EVENTS_PER_QUERY);
    for (const LightningEvent& event : afterLongGap) {
        REQUIRE(event.tick > 360);
        REQUIRE(event.tick <= 400);
    }
}

TEST_CASE("Lightning discovery caches each weather snapshot bucket",
          "[weather][lightning][cache]") {
    WeatherSystem system(764891, fillClimateGrid, WeatherPreset::STORM);
    const uint64_t firstRequest = system.requestSnapshot(0, 0, 0);
    REQUIRE(system.waitForSnapshot(firstRequest, std::chrono::seconds(10)));

    const std::vector<LightningEvent> first = system.lightningEvents(1, 39);
    const WeatherSystemStats afterFirst = system.stats();
    REQUIRE(afterFirst.lightningDiscoveryBuilds == 1);
    REQUIRE(afterFirst.lightningDiscoveryCacheHits == 0);

    const std::vector<LightningEvent> repeated = system.lightningEvents(1, 39);
    const WeatherSystemStats afterRepeated = system.stats();
    REQUIRE(repeated.size() == first.size());
    REQUIRE(afterRepeated.lightningDiscoveryBuilds == afterFirst.lightningDiscoveryBuilds);
    REQUIRE(afterRepeated.lightningDiscoveryCacheHits ==
            afterFirst.lightningDiscoveryCacheHits + 1);

    system.lightningEvents(40, 79);
    const WeatherSystemStats afterNextBucket = system.stats();
    REQUIRE(afterNextBucket.lightningDiscoveryBuilds == afterRepeated.lightningDiscoveryBuilds + 1);

    const uint64_t recenteredRequest = system.requestSnapshot(2'048, 0, 0);
    REQUIRE(recenteredRequest > firstRequest);
    REQUIRE(system.waitForSnapshot(recenteredRequest, std::chrono::seconds(10)));
    system.lightningEvents(1, 39);
    const WeatherSystemStats afterRecenter = system.stats();
    REQUIRE(afterRecenter.lightningDiscoveryBuilds == afterNextBucket.lightningDiscoveryBuilds + 1);
}

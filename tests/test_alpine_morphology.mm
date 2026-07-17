#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "world/alpine_morphology.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numbers>
#include <numeric>
#include <thread>
#include <vector>

namespace {

worldgen::AlpineTectonicContext strongRange(double x, double z) {
    return {
        .x = x,
        .z = z,
        .uplift = 1.0,
        .rockResistance = 1.2,
        .continentalFraction = 1.0,
    };
}

worldgen::AlpineSurfaceContext glacialValley(double channelDistance) {
    worldgen::AlpineSurfaceContext context;
    static_cast<worldgen::AlpineTectonicContext&>(context) = strongRange(137.0, -219.0);
    context.terrainHeight = 268.0;
    context.temperatureC = -3.0;
    context.annualPrecipitationMm = 1'250.0;
    context.flowX = 0.82;
    context.flowZ = 0.57;
    context.channelDistance = channelDistance;
    context.channelWidth = 12.0;
    context.channelGradient = 0.022;
    context.discharge = 1'200.0;
    context.erosionDepth = 8.0;
    context.footprintWidth = 1;
    return context;
}

double resistanceFor(worldgen::RockType rock) {
    switch (rock) {
        case worldgen::RockType::GRANITE:
            return 0.92;
        case worldgen::RockType::BASALT:
            return 1.12;
        case worldgen::RockType::LIMESTONE:
            return 0.56;
        case worldgen::RockType::SANDSTONE:
            return 0.42;
        case worldgen::RockType::VOLCANIC:
            return 1.20;
    }
    return 0.90;
}

} // namespace

TEST_CASE("Alpine process regimes preserve resistant divides selectively",
          "[worldgen][alpine][erosion][uplift][geology][climate]") {
    const worldgen::AlpineErosionResponse divide = worldgen::alpineErosionResponse({
        .uplift = 1.0,
        .rockResistance = 1.25,
        .terrainHeight = 280.0,
        .temperatureC = -1.0,
        .annualPrecipitationMm = 900.0,
        .drainageConvergence = 0.0,
        .slope = 0.92,
    });
    const worldgen::AlpineErosionResponse channel = worldgen::alpineErosionResponse({
        .uplift = 1.0,
        .rockResistance = 1.25,
        .terrainHeight = 280.0,
        .temperatureC = -1.0,
        .annualPrecipitationMm = 900.0,
        .drainageConvergence = 1.0,
        .slope = 0.92,
    });
    const worldgen::AlpineErosionResponse weakSoil = worldgen::alpineErosionResponse({
        .uplift = 0.12,
        .rockResistance = 0.30,
        .terrainHeight = 84.0,
        .temperatureC = 9.0,
        .annualPrecipitationMm = 900.0,
        .drainageConvergence = 0.72,
        .slope = 0.72,
    });
    const worldgen::AlpineErosionResponse warmRange = worldgen::alpineErosionResponse({
        .uplift = 1.0,
        .rockResistance = 1.25,
        .terrainHeight = 280.0,
        .temperatureC = 11.0,
        .annualPrecipitationMm = 900.0,
        .drainageConvergence = 0.0,
        .slope = 0.92,
    });
    const worldgen::AlpineErosionResponse coldLowland = worldgen::alpineErosionResponse({
        .uplift = 1.0,
        .rockResistance = 1.25,
        .terrainHeight = 82.0,
        .temperatureC = -1.0,
        .annualPrecipitationMm = 900.0,
        .drainageConvergence = 0.0,
        .slope = 0.92,
    });

    REQUIRE(divide.ridgePreservation > 0.80);
    REQUIRE(divide.thermalRelaxationScale < weakSoil.thermalRelaxationScale * 0.35);
    REQUIRE(channel.streamIncisionScale > divide.streamIncisionScale + 0.40);
    REQUIRE(channel.ridgePreservation < divide.ridgePreservation * 0.30);
    REQUIRE(divide.glacialCompetition > warmRange.glacialCompetition + 0.60);
    REQUIRE(divide.glacialCompetition > coldLowland.glacialCompetition + 0.60);
    REQUIRE(divide.periglacialWeathering > warmRange.periglacialWeathering + 0.60);
    REQUIRE(divide.criticalSlope > weakSoil.criticalSlope + 0.35);
}

TEST_CASE("Weak uplift leaves ordinary terrain unchanged at every footprint",
          "[worldgen][alpine][uplift][lod][invariance]") {
    worldgen::AlpineMorphologySampler sampler(42);
    constexpr std::array<int, 6> FOOTPRINTS = {1, 2, 4, 8, 16, 32};
    for (const int footprint : FOOTPRINTS) {
        worldgen::AlpineSurfaceContext context;
        context.x = -81'896.0;
        context.z = 126'960.0;
        context.uplift = 0.15;
        context.rockResistance = 1.25;
        context.continentalFraction = 1.0;
        context.terrainHeight = 146.0;
        context.temperatureC = 11.0;
        context.annualPrecipitationMm = 900.0;
        context.channelDistance = 500.0;
        context.footprintWidth = footprint;
        const auto tectonic = sampler.sampleTectonic(context);
        const auto surface = sampler.sampleSurface(context, tectonic);
        CAPTURE(footprint);
        REQUIRE(tectonic.elevationOffset == 0.0);
        REQUIRE(surface.ridgeDetail == 0.0);
        REQUIRE(surface.valleyCarve == 0.0);
        REQUIRE(surface.cirqueCarve == 0.0);
        REQUIRE(surface.elevationOffset == 0.0);
    }
}

TEST_CASE("Alpine morphology is identical across worker schedules",
          "[worldgen][alpine][determinism][concurrency]") {
    worldgen::AlpineMorphologySampler sampler(764891);
    constexpr int SAMPLE_COUNT = 4'096;
    std::array<double, SAMPLE_COUNT> reference{};
    std::array<double, SAMPLE_COUNT> concurrent{};
    for (int index = 0; index < SAMPLE_COUNT; ++index) {
        auto context = strongRange((index % 64) * 11.0 - 352.0, (index / 64) * 13.0 - 416.0);
        reference[static_cast<size_t>(index)] = sampler.sampleTectonic(context).elevationOffset;
    }

    std::array<std::thread, 4> workers;
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            for (int index = SAMPLE_COUNT - 1 - static_cast<int>(worker); index >= 0;
                 index -= static_cast<int>(workers.size())) {
                auto context =
                    strongRange((index % 64) * 11.0 - 352.0, (index / 64) * 13.0 - 416.0);
                concurrent[static_cast<size_t>(index)] =
                    sampler.sampleTectonic(context).elevationOffset;
            }
        });
    }
    for (std::thread& worker : workers)
        worker.join();
    REQUIRE(concurrent == reference);
}

TEST_CASE("Warped tectonic ridges stay connected and produce bounded sharp horns",
          "[worldgen][alpine][mountain][ridge][horn][continuity][determinism]") {
    worldgen::AlpineMorphologySampler sampler(42);
    constexpr int EDGE = 129;
    constexpr int SPACING = 16;
    constexpr int ORIGIN = -(EDGE / 2) * SPACING;
    std::vector<double> ridge(static_cast<size_t>(EDGE * EDGE));
    std::vector<uint8_t> active(ridge.size());
    double maximumHorn = 0.0;
    double maximumOffset = 0.0;
    int peakX = 0;
    int peakZ = 0;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const auto sample =
                sampler.sampleTectonic(strongRange(ORIGIN + x * SPACING, ORIGIN + z * SPACING));
            const size_t index = static_cast<size_t>(z * EDGE + x);
            ridge[index] = sample.ridgeStrength;
            active[index] = sample.ridgeStrength >= 0.58;
            REQUIRE(std::isfinite(sample.elevationOffset));
            REQUIRE(sample.elevationOffset >= 0.0);
            REQUIRE(sample.elevationOffset <= 80.0);
            if (sample.hornStrength > maximumHorn) {
                maximumHorn = sample.hornStrength;
                maximumOffset = sample.elevationOffset;
                peakX = ORIGIN + x * SPACING;
                peakZ = ORIGIN + z * SPACING;
            }
            REQUIRE(sampler.sampleTectonic(strongRange(ORIGIN + x * SPACING, ORIGIN + z * SPACING))
                        .elevationOffset == sample.elevationOffset);
        }
    }

    std::vector<uint8_t> visited(active.size());
    size_t largestComponent = 0;
    constexpr std::array<int, 8> DX = {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr std::array<int, 8> DZ = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const size_t start = static_cast<size_t>(z * EDGE + x);
            if (!active[start] || visited[start])
                continue;
            std::deque<size_t> pending{start};
            visited[start] = true;
            size_t component = 0;
            while (!pending.empty()) {
                const size_t index = pending.front();
                pending.pop_front();
                ++component;
                const int cellX = static_cast<int>(index % EDGE);
                const int cellZ = static_cast<int>(index / EDGE);
                for (size_t direction = 0; direction < DX.size(); ++direction) {
                    const int neighborX = cellX + DX[direction];
                    const int neighborZ = cellZ + DZ[direction];
                    if (neighborX < 0 || neighborX >= EDGE || neighborZ < 0 || neighborZ >= EDGE) {
                        continue;
                    }
                    const size_t neighbor = static_cast<size_t>(neighborZ * EDGE + neighborX);
                    if (!active[neighbor] || visited[neighbor])
                        continue;
                    visited[neighbor] = true;
                    pending.push_back(neighbor);
                }
            }
            largestComponent = std::max(largestComponent, component);
        }
    }

    size_t lowerDirections = 0;
    constexpr std::array<double, 8> ANGLES = {
        0.0,
        0.7853981633974483,
        1.5707963267948966,
        2.356194490192345,
        3.141592653589793,
        3.9269908169872414,
        4.71238898038469,
        5.497787143782138,
    };
    for (double angle : ANGLES) {
        const double x = peakX + std::cos(angle) * 112.0;
        const double z = peakZ + std::sin(angle) * 112.0;
        const double shoulder = sampler.sampleTectonic(strongRange(x, z)).elevationOffset;
        lowerDirections += shoulder + 5.0 < maximumOffset;
    }
    INFO("maximum horn " << maximumHorn << ", peak offset " << maximumOffset << ", connected cells "
                         << largestComponent << ", lower directions " << lowerDirections);
    REQUIRE(maximumHorn > 0.50);
    REQUIRE(largestComponent >= 80);
    REQUIRE(lowerDirections >= 4);
    for (int neighborZ = -1; neighborZ <= 1; ++neighborZ) {
        for (int neighborX = -1; neighborX <= 1; ++neighborX) {
            if (neighborX == 0 && neighborZ == 0)
                continue;
            const double neighbor =
                sampler.sampleTectonic(strongRange(peakX + neighborX, peakZ + neighborZ))
                    .elevationOffset;
            REQUIRE(std::abs(neighbor - maximumOffset) < 3.0);
        }
    }
    double windowMaximum = -1.0;
    std::array<double, 65 * 65> peakWindow{};
    size_t windowIndex = 0;
    for (int offsetZ = -32; offsetZ <= 32; ++offsetZ) {
        for (int offsetX = -32; offsetX <= 32; ++offsetX) {
            const double value =
                sampler.sampleTectonic(strongRange(peakX + offsetX, peakZ + offsetZ))
                    .elevationOffset;
            peakWindow[windowIndex++] = value;
            windowMaximum = std::max(windowMaximum, value);
        }
    }
    const size_t plateauSamples =
        static_cast<size_t>(std::count_if(peakWindow.begin(), peakWindow.end(), [&](double value) {
            return std::abs(value - windowMaximum) < 1.0e-5;
        }));
    const auto [windowMinimum, ignoredMaximum] =
        std::minmax_element(peakWindow.begin(), peakWindow.end());
    static_cast<void>(ignoredMaximum);
    REQUIRE(plateauSamples <= 4);
    REQUIRE(windowMaximum > *windowMinimum + 0.5);

    auto weakRock = strongRange(peakX, peakZ);
    weakRock.rockResistance = 0.30;
    auto weakUplift = strongRange(peakX, peakZ);
    weakUplift.uplift = 0.15;
    REQUIRE(sampler.sampleTectonic(weakRock).elevationOffset <= maximumOffset * 0.24);
    REQUIRE(sampler.sampleTectonic(weakUplift).elevationOffset == 0.0);
}

TEST_CASE("Glacial troughs retain broad floors steep walls and talus shoulders",
          "[worldgen][alpine][glacier][valley][talus][cross-section]") {
    worldgen::AlpineMorphologySampler sampler(764891);
    const auto center = sampler.sampleSurface(glacialValley(0.0));
    const auto inner = sampler.sampleSurface(glacialValley(18.0));
    const auto wall = sampler.sampleSurface(glacialValley(46.0));
    const auto shoulder = sampler.sampleSurface(glacialValley(72.0));
    const auto outside = sampler.sampleSurface(glacialValley(120.0));

    REQUIRE(center.glacialInfluence > 0.75);
    REQUIRE(center.valleyCarve > 16.0);
    REQUIRE(inner.valleyCarve == Catch::Approx(center.valleyCarve).margin(0.20));
    REQUIRE(wall.valleyCarve < inner.valleyCarve * 0.72);
    REQUIRE(shoulder.valleyCarve < wall.valleyCarve * 0.45);
    REQUIRE(outside.valleyCarve == Catch::Approx(0.0).margin(1.0e-8));
    REQUIRE(wall.talusDeposit > center.talusDeposit + 0.5);
    REQUIRE(wall.talusDeposit > outside.talusDeposit + 0.5);
}

TEST_CASE("Alpine carving requires compatible climate and geology",
          "[worldgen][alpine][glacier][cirque][climate][geology]") {
    worldgen::AlpineMorphologySampler sampler(42);
    double strongestColdCirque = 0.0;
    double matchingWarmCirque = 0.0;
    double matchingWeakCirque = 0.0;
    worldgen::AlpineSurfaceContext strongestContext;
    for (int z = -640; z <= 640; z += 32) {
        for (int x = -640; x <= 640; x += 32) {
            worldgen::AlpineSurfaceContext cold = glacialValley(500.0);
            static_cast<worldgen::AlpineTectonicContext&>(cold) = strongRange(x, z);
            const auto coldSample = sampler.sampleSurface(cold);
            if (coldSample.cirqueInfluence <= strongestColdCirque)
                continue;
            strongestColdCirque = coldSample.cirqueInfluence;
            strongestContext = cold;

            worldgen::AlpineSurfaceContext warm = cold;
            warm.temperatureC = 13.0;
            matchingWarmCirque = sampler.sampleSurface(warm).cirqueInfluence;

            worldgen::AlpineSurfaceContext weak = cold;
            weak.uplift = 0.04;
            weak.rockResistance = 0.30;
            matchingWeakCirque = sampler.sampleSurface(weak).cirqueInfluence;
        }
    }

    INFO("strongest cold cirque " << strongestColdCirque << ", warm " << matchingWarmCirque
                                  << ", weak " << matchingWeakCirque);
    REQUIRE(strongestColdCirque > 0.20);
    REQUIRE(matchingWarmCirque < strongestColdCirque * 0.05);
    REQUIRE(matchingWeakCirque < strongestColdCirque * 0.05);
    strongestContext.flowX = -0.31;
    strongestContext.flowZ = 0.95;
    REQUIRE(sampler.sampleSurface(strongestContext).cirqueInfluence == strongestColdCirque);
}

TEST_CASE("Alpine ridges avoid storage-axis orientation bias",
          "[worldgen][alpine][ridge][orientation][artifact]") {
    worldgen::AlpineMorphologySampler sampler(91);
    constexpr int EDGE = 96;
    constexpr int SPACING = 20;
    constexpr double DELTA = 5.0;
    std::array<size_t, 8> bins{};
    size_t classified = 0;
    for (int z = 1; z < EDGE - 1; ++z) {
        for (int x = 1; x < EDGE - 1; ++x) {
            const double worldX = (x - EDGE / 2) * SPACING;
            const double worldZ = (z - EDGE / 2) * SPACING;
            const double gx =
                sampler.sampleTectonic(strongRange(worldX + DELTA, worldZ)).ridgeStrength -
                sampler.sampleTectonic(strongRange(worldX - DELTA, worldZ)).ridgeStrength;
            const double gz =
                sampler.sampleTectonic(strongRange(worldX, worldZ + DELTA)).ridgeStrength -
                sampler.sampleTectonic(strongRange(worldX, worldZ - DELTA)).ridgeStrength;
            if (std::hypot(gx, gz) < 0.015)
                continue;
            double angle = std::atan2(gz, gx);
            if (angle < 0.0)
                angle += std::numbers::pi;
            if (angle >= std::numbers::pi)
                angle -= std::numbers::pi;
            const size_t bin = std::min<size_t>(
                bins.size() - 1,
                static_cast<size_t>(angle / std::numbers::pi * static_cast<double>(bins.size())));
            ++bins[bin];
            ++classified;
        }
    }

    std::array<size_t, 4> cardinalAndDiagonal = {bins[0] + bins[7], bins[2] + bins[1],
                                                 bins[4] + bins[3], bins[6] + bins[5]};
    const auto [minimum, maximum] =
        std::minmax_element(cardinalAndDiagonal.begin(), cardinalAndDiagonal.end());
    INFO("orientation bins " << bins[0] << ' ' << bins[1] << ' ' << bins[2] << ' ' << bins[3] << ' '
                             << bins[4] << ' ' << bins[5] << ' ' << bins[6] << ' ' << bins[7]);
    REQUIRE(classified > 500);
    REQUIRE(*minimum > 0);
    REQUIRE(static_cast<double>(*maximum) / static_cast<double>(*minimum) < 1.50);
}

TEST_CASE("Alpine detail filters by footprint under fixed landform ownership",
          "[worldgen][alpine][lod][footprint][determinism][performance]") {
    worldgen::AlpineMorphologySampler sampler(42);
    constexpr std::array<int, 6> FOOTPRINTS = {1, 2, 4, 8, 16, 32};
    std::array<double, FOOTPRINTS.size()> energy{};
    for (int z = -256; z <= 256; z += 16) {
        for (int x = -256; x <= 256; x += 16) {
            for (size_t index = 0; index < FOOTPRINTS.size(); ++index) {
                worldgen::AlpineSurfaceContext context = glacialValley(500.0);
                static_cast<worldgen::AlpineTectonicContext&>(context) = strongRange(x, z);
                context.temperatureC = 12.0;
                context.footprintWidth = FOOTPRINTS[index];
                const auto tectonic = sampler.sampleTectonic(context);
                const auto sample = sampler.sampleSurface(context, tectonic);
                energy[index] += sample.ridgeDetail * sample.ridgeDetail;
                REQUIRE(sample.ridgeStrength == sampler.sampleSurface(context).ridgeStrength);
                worldgen::AlpineSurfaceContext basinContext = context;
                basinContext.footprintWidth = 16;
                const double basinDetail = sampler.sampleRidgeDetail(basinContext, tectonic);
                const double refinement =
                    sampler.sampleRidgeDetail(context, tectonic) - basinDetail;
                REQUIRE(basinDetail + refinement == Catch::Approx(sample.ridgeDetail));
                REQUIRE(std::isfinite(sample.elevationOffset));
            }
        }
    }
    for (size_t index = 1; index < energy.size(); ++index) {
        INFO("footprint " << FOOTPRINTS[index - 1] << " energy " << energy[index - 1]
                          << ", footprint " << FOOTPRINTS[index] << " energy " << energy[index]);
        REQUIRE(energy[index] <= energy[index - 1] * 1.02);
    }
    STATIC_REQUIRE(worldgen::ALPINE_CIRQUE_CANDIDATE_LIMIT == 9);
    STATIC_REQUIRE(worldgen::ALPINE_RIDGE_DETAIL_BAND_COUNT == 4);
}

TEST_CASE("Production alpine terrain routes over taller filtered peaks",
          "[worldgen][alpine][integration][mountain][lod][determinism]") {
    constexpr double PEAK_X = -81'896.0;
    constexpr double PEAK_Z = 126'960.0;
    worldgen::MacroGenerationSampler sampler(42);

    const worldgen::GeologySample geology = sampler.sampleGeology(PEAK_X, PEAK_Z);
    const double resistance =
        resistanceFor(geology.lithology.primary) * (1.0 - geology.lithology.transition) +
        resistanceFor(geology.lithology.secondary) * geology.lithology.transition;
    worldgen::AlpineMorphologySampler alpine(42);
    const worldgen::AlpineTectonicSample tectonic = alpine.sampleTectonic({
        .x = PEAK_X,
        .z = PEAK_Z,
        .uplift = geology.uplift,
        .rockResistance = resistance,
        .continentalFraction = geology.continentalFraction,
    });
    const double preliminary = sampler.preliminaryElevation(PEAK_X, PEAK_Z);
    const worldgen::SurfaceSample exact =
        sampler.sampleSurface(PEAK_X, PEAK_Z, worldgen::SurfaceFootprint::BLOCK_1);
    const worldgen::SurfaceSample coarse =
        sampler.sampleSurface(PEAK_X, PEAK_Z, worldgen::SurfaceFootprint::BLOCK_16);

    INFO("preliminary " << preliminary << ", exact " << exact.terrainHeight << ", coarse "
                        << coarse.terrainHeight << ", alpine broad " << tectonic.elevationOffset
                        << ", uplift " << geology.uplift << ", resistance " << resistance);
    REQUIRE(geology.uplift > 0.90);
    REQUIRE(tectonic.elevationOffset > 4.0);
    REQUIRE(exact.terrainHeight >= 380.0);
    REQUIRE(exact.terrainHeight < 450.0);
    REQUIRE(coarse.terrainHeight >= 370.0);
    REQUIRE(std::abs(exact.terrainHeight - coarse.terrainHeight) < 16.0);
    REQUIRE_FALSE(exact.hydrology.ocean);
    REQUIRE_FALSE(exact.hydrology.lake);
    REQUIRE_FALSE(exact.hydrology.river);

    const uint64_t waterBody = exact.hydrology.waterBodyId;
    const double waterSurface = exact.waterSurface;
    sampler.clearMacroControlCache();
    sampler.clearBasinCache();
    const worldgen::SurfaceSample rebuilt =
        sampler.sampleSurface(PEAK_X, PEAK_Z, worldgen::SurfaceFootprint::BLOCK_1);
    REQUIRE(rebuilt.terrainHeight == exact.terrainHeight);
    REQUIRE(rebuilt.hydrology.waterBodyId == waterBody);
    REQUIRE(rebuilt.waterSurface == waterSurface);
}

TEST_CASE("Continuous bathymetry forms deep abyssal floors without moving water ownership",
          "[worldgen][alpine][integration][ocean][bathymetry][lod][determinism]") {
    constexpr double OCEAN_X = -518'996.0;
    constexpr double OCEAN_Z = -385'073.0;
    worldgen::MacroGenerationSampler sampler(42);
    constexpr std::array<worldgen::SurfaceFootprint, 5> FOOTPRINTS = {
        worldgen::SurfaceFootprint::BLOCK_1,  worldgen::SurfaceFootprint::BLOCK_2,
        worldgen::SurfaceFootprint::BLOCK_4,  worldgen::SurfaceFootprint::BLOCK_8,
        worldgen::SurfaceFootprint::BLOCK_16,
    };
    std::array<worldgen::SurfaceSample, FOOTPRINTS.size()> samples;
    for (size_t index = FOOTPRINTS.size(); index-- > 0;) {
        samples[index] = sampler.sampleSurface(OCEAN_X, OCEAN_Z, FOOTPRINTS[index]);
    }

    const worldgen::SurfaceSample& exact = samples.front();
    INFO("abyss floor " << exact.terrainHeight << ", depth "
                        << exact.waterSurface - exact.terrainHeight << ", preliminary "
                        << sampler.preliminaryElevation(OCEAN_X, OCEAN_Z));
    REQUIRE(exact.hydrology.ocean);
    REQUIRE(exact.waterSurface == SEA_LEVEL);
    REQUIRE(exact.terrainHeight <= -55.0);
    REQUIRE(exact.terrainHeight >= -105.0);
    REQUIRE(exact.waterSurface - exact.terrainHeight >= 115.0);
    for (const worldgen::SurfaceSample& sample : samples) {
        REQUIRE(std::isfinite(sample.terrainHeight));
        REQUIRE(sample.hydrology.ocean);
        REQUIRE(sample.hydrology.waterBodyId == exact.hydrology.waterBodyId);
        REQUIRE(sample.waterSurface == exact.waterSurface);
        REQUIRE(std::abs(sample.terrainHeight - exact.terrainHeight) < 10.0);
    }

    sampler.clearMacroControlCache();
    sampler.clearBasinCache();
    for (size_t index = 0; index < FOOTPRINTS.size(); ++index) {
        const worldgen::SurfaceSample rebuilt =
            sampler.sampleSurface(OCEAN_X, OCEAN_Z, FOOTPRINTS[index]);
        REQUIRE(rebuilt.terrainHeight == samples[index].terrainHeight);
        REQUIRE(rebuilt.hydrology.ocean == samples[index].hydrology.ocean);
        REQUIRE(rebuilt.hydrology.waterBodyId == samples[index].hydrology.waterBodyId);
    }
}

TEST_CASE("Production alpine fields stay gated bounded and fast",
          "[worldgen][alpine][integration][bounds][performance][geology]") {
    worldgen::MacroGenerationSampler sampler(42);
    worldgen::AlpineMorphologySampler alpine(42);
    constexpr int EDGE = 33;
    constexpr int SPACING = 32;
    constexpr double CENTER_X = -81'896.0;
    constexpr double CENTER_Z = 126'960.0;
    std::array<double, EDGE * EDGE> elevations{};

    const auto start = std::chrono::steady_clock::now();
    size_t index = 0;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const double sampleX = CENTER_X + (x - EDGE / 2) * SPACING;
            const double sampleZ = CENTER_Z + (z - EDGE / 2) * SPACING;
            elevations[index++] = sampler.preliminaryElevation(sampleX, sampleZ);
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double microsecondsPerSample =
        std::chrono::duration<double, std::micro>(elapsed).count() / elevations.size();
    INFO("production preliminary elevation " << microsecondsPerSample << " us/sample");
    REQUIRE(microsecondsPerSample < 100.0);

    const auto [minimum, maximum] = std::minmax_element(elevations.begin(), elevations.end());
    REQUIRE(std::isfinite(*minimum));
    REQUIRE(std::isfinite(*maximum));
    REQUIRE(*minimum > -112.0);
    REQUIRE(*maximum < 479.0);
    const size_t plateauCount = static_cast<size_t>(
        std::count_if(elevations.begin(), elevations.end(),
                      [&](double elevation) { return std::abs(elevation - *maximum) < 1.0e-9; }));
    REQUIRE(plateauCount <= 4);

    size_t weakSamples = 0;
    for (int z = -8; z <= 8; ++z) {
        for (int x = -8; x <= 8; ++x) {
            const double sampleX = x * 384.0;
            const double sampleZ = z * 384.0;
            const worldgen::GeologySample geology = sampler.sampleGeology(sampleX, sampleZ);
            if (geology.uplift >= 0.20)
                continue;
            const double resistance =
                resistanceFor(geology.lithology.primary) * (1.0 - geology.lithology.transition) +
                resistanceFor(geology.lithology.secondary) * geology.lithology.transition;
            const worldgen::AlpineTectonicSample tectonic = alpine.sampleTectonic({
                .x = sampleX,
                .z = sampleZ,
                .uplift = geology.uplift,
                .rockResistance = resistance,
                .continentalFraction = geology.continentalFraction,
            });
            REQUIRE(tectonic.elevationOffset == 0.0);
            REQUIRE(std::isfinite(sampler.preliminaryElevation(sampleX, sampleZ)));
            ++weakSamples;
        }
    }
    REQUIRE(weakSamples >= 64);
}

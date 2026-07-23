#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_helpers.hpp"
#include "world/artifact_analysis.hpp"
#include "world/chunk_generator.hpp"
#include "world/learned_terrain.hpp"
#include "world/native_hydrology.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace artifact = worldgen::artifact_analysis;

// Three independent anchors and two seeds still leave at least 780 former
// derivative samples and 6,240 paired controls at every required stride.
// Each exact derivative is compared with the coordinate-pure direct authority
// at the same points, so genuine terrain variation cannot masquerade as an
// old storage-line artifact. Keeping this bounded avoids materializing
// hundreds of unrelated 2,048-block fake hydrology pages merely to repeat the
// same statistical assertion.
constexpr int64_t CONTINUOUS_HALF_WINDOW = 64;
constexpr int64_t CATEGORICAL_HALF_WINDOW = 64;
constexpr std::array<uint64_t, 2> CONTINUOUS_SEEDS = {42, 764'891};
constexpr std::array<uint64_t, 2> CATEGORICAL_SEEDS = {42, 764'891};
constexpr std::array<int64_t, 9> CONTINUITY_GRID_SPACINGS = {
    8, 16, 32, 64, 256, 768, 1'024, 2'048, 8'192,
};
constexpr std::array<int64_t, 6> ABSOLUTE_AUTHORITY_GRID_SPACINGS = {
    16, 256, 768, 1'024, 2'048, 8'192,
};
constexpr std::array<int64_t, ABSOLUTE_AUTHORITY_GRID_SPACINGS.size()>
    ABSOLUTE_SHORE_ALONG_CENTERS = {-1'184, -1'120, -1'056, -992, -928, -864};
// Preserve the learned four-block native-pixel phase without allowing a
// nearby control to land on an eight-block former lattice line.
constexpr std::array<int64_t, 8> CONTINUOUS_NEARBY_OFFSETS = {
    -28, -20, -12, -4, 4, 12, 20, 28,
};
static_assert([] {
    for (const int64_t offset : CONTINUOUS_NEARBY_OFFSETS) {
        if (offset % 4 != 0 || offset % 8 == 0)
            return false;
    }
    return true;
}());

struct WindowAnchor {
    int64_t line;
    int64_t along;
};

constexpr std::array<WindowAnchor, 6> WINDOW_ANCHORS = {
    WindowAnchor{12'345, -9'327},    WindowAnchor{-54'321, 48'123},
    WindowAnchor{90'117, 32'761},    WindowAnchor{-130'019, -77'003},
    WindowAnchor{249'999, -180'011}, WindowAnchor{-333'337, 222'229},
};

constexpr std::array<WindowAnchor, 3> CONTINUOUS_WINDOW_ANCHORS = {
    WINDOW_ANCHORS[0],
    WINDOW_ANCHORS[2],
    WINDOW_ANCHORS[5],
};

struct ShoreWindowFixture {
    int64_t normal = 0;
    int64_t transitionNormal = 0;
    int64_t along = 0;
    bool vertical = true;
};

constexpr int64_t SHORE_TRANSITION_MIN_X = -512;
constexpr int64_t SHORE_TRANSITION_MAX_X = 512;
constexpr int64_t SHORE_TRANSITION_MIN_Z = -1'536;
constexpr int64_t SHORE_TRANSITION_MAX_Z = -512;
constexpr int64_t SHORE_MAX_CONTROL_DISTANCE = 96;

struct SamplePos {
    int64_t x = 0;
    int64_t z = 0;

    bool operator==(const SamplePos&) const = default;
};

struct SamplePosHash {
    size_t operator()(const SamplePos& position) const noexcept {
        size_t seed = world_coord::mix(0, static_cast<uint64_t>(position.x));
        return world_coord::mix(seed, static_cast<uint64_t>(position.z));
    }
};

struct CachedProbe {
    worldgen::SurfaceSample surface;
    std::optional<BlockType> material;
    std::optional<BlockType> emittedGround;
    std::optional<BlockType> emittedTop;
};

uint8_t waterClass(const worldgen::SurfaceSample& sample);

struct FinalSurfaceDigest {
    double terrainHeight = 0.0;
    double waterSurface = 0.0;
    double hydrologySurface = 0.0;
    double hydrologyWaterSurface = 0.0;
    double shoreDistance = 0.0;
    double discharge = 0.0;
    double sediment = 0.0;
    double channelDistance = 0.0;
    double channelWidth = 0.0;
    double channelDepth = 0.0;
    double precipitation = 0.0;
    double temperature = 0.0;
    double humidity = 0.0;
    double evapotranspiration = 0.0;
    double aridity = 0.0;
    double soilMoisture = 0.0;
    double soilFertility = 0.0;
    double soilDrainage = 0.0;
    double waterTable = 0.0;
    double slope = 0.0;
    double lithologyTransition = 0.0;
    double lithologyContact = 0.0;
    uint64_t plateId = 0;
    worldgen::WaterBodyId waterBodyId = worldgen::NO_WATER_BODY;
    worldgen::RockType primaryRock = worldgen::RockType::GRANITE;
    worldgen::RockType secondaryRock = worldgen::RockType::GRANITE;
    Biome primaryBiome = Biome::PLAINS;
    Biome secondaryBiome = Biome::PLAINS;
    double biomeTransition = 0.0;
    uint8_t water = 0;
    BlockType material = BlockType::AIR;
    std::array<float, static_cast<size_t>(Biome::COUNT)> suitability{};

    bool operator==(const FinalSurfaceDigest&) const = default;
};

FinalSurfaceDigest digest(const worldgen::SurfaceSample& sample,
                          BlockType material = BlockType::AIR) {
    return {
        sample.terrainHeight,
        sample.waterSurface,
        sample.hydrology.surfaceElevation,
        sample.hydrology.waterSurface,
        sample.hydrology.lakeShoreDistance,
        sample.hydrology.discharge,
        sample.hydrology.sediment,
        sample.hydrology.channelDistance,
        sample.hydrology.channelWidth,
        sample.hydrology.channelDepth,
        sample.climate.annualPrecipitationMm,
        sample.climate.temperatureC,
        sample.climate.relativeHumidity,
        sample.climate.potentialEvapotranspirationMm,
        sample.climate.aridity,
        sample.soil.moisture,
        sample.soil.fertility,
        sample.soil.drainage,
        sample.soil.waterTable,
        sample.slope,
        sample.geology.lithology.transition,
        sample.geology.lithology.contactDistance,
        sample.geology.plateId,
        sample.hydrology.waterBodyId,
        sample.geology.lithology.primary,
        sample.geology.lithology.secondary,
        sample.biome.primary,
        sample.biome.secondary,
        sample.biome.transition,
        waterClass(sample),
        material,
        sample.suitability.scores,
    };
}

const char* firstDigestDifference(const FinalSurfaceDigest& first,
                                  const FinalSurfaceDigest& second) {
#define RYCRAFT_COMPARE_DIGEST_FIELD(field)                                                        \
    if (first.field != second.field)                                                               \
    return #field
    RYCRAFT_COMPARE_DIGEST_FIELD(terrainHeight);
    RYCRAFT_COMPARE_DIGEST_FIELD(waterSurface);
    RYCRAFT_COMPARE_DIGEST_FIELD(hydrologySurface);
    RYCRAFT_COMPARE_DIGEST_FIELD(hydrologyWaterSurface);
    RYCRAFT_COMPARE_DIGEST_FIELD(shoreDistance);
    RYCRAFT_COMPARE_DIGEST_FIELD(discharge);
    RYCRAFT_COMPARE_DIGEST_FIELD(sediment);
    RYCRAFT_COMPARE_DIGEST_FIELD(channelDistance);
    RYCRAFT_COMPARE_DIGEST_FIELD(channelWidth);
    RYCRAFT_COMPARE_DIGEST_FIELD(channelDepth);
    RYCRAFT_COMPARE_DIGEST_FIELD(precipitation);
    RYCRAFT_COMPARE_DIGEST_FIELD(temperature);
    RYCRAFT_COMPARE_DIGEST_FIELD(humidity);
    RYCRAFT_COMPARE_DIGEST_FIELD(evapotranspiration);
    RYCRAFT_COMPARE_DIGEST_FIELD(aridity);
    RYCRAFT_COMPARE_DIGEST_FIELD(soilMoisture);
    RYCRAFT_COMPARE_DIGEST_FIELD(soilFertility);
    RYCRAFT_COMPARE_DIGEST_FIELD(soilDrainage);
    RYCRAFT_COMPARE_DIGEST_FIELD(waterTable);
    RYCRAFT_COMPARE_DIGEST_FIELD(slope);
    RYCRAFT_COMPARE_DIGEST_FIELD(lithologyTransition);
    RYCRAFT_COMPARE_DIGEST_FIELD(lithologyContact);
    RYCRAFT_COMPARE_DIGEST_FIELD(plateId);
    RYCRAFT_COMPARE_DIGEST_FIELD(waterBodyId);
    RYCRAFT_COMPARE_DIGEST_FIELD(primaryRock);
    RYCRAFT_COMPARE_DIGEST_FIELD(secondaryRock);
    RYCRAFT_COMPARE_DIGEST_FIELD(primaryBiome);
    RYCRAFT_COMPARE_DIGEST_FIELD(secondaryBiome);
    RYCRAFT_COMPARE_DIGEST_FIELD(biomeTransition);
    RYCRAFT_COMPARE_DIGEST_FIELD(water);
    RYCRAFT_COMPARE_DIGEST_FIELD(material);
    RYCRAFT_COMPARE_DIGEST_FIELD(suitability);
#undef RYCRAFT_COMPARE_DIGEST_FIELD
    return "none";
}

bool hasStandingWater(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.ocean || sample.hydrology.lake;
}

uint8_t waterClass(const worldgen::SurfaceSample& sample) {
    return static_cast<uint8_t>(
        (sample.hydrology.ocean ? 1U : 0U) | (sample.hydrology.lake ? 2U : 0U) |
        (sample.hydrology.river ? 4U : 0U) | (sample.hydrology.waterfall ? 8U : 0U) |
        (sample.hydrology.delta ? 16U : 0U));
}

bool isOutletOrChannelJunction(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.river || sample.hydrology.waterfall || sample.hydrology.delta;
}

bool isTaggedFault(const worldgen::SurfaceSample& first, const worldgen::SurfaceSample& second) {
    return first.geology.faultStrength > 0.45 || second.geology.faultStrength > 0.45 ||
           first.geology.boundary == worldgen::PlateBoundary::TRANSFORM ||
           second.geology.boundary == worldgen::PlateBoundary::TRANSFORM;
}

bool hasUsableShoreDistance(const worldgen::SurfaceSample& sample) {
    return std::isfinite(sample.hydrology.lakeShoreDistance) &&
           sample.hydrology.lakeShoreDistance > -32'000.0 &&
           (sample.hydrology.lake || sample.hydrology.lakeBank ||
            std::abs(sample.hydrology.lakeShoreDistance) <= 96.0);
}

bool isWaterLevelJunction(const worldgen::SurfaceSample& first,
                          const worldgen::SurfaceSample& second) {
    if (!hasStandingWater(first) || !hasStandingWater(second))
        return false;
    if (first.hydrology.waterBodyId == second.hydrology.waterBodyId)
        return false;
    return std::abs(first.hydrology.waterSurface - second.hydrology.waterSurface) > 0.125;
}

bool isIntentionalShoreline(const worldgen::SurfaceSample& first,
                            const worldgen::SurfaceSample& second) {
    if (waterClass(first) == waterClass(second))
        return false;
    if (isOutletOrChannelJunction(first) || isOutletOrChannelJunction(second))
        return true;

    if (first.hydrology.ocean != second.hydrology.ocean) {
        const double firstSeaDistance = first.terrainHeight - SEA_LEVEL;
        const double secondSeaDistance = second.terrainHeight - SEA_LEVEL;
        if (firstSeaDistance * secondSeaDistance <= 0.0 ||
            std::min(std::abs(firstSeaDistance), std::abs(secondSeaDistance)) <= 2.0) {
            return true;
        }
    }

    if (first.hydrology.lake != second.hydrology.lake && hasUsableShoreDistance(first) &&
        hasUsableShoreDistance(second)) {
        const double firstDistance = first.hydrology.lakeShoreDistance;
        const double secondDistance = second.hydrology.lakeShoreDistance;
        return firstDistance * secondDistance <= 0.0 ||
               std::min(std::abs(firstDistance), std::abs(secondDistance)) <= 2.0;
    }
    return false;
}

bool hasHardMaterialConstraint(const worldgen::SurfaceSample& sample) {
    return waterClass(sample) != 0 || sample.geology.volcanicActivity > 0.52 ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::CLIFF) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SCREE) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SNOWFIELD) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::GLACIER);
}

class SmoothFakeLearnedBackend final : public worldgen::learned::TerrainInferenceBackend {
public:
    worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>
    inferPage(const worldgen::learned::GenerationIdentity& identity,
              worldgen::learned::TerrainPageKey key) override {
        using namespace worldgen::learned;
        if (!identity.valid()) {
            return AuthorityResult<TerrainAuthorityPage>::failed({
                .code = GenerationFailureCode::INVALID_REQUEST,
                .message = "Invalid final-field fake learned identity",
                .retriable = false,
            });
        }

        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        const int64_t rowBegin = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE;
        const int64_t columnBegin = key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE;
        const double seedPhase = static_cast<double>(identity.seed % 65'521U) * 0.000'173;
        const auto signedQuantized = [](double value) {
            return static_cast<int16_t>(std::clamp<long long>(std::llround(value),
                                                              std::numeric_limits<int16_t>::min(),
                                                              std::numeric_limits<int16_t>::max()));
        };
        const auto unsignedQuantized = [](double value) {
            return static_cast<uint16_t>(
                std::clamp<long long>(std::llround(value), std::numeric_limits<uint16_t>::min(),
                                      std::numeric_limits<uint16_t>::max()));
        };
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const double worldRow = static_cast<double>(rowBegin + row);
                const double worldColumn = static_cast<double>(columnBegin + column);
                const double broad = 520.0 * std::sin(worldColumn / 211.0 + seedPhase) +
                                     430.0 * std::cos(worldRow / 263.0 - seedPhase * 0.7);
                const double basin =
                    260.0 * std::sin((worldColumn + worldRow) / 97.0 + seedPhase * 1.3) +
                    170.0 * std::cos((worldColumn - worldRow) / 131.0 - seedPhase * 0.4);
                // A stationary, multi-orientation residual keeps the fake
                // authority's own derivative energy neutral at every tested
                // storage spacing. Without this residual, the small set of
                // broad analytical waves correlates with the fixed spacing-32
                // probes before production postprocessing is involved.
                const double residual =
                    18.0 * std::sin((worldColumn * 0.809 + worldRow * 0.588) / 17.3 +
                                    seedPhase * 2.11) +
                    14.0 * std::cos((-worldColumn * 0.374 + worldRow * 0.927) / 23.7 -
                                    seedPhase * 1.73) +
                    10.0 * std::sin((worldColumn * 0.643 - worldRow * 0.766) / 31.1 +
                                    seedPhase * 0.91);
                const double unmodifiedElevation = broad + basin + residual - 180.0;
                const double worldX = worldColumn * worldgen::learned::MODEL_BLOCK_SCALE;
                const double worldZ = worldRow * worldgen::learned::MODEL_BLOCK_SCALE;
                // Offset the basin so its curved shore crosses the x=0 former
                // control line instead of centering a zero normal derivative
                // on that line. This keeps the statistical control honest at
                // the 2,048 and 8,192-block spacings, where x=0 is the only
                // aligned line in the bounded fixture.
                const double localX = worldX + 296.0;
                const double localZ = worldZ + 1'024.0;
                const double warpedX = localX + 13.0 * std::sin(localZ / 71.0);
                const double warpedZ = localZ + 9.0 * std::sin(localX / 61.0);
                const double lakeRadius = std::hypot(warpedX, warpedZ);
                const double lakeHeight = 126.0 + std::min(22.0, lakeRadius * 0.01);
                const double lakeElevationMeters =
                    (lakeHeight - worldgen::learned::LEARNED_SEA_LEVEL) *
                    worldgen::learned::WORLD_METERS_PER_BLOCK;
                const double elevation =
                    lakeRadius <= 384.0 ? lakeElevationMeters : unmodifiedElevation;
                const double temperature =
                    1'450.0 + 720.0 * std::sin(worldColumn / 1'103.0 + seedPhase) -
                    510.0 * std::sin(worldRow / 887.0 - seedPhase) +
                    95.0 * std::sin((worldColumn * 0.731 + worldRow * 0.682) / 19.1 +
                                    seedPhase * 1.37) +
                    70.0 * std::cos((-worldColumn * 0.526 + worldRow * 0.851) / 27.9 -
                                    seedPhase * 1.91) +
                    135.0 * std::sin((worldColumn * 0.963 + worldRow * 0.269) / 7.7 +
                                     seedPhase * 2.43) +
                    105.0 * std::cos((-worldColumn * 0.216 + worldRow * 0.976) / 11.3 -
                                     seedPhase * 0.67);
                const double unmodifiedPrecipitation =
                    1'250.0 + 380.0 * std::sin(worldColumn / 479.0 - seedPhase * 0.5) +
                    290.0 * std::cos(worldRow / 557.0 + seedPhase * 0.8) +
                    75.0 * std::sin((worldColumn * 0.914 - worldRow * 0.406) / 21.3 +
                                    seedPhase * 1.57) +
                    55.0 * std::cos((worldColumn * 0.438 + worldRow * 0.899) / 33.1 -
                                    seedPhase * 0.83) +
                    114.0 * std::cos((worldColumn * 0.847 + worldRow * 0.532) / 8.9 +
                                     seedPhase * 2.73) +
                    90.0 * std::sin((-worldColumn * 0.617 + worldRow * 0.787) / 13.7 -
                                    seedPhase * 1.11);
                const double precipitation = lakeRadius <= 384.0 ? 0.0 : unmodifiedPrecipitation;
                QuantizedTerrainSample& sample =
                    page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)];
                sample.elevationMeters = signedQuantized(elevation);
                sample.meanTemperatureCentidegrees = signedQuantized(temperature);
                sample.temperatureVariabilityCentidegrees = unsignedQuantized(
                    650.0 + 180.0 * std::sin((worldColumn - worldRow) / 733.0 + seedPhase) +
                    55.0 * std::sin((worldColumn * 0.342 + worldRow * 0.940) / 24.7 -
                                    seedPhase * 1.21));
                sample.annualPrecipitationMillimeters = unsignedQuantized(precipitation);
                sample.precipitationCoefficientBasisPoints = unsignedQuantized(
                    2'800.0 + 550.0 * std::cos((worldColumn + worldRow) / 941.0 - seedPhase));
                sample.lapseRateMicrodegreesPerMeter =
                    signedQuantized(-6'500.0 + 180.0 * std::sin(worldColumn / 1'271.0 + seedPhase));
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }
};

worldgen::learned::GenerationIdentity finalFieldIdentity(uint64_t seed) {
    worldgen::learned::GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash = worldgen::learned::parseSha256(
                                 "543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0")
                                 .value();
    identity.runtimeHash = worldgen::learned::parseSha256(
                               "e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6")
                               .value();
    return identity;
}

template <typename Operation> decltype(auto) awaitLearnedAuthority(Operation&& operation) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        try {
            return operation();
        } catch (const worldgen::learned::GenerationFailureException& error) {
            if (error.status() != worldgen::learned::AuthorityStatus::DEFERRED ||
                std::chrono::steady_clock::now() >= deadline) {
                throw;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

class FakeLearnedGenerator {
public:
    explicit FakeLearnedGenerator(uint64_t seed)
        : store_("final_field_authority"), identity_(finalFieldIdentity(seed)),
          backend_(std::make_shared<SmoothFakeLearnedBackend>()),
          authority_(std::make_shared<worldgen::learned::CachedTerrainAuthority>(
              identity_, store_.path(), backend_)),
          context_(std::make_shared<worldgen::learned::WorldGenerationContext>(
              identity_, authority_, worldgen::learned::AuthorityQuality::FINAL)),
          macro_(seed, context_), generator_(seed, context_) {}

    worldgen::SurfaceSample sampleSurface(int64_t x, int64_t z) {
        return awaitLearnedAuthority([&] { return generator_.sampleSurface(x, z); });
    }

    worldgen::SurfaceSample sampleMacroSurface(int64_t x, int64_t z) {
        return awaitLearnedAuthority(
            [&] { return macro_.sampleSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1); });
    }

    void nativeHydrologyAtPoints(std::span<const ColumnPos> positions,
                                 std::span<worldgen::HydrologySample> output) {
        awaitLearnedAuthority(
            [&] { generator_.sampleNativeHydrologyAuthorityPoints(positions, output); });
    }

    void nativeHydrologyTopologyGrid(int64_t originX, int64_t originZ, int cellWidth,
                                     int cellHeight,
                                     std::span<worldgen::NativeHydrologyTopologyCell> output) {
        awaitLearnedAuthority([&] {
            generator_.sampleNativeHydrologyTopologyGrid(originX, originZ, cellWidth, cellHeight,
                                                         output);
        });
    }

    BlockType surfaceMaterialAt(int64_t x, int64_t z) {
        return awaitLearnedAuthority([&] { return generator_.surfaceMaterialAt(x, z); });
    }

    std::shared_ptr<const ColumnPlan> getColumnPlan(ColumnPos column) {
        return awaitLearnedAuthority([&] { return generator_.getColumnPlan(column); });
    }

    void sampleFarHabitatPoints(std::span<const ColumnPos> positions,
                                std::span<worldgen::SurfaceSample> output) {
        awaitLearnedAuthority([&] {
            generator_.sampleFarHabitatPoints(positions, output);
            return true;
        });
    }

    void generate(Chunk& chunk) {
        awaitLearnedAuthority([&] { generator_.generate(chunk); });
    }

    void clearMacroCaches() { generator_.clearMacroCaches(); }

private:
    TempDir store_;
    worldgen::learned::GenerationIdentity identity_;
    std::shared_ptr<SmoothFakeLearnedBackend> backend_;
    std::shared_ptr<worldgen::learned::CachedTerrainAuthority> authority_;
    std::shared_ptr<worldgen::learned::WorldGenerationContext> context_;
    worldgen::MacroGenerationSampler macro_;
    ChunkGenerator generator_;
};

class ProbeCache {
public:
    explicit ProbeCache(uint64_t seed) : generator_(seed) {}

    const worldgen::SurfaceSample& surface(int64_t x, int64_t z) { return probe(x, z).surface; }

    const worldgen::SurfaceSample& directSurface(int64_t x, int64_t z) {
        const SamplePos position{x, z};
        if (auto found = directSurfaces_.find(position); found != directSurfaces_.end())
            return found->second;
        return directSurfaces_.emplace(position, generator_.sampleMacroSurface(x, z)).first->second;
    }

    void nativeHydrologyTopologyGrid(int64_t originX, int64_t originZ, int cellWidth,
                                     int cellHeight,
                                     std::span<worldgen::NativeHydrologyTopologyCell> output) {
        generator_.nativeHydrologyTopologyGrid(originX, originZ, cellWidth, cellHeight, output);
    }

    void sampleNativeHydrologyAuthorityPoints(std::span<const ColumnPos> positions,
                                              std::span<worldgen::HydrologySample> output) {
        generator_.nativeHydrologyAtPoints(positions, output);
    }

    BlockType material(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        if (!cached.material)
            cached.material = generator_.surfaceMaterialAt(x, z);
        return *cached.material;
    }

    BlockType emittedGround(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        populateEmittedBlocks(cached, x, z);
        return *cached.emittedGround;
    }

    BlockType emittedTop(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        populateEmittedBlocks(cached, x, z);
        return *cached.emittedTop;
    }

private:
    CachedProbe& probe(int64_t x, int64_t z) {
        const SamplePos position{x, z};
        if (auto found = probes_.find(position); found != probes_.end())
            return found->second;
        CachedProbe result;
        result.surface = generator_.sampleSurface(x, z);
        return probes_.emplace(position, std::move(result)).first->second;
    }

    BlockType blockAt(int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        auto [found, inserted] = cubes_.try_emplace(position);
        if (inserted) {
            found->second = std::make_unique<Chunk>(position);
            generator_.generate(*found->second);
        }
        return found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                       Chunk::worldToLocal(z));
    }

    void populateEmittedBlocks(CachedProbe& cached, int64_t x, int64_t z) {
        if (cached.emittedGround && cached.emittedTop)
            return;
        const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
        const std::shared_ptr<const ColumnPlan> plan = generator_.getColumnPlan(column);
        const int groundY = plan->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
        int topY = groundY;
        if (waterClass(cached.surface) != 0) {
            topY = std::max(topY, static_cast<int>(std::ceil(cached.surface.waterSurface)) - 1);
        }
        cached.emittedGround = blockAt(x, groundY, z);
        cached.emittedTop = blockAt(x, topY, z);
    }

    FakeLearnedGenerator generator_;
    std::unordered_map<SamplePos, CachedProbe, SamplePosHash> probes_;
    std::unordered_map<SamplePos, worldgen::SurfaceSample, SamplePosHash> directSurfaces_;
    std::unordered_map<ChunkPos, std::unique_ptr<Chunk>> cubes_;
};

struct EnergyAccumulator {
    double squaredSum = 0.0;
    size_t count = 0;

    void add(double derivative) {
        if (!std::isfinite(derivative))
            return;
        squaredSum += derivative * derivative;
        ++count;
    }

    void merge(const EnergyAccumulator& other) {
        squaredSum += other.squaredSum;
        count += other.count;
    }

    double mean() const { return count == 0 ? 0.0 : squaredSum / static_cast<double>(count); }
};

struct ContinuousMetric {
    EnergyAccumulator formerLine;
    EnergyAccumulator nearby;
    artifact::OrientationHistogram orientation;

    void merge(const ContinuousMetric& other) {
        formerLine.merge(other.formerLine);
        nearby.merge(other.nearby);
        artifact::add(orientation, other.orientation);
    }
};

enum class MeasuredField : uint8_t {
    TERRAIN,
    PRECIPITATION,
    LITHOLOGY,
    SHORE_DISTANCE,
    BIOME_SUITABILITY,
    COUNT,
};

using ContinuousMetrics = std::array<ContinuousMetric, static_cast<size_t>(MeasuredField::COUNT)>;

bool derivativeAllowed(const worldgen::SurfaceSample& first, const worldgen::SurfaceSample& second,
                       MeasuredField field) {
    if (field == MeasuredField::LITHOLOGY)
        return !isTaggedFault(first, second);
    if (field == MeasuredField::SHORE_DISTANCE) {
        if (!hasUsableShoreDistance(first) || !hasUsableShoreDistance(second))
            return false;
        if (isOutletOrChannelJunction(first) || isOutletOrChannelJunction(second))
            return false;
        if (isWaterLevelJunction(first, second))
            return false;
        const worldgen::WaterBodyId firstId = first.hydrology.waterBodyId;
        const worldgen::WaterBodyId secondId = second.hydrology.waterBodyId;
        return firstId == secondId || firstId == worldgen::NO_WATER_BODY ||
               secondId == worldgen::NO_WATER_BODY;
    }
    return !isTaggedFault(first, second) && !isWaterLevelJunction(first, second) &&
           !isIntentionalShoreline(first, second);
}

double scalarValue(const worldgen::SurfaceSample& sample, MeasuredField field) {
    switch (field) {
        case MeasuredField::TERRAIN:
            return artifact::fieldValue(sample, artifact::FinalField::TERRAIN_HEIGHT);
        case MeasuredField::PRECIPITATION:
            return artifact::fieldValue(sample, artifact::FinalField::ANNUAL_PRECIPITATION);
        case MeasuredField::LITHOLOGY:
            return artifact::fieldValue(sample, artifact::FinalField::LITHOLOGY_CONTACT);
        case MeasuredField::SHORE_DISTANCE:
            return artifact::fieldValue(sample, artifact::FinalField::LAKE_SHORE_DISTANCE);
        case MeasuredField::BIOME_SUITABILITY:
        case MeasuredField::COUNT:
            return 0.0;
    }
    return 0.0;
}

void addOrientation(artifact::OrientationBins& bins, double gradientX, double gradientZ) {
    if (!std::isfinite(gradientX) || !std::isfinite(gradientZ) ||
        std::hypot(gradientX, gradientZ) <= 1.0e-8) {
        return;
    }
    ++bins[artifact::orientationBin(gradientX, gradientZ)];
}

// ColumnPlan stores its compact canonical fields as floats while the direct
// learned query retains doubles. Ignore only their sub-tenth-milliblock
// conversion residue so a ratio of two numerical zeros cannot look like a
// control-line signal. Any visible terrain, climate, or ecology artifact
// exceeds this by several orders of magnitude.
constexpr double DIRECT_REFERENCE_RESIDUAL_EPSILON = 1.0e-4;

double directReferenceResidual(double value) {
    return std::abs(value) <= DIRECT_REFERENCE_RESIDUAL_EPSILON ? 0.0 : value;
}

struct AxialPoint {
    int64_t x = 0;
    int64_t z = 0;
};

AxialPoint axialPoint(bool vertical, int64_t normal, int64_t along) {
    return vertical ? AxialPoint{normal, along} : AxialPoint{along, normal};
}

void accumulateDerivative(ProbeCache& cache, ContinuousMetric& metric, MeasuredField field,
                          bool formerLine, bool vertical, int64_t normal, int64_t along) {
    const AxialPoint firstPosition = axialPoint(vertical, normal - 1, along);
    const AxialPoint secondPosition = axialPoint(vertical, normal + 1, along);
    const worldgen::SurfaceSample& first = cache.surface(firstPosition.x, firstPosition.z);
    const worldgen::SurfaceSample& second = cache.surface(secondPosition.x, secondPosition.z);
    const worldgen::SurfaceSample& directFirst =
        cache.directSurface(firstPosition.x, firstPosition.z);
    const worldgen::SurfaceSample& directSecond =
        cache.directSurface(secondPosition.x, secondPosition.z);
    if (!derivativeAllowed(first, second, field) ||
        !derivativeAllowed(directFirst, directSecond, field))
        return;
    EnergyAccumulator& destination = formerLine ? metric.formerLine : metric.nearby;
    if (field != MeasuredField::BIOME_SUITABILITY) {
        const double exactDerivative =
            (scalarValue(second, field) - scalarValue(first, field)) * 0.5;
        const double directDerivative =
            (scalarValue(directSecond, field) - scalarValue(directFirst, field)) * 0.5;
        destination.add(directReferenceResidual(exactDerivative - directDerivative));
        return;
    }
    for (size_t biome = 0; biome < static_cast<size_t>(Biome::COUNT); ++biome) {
        const Biome value = static_cast<Biome>(biome);
        const double exactDerivative = (artifact::biomeSuitabilityValue(second, value) -
                                        artifact::biomeSuitabilityValue(first, value)) *
                                       0.5;
        const double directDerivative = (artifact::biomeSuitabilityValue(directSecond, value) -
                                         artifact::biomeSuitabilityValue(directFirst, value)) *
                                        0.5;
        destination.add(directReferenceResidual(exactDerivative - directDerivative));
    }
}

void accumulateOrientation(ProbeCache& cache, ContinuousMetric& metric, MeasuredField field,
                           bool formerLine, int64_t x, int64_t z) {
    const worldgen::SurfaceSample& west = cache.surface(x - 1, z);
    const worldgen::SurfaceSample& east = cache.surface(x + 1, z);
    const worldgen::SurfaceSample& north = cache.surface(x, z - 1);
    const worldgen::SurfaceSample& south = cache.surface(x, z + 1);
    const worldgen::SurfaceSample& directWest = cache.directSurface(x - 1, z);
    const worldgen::SurfaceSample& directEast = cache.directSurface(x + 1, z);
    const worldgen::SurfaceSample& directNorth = cache.directSurface(x, z - 1);
    const worldgen::SurfaceSample& directSouth = cache.directSurface(x, z + 1);
    if (!derivativeAllowed(west, east, field) || !derivativeAllowed(north, south, field) ||
        !derivativeAllowed(directWest, directEast, field) ||
        !derivativeAllowed(directNorth, directSouth, field))
        return;
    artifact::OrientationBins& destination =
        formerLine ? metric.orientation.formerLine : metric.orientation.nearby;
    if (field != MeasuredField::BIOME_SUITABILITY) {
        const double exactX = (scalarValue(east, field) - scalarValue(west, field)) * 0.5;
        const double exactZ = (scalarValue(south, field) - scalarValue(north, field)) * 0.5;
        const double directX =
            (scalarValue(directEast, field) - scalarValue(directWest, field)) * 0.5;
        const double directZ =
            (scalarValue(directSouth, field) - scalarValue(directNorth, field)) * 0.5;
        addOrientation(destination, directReferenceResidual(exactX - directX),
                       directReferenceResidual(exactZ - directZ));
        return;
    }
    for (size_t biome = 0; biome < static_cast<size_t>(Biome::COUNT); ++biome) {
        const Biome value = static_cast<Biome>(biome);
        const double exactX = (artifact::biomeSuitabilityValue(east, value) -
                               artifact::biomeSuitabilityValue(west, value)) *
                              0.5;
        const double exactZ = (artifact::biomeSuitabilityValue(south, value) -
                               artifact::biomeSuitabilityValue(north, value)) *
                              0.5;
        const double directX = (artifact::biomeSuitabilityValue(directEast, value) -
                                artifact::biomeSuitabilityValue(directWest, value)) *
                               0.5;
        const double directZ = (artifact::biomeSuitabilityValue(directSouth, value) -
                                artifact::biomeSuitabilityValue(directNorth, value)) *
                               0.5;
        addOrientation(destination, directReferenceResidual(exactX - directX),
                       directReferenceResidual(exactZ - directZ));
    }
}

ContinuousMetrics measureContinuousWindow(ProbeCache& cache, int64_t line, int64_t alongCenter,
                                          bool vertical) {
    ContinuousMetrics result;
    // Every learned native pixel spans four blocks. Preserve that phase in
    // the nearby controls so ordinary native-pixel boundaries do not become
    // false positives at former lines whose spacing is also divisible by four.
    // Every offset is four-block aligned but deliberately not eight-block
    // aligned. That makes a control adjacent to, rather than coincident with,
    // the smallest former lattice line and consequently with every larger
    // tested spacing as well.
    for (int64_t offset = -CONTINUOUS_HALF_WINDOW; offset <= CONTINUOUS_HALF_WINDOW; offset += 2) {
        const int64_t along = alongCenter + offset;
        for (size_t field = 0; field < result.size(); ++field) {
            const MeasuredField measured = static_cast<MeasuredField>(field);
            accumulateDerivative(cache, result[field], measured, true, vertical, line, along);
            for (const int64_t nearbyOffset : CONTINUOUS_NEARBY_OFFSETS) {
                accumulateDerivative(cache, result[field], measured, false, vertical,
                                     line + nearbyOffset, along);
            }
            if (offset % 4 == 0) {
                const AxialPoint former = axialPoint(vertical, line, along);
                accumulateOrientation(cache, result[field], measured, true, former.x, former.z);
                for (const int64_t nearbyOffset : CONTINUOUS_NEARBY_OFFSETS) {
                    const AxialPoint nearby = axialPoint(vertical, line + nearbyOffset, along);
                    accumulateOrientation(cache, result[field], measured, false, nearby.x,
                                          nearby.z);
                }
            }
        }
    }
    return result;
}

ContinuousMetric measureShoreDistanceWindow(ProbeCache& cache, int64_t line, int64_t alongCenter,
                                            bool vertical) {
    ContinuousMetric result;
    constexpr MeasuredField field = MeasuredField::SHORE_DISTANCE;
    const auto accumulateAbsoluteDerivative = [&](bool formerLine, int64_t normal, int64_t along) {
        const AxialPoint firstPosition = axialPoint(vertical, normal - 1, along);
        const AxialPoint secondPosition = axialPoint(vertical, normal + 1, along);
        const worldgen::SurfaceSample& first = cache.surface(firstPosition.x, firstPosition.z);
        const worldgen::SurfaceSample& second = cache.surface(secondPosition.x, secondPosition.z);
        if (!derivativeAllowed(first, second, field))
            return;
        EnergyAccumulator& destination = formerLine ? result.formerLine : result.nearby;
        destination.add((scalarValue(second, field) - scalarValue(first, field)) * 0.5);
    };
    const auto accumulateAbsoluteOrientation = [&](bool formerLine, int64_t x, int64_t z) {
        const worldgen::SurfaceSample& west = cache.surface(x - 1, z);
        const worldgen::SurfaceSample& east = cache.surface(x + 1, z);
        const worldgen::SurfaceSample& north = cache.surface(x, z - 1);
        const worldgen::SurfaceSample& south = cache.surface(x, z + 1);
        if (!derivativeAllowed(west, east, field) || !derivativeAllowed(north, south, field))
            return;
        artifact::OrientationBins& destination =
            formerLine ? result.orientation.formerLine : result.orientation.nearby;
        addOrientation(destination, (scalarValue(east, field) - scalarValue(west, field)) * 0.5,
                       (scalarValue(south, field) - scalarValue(north, field)) * 0.5);
    };
    for (int64_t offset = -CONTINUOUS_HALF_WINDOW; offset <= CONTINUOUS_HALF_WINDOW; offset += 2) {
        const int64_t along = alongCenter + offset;
        accumulateAbsoluteDerivative(true, line, along);
        for (const int64_t nearbyOffset : CONTINUOUS_NEARBY_OFFSETS) {
            accumulateAbsoluteDerivative(false, line + nearbyOffset, along);
        }
        if (offset % 4 != 0)
            continue;
        const AxialPoint former = axialPoint(vertical, line, along);
        accumulateAbsoluteOrientation(true, former.x, former.z);
        for (const int64_t nearbyOffset : CONTINUOUS_NEARBY_OFFSETS) {
            const AxialPoint nearby = axialPoint(vertical, line + nearbyOffset, along);
            accumulateAbsoluteOrientation(false, nearby.x, nearby.z);
        }
    }
    return result;
}

bool hasCanonicalLakeToDryTransition(const worldgen::HydrologySample& first,
                                     const worldgen::HydrologySample& second) {
    const bool firstLake = first.lake || first.lakeBank;
    const bool secondLake = second.lake || second.lakeBank;
    if (firstLake == secondLake)
        return false;
    const auto hasOtherWaterFeature = [](const worldgen::HydrologySample& sample) {
        return sample.ocean || sample.river || sample.delta || sample.waterfall || sample.wetland ||
               sample.channelBank;
    };
    if (hasOtherWaterFeature(first) || hasOtherWaterFeature(second))
        return false;

    const worldgen::HydrologySample& wet = firstLake ? first : second;
    const worldgen::HydrologySample& dry = firstLake ? second : first;
    return wet.waterBodyId != worldgen::NO_WATER_BODY &&
           dry.waterBodyId == worldgen::NO_WATER_BODY && std::isfinite(wet.lakeShoreDistance) &&
           std::isfinite(dry.lakeShoreDistance) && wet.lakeShoreDistance > 0.0 &&
           dry.lakeShoreDistance <= 0.0 && wet.waterSurface > wet.surfaceElevation + 0.01 &&
           dry.waterSurface <= dry.surfaceElevation + 0.01;
}

bool hasCanonicalLakeToDryTransition(const worldgen::SurfaceSample& first,
                                     const worldgen::SurfaceSample& second) {
    return hasCanonicalLakeToDryTransition(first.hydrology, second.hydrology) &&
           !isWaterLevelJunction(first, second);
}

bool shoreFixtureLess(const ShoreWindowFixture& first, const ShoreWindowFixture& second) {
    if (first.normal != second.normal)
        return first.normal < second.normal;
    if (first.transitionNormal != second.transitionNormal)
        return first.transitionNormal < second.transitionNormal;
    if (first.along != second.along)
        return first.along < second.along;
    return first.vertical < second.vertical;
}

bool sameShoreFixture(const ShoreWindowFixture& first, const ShoreWindowFixture& second) {
    return first.normal == second.normal && first.transitionNormal == second.transitionNormal &&
           first.along == second.along && first.vertical == second.vertical;
}

struct NativeShoreTransition {
    int64_t normal = 0;
    int64_t along = 0;
    bool vertical = true;
};

bool nativeShoreTransitionLess(const NativeShoreTransition& first,
                               const NativeShoreTransition& second) {
    if (first.normal != second.normal)
        return first.normal < second.normal;
    if (first.along != second.along)
        return first.along < second.along;
    return first.vertical < second.vertical;
}

bool sameNativeShoreTransition(const NativeShoreTransition& first,
                               const NativeShoreTransition& second) {
    return first.normal == second.normal && first.along == second.along &&
           first.vertical == second.vertical;
}

std::vector<NativeShoreTransition> collectCanonicalShoreTransitions(ProbeCache& cache) {
    std::vector<NativeShoreTransition> transitions;
    std::vector<NativeShoreTransition> pending;
    std::vector<ColumnPos> positions;
    constexpr size_t BATCH_TRANSITIONS = 2'048;

    const auto flush = [&] {
        if (pending.empty())
            return;
        std::vector<worldgen::HydrologySample> samples(positions.size());
        cache.sampleNativeHydrologyAuthorityPoints(positions, samples);
        for (size_t index = 0; index < pending.size(); ++index) {
            if (hasCanonicalLakeToDryTransition(samples[index * 2], samples[index * 2 + 1]))
                transitions.push_back(pending[index]);
        }
        pending.clear();
        positions.clear();
    };
    const auto append = [&](bool vertical, int64_t normal, int64_t along) {
        pending.push_back({normal, along, vertical});
        const AxialPoint first = axialPoint(vertical, normal - 1, along);
        const AxialPoint second = axialPoint(vertical, normal + 1, along);
        positions.push_back({first.x, first.z});
        positions.push_back({second.x, second.z});
        if (pending.size() == BATCH_TRANSITIONS)
            flush();
    };

    // Native samples are centered on four-block coordinates. The categorical
    // lake edge is therefore halfway between adjacent centers, not on a
    // center itself. Probe one block on either side of that true boundary.
    constexpr int64_t NATIVE_HALF_SPACING = worldgen::NATIVE_HYDROLOGY_RASTER_SPACING / 2;
    static_assert(NATIVE_HALF_SPACING > 0);
    for (int64_t x = SHORE_TRANSITION_MIN_X + NATIVE_HALF_SPACING;
         x <= SHORE_TRANSITION_MAX_X - NATIVE_HALF_SPACING;
         x += worldgen::NATIVE_HYDROLOGY_RASTER_SPACING) {
        for (int64_t z = SHORE_TRANSITION_MIN_Z; z <= SHORE_TRANSITION_MAX_Z;
             z += worldgen::NATIVE_HYDROLOGY_RASTER_SPACING) {
            append(true, x, z);
        }
    }
    for (int64_t z = SHORE_TRANSITION_MIN_Z + NATIVE_HALF_SPACING;
         z <= SHORE_TRANSITION_MAX_Z - NATIVE_HALF_SPACING;
         z += worldgen::NATIVE_HYDROLOGY_RASTER_SPACING) {
        for (int64_t x = SHORE_TRANSITION_MIN_X; x <= SHORE_TRANSITION_MAX_X;
             x += worldgen::NATIVE_HYDROLOGY_RASTER_SPACING) {
            append(false, z, x);
        }
    }
    flush();

    std::sort(transitions.begin(), transitions.end(), nativeShoreTransitionLess);
    transitions.erase(
        std::unique(transitions.begin(), transitions.end(), sameNativeShoreTransition),
        transitions.end());
    return transitions;
}

int64_t nearestAlignedFormerLine(int64_t coordinate, int64_t spacing) {
    const int64_t lower = world_coord::floorDiv(coordinate, spacing) * spacing;
    const int64_t upper = lower + spacing;
    return coordinate - lower <= upper - coordinate ? lower : upper;
}

int64_t controlDistance(const ShoreWindowFixture& fixture) {
    return std::abs(fixture.normal - fixture.transitionNormal);
}

size_t orientationCount(const artifact::OrientationBins& bins) {
    size_t total = 0;
    for (const uint32_t count : bins)
        total += count;
    return total;
}

bool hasRequiredShoreCoverage(const ContinuousMetric& metric) {
    return metric.formerLine.count >= 24 && metric.nearby.count >= 96 &&
           orientationCount(metric.orientation.formerLine) >= 16 &&
           orientationCount(metric.orientation.nearby) >= 64;
}

struct LocatedShoreWindow {
    ShoreWindowFixture fixture;
    ContinuousMetric metric;
};

struct ShoreSearchDiagnostics {
    size_t nativeTransitions = 0;
    size_t controlCandidates = 0;
    size_t finalTransitions = 0;
    size_t coverageCandidates = 0;
    size_t bestFormerSamples = 0;
    size_t bestNearbySamples = 0;
    size_t bestFormerOrientations = 0;
    size_t bestNearbyOrientations = 0;
    size_t mergedFormerSamples = 0;
    size_t mergedNearbySamples = 0;
    size_t mergedFormerOrientations = 0;
    size_t mergedNearbyOrientations = 0;
    int64_t closestControlDistance = std::numeric_limits<int64_t>::max();
    std::optional<NativeShoreTransition> closestTransition;
    int64_t closestVerticalControlDistance = std::numeric_limits<int64_t>::max();
    int64_t closestHorizontalControlDistance = std::numeric_limits<int64_t>::max();
};

std::optional<LocatedShoreWindow>
locateCanonicalShoreWindow(ProbeCache& cache, std::span<const NativeShoreTransition> transitions,
                           int64_t spacing, ShoreSearchDiagnostics& diagnostics) {
    diagnostics.nativeTransitions = transitions.size();
    std::vector<ShoreWindowFixture> candidates;
    candidates.reserve(transitions.size());
    for (const NativeShoreTransition& transition : transitions) {
        const int64_t normal = nearestAlignedFormerLine(transition.normal, spacing);
        const int64_t distance = std::abs(normal - transition.normal);
        if (distance < diagnostics.closestControlDistance) {
            diagnostics.closestControlDistance = distance;
            diagnostics.closestTransition = transition;
        }
        int64_t& closestByOrientation = transition.vertical
                                            ? diagnostics.closestVerticalControlDistance
                                            : diagnostics.closestHorizontalControlDistance;
        closestByOrientation = std::min(closestByOrientation, distance);
        if (distance > SHORE_MAX_CONTROL_DISTANCE)
            continue;
        candidates.push_back({normal, transition.normal, transition.along, transition.vertical});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const ShoreWindowFixture& first, const ShoreWindowFixture& second) {
                  if (controlDistance(first) != controlDistance(second))
                      return controlDistance(first) < controlDistance(second);
                  return shoreFixtureLess(first, second);
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end(), sameShoreFixture),
                     candidates.end());
    diagnostics.controlCandidates = candidates.size();

    // Native authority selected the transition. Final exact samples prove
    // that no later terrain, material, or mesh-facing stage deleted it.
    std::optional<ShoreWindowFixture> firstFixture;
    ContinuousMetric mergedMetric;
    for (const ShoreWindowFixture& fixture : candidates) {
        const AxialPoint firstPosition =
            axialPoint(fixture.vertical, fixture.transitionNormal - 1, fixture.along);
        const AxialPoint secondPosition =
            axialPoint(fixture.vertical, fixture.transitionNormal + 1, fixture.along);
        if (!hasCanonicalLakeToDryTransition(cache.surface(firstPosition.x, firstPosition.z),
                                             cache.surface(secondPosition.x, secondPosition.z))) {
            continue;
        }
        ++diagnostics.finalTransitions;
        ContinuousMetric metric =
            measureShoreDistanceWindow(cache, fixture.normal, fixture.along, fixture.vertical);
        diagnostics.bestFormerSamples =
            std::max(diagnostics.bestFormerSamples, metric.formerLine.count);
        diagnostics.bestNearbySamples =
            std::max(diagnostics.bestNearbySamples, metric.nearby.count);
        diagnostics.bestFormerOrientations = std::max(
            diagnostics.bestFormerOrientations, orientationCount(metric.orientation.formerLine));
        diagnostics.bestNearbyOrientations = std::max(diagnostics.bestNearbyOrientations,
                                                      orientationCount(metric.orientation.nearby));
        if (!firstFixture)
            firstFixture = fixture;
        mergedMetric.merge(metric);
        ++diagnostics.coverageCandidates;
        diagnostics.mergedFormerSamples = mergedMetric.formerLine.count;
        diagnostics.mergedNearbySamples = mergedMetric.nearby.count;
        diagnostics.mergedFormerOrientations =
            orientationCount(mergedMetric.orientation.formerLine);
        diagnostics.mergedNearbyOrientations = orientationCount(mergedMetric.orientation.nearby);
        if (hasRequiredShoreCoverage(mergedMetric)) {
            return LocatedShoreWindow{*firstFixture, std::move(mergedMetric)};
        }
    }
    if (firstFixture) {
        return LocatedShoreWindow{*firstFixture, std::move(mergedMetric)};
    }
    return std::nullopt;
}

struct CategoricalRuns {
    int unexplainedWater = 0;
    int material = 0;
    int emittedGround = 0;
    int emittedTop = 0;
    int lithology = 0;
    int biome = 0;

    void include(const CategoricalRuns& other) {
        unexplainedWater = std::max(unexplainedWater, other.unexplainedWater);
        material = std::max(material, other.material);
        emittedGround = std::max(emittedGround, other.emittedGround);
        emittedTop = std::max(emittedTop, other.emittedTop);
        lithology = std::max(lithology, other.lithology);
        biome = std::max(biome, other.biome);
    }
};

CategoricalRuns measureCategoricalWindow(ProbeCache& cache, int64_t line, int64_t alongCenter,
                                         bool vertical) {
    CategoricalRuns longest;
    CategoricalRuns current;
    for (int64_t offset = -CATEGORICAL_HALF_WINDOW; offset <= CATEGORICAL_HALF_WINDOW; ++offset) {
        const AxialPoint firstPosition = axialPoint(vertical, line - 1, alongCenter + offset);
        const AxialPoint secondPosition = axialPoint(vertical, line + 1, alongCenter + offset);
        const worldgen::SurfaceSample& first = cache.surface(firstPosition.x, firstPosition.z);
        const worldgen::SurfaceSample& second = cache.surface(secondPosition.x, secondPosition.z);
        const bool intentionalShore = isIntentionalShoreline(first, second);
        const bool waterJunction = isWaterLevelJunction(first, second);
        const bool hardMaterial = hasHardMaterialConstraint(first) ||
                                  hasHardMaterialConstraint(second) || intentionalShore ||
                                  waterJunction || isTaggedFault(first, second);

        const bool unexplainedWater =
            waterClass(first) != waterClass(second) && !intentionalShore && !waterJunction;
        const bool materialBoundary =
            !hardMaterial && cache.material(firstPosition.x, firstPosition.z) !=
                                 cache.material(secondPosition.x, secondPosition.z);
        const bool groundBoundary =
            !hardMaterial && cache.emittedGround(firstPosition.x, firstPosition.z) !=
                                 cache.emittedGround(secondPosition.x, secondPosition.z);
        const bool topBoundary =
            !hardMaterial && cache.emittedTop(firstPosition.x, firstPosition.z) !=
                                 cache.emittedTop(secondPosition.x, secondPosition.z);
        const bool lithologyBoundary =
            !isTaggedFault(first, second) &&
            first.geology.lithology.primary != second.geology.lithology.primary;
        const bool biomeBoundary = !hardMaterial && first.biome.primary != second.biome.primary;

        current.unexplainedWater = unexplainedWater ? current.unexplainedWater + 1 : 0;
        current.material = materialBoundary ? current.material + 1 : 0;
        current.emittedGround = groundBoundary ? current.emittedGround + 1 : 0;
        current.emittedTop = topBoundary ? current.emittedTop + 1 : 0;
        current.lithology = lithologyBoundary ? current.lithology + 1 : 0;
        current.biome = biomeBoundary ? current.biome + 1 : 0;
        longest.include(current);
    }
    return longest;
}

size_t binTotal(const artifact::OrientationBins& bins) {
    size_t total = 0;
    for (const uint32_t count : bins)
        total += count;
    return total;
}

const char* fieldName(MeasuredField field) {
    switch (field) {
        case MeasuredField::TERRAIN:
            return "terrain";
        case MeasuredField::PRECIPITATION:
            return "precipitation";
        case MeasuredField::LITHOLOGY:
            return "lithology contact";
        case MeasuredField::SHORE_DISTANCE:
            return "shore distance";
        case MeasuredField::BIOME_SUITABILITY:
            return "biome suitability";
        case MeasuredField::COUNT:
            return "invalid";
    }
    return "invalid";
}

} // namespace

TEST_CASE("Final emitted world fields do not inherit former control-line derivative energy",
          "[worldgen][continuity][artifact][final-fields]") {
    std::array<ContinuousMetrics, CONTINUITY_GRID_SPACINGS.size()> bySpacing{};

    for (const uint64_t seed : CONTINUOUS_SEEDS) {
        ProbeCache cache(seed);
        for (const WindowAnchor anchor : CONTINUOUS_WINDOW_ANCHORS) {
            for (size_t spacingIndex = 0; spacingIndex < CONTINUITY_GRID_SPACINGS.size();
                 ++spacingIndex) {
                const int64_t spacing = CONTINUITY_GRID_SPACINGS[spacingIndex];
                const int64_t line = world_coord::floorDiv(anchor.line, spacing) * spacing;
                for (const bool vertical : {true, false}) {
                    const ContinuousMetrics window =
                        measureContinuousWindow(cache, line, anchor.along, vertical);
                    for (size_t field = 0; field < window.size(); ++field)
                        bySpacing[spacingIndex][field].merge(window[field]);
                }
            }
        }
    }

    for (size_t spacingIndex = 0; spacingIndex < CONTINUITY_GRID_SPACINGS.size(); ++spacingIndex) {
        const int64_t spacing = CONTINUITY_GRID_SPACINGS[spacingIndex];
        for (size_t field = 0; field < static_cast<size_t>(MeasuredField::COUNT); ++field) {
            const MeasuredField measured = static_cast<MeasuredField>(field);
            // Canonical shore distance needs a real curved lake, not an
            // incidental depression in this broad learned-field fixture. It
            // is qualified by the dedicated v4 curved-lake regression, which
            // checks exact subcell gradients, supported beds, no dry-bank
            // raising, and shared water authority through cube emission.
            if (measured == MeasuredField::SHORE_DISTANCE)
                continue;
            const ContinuousMetric& metric = bySpacing[spacingIndex][field];
            const double ratio =
                artifact::energyRatio(metric.formerLine.mean(), metric.nearby.mean());
            INFO("ratio " << ratio << " spacing " << spacing << " final field "
                          << fieldName(measured) << " former samples " << metric.formerLine.count
                          << " nearby samples " << metric.nearby.count
                          << " former direct-reference residual energy " << metric.formerLine.mean()
                          << " nearby direct-reference residual energy " << metric.nearby.mean());
            REQUIRE(metric.formerLine.count >= 100);
            REQUIRE(metric.nearby.count >= 400);
            CHECK(ratio >= artifact::DERIVATIVE_RATIO_MINIMUM);
            CHECK(ratio <= artifact::DERIVATIVE_RATIO_MAXIMUM);

            // The compact exact lithology authority stores contact distance
            // at one-eighth-block precision. Its scalar gradient therefore
            // collapses into cardinal bins even when the categorical contact
            // itself is curved. Derivative energy and the categorical run
            // test remain meaningful; orientation does not.
            const size_t formerOrientation = binTotal(metric.orientation.formerLine);
            const size_t nearbyOrientation = binTotal(metric.orientation.nearby);
            if (measured != MeasuredField::LITHOLOGY && formerOrientation >= 64 &&
                nearbyOrientation >= 256) {
                const double orientationRatio =
                    artifact::structuredOrientationRatio(metric.orientation);
                INFO("orientation " << orientationRatio << " spacing " << spacing << " final field "
                                    << fieldName(measured));
                CHECK(orientationRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);
            }
        }
    }
}

TEST_CASE("Canonical signed shoreline distance has no former-grid energy",
          "[worldgen][v4][hydrology][continuity][artifact][shore-distance][absolute]") {
    ProbeCache cache(42);
    const std::vector<NativeShoreTransition> transitions = collectCanonicalShoreTransitions(cache);
    REQUIRE_FALSE(transitions.empty());
    artifact::OrientationBins canonicalShoreOrientations{};
    artifact::OrientationBins expectedShoreOrientations{};
    const auto fixtureLakeRadius = [](double x, double z) {
        const double localX = x + 296.0;
        const double localZ = z + 1'024.0;
        const double warpedX = localX + 13.0 * std::sin(localZ / 71.0);
        const double warpedZ = localZ + 9.0 * std::sin(localX / 61.0);
        return std::hypot(warpedX, warpedZ);
    };
    for (int64_t z = -1'344; z <= -704; z += 4) {
        for (int64_t x = -512; x <= 64; x += 4) {
            const worldgen::SurfaceSample& center = cache.surface(x, z);
            if (!hasUsableShoreDistance(center) ||
                std::abs(center.hydrology.lakeShoreDistance) < 8.0 ||
                std::abs(center.hydrology.lakeShoreDistance) > 64.0) {
                continue;
            }
            const worldgen::SurfaceSample& west = cache.surface(x - 1, z);
            const worldgen::SurfaceSample& east = cache.surface(x + 1, z);
            const worldgen::SurfaceSample& north = cache.surface(x, z - 1);
            const worldgen::SurfaceSample& south = cache.surface(x, z + 1);
            if (!derivativeAllowed(west, east, MeasuredField::SHORE_DISTANCE) ||
                !derivativeAllowed(north, south, MeasuredField::SHORE_DISTANCE))
                continue;
            const double expectedGradientX =
                (fixtureLakeRadius(x + 1, z) - fixtureLakeRadius(x - 1, z)) * 0.5;
            const double expectedGradientZ =
                (fixtureLakeRadius(x, z + 1) - fixtureLakeRadius(x, z - 1)) * 0.5;
            // Keep the measured and analytic histograms paired. A stationary
            // point in the analytic fixture has no defined orientation and
            // therefore cannot provide a meaningful control bin.
            if (std::hypot(expectedGradientX, expectedGradientZ) <= 1.0e-8)
                continue;
            addOrientation(
                canonicalShoreOrientations,
                (east.hydrology.lakeShoreDistance - west.hydrology.lakeShoreDistance) * 0.5,
                (south.hydrology.lakeShoreDistance - north.hydrology.lakeShoreDistance) * 0.5);
            addOrientation(expectedShoreOrientations, expectedGradientX, expectedGradientZ);
        }
    }
    REQUIRE(orientationCount(canonicalShoreOrientations) >= 256);
    REQUIRE(orientationCount(expectedShoreOrientations) ==
            orientationCount(canonicalShoreOrientations));
    CAPTURE(canonicalShoreOrientations, expectedShoreOrientations);
    const double canonicalOrientationRatio = artifact::structuredOrientationRatio(
        artifact::orientationBias(canonicalShoreOrientations, expectedShoreOrientations));
    INFO("canonical signed shoreline orientation ratio " << canonicalOrientationRatio);
    CHECK(canonicalOrientationRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);

    for (size_t spacingIndex = 0; spacingIndex < ABSOLUTE_AUTHORITY_GRID_SPACINGS.size();
         ++spacingIndex) {
        const int64_t spacing = ABSOLUTE_AUTHORITY_GRID_SPACINGS[spacingIndex];
        // The page-backed fixture proves that a native wet-to-dry transition
        // survives exact surface publication at each former grid scale.
        ShoreSearchDiagnostics diagnostics;
        const std::optional<LocatedShoreWindow> located =
            locateCanonicalShoreWindow(cache, transitions, spacing, diagnostics);
        INFO("spacing " << spacing << " native transitions " << diagnostics.nativeTransitions
                        << " control candidates " << diagnostics.controlCandidates
                        << " final transitions " << diagnostics.finalTransitions
                        << " coverage candidates " << diagnostics.coverageCandidates
                        << " closest control distance " << diagnostics.closestControlDistance
                        << " closest vertical distance "
                        << diagnostics.closestVerticalControlDistance
                        << " closest horizontal distance "
                        << diagnostics.closestHorizontalControlDistance << " former samples "
                        << diagnostics.mergedFormerSamples << " nearby samples "
                        << diagnostics.mergedNearbySamples << " former orientations "
                        << diagnostics.mergedFormerOrientations << " nearby orientations "
                        << diagnostics.mergedNearbyOrientations);
        REQUIRE(located.has_value());
        REQUIRE(diagnostics.finalTransitions > 0);
        // Every required former grid contains x=0. Probe the absolute signed
        // shoreline field on that exact line through the bounded basin, then
        // compare it with same-phase controls on both sides. Transition
        // discovery above independently proves the wet-to-dry contour
        // survives native authority and exact surface publication.
        const ContinuousMetric metric =
            measureShoreDistanceWindow(cache, 0, ABSOLUTE_SHORE_ALONG_CENTERS[spacingIndex], true);
        INFO("spacing " << spacing << " exact x=0 signed shoreline samples "
                        << metric.formerLine.count << " nearby samples " << metric.nearby.count);
        CHECK(metric.formerLine.count >= 16);
        CHECK(metric.nearby.count >= 64);
        if (metric.formerLine.count < 16 || metric.nearby.count < 64)
            continue;
        const double derivativeRatio =
            artifact::energyRatio(metric.formerLine.mean(), metric.nearby.mean());
        CHECK(orientationCount(metric.orientation.formerLine) >= 16);
        CHECK(orientationCount(metric.orientation.nearby) >= 64);
        if (orientationCount(metric.orientation.formerLine) < 16 ||
            orientationCount(metric.orientation.nearby) < 64) {
            continue;
        }
        const double orientationRatio = artifact::structuredOrientationRatio(metric.orientation);
        INFO("spacing " << spacing << " signed shoreline derivative ratio " << derivativeRatio
                        << " orientation ratio " << orientationRatio << " former samples "
                        << metric.formerLine.count << " nearby samples " << metric.nearby.count);
        CHECK(derivativeRatio >= artifact::DERIVATIVE_RATIO_MINIMUM);
        CHECK(derivativeRatio <= artifact::DERIVATIVE_RATIO_MAXIMUM);
        CHECK(orientationRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);
    }
}

TEST_CASE("V4 exact surfaces retain coordinate-pure canonical hydrology fields",
          "[worldgen][v4][hydrology][column-plan][regression]") {
    FakeLearnedGenerator generator(42);
    constexpr std::array<SamplePos, 4> positions = {{
        {12'345, -9'327},
        {12'349, -9'323},
        {12'353, -9'319},
        {12'337, -9'341},
    }};

    for (const SamplePos position : positions) {
        const worldgen::SurfaceSample exact = generator.sampleSurface(position.x, position.z);
        const worldgen::SurfaceSample direct = generator.sampleMacroSurface(position.x, position.z);
        const worldgen::HydrologySample& actual = exact.hydrology;
        const worldgen::HydrologySample& expected = direct.hydrology;
        CAPTURE(position.x, position.z, actual.discharge, expected.discharge, actual.sediment,
                expected.sediment, actual.channelDistance, expected.channelDistance,
                actual.channelWidth, expected.channelWidth, actual.channelDepth,
                expected.channelDepth, actual.hydroperiod, expected.hydroperiod);
        CHECK(actual.waterBodyId == expected.waterBodyId);
        CHECK(actual.generatedFluidLevel == expected.generatedFluidLevel);
        CHECK(actual.transitionOwnerKind == expected.transitionOwnerKind);
        CHECK(actual.transitionOwnerId == expected.transitionOwnerId);
        CHECK(actual.ocean == expected.ocean);
        CHECK(actual.lake == expected.lake);
        CHECK(actual.river == expected.river);
        CHECK(actual.lakeBank == expected.lakeBank);
        CHECK(actual.channelBank == expected.channelBank);
        CHECK(actual.endorheic == expected.endorheic);
        CHECK(actual.wetland == expected.wetland);
        CHECK(actual.delta == expected.delta);
        CHECK(actual.estuary == expected.estuary);
        CHECK(actual.brackish == expected.brackish);
        CHECK(actual.waterfall == expected.waterfall);
        CHECK(actual.waterfallAnchor == expected.waterfallAnchor);
        CHECK(actual.perennial == expected.perennial);
        CHECK(actual.ephemeral == expected.ephemeral);
        CHECK(actual.streamOrder == expected.streamOrder);
        CHECK(actual.distributaryCount == expected.distributaryCount);
        // ColumnPlan stores native flow direction with the same signed
        // 16-bit components used by canonical waterfall authority. A unit
        // component margin covers component rounding and re-normalization.
        constexpr double FLOW_COMPONENT_MARGIN =
            1.0 / static_cast<double>(std::numeric_limits<int16_t>::max());
        CHECK(actual.flowDirection.x ==
              Catch::Approx(expected.flowDirection.x).margin(FLOW_COMPONENT_MARGIN));
        CHECK(actual.flowDirection.z ==
              Catch::Approx(expected.flowDirection.z).margin(FLOW_COMPONENT_MARGIN));
        CHECK(actual.surfaceElevation == Catch::Approx(expected.surfaceElevation));
        CHECK(actual.terrainSlope == Catch::Approx(expected.terrainSlope));
        CHECK(actual.waterSurface == Catch::Approx(expected.waterSurface));
        CHECK(actual.discharge == Catch::Approx(expected.discharge));
        CHECK(actual.sediment == Catch::Approx(expected.sediment));
        CHECK(actual.channelDistance == Catch::Approx(expected.channelDistance));
        CHECK(actual.channelWidth == Catch::Approx(expected.channelWidth));
        CHECK(actual.channelDepth == Catch::Approx(expected.channelDepth));
        CHECK(actual.channelGradient == Catch::Approx(expected.channelGradient));
        CHECK(actual.erosionDepth == Catch::Approx(expected.erosionDepth));
        CHECK(actual.lakeDepth == Catch::Approx(expected.lakeDepth));
        CHECK(actual.groundwaterHead == Catch::Approx(expected.groundwaterHead));
        CHECK(actual.hydroperiod == Catch::Approx(expected.hydroperiod));
        CHECK(actual.groundwaterRechargeMm == Catch::Approx(expected.groundwaterRechargeMm));
        CHECK(actual.lakeShoreDistance == Catch::Approx(expected.lakeShoreDistance));
        CHECK(actual.shoreWaterSurface == Catch::Approx(expected.shoreWaterSurface));
        CHECK(actual.lakeBankTarget == Catch::Approx(expected.lakeBankTarget));
        CHECK(actual.lakeBankInfluence == Catch::Approx(expected.lakeBankInfluence));
        CHECK(actual.lakeAreaSquareKilometers == Catch::Approx(expected.lakeAreaSquareKilometers));
        CHECK(actual.lakeVolumeCubicMeters == Catch::Approx(expected.lakeVolumeCubicMeters));
        CHECK(actual.lakeRunoffMmSquareKilometers ==
              Catch::Approx(expected.lakeRunoffMmSquareKilometers));
        CHECK(actual.lakeLossMm == Catch::Approx(expected.lakeLossMm));
        CHECK(actual.lakeOverflowMmSquareKilometers ==
              Catch::Approx(expected.lakeOverflowMmSquareKilometers));
        CHECK(actual.lakeSpillSurface == Catch::Approx(expected.lakeSpillSurface));
        CHECK(actual.baseflow == Catch::Approx(expected.baseflow));
        CHECK(actual.precipitationSeasonality == Catch::Approx(expected.precipitationSeasonality));
        CHECK(actual.waterfallTop == Catch::Approx(expected.waterfallTop));
        CHECK(actual.waterfallBottom == Catch::Approx(expected.waterfallBottom));
        CHECK(actual.waterfallWidth == Catch::Approx(expected.waterfallWidth));
    }
}

TEST_CASE("V4 exact surfaces retain coordinate-pure ecology at former control lines",
          "[worldgen][v4][ecology][column-plan][regression]") {
    FakeLearnedGenerator generator(42);
    // These points span both axes of the compact eight-block lattice at the
    // first continuity fixture. The learned authority is coordinate-pure, so
    // no physical, climate, soil, or suitability field may change merely
    // because an exact sample falls on a former control line.
    constexpr std::array<SamplePos, 8> positions = {{
        {12'343, -9'327},
        {12'344, -9'327},
        {12'345, -9'327},
        {12'352, -9'327},
        {12'345, -9'329},
        {12'345, -9'328},
        {12'345, -9'327},
        {12'345, -9'320},
    }};

    for (const SamplePos position : positions) {
        const worldgen::SurfaceSample exact = generator.sampleSurface(position.x, position.z);
        const worldgen::SurfaceSample direct = generator.sampleMacroSurface(position.x, position.z);
        CAPTURE(position.x, position.z, exact.terrainHeight, direct.terrainHeight,
                exact.climate.annualPrecipitationMm, direct.climate.annualPrecipitationMm,
                exact.climate.temperatureC, direct.climate.temperatureC, exact.slope, direct.slope,
                exact.soil.moisture, direct.soil.moisture, exact.soil.fertility,
                direct.soil.fertility, exact.soil.drainage, direct.soil.drainage,
                exact.soil.waterTable, direct.soil.waterTable);
        CHECK(exact.terrainHeight == Catch::Approx(direct.terrainHeight));
        CHECK(exact.slope == Catch::Approx(direct.slope));
        CHECK(exact.climate.temperatureC == Catch::Approx(direct.climate.temperatureC));
        CHECK(exact.climate.temperatureVariabilityC ==
              Catch::Approx(direct.climate.temperatureVariabilityC));
        CHECK(exact.climate.annualPrecipitationMm ==
              Catch::Approx(direct.climate.annualPrecipitationMm));
        CHECK(exact.climate.precipitationCoefficientOfVariation ==
              Catch::Approx(direct.climate.precipitationCoefficientOfVariation));
        CHECK(exact.climate.lapseRateCPerMeter == Catch::Approx(direct.climate.lapseRateCPerMeter));
        CHECK(exact.climate.potentialEvapotranspirationMm ==
              Catch::Approx(direct.climate.potentialEvapotranspirationMm));
        CHECK(exact.climate.aridity == Catch::Approx(direct.climate.aridity));
        CHECK(exact.climate.relativeHumidity == Catch::Approx(direct.climate.relativeHumidity));
        CHECK(exact.soil.moisture == Catch::Approx(direct.soil.moisture));
        CHECK(exact.soil.fertility == Catch::Approx(direct.soil.fertility));
        CHECK(exact.soil.drainage == Catch::Approx(direct.soil.drainage));
        CHECK(exact.soil.waterTable == Catch::Approx(direct.soil.waterTable));
        CHECK(exact.biome.primary == direct.biome.primary);
        CHECK(exact.biome.secondary == direct.biome.secondary);
        CHECK(exact.biome.transition == Catch::Approx(direct.biome.transition));
        for (size_t biome = 0; biome < exact.suitability.scores.size(); ++biome) {
            CHECK(exact.suitability.scores[biome] ==
                  Catch::Approx(direct.suitability.scores[biome]));
        }
    }
}

TEST_CASE("V4 far habitats retain exact canonical routing fields",
          "[worldgen][v4][hydrology][far-habitat][regression]") {
    FakeLearnedGenerator generator(42);
    constexpr std::array<ColumnPos, 4> positions = {{
        {12'345, -9'327},
        {12'349, -9'323},
        {12'353, -9'319},
        {12'337, -9'341},
    }};
    std::array<worldgen::SurfaceSample, positions.size()> habitats{};
    generator.sampleFarHabitatPoints(positions, habitats);

    constexpr double FLOW_COMPONENT_MARGIN =
        1.0 / static_cast<double>(std::numeric_limits<int16_t>::max());
    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        const worldgen::HydrologySample& exact =
            generator.sampleSurface(position.x, position.z).hydrology;
        const worldgen::HydrologySample& habitat = habitats[index].hydrology;
        CAPTURE(position.x, position.z, habitat.discharge, exact.discharge, habitat.sediment,
                exact.sediment, habitat.channelDistance, exact.channelDistance,
                habitat.channelWidth, exact.channelWidth, habitat.channelDepth, exact.channelDepth,
                habitat.hydroperiod, exact.hydroperiod);
        CHECK(habitat.waterBodyId == exact.waterBodyId);
        CHECK(habitat.ocean == exact.ocean);
        CHECK(habitat.lake == exact.lake);
        CHECK(habitat.river == exact.river);
        CHECK(habitat.wetland == exact.wetland);
        CHECK(habitat.delta == exact.delta);
        CHECK(habitat.waterfall == exact.waterfall);
        CHECK(habitat.perennial == exact.perennial);
        CHECK(habitat.ephemeral == exact.ephemeral);
        CHECK(habitat.streamOrder == exact.streamOrder);
        CHECK(habitat.distributaryCount == exact.distributaryCount);
        CHECK(habitat.flowDirection.x ==
              Catch::Approx(exact.flowDirection.x).margin(FLOW_COMPONENT_MARGIN));
        CHECK(habitat.flowDirection.z ==
              Catch::Approx(exact.flowDirection.z).margin(FLOW_COMPONENT_MARGIN));
        CHECK(habitat.discharge == Catch::Approx(exact.discharge));
        CHECK(habitat.sediment == Catch::Approx(exact.sediment));
        CHECK(habitat.channelDistance == Catch::Approx(exact.channelDistance));
        CHECK(habitat.channelWidth == Catch::Approx(exact.channelWidth));
        CHECK(habitat.channelDepth == Catch::Approx(exact.channelDepth));
        CHECK(habitat.channelGradient == Catch::Approx(exact.channelGradient));
        CHECK(habitat.erosionDepth == Catch::Approx(exact.erosionDepth));
        CHECK(habitat.baseflow == Catch::Approx(exact.baseflow));
        CHECK(habitat.precipitationSeasonality == Catch::Approx(exact.precipitationSeasonality));
        CHECK(habitat.groundwaterRechargeMm == Catch::Approx(exact.groundwaterRechargeMm));
        CHECK(habitat.groundwaterHead == Catch::Approx(exact.groundwaterHead));
        CHECK(habitat.hydroperiod == Catch::Approx(exact.hydroperiod));
    }
}

TEST_CASE("Final emitted categories do not run along former control lines",
          "[worldgen][continuity][artifact][final-fields][blocks]") {
    std::array<CategoricalRuns, artifact::FORMER_GRID_SPACINGS.size()> bySpacing{};
    for (const uint64_t seed : CATEGORICAL_SEEDS) {
        ProbeCache cache(seed);
        for (const WindowAnchor anchor : WINDOW_ANCHORS) {
            for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
                 ++spacingIndex) {
                const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
                const int64_t line = world_coord::floorDiv(anchor.line, spacing) * spacing;
                for (const bool vertical : {true, false}) {
                    bySpacing[spacingIndex].include(
                        measureCategoricalWindow(cache, line, anchor.along, vertical));
                }
            }
        }
    }

    for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
         ++spacingIndex) {
        const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
        const CategoricalRuns& runs = bySpacing[spacingIndex];
        INFO("spacing " << spacing << " unexplained water " << runs.unexplainedWater << " material "
                        << runs.material << " emitted ground " << runs.emittedGround
                        << " emitted top " << runs.emittedTop << " lithology " << runs.lithology
                        << " biome " << runs.biome);
        REQUIRE(runs.unexplainedWater <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.material <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.emittedGround <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.emittedTop <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.lithology <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.biome <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
    }
}

TEST_CASE("Column plans share exact final fields across negative and positive faces",
          "[worldgen][continuity][artifact][final-fields][seam]") {
    FakeLearnedGenerator generator(764'891);
    constexpr std::array<int64_t, 7> boundaries = {-64, -32, -16, 0, 16, 32, 64};
    constexpr std::array<int64_t, 8> alongPositions = {-65, -33, -17, -1, 0, 15, 31, 63};

    for (const int64_t boundary : boundaries) {
        const int64_t highColumn =
            world_coord::floorDiv(boundary, static_cast<int64_t>(CHUNK_EDGE));
        for (const int64_t along : alongPositions) {
            CAPTURE(boundary, along);
            const int64_t alongColumn =
                world_coord::floorDiv(along, static_cast<int64_t>(CHUNK_EDGE));
            const int localAlong = Chunk::worldToLocal(along);

            const auto west = generator.getColumnPlan(ColumnPos{highColumn - 1, alongColumn});
            const auto east = generator.getColumnPlan(ColumnPos{highColumn, alongColumn});
            const FinalSurfaceDigest westDigest = digest(west->sample(CHUNK_EDGE, localAlong));
            const FinalSurfaceDigest eastDigest = digest(east->sample(0, localAlong));
            INFO("west/east first difference " << firstDigestDifference(westDigest, eastDigest));
            REQUIRE(westDigest == eastDigest);

            const auto north = generator.getColumnPlan(ColumnPos{alongColumn, highColumn - 1});
            const auto south = generator.getColumnPlan(ColumnPos{alongColumn, highColumn});
            const FinalSurfaceDigest northDigest = digest(north->sample(localAlong, CHUNK_EDGE));
            const FinalSurfaceDigest southDigest = digest(south->sample(localAlong, 0));
            INFO("north/south first difference "
                 << firstDigestDifference(northDigest, southDigest));
            REQUIRE(northDigest == southDigest);
        }
    }
}

TEST_CASE("Final fields are identical after reverse-order macro cache eviction",
          "[worldgen][continuity][artifact][final-fields][determinism]") {
    constexpr std::array<SamplePos, 16> positions = {
        SamplePos{-8193, -2049}, SamplePos{-8192, -2048}, SamplePos{-8191, -2047},
        SamplePos{-2049, -65},   SamplePos{-2048, -64},   SamplePos{-2047, -63},
        SamplePos{-17, -17},     SamplePos{-16, -16},     SamplePos{-1, -1},
        SamplePos{0, 0},         SamplePos{15, 15},       SamplePos{16, 16},
        SamplePos{63, 65},       SamplePos{2048, -2048},  SamplePos{8192, 2048},
        SamplePos{8193, 8193},
    };

    FakeLearnedGenerator generator(764'891);
    std::array<FinalSurfaceDigest, positions.size()> forward;
    for (size_t index = 0; index < positions.size(); ++index) {
        const SamplePos position = positions[index];
        const worldgen::SurfaceSample sample = generator.sampleSurface(position.x, position.z);
        forward[index] = digest(sample, generator.surfaceMaterialAt(position.x, position.z));
    }

    generator.clearMacroCaches();
    for (size_t reverseIndex = positions.size(); reverseIndex > 0; --reverseIndex) {
        const size_t index = reverseIndex - 1;
        const SamplePos position = positions[index];
        CAPTURE(position.x, position.z);
        const worldgen::SurfaceSample sample = generator.sampleSurface(position.x, position.z);
        const FinalSurfaceDigest reverse =
            digest(sample, generator.surfaceMaterialAt(position.x, position.z));
        INFO("first difference " << firstDigestDifference(reverse, forward[index]));
        INFO("channel distance " << reverse.channelDistance << " / "
                                 << forward[index].channelDistance);
        REQUIRE(reverse == forward[index]);
    }
}

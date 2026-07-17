#include "common/random.hpp"
#include "render/far_terrain.hpp"
#include "world/artifact_analysis.hpp"
#include "world/chunk_generator.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace artifact = worldgen::artifact_analysis;

const char* biomeName(Biome biome) {
    constexpr std::array names = {
        "deep_ocean",
        "ocean",
        "plains",
        "forest",
        "taiga",
        "desert",
        "extreme_hills",
        "swamp",
        "mushroom_island",
        "ice_spikes",
        "beach",
        "river",
        "birch_forest",
        "flower_field",
        "savanna",
        "tropical_rainforest",
        "temperate_rainforest",
        "shrubland",
        "steppe",
        "cold_desert",
        "badlands",
        "tundra",
        "alpine",
        "mangrove",
        "frozen_ocean",
        "volcanic_barren",
        "glacier",
        "montane_grassland",
        "flooded_grassland",
        "mediterranean_woodland",
        "temperate_conifer_forest",
        "tropical_conifer_forest",
        "tropical_dry_forest",
    };
    static_assert(names.size() == static_cast<size_t>(Biome::COUNT));
    const size_t index = static_cast<size_t>(biome);
    return index < names.size() ? names[index] : "invalid";
}

uint64_t cubeHash(const Chunk& cube) {
    uint64_t hash = hash64(static_cast<uint64_t>(cube.chunkX));
    hash ^= hash64(static_cast<uint64_t>(static_cast<uint32_t>(cube.chunkY)));
    hash ^= hash64(static_cast<uint64_t>(cube.chunkZ));
    for (int y = 0; y < CHUNK_EDGE; ++y) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            for (int x = 0; x < CHUNK_EDGE; ++x) {
                const uint64_t address = static_cast<uint64_t>(Chunk::index(x, y, z));
                const uint64_t block = static_cast<uint8_t>(cube.getBlock(x, y, z));
                const uint64_t fluid = cube.getFluidState(x, y, z).packed();
                hash = hash64(hash ^ (address << 16) ^ (block << 8) ^ fluid);
            }
        }
    }
    return hash;
}

double elapsedMilliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double percentile95(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index =
        static_cast<size_t>(std::ceil(static_cast<double>(values.size()) * 0.95) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

const char* rockName(worldgen::RockType rock) {
    switch (rock) {
        case worldgen::RockType::GRANITE:
            return "granite";
        case worldgen::RockType::BASALT:
            return "basalt";
        case worldgen::RockType::LIMESTONE:
            return "limestone";
        case worldgen::RockType::SANDSTONE:
            return "sandstone";
        case worldgen::RockType::VOLCANIC:
            return "volcanic";
    }
    return "invalid";
}

void writeWaterBodyId(std::ostream& output, worldgen::WaterBodyId identity) {
    output << "\"0x" << std::hex << identity << std::dec << '"';
}

void writeMaterialPalette(std::ostream& output,
                          const worldgen::surface_material::SurfaceMaterialPalette& palette) {
    output << '[';
    for (size_t index = 0; index < palette.count; ++index) {
        const auto& entry = palette.entries[index];
        output << "{\"material\": " << static_cast<unsigned>(entry.material)
               << ", \"weight\": " << static_cast<unsigned>(entry.weight) << '}';
        if (index + 1 != palette.count) output << ", ";
    }
    output << ']';
}

struct FootprintMetric {
    worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1;
    worldgen::SurfaceSample sample;
    double blockWindowMinimum = 0.0;
    double blockWindowMaximum = 0.0;
    bool finite = false;
    bool topologyMatchesBlock = false;
    bool waterBodyMatchesBlock = false;
};

bool matchingWaterTopology(const worldgen::HydrologySample& first,
                           const worldgen::HydrologySample& second) {
    return first.ocean == second.ocean && first.river == second.river &&
           first.lake == second.lake && first.delta == second.delta &&
           first.waterfall == second.waterfall;
}

std::vector<FootprintMetric> measureFootprints(ChunkGenerator& generator, int64_t x, int64_t z) {
    constexpr std::array footprints = {
        worldgen::SurfaceFootprint::BLOCK_1,  worldgen::SurfaceFootprint::BLOCK_2,
        worldgen::SurfaceFootprint::BLOCK_4,  worldgen::SurfaceFootprint::BLOCK_8,
        worldgen::SurfaceFootprint::BLOCK_16,
    };
    const worldgen::SurfaceSample block =
        generator.sampleSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
    std::vector<FootprintMetric> result;
    result.reserve(footprints.size());
    for (worldgen::SurfaceFootprint footprint : footprints) {
        const int width = worldgen::surfaceFootprintWidth(footprint);
        const int64_t minimumX = x - width / 2;
        const int64_t minimumZ = z - width / 2;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (int localZ = 0; localZ < width; ++localZ) {
            for (int localX = 0; localX < width; ++localX) {
                const worldgen::SurfaceSample exact = generator.sampleSurface(
                    minimumX + localX, minimumZ + localZ, worldgen::SurfaceFootprint::BLOCK_1);
                minimum = std::min(minimum, exact.terrainHeight);
                maximum = std::max(maximum, exact.terrainHeight);
            }
        }
        worldgen::SurfaceSample sample = generator.sampleSurface(x, z, footprint);
        result.push_back({
            .footprint = footprint,
            .sample = sample,
            .blockWindowMinimum = minimum,
            .blockWindowMaximum = maximum,
            .finite = std::isfinite(sample.terrainHeight) && std::isfinite(sample.waterSurface) &&
                      std::isfinite(sample.geology.lithology.transition) &&
                      std::isfinite(sample.geology.lithology.contactDistance),
            .topologyMatchesBlock = matchingWaterTopology(block.hydrology, sample.hydrology),
            .waterBodyMatchesBlock = block.hydrology.waterBodyId == sample.hydrology.waterBodyId,
        });
    }
    return result;
}

struct ContinuityProbe {
    worldgen::SurfaceSample surface;
    BlockType dominantMaterial = BlockType::AIR;
};

using ContinuityProbeCache = std::map<std::pair<int64_t, int64_t>, ContinuityProbe>;

const ContinuityProbe& continuityProbe(ChunkGenerator& generator, ContinuityProbeCache& cache,
                                       int64_t x, int64_t z) {
    const std::pair key{x, z};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    ContinuityProbe probe;
    probe.surface = generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
    probe.dominantMaterial = generator.farSurfaceMaterialAt(x, z);
    return cache.emplace(key, std::move(probe)).first->second;
}

bool hasHardSurfaceMaterialConstraint(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.ocean || sample.hydrology.lake || sample.hydrology.river ||
           sample.hydrology.waterfall || sample.hydrology.delta ||
           sample.geology.volcanicActivity > 0.52 ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::CLIFF) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SCREE) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SNOWFIELD) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::GLACIER);
}

template <typename Predicate>
int maximumBoundaryRun(int64_t centerZ, Predicate&& boundaryAt) {
    int longest = 0;
    int current = 0;
    for (int64_t offset = -artifact::CATEGORICAL_HALF_WINDOW;
         offset <= artifact::CATEGORICAL_HALF_WINDOW; ++offset) {
        if (boundaryAt(centerZ + offset)) {
            longest = std::max(longest, ++current);
        } else {
            current = 0;
        }
    }
    return longest;
}

struct FormerGridMetric {
    int spacing = 0;
    int64_t lineX = 0;
    int64_t centerZ = 0;
    std::array<double, 3> boundaryEnergy{};
    std::array<double, 3> nearbyEnergy{};
    std::array<double, 3> energyRatios{};
    std::array<uint32_t, 8> formerLineOrientation{};
    std::array<uint32_t, 8> nearbyOrientation{};
    std::array<double, 8> orientationBias{};
    double structuredOrientationRatio = 0.0;
    int lakeBoundaryRun = 0;
    int materialBoundaryRun = 0;
    int untaggedLithologyBoundaryRun = 0;
};

FormerGridMetric measureFormerGridBoundary(ChunkGenerator& generator,
                                           worldgen::MacroGenerationSampler& macro, int spacing) {
    FormerGridMetric result;
    result.spacing = spacing;
    result.lineX = spacing;
    result.centerZ = spacing / 2;

    ContinuityProbeCache cache;
    for (size_t index = 0; index < artifact::FIELDS.size(); ++index) {
        result.boundaryEnergy[index] = artifact::derivativeEnergy(macro, artifact::FIELDS[index],
                                                                  result.lineX, result.centerZ);
        result.nearbyEnergy[index] = artifact::nearbyDerivativeEnergy(
            macro, artifact::FIELDS[index], result.lineX, result.centerZ);
        result.energyRatios[index] =
            artifact::energyRatio(result.boundaryEnergy[index], result.nearbyEnergy[index]);
        const artifact::OrientationHistogram histogram = artifact::orientationHistogram(
            macro, artifact::FIELDS[index], result.lineX, result.centerZ);
        for (size_t bin = 0; bin < histogram.formerLine.size(); ++bin) {
            result.formerLineOrientation[bin] += histogram.formerLine[bin];
            result.nearbyOrientation[bin] += histogram.nearby[bin];
        }
    }
    result.orientationBias =
        artifact::orientationBias(result.formerLineOrientation, result.nearbyOrientation);
    result.structuredOrientationRatio =
        artifact::structuredOrientationRatio(result.orientationBias);

    result.lakeBoundaryRun = maximumBoundaryRun(result.centerZ, [&](int64_t z) {
        const auto& left = continuityProbe(generator, cache, result.lineX - 1, z).surface;
        const auto& right = continuityProbe(generator, cache, result.lineX + 1, z).surface;
        if (left.hydrology.river || right.hydrology.river || left.hydrology.waterfall ||
            right.hydrology.waterfall || left.hydrology.delta || right.hydrology.delta) {
            return false;
        }
        const bool touchesLake = left.hydrology.lake || right.hydrology.lake;
        return touchesLake && (left.hydrology.lake != right.hydrology.lake ||
                               left.hydrology.waterBodyId != right.hydrology.waterBodyId);
    });
    result.materialBoundaryRun = maximumBoundaryRun(result.centerZ, [&](int64_t z) {
        const auto& left = continuityProbe(generator, cache, result.lineX - 1, z);
        const auto& right = continuityProbe(generator, cache, result.lineX + 1, z);
        const bool constrained = hasHardSurfaceMaterialConstraint(left.surface) ||
                                 hasHardSurfaceMaterialConstraint(right.surface);
        return !constrained && left.dominantMaterial != right.dominantMaterial;
    });
    result.untaggedLithologyBoundaryRun = maximumBoundaryRun(result.centerZ, [&](int64_t z) {
        const auto& left = continuityProbe(generator, cache, result.lineX - 1, z).surface.geology;
        const auto& right = continuityProbe(generator, cache, result.lineX + 1, z).surface.geology;
        const bool taggedFault = left.faultStrength > 0.45 || right.faultStrength > 0.45 ||
                                 left.boundary == worldgen::PlateBoundary::TRANSFORM ||
                                 right.boundary == worldgen::PlateBoundary::TRANSFORM;
        return !taggedFault && left.lithology.primary != right.lithology.primary;
    });
    return result;
}

struct FormerGridSummary {
    std::array<uint32_t, 8> formerLineOrientation{};
    std::array<uint32_t, 8> nearbyOrientation{};
    std::array<double, 8> orientationBias{};
    double formerLineStructuredBiasRatio = 0.0;
    std::array<uint32_t, 8> globalOrientation{};
    double globalStructuredOrientationRatio = 0.0;
    double minimumEnergyRatio = std::numeric_limits<double>::infinity();
    double maximumEnergyRatio = 0.0;
    int lakeBoundaryRun = 0;
    int materialBoundaryRun = 0;
    int untaggedLithologyBoundaryRun = 0;
    bool derivativeEnergyWithinLimit = true;
    bool orientationWithinLimit = true;
    bool categoricalRunsWithinLimit = true;
};

FormerGridSummary summarizeFormerGridMetrics(const std::vector<FormerGridMetric>& metrics) {
    FormerGridSummary result;
    for (const FormerGridMetric& metric : metrics) {
        for (double ratio : metric.energyRatios) {
            result.minimumEnergyRatio = std::min(result.minimumEnergyRatio, ratio);
            result.maximumEnergyRatio = std::max(result.maximumEnergyRatio, ratio);
            result.derivativeEnergyWithinLimit &= ratio >= artifact::DERIVATIVE_RATIO_MINIMUM &&
                                                  ratio <= artifact::DERIVATIVE_RATIO_MAXIMUM;
        }
        for (size_t bin = 0; bin < result.formerLineOrientation.size(); ++bin) {
            result.formerLineOrientation[bin] += metric.formerLineOrientation[bin];
            result.nearbyOrientation[bin] += metric.nearbyOrientation[bin];
        }
        result.lakeBoundaryRun = std::max(result.lakeBoundaryRun, metric.lakeBoundaryRun);
        result.materialBoundaryRun =
            std::max(result.materialBoundaryRun, metric.materialBoundaryRun);
        result.untaggedLithologyBoundaryRun =
            std::max(result.untaggedLithologyBoundaryRun, metric.untaggedLithologyBoundaryRun);
    }
    if (metrics.empty()) result.minimumEnergyRatio = 1.0;
    result.orientationBias =
        artifact::orientationBias(result.formerLineOrientation, result.nearbyOrientation);
    result.formerLineStructuredBiasRatio =
        artifact::structuredOrientationRatio(result.orientationBias);
    result.orientationWithinLimit =
        result.formerLineStructuredBiasRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT;
    result.categoricalRunsWithinLimit =
        result.lakeBoundaryRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT &&
        result.materialBoundaryRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT &&
        result.untaggedLithologyBoundaryRun <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT;
    return result;
}

size_t diskColumnCount(int radius) {
    size_t result = 0;
    for (int z = -radius; z <= radius; ++z) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + z * z <= radius * radius) ++result;
        }
    }
    return result;
}

template <typename Value>
void writeBins(std::ostream& output, const std::array<Value, 8>& bins) {
    output << '[';
    for (size_t index = 0; index < bins.size(); ++index) {
        output << bins[index];
        if (index + 1 != bins.size()) output << ", ";
    }
    output << ']';
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: rycraft_worldgen_inspect [seed] [sample_x sample_z]\n";
        return 0;
    }
    uint32_t seed = 42;
    if (const char* environmentSeed = std::getenv("RYCRAFT_WORLD_SEED")) {
        seed = static_cast<uint32_t>(std::strtoull(environmentSeed, nullptr, 0));
    }
    if (argc > 1) seed = static_cast<uint32_t>(std::strtoull(argv[1], nullptr, 0));
    std::optional<std::pair<int64_t, int64_t>> requestedSample;
    if (argc >= 4) {
        requestedSample = std::pair{static_cast<int64_t>(std::strtoll(argv[2], nullptr, 0)),
                                    static_cast<int64_t>(std::strtoll(argv[3], nullptr, 0))};
    }

    worldgen::MacroGenerationSampler macro(seed);
    auto generatorOwner = std::make_shared<ChunkGenerator>(seed);
    ChunkGenerator& generator = *generatorOwner;
    std::map<std::string, std::pair<int64_t, int64_t>> sightings;
    std::array<std::optional<std::pair<int64_t, int64_t>>, static_cast<size_t>(Biome::COUNT)>
        biomeSightings{};
    std::vector<std::pair<int64_t, int64_t>> probes;
    if (seed == 42) {
        // Stable acceptance-route landmarks keep the default diagnostic fast
        // and make every playtest capture reproducible.
        probes = {{-518996, -385073}, {-26355, 29672},  {3264, 480},      {-8235, 2976},
                  {-8240, 3088},      {-29568, 13328},  {-20809, -26567}, {-8348, 2281},
                  {-23904, 0},        {-10066, -23060}, {-14200, 27190},  {-13248, -38352},
                  {-12288, 2653},     {-12352, 2653},   {27037, -129},    {-27297, -17021},
                  {-24841, -9553},    {-9003, 21417},   {-4654, 22202},   {9429, 8254},
                  {15570, 6095}};
        probes.emplace_back(-81'792, 126'976);
    }
    const int searchSamples = seed == 42 ? 0 : 192;
    for (int index = 0; index < searchSamples; ++index) {
        probes.emplace_back(static_cast<int64_t>((index * 7'919) % 65'521) - 32'760,
                            static_cast<int64_t>((index * 15'485 + 7'721) % 65'519) - 32'759);
    }
    const auto searchStart = Clock::now();
    for (const auto& [x, z] : probes) {
        const worldgen::GeologySample geology = macro.sampleGeology(x, z);
        const worldgen::SurfaceSample macroSurface = macro.sampleSurface(x, z);
        const worldgen::SurfaceSample surface = generator.sampleFarSurface(x, z);
        const worldgen::HydrologySample& hydrology = surface.hydrology;
        const double east = generator.sampleFarTerrainHeight(x + 16, z);
        const double west = generator.sampleFarTerrainHeight(x - 16, z);
        const double north = generator.sampleFarTerrainHeight(x, z + 16);
        const double south = generator.sampleFarTerrainHeight(x, z - 16);
        const double slope = std::hypot((east - west) / 32.0, (north - south) / 32.0);
        auto record = [&](const char* name, bool present) {
            if (present && !sightings.contains(name)) sightings.emplace(name, std::pair{x, z});
        };
        auto recordBiome = [&](Biome biome) {
            auto& sighting = biomeSightings[static_cast<size_t>(biome)];
            if (!sighting.has_value()) sighting = std::pair{x, z};
        };
        recordBiome(surface.biome.primary);
        if (surface.biome.secondary != surface.biome.primary && surface.biome.transition >= 0.15) {
            recordBiome(surface.biome.secondary);
        }
        record("mountain", surface.terrainHeight > 145.0 && geology.uplift > 0.35);
        record("cliff", slope > 0.75);
        record("canyon", worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::CANYON));
        record("river", hydrology.river);
        record("confluence",
               hydrology.river && hydrology.streamOrder >= 2 && hydrology.discharge > 250.0);
        record("lake", hydrology.lake);
        record("endorheic_lake", hydrology.lake && hydrology.endorheic);
        record("waterfall", hydrology.waterfall);
        record("delta", hydrology.delta);
        record("volcano", surface.terrainHeight - macroSurface.terrainHeight > 12.0 &&
                              geology.volcanicActivity > 0.35);
        record("oceanic_island", geology.crust == worldgen::CrustType::OCEANIC &&
                                     surface.terrainHeight > SEA_LEVEL + 2.0 && !hydrology.ocean);
        record("biome_transition",
               surface.biome.primary != surface.biome.secondary && surface.biome.transition > 0.40);
        record("snow_peak",
               worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::SNOWFIELD) &&
                   (worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::ALPINE_ZONE) ||
                    worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::EXPOSED_PEAK)));
        const auto biomeSuitability = [&](Biome biome) {
            return surface.suitability.scores[static_cast<size_t>(biome)];
        };
        const float forestHabitat = std::max(
            {biomeSuitability(Biome::FOREST), biomeSuitability(Biome::BIRCH_FOREST),
             biomeSuitability(Biome::TROPICAL_RAINFOREST),
             biomeSuitability(Biome::TEMPERATE_RAINFOREST), biomeSuitability(Biome::MANGROVE),
             biomeSuitability(Biome::MEDITERRANEAN_WOODLAND),
             biomeSuitability(Biome::TEMPERATE_CONIFER_FOREST),
             biomeSuitability(Biome::TROPICAL_CONIFER_FOREST),
             biomeSuitability(Biome::TROPICAL_DRY_FOREST)});
        record("dense_flora", forestHabitat > 0.45F && surface.soil.fertility > 0.50 &&
                                  surface.soil.moisture > 0.52 && surface.slope < 0.45);
        const double waterDepth = hydrology.ocean || hydrology.lake || hydrology.river
                                      ? std::max(0.0, surface.waterSurface - surface.terrainHeight)
                                      : 0.0;
        record("deep_fish_water",
               (hydrology.ocean || hydrology.lake || hydrology.river) && waterDepth >= 4.0);
    }
    const double searchMs = elapsedMilliseconds(searchStart);

    constexpr std::array<ChunkPos, 8> route = {
        ChunkPos{0, 4, 0},    ChunkPos{1, 4, 0},   ChunkPos{-1, 4, -1},   ChunkPos{16, 5, -8},
        ChunkPos{-24, 2, 19}, ChunkPos{64, 8, 64}, ChunkPos{-96, -2, 48}, ChunkPos{128, 12, -128},
    };

    generator.clearMacroCaches();
    std::vector<std::pair<ChunkPos, uint64_t>> forward;
    std::vector<double> coldCubeMilliseconds;
    const auto coldStart = Clock::now();
    for (ChunkPos pos : route) {
        const auto cubeStart = Clock::now();
        Chunk cube(pos);
        generator.generateCube(cube);
        coldCubeMilliseconds.push_back(elapsedMilliseconds(cubeStart));
        forward.emplace_back(pos, cubeHash(cube));
    }
    const double coldMs = elapsedMilliseconds(coldStart);

    std::vector<std::pair<ChunkPos, uint64_t>> reverse;
    std::vector<double> warmCubeMilliseconds;
    const auto warmStart = Clock::now();
    for (auto iterator = route.rbegin(); iterator != route.rend(); ++iterator) {
        const auto cubeStart = Clock::now();
        Chunk cube(*iterator);
        generator.generateCube(cube);
        warmCubeMilliseconds.push_back(elapsedMilliseconds(cubeStart));
        reverse.emplace_back(*iterator, cubeHash(cube));
    }
    const double warmMs = elapsedMilliseconds(warmStart);
    const auto positionLess = [](const auto& left, const auto& right) {
        if (left.first.x != right.first.x) return left.first.x < right.first.x;
        if (left.first.y != right.first.y) return left.first.y < right.first.y;
        return left.first.z < right.first.z;
    };
    std::sort(forward.begin(), forward.end(), positionLess);
    std::sort(reverse.begin(), reverse.end(), positionLess);
    const bool deterministic = forward == reverse;
    uint64_t routeHash = 0;
    for (const auto& [pos, hash] : forward)
        routeHash = hash64(routeHash ^ hash);

    worldgen::MacroGenerationSampler basinBenchmark(seed);
    const int64_t basinX = seed == 42 ? -8235 : 0;
    const int64_t basinZ = seed == 42 ? 2976 : 0;
    const auto coldBasinStart = Clock::now();
    const worldgen::HydrologySample coldBasin = basinBenchmark.sampleHydrology(basinX, basinZ);
    const double coldBasinMs = elapsedMilliseconds(coldBasinStart);
    const auto warmBasinStart = Clock::now();
    const worldgen::HydrologySample warmBasin = basinBenchmark.sampleHydrology(basinX, basinZ);
    const double warmBasinMs = elapsedMilliseconds(warmBasinStart);

    const worldgen::SurfaceSample origin = generator.sampleFarSurface(0, 0);

    std::array<double, 5> farTileMilliseconds{};
    std::array<size_t, 5> farTileBytes{};
    std::array<uint64_t, 5> farTileHashes{};
    constexpr std::array<FarTerrainStep, 5> FAR_STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    FarTerrainSource farSource = FarTerrainMesher::generatorGeometrySource(generatorOwner);
    for (size_t index = 0; index < FAR_STEPS.size(); ++index) {
        const auto start = Clock::now();
        const std::shared_ptr<const FarTerrainMesh> tile =
            FarTerrainMesher::build({8, -5, FAR_STEPS[index]}, farSource);
        farTileMilliseconds[index] = elapsedMilliseconds(start);
        farTileBytes[index] = tile->byteSize();
        farTileHashes[index] = tile->deterministicHash;
    }

    std::vector<FarTerrainViewTile> farView;
    selectFarTerrainView(0.0, 0.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, farView);
    std::array<size_t, 5> farViewCounts{};
    for (const FarTerrainViewTile& tile : farView) {
        const auto found = std::find(FAR_STEPS.begin(), FAR_STEPS.end(), tile.key.step);
        if (found != FAR_STEPS.end()) {
            ++farViewCounts[static_cast<size_t>(std::distance(FAR_STEPS.begin(), found))];
        }
    }
    const size_t farBaseWanted = farView.size();
    std::vector<FarTerrainKey> farResidencyOrder;
    buildFarTerrainResidencyOrder(farView, farResidencyOrder);
    const size_t farRefinementWanted = farResidencyOrder.size() - farBaseWanted;

    std::pair<int64_t, int64_t> continuityLocation = requestedSample.value_or(std::pair{0, 0});
    if (!requestedSample) {
        if (const auto lake = sightings.find("lake"); lake != sightings.end()) {
            continuityLocation = lake->second;
        }
    }
    const auto continuityStart = Clock::now();
    const auto [continuityX, continuityZ] = continuityLocation;
    const std::vector<FootprintMetric> footprintMetrics =
        measureFootprints(generator, continuityX, continuityZ);
    const worldgen::SurfaceSample continuitySurface = footprintMetrics.front().sample;
    const worldgen::surface_material::SurfaceMaterialPalette continuityPalette =
        generator.surfaceMaterialPaletteAt(continuityX, continuityZ);
    std::vector<FormerGridMetric> formerGridMetrics;
    formerGridMetrics.reserve(artifact::FORMER_GRID_SPACINGS.size());
    for (const int64_t spacing : artifact::FORMER_GRID_SPACINGS) {
        formerGridMetrics.push_back(
            measureFormerGridBoundary(generator, macro, static_cast<int>(spacing)));
    }
    FormerGridSummary formerGridSummary = summarizeFormerGridMetrics(formerGridMetrics);
    formerGridSummary.globalOrientation = artifact::globalOrientationHistogram(macro);
    formerGridSummary.globalStructuredOrientationRatio =
        artifact::structuredOrientationRatio(formerGridSummary.globalOrientation);
    formerGridSummary.orientationWithinLimit &=
        formerGridSummary.globalStructuredOrientationRatio <=
        artifact::STRUCTURED_ORIENTATION_LIMIT;
    const double continuityScanMs = elapsedMilliseconds(continuityStart);
    const worldgen::BasinCacheMetrics basinCache = generator.basinCacheMetrics();
    const worldgen::MacroControlCacheMetrics macroControlCache =
        generator.macroControlCacheMetrics();
    const worldgen::MacroControlCacheMetrics farClimateControlCache =
        generator.farClimateControlCacheMetrics();

    constexpr int exactNominalRadius = FAR_TERRAIN_NEAR_CHUNK_RADIUS;
    constexpr int exactPlanningRadius = FAR_TERRAIN_NEAR_CHUNK_RADIUS + 1;
    const size_t nominalExactColumns = diskColumnCount(exactNominalRadius);
    const size_t planningExactColumns = diskColumnCount(exactPlanningRadius);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n";
    std::cout << "  \"seed\": " << seed << ",\n";
    std::cout << "  \"origin\": {\"height\": " << origin.terrainHeight
              << ", \"water_surface\": " << origin.waterSurface
              << ", \"temperature_c\": " << origin.climate.temperatureC
              << ", \"precipitation_mm\": " << origin.climate.annualPrecipitationMm
              << ", \"biome\": \"" << biomeName(origin.biome.primary)
              << "\", \"secondary_biome\": \"" << biomeName(origin.biome.secondary)
              << "\", \"biome_transition\": " << origin.biome.transition << "},\n";
    std::cout << "  \"features\": {\n";
    constexpr std::array<const char*, 15> featureNames = {
        "mountain",   "cliff",       "canyon",          "river",
        "confluence", "lake",        "endorheic_lake",  "waterfall",
        "delta",      "volcano",     "oceanic_island",  "biome_transition",
        "snow_peak",  "dense_flora", "deep_fish_water",
    };
    for (size_t index = 0; index < featureNames.size(); ++index) {
        const char* name = featureNames[index];
        auto position = sightings.find(name);
        std::cout << "    \"" << name << "\": ";
        if (position == sightings.end()) {
            std::cout << "null";
        } else {
            std::cout << '[' << position->second.first << ", " << position->second.second << ']';
        }
        std::cout << (index + 1 == featureNames.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"biome_sightings\": {\n";
    for (size_t index = 0; index < biomeSightings.size(); ++index) {
        std::cout << "    \"" << biomeName(static_cast<Biome>(index)) << "\": ";
        if (biomeSightings[index].has_value()) {
            const auto [x, z] = *biomeSightings[index];
            std::cout << '[' << x << ", " << z << ']';
        } else {
            std::cout << "null";
        }
        std::cout << (index + 1 == biomeSightings.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    std::cout << "  \"feature_samples\": {\n";
    for (size_t index = 0; index < featureNames.size(); ++index) {
        const char* name = featureNames[index];
        const auto position = sightings.find(name);
        std::cout << "    \"" << name << "\": ";
        if (position == sightings.end()) {
            std::cout << "null";
        } else {
            const auto [x, z] = position->second;
            const worldgen::SurfaceSample macroSample = macro.sampleSurface(x, z);
            const worldgen::SurfaceSample sample = generator.sampleFarSurface(x, z);
            const double waterDepth =
                sample.hydrology.ocean || sample.hydrology.lake || sample.hydrology.river
                    ? std::max(0.0, sample.waterSurface - sample.terrainHeight)
                    : 0.0;
            std::cout << "{\"x\": " << x << ", \"z\": " << z
                      << ", \"height\": " << sample.terrainHeight
                      << ", \"water_surface\": " << sample.waterSurface
                      << ", \"water_depth\": " << waterDepth
                      << ", \"temperature_c\": " << sample.climate.temperatureC
                      << ", \"precipitation_mm\": " << sample.climate.annualPrecipitationMm
                      << ", \"macro_height\": " << macroSample.terrainHeight
                      << ", \"volcanic_activity\": " << sample.geology.volcanicActivity
                      << ", \"uplift\": " << sample.geology.uplift
                      << ", \"ecotopes\": " << static_cast<uint32_t>(sample.ecotopes)
                      << ", \"biome\": \"" << biomeName(sample.biome.primary)
                      << "\", \"secondary_biome\": \"" << biomeName(sample.biome.secondary)
                      << "\", \"biome_transition\": " << sample.biome.transition
                      << ", \"river_order\": "
                      << static_cast<unsigned>(sample.hydrology.streamOrder)
                      << ", \"discharge\": " << sample.hydrology.discharge
                      << ", \"waterfall_top\": " << sample.hydrology.waterfallTop
                      << ", \"waterfall_bottom\": " << sample.hydrology.waterfallBottom << '}';
        }
        std::cout << (index + 1 == featureNames.size() ? "\n" : ",\n");
    }
    std::cout << "  },\n";
    if (requestedSample) {
        const auto [x, z] = *requestedSample;
        const worldgen::SurfaceSample macroSample = macro.sampleSurface(x, z);
        const worldgen::SurfaceSample sample = generator.sampleFarSurface(x, z);
        std::cout << "  \"requested_sample\": {\"x\": " << x << ", \"z\": " << z
                  << ", \"height\": " << sample.terrainHeight
                  << ", \"water_surface\": " << sample.waterSurface
                  << ", \"temperature_c\": " << sample.climate.temperatureC
                  << ", \"precipitation_mm\": " << sample.climate.annualPrecipitationMm
                  << ", \"fertility\": " << sample.soil.fertility << ", \"slope\": " << sample.slope
                  << ", \"volcanic_activity\": " << sample.geology.volcanicActivity
                  << ", \"surface_material\": "
                  << static_cast<unsigned>(generator.surfaceMaterialAt(x, z)) << ", \"biome\": \""
                  << biomeName(sample.biome.primary) << "\", \"secondary_biome\": \""
                  << biomeName(sample.biome.secondary)
                  << "\", \"biome_transition\": " << sample.biome.transition
                  << ", \"river_order\": " << static_cast<unsigned>(sample.hydrology.streamOrder)
                  << ", \"discharge\": " << sample.hydrology.discharge
                  << ", \"channel_distance\": " << sample.hydrology.channelDistance
                  << ", \"channel_width\": " << sample.hydrology.channelWidth
                  << ", \"channel_depth\": " << sample.hydrology.channelDepth
                  << ", \"flow_x\": " << sample.hydrology.flowDirection.x
                  << ", \"flow_z\": " << sample.hydrology.flowDirection.z
                  << ", \"generated_fluid_level\": "
                  << static_cast<unsigned>(sample.hydrology.generatedFluidLevel)
                  << ", \"transition_owner_kind\": "
                  << static_cast<unsigned>(sample.hydrology.transitionOwnerKind)
                  << ", \"transition_owner_id\": ";
        writeWaterBodyId(std::cout, sample.hydrology.transitionOwnerId);
        std::cout << ", \"river\": " << (sample.hydrology.river ? "true" : "false")
                  << ", \"lake\": " << (sample.hydrology.lake ? "true" : "false")
                  << ", \"delta\": " << (sample.hydrology.delta ? "true" : "false")
                  << ", \"waterfall\": " << (sample.hydrology.waterfall ? "true" : "false")
                  << ", \"waterfall_top\": " << sample.hydrology.waterfallTop
                  << ", \"waterfall_bottom\": " << sample.hydrology.waterfallBottom
                  << ", \"waterfall_width\": " << sample.hydrology.waterfallWidth
                  << ", \"endorheic\": " << (sample.hydrology.endorheic ? "true" : "false")
                  << ", \"water_body_id\": ";
        writeWaterBodyId(std::cout, sample.hydrology.waterBodyId);
        std::cout << ", \"shore_distance\": " << sample.hydrology.lakeShoreDistance
                  << ", \"lithology_primary\": \"" << rockName(sample.geology.lithology.primary)
                  << "\", \"lithology_secondary\": \""
                  << rockName(sample.geology.lithology.secondary)
                  << "\", \"lithology_transition\": " << sample.geology.lithology.transition
                  << ", \"lithology_contact_distance\": "
                  << sample.geology.lithology.contactDistance
                  << ", \"macro_height\": " << macroSample.terrainHeight
                  << ", \"macro_water_surface\": " << macroSample.waterSurface
                  << ", \"macro_delta\": " << (macroSample.hydrology.delta ? "true" : "false")
                  << "},\n";
    }
    std::cout << "  \"continuity_sample\": {\"x\": " << continuityX << ", \"z\": " << continuityZ
              << ", \"water_body_id\": ";
    writeWaterBodyId(std::cout, continuitySurface.hydrology.waterBodyId);
    std::cout << ", \"shore_distance\": " << continuitySurface.hydrology.lakeShoreDistance
              << ", \"lithology\": {\"primary\": \""
              << rockName(continuitySurface.geology.lithology.primary) << "\", \"secondary\": \""
              << rockName(continuitySurface.geology.lithology.secondary)
              << "\", \"transition\": " << continuitySurface.geology.lithology.transition
              << ", \"contact_distance\": " << continuitySurface.geology.lithology.contactDistance
              << "}, \"material_palette\": ";
    writeMaterialPalette(std::cout, continuityPalette);
    std::cout << "},\n";
    std::cout << "  \"surface_footprints\": {\"x\": " << continuityX << ", \"z\": " << continuityZ
              << ", \"samples\": [\n";
    for (size_t index = 0; index < footprintMetrics.size(); ++index) {
        const FootprintMetric& metric = footprintMetrics[index];
        const double undershoot =
            std::max(0.0, metric.blockWindowMinimum - metric.sample.terrainHeight);
        const double overshoot =
            std::max(0.0, metric.sample.terrainHeight - metric.blockWindowMaximum);
        std::cout << "    {\"width\": " << worldgen::surfaceFootprintWidth(metric.footprint)
                  << ", \"terrain_height\": " << metric.sample.terrainHeight
                  << ", \"water_surface\": " << metric.sample.waterSurface
                  << ", \"block_window_min\": " << metric.blockWindowMinimum
                  << ", \"block_window_max\": " << metric.blockWindowMaximum
                  << ", \"bound_undershoot\": " << undershoot
                  << ", \"bound_overshoot\": " << overshoot
                  << ", \"finite\": " << (metric.finite ? "true" : "false")
                  << ", \"topology_matches_block\": "
                  << (metric.topologyMatchesBlock ? "true" : "false")
                  << ", \"water_body_matches_block\": "
                  << (metric.waterBodyMatchesBlock ? "true" : "false") << ", \"water_body_id\": ";
        writeWaterBodyId(std::cout, metric.sample.hydrology.waterBodyId);
        std::cout << '}' << (index + 1 == footprintMetrics.size() ? "\n" : ",\n");
    }
    std::cout << "  ]},\n";
    std::cout << "  \"former_grid_artifacts\": {\"summary\": {\"derivative_energy_within_limit\": "
              << (formerGridSummary.derivativeEnergyWithinLimit ? "true" : "false")
              << ", \"minimum_derivative_ratio\": " << formerGridSummary.minimumEnergyRatio
              << ", \"maximum_derivative_ratio\": " << formerGridSummary.maximumEnergyRatio
              << ", \"former_line_orientation_bins\": ";
    writeBins(std::cout, formerGridSummary.formerLineOrientation);
    std::cout << ", \"nearby_orientation_bins\": ";
    writeBins(std::cout, formerGridSummary.nearbyOrientation);
    std::cout << ", \"former_line_orientation_bias\": ";
    writeBins(std::cout, formerGridSummary.orientationBias);
    std::cout << ", \"former_line_structured_bias_ratio\": "
              << formerGridSummary.formerLineStructuredBiasRatio
              << ", \"global_orientation_bins\": ";
    writeBins(std::cout, formerGridSummary.globalOrientation);
    std::cout << ", \"global_structured_to_median_unstructured\": "
              << formerGridSummary.globalStructuredOrientationRatio
              << ", \"orientation_within_limit\": "
              << (formerGridSummary.orientationWithinLimit ? "true" : "false")
              << ", \"max_lake_boundary_run\": " << formerGridSummary.lakeBoundaryRun
              << ", \"max_material_boundary_run\": " << formerGridSummary.materialBoundaryRun
              << ", \"max_untagged_lithology_boundary_run\": "
              << formerGridSummary.untaggedLithologyBoundaryRun
              << ", \"categorical_runs_within_limit\": "
              << (formerGridSummary.categoricalRunsWithinLimit ? "true" : "false")
              << "}, \"metrics\": [\n";
    constexpr std::array signalNames = {"terrain", "provisional_precipitation",
                                        "lithology_contact"};
    for (size_t metricIndex = 0; metricIndex < formerGridMetrics.size(); ++metricIndex) {
        const FormerGridMetric& metric = formerGridMetrics[metricIndex];
        std::cout << "    {\"spacing\": " << metric.spacing << ", \"line_x\": " << metric.lineX
                  << ", \"center_z\": " << metric.centerZ << ", \"derivative_energy\": {";
        for (size_t signalIndex = 0; signalIndex < signalNames.size(); ++signalIndex) {
            std::cout << (signalIndex == 0 ? "" : ", ") << '"' << signalNames[signalIndex]
                      << "\": {\"boundary\": " << metric.boundaryEnergy[signalIndex]
                      << ", \"nearby\": " << metric.nearbyEnergy[signalIndex]
                      << ", \"ratio\": " << metric.energyRatios[signalIndex] << '}';
        }
        std::cout << "}, \"former_line_orientation_bins\": ";
        writeBins(std::cout, metric.formerLineOrientation);
        std::cout << ", \"nearby_orientation_bins\": ";
        writeBins(std::cout, metric.nearbyOrientation);
        std::cout << ", \"former_line_orientation_bias\": ";
        writeBins(std::cout, metric.orientationBias);
        std::cout << ", \"former_line_structured_bias_ratio\": "
                  << metric.structuredOrientationRatio
                  << ", \"max_lake_boundary_run\": " << metric.lakeBoundaryRun
                  << ", \"max_material_boundary_run\": " << metric.materialBoundaryRun
                  << ", \"max_untagged_lithology_boundary_run\": "
                  << metric.untaggedLithologyBoundaryRun << '}'
                  << (metricIndex + 1 == formerGridMetrics.size() ? "\n" : ",\n");
    }
    std::cout << "  ]},\n";
    std::cout << "  \"exact_coverage_estimate\": {\"nominal_radius_chunks\": " << exactNominalRadius
              << ", \"planning_radius_chunks\": " << exactPlanningRadius
              << ", \"nominal_primary_columns\": " << nominalExactColumns
              << ", \"planning_primary_columns\": " << planningExactColumns
              << ", \"minimum_primary_sections\": " << planningExactColumns << "},\n";
    std::cout << "  \"cache\": {\"column_plans\": " << generator.cachedColumnPlanCount()
              << ", \"column_capacity\": " << DEFAULT_COLUMN_PLAN_CACHE_CAPACITY
              << ", \"basins\": " << basinCache.entries << ", \"basin_bytes\": " << basinCache.bytes
              << ", \"basin_hits\": " << basinCache.hits
              << ", \"basin_misses\": " << basinCache.misses
              << ", \"basin_builds\": " << basinCache.builds
              << ", \"basin_failures\": " << basinCache.failures
              << ", \"fallback_builds\": " << basinCache.fallbackBuilds
              << ", \"erosion_epochs\": " << basinCache.erosionEpochs
              << ", \"erosion_reroutes\": " << basinCache.erosionReroutes
              << ", \"erosion_receiver_changes\": " << basinCache.erosionReceiverChanges
              << ", \"shoreline_pages\": " << basinCache.shorelineEntries
              << ", \"shoreline_bytes\": " << basinCache.shorelineBytes
              << ", \"shoreline_hits\": " << basinCache.shorelineHits
              << ", \"shoreline_misses\": " << basinCache.shorelineMisses
              << ", \"shoreline_builds\": " << basinCache.shorelineBuilds
              << ", \"shoreline_failures\": " << basinCache.shorelineFailures
              << ", \"macro_control_tiles\": " << macroControlCache.entries
              << ", \"macro_control_bytes\": " << macroControlCache.bytes
              << ", \"macro_control_hits\": " << macroControlCache.hits
              << ", \"macro_control_misses\": " << macroControlCache.misses
              << ", \"macro_control_builds\": " << macroControlCache.builds
              << ", \"macro_control_evictions\": " << macroControlCache.evictions
              << ", \"macro_control_single_flight_waits\": " << macroControlCache.singleFlightWaits
              << ", \"peak_macro_control_builds\": " << macroControlCache.peakBuilds
              << ", \"far_climate_control_tiles\": " << farClimateControlCache.entries
              << ", \"far_climate_control_bytes\": " << farClimateControlCache.bytes
              << ", \"far_climate_control_hits\": " << farClimateControlCache.hits
              << ", \"far_climate_control_misses\": " << farClimateControlCache.misses
              << ", \"far_climate_control_builds\": " << farClimateControlCache.builds
              << ", \"far_climate_control_evictions\": " << farClimateControlCache.evictions
              << ", \"far_climate_control_single_flight_waits\": "
              << farClimateControlCache.singleFlightWaits
              << ", \"peak_far_climate_control_builds\": " << farClimateControlCache.peakBuilds
              << ", \"active_cold_basins\": " << basinCache.activeColdBuilds
              << ", \"peak_cold_basins\": " << basinCache.peakColdBuilds
              << ", \"throttled_cold_basins\": " << basinCache.throttledBuilds << "},\n";
    std::cout << "  \"benchmark\": {\"feature_search_ms\": " << searchMs
              << ", \"cold_route_ms\": " << coldMs << ", \"warm_route_ms\": " << warmMs
              << ", \"continuity_scan_ms\": " << continuityScanMs
              << ", \"cold_cube_p95_ms\": " << percentile95(coldCubeMilliseconds)
              << ", \"warm_cube_mean_ms\": " << warmMs / route.size()
              << ", \"warm_cube_p95_ms\": " << percentile95(warmCubeMilliseconds)
              << ", \"cold_basin_ms\": " << coldBasinMs << ", \"warm_basin_ms\": " << warmBasinMs
              << "},\n";
    std::cout << "  \"far_terrain\": {\"view_tiles\": " << farView.size()
              << ", \"full_disk_base_wanted\": " << farBaseWanted
              << ", \"refinement_wanted\": " << farRefinementWanted
              << ", \"total_unique_wanted\": " << farResidencyOrder.size()
              << ", \"step_2_tiles\": " << farViewCounts[0]
              << ", \"step_4_tiles\": " << farViewCounts[1]
              << ", \"step_8_tiles\": " << farViewCounts[2]
              << ", \"step_16_tiles\": " << farViewCounts[3]
              << ", \"step_32_parent_tiles\": " << farViewCounts[4]
              << ", \"worker_count\": " << FarTerrainScheduler::WORKER_COUNT
              << ", \"minimum_base_workers_during_coverage\": "
              << FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE
              << ", \"maximum_urgent_workers_during_coverage\": "
              << farTerrainUrgentWorkerLimit(FarTerrainScheduler::WORKER_COUNT, true)
              << ", \"tile_builds\": [\n";
    for (size_t index = 0; index < FAR_STEPS.size(); ++index) {
        std::cout << "    {\"step\": " << farTerrainStepSize(FAR_STEPS[index])
                  << ", \"milliseconds\": " << farTileMilliseconds[index]
                  << ", \"bytes\": " << farTileBytes[index] << ", \"hash\": \"0x" << std::hex
                  << farTileHashes[index] << std::dec << "\"}"
                  << (index + 1 == FAR_STEPS.size() ? "\n" : ",\n");
    }
    std::cout << "  ]},\n";
    std::cout << "  \"determinism\": {\"forward_reverse_equal\": "
              << (deterministic && coldBasin.surfaceElevation == warmBasin.surfaceElevation
                      ? "true"
                      : "false")
              << ", \"route_hash\": \"0x" << std::hex << routeHash << std::dec << "\"}\n";
    std::cout << "}\n";
    return deterministic ? 0 : 1;
}

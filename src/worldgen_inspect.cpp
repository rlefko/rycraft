#include "common/random.hpp"
#include "engine/v4_world_startup.hpp"
#include "render/far_terrain.hpp"
#include "world/artifact_analysis.hpp"
#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"
#include "world/features.hpp"
#include "world/macro_generation.hpp"
#include "world/save_manager.hpp"
#include "world/terrain_runtime.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
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

struct AuthorityDeltaSummary {
    bool requested = false;
    double milliseconds = 0.0;
    double previewMilliseconds = 0.0;
    double finalMilliseconds = 0.0;
    double meanSignedMeters = 0.0;
    double rootMeanSquareMeters = 0.0;
    double p95AbsoluteMeters = 0.0;
    double maximumAbsoluteMeters = 0.0;
    double boundaryP95AbsoluteMeters = 0.0;
    double boundaryMaximumAbsoluteMeters = 0.0;
    double p95AbsoluteBlocks = 0.0;
    double maximumAbsoluteBlocks = 0.0;
    uint64_t oceanSignDisagreements = 0;
    std::string previewPageHash;
    std::string finalPageHash;
    std::array<uint64_t, 3> modelCalls{};
    std::array<uint64_t, 3> previewModelCalls{};
    std::array<uint64_t, 3> finalModelCalls{};
};

bool pathIsWithin(const std::filesystem::path& child, const std::filesystem::path& parent) {
    std::error_code error;
    const std::filesystem::path normalizedChild = std::filesystem::weakly_canonical(child, error);
    if (error) return false;
    error.clear();
    const std::filesystem::path normalizedParent = std::filesystem::weakly_canonical(parent, error);
    if (error) return false;
    auto childPart = normalizedChild.begin();
    auto parentPart = normalizedParent.begin();
    for (; parentPart != normalizedParent.end(); ++parentPart, ++childPart) {
        if (childPart == normalizedChild.end() || *childPart != *parentPart) return false;
    }
    return true;
}

std::optional<std::filesystem::path>
prepareInspectorAuthorityProfile(const std::optional<std::filesystem::path>& requestedProfile,
                                 uint64_t seed, const worldgen::learned::Sha256Digest& fingerprint,
                                 std::string& failure) {
    std::filesystem::path profile;
    if (requestedProfile) {
        if (!requestedProfile->is_absolute()) {
            failure =
                "The --profile path must be absolute and outside the user application support root";
            return std::nullopt;
        }
        const std::filesystem::path applicationSupport =
            worldgen::bootstrap::defaultRycraftApplicationSupportPath();
        if (pathIsWithin(*requestedProfile, applicationSupport)) {
            failure = "The --profile path must be outside the user application support root";
            return std::nullopt;
        }
        profile = requestedProfile->lexically_normal();
    } else {
        std::error_code error;
        const std::filesystem::path temporaryRoot = std::filesystem::temp_directory_path(error);
        if (error) {
            failure = "Could not locate a temporary directory for the v4 inspector profile";
            return std::nullopt;
        }
        profile = temporaryRoot / "rycraft-worldgen-inspect-v4" /
                  ("seed-" + std::to_string(seed) + "-fingerprint-" +
                   worldgen::learned::sha256Hex(fingerprint));
    }

    std::error_code error;
    std::filesystem::create_directories(profile, error);
    if (error || !std::filesystem::is_directory(profile, error) || error) {
        failure = "Could not create the isolated v4 inspector profile directory";
        return std::nullopt;
    }
    return profile;
}

std::string jsonEscaped(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (const char character : value) {
        switch (character) {
            case '\\':
                result += "\\\\";
                break;
            case '"':
                result += "\\\"";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    result += "?";
                } else {
                    result += character;
                }
                break;
        }
    }
    return result;
}

std::optional<int> parseInspectorHorizonRadius(std::string_view value) {
    int radius = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), radius);
    if (error != std::errc{} || end != value.data() + value.size()) return std::nullopt;
    return radius;
}

// The normal inspector measures resident authority and total mesh time. This
// opt-in profile records which conservative step-32 water gate rejected each
// tile, without changing the mesher's inputs or scheduling. It is deliberately
// scoped to the inspector so release mesh construction stays free of counters.
struct HorizonWaterProfile {
    struct TileSamples {
        int step = 0;
        int sampleEdge = 0;
        worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1;
        std::vector<FarSurfaceSample> samples;
    };

    std::mutex mutex;
    std::map<std::pair<int64_t, int64_t>, TileSamples> pendingSamples;
    size_t evaluatedTiles = 0;
    size_t uniformStandingTiles = 0;
    size_t rejectedDryOrMixedTiles = 0;
    size_t rejectedMovingWaterTiles = 0;
    size_t rejectedFallTiles = 0;
    size_t rejectedAuthorityTiles = 0;
    size_t rejectedTopologyTiles = 0;
    size_t topologyMarkedParents = 0;
    size_t canonicalWaterGridCalls = 0;
    size_t canonicalWaterGridSamples = 0;
    size_t fullNativeWaterPageCalls = 0;
    size_t fullNativeWaterPageSamples = 0;
};

bool inspectorSampleIsWet(const FarTerrainGeometrySample& sample) {
    return (sample.ocean || sample.river || sample.lake || sample.wetland) &&
           sample.waterSurface > sample.terrainHeight + 0.01;
}

float inspectorVertexHeight(double height) {
    return static_cast<float>(static_cast<float16_t>(std::clamp(
        height, static_cast<double>(WORLD_MIN_Y), static_cast<double>(WORLD_MAX_Y + 1))));
}

struct InspectorStandingAuthority {
    worldgen::WaterBodyId bodyId = worldgen::NO_WATER_BODY;
    uint64_t transitionOwnerId = 0;
    float height = SEA_LEVEL;
    worldgen::WaterTransitionKind transitionOwnerKind = worldgen::WaterTransitionKind::NONE;
    uint8_t generatedFluidLevel = 0;
    bool lake = false;
    bool delta = false;

    bool operator==(const InspectorStandingAuthority&) const = default;
};

InspectorStandingAuthority inspectorStandingAuthority(const FarTerrainGeometrySample& sample) {
    return {
        .bodyId = sample.waterBodyId,
        .transitionOwnerId = sample.transitionOwnerId,
        .height = inspectorVertexHeight(sample.waterSurface),
        .transitionOwnerKind = sample.transitionOwnerKind,
        .generatedFluidLevel = sample.generatedFluidLevel,
        .lake = sample.lake,
        .delta = sample.delta,
    };
}

void profileStep32WaterGate(HorizonWaterProfile& profile, int64_t boundsOriginX,
                            int64_t boundsOriginZ, int step, int cellWidth, int cellHeight,
                            worldgen::SurfaceFootprint footprint,
                            std::span<const FarTerrainCellBounds> bounds) {
    if (step != 32 || footprint != worldgen::SurfaceFootprint::BLOCK_32 || cellWidth != 10 ||
        cellHeight != 10 || bounds.size() != 100) {
        return;
    }
    const std::pair sampleKey{boundsOriginX + step, boundsOriginZ + step};
    const auto found = profile.pendingSamples.find(sampleKey);
    if (found == profile.pendingSamples.end() || found->second.step != step ||
        found->second.sampleEdge != 9 ||
        found->second.footprint != worldgen::SurfaceFootprint::BLOCK_32 ||
        found->second.samples.size() != 81) {
        return;
    }

    const std::vector<FarSurfaceSample>& samples = found->second.samples;
    ++profile.evaluatedTiles;
    bool dryOrMixed = false;
    bool movingWater = false;
    bool falls = false;
    bool authorityMismatch = false;
    std::optional<InspectorStandingAuthority> reference;
    for (const FarSurfaceSample& sample : samples) {
        const FarTerrainGeometrySample& geometry = sample.geometry;
        if (!inspectorSampleIsWet(geometry)) {
            dryOrMixed = true;
            continue;
        }
        if (geometry.waterfall || geometry.waterfallAnchor) falls = true;
        if (!geometry.ocean && !geometry.lake) {
            movingWater = true;
            continue;
        }
        const InspectorStandingAuthority authority = inspectorStandingAuthority(geometry);
        if (!reference) {
            reference = authority;
        } else if (*reference != authority) {
            authorityMismatch = true;
        }
    }

    bool topology = false;
    for (int cellZ = 0; cellZ < 8; ++cellZ) {
        for (int cellX = 0; cellX < 8; ++cellX) {
            const FarTerrainCellBounds& cell =
                bounds[static_cast<size_t>((cellZ + 1) * cellWidth + cellX + 1)];
            if (cell.waterTopologyPossible || cell.volcanicWaterPossible ||
                cell.waterfallPossible) {
                topology = true;
                ++profile.topologyMarkedParents;
            }
        }
    }

    if (dryOrMixed) ++profile.rejectedDryOrMixedTiles;
    if (movingWater) ++profile.rejectedMovingWaterTiles;
    if (falls) ++profile.rejectedFallTiles;
    if (authorityMismatch) ++profile.rejectedAuthorityTiles;
    if (topology) ++profile.rejectedTopologyTiles;
    if (!dryOrMixed && !movingWater && !falls && !authorityMismatch && !topology && reference) {
        ++profile.uniformStandingTiles;
    }
    profile.pendingSamples.erase(found);
}

FarTerrainSource
instrumentHorizonWaterProfile(FarTerrainSource source,
                              const std::shared_ptr<HorizonWaterProfile>& profile) {
    const FarTerrainSource::GridSampleFunction sampleGrid = source.sampleGrid;
    const FarTerrainSource::CellBoundsGridFunction cellBoundsGrid = source.cellBoundsGrid;
    const FarTerrainSource::GeometryGridSampleFunction canonicalWaterGrid =
        source.canonicalWaterGrid;

    source.sampleGrid = [sampleGrid, profile](int64_t originX, int64_t originZ, int spacing,
                                              int sampleEdge, worldgen::SurfaceFootprint footprint,
                                              std::span<FarSurfaceSample> output) {
        sampleGrid(originX, originZ, spacing, sampleEdge, footprint, output);
        if (spacing != 32 || sampleEdge != 9 || footprint != worldgen::SurfaceFootprint::BLOCK_32) {
            return;
        }
        std::lock_guard lock(profile->mutex);
        profile->pendingSamples[{originX, originZ}] = {
            .step = spacing,
            .sampleEdge = sampleEdge,
            .footprint = footprint,
            .samples = std::vector<FarSurfaceSample>(output.begin(), output.end()),
        };
    };
    source.cellBoundsGrid = [cellBoundsGrid, profile](int64_t originX, int64_t originZ, int step,
                                                      int cellWidth, int cellHeight,
                                                      worldgen::SurfaceFootprint footprint,
                                                      std::span<FarTerrainCellBounds> output) {
        cellBoundsGrid(originX, originZ, step, cellWidth, cellHeight, footprint, output);
        std::lock_guard lock(profile->mutex);
        profileStep32WaterGate(*profile, originX, originZ, step, cellWidth, cellHeight, footprint,
                               output);
    };
    source.canonicalWaterGrid = [canonicalWaterGrid,
                                 profile](int64_t originX, int64_t originZ, int spacingX,
                                          int spacingZ, int sampleWidth, int sampleHeight,
                                          worldgen::SurfaceFootprint footprint,
                                          std::span<FarTerrainGeometrySample> output) {
        {
            std::lock_guard lock(profile->mutex);
            ++profile->canonicalWaterGridCalls;
            profile->canonicalWaterGridSamples += output.size();
            constexpr int nativeWaterPageEdge =
                FAR_TERRAIN_TILE_EDGE / worldgen::NATIVE_HYDROLOGY_RASTER_SPACING + 2;
            if (spacingX == worldgen::NATIVE_HYDROLOGY_RASTER_SPACING &&
                spacingZ == worldgen::NATIVE_HYDROLOGY_RASTER_SPACING &&
                sampleWidth == nativeWaterPageEdge && sampleHeight == nativeWaterPageEdge &&
                footprint == worldgen::SurfaceFootprint::BLOCK_1) {
                ++profile->fullNativeWaterPageCalls;
                profile->fullNativeWaterPageSamples += output.size();
            }
        }
        canonicalWaterGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
                           footprint, output);
    };
    return source;
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
           first.lake == second.lake && first.wetland == second.wetland &&
           first.delta == second.delta && first.waterfall == second.waterfall;
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
           sample.hydrology.wetland || sample.hydrology.waterfall || sample.hydrology.delta ||
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

int runV4Inspector(int argc, char** argv) {
    using namespace worldgen;
    using namespace worldgen::bootstrap;
    using namespace worldgen::learned;
    using namespace worldgen::runtime;

    if (argc < 3) {
        std::cerr << "Usage: rycraft_worldgen_inspect --v4-model MODEL_PACK [seed] [x z] "
                     "[preview|final] [--profile ABSOLUTE_PATH] [--hydrology] [--flora] "
                     "[--dry-spawn] [--authority-delta] "
                     "[--horizon|--horizon-hydrology|--horizon-mesh|"
                     "--horizon-water-profile] [--horizon-radius CHUNKS]\n";
        return 2;
    }
    const std::filesystem::path modelPack = argv[2];
    uint64_t seed = 42;
    int64_t worldX = 0;
    int64_t worldZ = 0;
    AuthorityQuality quality = AuthorityQuality::FINAL;
    bool includeHydrology = false;
    bool includeFlora = false;
    bool includeHorizon = false;
    bool includeHorizonHydrology = false;
    bool includeHorizonMesh = false;
    bool includeHorizonWaterProfile = false;
    bool includeDrySpawn = false;
    bool includeAuthorityDelta = false;
    int horizonChunkRadius = FAR_TERRAIN_MAX_CHUNK_RADIUS;
    bool horizonRadiusSpecified = false;
    std::optional<std::filesystem::path> requestedProfile;
    if (argc >= 4) seed = std::strtoull(argv[3], nullptr, 0);
    if (argc >= 6) {
        worldX = std::strtoll(argv[4], nullptr, 0);
        worldZ = std::strtoll(argv[5], nullptr, 0);
    }
    for (int index = 6; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "preview")
            quality = AuthorityQuality::PREVIEW;
        else if (argument == "final")
            quality = AuthorityQuality::FINAL;
        else if (argument == "--hydrology")
            includeHydrology = true;
        else if (argument == "--flora")
            includeFlora = true;
        else if (argument == "--dry-spawn")
            includeDrySpawn = true;
        else if (argument == "--authority-delta")
            includeAuthorityDelta = true;
        else if (argument == "--profile") {
            if (++index >= argc) {
                std::cerr << "The --profile option requires an absolute directory path\n";
                return 2;
            }
            if (requestedProfile) {
                std::cerr << "The --profile option may be specified only once\n";
                return 2;
            }
            requestedProfile = std::filesystem::path(argv[index]);
        } else if (argument == "--horizon")
            includeHorizon = true;
        else if (argument == "--horizon-hydrology") {
            includeHorizon = true;
            includeHorizonHydrology = true;
        } else if (argument == "--horizon-mesh") {
            includeHorizon = true;
            includeHorizonHydrology = true;
            includeHorizonMesh = true;
        } else if (argument == "--horizon-water-profile") {
            includeHorizon = true;
            includeHorizonHydrology = true;
            includeHorizonMesh = true;
            includeHorizonWaterProfile = true;
        } else if (argument == "--horizon-radius") {
            if (++index >= argc) {
                std::cerr << "The --horizon-radius option requires a chunk radius\n";
                return 2;
            }
            if (horizonRadiusSpecified) {
                std::cerr << "The --horizon-radius option may be specified only once\n";
                return 2;
            }
            const std::optional<int> parsed = parseInspectorHorizonRadius(argv[index]);
            if (!parsed || !farTerrainHorizonRadiusValid(*parsed)) {
                std::cerr << "The --horizon-radius value must be an integer from "
                          << FAR_TERRAIN_NEAR_CHUNK_RADIUS << " through "
                          << FAR_TERRAIN_MAX_CHUNK_RADIUS << " chunks\n";
                return 2;
            }
            horizonChunkRadius = *parsed;
            horizonRadiusSpecified = true;
        } else {
            std::cerr << "Unknown v4 inspector option: " << argument << '\n';
            return 2;
        }
    }
    if (includeDrySpawn && quality != AuthorityQuality::FINAL) {
        std::cerr << "Dry-spawn qualification requires final authority\n";
        return 2;
    }
    if (includeFlora && quality != AuthorityQuality::FINAL) {
        std::cerr << "Flora qualification requires final authority\n";
        return 2;
    }
    if (horizonRadiusSpecified && !includeHorizon) {
        std::cerr << "The --horizon-radius option requires a --horizon inspection mode\n";
        return 2;
    }

    Sha256TerrainAssetVerifier verifier;
    for (const TerrainAssetSpec& asset : pinnedTerrainAssets()) {
        const TerrainVerificationResult verified =
            verifier.verify(modelPack / asset.fileName, asset, nullptr);
        if (!verified.valid) {
            std::cerr << "Model verification failed for " << asset.fileName << ": "
                      << verified.message << '\n';
            return 3;
        }
    }

    TerrainBootstrapCancellation cancellation;
    ProductionTerrainRuntime runtime(seed);
    const auto setupStart = Clock::now();
    TerrainRuntimeStepResult step = runtime.qualifyPlatform();
    if (step.succeeded) step = runtime.compile(modelPack, cancellation);
    if (step.succeeded) step = runtime.loadAndQualify(modelPack, cancellation);
    if (!step.succeeded) {
        std::cerr << "Runtime qualification failed: " << step.failure.message << '\n';
        return 4;
    }
    std::string profileFailure;
    const std::optional<std::filesystem::path> authorityProfile = prepareInspectorAuthorityProfile(
        requestedProfile, seed, runtime.generationIdentity().fingerprint(), profileFailure);
    if (!authorityProfile) {
        std::cerr << "Inspector profile preparation failed: " << profileFailure << '\n';
        return 4;
    }
    step = runtime.bindWorldProfile(*authorityProfile);
    if (!step.succeeded) {
        std::cerr << "Inspector profile binding failed: " << step.failure.message << '\n';
        return 4;
    }
    const double setupMilliseconds = elapsedMilliseconds(setupStart);
    std::shared_ptr<WorldGenerationContext> context = runtime.qualifiedGenerationContext();
    if (!context) {
        std::cerr << "Runtime qualification did not publish a generation context\n";
        return 4;
    }
    if (quality != context->quality()) context = context->withQuality(quality);
    if (includeHorizonHydrology && quality != AuthorityQuality::PREVIEW) {
        std::cerr << "Horizon hydrology qualification requires preview authority\n";
        return 2;
    }

    // Exercise the same learned selector and canonical-water screen used by
    // the title-screen bootstrap. This intentionally precedes the requested
    // sample and horizon work: a fresh world must locate a candidate that is
    // both learned land and dry canonical hydrology before it spends work on
    // any visual horizon. Exact collision remains the subsequent World-level
    // authority and is intentionally not fabricated by this inspector.
    std::optional<Vec3> drySpawn;
    uint32_t drySpawnOrdinal = 0;
    uint32_t drySpawnWaterRejections = 0;
    uint32_t drySpawnLocalRelocations = 0;
    double drySpawnMilliseconds = 0.0;
    double drySpawnSelectionMilliseconds = 0.0;
    double drySpawnWaterScreenMilliseconds = 0.0;
    std::optional<std::string> drySpawnFailure;
    if (includeDrySpawn) {
        V4SpawnWaterScreen waterScreen;
        const auto drySpawnStart = Clock::now();
        const auto drySpawnDeadline = drySpawnStart + std::chrono::minutes(5);
        while (drySpawnOrdinal < V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES &&
               Clock::now() < drySpawnDeadline) {
            const auto selectionStart = Clock::now();
            AuthorityResult<std::optional<Vec3>> selected =
                findV4DryLandSpawnCandidate(context, worldX, worldZ, drySpawnOrdinal);
            while (selected.status() == AuthorityStatus::DEFERRED &&
                   Clock::now() < drySpawnDeadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                selected = findV4DryLandSpawnCandidate(context, worldX, worldZ, drySpawnOrdinal);
            }
            drySpawnSelectionMilliseconds += elapsedMilliseconds(selectionStart);
            if (selected.status() == AuthorityStatus::FAILED) {
                drySpawnFailure =
                    selected.failure() ? selected.failure()->message : "unknown dry-spawn failure";
                break;
            }
            if (!selected.isReady()) break;
            if (selected.value() && *selected.value()) {
                const Vec3 candidate = **selected.value();
                const auto waterScreenStart = Clock::now();
                V4SpawnWaterScreenResult water = waterScreen.screen(context, candidate);
                while (water.deferred() && Clock::now() < drySpawnDeadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    water = waterScreen.screen(context, candidate);
                }
                drySpawnWaterScreenMilliseconds += elapsedMilliseconds(waterScreenStart);
                if (water.failed()) {
                    drySpawnFailure = water.failure ? water.failure->message
                                                    : "canonical dry-spawn water screen failed";
                    break;
                }
                if (water.deferred()) break;
                if (water.water()) {
                    ++drySpawnWaterRejections;
                    waterScreen.reset();
                    ++drySpawnOrdinal;
                    continue;
                }
                const Vec3 resolvedCandidate = water.resolvedCandidate.value_or(candidate);
                if (static_cast<int64_t>(std::floor(resolvedCandidate.x)) !=
                        static_cast<int64_t>(std::floor(candidate.x)) ||
                    static_cast<int64_t>(std::floor(resolvedCandidate.z)) !=
                        static_cast<int64_t>(std::floor(candidate.z))) {
                    ++drySpawnLocalRelocations;
                }
                waterScreen.reset();
                drySpawn = resolvedCandidate;
                break;
            }
            ++drySpawnOrdinal;
        }
        drySpawnMilliseconds = elapsedMilliseconds(drySpawnStart);
        if (!drySpawn && !drySpawnFailure) {
            drySpawnFailure = Clock::now() >= drySpawnDeadline
                                  ? "dry-spawn selection exceeded five minutes"
                                  : "no canonical dry-land candidate was found";
        }
    }

    size_t horizonPageCount = 0;
    double horizonMilliseconds = 0.0;
    std::vector<FarTerrainViewTile> horizonTiles;
    if (includeHorizon) {
        selectFarTerrainView(static_cast<double>(worldX), static_cast<double>(worldZ),
                             horizonChunkRadius, horizonTiles);
        const std::vector<TerrainPageCoordinate> pages = farTerrainCoarseAuthorityPages(
            horizonTiles, static_cast<double>(worldX), static_cast<double>(worldZ));
        const std::shared_ptr<WorldGenerationContext> previewContext =
            context->quality() == AuthorityQuality::PREVIEW
                ? context
                : context->withQuality(AuthorityQuality::PREVIEW);
        const auto horizonStart = Clock::now();
        const auto horizonDeadline = horizonStart + std::chrono::minutes(5);
        size_t requested = 0;
        std::vector<TerrainPageCoordinate> outstanding;
        outstanding.reserve(std::min(pages.size(), MAXIMUM_AUTHORITY_QUEUED_REQUESTS));
        std::optional<std::string> horizonFailure;
        const auto pollOutstanding = [&] {
            bool progressed = false;
            for (size_t index = 0; index < outstanding.size();) {
                const TerrainPageCoordinate coordinate = outstanding[index];
                const AuthorityResult<bool> prepared = previewContext->requestAuthorityPage(
                    coordinate, AuthorityRequestPriority::COARSE_PREVIEW);
                if (prepared.status() == AuthorityStatus::FAILED) {
                    horizonFailure =
                        "Horizon authority page row " + std::to_string(coordinate.row) +
                        " column " + std::to_string(coordinate.column) + " failed: " +
                        (prepared.failure() ? prepared.failure()->message : "unknown failure");
                    return false;
                }
                if (prepared.isReady()) {
                    outstanding[index] = outstanding.back();
                    outstanding.pop_back();
                    progressed = true;
                    continue;
                }
                ++index;
            }
            return progressed;
        };
        while (requested < pages.size() && Clock::now() < horizonDeadline) {
            const AuthorityResult<bool> prepared = previewContext->requestAuthorityPage(
                pages[requested], AuthorityRequestPriority::COARSE_PREVIEW);
            if (prepared.status() == AuthorityStatus::FAILED) {
                std::cerr << "Horizon authority preparation failed: "
                          << (prepared.failure() ? prepared.failure()->message : "unknown failure")
                          << '\n';
                return 5;
            }
            if (prepared.status() == AuthorityStatus::DEFERRED && prepared.failure() &&
                prepared.failure()->code == GenerationFailureCode::QUEUE_FULL) {
                const bool progressed = pollOutstanding();
                if (horizonFailure) {
                    std::cerr << *horizonFailure << '\n';
                    return 5;
                }
                if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (prepared.status() == AuthorityStatus::DEFERRED)
                outstanding.push_back(pages[requested]);
            ++requested;
        }
        if (requested != pages.size()) {
            std::cerr << "Horizon authority enqueue exceeded five minutes\n";
            return 5;
        }
        while (!outstanding.empty() && Clock::now() < horizonDeadline) {
            const bool progressed = pollOutstanding();
            if (horizonFailure) {
                std::cerr << *horizonFailure << '\n';
                return 5;
            }
            if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!outstanding.empty()) {
            std::cerr << "Horizon authority pages did not become ready before the deadline\n";
            return 5;
        }
        horizonPageCount = pages.size();
        horizonMilliseconds = elapsedMilliseconds(horizonStart);
    }

    size_t horizonHydrologyPageCount = 0;
    double horizonHydrologyMilliseconds = 0.0;
    if (includeHorizonHydrology) {
        std::set<std::pair<int64_t, int64_t>> owners;
        constexpr int64_t SAMPLING_APRON = farTerrainStepSize(FAR_TERRAIN_BASE_STEP);
        for (const FarTerrainViewTile& tile : horizonTiles) {
            const int64_t firstOwnerX =
                world_coord::floorDiv(tile.bounds.minX - SAMPLING_APRON,
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t lastOwnerX =
                world_coord::floorDiv(tile.bounds.maxX + SAMPLING_APRON - 1,
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t firstOwnerZ =
                world_coord::floorDiv(tile.bounds.minZ - SAMPLING_APRON,
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t lastOwnerZ =
                world_coord::floorDiv(tile.bounds.maxZ + SAMPLING_APRON - 1,
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
            for (int64_t ownerZ = firstOwnerZ; ownerZ <= lastOwnerZ; ++ownerZ)
                for (int64_t ownerX = firstOwnerX; ownerX <= lastOwnerX; ++ownerX)
                    owners.emplace(ownerX, ownerZ);
        }
        const auto horizonHydrologyStart = Clock::now();
        constexpr size_t BATCH_SIZE = NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS;
        std::vector<std::pair<int64_t, int64_t>> ownerList(owners.begin(), owners.end());
        for (size_t begin = 0; begin < ownerList.size(); begin += BATCH_SIZE) {
            const size_t end = std::min(ownerList.size(), begin + BATCH_SIZE);
            std::vector<std::future<HydrologySample>> requests;
            requests.reserve(end - begin);
            for (size_t index = begin; index < end; ++index) {
                const auto [ownerX, ownerZ] = ownerList[index];
                requests.push_back(std::async(std::launch::async, [context, seed, ownerX, ownerZ] {
                    MacroGenerationSampler sampler(seed, context);
                    const double x = static_cast<double>(ownerX * NATIVE_HYDROLOGY_PAGE_EDGE +
                                                         NATIVE_HYDROLOGY_PAGE_EDGE / 2);
                    const double z = static_cast<double>(ownerZ * NATIVE_HYDROLOGY_PAGE_EDGE +
                                                         NATIVE_HYDROLOGY_PAGE_EDGE / 2);
                    return sampler.sampleHydrology(x, z);
                }));
            }
            for (std::future<HydrologySample>& request : requests)
                static_cast<void>(request.get());
        }
        horizonHydrologyPageCount = ownerList.size();
        horizonHydrologyMilliseconds = elapsedMilliseconds(horizonHydrologyStart);
    }

    size_t horizonMeshCount = 0;
    size_t horizonMeshBytes = 0;
    double horizonMeshMilliseconds = 0.0;
    worldgen::MacroControlCacheMetrics horizonMacroControlCache;
    worldgen::MacroControlCacheMetrics horizonFarClimateControlCache;
    const std::shared_ptr<HorizonWaterProfile> horizonWaterProfile =
        includeHorizonWaterProfile ? std::make_shared<HorizonWaterProfile>() : nullptr;
    if (includeHorizonMesh) {
        const auto horizonMeshStart = Clock::now();
        const auto generator = std::make_shared<ChunkGenerator>(seed, context);
        FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        if (horizonWaterProfile)
            source = instrumentHorizonWaterProfile(std::move(source), horizonWaterProfile);
        constexpr size_t BATCH_SIZE = NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS;
        for (size_t begin = 0; begin < horizonTiles.size(); begin += BATCH_SIZE) {
            const size_t end = std::min(horizonTiles.size(), begin + BATCH_SIZE);
            struct MeshRequest {
                FarTerrainViewTile tile;
                std::future<std::shared_ptr<const FarTerrainMesh>> future;
            };
            std::vector<MeshRequest> requests;
            requests.reserve(end - begin);
            for (size_t index = begin; index < end; ++index) {
                const FarTerrainViewTile tile = horizonTiles[index];
                requests.push_back({
                    .tile = tile,
                    .future = std::async(std::launch::async,
                                         [source, tile] {
                                             return FarTerrainMesher::build(
                                                 {.tileX = tile.key.tileX,
                                                  .tileZ = tile.key.tileZ,
                                                  .step = FAR_TERRAIN_BASE_STEP},
                                                 source);
                                         }),
                });
            }
            for (MeshRequest& request : requests) {
                std::shared_ptr<const FarTerrainMesh> mesh;
                try {
                    mesh = request.future.get();
                } catch (const std::exception& exception) {
                    std::cerr << "Horizon base mesh failed for tile (" << request.tile.key.tileX
                              << ", " << request.tile.key.tileZ << "): " << exception.what()
                              << '\n';
                    return 6;
                }
                if (!mesh) {
                    std::cerr << "Horizon base mesh was not produced for tile ("
                              << request.tile.key.tileX << ", " << request.tile.key.tileZ << ")\n";
                    return 6;
                }
                ++horizonMeshCount;
                horizonMeshBytes += mesh->byteSize();
            }
        }
        horizonMacroControlCache = generator->macroControlCacheMetrics();
        horizonFarClimateControlCache = generator->farClimateControlCacheMetrics();
        horizonMeshMilliseconds = elapsedMilliseconds(horizonMeshStart);
    }

    AuthorityDeltaSummary authorityDelta{.requested = includeAuthorityDelta};
    if (includeAuthorityDelta) {
        const NativePoint deltaNative = worldBlockToNative(worldX, worldZ);
        const TerrainPageCoordinate deltaCoordinate = terrainPageCoordinateFor(deltaNative);
        const std::shared_ptr<WorldGenerationContext> previewContext =
            context->quality() == AuthorityQuality::PREVIEW
                ? context
                : context->withQuality(AuthorityQuality::PREVIEW);
        const std::shared_ptr<WorldGenerationContext> finalContext =
            context->quality() == AuthorityQuality::FINAL
                ? context
                : context->withQuality(AuthorityQuality::FINAL);
        const TerrainRuntimeMetrics beforeDelta = runtime.metrics();
        const auto deltaStart = Clock::now();
        const auto prepareDeltaPage = [&](const std::shared_ptr<WorldGenerationContext>& owner,
                                          AuthorityRequestPriority priority) {
            AuthorityResult<bool> result = owner->requestAuthorityPage(deltaCoordinate, priority);
            const auto deltaDeadline = Clock::now() + std::chrono::minutes(5);
            while (result.status() == AuthorityStatus::DEFERRED && Clock::now() < deltaDeadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                result = owner->requestAuthorityPage(deltaCoordinate, priority);
            }
            return result;
        };
        const auto previewDeltaStart = Clock::now();
        const AuthorityResult<bool> previewPrepared =
            prepareDeltaPage(previewContext, AuthorityRequestPriority::COARSE_PREVIEW);
        authorityDelta.previewMilliseconds = elapsedMilliseconds(previewDeltaStart);
        const TerrainRuntimeMetrics afterPreviewDelta = runtime.metrics();
        const auto finalDeltaStart = Clock::now();
        const AuthorityResult<bool> finalPrepared =
            prepareDeltaPage(finalContext, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
        authorityDelta.finalMilliseconds = elapsedMilliseconds(finalDeltaStart);
        if (!previewPrepared.isReady() || !finalPrepared.isReady()) {
            std::cerr << "Authority delta pages did not become ready: preview="
                      << (previewPrepared.failure() ? previewPrepared.failure()->message : "ready")
                      << " final="
                      << (finalPrepared.failure() ? finalPrepared.failure()->message : "ready")
                      << '\n';
            return 5;
        }
        TerrainPageStore deltaStore(*authorityProfile / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
                                    context->identity());
        const AuthorityResult<TerrainAuthorityPage> previewPage = deltaStore.loadPage(
            {.quality = AuthorityQuality::PREVIEW, .coordinate = deltaCoordinate});
        const AuthorityResult<TerrainAuthorityPage> finalPage = deltaStore.loadPage(
            {.quality = AuthorityQuality::FINAL, .coordinate = deltaCoordinate});
        if (!previewPage.isReady() || !finalPage.isReady() ||
            previewPage.value()->samples.size() != finalPage.value()->samples.size()) {
            std::cerr << "Authority delta pages could not be compared after publication\n";
            return 5;
        }
        const auto pageHash = [](const TerrainAuthorityPage& page) {
            const std::span<const uint8_t> bytes{
                reinterpret_cast<const uint8_t*>(page.samples.data()), page.byteSize()};
            return sha256Hex(sha256(bytes));
        };
        authorityDelta.previewPageHash = pageHash(*previewPage.value());
        authorityDelta.finalPageHash = pageHash(*finalPage.value());
        std::vector<double> absoluteMeters;
        std::vector<double> boundaryAbsoluteMeters;
        std::vector<double> absoluteBlocks;
        absoluteMeters.reserve(previewPage.value()->samples.size());
        boundaryAbsoluteMeters.reserve(4 * learned::AUTHORITY_PAGE_NATIVE_EDGE);
        absoluteBlocks.reserve(previewPage.value()->samples.size());
        double signedTotal = 0.0;
        double squaredTotal = 0.0;
        for (size_t index = 0; index < previewPage.value()->samples.size(); ++index) {
            const double previewMeters = previewPage.value()->samples[index].elevationMeters;
            const double finalMeters = finalPage.value()->samples[index].elevationMeters;
            const double deltaMeters = finalMeters - previewMeters;
            const double absolute = std::abs(deltaMeters);
            absoluteMeters.push_back(absolute);
            signedTotal += deltaMeters;
            squaredTotal += deltaMeters * deltaMeters;
            const size_t row = index / learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const size_t column = index % learned::AUTHORITY_PAGE_NATIVE_EDGE;
            if (row == 0 || column == 0 || row + 1 == learned::AUTHORITY_PAGE_NATIVE_EDGE ||
                column + 1 == learned::AUTHORITY_PAGE_NATIVE_EDGE) {
                boundaryAbsoluteMeters.push_back(absolute);
            }
            absoluteBlocks.push_back(std::abs(learnedElevationMetersToWorldHeight(finalMeters) -
                                              learnedElevationMetersToWorldHeight(previewMeters)));
            if ((previewMeters < 0.0) != (finalMeters < 0.0))
                ++authorityDelta.oceanSignDisagreements;
        }
        const double sampleCount = static_cast<double>(absoluteMeters.size());
        authorityDelta.meanSignedMeters = signedTotal / sampleCount;
        authorityDelta.rootMeanSquareMeters = std::sqrt(squaredTotal / sampleCount);
        authorityDelta.p95AbsoluteMeters = percentile95(absoluteMeters);
        authorityDelta.maximumAbsoluteMeters =
            *std::max_element(absoluteMeters.begin(), absoluteMeters.end());
        authorityDelta.boundaryP95AbsoluteMeters = percentile95(boundaryAbsoluteMeters);
        authorityDelta.boundaryMaximumAbsoluteMeters =
            *std::max_element(boundaryAbsoluteMeters.begin(), boundaryAbsoluteMeters.end());
        authorityDelta.p95AbsoluteBlocks = percentile95(absoluteBlocks);
        authorityDelta.maximumAbsoluteBlocks =
            *std::max_element(absoluteBlocks.begin(), absoluteBlocks.end());
        authorityDelta.milliseconds = elapsedMilliseconds(deltaStart);
        const TerrainRuntimeMetrics afterDelta = runtime.metrics();
        for (size_t model = 0; model < authorityDelta.modelCalls.size(); ++model) {
            authorityDelta.modelCalls[model] =
                afterDelta.models[model].calls - beforeDelta.models[model].calls;
            authorityDelta.previewModelCalls[model] =
                afterPreviewDelta.models[model].calls - beforeDelta.models[model].calls;
            authorityDelta.finalModelCalls[model] =
                afterDelta.models[model].calls - afterPreviewDelta.models[model].calls;
        }
    }

    const auto authorityStart = Clock::now();
    AuthorityResult<bool> prepared = context->requestWorldPage(
        worldX, worldZ,
        quality == AuthorityQuality::FINAL ? AuthorityRequestPriority::EXPLORATION_EXACT
                                           : AuthorityRequestPriority::COARSE_PREVIEW);
    const auto deadline = Clock::now() + std::chrono::minutes(5);
    while (prepared.status() == AuthorityStatus::DEFERRED && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        prepared = context->requestWorldPage(worldX, worldZ,
                                             quality == AuthorityQuality::FINAL
                                                 ? AuthorityRequestPriority::EXPLORATION_EXACT
                                                 : AuthorityRequestPriority::COARSE_PREVIEW);
    }
    if (!prepared.isReady()) {
        std::cerr << "Authority page did not become ready: "
                  << (prepared.failure() ? prepared.failure()->message : "unknown failure") << '\n';
        return 5;
    }
    const double authorityMilliseconds = elapsedMilliseconds(authorityStart);
    const AuthorityResult<PhysicalTerrainSample> sampled = context->sampleWorld(worldX, worldZ);
    if (!sampled.isReady()) {
        std::cerr << "Authority sample failed after page publication\n";
        return 5;
    }

    const NativePoint native = worldBlockToNative(worldX, worldZ);
    const TerrainPageKey pageKey{.quality = quality,
                                 .coordinate = terrainPageCoordinateFor(native)};
    TerrainPageStore store(*authorityProfile / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
                           context->identity());
    const AuthorityResult<TerrainAuthorityPage> page = store.loadPage(pageKey);
    if (!page.isReady()) {
        std::cerr << "Published authority page could not be reloaded\n";
        return 5;
    }
    const auto pageBytes = std::span(reinterpret_cast<const uint8_t*>(page.value()->samples.data()),
                                     page.value()->byteSize());
    const std::string pageHash = sha256Hex(sha256(pageBytes));

    std::optional<SurfaceSample> hydrology;
    double hydrologyMilliseconds = 0.0;
    if (includeHydrology) {
        ChunkGenerator generator(seed, context);
        const auto hydrologyStart = Clock::now();
        while (Clock::now() < deadline) {
            try {
                hydrology = generator.sampleFarSurface(worldX, worldZ, SurfaceFootprint::BLOCK_1);
                break;
            } catch (const GenerationFailureException& exception) {
                if (exception.status() != AuthorityStatus::DEFERRED) {
                    std::cerr << "Hydrology inspection failed: " << exception.what() << '\n';
                    return 6;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        if (!hydrology) {
            std::cerr << "Hydrology inspection exceeded five minutes\n";
            return 6;
        }
        hydrologyMilliseconds = elapsedMilliseconds(hydrologyStart);
    }

    const int64_t floraMinimumX =
        world_coord::floorMultiple(worldX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    const int64_t floraMinimumZ =
        world_coord::floorMultiple(worldZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    std::optional<std::vector<FarCanopy>> stepTwoCanopies;
    std::optional<std::vector<FarCanopy>> stepFourCanopies;
    std::optional<std::vector<FarCanopy>> stepEightCanopies;
    std::optional<std::vector<FarCanopy>> stepSixteenCanopies;
    std::optional<std::vector<FarCanopy>> stepThirtyTwoCanopies;
    double floraMilliseconds = 0.0;
    if (includeFlora) {
        ChunkGenerator generator(seed, context);
        const auto floraStart = Clock::now();
        const auto floraDeadline = floraStart + std::chrono::minutes(5);
        const auto collect = [&](int step, std::optional<std::vector<FarCanopy>>& output) {
            while (!output && Clock::now() < floraDeadline) {
                try {
                    output = generator.collectFarCanopiesForLod(
                        floraMinimumX, floraMinimumZ, floraMinimumX + FAR_TERRAIN_TILE_EDGE,
                        floraMinimumZ + FAR_TERRAIN_TILE_EDGE, step);
                } catch (const GenerationFailureException& exception) {
                    if (exception.status() != AuthorityStatus::DEFERRED) throw;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        };
        try {
            collect(2, stepTwoCanopies);
            collect(4, stepFourCanopies);
            collect(8, stepEightCanopies);
            collect(16, stepSixteenCanopies);
            collect(32, stepThirtyTwoCanopies);
        } catch (const GenerationFailureException& exception) {
            std::cerr << "Flora inspection failed: " << exception.what() << '\n';
            return 6;
        }
        if (!stepTwoCanopies || !stepFourCanopies || !stepEightCanopies || !stepSixteenCanopies ||
            !stepThirtyTwoCanopies) {
            std::cerr << "Flora inspection exceeded five minutes\n";
            return 6;
        }
        floraMilliseconds = elapsedMilliseconds(floraStart);
    }

    const TerrainRuntimeMetrics runtimeMetrics = runtime.metrics();
    const WorldGenerationMetrics generationMetrics = context->metrics();
    const NativeHydrologyCacheMetrics hydrologyMetrics =
        context->nativeHydrologyRouter()->cacheMetrics();
    const PhysicalTerrainSample& sample = *sampled.value();
    const GenerationIdentity& identity = context->identity();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n"
              << "  \"mode\": \"generator-v4\",\n"
              << "  \"seed\": " << seed << ",\n"
              << "  \"quality\": \"" << (quality == AuthorityQuality::FINAL ? "final" : "preview")
              << "\",\n"
              << "  \"fingerprint\": \"" << sha256Hex(context->fingerprint()) << "\",\n"
              << "  \"identity\": {\"generator_version\": " << identity.generatorVersion
              << ", \"model_pack_hash\": \"" << sha256Hex(identity.modelPackHash)
              << "\", \"runtime_hash\": \"" << sha256Hex(identity.runtimeHash)
              << "\", \"runtime_dylib\": \""
              << jsonEscaped(runtime.runtimeDylibPath().generic_string())
              << "\", \"provider\": " << static_cast<unsigned>(identity.provider.provider)
              << ", \"onnx_runtime\": [" << identity.provider.onnxRuntimeMajorVersion << ", "
              << identity.provider.onnxRuntimeMinorVersion << ", "
              << identity.provider.onnxRuntimePatchVersion
              << "], \"provider_flags\": " << identity.provider.flags
              << ", \"model_block_scale\": " << identity.modelBlockScale
              << ", \"rng_revision\": " << identity.rngRevision
              << ", \"quantization_revision\": " << identity.quantizationRevision
              << ", \"hydrology_revision\": " << identity.hydrologyRevision
              << ", \"postprocessing_revision\": " << identity.postprocessingRevision
              << ", \"windows\": {\"coarse\": [" << identity.coarseWindow.edge << ", "
              << identity.coarseWindow.stride << ", " << identity.coarseWindow.inferenceSteps
              << ", " << identity.coarseWindow.batchSize << "], \"latent\": ["
              << identity.latentWindow.edge << ", " << identity.latentWindow.stride << ", "
              << identity.latentWindow.inferenceSteps << ", " << identity.latentWindow.batchSize
              << "], \"decoder\": [" << identity.decoderWindow.edge << ", "
              << identity.decoderWindow.stride << ", " << identity.decoderWindow.inferenceSteps
              << ", " << identity.decoderWindow.batchSize << "]}},\n"
              << "  \"authority_profile\": \"" << jsonEscaped(authorityProfile->generic_string())
              << "\",\n"
              << "  \"qualification_hash\": \""
              << (runtimeMetrics.qualificationDigest
                      ? sha256Hex(*runtimeMetrics.qualificationDigest)
                      : std::string{})
              << "\",\n"
              << "  \"authority_page_hash\": \"" << pageHash << "\",\n"
              << "  \"timing_ms\": {\"setup\": " << setupMilliseconds
              << ", \"authority_page\": " << authorityMilliseconds
              << ", \"hydrology\": " << hydrologyMilliseconds
              << ", \"flora\": " << floraMilliseconds << ", \"dry_spawn\": " << drySpawnMilliseconds
              << ", \"dry_spawn_selection\": " << drySpawnSelectionMilliseconds
              << ", \"dry_spawn_water_screen\": " << drySpawnWaterScreenMilliseconds
              << ", \"horizon\": " << horizonMilliseconds
              << ", \"horizon_hydrology\": " << horizonHydrologyMilliseconds
              << ", \"horizon_mesh\": " << horizonMeshMilliseconds
              << ", \"authority_delta\": " << authorityDelta.milliseconds << "},\n"
              << "  \"authority_delta\": ";
    if (!authorityDelta.requested) {
        std::cout << "null,\n";
    } else {
        std::cout << "{\"mean_signed_m\": " << authorityDelta.meanSignedMeters
                  << ", \"rms_m\": " << authorityDelta.rootMeanSquareMeters
                  << ", \"p95_abs_m\": " << authorityDelta.p95AbsoluteMeters
                  << ", \"max_abs_m\": " << authorityDelta.maximumAbsoluteMeters
                  << ", \"boundary_p95_abs_m\": " << authorityDelta.boundaryP95AbsoluteMeters
                  << ", \"boundary_max_abs_m\": " << authorityDelta.boundaryMaximumAbsoluteMeters
                  << ", \"p95_abs_blocks\": " << authorityDelta.p95AbsoluteBlocks
                  << ", \"max_abs_blocks\": " << authorityDelta.maximumAbsoluteBlocks
                  << ", \"ocean_sign_disagreements\": " << authorityDelta.oceanSignDisagreements
                  << ", \"preview_page_hash\": \"" << authorityDelta.previewPageHash
                  << "\", \"final_page_hash\": \"" << authorityDelta.finalPageHash << "\""
                  << ", \"preview_ms\": " << authorityDelta.previewMilliseconds
                  << ", \"final_ms\": " << authorityDelta.finalMilliseconds
                  << ", \"model_calls\": {\"coarse\": " << authorityDelta.modelCalls[0]
                  << ", \"base\": " << authorityDelta.modelCalls[1]
                  << ", \"decoder\": " << authorityDelta.modelCalls[2]
                  << "}, \"preview_model_calls\": {\"coarse\": "
                  << authorityDelta.previewModelCalls[0]
                  << ", \"base\": " << authorityDelta.previewModelCalls[1]
                  << ", \"decoder\": " << authorityDelta.previewModelCalls[2]
                  << "}, \"final_model_calls\": {\"coarse\": " << authorityDelta.finalModelCalls[0]
                  << ", \"base\": " << authorityDelta.finalModelCalls[1]
                  << ", \"decoder\": " << authorityDelta.finalModelCalls[2] << "}},\n";
    }
    std::cout << "  \"dry_spawn\": ";
    if (!includeDrySpawn) {
        std::cout << "null,\n";
    } else if (drySpawn) {
        std::cout << "{\"ordinal\": " << drySpawnOrdinal
                  << ", \"canonical_water_rejections\": " << drySpawnWaterRejections
                  << ", \"canonical_local_relocations\": " << drySpawnLocalRelocations
                  << ", \"x\": " << drySpawn->x << ", \"y\": " << drySpawn->y
                  << ", \"z\": " << drySpawn->z << "},\n";
    } else {
        std::cout << "{\"ordinal\": " << drySpawnOrdinal << ", \"failure\": \""
                  << drySpawnFailure.value_or("unknown dry-spawn failure") << "\"},\n";
    }
    std::cout << "  \"horizon\": {\"radius_chunks\": " << horizonChunkRadius
              << ", \"preview_pages\": " << horizonPageCount
              << ", \"hydrology_pages\": " << horizonHydrologyPageCount
              << ", \"base_meshes\": " << horizonMeshCount
              << ", \"base_mesh_bytes\": " << horizonMeshBytes
              << ", \"macro_control_builds\": " << horizonMacroControlCache.builds
              << ", \"far_climate_builds\": " << horizonFarClimateControlCache.builds
              << ", \"far_climate_evictions\": " << horizonFarClimateControlCache.evictions
              << ", \"far_climate_single_flight_waits\": "
              << horizonFarClimateControlCache.singleFlightWaits << "},\n";
    std::cout << "  \"flora\": ";
    if (!includeFlora) {
        std::cout << "null,\n";
    } else {
        std::cout << "{\"tile_minimum_x\": " << floraMinimumX
                  << ", \"tile_minimum_z\": " << floraMinimumZ
                  << ", \"step_2_canopies\": " << stepTwoCanopies->size()
                  << ", \"step_4_canopies\": " << stepFourCanopies->size()
                  << ", \"step_8_canopies\": " << stepEightCanopies->size()
                  << ", \"step_16_canopies\": " << stepSixteenCanopies->size()
                  << ", \"step_32_canopies\": " << stepThirtyTwoCanopies->size() << "},\n";
    }
    if (horizonWaterProfile) {
        std::lock_guard lock(horizonWaterProfile->mutex);
        std::cout << "  \"horizon_water_profile\": {\"evaluated_tiles\": "
                  << horizonWaterProfile->evaluatedTiles
                  << ", \"uniform_standing_tiles\": " << horizonWaterProfile->uniformStandingTiles
                  << ", \"rejected_dry_or_mixed_tiles\": "
                  << horizonWaterProfile->rejectedDryOrMixedTiles
                  << ", \"rejected_moving_water_tiles\": "
                  << horizonWaterProfile->rejectedMovingWaterTiles
                  << ", \"rejected_fall_tiles\": " << horizonWaterProfile->rejectedFallTiles
                  << ", \"rejected_authority_tiles\": "
                  << horizonWaterProfile->rejectedAuthorityTiles
                  << ", \"rejected_topology_tiles\": " << horizonWaterProfile->rejectedTopologyTiles
                  << ", \"topology_marked_parents\": " << horizonWaterProfile->topologyMarkedParents
                  << ", \"canonical_water_grid_calls\": "
                  << horizonWaterProfile->canonicalWaterGridCalls
                  << ", \"canonical_water_grid_samples\": "
                  << horizonWaterProfile->canonicalWaterGridSamples
                  << ", \"full_native_water_page_calls\": "
                  << horizonWaterProfile->fullNativeWaterPageCalls
                  << ", \"full_native_water_page_samples\": "
                  << horizonWaterProfile->fullNativeWaterPageSamples << "},\n";
    }
    std::cout
        << "  \"provider\": {\"core_ml_partitions\": " << runtimeMetrics.coreMlPartitions
        << ", \"core_ml_nodes\": " << runtimeMetrics.coreMlNodes
        << ", \"cpu_fallback_partitions\": " << runtimeMetrics.cpuFallbackPartitions
        << ", \"cpu_fallback_nodes\": " << runtimeMetrics.cpuFallbackNodes
        << ", \"static_base_batch\": "
        << ((context->identity().provider.flags & CORE_ML_STATIC_BASE_BATCH_FOUR) != 0
                ? LATENT_WINDOW.batchSize
                : 0)
        << ", \"cpu_fallback_intra_op_threads\": " << runtimeMetrics.cpuFallbackIntraOpThreads
        << ", \"maximum_concurrent_calls\": " << runtimeMetrics.maximumConcurrentInferenceCalls
        << "},\n"
        << "  \"sessions\": {\"compiled\": " << runtimeMetrics.compiledSessions
        << ", \"resident\": " << runtimeMetrics.residentSessions
        << ", \"peak_resident\": " << runtimeMetrics.peakResidentSessions
        << ", \"creations\": {\"coarse\": " << runtimeMetrics.models[0].sessionCreations
        << ", \"base\": " << runtimeMetrics.models[1].sessionCreations
        << ", \"decoder\": " << runtimeMetrics.models[2].sessionCreations << "}},\n"
        << "  \"inference_calls\": {\"coarse\": " << runtimeMetrics.models[0].calls
        << ", \"base\": " << runtimeMetrics.models[1].calls
        << ", \"decoder\": " << runtimeMetrics.models[2].calls << "},\n"
        << "  \"inference_ms\": {\"coarse\": {\"total\": "
        << static_cast<double>(runtimeMetrics.models[0].inferenceNanoseconds) / 1'000'000.0
        << ", \"maximum\": "
        << static_cast<double>(runtimeMetrics.models[0].maximumInferenceNanoseconds) / 1'000'000.0
        << "}, \"base\": {\"total\": "
        << static_cast<double>(runtimeMetrics.models[1].inferenceNanoseconds) / 1'000'000.0
        << ", \"maximum\": "
        << static_cast<double>(runtimeMetrics.models[1].maximumInferenceNanoseconds) / 1'000'000.0
        << "}, \"decoder\": {\"total\": "
        << static_cast<double>(runtimeMetrics.models[2].inferenceNanoseconds) / 1'000'000.0
        << ", \"maximum\": "
        << static_cast<double>(runtimeMetrics.models[2].maximumInferenceNanoseconds) / 1'000'000.0
        << "}},\n"
        << "  \"authority_cache\": {\"entries\": " << generationMetrics.authorityCache.entries
        << ", \"bytes\": " << generationMetrics.authorityCache.bytes
        << ", \"queued\": " << generationMetrics.authorityCache.queuedBuilds
        << ", \"active\": " << generationMetrics.authorityCache.activeBuilds
        << ", \"publication_queued\": " << generationMetrics.authorityCache.queuedPublications
        << ", \"publication_active\": " << generationMetrics.authorityCache.activePublications
        << ", \"publication_peak\": " << generationMetrics.authorityCache.peakConcurrentPublications
        << ", \"publication_writes\": " << generationMetrics.authorityCache.publicationWrites
        << ", \"batches\": " << generationMetrics.authorityCache.batches
        << ", \"batched_pages\": " << generationMetrics.authorityCache.batchedPages << "},\n"
        << "  \"sample\": {\"x\": " << worldX << ", \"z\": " << worldZ
        << ", \"elevation_m\": " << sample.elevationMeters
        << ", \"temperature_c\": " << sample.meanTemperatureC
        << ", \"temperature_variability_c\": " << sample.temperatureVariabilityC
        << ", \"precipitation_mm\": " << sample.annualPrecipitationMm
        << ", \"precipitation_cv\": " << sample.precipitationCoefficientOfVariation
        << ", \"lapse_rate_c_per_m\": " << sample.lapseRateCPerMeter << "},\n"
        << "  \"hydrology_cache\": {\"entries\": " << hydrologyMetrics.entries
        << ", \"bytes\": " << hydrologyMetrics.bytes << ", \"builds\": " << hydrologyMetrics.builds
        << ", \"persisted_loads\": " << hydrologyMetrics.persistedLoads
        << ", \"parallel_peak\": " << hydrologyMetrics.peakConcurrentBuilds
        << ", \"admission_waits\": " << hydrologyMetrics.buildAdmissionWaits
        << ", \"connected_wetland_entries\": " << hydrologyMetrics.connectedWetlandEntries
        << ", \"sea_backwater_entries\": " << hydrologyMetrics.seaBackwaterEntries << "},\n"
        << "  \"hydrology\": ";
    if (!hydrology) {
        std::cout << "null\n";
    } else {
        std::cout << "{\"terrain_height\": " << hydrology->terrainHeight
                  << ", \"emitted_terrain_height\": " << geometryTerrainHeight(*hydrology)
                  << ", \"bed_elevation\": " << hydrology->hydrology.surfaceElevation
                  << ", \"water_surface\": " << hydrology->waterSurface
                  << ", \"generated_fluid_level\": "
                  << static_cast<unsigned>(hydrology->hydrology.generatedFluidLevel)
                  << ", \"water_body_id\": ";
        writeWaterBodyId(std::cout, hydrology->hydrology.waterBodyId);
        std::cout << ", \"transition_owner_kind\": "
                  << static_cast<unsigned>(hydrology->hydrology.transitionOwnerKind)
                  << ", \"transition_owner_id\": ";
        writeWaterBodyId(std::cout, hydrology->hydrology.transitionOwnerId);
        std::cout << ", \"flow_direction\": [" << hydrology->hydrology.flowDirection.x << ", "
                  << hydrology->hydrology.flowDirection.z << "]"
                  << ", \"terrain_slope\": " << hydrology->hydrology.terrainSlope
                  << ", \"river\": " << (hydrology->hydrology.river ? "true" : "false")
                  << ", \"lake\": " << (hydrology->hydrology.lake ? "true" : "false")
                  << ", \"ocean\": " << (hydrology->hydrology.ocean ? "true" : "false")
                  << ", \"wetland\": " << (hydrology->hydrology.wetland ? "true" : "false")
                  << ", \"delta\": " << (hydrology->hydrology.delta ? "true" : "false")
                  << ", \"estuary\": " << (hydrology->hydrology.estuary ? "true" : "false")
                  << ", \"brackish\": " << (hydrology->hydrology.brackish ? "true" : "false")
                  << ", \"endorheic\": " << (hydrology->hydrology.endorheic ? "true" : "false")
                  << ", \"perennial\": " << (hydrology->hydrology.perennial ? "true" : "false")
                  << ", \"ephemeral\": " << (hydrology->hydrology.ephemeral ? "true" : "false")
                  << ", \"discharge\": " << hydrology->hydrology.discharge
                  << ", \"baseflow\": " << hydrology->hydrology.baseflow
                  << ", \"channel_distance\": " << hydrology->hydrology.channelDistance
                  << ", \"channel_width\": " << hydrology->hydrology.channelWidth
                  << ", \"channel_depth\": " << hydrology->hydrology.channelDepth
                  << ", \"channel_gradient\": " << hydrology->hydrology.channelGradient
                  << ", \"signed_shoreline_distance\": " << hydrology->hydrology.lakeShoreDistance
                  << ", \"shore_water_surface\": " << hydrology->hydrology.shoreWaterSurface
                  << ", \"lake_spill_surface\": " << hydrology->hydrology.lakeSpillSurface
                  << ", \"hydroperiod\": " << hydrology->hydrology.hydroperiod
                  << ", \"groundwater_head\": " << hydrology->hydrology.groundwaterHead
                  << ", \"waterfall_top\": " << hydrology->hydrology.waterfallTop
                  << ", \"waterfall_bottom\": " << hydrology->hydrology.waterfallBottom
                  << ", \"waterfall_width\": " << hydrology->hydrology.waterfallWidth
                  << ", \"stream_order\": "
                  << static_cast<unsigned>(hydrology->hydrology.streamOrder)
                  << ", \"distributary_count\": "
                  << static_cast<unsigned>(hydrology->hydrology.distributaryCount)
                  << ", \"temperature_c\": " << hydrology->climate.temperatureC
                  << ", \"precipitation_mm\": " << hydrology->climate.annualPrecipitationMm
                  << ", \"soil_moisture\": " << hydrology->soil.moisture
                  << ", \"soil_fertility\": " << hydrology->soil.fertility
                  << ", \"primary_biome\": \"" << biomeName(hydrology->biome.primary)
                  << "\", \"secondary_biome\": \"" << biomeName(hydrology->biome.secondary)
                  << "\", \"tree_cover\": " << feature_generation::treeCoverDensity(*hydrology)
                  << "}\n";
    }
    std::cout << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--v4-model") {
        try {
            return runV4Inspector(argc, argv);
        } catch (const std::exception& exception) {
            std::cerr << "Generator v4 inspection failed: " << exception.what() << '\n';
            return 6;
        }
    }
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: rycraft_worldgen_inspect [seed] [sample_x sample_z]\n"
                     "       rycraft_worldgen_inspect --v4-model MODEL_PACK [seed] [x z] "
                     "[preview|final] [--profile ABSOLUTE_PATH] [--hydrology] [--flora] "
                     "[--dry-spawn] [--authority-delta] "
                     "[--horizon|--horizon-hydrology|--horizon-mesh|"
                     "--horizon-water-profile] [--horizon-radius CHUNKS]\n";
        return 0;
    }
    uint64_t seed = 42;
    if (const char* environmentSeed = std::getenv("RYCRAFT_WORLD_SEED")) {
        seed = std::strtoull(environmentSeed, nullptr, 0);
    }
    if (argc > 1) seed = std::strtoull(argv[1], nullptr, 0);
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
        record("estuary", hydrology.estuary);
        record("wetland", hydrology.wetland);
        record("groundwater_interface",
               hydrology.wetland && hydrology.groundwaterHead >= hydrology.waterSurface);
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
        const double waterDepth =
            hydrology.ocean || hydrology.lake || hydrology.river || hydrology.wetland
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
                sample.hydrology.ocean || sample.hydrology.lake || sample.hydrology.river ||
                        sample.hydrology.wetland
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

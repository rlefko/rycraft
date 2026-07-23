#include "world/features.hpp"

#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace {

constexpr int MAX_TREE_SPACING = 14;
constexpr int TREE_COMPETITOR_CELL_RADIUS =
    (MAX_TREE_SPACING + feature_generation::TREE_CELL_EDGE - 1) /
        feature_generation::TREE_CELL_EDGE +
    1;
constexpr uint64_t TREE_CANDIDATE_STREAM = 0x5452454543414E44ULL;
constexpr uint64_t TREE_PRIORITY_STREAM = 0x545245455052494FULL;
constexpr uint64_t TREE_SHAPE_STREAM = 0x5452454553484150ULL;
constexpr uint64_t FAR_CANOPY_CLUSTER_STREAM = 0x46415243414E4F50ULL;
constexpr uint64_t FAR_FLORA_CANDIDATE_STREAM = 0x464152464C4F5241ULL;
constexpr uint64_t FAR_FLORA_RETENTION_STREAM = 0x464C4F52414C4F44ULL;
constexpr uint64_t FLORA_STREAM = 0x464C4F524143454CULL;
constexpr uint64_t FLORA_BIOME_STREAM = 0x464C4F524142494FULL;
constexpr int64_t FAR_CANOPY_CLUSTER_CELL_EDGE = 64;
constexpr int FAR_CANOPY_CLUSTER_CANDIDATE_COUNT = 6;
constexpr int64_t FAR_FLORA_CELL_EDGE = 8;
constexpr int FAR_FLORA_CANDIDATE_COUNT = 4;
constexpr double FAR_CANOPY_EXACT_OPPORTUNITIES_PER_CLUSTER_CANDIDATE =
    static_cast<double>((FAR_CANOPY_CLUSTER_CELL_EDGE / feature_generation::TREE_CELL_EDGE) *
                        (FAR_CANOPY_CLUSTER_CELL_EDGE / feature_generation::TREE_CELL_EDGE)) /
    FAR_CANOPY_CLUSTER_CANDIDATE_COUNT;

using TreeKind = feature_generation::TreeSpecies;

struct SpeciesTraits {
    double minimumTemperatureC;
    double maximumTemperatureC;
    double minimumPrecipitationMm;
    double maximumPrecipitationMm;
    double minimumMoisture;
    double minimumFertility;
    double maximumSlope;
    double minimumAltitude;
    double maximumAltitude;
    double minimumLight;
    bool toleratesFlooding;
    int maximumFloodDepth;
    int spacing;
};

constexpr SpeciesTraits traitsFor(TreeKind kind) {
    switch (kind) {
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK:
            return {-8.0, 29.0, 450.0, 2800.0, 0.28, 0.22, 1.20, 48.0, 190.0, 0.45, false, 0, 9};
        case TreeKind::BIRCH:
            return {-14.0, 22.0, 380.0, 2400.0, 0.30, 0.18, 1.30, 50.0, 210.0, 0.52, false, 0, 8};
        case TreeKind::SPRUCE:
            return {-24.0, 31.0, 280.0, 2800.0, 0.30, 0.12, 1.45, 45.0, 285.0, 0.35, false, 0, 8};
        case TreeKind::ACACIA:
            return {15.0, 38.0, 180.0, 1700.0, 0.12, 0.12, 1.10, 48.0, 170.0, 0.70, false, 0, 11};
        case TreeKind::JUNGLE:
            return {18.0, 38.0, 1250.0, 3800.0, 0.65, 0.35, 1.05, 48.0, 175.0, 0.28, false, 0, 12};
        case TreeKind::MANGROVE:
            return {16.0, 36.0, 1000.0, 3800.0, 0.68, 0.22, 0.75, 48.0, 84.0, 0.38, true, 3, 9};
        case TreeKind::PALM:
            return {17.0, 39.0, 350.0, 3300.0, 0.24, 0.10, 0.85, 48.0, 105.0, 0.64, false, 0, 12};
        case TreeKind::WILLOW:
            return {-2.0, 29.0, 650.0, 3500.0, 0.54, 0.22, 0.85, 48.0, 125.0, 0.32, true, 2, 10};
        case TreeKind::ALPINE_SCRUB:
            return {-24.0, 13.0, 120.0, 1900.0, 0.10, 0.05, 1.55, 82.0, 350.0, 0.58, false, 0, 7};
        case TreeKind::FALLEN_LOG:
            return {-12.0, 34.0, 350.0, 3300.0, 0.25, 0.10, 0.70, 48.0, 190.0, 0.18, false, 0, 14};
        case TreeKind::COUNT:
            break;
    }
    return {};
}

static_assert(traitsFor(TreeKind::SPRUCE).minimumTemperatureC <= 22.0 &&
              traitsFor(TreeKind::SPRUCE).maximumTemperatureC >= 22.0);

double rangeSuitability(double value, double minimum, double maximum) {
    if (value < minimum || value > maximum) return 0.0;
    const double middle = (minimum + maximum) * 0.5;
    const double halfRange = std::max((maximum - minimum) * 0.5, 0.001);
    return std::clamp(1.0 - std::abs(value - middle) / halfRange * 0.35, 0.0, 1.0);
}

double biomeTreeDensity(Biome biome) {
    switch (biome) {
        case Biome::TROPICAL_RAINFOREST:
            return 0.92;
        case Biome::TEMPERATE_RAINFOREST:
            return 0.82;
        case Biome::FOREST:
        case Biome::BIRCH_FOREST:
            return 0.72;
        case Biome::TAIGA:
            return 0.68;
        case Biome::TEMPERATE_CONIFER_FOREST:
            return 0.74;
        case Biome::TROPICAL_CONIFER_FOREST:
            return 0.62;
        case Biome::TROPICAL_DRY_FOREST:
            return 0.38;
        case Biome::MANGROVE:
            return 0.78;
        case Biome::SWAMP:
            return 0.52;
        case Biome::FLOODED_GRASSLAND:
            return 0.16;
        case Biome::SAVANNA:
            return 0.34;
        case Biome::MEDITERRANEAN_WOODLAND:
            return 0.28;
        case Biome::SHRUBLAND:
            return 0.18;
        case Biome::ALPINE:
            return 0.24;
        case Biome::MONTANE_GRASSLAND:
            return 0.14;
        case Biome::PLAINS:
        case Biome::FLOWER_FIELD:
        case Biome::BEACH:
            return 0.09;
        default:
            return 0.0;
    }
}

double blendedTreeDensity(const worldgen::SurfaceSample& surface) {
    const double primaryDensity = biomeTreeDensity(surface.biome.primary);
    const double secondaryDensity = biomeTreeDensity(surface.biome.secondary);
    const double secondaryWeight =
        worldgen::biomeBlendWeight(surface.biome, surface.biome.secondary);
    if (surface.biome.primary == surface.biome.secondary) return primaryDensity;
    return primaryDensity * (1.0 - secondaryWeight) + secondaryDensity * secondaryWeight;
}

double maximumBiomeSuitability(const worldgen::SurfaceSample& surface) {
    return static_cast<double>(
        *std::max_element(surface.suitability.scores.begin(), surface.suitability.scores.end()));
}

double normalizedBiomeSuitability(const worldgen::SurfaceSample& surface, Biome biome,
                                  double maximum) {
    if (!(maximum > 1.0e-6)) return 0.0;
    return std::clamp(static_cast<double>(surface.suitability.scores[static_cast<size_t>(biome)]) /
                          maximum,
                      0.0, 1.0);
}

double biomeSignal(const worldgen::SurfaceSample& surface, Biome biome, double maximumSuitability) {
    return std::max(normalizedBiomeSuitability(surface, biome, maximumSuitability),
                    worldgen::biomeBlendWeight(surface.biome, biome));
}

double continuousTreeDensity(const worldgen::SurfaceSample& surface) {
    double maximumScore = 0.0;
    double forestScore = 0.0;
    for (size_t index = 0; index < surface.suitability.scores.size(); ++index) {
        const double score = std::max(0.0, static_cast<double>(surface.suitability.scores[index]));
        maximumScore = std::max(maximumScore, score);
        forestScore = std::max(forestScore, score * biomeTreeDensity(static_cast<Biome>(index)));
    }
    const double suitabilityDensity = maximumScore > 1.0e-9 ? forestScore / maximumScore : 0.0;
    const double biomeDensity = std::max(blendedTreeDensity(surface), suitabilityDensity * 0.94);
    const double moistureCover = std::clamp((surface.soil.moisture - 0.08) / 0.52, 0.0, 1.0);
    const double fertilityCover = std::clamp((surface.soil.fertility - 0.05) / 0.48, 0.0, 1.0);
    const double precipitationCover =
        std::clamp((surface.climate.annualPrecipitationMm - 180.0) / 1350.0, 0.0, 1.0);
    const double slopeCover =
        std::clamp(1.0 - std::max(0.0, surface.slope - 0.18) / 1.55, 0.08, 1.0);
    const double floodplain =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::FLOODPLAIN);
    const double riparian = std::max(
        {worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::LAKESHORE),
         floodplain});
    const double volcanicStress =
        std::clamp((surface.geology.volcanicActivity - 0.24) / 0.50, 0.0, 1.0);
    const double tectonicStress =
        std::clamp(volcanicStress * 0.82 + surface.geology.faultStrength * 0.18 +
                       std::max(0.0, surface.geology.uplift - 0.65) * 0.20,
                   0.0, 0.97);
    const double resourceFit =
        0.42 + moistureCover * 0.24 + fertilityCover * 0.20 + precipitationCover * 0.14;
    return std::clamp((biomeDensity * resourceFit + riparian * 0.10) * slopeCover *
                          (1.0 - tectonicStress),
                      0.0, 1.0);
}

double speciesBiomeAffinity(TreeKind kind, const worldgen::SurfaceSample& surface,
                            double maximumSuitability) {
    const auto signal = [&](Biome biome, double weight = 1.0) {
        return biomeSignal(surface, biome, maximumSuitability) * weight;
    };
    switch (kind) {
        case TreeKind::OAK:
            return std::max({signal(Biome::FOREST), signal(Biome::MEDITERRANEAN_WOODLAND, 0.88),
                             signal(Biome::TEMPERATE_RAINFOREST, 0.78), signal(Biome::PLAINS, 0.22),
                             signal(Biome::FLOWER_FIELD, 0.18)});
        case TreeKind::LARGE_OAK:
            return std::max({signal(Biome::FOREST, 0.92), signal(Biome::TEMPERATE_RAINFOREST),
                             signal(Biome::MEDITERRANEAN_WOODLAND, 0.42)});
        case TreeKind::BIRCH:
            return std::max({signal(Biome::BIRCH_FOREST), signal(Biome::FOREST, 0.48),
                             signal(Biome::TAIGA, 0.28)});
        case TreeKind::SPRUCE:
            return std::max({signal(Biome::TAIGA), signal(Biome::TEMPERATE_CONIFER_FOREST),
                             signal(Biome::TROPICAL_CONIFER_FOREST, 0.72),
                             signal(Biome::MONTANE_GRASSLAND, 0.52)});
        case TreeKind::ACACIA:
            return std::max({signal(Biome::SAVANNA), signal(Biome::TROPICAL_DRY_FOREST),
                             signal(Biome::MEDITERRANEAN_WOODLAND, 0.62),
                             signal(Biome::SHRUBLAND, 0.25)});
        case TreeKind::JUNGLE:
            return std::max({signal(Biome::TROPICAL_RAINFOREST),
                             signal(Biome::TROPICAL_CONIFER_FOREST, 0.66),
                             signal(Biome::TROPICAL_DRY_FOREST, 0.38)});
        case TreeKind::MANGROVE:
            return std::max({signal(Biome::MANGROVE), signal(Biome::SWAMP, 0.48),
                             signal(Biome::FLOODED_GRASSLAND, 0.42)});
        case TreeKind::PALM:
            return std::max({signal(Biome::BEACH), signal(Biome::TROPICAL_DRY_FOREST, 0.58),
                             signal(Biome::TROPICAL_RAINFOREST, 0.38)});
        case TreeKind::WILLOW:
            return std::max({signal(Biome::TEMPERATE_RAINFOREST), signal(Biome::SWAMP),
                             signal(Biome::FLOODED_GRASSLAND, 0.82), signal(Biome::FOREST, 0.30)});
        case TreeKind::ALPINE_SCRUB:
            return std::max({signal(Biome::ALPINE), signal(Biome::MONTANE_GRASSLAND),
                             signal(Biome::TUNDRA, 0.36)});
        case TreeKind::FALLEN_LOG:
            return std::max({signal(Biome::FOREST), signal(Biome::TEMPERATE_RAINFOREST),
                             signal(Biome::TAIGA, 0.74),
                             signal(Biome::TEMPERATE_CONIFER_FOREST, 0.82)});
        case TreeKind::COUNT:
            return 0.0;
    }
    return 0.0;
}

double lithologyRootFit(const worldgen::SurfaceSample& surface) {
    double fit = 1.0;
    switch (surface.geology.rock) {
        case worldgen::RockType::LIMESTONE:
            fit = 1.08;
            break;
        case worldgen::RockType::BASALT:
            fit = 1.04;
            break;
        case worldgen::RockType::SANDSTONE:
            fit = 0.82;
            break;
        case worldgen::RockType::VOLCANIC:
            fit = 0.76;
            break;
        default:
            break;
    }
    const double contactStress =
        std::clamp((28.0 - std::abs(surface.geology.lithology.contactDistance)) / 28.0, 0.0, 1.0);
    return std::clamp(fit - contactStress * surface.geology.faultStrength * 0.16, 0.55, 1.10);
}

std::pair<BlockType, BlockType> canopyBlocks(TreeKind kind) {
    switch (kind) {
        case TreeKind::BIRCH:
            return {BlockType::BIRCH_LOG, BlockType::BIRCH_LEAVES};
        case TreeKind::SPRUCE:
            return {BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES};
        case TreeKind::ALPINE_SCRUB:
            return {BlockType::AIR, BlockType::SHRUB};
        case TreeKind::ACACIA:
            return {BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES};
        case TreeKind::JUNGLE:
            return {BlockType::JUNGLE_LOG, BlockType::JUNGLE_LEAVES};
        case TreeKind::MANGROVE:
            return {BlockType::MANGROVE_LOG, BlockType::MANGROVE_LEAVES};
        case TreeKind::PALM:
            return {BlockType::PALM_LOG, BlockType::PALM_LEAVES};
        case TreeKind::WILLOW:
            return {BlockType::WILLOW_LOG, BlockType::WILLOW_LEAVES};
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK:
        case TreeKind::FALLEN_LOG:
        case TreeKind::COUNT:
            return {BlockType::LOG, BlockType::LEAVES};
    }
    return {BlockType::LOG, BlockType::LEAVES};
}

struct CoarseCanopyDimensions {
    int topOffset = 0;
    int crownBottomOffset = 0;
    uint8_t radius = 0;
};

CoarseCanopyDimensions coarseCanopyDimensions(TreeKind kind, BlockType leafBlock,
                                              int heightVariation) {
    if (kind == TreeKind::ALPINE_SCRUB) {
        return {.topOffset = 1 + heightVariation,
                .crownBottomOffset = 0,
                .radius = static_cast<uint8_t>(1 + heightVariation)};
    }
    switch (leafBlock) {
        case BlockType::SPRUCE_LEAVES:
            return {.topOffset = 10 + heightVariation, .crownBottomOffset = 2, .radius = 3};
        case BlockType::ACACIA_LEAVES:
            return {.topOffset = 8 + heightVariation,
                    .crownBottomOffset = 7 + heightVariation,
                    .radius = 3};
        case BlockType::JUNGLE_LEAVES:
            return {.topOffset = 11 + heightVariation,
                    .crownBottomOffset = 8 + heightVariation,
                    .radius = 3};
        case BlockType::MANGROVE_LEAVES:
            return {.topOffset = 8 + heightVariation,
                    .crownBottomOffset = 5 + heightVariation,
                    .radius = 3};
        case BlockType::PALM_LEAVES:
            return {.topOffset = 10 + heightVariation,
                    .crownBottomOffset = 8 + heightVariation,
                    .radius = 3};
        case BlockType::WILLOW_LEAVES:
            return {.topOffset = 8 + heightVariation,
                    .crownBottomOffset = 4 + heightVariation,
                    .radius = 3};
        case BlockType::BIRCH_LEAVES:
        case BlockType::LEAVES:
        default:
            return {.topOffset = 7 + heightVariation,
                    .crownBottomOffset = 4 + heightVariation,
                    .radius = 2};
    }
}

Biome ditheredBiome(const worldgen::SurfaceSample& surface, const CounterRng& random,
                    uint64_t stream, int64_t x, int64_t z, uint32_t index = 0) {
    const double secondaryWeight = std::clamp(surface.biome.transition, 0.0, 0.5);
    return worldgen::multiscaleDitherThreshold(random, stream, x, z, index) < secondaryWeight
               ? surface.biome.secondary
               : surface.biome.primary;
}

double availableSurfaceLight(const worldgen::SurfaceSample& surface, double treeCover) {
    const double canopyShade = treeCover * 0.55;
    const double cloudShade = surface.climate.relativeHumidity * 0.10;
    const double terrainShade = std::min(surface.slope, 2.0) * 0.07;
    return std::clamp(1.0 - canopyShade - cloudShade - terrainShade, 0.18, 1.0);
}

double availableSurfaceLight(const worldgen::SurfaceSample& surface) {
    return availableSurfaceLight(surface, continuousTreeDensity(surface));
}

int generatedWaterDepth(const worldgen::SurfaceSample& surface, int groundSurfaceY) {
    if (!(surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake ||
          surface.hydrology.wetland) ||
        !std::isfinite(surface.waterSurface)) {
        return 0;
    }
    const double boundedWater = std::clamp(surface.waterSurface, static_cast<double>(WORLD_MIN_Y),
                                           static_cast<double>(WORLD_MAX_Y + 1));
    const int waterTopY = static_cast<int>(std::ceil(boundedWater)) - 1;
    return std::max(0, waterTopY - groundSurfaceY);
}

double floodHabitatAffinity(TreeKind kind, const worldgen::SurfaceSample& surface,
                            double speciesAffinity) {
    const double riverbank =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK);
    const double lakeshore =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::LAKESHORE);
    const double floodplain =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::FLOODPLAIN);
    const double delta =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::DELTA);
    const double coast =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::COAST);
    if (kind == TreeKind::MANGROVE) {
        return std::max({speciesAffinity, delta, coast * 0.78, lakeshore * 0.55, riverbank * 0.46});
    }
    if (kind == TreeKind::WILLOW) {
        if (surface.hydrology.ocean) return 0.0;
        return std::max({speciesAffinity, riverbank, lakeshore, floodplain * 0.82});
    }
    return 0.0;
}

int habitatGroundSurfaceY(const worldgen::SurfaceSample& surface) {
    // Final block-footprint samples report the emitted top face. The rooting
    // block is the voxel immediately below that face for dry and wet terrain.
    return static_cast<int>(std::ceil(worldgen::geometryTerrainHeight(surface))) - 1;
}

feature_generation::TreeHabitatEvaluation
evaluateTreeHabitatWithContext(TreeKind species, const worldgen::SurfaceSample& surface,
                               int groundSurfaceY, double cover, double maximumSuitability) {
    const SpeciesTraits traits = traitsFor(species);
    feature_generation::TreeHabitatEvaluation result;
    result.waterDepthBlocks = generatedWaterDepth(surface, groundSurfaceY);
    result.submerged = result.waterDepthBlocks > 0;

    const double affinity = speciesBiomeAffinity(species, surface, maximumSuitability);
    result.spacing =
        std::clamp(traits.spacing - static_cast<int>(std::lround(cover * 2.0 + affinity * 0.45)), 6,
                   MAX_TREE_SPACING);

    if (species == TreeKind::COUNT || surface.hydrology.waterfall ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL) ||
        surface.geology.volcanicActivity > 0.78 || surface.slope > traits.maximumSlope ||
        groundSurfaceY < traits.minimumAltitude || groundSurfaceY > traits.maximumAltitude ||
        surface.soil.moisture < traits.minimumMoisture ||
        surface.soil.fertility < traits.minimumFertility ||
        surface.climate.temperatureC < traits.minimumTemperatureC ||
        surface.climate.temperatureC > traits.maximumTemperatureC ||
        surface.climate.annualPrecipitationMm < traits.minimumPrecipitationMm ||
        surface.climate.annualPrecipitationMm > traits.maximumPrecipitationMm ||
        availableSurfaceLight(surface, cover) < traits.minimumLight || affinity < 0.055) {
        return result;
    }

    double hydrologyFit = 1.0;
    if (result.submerged) {
        const double floodedAffinity = floodHabitatAffinity(species, surface, affinity);
        if (!traits.toleratesFlooding || result.waterDepthBlocks > traits.maximumFloodDepth ||
            floodedAffinity < 0.22) {
            return result;
        }
        hydrologyFit = std::clamp(0.55 + floodedAffinity * 0.45, 0.0, 1.0);
    } else if (species == TreeKind::MANGROVE || species == TreeKind::WILLOW) {
        hydrologyFit =
            std::clamp(0.45 + floodHabitatAffinity(species, surface, affinity) * 0.55, 0.0, 1.0);
    } else {
        const double floodplain = worldgen::MacroGenerationSampler::ecotopeInfluence(
            surface, worldgen::Ecotope::FLOODPLAIN);
        hydrologyFit = std::clamp(1.0 - floodplain * 0.38, 0.0, 1.0);
    }

    const double climateFit = rangeSuitability(
        surface.climate.temperatureC, traits.minimumTemperatureC, traits.maximumTemperatureC);
    const double precipitationFit =
        rangeSuitability(surface.climate.annualPrecipitationMm, traits.minimumPrecipitationMm,
                         traits.maximumPrecipitationMm);
    const double moistureFit =
        std::clamp((surface.soil.moisture - traits.minimumMoisture) * 1.55 + 0.52, 0.0, 1.0);
    const double fertilityFit =
        std::clamp((surface.soil.fertility - traits.minimumFertility) * 1.70 + 0.50, 0.0, 1.0);
    const double slopeFit =
        std::clamp(1.0 - surface.slope / std::max(0.01, traits.maximumSlope) * 0.62, 0.0, 1.0);
    const double volcanicStress =
        std::clamp((surface.geology.volcanicActivity - 0.22) / 0.56, 0.0, 1.0);
    const double tectonicStress =
        std::clamp(volcanicStress * 0.72 + surface.geology.faultStrength * 0.18 +
                       std::max(0.0, surface.geology.uplift - 0.62) * 0.22,
                   0.0, 0.94);
    const double resourceFit =
        0.46 + moistureFit * 0.20 + fertilityFit * 0.18 + precipitationFit * 0.16;
    result.suitability =
        std::clamp(affinity * (0.62 + climateFit * 0.38) * resourceFit * (0.58 + slopeFit * 0.42) *
                       hydrologyFit * lithologyRootFit(surface) * (1.0 - tectonicStress),
                   0.0, 1.0);
    result.allowed = result.suitability > 0.035;
    return result;
}

} // namespace

double feature_generation::treeCoverDensity(const worldgen::SurfaceSample& surface) {
    return continuousTreeDensity(surface);
}

feature_generation::TreeHabitatEvaluation
feature_generation::evaluateTreeHabitat(TreeSpecies species, const worldgen::SurfaceSample& surface,
                                        int groundSurfaceY) {
    const double cover = treeCoverDensity(surface);
    return evaluateTreeHabitatWithContext(species, surface, groundSurfaceY, cover,
                                          maximumBiomeSuitability(surface));
}

double feature_generation::farCanopyAggregateAcceptance(double exactCandidateAcceptance) {
    const double bounded = std::clamp(exactCandidateAcceptance, 0.0, 1.0);
    return 1.0 - std::pow(1.0 - bounded, FAR_CANOPY_EXACT_OPPORTUNITIES_PER_CLUSTER_CANDIDATE);
}

bool feature_generation::previewFarEcologyRejectsRoot(
    const worldgen::SurfaceSample& surface) noexcept {
    return surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake ||
           surface.hydrology.wetland || surface.hydrology.waterfall;
}

namespace {

struct SelectedTreeKind {
    TreeKind kind = TreeKind::OAK;
    feature_generation::TreeHabitatEvaluation habitat;
};

std::optional<SelectedTreeKind> selectTreeKind(const worldgen::SurfaceSample& surface,
                                               int groundSurfaceY, double roll,
                                               bool includeFallenLogs = true) {
    constexpr size_t SPECIES_COUNT = static_cast<size_t>(TreeKind::COUNT);
    std::array<double, SPECIES_COUNT> weights{};
    std::array<feature_generation::TreeHabitatEvaluation, SPECIES_COUNT> habitats{};
    const double cover = continuousTreeDensity(surface);
    const double maximumSuitability = maximumBiomeSuitability(surface);
    double total = 0.0;
    for (size_t index = 0; index < SPECIES_COUNT; ++index) {
        const TreeKind kind = static_cast<TreeKind>(index);
        if (!includeFallenLogs && kind == TreeKind::FALLEN_LOG) continue;
        const feature_generation::TreeHabitatEvaluation habitat = evaluateTreeHabitatWithContext(
            kind, surface, groundSurfaceY, cover, maximumSuitability);
        if (!habitat.allowed) continue;
        double formWeight = 1.0;
        if (kind == TreeKind::LARGE_OAK) formWeight = 0.22;
        if (kind == TreeKind::FALLEN_LOG) formWeight = 0.025;
        habitats[index] = habitat;
        weights[index] = habitat.suitability * formWeight;
        total += weights[index];
    }
    if (!(total > 1.0e-9)) return std::nullopt;
    double threshold = std::clamp(roll, 0.0, std::nextafter(1.0, 0.0)) * total;
    for (size_t index = 0; index < SPECIES_COUNT; ++index) {
        if (threshold < weights[index]) {
            return SelectedTreeKind{.kind = static_cast<TreeKind>(index),
                                    .habitat = habitats[index]};
        }
        threshold -= weights[index];
    }
    const size_t fallback = SPECIES_COUNT - 1;
    return SelectedTreeKind{.kind = static_cast<TreeKind>(fallback), .habitat = habitats[fallback]};
}

struct TreeCandidate {
    int64_t cellX = 0;
    int64_t cellZ = 0;
    int64_t x = 0;
    int64_t z = 0;
    TreeKind kind = TreeKind::OAK;
    int sampledSurfaceY = SEA_LEVEL;
    double priority = 0.0;
    int spacing = 8;
};

ColumnPos candidatePosition(const CounterRng& random, int64_t cellX, int64_t cellZ) {
    const int offsetX = random.uniformInt(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 0, 0,
                                          feature_generation::TREE_CELL_EDGE - 1);
    const int offsetZ = random.uniformInt(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 1, 0,
                                          feature_generation::TREE_CELL_EDGE - 1);
    return ColumnPos{cellX * feature_generation::TREE_CELL_EDGE + offsetX,
                     cellZ * feature_generation::TREE_CELL_EDGE + offsetZ};
}

std::optional<TreeCandidate> makeCandidateFromSurface(const CounterRng& random, int64_t cellX,
                                                      int64_t cellZ,
                                                      const worldgen::SurfaceSample& surface,
                                                      BlockType rootMaterial) {
    const ColumnPos position = candidatePosition(random, cellX, cellZ);
    const int64_t x = position.x;
    const int64_t z = position.z;
    const double density = feature_generation::treeCoverDensity(surface);
    const double barrenWeight = worldgen::biomeBlendWeight(surface.biome, Biome::VOLCANIC_BARREN);
    if (density <= 0.0 || barrenWeight >= 0.7 ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL) ||
        !worldgen::surface_material::supportsTreeRooting(rootMaterial)) {
        return std::nullopt;
    }

    const int groundSurfaceY = habitatGroundSurfaceY(surface);
    const int habitatGroundY = habitatGroundSurfaceY(surface);
    const double kindRoll = random.uniform01(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 2);
    const std::optional<SelectedTreeKind> selectedKind =
        selectTreeKind(surface, habitatGroundY, kindRoll);
    if (!selectedKind.has_value()) return std::nullopt;
    const TreeKind kind = selectedKind->kind;
    const feature_generation::TreeHabitatEvaluation& habitat = selectedKind->habitat;
    if (!habitat.allowed) return std::nullopt;
    const double acceptance = std::clamp(density * (0.62 + habitat.suitability * 0.55), 0.0, 1.0);
    if (random.uniform01(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 3) >= acceptance) {
        return std::nullopt;
    }
    return TreeCandidate{
        .cellX = cellX,
        .cellZ = cellZ,
        .x = x,
        .z = z,
        .kind = kind,
        .sampledSurfaceY = groundSurfaceY,
        .priority = random.uniform01(TREE_PRIORITY_STREAM, cellX, 0, cellZ),
        .spacing = habitat.spacing,
    };
}

using TreeCandidateCache = std::unordered_map<ColumnPos, std::optional<TreeCandidate>>;

void populateCandidateCache(const CounterRng& random, const ChunkGenerator& generator,
                            int64_t minimumCellX, int64_t minimumCellZ, int64_t maximumCellX,
                            int64_t maximumCellZ, TreeCandidateCache& cache) {
    if (minimumCellX > maximumCellX || minimumCellZ > maximumCellZ) return;
    const int64_t width = maximumCellX - minimumCellX + 1;
    const int64_t height = maximumCellZ - minimumCellZ + 1;
    std::vector<ColumnPos> rootPositions;
    rootPositions.reserve(static_cast<size_t>(width * height));
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX)
            rootPositions.push_back(candidatePosition(random, cellX, cellZ));
    }
    std::vector<worldgen::SurfaceSample> rootSurfaces(rootPositions.size());
    generator.sampleFarHabitatPoints(rootPositions, rootSurfaces);
    cache.reserve(cache.size() + static_cast<size_t>(width * height));
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const size_t rootIndex =
                static_cast<size_t>((cellZ - minimumCellZ) * width + cellX - minimumCellX);
            const ColumnPos root = rootPositions[rootIndex];
            const worldgen::SurfaceSample& surface = rootSurfaces[rootIndex];
            const BlockType material = generator.farSurfaceMaterialAt(root.x, root.z, surface);
            cache.emplace(ColumnPos{cellX, cellZ},
                          makeCandidateFromSurface(random, cellX, cellZ, surface, material));
        }
    }
}

bool candidateWins(const TreeCandidate& candidate, const TreeCandidateCache& cache) {
    for (int offsetZ = -TREE_COMPETITOR_CELL_RADIUS; offsetZ <= TREE_COMPETITOR_CELL_RADIUS;
         ++offsetZ) {
        for (int offsetX = -TREE_COMPETITOR_CELL_RADIUS; offsetX <= TREE_COMPETITOR_CELL_RADIUS;
             ++offsetX) {
            const ColumnPos cell{candidate.cellX + offsetX, candidate.cellZ + offsetZ};
            const auto found = cache.find(cell);
            if (found == cache.end()) {
                throw std::logic_error("tree competitor cache does not cover the priority radius");
            }
            if (!found->second.has_value()) continue;
            const TreeCandidate& competitor = *found->second;
            if (competitor.cellX == candidate.cellX && competitor.cellZ == candidate.cellZ)
                continue;
            const int64_t dx = competitor.x - candidate.x;
            const int64_t dz = competitor.z - candidate.z;
            const int spacing = std::max(candidate.spacing, competitor.spacing);
            if (dx * dx + dz * dz >= static_cast<int64_t>(spacing * spacing)) continue;
            if (competitor.priority > candidate.priority ||
                (competitor.priority == candidate.priority &&
                 (competitor.cellX < candidate.cellX ||
                  (competitor.cellX == candidate.cellX && competitor.cellZ < candidate.cellZ)))) {
                return false;
            }
        }
    }
    return true;
}

struct TreeWriter {
    Chunk& chunk;
    int64_t baseX;
    int baseY;
    int64_t baseZ;
    bool replaceGeneratedWater = false;

    void log(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return;
        }
        const BlockType current = chunk.getBlock(lx, ly, lz);
        if (current == BlockType::AIR || isLeafBlock(current) || isFlora(current) ||
            (replaceGeneratedWater && current == BlockType::WATER)) {
            chunk.setBlock(lx, ly, lz, block);
        }
    }

    void leaves(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return;
        }
        if (chunk.getBlock(lx, ly, lz) == BlockType::AIR) chunk.setBlock(lx, ly, lz, block);
    }
};

struct TreeBoundsWriter {
    int64_t minimumLeafX = std::numeric_limits<int64_t>::max();
    int64_t maximumLeafX = std::numeric_limits<int64_t>::min();
    int minimumLeafY = std::numeric_limits<int>::max();
    int maximumLeafY = std::numeric_limits<int>::min();
    int64_t minimumLeafZ = std::numeric_limits<int64_t>::max();
    int64_t maximumLeafZ = std::numeric_limits<int64_t>::min();
    int64_t minimumLogX = std::numeric_limits<int64_t>::max();
    int64_t maximumLogX = std::numeric_limits<int64_t>::min();
    int minimumLogY = std::numeric_limits<int>::max();
    int maximumLogY = std::numeric_limits<int>::min();
    int64_t minimumLogZ = std::numeric_limits<int64_t>::max();
    int64_t maximumLogZ = std::numeric_limits<int64_t>::min();
    int topY = std::numeric_limits<int>::min();
    BlockType logBlock = BlockType::AIR;
    BlockType leafBlock = BlockType::AIR;
    bool hasLog = false;
    bool hasFoliage = false;

    void log(int64_t x, int y, int64_t z, BlockType block) {
        if (logBlock == BlockType::AIR) logBlock = block;
        hasLog = true;
        minimumLogX = std::min(minimumLogX, x);
        maximumLogX = std::max(maximumLogX, x);
        minimumLogY = std::min(minimumLogY, y);
        maximumLogY = std::max(maximumLogY, y);
        minimumLogZ = std::min(minimumLogZ, z);
        maximumLogZ = std::max(maximumLogZ, z);
        topY = std::max(topY, y);
    }

    void leaves(int64_t x, int y, int64_t z, BlockType block) {
        if (leafBlock == BlockType::AIR) leafBlock = block;
        hasFoliage = true;
        minimumLeafX = std::min(minimumLeafX, x);
        maximumLeafX = std::max(maximumLeafX, x);
        minimumLeafY = std::min(minimumLeafY, y);
        maximumLeafY = std::max(maximumLeafY, y);
        minimumLeafZ = std::min(minimumLeafZ, z);
        maximumLeafZ = std::max(maximumLeafZ, z);
        topY = std::max(topY, y);
    }
};

int shapeInt(const CounterRng& random, const TreeCandidate& candidate, uint32_t index, int minimum,
             int maximum) {
    return random.uniformInt(TREE_SHAPE_STREAM, candidate.cellX, 0, candidate.cellZ, index, minimum,
                             maximum);
}

double shapeUnit(const CounterRng& random, const TreeCandidate& candidate, uint32_t index) {
    return random.uniform01(TREE_SHAPE_STREAM, candidate.cellX, 0, candidate.cellZ, index);
}

template <typename Writer>
void roundedCanopy(Writer& out, const CounterRng& random, const TreeCandidate& candidate, int64_t x,
                   int top, int64_t z, int radius, BlockType leaves) {
    for (int dy = -2; dy <= 1; ++dy) {
        const int layerRadius = dy >= 1 ? std::max(1, radius - 1) : radius;
        for (int dz = -layerRadius; dz <= layerRadius; ++dz) {
            for (int dx = -layerRadius; dx <= layerRadius; ++dx) {
                const bool corner = std::abs(dx) == layerRadius && std::abs(dz) == layerRadius;
                const uint32_t shapeIndex = 512U + static_cast<uint32_t>(candidate.kind);
                if (corner && random.uniform01(TREE_SHAPE_STREAM, x + dx, top + dy, z + dz,
                                               shapeIndex) < 0.75) {
                    continue;
                }
                out.leaves(x + dx, top + dy, z + dz, leaves);
            }
        }
    }
    out.leaves(x, top + 2, z, leaves);
}

template <typename Writer>
void flatCanopy(Writer& out, int64_t x, int y, int64_t z, int radius, BlockType leaves) {
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz > radius * radius + 1) continue;
            out.leaves(x + dx, y, z + dz, leaves);
            if (std::abs(dx) + std::abs(dz) < radius) out.leaves(x + dx, y + 1, z + dz, leaves);
        }
    }
}

// Voxel limbs must advance through one face at a time. Moving laterally and
// vertically in a single emitted sample leaves logs touching only along an
// edge or corner, which reads as a floating staircase once exact cubes take
// ownership from a far silhouette.
template <typename Writer>
void connectedLogStep(Writer& out, int64_t fromX, int fromY, int64_t fromZ, int64_t toX, int toY,
                      int64_t toZ, BlockType block, bool horizontalBeforeVertical = false) {
    const auto emit = [&](int64_t x, int y, int64_t z) { out.log(x, y, z, block); };
    int64_t x = fromX;
    int y = fromY;
    int64_t z = fromZ;
    emit(x, y, z);
    const auto moveHorizontal = [&] {
        while (x != toX) {
            x += x < toX ? 1 : -1;
            emit(x, y, z);
        }
        while (z != toZ) {
            z += z < toZ ? 1 : -1;
            emit(x, y, z);
        }
    };
    const auto moveVertical = [&] {
        while (y != toY) {
            y += y < toY ? 1 : -1;
            emit(x, y, z);
        }
    };
    if (horizontalBeforeVertical) {
        moveHorizontal();
        moveVertical();
    } else {
        moveVertical();
        moveHorizontal();
    }
}

template <typename Writer>
void connectedLeafStep(Writer& out, int64_t fromX, int fromY, int64_t fromZ, int64_t toX, int toY,
                       int64_t toZ, BlockType block) {
    int64_t x = fromX;
    int y = fromY;
    int64_t z = fromZ;
    out.leaves(x, y, z, block);
    while (x != toX) {
        x += x < toX ? 1 : -1;
        out.leaves(x, y, z, block);
    }
    while (z != toZ) {
        z += z < toZ ? 1 : -1;
        out.leaves(x, y, z, block);
    }
    while (y != toY) {
        y += y < toY ? 1 : -1;
        out.leaves(x, y, z, block);
    }
}

template <typename Writer>
void buildTree(const TreeCandidate& candidate, const CounterRng& random, int baseY, Writer& out) {
    const TreeKind kind = candidate.kind;
    const int64_t x = candidate.x;
    const int64_t z = candidate.z;
    switch (kind) {
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK: {
            const bool large = kind == TreeKind::LARGE_OAK;
            const int height = large ? shapeInt(random, candidate, 0, 8, 11)
                                     : shapeInt(random, candidate, 0, 5, 7);
            const int top = baseY + height;
            roundedCanopy(out, random, candidate, x, top, z, large ? 3 : 2, BlockType::LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::LOG);
            if (large) {
                constexpr std::array<std::array<int, 2>, 4> branches{{
                    {{1, 0}},
                    {{-1, 0}},
                    {{0, 1}},
                    {{0, -1}},
                }};
                for (size_t index = 0; index < branches.size(); ++index) {
                    if (shapeUnit(random, candidate, 16U + static_cast<uint32_t>(index)) < 0.25) {
                        continue;
                    }
                    const int branchY = top - 3 + static_cast<int>(index & 1U);
                    connectedLogStep(out, x, branchY, z, x + branches[index][0], branchY,
                                     z + branches[index][1], BlockType::LOG);
                    connectedLogStep(out, x + branches[index][0], branchY, z + branches[index][1],
                                     x + branches[index][0] * 2, branchY + 1,
                                     z + branches[index][1] * 2, BlockType::LOG);
                }
            }
            break;
        }
        case TreeKind::BIRCH: {
            const int height = shapeInt(random, candidate, 0, 6, 9);
            const int top = baseY + height;
            roundedCanopy(out, random, candidate, x, top, z, 2, BlockType::BIRCH_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::BIRCH_LOG);
            break;
        }
        case TreeKind::SPRUCE: {
            const int height = shapeInt(random, candidate, 0, 8, 12);
            const int top = baseY + height;
            for (int dy = 0; dy <= height - 2; ++dy) {
                int radius = std::min(3, 1 + dy / 2);
                if ((dy & 1) != 0) radius = std::max(1, radius - 1);
                for (int dz = -radius; dz <= radius; ++dz) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                            continue;
                        out.leaves(x + dx, top - dy, z + dz, BlockType::SPRUCE_LEAVES);
                    }
                }
            }
            out.leaves(x, top + 1, z, BlockType::SPRUCE_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::SPRUCE_LOG);
            break;
        }
        case TreeKind::ACACIA: {
            const int height = shapeInt(random, candidate, 0, 6, 9);
            const int bendX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
            const int bendZ = shapeUnit(random, candidate, 2) < 0.5 ? -1 : 1;
            int64_t previousX = x;
            int previousY = baseY;
            int64_t previousZ = z;
            for (int y = 0; y < height; ++y) {
                const int step = y > height / 2 ? (y - height / 2 + 1) / 2 : 0;
                const int64_t trunkX = x + bendX * step;
                const int trunkY = baseY + y;
                const int64_t trunkZ = z + bendZ * step;
                connectedLogStep(out, previousX, previousY, previousZ, trunkX, trunkY, trunkZ,
                                 BlockType::ACACIA_LOG);
                previousX = trunkX;
                previousY = trunkY;
                previousZ = trunkZ;
            }
            const int64_t crownX = x + bendX * ((height - height / 2) / 2);
            const int64_t crownZ = z + bendZ * ((height - height / 2) / 2);
            flatCanopy(out, crownX, baseY + height, crownZ, 3, BlockType::ACACIA_LEAVES);
            connectedLogStep(out, crownX, baseY + height - 1, crownZ, crownX - bendZ,
                             baseY + height - 1, crownZ + bendX, BlockType::ACACIA_LOG);
            flatCanopy(out, crownX - bendZ * 2, baseY + height, crownZ + bendX * 2, 2,
                       BlockType::ACACIA_LEAVES);
            break;
        }
        case TreeKind::JUNGLE: {
            const int height = shapeInt(random, candidate, 0, 12, 18);
            for (int y = 0; y < height; ++y) {
                out.log(x, baseY + y, z, BlockType::JUNGLE_LOG);
                out.log(x + 1, baseY + y, z, BlockType::JUNGLE_LOG);
                out.log(x, baseY + y, z + 1, BlockType::JUNGLE_LOG);
                out.log(x + 1, baseY + y, z + 1, BlockType::JUNGLE_LOG);
            }
            for (int direction = 0; direction < 4; ++direction) {
                const int dx = direction == 0 ? 1 : direction == 1 ? -1 : 0;
                const int dz = direction == 2 ? 1 : direction == 3 ? -1 : 0;
                connectedLogStep(out, x, baseY, z, x + dx, baseY, z + dz, BlockType::JUNGLE_LOG,
                                 true);
                connectedLogStep(out, x + dx, baseY, z + dz, x + dx * 2, baseY - 1, z + dz * 2,
                                 BlockType::JUNGLE_LOG, true);
                const int branchY = baseY + height - 3 - (direction & 1);
                int64_t branchX = x;
                int branchCurrentY = branchY;
                int64_t branchZ = z;
                for (int step = 1; step <= 3; ++step) {
                    const int64_t nextX = x + dx * step;
                    const int nextY = branchY + step / 2;
                    const int64_t nextZ = z + dz * step;
                    connectedLogStep(out, branchX, branchCurrentY, branchZ, nextX, nextY, nextZ,
                                     BlockType::JUNGLE_LOG);
                    branchX = nextX;
                    branchCurrentY = nextY;
                    branchZ = nextZ;
                }
                flatCanopy(out, x + dx * 3, branchY + 2, z + dz * 3, 2, BlockType::JUNGLE_LEAVES);
            }
            flatCanopy(out, x, baseY + height, z, 4, BlockType::JUNGLE_LEAVES);
            break;
        }
        case TreeKind::MANGROVE: {
            const int height = shapeInt(random, candidate, 0, 7, 10);
            for (int y = 0; y < height; ++y)
                out.log(x, baseY + y, z, BlockType::MANGROVE_LOG);
            constexpr std::array<std::array<int, 2>, 8> roots{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& root : roots) {
                connectedLogStep(out, x, baseY, z, x + root[0], baseY, z + root[1],
                                 BlockType::MANGROVE_LOG, true);
                connectedLogStep(out, x + root[0], baseY, z + root[1], x + root[0] * 2, baseY - 1,
                                 z + root[1] * 2, BlockType::MANGROVE_LOG, true);
            }
            roundedCanopy(out, random, candidate, x, baseY + height, z, 3,
                          BlockType::MANGROVE_LEAVES);
            break;
        }
        case TreeKind::PALM: {
            const int height = shapeInt(random, candidate, 0, 8, 12);
            const int leanX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
            int64_t previousX = x;
            int previousY = baseY;
            for (int y = 0; y < height; ++y) {
                const int bend = y > height * 2 / 3 ? 1 : 0;
                const int64_t trunkX = x + leanX * bend;
                const int trunkY = baseY + y;
                connectedLogStep(out, previousX, previousY, z, trunkX, trunkY, z,
                                 BlockType::PALM_LOG);
                previousX = trunkX;
                previousY = trunkY;
            }
            const int64_t topX = x + leanX;
            const int topY = baseY + height;
            out.log(topX, topY, z, BlockType::PALM_LOG);
            out.leaves(topX, topY + 1, z, BlockType::PALM_LEAVES);
            constexpr std::array<std::array<int, 2>, 8> fronds{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& frond : fronds) {
                int64_t previousX = topX;
                int previousY = topY;
                int64_t previousZ = z;
                for (int step = 1; step <= 4; ++step) {
                    const int drop = step >= 3 ? 1 : 0;
                    const int64_t nextX = topX + frond[0] * step;
                    const int nextY = topY - drop;
                    const int64_t nextZ = z + frond[1] * step;
                    connectedLeafStep(out, previousX, previousY, previousZ, nextX, nextY, nextZ,
                                      BlockType::PALM_LEAVES);
                    previousX = nextX;
                    previousY = nextY;
                    previousZ = nextZ;
                }
            }
            break;
        }
        case TreeKind::WILLOW: {
            const int height = shapeInt(random, candidate, 0, 7, 10);
            const int top = baseY + height;
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::WILLOW_LOG);
            flatCanopy(out, x, top, z, 3, BlockType::WILLOW_LEAVES);
            constexpr std::array<std::array<int, 2>, 8> directions{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& direction : directions) {
                connectedLogStep(out, x, top - 2, z, x + direction[0], top - 2, z + direction[1],
                                 BlockType::WILLOW_LOG);
                // The diagonal curtain begins beyond the rounded edge of the
                // crown. Bridge it through faces so a willow never contains
                // a visually floating column of leaves.
                connectedLeafStep(out, x + direction[0] * 2, top, z + direction[1] * 2,
                                  x + direction[0] * 3, top - 3, z + direction[1] * 3,
                                  BlockType::WILLOW_LEAVES);
            }
            break;
        }
        case TreeKind::ALPINE_SCRUB: {
            const int radius = shapeInt(random, candidate, 0, 1, 2);
            for (int dz = -radius; dz <= radius; ++dz) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dz * dz > radius * radius + 1) continue;
                    const uint32_t index = 160U + static_cast<uint32_t>((dx + 2) * 5 + dz + 2);
                    if (shapeUnit(random, candidate, index) < 0.28) continue;
                    out.leaves(x + dx, baseY, z + dz, BlockType::SHRUB);
                    if (dx == 0 && dz == 0 && shapeUnit(random, candidate, index + 32U) > 0.42) {
                        out.leaves(x, baseY + 1, z, BlockType::SHRUB);
                    }
                }
            }
            break;
        }
        case TreeKind::FALLEN_LOG: {
            const bool alongX = shapeUnit(random, candidate, 0) < 0.5;
            const int length = shapeInt(random, candidate, 1, 4, 7);
            const BlockType log =
                shapeUnit(random, candidate, 2) < 0.5 ? BlockType::LOG : BlockType::WILLOW_LOG;
            for (int step = 0; step < length; ++step) {
                out.log(x + (alongX ? step : 0), baseY, z + (alongX ? 0 : step), log);
            }
            break;
        }
        case TreeKind::COUNT:
            break;
    }
}

std::optional<int> acceptedTreeBaseY(const TreeCandidate& candidate,
                                     const ChunkGenerator& generator,
                                     const StructurePlacer& structures, GenScratch& scratch) {
    const int surfaceY = generator.surfaceYAt(candidate.x, candidate.z, scratch);
    const int64_t anchorChunkX = Chunk::worldToChunk(candidate.x);
    const int64_t anchorChunkZ = Chunk::worldToChunk(candidate.z);
    if (structures.insideStructure(candidate.x, candidate.z, anchorChunkX, anchorChunkZ, 1)) {
        return std::nullopt;
    }
    return surfaceY + 1;
}

struct DescribedFarCanopy {
    FarCanopy canopy;
    int64_t minimumX = 0;
    int64_t maximumX = 0;
    int64_t minimumZ = 0;
    int64_t maximumZ = 0;
};

std::optional<DescribedFarCanopy> describeFarCanopy(const TreeCandidate& candidate,
                                                    const CounterRng& random, int baseY) {
    TreeBoundsWriter bounds;
    buildTree(candidate, random, baseY, bounds);
    if (!bounds.hasFoliage && !bounds.hasLog) return std::nullopt;

    const int64_t canopyCenterX =
        bounds.hasFoliage ? bounds.minimumLeafX + (bounds.maximumLeafX - bounds.minimumLeafX) / 2
                          : candidate.x;
    const int64_t canopyCenterZ =
        bounds.hasFoliage ? bounds.minimumLeafZ + (bounds.maximumLeafZ - bounds.minimumLeafZ) / 2
                          : candidate.z;
    const int64_t radius =
        bounds.hasFoliage
            ? std::max({canopyCenterX - bounds.minimumLeafX, bounds.maximumLeafX - canopyCenterX,
                        canopyCenterZ - bounds.minimumLeafZ, bounds.maximumLeafZ - canopyCenterZ})
            : 0;

    int formX = 0;
    int formZ = 0;
    int formExtent = 0;
    if (candidate.kind == TreeKind::ACACIA) {
        formX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
        formZ = shapeUnit(random, candidate, 2) < 0.5 ? -1 : 1;
    } else if (candidate.kind == TreeKind::PALM) {
        formX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
    } else if (candidate.kind == TreeKind::FALLEN_LOG) {
        const bool alongX = shapeUnit(random, candidate, 0) < 0.5;
        formX = alongX ? 1 : 0;
        formZ = alongX ? 0 : 1;
        formExtent = shapeInt(random, candidate, 1, 4, 7);
    }

    DescribedFarCanopy result;
    result.canopy = {
        .x = candidate.x,
        .z = candidate.z,
        .baseY = baseY,
        .topY = bounds.topY,
        .canopyMinimumY = bounds.hasFoliage ? bounds.minimumLeafY : bounds.minimumLogY,
        .canopyMaximumY = bounds.hasFoliage ? bounds.maximumLeafY : bounds.maximumLogY,
        .canopyOffsetX = static_cast<int8_t>(canopyCenterX - candidate.x),
        .canopyOffsetZ = static_cast<int8_t>(canopyCenterZ - candidate.z),
        .canopyRadius = static_cast<uint8_t>(radius),
        .logBlock = bounds.logBlock,
        .leafBlock = bounds.leafBlock,
        .anchorId = random.u64(TREE_PRIORITY_STREAM, candidate.cellX, 0, candidate.cellZ),
        .species = candidate.kind,
        .formX = static_cast<int8_t>(formX),
        .formZ = static_cast<int8_t>(formZ),
        .formExtent = static_cast<uint8_t>(formExtent),
    };
    result.minimumX = bounds.hasFoliage ? bounds.minimumLeafX : bounds.minimumLogX;
    result.maximumX = bounds.hasFoliage ? bounds.maximumLeafX : bounds.maximumLogX;
    result.minimumZ = bounds.hasFoliage ? bounds.minimumLeafZ : bounds.minimumLogZ;
    result.maximumZ = bounds.hasFoliage ? bounds.maximumLeafZ : bounds.maximumLogZ;
    return result;
}

bool intersectsCubeVertically(TreeKind kind, int baseY, int cubeBaseY) {
    int minimum = baseY + feature_generation::TREE_MINIMUM_VERTICAL_OFFSET;
    if (kind != TreeKind::JUNGLE && kind != TreeKind::MANGROVE) minimum = baseY;
    int maximum = baseY + feature_generation::TREE_MAXIMUM_VERTICAL_OFFSET;
    return maximum >= cubeBaseY && minimum < cubeBaseY + CHUNK_EDGE;
}

struct FloraWriter {
    Chunk& chunk;
    int64_t baseX;
    int baseY;
    int64_t baseZ;

    bool setIfAir(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return false;
        }
        if (chunk.getBlock(lx, ly, lz) != BlockType::AIR) return false;
        chunk.setBlock(lx, ly, lz, block);
        return true;
    }
};

struct GroundFloraDescription {
    BlockType block = BlockType::AIR;
    uint8_t height = 1;
};

template <typename SupportFunction>
std::optional<GroundFloraDescription>
describeGroundFlora(const CounterRng& random, const worldgen::SurfaceSample& surface, int64_t x,
                    int terrainY, int64_t z, SupportFunction&& supportsFlora) {
    const double roll = random.uniform01(FLORA_STREAM, x, terrainY, z, 0);
    const double kindRoll = random.uniform01(FLORA_STREAM, x, terrainY, z, 1);
    const Biome biome = ditheredBiome(surface, random, FLORA_BIOME_STREAM, x, z);
    const Biome substrateBiome = worldgen::surface_material::materialBiome(surface, random, x, z);
    const double barrenWeight = worldgen::biomeBlendWeight(surface.biome, Biome::VOLCANIC_BARREN);
    const double volcanicStress = std::max(
        barrenWeight, std::clamp((surface.geology.volcanicActivity - 0.32) / 0.42, 0.0, 1.0));
    const double growthFit = 1.0 - volcanicStress;
    if (growthFit <= 0.01 || substrateBiome == Biome::VOLCANIC_BARREN ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL)) {
        return std::nullopt;
    }

    const double riparianInfluence = std::max(
        {worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::RIVERBANK),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::LAKESHORE),
         worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::FLOODPLAIN),
         worldgen::biomeBlendWeight(surface.biome, Biome::MANGROVE),
         worldgen::biomeBlendWeight(surface.biome, Biome::FLOODED_GRASSLAND)});
    if (surface.soil.moisture > 0.50 && roll < 0.22 * riparianInfluence * growthFit) {
        if (!supportsFlora()) return std::nullopt;
        const BlockType plant = kindRoll < 0.58 ? BlockType::CATTAIL : BlockType::REED;
        const int height = plant == BlockType::REED ? 2 + static_cast<int>(kindRoll * 3.0) : 2;
        return GroundFloraDescription{plant, static_cast<uint8_t>(height)};
    }

    if (biome == Biome::ALPINE || biome == Biome::MONTANE_GRASSLAND) {
        const double slopeFit = std::clamp(1.0 - surface.slope / 1.65, 0.0, 1.0);
        const double scrubSuitability =
            std::clamp(surface.soil.moisture * 0.30 + surface.soil.fertility * 0.34 +
                           slopeFit * 0.24 + availableSurfaceLight(surface) * 0.12,
                       0.0, 1.0);
        if (roll < scrubSuitability * 0.26 * growthFit && supportsFlora()) {
            return GroundFloraDescription{BlockType::SHRUB, 1};
        }
        if (roll < scrubSuitability * 0.34 * growthFit && kindRoll > 0.55 && supportsFlora()) {
            return GroundFloraDescription{BlockType::FERN, 1};
        }
        return std::nullopt;
    }

    if (biome == Biome::DESERT || biome == Biome::COLD_DESERT || biome == Biome::BADLANDS) {
        if (roll < 0.018 * growthFit && supportsFlora()) {
            return GroundFloraDescription{
                BlockType::CACTUS, static_cast<uint8_t>(1 + static_cast<int>(kindRoll * 3.0))};
        }
        if (roll < 0.055 * growthFit && supportsFlora()) {
            return GroundFloraDescription{
                kindRoll < 0.45 ? BlockType::SUCCULENT : BlockType::DEAD_BUSH, 1};
        }
        return std::nullopt;
    }

    const double vegetation =
        std::clamp(surface.soil.moisture * 0.45 + surface.soil.fertility * 0.45 +
                       surface.climate.relativeHumidity * 0.10 - surface.slope * 0.12,
                   0.0, 0.72) *
        growthFit;
    if (roll >= vegetation || !supportsFlora()) return std::nullopt;
    BlockType plant = BlockType::TALL_GRASS;
    if (biome == Biome::TROPICAL_RAINFOREST || biome == Biome::TEMPERATE_RAINFOREST ||
        biome == Biome::TAIGA || biome == Biome::TEMPERATE_CONIFER_FOREST ||
        biome == Biome::TROPICAL_CONIFER_FOREST) {
        plant = kindRoll < 0.62 ? BlockType::FERN : BlockType::SHRUB;
    } else if (biome == Biome::SHRUBLAND || biome == Biome::STEPPE || biome == Biome::SAVANNA ||
               biome == Biome::MEDITERRANEAN_WOODLAND || biome == Biome::TROPICAL_DRY_FOREST) {
        plant = kindRoll < 0.52 ? BlockType::SHRUB : BlockType::TALL_GRASS;
    } else if (kindRoll > 0.87) {
        plant = BlockType::FLOWER_BLUE;
    } else if (kindRoll > 0.77) {
        plant = BlockType::FLOWER_RED;
    } else if (kindRoll > 0.67) {
        plant = BlockType::FLOWER_YELLOW;
    } else if (surface.climate.relativeHumidity > 0.78 && kindRoll < 0.08) {
        plant = BlockType::MUSHROOM_BROWN;
    }
    return GroundFloraDescription{plant, 1};
}

} // namespace

FeaturePlacer::FeaturePlacer(uint32_t worldSeed) : random_(worldSeed) {}

void FeaturePlacer::placeTrees(Chunk& chunk, const ChunkGenerator& generator,
                               const StructurePlacer& structures, GenScratch& scratch) const {
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    TreeCandidateCache cache;

    const int64_t minimumCellX =
        world_coord::floorDiv(baseX - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellX = world_coord::floorDiv(
        baseX + CHUNK_EDGE - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
        static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t minimumCellZ =
        world_coord::floorDiv(baseZ - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellZ = world_coord::floorDiv(
        baseZ + CHUNK_EDGE - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
        static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    populateCandidateCache(random_, generator, minimumCellX - TREE_COMPETITOR_CELL_RADIUS,
                           minimumCellZ - TREE_COMPETITOR_CELL_RADIUS,
                           maximumCellX + TREE_COMPETITOR_CELL_RADIUS,
                           maximumCellZ + TREE_COMPETITOR_CELL_RADIUS, cache);

    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const ColumnPos cell{cellX, cellZ};
            const auto found = cache.find(cell);
            if (found == cache.end()) {
                throw std::logic_error("tree candidate cache does not cover the emission range");
            }
            if (!found->second.has_value()) continue;
            const TreeCandidate candidate = *found->second;
            if (!candidateWins(candidate, cache)) continue;

            const std::optional<int> treeBaseY =
                acceptedTreeBaseY(candidate, generator, structures, scratch);
            if (!treeBaseY.has_value()) continue;
            if (!intersectsCubeVertically(candidate.kind, *treeBaseY, baseY)) continue;
            TreeWriter writer{chunk, baseX, baseY, baseZ,
                              traitsFor(candidate.kind).toleratesFlooding};
            buildTree(candidate, random_, *treeBaseY, writer);
        }
    }
}

std::vector<FarCanopy> FeaturePlacer::collectFarCanopyAnchors(int64_t minimumX, int64_t minimumZ,
                                                              int64_t maximumX, int64_t maximumZ,
                                                              const ChunkGenerator& generator,
                                                              const StructurePlacer& structures,
                                                              GenScratch&) const {
    std::vector<FarCanopy> result;
    if (minimumX >= maximumX || minimumZ >= maximumZ) return result;

    constexpr int64_t REACH = feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH;
    const int64_t expandedMinimumX = minimumX < std::numeric_limits<int64_t>::min() + REACH
                                         ? std::numeric_limits<int64_t>::min()
                                         : minimumX - REACH;
    const int64_t expandedMinimumZ = minimumZ < std::numeric_limits<int64_t>::min() + REACH
                                         ? std::numeric_limits<int64_t>::min()
                                         : minimumZ - REACH;
    const int64_t lastX = maximumX - 1;
    const int64_t lastZ = maximumZ - 1;
    const int64_t expandedMaximumX = lastX > std::numeric_limits<int64_t>::max() - REACH
                                         ? std::numeric_limits<int64_t>::max()
                                         : lastX + REACH;
    const int64_t expandedMaximumZ = lastZ > std::numeric_limits<int64_t>::max() - REACH
                                         ? std::numeric_limits<int64_t>::max()
                                         : lastZ + REACH;

    const int64_t minimumCellX = world_coord::floorDiv(
        expandedMinimumX, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellX = world_coord::floorDiv(
        expandedMaximumX, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t minimumCellZ = world_coord::floorDiv(
        expandedMinimumZ, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellZ = world_coord::floorDiv(
        expandedMaximumZ, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));

    TreeCandidateCache cache;
    populateCandidateCache(random_, generator, minimumCellX - TREE_COMPETITOR_CELL_RADIUS,
                           minimumCellZ - TREE_COMPETITOR_CELL_RADIUS,
                           maximumCellX + TREE_COMPETITOR_CELL_RADIUS,
                           maximumCellZ + TREE_COMPETITOR_CELL_RADIUS, cache);
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const ColumnPos cell{cellX, cellZ};
            const auto found = cache.find(cell);
            if (found == cache.end()) {
                throw std::logic_error("tree candidate cache does not cover the far canopy range");
            }
            if (!found->second.has_value()) continue;
            const TreeCandidate candidate = *found->second;
            if (!candidateWins(candidate, cache)) continue;

            const int64_t anchorChunkX = Chunk::worldToChunk(candidate.x);
            const int64_t anchorChunkZ = Chunk::worldToChunk(candidate.z);
            const bool insideStructure =
                structures.insideStructure(candidate.x, candidate.z, anchorChunkX, anchorChunkZ, 1);
            if (insideStructure) continue;
            const int treeBaseY = candidate.sampledSurfaceY + 1;
            const std::optional<DescribedFarCanopy> described =
                describeFarCanopy(candidate, random_, treeBaseY);
            if (!described.has_value()) continue;
            if (described->maximumX < minimumX || described->minimumX >= maximumX ||
                described->maximumZ < minimumZ || described->minimumZ >= maximumZ) {
                continue;
            }
            result.push_back(described->canopy);
        }
    }
    return result;
}

std::vector<FarCanopy> FeaturePlacer::collectFarCanopies(int64_t minimumX, int64_t minimumZ,
                                                         int64_t maximumX, int64_t maximumZ,
                                                         const ChunkGenerator& generator,
                                                         const StructurePlacer& structures,
                                                         GenScratch& scratch) const {
    std::vector<FarCanopy> result = collectFarCanopyAnchors(minimumX, minimumZ, maximumX, maximumZ,
                                                            generator, structures, scratch);
    for (FarCanopy& canopy : result) {
        const int exactBaseY = generator.surfaceYAt(canopy.x, canopy.z, scratch) + 1;
        const int verticalOffset = exactBaseY - canopy.baseY;
        canopy.baseY = exactBaseY;
        canopy.topY += verticalOffset;
        canopy.canopyMinimumY += verticalOffset;
        canopy.canopyMaximumY += verticalOffset;
    }
    return result;
}

std::vector<FarCanopy>
FeaturePlacer::collectFarCanopyClusters(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                        int64_t maximumZ, int lodStep,
                                        const ChunkGenerator& generator) const {
    std::vector<FarCanopy> result;
    if (minimumX >= maximumX || minimumZ >= maximumZ) return result;
    constexpr int64_t CELL_EDGE = FAR_CANOPY_CLUSTER_CELL_EDGE;
    constexpr int CANDIDATE_COUNT = FAR_CANOPY_CLUSTER_CANDIDATE_COUNT;
    const auto crownLimitForStep = [](int step) -> size_t {
        if (step <= 2) return 6;
        if (step <= 4) return 5;
        if (step <= 8) return 4;
        if (step <= 16) return 3;
        return 2;
    };
    const int64_t cellEdge = CELL_EDGE;
    const int64_t minimumCellX = world_coord::floorDiv(minimumX, cellEdge);
    const int64_t maximumCellX = world_coord::floorDiv(maximumX - 1, cellEdge);
    const int64_t minimumCellZ = world_coord::floorDiv(minimumZ, cellEdge);
    const int64_t maximumCellZ = world_coord::floorDiv(maximumZ - 1, cellEdge);
    result.reserve(static_cast<size_t>((maximumCellX - minimumCellX + 1) *
                                       (maximumCellZ - minimumCellZ + 1) * CANDIDATE_COUNT));

    struct RankedCanopy {
        FarCanopy canopy;
        uint64_t retentionRank = 0;
        int slot = 0;
    };

    const int64_t cellCountX = maximumCellX - minimumCellX + 1;
    const int64_t cellCountZ = maximumCellZ - minimumCellZ + 1;
    const size_t candidateCount = static_cast<size_t>(cellCountX * cellCountZ * CANDIDATE_COUNT);
    std::vector<ColumnPos> candidatePositions(candidateCount);
    const auto candidateIndex = [&](int64_t cellX, int64_t cellZ, int slot) {
        return (static_cast<size_t>(cellZ - minimumCellZ) * static_cast<size_t>(cellCountX) +
                static_cast<size_t>(cellX - minimumCellX)) *
                   CANDIDATE_COUNT +
               static_cast<size_t>(slot);
    };
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            for (int slot = 0; slot < CANDIDATE_COUNT; ++slot) {
                const int offsetX =
                    12 + (slot % 3) * 20 +
                    random_.uniformInt(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 0, -4, 4);
                const int offsetZ =
                    18 + (slot / 3) * 28 +
                    random_.uniformInt(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 1, -5, 5);
                candidatePositions[candidateIndex(cellX, cellZ, slot)] = {
                    cellX * cellEdge + offsetX, cellZ * cellEdge + offsetZ};
            }
        }
    }
    std::vector<worldgen::SurfaceSample> candidateSurfaces(candidatePositions.size());
    generator.sampleFarEcologyPoints(candidatePositions, lodStep, candidateSurfaces);

    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            std::array<RankedCanopy, CANDIDATE_COUNT> cellCanopies{};
            size_t cellCanopyCount = 0;
            for (int slot = 0; slot < CANDIDATE_COUNT; ++slot) {
                const size_t index = candidateIndex(cellX, cellZ, slot);
                const int64_t x = candidatePositions[index].x;
                const int64_t z = candidatePositions[index].z;

                // Every coarse tier evaluates the same candidates against
                // block-resolution habitat authority. More distant tiers
                // retain strict subsets without moving or resizing surviving
                // crowns, and filtered terrain cannot introduce a tree that
                // exact ecology rejects at a shoreline or material contact.
                const worldgen::SurfaceSample& surface = candidateSurfaces[index];
                if (generator.usesPreviewAuthority() &&
                    feature_generation::previewFarEcologyRejectsRoot(surface)) {
                    continue;
                }
                const Biome substrateBiome =
                    worldgen::surface_material::materialBiome(surface, random_, x, z);
                const double density = feature_generation::treeCoverDensity(surface);
                const int groundSurfaceY = habitatGroundSurfaceY(surface);
                const int habitatGroundY = habitatGroundSurfaceY(surface);
                const std::optional<SelectedTreeKind> selectedKind = selectTreeKind(
                    surface, habitatGroundY,
                    random_.uniform01(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 6), false);
                if (density <= 0.0 || !selectedKind.has_value() ||
                    substrateBiome == Biome::VOLCANIC_BARREN ||
                    !worldgen::surface_material::supportsTreeRooting(
                        generator.farSurfaceMaterialAt(x, z, surface))) {
                    continue;
                }
                const TreeKind kind = selectedKind->kind;
                const feature_generation::TreeHabitatEvaluation& habitat = selectedKind->habitat;
                if (!habitat.allowed || habitat.submerged) continue;
                const double exactCandidateAcceptance =
                    std::clamp(density * (0.70 + habitat.suitability * 0.52), 0.0, 1.0);
                const double acceptance =
                    feature_generation::farCanopyAggregateAcceptance(exactCandidateAcceptance);
                if (random_.uniform01(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 2) >=
                    acceptance) {
                    continue;
                }

                const auto [logBlock, leafBlock] = canopyBlocks(kind);
                const int heightVariation =
                    random_.uniformInt(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 3, 0, 1);
                const CoarseCanopyDimensions dimensions =
                    coarseCanopyDimensions(kind, leafBlock, heightVariation);
                const int baseY = groundSurfaceY + 1;
                const int formX =
                    kind == TreeKind::ACACIA || kind == TreeKind::PALM
                        ? (random_.uniform01(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 7) < 0.5
                               ? -1
                               : 1)
                        : 0;
                const int formZ =
                    kind == TreeKind::ACACIA
                        ? (random_.uniform01(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 8) < 0.5
                               ? -1
                               : 1)
                        : 0;
                cellCanopies[cellCanopyCount++] = {
                    .canopy =
                        {
                            .x = x,
                            .z = z,
                            .baseY = baseY,
                            .topY = baseY + dimensions.topOffset,
                            .canopyMinimumY = baseY + dimensions.crownBottomOffset,
                            .canopyMaximumY = baseY + dimensions.topOffset,
                            .canopyOffsetX = 0,
                            .canopyOffsetZ = 0,
                            .canopyRadius = dimensions.radius,
                            .logBlock = logBlock,
                            .leafBlock = leafBlock,
                            .anchorId =
                                random_.u64(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 4),
                            .aggregate = true,
                            .species = kind,
                            .formX = static_cast<int8_t>(formX),
                            .formZ = static_cast<int8_t>(formZ),
                        },
                    .retentionRank = random_.u64(FAR_CANOPY_CLUSTER_STREAM, cellX, slot, cellZ, 5),
                    .slot = slot,
                };
            }

            const size_t crownLimit = crownLimitForStep(lodStep);
            if (cellCanopyCount > crownLimit) {
                std::sort(cellCanopies.begin(), cellCanopies.begin() + cellCanopyCount,
                          [](const RankedCanopy& first, const RankedCanopy& second) {
                              return first.retentionRank < second.retentionRank;
                          });
                cellCanopyCount = crownLimit;
                std::sort(cellCanopies.begin(), cellCanopies.begin() + cellCanopyCount,
                          [](const RankedCanopy& first, const RankedCanopy& second) {
                              return first.slot < second.slot;
                          });
            }
            for (size_t index = 0; index < cellCanopyCount; ++index) {
                const RankedCanopy& ranked = cellCanopies[index];
                const FarCanopy& canopy = ranked.canopy;
                if (canopy.x < minimumX || canopy.x >= maximumX || canopy.z < minimumZ ||
                    canopy.z >= maximumZ) {
                    continue;
                }
                result.push_back(canopy);
            }
        }
    }
    return result;
}

std::vector<FarFlora> FeaturePlacer::collectFarFlora(int64_t minimumX, int64_t minimumZ,
                                                     int64_t maximumX, int64_t maximumZ,
                                                     int lodStep,
                                                     const ChunkGenerator& generator) const {
    std::vector<FarFlora> result;
    if (minimumX >= maximumX || minimumZ >= maximumZ) return result;

    const int64_t minimumCellX = world_coord::floorDiv(minimumX, FAR_FLORA_CELL_EDGE);
    const int64_t maximumCellX = world_coord::floorDiv(maximumX - 1, FAR_FLORA_CELL_EDGE);
    const int64_t minimumCellZ = world_coord::floorDiv(minimumZ, FAR_FLORA_CELL_EDGE);
    const int64_t maximumCellZ = world_coord::floorDiv(maximumZ - 1, FAR_FLORA_CELL_EDGE);
    const uint8_t retentionSlots = lodStep <= 2    ? 8
                                   : lodStep <= 4  ? 6
                                   : lodStep <= 8  ? 4
                                   : lodStep <= 16 ? 2
                                                   : 1;

    struct Candidate {
        ColumnPos position;
        int64_t cellX = 0;
        int64_t cellZ = 0;
        int slot = 0;
    };
    std::vector<Candidate> candidates;
    const auto appendCoordinate = [](int64_t cell, int offset) -> std::optional<int64_t> {
        const __int128 coordinate = static_cast<__int128>(cell) * FAR_FLORA_CELL_EDGE + offset;
        if (coordinate < std::numeric_limits<int64_t>::min() ||
            coordinate > std::numeric_limits<int64_t>::max()) {
            return std::nullopt;
        }
        return static_cast<int64_t>(coordinate);
    };
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            for (int slot = 0; slot < FAR_FLORA_CANDIDATE_COUNT; ++slot) {
                const uint64_t retentionRank =
                    random_.u64(FAR_FLORA_RETENTION_STREAM, cellX, slot, cellZ, 0);
                if ((retentionRank & 7U) >= retentionSlots) continue;
                const int quadrantX = (slot & 1) * 4;
                const int quadrantZ = (slot >> 1) * 4;
                const int offsetX = quadrantX + random_.uniformInt(FAR_FLORA_CANDIDATE_STREAM,
                                                                   cellX, slot, cellZ, 0, 0, 3);
                const int offsetZ = quadrantZ + random_.uniformInt(FAR_FLORA_CANDIDATE_STREAM,
                                                                   cellX, slot, cellZ, 1, 0, 3);
                const std::optional<int64_t> x = appendCoordinate(cellX, offsetX);
                const std::optional<int64_t> z = appendCoordinate(cellZ, offsetZ);
                if (!x.has_value() || !z.has_value() || *x < minimumX || *x >= maximumX ||
                    *z < minimumZ || *z >= maximumZ) {
                    continue;
                }
                candidates.push_back(
                    {.position = {*x, *z}, .cellX = cellX, .cellZ = cellZ, .slot = slot});
            }
        }
    }
    if (candidates.empty()) return result;

    std::vector<ColumnPos> positions;
    positions.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
        positions.push_back(candidate.position);
    std::vector<worldgen::SurfaceSample> surfaces(positions.size());
    generator.sampleFarEcologyPoints(positions, lodStep, surfaces);
    result.reserve(candidates.size() / 2);
    for (size_t index = 0; index < candidates.size(); ++index) {
        const Candidate& candidate = candidates[index];
        const worldgen::SurfaceSample& surface = surfaces[index];
        const int terrainY = habitatGroundSurfaceY(surface);
        // Exact flora emission cannot replace a generated water block. Keep
        // that same rule in the aggregate authority so submerged grass never
        // protrudes through a lake while the exact cube is loading.
        if ((generator.usesPreviewAuthority() &&
             feature_generation::previewFarEcologyRejectsRoot(surface)) ||
            generatedWaterDepth(surface, terrainY) > 0 || surface.hydrology.waterfall) {
            continue;
        }
        std::optional<bool> supportsFlora;
        const auto description = describeGroundFlora(
            random_, surface, candidate.position.x, terrainY, candidate.position.z, [&] {
                if (!supportsFlora.has_value()) {
                    supportsFlora = worldgen::surface_material::supportsSurfaceFlora(
                        generator.farSurfaceMaterialAt(candidate.position.x, candidate.position.z,
                                                       surface));
                }
                return *supportsFlora;
            });
        if (!description.has_value() || !rendersAsCross(description->block)) continue;
        result.push_back({.x = candidate.position.x,
                          .z = candidate.position.z,
                          .baseY = terrainY + 1,
                          .block = description->block,
                          .height = description->height,
                          .anchorId = random_.u64(FAR_FLORA_CANDIDATE_STREAM, candidate.cellX,
                                                  candidate.slot, candidate.cellZ, 2)});
    }
    return result;
}

void FeaturePlacer::placeFlora(Chunk& chunk, const ChunkGenerator& generator,
                               GenScratch& scratch) const {
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    FloraWriter writer{chunk, baseX, baseY, baseZ};

    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const int64_t x = baseX + localX;
            const int64_t z = baseZ + localZ;
            const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);
            const int terrainY = generator.surfaceYAt(x, z, scratch);
            const double roll = random_.uniform01(FLORA_STREAM, x, terrainY, z, 0);
            const Biome biome = ditheredBiome(surface, random_, FLORA_BIOME_STREAM, x, z);
            const Biome substrateBiome =
                worldgen::surface_material::materialBiome(surface, random_, x, z);
            const double barrenWeight =
                worldgen::biomeBlendWeight(surface.biome, Biome::VOLCANIC_BARREN);
            const double volcanicStress =
                std::max(barrenWeight,
                         std::clamp((surface.geology.volcanicActivity - 0.32) / 0.42, 0.0, 1.0));
            const double growthFit = 1.0 - volcanicStress;
            if (growthFit <= 0.01 || substrateBiome == Biome::VOLCANIC_BARREN ||
                worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL)) {
                continue;
            }
            std::optional<bool> floraSupport;
            auto supportsFlora = [&] {
                if (!floraSupport.has_value()) {
                    floraSupport = worldgen::surface_material::supportsSurfaceFlora(
                        generator.surfaceMaterialAt(x, z));
                }
                return *floraSupport;
            };

            if ((surface.hydrology.lake || biome == Biome::SWAMP || biome == Biome::MANGROVE) &&
                !surface.hydrology.waterfall) {
                const int waterTop = static_cast<int>(std::ceil(surface.waterSurface)) - 1;
                if (roll < 0.055 * growthFit) {
                    writer.setIfAir(x, waterTop + 1, z, BlockType::LILY_PAD);
                }
            }
            const std::optional<GroundFloraDescription> description =
                describeGroundFlora(random_, surface, x, terrainY, z, supportsFlora);
            if (!description.has_value()) continue;
            for (int offset = 1; offset <= description->height; ++offset) {
                if (!writer.setIfAir(x, terrainY + offset, z, description->block)) break;
            }
        }
    }
}

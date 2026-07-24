#pragma once

#include "world/block_properties.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace worldgen::surface_material {

inline constexpr uint64_t DITHER_STREAM = 0x535552464143454DULL;

struct VolcanicSignals {
    double basaltField = 0.0;
    double craterFactor = 0.0;
    bool conduitExposure = false;
};

struct SurfaceMaterialPaletteEntry {
    BlockType material = BlockType::AIR;
    uint8_t weight = 0;
};

struct SurfaceMaterialPalette {
    std::array<SurfaceMaterialPaletteEntry, 4> entries{};
    uint8_t count = 0;
};

struct WeightedBiome {
    Biome biome = Biome::PLAINS;
    double weight = 0.0;
};

inline std::array<WeightedBiome, 4> weightedMaterialBiomes(const SurfaceSample& sample) {
    std::array<WeightedBiome, 4> result{};
    for (size_t index = 0; index < sample.suitability.scores.size(); ++index) {
        // Squaring preserves a clear regional identity while allowing the
        // third and fourth climate fits to appear naturally near broad
        // transitions. It also avoids discontinuities when second place and
        // third place exchange rank.
        const double score = static_cast<double>(sample.suitability.scores[index]);
        const double weight = score * score;
        if (weight <= result.back().weight) continue;
        size_t destination = result.size() - 1;
        while (destination > 0 && weight > result[destination - 1].weight) {
            result[destination] = result[destination - 1];
            --destination;
        }
        result[destination] = {static_cast<Biome>(index), weight};
    }
    double total = 0.0;
    for (const WeightedBiome entry : result)
        total += entry.weight;
    if (total <= 1.0e-12) {
        const double secondary = std::clamp(sample.biome.transition, 0.0, 0.5);
        result = {{{sample.biome.primary, 1.0 - secondary},
                   {sample.biome.secondary, secondary},
                   {sample.biome.primary, 0.0},
                   {sample.biome.primary, 0.0}}};
        return result;
    }
    for (WeightedBiome& entry : result)
        entry.weight /= total;
    return result;
}

inline Biome materialBiome(const SurfaceSample& sample, const CounterRng& random, int64_t x,
                           int64_t z) {
    const std::array<WeightedBiome, 4> biomes = weightedMaterialBiomes(sample);
    const double rank = multiscaleDitherThreshold(random, DITHER_STREAM, x, z);
    double cumulative = 0.0;
    for (const WeightedBiome entry : biomes) {
        cumulative += entry.weight;
        if (rank < cumulative) return entry.biome;
    }
    return biomes.front().biome;
}

inline bool frozen(const SurfaceSample& sample, Biome biome) {
    const double frozenBiomeWeight = biomeBlendWeight(sample.biome, Biome::ICE_SPIKES) +
                                     biomeBlendWeight(sample.biome, Biome::GLACIER) +
                                     biomeBlendWeight(sample.biome, Biome::FROZEN_OCEAN);
    return sample.climate.temperatureC <= -1.0 &&
           (sample.climate.annualPrecipitationMm >= 120.0 || frozenBiomeWeight >= 0.5 ||
            biome == Biome::ICE_SPIKES || biome == Biome::GLACIER || biome == Biome::FROZEN_OCEAN);
}

inline bool submerged(const SurfaceSample& sample) {
    if (!sample.hydrology.ocean && !sample.hydrology.river && !sample.hydrology.lake &&
        !sample.hydrology.wetland) {
        return false;
    }
    const int terrainY = static_cast<int>(std::floor(geometryTerrainHeight(sample)));
    const int waterTopY = static_cast<int>(std::ceil(sample.waterSurface)) - 1;
    return terrainY < waterTopY;
}

inline double weathering(const SurfaceSample& sample) {
    const double precipitation =
        std::clamp(sample.climate.annualPrecipitationMm / 2400.0, 0.0, 1.0);
    return std::clamp(sample.soil.moisture * 0.34 + sample.climate.relativeHumidity * 0.18 +
                          precipitation * 0.18 + sample.geology.crustAge * 0.16 -
                          std::clamp(sample.climate.aridity - 0.8, 0.0, 1.5) * 0.12,
                      0.0, 1.0);
}

inline BlockType outcrop(const GeologySample& geology, RockType rock) {
    switch (rock) {
        case RockType::BASALT:
            return BlockType::BASALT;
        case RockType::VOLCANIC:
            if (geology.crust == CrustType::CONTINENTAL &&
                (geology.boundary == PlateBoundary::CONVERGENT ||
                 geology.volcanicActivity < 0.82)) {
                return BlockType::ANDESITE;
            }
            return BlockType::BASALT;
        case RockType::LIMESTONE:
            return BlockType::LIMESTONE;
        case RockType::SANDSTONE:
            return BlockType::SANDSTONE;
        case RockType::GRANITE:
            if (geology.crust == CrustType::CONTINENTAL &&
                geology.boundary == PlateBoundary::CONVERGENT && geology.volcanicActivity > 0.16) {
                return BlockType::ANDESITE;
            }
            return BlockType::STONE;
    }
}

inline BlockType outcrop(const GeologySample& geology) {
    return outcrop(geology, geology.rock);
}

inline bool supportsSurfaceFlora(BlockType substrate) {
    switch (substrate) {
        case BlockType::GRASS:
        case BlockType::DIRT:
        case BlockType::SAND:
        case BlockType::MUD:
        case BlockType::CLAY:
        case BlockType::SILT:
            return true;
        default:
            return false;
    }
}

inline bool supportsTreeRooting(BlockType substrate) {
    return supportsSurfaceFlora(substrate) || substrate == BlockType::SNOW;
}

inline BlockType surface(const SurfaceSample& sample, Biome biome, const VolcanicSignals& volcanic,
                         bool isFrozen, bool isSubmerged, bool alluvialDeposit = false) {
    const double geothermal = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::GEOTHERMAL);
    const double cliff = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::CLIFF);
    const double canyon = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::CANYON);
    const double scree = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::SCREE);
    const double exposedPeak =
        MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::EXPOSED_PEAK);
    const double altered = std::max(volcanic.basaltField, sample.geology.volcanicActivity);
    const bool activeVent = volcanic.conduitExposure || volcanic.craterFactor > 0.58;

    if (isSubmerged) {
        // Sediment blankets ordinary submerged lava fields. Only an active
        // crater or conduit remains exposed through the depositional cover.
        if (activeVent) return BlockType::BASALT;
        if (sample.hydrology.delta) return BlockType::SILT;
        if (sample.hydrology.river || sample.hydrology.lake || sample.hydrology.wetland) {
            return sample.soil.moisture > 0.62 ? BlockType::MUD : BlockType::CLAY;
        }
        if (sample.terrainHeight < SEA_LEVEL - 18.0 || sample.slope > 0.55) {
            return BlockType::GRAVEL;
        }
        return BlockType::SAND;
    }

    if (isFrozen && !activeVent) return BlockType::SNOW;

    // Channel and floodplain deposits overlie weathered volcanic ground.
    // Fresh vent rock is the exception because it postdates the sediment.
    if (!activeVent && sample.hydrology.delta) return BlockType::SILT;
    if (!activeVent && alluvialDeposit) {
        return sample.soil.moisture > 0.68 ? BlockType::MUD : BlockType::SILT;
    }

    const bool rareGlass = geothermal > 0.5 && sample.geology.volcanicActivity > 0.74 &&
                           activeVent && (cliff > 0.25 || sample.slope > 0.38);
    if (rareGlass) return BlockType::OBSIDIAN;

    const bool volcanicGround =
        altered > 0.50 || volcanic.basaltField > 0.16 || biome == Biome::VOLCANIC_BARREN;
    if (volcanicGround) {
        const bool depositionalAsh =
            biome == Biome::VOLCANIC_BARREN && sample.slope < 0.62 && volcanic.craterFactor < 0.58;
        if (depositionalAsh) return BlockType::VOLCANIC_ASH;
        return outcrop(sample.geology);
    }

    if (scree > 0.45 && sample.slope > 0.52) return BlockType::GRAVEL;

    const double exposure = sample.slope * 0.72 + cliff * 0.42 + canyon * 0.24 +
                            std::clamp((sample.terrainHeight - 105.0) / 150.0, 0.0, 0.32) -
                            weathering(sample) * 0.30 - sample.soil.fertility * 0.08;
    if (exposure > 0.58 || exposedPeak > 0.58) {
        return outcrop(sample.geology);
    }

    switch (biome) {
        case Biome::DESERT:
        case Biome::BEACH:
        case Biome::COLD_DESERT:
            return BlockType::SAND;
        case Biome::BADLANDS:
            return BlockType::SANDSTONE;
        case Biome::SWAMP:
        case Biome::MANGROVE:
            return BlockType::MUD;
        case Biome::FLOODED_GRASSLAND:
            return sample.soil.moisture > 0.72 ? BlockType::MUD : BlockType::GRASS;
        default:
            return BlockType::GRASS;
    }
}

inline BlockType subsurface(const SurfaceSample& sample, Biome biome,
                            const VolcanicSignals& volcanic, bool isSubmerged,
                            bool alluvialDeposit = false) {
    const double altered = std::max(volcanic.basaltField, sample.geology.volcanicActivity);
    const bool activeVent = volcanic.conduitExposure || volcanic.craterFactor > 0.58;
    if (isSubmerged) {
        if (activeVent) return outcrop(sample.geology);
        if (sample.hydrology.delta) return BlockType::SILT;
        if (sample.hydrology.river || sample.hydrology.lake || sample.hydrology.wetland)
            return BlockType::CLAY;
        return sample.terrainHeight < SEA_LEVEL - 18.0 ? BlockType::GRAVEL : BlockType::SAND;
    }
    if (!activeVent && sample.hydrology.delta) return BlockType::SILT;
    if (!activeVent && alluvialDeposit) return BlockType::CLAY;
    if (altered > 0.46 || volcanic.basaltField > 0.08 || biome == Biome::VOLCANIC_BARREN) {
        return outcrop(sample.geology);
    }
    if (MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::CLIFF) > 0.55 &&
        sample.slope > 0.82) {
        return outcrop(sample.geology);
    }
    switch (biome) {
        case Biome::DESERT:
        case Biome::BEACH:
        case Biome::COLD_DESERT:
            return BlockType::SAND;
        case Biome::BADLANDS:
            return BlockType::SANDSTONE;
        case Biome::SWAMP:
        case Biome::MANGROVE:
        case Biome::FLOODED_GRASSLAND:
            return BlockType::MUD;
        default:
            return BlockType::DIRT;
    }
}

inline bool exposesLithology(const SurfaceSample& sample, Biome biome,
                             const VolcanicSignals& volcanic, bool isFrozen, bool isSubmerged,
                             bool alluvialDeposit = false) {
    const double geothermal = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::GEOTHERMAL);
    const double cliff = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::CLIFF);
    const double canyon = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::CANYON);
    const double scree = MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::SCREE);
    const double exposedPeak =
        MacroGenerationSampler::ecotopeInfluence(sample, Ecotope::EXPOSED_PEAK);
    const double altered = std::max(volcanic.basaltField, sample.geology.volcanicActivity);
    const bool activeVent = volcanic.conduitExposure || volcanic.craterFactor > 0.58;
    if (isSubmerged || (isFrozen && !activeVent) || (!activeVent && sample.hydrology.delta) ||
        (!activeVent && alluvialDeposit)) {
        return false;
    }
    const bool rareGlass = geothermal > 0.5 && sample.geology.volcanicActivity > 0.74 &&
                           activeVent && (cliff > 0.25 || sample.slope > 0.38);
    if (rareGlass) return false;
    const bool volcanicGround =
        altered > 0.50 || volcanic.basaltField > 0.16 || biome == Biome::VOLCANIC_BARREN;
    if (volcanicGround) {
        const bool depositionalAsh =
            biome == Biome::VOLCANIC_BARREN && sample.slope < 0.62 && volcanic.craterFactor < 0.58;
        return !depositionalAsh;
    }
    if (scree > 0.45 && sample.slope > 0.52) return false;
    const double exposure = sample.slope * 0.72 + cliff * 0.42 + canyon * 0.24 +
                            std::clamp((sample.terrainHeight - 105.0) / 150.0, 0.0, 0.32) -
                            weathering(sample) * 0.30 - sample.soil.fertility * 0.08;
    return exposure > 0.58 || exposedPeak > 0.58;
}

inline SurfaceMaterialPalette materialPalette(const SurfaceSample& sample,
                                              const VolcanicSignals& volcanic, bool isFrozen,
                                              bool isSubmerged, bool alluvialDeposit = false) {
    struct WeightedMaterial {
        BlockType material = BlockType::AIR;
        double weight = 0.0;
    };
    std::array<WeightedMaterial, 8> candidates{};
    size_t candidateCount = 0;
    const auto addCandidate = [&](BlockType material, double weight) {
        if (weight <= 0.0) return;
        size_t destination = 0;
        while (destination < candidateCount && candidates[destination].material != material) {
            ++destination;
        }
        if (destination == candidateCount) {
            if (candidateCount >= candidates.size()) return;
            candidates[candidateCount].material = material;
            ++candidateCount;
        }
        candidates[destination].weight += weight;
    };

    const std::array<WeightedBiome, 4> biomes = weightedMaterialBiomes(sample);
    for (const WeightedBiome entry : biomes) {
        if (entry.weight <= 0.0) continue;
        if (exposesLithology(sample, entry.biome, volcanic, isFrozen, isSubmerged,
                             alluvialDeposit) &&
            sample.geology.lithology.primary != sample.geology.lithology.secondary) {
            const double secondaryWeight =
                std::clamp(sample.geology.lithology.transition, 0.0, 0.5);
            addCandidate(outcrop(sample.geology, sample.geology.lithology.primary),
                         entry.weight * (1.0 - secondaryWeight));
            addCandidate(outcrop(sample.geology, sample.geology.lithology.secondary),
                         entry.weight * secondaryWeight);
        } else {
            addCandidate(
                surface(sample, entry.biome, volcanic, isFrozen, isSubmerged, alluvialDeposit),
                entry.weight);
        }
    }
    if (candidateCount == 0) {
        addCandidate(
            surface(sample, sample.biome.primary, volcanic, isFrozen, isSubmerged, alluvialDeposit),
            1.0);
    }

    std::sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidateCount),
              [](const WeightedMaterial& lhs, const WeightedMaterial& rhs) {
                  if (lhs.weight != rhs.weight) return lhs.weight > rhs.weight;
                  return static_cast<uint8_t>(lhs.material) < static_cast<uint8_t>(rhs.material);
              });
    SurfaceMaterialPalette result;
    result.count = static_cast<uint8_t>(std::min(candidateCount, result.entries.size()));
    double remainingWeight = 0.0;
    for (size_t index = 0; index < result.count; ++index)
        remainingWeight += candidates[index].weight;
    int remainingUnits = 255;
    for (size_t index = 0; index < result.count; ++index) {
        const int entriesAfter = static_cast<int>(result.count - index - 1);
        const int units =
            entriesAfter == 0
                ? remainingUnits
                : std::clamp(static_cast<int>(std::lround(candidates[index].weight /
                                                          remainingWeight * remainingUnits)),
                             1, remainingUnits - entriesAfter);
        result.entries[index] = {candidates[index].material, static_cast<uint8_t>(units)};
        remainingUnits -= units;
        remainingWeight -= candidates[index].weight;
    }
    return result;
}

inline BlockType selectMaterial(const SurfaceMaterialPalette& palette, double rank) {
    if (palette.count == 0) return BlockType::AIR;
    const int target = std::clamp(static_cast<int>(rank * 255.0), 0, 254);
    int cumulative = 0;
    for (size_t index = 0; index < palette.count; ++index) {
        cumulative += palette.entries[index].weight;
        if (target < cumulative) return palette.entries[index].material;
    }
    return palette.entries[palette.count - 1].material;
}

} // namespace worldgen::surface_material

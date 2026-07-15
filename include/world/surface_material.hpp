#pragma once

#include "world/block_properties.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <cmath>

namespace worldgen::surface_material {

inline constexpr uint64_t DITHER_STREAM = 0x535552464143454DULL;

struct VolcanicSignals {
    double basaltField = 0.0;
    double craterFactor = 0.0;
    bool conduitExposure = false;
};

inline Biome materialBiome(const SurfaceSample& sample, const CounterRng& random, int64_t x,
                           int64_t z) {
    const double secondaryWeight = std::clamp(sample.biome.transition, 0.0, 0.5);
    return multiscaleDitherThreshold(random, DITHER_STREAM, x, z) < secondaryWeight
               ? sample.biome.secondary
               : sample.biome.primary;
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
    if (!sample.hydrology.ocean && !sample.hydrology.river && !sample.hydrology.lake) return false;
    const int terrainY = static_cast<int>(std::floor(sample.terrainHeight));
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

inline BlockType outcrop(const GeologySample& geology) {
    switch (geology.rock) {
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
        if (sample.hydrology.river || sample.hydrology.lake) {
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
        if (sample.hydrology.river || sample.hydrology.lake) return BlockType::CLAY;
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

} // namespace worldgen::surface_material

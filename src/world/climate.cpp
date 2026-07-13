#include "world/climate.hpp"

#include "world/gen_seeds.hpp"

#include <algorithm>
#include <cmath>

namespace {

// Piecewise-linear spline over sorted control points; clamps outside.
struct SplinePoint {
    double in;
    double out;
};

template <size_t N>
double spline(const SplinePoint (&pts)[N], double v) {
    if (v <= pts[0].in) return pts[0].out;
    for (size_t i = 1; i < N; ++i) {
        if (v <= pts[i].in) {
            double t = (v - pts[i - 1].in) / (pts[i].in - pts[i - 1].in);
            return pts[i - 1].out + (pts[i].out - pts[i - 1].out) * t;
        }
    }
    return pts[N - 1].out;
}

double smoothstep(double edge0, double edge1, double v) {
    double t = std::clamp((v - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Base height from continentalness: deep ocean → coast → inland plateau.
constexpr SplinePoint BASE_HEIGHT[] = {
    {-1.00, 38.0}, {-0.45, 50.0}, {-0.19, 61.0}, {-0.10, 64.0},
    {0.03, 68.0},  {0.30, 77.0},  {1.00, 96.0},
};

// Mountain amplitude from erosion: young ranges → worn flat.
constexpr SplinePoint MOUNTAIN_AMP[] = {
    {-1.00, 70.0}, {-0.40, 34.0}, {0.10, 14.0}, {0.60, 6.0}, {1.00, 2.0},
};

} // namespace

ClimateSampler::ClimateSampler(uint32_t worldSeed)
    : continents_(genseed::subSeed(worldSeed, genseed::CONTINENTS))
    , erosion_(genseed::subSeed(worldSeed, genseed::EROSION))
    , ridges_(genseed::subSeed(worldSeed, genseed::RIDGES))
    , temperature_(genseed::subSeed(worldSeed, genseed::TEMPERATURE))
    , humidity_(genseed::subSeed(worldSeed, genseed::HUMIDITY))
    , entrance_(genseed::subSeed(worldSeed, genseed::ENTRANCE))
    , ravine_(genseed::subSeed(worldSeed, genseed::RAVINE)) {}

ColumnShape ClimateSampler::shapeColumn(double x, double z) const {
    ColumnShape shape;
    ClimateSample& c = shape.climate;
    c.continentalness = continents_.octave2D(x / 1200.0, z / 1200.0, 4);
    c.erosion = erosion_.octave2D(x / 900.0, z / 900.0, 4);
    c.ridges = ridges_.octave2D(x / 500.0, z / 500.0, 3);
    c.temperature = temperature_.octave2D(x / 1400.0, z / 1400.0, 3);
    c.humidity = humidity_.octave2D(x / 1000.0, z / 1000.0, 3);

    // Peaks-and-valleys fold: |R| = 2/3 is a ridge line, |R| = 0 a valley
    // floor. Remapped to [0,1] so it only ever ADDS height — valley floors
    // sit at the continental base, which keeps them above sea level inland.
    double pv = 1.0 - std::abs(3.0 * std::abs(c.ridges) - 2.0);
    double pv01 = (pv + 1.0) * 0.5;

    double base = spline(BASE_HEIGHT, c.continentalness);
    double landFactor = smoothstep(-0.15, 0.10, c.continentalness);
    double mountainAmp = spline(MOUNTAIN_AMP, c.erosion) * landFactor;
    double height = base + mountainAmp * pv01;

    // Rare shallow islands far out in deep ocean (mushroom island habitat)
    height += smoothstep(-0.72, -0.90, c.continentalness) * 26.0 * pv01;

    // Rivers: carve a channel along ridge-noise zero lines on land. The
    // channel floor sits at 59 so the water fill (sea level 64) makes a
    // 4-5 block deep river; smoothstep gives sloped banks.
    if (c.continentalness > -0.10 && height > 59.0) {
        double r = std::abs(c.ridges);
        // Band width trades river coverage against bank steepness: 0.02 of
        // channel keeps rivers at a few percent of the land, and the 0.04
        // ramp keeps banks under the continuity test's slope bound.
        double cutFactor = smoothstep(0.060, 0.020, r);
        shape.riverCut = cutFactor * (height - 59.0);
        height -= shape.riverCut;
    }

    shape.height = std::min(height, 240.0);

    // 3D detail amplitude: craggy on mountains, gentle on plains, and damped
    // to zero at/below sea level so coasts and river banks stay clean.
    double mNorm = mountainAmp * pv01 / 70.0;
    shape.detailAmp = (2.5 + 11.0 * mNorm) * smoothstep(62.0, 70.0, shape.height);

    // Cave entrance mask: where this exceeds 0.4 (and the column is well
    // above sea level) caves keep full strength up to the surface.
    shape.entrance = entrance_.octave2D(x / 140.0, z / 140.0, 2);

    // Ravines: thin ridged lines; edge ramps 0 → 1 across the canyon lip.
    double rav = ravine_.ridged2D(x / 280.0, z / 280.0, 2);
    shape.ravineEdge = smoothstep(0.82, 0.96, rav);
    shape.ravineFloor = std::max(12.0, shape.height - 46.0 * shape.ravineEdge);

    return shape;
}

Biome ClimateSampler::selectBiome(const ColumnShape& shape) {
    const ClimateSample& c = shape.climate;
    double h = shape.height;

    if (c.continentalness < -0.68 && h >= 60.0) return Biome::MUSHROOM_ISLAND;
    // Rivers sit below 62 too — classify them before the ocean bands
    if (shape.riverCut > 1.5 && h < 66.0) return Biome::RIVER;
    if (h < 50.0) return Biome::DEEP_OCEAN;
    if (h < 62.0) return Biome::OCEAN;
    if (h < 65.0 && c.continentalness < 0.0 && c.temperature > -0.35) return Biome::BEACH;

    double pv01 = (1.0 - std::abs(3.0 * std::abs(c.ridges) - 2.0) + 1.0) * 0.5;
    if (h > 102.0 && pv01 > 0.55) return Biome::EXTREME_HILLS;

    if (c.temperature < -0.45) return c.humidity < -0.1 ? Biome::ICE_SPIKES : Biome::TAIGA;
    if (c.temperature > 0.4 && c.humidity < -0.15) return Biome::DESERT;
    if (c.humidity > 0.55 && h < 70.0) return Biome::SWAMP;
    if (c.humidity > 0.30) return c.temperature < -0.05 ? Biome::BIRCH_FOREST : Biome::FOREST;
    if (c.humidity > 0.0 && c.humidity < 0.22 && c.temperature > 0.15) return Biome::FLOWER_FIELD;
    return Biome::PLAINS;
}

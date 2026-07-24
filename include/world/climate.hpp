#pragma once

#include "world/chunk.hpp"
#include "world/noise.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Climate fields and terrain shaping — the 2D half of world generation.
//
// Five low-frequency fields (continentalness, erosion, ridges, temperature,
// humidity) drive both the terrain shape and the biome choice. Terrain
// height reads continuous splines of the fields, and the biome ID is derived
// from the SAME fields but used only for materials/flora/trees — so biome
// borders can never produce height cliffs (the old design added per-biome
// height offsets after a discrete biome pick, which did).
//
// Everything here is a pure function of world coordinates + seed: that is
// what lets neighboring chunks — and cross-chunk feature placement — agree
// on shared columns without any generation-order dependency.
// ---------------------------------------------------------------------------

struct ClimateSample {
    double continentalness = 0.0; // [-1,1] ocean floor → continent interior
    double erosion = 0.0;         // [-1,1] mountainous → worn flat
    double ridges = 0.0;          // [-1,1] folded into peaks/valleys and rivers
    double temperature = 0.0;     // [-1,1] frozen → hot
    double humidity = 0.0;        // [-1,1] arid → wet
};

// Everything the density function and surface pass need to know about one
// column. All fields are plain doubles so per-block columns can be produced
// by bilinear interpolation of lattice columns.
struct ColumnShape {
    ClimateSample climate;
    double height = 64.0;      // pre-cave surface height H (after river cut)
    double detailAmp = 0.0;    // 3D detail amplitude in blocks
    double entrance = 0.0;     // cave entrance mask (raw noise value)
    double riverCut = 0.0;     // depth removed by the river channel (blocks)
    double ravineEdge = 0.0;   // 0 outside ravines → 1 at the canyon core
    double ravineFloor = 64.0; // carve floor when ravineEdge > 0
};

class ClimateSampler {
public:
    explicit ClimateSampler(uint32_t worldSeed);

    // Full column shape at any world column (pure).
    ColumnShape shapeColumn(double x, double z) const;

    // Bounded cubic generation needs only this mask. Exact v4 density must
    // not evaluate the legacy continent, erosion, climate, river, and ravine
    // fields merely to retain coordinate-pure cave entrances.
    double caveEntrance(double x, double z) const;

    // Biome from an (interpolated) column shape (pure). The rules read the
    // climate fields plus the resulting height, never a neighbor.
    static Biome selectBiome(const ColumnShape& shape);

    // Frozen columns get ice caps and snow tops.
    static bool isFrozen(const ColumnShape& shape) { return shape.climate.temperature < -0.45; }

private:
    SimplexNoise continents_;
    SimplexNoise erosion_;
    SimplexNoise ridges_;
    SimplexNoise temperature_;
    SimplexNoise humidity_;
    SimplexNoise entrance_;
    SimplexNoise ravine_;
};

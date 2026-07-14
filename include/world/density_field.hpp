#pragma once

#include "world/climate.hpp"
#include "world/noise.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// The density function — the 3D half of world generation.
//
// A voxel is solid iff D(x,y,z) > 0. Terrain shape and every cave type
// combine into the one function (caves contribute negative density through
// min()), so carving can never orphan a surface layer or disagree with the
// height map: there is no separate carve pass to get out of sync.
//
// D is evaluated only at world-aligned lattice points (LATTICE_XZ x
// LATTICE_Y x LATTICE_XZ spacing) and trilinearly interpolated to voxels.
// Because the lattice is world-aligned, interpolated density is a pure
// function of world position — identical no matter which chunk computes it.
// All component densities are clamped to ±DENSITY_CAP block units so the
// min() combine and the interpolation stay well-behaved.
// ---------------------------------------------------------------------------

// Lattice spacing must stay a power of two: latticeFloor masks with
// (LATTICE_XZ - 1) to floor negative coordinates.
inline constexpr int LATTICE_XZ = 4;
inline constexpr int LATTICE_Y = 4;
inline constexpr double DENSITY_CAP = 32.0;

// One fixed operation order for density interpolation: bilinear in xz at the
// two lattice y-levels, then linear in y. Every consumer (bulk chunk fill
// and single-column surfaceYAt queries) must interpolate through THIS
// function — a second code path with reordered float ops would produce
// different bits and visible chunk seams.
inline double lerpDensity(double a, double b, double t) {
    return a + (b - a) * t;
}

inline double bilerpDensity(double d00, double d10, double d01, double d11, double fx, double fz) {
    return lerpDensity(lerpDensity(d00, d10, fx), lerpDensity(d01, d11, fx), fz);
}

class DensityField {
public:
    explicit DensityField(uint32_t worldSeed);

    // Density at a lattice point, given that column's shape (pure).
    double density(double x, double y, double z, const ColumnShape& col) const;

private:
    SimplexNoise detail_;
    SimplexNoise cheese_;
    SimplexNoise spaghetti1_;
    SimplexNoise spaghetti2_;
    SimplexNoise noodle1_;
    SimplexNoise noodle2_;
};

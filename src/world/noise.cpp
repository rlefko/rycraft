#include "world/noise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

// Simple LCG PRNG for permutation table initialization
static uint32_t lcgNext(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (state >> 16) & 0x7fff;
}

SimplexNoise::SimplexNoise(uint32_t seed) {
    initPerm(seed);
}

void SimplexNoise::initPerm(uint32_t seed) {
    // Initialize with identity permutation [0, 1, 2, ..., 255]
    uint8_t base[256];
    for (int i = 0; i < 256; ++i) {
        base[i] = static_cast<uint8_t>(i);
    }

    // Fisher-Yates shuffle seeded by LCG
    uint32_t state = seed;
    for (int i = 255; i > 0; --i) {
        int j = static_cast<int>(lcgNext(state)) % (i + 1);
        std::swap(base[i], base[j]);
    }

    // Double the table for modulo-free access
    for (int i = 0; i < 512; ++i) {
        perm[i] = base[i & 255];
    }
}

double SimplexNoise::dot3(int hash, double x, double y, double z) const {
    int idx = hash & 0x0F;
    return x * GRAD3[idx][0] + y * GRAD3[idx][1] + z * GRAD3[idx][2];
}

double SimplexNoise::noise3D(double x, double y, double z) const {
    // Skewing factors for 3D simplex
    constexpr double F3 = 1.0 / 3.0;
    constexpr double G3 = 1.0 / 6.0;

    // Skew input space to determine simplex cell
    double s = (x + y + z) * F3;
    int i = static_cast<int>(std::floor(x + s));
    int j = static_cast<int>(std::floor(y + s));
    int k = static_cast<int>(std::floor(z + s));

    // Unskew back to get cubic cell origin
    double t = (i + j + k) * G3;
    double X0 = i - t;
    double Y0 = j - t;
    double Z0 = k - t;

    double x0 = x - X0;
    double y0 = y - Y0;
    double z0 = z - Z0;

    // Determine simplex offsets (which corner of the tetrahedron)
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) {
            // x >= y >= z
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        } else if (x0 >= z0) {
            // x >= z >= y
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        } else {
            // z >= x >= y
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        }
    } else {
        if (y0 < z0) {
            // z >= y >= x
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 0;
            j2 = 1;
            k2 = 1;
        } else if (x0 < z0) {
            // y >= z >= x
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 0;
            j2 = 1;
            k2 = 1;
        } else {
            // y >= x >= z
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        }
    }

    // Offsets for the other three corners
    double x1 = x0 - i1 + G3;
    double y1 = y0 - j1 + G3;
    double z1 = z0 - k1 + G3;
    double x2 = x0 - i2 + 2.0 * G3;
    double y2 = y0 - j2 + 2.0 * G3;
    double z2 = z0 - k2 + 2.0 * G3;
    double x3 = x0 - 1.0 + 3.0 * G3;
    double y3 = y0 - 1.0 + 3.0 * G3;
    double z3 = z0 - 1.0 + 3.0 * G3;

    // Hash coordinates to gradient indices
    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;

    int gi0 = perm[ii + perm[jj + perm[kk]]] % 12;
    int gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
    int gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
    int gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;

    // Contribution from corner 0
    double t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    double contrib0 = 0.0;
    if (t0 > 0.0) {
        t0 *= t0;
        contrib0 = t0 * t0 * dot3(gi0, x0, y0, z0);
    }

    // Contribution from corner 1
    double t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    double contrib1 = 0.0;
    if (t1 > 0.0) {
        t1 *= t1;
        contrib1 = t1 * t1 * dot3(gi1, x1, y1, z1);
    }

    // Contribution from corner 2
    double t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    double contrib2 = 0.0;
    if (t2 > 0.0) {
        t2 *= t2;
        contrib2 = t2 * t2 * dot3(gi2, x2, y2, z2);
    }

    // Contribution from corner 3
    double t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    double contrib3 = 0.0;
    if (t3 > 0.0) {
        t3 *= t3;
        contrib3 = t3 * t3 * dot3(gi3, x3, y3, z3);
    }

    // Scale to approximately [-1, 1]
    return 32.0 * (contrib0 + contrib1 + contrib2 + contrib3);
}

double SimplexNoise::noise2D(double x, double y) const {
    return noise3D(x, y, 0.0);
}

double SimplexNoise::operator()(double x, double y) const {
    return noise2D(x, y);
}

double SimplexNoise::octave2D(double x, double y, int octaves, double persistence,
                              double lacunarity) const {
    if (octaves <= 0) return 0.0;

    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    double maxAmplitude = 0.0;

    for (int i = 0; i < octaves; ++i) {
        total += noise2D(x * frequency, y * frequency) * amplitude;
        maxAmplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize to [-1, 1]
    return total / maxAmplitude;
}

double SimplexNoise::octave3D(double x, double y, double z, int octaves, double persistence,
                              double lacunarity) const {
    if (octaves <= 0) return 0.0;

    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    double maxAmplitude = 0.0;

    for (int i = 0; i < octaves; ++i) {
        total += noise3D(x * frequency, y * frequency, z * frequency) * amplitude;
        maxAmplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total / maxAmplitude;
}

double SimplexNoise::ridged2D(double x, double y, int octaves, double persistence,
                              double lacunarity) const {
    if (octaves <= 0) return 0.0;

    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    double previous = 1.0;
    double maxAmplitude = 0.0;

    for (int i = 0; i < octaves; ++i) {
        double signal = noise2D(x * frequency, y * frequency);
        signal = std::abs(signal);
        signal = 1.0 - signal;
        signal *= signal; // Square for sharper ridges
        signal *= previous;
        signal *= amplitude;

        total += signal;
        maxAmplitude += amplitude;
        previous = signal;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total / maxAmplitude;
}

double SimplexNoise::ridged3D(double x, double y, double z, int octaves, double persistence,
                              double lacunarity) const {
    if (octaves <= 0) return 0.0;

    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    double previous = 1.0;
    double maxAmplitude = 0.0;

    for (int i = 0; i < octaves; ++i) {
        double signal = noise3D(x * frequency, y * frequency, z * frequency);
        signal = std::abs(signal);
        signal = 1.0 - signal;
        signal *= signal;
        signal *= previous;
        signal *= amplitude;

        total += signal;
        maxAmplitude += amplitude;
        previous = signal;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total / maxAmplitude;
}

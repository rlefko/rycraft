#pragma once

#include <cstdint>

class SimplexNoise {
public:
    explicit SimplexNoise(uint32_t seed = 0);

    // 2D noise (wrapper calling 3D with z=0)
    double operator()(double x, double y) const;
    double noise2D(double x, double y) const;

    // 3D noise
    double noise3D(double x, double y, double z) const;

    // Octave fBm support
    double octave2D(double x, double y, int octaves, double persistence = 0.5,
                    double lacunarity = 2.0) const;
    double octave3D(double x, double y, double z, int octaves, double persistence = 0.5,
                    double lacunarity = 2.0) const;

    // Ridged noise (for caves/ridges)
    double ridged2D(double x, double y, int octaves, double persistence = 0.5,
                    double lacunarity = 2.0) const;
    double ridged3D(double x, double y, double z, int octaves, double persistence = 0.5,
                    double lacunarity = 2.0) const;

private:
    // Permutation table — 512 entries (doubled for modulo-free access)
    uint8_t perm[512];

    // Gradient vectors for 3D simplex noise
    static constexpr double GRAD3[12][3] = {{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
                                            {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
                                            {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}};

    void initPerm(uint32_t seed);
    double dot3(int hash, double x, double y, double z) const;
};

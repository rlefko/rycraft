#pragma once
#include "world/chunk.hpp"
#include "world/noise.hpp"
#include <cstdint>

struct CaveConfig {
    // Cheese caves — large open spaces
    double cheeseThreshold = 0.05;
    double cheeseScale = 16.0;

    // Spaghetti caves — thin tunnels
    double spaghettiThreshold = 0.10;
    double spaghettiScale = 8.0;

    // Noodle caves — ridged intersection
    double noodleThreshold = 0.08;
    double noodleScale = 4.0;

    // Height distribution
    double caveCeiling = 128.0; // No caves above this
    double caveFloor = 4.0;     // No caves below this
};

class CaveGenerator {
public:
    explicit CaveGenerator(uint32_t seed);

    // Carve caves into a chunk
    void carve(Chunk& chunk, const CaveConfig& config = {}) const;

    // Get cave noise value at world position
    double getNoise(int x, int y, int z, const CaveConfig& config) const;

private:
    SimplexNoise cheeseNoise_;
    SimplexNoise spaghettiNoise1_;
    SimplexNoise spaghettiNoise2_;
    SimplexNoise noodleNoise1_;
    SimplexNoise noodleNoise2_;

    bool isCheeseCave(int x, int y, int z, const CaveConfig& config) const;
    bool isSpaghettiCave(int x, int y, int z, const CaveConfig& config) const;
    bool isNoodleCave(int x, int y, int z, const CaveConfig& config) const;
};

#include "world/caves.hpp"

#include <cmath>

CaveGenerator::CaveGenerator(uint32_t seed)
    : cheeseNoise_(seed)
    , spaghettiNoise1_(seed + 1)
    , spaghettiNoise2_(seed + 2)
    , noodleNoise1_(seed + 3)
    , noodleNoise2_(seed + 4) {}

bool CaveGenerator::isCheeseCave(int x, int y, int z, const CaveConfig& config) const {
    double nx = static_cast<double>(x) / config.cheeseScale;
    double ny = static_cast<double>(y) / config.cheeseScale;
    double nz = static_cast<double>(z) / config.cheeseScale;

    double val = cheeseNoise_.noise3D(nx, ny, nz);
    // Map from [-1, 1] to [0, 1]
    val = (val + 1.0) * 0.5;

    return val > config.cheeseThreshold;
}

bool CaveGenerator::isSpaghettiCave(int x, int y, int z, const CaveConfig& config) const {
    double nx = static_cast<double>(x) / config.spaghettiScale;
    double ny = static_cast<double>(y) / config.spaghettiScale;
    double nz = static_cast<double>(z) / config.spaghettiScale;

    double val1 = spaghettiNoise1_.noise3D(nx, ny, nz);
    double val2 = spaghettiNoise2_.noise3D(nx, ny, nz);

    // Map from [-1, 1] to [0, 1]
    val1 = (val1 + 1.0) * 0.5;
    val2 = (val2 + 1.0) * 0.5;

    // Dual intersection: both must exceed threshold
    return val1 > config.spaghettiThreshold && val2 > config.spaghettiThreshold;
}

bool CaveGenerator::isNoodleCave(int x, int y, int z, const CaveConfig& config) const {
    double nx = static_cast<double>(x) / config.noodleScale;
    double ny = static_cast<double>(y) / config.noodleScale;
    double nz = static_cast<double>(z) / config.noodleScale;

    // Ridged noise: abs(1 - |noise|)^2
    double raw1 = noodleNoise1_.noise3D(nx, ny, nz);
    double raw2 = noodleNoise2_.noise3D(nx, ny, nz);

    double ridged1 = std::abs(raw1);
    ridged1 = 1.0 - ridged1;
    ridged1 = ridged1 * ridged1;

    double ridged2 = std::abs(raw2);
    ridged2 = 1.0 - ridged2;
    ridged2 = ridged2 * ridged2;

    // Ridged intersection: both must exceed threshold
    return ridged1 > config.noodleThreshold && ridged2 > config.noodleThreshold;
}

double CaveGenerator::getNoise(int x, int y, int z, const CaveConfig& config) const {
    // Combined cave noise for debugging
    double cheese = isCheeseCave(x, y, z, config) ? 1.0 : 0.0;
    double spaghetti = isSpaghettiCave(x, y, z, config) ? 1.0 : 0.0;
    double noodle = isNoodleCave(x, y, z, config) ? 1.0 : 0.0;

    return (cheese + spaghetti + noodle) / 3.0;
}

void CaveGenerator::carve(Chunk& chunk, const CaveConfig& config) const {
    int worldBaseX = chunk.chunkX * CHUNK_WIDTH;
    int worldBaseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int localZ = 0; localZ < CHUNK_DEPTH; ++localZ) {
        for (int localX = 0; localX < CHUNK_WIDTH; ++localX) {
            int worldX = worldBaseX + localX;
            int worldZ = worldBaseZ + localZ;

            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                // Skip heights outside cave range
                if (static_cast<double>(y) >= config.caveCeiling) continue;
                if (static_cast<double>(y) <= config.caveFloor) continue;

                // Never carve through bedrock
                if (y < 2) continue;

                // Only carve stone blocks (preserve surface, water, etc.)
                BlockType current = chunk.getBlock(localX, y, localZ);
                if (current != BlockType::STONE) continue;

                // Check each cave type
                bool cave = isCheeseCave(worldX, y, worldZ, config) ||
                            isSpaghettiCave(worldX, y, worldZ, config) ||
                            isNoodleCave(worldX, y, worldZ, config);

                if (cave) {
                    chunk.setBlock(localX, y, localZ, BlockType::AIR);
                }
            }
        }
    }
}

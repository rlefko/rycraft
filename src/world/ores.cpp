#include "world/ores.hpp"

#include <cmath>
#include <cstdlib>

// Simple deterministic PRNG for ore placement
static uint32_t oreLcg(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (state >> 16) & 0x7fff;
}

OreGenerator::OreGenerator(uint32_t seed) : oreNoise_(seed) {}

double OreGenerator::getOreDensity(double y, const OreConfig::OreDistribution& dist) const {
    // Trapezoid height distribution:
    // - 0 below minHeight
    // - linearly increases to 1.0 at falloffHeight
    // - stays at 1.0 until maxHeight * 0.75 (mid-peak)
    // - linearly decreases to 0 at maxHeight
    if (y < dist.minHeight || y > dist.maxHeight) {
        return 0.0;
    }

    // Rising edge: minHeight -> falloffHeight
    if (y < dist.falloffHeight) {
        return (y - dist.minHeight) / (dist.falloffHeight - dist.minHeight);
    }

    // Falling edge: maxHeight * 0.75 -> maxHeight
    double fallStart = dist.maxHeight * 0.75;
    if (y > fallStart) {
        return (dist.maxHeight - y) / (dist.maxHeight - fallStart);
    }

    // Peak plateau
    return 1.0;
}

void OreGenerator::generateOreVein(Chunk& chunk, int x, int y, int z, int veinSize,
                                   BlockType ore) const {
    // Generate a sphere of ore replacing only STONE blocks
    int radiusSq = veinSize * veinSize;

    for (int dx = -veinSize; dx <= veinSize; ++dx) {
        for (int dy = -veinSize; dy <= veinSize; ++dy) {
            for (int dz = -veinSize; dz <= veinSize; ++dz) {
                int distSq = dx * dx + dy * dy + dz * dz;
                if (distSq > radiusSq) continue;

                int targetX = x + dx;
                int targetY = y + dy;
                int targetZ = z + dz;

                // Bounds check
                if (targetX < 0 || targetX >= CHUNK_WIDTH) continue;
                if (targetY < 0 || targetY >= CHUNK_HEIGHT) continue;
                if (targetZ < 0 || targetZ >= CHUNK_DEPTH) continue;

                // Only replace STONE blocks
                if (chunk.getBlock(targetX, targetY, targetZ) == BlockType::STONE) {
                    chunk.setBlock(targetX, targetY, targetZ, ore);
                }
            }
        }
    }
}

void OreGenerator::generate(Chunk& chunk, const OreConfig& config) const {
    int worldBaseX = chunk.chunkX * CHUNK_WIDTH;
    int worldBaseZ = chunk.chunkZ * CHUNK_DEPTH;

    // Seed for deterministic PRNG within this chunk
    uint32_t state = static_cast<uint32_t>(worldBaseX * 374761393u + worldBaseZ * 668265263u);

    for (const auto& oreDist : config.ores) {
        for (int cluster = 0; cluster < oreDist.clustersPerChunk; ++cluster) {
            // Use noise to determine cluster placement
            // Spread clusters across a 3x3 chunk area to avoid seams
            int noiseX = worldBaseX + (oreLcg(state) % (CHUNK_WIDTH * 3));
            int noiseZ = worldBaseZ + (oreLcg(state) % (CHUNK_DEPTH * 3));

            double noiseVal = oreNoise_.noise3D(static_cast<double>(noiseX) * 0.1,
                                                static_cast<double>(cluster) * 0.5,
                                                static_cast<double>(noiseZ) * 0.1);

            // Map from [-1, 1] to [0, 1]
            double normalized = (noiseVal + 1.0) * 0.5;

            // Discard cluster if noise is too high
            if (normalized > oreDist.discardThreshold) {
                continue;
            }

            // Determine cluster center position
            int centerX = (oreLcg(state) % (CHUNK_WIDTH * 3)) - CHUNK_WIDTH;
            int centerZ = (oreLcg(state) % (CHUNK_DEPTH * 3)) - CHUNK_DEPTH;

            // Height based on ore distribution
            double heightRange = oreDist.maxHeight - oreDist.minHeight;
            double heightT = (static_cast<double>(oreLcg(state)) / 32767.0) * heightRange;
            int centerY = static_cast<int>(oreDist.minHeight + heightT);

            // Apply density weighting
            double density = getOreDensity(static_cast<double>(centerY), oreDist);
            if (density <= 0.0) continue;

            // Vein size scaled by density
            int veinRange = oreDist.maxVeinSize - oreDist.minVeinSize;
            int veinSize = oreDist.minVeinSize +
                           static_cast<int>((static_cast<double>(oreLcg(state)) / 32767.0) *
                                            veinRange * density);
            veinSize = std::max(1, veinSize);

            // Generate the ore vein
            generateOreVein(chunk, centerX, centerY, centerZ, veinSize, oreDist.ore);
        }
    }
}

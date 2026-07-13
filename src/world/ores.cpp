#include "world/ores.hpp"

#include "world/gen_seeds.hpp"

#include <algorithm>

namespace {

struct OreKind {
    BlockType block;
    int minY;
    int maxY;
    int attemptsPerChunk;
    int blobMin;
    int blobMax; // walk steps, capped ≤ 12 (see header)
};

// Depth bands: common coal up high, precious metals compressed toward
// bedrock so deep caves are worth exploring.
constexpr OreKind ORE_KINDS[] = {
    {BlockType::COAL_ORE, 48, 131, 14, 6, 12},
    {BlockType::IRON_ORE, 8, 71, 10, 4, 8},
    {BlockType::GOLD_ORE, 4, 35, 5, 3, 6},
    {BlockType::DIAMOND_ORE, 2, 17, 4, 2, 5},
};

} // namespace

OrePlacer::OrePlacer(uint32_t worldSeed) : seed_(worldSeed) {}

void OrePlacer::place(Chunk& chunk) const {
    const int baseX = chunk.chunkX * CHUNK_WIDTH;
    const int baseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sourceChunkX = chunk.chunkX + dx;
            int sourceChunkZ = chunk.chunkZ + dz;
            int sourceBaseX = sourceChunkX * CHUNK_WIDTH;
            int sourceBaseZ = sourceChunkZ * CHUNK_DEPTH;

            for (size_t oreIndex = 0; oreIndex < std::size(ORE_KINDS); ++oreIndex) {
                const OreKind& ore = ORE_KINDS[oreIndex];
                SeededRng rng(hashCoords(sourceChunkX, sourceChunkZ,
                                         genseed::subSeed(seed_, genseed::ORES + oreIndex)));

                for (int attempt = 0; attempt < ore.attemptsPerChunk; ++attempt) {
                    // Fixed draw order — every neighbor re-rolls the same
                    // sequence, so a skipped attempt costs the same draws.
                    int x = sourceBaseX + rng.nextInt(0, CHUNK_WIDTH - 1);
                    int z = sourceBaseZ + rng.nextInt(0, CHUNK_DEPTH - 1);
                    int y = rng.nextInt(ore.minY, ore.maxY);
                    int steps = rng.nextInt(ore.blobMin, ore.blobMax);

                    for (int i = 0; i < steps; ++i) {
                        int lx = x - baseX;
                        int lz = z - baseZ;
                        if (lx >= 0 && lx < CHUNK_WIDTH && lz >= 0 && lz < CHUNK_DEPTH &&
                            chunk.getBlock(lx, y, lz) == BlockType::STONE) {
                            chunk.setBlock(lx, y, lz, ore.block);
                        }
                        // Random-walk one axis step; y stays inside the band
                        int axis = rng.nextInt(0, 2);
                        int dir = rng.nextInt(0, 1) == 0 ? -1 : 1;
                        if (axis == 0) {
                            x += dir;
                        } else if (axis == 1) {
                            y = std::clamp(y + dir, ore.minY, ore.maxY);
                        } else {
                            z += dir;
                        }
                    }
                }
            }
        }
    }
}

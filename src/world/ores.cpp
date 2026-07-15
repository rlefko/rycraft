#include "world/ores.hpp"

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
    {BlockType::COAL_ORE, -32, 160, 14, 6, 12},
    {BlockType::IRON_ORE, -64, 96, 10, 4, 8},
    {BlockType::GOLD_ORE, -112, 32, 5, 3, 6},
    {BlockType::DIAMOND_ORE, -120, -32, 4, 2, 5},
};

// Streams are stable addresses, not seeds consumed in sequence. Each ore
// kind receives its own stream domain so adding a kind cannot perturb any
// existing vein.
constexpr uint64_t ORE_ANCHOR_X_STREAM = 0x4F52455F414E4358ULL;
constexpr uint64_t ORE_ANCHOR_Z_STREAM = 0x4F52455F414E435AULL;
constexpr uint64_t ORE_ANCHOR_Y_STREAM = 0x4F52455F414E4359ULL;
constexpr uint64_t ORE_LENGTH_STREAM = 0x4F52455F4C454E47ULL;
constexpr uint64_t ORE_WALK_AXIS_STREAM = 0x4F52455F57414C4BULL;
constexpr uint64_t ORE_WALK_SIGN_STREAM = 0x4F52455F5349474EULL;
constexpr uint32_t MAX_WALK_STEPS = 12;

constexpr uint64_t oreStream(uint64_t stream, size_t oreIndex) {
    return stream ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(oreIndex + 1));
}

constexpr uint32_t walkIndex(int attempt, int step) {
    return static_cast<uint32_t>(attempt) * MAX_WALK_STEPS + static_cast<uint32_t>(step);
}

} // namespace

OrePlacer::OrePlacer(uint32_t worldSeed) : random_(worldSeed) {}

void OrePlacer::place(Chunk& chunk) const {
    const int64_t baseX = chunk.chunkX * CHUNK_WIDTH;
    const int baseY = chunk.chunkY * CHUNK_HEIGHT;
    const int64_t baseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            int64_t sourceChunkX = chunk.chunkX + dx;
            int64_t sourceChunkZ = chunk.chunkZ + dz;
            int64_t sourceBaseX = sourceChunkX * CHUNK_WIDTH;
            int64_t sourceBaseZ = sourceChunkZ * CHUNK_DEPTH;

            for (size_t oreIndex = 0; oreIndex < std::size(ORE_KINDS); ++oreIndex) {
                const OreKind& ore = ORE_KINDS[oreIndex];

                for (int attempt = 0; attempt < ore.attemptsPerChunk; ++attempt) {
                    const uint32_t candidateIndex = static_cast<uint32_t>(attempt);
                    int64_t x =
                        sourceBaseX + random_.uniformInt(oreStream(ORE_ANCHOR_X_STREAM, oreIndex),
                                                         sourceChunkX, 0, sourceChunkZ,
                                                         candidateIndex, 0, CHUNK_WIDTH - 1);
                    int64_t z =
                        sourceBaseZ + random_.uniformInt(oreStream(ORE_ANCHOR_Z_STREAM, oreIndex),
                                                         sourceChunkX, 0, sourceChunkZ,
                                                         candidateIndex, 0, CHUNK_DEPTH - 1);
                    int y =
                        random_.uniformInt(oreStream(ORE_ANCHOR_Y_STREAM, oreIndex), sourceChunkX,
                                           0, sourceChunkZ, candidateIndex, ore.minY, ore.maxY);
                    int steps =
                        random_.uniformInt(oreStream(ORE_LENGTH_STREAM, oreIndex), sourceChunkX, 0,
                                           sourceChunkZ, candidateIndex, ore.blobMin, ore.blobMax);

                    for (int i = 0; i < steps; ++i) {
                        int lx = static_cast<int>(x - baseX);
                        int ly = y - baseY;
                        int lz = static_cast<int>(z - baseZ);
                        if (lx >= 0 && lx < CHUNK_WIDTH && ly >= 0 && ly < CHUNK_HEIGHT &&
                            lz >= 0 && lz < CHUNK_DEPTH &&
                            chunk.getBlock(lx, ly, lz) == BlockType::STONE) {
                            chunk.setBlock(lx, ly, lz, ore.block);
                        }
                        // Random-walk one axis step; y stays inside the band
                        const uint32_t stepIndex = walkIndex(attempt, i);
                        int axis =
                            random_.uniformInt(oreStream(ORE_WALK_AXIS_STREAM, oreIndex),
                                               sourceChunkX, 0, sourceChunkZ, stepIndex, 0, 2);
                        int dir =
                            random_.uniformInt(oreStream(ORE_WALK_SIGN_STREAM, oreIndex),
                                               sourceChunkX, 0, sourceChunkZ, stepIndex, 0, 1) == 0
                                ? -1
                                : 1;
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

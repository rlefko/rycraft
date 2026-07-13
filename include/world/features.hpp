#pragma once

#include "world/chunk.hpp"
#include "world/structures.hpp"

#include <cstdint>

class ChunkGenerator;
struct GenScratch;

// ---------------------------------------------------------------------------
// Features — trees and flora.
//
// Trees use the same neighborhood-deterministic pattern as ores: every chunk
// re-rolls the 12 tree attempts of its 3×3 neighborhood from each source
// chunk's world-coordinate hash and emits only the blocks inside itself, so
// canopies span chunk borders seamlessly. Placement reads ONLY the pure
// terrain queries (surfaceYAt is post-cave, so trees never float over cave
// mouths). Each accepted attempt rebuilds its whole tree from a private
// RNG stream, which keeps the per-attempt draw count on the chunk stream
// fixed — the RNG-order rule that makes skipped attempts free.
//
// Flora (grass tufts, flowers, mushrooms, cacti, dead bushes, reeds) is
// strictly chunk-local and placed on the chunk's real final blocks.
// ---------------------------------------------------------------------------

class FeaturePlacer {
public:
    explicit FeaturePlacer(uint32_t worldSeed);

    void placeTrees(Chunk& chunk, const ChunkGenerator& gen, const StructurePlacer& structures,
                    GenScratch& scratch) const;
    void placeFlora(Chunk& chunk) const;

private:
    uint32_t seed_;
};

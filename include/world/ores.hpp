#pragma once

#include "world/chunk.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Ore placement — neighborhood-deterministic random-walk blobs.
//
// When generating chunk C, the placer re-rolls the ore attempts of every
// chunk in the 3×3 neighborhood from each chunk's own coordinate hash and
// writes only the blocks that land inside C. Both sides of a chunk border
// therefore agree on every vein without either chunk reading the other.
// Blob walks are capped at 12 steps, which is what makes radius 1 enough.
// ---------------------------------------------------------------------------

class OrePlacer {
public:
    explicit OrePlacer(uint32_t worldSeed);

    void place(Chunk& chunk) const;

private:
    uint32_t seed_;
};

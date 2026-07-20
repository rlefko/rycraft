#pragma once

#include "common/counter_rng.hpp"
#include "world/chunk.hpp"

#include <cstdint>

class ChunkGenerator;
struct GenScratch;

// ---------------------------------------------------------------------------
// Structures — one deterministic attempt per 8×8-chunk region.
//
// A region rolls its structure (anchor, kind, rotation) purely from the
// region hash, then validates against the pure terrain queries (flatness,
// dry land). Every chunk the footprint touches recomputes the identical
// placement and emits only its own blocks, so buildings span chunk borders
// with no generation-order dependency. Footprints are capped so a radius-1
// chunk neighborhood always sees every structure that can reach it.
// ---------------------------------------------------------------------------

inline constexpr int STRUCTURE_REGION_CHUNKS = 8;

enum class StructureKind : uint8_t { RUIN = 0, WELL = 1, HOUSE = 2 };

struct StructurePlacement {
    bool valid = false;
    StructureKind kind = StructureKind::RUIN;
    int rotation = 0;    // quarter turns
    int64_t anchorX = 0; // world coords of the footprint center
    int64_t anchorZ = 0;
    int floorY = 0;
    int halfX = 0; // rotated half-extents of the footprint
    int halfZ = 0;
};

class StructurePlacer {
public:
    // A disabled placer emits nothing and reserves no footprints, so trees
    // fill former structure sites deterministically for that toggle value.
    explicit StructurePlacer(uint32_t worldSeed, bool enabled = true);

    // Emit every structure that intersects this chunk.
    void place(Chunk& chunk, const ChunkGenerator& gen, GenScratch& scratch) const;

    // Pure, cached placement for one region (used by tree rejection too).
    const StructurePlacement& regionPlacement(int64_t regionX, int64_t regionZ,
                                              const ChunkGenerator& gen, GenScratch& scratch) const;

    // True when (x, z) lies inside any structure footprint (plus margin)
    // that could reach the chunk neighborhood of (chunkX, chunkZ).
    bool insideStructure(int64_t x, int64_t z, int64_t chunkX, int64_t chunkZ, int margin) const;

private:
    CounterRng random_;
    bool enabled_ = true;
};

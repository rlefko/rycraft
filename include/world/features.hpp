#pragma once

#include "common/counter_rng.hpp"
#include "world/chunk.hpp"
#include "world/structures.hpp"

#include <cstdint>
#include <vector>

class ChunkGenerator;
struct GenScratch;

namespace feature_generation {

// These are reconstruction bounds, not placement density controls. Column
// planning uses the same values as cube emission so a plant rooted in a
// neighboring column cannot be clipped by a sparse-section gate.
inline constexpr int TREE_CELL_EDGE = 8;
inline constexpr int TREE_MAXIMUM_HORIZONTAL_REACH = 6;
inline constexpr int TREE_MINIMUM_VERTICAL_OFFSET = -2;
inline constexpr int TREE_MAXIMUM_VERTICAL_OFFSET = 20;

// Density can expose a surface above or below its climate terrain sample.
// Rejecting larger departures also prevents trees from growing on deep cave
// floors and gives sparse column planning a finite, exact vertical envelope.
inline constexpr int TREE_MAXIMUM_SURFACE_DEVIATION = 24;

} // namespace feature_generation

// Large plants come from world-aligned candidate cells. A candidate is
// accepted only when its counter-based priority wins against every candidate
// inside the larger species spacing radius. Each cube queries an expanded
// footprint and clips the resulting plant locally, so trunks, roots, branches,
// canopies, and fallen logs cross cube faces without shared mutable state.
// Climate and soil traits gate every candidate. Biome transition weights are
// dithered from world coordinates for both tree forms and smaller flora.

// Compact, coordinate-pure description of one accepted large-plant canopy.
// The anchor is the same world-space root used by cube emission. Canopy
// offsets and radius enclose every emitted foliage candidate, including bent
// and branching species, so far terrain can build conservative impostors
// without generating cubes or maintaining a second forest distribution.
struct FarCanopy {
    int64_t x = 0;
    int64_t z = 0;
    int32_t baseY = 0;
    int32_t topY = 0;
    int32_t canopyMinimumY = 0;
    int32_t canopyMaximumY = 0;
    int8_t canopyOffsetX = 0;
    int8_t canopyOffsetZ = 0;
    uint8_t canopyRadius = 0;
    BlockType logBlock = BlockType::AIR;
    BlockType leafBlock = BlockType::AIR;
    uint64_t anchorId = 0;
    bool aggregate = false;

    bool operator==(const FarCanopy&) const = default;
};

class FeaturePlacer {
public:
    explicit FeaturePlacer(uint32_t worldSeed);

    void placeTrees(Chunk& chunk, const ChunkGenerator& gen, const StructurePlacer& structures,
                    GenScratch& scratch) const;
    void placeFlora(Chunk& chunk, const ChunkGenerator& gen, GenScratch& scratch) const;
    // Returns every accepted canopy whose exact foliage bounds intersect the
    // half-open XZ rectangle [minimum, maximum). Anchors just outside the
    // rectangle are included when their canopies cross an edge.
    std::vector<FarCanopy> collectFarCanopies(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                              int64_t maximumZ, const ChunkGenerator& gen,
                                              const StructurePlacer& structures,
                                              GenScratch& scratch) const;
    // Coarse LODs represent deterministic canopy cover with globally anchored
    // clusters. Exact accepted tree anchors remain in the two near tiers,
    // while distant tiers avoid constructing hundreds of cube-density column
    // plans per tile.
    std::vector<FarCanopy> collectFarCanopyClusters(int64_t minimumX, int64_t minimumZ,
                                                    int64_t maximumX, int64_t maximumZ, int lodStep,
                                                    const ChunkGenerator& gen) const;

private:
    CounterRng random_;
};

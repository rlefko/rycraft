#pragma once

#include "common/counter_rng.hpp"
#include "world/chunk.hpp"
#include "world/structures.hpp"

#include <cstdint>
#include <vector>

class ChunkGenerator;
struct GenScratch;

namespace worldgen {
struct SurfaceSample;
}

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

enum class TreeSpecies : uint8_t {
    OAK,
    LARGE_OAK,
    BIRCH,
    SPRUCE,
    ACACIA,
    JUNGLE,
    MANGROVE,
    PALM,
    WILLOW,
    ALPINE_SCRUB,
    FALLEN_LOG,
    COUNT,
};

// Shared ecology result used by exact candidates and aggregate far crowns.
// The supplied ground Y is the actual solid surface block, not a water plane.
// Flooded roots are accepted only for explicitly adapted species in shallow,
// supported habitat. Suitability remains continuous so climate and biome
// transitions change forest composition without categorical seams.
struct TreeHabitatEvaluation {
    double suitability = 0.0;
    int waterDepthBlocks = 0;
    int spacing = 0;
    bool submerged = false;
    bool allowed = false;
};

TreeHabitatEvaluation evaluateTreeHabitat(TreeSpecies species,
                                          const worldgen::SurfaceSample& surface,
                                          int groundSurfaceY);
double treeCoverDensity(const worldgen::SurfaceSample& surface);

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
// and branching species. Far terrain keeps that envelope fixed while using
// the authoritative species plus its log and leaf materials to build a
// compact, layered silhouette.
// An aggregate entry is one small crown in a coarse forest cell, never a box
// covering the complete cell.
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
    feature_generation::TreeSpecies species = feature_generation::TreeSpecies::OAK;
    // Signed horizontal form direction for bent trunks, asymmetric crowns,
    // palm lean, and fallen logs. Form extent is the block length of a
    // horizontal ground form and zero for ordinary standing trees.
    int8_t formX = 0;
    int8_t formZ = 0;
    uint8_t formExtent = 0;

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
    // Near far-terrain canopies share the exact accepted 8-block anchor set,
    // species, and shape without constructing a complete ColumnPlan for each
    // potential root. The mesher grounds each returned silhouette against its
    // own terrain sample; exact cube emission resolves the same anchor to the
    // density surface only after local-priority acceptance.
    std::vector<FarCanopy> collectFarCanopyAnchors(int64_t minimumX, int64_t minimumZ,
                                                   int64_t maximumX, int64_t maximumZ,
                                                   const ChunkGenerator& gen,
                                                   const StructurePlacer& structures,
                                                   GenScratch& scratch) const;
    // Far LODs represent deterministic canopy cover with globally anchored
    // clusters. Every tier evaluates one fixed candidate set and retains a
    // strict subset of the preceding tier, so rendering never makes a second
    // thinning decision. Exact anchors remain the block-resolution authority.
    std::vector<FarCanopy> collectFarCanopyClusters(int64_t minimumX, int64_t minimumZ,
                                                    int64_t maximumX, int64_t maximumZ, int lodStep,
                                                    const ChunkGenerator& gen) const;

private:
    CounterRng random_;
};

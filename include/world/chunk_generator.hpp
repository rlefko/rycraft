#pragma once

#include "world/chunk.hpp"
#include "world/climate.hpp"
#include "world/density_field.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// ChunkGenerator — the world generation façade.
//
// generate() fills one chunk completely and independently: climate lattice →
// density lattice → trilinear block fill → liquids and surface materials.
// The pure single-column queries (baseHeightAt / biomeAt / surfaceYAt)
// reproduce exactly what generate() would put at any world column WITHOUT
// generating the chunk — that contract is what cross-chunk features (trees,
// structures) and gameplay queries build on. They share one lattice +
// interpolation code path with the bulk fill; see density_field.hpp for why
// there must never be a second one.
// ---------------------------------------------------------------------------

// Reusable cache of lattice columns for a burst of related queries (one
// chunk generation, or a stream of gameplay lookups). Owner-tagged so a
// thread-local scratch can't leak columns across generators/seeds.
struct GenScratch {
    const void* owner = nullptr;
    std::unordered_map<uint64_t, ColumnShape> shapes;
    std::unordered_map<uint64_t, std::vector<double>> densityColumns;

    void reset(const void* newOwner) {
        owner = newOwner;
        shapes.clear();
        densityColumns.clear();
    }
};

class ChunkGenerator {
public:
    explicit ChunkGenerator(uint32_t worldSeed);

    void generate(Chunk& chunk) const;

    // ---- Pure single-column queries (world coordinates) ----
    // Interpolated pre-cave surface height H.
    double baseHeightAt(int x, int z, GenScratch& scratch) const;
    Biome biomeAt(int x, int z, GenScratch& scratch) const;
    // Topmost solid block the density fill produces (post-cave/ravine,
    // pre-decoration). Matches Chunk::heightMap right after generate().
    int surfaceYAt(int x, int z, GenScratch& scratch) const;
    // Bilinearly interpolated shape for one block column.
    ColumnShape columnShapeAt(int x, int z, GenScratch& scratch) const;

    // Convenience overloads with a bounded thread-local scratch, for
    // occasional gameplay queries (spawner, particles).
    double baseHeightAt(int x, int z) const;
    Biome biomeAt(int x, int z) const;

    uint32_t seed() const { return seed_; }

private:
    uint32_t seed_;
    uint32_t bedrockSeed_;
    uint32_t surfaceSeed_;
    ClimateSampler climate_;
    DensityField density_;

    const ColumnShape& latticeShape(int lx, int lz, GenScratch& scratch) const;
    const std::vector<double>& latticeDensityColumn(int lx, int lz, GenScratch& scratch) const;
    void applyColumnSurface(Chunk& chunk, int lx, int lz, const ColumnShape& shape,
                            Biome biome) const;
    GenScratch& threadScratch() const;
};

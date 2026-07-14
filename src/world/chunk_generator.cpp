#include "world/chunk_generator.hpp"

#include "world/chunk_pos.hpp"
#include "world/gen_seeds.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr int LAVA_LEVEL = 10;     // sealed cave air at or below becomes lava
constexpr int WORLD_AIR_CAP = 250; // top of the world stays open
constexpr int LATTICE_LEVELS = CHUNK_HEIGHT / LATTICE_Y + 1;

// Lattice columns are keyed by their packed world coordinates — through
// ChunkPos::packed(), THE bit layout for xz keys.
uint64_t latticeKey(int lx, int lz) {
    return ChunkPos{lx, lz}.packed();
}

// Floor to the containing lattice column (works for negatives: two's
// complement AND clears the low bits downward).
int latticeFloor(int v) {
    return v & ~(LATTICE_XZ - 1);
}

// One fixed op order for voxel density — see density_field.hpp.
double voxelDensity(const std::vector<double>& c00, const std::vector<double>& c10,
                    const std::vector<double>& c01, const std::vector<double>& c11, double fx,
                    double fz, int y) {
    int level = y / LATTICE_Y;
    double fy = static_cast<double>(y - level * LATTICE_Y) / LATTICE_Y;
    double below = bilerpDensity(c00[level], c10[level], c01[level], c11[level], fx, fz);
    double above =
        bilerpDensity(c00[level + 1], c10[level + 1], c01[level + 1], c11[level + 1], fx, fz);
    return lerpDensity(below, above, fy);
}

bool solidAt(double density, int y) {
    if (y <= 2) return true;
    if (y >= WORLD_AIR_CAP) return false;
    return density > 0.0;
}

// Surface material sets. Submerged columns get a sea/river bed instead of
// their biome's dry surface.
BlockType surfaceBlockFor(Biome biome, int y, bool submerged, bool gravelPatch) {
    if (submerged) {
        return gravelPatch ? BlockType::GRAVEL : BlockType::SAND;
    }
    if (y >= SNOW_LINE) return BlockType::SNOW;
    switch (biome) {
        case Biome::DESERT:
        case Biome::BEACH:
            return BlockType::SAND;
        case Biome::ICE_SPIKES:
            return BlockType::SNOW;
        case Biome::EXTREME_HILLS:
            return BlockType::STONE;
        default:
            return BlockType::GRASS;
    }
}

BlockType subsurfaceBlockFor(Biome biome, bool submerged, bool gravelPatch) {
    if (submerged) {
        return gravelPatch ? BlockType::GRAVEL : BlockType::SAND;
    }
    switch (biome) {
        case Biome::DESERT:
        case Biome::BEACH:
            return BlockType::SAND;
        case Biome::EXTREME_HILLS:
            return BlockType::STONE;
        default:
            return BlockType::DIRT;
    }
}

} // namespace

ChunkGenerator::ChunkGenerator(uint32_t worldSeed)
    : seed_(worldSeed)
    , bedrockSeed_(genseed::subSeed(worldSeed, genseed::BEDROCK))
    , surfaceSeed_(genseed::subSeed(worldSeed, genseed::SURFACE))
    , climate_(worldSeed)
    , density_(worldSeed)
    , ores_(worldSeed)
    , structures_(worldSeed)
    , features_(worldSeed) {}

const ColumnShape& ChunkGenerator::latticeShape(int lx, int lz, GenScratch& scratch) const {
    uint64_t key = latticeKey(lx, lz);
    auto it = scratch.shapes.find(key);
    if (it != scratch.shapes.end()) return it->second;
    return scratch.shapes
        .emplace(key, climate_.shapeColumn(static_cast<double>(lx), static_cast<double>(lz)))
        .first->second;
}

const std::vector<double>& ChunkGenerator::latticeDensityColumn(int lx, int lz,
                                                                GenScratch& scratch) const {
    uint64_t key = latticeKey(lx, lz);
    auto it = scratch.densityColumns.find(key);
    if (it != scratch.densityColumns.end()) return it->second;

    const ColumnShape& shape = latticeShape(lx, lz, scratch);
    std::vector<double> column(LATTICE_LEVELS);
    // Above height + detail there is provably no terrain: every density
    // component is ≥ 4 blocks negative there, so a flat cap keeps the
    // interpolated sign identical while skipping the noise evals.
    double yCap = shape.height + shape.detailAmp + 5.0;
    for (int level = 0; level < LATTICE_LEVELS; ++level) {
        double y = static_cast<double>(level * LATTICE_Y);
        column[level] =
            y > yCap ? -DENSITY_CAP
                     : density_.density(static_cast<double>(lx), y, static_cast<double>(lz), shape);
    }
    return scratch.densityColumns.emplace(key, std::move(column)).first->second;
}

ColumnShape ChunkGenerator::columnShapeAt(int x, int z, GenScratch& scratch) const {
    int lx0 = latticeFloor(x);
    int lz0 = latticeFloor(z);
    double fx = static_cast<double>(x - lx0) / LATTICE_XZ;
    double fz = static_cast<double>(z - lz0) / LATTICE_XZ;
    const ColumnShape& s00 = latticeShape(lx0, lz0, scratch);
    const ColumnShape& s10 = latticeShape(lx0 + LATTICE_XZ, lz0, scratch);
    const ColumnShape& s01 = latticeShape(lx0, lz0 + LATTICE_XZ, scratch);
    const ColumnShape& s11 = latticeShape(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, scratch);

    auto blend = [&](auto field) {
        return bilerpDensity(field(s00), field(s10), field(s01), field(s11), fx, fz);
    };
    ColumnShape out;
    out.climate.continentalness =
        blend([](const ColumnShape& s) { return s.climate.continentalness; });
    out.climate.erosion = blend([](const ColumnShape& s) { return s.climate.erosion; });
    out.climate.ridges = blend([](const ColumnShape& s) { return s.climate.ridges; });
    out.climate.temperature = blend([](const ColumnShape& s) { return s.climate.temperature; });
    out.climate.humidity = blend([](const ColumnShape& s) { return s.climate.humidity; });
    out.height = blend([](const ColumnShape& s) { return s.height; });
    out.detailAmp = blend([](const ColumnShape& s) { return s.detailAmp; });
    out.entrance = blend([](const ColumnShape& s) { return s.entrance; });
    out.riverCut = blend([](const ColumnShape& s) { return s.riverCut; });
    out.ravineEdge = blend([](const ColumnShape& s) { return s.ravineEdge; });
    out.ravineFloor = blend([](const ColumnShape& s) { return s.ravineFloor; });
    return out;
}

double ChunkGenerator::baseHeightAt(int x, int z, GenScratch& scratch) const {
    return columnShapeAt(x, z, scratch).height;
}

Biome ChunkGenerator::biomeAt(int x, int z, GenScratch& scratch) const {
    return ClimateSampler::selectBiome(columnShapeAt(x, z, scratch));
}

int ChunkGenerator::surfaceYAt(int x, int z, GenScratch& scratch) const {
    int lx0 = latticeFloor(x);
    int lz0 = latticeFloor(z);
    const std::vector<double>& c00 = latticeDensityColumn(lx0, lz0, scratch);
    const std::vector<double>& c10 = latticeDensityColumn(lx0 + LATTICE_XZ, lz0, scratch);
    const std::vector<double>& c01 = latticeDensityColumn(lx0, lz0 + LATTICE_XZ, scratch);
    const std::vector<double>& c11 =
        latticeDensityColumn(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, scratch);
    double fx = static_cast<double>(x - lx0) / LATTICE_XZ;
    double fz = static_cast<double>(z - lz0) / LATTICE_XZ;

    ColumnShape shape = columnShapeAt(x, z, scratch);
    int yStart =
        std::min(WORLD_AIR_CAP - 1, static_cast<int>(shape.height + shape.detailAmp + 4.0));
    for (int y = yStart; y > 2; --y) {
        if (solidAt(voxelDensity(c00, c10, c01, c11, fx, fz, y), y)) return y;
    }
    return 2;
}

GenScratch& ChunkGenerator::threadScratch() const {
    thread_local GenScratch scratch;
    // Re-key on generator change (tests spin up several worlds per thread)
    // and bound the cache so long play sessions can't grow it forever.
    if (scratch.owner != this || scratch.shapes.size() > 4096) {
        scratch.reset(this);
    }
    return scratch;
}

double ChunkGenerator::baseHeightAt(int x, int z) const {
    return baseHeightAt(x, z, threadScratch());
}

Biome ChunkGenerator::biomeAt(int x, int z) const {
    return biomeAt(x, z, threadScratch());
}

// ---------------------------------------------------------------------------
// Surface pass: one top-down sweep per column over the freshly filled
// blocks. Open-to-sky air below sea level floods (ice-capped when frozen),
// sealed cave air at the bottom becomes lava, and every air→solid
// transition that sees the sky gets its biome's surface + subsoil. Working
// top-down over an already-contiguous column makes a floating-surface gap
// (the old off-by-one) impossible by construction.
// ---------------------------------------------------------------------------
void ChunkGenerator::applyColumnSurface(Chunk& chunk, int lx, int lz, const ColumnShape& shape,
                                        Biome biome) const {
    int wx = chunk.chunkX * CHUNK_WIDTH + lx;
    int wz = chunk.chunkZ * CHUNK_DEPTH + lz;
    uint64_t columnHash = hashCoords(wx, wz, surfaceSeed_);
    int subsoilDepth = 2 + static_cast<int>(columnHash % 3);
    bool gravelPatch = ((columnHash >> 8) % 3) == 0;
    bool frozen = ClimateSampler::isFrozen(shape);
    bool desertLike = biome == Biome::DESERT || biome == Biome::BEACH;

    bool openToSky = true;
    bool prevAirLike = true; // above the world counts as air
    int topSolid = -1;
    int subsoilLeft = 0;
    int sandstoneLeft = 0;
    BlockType subsoilBlock = BlockType::DIRT;

    for (int y = CHUNK_HEIGHT - 1; y >= 0; --y) {
        BlockType block = chunk.getBlock(lx, y, lz);
        if (block == BlockType::AIR) {
            if (openToSky && y < SEA_LEVEL) {
                chunk.setBlock(lx, y, lz,
                               (frozen && y == SEA_LEVEL - 1) ? BlockType::ICE : BlockType::WATER);
            } else if (!openToSky && y <= LAVA_LEVEL) {
                chunk.setBlock(lx, y, lz, BlockType::LAVA);
            }
            prevAirLike = true;
            subsoilLeft = 0;
            sandstoneLeft = 0;
            continue;
        }

        if (topSolid < 0) topSolid = y;

        if (block == BlockType::STONE) {
            if (prevAirLike && openToSky) {
                // The real surface: everything under overhangs or inside
                // caves (openToSky already false) stays stone.
                bool submerged = y < SEA_LEVEL - 1;
                chunk.setBlock(lx, y, lz, surfaceBlockFor(biome, y, submerged, gravelPatch));
                subsoilBlock = subsurfaceBlockFor(biome, submerged, gravelPatch);
                subsoilLeft = subsoilDepth;
                sandstoneLeft = desertLike && !submerged ? 3 : 0;
            } else if (subsoilLeft > 0) {
                chunk.setBlock(lx, y, lz, subsoilBlock);
                if (--subsoilLeft == 0 && sandstoneLeft > 0) {
                    subsoilBlock = BlockType::SANDSTONE;
                    subsoilLeft = sandstoneLeft;
                    sandstoneLeft = 0;
                }
            }
        }

        openToSky = false;
        prevAirLike = false;
    }

    chunk.heightMap[lx + lz * CHUNK_WIDTH] = topSolid >= 0 ? topSolid : 0;
}

void ChunkGenerator::generate(Chunk& chunk) const {
    GenScratch scratch;
    scratch.reset(this);
    const int baseX = chunk.chunkX * CHUNK_WIDTH;
    const int baseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
        for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
            int wx = baseX + lx;
            int wz = baseZ + lz;
            int lx0 = latticeFloor(wx);
            int lz0 = latticeFloor(wz);
            // unordered_map keeps references valid across later inserts
            const std::vector<double>& c00 = latticeDensityColumn(lx0, lz0, scratch);
            const std::vector<double>& c10 = latticeDensityColumn(lx0 + LATTICE_XZ, lz0, scratch);
            const std::vector<double>& c01 = latticeDensityColumn(lx0, lz0 + LATTICE_XZ, scratch);
            const std::vector<double>& c11 =
                latticeDensityColumn(lx0 + LATTICE_XZ, lz0 + LATTICE_XZ, scratch);
            double fx = static_cast<double>(wx - lx0) / LATTICE_XZ;
            double fz = static_cast<double>(wz - lz0) / LATTICE_XZ;

            ColumnShape shape = columnShapeAt(wx, wz, scratch);
            Biome biome = ClimateSampler::selectBiome(shape);
            chunk.biomes[lx + lz * CHUNK_WIDTH] = biome;

            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                BlockType block = BlockType::AIR;
                if (y <= 1) {
                    block = BlockType::BEDROCK;
                } else if (y == 2) {
                    // Dithered floor: half bedrock, half stone
                    block = (hashCoords(wx, wz, bedrockSeed_) & 1) ? BlockType::BEDROCK
                                                                   : BlockType::STONE;
                } else if (solidAt(voxelDensity(c00, c10, c01, c11, fx, fz, y), y)) {
                    block = BlockType::STONE;
                }
                chunk.setBlock(lx, y, lz, block);
            }

            applyColumnSurface(chunk, lx, lz, shape, biome);
        }
    }

    // Decoration: ores → structures → trees → flora. Structures go before
    // trees so tree placement can reject anchors inside their footprints;
    // flora is last so it reads the chunk's real final surface.
    ores_.place(chunk);
    structures_.place(chunk, *this, scratch);
    features_.placeTrees(chunk, *this, structures_, scratch);
    features_.placeFlora(chunk);

    chunk.generated = true;
    chunk.needsMeshUpdate = true;
}

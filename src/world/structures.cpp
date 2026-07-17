#include "world/structures.hpp"

#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"

#include <algorithm>
#include <array>

namespace {

constexpr uint64_t STRUCTURE_ANCHOR_CHUNK_X_STREAM = 0x5354525F43484E58ULL;
constexpr uint64_t STRUCTURE_ANCHOR_CHUNK_Z_STREAM = 0x5354525F43484E5AULL;
constexpr uint64_t STRUCTURE_ANCHOR_LOCAL_X_STREAM = 0x5354525F4C4F4358ULL;
constexpr uint64_t STRUCTURE_ANCHOR_LOCAL_Z_STREAM = 0x5354525F4C4F435AULL;
constexpr uint64_t STRUCTURE_KIND_STREAM = 0x5354525F4B494E44ULL;
constexpr uint64_t STRUCTURE_ROTATION_STREAM = 0x5354525F524F544EULL;
constexpr uint64_t RUIN_DECAY_STREAM = 0x5354525F44454359ULL;

int64_t floorDiv(int64_t a, int64_t b) {
    return world_coord::floorDiv(a, b);
}

// Rotate footprint-local (u, v) by quarter turns around the anchor.
void rotate(int u, int v, int rotation, int& outDx, int& outDz) {
    switch (rotation & 3) {
        case 0:
            outDx = u;
            outDz = v;
            break;
        case 1:
            outDx = v;
            outDz = -u;
            break;
        case 2:
            outDx = -u;
            outDz = -v;
            break;
        default:
            outDx = -v;
            outDz = u;
            break;
    }
}

// Unrotated half-extents per kind (rotation swaps them).
void kindExtents(StructureKind kind, int& halfU, int& halfV) {
    switch (kind) {
        case StructureKind::HOUSE:
            halfU = 3;
            halfV = 3;
            break;
        case StructureKind::WELL:
            halfU = 2;
            halfV = 2;
            break;
        case StructureKind::RUIN:
            halfU = 3;
            halfV = 4;
            break;
    }
}

StructurePlacement unvalidatedPlacement(const CounterRng& random, int64_t regionX,
                                        int64_t regionZ) {
    const int64_t anchorChunkX =
        regionX * STRUCTURE_REGION_CHUNKS +
        random.uniformInt(STRUCTURE_ANCHOR_CHUNK_X_STREAM, regionX, 0, regionZ, 0, 1, 6);
    const int64_t anchorChunkZ =
        regionZ * STRUCTURE_REGION_CHUNKS +
        random.uniformInt(STRUCTURE_ANCHOR_CHUNK_Z_STREAM, regionX, 0, regionZ, 0, 1, 6);
    const int anchorLocalX =
        random.uniformInt(STRUCTURE_ANCHOR_LOCAL_X_STREAM, regionX, 0, regionZ, 0, 2, 13);
    const int anchorLocalZ =
        random.uniformInt(STRUCTURE_ANCHOR_LOCAL_Z_STREAM, regionX, 0, regionZ, 0, 2, 13);
    const double kindRoll = random.uniform01(STRUCTURE_KIND_STREAM, regionX, 0, regionZ);
    const int rotation = random.uniformInt(STRUCTURE_ROTATION_STREAM, regionX, 0, regionZ, 0, 0, 3);

    StructurePlacement placement;
    placement.kind = kindRoll < 0.4   ? StructureKind::RUIN
                     : kindRoll < 0.7 ? StructureKind::WELL
                                      : StructureKind::HOUSE;
    placement.rotation = rotation;
    placement.anchorX = anchorChunkX * CHUNK_WIDTH + anchorLocalX;
    placement.anchorZ = anchorChunkZ * CHUNK_DEPTH + anchorLocalZ;
    int halfU = 0;
    int halfV = 0;
    kindExtents(placement.kind, halfU, halfV);
    placement.halfX = (rotation & 1) != 0 ? halfV : halfU;
    placement.halfZ = (rotation & 1) != 0 ? halfU : halfV;
    return placement;
}

bool isLandBiome(Biome biome) {
    return biome != Biome::OCEAN && biome != Biome::DEEP_OCEAN && biome != Biome::FROZEN_OCEAN &&
           biome != Biome::RIVER && biome != Biome::COUNT;
}

// A per-chunk writer that silently drops blocks outside the chunk — the
// filter that makes footprints span borders safely.
struct ChunkWriter {
    Chunk& chunk;
    int64_t baseX;
    int baseY;
    int64_t baseZ;

    void force(int64_t x, int y, int64_t z, BlockType block) const {
        int lx = static_cast<int>(x - baseX);
        int ly = y - baseY;
        int lz = static_cast<int>(z - baseZ);
        if (lx >= 0 && lx < CHUNK_WIDTH && ly >= 0 && ly < CHUNK_HEIGHT && lz >= 0 &&
            lz < CHUNK_DEPTH) {
            chunk.setBlock(lx, ly, lz, block);
        }
    }
};

} // namespace

StructurePlacer::StructurePlacer(uint32_t worldSeed) : random_(worldSeed) {}

const StructurePlacement& StructurePlacer::regionPlacement(int64_t regionX, int64_t regionZ,
                                                           const ChunkGenerator& gen,
                                                           GenScratch& scratch) const {
    const ColumnPos key{regionX, regionZ};
    auto it = scratch.structurePlacements.find(key);
    if (it != scratch.structurePlacements.end()) return it->second;

    // The anchor margins (chunk offset 1..6 in the region, block offset
    // 2..13 in the chunk) bound footprint spill to under one chunk, which lets a
    // radius-1 chunk neighborhood see every structure that can reach it.
    StructurePlacement placement = unvalidatedPlacement(random_, regionX, regionZ);

    // Terrain adaptation: probe the corners + center, require dry and
    // near-flat ground (ruins tolerate rougher sites, half-buried).
    std::array<int, 5> probes{};
    probes[0] = gen.surfaceYAt(placement.anchorX, placement.anchorZ, scratch);
    probes[1] = gen.surfaceYAt(placement.anchorX - placement.halfX,
                               placement.anchorZ - placement.halfZ, scratch);
    probes[2] = gen.surfaceYAt(placement.anchorX + placement.halfX,
                               placement.anchorZ - placement.halfZ, scratch);
    probes[3] = gen.surfaceYAt(placement.anchorX - placement.halfX,
                               placement.anchorZ + placement.halfZ, scratch);
    probes[4] = gen.surfaceYAt(placement.anchorX + placement.halfX,
                               placement.anchorZ + placement.halfZ, scratch);
    std::sort(probes.begin(), probes.end());
    int spread = probes[4] - probes[0];
    int maxSpread = placement.kind == StructureKind::RUIN ? 5 : 2;

    placement.floorY = probes[2]; // median
    // Dry land (above sea level), and not on extreme peaks where the
    // foundation cap can't bridge the terrain
    placement.valid = spread <= maxSpread && probes[0] >= SEA_LEVEL && placement.floorY < 180 &&
                      isLandBiome(gen.biomeAt(placement.anchorX, placement.anchorZ, scratch));

    return scratch.structurePlacements.emplace(key, placement).first->second;
}

bool StructurePlacer::insideStructure(int64_t x, int64_t z, int64_t chunkX, int64_t chunkZ,
                                      int margin) const {
    int64_t minRegionX = floorDiv(chunkX - 1, STRUCTURE_REGION_CHUNKS);
    int64_t maxRegionX = floorDiv(chunkX + 1, STRUCTURE_REGION_CHUNKS);
    int64_t minRegionZ = floorDiv(chunkZ - 1, STRUCTURE_REGION_CHUNKS);
    int64_t maxRegionZ = floorDiv(chunkZ + 1, STRUCTURE_REGION_CHUNKS);
    for (int64_t rz = minRegionZ; rz <= maxRegionZ; ++rz) {
        for (int64_t rx = minRegionX; rx <= maxRegionX; ++rx) {
            const StructurePlacement candidate = unvalidatedPlacement(random_, rx, rz);
            if (std::abs(x - candidate.anchorX) > candidate.halfX + margin ||
                std::abs(z - candidate.anchorZ) > candidate.halfZ + margin) {
                continue;
            }
            // Reserve every deterministic candidate footprint. Whether the
            // terrain probe ultimately emits the structure does not alter
            // large-plant ownership, so exact cubes and far canopy pages can
            // agree without constructing distant ColumnPlans.
            return true;
        }
    }
    return false;
}

void StructurePlacer::place(Chunk& chunk, const ChunkGenerator& gen, GenScratch& scratch) const {
    ChunkWriter out{chunk, chunk.chunkX * CHUNK_WIDTH, chunk.chunkY * CHUNK_HEIGHT,
                    chunk.chunkZ * CHUNK_DEPTH};

    int64_t minRegionX = floorDiv(chunk.chunkX - 1, STRUCTURE_REGION_CHUNKS);
    int64_t maxRegionX = floorDiv(chunk.chunkX + 1, STRUCTURE_REGION_CHUNKS);
    int64_t minRegionZ = floorDiv(chunk.chunkZ - 1, STRUCTURE_REGION_CHUNKS);
    int64_t maxRegionZ = floorDiv(chunk.chunkZ + 1, STRUCTURE_REGION_CHUNKS);

    for (int64_t rz = minRegionZ; rz <= maxRegionZ; ++rz) {
        for (int64_t rx = minRegionX; rx <= maxRegionX; ++rx) {
            const StructurePlacement& p = regionPlacement(rx, rz, gen, scratch);
            if (!p.valid) continue;
            // Skip footprints that cannot touch this chunk
            if (p.anchorX + p.halfX < out.baseX - 1 ||
                p.anchorX - p.halfX > out.baseX + CHUNK_WIDTH ||
                p.anchorZ + p.halfZ < out.baseZ - 1 ||
                p.anchorZ - p.halfZ > out.baseZ + CHUNK_DEPTH) {
                continue;
            }

            bool desert = gen.biomeAt(p.anchorX, p.anchorZ, scratch) == Biome::DESERT;
            BlockType stoneBlock = desert ? BlockType::SANDSTONE : BlockType::COBBLESTONE;
            BlockType mossyBlock = desert ? BlockType::SANDSTONE : BlockType::MOSSY_COBBLESTONE;

            int halfU = 0, halfV = 0;
            kindExtents(p.kind, halfU, halfV);

            // Foundation: fill from just under the floor down to the real
            // terrain at every footprint column (capped — a probe gate of
            // ≤2/≤5 blocks of spread keeps this shallow in practice).
            for (int v = -halfV; v <= halfV; ++v) {
                for (int u = -halfU; u <= halfU; ++u) {
                    int dx = 0, dz = 0;
                    rotate(u, v, p.rotation, dx, dz);
                    int64_t x = p.anchorX + dx;
                    int64_t z = p.anchorZ + dz;
                    int ground = gen.surfaceYAt(x, z, scratch);
                    for (int y = p.floorY - 1; y > ground && y > p.floorY - 9; --y) {
                        out.force(x, y, z, stoneBlock);
                    }
                }
            }

            switch (p.kind) {
                case StructureKind::HOUSE: {
                    for (int v = -3; v <= 3; ++v) {
                        for (int u = -3; u <= 3; ++u) {
                            int dx = 0, dz = 0;
                            rotate(u, v, p.rotation, dx, dz);
                            int64_t x = p.anchorX + dx;
                            int64_t z = p.anchorZ + dz;

                            out.force(x, p.floorY, z, BlockType::PLANKS);

                            bool perimeter = std::abs(u) == 3 || std::abs(v) == 3;
                            for (int dy = 1; dy <= 3; ++dy) {
                                BlockType block = BlockType::AIR; // carved interior
                                if (perimeter) {
                                    bool corner = std::abs(u) == 3 && std::abs(v) == 3;
                                    bool door = v == -3 && u == 0 && dy <= 2;
                                    bool window = dy == 2 && ((std::abs(u) == 3 && v == 0) ||
                                                              (v == 3 && u == 0));
                                    block = corner   ? stoneBlock
                                            : door   ? BlockType::AIR
                                            : window ? BlockType::GLASS
                                                     : BlockType::PLANKS;
                                }
                                out.force(x, p.floorY + dy, z, block);
                            }

                            // Stepped pyramid roof
                            out.force(x, p.floorY + 4, z, BlockType::PLANKS);
                            if (std::abs(u) <= 2 && std::abs(v) <= 2)
                                out.force(x, p.floorY + 5, z, BlockType::PLANKS);
                            if (std::abs(u) <= 1 && std::abs(v) <= 1)
                                out.force(x, p.floorY + 6, z, BlockType::PLANKS);
                        }
                    }
                    break;
                }

                case StructureKind::WELL: {
                    for (int v = -2; v <= 2; ++v) {
                        for (int u = -2; u <= 2; ++u) {
                            int dx = 0, dz = 0;
                            rotate(u, v, p.rotation, dx, dz);
                            int64_t x = p.anchorX + dx;
                            int64_t z = p.anchorZ + dz;

                            if (u == 0 && v == 0) {
                                // Water shaft with a stone bottom
                                out.force(x, p.floorY, z, BlockType::WATER);
                                out.force(x, p.floorY - 1, z, BlockType::WATER);
                                out.force(x, p.floorY - 2, z, BlockType::WATER);
                                out.force(x, p.floorY - 3, z, stoneBlock);
                            } else {
                                out.force(x, p.floorY, z, stoneBlock);
                            }

                            int ring = std::max(std::abs(u), std::abs(v));
                            if (ring == 1) {
                                out.force(x, p.floorY + 1, z, stoneBlock); // rim
                                if (std::abs(u) == 1 && std::abs(v) == 1) {
                                    out.force(x, p.floorY + 2, z, BlockType::LOG); // posts
                                    out.force(x, p.floorY + 3, z, BlockType::PLANKS);
                                }
                            }
                            if (u == 0 && v == 0)
                                out.force(x, p.floorY + 3, z, BlockType::PLANKS); // roof center
                        }
                    }
                    break;
                }

                case StructureKind::RUIN: {
                    for (int v = -4; v <= 4; ++v) {
                        for (int u = -3; u <= 3; ++u) {
                            int dx = 0, dz = 0;
                            rotate(u, v, p.rotation, dx, dz);
                            int64_t x = p.anchorX + dx;
                            int64_t z = p.anchorZ + dz;
                            uint64_t cell = random_.u64(RUIN_DECAY_STREAM, x, p.floorY, z,
                                                        static_cast<uint32_t>(p.rotation));

                            if (cell % 10 < 6)
                                out.force(x, p.floorY, z,
                                          (cell >> 4) % 3 == 0 ? mossyBlock : stoneBlock);

                            bool perimeter = std::abs(u) == 3 || std::abs(v) == 4;
                            if (perimeter) {
                                int wallHeight = static_cast<int>((cell >> 8) % 4); // 0..3
                                for (int dy = 1; dy <= wallHeight; ++dy) {
                                    out.force(x, p.floorY + dy, z,
                                              (cell >> (10 + dy)) % 3 == 0 ? mossyBlock
                                                                           : stoneBlock);
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

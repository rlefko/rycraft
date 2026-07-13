#include "world/structures.hpp"

#include "world/chunk_generator.hpp"
#include "world/gen_seeds.hpp"

#include <algorithm>
#include <array>

namespace {

int floorDiv(int a, int b) {
    return (a >= 0) ? a / b : -((-a + b - 1) / b);
}

uint64_t regionKey(int regionX, int regionZ) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(regionX)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(regionZ));
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

bool isLandBiome(Biome biome) {
    switch (biome) {
        case Biome::OCEAN:
        case Biome::DEEP_OCEAN:
        case Biome::RIVER:
            return false;
        default:
            return true;
    }
}

// A per-chunk writer that silently drops blocks outside the chunk — the
// filter that makes footprints span borders safely.
struct ChunkWriter {
    Chunk& chunk;
    int baseX;
    int baseZ;

    void force(int x, int y, int z, BlockType block) const {
        int lx = x - baseX;
        int lz = z - baseZ;
        if (lx >= 0 && lx < CHUNK_WIDTH && lz >= 0 && lz < CHUNK_DEPTH && y >= 0 &&
            y < CHUNK_HEIGHT) {
            chunk.setBlock(lx, y, lz, block);
        }
    }
};

} // namespace

StructurePlacer::StructurePlacer(uint32_t worldSeed) : seed_(worldSeed) {}

const StructurePlacement& StructurePlacer::regionPlacement(int regionX, int regionZ,
                                                           const ChunkGenerator& gen,
                                                           GenScratch& scratch) const {
    uint64_t key = regionKey(regionX, regionZ);
    auto it = scratch.structurePlacements.find(key);
    if (it != scratch.structurePlacements.end()) return it->second;

    SeededRng rng(hashCoords(regionX, regionZ, genseed::subSeed(seed_, genseed::STRUCTURES)));

    // Fixed draw order (see the ores placer for why)
    int anchorChunkX = regionX * STRUCTURE_REGION_CHUNKS + rng.nextInt(1, 6);
    int anchorChunkZ = regionZ * STRUCTURE_REGION_CHUNKS + rng.nextInt(1, 6);
    int anchorLocalX = rng.nextInt(2, 13);
    int anchorLocalZ = rng.nextInt(2, 13);
    float kindRoll = rng.nextFloat();
    int rotation = rng.nextInt(0, 3);

    StructurePlacement placement;
    placement.kind = kindRoll < 0.4f   ? StructureKind::RUIN
                     : kindRoll < 0.7f ? StructureKind::WELL
                                       : StructureKind::HOUSE;
    placement.rotation = rotation;
    placement.anchorX = anchorChunkX * CHUNK_WIDTH + anchorLocalX;
    placement.anchorZ = anchorChunkZ * CHUNK_DEPTH + anchorLocalZ;

    int halfU = 0, halfV = 0;
    kindExtents(placement.kind, halfU, halfV);
    placement.halfX = (rotation & 1) ? halfV : halfU;
    placement.halfZ = (rotation & 1) ? halfU : halfV;

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
    placement.valid = spread <= maxSpread && probes[0] >= 64 && placement.floorY < 180 &&
                      isLandBiome(gen.biomeAt(placement.anchorX, placement.anchorZ, scratch));

    return scratch.structurePlacements.emplace(key, placement).first->second;
}

bool StructurePlacer::insideStructure(int x, int z, int chunkX, int chunkZ,
                                      const ChunkGenerator& gen, GenScratch& scratch,
                                      int margin) const {
    int minRegionX = floorDiv(chunkX - 1, STRUCTURE_REGION_CHUNKS);
    int maxRegionX = floorDiv(chunkX + 1, STRUCTURE_REGION_CHUNKS);
    int minRegionZ = floorDiv(chunkZ - 1, STRUCTURE_REGION_CHUNKS);
    int maxRegionZ = floorDiv(chunkZ + 1, STRUCTURE_REGION_CHUNKS);
    for (int rz = minRegionZ; rz <= maxRegionZ; ++rz) {
        for (int rx = minRegionX; rx <= maxRegionX; ++rx) {
            const StructurePlacement& p = regionPlacement(rx, rz, gen, scratch);
            if (!p.valid) continue;
            if (std::abs(x - p.anchorX) <= p.halfX + margin &&
                std::abs(z - p.anchorZ) <= p.halfZ + margin) {
                return true;
            }
        }
    }
    return false;
}

void StructurePlacer::place(Chunk& chunk, const ChunkGenerator& gen, GenScratch& scratch) const {
    ChunkWriter out{chunk, chunk.chunkX * CHUNK_WIDTH, chunk.chunkZ * CHUNK_DEPTH};

    int minRegionX = floorDiv(chunk.chunkX - 1, STRUCTURE_REGION_CHUNKS);
    int maxRegionX = floorDiv(chunk.chunkX + 1, STRUCTURE_REGION_CHUNKS);
    int minRegionZ = floorDiv(chunk.chunkZ - 1, STRUCTURE_REGION_CHUNKS);
    int maxRegionZ = floorDiv(chunk.chunkZ + 1, STRUCTURE_REGION_CHUNKS);

    for (int rz = minRegionZ; rz <= maxRegionZ; ++rz) {
        for (int rx = minRegionX; rx <= maxRegionX; ++rx) {
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
                    int x = p.anchorX + dx;
                    int z = p.anchorZ + dz;
                    int ground = gen.surfaceYAt(x, z, scratch);
                    for (int y = p.floorY - 1; y > ground && y > p.floorY - 9; --y) {
                        out.force(x, y, z, stoneBlock);
                    }
                }
            }

            uint64_t decaySeed =
                hashCoords(p.anchorX, p.anchorZ, genseed::subSeed(seed_, genseed::STRUCTURES));

            switch (p.kind) {
                case StructureKind::HOUSE: {
                    for (int v = -3; v <= 3; ++v) {
                        for (int u = -3; u <= 3; ++u) {
                            int dx = 0, dz = 0;
                            rotate(u, v, p.rotation, dx, dz);
                            int x = p.anchorX + dx;
                            int z = p.anchorZ + dz;

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
                            int x = p.anchorX + dx;
                            int z = p.anchorZ + dz;

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
                            int x = p.anchorX + dx;
                            int z = p.anchorZ + dz;
                            uint64_t cell = hashCoords(x, z, static_cast<uint32_t>(decaySeed));

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

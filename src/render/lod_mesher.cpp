#include "render/lod_mesher.hpp"

#include "render/block_textures.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>

// ==========================================================================
// Generic greedy mesher — works with any grid dimensions.
//
// Replaces the hardcoded 16×16×256 constants with template parameters and
// uses a block accessor callback instead of a Chunk reference. This allows
// the same meshing pipeline to run on full-resolution and coarse-resolution
// grids without code duplication.
//
// Flat 1D arrays used throughout to avoid vector<vector<T>> allocation
// overhead. Indexing: idx(row, col, width) = row * width + col.
// ==========================================================================

using BlockAccessor = std::function<BlockType(int, int, int)>;
inline static int idx(int row, int col, int width) {
    return row * width + col;
}

// A face of `cur` toward `neighbor` renders when cur has geometry (isSolid),
// the neighbor doesn't fully hide it (isOpaque), and the two blocks differ —
// interior faces between identical cutout blocks (leaf-leaf, glass-glass)
// stay culled so foliage doesn't render its own inner walls.
inline static bool faceVisible(BlockType cur, BlockType neighbor) {
    return isSolid(cur) && !isOpaque(neighbor) && neighbor != cur;
}

// One vertex of a quad corner: position + UV (UVs span the quad extent in
// blocks so the repeat sampler tiles the texture per block).
struct QuadCorner {
    float x, y, z;
    float u, v;
};

// Append one greedy quad (4 vertices + 6 indices). Corners arrive in the
// same winding the face's caller always used.
static void pushQuad(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs, FaceNormal face,
                     BlockType bt, uint8_t skyLight, const QuadCorner (&corners)[4]) {
    const uint32_t attr = packFaceAttr(face, textureLayerFor(bt, face), skyLight);
    for (const QuadCorner& c : corners) {
        verts.push_back(Vertex{attr, static_cast<float16_t>(c.x), static_cast<float16_t>(c.y),
                               static_cast<float16_t>(c.z), static_cast<float16_t>(c.u),
                               static_cast<float16_t>(c.v)});
    }
    uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
    idxs.push_back(bi);
    idxs.push_back(bi + 1);
    idxs.push_back(bi + 2);
    idxs.push_back(bi);
    idxs.push_back(bi + 2);
    idxs.push_back(bi + 3);
}

// Greedy merge on a face plane of arbitrary dimensions.
//
// faceMask[row*faceWidth + col] == true means that face cell is exposed.
// blockTypes[row*faceWidth + col] stores the block type at each exposed face cell.
//
// For each unmerged exposed cell, extend right as far as possible with
// matching block type, then extend down as far as possible with the same
// width and matching block type. Each merged rectangle becomes 1 quad.
static void meshFaceGeneric(int faceHeight, int faceWidth, const std::vector<bool>& faceMask,
                            const std::vector<BlockType>& blockTypes,
                            const std::vector<uint8_t>& cellLight, std::vector<bool>& merged,
                            FaceNormal face, std::vector<Vertex>& vertices,
                            std::vector<uint32_t>& indices, const auto& emitQuadFn) {
    merged.assign(faceHeight * faceWidth, false);

    for (int row = 0; row < faceHeight; ++row) {
        for (int col = 0; col < faceWidth; ++col) {
            int i = idx(row, col, faceWidth);
            if (!faceMask[i] || merged[i]) {
                continue;
            }

            BlockType leadType = blockTypes[i];
            uint8_t leadLight = cellLight[i];

            // Extend right (horizontal) while type AND light match — a quad
            // carries one light value, so shading boundaries end the merge
            int width = 1;
            while (col + width < faceWidth && faceMask[idx(row, col + width, faceWidth)] &&
                   !merged[idx(row, col + width, faceWidth)] &&
                   blockTypes[idx(row, col + width, faceWidth)] == leadType &&
                   cellLight[idx(row, col + width, faceWidth)] == leadLight) {
                ++width;
            }

            // Extend down (vertical) as far as possible with same width, type, light
            int height = 1;
            while (row + height < faceHeight) {
                bool rowValid = true;
                for (int w = 0; w < width; ++w) {
                    int j = idx(row + height, col + w, faceWidth);
                    if (!faceMask[j] || merged[j] || blockTypes[j] != leadType ||
                        cellLight[j] != leadLight) {
                        rowValid = false;
                        break;
                    }
                }
                if (!rowValid) break;
                ++height;
            }

            // Mark merged
            for (int dr = 0; dr < height; ++dr) {
                for (int dc = 0; dc < width; ++dc) {
                    merged[idx(row + dr, col + dc, faceWidth)] = true;
                }
            }

            // Emit quad via callback
            emitQuadFn(col, row, width, height, face, leadType, leadLight, vertices, indices);
        }
    }
}

static MeshOutput buildGenericMesh(int gridW, int gridH, int gridD, const BlockAccessor& getBlock) {
    MeshOutput output;

    // Pre-allocate reasonable capacity (6 faces × grid area × 4 verts)
    output.vertices.reserve(6 * gridW * gridD * gridH / 4);
    output.indices.reserve(6 * gridW * gridD * gridH / 2);

    // ---- Column skylight ----
    // The first open Y above the topmost solid block per column. A face is
    // lit by how close its exposure cell sits to that height: open sky is
    // 15, shade under a canopy or overhang steps down, caves bottom out at 4.
    std::vector<int> skyHeight(gridW * gridD, 0);
    for (int z = 0; z < gridD; ++z) {
        for (int x = 0; x < gridW; ++x) {
            for (int y = gridH - 1; y >= 0; --y) {
                if (isSolid(getBlock(x, y, z))) {
                    skyHeight[idx(z, x, gridW)] = y + 1;
                    break;
                }
            }
        }
    }
    auto lightAt = [&](int x, int y, int z) -> uint8_t {
        x = std::clamp(x, 0, gridW - 1);
        z = std::clamp(z, 0, gridD - 1);
        int depth = skyHeight[idx(z, x, gridW)] - y;
        if (depth <= 0) return 15;
        return static_cast<uint8_t>(std::max(12 - depth, 4));
    };

    // Reusable flat buffers for face processing (avoid reallocation per face)
    std::vector<bool> faceMask;
    std::vector<BlockType> blockTypes;
    std::vector<uint8_t> cellLight;
    std::vector<bool> merged;

    // ======================================================================
    // Face: +Y (top) — visible when the block above doesn't hide it
    // ======================================================================
    for (int ly = 0; ly < gridH - 1; ++ly) {
        faceMask.assign(gridD * gridW, false);
        blockTypes.assign(gridD * gridW, BlockType::AIR);
        cellLight.assign(gridD * gridW, 15);

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (faceVisible(cur, getBlock(x, ly + 1, z))) {
                    faceMask[idx(z, x, gridW)] = true;
                    blockTypes[idx(z, x, gridW)] = cur;
                    cellLight[idx(z, x, gridW)] = lightAt(x, ly + 1, z);
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // +Y face: y = ly+1, CCW from above
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(ly + 1), static_cast<float>(row), 0.f,
                 0.f},
                {static_cast<float>(col + width), static_cast<float>(ly + 1),
                 static_cast<float>(row), fw, 0.f},
                {static_cast<float>(col + width), static_cast<float>(ly + 1),
                 static_cast<float>(row + height), fw, fh},
                {static_cast<float>(col), static_cast<float>(ly + 1),
                 static_cast<float>(row + height), 0.f, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridD, gridW, faceMask, blockTypes, cellLight, merged, FaceNormal::PLUS_Y,
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -Y (bottom) — visible when the block below doesn't hide it
    // ======================================================================
    for (int ly = 1; ly < gridH; ++ly) {
        faceMask.assign(gridD * gridW, false);
        blockTypes.assign(gridD * gridW, BlockType::AIR);
        cellLight.assign(gridD * gridW, 15);

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (faceVisible(cur, getBlock(x, ly - 1, z))) {
                    faceMask[idx(z, x, gridW)] = true;
                    blockTypes[idx(z, x, gridW)] = cur;
                    cellLight[idx(z, x, gridW)] = lightAt(x, ly - 1, z);
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // -Y face: y = ly, CCW from below
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(ly), static_cast<float>(row), 0.f,
                 0.f},
                {static_cast<float>(col + width), static_cast<float>(ly), static_cast<float>(row),
                 fw, 0.f},
                {static_cast<float>(col + width), static_cast<float>(ly),
                 static_cast<float>(row + height), fw, fh},
                {static_cast<float>(col), static_cast<float>(ly), static_cast<float>(row + height),
                 0.f, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridD, gridW, faceMask, blockTypes, cellLight, merged, FaceNormal::MINUS_Y,
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: +X (right) — exposed when solid AND block to +X is AIR
    // ======================================================================
    for (int lx = 0; lx < gridW - 1; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);
        cellLight.assign(gridH * gridD, 15);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (faceVisible(cur, getBlock(lx + 1, y, z))) {
                    faceMask[idx(y, z, gridD)] = true;
                    blockTypes[idx(y, z, gridD)] = cur;
                    cellLight[idx(y, z, gridD)] = lightAt(lx + 1, y, z);
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // +X face: x = lx+1, CCW from +X (rows are Y, cols are Z).
            // Texture v runs downward in Metal, so the TOP of the face
            // carries v=0 — otherwise side textures (the grass strip)
            // render upside down. Same for the other three side faces.
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(lx + 1), static_cast<float>(row), static_cast<float>(col), 0.f,
                 fh},
                {static_cast<float>(lx + 1), static_cast<float>(row + height),
                 static_cast<float>(col), 0.f, 0.f},
                {static_cast<float>(lx + 1), static_cast<float>(row + height),
                 static_cast<float>(col + width), fw, 0.f},
                {static_cast<float>(lx + 1), static_cast<float>(row),
                 static_cast<float>(col + width), fw, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridH, gridD, faceMask, blockTypes, cellLight, merged, FaceNormal::PLUS_X,
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -X (left) — exposed when solid AND block to -X is AIR
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);
        cellLight.assign(gridH * gridD, 15);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (faceVisible(cur, getBlock(lx - 1, y, z))) {
                    faceMask[idx(y, z, gridD)] = true;
                    blockTypes[idx(y, z, gridD)] = cur;
                    cellLight[idx(y, z, gridD)] = lightAt(lx - 1, y, z);
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // -X face: the face plane of block lx is x = lx (the old code
            // emitted at lx-1, one unit inside the neighbor)
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(lx), static_cast<float>(row), static_cast<float>(col), 0.f, fh},
                {static_cast<float>(lx), static_cast<float>(row), static_cast<float>(col + width),
                 fw, fh},
                {static_cast<float>(lx), static_cast<float>(row + height),
                 static_cast<float>(col + width), fw, 0.f},
                {static_cast<float>(lx), static_cast<float>(row + height), static_cast<float>(col),
                 0.f, 0.f},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridH, gridD, faceMask, blockTypes, cellLight, merged, FaceNormal::MINUS_X,
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: +Z (front) — exposed when solid AND block to +Z is AIR
    // ======================================================================
    for (int lz = 0; lz < gridD - 1; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);
        cellLight.assign(gridH * gridW, 15);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (faceVisible(cur, getBlock(x, y, lz + 1))) {
                    faceMask[idx(y, x, gridW)] = true;
                    blockTypes[idx(y, x, gridW)] = cur;
                    cellLight[idx(y, x, gridW)] = lightAt(x, y, lz + 1);
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // +Z face: z = lz+1 (the old code emitted at lz, coplanar with
            // the block interior)
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(row), static_cast<float>(lz + 1), 0.f,
                 fh},
                {static_cast<float>(col), static_cast<float>(row + height),
                 static_cast<float>(lz + 1), 0.f, 0.f},
                {static_cast<float>(col + width), static_cast<float>(row + height),
                 static_cast<float>(lz + 1), fw, 0.f},
                {static_cast<float>(col + width), static_cast<float>(row),
                 static_cast<float>(lz + 1), fw, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridH, gridW, faceMask, blockTypes, cellLight, merged, FaceNormal::PLUS_Z,
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -Z (back) — exposed when solid AND block to -Z is AIR
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);
        cellLight.assign(gridH * gridW, 15);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (faceVisible(cur, getBlock(x, y, lz - 1))) {
                    faceMask[idx(y, x, gridW)] = true;
                    blockTypes[idx(y, x, gridW)] = cur;
                    cellLight[idx(y, x, gridW)] = lightAt(x, y, lz - 1);
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, std::vector<Vertex>& verts,
                             std::vector<uint32_t>& idxs) {
            // -Z face: the face plane of block lz is z = lz (the old code
            // emitted at lz-1, one unit inside the neighbor)
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(row), static_cast<float>(lz), 0.f, fh},
                {static_cast<float>(col + width), static_cast<float>(row), static_cast<float>(lz),
                 fw, fh},
                {static_cast<float>(col + width), static_cast<float>(row + height),
                 static_cast<float>(lz), fw, 0.f},
                {static_cast<float>(col), static_cast<float>(row + height), static_cast<float>(lz),
                 0.f, 0.f},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridH, gridW, faceMask, blockTypes, cellLight, merged, FaceNormal::MINUS_Z,
                        output.vertices, output.indices, emitQuad);
    }

    return output;
}

// ==========================================================================
// LODMesher implementation
// ==========================================================================

MeshOutput LODMesher::buildMesh(const Chunk& chunk, int lodLevel) {
    // Beyond render distance — return empty mesh (distance culling)
    if (lodLevel >= static_cast<int>(ChunkLOD::COUNT)) {
        return MeshOutput{};
    }

    switch (static_cast<ChunkLOD>(lodLevel)) {
        case ChunkLOD::FULL: {
            // LOD 0: Full resolution greedy meshing (16×16×256)
            auto blockFn = [&chunk](int x, int y, int z) -> BlockType {
                return chunk.getBlock(x, y, z);
            };
            return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn);
        }

        case ChunkLOD::MEDIUM: {
            // LOD 1: 2× downsampling (8×8×128)
            // Each coarse block represents a 2×2×2 group of original blocks.
            // Coarse block is solid if majority of group is solid.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                int gx = cx * 2;
                int gy = cy * 2;
                int gz = cz * 2;

                std::unordered_map<BlockType, int> typeCounts;
                for (int dz = 0; dz < 2; ++dz) {
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            BlockType bt = chunk.getBlock(gx + dx, gy + dy, gz + dz);
                            typeCounts[bt]++;
                        }
                    }
                }

                // Return the most common block type in the group
                BlockType dominant = BlockType::AIR;
                int maxCount = 0;
                for (auto& [bt, count] : typeCounts) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominant = bt;
                    }
                }
                return dominant;
            };
            return buildGenericMesh(8, 128, 8, blockFn);
        }

        case ChunkLOD::COARSE: {
            // LOD 2: 4× downsampling (4×4×64)
            // Each coarse block represents a 4×4×4 group of original blocks.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                int gx = cx * 4;
                int gy = cy * 4;
                int gz = cz * 4;

                std::unordered_map<BlockType, int> typeCounts;
                for (int dz = 0; dz < 4; ++dz) {
                    for (int dy = 0; dy < 4; ++dy) {
                        for (int dx = 0; dx < 4; ++dx) {
                            BlockType bt = chunk.getBlock(gx + dx, gy + dy, gz + dz);
                            typeCounts[bt]++;
                        }
                    }
                }

                // Return the most common block type in the group
                BlockType dominant = BlockType::AIR;
                int maxCount = 0;
                for (auto& [bt, count] : typeCounts) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominant = bt;
                    }
                }
                return dominant;
            };
            return buildGenericMesh(4, 64, 4, blockFn);
        }

        default:
            return MeshOutput{};
    }
}

int LODMesher::computeLODLevel(int distanceBlocks) {
    if (distanceBlocks < LOD0_MAX_DISTANCE) return static_cast<int>(ChunkLOD::FULL);
    if (distanceBlocks < LOD1_MAX_DISTANCE) return static_cast<int>(ChunkLOD::MEDIUM);
    if (distanceBlocks < LOD2_MAX_DISTANCE) return static_cast<int>(ChunkLOD::COARSE);
    return static_cast<int>(ChunkLOD::COUNT); // Beyond render distance
}

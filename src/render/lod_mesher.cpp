#include "render/lod_mesher.hpp"

#include "render/block_textures.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_map>

// ==========================================================================
// Generic greedy mesher — works with any grid dimensions.
//
// The block accessor is a template parameter (a chunk read used to go
// through std::function — ~1M indirect calls per full build, the mesher's
// documented hot-path debt). The six directional passes run twice per build
// with different visibility predicates: once for the opaque cube section,
// once for water surfaces (drawn by the dedicated water pass).
//
// `padded` builds (MeshSnapshot) read one real neighbor block beyond every
// XZ edge: boundary faces emit exactly when the neighbor doesn't hide them
// — the same rule as interior faces. Unpadded builds (coarse LODs, unit
// tests) treat out-of-grid as air and skip the +X/+Z boundary layer, which
// over-draws interior walls but can't read neighbors it doesn't have.
//
// Flat 1D arrays used throughout to avoid vector<vector<T>> allocation
// overhead. Indexing: idx(row, col, width) = row * width + col.
// ==========================================================================

inline static int idx(int row, int col, int width) {
    return row * width + col;
}

// A face of `cur` toward `neighbor` renders when cur has cube geometry,
// the neighbor doesn't fully hide it (isOpaque), and the two blocks differ —
// interior faces between identical cutout blocks (leaf-leaf, glass-glass)
// stay culled so foliage doesn't render its own inner walls.
inline static bool cubeFaceVisible(BlockType cur, BlockType neighbor) {
    return rendersAsCube(cur) && !isOpaque(neighbor) && neighbor != cur;
}

// Water surfaces: every boundary between water and a non-water, non-hiding
// cell (the sea top under air, sides at waterfall lips, ceilings of flooded
// caves). Interior water-water faces stay culled.
inline static bool waterFaceVisible(BlockType cur, BlockType neighbor) {
    return cur == BlockType::WATER && neighbor != BlockType::WATER && !isOpaque(neighbor);
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

// The six directional passes for one visibility predicate. topDrop lowers
// the +Y face plane (the water surface sits 0.125 below the cell top —
// fp16-exact at every chunk-local magnitude, so no cracks).
template <typename Access, typename Visible>
static void runGreedyPasses(int gridW, int gridH, int gridD, const Access& getBlock,
                            const Visible& visible, const auto& lightAt, float topDrop, bool padded,
                            MeshScratch& scratch, std::vector<Vertex>& outVertices,
                            std::vector<uint32_t>& outIndices) {
    std::vector<bool>& faceMask = scratch.faceMask;
    std::vector<BlockType>& blockTypes = scratch.blockTypes;
    std::vector<uint8_t>& cellLight = scratch.cellLight;
    std::vector<bool>& merged = scratch.merged;

    // Padded builds know their +X/+Z neighbor walls, so the boundary layer
    // meshes like any other; unpadded builds must skip it (assuming air
    // there would paint a wall inside the neighbor).
    const int xEnd = padded ? gridW : gridW - 1;
    const int zEnd = padded ? gridD : gridD - 1;

    // ======================================================================
    // Face: +Y (top) — visible when the block above doesn't hide it
    // (the world's top layer reads air above and gets a lid)
    // ======================================================================
    for (int ly = 0; ly < gridH; ++ly) {
        faceMask.assign(gridD * gridW, false);
        blockTypes.assign(gridD * gridW, BlockType::AIR);
        cellLight.assign(gridD * gridW, 15);

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (visible(cur, getBlock(x, ly + 1, z))) {
                    faceMask[idx(z, x, gridW)] = true;
                    blockTypes[idx(z, x, gridW)] = cur;
                    cellLight[idx(z, x, gridW)] = lightAt(x, ly + 1, z);
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly, topDrop](int col, int row, int width, int height, FaceNormal face,
                                      BlockType bt, uint8_t skyLight, std::vector<Vertex>& verts,
                                      std::vector<uint32_t>& idxs) {
            // +Y face: y = ly+1 (minus the water-surface drop), CCW from above
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const float y = static_cast<float>(ly + 1) - topDrop;
            const QuadCorner corners[4] = {
                {static_cast<float>(col), y, static_cast<float>(row), 0.f, 0.f},
                {static_cast<float>(col + width), y, static_cast<float>(row), fw, 0.f},
                {static_cast<float>(col + width), y, static_cast<float>(row + height), fw, fh},
                {static_cast<float>(col), y, static_cast<float>(row + height), 0.f, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners);
        };

        meshFaceGeneric(gridD, gridW, faceMask, blockTypes, cellLight, merged, FaceNormal::PLUS_Y,
                        outVertices, outIndices, emitQuad);
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
                if (visible(cur, getBlock(x, ly - 1, z))) {
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
                        outVertices, outIndices, emitQuad);
    }

    // ======================================================================
    // Face: +X (right) — exposed when visible toward the +X neighbor
    // ======================================================================
    for (int lx = 0; lx < xEnd; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);
        cellLight.assign(gridH * gridD, 15);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx + 1, y, z))) {
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
                        outVertices, outIndices, emitQuad);
    }

    // ======================================================================
    // Face: -X (left) — exposed when visible toward the -X neighbor
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);
        cellLight.assign(gridH * gridD, 15);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx - 1, y, z))) {
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
                        outVertices, outIndices, emitQuad);
    }

    // ======================================================================
    // Face: +Z (front) — exposed when visible toward the +Z neighbor
    // ======================================================================
    for (int lz = 0; lz < zEnd; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);
        cellLight.assign(gridH * gridW, 15);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz + 1))) {
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
                        outVertices, outIndices, emitQuad);
    }

    // ======================================================================
    // Face: -Z (back) — exposed when visible toward the -Z neighbor
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);
        cellLight.assign(gridH * gridW, 15);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz - 1))) {
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
                        outVertices, outIndices, emitQuad);
    }
}

// Flora cross-quads: two diagonal quads spanning the cell, inset 0.125 from
// the walls (0.125 is exactly representable in fp16 at every chunk-local
// magnitude, so the X stays crack-free). Single winding per quad — the
// scene pass renders with cull mode None, which makes them double-sided.
static void emitFloraCross(int x, int y, int z, BlockType bt, uint8_t skyLight,
                           std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
    const float x0 = static_cast<float>(x) + 0.125f;
    const float x1 = static_cast<float>(x) + 0.875f;
    const float z0 = static_cast<float>(z) + 0.125f;
    const float z1 = static_cast<float>(z) + 0.875f;
    const float y0 = static_cast<float>(y);
    const float y1 = static_cast<float>(y + 1);

    // v = 0 at the TOP (Metal v runs downward — see the +X face comment)
    const QuadCorner diagonalA[4] = {
        {x0, y0, z0, 0.f, 1.f},
        {x1, y0, z1, 1.f, 1.f},
        {x1, y1, z1, 1.f, 0.f},
        {x0, y1, z0, 0.f, 0.f},
    };
    const QuadCorner diagonalB[4] = {
        {x0, y0, z1, 0.f, 1.f},
        {x1, y0, z0, 1.f, 1.f},
        {x1, y1, z0, 1.f, 0.f},
        {x0, y1, z1, 0.f, 0.f},
    };
    pushQuad(verts, idxs, FaceNormal::CROSS, bt, skyLight, diagonalA);
    pushQuad(verts, idxs, FaceNormal::CROSS, bt, skyLight, diagonalB);
}

template <typename Access>
static MeshOutput buildGenericMesh(int gridW, int gridH, int gridD, const Access& getBlock,
                                   bool padded, bool emitFlora, MeshScratch& scratch) {
    MeshOutput output;

    // Typical full chunks mesh to a few thousand vertices; growth beyond
    // this is amortized (the old code reserved 1.5 MB per build)
    output.vertices.reserve(8192);
    output.indices.reserve(12288);

    // ---- Column skylight ----
    // The first open Y above the topmost solid block per column, computed
    // over the padded ring too so border faces read their real neighbor
    // column instead of a clamped copy (that clamp painted a visible light
    // seam along every chunk edge). A face is lit by how close its exposure
    // cell sits to that height: open sky is 15, shade under a canopy or
    // overhang steps down, caves bottom out at 4.
    const int ringW = gridW + 2;
    const int ringD = gridD + 2;
    std::vector<int>& skyHeight = scratch.skyHeight;
    skyHeight.assign(ringW * ringD, 0);
    for (int z = -1; z <= gridD; ++z) {
        for (int x = -1; x <= gridW; ++x) {
            for (int y = gridH - 1; y >= 0; --y) {
                if (isSolid(getBlock(x, y, z))) {
                    skyHeight[idx(z + 1, x + 1, ringW)] = y + 1;
                    break;
                }
            }
        }
    }
    auto lightAt = [&](int x, int y, int z) -> uint8_t {
        int depth = skyHeight[idx(z + 1, x + 1, ringW)] - y;
        if (depth <= 0) return 15;
        return static_cast<uint8_t>(std::max(12 - depth, 4));
    };

    // ---- Opaque section: cubes, then flora crosses (full LOD only) ----
    runGreedyPasses(gridW, gridH, gridD, getBlock, cubeFaceVisible, lightAt, 0.f, padded, scratch,
                    output.vertices, output.indices);

    if (emitFlora) {
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                for (int y = 0; y < gridH; ++y) {
                    BlockType bt = getBlock(x, y, z);
                    if (isFlora(bt)) {
                        emitFloraCross(x, y, z, bt, lightAt(x, y, z), output.vertices,
                                       output.indices);
                    }
                }
            }
        }
    }

    // ---- Water section: everything after this index draws in the water
    // pass. Padded builds read real neighbor water; unpadded builds assume
    // water continues past the edge (oceans virtually always do, and a wall
    // there painted phantom stripes along every chunk border).
    output.opaqueIndexCount = static_cast<uint32_t>(output.indices.size());
    if (padded) {
        runGreedyPasses(gridW, gridH, gridD, getBlock, waterFaceVisible, lightAt, 0.125f, padded,
                        scratch, output.vertices, output.indices);
    } else {
        auto waterEdgeBlock = [&getBlock, gridW, gridD](int x, int y, int z) -> BlockType {
            if (x < 0 || x >= gridW || z < 0 || z >= gridD) return BlockType::WATER;
            return getBlock(x, y, z);
        };
        runGreedyPasses(gridW, gridH, gridD, waterEdgeBlock, waterFaceVisible, lightAt, 0.125f,
                        padded, scratch, output.vertices, output.indices);
    }

    return output;
}

// ==========================================================================
// LODMesher implementation
// ==========================================================================

MeshOutput LODMesher::buildMesh(const MeshSnapshot& snapshot, MeshScratch& scratch) {
    auto blockFn = [&snapshot](int x, int y, int z) -> BlockType { return snapshot.at(x, y, z); };
    return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn, /*padded=*/true,
                            /*emitFlora=*/true, scratch);
}

MeshOutput LODMesher::buildMesh(const Chunk& chunk, int lodLevel) {
    // Beyond render distance — return empty mesh (distance culling)
    if (lodLevel >= static_cast<int>(ChunkLOD::COUNT)) {
        return MeshOutput{};
    }

    thread_local MeshScratch scratch;

    switch (static_cast<ChunkLOD>(lodLevel)) {
        case ChunkLOD::FULL: {
            // LOD 0: Full resolution greedy meshing (16×16×256),
            // neighbor-blind (tests and tools; the game uses the
            // MeshSnapshot overload)
            auto blockFn = [&chunk](int x, int y, int z) -> BlockType {
                return chunk.getBlock(x, y, z);
            };
            return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn,
                                    /*padded=*/false, /*emitFlora=*/true, scratch);
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
                            // Flora is skipped at coarse LODs — a meadow must
                            // not majority-pick into phantom solid cells
                            if (isFlora(bt)) bt = BlockType::AIR;
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
            return buildGenericMesh(8, 128, 8, blockFn, /*padded=*/false, /*emitFlora=*/false,
                                    scratch);
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
                            if (isFlora(bt)) bt = BlockType::AIR;
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
            return buildGenericMesh(4, 64, 4, blockFn, /*padded=*/false, /*emitFlora=*/false,
                                    scratch);
        }

        default:
            return MeshOutput{};
    }
}

#include "render/lod_mesher.hpp"

#include "common/random.hpp"
#include "render/block_textures.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>

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
// XZ edge: boundary faces emit exactly when the neighbor doesn't hide them,
// using the same rule as interior faces. Unpadded builds (coarse LODs and
// unit tests) treat out-of-grid cells as air and emit all six boundaries.
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

// Classic voxel corner AO: 0 (fully enclosed) .. 3 (open). side1/side2 are the
// two edge-adjacent occluders one step along the outward normal, corner the
// diagonal one; two solid sides bury the vertex regardless of the corner.
inline static uint8_t aoVertex(bool side1, bool side2, bool corner) {
    if (side1 && side2) {
        return 0;
    }
    return static_cast<uint8_t>(3 - (int(side1) + int(side2) + int(corner)));
}

// Pack/unpack four 2-bit corner-AO values (emit-corner order) in one byte.
// AO_ALL_OPEN is the "no occlusion, all four corners open" sentinel.
inline static uint8_t packAO(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return static_cast<uint8_t>(a0 | (a1 << 2) | (a2 << 4) | (a3 << 6));
}
inline static uint8_t cornerAOAt(uint8_t packed, int corner) {
    return static_cast<uint8_t>((packed >> (corner * 2)) & 3u);
}
constexpr uint8_t AO_ALL_OPEN = 0xFF; // packAO(3, 3, 3, 3)

// Block-light accessor for the coarse LODs, which downsample block types only
// and carry no baked block light.
inline static uint8_t noBlockLight(int, int, int) {
    return 0;
}

static Vertex makeVertex(uint32_t faceAttr, const QuadCorner& corner) {
    Vertex vertex;
    std::memset(&vertex, 0, sizeof(vertex));
    vertex.faceAttr = faceAttr;
    vertex.px = static_cast<float16_t>(corner.x);
    vertex.py = static_cast<float16_t>(corner.y);
    vertex.pz = static_cast<float16_t>(corner.z);
    vertex.u = static_cast<float16_t>(corner.u);
    vertex.v = static_cast<float16_t>(corner.v);
    return vertex;
}

static void reserveInitialMeshStorage(std::vector<Vertex>& vertices,
                                      std::vector<uint32_t>& indices) {
    if (vertices.capacity() == 0) vertices.reserve(4608);
    if (indices.capacity() == 0) indices.reserve(6912);
}

// Append one outward-wound greedy quad. packedAO follows the corner order.
// When the diagonals differ, rotate the quad cyclically so triangulation uses
// the brighter diagonal without changing its winding.
static void pushQuad(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs, FaceNormal face,
                     BlockType bt, uint8_t skyLight, const QuadCorner (&corners)[4],
                     uint8_t packedAO, uint8_t blockLight) {
    reserveInitialMeshStorage(verts, idxs);
    const bool flip = (cornerAOAt(packedAO, 0) + cornerAOAt(packedAO, 2)) <
                      (cornerAOAt(packedAO, 1) + cornerAOAt(packedAO, 3));
    const uint8_t layer = textureLayerFor(bt, face);
    for (int index = 0; index < 4; ++index) {
        const int cornerIndex = flip ? (index + 1) & 3 : index;
        const uint32_t attr = packFaceAttr(face, layer, skyLight, cornerAOAt(packedAO, cornerIndex),
                                           blockLight, isEmissive(bt), swayClass(bt));
        verts.push_back(makeVertex(attr, corners[cornerIndex]));
    }
    uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
    idxs.push_back(bi);
    idxs.push_back(bi + 1);
    idxs.push_back(bi + 2);
    idxs.push_back(bi);
    idxs.push_back(bi + 2);
    idxs.push_back(bi + 3);
}

static void pushDoubleSidedQuad(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs,
                                FaceNormal face, BlockType bt, uint8_t skyLight,
                                const QuadCorner (&corners)[4], uint8_t packedAO,
                                uint8_t blockLight) {
    pushQuad(verts, idxs, face, bt, skyLight, corners, packedAO, blockLight);
    const uint32_t base = static_cast<uint32_t>(verts.size()) - 4;
    idxs.push_back(base);
    idxs.push_back(base + 2);
    idxs.push_back(base + 1);
    idxs.push_back(base);
    idxs.push_back(base + 3);
    idxs.push_back(base + 2);
}

static void pushFluidQuad(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs, FaceNormal face,
                          uint8_t skyLight, uint8_t blockLight, uint8_t flowDirection, bool falling,
                          const QuadCorner (&corners)[4]) {
    reserveInitialMeshStorage(verts, idxs);
    const uint32_t attr = packFluidFaceAttr(face, skyLight, flowDirection, falling, blockLight);
    for (const QuadCorner& corner : corners)
        verts.push_back(makeVertex(attr, corner));
    const uint32_t base = static_cast<uint32_t>(verts.size()) - 4;
    idxs.insert(idxs.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

// Greedy merge on a face plane of arbitrary dimensions.
//
// A nonzero key combines every property that must remain constant across a
// greedy quad. Consumed rectangles are cleared in place.
static uint32_t packFaceKey(BlockType block, uint8_t skyLight, uint8_t blockLight,
                            uint8_t packedAO) {
    return (static_cast<uint32_t>(block) + 1U) | (static_cast<uint32_t>(skyLight & 0x0FU) << 8U) |
           (static_cast<uint32_t>(blockLight & 0x0FU) << 12U) |
           (static_cast<uint32_t>(packedAO) << 16U);
}

static BlockType faceKeyBlock(uint32_t key) {
    return static_cast<BlockType>((key & 0xFFU) - 1U);
}

static uint8_t faceKeySkyLight(uint32_t key) {
    return static_cast<uint8_t>((key >> 8U) & 0x0FU);
}

static uint8_t faceKeyBlockLight(uint32_t key) {
    return static_cast<uint8_t>((key >> 12U) & 0x0FU);
}

static uint8_t faceKeyAO(uint32_t key) {
    return static_cast<uint8_t>((key >> 16U) & 0xFFU);
}

static_assert(BLOCK_TYPE_COUNT <= 255);

static void meshFaceGeneric(int faceHeight, int faceWidth, uint32_t* faceKeys, FaceNormal face,
                            std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                            const auto& emitQuadFn) {
    for (int row = 0; row < faceHeight; ++row) {
        for (int col = 0; col < faceWidth; ++col) {
            const int i = idx(row, col, faceWidth);
            const uint32_t leadKey = faceKeys[i];
            if (leadKey == 0) continue;

            int width = 1;
            while (col + width < faceWidth &&
                   faceKeys[idx(row, col + width, faceWidth)] == leadKey) {
                ++width;
            }

            int height = 1;
            while (row + height < faceHeight) {
                bool rowValid = true;
                for (int w = 0; w < width; ++w) {
                    if (faceKeys[idx(row + height, col + w, faceWidth)] != leadKey) {
                        rowValid = false;
                        break;
                    }
                }
                if (!rowValid) break;
                ++height;
            }

            for (int dr = 0; dr < height; ++dr) {
                std::fill_n(faceKeys + idx(row + dr, col, faceWidth), width, uint32_t{0});
            }

            emitQuadFn(col, row, width, height, face, faceKeyBlock(leadKey),
                       faceKeySkyLight(leadKey), faceKeyBlockLight(leadKey), faceKeyAO(leadKey),
                       vertices, indices);
        }
    }
}

// The six directional passes for one visibility predicate. topDrop lowers
// the +Y face plane when a caller has an explicit partial fluid height.
template <typename Access, typename LightAccess, typename Visible>
static void runGreedyPasses(int gridW, int gridH, int gridD, const Access& getBlock,
                            const LightAccess& getBlockLight, const Visible& visible,
                            const auto& lightAt, float topDrop, bool bakeAO, MeshScratch& scratch,
                            std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices) {
    uint32_t* faceKeys = scratch.faceKeys.data();
    assert(gridW * gridD <= static_cast<int>(scratch.faceKeys.size()));
    assert(gridH * gridD <= static_cast<int>(scratch.faceKeys.size()));
    assert(gridH * gridW <= static_cast<int>(scratch.faceKeys.size()));

    // Baked corner AO reads the eight occluders in the plane one step along
    // each face's outward normal (isOpaque, so leaves don't cast AO — the same
    // rule as skylight). Each helper folds them into the four quad corners in
    // that face's emit order; unexposed cells and the water pass stay
    // AO_ALL_OPEN (no darkening).
    auto occ = [&](int gx, int gy, int gz) { return isOpaque(getBlock(gx, gy, gz)); };
    auto aoPlusY = [&](int cx, int py, int cz) -> uint8_t {
        bool xm = occ(cx - 1, py, cz), xp = occ(cx + 1, py, cz);
        bool zm = occ(cx, py, cz - 1), zp = occ(cx, py, cz + 1);
        bool mm = occ(cx - 1, py, cz - 1), pm = occ(cx + 1, py, cz - 1);
        bool pp = occ(cx + 1, py, cz + 1), mp = occ(cx - 1, py, cz + 1);
        return packAO(aoVertex(xm, zm, mm), aoVertex(xm, zp, mp), aoVertex(xp, zp, pp),
                      aoVertex(xp, zm, pm));
    };
    auto aoMinusY = [&](int cx, int py, int cz) -> uint8_t {
        bool xm = occ(cx - 1, py, cz), xp = occ(cx + 1, py, cz);
        bool zm = occ(cx, py, cz - 1), zp = occ(cx, py, cz + 1);
        bool mm = occ(cx - 1, py, cz - 1), pm = occ(cx + 1, py, cz - 1);
        bool pp = occ(cx + 1, py, cz + 1), mp = occ(cx - 1, py, cz + 1);
        return packAO(aoVertex(xm, zm, mm), aoVertex(xp, zm, pm), aoVertex(xp, zp, pp),
                      aoVertex(xm, zp, mp));
    };
    auto aoPlusX = [&](int px, int cy, int cz) -> uint8_t { // +X, outward plane px
        bool ym = occ(px, cy - 1, cz), yp = occ(px, cy + 1, cz);
        bool zm = occ(px, cy, cz - 1), zp = occ(px, cy, cz + 1);
        bool mm = occ(px, cy - 1, cz - 1), pm = occ(px, cy + 1, cz - 1);
        bool pp = occ(px, cy + 1, cz + 1), mp = occ(px, cy - 1, cz + 1);
        return packAO(aoVertex(ym, zm, mm), aoVertex(yp, zm, pm), aoVertex(yp, zp, pp),
                      aoVertex(ym, zp, mp));
    };
    auto aoMinusX = [&](int px, int cy, int cz) -> uint8_t { // -X, outward plane px
        bool ym = occ(px, cy - 1, cz), yp = occ(px, cy + 1, cz);
        bool zm = occ(px, cy, cz - 1), zp = occ(px, cy, cz + 1);
        bool mm = occ(px, cy - 1, cz - 1), pm = occ(px, cy + 1, cz - 1);
        bool pp = occ(px, cy + 1, cz + 1), mp = occ(px, cy - 1, cz + 1);
        return packAO(aoVertex(ym, zm, mm), aoVertex(ym, zp, mp), aoVertex(yp, zp, pp),
                      aoVertex(yp, zm, pm));
    };
    auto aoPlusZ = [&](int cx, int cy, int pz) -> uint8_t { // +Z, outward plane pz
        bool xm = occ(cx - 1, cy, pz), xp = occ(cx + 1, cy, pz);
        bool ym = occ(cx, cy - 1, pz), yp = occ(cx, cy + 1, pz);
        bool mm = occ(cx - 1, cy - 1, pz), pm = occ(cx + 1, cy - 1, pz);
        bool pp = occ(cx + 1, cy + 1, pz), mp = occ(cx - 1, cy + 1, pz);
        return packAO(aoVertex(xm, ym, mm), aoVertex(xp, ym, pm), aoVertex(xp, yp, pp),
                      aoVertex(xm, yp, mp));
    };
    auto aoMinusZ = [&](int cx, int cy, int pz) -> uint8_t { // -Z, outward plane pz
        bool xm = occ(cx - 1, cy, pz), xp = occ(cx + 1, cy, pz);
        bool ym = occ(cx, cy - 1, pz), yp = occ(cx, cy + 1, pz);
        bool mm = occ(cx - 1, cy - 1, pz), pm = occ(cx + 1, cy - 1, pz);
        bool pp = occ(cx + 1, cy + 1, pz), mp = occ(cx - 1, cy + 1, pz);
        return packAO(aoVertex(xm, ym, mm), aoVertex(xm, yp, mp), aoVertex(xp, yp, pp),
                      aoVertex(xp, ym, pm));
    };

    // ======================================================================
    // Face: +Y (top) — visible when the block above doesn't hide it
    // (the world's top layer reads air above and gets a lid)
    // ======================================================================
    for (int ly = 0; ly < gridH; ++ly) {
        std::fill_n(faceKeys, gridD * gridW, uint32_t{0});

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (visible(cur, getBlock(x, ly + 1, z))) {
                    faceKeys[idx(z, x, gridW)] =
                        packFaceKey(cur, lightAt(x, ly + 1, z), getBlockLight(x, ly + 1, z),
                                    bakeAO ? aoPlusY(x, ly + 1, z) : AO_ALL_OPEN);
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly, topDrop](int col, int row, int width, int height, FaceNormal face,
                                      BlockType bt, uint8_t skyLight, uint8_t blockLight,
                                      uint8_t ao, std::vector<Vertex>& verts,
                                      std::vector<uint32_t>& idxs) {
            // +Y face: y = ly+1 (minus the water-surface drop), CCW from above
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const float y = static_cast<float>(ly + 1) - topDrop;
            const QuadCorner corners[4] = {
                {static_cast<float>(col), y, static_cast<float>(row), 0.f, 0.f},
                {static_cast<float>(col), y, static_cast<float>(row + height), 0.f, fh},
                {static_cast<float>(col + width), y, static_cast<float>(row + height), fw, fh},
                {static_cast<float>(col + width), y, static_cast<float>(row), fw, 0.f},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridD, gridW, faceKeys, FaceNormal::PLUS_Y, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -Y (bottom) — visible when the block below doesn't hide it
    // ======================================================================
    for (int ly = 0; ly < gridH; ++ly) {
        std::fill_n(faceKeys, gridD * gridW, uint32_t{0});

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (visible(cur, getBlock(x, ly - 1, z))) {
                    faceKeys[idx(z, x, gridW)] =
                        packFaceKey(cur, lightAt(x, ly - 1, z), getBlockLight(x, ly - 1, z),
                                    bakeAO ? aoMinusY(x, ly - 1, z) : AO_ALL_OPEN);
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, uint8_t blockLight, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
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
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridD, gridW, faceKeys, FaceNormal::MINUS_Y, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: +X (right) — exposed when visible toward the +X neighbor
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        std::fill_n(faceKeys, gridH * gridD, uint32_t{0});

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx + 1, y, z))) {
                    faceKeys[idx(y, z, gridD)] =
                        packFaceKey(cur, lightAt(lx + 1, y, z), getBlockLight(lx + 1, y, z),
                                    bakeAO ? aoPlusX(lx + 1, y, z) : AO_ALL_OPEN);
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, uint8_t blockLight, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
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
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridH, gridD, faceKeys, FaceNormal::PLUS_X, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -X (left) — exposed when visible toward the -X neighbor
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        std::fill_n(faceKeys, gridH * gridD, uint32_t{0});

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx - 1, y, z))) {
                    faceKeys[idx(y, z, gridD)] =
                        packFaceKey(cur, lightAt(lx - 1, y, z), getBlockLight(lx - 1, y, z),
                                    bakeAO ? aoMinusX(lx - 1, y, z) : AO_ALL_OPEN);
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, uint8_t blockLight, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
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
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridH, gridD, faceKeys, FaceNormal::MINUS_X, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: +Z (front) — exposed when visible toward the +Z neighbor
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        std::fill_n(faceKeys, gridH * gridW, uint32_t{0});

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz + 1))) {
                    faceKeys[idx(y, x, gridW)] =
                        packFaceKey(cur, lightAt(x, y, lz + 1), getBlockLight(x, y, lz + 1),
                                    bakeAO ? aoPlusZ(x, y, lz + 1) : AO_ALL_OPEN);
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, uint8_t blockLight, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
            // +Z face: z = lz+1 (the old code emitted at lz, coplanar with
            // the block interior)
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(row), static_cast<float>(lz + 1), 0.f,
                 fh},
                {static_cast<float>(col + width), static_cast<float>(row),
                 static_cast<float>(lz + 1), fw, fh},
                {static_cast<float>(col + width), static_cast<float>(row + height),
                 static_cast<float>(lz + 1), fw, 0.f},
                {static_cast<float>(col), static_cast<float>(row + height),
                 static_cast<float>(lz + 1), 0.f, 0.f},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridH, gridW, faceKeys, FaceNormal::PLUS_Z, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -Z (back) — exposed when visible toward the -Z neighbor
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        std::fill_n(faceKeys, gridH * gridW, uint32_t{0});

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz - 1))) {
                    faceKeys[idx(y, x, gridW)] =
                        packFaceKey(cur, lightAt(x, y, lz - 1), getBlockLight(x, y, lz - 1),
                                    bakeAO ? aoMinusZ(x, y, lz - 1) : AO_ALL_OPEN);
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint8_t skyLight, uint8_t blockLight, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
            // -Z face: the face plane of block lz is z = lz (the old code
            // emitted at lz-1, one unit inside the neighbor)
            const float fw = static_cast<float>(width);
            const float fh = static_cast<float>(height);
            const QuadCorner corners[4] = {
                {static_cast<float>(col), static_cast<float>(row), static_cast<float>(lz), 0.f, fh},
                {static_cast<float>(col), static_cast<float>(row + height), static_cast<float>(lz),
                 0.f, 0.f},
                {static_cast<float>(col + width), static_cast<float>(row + height),
                 static_cast<float>(lz), fw, 0.f},
                {static_cast<float>(col + width), static_cast<float>(row), static_cast<float>(lz),
                 fw, fh},
            };
            pushQuad(verts, idxs, face, bt, skyLight, corners, ao, blockLight);
        };

        meshFaceGeneric(gridH, gridW, faceKeys, FaceNormal::MINUS_Z, outVertices, outIndices,
                        emitQuad);
    }
}

// Dense flora exposed a regular planting-row artifact because every accepted
// plant used the same two diagonals at the exact center of its block. Vary the
// visual pose from global coordinates without moving the generated anchor or
// consuming mutable randomness. Every value is a multiple of 1/32 so the
// resulting chunk-local positions stay exact in fp16.
struct FloraPose {
    float centerX;
    float centerZ;
    float axisX;
    float axisZ;
};

static FloraPose floraPose(int x, int z, int64_t worldX, int64_t worldZ, BlockType block) {
    constexpr uint64_t FLORA_POSE_SALT = 0x464C4F5241504F53ULL;
    constexpr std::array<std::array<int8_t, 2>, 4> AXES = {
        {{{5, 0}}, {{5, 3}}, {{5, 5}}, {{3, 5}}}};
    constexpr std::array<float, 4> JITTER = {-0.1875F, -0.0625F, 0.0625F, 0.1875F};

    const uint64_t poseHash =
        hash64(hash64(static_cast<uint64_t>(worldX) ^ FLORA_POSE_SALT) ^
               hash64(static_cast<uint64_t>(worldZ)) ^ (static_cast<uint64_t>(block) << 48U));
    const auto& axis = AXES[poseHash & 3U];
    return {
        .centerX = static_cast<float>(x) + 0.5F + JITTER[(poseHash >> 8U) & 3U],
        .centerZ = static_cast<float>(z) + 0.5F + JITTER[(poseHash >> 16U) & 3U],
        .axisX = static_cast<float>(axis[0]) * 0.0625F,
        .axisZ = static_cast<float>(axis[1]) * 0.0625F,
    };
}

// Two perpendicular vertical quads share one coordinate-hashed pose. Explicit
// reverse winding keeps every plant visible with back-face culling enabled.
static void emitFloraCross(int x, int y, int z, int64_t worldX, int64_t worldZ, BlockType bt,
                           uint8_t skyLight, uint8_t blockLight, std::vector<Vertex>& verts,
                           std::vector<uint32_t>& idxs) {
    const FloraPose pose = floraPose(x, z, worldX, worldZ, bt);
    const float y0 = static_cast<float>(y);
    const float y1 = static_cast<float>(y + 1);

    // v = 0 at the top because Metal texture coordinates run downward.
    const QuadCorner diagonalA[4] = {
        {pose.centerX - pose.axisX, y0, pose.centerZ - pose.axisZ, 0.f, 1.f},
        {pose.centerX + pose.axisX, y0, pose.centerZ + pose.axisZ, 1.f, 1.f},
        {pose.centerX + pose.axisX, y1, pose.centerZ + pose.axisZ, 1.f, 0.f},
        {pose.centerX - pose.axisX, y1, pose.centerZ - pose.axisZ, 0.f, 0.f},
    };
    const QuadCorner diagonalB[4] = {
        {pose.centerX + pose.axisZ, y0, pose.centerZ - pose.axisX, 0.f, 1.f},
        {pose.centerX - pose.axisZ, y0, pose.centerZ + pose.axisX, 1.f, 1.f},
        {pose.centerX - pose.axisZ, y1, pose.centerZ + pose.axisX, 1.f, 0.f},
        {pose.centerX + pose.axisZ, y1, pose.centerZ - pose.axisX, 0.f, 0.f},
    };
    // Flora is unshaded by AO (cross-quads have no face plane to occlude, and
    // the shader gives CROSS a fixed light); pass fully-open corners. Block
    // light still tints it so grass near lava glows.
    pushDoubleSidedQuad(verts, idxs, FaceNormal::CROSS, bt, skyLight, diagonalA, AO_ALL_OPEN,
                        blockLight);
    pushDoubleSidedQuad(verts, idxs, FaceNormal::CROSS, bt, skyLight, diagonalB, AO_ALL_OPEN,
                        blockLight);
}

static void emitFlatFlora(int x, int y, int z, BlockType bt, uint8_t skyLight, uint8_t blockLight,
                          std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
    const float surface = static_cast<float>(y) + 0.125f;
    const QuadCorner corners[4] = {
        {static_cast<float>(x), surface, static_cast<float>(z), 0.f, 0.f},
        {static_cast<float>(x), surface, static_cast<float>(z + 1), 0.f, 1.f},
        {static_cast<float>(x + 1), surface, static_cast<float>(z + 1), 1.f, 1.f},
        {static_cast<float>(x + 1), surface, static_cast<float>(z), 1.f, 0.f},
    };
    pushDoubleSidedQuad(verts, idxs, FaceNormal::PLUS_Y, bt, skyLight, corners, AO_ALL_OPEN,
                        blockLight);
}

static void markExteriorAir(const MeshSnapshot& snapshot, MeshScratch& scratch) {
    auto& exteriorAir = scratch.exteriorAir;
    auto& frontier = scratch.exteriorFrontier;
    exteriorAir.fill(0);
    size_t frontierRead = 0;
    size_t frontierWrite = 0;
    const int32_t cubeBaseY = snapshot.pos.y * CHUNK_EDGE;

    const auto enqueue = [&](int x, int y, int z) {
        const int index = MeshSnapshot::index(x, y, z);
        if (exteriorAir[static_cast<size_t>(index)] != 0 || isOpaque(snapshot.at(x, y, z))) {
            return;
        }
        exteriorAir[static_cast<size_t>(index)] = 1;
        frontier[frontierWrite++] = static_cast<uint16_t>(index);
    };

    // Generated cutoffs identify air with a direct path to the sky. Starting
    // from every such cell in the padded snapshot also admits outdoor air
    // through a loaded side halo, which is important beneath arches and
    // overhangs where a column-only cutoff would misclassify a cliff face as
    // an underground wall.
    for (int y = -1; y <= CHUNK_EDGE; ++y) {
        const int32_t worldY = cubeBaseY + y;
        for (int z = -1; z <= CHUNK_EDGE; ++z) {
            for (int x = -1; x <= CHUNK_EDGE; ++x) {
                int32_t cutoff = snapshot.generatedSurfaceCutoffAt(x, z);
                const int32_t skyCutoff = snapshot.skyCutoffAt(x, z);
                // A real edited or structure roof raises skyCutoff above the
                // generated surface and must remain an opaque barrier. The
                // SKY_CUTOFF_INCOMPLETE is instead the conservative marker for
                // an incomplete vertical load; use generated authority in that
                // case so outdoor streaming caps do not turn black.
                if (skyCutoff != MeshSnapshot::SKY_CUTOFF_UNKNOWN &&
                    skyCutoff != MeshSnapshot::SKY_CUTOFF_INCOMPLETE) {
                    cutoff = skyCutoff;
                }
                if (cutoff != MeshSnapshot::SKY_CUTOFF_UNKNOWN && worldY >= cutoff) {
                    enqueue(x, y, z);
                }
            }
        }
    }

    constexpr int directions[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    };
    while (frontierRead < frontierWrite) {
        const int linear = frontier[frontierRead++];
        const int paddedY = linear / (MeshSnapshot::PADDED_EDGE * MeshSnapshot::PADDED_EDGE);
        const int remainder = linear % (MeshSnapshot::PADDED_EDGE * MeshSnapshot::PADDED_EDGE);
        const int paddedZ = remainder / MeshSnapshot::PADDED_EDGE;
        const int paddedX = remainder % MeshSnapshot::PADDED_EDGE;
        const int x = paddedX - 1;
        const int y = paddedY - 1;
        const int z = paddedZ - 1;
        for (const auto& direction : directions) {
            const int nextX = x + direction[0];
            const int nextY = y + direction[1];
            const int nextZ = z + direction[2];
            if (nextX < -1 || nextX > CHUNK_EDGE || nextY < -1 || nextY > CHUNK_EDGE ||
                nextZ < -1 || nextZ > CHUNK_EDGE) {
                continue;
            }
            enqueue(nextX, nextY, nextZ);
        }
    }
}

static bool hasMissingLateralCapCandidate(const MeshSnapshot& snapshot) {
    const int32_t cubeBaseY = snapshot.pos.y * CHUNK_EDGE;
    const auto candidate = [&](uint8_t mask, int selfX, int selfZ, int neighborX, int neighborZ,
                               int y) {
        if ((snapshot.missingNeighborFaces & mask) == 0 || isOpaque(snapshot.at(selfX, y, selfZ))) {
            return false;
        }
        const int32_t neighborCutoff = snapshot.generatedSurfaceCutoffAt(neighborX, neighborZ);
        return neighborCutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN || cubeBaseY + y < neighborCutoff;
    };
    for (int y = 0; y < CHUNK_EDGE; ++y) {
        for (int coordinate = 0; coordinate < CHUNK_EDGE; ++coordinate) {
            if (candidate(MeshSnapshot::MISSING_PLUS_X, CHUNK_EDGE - 1, coordinate, CHUNK_EDGE,
                          coordinate, y) ||
                candidate(MeshSnapshot::MISSING_MINUS_X, 0, coordinate, -1, coordinate, y) ||
                candidate(MeshSnapshot::MISSING_PLUS_Z, coordinate, CHUNK_EDGE - 1, coordinate,
                          CHUNK_EDGE, y) ||
                candidate(MeshSnapshot::MISSING_MINUS_Z, coordinate, 0, coordinate, -1, y)) {
                return true;
            }
        }
    }
    return false;
}

static void emitMissingNeighborCaps(const MeshSnapshot& snapshot, MeshScratch& scratch,
                                    MeshOutput& output) {
    if (snapshot.missingNeighborFaces == 0) return;
    constexpr float edge = static_cast<float>(CHUNK_EDGE);
    const int32_t cubeBaseY = snapshot.pos.y * CHUNK_EDGE;
    constexpr uint8_t lateralFaces = MeshSnapshot::MISSING_PLUS_X | MeshSnapshot::MISSING_MINUS_X |
                                     MeshSnapshot::MISSING_PLUS_Z | MeshSnapshot::MISSING_MINUS_Z;
    if ((snapshot.missingNeighborFaces & lateralFaces) != 0 &&
        hasMissingLateralCapCandidate(snapshot)) {
        markExteriorAir(snapshot, scratch);
    }
    auto emit = [&](uint8_t mask, FaceNormal inwardFace, BlockType current, int32_t worldY,
                    int selfX, int selfZ, int neighborX, int neighborZ,
                    const QuadCorner(&corners)[4]) {
        if ((snapshot.missingNeighborFaces & mask) == 0 || isOpaque(current)) return;
        const int32_t neighborCutoff = snapshot.generatedSurfaceCutoffAt(neighborX, neighborZ);
        const bool neighborIsPlannedSolid =
            neighborCutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN || worldY < neighborCutoff;
        if (!neighborIsPlannedSolid) return;

        const bool lateralFace =
            inwardFace == FaceNormal::MINUS_X || inwardFace == FaceNormal::PLUS_X ||
            inwardFace == FaceNormal::MINUS_Z || inwardFace == FaceNormal::PLUS_Z;
        const int localY = worldY - cubeBaseY;
        const bool surfaceOpening =
            lateralFace && localY >= 0 && localY < CHUNK_EDGE &&
            scratch.exteriorAir[static_cast<size_t>(MeshSnapshot::index(selfX, localY, selfZ))] !=
                0;
        BlockType material = surfaceOpening
                                 ? snapshot.generatedSurfaceMaterialAt(neighborX, neighborZ)
                                 : (lateralFace ? BlockType::STONE : BlockType::BEDROCK);
        if (!rendersAsCube(material) || material == BlockType::WATER) material = BlockType::STONE;
        const uint8_t skyLight = surfaceOpening ? 15 : 0;
        const uint8_t occlusion = surfaceOpening || lateralFace ? AO_ALL_OPEN : 0;
        const uint8_t blockLight =
            surfaceOpening ? snapshot.lightAt(selfX, worldY - cubeBaseY, selfZ) : 0;
        pushQuad(output.vertices, output.indices, inwardFace, material, skyLight, corners,
                 occlusion, blockLight);
    };

    for (int y = 0; y < CHUNK_EDGE; ++y) {
        const float y0 = static_cast<float>(y);
        const float y1 = static_cast<float>(y + 1);
        const int32_t worldY = cubeBaseY + y;
        for (int coordinate = 0; coordinate < CHUNK_EDGE; ++coordinate) {
            const float c0 = static_cast<float>(coordinate);
            const float c1 = static_cast<float>(coordinate + 1);
            const QuadCorner plusX[4] = {
                {edge, y0, c0, 0.f, 1.f},
                {edge, y0, c1, 1.f, 1.f},
                {edge, y1, c1, 1.f, 0.f},
                {edge, y1, c0, 0.f, 0.f},
            };
            emit(MeshSnapshot::MISSING_PLUS_X, FaceNormal::MINUS_X,
                 snapshot.at(CHUNK_EDGE - 1, y, coordinate), worldY, CHUNK_EDGE - 1, coordinate,
                 CHUNK_EDGE, coordinate, plusX);

            const QuadCorner minusX[4] = {
                {0.f, y0, c0, 0.f, 1.f},
                {0.f, y1, c0, 0.f, 0.f},
                {0.f, y1, c1, 1.f, 0.f},
                {0.f, y0, c1, 1.f, 1.f},
            };
            emit(MeshSnapshot::MISSING_MINUS_X, FaceNormal::PLUS_X, snapshot.at(0, y, coordinate),
                 worldY, 0, coordinate, -1, coordinate, minusX);

            const QuadCorner plusZ[4] = {
                {c0, y0, edge, 0.f, 1.f},
                {c0, y1, edge, 0.f, 0.f},
                {c1, y1, edge, 1.f, 0.f},
                {c1, y0, edge, 1.f, 1.f},
            };
            emit(MeshSnapshot::MISSING_PLUS_Z, FaceNormal::MINUS_Z,
                 snapshot.at(coordinate, y, CHUNK_EDGE - 1), worldY, coordinate, CHUNK_EDGE - 1,
                 coordinate, CHUNK_EDGE, plusZ);

            const QuadCorner minusZ[4] = {
                {c0, y0, 0.f, 0.f, 1.f},
                {c1, y0, 0.f, 1.f, 1.f},
                {c1, y1, 0.f, 1.f, 0.f},
                {c0, y1, 0.f, 0.f, 0.f},
            };
            emit(MeshSnapshot::MISSING_MINUS_Z, FaceNormal::PLUS_Z, snapshot.at(coordinate, y, 0),
                 worldY, coordinate, 0, coordinate, -1, minusZ);
        }
    }

    for (int z = 0; z < CHUNK_EDGE; ++z) {
        const float z0 = static_cast<float>(z);
        const float z1 = static_cast<float>(z + 1);
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            const float x0 = static_cast<float>(x);
            const float x1 = static_cast<float>(x + 1);
            const QuadCorner plusY[4] = {
                {x0, edge, z0, 0.f, 0.f},
                {x1, edge, z0, 1.f, 0.f},
                {x1, edge, z1, 1.f, 1.f},
                {x0, edge, z1, 0.f, 1.f},
            };
            emit(MeshSnapshot::MISSING_PLUS_Y, FaceNormal::MINUS_Y,
                 snapshot.at(x, CHUNK_EDGE - 1, z), cubeBaseY + CHUNK_EDGE, x, z, x, z, plusY);

            const QuadCorner minusY[4] = {
                {x0, 0.f, z0, 0.f, 0.f},
                {x0, 0.f, z1, 0.f, 1.f},
                {x1, 0.f, z1, 1.f, 1.f},
                {x1, 0.f, z0, 1.f, 0.f},
            };
            emit(MeshSnapshot::MISSING_MINUS_Y, FaceNormal::PLUS_Y, snapshot.at(x, 0, z),
                 cubeBaseY - 1, x, z, x, z, minusY);
        }
    }
}

static float snapshotFluidHeight(const MeshSnapshot& snapshot, int x, int y, int z) {
    if (snapshot.at(x, y, z) != BlockType::WATER) return 0.0f;
    if (snapshot.at(x, y + 1, z) == BlockType::WATER) return 1.0f;
    return fluidSurfaceHeight(snapshot.fluidAt(x, y, z));
}

static float cornerFluidHeight(const MeshSnapshot& snapshot, int x, int y, int z, int cornerX,
                               int cornerZ) {
    const int cornerWorldX = x + cornerX;
    const int cornerWorldZ = z + cornerZ;
    float total = 0.0f;
    int samples = 0;
    for (int offsetZ = -1; offsetZ <= 0; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 0; ++offsetX) {
            const int sampleX = cornerWorldX + offsetX;
            const int sampleZ = cornerWorldZ + offsetZ;
            if (snapshot.at(sampleX, y, sampleZ) != BlockType::WATER) continue;
            if (snapshot.at(sampleX, y + 1, sampleZ) == BlockType::WATER) return 1.0f;
            total += snapshotFluidHeight(snapshot, sampleX, y, sampleZ);
            ++samples;
        }
    }
    return samples == 0 ? snapshotFluidHeight(snapshot, x, y, z)
                        : total / static_cast<float>(samples);
}

static uint8_t fluidFlowDirection(const MeshSnapshot& snapshot, int x, int y, int z) {
    const FluidState state = snapshot.fluidAt(x, y, z);
    if (state.isFalling()) return 0;
    const float center = snapshotFluidHeight(snapshot, x, y, z);
    constexpr std::array<std::array<int, 2>, 4> offsets{{
        {{-1, 0}},
        {{1, 0}},
        {{0, -1}},
        {{0, 1}},
    }};
    float lowest = center;
    uint8_t direction = 0;
    for (uint8_t i = 0; i < offsets.size(); ++i) {
        const int nx = x + offsets[i][0];
        const int nz = z + offsets[i][1];
        const float neighbor = snapshotFluidHeight(snapshot, nx, y, nz);
        if (neighbor > 0.0f && neighbor + 0.001f < lowest) {
            lowest = neighbor;
            direction = static_cast<uint8_t>(i + 1);
        }
    }
    return direction;
}

template <typename SkyAccess, typename LightAccess>
static void emitPartialWater(const MeshSnapshot& snapshot, MeshOutput& output,
                             const SkyAccess& skyLightAt, const LightAccess& blockLightAt) {
    for (int y = 0; y < CHUNK_EDGE; ++y) {
        for (int z = 0; z < CHUNK_EDGE; ++z) {
            for (int x = 0; x < CHUNK_EDGE; ++x) {
                if (snapshot.at(x, y, z) != BlockType::WATER) continue;
                const FluidState state = snapshot.fluidAt(x, y, z);
                const uint8_t flow = fluidFlowDirection(snapshot, x, y, z);
                const float h00 = cornerFluidHeight(snapshot, x, y, z, 0, 0);
                const float h10 = cornerFluidHeight(snapshot, x, y, z, 1, 0);
                const float h11 = cornerFluidHeight(snapshot, x, y, z, 1, 1);
                const float h01 = cornerFluidHeight(snapshot, x, y, z, 0, 1);
                const float baseY = static_cast<float>(y);

                if (snapshot.at(x, y + 1, z) != BlockType::WATER) {
                    const QuadCorner top[4] = {
                        {static_cast<float>(x), baseY + h00, static_cast<float>(z), 0.f, 0.f},
                        {static_cast<float>(x), baseY + h01, static_cast<float>(z + 1), 0.f, 1.f},
                        {static_cast<float>(x + 1), baseY + h11, static_cast<float>(z + 1), 1.f,
                         1.f},
                        {static_cast<float>(x + 1), baseY + h10, static_cast<float>(z), 1.f, 0.f},
                    };
                    pushFluidQuad(output.vertices, output.indices, FaceNormal::PLUS_Y,
                                  skyLightAt(x, y + 1, z), blockLightAt(x, y + 1, z), flow,
                                  state.isFalling(), top);
                }

                auto emitSide = [&](FaceNormal face, int neighborX, int neighborY, int neighborZ,
                                    const QuadCorner(&corners)[4]) {
                    const BlockType neighbor = snapshot.at(neighborX, neighborY, neighborZ);
                    if (neighbor == BlockType::WATER || isOpaque(neighbor)) return;
                    pushFluidQuad(output.vertices, output.indices, face,
                                  skyLightAt(neighborX, neighborY, neighborZ),
                                  blockLightAt(neighborX, neighborY, neighborZ), flow,
                                  state.isFalling(), corners);
                };
                // Stable source and horizontal-flow surfaces end at the bank;
                // their analytical terrain owns the shoreline. Full-height
                // side sheets are reserved for explicit falling columns, so
                // lake, river, ocean, cube, and LOD edges cannot become walls
                // of water when the adjacent ground is lower.
                if (state.isFalling()) {
                    const QuadCorner west[4] = {
                        {static_cast<float>(x), baseY, static_cast<float>(z), 0.f, 1.f},
                        {static_cast<float>(x), baseY, static_cast<float>(z + 1), 1.f, 1.f},
                        {static_cast<float>(x), baseY + h01, static_cast<float>(z + 1), 1.f, 0.f},
                        {static_cast<float>(x), baseY + h00, static_cast<float>(z), 0.f, 0.f},
                    };
                    emitSide(FaceNormal::MINUS_X, x - 1, y, z, west);
                    const QuadCorner east[4] = {
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z + 1), 0.f, 1.f},
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z), 1.f, 1.f},
                        {static_cast<float>(x + 1), baseY + h10, static_cast<float>(z), 1.f, 0.f},
                        {static_cast<float>(x + 1), baseY + h11, static_cast<float>(z + 1), 0.f,
                         0.f},
                    };
                    emitSide(FaceNormal::PLUS_X, x + 1, y, z, east);
                    const QuadCorner north[4] = {
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z), 0.f, 1.f},
                        {static_cast<float>(x), baseY, static_cast<float>(z), 1.f, 1.f},
                        {static_cast<float>(x), baseY + h00, static_cast<float>(z), 1.f, 0.f},
                        {static_cast<float>(x + 1), baseY + h10, static_cast<float>(z), 0.f, 0.f},
                    };
                    emitSide(FaceNormal::MINUS_Z, x, y, z - 1, north);
                    const QuadCorner south[4] = {
                        {static_cast<float>(x), baseY, static_cast<float>(z + 1), 0.f, 1.f},
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z + 1), 1.f, 1.f},
                        {static_cast<float>(x + 1), baseY + h11, static_cast<float>(z + 1), 1.f,
                         0.f},
                        {static_cast<float>(x), baseY + h01, static_cast<float>(z + 1), 0.f, 0.f},
                    };
                    emitSide(FaceNormal::PLUS_Z, x, y, z + 1, south);
                }

                if (snapshot.at(x, y - 1, z) != BlockType::WATER &&
                    !isOpaque(snapshot.at(x, y - 1, z))) {
                    const QuadCorner bottom[4] = {
                        {static_cast<float>(x), baseY, static_cast<float>(z), 0.f, 0.f},
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z), 1.f, 0.f},
                        {static_cast<float>(x + 1), baseY, static_cast<float>(z + 1), 1.f, 1.f},
                        {static_cast<float>(x), baseY, static_cast<float>(z + 1), 0.f, 1.f},
                    };
                    pushFluidQuad(output.vertices, output.indices, FaceNormal::MINUS_Y,
                                  skyLightAt(x, y - 1, z), blockLightAt(x, y - 1, z), flow,
                                  state.isFalling(), bottom);
                }
            }
        }
    }
}

template <typename Access, typename LightAccess, typename SkyCutoffAccess>
static MeshOutput buildGenericMesh(int gridW, int gridH, int gridD, const Access& getBlock,
                                   const LightAccess& getBlockLight,
                                   const SkyCutoffAccess& getSkyCutoff, int64_t worldBaseX,
                                   int32_t sectionBaseY, int64_t worldBaseZ, bool padded,
                                   bool emitFlora, bool emitGreedyWater,
                                   const MeshSnapshot* partialWater, MeshScratch& scratch) {
    MeshOutput output;

    // ---- Column skylight ----
    // The first open Y above the topmost OPAQUE block per column, computed
    // over the padded ring too so border faces read their real neighbor
    // column instead of a clamped copy (that clamp painted a visible light
    // seam along every chunk edge). Only opaque blocks block the sky: a tree
    // canopy is non-opaque leaves, so it must NOT darken the ground below with
    // a fake column shadow — the real cascade shadow does that, and doubling
    // them put a second shadow directly under every tree. Genuine cover
    // (opaque stone/dirt: caves, overhangs) blocks direct skylight; open sky
    // is 15. Horizontal sky propagation is not synthesized through missing
    // cubes, so unavailable cover stays conservatively dark.
    const int ringW = gridW + 2;
    const int ringD = gridD + 2;
    assert(ringW * ringD <= static_cast<int>(scratch.skyHeight.size()));
    int32_t* skyHeight = scratch.skyHeight.data();
    std::fill_n(skyHeight, ringW * ringD, int32_t{0});
    for (int z = -1; z <= gridD; ++z) {
        for (int x = -1; x <= gridW; ++x) {
            int32_t cutoffY = getSkyCutoff(x, z);
            if (cutoffY == MeshSnapshot::SKY_CUTOFF_UNKNOWN) {
                cutoffY = sectionBaseY;
                for (int y = gridH; y >= 0; --y) {
                    if (isOpaque(getBlock(x, y, z))) {
                        cutoffY = sectionBaseY + y + 1;
                        break;
                    }
                }
            }
            skyHeight[idx(z + 1, x + 1, ringW)] = cutoffY - sectionBaseY;
        }
    }
    auto lightAt = [&](int x, int y, int z) -> uint8_t {
        const int32_t depth = skyHeight[idx(z + 1, x + 1, ringW)] - y;
        return depth <= 0 ? 15 : 0;
    };

    // ---- Opaque section: cubes (with baked corner AO), then flora crosses ----
    runGreedyPasses(gridW, gridH, gridD, getBlock, getBlockLight, cubeFaceVisible, lightAt, 0.f,
                    /*bakeAO=*/true, scratch, output.vertices, output.indices);

    if (emitFlora) {
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                for (int y = 0; y < gridH; ++y) {
                    BlockType bt = getBlock(x, y, z);
                    if (blockDefinition(bt).renderShape == BlockRenderShape::CROSS) {
                        emitFloraCross(x, y, z, worldBaseX + x, worldBaseZ + z, bt,
                                       lightAt(x, y, z), getBlockLight(x, y, z), output.vertices,
                                       output.indices);
                    } else if (blockDefinition(bt).renderShape == BlockRenderShape::FLAT) {
                        emitFlatFlora(x, y, z, bt, lightAt(x, y, z), getBlockLight(x, y, z),
                                      output.vertices, output.indices);
                    }
                }
            }
        }
    }
    if (partialWater != nullptr) emitMissingNeighborCaps(*partialWater, scratch, output);

    // ---- Water section: everything after this index draws in the water
    // pass. Padded builds read real neighbor water; unpadded builds assume
    // water continues past the edge (oceans virtually always do, and a wall
    // there painted phantom stripes along every chunk border).
    output.opaqueIndexCount = static_cast<uint32_t>(output.indices.size());
    if (partialWater != nullptr) {
        emitPartialWater(*partialWater, output, lightAt, getBlockLight);
        return output;
    }
    if (!emitGreedyWater) return output;
    constexpr float SOURCE_TOP_DROP = 1.0F - fluidSurfaceHeight(FluidState::source());
    if (padded) {
        runGreedyPasses(gridW, gridH, gridD, getBlock, getBlockLight, waterFaceVisible, lightAt,
                        SOURCE_TOP_DROP, /*bakeAO=*/false, scratch, output.vertices,
                        output.indices);
    } else {
        auto waterEdgeBlock = [&getBlock, gridW, gridD](int x, int y, int z) -> BlockType {
            if (x < 0 || x >= gridW || z < 0 || z >= gridD) return BlockType::WATER;
            return getBlock(x, y, z);
        };
        runGreedyPasses(gridW, gridH, gridD, waterEdgeBlock, getBlockLight, waterFaceVisible,
                        lightAt, SOURCE_TOP_DROP, /*bakeAO=*/false, scratch, output.vertices,
                        output.indices);
    }

    return output;
}

template <int SCALE>
static BlockType dominantDownsampledBlock(const Chunk& chunk, int cellX, int cellY, int cellZ) {
    std::array<uint8_t, BLOCK_TYPE_COUNT> counts{};
    const int baseX = cellX * SCALE;
    const int baseY = cellY * SCALE;
    const int baseZ = cellZ * SCALE;
    for (int dz = 0; dz < SCALE; ++dz) {
        for (int dy = 0; dy < SCALE; ++dy) {
            for (int dx = 0; dx < SCALE; ++dx) {
                BlockType block = chunk.getBlock(baseX + dx, baseY + dy, baseZ + dz);
                if (isFlora(block)) block = BlockType::AIR;
                ++counts[static_cast<size_t>(block)];
            }
        }
    }

    size_t dominantIndex = 0;
    for (size_t index = 1; index < counts.size(); ++index) {
        if (counts[index] > counts[dominantIndex]) dominantIndex = index;
    }
    return static_cast<BlockType>(dominantIndex);
}

// ==========================================================================
// LODMesher implementation
// ==========================================================================

MeshOutput LODMesher::buildMesh(const MeshSnapshot& snapshot, MeshScratch& scratch) {
    auto blockFn = [&snapshot](int x, int y, int z) -> BlockType { return snapshot.at(x, y, z); };
    auto lightFn = [&snapshot](int x, int y, int z) -> uint8_t {
        return snapshot.lightAt(x, y, z);
    };
    auto skyFn = [&snapshot](int x, int z) -> int32_t { return snapshot.skyCutoffAt(x, z); };
    return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn, lightFn, skyFn,
                            snapshot.pos.x * CHUNK_EDGE, snapshot.pos.y * CHUNK_EDGE,
                            snapshot.pos.z * CHUNK_EDGE, /*padded=*/true,
                            /*emitFlora=*/true, /*emitGreedyWater=*/false, &snapshot, scratch);
}

MeshOutput LODMesher::buildMesh(const Chunk& chunk, int lodLevel) {
    // Beyond render distance — return empty mesh (distance culling)
    if (lodLevel >= static_cast<int>(ChunkLOD::COUNT)) {
        return MeshOutput{};
    }

    thread_local MeshScratch scratch;
    const auto noSkyCutoff = [](int, int) -> int32_t { return MeshSnapshot::SKY_CUTOFF_UNKNOWN; };

    switch (static_cast<ChunkLOD>(lodLevel)) {
        case ChunkLOD::FULL: {
            // LOD 0: Full resolution greedy meshing (16x16x16),
            // neighbor-blind (tests and tools; the game uses the
            // MeshSnapshot overload)
            auto blockFn = [&chunk](int x, int y, int z) -> BlockType {
                return chunk.getBlock(x, y, z);
            };
            auto lightFn = [&chunk](int x, int y, int z) -> uint8_t {
                return chunk.getBlockLight(x, y, z);
            };
            return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn, lightFn,
                                    noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                                    chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE,
                                    /*padded=*/false, /*emitFlora=*/true,
                                    /*emitGreedyWater=*/true, nullptr, scratch);
        }

        case ChunkLOD::MEDIUM: {
            // LOD 1: 2x downsampling. Flora does not contribute to the
            // dominant block in a coarse cell.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                return dominantDownsampledBlock<2>(chunk, cx, cy, cz);
            };
            return buildGenericMesh(
                8, 8, 8, blockFn, noBlockLight, noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE, /*padded=*/false,
                /*emitFlora=*/false, /*emitGreedyWater=*/true, nullptr, scratch);
        }

        case ChunkLOD::COARSE: {
            // LOD 2: 4x downsampling.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                return dominantDownsampledBlock<4>(chunk, cx, cy, cz);
            };
            return buildGenericMesh(
                4, 4, 4, blockFn, noBlockLight, noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE, /*padded=*/false,
                /*emitFlora=*/false, /*emitGreedyWater=*/true, nullptr, scratch);
        }

        default:
            return MeshOutput{};
    }
}

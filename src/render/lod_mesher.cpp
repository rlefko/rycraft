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
// Generic greedy mesher, works with any grid dimensions.
//
// The block accessor is a template parameter (a chunk read used to go
// through std::function, ~1M indirect calls per full build, the mesher's
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
// the neighbor doesn't fully hide it (isOpaque), and the two blocks differ,
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

// Pack/unpack four 4-bit corner light values (emit-corner order) in a uint16.
// One packing serves both skylight and block light. Smooth lighting writes a
// distinct per-corner nibble into each of a quad's four vertices; the
// rasterizer then interpolates it across the face.
inline static uint16_t packCornerLight(uint8_t l0, uint8_t l1, uint8_t l2, uint8_t l3) {
    return static_cast<uint16_t>((l0 & 0x0Fu) | ((l1 & 0x0Fu) << 4u) | ((l2 & 0x0Fu) << 8u) |
                                 ((l3 & 0x0Fu) << 12u));
}
inline static uint8_t cornerLightAt(uint16_t packed, int corner) {
    return static_cast<uint8_t>((packed >> (corner * 4)) & 0x0Fu);
}
// Same light on all four corners, for faces that stay flat (flora, water
// sides, missing-neighbor caps, and the coarse LODs' light-less passes).
inline static uint16_t broadcastLight(uint8_t light) {
    return packCornerLight(light, light, light, light);
}

// One corner's light: average the light over the cells touching the corner in
// the outward plane, excluding opaque cells (they store no propagated light,
// so counting them as zero would wrongly darken corners against a wall). The
// face cell is transparent by construction and always counts; the diagonal is
// dropped when both sides are opaque, mirroring the AO short-circuit so light
// cannot leak around a solid corner.
inline static uint8_t avgCornerLight(uint8_t face, uint8_t side1, uint8_t side2, uint8_t diagonal,
                                     bool side1Opaque, bool side2Opaque, bool diagonalOpaque) {
    uint32_t sum = face;
    uint32_t count = 1;
    if (!side1Opaque) {
        sum += side1;
        ++count;
    }
    if (!side2Opaque) {
        sum += side2;
        ++count;
    }
    if (!diagonalOpaque && !(side1Opaque && side2Opaque)) {
        sum += diagonal;
        ++count;
    }
    return static_cast<uint8_t>(sum / count);
}

// The three per-vertex terms a face contributes to the greedy key: four 2-bit
// AO corners, four 4-bit skylight corners, and four 4-bit block-light corners.
struct FaceCorners {
    uint8_t packedAO;
    uint16_t packedSky;
    uint16_t packedBlock;
};

// One outward-plane cell as the mesher reads it: propagated skylight and block
// light plus opacity, gathered once so AO and both light channels share the
// same nine reads per exposed face.
struct CellLight {
    uint8_t sky;
    uint8_t block;
    bool opaque;
};

// One corner's AO and averaged light from the face cell and its two side cells
// and diagonal. side1/side2/diagonal follow the same order aoVertex expects.
inline static std::array<uint8_t, 3> cornerTerms(const CellLight& face, const CellLight& side1,
                                                 const CellLight& side2,
                                                 const CellLight& diagonal) {
    return {aoVertex(side1.opaque, side2.opaque, diagonal.opaque),
            avgCornerLight(face.sky, side1.sky, side2.sky, diagonal.sky, side1.opaque, side2.opaque,
                           diagonal.opaque),
            avgCornerLight(face.block, side1.block, side2.block, diagonal.block, side1.opaque,
                           side2.opaque, diagonal.opaque)};
}

// Fold the four corner terms (emit order) into the packed AO/sky/block trio.
inline static FaceCorners packFaceCorners(const std::array<uint8_t, 3>& c0,
                                          const std::array<uint8_t, 3>& c1,
                                          const std::array<uint8_t, 3>& c2,
                                          const std::array<uint8_t, 3>& c3) {
    return {packAO(c0[0], c1[0], c2[0], c3[0]), packCornerLight(c0[1], c1[1], c2[1], c3[1]),
            packCornerLight(c0[2], c1[2], c2[2], c3[2])};
}

// Block-light accessor for the coarse LODs, which downsample block types only
// and carry no baked block light.
inline static uint8_t noBlockLight(int, int, int) {
    return 0;
}

inline static uint8_t noSkyLight(int, int, int) {
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
                     BlockType bt, uint16_t packedSky, const QuadCorner (&corners)[4],
                     uint8_t packedAO, uint16_t packedBlock) {
    reserveInitialMeshStorage(verts, idxs);
    const bool flip = (cornerAOAt(packedAO, 0) + cornerAOAt(packedAO, 2)) <
                      (cornerAOAt(packedAO, 1) + cornerAOAt(packedAO, 3));
    const uint8_t layer = textureLayerFor(bt, face);
    for (int index = 0; index < 4; ++index) {
        const int cornerIndex = flip ? (index + 1) & 3 : index;
        const uint32_t attr = packFaceAttr(
            face, layer, cornerLightAt(packedSky, cornerIndex), cornerAOAt(packedAO, cornerIndex),
            cornerLightAt(packedBlock, cornerIndex), isEmissive(bt), swayClass(bt));
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
    pushQuad(verts, idxs, face, bt, broadcastLight(skyLight), corners, packedAO,
             broadcastLight(blockLight));
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
                          bool exteriorSky, const QuadCorner (&corners)[4]) {
    reserveInitialMeshStorage(verts, idxs);
    const uint32_t attr =
        packFluidFaceAttr(face, skyLight, flowDirection, falling, blockLight, exteriorSky);
    for (const QuadCorner& corner : corners)
        verts.push_back(makeVertex(attr, corner));
    const uint32_t base = static_cast<uint32_t>(verts.size()) - 4;
    idxs.insert(idxs.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

// Same as pushFluidQuad but with a distinct skylight nibble per corner, so the
// water surface's ambient blends smoothly under a cave mouth or overhang the
// way the opaque faces now do. Corner nibbles follow the quad's corner order.
static void pushFluidQuadSmoothSky(std::vector<Vertex>& verts, std::vector<uint32_t>& idxs,
                                   FaceNormal face, uint16_t packedSky, uint8_t blockLight,
                                   uint8_t flowDirection, bool falling, bool exteriorSky,
                                   const QuadCorner (&corners)[4]) {
    reserveInitialMeshStorage(verts, idxs);
    for (int corner = 0; corner < 4; ++corner) {
        const uint32_t attr = packFluidFaceAttr(face, cornerLightAt(packedSky, corner),
                                                flowDirection, falling, blockLight, exteriorSky);
        verts.push_back(makeVertex(attr, corners[corner]));
    }
    const uint32_t base = static_cast<uint32_t>(verts.size()) - 4;
    idxs.insert(idxs.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

// Greedy merge on a face plane of arbitrary dimensions.
//
// A nonzero key combines every property that must remain constant across a
// greedy quad. Consumed rectangles are cleared in place.
// The key is 64-bit: block identity in bits 0-7, the four 2-bit AO corners in
// bits 8-15, the four 4-bit skylight corners in bits 16-31, and the four 4-bit
// block-light corners in bits 32-47. Two cells merge only when every per-corner
// value matches, so a light gradient keeps its per-vertex detail instead of
// bleeding across one large merged quad.
static uint64_t packFaceKey(BlockType block, uint16_t packedSky, uint16_t packedBlock,
                            uint8_t packedAO) {
    return (static_cast<uint64_t>(block) + 1ULL) | (static_cast<uint64_t>(packedAO) << 8U) |
           (static_cast<uint64_t>(packedSky) << 16U) | (static_cast<uint64_t>(packedBlock) << 32U);
}

static uint64_t packFaceKey(BlockType block, const FaceCorners& corners) {
    return packFaceKey(block, corners.packedSky, corners.packedBlock, corners.packedAO);
}

static BlockType faceKeyBlock(uint64_t key) {
    return static_cast<BlockType>((key & 0xFFULL) - 1ULL);
}

static uint16_t faceKeyCornerSky(uint64_t key) {
    return static_cast<uint16_t>((key >> 16U) & 0xFFFFULL);
}

static uint16_t faceKeyCornerBlock(uint64_t key) {
    return static_cast<uint16_t>((key >> 32U) & 0xFFFFULL);
}

static uint8_t faceKeyAO(uint64_t key) {
    return static_cast<uint8_t>((key >> 8U) & 0xFFULL);
}

static_assert(BLOCK_TYPE_COUNT <= 255);

static void meshFaceGeneric(int faceHeight, int faceWidth, uint64_t* faceKeys, FaceNormal face,
                            std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                            const auto& emitQuadFn) {
    for (int row = 0; row < faceHeight; ++row) {
        for (int col = 0; col < faceWidth; ++col) {
            const int i = idx(row, col, faceWidth);
            const uint64_t leadKey = faceKeys[i];
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
                std::fill_n(faceKeys + idx(row + dr, col, faceWidth), width, uint64_t{0});
            }

            emitQuadFn(col, row, width, height, face, faceKeyBlock(leadKey),
                       faceKeyCornerSky(leadKey), faceKeyCornerBlock(leadKey), faceKeyAO(leadKey),
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
    uint64_t* faceKeys = scratch.faceKeys.data();
    assert(gridW * gridD <= static_cast<int>(scratch.faceKeys.size()));
    assert(gridH * gridD <= static_cast<int>(scratch.faceKeys.size()));
    assert(gridH * gridW <= static_cast<int>(scratch.faceKeys.size()));

    // Each face reads the nine cells in the plane one step along its outward
    // normal (the face cell plus its eight neighbors) exactly once, folding
    // them into per-corner AO and smoothed skylight and block light in that
    // face's emit order. Opaque cells cast AO (isOpaque, so leaves don't) and
    // are excluded from the light average. Unexposed cells and the water pass
    // pass bake=false, keeping AO_ALL_OPEN and broadcasting the face cell's
    // light to all four corners (no darkening, no smoothing).
    auto cell = [&](int gx, int gy, int gz) -> CellLight {
        return {lightAt(gx, gy, gz), getBlockLight(gx, gy, gz), isOpaque(getBlock(gx, gy, gz))};
    };
    auto flatCorners = [&](int cx, int cy, int cz) -> FaceCorners {
        return {AO_ALL_OPEN, broadcastLight(lightAt(cx, cy, cz)),
                broadcastLight(getBlockLight(cx, cy, cz))};
    };
    auto cornersPlusY = [&](int cx, int py, int cz, bool bake) -> FaceCorners {
        if (!bake) return flatCorners(cx, py, cz);
        const CellLight f = cell(cx, py, cz);
        const CellLight xm = cell(cx - 1, py, cz), xp = cell(cx + 1, py, cz);
        const CellLight zm = cell(cx, py, cz - 1), zp = cell(cx, py, cz + 1);
        const CellLight mm = cell(cx - 1, py, cz - 1), pm = cell(cx + 1, py, cz - 1);
        const CellLight pp = cell(cx + 1, py, cz + 1), mp = cell(cx - 1, py, cz + 1);
        return packFaceCorners(cornerTerms(f, xm, zm, mm), cornerTerms(f, xm, zp, mp),
                               cornerTerms(f, xp, zp, pp), cornerTerms(f, xp, zm, pm));
    };
    auto cornersMinusY = [&](int cx, int py, int cz, bool bake) -> FaceCorners {
        if (!bake) return flatCorners(cx, py, cz);
        const CellLight f = cell(cx, py, cz);
        const CellLight xm = cell(cx - 1, py, cz), xp = cell(cx + 1, py, cz);
        const CellLight zm = cell(cx, py, cz - 1), zp = cell(cx, py, cz + 1);
        const CellLight mm = cell(cx - 1, py, cz - 1), pm = cell(cx + 1, py, cz - 1);
        const CellLight pp = cell(cx + 1, py, cz + 1), mp = cell(cx - 1, py, cz + 1);
        return packFaceCorners(cornerTerms(f, xm, zm, mm), cornerTerms(f, xp, zm, pm),
                               cornerTerms(f, xp, zp, pp), cornerTerms(f, xm, zp, mp));
    };
    auto cornersPlusX = [&](int px, int cy, int cz, bool bake) -> FaceCorners { // outward plane px
        if (!bake) return flatCorners(px, cy, cz);
        const CellLight f = cell(px, cy, cz);
        const CellLight ym = cell(px, cy - 1, cz), yp = cell(px, cy + 1, cz);
        const CellLight zm = cell(px, cy, cz - 1), zp = cell(px, cy, cz + 1);
        const CellLight mm = cell(px, cy - 1, cz - 1), pm = cell(px, cy + 1, cz - 1);
        const CellLight pp = cell(px, cy + 1, cz + 1), mp = cell(px, cy - 1, cz + 1);
        return packFaceCorners(cornerTerms(f, ym, zm, mm), cornerTerms(f, yp, zm, pm),
                               cornerTerms(f, yp, zp, pp), cornerTerms(f, ym, zp, mp));
    };
    auto cornersMinusX = [&](int px, int cy, int cz, bool bake) -> FaceCorners { // outward plane px
        if (!bake) return flatCorners(px, cy, cz);
        const CellLight f = cell(px, cy, cz);
        const CellLight ym = cell(px, cy - 1, cz), yp = cell(px, cy + 1, cz);
        const CellLight zm = cell(px, cy, cz - 1), zp = cell(px, cy, cz + 1);
        const CellLight mm = cell(px, cy - 1, cz - 1), pm = cell(px, cy + 1, cz - 1);
        const CellLight pp = cell(px, cy + 1, cz + 1), mp = cell(px, cy - 1, cz + 1);
        return packFaceCorners(cornerTerms(f, ym, zm, mm), cornerTerms(f, ym, zp, mp),
                               cornerTerms(f, yp, zp, pp), cornerTerms(f, yp, zm, pm));
    };
    auto cornersPlusZ = [&](int cx, int cy, int pz, bool bake) -> FaceCorners { // outward plane pz
        if (!bake) return flatCorners(cx, cy, pz);
        const CellLight f = cell(cx, cy, pz);
        const CellLight xm = cell(cx - 1, cy, pz), xp = cell(cx + 1, cy, pz);
        const CellLight ym = cell(cx, cy - 1, pz), yp = cell(cx, cy + 1, pz);
        const CellLight mm = cell(cx - 1, cy - 1, pz), pm = cell(cx + 1, cy - 1, pz);
        const CellLight pp = cell(cx + 1, cy + 1, pz), mp = cell(cx - 1, cy + 1, pz);
        return packFaceCorners(cornerTerms(f, xm, ym, mm), cornerTerms(f, xp, ym, pm),
                               cornerTerms(f, xp, yp, pp), cornerTerms(f, xm, yp, mp));
    };
    auto cornersMinusZ = [&](int cx, int cy, int pz, bool bake) -> FaceCorners { // outward plane pz
        if (!bake) return flatCorners(cx, cy, pz);
        const CellLight f = cell(cx, cy, pz);
        const CellLight xm = cell(cx - 1, cy, pz), xp = cell(cx + 1, cy, pz);
        const CellLight ym = cell(cx, cy - 1, pz), yp = cell(cx, cy + 1, pz);
        const CellLight mm = cell(cx - 1, cy - 1, pz), pm = cell(cx + 1, cy - 1, pz);
        const CellLight pp = cell(cx + 1, cy + 1, pz), mp = cell(cx - 1, cy + 1, pz);
        return packFaceCorners(cornerTerms(f, xm, ym, mm), cornerTerms(f, xm, yp, mp),
                               cornerTerms(f, xp, yp, pp), cornerTerms(f, xp, ym, pm));
    };

    // ======================================================================
    // Face: +Y (top), visible when the block above doesn't hide it
    // (the world's top layer reads air above and gets a lid)
    // ======================================================================
    for (int ly = 0; ly < gridH; ++ly) {
        std::fill_n(faceKeys, gridD * gridW, uint64_t{0});

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (visible(cur, getBlock(x, ly + 1, z))) {
                    faceKeys[idx(z, x, gridW)] =
                        packFaceKey(cur, cornersPlusY(x, ly + 1, z, bakeAO));
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly, topDrop](int col, int row, int width, int height, FaceNormal face,
                                      BlockType bt, uint16_t packedSky, uint16_t packedBlock,
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
        };

        meshFaceGeneric(gridD, gridW, faceKeys, FaceNormal::PLUS_Y, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -Y (bottom), visible when the block below doesn't hide it
    // ======================================================================
    for (int ly = 0; ly < gridH; ++ly) {
        std::fill_n(faceKeys, gridD * gridW, uint64_t{0});

        bool anyExposed = false;
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                BlockType cur = getBlock(x, ly, z);
                if (visible(cur, getBlock(x, ly - 1, z))) {
                    faceKeys[idx(z, x, gridW)] =
                        packFaceKey(cur, cornersMinusY(x, ly - 1, z, bakeAO));
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        auto emitQuad = [ly](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint16_t packedSky, uint16_t packedBlock, uint8_t ao,
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
        };

        meshFaceGeneric(gridD, gridW, faceKeys, FaceNormal::MINUS_Y, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: +X (right), exposed when visible toward the +X neighbor
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        std::fill_n(faceKeys, gridH * gridD, uint64_t{0});

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx + 1, y, z))) {
                    faceKeys[idx(y, z, gridD)] =
                        packFaceKey(cur, cornersPlusX(lx + 1, y, z, bakeAO));
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint16_t packedSky, uint16_t packedBlock, uint8_t ao,
                             std::vector<Vertex>& verts, std::vector<uint32_t>& idxs) {
            // +X face: x = lx+1, CCW from +X (rows are Y, cols are Z).
            // Texture v runs downward in Metal, so the TOP of the face
            // carries v=0, otherwise side textures (the grass strip)
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
        };

        meshFaceGeneric(gridH, gridD, faceKeys, FaceNormal::PLUS_X, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -X (left), exposed when visible toward the -X neighbor
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        std::fill_n(faceKeys, gridH * gridD, uint64_t{0});

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                BlockType cur = getBlock(lx, y, z);
                if (visible(cur, getBlock(lx - 1, y, z))) {
                    faceKeys[idx(y, z, gridD)] =
                        packFaceKey(cur, cornersMinusX(lx - 1, y, z, bakeAO));
                }
            }
        }

        auto emitQuad = [lx](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint16_t packedSky, uint16_t packedBlock, uint8_t ao,
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
        };

        meshFaceGeneric(gridH, gridD, faceKeys, FaceNormal::MINUS_X, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: +Z (front), exposed when visible toward the +Z neighbor
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        std::fill_n(faceKeys, gridH * gridW, uint64_t{0});

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz + 1))) {
                    faceKeys[idx(y, x, gridW)] =
                        packFaceKey(cur, cornersPlusZ(x, y, lz + 1, bakeAO));
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint16_t packedSky, uint16_t packedBlock, uint8_t ao,
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
        };

        meshFaceGeneric(gridH, gridW, faceKeys, FaceNormal::PLUS_Z, outVertices, outIndices,
                        emitQuad);
    }

    // ======================================================================
    // Face: -Z (back), exposed when visible toward the -Z neighbor
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        std::fill_n(faceKeys, gridH * gridW, uint64_t{0});

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                BlockType cur = getBlock(x, y, lz);
                if (visible(cur, getBlock(x, y, lz - 1))) {
                    faceKeys[idx(y, x, gridW)] =
                        packFaceKey(cur, cornersMinusZ(x, y, lz - 1, bakeAO));
                }
            }
        }

        auto emitQuad = [lz](int col, int row, int width, int height, FaceNormal face, BlockType bt,
                             uint16_t packedSky, uint16_t packedBlock, uint8_t ao,
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
            pushQuad(verts, idxs, face, bt, packedSky, corners, ao, packedBlock);
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
        const uint8_t skyLight = surfaceOpening
                                     ? (snapshot.derivedSkyLightValid
                                            ? snapshot.skyLightAt(selfX, worldY - cubeBaseY, selfZ)
                                            : 15)
                                     : 0;
        const uint8_t occlusion = surfaceOpening || lateralFace ? AO_ALL_OPEN : 0;
        const uint8_t blockLight =
            surfaceOpening ? snapshot.blockLightAt(selfX, worldY - cubeBaseY, selfZ) : 0;
        pushQuad(output.vertices, output.indices, inwardFace, material, broadcastLight(skyLight),
                 corners, occlusion, broadcastLight(blockLight));
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

// Skylight just above the water surface, averaged over the (up to four) columns
// meeting a top-face corner, so the water surface's ambient blends smoothly the
// way the opaque faces now do. Opaque columns above the surface contribute no
// skylight and are skipped, mirroring the corner-light rule for solid faces.
static uint8_t cornerSurfaceSkyLight(const MeshSnapshot& snapshot, int x, int y, int z, int cornerX,
                                     int cornerZ) {
    const int cornerWorldX = x + cornerX;
    const int cornerWorldZ = z + cornerZ;
    uint32_t total = 0;
    int samples = 0;
    for (int offsetZ = -1; offsetZ <= 0; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 0; ++offsetX) {
            const int sampleX = cornerWorldX + offsetX;
            const int sampleZ = cornerWorldZ + offsetZ;
            if (isOpaque(snapshot.at(sampleX, y, sampleZ))) continue;
            total += snapshot.skyLightAt(sampleX, y, sampleZ);
            ++samples;
        }
    }
    return samples == 0 ? snapshot.skyLightAt(x, y, z) : static_cast<uint8_t>(total / samples);
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

// Packed skylight remains conservative while streaming is incomplete: it may
// be zero above a lake even though the generated terrain column is visibly
// exterior. Water needs a separate binary interface authority so that an
// unresolved vertical path cannot turn exact 16-block sections into opaque
// reflection panes. This never seeds or raises ordinary skylight. A complete
// edited roof remains in visualSkyCutoffY and therefore keeps covered water
// dark until real propagated light reaches an opening.
static bool waterFaceHasExteriorSky(const MeshSnapshot& snapshot, int x, int y, int z) {
    if (snapshot.skyLightAt(x, y, z) != 0U) return true;
    const int32_t cutoff = snapshot.visualSkyCutoffAt(x, z);
    if (cutoff == MeshSnapshot::SKY_CUTOFF_UNKNOWN ||
        cutoff == MeshSnapshot::SKY_CUTOFF_INCOMPLETE) {
        return false;
    }
    const int32_t worldY = snapshot.pos.y * CHUNK_EDGE + y;
    return worldY >= cutoff;
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
                    const uint16_t topSky =
                        packCornerLight(cornerSurfaceSkyLight(snapshot, x, y + 1, z, 0, 0),
                                        cornerSurfaceSkyLight(snapshot, x, y + 1, z, 0, 1),
                                        cornerSurfaceSkyLight(snapshot, x, y + 1, z, 1, 1),
                                        cornerSurfaceSkyLight(snapshot, x, y + 1, z, 1, 0));
                    pushFluidQuadSmoothSky(output.vertices, output.indices, FaceNormal::PLUS_Y,
                                           topSky, blockLightAt(x, y + 1, z), flow,
                                           state.isFalling(),
                                           waterFaceHasExteriorSky(snapshot, x, y + 1, z), top);
                }

                auto emitSide = [&](FaceNormal face, int neighborX, int neighborY, int neighborZ,
                                    const QuadCorner(&corners)[4]) {
                    const BlockType neighbor = snapshot.at(neighborX, neighborY, neighborZ);
                    if (neighbor == BlockType::WATER || isOpaque(neighbor)) return;
                    pushFluidQuad(
                        output.vertices, output.indices, face,
                        skyLightAt(neighborX, neighborY, neighborZ),
                        blockLightAt(neighborX, neighborY, neighborZ), flow, state.isFalling(),
                        waterFaceHasExteriorSky(snapshot, neighborX, neighborY, neighborZ),
                        corners);
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
                                  state.isFalling(), false, bottom);
                }
            }
        }
    }
}

template <typename Access, typename LightAccess, typename SkyLightAccess, typename SkyCutoffAccess>
static MeshOutput
buildGenericMesh(int gridW, int gridH, int gridD, const Access& getBlock,
                 const LightAccess& getBlockLight, const SkyLightAccess& getSkyLight,
                 const SkyCutoffAccess& getSkyCutoff, int64_t worldBaseX, int32_t sectionBaseY,
                 int64_t worldBaseZ, bool padded, bool useDerivedSkyLight, bool emitFlora,
                 bool emitGreedyWater, const MeshSnapshot* partialWater, MeshScratch& scratch) {
    MeshOutput output;

    // ---- Column skylight ----
    // The first open Y above the topmost OPAQUE block per column, computed
    // over the padded ring too so border faces read their real neighbor
    // column instead of a clamped copy (that clamp painted a visible light
    // seam along every chunk edge). Only opaque blocks block the sky: a tree
    // canopy is non-opaque leaves, so it must NOT darken the ground below with
    // a fake column shadow, the real cascade shadow does that, and doubling
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
        if (useDerivedSkyLight) return getSkyLight(x, y, z);
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
        return snapshot.blockLightAt(x, y, z);
    };
    auto skyLightFn = [&snapshot](int x, int y, int z) -> uint8_t {
        return snapshot.skyLightAt(x, y, z);
    };
    auto skyFn = [&snapshot](int x, int z) -> int32_t { return snapshot.skyCutoffAt(x, z); };
    return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, blockFn, lightFn, skyLightFn,
                            skyFn, snapshot.pos.x * CHUNK_EDGE, snapshot.pos.y * CHUNK_EDGE,
                            snapshot.pos.z * CHUNK_EDGE, /*padded=*/true,
                            /*useDerivedSkyLight=*/snapshot.derivedSkyLightValid,
                            /*emitFlora=*/true, /*emitGreedyWater=*/false, &snapshot, scratch);
}

MeshOutput LODMesher::buildMesh(const Chunk& chunk, int lodLevel) {
    // Beyond render distance, return empty mesh (distance culling)
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
                                    noSkyLight, noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                                    chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE,
                                    /*padded=*/false, /*useDerivedSkyLight=*/false,
                                    /*emitFlora=*/true,
                                    /*emitGreedyWater=*/true, nullptr, scratch);
        }

        case ChunkLOD::MEDIUM: {
            // LOD 1: 2x downsampling. Flora does not contribute to the
            // dominant block in a coarse cell.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                return dominantDownsampledBlock<2>(chunk, cx, cy, cz);
            };
            return buildGenericMesh(
                8, 8, 8, blockFn, noBlockLight, noSkyLight, noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE, /*padded=*/false,
                /*useDerivedSkyLight=*/false,
                /*emitFlora=*/false, /*emitGreedyWater=*/true, nullptr, scratch);
        }

        case ChunkLOD::COARSE: {
            // LOD 2: 4x downsampling.
            auto blockFn = [&chunk](int cx, int cy, int cz) -> BlockType {
                return dominantDownsampledBlock<4>(chunk, cx, cy, cz);
            };
            return buildGenericMesh(
                4, 4, 4, blockFn, noBlockLight, noSkyLight, noSkyCutoff, chunk.chunkX * CHUNK_EDGE,
                chunk.chunkY * CHUNK_EDGE, chunk.chunkZ * CHUNK_EDGE, /*padded=*/false,
                /*useDerivedSkyLight=*/false,
                /*emitFlora=*/false, /*emitGreedyWater=*/true, nullptr, scratch);
        }

        default:
            return MeshOutput{};
    }
}

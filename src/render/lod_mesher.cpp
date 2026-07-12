#include "render/lod_mesher.hpp"

#include "world/chunk.hpp"

#include <array>
#include <algorithm>
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
inline static int idx(int row, int col, int width) { return row * width + col; }

// Greedy merge on a face plane of arbitrary dimensions.
//
// faceMask[row*faceWidth + col] == true means that face cell is exposed.
// blockTypes[row*faceWidth + col] stores the block type at each exposed face cell.
//
// For each unmerged exposed cell, extend right as far as possible with
// matching block type, then extend down as far as possible with the same
// width and matching block type. Each merged rectangle becomes 1 quad.
static void meshFaceGeneric(
    int faceHeight, int faceWidth,
    const std::vector<bool>& faceMask,
    const std::vector<BlockType>& blockTypes,
    std::vector<bool>& merged,
    uint8_t normalIdx,
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    const auto& emitQuadFn
) {
    merged.assign(faceHeight * faceWidth, false);

    for (int row = 0; row < faceHeight; ++row) {
        for (int col = 0; col < faceWidth; ++col) {
            int i = idx(row, col, faceWidth);
            if (!faceMask[i] || merged[i]) {
                continue;
            }

            BlockType leadType = blockTypes[i];

            // Extend right (horizontal) as far as possible with same block type
            int width = 1;
            while (col + width < faceWidth &&
                   faceMask[idx(row, col + width, faceWidth)] &&
                   !merged[idx(row, col + width, faceWidth)] &&
                   blockTypes[idx(row, col + width, faceWidth)] == leadType) {
                ++width;
            }

            // Extend down (vertical) as far as possible with same width and type
            int height = 1;
            while (row + height < faceHeight) {
                bool rowValid = true;
                for (int w = 0; w < width; ++w) {
                    int j = idx(row + height, col + w, faceWidth);
                    if (!faceMask[j] || merged[j] || blockTypes[j] != leadType) {
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
            emitQuadFn(col, row, width, height, normalIdx, leadType,
                       vertices, indices);
        }
    }
}

static MeshOutput buildGenericMesh(
    int gridW, int gridH, int gridD,
    const BlockAccessor& getBlock,
    float worldX, float worldZ
) {
    MeshOutput output;

    // Pre-allocate reasonable capacity (6 faces × grid area × 4 verts)
    output.vertices.reserve(6 * gridW * gridD * gridH / 4);
    output.indices.reserve(6 * gridW * gridD * gridH / 2);

    // ======================================================================
    // Pass 1: Build Y-occupancy masks (flat bool[gridH][gridD][gridW])
    // ======================================================================
    std::vector<bool> solidY(gridH * gridD * gridW, false);

    for (int y = 0; y < gridH; ++y) {
        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                if (isSolid(getBlock(x, y, z))) {
                    solidY[idx(y, idx(z, x, gridW), gridD * gridW)] = true;
                }
            }
        }
    }

    // Reusable flat buffers for face processing (avoid reallocation per face)
    std::vector<bool> faceMask;
    std::vector<BlockType> blockTypes;
    std::vector<bool> merged;

    // ======================================================================
    // Face: +Y (top) — exposed when solid AND block above is AIR
    // ======================================================================
    for (int ly = 0; ly < gridH - 1; ++ly) {
        bool anyExposed = false;
        for (int z = 0; z < gridD && !anyExposed; ++z) {
            for (int x = 0; x < gridW && !anyExposed; ++x) {
                if (solidY[idx(ly, idx(z, x, gridW), gridD * gridW)] &&
                    !solidY[idx(ly + 1, idx(z, x, gridW), gridD * gridW)]) {
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        faceMask.assign(gridD * gridW, false);
        blockTypes.assign(gridD * gridW, BlockType::AIR);

        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                if (solidY[idx(ly, idx(z, x, gridW), gridD * gridW)] &&
                    !solidY[idx(ly + 1, idx(z, x, gridW), gridD * gridW)]) {
                    faceMask[idx(z, x, gridW)] = true;
                    blockTypes[idx(z, x, gridW)] = getBlock(x, ly, z);
                }
            }
        }

        auto emitQuad = [ly, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // +Y face: y = ly+1, CCW from above
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(ly + 1),
                static_cast<float16_t>(row + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(ly + 1),
                static_cast<float16_t>(row + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(ly + 1),
                static_cast<float16_t>(row + height + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(ly + 1),
                static_cast<float16_t>(row + height + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridD, gridW, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::PlusY),
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -Y (bottom) — exposed when solid AND block below is AIR
    // ======================================================================
    for (int ly = 1; ly < gridH; ++ly) {
        bool anyExposed = false;
        for (int z = 0; z < gridD && !anyExposed; ++z) {
            for (int x = 0; x < gridW && !anyExposed; ++x) {
                if (solidY[idx(ly, idx(z, x, gridW), gridD * gridW)] &&
                    !solidY[idx(ly - 1, idx(z, x, gridW), gridD * gridW)]) {
                    anyExposed = true;
                }
            }
        }
        if (!anyExposed) continue;

        faceMask.assign(gridD * gridW, false);
        blockTypes.assign(gridD * gridW, BlockType::AIR);

        for (int z = 0; z < gridD; ++z) {
            for (int x = 0; x < gridW; ++x) {
                if (solidY[idx(ly, idx(z, x, gridW), gridD * gridW)] &&
                    !solidY[idx(ly - 1, idx(z, x, gridW), gridD * gridW)]) {
                    faceMask[idx(z, x, gridW)] = true;
                    blockTypes[idx(z, x, gridW)] = getBlock(x, ly, z);
                }
            }
        }

        auto emitQuad = [ly, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // -Y face: y = ly, CCW from below
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(ly),
                static_cast<float16_t>(row + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(ly),
                static_cast<float16_t>(row + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(ly),
                static_cast<float16_t>(row + height + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(ly),
                static_cast<float16_t>(row + height + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridD, gridW, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::MinusY),
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: +X (right) — exposed when solid AND block to +X is AIR
    // ======================================================================
    for (int lx = 0; lx < gridW - 1; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                if (isSolid(getBlock(lx, y, z)) &&
                    !isSolid(getBlock(lx + 1, y, z))) {
                    faceMask[idx(y, z, gridD)] = true;
                    blockTypes[idx(y, z, gridD)] = getBlock(lx, y, z);
                }
            }
        }

        auto emitQuad = [lx, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // +X face: x = lx+1, CCW from +X
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(col + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(col + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(col + width + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(col + width + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridH, gridD, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::PlusX),
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -X (left) — exposed when solid AND block to -X is AIR
    // ======================================================================
    for (int lx = 0; lx < gridW; ++lx) {
        faceMask.assign(gridH * gridD, false);
        blockTypes.assign(gridH * gridD, BlockType::AIR);

        for (int y = 0; y < gridH; ++y) {
            for (int z = 0; z < gridD; ++z) {
                if (isSolid(getBlock(lx, y, z)) &&
                    !isSolid(getBlock(lx - 1, y, z))) {
                    faceMask[idx(y, z, gridD)] = true;
                    blockTypes[idx(y, z, gridD)] = getBlock(lx, y, z);
                }
            }
        }

        auto emitQuad = [lx, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // -X face: x = lx, CCW from -X
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx - 1 + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(col + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx - 1 + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(col + width + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx - 1 + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(col + width + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(lx - 1 + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(col + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridH, gridD, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::MinusX),
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: +Z (front) — exposed when solid AND block to +Z is AIR
    // ======================================================================
    for (int lz = 0; lz < gridD - 1; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                if (isSolid(getBlock(x, y, lz)) &&
                    !isSolid(getBlock(x, y, lz + 1))) {
                    faceMask[idx(y, x, gridW)] = true;
                    blockTypes[idx(y, x, gridW)] = getBlock(x, y, lz);
                }
            }
        }

        auto emitQuad = [lz, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // +Z face: z = lz+1, CCW from +Z
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(lz + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(lz + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(lz + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(lz + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridH, gridW, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::PlusZ),
                        output.vertices, output.indices, emitQuad);
    }

    // ======================================================================
    // Face: -Z (back) — exposed when solid AND block to -Z is AIR
    // ======================================================================
    for (int lz = 0; lz < gridD; ++lz) {
        faceMask.assign(gridH * gridW, false);
        blockTypes.assign(gridH * gridW, BlockType::AIR);

        for (int x = 0; x < gridW; ++x) {
            for (int y = 0; y < gridH; ++y) {
                if (isSolid(getBlock(x, y, lz)) &&
                    !isSolid(getBlock(x, y, lz - 1))) {
                    faceMask[idx(y, x, gridW)] = true;
                    blockTypes[idx(y, x, gridW)] = getBlock(x, y, lz);
                }
            }
        }

        auto emitQuad = [lz, worldX, worldZ](
            int col, int row, int width, int height,
            uint8_t normalIdx, BlockType bt,
            std::vector<Vertex>& verts,
            std::vector<uint32_t>& idxs) {
            float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
            float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
            // -Z face: z = lz, CCW from -Z
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(lz - 1 + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(row),
                static_cast<float16_t>(lz - 1 + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + width + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(lz - 1 + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            verts.push_back(Vertex{normalIdx,
                static_cast<float16_t>(col + worldX),
                static_cast<float16_t>(row + height),
                static_cast<float16_t>(lz - 1 + worldZ),
                static_cast<float16_t>(u), static_cast<float16_t>(v)});
            uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
            idxs.push_back(bi); idxs.push_back(bi + 1); idxs.push_back(bi + 2);
            idxs.push_back(bi); idxs.push_back(bi + 2); idxs.push_back(bi + 3);
        };

        meshFaceGeneric(gridH, gridW, faceMask, blockTypes, merged,
                        static_cast<uint8_t>(FaceNormal::MinusZ),
                        output.vertices, output.indices, emitQuad);
    }

    return output;
}

// ==========================================================================
// LODMesher implementation
// ==========================================================================

MeshOutput LODMesher::buildMesh(const Chunk& chunk, int lodLevel) {
    // Beyond render distance — return empty mesh (distance culling)
    if (lodLevel >= static_cast<int>(ChunkLOD::Count)) {
        return MeshOutput{};
    }

    float worldX = static_cast<float>(chunk.chunkX * CHUNK_WIDTH);
    float worldZ = static_cast<float>(chunk.chunkZ * CHUNK_DEPTH);

    switch (static_cast<ChunkLOD>(lodLevel)) {
        case ChunkLOD::Full: {
            // LOD 0: Full resolution greedy meshing (16×16×256)
            auto blockFn = [&chunk](int x, int y, int z) -> BlockType {
                return chunk.getBlock(x, y, z);
            };
            return buildGenericMesh(CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH,
                                    blockFn, worldX, worldZ);
        }

        case ChunkLOD::Medium: {
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
                            BlockType bt = chunk.getBlock(
                                gx + dx, gy + dy, gz + dz);
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
            return buildGenericMesh(8, 128, 8, blockFn, worldX, worldZ);
        }

        case ChunkLOD::Coarse: {
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
                            BlockType bt = chunk.getBlock(
                                gx + dx, gy + dy, gz + dz);
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
            return buildGenericMesh(4, 64, 4, blockFn, worldX, worldZ);
        }

        default:
            return MeshOutput{};
    }
}

int LODMesher::computeLODLevel(int distanceBlocks) {
    if (distanceBlocks < LOD0_MAX_DISTANCE) return static_cast<int>(ChunkLOD::Full);
    if (distanceBlocks < LOD1_MAX_DISTANCE) return static_cast<int>(ChunkLOD::Medium);
    if (distanceBlocks < LOD2_MAX_DISTANCE) return static_cast<int>(ChunkLOD::Coarse);
    return static_cast<int>(ChunkLOD::Count); // Beyond render distance
}

#include "render/mesher.hpp"

#include "world/chunk.hpp"

#include <array>
#include <algorithm>

// Greedy merge on a 16×16 face plane.
//
// faceMask[row][col] == true means that face cell is exposed.
// row is the vertical scan axis (top→bottom), col is horizontal (left→right).
// blockTypes[row][col] stores the block type at each exposed face cell.
//
// For each unmerged exposed cell, extend right as far as possible with
// matching block type, then extend down as far as possible with the same
// width and matching block type. Each merged rectangle becomes 1 quad.
static void meshFace(
    const std::array<std::array<bool, 16>, 16>& faceMask,
    const std::array<std::array<BlockType, 16>, 16>& blockTypes,
    uint8_t normalIdx,
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    const auto& emitQuadFn
) {
    std::array<std::array<bool, 16>, 16> merged{};
    for (auto& row : merged) {
        std::fill(row.begin(), row.end(), false);
    }

    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            if (!faceMask[row][col] || merged[row][col]) {
                continue;
            }

            BlockType leadType = blockTypes[row][col];

            // Extend right (horizontal) as far as possible with same block type
            int width = 1;
            while (col + width < 16 &&
                   faceMask[row][col + width] &&
                   !merged[row][col + width] &&
                   blockTypes[row][col + width] == leadType) {
                ++width;
            }

            // Extend down (vertical) as far as possible with same width and type
            int height = 1;
            while (row + height < 16) {
                bool rowValid = true;
                for (int w = 0; w < width; ++w) {
                    if (!faceMask[row + height][col + w] ||
                        merged[row + height][col + w] ||
                        blockTypes[row + height][col + w] != leadType) {
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
                    merged[row + dr][col + dc] = true;
                }
            }

            // Emit quad via callback
            emitQuadFn(col, row, width, height, normalIdx, leadType, vertices, indices);
        }
    }
}

MeshOutput GreedyMesher::buildMesh(Chunk& chunk) {
    MeshOutput output;

    // Pre-allocate reasonable capacity
    output.vertices.reserve(6 * 16 * 16 * 4);
    output.indices.reserve(6 * 16 * 16 * 6);

    float wx = static_cast<float>(chunk.chunkX * CHUNK_WIDTH);
    float wz = static_cast<float>(chunk.chunkZ * CHUNK_DEPTH);

    // ========================================================================
    // Pass 1: Build Y-occupancy masks (uint64_t[16] per Y level)
    // masksY[y][z]: bit x set if block(x,y,z) is solid
    // Each Z-row fits in one uint64_t (16 bits used)
    // ========================================================================
    std::array<std::array<uint64_t, 16>, CHUNK_HEIGHT> masksY{};
    for (auto& zRow : masksY) {
        zRow.fill(0);
    }

    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                if (isSolid(chunk.getBlock(x, y, z))) {
                    masksY[y][z] |= uint64_t{1} << x;
                }
            }
        }
    }

    // ========================================================================
    // Face: +Y (top) — exposed when block(x,y,z) solid AND block(x,y+1,z) AIR
    // Plane: XZ, row=Z (top→bottom), col=X (left→right)
    // Process each Y level independently (each has its own XZ plane)
    // ========================================================================
    {
        for (int ly = 0; ly < CHUNK_HEIGHT - 1; ++ly) {
            bool anyExposed = false;
            for (int z = 0; z < CHUNK_DEPTH; ++z) {
                uint64_t exposed = masksY[ly][z] & ~masksY[ly + 1][z];
                if (exposed != 0) { anyExposed = true; break; }
            }
            if (!anyExposed) continue;

            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            for (int z = 0; z < CHUNK_DEPTH; ++z) {
                uint64_t exposed = masksY[ly][z] & ~masksY[ly + 1][z];
                for (int x = 0; x < CHUNK_WIDTH; ++x) {
                    if ((exposed >> x) & 1) {
                        faceMask[z][x] = true;
                        blockTypes[z][x] = chunk.getBlock(x, ly, z);
                    }
                }
            }

            auto emitQuad = [ly, wx, wz](int col, int row, int width, int height,
                                         uint8_t normalIdx, BlockType bt,
                                         std::vector<Vertex>& verts,
                                         std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                // +Y face: y = ly+1, CCW from above
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(ly + 1),
                    static_cast<float16_t>(row + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(ly + 1),
                    static_cast<float16_t>(row + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(ly + 1),
                    static_cast<float16_t>(row + height + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(ly + 1),
                    static_cast<float16_t>(row + height + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::PlusY),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // ========================================================================
    // Face: -Y (bottom) — exposed when block(x,y,z) solid AND block(x,y-1,z) AIR
    // Plane: XZ, row=Z (top→bottom), col=X (left→right)
    // ========================================================================
    {
        for (int ly = 1; ly < CHUNK_HEIGHT; ++ly) {
            bool anyExposed = false;
            for (int z = 0; z < CHUNK_DEPTH; ++z) {
                uint64_t exposed = masksY[ly][z] & ~masksY[ly - 1][z];
                if (exposed != 0) { anyExposed = true; break; }
            }
            if (!anyExposed) continue;

            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            for (int z = 0; z < CHUNK_DEPTH; ++z) {
                uint64_t exposed = masksY[ly][z] & ~masksY[ly - 1][z];
                for (int x = 0; x < CHUNK_WIDTH; ++x) {
                    if ((exposed >> x) & 1) {
                        faceMask[z][x] = true;
                        blockTypes[z][x] = chunk.getBlock(x, ly, z);
                    }
                }
            }

            auto emitQuad = [ly, wx, wz](int col, int row, int width, int height,
                                         uint8_t normalIdx, BlockType bt,
                                         std::vector<Vertex>& verts,
                                         std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                // -Y face: y = ly, CCW from below
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(ly),
                    static_cast<float16_t>(row + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(ly),
                    static_cast<float16_t>(row + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(ly),
                    static_cast<float16_t>(row + height + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(ly),
                    static_cast<float16_t>(row + height + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::MinusY),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // ========================================================================
    // Face: +X (right) — exposed when block(x,y,z) solid AND block(x+1,y,z) AIR
    // Plane: YZ, row=Y (top→bottom), col=Z (left→right)
    // Process in 16-block Y slices (faceMask is 16×16)
    // ========================================================================
    {
        for (int yBase = 0; yBase < CHUNK_HEIGHT; yBase += 16) {
            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            int yEnd = std::min(yBase + 16, CHUNK_HEIGHT);
            for (int x = 0; x < CHUNK_WIDTH - 1; ++x) {
                for (int y = yBase; y < yEnd; ++y) {
                    for (int z = 0; z < CHUNK_DEPTH; ++z) {
                        if (isSolid(chunk.getBlock(x, y, z)) &&
                            !isSolid(chunk.getBlock(x + 1, y, z))) {
                            faceMask[y - yBase][z] = true;
                            blockTypes[y - yBase][z] = chunk.getBlock(x, y, z);
                        }
                    }
                }
            }

            auto emitQuad = [yBase, wx, wz](int col, int row, int width, int height,
                                            uint8_t normalIdx, BlockType bt,
                                            std::vector<Vertex>& verts,
                                            std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                int lx = CHUNK_WIDTH - 1;
                // +X face: x = lx+1, CCW from +X
                int y0 = yBase + row;
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(col + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(col + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(col + width + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(col + width + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::PlusX),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // ========================================================================
    // Face: -X (left) — exposed when block(x,y,z) solid AND block(x-1,y,z) AIR
    // Plane: YZ, row=Y (top→bottom), col=Z (left→right)
    // ========================================================================
    {
        for (int yBase = 0; yBase < CHUNK_HEIGHT; yBase += 16) {
            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            int yEnd = std::min(yBase + 16, CHUNK_HEIGHT);
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                for (int y = yBase; y < yEnd; ++y) {
                    for (int z = 0; z < CHUNK_DEPTH; ++z) {
                        if (isSolid(chunk.getBlock(x, y, z)) &&
                            !isSolid(chunk.getBlock(x - 1, y, z))) {
                            faceMask[y - yBase][z] = true;
                            blockTypes[y - yBase][z] = chunk.getBlock(x, y, z);
                        }
                    }
                }
            }

            auto emitQuad = [yBase, wx, wz](int col, int row, int width, int height,
                                            uint8_t normalIdx, BlockType bt,
                                            std::vector<Vertex>& verts,
                                            std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                int lx = 0;
                // -X face: x = lx, CCW from -X
                int y0 = yBase + row;
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(col + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(col + width + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(col + width + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(lx + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(col + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::MinusX),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // ========================================================================
    // Face: +Z (front) — exposed when block(x,y,z) solid AND block(x,y,z+1) AIR
    // Plane: XY, row=Y (top→bottom), col=X (left→right)
    // ========================================================================
    {
        for (int yBase = 0; yBase < CHUNK_HEIGHT; yBase += 16) {
            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            int yEnd = std::min(yBase + 16, CHUNK_HEIGHT);
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                for (int y = yBase; y < yEnd; ++y) {
                    for (int z = 0; z < CHUNK_DEPTH - 1; ++z) {
                        if (isSolid(chunk.getBlock(x, y, z)) &&
                            !isSolid(chunk.getBlock(x, y, z + 1))) {
                            faceMask[y - yBase][x] = true;
                            blockTypes[y - yBase][x] = chunk.getBlock(x, y, z);
                        }
                    }
                }
            }

            auto emitQuad = [yBase, wx, wz](int col, int row, int width, int height,
                                            uint8_t normalIdx, BlockType bt,
                                            std::vector<Vertex>& verts,
                                            std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                int lz = CHUNK_DEPTH - 1;
                // +Z face: z = lz+1, CCW from +Z
                int y0 = yBase + row;
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::PlusZ),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // ========================================================================
    // Face: -Z (back) — exposed when block(x,y,z) solid AND block(x,y,z-1) AIR
    // Plane: XY, row=Y (top→bottom), col=X (left→right)
    // ========================================================================
    {
        for (int yBase = 0; yBase < CHUNK_HEIGHT; yBase += 16) {
            std::array<std::array<bool, 16>, 16> faceMask{};
            std::array<std::array<BlockType, 16>, 16> blockTypes{};

            int yEnd = std::min(yBase + 16, CHUNK_HEIGHT);
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                for (int y = yBase; y < yEnd; ++y) {
                    for (int z = 0; z < CHUNK_DEPTH; ++z) {
                        if (isSolid(chunk.getBlock(x, y, z)) &&
                            !isSolid(chunk.getBlock(x, y, z - 1))) {
                            faceMask[y - yBase][x] = true;
                            blockTypes[y - yBase][x] = chunk.getBlock(x, y, z);
                        }
                    }
                }
            }

            auto emitQuad = [yBase, wx, wz](int col, int row, int width, int height,
                                            uint8_t normalIdx, BlockType bt,
                                            std::vector<Vertex>& verts,
                                            std::vector<uint32_t>& idx) {
                float u = static_cast<float>(static_cast<int>(bt) % 16) / 16.0f;
                float v = static_cast<float>(static_cast<int>(bt) / 16) / 16.0f;
                int lz = 0;
                // -Z face: z = lz, CCW from -Z
                int y0 = yBase + row;
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(y0),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + width + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                verts.push_back(Vertex{
                    normalIdx,
                    static_cast<float16_t>(col + wx),
                    static_cast<float16_t>(y0 + height),
                    static_cast<float16_t>(lz + wz),
                    static_cast<float16_t>(u),
                    static_cast<float16_t>(v)
                });
                uint32_t bi = static_cast<uint32_t>(verts.size()) - 4;
                idx.push_back(bi);
                idx.push_back(bi + 1);
                idx.push_back(bi + 2);
                idx.push_back(bi);
                idx.push_back(bi + 2);
                idx.push_back(bi + 3);
            };

            meshFace(faceMask, blockTypes, static_cast<uint8_t>(FaceNormal::MinusZ),
                     output.vertices, output.indices, emitQuad);
        }
    }

    // Mark chunk as meshed
    chunk.setMeshed(true);

    return output;
}

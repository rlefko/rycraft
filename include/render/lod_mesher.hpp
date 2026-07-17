#pragma once

#include "render/vertex.hpp"
#include "world/block_properties.hpp"
#include "world/mesh_snapshot.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Forward declaration
class Chunk;

// Output of a single chunk mesh build. One vertex/index stream holds two
// sections: indices [0, opaqueIndexCount) draw in the opaque chunk pass,
// [opaqueIndexCount, indices.size()) are the chunk's water surfaces, drawn
// by the dedicated water pass. One MegaBuffer allocation serves both.
struct MeshOutput {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t opaqueIndexCount = 0;

    MeshOutput() = default;
    MeshOutput(MeshOutput&&) = default;
    MeshOutput& operator=(MeshOutput&&) = default;
};

// Level of Detail enum for chunk meshing.
enum class ChunkLOD : int {
    FULL = 0,   // 16x16x16, full greedy meshing
    MEDIUM = 1, // 8x8x8, 2x downsampling
    COARSE = 2, // 4x4x4, 4x downsampling
    COUNT = 3
};

// Fixed-size per-thread buffers for one cubic mesh plane and its skylight
// halo. A packed nonzero face key stores block type, sky light, block light,
// and four corner-AO values. Zero represents an empty or consumed mask cell.
struct MeshScratch {
    static constexpr size_t MAX_FACE_CELLS = CHUNK_EDGE * CHUNK_EDGE;
    static constexpr size_t MAX_SKY_COLUMNS = (CHUNK_EDGE + 2) * (CHUNK_EDGE + 2);

    std::array<uint32_t, MAX_FACE_CELLS> faceKeys{};
    std::array<int32_t, MAX_SKY_COLUMNS> skyHeight{};
    std::array<uint8_t, MeshSnapshot::PADDED_VOLUME> exteriorAir{};
    std::array<uint16_t, MeshSnapshot::PADDED_VOLUME> exteriorFrontier{};
};

// Exact-cube greedy mesher. Production exact terrain uses the MeshSnapshot
// overload at full 16x16x16 block resolution. The direct Chunk overload
// retains legacy downsample modes for isolated unit tests:
//
//   FULL:   full greedy meshing (16x16x16)
//   MEDIUM: 2x downsampling (8x8x8)
//   COARSE: 4x downsampling (4x4x4)
//
// Exact terrain meshes through the MeshSnapshot overload. Real neighbor walls
// make chunk-boundary faces symmetric (no hidden interior walls between
// solid chunks, no holes or light seams at borders). The direct Chunk overload
// treats out-of-chunk as air.
//
// The separate far-terrain pipeline emits world-scale voxel tiers at 32-,
// 16-, 8-, 4-, and 2-block footprints. ChunkLOD does not select those tiers.
class LODMesher {
public:
    // Build an isolated chunk at a legacy unit-test resolution. Returns an
    // empty mesh when lodLevel is outside ChunkLOD.
    MeshOutput buildMesh(const Chunk& chunk, int lodLevel);

    // Neighbor-aware full-detail exact build. This is pure CPU work and is
    // safe to run on any thread with per-thread scratch storage.
    static MeshOutput buildMesh(const MeshSnapshot& snapshot, MeshScratch& scratch);
};

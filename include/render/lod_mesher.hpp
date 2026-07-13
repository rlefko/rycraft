#pragma once

#include "render/vertex.hpp"
#include <cstdint>
#include <vector>

// Forward declaration
struct Chunk;

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

// Distance thresholds (in blocks, not chunk units).
// A chunk is 16 blocks wide, so 8 chunks = 128 blocks.
static constexpr int LOD0_MAX_DISTANCE = 128; // 0-8 chunks: full detail
static constexpr int LOD1_MAX_DISTANCE = 256; // 8-16 chunks: 2x coarse
static constexpr int LOD2_MAX_DISTANCE = 512; // 16-32 chunks: 4x coarse
// Beyond 512 blocks (32 chunks): no rendering

// Level of Detail enum for chunk meshing.
enum class ChunkLOD : int {
    FULL = 0,   // 16x16x256, full greedy meshing
    MEDIUM = 1, // 8x8x128, 2x downsampling
    COARSE = 2, // 4x4x64, 4x downsampling
    COUNT = 3
};

// Greedy mesher with level-of-detail support. This is the single mesher in
// the engine: ChunkLOD::FULL runs the standard 16×16×256 greedy meshing, the
// coarser levels sample the chunk at reduced resolution first.
//
//   LOD 0 (near,  < 128 blocks):  full greedy meshing (16×16×256)
//   LOD 1 (mid,   128-256 blocks): 2× downsampling (8×8×128)
//   LOD 2 (far,   256-512 blocks): 4× downsampling (4×4×64)
//   Beyond 512 blocks: returns empty mesh (distance culling)
//
// NOTE: the renderer currently draws everything at ChunkLOD::FULL — the
// coarse levels emit geometry at grid scale (not world scale) and switching
// levels invalidates nothing, so LOD selection is parked until those are
// reworked (see docs/rendering-conventions.md).
class LODMesher {
public:
    // Build mesh for the given chunk at the specified LOD level.
    // Returns empty mesh when lodLevel >= ChunkLOD::COUNT (distance-culled).
    MeshOutput buildMesh(const Chunk& chunk, int lodLevel);

    // Determine LOD level from distance (in blocks) to chunk center.
    // Returns ChunkLOD::COUNT when chunk is beyond render distance.
    static int computeLODLevel(int distanceBlocks);
};

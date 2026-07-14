#pragma once

#include "render/vertex.hpp"
#include "world/block_properties.hpp"
#include "world/mesh_snapshot.hpp"
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

// Level of Detail enum for chunk meshing.
enum class ChunkLOD : int {
    FULL = 0,   // 16x16x256, full greedy meshing
    MEDIUM = 1, // 8x8x128, 2x downsampling
    COARSE = 2, // 4x4x64, 4x downsampling
    COUNT = 3
};

// Reusable per-thread meshing buffers. Building a full chunk mesh used to
// heap-allocate ~2.25 MB of scratch per build (~36 MB/frame during
// streaming bursts); assign() on long-lived vectors keeps the capacity.
struct MeshScratch {
    std::vector<bool> faceMask;
    std::vector<BlockType> blockTypes;
    std::vector<uint8_t> cellLight;
    std::vector<bool> merged;
    std::vector<int> skyHeight;
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
// The game meshes through the MeshSnapshot overload: real neighbor walls
// make chunk-boundary faces symmetric (no hidden interior walls between
// solid chunks, no holes or light seams at borders). The Chunk overload
// treats out-of-chunk as air and remains for the coarse LODs and for
// single-chunk unit tests.
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

    // Neighbor-aware full-detail build — the game's meshing path. Pure CPU
    // (no Metal), safe to run on any thread with a per-thread scratch.
    static MeshOutput buildMesh(const MeshSnapshot& snapshot, MeshScratch& scratch);
};

#pragma once

#include <vector>
#include <cstdint>
#include "render/vertex.hpp"
#include "render/mesher.hpp"

// Forward declaration
struct Chunk;

// Distance thresholds (in blocks, not chunk units).
// A chunk is 16 blocks wide, so 8 chunks = 128 blocks.
static constexpr int LOD0_MAX_DISTANCE = 128; // 0-8 chunks: full detail
static constexpr int LOD1_MAX_DISTANCE = 256; // 8-16 chunks: 2x coarse
static constexpr int LOD2_MAX_DISTANCE = 512; // 16-32 chunks: 4x coarse
// Beyond 512 blocks (32 chunks): no rendering

// Level of Detail enum for chunk meshing.
enum class ChunkLOD : int {
    Full = 0,   // 16x16x256, full greedy meshing
    Medium = 1, // 8x8x128, 2x downsampling
    Coarse = 2, // 4x4x64, 4x downsampling
    Count = 3
};

// Level of Detail mesher — reduces polygon count for distant chunks.
//
// Selects mesh resolution based on camera-to-chunk distance:
//   LOD 0 (near,  < 128 blocks):  full greedy meshing (16×16×256)
//   LOD 1 (mid,   128-256 blocks): 2× downsampling (8×8×128)
//   LOD 2 (far,   256-512 blocks): 4× downsampling (4×4×64)
//   Beyond 512 blocks: returns empty mesh (distance culling)
//
// Each LOD level builds a coarse representation by sampling the chunk at
// reduced resolution, then runs greedy meshing on the coarse grid.
class LODMesher {
public:
    // Build mesh for the given chunk at the specified LOD level.
    // Returns empty mesh when lodLevel >= ChunkLOD::Count (distance-culled).
    MeshOutput buildMesh(const Chunk& chunk, int lodLevel);

    // Determine LOD level from distance (in blocks) to chunk center.
    // Returns ChunkLOD::Count when chunk is beyond render distance.
    static int computeLODLevel(int distanceBlocks);
};

#pragma once

#include "world/chunk.hpp"

#include <vector>

// ---------------------------------------------------------------------------
// MeshSnapshot — a chunk plus a one-block ring from all eight neighbors (four
// face walls + four diagonal corner columns), copied under chunksMutex_ in one
// bounded memcpy (~83 KB, microseconds — the corner columns are four extra
// byte-writes per layer into the ring that resize() already allocated).
//
// Meshing reads it lock-free afterwards: block data only mutates before a
// chunk is inserted into the world or under chunksMutex_, so the copy is
// always internally consistent. The padding ring is what lets the mesher
// emit chunk-boundary faces symmetrically from REAL neighbor blocks —
// treating the neighbor as air produced both hidden interior walls between
// solid chunks and holes/light seams at borders.
//
// x and z accept [-1, CHUNK_WIDTH] / [-1, CHUNK_DEPTH], corner columns
// included: baked corner AO samples the diagonal neighbor of each face vertex,
// so leaving the four corners as air would put a bright AO seam along every
// chunk edge wherever a corner occluder straddles the border.
// ---------------------------------------------------------------------------
struct MeshSnapshot {
    static constexpr int PADDED_WIDTH = CHUNK_WIDTH + 2;
    static constexpr int PADDED_DEPTH = CHUNK_DEPTH + 2;

    int chunkX = 0;
    int chunkZ = 0;
    uint32_t version = 0;          // chunk revision captured with the blocks
    std::vector<BlockType> blocks; // PADDED_WIDTH × PADDED_DEPTH × CHUNK_HEIGHT

    void resize() { blocks.assign(PADDED_WIDTH * PADDED_DEPTH * CHUNK_HEIGHT, BlockType::AIR); }

    static int index(int x, int y, int z) {
        return (x + 1) + (z + 1) * PADDED_WIDTH + y * PADDED_WIDTH * PADDED_DEPTH;
    }

    BlockType at(int x, int y, int z) const {
        if (y < 0 || y >= CHUNK_HEIGHT || x < -1 || x > CHUNK_WIDTH || z < -1 || z > CHUNK_DEPTH)
            return BlockType::AIR;
        return blocks[index(x, y, z)];
    }
};

#pragma once

#include "world/chunk.hpp"

#include <array>
#include <limits>

// ---------------------------------------------------------------------------
// MeshSnapshot holds one 16x16x16 cube plus a one-block halo on every face,
// edge, and corner. Meshing consumes the immutable 18x18x18 block, fluid, and
// derived packed-light fields without holding the world mutex.
//
// Coordinates accept [-1, CHUNK_EDGE] on all axes. Loaded halo cells keep
// face culling, corner AO, partial-fluid corners, and light continuous across
// cubic chunk boundaries. Missing in-range cells below the planned terrain
// silhouette are opaque placeholders. Cells above it remain air so surface
// streaming cannot flash a full dark cube face. Cardinal cap bits let the
// mesher close only genuine underground openings until their halo loads.
// ---------------------------------------------------------------------------
struct MeshSnapshot {
    static constexpr int PADDED_EDGE = CHUNK_EDGE + 2;
    static constexpr int PADDED_VOLUME = PADDED_EDGE * PADDED_EDGE * PADDED_EDGE;
    static constexpr int SKY_COLUMNS = PADDED_EDGE * PADDED_EDGE;
    static constexpr int32_t SKY_CUTOFF_UNKNOWN = std::numeric_limits<int32_t>::min();
    // A real opaque block at WORLD_MAX_Y has cutoff WORLD_MAX_Y + 1. Keep the
    // conservative incomplete-load marker distinct so top-of-world roofs do
    // not become indistinguishable from a missing vertical section.
    static constexpr int32_t SKY_CUTOFF_INCOMPLETE = INCOMPLETE_SKY_PATH_CUTOFF;

    enum MissingFace : uint8_t {
        MISSING_PLUS_X = 1U << 0U,
        MISSING_MINUS_X = 1U << 1U,
        MISSING_PLUS_Z = 1U << 2U,
        MISSING_MINUS_Z = 1U << 3U,
        MISSING_PLUS_Y = 1U << 4U,
        MISSING_MINUS_Y = 1U << 5U,
    };

    ChunkPos pos{};
    uint32_t version = 0;
    // Production world snapshots set this after copying the packed halo.
    // Standalone tests and tools may leave it false to derive bounded column
    // light from skyCutoffY.
    bool derivedSkyLightValid = false;
    // Cardinal in-range neighbors absent when the snapshot was published.
    // The mesher seals underground openings and reconstructs a lit planned
    // surface silhouette. A later neighbor load dirties this mesh and replaces
    // the provisional boundary with the real shared face.
    uint8_t missingNeighborFaces = 0;
    std::array<BlockType, PADDED_VOLUME> blocks{};
    std::array<uint8_t, PADDED_VOLUME> fluidStates{};
    // High nibble is skylight and low nibble is block light, matching Chunk.
    std::array<uint8_t, PADDED_VOLUME> packedLight{};

    // Immutable generated-terrain cutoff and top material for each padded XZ
    // column. Unlike skyCutoffY, this cutoff is never made conservative by an
    // unloaded vertical section, so it can distinguish an aboveground loading
    // frontier from a genuinely underground opening.
    std::array<int32_t, SKY_COLUMNS> generatedSurfaceCutoffY{};
    std::array<BlockType, SKY_COLUMNS> generatedSurfaceMaterial{};

    // World-space Y of the first cell above the highest opaque block in each
    // padded XZ column. This preserves full-column skylight across 16-high
    // cubes. SKY_CUTOFF_UNKNOWN asks standalone tests/tools to derive a
    // bounded cutoff from the blocks available in this snapshot.
    std::array<int32_t, SKY_COLUMNS> skyCutoffY{};
    // The unoccluded geometric cutoff before an incomplete vertical path
    // turns skyCutoffY into SKY_CUTOFF_INCOMPLETE. Water uses this only to
    // classify a visible exterior interface while keeping light propagation
    // conservatively dark. Edited roofs remain part of this authority.
    std::array<int32_t, SKY_COLUMNS> visualSkyCutoffY{};

    MeshSnapshot() { clear(); }

    void clear() {
        blocks.fill(BlockType::AIR);
        fluidStates.fill(FluidState::source().packed());
        packedLight.fill(0);
        generatedSurfaceCutoffY.fill(SKY_CUTOFF_UNKNOWN);
        generatedSurfaceMaterial.fill(BlockType::STONE);
        skyCutoffY.fill(SKY_CUTOFF_UNKNOWN);
        visualSkyCutoffY.fill(SKY_CUTOFF_UNKNOWN);
        missingNeighborFaces = 0;
        derivedSkyLightValid = false;
    }
    void resize() { clear(); }

    static constexpr int index(int x, int y, int z) {
        return (x + 1) + (z + 1) * PADDED_EDGE + (y + 1) * PADDED_EDGE * PADDED_EDGE;
    }

    static constexpr int skyIndex(int x, int z) { return (x + 1) + (z + 1) * PADDED_EDGE; }

    int32_t skyCutoffAt(int x, int z) const {
        if (x < -1 || x > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return SKY_CUTOFF_UNKNOWN;
        }
        return skyCutoffY[skyIndex(x, z)];
    }

    int32_t visualSkyCutoffAt(int x, int z) const {
        if (x < -1 || x > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return SKY_CUTOFF_UNKNOWN;
        }
        return visualSkyCutoffY[skyIndex(x, z)];
    }

    int32_t generatedSurfaceCutoffAt(int x, int z) const {
        if (x < -1 || x > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return SKY_CUTOFF_UNKNOWN;
        }
        return generatedSurfaceCutoffY[skyIndex(x, z)];
    }

    BlockType generatedSurfaceMaterialAt(int x, int z) const {
        if (x < -1 || x > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return BlockType::STONE;
        }
        return generatedSurfaceMaterial[skyIndex(x, z)];
    }

    BlockType at(int x, int y, int z) const {
        if (x < -1 || x > CHUNK_EDGE || y < -1 || y > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return BlockType::AIR;
        }
        return blocks[index(x, y, z)];
    }

    uint8_t packedLightAt(int x, int y, int z) const {
        if (x < -1 || x > CHUNK_EDGE || y < -1 || y > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return 0;
        }
        return packedLight[index(x, y, z)];
    }

    uint8_t skyLightAt(int x, int y, int z) const {
        return derivedSkyLight(packedLightAt(x, y, z));
    }

    uint8_t blockLightAt(int x, int y, int z) const {
        return derivedBlockLight(packedLightAt(x, y, z));
    }

    FluidState fluidAt(int x, int y, int z) const {
        if (x < -1 || x > CHUNK_EDGE || y < -1 || y > CHUNK_EDGE || z < -1 || z > CHUNK_EDGE) {
            return FluidState::source();
        }
        return FluidState::fromPacked(fluidStates[index(x, y, z)]);
    }
};

static_assert(MeshSnapshot::PADDED_EDGE == 18);
static_assert(MeshSnapshot::PADDED_VOLUME == 18 * 18 * 18);

#pragma once

#include "world/chunk.hpp"

#include <array>

// ---------------------------------------------------------------------------
// LightEngine derives packed skylight and block light by flooding through
// transparent cells. Light is derived state: it is never serialized and is
// recomputed on generation, load, and edits.
//
// floodChunk recomputes a whole chunk from scratch each call (self emitters +
// the border light of its neighbors), so it needs no separate add/remove
// passes, placing or breaking a light source is just a re-flood. The flood is
// the unique max-light fixed point for a fixed set of blocks and neighbor
// borders, so repeated reconciliation converges to the same global result
// regardless of the order chunks stream in (World::reconcileLight drives it).
// ---------------------------------------------------------------------------
namespace LightEngine {

// The six face neighbors whose border light spills into a cubic chunk, in
// -X, +X, -Z, +Z, -Y, +Y order. Null entries are treated as dark. A neighbor
// that loads later participates when the world reconciles the shared face.
using FaceNeighbors = std::array<const Chunk*, 6>;

enum ChangedFace : uint8_t {
    CHANGED_MINUS_X = 1U << 0U,
    CHANGED_PLUS_X = 1U << 1U,
    CHANGED_MINUS_Z = 1U << 2U,
    CHANGED_PLUS_Z = 1U << 3U,
    CHANGED_MINUS_Y = 1U << 4U,
    CHANGED_PLUS_Y = 1U << 5U,
};

struct FloodResult {
    bool changedState = false;
    uint8_t changedFaceMask = 0;

    constexpr explicit operator bool() const { return changedState; }
};

// World-space first-open Y for each local XZ column. BLOCKED represents a
// column whose complete path to the sky has not been proven yet.
struct SkyLightSeedColumns {
    static constexpr int32_t BLOCKED = INCOMPLETE_SKY_PATH_CUTOFF;
    std::array<int32_t, CHUNK_EDGE * CHUNK_EDGE> cutoffY{};

    SkyLightSeedColumns() { cutoffY.fill(BLOCKED); }

    static constexpr int index(int x, int z) { return x + z * CHUNK_EDGE; }
    int32_t at(int x, int z) const { return cutoffY[index(x, z)]; }
    void set(int x, int z, int32_t cutoff) { cutoffY[index(x, z)] = cutoff; }
};

// Recompute both channels from direct sky columns, self emitters, and neighbor
// borders. Reports whether the packed field changed and which border planes
// changed, allowing the world to remesh and reconcile only affected neighbors.
FloodResult floodChunk(Chunk& chunk, const FaceNeighbors& neighbors,
                       const SkyLightSeedColumns& skySeeds = {});

// Convenience for tests and tools that only need self emitters. World
// publication supplies authoritative sky columns through floodChunk.
inline FloodResult computeSelfLight(Chunk& chunk) {
    return floodChunk(chunk, FaceNeighbors{}, SkyLightSeedColumns{});
}

} // namespace LightEngine

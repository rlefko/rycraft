#pragma once

#include <vector>
#include <cstdint>
#include "render/vertex.hpp"

// Forward declaration
struct Chunk;

// Output of a single chunk mesh build.
struct MeshOutput {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    MeshOutput() = default;
    MeshOutput(MeshOutput&&) = default;
    MeshOutput& operator=(MeshOutput&&) = default;
};

// Binary greedy mesher — builds triangle mesh from a Chunk's block data.
//
// Uses a 3-pass pipeline with uint64_t bitmasks for 16-wide rows:
//   Pass 1: Build occupancy masks per Y level.
//   Pass 2: Compute face-exposure masks for all 6 directions.
//   Pass 3: Greedily merge exposed faces into maximal quads.
//
// Target: <200 us per 16×16×256 chunk.
class GreedyMesher {
public:
    // Build mesh for the given chunk (pure: no side effects on chunk).
    MeshOutput buildMesh(const Chunk& chunk);
};

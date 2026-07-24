#pragma once

#include <cstdint>

// Half-precision float type alias.
// std::float16_t (C++23) is not yet available on Apple clang, so we use
// the compiler builtin __fp16 which is fully supported on Apple Silicon.
using float16_t = __fp16;

// 16-byte packed vertex for Metal vertex buffer.
//
// Fields are ordered so 4-byte aligned types come first, preventing
// compiler-inserted padding. Metal vertex descriptor reads each field
// with a fixed-size format that matches its byte width.
//
// Layout (16 bytes total, 16-byte aligned):
//   faceAttr     : uint32_t       (4 bytes)  — face normal (bits 0-2) +
//                                              texture array layer (bits 3+),
//                                              packed by packFaceAttr()
//   px, py, pz   : float16_t × 3  (6 bytes)  — CHUNK-LOCAL position (0..256,
//                                              exact in fp16; the per-draw
//                                              ChunkOrigin restores world
//                                              space in the vertex shader)
//   u, v         : float16_t × 2  (4 bytes)  — UV spanning the quad extent
//                                              in blocks; the repeat sampler
//                                              tiles greedy-merged quads
//
// Metal vertex descriptor offsets:
//   attribute(0) faceAttr:  offset 0,  format UInt   (4 bytes)
//   attribute(1) position:  offset 4,  format Half3  (6 bytes)
//   attribute(2) uv:        offset 10, format Half2  (4 bytes)
//   stride = 16 bytes
//
// Face normal indices:
//   0 = +X, 1 = -X, 2 = +Z, 3 = -Z, 4 = +Y, 5 = -Y
//   6 = flora cross, 7 = floor-torch cross

struct alignas(16) Vertex {
    // Packed face normal + texture layer — offset 0
    uint32_t faceAttr;

    // Position: 3 × float16_t = 6 bytes — offset 4
    float16_t px;
    float16_t py;
    float16_t pz;

    // UV: 2 × float16_t = 4 bytes — offset 10
    float16_t u;
    float16_t v;
};

static_assert(sizeof(Vertex) == 16, "Vertex must be 16 bytes");
static_assert(alignof(Vertex) == 16, "Vertex must be 16-byte aligned");

// Face normal index constants. CROSS marks flora cross-quads: the vertex
// shader gives them a fixed up-facing light instead of face shading (so the
// two diagonal quads of one plant never shade differently). TORCH_CROSS keeps
// the same double-sided geometry without applying plant-facing or subsurface
// lighting to the authored flame and stick.
enum class FaceNormal : uint8_t {
    PLUS_X = 0,
    MINUS_X = 1,
    PLUS_Z = 2,
    MINUS_Z = 3,
    PLUS_Y = 4,
    MINUS_Y = 5,
    CROSS = 6,
    TORCH_CROSS = 7,
};

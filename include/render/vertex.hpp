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
//   normalIdx    : uint32_t       (4 bytes)  — index into 6 face normals
//   px, py, pz   : float16_t × 3  (6 bytes)  — position
//   u, v         : float16_t × 2  (4 bytes)  — texture atlas UV
//
// Metal vertex descriptor offsets:
//   attribute(0) normalIdx: offset 0,  format UInt   (4 bytes)
//   attribute(1) position:  offset 4,  format Half3  (6 bytes)
//   attribute(2) uv:        offset 10, format Half2  (4 bytes)
//   stride = 16 bytes
//
// Face normal indices:
//   0 = +X, 1 = -X, 2 = +Z, 3 = -Z, 4 = +Y, 5 = -Y

struct alignas(16) Vertex {
    // Normal index: uint32_t (4 bytes) — offset 0
    uint32_t normalIdx;

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

// Face normal index constants
enum class FaceNormal : uint8_t {
    PlusX = 0,
    MinusX = 1,
    PlusZ = 2,
    MinusZ = 3,
    PlusY = 4,
    MinusY = 5,
};

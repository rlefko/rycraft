#pragma once

#include <cstdint>

// Half-precision float type alias.
// std::float16_t (C++23) is not yet available on Apple clang, so we use
// the compiler builtin __fp16 which is fully supported on Apple Silicon.
using float16_t = __fp16;

// 16-byte packed vertex for Metal vertex buffer.
//
// Layout (16 bytes total, 16-byte aligned):
//   px, py, pz   : float16_t × 3  (6 bytes)  — position
//   normalIdx    : uint8_t         (1 byte)   — index into 6 face normals
//   (1 byte padding)
//   u, v         : float16_t × 2  (4 bytes)  — texture atlas UV
//   color        : uint32_t        (4 bytes)  — ABGR color tint
//
// Face normal indices:
//   0 = +X, 1 = -X, 2 = +Z, 3 = -Z, 4 = +Y, 5 = -Y

struct alignas(16) Vertex {
    // Position: SIMD3<Float16> = 6 bytes
    float16_t px;
    float16_t py;
    float16_t pz;

    // Normal index: 1 byte (indices into 6 face normals)
    uint8_t normalIdx;

    // UV: SIMD2<Float16> = 4 bytes
    float16_t u;
    float16_t v;

    // Color: UInt32 = 4 bytes (ABGR for Metal)
    uint32_t color;
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

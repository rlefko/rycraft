#pragma once

#include <cstdint>

// GPU-side uniform buffer — matches the Metal shader's Uniforms struct exactly.
// Bound at buffer(1) (buffer 0 is reserved for vertex data).
struct alignas(16) Uniforms {
    float modelMatrix[16];
    float viewMatrix[16];
    float projectionMatrix[16];
    float sunDirection[3];
    float _pad0;
    float sunColor[3];
    float _pad1;
    float ambientColor[3];
    float _pad2;
};

static_assert(sizeof(Uniforms) <= 256, "Uniforms must fit in 256-byte buffer");

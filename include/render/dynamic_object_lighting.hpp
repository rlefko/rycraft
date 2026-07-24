#pragma once

#include "world/chunk.hpp"

#include <simd/simd.h>

#include <cstdint>

// Gameplay objects cache this packed probe on the fixed tick. Rendering only
// decodes immutable object state, so a high-refresh frame never locks World.
constexpr float dynamicObjectSkyLight(uint8_t packedLight) noexcept {
    return normalizedDerivedLight(derivedSkyLight(packedLight));
}

constexpr float dynamicObjectBlockLight(uint8_t packedLight) noexcept {
    return normalizedDerivedLight(derivedBlockLight(packedLight));
}

inline simd_float4 dynamicObjectLighting(uint8_t packedLight) noexcept {
    return simd_make_float4(dynamicObjectSkyLight(packedLight),
                            dynamicObjectBlockLight(packedLight), 1.0F, 0.0F);
}

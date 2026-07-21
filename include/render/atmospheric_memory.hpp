#pragma once

#include <cstdint>

struct AtmosphericMemoryFootprint {
    uint64_t sceneTargetBytes = 0;
    uint64_t shadowBytes = 0;
    uint64_t indirectBytes = 0;
    uint64_t atmosphereBytes = 0;
    uint64_t cloudBytes = 0;
    uint64_t volumetricBytes = 0;
    uint64_t lightningBytes = 0;

    uint64_t totalBytes() const noexcept {
        return sceneTargetBytes + shadowBytes + indirectBytes + atmosphereBytes + cloudBytes +
               volumetricBytes + lightningBytes;
    }
};

// The water reflection source keeps a complete HDR mip pyramid. It is copied
// once after opaque rendering, then filtered explicitly by the grazing-angle
// SSR march. Account for every allocated level rather than assuming the
// idealized four-thirds series: odd drawable dimensions round down at each
// level on Metal.
constexpr uint64_t waterReflectionPyramidMemoryBytes(uint32_t width, uint32_t height) noexcept {
    if (width == 0U || height == 0U) {
        return 0U;
    }
    uint64_t texels = 0U;
    for (;;) {
        texels += static_cast<uint64_t>(width) * height;
        if (width == 1U && height == 1U) {
            break;
        }
        width = width > 1U ? width / 2U : 1U;
        height = height > 1U ? height / 2U : 1U;
    }
    return texels * 8U; // RGBA16Float
}

constexpr uint64_t atmosphericSceneTargetMemoryBytes(uint32_t width, uint32_t height) noexcept {
    // HDR resolve is 8 B; the water refraction and SSR source is its complete
    // 8 B-per-texel mip pyramid; surface data is 4 B; opaque and media depth
    // are 4 B each.
    return static_cast<uint64_t>(width) * height * (8U + 4U + 4U + 4U) +
           waterReflectionPyramidMemoryBytes(width, height);
}

// Persistent payload retained by the integrated atmospheric frame graph.
// Memoryless MSAA attachments and device-specific page alignment are excluded.
AtmosphericMemoryFootprint atmosphericMemoryFootprint(uint32_t width, uint32_t height,
                                                      int quality) noexcept;

#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

// The frame's pixel formats are defined once so every opaque pipeline and
// render pass shares the same material and temporal-history contract.
namespace PixelFormats {
inline constexpr MTLPixelFormat SCENE_HDR = MTLPixelFormatRGBA16Float;
inline constexpr MTLPixelFormat SCENE_DEPTH = MTLPixelFormatDepth32Float;
inline constexpr MTLPixelFormat SURFACE = MTLPixelFormatRGBA8Unorm;
inline constexpr MTLPixelFormat REACTIVE = MTLPixelFormatR8Unorm;
inline constexpr MTLPixelFormat RESOLVE_DEPTH_KEY = MTLPixelFormatR32Float;
inline constexpr MTLPixelFormat BLOOM = MTLPixelFormatRG11B10Float;
inline constexpr MTLPixelFormat DRAWABLE = MTLPixelFormatBGRA8Unorm;

inline constexpr NSUInteger SCENE_SAMPLE_COUNT = 4;
inline constexpr NSUInteger RESOLVED_SCENE_SAMPLE_COUNT = 1;

constexpr uint16_t sceneResolveCoverageMask(uint32_t sampleCount) noexcept {
    return sampleCount == 0U || sampleCount > 16U
               ? uint16_t{0}
               : static_cast<uint16_t>((uint32_t{1} << sampleCount) - 1U);
}

constexpr bool sceneResolveUsesTileShader(uint32_t sampleCount) noexcept {
    return sampleCount > 1U;
}

// CPU golden for the tile resolver's full-precision minimum-depth selection.
constexpr size_t sceneResolveNearestDepthIndex(std::span<const float> deviceDepths) noexcept {
    size_t nearest = 0;
    for (size_t index = 1; index < deviceDepths.size(); ++index) {
        if (deviceDepths[index] < deviceDepths[nearest]) nearest = index;
    }
    return nearest;
}

constexpr uint64_t sceneColorPyramidBytes(uint32_t width, uint32_t height) noexcept {
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    uint64_t texels = 0U;
    for (;;) {
        texels += static_cast<uint64_t>(width) * height;
        if (width == 1U && height == 1U) break;
        width = width > 1U ? width / 2U : 1U;
        height = height > 1U ? height / 2U : 1U;
    }
    return texels * 8U;
}

// Driver allocation alignment is device-specific. This is the exact retained
// texture payload; every multisample attachment remains memoryless.
struct SceneTargetMemoryFootprint {
    uint64_t colorResolveBytes = 0;
    uint64_t surfaceResolveBytes = 0;
    uint64_t reactiveResolveBytes = 0;
    uint64_t depthResolveBytes = 0;
    uint64_t mediaDepthResolveBytes = 0;
    uint64_t sceneColorCopyBytes = 0;
    uint64_t persistentMultisampleBytes = 0;

    constexpr uint64_t totalBytes() const noexcept {
        return colorResolveBytes + surfaceResolveBytes + reactiveResolveBytes + depthResolveBytes +
               mediaDepthResolveBytes + sceneColorCopyBytes + persistentMultisampleBytes;
    }
};

constexpr SceneTargetMemoryFootprint sceneTargetMemoryFootprint(uint32_t width,
                                                                uint32_t height) noexcept {
    const uint64_t pixels =
        static_cast<uint64_t>(std::max(width, 1U)) * static_cast<uint64_t>(std::max(height, 1U));
    return {
        .colorResolveBytes = pixels * 8U,
        .surfaceResolveBytes = pixels * 4U,
        .reactiveResolveBytes = pixels,
        .depthResolveBytes = pixels * 4U,
        .mediaDepthResolveBytes = pixels * 4U,
        .sceneColorCopyBytes = sceneColorPyramidBytes(width, height),
        .persistentMultisampleBytes = 0U,
    };
}

// Every opaque scene pipeline declares all four color attachments and depth.
// The tile shader averages HDR while selecting surface and reactive data from
// the same nearest covered depth sample.
inline void configureScenePassPipeline(MTLRenderPipelineDescriptor* descriptor) {
    descriptor.colorAttachments[0].pixelFormat = SCENE_HDR;
    descriptor.colorAttachments[1].pixelFormat = SURFACE;
    descriptor.colorAttachments[2].pixelFormat = REACTIVE;
    descriptor.colorAttachments[3].pixelFormat = RESOLVE_DEPTH_KEY;
    descriptor.depthAttachmentPixelFormat = SCENE_DEPTH;
    descriptor.rasterSampleCount = SCENE_SAMPLE_COUNT;
}

inline void configureResolvedScenePassPipeline(MTLRenderPipelineDescriptor* descriptor) {
    descriptor.colorAttachments[0].pixelFormat = SCENE_HDR;
    descriptor.depthAttachmentPixelFormat = SCENE_DEPTH;
    descriptor.rasterSampleCount = RESOLVED_SCENE_SAMPLE_COUNT;
}
} // namespace PixelFormats

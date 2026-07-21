#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"
#include "world/weather.hpp"

#include <cstdint>
#include <span>

class GpuFrameTimer;

float lightningFlashIntensity(const LightningEvent& event, uint64_t currentWorldTick,
                              float ticksPerSecond = 20.0F) noexcept;
uint32_t lightningBoltSegmentCount(uint64_t eventId) noexcept;
inline constexpr uint64_t LIGHTNING_RENDERER_MEMORY_BYTES = 2U;

struct LightningRenderStats {
    uint64_t lastEventId = 0;
    uint32_t renderedEventCount = 0;
    float peakFlashIntensity = 0.0F;
};

// Draws deterministic procedural line bolts into resolved HDR after clouds.
// Geometry depth rejects bolts behind terrain. Optional quarter-resolution
// cloud hit depth dims the core while preserving a diffuse in-cloud glow, so
// strikes can read both in front of and behind the cloud layer.
class LightningRenderer {
public:
    LightningRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~LightningRenderer();

    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> sceneDepth, id<MTLTexture> cloudHitDepth,
                const simd_float4x4& viewProjection, simd_float3 cameraPosition,
                std::span<const LightningEvent> events, uint64_t currentWorldTick,
                float ticksPerSecond = 20.0F, GpuFrameTimer* timer = nullptr);

    LightningRenderStats stats() const noexcept { return _stats; }

private:
    id<MTLRenderPipelineState> _boltPipeline;
    id<MTLRenderPipelineState> _flashPipeline;
    id<MTLDepthStencilState> _depthState;
    id<MTLTexture> _neutralCloudDepth;
    LightningRenderStats _stats;
};

#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"

// ---------------------------------------------------------------------------
// Ssao — screen-space ambient occlusion.
//
// Owns a half-resolution R8 occlusion target and two pipelines: generate
// (hemisphere sampling from the resolved depth) and apply (multiply-blends the
// box-upsampled AO onto the HDR scene). Runs on the resolved opaque scene
// before the water and volumetric passes so it darkens only opaque ambient,
// never translucent water or additive light shafts. Skipped when the SSAO
// setting is off.
// ---------------------------------------------------------------------------
class Ssao {
public:
    Ssao(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width, uint32_t height);

    void resize(uint32_t width, uint32_t height);

    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> depthResolve, const SsaoUniforms& uniforms);

    // The generate target's dimensions — the one source the shader's texel
    // size must match, so callers fill SsaoUniforms.resolution from here.
    simd_float2 resolution() const {
        return simd_make_float2(static_cast<float>(_halfWidth), static_cast<float>(_halfHeight));
    }

private:
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _generatePipeline;
    id<MTLRenderPipelineState> _applyPipeline;
    id<MTLTexture> _aoTex; // half-res R8
    uint32_t _halfWidth;
    uint32_t _halfHeight;

    void allocateTarget();
};

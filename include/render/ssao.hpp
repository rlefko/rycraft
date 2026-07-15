#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"

// ---------------------------------------------------------------------------
// Ssao — screen-space ambient occlusion.
//
// Owns two half-resolution R8 occlusion targets and three pipelines: generate
// (hemisphere sampling from the resolved depth), a depth-aware bilateral blur
// (the generate pass dithers its kernel rotation with IGN and MSAA keeps no
// temporal history to hide that under, so unblurred AO printed diagonal scan
// lines on grazing surfaces), and apply (multiply-blends the blurred AO onto
// the HDR scene). Runs on the resolved opaque scene before the water and
// volumetric passes so it darkens only opaque ambient, never translucent
// water or additive light shafts. Skipped when the SSAO setting is off.
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
    id<MTLRenderPipelineState> _blurPipeline;
    id<MTLRenderPipelineState> _applyPipeline;
    id<MTLTexture> _aoTex;     // half-res R8, raw generate output
    id<MTLTexture> _aoBlurTex; // half-res R8, bilateral-blurred
    uint32_t _halfWidth;
    uint32_t _halfHeight;

    void allocateTarget();
};

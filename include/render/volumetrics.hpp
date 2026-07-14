#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"

// ---------------------------------------------------------------------------
// Volumetrics — ray-marched sun/moon light shafts.
//
// Owns a half-resolution RG11B10 target and two pipelines: the march (samples
// the shadow cascades along each view ray) and the additive composite that
// upsamples the shafts onto the HDR scene. Runs after the water pass so shafts
// sit over the resolved opaque + water color. Skipped entirely when the
// volumetric-light setting is off.
// ---------------------------------------------------------------------------
class Volumetrics {
public:
    Volumetrics(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                uint32_t height);

    void resize(uint32_t width, uint32_t height);

    // March into the half-res target, then composite additively onto sceneHDR.
    // shadowUniforms is the frame's cascade block; the shadow depth array +
    // comparison sampler come from the ShadowMap.
    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> depthResolve, id<MTLTexture> shadowDepth,
                id<MTLSamplerState> shadowSampler, const VolumetricUniforms& uniforms,
                const ShadowUniforms& shadowUniforms);

private:
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _marchPipeline;
    id<MTLRenderPipelineState> _compositePipeline;
    id<MTLTexture> _volumetricTex; // half-res inscatter
    uint32_t _halfWidth;
    uint32_t _halfHeight;

    void allocateTarget();
};

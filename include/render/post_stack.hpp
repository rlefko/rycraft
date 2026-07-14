#pragma once

#import <Metal/Metal.h>

#include "render/graphics_settings.hpp"

// ---------------------------------------------------------------------------
// PostStack — the terminal display pass.
//
// One fullscreen pass converts the linear HDR scene to the BGRA8 drawable:
// exposure → bloom add → Uchimura tonemap → vibrance grade → optional CAS
// sharpen → dither. It ALWAYS runs (the pre-HDR pipeline blitted raw scene
// colors when bloom was off, so the no-bloom path was never tonemapped);
// with bloom disabled the caller passes the class's own 4×4 black fallback
// as the bloom input so the pipeline never forks.
//
// Later commits grow this class with the exposure and lens-flare compute
// dispatches and the scene-apply (AO/cloud) pass; for now it owns only the
// composite.
// ---------------------------------------------------------------------------
class PostStack {
public:
    PostStack(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);

    // Measure scene luminance and ease the persistent exposure toward it
    // (eye adaptation). Run after the scene + water are composited, before
    // the composite reads the exposure.
    void encodeExposure(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR);

    // Encode the composite from sceneHDR (+ bloom) into outputTexture.
    // Pass a nil bloomTexture to composite with no bloom (the black
    // fallback is substituted). `frameIndex` drives the deterministic dither.
    void encodeComposite(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                         id<MTLTexture> bloomTexture, id<MTLTexture> outputTexture,
                         const GraphicsSettings& gfx, uint32_t frameIndex);

private:
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _compositePipelineState;
    id<MTLComputePipelineState> _exposurePipelineState;
    id<MTLBuffer> _exposureBuffer; // persistent ExposureState, GPU-only
    id<MTLTexture> _blackFallback; // 4×4, bound when bloom is off
    id<MTLSamplerState> _linearSampler;
};

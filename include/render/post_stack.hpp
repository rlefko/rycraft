#pragma once

#import <Metal/Metal.h>

#include "render/graphics_settings.hpp"
#include <simd/simd.h>

// ---------------------------------------------------------------------------
// PostStack, the terminal display pass.
//
// Compute passes update persistent exposure and cloud-aware sun-flare
// visibility. One fullscreen pass converts the linear HDR scene to the BGRA8 drawable:
// exposure → bloom add → Hable filmic tonemap → vibrance grade → optional CAS
// sharpen → dither. It ALWAYS runs (the pre-HDR pipeline blitted raw scene
// colors when bloom was off, so the no-bloom path was never tonemapped);
// with bloom disabled the caller passes the class's own 4×4 black fallback
// as the bloom input so the pipeline never forks.
// ---------------------------------------------------------------------------
class PostStack {
public:
    PostStack(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~PostStack();

    // Measure scene luminance and ease the persistent exposure toward it
    // (eye adaptation). Run after the scene + water are composited, before
    // the composite reads the exposure.
    void encodeExposure(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR);

    // Probe the sun's occlusion for the lens flare: 16 depth taps around
    // sunScreenUV ease the persistent visibility. Skip the call entirely
    // when the flare is off (visibility simply keeps its last value).
    void encodeFlareProbe(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneDepth,
                          id<MTLTexture> resolvedCloud, simd_float2 sunScreenUV);

    // Encode the composite from sceneHDR (+ bloom) into outputTexture.
    // Pass a nil bloomTexture to composite with no bloom (the black
    // fallback is substituted). `frameIndex` drives the deterministic dither.
    // flareStrength 0 disables the lens-flare overlay (setting off, night,
    // or the sun behind the camera); sunScreenUV is its screen anchor.
    void encodeComposite(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                         id<MTLTexture> bloomTexture, id<MTLTexture> outputTexture,
                         const GraphicsSettings& gfx, uint32_t frameIndex, float flareStrength,
                         simd_float2 sunScreenUV);

private:
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _compositePipelineState{};
    id<MTLComputePipelineState> _exposurePipelineState{};
    id<MTLComputePipelineState> _flarePipelineState{};
    id<MTLBuffer> _exposureBuffer{}; // persistent ExposureState, GPU-only
    id<MTLBuffer> _flareBuffer{};    // persistent FlareState (sun visibility)
    id<MTLTexture> _blackFallback{}; // 4×4, bound when bloom is off
    id<MTLTexture> _whiteFallback{}; // neutral cloud transmittance
    id<MTLSamplerState> _linearSampler{};
};

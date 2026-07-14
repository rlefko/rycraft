#pragma once

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

// ---------------------------------------------------------------------------
// Bloom — HDR bright-pass + Kawase blur pyramid.
//
// Pipeline (all RG11B10Float, half-resolution and below):
//   1. Extract pass — soft-threshold the HDR scene (radiance above ~1.0)
//   2. Kawase blur — 4-level pyramid, separable 8-tap blur
// The blurred result is exposed through `bloomTexture()`; the final
// composite in post.metal owns exposure, the bloom add, and tonemapping.
// Per-pass constants ride setFragmentBytes (no shared per-frame buffer, so
// no ring-buffering needed — the pass runs to completion within one encode).
// ---------------------------------------------------------------------------
class Bloom {
public:
    Bloom(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width, uint32_t height);

    ~Bloom();

    // Run extract + blur from the HDR scene into the pyramid. The bloom
    // result is bloomTexture(); a no-op when intensity is zero (the caller
    // binds the black fallback instead).
    void renderBloom(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture);

    // The finest pyramid level holding the accumulated bloom, for the
    // composite to sample. Valid after renderBloom.
    id<MTLTexture> bloomTexture() const { return _blurPyramid[0][0]; }

    // Reallocate textures for new resolution.
    void resize(uint32_t width, uint32_t height);

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When intensity is 0, renderBloom skips the entire pipeline as early exit.
    void setIntensity(float intensity) { _intensity = intensity; }
    float getIntensity() const { return _intensity; }

private:
    id<MTLDevice> _device;

    // ---- Pipeline states ----
    id<MTLRenderPipelineState> _extractPipelineState;
    id<MTLRenderPipelineState> _blurPipelineState;

    // ---- Extract pass texture (half-res) ----
    id<MTLTexture> _extractTexture;

    // ---- Blur pyramid (4 mip levels, half-resolution each step) ----
    static constexpr int PYRAMID_LEVELS = 4;
    id<MTLTexture> _blurPyramid[PYRAMID_LEVELS][2]; // [level][ping/pong]

    // ---- Sampler state (linear for blur) ----
    id<MTLSamplerState> _linearSampler;

    uint32_t _width;
    uint32_t _height;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _intensity;

    // ---- Texture allocation helpers ----
    void allocateExtractTexture();
    void allocateBlurPyramid();

    // ---- Render pass helpers ----
    void renderExtractPass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture);

    void renderBlurPass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> source,
                        id<MTLTexture> destination, float blurRadius);
};

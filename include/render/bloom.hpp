#pragma once

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

// ---------------------------------------------------------------------------
// Bloom — Post-processing bloom effect with ACES tone mapping.
//
// Pipeline:
//   1. Extract pass — threshold bright pixels (luminance > 1.0)
//   2. Kawase blur — 4-level mip pyramid, separable 8-tap blur
//   3. Composite pass — additive blend bloom + ACES tone mapping
//
// Responsibilities:
//   • Allocate MRT textures for extract and blur pyramid
//   • Create blur and composite pipeline states
//   • Execute full bloom pipeline per frame
// ---------------------------------------------------------------------------
class Bloom {
public:
    Bloom(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
          uint32_t width, uint32_t height);

    ~Bloom();

    // Execute the bloom post-processing pipeline.
    // sceneTexture: the resolved MSAA scene color
    // outputTexture: the final display texture
    void renderBloom(id<MTLCommandBuffer> commandBuffer,
                     id<MTLTexture> sceneTexture,
                     id<MTLTexture> outputTexture);

    // Reallocate textures for new resolution.
    void resize(uint32_t width, uint32_t height);

    // Get the intermediate bloom output texture (for chaining effects).
    id<MTLTexture> bloomOutputTexture() const { return _bloomOutput; }

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When intensity is 0, renderBloom skips the entire pipeline as early exit.
    void setIntensity(float intensity) { _intensity = intensity; }
    float getIntensity() const { return _intensity; }

private:
    id<MTLDevice> _device;

    // ---- Pipeline states ----
    id<MTLRenderPipelineState> _extractPipelineState;
    id<MTLRenderPipelineState> _blurPipelineState;
    id<MTLRenderPipelineState> _compositePipelineState;

    // ---- Extract pass texture ----
    id<MTLTexture> _extractTexture;

    // ---- Blur pyramid (4 mip levels, half-resolution each step) ----
    static constexpr int PYRAMID_LEVELS = 4;
    id<MTLTexture> _blurPyramid[PYRAMID_LEVELS][2]; // [level][ping/pong]

    // ---- Composite output ----
    id<MTLTexture> _bloomOutput;

    // ---- Uniform buffers ----
    id<MTLBuffer> _uniformsBuffer;

    // ---- Sampler state (linear for blur/composite) ----
    id<MTLSamplerState> _linearSampler;

    uint32_t _width;
    uint32_t _height;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _intensity;

    // ---- Texture allocation helpers ----
    void allocateExtractTexture();
    void allocateBlurPyramid();
    void allocateCompositeOutput();

    // ---- Render pass helpers ----
    void renderExtractPass(id<MTLCommandBuffer> commandBuffer,
                           id<MTLTexture> sceneTexture);

    void renderBlurPass(id<MTLCommandBuffer> commandBuffer,
                        id<MTLTexture> source,
                        id<MTLTexture> destination,
                        float blurRadius);

    void renderCompositePass(id<MTLCommandBuffer> commandBuffer,
                             id<MTLTexture> sceneTexture,
                             id<MTLTexture> bloomTexture,
                             id<MTLTexture> outputTexture);

    // Upload bloom uniforms
    void uploadUniforms(float resolution[2], float texelSize[2],
                        float threshold, float intensity, float blurRadius);
};

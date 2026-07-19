#import "render/bloom.hpp"

#include "common/error.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"
#include "render/shader_types.hpp"

#include <algorithm>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
Bloom::Bloom(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width, uint32_t height)
    : _device(device), _blurPyramid{}, _width(width), _height(height), _intensity(1.0f) {
    // ---- Load shader functions ----
    id<MTLFunction> extractVertexFunc = [shaderLibrary newFunctionWithName:@"bloomExtractVertex"];
    id<MTLFunction> extractFragmentFunc =
        [shaderLibrary newFunctionWithName:@"bloomExtractFragment"];
    id<MTLFunction> blurVertexFunc = [shaderLibrary newFunctionWithName:@"bloomBlurVertex"];
    id<MTLFunction> blurFragmentFunc = [shaderLibrary newFunctionWithName:@"bloomBlurFragment"];

    if (!extractVertexFunc || !extractFragmentFunc || !blurVertexFunc || !blurFragmentFunc) {
        RY_LOG_FATAL("Failed to load bloom shader functions");
    }

    // ---- Extract pipeline state (HDR) ----
    auto extractDesc = [[MTLRenderPipelineDescriptor alloc] init];
    extractDesc.vertexFunction = extractVertexFunc;
    extractDesc.fragmentFunction = extractFragmentFunc;
    extractDesc.colorAttachments[0].pixelFormat = PixelFormats::BLOOM;

    NSError* error = nil;
    _extractPipelineState = [_device newRenderPipelineStateWithDescriptor:extractDesc error:&error];
    resetMetalObject(extractDesc);
    resetMetalObject(extractVertexFunc);
    resetMetalObject(extractFragmentFunc);
    if (!_extractPipelineState) {
        RY_LOG_FATAL("Failed to create bloom extract pipeline state");
    }

    // ---- Blur pipeline state (HDR) ----
    auto blurDesc = [[MTLRenderPipelineDescriptor alloc] init];
    blurDesc.vertexFunction = blurVertexFunc;
    blurDesc.fragmentFunction = blurFragmentFunc;
    blurDesc.colorAttachments[0].pixelFormat = PixelFormats::BLOOM;

    _blurPipelineState = [_device newRenderPipelineStateWithDescriptor:blurDesc error:&error];
    resetMetalObject(blurDesc);
    resetMetalObject(blurVertexFunc);
    resetMetalObject(blurFragmentFunc);
    if (!_blurPipelineState) {
        RY_LOG_FATAL("Failed to create bloom blur pipeline state");
    }

    // ---- Linear sampler ----
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.mipFilter = MTLSamplerMipFilterNearest;
    samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    _linearSampler = [_device newSamplerStateWithDescriptor:samplerDesc];
    resetMetalObject(samplerDesc);
    if (!_linearSampler) {
        RY_LOG_FATAL("Failed to create bloom linear sampler");
    }

    // ---- Allocate textures ----
    allocateExtractTexture();
    allocateBlurPyramid();
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
Bloom::~Bloom() {
    resetMetalObject(_extractTexture);
    for (int i = 0; i < PYRAMID_LEVELS; ++i) {
        resetMetalObject(_blurPyramid[i][0]);
        resetMetalObject(_blurPyramid[i][1]);
    }
    resetMetalObject(_extractPipelineState);
    resetMetalObject(_blurPipelineState);
    resetMetalObject(_linearSampler);
}

// ---------------------------------------------------------------------------
// allocateExtractTexture, extract runs at half resolution so bloom never
// pays the full-res bandwidth (a wide blur wants a small source anyway).
// ---------------------------------------------------------------------------
void Bloom::allocateExtractTexture() {
    uint32_t w = std::max(_width / 2, 1u);
    uint32_t h = std::max(_height / 2, 1u);
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::BLOOM
                                                                   width:w
                                                                  height:h
                                                               mipmapped:false];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    _extractTexture = [_device newTextureWithDescriptor:desc];
    if (!_extractTexture) {
        RY_LOG_FATAL("Failed to allocate bloom extract texture");
    }
}

// ---------------------------------------------------------------------------
// allocateBlurPyramid
// ---------------------------------------------------------------------------
void Bloom::allocateBlurPyramid() {
    // Level 0 sits at the extract's half resolution; each level halves again.
    uint32_t w = std::max(_width / 2, 1u);
    uint32_t h = std::max(_height / 2, 1u);

    for (int level = 0; level < PYRAMID_LEVELS; ++level) {
        for (int pingpong = 0; pingpong < 2; ++pingpong) {
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::BLOOM
                                                                           width:w
                                                                          height:h
                                                                       mipmapped:false];
            desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModePrivate;
            _blurPyramid[level][pingpong] = [_device newTextureWithDescriptor:desc];
            if (!_blurPyramid[level][pingpong]) {
                RY_LOG_FATAL("Failed to allocate bloom blur pyramid texture");
            }
        }
        w = std::max(w / 2, 1u);
        h = std::max(h / 2, 1u);
    }
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------
void Bloom::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height)
        return;

    _width = width;
    _height = height;

    // Release and reallocate all textures
    resetMetalObject(_extractTexture);
    for (int i = 0; i < PYRAMID_LEVELS; ++i) {
        resetMetalObject(_blurPyramid[i][0]);
        resetMetalObject(_blurPyramid[i][1]);
    }

    allocateExtractTexture();
    allocateBlurPyramid();
}

// ---------------------------------------------------------------------------
// renderExtractPass
// ---------------------------------------------------------------------------
void Bloom::renderExtractPass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture) {
    if (!commandBuffer || !sceneTexture)
        return;

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = _extractTexture;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder) {
        resetMetalObject(passDesc);
        return;
    }

    [encoder setRenderPipelineState:_extractPipelineState];
    [encoder setFragmentTexture:sceneTexture atIndex:0];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];

    BloomUniforms uniforms{};
    uniforms.resolution = simd_make_float2(static_cast<float>(_extractTexture.width),
                                           static_cast<float>(_extractTexture.height));
    uniforms.texelSize =
        simd_make_float2(1.0f / uniforms.resolution.x, 1.0f / uniforms.resolution.y);
    uniforms.threshold = 1.0f; // HDR working space: extract radiance above ~1
    uniforms.intensity = 1.0f;
    uniforms.blurRadius = 1.0f;
    [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
    resetMetalObject(passDesc);
}

// ---------------------------------------------------------------------------
// renderBlurPass
// ---------------------------------------------------------------------------
void Bloom::renderBlurPass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> source,
                           id<MTLTexture> destination, float blurRadius) {
    if (!commandBuffer || !source || !destination)
        return;

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = destination;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder) {
        resetMetalObject(passDesc);
        return;
    }

    [encoder setRenderPipelineState:_blurPipelineState];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];

    BloomUniforms uniforms{};
    uniforms.resolution = simd_make_float2(static_cast<float>(destination.width),
                                           static_cast<float>(destination.height));
    uniforms.texelSize =
        simd_make_float2(1.0f / uniforms.resolution.x, 1.0f / uniforms.resolution.y);
    uniforms.threshold = 1.0f;
    uniforms.intensity = 1.0f;
    uniforms.blurRadius = blurRadius;
    [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
    resetMetalObject(passDesc);
}

// ---------------------------------------------------------------------------
// renderBloom, extract + blur pyramid, leaving the result in _blurPyramid[0][0]
// ---------------------------------------------------------------------------
void Bloom::renderBloom(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture) {
    if (!commandBuffer || !sceneTexture)
        return;

    // Early exit: zero intensity means bloom is effectively disabled. The
    // caller binds a black fallback in the composite instead.
    if (_intensity <= 0.0f)
        return;

    // 1. Extract bright pixels (half-res)
    renderExtractPass(commandBuffer, sceneTexture);

    // 2. Kawase blur pyramid, each level ping-pongs twice, growing radius
    renderBlurPass(commandBuffer, _extractTexture, _blurPyramid[0][0], 1.0f);
    renderBlurPass(commandBuffer, _blurPyramid[0][0], _blurPyramid[0][1], 1.0f);

    renderBlurPass(commandBuffer, _blurPyramid[0][1], _blurPyramid[1][0], 2.0f);
    renderBlurPass(commandBuffer, _blurPyramid[1][0], _blurPyramid[1][1], 2.0f);

    renderBlurPass(commandBuffer, _blurPyramid[1][1], _blurPyramid[2][0], 4.0f);
    renderBlurPass(commandBuffer, _blurPyramid[2][0], _blurPyramid[2][1], 4.0f);

    renderBlurPass(commandBuffer, _blurPyramid[2][1], _blurPyramid[3][0], 8.0f);
    renderBlurPass(commandBuffer, _blurPyramid[3][0], _blurPyramid[3][1], 8.0f);

    // Up-sample: fold coarse levels back down into level 0 (additive blur)
    renderBlurPass(commandBuffer, _blurPyramid[3][1], _blurPyramid[2][0], 4.0f);
    renderBlurPass(commandBuffer, _blurPyramid[2][0], _blurPyramid[1][0], 2.0f);
    renderBlurPass(commandBuffer, _blurPyramid[1][0], _blurPyramid[0][0], 1.0f);
}

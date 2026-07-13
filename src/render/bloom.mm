#import "render/bloom.hpp"

#include "common/error.hpp"
#include "render/shader_types.hpp"

#include <cstring>

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
    id<MTLFunction> compositeVertexFunc =
        [shaderLibrary newFunctionWithName:@"bloomCompositeVertex"];
    id<MTLFunction> compositeFragmentFunc =
        [shaderLibrary newFunctionWithName:@"bloomCompositeFragment"];

    if (!extractVertexFunc || !extractFragmentFunc || !blurVertexFunc || !blurFragmentFunc ||
        !compositeVertexFunc || !compositeFragmentFunc) {
        RY_LOG_FATAL("Failed to load bloom shader functions");
    }

    // ---- Extract pipeline state ----
    auto extractDesc = [[MTLRenderPipelineDescriptor alloc] init];
    extractDesc.vertexFunction = extractVertexFunc;
    extractDesc.fragmentFunction = extractFragmentFunc;
    extractDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

    NSError* error = nil;
    _extractPipelineState = [_device newRenderPipelineStateWithDescriptor:extractDesc error:&error];
    if (!_extractPipelineState) {
        RY_LOG_FATAL("Failed to create bloom extract pipeline state");
    }

    // ---- Blur pipeline state ----
    auto blurDesc = [[MTLRenderPipelineDescriptor alloc] init];
    blurDesc.vertexFunction = blurVertexFunc;
    blurDesc.fragmentFunction = blurFragmentFunc;
    blurDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

    _blurPipelineState = [_device newRenderPipelineStateWithDescriptor:blurDesc error:&error];
    if (!_blurPipelineState) {
        RY_LOG_FATAL("Failed to create bloom blur pipeline state");
    }

    // ---- Composite pipeline state ----
    auto compositeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    compositeDesc.vertexFunction = compositeVertexFunc;
    compositeDesc.fragmentFunction = compositeFragmentFunc;
    compositeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    _compositePipelineState = [_device newRenderPipelineStateWithDescriptor:compositeDesc
                                                                      error:&error];
    if (!_compositePipelineState) {
        RY_LOG_FATAL("Failed to create bloom composite pipeline state");
    }

    // ---- Uniforms buffer ----
    _uniformsBuffer = [_device newBufferWithLength:sizeof(BloomUniforms)
                                           options:MTLResourceStorageModeShared];
    if (!_uniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate bloom uniforms buffer");
    }

    // ---- Linear sampler ----
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.mipFilter = MTLSamplerMipFilterNearest;
    samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    _linearSampler = [_device newSamplerStateWithDescriptor:samplerDesc];
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
    // Metal objects are reference-counted; nil assignment releases them
    _extractTexture = nil;
    for (int i = 0; i < PYRAMID_LEVELS; ++i) {
        _blurPyramid[i][0] = nil;
        _blurPyramid[i][1] = nil;
    }
}

// ---------------------------------------------------------------------------
// allocateExtractTexture
// ---------------------------------------------------------------------------
void Bloom::allocateExtractTexture() {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:_width
                                                                  height:_height
                                                               mipmapped:false];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _extractTexture = [_device newTextureWithDescriptor:desc];
    if (!_extractTexture) {
        RY_LOG_FATAL("Failed to allocate bloom extract texture");
    }
}

// ---------------------------------------------------------------------------
// allocateBlurPyramid
// ---------------------------------------------------------------------------
void Bloom::allocateBlurPyramid() {
    uint32_t w = _width;
    uint32_t h = _height;

    for (int level = 0; level < PYRAMID_LEVELS; ++level) {
        // Half-resolution each step
        w = std::max(w / 2, 1u);
        h = std::max(h / 2, 1u);

        for (int pingpong = 0; pingpong < 2; ++pingpong) {
            auto desc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:w
                                                                  height:h
                                                               mipmapped:false];
            desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            _blurPyramid[level][pingpong] = [_device newTextureWithDescriptor:desc];
            if (!_blurPyramid[level][pingpong]) {
                RY_LOG_FATAL("Failed to allocate bloom blur pyramid texture");
            }
        }
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
    _extractTexture = nil;
    for (int i = 0; i < PYRAMID_LEVELS; ++i) {
        _blurPyramid[i][0] = nil;
        _blurPyramid[i][1] = nil;
    }

    allocateExtractTexture();
    allocateBlurPyramid();
}

// ---------------------------------------------------------------------------
// uploadUniforms
// ---------------------------------------------------------------------------
void Bloom::uploadUniforms(float resolution[2], float texelSize[2], float threshold,
                           float intensity, float blurRadius) {
    BloomUniforms uniforms{};
    uniforms.resolution = simd_make_float2(resolution[0], resolution[1]);
    uniforms.texelSize = simd_make_float2(texelSize[0], texelSize[1]);
    uniforms.threshold = threshold;
    uniforms.intensity = intensity;
    uniforms.blurRadius = blurRadius;

    std::memcpy((void*)_uniformsBuffer.contents, &uniforms, sizeof(uniforms));
}

// ---------------------------------------------------------------------------
// renderExtractPass
// ---------------------------------------------------------------------------
void Bloom::renderExtractPass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture) {
    if (!commandBuffer || !sceneTexture)
        return;

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = _extractTexture;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder)
        return;

    [encoder setRenderPipelineState:_extractPipelineState];
    [encoder setFragmentTexture:sceneTexture atIndex:0];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];

    // Upload uniforms
    float resolution[2] = {static_cast<float>(_width), static_cast<float>(_height)};
    float texelSize[2] = {1.0f / static_cast<float>(_width), 1.0f / static_cast<float>(_height)};
    uploadUniforms(resolution, texelSize, 1.0f, 1.0f, 1.0f);
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:0];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
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
    passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder)
        return;

    [encoder setRenderPipelineState:_blurPipelineState];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];

    // Upload uniforms
    float resolution[2] = {static_cast<float>(destination.width),
                           static_cast<float>(destination.height)};
    float texelSize[2] = {1.0f / static_cast<float>(destination.width),
                          1.0f / static_cast<float>(destination.height)};
    uploadUniforms(resolution, texelSize, 1.0f, 1.0f, blurRadius);
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:0];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
}

// ---------------------------------------------------------------------------
// renderCompositePass
// ---------------------------------------------------------------------------
void Bloom::renderCompositePass(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture,
                                id<MTLTexture> bloomTexture, id<MTLTexture> outputTexture) {
    if (!commandBuffer || !sceneTexture || !bloomTexture || !outputTexture)
        return;

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = outputTexture;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder)
        return;

    [encoder setRenderPipelineState:_compositePipelineState];
    [encoder setFragmentTexture:sceneTexture atIndex:0];
    [encoder setFragmentTexture:bloomTexture atIndex:1];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];
    [encoder setFragmentSamplerState:_linearSampler atIndex:1];

    // Upload uniforms — intensity controls bloom strength in composite shader.
    float resolution[2] = {static_cast<float>(outputTexture.width),
                           static_cast<float>(outputTexture.height)};
    float texelSize[2] = {1.0f / static_cast<float>(outputTexture.width),
                          1.0f / static_cast<float>(outputTexture.height)};
    uploadUniforms(resolution, texelSize, 1.0f, _intensity, 1.0f);
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:0];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
}

// ---------------------------------------------------------------------------
// renderBloom — Full bloom pipeline
// ---------------------------------------------------------------------------
void Bloom::renderBloom(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneTexture,
                        id<MTLTexture> outputTexture) {
    if (!commandBuffer || !sceneTexture || !outputTexture)
        return;

    // Early exit: zero intensity means bloom is effectively disabled.
    // Avoids 13 render passes (1 extract + 8 blur + 3 upsample + 1 composite).
    if (_intensity <= 0.0f)
        return;

    // 1. Extract bright pixels
    renderExtractPass(commandBuffer, sceneTexture);

    // 2. Kawase blur pyramid
    // Level 0: blur extract → pyramid[0][0], then pyramid[0][0] → pyramid[0][1]
    renderBlurPass(commandBuffer, _extractTexture, _blurPyramid[0][0], 1.0f);
    renderBlurPass(commandBuffer, _blurPyramid[0][0], _blurPyramid[0][1], 1.0f);

    // Level 1: blur pyramid[0][1] → pyramid[1][0], then pyramid[1][0] → pyramid[1][1]
    renderBlurPass(commandBuffer, _blurPyramid[0][1], _blurPyramid[1][0], 2.0f);
    renderBlurPass(commandBuffer, _blurPyramid[1][0], _blurPyramid[1][1], 2.0f);

    // Level 2
    renderBlurPass(commandBuffer, _blurPyramid[1][1], _blurPyramid[2][0], 4.0f);
    renderBlurPass(commandBuffer, _blurPyramid[2][0], _blurPyramid[2][1], 4.0f);

    // Level 3
    renderBlurPass(commandBuffer, _blurPyramid[2][1], _blurPyramid[3][0], 8.0f);
    renderBlurPass(commandBuffer, _blurPyramid[3][0], _blurPyramid[3][1], 8.0f);

    // Up-sample: combine levels (additive)
    // Start with finest level, add coarser levels
    // pyramid[2][1] → pyramid[2][0] (reuse as accumulator)
    renderBlurPass(commandBuffer, _blurPyramid[3][1], _blurPyramid[2][0], 4.0f);
    renderBlurPass(commandBuffer, _blurPyramid[2][0], _blurPyramid[1][0], 2.0f);
    renderBlurPass(commandBuffer, _blurPyramid[1][0], _blurPyramid[0][0], 1.0f);

    // 3. Composite: scene + bloom → output
    renderCompositePass(commandBuffer, sceneTexture, _blurPyramid[0][0], outputTexture);
}

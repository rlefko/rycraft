#import "render/volumetrics.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>

Volumetrics::Volumetrics(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                         uint32_t height)
    : _device(device) {
    id<MTLFunction> marchVertex = [shaderLibrary newFunctionWithName:@"volumetricVertex"];
    id<MTLFunction> marchFragment = [shaderLibrary newFunctionWithName:@"volumetricFragment"];
    id<MTLFunction> compVertex = [shaderLibrary newFunctionWithName:@"volumetricCompositeVertex"];
    id<MTLFunction> compFragment =
        [shaderLibrary newFunctionWithName:@"volumetricCompositeFragment"];
    if (!marchVertex || !marchFragment || !compVertex || !compFragment) {
        RY_LOG_FATAL("Failed to load volumetric shader functions");
    }

    NSError* error = nil;
    auto marchDesc = [[MTLRenderPipelineDescriptor alloc] init];
    marchDesc.vertexFunction = marchVertex;
    marchDesc.fragmentFunction = marchFragment;
    marchDesc.colorAttachments[0].pixelFormat = PixelFormats::BLOOM; // RG11B10, radiance only
    _marchPipeline = [_device newRenderPipelineStateWithDescriptor:marchDesc error:&error];
    if (!_marchPipeline) {
        RY_LOG_FATAL("Failed to create volumetric march pipeline state");
    }

    // The composite adds the shafts onto the HDR scene (one + one blend).
    auto compDesc = [[MTLRenderPipelineDescriptor alloc] init];
    compDesc.vertexFunction = compVertex;
    compDesc.fragmentFunction = compFragment;
    compDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    compDesc.colorAttachments[0].blendingEnabled = true;
    compDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    compDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    compDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    compDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    compDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    compDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    _compositePipeline = [_device newRenderPipelineStateWithDescriptor:compDesc error:&error];
    if (!_compositePipeline) {
        RY_LOG_FATAL("Failed to create volumetric composite pipeline state");
    }

    _halfWidth = std::max(width / 2, 1u);
    _halfHeight = std::max(height / 2, 1u);
    allocateTarget();
}

void Volumetrics::allocateTarget() {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::BLOOM
                                                                   width:_halfWidth
                                                                  height:_halfHeight
                                                               mipmapped:false];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    _volumetricTex = [_device newTextureWithDescriptor:desc];
    if (!_volumetricTex) {
        RY_LOG_FATAL("Failed to allocate volumetric target");
    }
}

void Volumetrics::resize(uint32_t width, uint32_t height) {
    uint32_t hw = std::max(width / 2, 1u);
    uint32_t hh = std::max(height / 2, 1u);
    if (hw == _halfWidth && hh == _halfHeight) {
        return;
    }
    _halfWidth = hw;
    _halfHeight = hh;
    allocateTarget();
}

void Volumetrics::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                         id<MTLTexture> depthResolve, id<MTLTexture> shadowDepth,
                         id<MTLSamplerState> shadowSampler, const VolumetricUniforms& uniforms,
                         const ShadowUniforms& shadowUniforms) {
    if (!commandBuffer || !sceneHDR || !depthResolve || !shadowDepth) {
        return;
    }

    // ---- March into the half-res target ----
    auto marchPass = [[MTLRenderPassDescriptor alloc] init];
    marchPass.colorAttachments[0].texture = _volumetricTex;
    marchPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    marchPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> march =
        [commandBuffer renderCommandEncoderWithDescriptor:marchPass];
    if (!march) {
        return;
    }
    [march setRenderPipelineState:_marchPipeline];
    [march setFragmentTexture:depthResolve atIndex:0];
    [march setFragmentTexture:shadowDepth atIndex:1];
    [march setFragmentSamplerState:shadowSampler atIndex:1];
    [march setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [march setFragmentBytes:&shadowUniforms length:sizeof(shadowUniforms) atIndex:1];
    [march drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [march endEncoding];

    // ---- Additive composite onto the HDR scene ----
    auto compPass = [[MTLRenderPassDescriptor alloc] init];
    compPass.colorAttachments[0].texture = sceneHDR;
    compPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    compPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> comp = [commandBuffer renderCommandEncoderWithDescriptor:compPass];
    if (!comp) {
        return;
    }
    [comp setRenderPipelineState:_compositePipeline];
    [comp setFragmentTexture:_volumetricTex atIndex:0];
    [comp drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [comp endEncoding];
}

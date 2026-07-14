#import "render/ssao.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>

Ssao::Ssao(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width, uint32_t height)
    : _device(device) {
    id<MTLFunction> genVertex = [shaderLibrary newFunctionWithName:@"ssaoVertex"];
    id<MTLFunction> genFragment = [shaderLibrary newFunctionWithName:@"ssaoGenerateFragment"];
    id<MTLFunction> applyVertex = [shaderLibrary newFunctionWithName:@"ssaoApplyVertex"];
    id<MTLFunction> applyFragment = [shaderLibrary newFunctionWithName:@"ssaoApplyFragment"];
    if (!genVertex || !genFragment || !applyVertex || !applyFragment) {
        RY_LOG_FATAL("Failed to load SSAO shader functions");
    }

    NSError* error = nil;
    auto genDesc = [[MTLRenderPipelineDescriptor alloc] init];
    genDesc.vertexFunction = genVertex;
    genDesc.fragmentFunction = genFragment;
    genDesc.colorAttachments[0].pixelFormat = PixelFormats::AO;
    _generatePipeline = [_device newRenderPipelineStateWithDescriptor:genDesc error:&error];
    if (!_generatePipeline) {
        RY_LOG_FATAL("Failed to create SSAO generate pipeline state");
    }

    // The apply multiplies AO onto the HDR scene (dst * src, no self-read).
    auto applyDesc = [[MTLRenderPipelineDescriptor alloc] init];
    applyDesc.vertexFunction = applyVertex;
    applyDesc.fragmentFunction = applyFragment;
    applyDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    applyDesc.colorAttachments[0].blendingEnabled = true;
    applyDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    applyDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    applyDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
    applyDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorDestinationAlpha;
    applyDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorZero;
    applyDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
    _applyPipeline = [_device newRenderPipelineStateWithDescriptor:applyDesc error:&error];
    if (!_applyPipeline) {
        RY_LOG_FATAL("Failed to create SSAO apply pipeline state");
    }

    _halfWidth = std::max(width / 2, 1u);
    _halfHeight = std::max(height / 2, 1u);
    allocateTarget();
}

void Ssao::allocateTarget() {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::AO
                                                                   width:_halfWidth
                                                                  height:_halfHeight
                                                               mipmapped:false];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    _aoTex = [_device newTextureWithDescriptor:desc];
    if (!_aoTex) {
        RY_LOG_FATAL("Failed to allocate SSAO target");
    }
}

void Ssao::resize(uint32_t width, uint32_t height) {
    uint32_t hw = std::max(width / 2, 1u);
    uint32_t hh = std::max(height / 2, 1u);
    if (hw == _halfWidth && hh == _halfHeight) {
        return;
    }
    _halfWidth = hw;
    _halfHeight = hh;
    allocateTarget();
}

void Ssao::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                  id<MTLTexture> depthResolve, const SsaoUniforms& uniforms) {
    if (!commandBuffer || !sceneHDR || !depthResolve) {
        return;
    }

    // ---- Generate the half-res occlusion ----
    auto genPass = [[MTLRenderPassDescriptor alloc] init];
    genPass.colorAttachments[0].texture = _aoTex;
    genPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    genPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> gen = [commandBuffer renderCommandEncoderWithDescriptor:genPass];
    if (!gen) {
        return;
    }
    [gen setRenderPipelineState:_generatePipeline];
    [gen setFragmentTexture:depthResolve atIndex:0];
    [gen setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [gen drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [gen endEncoding];

    // ---- Multiply the AO onto the HDR scene ----
    auto applyPass = [[MTLRenderPassDescriptor alloc] init];
    applyPass.colorAttachments[0].texture = sceneHDR;
    applyPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    applyPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> apply =
        [commandBuffer renderCommandEncoderWithDescriptor:applyPass];
    if (!apply) {
        return;
    }
    [apply setRenderPipelineState:_applyPipeline];
    [apply setFragmentTexture:_aoTex atIndex:0];
    [apply drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [apply endEncoding];
}

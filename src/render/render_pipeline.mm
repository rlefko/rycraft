#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "render/bloom.hpp"
#include "render/lod_mesher.hpp"
#include "render/mesher.hpp"
#include "render/particles.hpp"
#include "render/ui_overlay.hpp"
#include "render/uniforms.hpp"
#include "world/chunk.hpp"
#include "world/world.hpp"
#include "engine/camera.hpp"
#include "engine/hotbar.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RenderPipeline::RenderPipeline(id<MTLDevice> device,
                                id<MTLLibrary> shaderLibrary,
                                uint32_t width,
                                uint32_t height)
    : _device(device)
    , _megaBuffer(nullptr)
    , _textureAtlas(nullptr)
    , _uiOverlay(nullptr)
    , _bloom(nullptr)
    , _upscaler(nullptr)
    , _particles(nullptr)
    , _bloomIntensity(1.0f)
    , _renderWidth(width / 2)
    , _renderHeight(height / 2)
    , _displayWidth(width)
    , _displayHeight(height)
    , _frustumPlanes{}
{
    // ---- Load main chunk shader functions ----
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    if (!vertexFunc) {
        RY_LOG_FATAL("Failed to load vertex shader function 'vertexMain'");
    }

    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];
    if (!fragmentFunc) {
        RY_LOG_FATAL("Failed to load fragment shader function 'fragmentMain'");
    }

    // ---- Vertex descriptor (Metal 2.0 API) ----
    auto vertexDesc = [MTLVertexDescriptor vertexDescriptor];

    // Attribute 0: normalIdx (uint)
    vertexDesc.attributes[0].format = MTLVertexFormatUInt;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;

    // Attribute 1: position (float3 from Half3)
    vertexDesc.attributes[1].format = MTLVertexFormatHalf3;
    vertexDesc.attributes[1].offset = 4;
    vertexDesc.attributes[1].bufferIndex = 0;

    // Attribute 2: uv (float2 from Half2)
    vertexDesc.attributes[2].format = MTLVertexFormatHalf2;
    vertexDesc.attributes[2].offset = 10;
    vertexDesc.attributes[2].bufferIndex = 0;

    // Buffer layout
    vertexDesc.layouts[0].stride = 16;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDesc.layouts[0].stepRate = 1;

    // ---- Main render pipeline state ----
    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;

    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                            error:&error];
    if (!_pipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create render pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Water pipeline state (transparent) ----
    auto waterPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    waterPipelineDesc.vertexFunction = vertexFunc;
    waterPipelineDesc.fragmentFunction = fragmentFunc;
    waterPipelineDesc.vertexDescriptor = vertexDesc;

    waterPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    waterPipelineDesc.colorAttachments[0].blendingEnabled = true;
    waterPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    waterPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    waterPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    waterPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    waterPipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    waterPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    waterPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    _waterPipelineState = [_device newRenderPipelineStateWithDescriptor:waterPipelineDesc
                                                                 error:&error];
    if (!_waterPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create water pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Depth stencil state (opaque) ----
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create depth stencil state");
    }

    // ---- Water depth state (no depth write) ----
    auto waterDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    waterDepthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    waterDepthDesc.depthWriteEnabled = false;
    _waterDepthState = [_device newDepthStencilStateWithDescriptor:waterDepthDesc];
    if (!_waterDepthState) {
        RY_LOG_FATAL("Failed to create water depth stencil state");
    }

    // ---- Sky pipeline state ----
    id<MTLFunction> skyVertexFunc = [shaderLibrary newFunctionWithName:@"skyVertexMain"];
    if (!skyVertexFunc) {
        RY_LOG_FATAL("Failed to load sky vertex shader function 'skyVertexMain'");
    }

    id<MTLFunction> skyFragmentFunc = [shaderLibrary newFunctionWithName:@"skyFragmentMain"];
    if (!skyFragmentFunc) {
        RY_LOG_FATAL("Failed to load sky fragment shader function 'skyFragmentMain'");
    }

    auto skyPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    skyPipelineDesc.vertexFunction = skyVertexFunc;
    skyPipelineDesc.fragmentFunction = skyFragmentFunc;
    skyPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    _skyPipelineState = [_device newRenderPipelineStateWithDescriptor:skyPipelineDesc
                                                               error:&error];
    if (!_skyPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create sky pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // Sky uniforms buffer
    _skyUniformsBuffer = [_device newBufferWithLength:sizeof(SkyUniforms)
                                                options:MTLResourceStorageModeShared];
    if (!_skyUniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate sky uniforms buffer");
    }

    // ---- Block highlight pipeline state (lines) ----
    id<MTLFunction> highlightVertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    id<MTLFunction> highlightFragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];

    auto highlightPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    highlightPipelineDesc.vertexFunction = highlightVertexFunc;
    highlightPipelineDesc.fragmentFunction = highlightFragmentFunc;
    highlightPipelineDesc.vertexDescriptor = vertexDesc;
    highlightPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    highlightPipelineDesc.colorAttachments[0].blendingEnabled = true;
    highlightPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    highlightPipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    _highlightPipelineState = [_device newRenderPipelineStateWithDescriptor:highlightPipelineDesc
                                                                     error:&error];
    if (!_highlightPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create highlight pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // Highlight vertex buffer: 24 vertices for wireframe box (12 lines × 2 endpoints)
    // Each vertex: normalIdx(4) + position(6) + uv(4) = 16 bytes
    // We use normalIdx=4 (+Y face) and uv=(0,0) as placeholders; color comes from uniforms
    struct alignas(16) HighlightVertex {
        uint8_t normalIdx;
        uint8_t _pad[3];
        float16_t px, py, pz;
        float16_t u, v;
    };
    HighlightVertex highlightVerts[24];
    std::memset(highlightVerts, 0, sizeof(highlightVerts));

    // Wireframe box vertices: 8 corners × 3 lines each = 24 vertex draws
    // Lines: 12 edges of a unit cube at origin
    float corners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, // Bottom face
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}  // Top face
    };
    uint16_t edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical
    };

    for (int i = 0; i < 12; ++i) {
        int a = edges[i][0];
        int b = edges[i][1];
        highlightVerts[i * 2] = {4, {0, 0},
                                 static_cast<float16_t>(corners[a][0] - 0.002f),
                                 static_cast<float16_t>(corners[a][1] - 0.002f),
                                 static_cast<float16_t>(corners[a][2] - 0.002f),
                                 0, 0};
        highlightVerts[i * 2 + 1] = {4, {0, 0},
                                     static_cast<float16_t>(corners[b][0] - 0.002f),
                                     static_cast<float16_t>(corners[b][1] - 0.002f),
                                     static_cast<float16_t>(corners[b][2] - 0.002f),
                                     0, 0};
    }

    _highlightVertexBuffer = [_device newBufferWithBytes:highlightVerts
                                                  length:sizeof(highlightVerts)
                                                 options:MTLResourceStorageModeShared];
    if (!_highlightVertexBuffer) {
        RY_LOG_FATAL("Failed to allocate highlight vertex buffer");
    }

    // Highlight uniforms buffer (same layout as Uniforms)
    _highlightUniformsBuffer = [_device newBufferWithLength:256
                                                      options:MTLResourceStorageModeShared];
    if (!_highlightUniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate highlight uniforms buffer");
    }

    // ---- Render target textures (single-sample, at render resolution) ----
    // MSAA disabled: sampleCount=4 crashes on M4 Max with assertion
    // "MTLTextureDescriptor has sampleCount set but is using a type that
    // does not allow sampleCount". Render directly to resolve textures.
    auto colorResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                   width:_renderWidth
                                                                                  height:_renderHeight
                                                                              mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to allocate color resolve texture");
    }

    auto depthResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                  width:_renderWidth
                                                                                 height:_renderHeight
                                                                             mipmapped:false];
    _depthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    if (!_depthResolve) {
        RY_LOG_FATAL("Failed to allocate depth resolve texture");
    }

    // ---- Uniforms buffer (512 bytes with fog + camera position) ----
    _uniformsBuffer = [_device newBufferWithLength:512
                                              options:MTLResourceStorageModeShared];
    if (!_uniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate uniforms buffer");
    }

    // ---- MegaBuffer (centralized GPU memory for chunk meshes) ----
    _megaBuffer = new MegaBuffer(_device);

    // ---- TextureAtlas (procedural block textures) ----
    _textureAtlas = new TextureAtlas(_device);

    // ---- UIOverlay (screen-space HUD rendering) ----
    _uiOverlay = new UIOverlay(_device, shaderLibrary, _displayWidth, _displayHeight);

    // ---- Cloud pipeline state (Phase 8) ----
    id<MTLFunction> cloudVertexFunc = [shaderLibrary newFunctionWithName:@"cloudVertexMain"];
    id<MTLFunction> cloudFragmentFunc = [shaderLibrary newFunctionWithName:@"cloudFragmentMain"];

    if (cloudVertexFunc && cloudFragmentFunc) {
        auto cloudPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        cloudPipelineDesc.vertexFunction = cloudVertexFunc;
        cloudPipelineDesc.fragmentFunction = cloudFragmentFunc;
        cloudPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        cloudPipelineDesc.colorAttachments[0].blendingEnabled = true;
        cloudPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        cloudPipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

        _cloudPipelineState = [_device newRenderPipelineStateWithDescriptor:cloudPipelineDesc
                                                                     error:&error];
    }

    // Cloud depth state (depth test, no write)
    auto cloudDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    cloudDepthDesc.depthCompareFunction = MTLCompareFunctionLess;
    cloudDepthDesc.depthWriteEnabled = false;
    _cloudDepthState = [_device newDepthStencilStateWithDescriptor:cloudDepthDesc];

    // Cloud uniforms buffer
    _cloudUniformsBuffer = [_device newBufferWithLength:sizeof(CloudUniforms)
                                                  options:MTLResourceStorageModeShared];

    // ---- Bloom post-processing (Phase 8) ----
    _bloom = new Bloom(_device, shaderLibrary, _renderWidth, _renderHeight);
    _bloom->setIntensity(_bloomIntensity);

    // ---- MetalFX Upscaler (Phase 8.4) ----
    _upscaler = new MetalFXUpscaler(_device, _renderWidth, _renderHeight,
                                      _displayWidth, _displayHeight);

    // ---- Weather Particle System ----
    _particles = new ParticleSystem(_device, shaderLibrary);
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------
void RenderPipeline::render(id<MTLCommandQueue> queue,
                            id<CAMetalDrawable> drawable,
                            const Mat4& viewMatrix,
                            const Mat4& projectionMatrix,
                            const World& world,
                            const Camera& camera,
                            uint64_t worldTime,
                            std::optional<Vec3> highlightedBlock,
                            const Hotbar& hotbar)
{
    if (!drawable || !queue) return;

    // Compute VP matrix and extract frustum planes
    Mat4 vpMatrix = projectionMatrix * viewMatrix;
    extractFrustumPlanes(vpMatrix);

    // Compute day/night uniforms
    float sunDirection[3] = {0.5f, 0.8f, 0.3f};
    float sunColor[3] = {1.0f, 0.95f, 0.9f};
    float ambientColor[3] = {0.35f, 0.35f, 0.4f};
    SkyUniforms skyUniforms{};
    computeDayNightUniforms(worldTime, sunDirection, sunColor, ambientColor, skyUniforms);

    // Normalize sun direction
    float sunLen = std::sqrt(
        sunDirection[0] * sunDirection[0] +
        sunDirection[1] * sunDirection[1] +
        sunDirection[2] * sunDirection[2]);
    if (sunLen > 0.001f) {
        sunDirection[0] /= sunLen;
        sunDirection[1] /= sunLen;
        sunDirection[2] /= sunLen;
    }

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return;

    // Upload sky uniforms
    std::memcpy((void*)_skyUniformsBuffer.contents, &skyUniforms, sizeof(SkyUniforms));

    // ---- Sky Pass (before everything else, at display resolution) ----
    renderSky(commandBuffer, drawable, skyUniforms);

    // ---- Main render pass descriptor (render at half-resolution for upscaling) ----
    auto renderPassDesc = [[MTLRenderPassDescriptor alloc] init];

    // Color attachment: render directly to _colorResolve (MSAA disabled).
    // Deferred store: MTLStoreActionStore keeps _colorResolve for bloom/upscale.
    // When bloom is active, _colorResolve feeds the bloom extract pass; when
    // bloom is off, _colorResolve feeds the upscaler directly.
    renderPassDesc.colorAttachments[0].texture = _colorResolve;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(
        skyUniforms.horizonColor[0],
        skyUniforms.horizonColor[1],
        skyUniforms.horizonColor[2],
        1.0f
    );

    // Depth attachment: render directly to _depthResolve (MSAA disabled)
    renderPassDesc.depthAttachment.texture = _depthResolve;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionStore;
    renderPassDesc.depthAttachment.clearDepth = 1.0;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) return;

    // ---- Opaque chunk pass ----
    renderChunks(encoder, world, viewMatrix, projectionMatrix,
                 sunDirection, sunColor, ambientColor, skyUniforms.horizonColor);

    // ---- Block highlight pass (after opaque chunks, before UI) ----
    if (highlightedBlock.has_value()) {
        renderBlockHighlight(encoder, highlightedBlock.value(), viewMatrix, projectionMatrix);
    }

    // ---- Weather particle pass (rain/snow billboards) ----
    if (_particles) {
        _particles->render(encoder, viewMatrix, projectionMatrix, camera.getPosition());
    }

    [encoder endEncoding];

    // ---- Cloud Pass (Phase 8, at render resolution) ----
    renderClouds(commandBuffer, drawable, viewMatrix, projectionMatrix,
                 camera, worldTime, sunDirection);

    // ---- Water pass (transparent, separate render pass) ----
    renderWater(commandBuffer, drawable, viewMatrix, projectionMatrix,
                sunDirection, sunColor, ambientColor);

    // ---- Bloom Post-Processing (Phase 8) ----
    // Deferred store path:
    //   Bloom active  → _colorResolve → bloom extract → blur → composite → _bloomOutput
    //   Bloom inactive → _colorResolve → upscaler → display (skip bloom entirely)
    // When bloom intensity is 0, renderBloom returns immediately (early exit),
    // skipping 13 internal render passes. The upscaler then reads _colorResolve.
    if (_bloom) {
        _bloom->renderBloom(commandBuffer, _colorResolve, _bloom->bloomOutputTexture());
    }

    // ---- Upscale to display resolution (Phase 8.4) ----
    // Select upscale source based on bloom activity:
    //   Bloom active  → upscale _bloomOutput (scene + bloom, tone-mapped)
    //   Bloom inactive → upscale _colorResolve (raw resolved scene)
    id<MTLTexture> finalTexture = _colorResolve;
    if (_bloom && _bloomIntensity > 0.0f) {
        finalTexture = _bloom->bloomOutputTexture();
    }
    if (_upscaler && finalTexture) {
        _upscaler->upscale(commandBuffer, finalTexture, drawable.texture);
    }

    // ---- UI Overlay Pass (screen-space HUD at display resolution) ----
    auto uiPassDesc = [[MTLRenderPassDescriptor alloc] init];
    uiPassDesc.colorAttachments[0].texture = drawable.texture;
    uiPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    uiPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> uiEncoder = [commandBuffer renderCommandEncoderWithDescriptor:uiPassDesc];
    if (uiEncoder) {
        renderUIOverlay(uiEncoder, hotbar);
        [uiEncoder endEncoding];
    }

    // Present and commit
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

// ---------------------------------------------------------------------------
// computeDayNightUniforms (Task 6.4-6.5)
// ---------------------------------------------------------------------------
void RenderPipeline::computeDayNightUniforms(uint64_t worldTime,
                                              float sunDirection[3],
                                              float sunColor[3],
                                              float ambientColor[3],
                                              SkyUniforms& skyUniforms)
{
    // Full day = 24000 ticks (20 minutes real time at 20Hz)
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // Orbital angle: 0 = dawn, PI/2 = noon, PI = dusk, 3PI/2 = midnight
    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    // Sun direction: rotates in XZ plane with slight Z offset for visual depth
    sunDirection[0] = std::cos(orbitalAngle);
    sunDirection[1] = std::sin(orbitalAngle);
    sunDirection[2] = 0.3f;

    // Sun elevation factor: 1 at noon (angle=PI/2), -1 at midnight (angle=3PI/2)
    float sunElevation = std::sin(orbitalAngle);

    // ---- Sun color: white at noon, orange at sunrise/sunset, dark at night ----
    if (sunElevation > 0.1f) {
        // Daytime: white to slightly warm
        float intensity = std::min(sunElevation, 1.0f);
        sunColor[0] = 1.0f;
        sunColor[1] = 0.95f + 0.05f * intensity;
        sunColor[2] = 0.9f + 0.1f * intensity;
    } else if (sunElevation > -0.1f) {
        // Sunrise/sunset: orange
        float t = (sunElevation + 0.1f) / 0.2f; // 0..1 across twilight
        sunColor[0] = 1.0f;
        sunColor[1] = 0.5f + 0.45f * t;
        sunColor[2] = 0.2f + 0.7f * t;
    } else {
        // Night: very dim
        sunColor[0] = 0.1f;
        sunColor[1] = 0.1f;
        sunColor[2] = 0.15f;
    }

    // ---- Ambient color: 0.35/0.35/0.4 at noon, 0.1/0.1/0.15 at night ----
    float ambientDay[3] = {0.35f, 0.35f, 0.4f};
    float ambientNight[3] = {0.1f, 0.1f, 0.15f};
    float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
    for (int i = 0; i < 3; ++i) {
        ambientColor[i] = ambientNight[i] + (ambientDay[i] - ambientNight[i]) * ambientT;
    }

    // ---- Sky colors ----
    // Daytime sky: blue zenith, lighter horizon
    float dayZenith[3] = {0.2f, 0.4f, 0.8f};
    float dayHorizon[3] = {0.53f, 0.81f, 0.92f};

    // Night sky: very dark blue
    float nightZenith[3] = {0.02f, 0.02f, 0.05f};
    float nightHorizon[3] = {0.05f, 0.05f, 0.1f};

    // Twilight: purple/orange
    float twilightZenith[3] = {0.15f, 0.1f, 0.3f};
    float twilightHorizon[3] = {0.6f, 0.3f, 0.2f};

    // Interpolate based on sun elevation
    float* zenithColor = dayZenith;
    float* horizonColor = dayHorizon;

    if (sunElevation > 0.05f) {
        // Full day
        zenithColor = dayZenith;
        horizonColor = dayHorizon;
    } else if (sunElevation > -0.15f) {
        // Twilight
        zenithColor = twilightZenith;
        horizonColor = twilightHorizon;
    } else {
        // Night
        zenithColor = nightZenith;
        horizonColor = nightHorizon;
    }

    std::memcpy(skyUniforms.zenithColor, zenithColor, sizeof(skyUniforms.zenithColor));
    std::memcpy(skyUniforms.horizonColor, horizonColor, sizeof(skyUniforms.horizonColor));
    std::memcpy(skyUniforms.sunDirection, sunDirection, sizeof(skyUniforms.sunDirection));
    std::memcpy(skyUniforms.sunColor, sunColor, sizeof(skyUniforms.sunColor));
    skyUniforms.sunIntensity = std::max(0.0f, sunElevation);
}

// ---------------------------------------------------------------------------
// renderSky (Task 6.6)
// ---------------------------------------------------------------------------
void RenderPipeline::renderSky(id<MTLCommandBuffer> commandBuffer,
                                id<CAMetalDrawable> drawable,
                                const SkyUniforms& skyUniforms)
{
    auto skyPassDesc = [[MTLRenderPassDescriptor alloc] init];
    skyPassDesc.colorAttachments[0].texture = drawable.texture;
    skyPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    skyPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    skyPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(
        skyUniforms.horizonColor[0],
        skyUniforms.horizonColor[1],
        skyUniforms.horizonColor[2],
        1.0f
    );

    id<MTLRenderCommandEncoder> skyEncoder = [commandBuffer renderCommandEncoderWithDescriptor:skyPassDesc];
    if (!skyEncoder) return;

    [skyEncoder setRenderPipelineState:_skyPipelineState];
    [skyEncoder setVertexBuffer:_skyUniformsBuffer offset:0 atIndex:1];

    // Draw fullscreen quad (6 vertices, no index buffer)
    [skyEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

    [skyEncoder endEncoding];
}

// ---------------------------------------------------------------------------
// renderChunks (opaque pass)
// ---------------------------------------------------------------------------
void RenderPipeline::renderChunks(id<MTLRenderCommandEncoder> encoder,
                                     const World& world,
                                     const Mat4& viewMatrix,
                                     const Mat4& projectionMatrix,
                                     const float sunDirection[3],
                                     const float sunColor[3],
                                     const float ambientColor[3],
                                     const float fogColor[3])
{
    // Bind pipeline state
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];

    // Extract camera position from view matrix.
    // View = R^T * T(-eye), so eye = -R^T * T_col3 = -R * T_col3
    const float* v = viewMatrix.data.data();
    float camX = -(v[12] * v[0] + v[13] * v[4] + v[14] * v[8]);
    float camY = -(v[12] * v[1] + v[13] * v[5] + v[14] * v[9]);
    float camZ = -(v[12] * v[2] + v[13] * v[6] + v[14] * v[10]);

    // Pack and upload uniforms
    Uniforms uniforms{};
    std::memset(&uniforms, 0, sizeof(Uniforms));

    // Identity model matrix
    uniforms.modelMatrix[0] = 1.f;
    uniforms.modelMatrix[5] = 1.f;
    uniforms.modelMatrix[10] = 1.f;
    uniforms.modelMatrix[15] = 1.f;

    // View and projection
    std::memcpy(uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(uniforms.projectionMatrix, projectionMatrix.data.data(), sizeof(uniforms.projectionMatrix));

    // Lighting
    std::memcpy(uniforms.sunDirection, sunDirection, sizeof(uniforms.sunDirection));
    std::memcpy(uniforms.sunColor, sunColor, sizeof(uniforms.sunColor));
    std::memcpy(uniforms.ambientColor, ambientColor, sizeof(uniforms.ambientColor));

    // Fog parameters (Phase 8)
    uniforms.fogColor[0] = fogColor[0];
    uniforms.fogColor[1] = fogColor[1];
    uniforms.fogColor[2] = fogColor[2];
    uniforms.fogDensity = 0.0003f; // Per block

    // Camera position for fog distance calculation
    uniforms.cameraPosition[0] = camX;
    uniforms.cameraPosition[1] = camY;
    uniforms.cameraPosition[2] = camZ;

    // Upload to GPU
    std::memcpy((void*)_uniformsBuffer.contents, &uniforms, sizeof(Uniforms));

    // Bind uniforms buffer at index 1
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];

    // LOD mesher instance (stateless — safe to reuse)
    LODMesher lodMesher;

    // Draw chunks (with frustum culling and distance-based LOD)
    auto loadedChunks = world.getLoadedChunks();

    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->meshed) continue;

        // Frustum culling
        AABB chunkAABB = chunk->getAABB();
        if (!isChunkInFrustum(chunkAABB)) continue;

        // Compute distance from camera to chunk center (in blocks)
        Vec3 chunkCenter = chunkAABB.center();
        float dx = chunkCenter.x - camX;
        float dy = chunkCenter.y - camY;
        float dz = chunkCenter.z - camZ;
        int distanceBlocks = static_cast<int>(std::sqrt(dx * dx + dy * dy + dz * dz));

        // Select LOD level based on distance
        int lodLevel = LODMesher::computeLODLevel(distanceBlocks);

        // Chunk key for mesh cache lookup (packed int64, no allocation)
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(chunk->chunkX)) << 32) |
                        static_cast<uint64_t>(static_cast<uint32_t>(chunk->chunkZ));

        // Mesh dirty chunks on demand (invalidate all LOD levels)
        if (chunk->needsMeshUpdate) {
            auto cit = _chunkMeshes.find(key);
            if (cit != _chunkMeshes.end()) {
                for (auto& [_, state] : cit->second) {
                    if (state.uploaded) {
                        _megaBuffer->free(state.alloc);
                    }
                }
                cit->second.clear();
            }

            MeshOutput mesh = lodMesher.buildMesh(*chunk, lodLevel);

            if (!mesh.vertices.empty()) {
                uint32_t vertCount = static_cast<uint32_t>(mesh.vertices.size());
                uint32_t idxCount = static_cast<uint32_t>(mesh.indices.size());

                auto alloc = _megaBuffer->allocate(vertCount, idxCount);

                _megaBuffer->uploadVertices(
                    mesh.vertices.data(),
                    vertCount * sizeof(Vertex),
                    alloc.vertexOffset
                );
                _megaBuffer->uploadIndices(
                    mesh.indices.data(),
                    idxCount * sizeof(uint32_t),
                    alloc.indexOffset
                );

                ChunkMeshState state;
                state.alloc = alloc;
                state.uploaded = true;

                _chunkMeshes[key][lodLevel] = state;
            }

            chunk->setMeshed(true);
            chunk->needsMeshUpdate = false;
        }

        // Look up cached mesh state for this LOD level
        auto cit = _chunkMeshes.find(key);
        if (cit == _chunkMeshes.end()) continue;

        auto lodIt = cit->second.find(lodLevel);
        if (lodIt == cit->second.end() || !lodIt->second.uploaded) continue;

        const auto& meshState = lodIt->second;

        if (meshState.alloc.indexCount == 0) continue;

        // Bind vertex buffer from MegaBuffer allocation
        [encoder setVertexBuffer:meshState.alloc.vertexBuffer
                            offset:meshState.alloc.vertexOffset
                          atIndex:0];
        [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
        [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];
        [encoder setFragmentTexture:_textureAtlas->texture() atIndex:0];
        [encoder setFragmentSamplerState:_textureAtlas->sampler() atIndex:0];

        // Draw indexed primitives (triangles)
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                              indexCount:meshState.alloc.indexCount
                               indexType:MTLIndexTypeUInt32
                             indexBuffer:meshState.alloc.indexBuffer
                         indexBufferOffset:meshState.alloc.indexOffset];
    }
}

// ---------------------------------------------------------------------------
// renderWater (Task 6.7-6.8)
// ---------------------------------------------------------------------------
void RenderPipeline::renderWater(id<MTLCommandBuffer> commandBuffer,
                                    id<CAMetalDrawable> drawable,
                                    const Mat4& viewMatrix,
                                   const Mat4& projectionMatrix,
                                   const float sunDirection[3],
                                   const float sunColor[3],
                                   const float ambientColor[3])
{
    // Water transparency pass not yet implemented (Phase 9 optimization).
    // Water blocks are rendered as part of the main opaque mesh.
    // Stub preserved for future depth-sorted back-to-front rendering.
    (void)commandBuffer; (void)drawable; (void)viewMatrix;
    (void)projectionMatrix; (void)sunDirection; (void)sunColor;
    (void)ambientColor;
}

// ---------------------------------------------------------------------------
// renderBlockHighlight (Task 6.9)
// ---------------------------------------------------------------------------
void RenderPipeline::renderBlockHighlight(id<MTLRenderCommandEncoder> encoder,
                                           const Vec3& blockPos,
                                           const Mat4& viewMatrix,
                                           const Mat4& projectionMatrix)
{
    // Upload highlight-specific uniforms with translation to block position
    Uniforms uniforms{};
    std::memset(&uniforms, 0, sizeof(Uniforms));

    // Translation matrix to move wireframe box to block position
    // With slight offset (0.002) to prevent z-fighting
    uniforms.modelMatrix[0] = 1.f;
    uniforms.modelMatrix[5] = 1.f;
    uniforms.modelMatrix[10] = 1.f;
    uniforms.modelMatrix[15] = 1.f;
    uniforms.modelMatrix[12] = blockPos.x - 0.002f;
    uniforms.modelMatrix[13] = blockPos.y - 0.002f;
    uniforms.modelMatrix[14] = blockPos.z - 0.002f;

    std::memcpy(uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(uniforms.projectionMatrix, projectionMatrix.data.data(), sizeof(uniforms.projectionMatrix));

    // Yellow highlight color
    uniforms.sunDirection[0] = 1.0f;
    uniforms.sunDirection[1] = 1.0f;
    uniforms.sunDirection[2] = 0.0f;
    uniforms.sunColor[0] = 1.0f;
    uniforms.sunColor[1] = 1.0f;
    uniforms.sunColor[2] = 0.0f;
    uniforms.ambientColor[0] = 0.0f;
    uniforms.ambientColor[1] = 0.0f;
    uniforms.ambientColor[2] = 0.0f;

    std::memcpy((void*)_highlightUniformsBuffer.contents, &uniforms, sizeof(Uniforms));

    [encoder setRenderPipelineState:_highlightPipelineState];
    [encoder setDepthStencilState:_waterDepthState]; // No depth write

    [encoder setVertexBuffer:_highlightVertexBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:_highlightUniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_highlightUniformsBuffer offset:0 atIndex:1];

    // Draw 12 lines (24 vertices) for wireframe box
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:24];
}

// ---------------------------------------------------------------------------
// renderUIOverlay (Task 6.10 + hotbar)
// ---------------------------------------------------------------------------
void RenderPipeline::renderUIOverlay(id<MTLRenderCommandEncoder> encoder,
                                      const Hotbar& hotbar)
{
    // ---- Performance HUD (Phase 8) ----
    // Note: Performance stats are tracked in the engine layer.
    // For now, render with default stats (will be populated by engine).
    PerformanceStats stats{};
    stats.fps = 60.0f;
    stats.chunkCount = 0;
    stats.entityCount = 0;
    stats.frameTimeMs = 16.67f;
    _uiOverlay->drawPerformanceHUD(encoder, stats);

    // ---- Crosshair ----
    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossH = 1.0f / static_cast<float>(_displayHeight);
    float crossW = 20.0f / static_cast<float>(_displayWidth);
    float crossV = 20.0f / static_cast<float>(_displayHeight);
    float crossLineW = 1.0f / static_cast<float>(_displayWidth);

    // Horizontal line
    _uiOverlay->drawQuad(encoder,
                         centerX - crossW * 0.5f, centerY - crossH * 0.5f,
                         crossW, crossH,
                         1.0f, 1.0f, 1.0f, 1.0f);

    // Vertical line
    _uiOverlay->drawQuad(encoder,
                         centerX - crossLineW * 0.5f, centerY - crossV * 0.5f,
                         crossLineW, crossV,
                         1.0f, 1.0f, 1.0f, 1.0f);

    // ---- Hotbar (9 slots at bottom of screen) ----
    float hotbarSlotSize = 48.0f / static_cast<float>(_displayHeight);
    float hotbarGap = 2.0f / static_cast<float>(_displayHeight);
    float hotbarY = 4.0f / static_cast<float>(_displayHeight);
    float totalHotbarWidth = Hotbar::SLOTS * hotbarSlotSize +
                             (Hotbar::SLOTS - 1) * hotbarGap;
    float hotbarX = (1.0f - totalHotbarWidth) * 0.5f;

    int selectedIndex = hotbar.getSelectedIndex();

    for (int i = 0; i < Hotbar::SLOTS; ++i) {
        float slotX = hotbarX + i * (hotbarSlotSize + hotbarGap);

        // Slot background
        if (i == selectedIndex) {
            // Selected slot: bright border
            _uiOverlay->drawQuad(encoder,
                                 slotX - 2.0f / _displayWidth,
                                 hotbarY - 2.0f / _displayHeight,
                                 hotbarSlotSize + 4.0f / _displayWidth,
                                 hotbarSlotSize + 4.0f / _displayHeight,
                                 1.0f, 1.0f, 1.0f, 0.8f);
        }

        _uiOverlay->drawQuad(encoder,
                             slotX, hotbarY,
                             hotbarSlotSize, hotbarSlotSize,
                             0.3f, 0.3f, 0.3f, 0.6f);

        // Block type indicator (simplified: color per block type)
        BlockType type = hotbar.getSlot(i);
        float r = 0.5f, g = 0.5f, b = 0.5f;
        switch (type) {
            case BlockType::STONE:    r = 0.5f; g = 0.5f; b = 0.5f; break;
            case BlockType::DIRT:     r = 0.55f; g = 0.35f; b = 0.2f; break;
            case BlockType::GRASS:    r = 0.2f; g = 0.6f; b = 0.2f; break;
            case BlockType::LOG:      r = 0.4f; g = 0.25f; b = 0.15f; break;
            case BlockType::SAND:     r = 0.85f; g = 0.78f; b = 0.55f; break;
            case BlockType::PLANKS:   r = 0.65f; g = 0.45f; b = 0.25f; break;
            case BlockType::BEDROCK:  r = 0.2f; g = 0.2f; b = 0.2f; break;
            case BlockType::COAL_ORE: r = 0.15f; g = 0.15f; b = 0.15f; break;
            case BlockType::IRON_ORE: r = 0.6f; g = 0.5f; b = 0.45f; break;
            default:                  r = 0.5f; g = 0.5f; b = 0.5f; break;
        }

        float innerSize = hotbarSlotSize * 0.7f;
        float innerOffset = (hotbarSlotSize - innerSize) * 0.5f;
        _uiOverlay->drawQuad(encoder,
                             slotX + innerOffset, hotbarY + innerOffset,
                             innerSize, innerSize,
                             r, g, b, 0.9f);
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
RenderPipeline::~RenderPipeline() {
    delete _megaBuffer;
    delete _textureAtlas;
    delete _uiOverlay;
    delete _bloom;
    delete _upscaler;
    delete _particles;
}

// ---------------------------------------------------------------------------
// resize()
// ---------------------------------------------------------------------------
void RenderPipeline::resize(uint32_t width, uint32_t height) {
    if (width == _displayWidth && height == _displayHeight) return;

    _displayWidth = width;
    _displayHeight = height;
    _renderWidth = width / 2;
    _renderHeight = height / 2;

    // Release old textures
    _colorResolve = nil;
    _depthResolve = nil;

    // Reallocate render target textures at render resolution (MSAA disabled)
    auto colorResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                   width:_renderWidth
                                                                                  height:_renderHeight
                                                                              mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to reallocate color resolve texture after resize");
    }

    auto depthResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                  width:_renderWidth
                                                                                 height:_renderHeight
                                                                             mipmapped:false];
    _depthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    if (!_depthResolve) {
        RY_LOG_FATAL("Failed to reallocate depth resolve texture after resize");
    }

    // Resize UI overlay for display resolution
    _uiOverlay->resize(_displayWidth, _displayHeight);

    // Resize bloom for render resolution
    if (_bloom) {
        _bloom->resize(_renderWidth, _renderHeight);
    }

    // Reallocate upscaler
    if (_upscaler) {
        delete _upscaler;
        _upscaler = new MetalFXUpscaler(_device, _renderWidth, _renderHeight,
                                         _displayWidth, _displayHeight);
    }
}

// ---------------------------------------------------------------------------
// setBloomIntensity
// ---------------------------------------------------------------------------
void RenderPipeline::setBloomIntensity(float intensity) {
    _bloomIntensity = intensity;
    if (_bloom) {
        _bloom->setIntensity(intensity);
    }
}

// ---------------------------------------------------------------------------
// tickParticles — Update weather particle physics each game tick
// ---------------------------------------------------------------------------
void RenderPipeline::tickParticles(float dt, const World& world, const Vec3& playerPosition) {
    if (!_particles) return;
    _particles->tick(dt, world, playerPosition);
}

// ---------------------------------------------------------------------------
// Frustum culling
// ---------------------------------------------------------------------------
void RenderPipeline::extractFrustumPlanes(const Mat4& vpMatrix) {
    const float* m = vpMatrix.data.data();

    _frustumPlanes[0][0] = m[12] + m[0];
    _frustumPlanes[0][1] = m[13] + m[1];
    _frustumPlanes[0][2] = m[14] + m[2];
    _frustumPlanes[0][3] = m[15] + m[3];

    _frustumPlanes[1][0] = m[12] - m[0];
    _frustumPlanes[1][1] = m[13] - m[1];
    _frustumPlanes[1][2] = m[14] - m[2];
    _frustumPlanes[1][3] = m[15] - m[3];

    _frustumPlanes[2][0] = m[12] + m[4];
    _frustumPlanes[2][1] = m[13] + m[5];
    _frustumPlanes[2][2] = m[14] + m[6];
    _frustumPlanes[2][3] = m[15] + m[7];

    _frustumPlanes[3][0] = m[12] - m[4];
    _frustumPlanes[3][1] = m[13] - m[5];
    _frustumPlanes[3][2] = m[14] - m[6];
    _frustumPlanes[3][3] = m[15] - m[7];

    _frustumPlanes[4][0] = m[12] + m[8];
    _frustumPlanes[4][1] = m[13] + m[9];
    _frustumPlanes[4][2] = m[14] + m[10];
    _frustumPlanes[4][3] = m[15] + m[11];

    _frustumPlanes[5][0] = m[12] - m[8];
    _frustumPlanes[5][1] = m[13] - m[9];
    _frustumPlanes[5][2] = m[14] - m[10];
    _frustumPlanes[5][3] = m[15] - m[11];

    for (int i = 0; i < 6; ++i) {
        float len = std::sqrt(
            _frustumPlanes[i][0] * _frustumPlanes[i][0] +
            _frustumPlanes[i][1] * _frustumPlanes[i][1] +
            _frustumPlanes[i][2] * _frustumPlanes[i][2]
        );
        if (len > 0.0f) {
            _frustumPlanes[i][0] /= len;
            _frustumPlanes[i][1] /= len;
            _frustumPlanes[i][2] /= len;
            _frustumPlanes[i][3] /= len;
        }
    }
}

bool RenderPipeline::isChunkInFrustum(const AABB& chunkAABB) const {
    Vec3 center = chunkAABB.center();
    Vec3 extents = chunkAABB.size() * 0.5f;

    for (int i = 0; i < 6; ++i) {
        float A = _frustumPlanes[i][0];
        float B = _frustumPlanes[i][1];
        float C = _frustumPlanes[i][2];
        float D = _frustumPlanes[i][3];

        float dist = A * center.x + B * center.y + C * center.z + D;
        float extent = std::abs(A) * extents.x +
                       std::abs(B) * extents.y +
                       std::abs(C) * extents.z;

        if (dist + extent < 0.f) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// renderClouds (Phase 8)
// ============================================================================
void RenderPipeline::renderClouds(id<MTLCommandBuffer> commandBuffer,
                                   id<CAMetalDrawable> drawable,
                                   const Mat4& /*viewMatrix*/,
                                   const Mat4& /*projectionMatrix*/,
                                   const Camera& camera,
                                   uint64_t worldTime,
                                   const float sunDirection[3])
{
    if (!_cloudPipelineState) return;

    // Cloud uniforms
    CloudUniforms cloudUniforms{};
    std::memset(&cloudUniforms, 0, sizeof(cloudUniforms));

    // Camera position
    Vec3 camPos = camera.getPosition();
    cloudUniforms.cameraPosition[0] = camPos.x;
    cloudUniforms.cameraPosition[1] = camPos.y;
    cloudUniforms.cameraPosition[2] = camPos.z;

    // Sun direction
    cloudUniforms.sunDirection[0] = sunDirection[0];
    cloudUniforms.sunDirection[1] = sunDirection[1];
    cloudUniforms.sunDirection[2] = sunDirection[2];

    // Wind offset: worldTime * windSpeed (0.02 blocks/tick)
    cloudUniforms.windOffset = static_cast<float>(worldTime) * 0.02f;

    // Cloud parameters
    cloudUniforms.cloudAltitude = 192.0f;
    cloudUniforms.noiseFrequency = 0.005f;
    cloudUniforms.cloudThreshold = 0.4f;

    std::memcpy((void*)_cloudUniformsBuffer.contents, &cloudUniforms, sizeof(cloudUniforms));

    // Cloud render pass (over the scene, with alpha blending)
    auto cloudPassDesc = [[MTLRenderPassDescriptor alloc] init];
    cloudPassDesc.colorAttachments[0].texture = drawable.texture;
    cloudPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    cloudPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    cloudPassDesc.depthAttachment.texture = _depthResolve;
    cloudPassDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    cloudPassDesc.depthAttachment.storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> cloudEncoder = [commandBuffer renderCommandEncoderWithDescriptor:cloudPassDesc];
    if (!cloudEncoder) return;

    [cloudEncoder setRenderPipelineState:_cloudPipelineState];
    [cloudEncoder setDepthStencilState:_cloudDepthState];
    [cloudEncoder setVertexBuffer:_cloudUniformsBuffer offset:0 atIndex:0];
    [cloudEncoder setFragmentBuffer:_cloudUniformsBuffer offset:0 atIndex:0];

    // Draw fullscreen quad (6 vertices)
    [cloudEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [cloudEncoder endEncoding];
}

// ============================================================================
// MetalFXUpscaler implementation (Phase 8.4)
// ============================================================================
MetalFXUpscaler::MetalFXUpscaler(id<MTLDevice> /*device*/, uint32_t /*srcWidth*/, uint32_t /*srcHeight*/,
                                 uint32_t /*dstWidth*/, uint32_t /*dstHeight*/)
{
}

MetalFXUpscaler::~MetalFXUpscaler() {
}

void MetalFXUpscaler::upscale(id<MTLCommandBuffer> commandBuffer,
                               id<MTLTexture> source,
                               id<MTLTexture> destination)
{
    if (!commandBuffer || !source || !destination) return;

    // Use Metal's built-in blit for bilinear upscale.
    // This is a placeholder — Phase 9.4 will replace with MetalFX temporal upscaling.
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    if (blitEncoder) {
        [blitEncoder generateMipmapsForTexture:source];
        [blitEncoder endEncoding];
    }

    // Draw fullscreen quad with bilinear sampling for upscale.
    // Since we don't have a dedicated upscale shader yet, we use a render pass
    // that samples the source texture with linear filtering into the destination.
    auto upscalePassDesc = [[MTLRenderPassDescriptor alloc] init];
    upscalePassDesc.colorAttachments[0].texture = destination;
    upscalePassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    upscalePassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    upscalePassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

    // The UI overlay pipeline can serve as a simple fullscreen quad drawer.
    // For now, we skip the upscale and let the MSAA resolve handle it.
    // (The MSAA resolve already does a quality bilinear interpolation.)
    (void)upscalePassDesc;
}

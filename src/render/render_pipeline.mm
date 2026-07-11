#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "render/mesher.hpp"
#include "world/chunk.hpp"
#include "world/world.hpp"
#include "engine/camera.hpp"

#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Uniforms — matches the Metal shader's Uniforms struct exactly.
// ---------------------------------------------------------------------------
struct alignas(16) Uniforms {
    float modelMatrix[16];
    float viewMatrix[16];
    float projectionMatrix[16];
    float sunDirection[3];
    float _pad0;
    float sunColor[3];
    float _pad1;
    float ambientColor[3];
    float _pad2;
};

static_assert(sizeof(Uniforms) <= 256, "Uniforms must fit in 256-byte buffer");

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RenderPipeline::RenderPipeline(id<MTLDevice> device,
                               id<MTLLibrary> shaderLibrary,
                               uint32_t width,
                               uint32_t height)
    : _device(device)
    , _width(width)
    , _height(height)
    , _frustumPlanes{}
{
    // ---- Load shader functions ----
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    if (!vertexFunc) {
        RY_LOG_FATAL("Failed to load vertex shader function 'vertexMain'");
    }

    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];
    if (!fragmentFunc) {
        RY_LOG_FATAL("Failed to load fragment shader function 'fragmentMain'");
    }

    // ---- Vertex descriptor (Metal 2.0 API) ----
    // Matches Vertex struct layout:
    //   offset 0:  normalIdx   (uint32_t = 4 bytes)   → UInt
    //   offset 4:  px, py, pz  (3 × float16_t = 6 bytes) → Half3
    //   offset 10: u, v        (2 × float16_t = 4 bytes) → Half2
    //   stride = 16 bytes

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

    // ---- Render pipeline state ----
    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;

    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                            error:&error];
    if (_pipelineState == nil) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create render pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Depth stencil state ----
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create depth stencil state");
    }

    // ---- MSAA textures ----
    auto colorMSAADesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                            width:_width
                                                                           height:_height
                                                                       mipmapped:false];
    colorMSAADesc.usage = MTLTextureUsageRenderTarget;
    colorMSAADesc.sampleCount = 4;
    _colorMSAA = [_device newTextureWithDescriptor:colorMSAADesc];
    if (!_colorMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA color texture");
    }

    auto depthMSAADesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                            width:_width
                                                                           height:_height
                                                                       mipmapped:false];
    depthMSAADesc.usage = MTLTextureUsageRenderTarget;
    depthMSAADesc.sampleCount = 4;
    _depthMSAA = [_device newTextureWithDescriptor:depthMSAADesc];
    if (!_depthMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA depth texture");
    }

    // ---- Resolve textures (single-sample) ----
    auto colorResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                               width:_width
                                                                              height:_height
                                                                          mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to allocate color resolve texture");
    }

    auto depthResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                width:_width
                                                                               height:_height
                                                                           mipmapped:false];
    _depthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    if (!_depthResolve) {
        RY_LOG_FATAL("Failed to allocate depth resolve texture");
    }

    // ---- Uniforms buffer (256 bytes) ----
    _uniformsBuffer = [_device newBufferWithLength:256
                                            options:MTLResourceStorageModeShared];
    if (!_uniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate uniforms buffer");
    }
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------
void RenderPipeline::render(id<MTLCommandQueue> queue,
                            id<CAMetalDrawable> drawable,
                            const Mat4& viewMatrix,
                            const Mat4& projectionMatrix,
                            const World& world,
                            const Camera& /*camera*/)
{
    if (!drawable || !queue) return;

    // Compute VP matrix and extract frustum planes
    Mat4 vpMatrix = projectionMatrix * viewMatrix;
    extractFrustumPlanes(vpMatrix);

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return;

    // ---- Render pass descriptor ----
    auto renderPassDesc = [[MTLRenderPassDescriptor alloc] init];

    // Color attachment: render to MSAA, resolve to drawable
    renderPassDesc.colorAttachments[0].texture = _colorMSAA;
    renderPassDesc.colorAttachments[0].resolveTexture = drawable.texture;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.53f, 0.81f, 0.92f, 1.0f);

    // Depth attachment: render to MSAA depth
    renderPassDesc.depthAttachment.texture = _depthMSAA;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionStore;
    renderPassDesc.depthAttachment.clearDepth = 1.0;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) return;

    // Bind pipeline state
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];

    // ---- Pack and upload uniforms ----
    Uniforms uniforms{};
    std::memset(&uniforms, 0, sizeof(Uniforms));

    // Identity model matrix
    uniforms.modelMatrix[0] = 1.f;
    uniforms.modelMatrix[5] = 1.f;
    uniforms.modelMatrix[10] = 1.f;
    uniforms.modelMatrix[15] = 1.f;

    // View and projection (column-major, matches Mat4::data layout)
    std::memcpy(uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(uniforms.projectionMatrix, projectionMatrix.data.data(), sizeof(uniforms.projectionMatrix));

    // Default lighting: sun from upper-right-front
    uniforms.sunDirection[0] = 0.5f;
    uniforms.sunDirection[1] = 0.8f;
    uniforms.sunDirection[2] = 0.3f;
    float sunLen = std::sqrt(
        uniforms.sunDirection[0] * uniforms.sunDirection[0] +
        uniforms.sunDirection[1] * uniforms.sunDirection[1] +
        uniforms.sunDirection[2] * uniforms.sunDirection[2]);
    uniforms.sunDirection[0] /= sunLen;
    uniforms.sunDirection[1] /= sunLen;
    uniforms.sunDirection[2] /= sunLen;

    uniforms.sunColor[0] = 1.0f;
    uniforms.sunColor[1] = 0.95f;
    uniforms.sunColor[2] = 0.9f;

    uniforms.ambientColor[0] = 0.35f;
    uniforms.ambientColor[1] = 0.35f;
    uniforms.ambientColor[2] = 0.4f;

    // Upload to GPU
    std::memcpy((void*)_uniformsBuffer.contents, &uniforms, sizeof(Uniforms));

    // Bind uniforms buffer to both vertex and fragment stages
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:0];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:0];

    // ---- Draw chunks (with frustum culling) ----
    auto loadedChunks = world.getLoadedChunks();

    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->meshed) continue;

        // Frustum culling
        AABB chunkAABB = chunk->getAABB();
        if (!isChunkInFrustum(chunkAABB)) continue;

        // Chunk key for mesh cache lookup
        std::string key = std::to_string(chunk->chunkX) + "," +
                          std::to_string(chunk->chunkZ);

        // Mesh dirty chunks on demand
        if (chunk->needsMeshUpdate) {
            auto it = _chunkMeshes.find(key);
            if (it != _chunkMeshes.end() && it->second.uploaded) {
                // TODO: free old allocation via MegaBuffer
            }

            GreedyMesher mesher;
            MeshOutput mesh = mesher.buildMesh(*chunk);

            if (!mesh.vertices.empty()) {
                ChunkMeshState state;
                state.alloc.vertexBuffer = nullptr;
                state.alloc.indexBuffer = nullptr;
                state.alloc.vertexOffset = 0;
                state.alloc.indexOffset = 0;
                state.alloc.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
                state.alloc.indexCount = static_cast<uint32_t>(mesh.indices.size());
                state.uploaded = true;

                // TODO: allocate via MegaBuffer and upload vertices/indices
                // For now, store metadata; actual GPU upload will be wired
                // when MegaBuffer is integrated into the render pipeline.

                _chunkMeshes[key] = state;
            }

            chunk->setMeshed(true);
            chunk->needsMeshUpdate = false;
        }

        // Look up cached mesh state
        auto it = _chunkMeshes.find(key);
        if (it == _chunkMeshes.end() || !it->second.uploaded) continue;

        const auto& meshState = it->second;

        // TODO: bind vertex/index buffers from MegaBuffer allocation
        // and call drawPrimitives:triangle vertexCount:...
        // Placeholder draw call (will be filled when MegaBuffer is wired):
        if (meshState.alloc.vertexCount > 0 && meshState.alloc.vertexBuffer) {
            [encoder setVertexBuffer:meshState.alloc.vertexBuffer
                              offset:meshState.alloc.vertexOffset
                            atIndex:0];
            [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];

            [encoder setVertexBuffer:meshState.alloc.indexBuffer
                              offset:meshState.alloc.indexOffset
                            atIndex:0];

            [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                         vertexStart:0
                       vertexCount:meshState.alloc.vertexCount];
        }
    }

    [encoder endEncoding];

    // Present and commit
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

// ---------------------------------------------------------------------------
// resize()
// ---------------------------------------------------------------------------
void RenderPipeline::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height) return;

    _width = width;
    _height = height;

    // Release old textures
    _colorMSAA = nil;
    _colorResolve = nil;
    _depthMSAA = nil;
    _depthResolve = nil;

    // Reallocate MSAA textures
    auto colorMSAADesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                            width:_width
                                                                           height:_height
                                                                       mipmapped:false];
    colorMSAADesc.usage = MTLTextureUsageRenderTarget;
    colorMSAADesc.sampleCount = 4;
    _colorMSAA = [_device newTextureWithDescriptor:colorMSAADesc];
    if (!_colorMSAA) {
        RY_LOG_FATAL("Failed to reallocate MSAA color texture after resize");
    }

    auto depthMSAADesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                            width:_width
                                                                           height:_height
                                                                       mipmapped:false];
    depthMSAADesc.usage = MTLTextureUsageRenderTarget;
    depthMSAADesc.sampleCount = 4;
    _depthMSAA = [_device newTextureWithDescriptor:depthMSAADesc];
    if (!_depthMSAA) {
        RY_LOG_FATAL("Failed to reallocate MSAA depth texture after resize");
    }

    // Reallocate resolve textures
    auto colorResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                               width:_width
                                                                              height:_height
                                                                          mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to reallocate color resolve texture after resize");
    }

    auto depthResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                width:_width
                                                                               height:_height
                                                                           mipmapped:false];
    _depthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    if (!_depthResolve) {
        RY_LOG_FATAL("Failed to reallocate depth resolve texture after resize");
    }
}

// ---------------------------------------------------------------------------
// Frustum culling
// ---------------------------------------------------------------------------
void RenderPipeline::extractFrustumPlanes(const Mat4& vpMatrix) {
    // Extract 6 frustum planes from view-projection matrix.
    // Each plane is stored as [A, B, C, D] representing Ax + By + Cz + D = 0.
    //
    // VP matrix is column-major: data[col * 4 + row]
    // Column 0: data[0..3],  Column 1: data[4..7],
    // Column 2: data[8..11], Column 3: data[12..15]

    const float* m = vpMatrix.data.data();

    // Left plane:  col3 + col0
    _frustumPlanes[0][0] = m[12] + m[0];
    _frustumPlanes[0][1] = m[13] + m[1];
    _frustumPlanes[0][2] = m[14] + m[2];
    _frustumPlanes[0][3] = m[15] + m[3];

    // Right plane:  col3 - col0
    _frustumPlanes[1][0] = m[12] - m[0];
    _frustumPlanes[1][1] = m[13] - m[1];
    _frustumPlanes[1][2] = m[14] - m[2];
    _frustumPlanes[1][3] = m[15] - m[3];

    // Bottom plane:  col3 + col1
    _frustumPlanes[2][0] = m[12] + m[4];
    _frustumPlanes[2][1] = m[13] + m[5];
    _frustumPlanes[2][2] = m[14] + m[6];
    _frustumPlanes[2][3] = m[15] + m[7];

    // Top plane:  col3 - col1
    _frustumPlanes[3][0] = m[12] - m[4];
    _frustumPlanes[3][1] = m[13] - m[5];
    _frustumPlanes[3][2] = m[14] - m[6];
    _frustumPlanes[3][3] = m[15] - m[7];

    // Near plane:  col3 + col2
    _frustumPlanes[4][0] = m[12] + m[8];
    _frustumPlanes[4][1] = m[13] + m[9];
    _frustumPlanes[4][2] = m[14] + m[10];
    _frustumPlanes[4][3] = m[15] + m[11];

    // Far plane:  col3 - col2
    _frustumPlanes[5][0] = m[12] - m[8];
    _frustumPlanes[5][1] = m[13] - m[9];
    _frustumPlanes[5][2] = m[14] - m[10];
    _frustumPlanes[5][3] = m[15] - m[11];
}

bool RenderPipeline::isChunkInFrustum(const AABB& chunkAABB) const {
    Vec3 center = chunkAABB.center();
    Vec3 extents = chunkAABB.size() * 0.5f;

    // Test AABB against all 6 frustum planes.
    // For each plane (A, B, C, D), compute the signed distance from the
    // AABB center to the plane. If the distance + maximum projection of
    // the AABB extents onto the plane normal is < 0, the AABB is entirely
    // outside the plane — not visible.
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

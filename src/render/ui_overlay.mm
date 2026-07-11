#import "render/ui_overlay.hpp"

#include "common/error.hpp"

#include <cstring>

// ---------------------------------------------------------------------------
// UIOverlay vertex layout (matches Metal shader):
//   attribute(0) float2 position  — offset 0
//   attribute(1) float4 color     — offset 8
//   stride = 16 bytes per vertex
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
UIOverlay::UIOverlay(id<MTLDevice> device,
                     id<MTLLibrary> shaderLibrary,
                     uint32_t width,
                     uint32_t height)
    : _device(device)
    , _width(width)
    , _height(height)
{
    // ---- Load shader functions ----
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"uiVertexMain"];
    if (!vertexFunc) {
        RY_LOG_FATAL("Failed to load UI vertex shader function 'uiVertexMain'");
    }

    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"uiFragmentMain"];
    if (!fragmentFunc) {
        RY_LOG_FATAL("Failed to load UI fragment shader function 'uiFragmentMain'");
    }

    // ---- Vertex descriptor ----
    auto vertexDesc = [MTLVertexDescriptor vertexDescriptor];

    // Attribute 0: position (float2)
    vertexDesc.attributes[0].format = MTLVertexFormatFloat2;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;

    // Attribute 1: color (float4)
    vertexDesc.attributes[1].format = MTLVertexFormatFloat4;
    vertexDesc.attributes[1].offset = 8;
    vertexDesc.attributes[1].bufferIndex = 0;

    // Buffer layout: 16 bytes per vertex
    vertexDesc.layouts[0].stride = 16;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDesc.layouts[0].stepRate = 1;

    // ---- Render pipeline state ----
    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;

    // Blend with existing content (alpha blending)
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.colorAttachments[0].blendingEnabled = true;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                            error:&error];
    if (!_pipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create UI overlay pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Allocate buffers ----
    // Dynamic buffer: 4 vertices × 16 bytes = 64 bytes
    _dynamicBuffer = [_device newBufferWithLength:64
                                           options:MTLResourceStorageModeShared];
    if (!_dynamicBuffer) {
        RY_LOG_FATAL("Failed to allocate UI overlay dynamic buffer");
    }

    // Index buffer: 6 indices × 2 bytes = 12 bytes
    _quadIndexBuffer = [_device newBufferWithLength:12
                                             options:MTLResourceStorageModeShared];
    if (!_quadIndexBuffer) {
        RY_LOG_FATAL("Failed to allocate UI overlay index buffer");
    }

    // Indices for a triangle strip quad (0,1,2, 0,2,3)
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};
    std::memcpy((void*)_quadIndexBuffer.contents, indices, sizeof(indices));

    // Projection matrix buffer (16 floats = 64 bytes)
    _projectionBuffer = [_device newBufferWithLength:64
                                              options:MTLResourceStorageModeShared];
    if (!_projectionBuffer) {
        RY_LOG_FATAL("Failed to allocate UI overlay projection buffer");
    }
    buildProjectionMatrix();
}

// ---------------------------------------------------------------------------
// drawQuad()
// ---------------------------------------------------------------------------
void UIOverlay::drawQuad(id<MTLRenderCommandEncoder> encoder,
                         float x, float y,
                         float w, float h,
                         float r, float g, float b, float a)
{
    if (!encoder) return;

    // Build quad vertices and upload to dynamic buffer
    buildQuadVertices(x, y, w, h, r, g, b, a);

    // Bind pipeline and buffers
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setVertexBuffer:_dynamicBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:_projectionBuffer offset:0 atIndex:1];

    // Draw indexed quad (2 triangles)
    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:6
                          indexType:MTLIndexTypeUInt16
                        indexBuffer:_quadIndexBuffer
                    indexBufferOffset:0];
}

// ---------------------------------------------------------------------------
// resize()
// ---------------------------------------------------------------------------
void UIOverlay::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height) return;

    _width = width;
    _height = height;

    buildProjectionMatrix();
}

// ---------------------------------------------------------------------------
// buildQuadVertices()
// ---------------------------------------------------------------------------
void UIOverlay::buildQuadVertices(float x, float y, float w, float h,
                                  float r, float g, float b, float a) {
    // Screen-space quad vertices (bottom-left origin, normalized coords).
    // Layout: [x, y, r, g, b, a, 0, 0] — 16 bytes per vertex.
    struct alignas(16) QuadVertex {
        float px, py;
        float cr, cg, cb, ca;
    };

    QuadVertex vertices[4];

    // Bottom-left
    vertices[0] = {x, y, r, g, b, a};
    // Top-left
    vertices[1] = {x, y + h, r, g, b, a};
    // Bottom-right
    vertices[2] = {x + w, y, r, g, b, a};
    // Top-right
    vertices[3] = {x + w, y + h, r, g, b, a};

    std::memcpy((void*)_dynamicBuffer.contents, vertices, sizeof(vertices));
}

// ---------------------------------------------------------------------------
// buildProjectionMatrix()
// ---------------------------------------------------------------------------
void UIOverlay::buildProjectionMatrix() {
    // Orthographic projection that maps normalized screen coords [0,1] to NDC.
    // This creates a 2×2 identity-like matrix so that screen coords pass through
    // directly to clip space (Metal's NDC is [-1,1] in x,y).
    //
    // We map: screen(0,0) → NDC(-1,-1), screen(1,1) → NDC(1,1)
    //   x_ndc = 2*x_screen - 1
    //   y_ndc = 2*y_screen - 1
    //
    // Column-major 4×4:
    float proj[16] = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
       -1.0f,-1.0f, 0.0f, 1.0f,
    };

    std::memcpy((void*)_projectionBuffer.contents, proj, sizeof(proj));
}

#import "render/ui_overlay.hpp"

#include "common/error.hpp"

#include <array>
#include <cstring>
#include <cmath>

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

// ============================================================================
// Performance HUD (Phase 8)
// ============================================================================

// ---- 8×8 bitmap font data ----
// Each character is 8 bytes (one byte per row), bits 7-0 = left-to-right pixels
// Using std::array to avoid C99 compound literal issues
static uint8_t getCharRow(char c, int row) {
    if (row < 0 || row >= 8) return 0x00;
    switch (c) {
        case '0': { uint8_t d[] = {0x1C,0x22,0x22,0x22,0x22,0x22,0x1C,0x00}; return d[row]; }
        case '1': { uint8_t d[] = {0x08,0x18,0x08,0x08,0x08,0x08,0x1E,0x00}; return d[row]; }
        case '2': { uint8_t d[] = {0x1C,0x22,0x02,0x0C,0x10,0x20,0x3E,0x00}; return d[row]; }
        case '3': { uint8_t d[] = {0x1C,0x22,0x02,0x1C,0x02,0x22,0x1C,0x00}; return d[row]; }
        case '4': { uint8_t d[] = {0x04,0x0C,0x14,0x24,0x3E,0x04,0x04,0x00}; return d[row]; }
        case '5': { uint8_t d[] = {0x3E,0x20,0x38,0x02,0x02,0x22,0x1C,0x00}; return d[row]; }
        case '6': { uint8_t d[] = {0x1C,0x20,0x38,0x22,0x22,0x22,0x1C,0x00}; return d[row]; }
        case '7': { uint8_t d[] = {0x3E,0x22,0x04,0x08,0x10,0x10,0x10,0x00}; return d[row]; }
        case '8': { uint8_t d[] = {0x1C,0x22,0x22,0x1C,0x22,0x22,0x1C,0x00}; return d[row]; }
        case '9': { uint8_t d[] = {0x1C,0x22,0x22,0x1E,0x02,0x02,0x1C,0x00}; return d[row]; }
        case 'A': { uint8_t d[] = {0x10,0x18,0x14,0x12,0x3E,0x12,0x12,0x00}; return d[row]; }
        case 'B': { uint8_t d[] = {0x3C,0x22,0x22,0x3C,0x22,0x22,0x3C,0x00}; return d[row]; }
        case 'C': { uint8_t d[] = {0x1C,0x22,0x20,0x20,0x20,0x22,0x1C,0x00}; return d[row]; }
        case 'D': { uint8_t d[] = {0x38,0x24,0x22,0x22,0x24,0x28,0x30,0x00}; return d[row]; }
        case 'E': { uint8_t d[] = {0x3E,0x20,0x20,0x38,0x20,0x20,0x3E,0x00}; return d[row]; }
        case 'F': { uint8_t d[] = {0x3E,0x20,0x20,0x38,0x20,0x20,0x20,0x00}; return d[row]; }
        case 'H': { uint8_t d[] = {0x22,0x22,0x22,0x3E,0x22,0x22,0x22,0x00}; return d[row]; }
        case 'K': { uint8_t d[] = {0x22,0x24,0x28,0x30,0x28,0x24,0x22,0x00}; return d[row]; }
        case 'M': { uint8_t d[] = {0x22,0x36,0x3A,0x2A,0x22,0x22,0x22,0x00}; return d[row]; }
        case 'N': { uint8_t d[] = {0x22,0x32,0x3A,0x36,0x32,0x32,0x32,0x00}; return d[row]; }
        case 'P': { uint8_t d[] = {0x3C,0x22,0x22,0x3C,0x20,0x20,0x20,0x00}; return d[row]; }
        case 'R': { uint8_t d[] = {0x3C,0x22,0x22,0x3C,0x24,0x22,0x22,0x00}; return d[row]; }
        case 'S': { uint8_t d[] = {0x1C,0x22,0x20,0x1C,0x02,0x22,0x1C,0x00}; return d[row]; }
        case 'T': { uint8_t d[] = {0x3E,0x08,0x08,0x08,0x08,0x08,0x08,0x00}; return d[row]; }
        case 'c': { uint8_t d[] = {0x02,0x22,0x3C,0x02,0x22,0x02,0x1C,0x00}; return d[row]; }
        case 'e': { uint8_t d[] = {0x1C,0x20,0x3E,0x22,0x22,0x1C,0x02,0x00}; return d[row]; }
        case 'f': { uint8_t d[] = {0x0C,0x04,0x3C,0x24,0x24,0x24,0x24,0x00}; return d[row]; }
        case 'h': { uint8_t d[] = {0x22,0x22,0x22,0x2C,0x32,0x32,0x32,0x00}; return d[row]; }
        case 'k': { uint8_t d[] = {0x22,0x24,0x28,0x30,0x28,0x24,0x22,0x00}; return d[row]; }
        case 'm': { uint8_t d[] = {0x32,0x3A,0x3A,0x32,0x32,0x32,0x32,0x00}; return d[row]; }
        case 'n': { uint8_t d[] = {0x36,0x32,0x32,0x32,0x32,0x32,0x32,0x00}; return d[row]; }
        case 's': { uint8_t d[] = {0x1C,0x20,0x1C,0x02,0x3E,0x00,0x00,0x00}; return d[row]; }
        case 't': { uint8_t d[] = {0x08,0x08,0x3C,0x08,0x08,0x08,0x1C,0x00}; return d[row]; }
        case '.': { uint8_t d[] = {0x00,0x00,0x00,0x00,0x00,0x08,0x08,0x00}; return d[row]; }
        case ':': { uint8_t d[] = {0x00,0x08,0x08,0x00,0x00,0x08,0x08,0x00}; return d[row]; }
        case '/': { uint8_t d[] = {0x00,0x02,0x04,0x08,0x10,0x20,0x00,0x00}; return d[row]; }
        default: return 0x00;
    }
}

std::array<uint8_t, 8> UIOverlay::getCharBitmap(char c) {
    std::array<uint8_t, 8> bitmap{};
    for (int i = 0; i < 8; ++i) {
        bitmap[i] = getCharRow(c, i);
    }
    return bitmap;
}

void UIOverlay::drawChar(id<MTLRenderCommandEncoder> encoder,
                          char c, float x, float y,
                          float r, float g, float b)
{
    if (!encoder) return;

    auto bitmap = getCharBitmap(c);

    // Draw each lit pixel as a small quad
    float pixelW = 1.0f / static_cast<float>(_width);
    float pixelH = 1.0f / static_cast<float>(_height);

    for (int row = 0; row < FONT_HEIGHT; ++row) {
        uint8_t rowBits = bitmap[row];
        for (int col = 0; col < FONT_WIDTH; ++col) {
            if (rowBits & (0x80 >> col)) {
                float px = x + col * pixelW;
                float py = y + (FONT_HEIGHT - 1 - row) * pixelH;
                drawQuad(encoder, px, py, pixelW, pixelH, r, g, b, 1.0f);
            }
        }
    }
}

float UIOverlay::drawString(id<MTLRenderCommandEncoder> encoder,
                             const char* str, float x, float y,
                             float r, float g, float b)
{
    if (!encoder || !str) return 0.f;

    float cursorX = x;

    while (*str) {
        drawChar(encoder, *str, cursorX, y, r, g, b);
        cursorX += (FONT_WIDTH + 1) / static_cast<float>(_width);
        ++str;
    }

    return cursorX - x;
}

void UIOverlay::intToString(int value, char* buf, size_t bufSize) {
    if (bufSize < 2) return;
    char tmp[20];
    int len = 0;
    if (value == 0) {
        tmp[len++] = '0';
    } else {
        int v = value < 0 ? -value : value;
        while (v > 0) {
            tmp[len++] = '0' + (v % 10);
            v /= 10;
        }
        if (value < 0) tmp[len++] = '-';
        // Reverse
        for (int i = 0; i < len / 2; ++i) {
            char t = tmp[i];
            tmp[i] = tmp[len - 1 - i];
            tmp[len - 1 - i] = t;
        }
    }
    size_t copyLen = len < static_cast<int>(bufSize - 1) ? len : bufSize - 1;
    std::memcpy(buf, tmp, copyLen);
    buf[copyLen] = '\0';
}

void UIOverlay::floatToString(float value, char* buf, size_t bufSize) {
    if (bufSize < 2) return;
    // Simple float formatting: one decimal place
    int intPart = static_cast<int>(std::floor(value));
    int fracPart = static_cast<int>((value - std::floor(value)) * 10);

    char tmp[20];
    intToString(intPart, tmp, sizeof(tmp));
    size_t len = std::strlen(tmp);
    if (len + 3 < bufSize) {
        std::memcpy(buf, tmp, len);
        buf[len] = '.';
        buf[len + 1] = '0' + fracPart;
        buf[len + 2] = '\0';
    } else {
        std::memcpy(buf, tmp, len < bufSize - 1 ? len : bufSize - 1);
        buf[len < bufSize - 1 ? len : bufSize - 1] = '\0';
    }
}

void UIOverlay::drawPerformanceHUD(id<MTLRenderCommandEncoder> encoder,
                                    const PerformanceStats& stats)
{
    if (!encoder) return;

    // HUD position: top-left corner
    float hudX = 8.0f / static_cast<float>(_width);
    float hudY = 1.0f - 8.0f / static_cast<float>(_height); // Top of screen

    // Background: semi-transparent dark rectangle
    float bgWidth = 220.0f / static_cast<float>(_width);
    float bgHeight = 80.0f / static_cast<float>(_height);
    drawQuad(encoder, hudX - 4.0f / _width, hudY - bgHeight,
             bgWidth, bgHeight,
             0.0f, 0.0f, 0.0f, 0.6f);

    // Line height (including spacing)
    float lineHeight = (FONT_HEIGHT + 2) / static_cast<float>(_height);
    float textX = hudX;
    float textY = hudY - bgHeight + 2.0f / _height;

    // FPS
    char fpsBuf[16];
    floatToString(stats.fps, fpsBuf, sizeof(fpsBuf));
    drawString(encoder, "FPS: ", textX, textY, 1.0f, 1.0f, 0.2f);
    drawString(encoder, fpsBuf, textX + 40.0f / _width, textY, 1.0f, 1.0f, 0.2f);
    textY -= lineHeight;

    // Chunks
    char chunkBuf[16];
    intToString(static_cast<int>(stats.chunkCount), chunkBuf, sizeof(chunkBuf));
    drawString(encoder, "Chunks: ", textX, textY, 0.2f, 1.0f, 0.8f);
    drawString(encoder, chunkBuf, textX + 72.0f / _width, textY, 0.2f, 1.0f, 0.8f);
    textY -= lineHeight;

    // Entities
    char entityBuf[16];
    intToString(static_cast<int>(stats.entityCount), entityBuf, sizeof(entityBuf));
    drawString(encoder, "Entities: ", textX, textY, 0.8f, 0.4f, 1.0f);
    drawString(encoder, entityBuf, textX + 88.0f / _width, textY, 0.8f, 0.4f, 1.0f);
    textY -= lineHeight;

    // Frame time
    char ftBuf[16];
    floatToString(stats.frameTimeMs, ftBuf, sizeof(ftBuf));
    drawString(encoder, "Frame: ", textX, textY, 1.0f, 0.8f, 0.4f);
    drawString(encoder, ftBuf, textX + 60.0f / _width, textY, 1.0f, 0.8f, 0.4f);
    (void)textY; // Suppress unused variable warning
}

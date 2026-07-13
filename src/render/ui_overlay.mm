#import "render/ui_overlay.hpp"

#include "common/error.hpp"

#include <cmath>
#include <cstring>

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

    // ---- Vertex descriptor (matches UIVertex: float2 pos, float4 color) ----
    auto vertexDesc = [MTLVertexDescriptor vertexDescriptor];

    vertexDesc.attributes[0].format = MTLVertexFormatFloat2;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;

    vertexDesc.attributes[1].format = MTLVertexFormatFloat4;
    vertexDesc.attributes[1].offset = 8;
    vertexDesc.attributes[1].bufferIndex = 0;

    vertexDesc.layouts[0].stride = sizeof(UIVertex);
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

    // ---- Projection matrix buffer (16 floats = 64 bytes) ----
    _projectionBuffer = [_device newBufferWithLength:64
                                              options:MTLResourceStorageModeShared];
    if (!_projectionBuffer) {
        RY_LOG_FATAL("Failed to allocate UI overlay projection buffer");
    }
    buildProjectionMatrix();

    // Vertex ring slots are allocated lazily in flush() (grow on demand).
    for (int i = 0; i < RING_SLOTS; ++i) {
        _vertexRing[i] = nil;
    }
    _vertices.reserve(4096);
}

// ---------------------------------------------------------------------------
// Frame batching
// ---------------------------------------------------------------------------
void UIOverlay::beginFrame() {
    _vertices.clear();
}

void UIOverlay::flush(id<MTLRenderCommandEncoder> encoder) {
    if (!encoder || _vertices.empty()) return;

    const size_t byteCount = _vertices.size() * sizeof(UIVertex);
    id<MTLBuffer>& slot = _vertexRing[_frameIndex % RING_SLOTS];
    ++_frameIndex;

    if (!slot || slot.length < byteCount) {
        // Grow to the next power of two so steady-state frames never realloc
        size_t capacity = 16384;
        while (capacity < byteCount) capacity *= 2;
        slot = [_device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
        if (!slot) {
            RY_LOG_ERROR("Failed to allocate UI overlay vertex buffer");
            _vertices.clear();
            return;
        }
    }

    std::memcpy((void*)slot.contents, _vertices.data(), byteCount);

    [encoder setRenderPipelineState:_pipelineState];
    [encoder setVertexBuffer:slot offset:0 atIndex:0];
    [encoder setVertexBuffer:_projectionBuffer offset:0 atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:_vertices.size()];

    _vertices.clear();
}

// ---------------------------------------------------------------------------
// drawQuad()
// ---------------------------------------------------------------------------
void UIOverlay::drawQuad(float x, float y,
                         float w, float h,
                         float r, float g, float b, float a)
{
    const UIVertex bl{x, y, r, g, b, a};
    const UIVertex tl{x, y + h, r, g, b, a};
    const UIVertex br{x + w, y, r, g, b, a};
    const UIVertex tr{x + w, y + h, r, g, b, a};

    _vertices.push_back(bl);
    _vertices.push_back(tl);
    _vertices.push_back(br);
    _vertices.push_back(tl);
    _vertices.push_back(tr);
    _vertices.push_back(br);
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
// buildProjectionMatrix()
// ---------------------------------------------------------------------------
void UIOverlay::buildProjectionMatrix() {
    // Orthographic projection that maps normalized screen coords [0,1] to NDC.
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
// Bitmap font text
// ============================================================================

// ---- 8×8 bitmap font data ----
// Each character is 8 bytes (one byte per row), bits 7-0 = left-to-right pixels
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

void UIOverlay::drawChar(char c, float x, float y, float scale,
                          float r, float g, float b)
{
    auto bitmap = getCharBitmap(c);

    // Each lit font pixel becomes one small quad
    float pixelW = scale / static_cast<float>(_width);
    float pixelH = scale / static_cast<float>(_height);

    for (int row = 0; row < FONT_HEIGHT; ++row) {
        uint8_t rowBits = bitmap[row];
        for (int col = 0; col < FONT_WIDTH; ++col) {
            if (rowBits & (0x80 >> col)) {
                float px = x + col * pixelW;
                float py = y + (FONT_HEIGHT - 1 - row) * pixelH;
                drawQuad(px, py, pixelW, pixelH, r, g, b, 1.0f);
            }
        }
    }
}

float UIOverlay::drawString(const char* str, float x, float y, float scale,
                             float r, float g, float b)
{
    if (!str) return 0.f;

    float cursorX = x;

    while (*str) {
        drawChar(*str, cursorX, y, scale, r, g, b);
        cursorX += (FONT_WIDTH + 1) * scale / static_cast<float>(_width);
        ++str;
    }

    return cursorX - x;
}

float UIOverlay::measureString(const char* str, float scale) const {
    if (!str) return 0.f;
    const float advance = (FONT_WIDTH + 1) * scale / static_cast<float>(_width);
    return advance * static_cast<float>(std::strlen(str));
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

void UIOverlay::drawPerformanceHUD(const PerformanceStats& stats)
{
    // HUD position: top-left corner
    float hudX = 8.0f / static_cast<float>(_width);
    float hudY = 1.0f - 8.0f / static_cast<float>(_height); // Top of screen

    // Background: semi-transparent dark rectangle
    float bgWidth = 220.0f / static_cast<float>(_width);
    float bgHeight = 80.0f / static_cast<float>(_height);
    drawQuad(hudX - 4.0f / _width, hudY - bgHeight,
             bgWidth, bgHeight,
             0.0f, 0.0f, 0.0f, 0.6f);

    // Line height (including spacing)
    float lineHeight = (FONT_HEIGHT + 2) / static_cast<float>(_height);
    float textX = hudX;
    float textY = hudY - bgHeight + 2.0f / _height;

    // FPS
    char fpsBuf[16];
    floatToString(stats.fps, fpsBuf, sizeof(fpsBuf));
    drawString("FPS: ", textX, textY, 1.0f, 1.0f, 1.0f, 0.2f);
    drawString(fpsBuf, textX + 40.0f / _width, textY, 1.0f, 1.0f, 1.0f, 0.2f);
    textY -= lineHeight;

    // Chunks
    char chunkBuf[16];
    intToString(static_cast<int>(stats.chunkCount), chunkBuf, sizeof(chunkBuf));
    drawString("Chunks: ", textX, textY, 1.0f, 0.2f, 1.0f, 0.8f);
    drawString(chunkBuf, textX + 72.0f / _width, textY, 1.0f, 0.2f, 1.0f, 0.8f);
    textY -= lineHeight;

    // Entities
    char entityBuf[16];
    intToString(static_cast<int>(stats.entityCount), entityBuf, sizeof(entityBuf));
    drawString("Entities: ", textX, textY, 1.0f, 0.8f, 0.4f, 1.0f);
    drawString(entityBuf, textX + 88.0f / _width, textY, 1.0f, 0.8f, 0.4f, 1.0f);
    textY -= lineHeight;

    // Frame time
    char ftBuf[16];
    floatToString(stats.frameTimeMs, ftBuf, sizeof(ftBuf));
    drawString("Frame: ", textX, textY, 1.0f, 1.0f, 0.8f, 0.4f);
    drawString(ftBuf, textX + 60.0f / _width, textY, 1.0f, 1.0f, 0.8f, 0.4f);
    (void)textY; // Suppress unused variable warning
}

#import "render/ui_overlay.hpp"

#include "common/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
UIOverlay::UIOverlay(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                     uint32_t height)
    : _device(device), _width(width), _height(height) {
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
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_pipelineState) {
        NSString* msg =
            [NSString stringWithFormat:@"Failed to create UI overlay pipeline state: %@",
                                       error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Projection matrix buffer (16 floats = 64 bytes) ----
    _projectionBuffer = [_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
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
    if (!encoder || _vertices.empty())
        return;

    const size_t byteCount = _vertices.size() * sizeof(UIVertex);
    id<MTLBuffer>& slot = _vertexRing[_frameIndex % RING_SLOTS];
    ++_frameIndex;

    if (!slot || slot.length < byteCount) {
        // Grow to the next power of two so steady-state frames never realloc
        size_t capacity = 16384;
        while (capacity < byteCount)
            capacity *= 2;
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
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:_vertices.size()];

    _vertices.clear();
}

// ---------------------------------------------------------------------------
// drawQuad()
// ---------------------------------------------------------------------------
void UIOverlay::drawQuad(float x, float y, float w, float h, float r, float g, float b, float a) {
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
    if (width == _width && height == _height)
        return;

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
        2.0f, 0.0f, 0.0f, 0.0f, 0.0f,  2.0f,  0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f,
    };

    std::memcpy((void*)_projectionBuffer.contents, proj, sizeof(proj));
}

// ============================================================================
// Bitmap font text
// ============================================================================

// ---- 8×8 bitmap font data ----
// One glyph per row: 8 bytes top-to-bottom, bits 5-1 = left-to-right pixels
// (a 5×7 face inside the 8×8 cell, one column of side bearing each side).
namespace {

struct Glyph {
    char c;
    uint8_t rows[8];
};

constexpr Glyph FONT[] = {
    {'0', {0x1C, 0x22, 0x26, 0x2A, 0x32, 0x22, 0x1C, 0x00}},
    {'1', {0x08, 0x18, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00}},
    {'2', {0x1C, 0x22, 0x02, 0x0C, 0x10, 0x20, 0x3E, 0x00}},
    {'3', {0x1C, 0x22, 0x02, 0x0C, 0x02, 0x22, 0x1C, 0x00}},
    {'4', {0x04, 0x0C, 0x14, 0x24, 0x3E, 0x04, 0x04, 0x00}},
    {'5', {0x3E, 0x20, 0x3C, 0x02, 0x02, 0x22, 0x1C, 0x00}},
    {'6', {0x1C, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x1C, 0x00}},
    {'7', {0x3E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x00}},
    {'8', {0x1C, 0x22, 0x22, 0x1C, 0x22, 0x22, 0x1C, 0x00}},
    {'9', {0x1C, 0x22, 0x22, 0x1E, 0x02, 0x02, 0x1C, 0x00}},

    {'A', {0x1C, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22, 0x00}},
    {'B', {0x3C, 0x22, 0x22, 0x3C, 0x22, 0x22, 0x3C, 0x00}},
    {'C', {0x1C, 0x22, 0x20, 0x20, 0x20, 0x22, 0x1C, 0x00}},
    {'D', {0x3C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x3C, 0x00}},
    {'E', {0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x3E, 0x00}},
    {'F', {0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x20, 0x00}},
    {'G', {0x1C, 0x22, 0x20, 0x2E, 0x22, 0x22, 0x1E, 0x00}},
    {'H', {0x22, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22, 0x00}},
    {'I', {0x1C, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00}},
    {'J', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x24, 0x18, 0x00}},
    {'K', {0x22, 0x24, 0x28, 0x30, 0x28, 0x24, 0x22, 0x00}},
    {'L', {0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x3E, 0x00}},
    {'M', {0x22, 0x36, 0x2A, 0x2A, 0x22, 0x22, 0x22, 0x00}},
    {'N', {0x22, 0x32, 0x2A, 0x26, 0x22, 0x22, 0x22, 0x00}},
    {'O', {0x1C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00}},
    {'P', {0x3C, 0x22, 0x22, 0x3C, 0x20, 0x20, 0x20, 0x00}},
    {'Q', {0x1C, 0x22, 0x22, 0x22, 0x2A, 0x24, 0x1A, 0x00}},
    {'R', {0x3C, 0x22, 0x22, 0x3C, 0x28, 0x24, 0x22, 0x00}},
    {'S', {0x1E, 0x20, 0x20, 0x1C, 0x02, 0x02, 0x3C, 0x00}},
    {'T', {0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00}},
    {'U', {0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00}},
    {'V', {0x22, 0x22, 0x22, 0x22, 0x22, 0x14, 0x08, 0x00}},
    {'W', {0x22, 0x22, 0x22, 0x2A, 0x2A, 0x2A, 0x14, 0x00}},
    {'X', {0x22, 0x22, 0x14, 0x08, 0x14, 0x22, 0x22, 0x00}},
    {'Y', {0x22, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x00}},
    {'Z', {0x3E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3E, 0x00}},

    {'a', {0x00, 0x00, 0x1C, 0x02, 0x1E, 0x22, 0x1E, 0x00}},
    {'b', {0x20, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x3C, 0x00}},
    {'c', {0x00, 0x00, 0x1E, 0x20, 0x20, 0x20, 0x1E, 0x00}},
    {'d', {0x02, 0x02, 0x1E, 0x22, 0x22, 0x22, 0x1E, 0x00}},
    {'e', {0x00, 0x00, 0x1C, 0x22, 0x3E, 0x20, 0x1E, 0x00}},
    {'f', {0x0C, 0x12, 0x10, 0x3C, 0x10, 0x10, 0x10, 0x00}},
    {'g', {0x00, 0x00, 0x1E, 0x22, 0x22, 0x1E, 0x02, 0x1C}},
    {'h', {0x20, 0x20, 0x3C, 0x22, 0x22, 0x22, 0x22, 0x00}},
    {'i', {0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x1C, 0x00}},
    {'j', {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x38, 0x00}},
    {'k', {0x20, 0x20, 0x22, 0x24, 0x38, 0x24, 0x22, 0x00}},
    {'l', {0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00}},
    {'m', {0x00, 0x00, 0x34, 0x2A, 0x2A, 0x2A, 0x2A, 0x00}},
    {'n', {0x00, 0x00, 0x3C, 0x22, 0x22, 0x22, 0x22, 0x00}},
    {'o', {0x00, 0x00, 0x1C, 0x22, 0x22, 0x22, 0x1C, 0x00}},
    {'p', {0x00, 0x00, 0x3C, 0x22, 0x22, 0x3C, 0x20, 0x20}},
    {'q', {0x00, 0x00, 0x1E, 0x22, 0x22, 0x1E, 0x02, 0x02}},
    {'r', {0x00, 0x00, 0x2E, 0x30, 0x20, 0x20, 0x20, 0x00}},
    {'s', {0x00, 0x00, 0x1E, 0x20, 0x1C, 0x02, 0x3C, 0x00}},
    {'t', {0x10, 0x10, 0x3C, 0x10, 0x10, 0x12, 0x0C, 0x00}},
    {'u', {0x00, 0x00, 0x22, 0x22, 0x22, 0x22, 0x1E, 0x00}},
    {'v', {0x00, 0x00, 0x22, 0x22, 0x22, 0x14, 0x08, 0x00}},
    {'w', {0x00, 0x00, 0x2A, 0x2A, 0x2A, 0x2A, 0x14, 0x00}},
    {'x', {0x00, 0x00, 0x22, 0x14, 0x08, 0x14, 0x22, 0x00}},
    {'y', {0x00, 0x00, 0x22, 0x22, 0x22, 0x1E, 0x02, 0x1C}},
    {'z', {0x00, 0x00, 0x3E, 0x04, 0x08, 0x10, 0x3E, 0x00}},

    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00}},
    {':', {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00}},
    {'/', {0x02, 0x02, 0x04, 0x08, 0x10, 0x20, 0x20, 0x00}},
    {'-', {0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00}},
    {'+', {0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00, 0x00}},
    {',', {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x08, 0x10}},
    {'!', {0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x08, 0x00}},
    {'?', {0x1C, 0x22, 0x02, 0x0C, 0x08, 0x00, 0x08, 0x00}},
    {'(', {0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04, 0x00}},
    {')', {0x10, 0x08, 0x04, 0x04, 0x04, 0x08, 0x10, 0x00}},
    {'%', {0x32, 0x34, 0x08, 0x08, 0x16, 0x26, 0x00, 0x00}},
    {'_', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00}},
};

} // namespace

std::array<uint8_t, 8> UIOverlay::getCharBitmap(char c) {
    for (const Glyph& glyph : FONT) {
        if (glyph.c == c) {
            std::array<uint8_t, 8> bitmap;
            for (int i = 0; i < 8; ++i)
                bitmap[static_cast<size_t>(i)] = glyph.rows[i];
            return bitmap;
        }
    }
    return {};
}

void UIOverlay::drawChar(char c, float x, float y, float scale, float r, float g, float b) {
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

float UIOverlay::drawString(const char* str, float x, float y, float scale, float r, float g,
                            float b) {
    if (!str)
        return 0.f;

    float cursorX = x;

    while (*str) {
        drawChar(*str, cursorX, y, scale, r, g, b);
        cursorX += (FONT_WIDTH + 1) * scale / static_cast<float>(_width);
        ++str;
    }

    return cursorX - x;
}

float UIOverlay::measureString(const char* str, float scale) const {
    if (!str)
        return 0.f;
    const float advance = (FONT_WIDTH + 1) * scale / static_cast<float>(_width);
    return advance * static_cast<float>(std::strlen(str));
}

void UIOverlay::intToString(int value, char* buf, size_t bufSize) {
    if (bufSize < 2)
        return;
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
        if (value < 0)
            tmp[len++] = '-';
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
    if (bufSize == 0)
        return;
    std::snprintf(buf, bufSize, "%.1f", static_cast<double>(value));
}

void UIOverlay::drawPerformanceHUD(const PerformanceStats& stats) {
    constexpr const char* biomeNames[] = {
        "Deep Ocean",
        "Ocean",
        "Plains",
        "Forest",
        "Taiga",
        "Desert",
        "Hills",
        "Swamp",
        "Mushroom",
        "Ice Spikes",
        "Beach",
        "River",
        "Birch Forest",
        "Flower Field",
        "Savanna",
        "Tropical Rainforest",
        "Temperate Rainforest",
        "Shrubland",
        "Steppe",
        "Cold Desert",
        "Badlands",
        "Tundra",
        "Alpine",
        "Mangrove",
        "Frozen Ocean",
        "Volcanic Barren",
        "Glacier",
        "Montane Grassland",
        "Flooded Grassland",
        "Mediterranean Woodland",
        "Temperate Conifer Forest",
        "Tropical Conifer Forest",
        "Tropical Dry Forest",
    };
    constexpr const char* boundaryNames[] = {"None", "Convergent", "Divergent", "Transform"};
    static_assert(static_cast<size_t>(Biome::COUNT) == std::size(biomeNames));
    static_assert(std::size(boundaryNames) == 4);
    const float hudX = 8.0f / static_cast<float>(_width);
    const float hudY = 1.0f - 8.0f / static_cast<float>(_height);
    const float bgWidth = 440.0f / static_cast<float>(_width);
    const float bgHeight = 220.0f / static_cast<float>(_height);
    const float lineHeight = (FONT_HEIGHT + 2) / static_cast<float>(_height);
    float textY = hudY - lineHeight;
    drawQuad(hudX - 4.0f / _width, hudY - bgHeight, bgWidth, bgHeight, 0.0f, 0.0f, 0.0f, 0.72f);

    auto integerText = [&](int64_t value, char* buffer, size_t size) {
        std::snprintf(buffer, size, "%lld", static_cast<long long>(value));
        return buffer;
    };
    auto unsignedText = [&](uint64_t value, char* buffer, size_t size) {
        std::snprintf(buffer, size, "%llu", static_cast<unsigned long long>(value));
        return buffer;
    };
    char a[24], b[24], c[24];
    auto nextLine = [&] { textY -= lineHeight; };

    drawString("FPS", hudX, textY, 1.0f, 1.0f, 1.0f, 0.3f);
    floatToString(stats.fps, a, sizeof(a));
    drawString(a, hudX + 40.0f / _width, textY, 1.0f, 1.0f, 1.0f, 0.3f);
    drawString("Frame", hudX + 120.0f / _width, textY, 1.0f, 1.0f, 0.8f, 0.4f);
    floatToString(stats.frameTimeMs, b, sizeof(b));
    drawString(b, hudX + 176.0f / _width, textY, 1.0f, 1.0f, 0.8f, 0.4f);
    drawString("GPU", hudX + 248.0f / _width, textY, 1.0f, 1.0f, 0.6f, 0.6f);
    floatToString(stats.gpuFrameMs, c, sizeof(c));
    drawString(c, hudX + 288.0f / _width, textY, 1.0f, 1.0f, 0.6f, 0.6f);
    nextLine();

    drawString("Cubes loaded/meshed", hudX, textY, 1.0f, 0.3f, 1.0f, 0.8f);
    drawString(integerText(static_cast<int>(stats.chunkCount), a, sizeof(a)),
               hudX + 184.0f / _width, textY, 1.0f, 0.3f, 1.0f, 0.8f);
    drawString("/", hudX + 240.0f / _width, textY, 1.0f, 0.3f, 1.0f, 0.8f);
    drawString(integerText(static_cast<int>(stats.meshedCubeCount), b, sizeof(b)),
               hudX + 252.0f / _width, textY, 1.0f, 0.3f, 1.0f, 0.8f);
    drawString("Entities", hudX + 304.0f / _width, textY, 1.0f, 1.0f, 0.7f, 0.4f);
    drawString(integerText(stats.entityCount, c, sizeof(c)), hudX + 384.0f / _width, textY, 1.0f,
               1.0f, 0.7f, 0.4f);
    nextLine();

    drawString("Cube XYZ", hudX, textY, 1.0f, 0.7f, 0.9f, 1.0f);
    drawString(integerText(stats.cubeX, a, sizeof(a)), hudX + 80.0f / _width, textY, 1.0f, 0.7f,
               0.9f, 1.0f);
    drawString(integerText(stats.cubeY, b, sizeof(b)), hudX + 152.0f / _width, textY, 1.0f, 0.7f,
               0.9f, 1.0f);
    drawString(integerText(stats.cubeZ, c, sizeof(c)), hudX + 224.0f / _width, textY, 1.0f, 0.7f,
               0.9f, 1.0f);
    nextLine();

    drawString("Gen pending/ms", hudX, textY, 1.0f, 0.5f, 0.9f, 1.0f);
    drawString(integerText(static_cast<int>(stats.pendingChunks), a, sizeof(a)),
               hudX + 128.0f / _width, textY, 1.0f, 0.5f, 0.9f, 1.0f);
    floatToString(stats.genMsAvg, b, sizeof(b));
    drawString(b, hudX + 192.0f / _width, textY, 1.0f, 0.5f, 0.9f, 1.0f);
    drawString("Mesh", hudX + 264.0f / _width, textY, 1.0f, 0.4f, 0.9f, 0.6f);
    floatToString(stats.meshMsAvg, c, sizeof(c));
    drawString(c, hudX + 312.0f / _width, textY, 1.0f, 0.4f, 0.9f, 0.6f);
    drawString(integerText(stats.meshBuildsFrame, a, sizeof(a)), hudX + 368.0f / _width, textY,
               1.0f, 0.4f, 0.9f, 0.6f);
    nextLine();

    drawString("Plate", hudX, textY, 1.0f, 0.9f, 0.6f, 0.3f);
    drawString(unsignedText(stats.plateId, a, sizeof(a)), hudX + 48.0f / _width, textY, 1.0f, 0.9f,
               0.6f, 0.3f);
    const size_t boundaryIndex =
        std::min(static_cast<size_t>(stats.boundary), std::size(boundaryNames) - 1);
    drawString(boundaryNames[boundaryIndex], hudX + 160.0f / _width, textY, 1.0f, 0.9f, 0.6f, 0.3f);
    nextLine();

    drawString("Temp C / rain mm", hudX, textY, 1.0f, 0.4f, 0.8f, 1.0f);
    floatToString(stats.temperatureC, a, sizeof(a));
    floatToString(stats.precipitationMm, b, sizeof(b));
    drawString(a, hudX + 152.0f / _width, textY, 1.0f, 0.4f, 0.8f, 1.0f);
    drawString(b, hudX + 224.0f / _width, textY, 1.0f, 0.4f, 0.8f, 1.0f);
    nextLine();

    drawString("Biome", hudX, textY, 1.0f, 0.5f, 1.0f, 0.5f);
    const size_t primaryBiome =
        std::min(static_cast<size_t>(stats.primaryBiome), std::size(biomeNames) - 1);
    const size_t secondaryBiome =
        std::min(static_cast<size_t>(stats.secondaryBiome), std::size(biomeNames) - 1);
    drawString(biomeNames[primaryBiome], hudX + 56.0f / _width, textY, 1.0f, 0.5f, 1.0f, 0.5f);
    nextLine();
    drawString("Blend", hudX, textY, 1.0f, 0.5f, 1.0f, 0.5f);
    drawString(biomeNames[secondaryBiome], hudX + 56.0f / _width, textY, 1.0f, 0.5f, 1.0f, 0.5f);
    floatToString(stats.biomeTransition, a, sizeof(a));
    drawString(a, hudX + 280.0f / _width, textY, 1.0f, 0.5f, 1.0f, 0.5f);
    nextLine();

    drawString("River order", hudX, textY, 1.0f, 0.3f, 0.8f, 1.0f);
    drawString(integerText(stats.riverOrder, a, sizeof(a)), hudX + 112.0f / _width, textY, 1.0f,
               0.3f, 0.8f, 1.0f);
    drawString("Cache", hudX + 176.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    drawString(integerText(static_cast<int>(stats.macroCacheEntries), b, sizeof(b)),
               hudX + 232.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    floatToString(stats.macroCacheMB, b, sizeof(b));
    drawString(b, hudX + 272.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    drawString("Fluid", hudX + 320.0f / _width, textY, 1.0f, 0.2f, 0.7f, 1.0f);
    drawString(integerText(static_cast<int>(stats.pendingFluids), c, sizeof(c)),
               hudX + 368.0f / _width, textY, 1.0f, 0.2f, 0.7f, 1.0f);
    nextLine();

    drawString("Fluid drops U/F", hudX, textY, 1.0f, 0.4f, 0.7f, 1.0f);
    drawString(unsignedText(stats.droppedFluidUpdates, a, sizeof(a)), hudX + 144.0f / _width, textY,
               1.0f, 0.4f, 0.7f, 1.0f);
    drawString("/", hudX + 240.0f / _width, textY, 1.0f, 0.4f, 0.7f, 1.0f);
    drawString(unsignedText(stats.droppedFluidFrontiers, b, sizeof(b)), hudX + 256.0f / _width,
               textY, 1.0f, 0.4f, 0.7f, 1.0f);
    nextLine();

    drawString("Exact mesh MB", hudX, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    floatToString(stats.megaUsedMB, a, sizeof(a));
    floatToString(stats.megaCapMB, b, sizeof(b));
    drawString(a, hudX + 120.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    drawString("/", hudX + 176.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    drawString(b, hudX + 192.0f / _width, textY, 1.0f, 0.9f, 0.9f, 0.5f);
    nextLine();

    drawString("Exact ready/req", hudX, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    drawString(integerText(stats.exactSurfaceReady, a, sizeof(a)), hudX + 120.0f / _width, textY,
               1.0f, 0.8f, 0.9f, 1.0f);
    drawString("/", hudX + 168.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    drawString(integerText(stats.exactSurfaceRequired, b, sizeof(b)), hudX + 184.0f / _width, textY,
               1.0f, 0.8f, 0.9f, 1.0f);
    drawString("Unresolved", hudX + 240.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    drawString(integerText(stats.exactSurfaceUnresolvedColumns, c, sizeof(c)),
               hudX + 328.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    nextLine();

    drawString("Handoff/frontier", hudX, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    floatToString(stats.exactSurfaceHandoffBlocks, a, sizeof(a));
    floatToString(stats.farCoverageFrontierBlocks, b, sizeof(b));
    drawString(a, hudX + 136.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    drawString("/", hudX + 200.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    drawString(b, hudX + 216.0f / _width, textY, 1.0f, 0.8f, 0.9f, 1.0f);
    nextLine();

    drawString("Far W/R/D", hudX, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farWantedTiles, a, sizeof(a)), hudX + 88.0f / _width, textY, 1.0f,
               0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 128.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farResidentTiles, b, sizeof(b)), hudX + 144.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 184.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farDrawnTiles, c, sizeof(c)), hudX + 200.0f / _width, textY, 1.0f,
               0.7f, 0.8f, 1.0f);
    drawString("F/O", hudX + 248.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farFrustumCulledTiles, a, sizeof(a)), hudX + 288.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 336.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farOcclusionCulledTiles, b, sizeof(b)), hudX + 352.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    nextLine();

    drawString("Base W/R/D/M", hudX, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farBaseWantedTiles, a, sizeof(a)), hudX + 112.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 152.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farBaseResidentTiles, b, sizeof(b)), hudX + 168.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 208.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farBaseDrawnTiles, c, sizeof(c)), hudX + 224.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 264.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farBaseMissingTiles, a, sizeof(a)), hudX + 280.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("Cached", hudX + 328.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farCachedBaseTiles, b, sizeof(b)), hudX + 384.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    nextLine();

    drawString("Refine W/R/D", hudX, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farRefinementWantedTiles, a, sizeof(a)), hudX + 112.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 152.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farRefinementResidentTiles, b, sizeof(b)), hudX + 168.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 208.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farRefinementDrawnTiles, c, sizeof(c)), hudX + 224.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("Q B/R", hudX + 272.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farQueuedBaseTiles, a, sizeof(a)), hudX + 328.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 368.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farQueuedRefinementTiles, b, sizeof(b)), hudX + 384.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    nextLine();

    drawString("Workers B/R/U/T", hudX, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farActiveBaseWorkers, a, sizeof(a)), hudX + 128.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 160.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farReservedBaseWorkers, b, sizeof(b)), hudX + 176.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 208.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farActiveUrgentRefinements, c, sizeof(c)), hudX + 224.0f / _width,
               textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("/", hudX + 256.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farWorkerBudget, a, sizeof(a)), hudX + 272.0f / _width, textY,
               1.0f, 0.7f, 0.8f, 1.0f);
    nextLine();

    drawString("Far pending", hudX, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString(integerText(stats.farPendingTiles, a, sizeof(a)), hudX + 96.0f / _width, textY, 1.0f,
               0.7f, 0.8f, 1.0f);
    drawString("Cache MB", hudX + 144.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    floatToString(stats.farCacheMB, b, sizeof(b));
    drawString(b, hudX + 216.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    drawString("Arena MB", hudX + 280.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
    floatToString(stats.farMeshMB, c, sizeof(c));
    drawString(c, hudX + 360.0f / _width, textY, 1.0f, 0.7f, 0.8f, 1.0f);
}

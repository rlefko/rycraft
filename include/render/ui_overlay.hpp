#pragma once

#import <Metal/Metal.h>

#include <cstdint>
#include <array>

// ---------------------------------------------------------------------------
// UIOverlay — Screen-space orthographic render pass for HUD elements.
//
// Renders solid-color quads in normalized screen coordinates (0..1 range).
// Uses a separate render pass after the main 3D pass completes.
//
// Responsibilities:
//   • Pre-allocate fullscreen quad geometry
//   • Provide drawQuad() for screen-space colored rectangles
//   • Manage its own pipeline state (orthographic projection)
//   • Render performance HUD (FPS, chunk count, entity count, frame time)
// ---------------------------------------------------------------------------

// Performance HUD state (Phase 8)
struct PerformanceStats {
    float fps = 0.f;           // Rolling average FPS (60 frames)
    uint32_t chunkCount = 0;   // Loaded chunks
    uint32_t entityCount = 0;  // Active entities
    float frameTimeMs = 0.f;   // Frame time in milliseconds
};

class UIOverlay {
public:
    UIOverlay(id<MTLDevice> device,
              id<MTLLibrary> shaderLibrary,
              uint32_t width,
              uint32_t height);

    // Draw a screen-space quad with a solid color.
    // x, y: bottom-left corner in normalized [0, 1] coordinates
    // w, h: width and height in normalized [0, 1] coordinates
    // r, g, b, a: color components in [0, 1] range
    void drawQuad(id<MTLRenderCommandEncoder> encoder,
                  float x, float y,
                  float w, float h,
                  float r, float g, float b, float a);

    // Draw performance HUD (Phase 8)
    void drawPerformanceHUD(id<MTLRenderCommandEncoder> encoder,
                            const PerformanceStats& stats);

    // Regenerate quad buffers for new viewport size.
    void resize(uint32_t width, uint32_t height);

private:
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;

    // Pre-allocated quad vertices: 4 vertices, 6 indices.
    // Each vertex is 8 floats: [x, y, r, g, b, a, _pad0, _pad1]
    id<MTLBuffer> _quadIndexBuffer;

    // Dynamic per-draw color/position buffer (4 vertices × 32 bytes).
    id<MTLBuffer> _dynamicBuffer;

    // Projection matrix constant buffer.
    id<MTLBuffer> _projectionBuffer;

    uint32_t _width;
    uint32_t _height;

    // Build quad vertex data for a screen-space rectangle.
    void buildQuadVertices(float x, float y, float w, float h,
                           float r, float g, float b, float a);

    // Build orthographic projection for normalized screen coords.
    void buildProjectionMatrix();

    // ---- Performance HUD helpers (Phase 8) ----

    // 8×8 bitmap font atlas data (procedural characters)
    static const uint8_t FONT_WIDTH = 8;
    static const uint8_t FONT_HEIGHT = 8;

    // Get bitmap data for a character (returns 8 bytes, one per row)
    static std::array<uint8_t, 8> getCharBitmap(char c);

    // Draw a single character at screen position
    void drawChar(id<MTLRenderCommandEncoder> encoder,
                  char c, float x, float y,
                  float r, float g, float b);

    // Draw a string at screen position (returns width in pixels)
    float drawString(id<MTLRenderCommandEncoder> encoder,
                     const char* str, float x, float y,
                     float r, float g, float b);

    // Convert number to string buffer
    static void intToString(int value, char* buf, size_t bufSize);
    static void floatToString(float value, char* buf, size_t bufSize);
};

#pragma once

#import <Metal/Metal.h>

#include "render/ui_menu.hpp"

#include <array>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// UIOverlay — Screen-space orthographic renderer for HUD and menu elements.
//
// Quads queue up between beginFrame() and flush(); flush() uploads the whole
// batch into one ring-buffer slot and issues a single draw call. Batching is
// required for correctness, not just speed: Metal reads vertex buffers at
// command-buffer execution time, so a single reused buffer rewritten per
// draw call renders every quad with the LAST quad's vertices.
//
// Coordinates are normalized [0, 1] with a bottom-left origin.
// ---------------------------------------------------------------------------

class UIOverlay {
public:
    UIOverlay(id<MTLDevice> device,
              id<MTLLibrary> shaderLibrary,
              uint32_t width,
              uint32_t height);

    // Start a new frame's batch (drops any unflushed quads).
    void beginFrame();

    // Upload the queued quads and draw them with a single call.
    void flush(id<MTLRenderCommandEncoder> encoder);

    // Queue a screen-space quad with a solid color.
    // x, y: bottom-left corner in normalized [0, 1] coordinates
    // w, h: width and height in normalized [0, 1] coordinates
    void drawQuad(float x, float y,
                  float w, float h,
                  float r, float g, float b, float a);

    // Queue a single 8×8 bitmap-font character. `scale` multiplies the glyph
    // pixel size (1.0 = 8px tall on screen).
    void drawChar(char c, float x, float y, float scale,
                  float r, float g, float b);

    // Queue a string; returns its normalized advance width.
    float drawString(const char* str, float x, float y, float scale,
                     float r, float g, float b);

    // Normalized width a string will occupy at the given scale.
    float measureString(const char* str, float scale) const;

    // Queue the performance HUD (top-left corner).
    void drawPerformanceHUD(const PerformanceStats& stats);

    // Update the projection for a new viewport size.
    void resize(uint32_t width, uint32_t height);

    // Number-to-text helpers for HUD labels.
    static void intToString(int value, char* buf, size_t bufSize);
    static void floatToString(float value, char* buf, size_t bufSize);

    static constexpr int FONT_WIDTH = 8;
    static constexpr int FONT_HEIGHT = 8;

    // Bitmap for one 8×8 glyph (a zero row for unknown characters).
    // Public so tests can verify the font covers every menu string.
    static std::array<uint8_t, 8> getCharBitmap(char c);

private:
    // One batched vertex: position (normalized) + color. Matches the vertex
    // descriptor in ui_overlay.mm (float2 @0, float4 @8, stride 24).
    struct UIVertex {
        float px, py;
        float cr, cg, cb, ca;
    };
    static_assert(sizeof(UIVertex) == 24);

    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;

    // CPU-side batch for the current frame.
    std::vector<UIVertex> _vertices;

    // Triple-buffered vertex upload ring so the CPU never rewrites a buffer
    // the GPU is still reading.
    static constexpr int RING_SLOTS = 3;
    id<MTLBuffer> _vertexRing[RING_SLOTS];
    uint64_t _frameIndex = 0;

    // Projection matrix constant buffer.
    id<MTLBuffer> _projectionBuffer;

    uint32_t _width;
    uint32_t _height;

    // Build orthographic projection for normalized screen coords.
    void buildProjectionMatrix();
};

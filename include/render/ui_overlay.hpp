#pragma once

#import <Metal/Metal.h>
#include <cstdint>

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
// ---------------------------------------------------------------------------
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
};

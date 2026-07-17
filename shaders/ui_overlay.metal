#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// UI Overlay — Screen-space orthographic shaders for HUD rendering.
//
// Vertex input (per-vertex buffer):
//   attribute(0) float2 position  — screen-space coords [0, 1]
//   attribute(1) float4 color     — RGBA color
//
// Buffer(1): orthographic projection matrix (float4x4)
// ---------------------------------------------------------------------------

struct UIVertexInput {
    float2 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct UIVertexOutput {
    float4 clipPosition [[position]];
    float4 vColor;
};

// ---------------------------------------------------------------------------
// Vertex shader — apply orthographic projection to screen-space coords
// ---------------------------------------------------------------------------
vertex UIVertexOutput uiVertexMain(
    UIVertexInput in [[stage_in]],
    constant float4x4 &projection [[buffer(1)]]
) {
    UIVertexOutput out;

    // Apply orthographic projection: maps [0,1] screen coords to NDC [-1,1]
    float4 screenPos = float4(in.position, 0.0, 1.0);
    out.clipPosition = projection * screenPos;

    // Pass color through
    out.vColor = in.color;

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — output solid color with alpha blending
// ---------------------------------------------------------------------------
fragment float4 uiFragmentMain(
    UIVertexOutput in [[stage_in]]
) {
    return in.vColor;
}

// ---------------------------------------------------------------------------
// Icon pipeline — textured quads sampling the block/item texture array.
// Runs between the two solid-color phases of the overlay batch.
// ---------------------------------------------------------------------------

struct UIIconVertexInput {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 tint [[attribute(2)]];
    uint layer [[attribute(3)]];
};

struct UIIconVertexOutput {
    float4 clipPosition [[position]];
    float2 vUv;
    float4 vTint;
    uint vLayer [[flat]];
};

vertex UIIconVertexOutput uiIconVertexMain(
    UIIconVertexInput in [[stage_in]],
    constant float4x4 &projection [[buffer(1)]]
) {
    UIIconVertexOutput out;
    out.clipPosition = projection * float4(in.position, 0.0, 1.0);
    out.vUv = in.uv;
    out.vTint = in.tint;
    out.vLayer = in.layer;
    return out;
}

fragment float4 uiIconFragmentMain(
    UIIconVertexOutput in [[stage_in]],
    texture2d_array<float> icons [[texture(0)]],
    sampler iconSampler [[sampler(0)]]
) {
    float4 sampled = icons.sample(iconSampler, in.vUv, in.vLayer);
    return float4(sampled.rgb * in.vTint.rgb, sampled.a * in.vTint.a);
}

#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Sky Shader — Full-screen gradient for day/night sky rendering
//
// Vertex shader outputs a fullscreen quad in clip space.
// Fragment shader interpolates between zenith and horizon colors
// based on the vertical screen position.
//
// Uniforms (buffer 1):
//   float3 zenithColor    — sky color at top of screen
//   float3 horizonColor   — sky color at horizon line
//   float3 sunDirection   — normalized direction to sun (for sun disc)
//   float3 sunColor       — color of the sun
//   float sunIntensity    — brightness multiplier (0 at night, 1 at noon)
// ---------------------------------------------------------------------------

struct SkyVertexOutput {
    float4 clipPosition [[position]];
    float2 vScreenUV;
};

// ---------------------------------------------------------------------------
// Vertex shader — fullscreen quad
// ---------------------------------------------------------------------------
vertex SkyVertexOutput skyVertexMain(
    uint vertexID [[vertex_id]]
) {
    // Two triangles forming a fullscreen quad
    // Using clip-space directly: [-1,-1] to [1,1]

    const float2 positions[6] = {
        float2(-1.0f, -1.0f),   // bottom-left
        float2( 1.0f, -1.0f),   // bottom-right
        float2( 1.0f,  1.0f),   // top-right
        float2(-1.0f, -1.0f),   // bottom-left
        float2( 1.0f,  1.0f),   // top-right
        float2(-1.0f,  1.0f)    // top-left
    };

    SkyVertexOutput out;
    float2 pos = positions[vertexID];
    out.clipPosition = float4(pos, 0.0, 1.0);

    // Screen UV: bottom-left is (0,0), top-right is (1,1)
    out.vScreenUV = pos * 0.5f + 0.5f;

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — vertical gradient with optional sun disc
// ---------------------------------------------------------------------------
fragment float4 skyFragmentMain(
    SkyVertexOutput in [[stage_in]],
    constant SkyUniforms &uniforms [[buffer(1)]]
) {
    // Interpolate between horizon (bottom, v=0) and zenith (top, v=1)
    float t = in.vScreenUV.y;
    float3 skyColor = mix(uniforms.horizonColor, uniforms.zenithColor, t);

    // Sun disc — project sun direction onto screen
    float3 sunScreen = uniforms.sunDirection;
    float sunHeight = sunScreen.y; // Higher = more visible

    // Only draw sun when it's above horizon
    if (sunHeight > 0.0f && uniforms.sunIntensity > 0.01f) {
        // Map sun position to screen UV
        // Sun moves across the sky: x position based on horizontal direction
        float sunUVX = sunScreen.x * 0.3f + 0.5f;
        float sunUVY = sunHeight * 0.5f + 0.1f;

        float2 sunUV = float2(sunUVX, sunUVY);
        float dist = distance(in.vScreenUV, sunUV);

        // Sun disc with soft edge
        float discRadius = 0.03f;
        float sunMask = 1.0f - smoothstep(discRadius * 0.8f, discRadius, dist);
        sunMask *= uniforms.sunIntensity;

        skyColor = mix(skyColor, uniforms.sunColor, sunMask * 0.8f);
    }

    return float4(skyColor, 1.0);
}

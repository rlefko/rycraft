#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Particle Shader, Billboard rain and snow particles
//
// Vertex shader: transforms particle world position to clip space, passes
// particle type to fragment shader.
//
// Fragment shader: renders rain as thin vertical lines or snow as soft
// circles with alpha falloff. Uses point_coord for per-pixel position
// within the point sprite.
//
// Uniforms (buffer 1): ParticleUniforms (camera transforms, extinction, physical scale)
// Vertices (buffer 0): GPUParticle array (position, velocity, lifetime, type)
// ---------------------------------------------------------------------------

struct ParticleVertexOutput {
    float4 clipPosition [[position]];
    float pointSize [[point_size]];
    float particleType;
    float viewDistanceMeters;
    float atmosphericExtinction;
};

// ---------------------------------------------------------------------------
// Vertex shader, transform particle to clip space
// ---------------------------------------------------------------------------
vertex ParticleVertexOutput particleVertexMain(device const GPUParticle* particles [[buffer(0)]],
                                               constant ParticleUniforms& uniforms [[buffer(1)]],
                                               uint vertexID [[vertex_id]]) {
    device const GPUParticle& particle = particles[vertexID];

    // Transform particle world position to clip space
    float4 worldPos = float4(particle.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.projectionMatrix * viewPos;

    ParticleVertexOutput out;
    out.clipPosition = clipPos;
    out.particleType = particle.type;
    const float viewDistanceBlocks = length(viewPos.xyz);
    out.viewDistanceMeters =
        particleOpticalDistanceMeters(viewDistanceBlocks, uniforms.metersPerBlock);
    out.atmosphericExtinction = uniforms.atmosphericExtinction;

    // Billboard sizing remains in block space. Physical scale changes only
    // the Beer-Lambert path used by the fragment stage.
    out.pointSize = particleBillboardPointSize(viewDistanceBlocks);

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader, rain (thin line) or snow (soft circle)
// ---------------------------------------------------------------------------
fragment float4 particleFragmentMain(ParticleVertexOutput in [[stage_in]],
                                     float2 pointCoord [[point_coord]]) {
    // pointCoord: [0,1] × [0,1] within the point sprite
    // Center of point: (0.5, 0.5)
    float2 center = float2(0.5);
    float2 delta = pointCoord - center;

    // Rain: type == 0.0
    // Thin vertical line: narrow in X, full height
    if (in.particleType < 0.5) {
        // Horizontal falloff: very narrow line (~4px wide in point space)
        float lineAlpha = 1.0 - smoothstep(0.0, 0.15, abs(delta.x));

        // Vertical falloff: slight fade at top and bottom
        float vertAlpha = 1.0 - smoothstep(0.35, 0.5, abs(delta.y));

        // Rain color: blue-gray
        float3 rainColor = float3(0.55, 0.60, 0.75);
        float alpha = lineAlpha * vertAlpha * 0.6;

        const float transmittance =
            beerLambertTransmittance(in.atmosphericExtinction, in.viewDistanceMeters);
        // Straight-alpha blending applies alpha to RGB once. Attenuate alpha
        // only so atmospheric transmittance is not squared by the blend unit.
        return float4(rainColor, alpha * transmittance);
    }

    // Snow: type == 1.0
    // Soft circle with radial alpha falloff
    float dist = length(delta);
    float snowAlpha = 1.0 - smoothstep(0.2, 0.5, dist);

    // Snow color: white with slight blue tint
    float3 snowColor = float3(0.95, 0.95, 0.98);
    float alpha = snowAlpha * 0.8;

    const float transmittance =
        beerLambertTransmittance(in.atmosphericExtinction, in.viewDistanceMeters);
    return float4(snowColor, alpha * transmittance);
}

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Uniforms — bound as a constant buffer via [[buffer(N)]]
// ---------------------------------------------------------------------------
struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3 sunDirection;   // Normalized direction to the sun
    float3 sunColor;
    float3 ambientColor;
    float _padding;
    // Fog (Phase 8)
    float3 fogColor;
    float fogDensity;
    float3 cameraPosition;
    float _padding2;
};

// ---------------------------------------------------------------------------
// Vertex input — bound through the vertex descriptor (not argument buffer)
//
// Attribute layout matches include/render/vertex.hpp:
//   attribute(0)  uint     normalIdx         offset 0   4 bytes  (UInt)
//   attribute(1)  float3   px, py, pz        offset 4   6 bytes  (Half3)
//   attribute(2)  float2   u, v              offset 10  4 bytes  (Half2)
//   stride = 16 bytes
// ---------------------------------------------------------------------------
struct VertexInput {
    uint normalIdx [[attribute(0)]];
    float3 position [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

// ---------------------------------------------------------------------------
// Vertex → Fragment inter-stage
// ---------------------------------------------------------------------------
struct VertexOutput {
    float4 clipPosition [[position]];
    float3 vNormal;
    float2 vUV;
    float vLight;
    float3 vWorldPosition; // World-space position for fog calculation
};

// ---------------------------------------------------------------------------
// Face normal lookup — indices match FaceNormal enum (0=+X … 5=-Y)
// ---------------------------------------------------------------------------
float3 getFaceNormal(uint index) {
    switch (index) {
        case 0: return float3( 1.0,  0.0,  0.0); // +X
        case 1: return float3(-1.0,  0.0,  0.0); // -X
        case 2: return float3( 0.0,  0.0,  1.0); // +Z
        case 3: return float3( 0.0,  0.0, -1.0); // -Z
        case 4: return float3( 0.0,  1.0,  0.0); // +Y
        case 5: return float3( 0.0, -1.0,  0.0); // -Y
        default: return float3(0.0, 1.0, 0.0);    // Fallback: +Y
    }
}

// ---------------------------------------------------------------------------
// Vertex shader — passes world position for fog calculation
// ---------------------------------------------------------------------------
vertex VertexOutput vertexMain(
    VertexInput in [[stage_in]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    VertexOutput out;

    // Transform position through MVP
    float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;

    // Pass through UV
    out.vUV = in.uv;

    // Look up face normal and transform to world space.
    // Multiply by float4(normal, 0.0) to apply only the rotation/scale
    // portion of the model matrix (no translation).
    float3 normal = getFaceNormal(in.normalIdx);
    out.vNormal = normalize(
        (uniforms.modelMatrix * float4(normal, 0.0)).xyz
    );

    // Per-vertex diffuse lighting
    float ndotl = dot(out.vNormal, uniforms.sunDirection);
    out.vLight = max(ndotl, 0.0);

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — with distance fog (Phase 8)
// ---------------------------------------------------------------------------
fragment float4 fragmentMain(
    VertexOutput in [[stage_in]],
    texture2d<float> atlas [[texture(0)]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    // Nearest-neighbor sampling for crisp voxel textures
    constexpr sampler atlasSampler(mag_filter::nearest,
                                    min_filter::nearest);

    float4 texColor = atlas.sample(atlasSampler, in.vUV);

    // Combine directional sun light with ambient
    float3 litColor = texColor.rgb * (uniforms.sunColor * in.vLight + uniforms.ambientColor);

    // ---- Distance fog (Phase 8) ----
    // Use world position passed from vertex shader
    float distanceToFrag = distance(in.vWorldPosition, uniforms.cameraPosition);

    // Exponential fog: fogFactor = 1.0 - exp(-density * distance)
    float fogFactor = 1.0f - exp(-uniforms.fogDensity * distanceToFrag);
    fogFactor = clamp(fogFactor, 0.0f, 1.0f);

    // Blend between fog color and lit color
    float3 finalColor = mix(litColor, uniforms.fogColor, fogFactor);

    return float4(finalColor, 1.0);
}

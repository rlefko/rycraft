#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Vertex input — bound through the vertex descriptor (not argument buffer)
//
// Attribute layout matches include/render/vertex.hpp:
//   attribute(0)  uint     faceAttr          offset 0   4 bytes  (UInt)
//                          face normal in bits 0-2, texture layer in bits 3+
//   attribute(1)  float3   px, py, pz        offset 4   6 bytes  (Half3)
//                          CHUNK-LOCAL position; ChunkOrigin restores world
//   attribute(2)  float2   u, v              offset 10  4 bytes  (Half2)
//                          spans the quad extent in blocks (repeat-sampled)
//   stride = 16 bytes
// ---------------------------------------------------------------------------
struct VertexInput {
    uint faceAttr [[attribute(0)]];
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
    uint vTextureLayer [[flat]];
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
    constant Uniforms &uniforms [[buffer(1)]],
    constant ChunkOrigin &chunkOrigin [[buffer(2)]]
) {
    VertexOutput out;

    // Restore world space from the chunk-local position, then run MVP
    float4 worldPos =
        uniforms.modelMatrix * float4(in.position + chunkOrigin.origin.xyz, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;

    // Pass through UV
    out.vUV = in.uv;

    // Unpack face normal (bits 0-2) and texture layer (bits 3+)
    out.vTextureLayer = in.faceAttr >> 3;
    float3 normal = getFaceNormal(in.faceAttr & 7u);
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
    texture2d_array<float> blockTextures [[texture(0)]],
    sampler blockSampler [[sampler(0)]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    // The bound sampler uses repeat addressing + nearest filtering: UVs span
    // the quad extent in blocks, so each block gets one full texture tile.
    float4 texColor = blockTextures.sample(blockSampler, in.vUV, in.vTextureLayer);

    // Alpha cutout for foliage/glass: transparent texels simply don't exist.
    // Runs in the opaque pass with depth writes, so no sorting is needed.
    if (texColor.a < 0.5f) {
        discard_fragment();
    }

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

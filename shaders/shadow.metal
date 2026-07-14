#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Shadow depth pass — renders scene geometry from the sun/moon into one
// cascade's depth slice. Chunk vertices carry their chunk-local fp16 position
// (ChunkOrigin restores world space, exactly like the main pass) and the
// texture layer, so a cutout fragment can discard leaf/flora holes — solid
// shadow blobs otherwise.
// ---------------------------------------------------------------------------

struct ShadowVertexInput {
    uint faceAttr [[attribute(0)]];
    float3 position [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct ShadowVertexOutput {
    float4 clipPosition [[position]];
    float2 vUV;
    uint vTextureLayer [[flat]];
};

vertex ShadowVertexOutput shadowVertexMain(ShadowVertexInput in [[stage_in]],
                                           constant ShadowPassUniforms& shadow [[buffer(1)]],
                                           constant ChunkOrigin& chunkOrigin [[buffer(2)]]) {
    ShadowVertexOutput out;
    float3 worldPos = in.position + chunkOrigin.origin.xyz;
    // The same sway as the scene pass, or foliage shadows detach from blades.
    worldPos =
        applySway(worldPos, (in.faceAttr >> 22) & 3u, in.uv.y, shadow.time, shadow.swayStrength);
    out.clipPosition = shadow.lightViewProj * float4(worldPos, 1.0);
    out.vUV = in.uv;
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    return out;
}

// Alpha cutout so leaves and flora cast hole-punched shadows. Solid faces keep
// their texel alpha at 1, so only genuine holes discard.
fragment void shadowCutoutFragment(ShadowVertexOutput in [[stage_in]],
                                   texture2d_array<float> blockTextures [[texture(0)]],
                                   sampler blockSampler [[sampler(0)]]) {
    float alpha = blockTextures.sample(blockSampler, in.vUV, in.vTextureLayer).a;
    if (alpha < 0.5f) {
        discard_fragment();
    }
}

#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Shadow depth pass, renders scene geometry from the sun/moon into one
// cascade's depth slice. Chunk vertices carry their chunk-local fp16 position
// (ChunkOrigin restores world space, exactly like the main pass) and the
// texture layer, so a cutout fragment can discard leaf/flora holes, solid
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
    float2 vLodWorldPosition;
    float2 vFarLocalPosition;
    uint vTextureLayer [[flat]];
    uint vFace [[flat]];
    uint vFarCanopy [[flat]];
    uint vFarTerrain [[flat]];
    float vLodTransitionProgress [[flat]];
};

vertex ShadowVertexOutput shadowVertexMain(ShadowVertexInput in [[stage_in]],
                                           constant ShadowPassUniforms& shadow [[buffer(1)]],
                                           constant ChunkOrigin& chunkOrigin [[buffer(2)]]) {
    ShadowVertexOutput out;
    const float3 worldPos = in.position + chunkOrigin.origin.xyz;
    out.vLodWorldPosition = worldPos.xz;
    out.vFarLocalPosition = in.position.xz;
    // The same sway as the scene pass, or foliage shadows detach from blades.
    const float3 displacedWorldPos =
        applySway(worldPos, (in.faceAttr >> 22) & 3u, in.uv.y, shadow.time, shadow.foliageWind);
    const float3 relativePosition = in.position +
                                    (chunkOrigin.origin.xyz - shadow.projectionOrigin.xyz) +
                                    (displacedWorldPos - worldPos);
    out.clipPosition = shadow.lightViewProj * float4(relativePosition, 1.0);
    out.vUV = in.uv;
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    out.vFace = in.faceAttr & 7u;
    out.vFarCanopy = (in.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0u;
    out.vFarTerrain = chunkOrigin.farMetadata.w;
    out.vLodTransitionProgress = as_type<float>(chunkOrigin.farMetadata.z);
    return out;
}

// Alpha cutout so leaves and flora cast hole-punched shadows. Solid faces keep
// their texel alpha at 1, so only genuine holes discard.
fragment void shadowCutoutFragment(ShadowVertexOutput in [[stage_in]],
                                   texture2d_array<float> blockTextures [[texture(0)]],
                                   sampler blockSampler [[sampler(0)]],
                                   constant FarTerrainOwnershipUniforms& ownership [[buffer(5)]]) {
    if (in.vFarTerrain != 0u) {
        const bool useEmittingColumn =
            farTerrainOpaqueRiserUsesEmittingColumn(in.vFace, in.vFarCanopy != 0u);
        if (farTerrainExactColumnOwnsFragment(in.vFarLocalPosition, in.vFace, useEmittingColumn,
                                              in.vFarCanopy != 0u, ownership)) {
            discard_fragment();
        }
        const float threshold = interleavedGradientNoise(floor(in.vLodWorldPosition));
        const bool visible =
            in.vFarCanopy != 0u
                ? farTerrainLodCanopyVisible(in.vLodTransitionProgress, threshold, in.vFarTerrain)
                : farTerrainLodTerrainVisible(in.vLodTransitionProgress, in.vFarTerrain);
        if (!visible) {
            discard_fragment();
        }
    }
    float alpha = blockTextures.sample(blockSampler, in.vUV, in.vTextureLayer).a;
    if (alpha < 0.5f) {
        discard_fragment();
    }
}

struct EntityShadowOutput {
    float4 clipPosition [[position]];
};

vertex EntityShadowOutput entityShadowVertexMain(device const EntityVertex* vertices [[buffer(0)]],
                                                 constant ShadowPassUniforms& shadow [[buffer(1)]],
                                                 constant EntityModel& entityModel [[buffer(2)]],
                                                 uint vertexID [[vertex_id]]) {
    const float4 worldPosition = entityModel.model * float4(vertices[vertexID].position, 1.0f);
    EntityShadowOutput output;
    output.clipPosition =
        shadow.lightViewProj * float4(worldPosition.xyz - shadow.projectionOrigin.xyz, 1.0f);
    return output;
}

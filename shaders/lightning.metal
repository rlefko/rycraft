#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

struct BoltVertexOutput {
    float4 position [[position]];
    float worldDistance;
};

struct FullscreenVertexOutput {
    float4 position [[position]];
    float2 uv;
};

namespace {

constant uint MAIN_SEGMENTS = 48U;
constant uint BRANCH_SEGMENTS = 8U;

uint hashWord(uint value) {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    return value ^ (value >> 16U);
}

float unitHash(uint value) {
    return float(hashWord(value) & 0x00FFFFFFU) / float(0x01000000U);
}

uint eventHash(constant LightningUniforms& uniforms, uint stream) {
    return hashWord(uniforms.eventAndShape.x ^ hashWord(uniforms.eventAndShape.y + stream));
}

float3 mainBoltPoint(uint node, constant LightningUniforms& uniforms) {
    const float amount = float(node) / float(MAIN_SEGMENTS);
    const float cloudY = as_type<float>(uniforms.eventAndShape.z);
    const float3 top = float3(uniforms.strikePosition.x, cloudY, uniforms.strikePosition.z);
    const float3 bottom = uniforms.strikePosition;
    const float taper = sin(amount * M_PI_F);
    const uint base = eventHash(uniforms, node * 0x9E3779B9U + 11U);
    const float2 jitter = (float2(unitHash(base), unitHash(base ^ 0xA511E9B3U)) * 2.0f - 1.0f) *
                          (2.0f + 8.0f * taper);
    float3 point = mix(top, bottom, amount);
    point.xz += jitter;
    return point;
}

float3 branchBoltPoint(uint branch, uint node, constant LightningUniforms& uniforms) {
    const uint branchSeed = eventHash(uniforms, 0xB5297A4DU + branch * 0x68E31DA4U);
    const uint anchorNode = 10U + (branchSeed % 25U);
    const float3 anchor = mainBoltPoint(anchorNode, uniforms);
    const float angle = unitHash(branchSeed ^ 0x1B56C4E9U) * (2.0f * M_PI_F);
    const float length = 18.0f + unitHash(branchSeed ^ 0xC6BC2796U) * 26.0f;
    const float amount = float(node) / float(BRANCH_SEGMENTS);
    const float3 direction = float3(cos(angle), -0.58f, sin(angle));
    const uint nodeSeed = hashWord(branchSeed + node * 0x9E3779B9U);
    const float2 jitter =
        (float2(unitHash(nodeSeed), unitHash(nodeSeed ^ 0xD1B54A35U)) * 2.0f - 1.0f) *
        (1.0f + 2.2f * amount);
    float3 point = anchor + direction * (length * amount);
    point.xz += jitter;
    return point;
}

float2 viewportSize(constant LightningUniforms& uniforms) {
    const uint packed = uniforms.eventAndShape.w;
    return max(float2(float(packed & 0xFFFFU), float(packed >> 16U)), 1.0f);
}

} // namespace

vertex BoltVertexOutput lightningBoltVertex(constant LightningUniforms& uniforms [[buffer(0)]],
                                            uint vertexID [[vertex_id]]) {
    const uint segment = vertexID / 2U;
    const uint endpoint = vertexID & 1U;
    float3 worldPosition;
    if (segment < MAIN_SEGMENTS) {
        worldPosition = mainBoltPoint(segment + endpoint, uniforms);
    } else {
        const uint branchSegment = segment - MAIN_SEGMENTS;
        const uint branch = branchSegment / BRANCH_SEGMENTS;
        const uint node = branchSegment % BRANCH_SEGMENTS;
        worldPosition = branchBoltPoint(branch, node + endpoint, uniforms);
    }

    BoltVertexOutput output;
    output.position = uniforms.viewProjection * float4(worldPosition, 1.0f);
    output.worldDistance = length(worldPosition - uniforms.cameraPosition);
    return output;
}

fragment float4 lightningBoltFragment(BoltVertexOutput input [[stage_in]],
                                      texture2d<float> cloudHitDepth [[texture(0)]],
                                      constant LightningUniforms& uniforms [[buffer(0)]]) {
    constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = input.position.xy / viewportSize(uniforms);
    const float cloudDistance = cloudHitDepth.sample(linearClamp, uv).r;
    const float behindCloud =
        cloudDistance > 0.0f && input.worldDistance > cloudDistance ? 1.0f : 0.0f;
    const float cloudTransmission = mix(1.0f, 0.18f, behindCloud);
    const float3 boltColor =
        uniforms.colorAndIntensity.rgb * (7.5f * uniforms.colorAndIntensity.a * cloudTransmission);
    return float4(boltColor, 0.0f);
}

vertex FullscreenVertexOutput lightningFullscreenVertex(constant LightningUniforms& uniforms
                                                        [[buffer(0)]],
                                                        uint vertexID [[vertex_id]]) {
    const float2 positions[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    FullscreenVertexOutput output;
    output.position = float4(positions[vertexID], 0.0f, 1.0f);
    output.uv = float2(positions[vertexID].x * 0.5f + 0.5f, 0.5f - positions[vertexID].y * 0.5f);
    return output;
}

fragment float4 lightningFlashFragment(FullscreenVertexOutput input [[stage_in]],
                                       depth2d<float> sceneDepth [[texture(0)]],
                                       texture2d<float> cloudHitDepth [[texture(1)]],
                                       constant LightningUniforms& uniforms [[buffer(0)]]) {
    constexpr sampler pointClamp(coord::normalized, address::clamp_to_edge, filter::nearest);
    constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    const float sceneZ = sceneDepth.sample(pointClamp, input.uv);
    const float cloudDistance = cloudHitDepth.sample(linearClamp, input.uv).r;
    const float cloudGlow = cloudDistance > 0.0f ? 1.0f : 0.0f;

    const float cloudY = as_type<float>(uniforms.eventAndShape.z);
    const float3 centerPosition =
        float3(uniforms.strikePosition.x, cloudY, uniforms.strikePosition.z);
    const float4 centerClip = uniforms.viewProjection * float4(centerPosition, 1.0f);
    float radial = 0.0f;
    if (centerClip.w > 0.0f) {
        const float2 centerNdc = centerClip.xy / centerClip.w;
        const float2 centerUv = float2(centerNdc.x * 0.5f + 0.5f, 0.5f - centerNdc.y * 0.5f);
        radial = exp(-length(input.uv - centerUv) * 7.0f);
    }
    const float geometryTransmission = sceneZ < 0.9999f ? 0.38f : 1.0f;
    const float flash = uniforms.colorAndIntensity.a * geometryTransmission *
                        (0.045f + radial * 0.32f + cloudGlow * (0.12f + radial * 0.24f));
    return float4(uniforms.colorAndIntensity.rgb * flash, 0.0f);
}

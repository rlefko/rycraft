#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Atmospheric Sky
//
// A fullscreen quad reconstructs a per-pixel view ray from the camera basis
// (matching clouds.metal) and shades the sky analytically:
//   • a Rayleigh-style vertical gradient (saturated blue up, hazy at horizon)
//   • a Mie forward-scatter glow around the sun for sunrise/sunset warmth
//   • true direction-projected sun and moon discs (HDR-bright so they bloom)
//   • a procedural star field that fades in as the sun sets
// Outputs HDR — the sun disc exceeds 1.0 so the bloom pass catches it.
// ---------------------------------------------------------------------------

struct SkyVertexOut {
    float4 clipPosition [[position]];
    float2 vNdc;
};

vertex SkyVertexOut skyVertexMain(uint vertexID [[vertex_id]]) {
    const float2 positions[6] = {float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f),
                                 float2(-1.0f, -1.0f), float2(1.0f, 1.0f),  float2(-1.0f, 1.0f)};
    SkyVertexOut out;
    float2 pos = positions[vertexID];
    out.clipPosition = float4(pos, 0.0f, 1.0f);
    out.vNdc = pos;
    return out;
}

// Hash a direction cell to a pseudo-random value for the star field.
static float hashDir(float3 c) {
    float h = dot(c, float3(127.1f, 311.7f, 74.7f));
    return fract(sin(h) * 43758.5453f);
}

fragment float4 skyFragmentMain(SkyVertexOut in [[stage_in]],
                                constant SkyUniforms &sky [[buffer(1)]]) {
    // View ray from the camera basis + projection shape (matches clouds.metal)
    float3 dir = normalize(sky.cameraForward + in.vNdc.x * sky.aspect * sky.tanHalfFov *
                                                   sky.cameraRight +
                           in.vNdc.y * sky.tanHalfFov * sky.cameraUp);

    // The stored sun/moon vectors carry a fixed z tilt and aren't unit
    // length; normalize so the dot-product disc tests are exact.
    float3 sunDir = normalize(sky.sunDirection);
    float3 moonDir = normalize(sky.moonDirection);

    // Vertical gradient in view space: horizon at dir.y=0, zenith at dir.y=1.
    // pow<1 lifts the horizon band so the transition reads like real haze.
    float up = clamp(dir.y, 0.0f, 1.0f);
    float grad = pow(up, 0.42f);
    float3 color = mix(sky.horizonColor, sky.zenithColor, grad);

    // Rayleigh-ish saturation: deepen the zenith blue a touch during the day.
    color = mix(color, color * float3(0.85f, 0.92f, 1.12f), grad * sky.sunIntensity * 0.5f);

    float sunCos = dot(dir, sunDir);

    // Mie forward scatter: a warm glow around the sun, strongest when the sun
    // sits low (sunrise/sunset), so dawn and dusk bleed orange. The lobe is
    // deliberately tight (pow 24) at moderate amplitude: the old broad pow-8
    // full-strength halo washed a quarter of the frame to white whenever the
    // sun was in view, reading as glare no exposure could recover.
    float mie = pow(max(sunCos, 0.0f), 24.0f);
    float lowSun = 1.0f - clamp(sky.sunDirection.y * 2.0f, 0.0f, 1.0f);
    color += sky.sunColor * mie * (0.22f + 0.45f * lowSun) * sky.sunIntensity;

    // Stars: hash a coarse direction lattice; only the brightest cells light,
    // and only above the horizon, fading in as the sun drops.
    if (sky.starStrength > 0.001f && dir.y > 0.0f) {
        float3 cell = floor(dir * 200.0f);
        float star = hashDir(cell);
        float twinkle = step(0.995f, star) * (star - 0.995f) / 0.005f;
        color += float3(0.9f, 0.95f, 1.0f) * twinkle * sky.starStrength * up;
    }

    // Moon disc: soft, cool, with a faint halo. Drawn before the sun so a
    // (rare) alignment lets the sun win.
    float moonCos = dot(dir, moonDir);
    float moonDisc = smoothstep(0.9994f, 0.9997f, moonCos);
    float moonHalo = pow(max(moonCos, 0.0f), 800.0f) * 0.3f;
    color = mix(color, float3(0.85f, 0.88f, 1.0f), moonDisc);
    color += float3(0.6f, 0.65f, 0.8f) * moonHalo * sky.starStrength;

    // Sun disc: sharp, limb-darkened, HDR-bright so the bloom pass halos it.
    // HDR 8 (was 12) stays far above the bloom threshold but keeps the limb
    // gradient visible once the highlight-aware exposure stops down, instead
    // of plateauing deep in the tonemap shoulder as flat white.
    if (sky.sunIntensity > 0.001f) {
        float sunDisc = smoothstep(0.9992f, 0.9996f, sunCos);
        float limb = 0.7f + 0.3f * smoothstep(0.9992f, 1.0f, sunCos);
        color = mix(color, sky.sunColor * 8.0f * limb, sunDisc);
    }

    return float4(color, 1.0f);
}

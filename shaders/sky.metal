#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Atmospheric Sky
//
// A fullscreen quad reconstructs a per-pixel view ray from the camera basis
// and samples the production atmosphere sky-view LUT. Rayleigh, Mie, ozone,
// altitude, and weather aerosols are integrated when the LUT is built. The
// sun uses its physical angular radius; moon and stars remain night overlays.
// Outputs HDR, the sun disc exceeds 1.0 so the bloom pass catches it.
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
                                constant SkyUniforms& sky [[buffer(1)]],
                                texture2d<float> skyViewLut [[texture(0)]],
                                texture2d<float> transmittanceLut [[texture(1)]]) {
    constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    // View ray from the camera basis + projection shape (matches clouds.metal)
    float3 dir =
        normalize(sky.cameraForward + in.vNdc.x * sky.aspect * sky.tanHalfFov * sky.cameraRight +
                  in.vNdc.y * sky.tanHalfFov * sky.cameraUp);

    // Normalize defensively so the dot-product disc tests remain exact across
    // CPU and GPU layout conversions.
    float3 sunDir = normalize(sky.sunDirection);
    float3 moonDir = normalize(sky.moonDirection);

    float2 viewHorizontal = normalize(dir.xz + float2(1.0e-6f, 0.0f));
    float2 sunHorizontal = normalize(sunDir.xz + float2(1.0e-6f, 0.0f));
    float signedAzimuth =
        atan2(viewHorizontal.x * sunHorizontal.y - viewHorizontal.y * sunHorizontal.x,
              dot(viewHorizontal, sunHorizontal));
    float2 skyUv =
        float2(signedAzimuth / (2.0f * M_PI_F) + 0.5f, saturate((dir.y + 0.08f) / 1.08f));
    float3 physicalDay = skyViewLut.sample(lutSampler, skyUv).rgb;
    float up = clamp(dir.y, 0.0f, 1.0f);
    float3 nightBase = mix(sky.horizonColor, sky.zenithColor, pow(up, 0.42f));
    const float starVisibility = sky.visibilityAndPhase.w;
    float3 color = mix(physicalDay, nightBase, starVisibility);

    // Stars: hash a coarse direction lattice; only the brightest cells light,
    // and only above the horizon, fading in as the sun drops.
    if (starVisibility > 0.001f && dir.y > 0.0f) {
        float3 cell = floor(dir * 200.0f);
        float star = hashDir(cell);
        float twinkle = step(0.995f, star) * (star - 0.995f) / 0.005f;
        color += float3(0.9f, 0.95f, 1.0f) * twinkle * starVisibility * up;
    }

    // The Moon is a projected sphere lit by the true Sun. This produces a
    // continuous terminator instead of a full-bright disc at every phase.
    // A small earthshine floor keeps the dark hemisphere barely legible only
    // after twilight. The disc is drawn at three and a half times its
    // physical angular radius: the true half-degree disc reads as a faint
    // dot on screen, while the eye's moon illusion expects a clearly
    // visible moon.
    float moonCos = dot(dir, moonDir);
    constexpr float LUNAR_ANGULAR_RADIUS = 0.01582f;
    const float3 referenceAxis =
        abs(moonDir.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 moonRight = normalize(cross(referenceAxis, moonDir));
    const float3 moonUp = normalize(cross(moonDir, moonRight));
    const float2 moonPoint =
        float2(dot(dir, moonRight), dot(dir, moonUp)) / sin(LUNAR_ANGULAR_RADIUS);
    const float moonRadius = length(moonPoint);
    const float moonEdgeWidth = max(fwidth(moonRadius), 0.002f);
    const float moonDisc =
        step(0.0f, moonCos) *
        (1.0f - smoothstep(1.0f - moonEdgeWidth, 1.0f + moonEdgeWidth, moonRadius));
    if (moonDisc > 0.0f) {
        const float sphereDepth = sqrt(max(1.0f - dot(moonPoint, moonPoint), 0.0f));
        const float3 lunarNormal =
            normalize(moonPoint.x * moonRight + moonPoint.y * moonUp - sphereDepth * moonDir);
        const float sunlit = saturate(dot(lunarNormal, sunDir));
        const float surfaceVariation = 0.72f + 0.28f * hashDir(floor(lunarNormal * 96.0f));
        const float earthshine = 0.012f * starVisibility;
        const float3 lunarRadiance =
            sky.moonColor * surfaceVariation * (earthshine + sunlit * 0.62f);
        color = mix(color, lunarRadiance, moonDisc * sky.visibilityAndPhase.y);
    }
    const float moonHalo = pow(max(moonCos, 0.0f), 800.0f) * 0.08f;
    color += sky.moonColor * moonHalo * sky.visibilityAndPhase.y * sky.visibilityAndPhase.z;

    // The solar half-angle is 0.2679 degrees (0.004675 radians). Sample the
    // transmittance LUT at the current solar elevation so low sunlight reddens
    // and attenuates through the same atmosphere as the sky.
    if (sky.visibilityAndPhase.x > 0.001f) {
        constexpr float SOLAR_ANGULAR_RADIUS = 0.004675f;
        const float3 referenceAxis =
            abs(sunDir.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
        const float3 sunRight = normalize(cross(referenceAxis, sunDir));
        const float3 sunUp = normalize(cross(sunDir, sunRight));
        const float2 sunPoint =
            float2(dot(dir, sunRight), dot(dir, sunUp)) / sin(SOLAR_ANGULAR_RADIUS);
        const float sunRadius = length(sunPoint);
        const float sunEdgeWidth = max(fwidth(sunRadius), 0.002f);
        const float sunDisc =
            step(0.0f, dot(dir, sunDir)) *
            (1.0f - smoothstep(1.0f - sunEdgeWidth, 1.0f + sunEdgeWidth, sunRadius));
        const float limb = 0.65f + 0.35f * sqrt(saturate(1.0f - sunRadius * sunRadius));
        float transU = atmosphereTransmittanceMuUv(sunDir.y);
        float3 solarTransmittance = transmittanceLut.sample(lutSampler, float2(transU, 0.0f)).rgb;
        color = mix(color, sky.sunColor * solarTransmittance * 18.0f * limb,
                    sunDisc * sky.visibilityAndPhase.x);
    }

    return float4(color, 1.0f);
}

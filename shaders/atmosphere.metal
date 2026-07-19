#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

namespace {
float ozoneDensity(float altitudeKm, constant AtmosphereUniforms& atmosphere) {
    const float center = atmosphere.ozoneAbsorptionAndCenter.w;
    return saturate(1.0f - abs(altitudeKm - center) / 15.0f);
}

float3 mediumExtinction(float altitudeKm, constant AtmosphereUniforms& atmosphere) {
    const float rayleighDensity =
        exp(-max(altitudeKm, 0.0f) / atmosphere.rayleighScatteringAndScaleHeight.w);
    const float mieDensity = exp(-max(altitudeKm, 0.0f) / atmosphere.mieScatteringAndScaleHeight.w);
    const float humidityScale = 1.0f + atmosphere.weatherOptics.y * 1.5f;
    const float3 rayleigh = atmosphere.rayleighScatteringAndScaleHeight.xyz * rayleighDensity;
    const float3 mie =
        atmosphere.mieScatteringAndScaleHeight.xyz * mieDensity * humidityScale * 1.11f;
    const float3 ozone =
        atmosphere.ozoneAbsorptionAndCenter.xyz * ozoneDensity(altitudeKm, atmosphere);
    return rayleigh + mie + ozone;
}

float transmittanceAltitudeUv(float radius, constant AtmosphereUniforms& atmosphere) {
    return sqrt(saturate((radius - atmosphere.atmosphereRadii.x) /
                         (atmosphere.atmosphereRadii.y - atmosphere.atmosphereRadii.x)));
}

float3 integrateTransmittance(float radius, float mu, constant AtmosphereUniforms& atmosphere) {
    const float groundRadius = atmosphere.atmosphereRadii.x;
    const float topRadius = atmosphere.atmosphereRadii.y;
    const float distance = atmosphereTransmittancePathLength(radius, mu, groundRadius, topRadius);
    const float stepLength = distance / 32.0f;
    float3 opticalDepth = 0.0f;
    for (uint step = 0; step < 32; ++step) {
        const float t = (float(step) + 0.5f) * stepLength;
        const float sampleRadius =
            sqrt(max(radius * radius + t * t + 2.0f * radius * mu * t, 0.0f));
        opticalDepth += mediumExtinction(sampleRadius - groundRadius, atmosphere) * stepLength;
    }
    return exp(-opticalDepth);
}

float rayleighPhase(float cosine) {
    return 3.0f * (1.0f + cosine * cosine) / (16.0f * M_PI_F);
}

float miePhase(float cosine, float g) {
    const float g2 = g * g;
    return (1.0f - g2) / (4.0f * M_PI_F * pow(max(1.0f + g2 - 2.0f * g * cosine, 1.0e-4f), 1.5f));
}

} // namespace

kernel void atmosphereTransmittanceKernel(texture2d<half, access::write> output [[texture(0)]],
                                          constant AtmosphereUniforms& atmosphere [[buffer(0)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    const float2 uv = (float2(gid) + 0.5f) / float2(output.get_width(), output.get_height());
    const float radius = mix(atmosphere.atmosphereRadii.x + 0.001f,
                             atmosphere.atmosphereRadii.y - 0.001f, uv.y * uv.y);
    const float mu = atmosphereTransmittanceUvMu(uv.x);
    output.write(half4(half3(integrateTransmittance(radius, mu, atmosphere)), half(1.0f)), gid);
}

kernel void atmosphereMultipleScatteringKernel(texture2d<half, access::sample> transmittance
                                               [[texture(0)]],
                                               texture2d<half, access::write> output [[texture(1)]],
                                               constant AtmosphereUniforms& atmosphere
                                               [[buffer(0)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = (float2(gid) + 0.5f) / float2(output.get_width(), output.get_height());
    float3 meanTransmittance = 0.0f;
    for (uint direction = 0; direction < 16; ++direction) {
        const float mu = (float(direction) + 0.5f) / 16.0f;
        meanTransmittance += float3(
            transmittance.sample(lutSampler, float2(atmosphereTransmittanceMuUv(mu), uv.y)).rgb);
    }
    meanTransmittance /= 16.0f;
    const float3 scatterAlbedo = atmosphere.rayleighScatteringAndScaleHeight.xyz +
                                 atmosphere.mieScatteringAndScaleHeight.xyz;
    const float3 energy = scatterAlbedo * (1.0f - meanTransmittance) /
                          max(1.0f - 0.65f * (1.0f - meanTransmittance), 0.1f);
    output.write(half4(half3(energy * atmosphere.sunRadiance), half(1.0f)), gid);
}

kernel void atmosphereSkyViewKernel(texture2d<half, access::sample> transmittance [[texture(0)]],
                                    texture2d<half, access::sample> multipleScattering
                                    [[texture(1)]],
                                    texture2d<half, access::write> output [[texture(2)]],
                                    constant AtmosphereUniforms& atmosphere [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = (float2(gid) + 0.5f) / float2(output.get_width(), output.get_height());
    const float viewMu = mix(-0.08f, 1.0f, uv.y);
    const float relativeAzimuth = (uv.x * 2.0f - 1.0f) * M_PI_F;
    const float sunMu = clamp(atmosphere.sunDirection.y, -1.0f, 1.0f);
    // Preserve the true solar source through civil and nautical twilight,
    // then extinguish it once the Sun is 18 degrees below the horizon. This
    // keeps the atmosphere authoritative without a nonzero nighttime floor.
    const float twilightVisibility = smoothstep(-0.309017f, 0.008727f, sunMu);
    const float viewSin = sqrt(max(1.0f - viewMu * viewMu, 0.0f));
    const float sunSin = sqrt(max(1.0f - sunMu * sunMu, 0.0f));
    const float viewSunCos =
        clamp(viewMu * sunMu + viewSin * sunSin * cos(relativeAzimuth), -1.0f, 1.0f);

    const float groundRadius = atmosphere.atmosphereRadii.x;
    const float topRadius = atmosphere.atmosphereRadii.y;
    const float cameraRadius = atmosphere.cameraPositionKm.y;
    const float3 viewDirection =
        float3(viewSin * cos(relativeAzimuth), viewMu, viewSin * sin(relativeAzimuth));
    const float3 sunDirection = float3(sunSin, sunMu, 0.0f);
    const float pathLength =
        atmosphereTransmittancePathLength(cameraRadius, viewMu, groundRadius, topRadius);
    const bool viewHitsGround = atmosphereRayHitsGround(cameraRadius, viewMu, groundRadius);

    constexpr uint VIEW_STEPS = 24u;
    const float stepLength = pathLength / float(VIEW_STEPS);
    float3 viewTransmittance = 1.0f;
    float3 radiance = 0.0f;
    for (uint step = 0u; step < VIEW_STEPS; ++step) {
        const float distance = (float(step) + 0.5f) * stepLength;
        const float3 samplePosition = float3(0.0f, cameraRadius, 0.0f) + viewDirection * distance;
        const float sampleRadius = length(samplePosition);
        const float altitude = max(sampleRadius - groundRadius, 0.0f);
        const float3 localUp = samplePosition / max(sampleRadius, 1.0f);
        const float localSunMu = dot(localUp, sunDirection);
        float3 sunTransmittance = 0.0f;
        if (!atmosphereRayHitsGround(sampleRadius, localSunMu, groundRadius)) {
            const float2 sunTransmittanceUv =
                float2(atmosphereTransmittanceMuUv(localSunMu),
                       transmittanceAltitudeUv(sampleRadius, atmosphere));
            sunTransmittance = float3(transmittance.sample(lutSampler, sunTransmittanceUv).rgb);
        }
        const float rayleighDensity =
            exp(-altitude / atmosphere.rayleighScatteringAndScaleHeight.w);
        const float mieDensity = exp(-altitude / atmosphere.mieScatteringAndScaleHeight.w);
        const float3 rayleighScatter = atmosphere.rayleighScatteringAndScaleHeight.xyz *
                                       rayleighDensity * rayleighPhase(viewSunCos);
        const float3 mieScatter = atmosphere.mieScatteringAndScaleHeight.xyz * mieDensity *
                                  (1.0f + atmosphere.weatherOptics.y * 1.5f) *
                                  miePhase(viewSunCos, atmosphere.weatherOptics.w);
        radiance += viewTransmittance * (rayleighScatter + mieScatter) * sunTransmittance *
                    atmosphere.sunRadiance * stepLength;
        viewTransmittance *= exp(-mediumExtinction(altitude, atmosphere) * stepLength);
    }

    const float altitudeUv = transmittanceAltitudeUv(cameraRadius, atmosphere);
    const float3 multi = float3(
        multipleScattering.sample(lutSampler, float2(saturate(sunMu * 0.5f + 0.5f), altitudeUv))
            .rgb);
    radiance += multi * (1.0f - viewTransmittance) * 0.35f;

    // A ray beneath the geometric horizon ends at the planet surface. The
    // old LUT only integrated the air on that segment, leaving a near-black
    // band beneath clear skies whenever far terrain had not yet streamed in.
    // Give the lower boundary a sunlit Lambertian response and retain its
    // aerial transmittance so it joins the horizon smoothly.
    if (viewHitsGround) {
        const float3 groundPosition = float3(0.0f, cameraRadius, 0.0f) + viewDirection * pathLength;
        const float groundHitRadius = length(groundPosition);
        const float3 groundUp = groundPosition / max(groundHitRadius, 1.0f);
        const float groundSunMu = dot(groundUp, sunDirection);
        float3 groundSunTransmittance = 0.0f;
        if (!atmosphereRayHitsGround(groundHitRadius, groundSunMu, groundRadius)) {
            const float2 groundSunUv = float2(atmosphereTransmittanceMuUv(groundSunMu),
                                              transmittanceAltitudeUv(groundHitRadius, atmosphere));
            groundSunTransmittance =
                float3(transmittance.sample(lutSampler, groundSunUv).rgb) * atmosphere.sunRadiance;
        }
        const float3 groundMulti = float3(
            multipleScattering
                .sample(lutSampler, float2(saturate(groundSunMu * 0.5f + 0.5f),
                                           transmittanceAltitudeUv(groundHitRadius, atmosphere)))
                .rgb);
        radiance += viewTransmittance * atmosphereGroundRadiance(atmosphere.groundAlbedo,
                                                                 groundSunTransmittance,
                                                                 groundSunMu, groundMulti);
    }
    // The physical phase functions above are normalized over 4 pi
    // steradians. Convert their per-steradian result into Rycraft's shared
    // HDR scene range, where the broad sky must meter against a tiny solar
    // disc and directly lit terrain. Without this calibration the LUT was
    // more than three stops too dark even in daylight.
    radiance = atmosphereSceneRadiance(radiance) * twilightVisibility;
    radiance *= mix(0.42f, 1.0f, saturate(viewMu + 0.15f));
    // Volumetric cloud transmittance is composited directionally after the
    // sky. Camera-local coverage here would dim clear gaps and attenuate
    // cloudy pixels a second time in the cloud pass.
    output.write(half4(half3(max(radiance, 0.0f)), half(1.0f)), gid);
}
